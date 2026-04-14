import os
import time
import threading
import datetime
import yaml
import pandas as pd
import numpy as np
from collections import Counter
from openai import OpenAI

from hw_router.hardware_monitor import start_metrics_watcher, model_metrics
from hw_router.load_patterns import RequestPattern
from hw_router.cost_predictor import HardwareCostPredictor
from hw_router.model_registry import get_model_id, get_model_hugging_face_name
from hw_router.constants import (
    MODEL_PRICES,
    LAT_P95_LOG,
    STATIC_COST_P95,
    STATIC_COST_P95_IRT,
    DEFAULT_LAMBDA,
)

# =========================================================
# Normalization constants (from hw_router.constants)
# =========================================================
STATIC_P95 = STATIC_COST_P95
STATIC_P95_IRT = STATIC_COST_P95_IRT

# Fixed score lambda
SCORE_LAMBDA = DEFAULT_LAMBDA


def parse_float_list(s: str):
    """Parse comma-separated float list."""
    return [float(x.strip()) for x in s.split(",") if x.strip()]


# =========================================================
#  Send request and measure TTFT, TPOT, latency
# =========================================================
def send_request_and_measure(openai_client, model_name, prompt):
    start = time.time()
    stream = openai_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=1024,
    )

    first_token_time, total_tokens = None, 0
    for chunk in stream:
        if hasattr(chunk, "choices"):
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", "")
            if text:
                total_tokens += len(text.split())
                if first_token_time is None:
                    first_token_time = time.time()

    end = time.time()
    ttft = (first_token_time - start) if first_token_time else 0.0
    latency = end - start
    tpot = (latency - ttft) / max(total_tokens, 1) if ttft > 0 else 0.0

    return ttft, tpot, latency, total_tokens


# =========================================================
# GPU/model-level monitor (running / waiting / kv-cache)
# =========================================================
def start_gpu_monitor(model_to_gpu, model_stats_list, interval=1.0):
    """
    Periodically snapshot per-model hardware stats:
    - running / waiting queue
    - kv_cache usage
    """
    stop_event = threading.Event()

    def monitor():
        while not stop_event.is_set():
            now = time.time()
            for local_model_name, gid in model_to_gpu.items():
                snap = model_metrics.get(local_model_name, {})
                running = snap.get("num_requests_running", 0)
                waiting = snap.get("num_requests_waiting", 0)
                kv_cache = snap.get("kv_cache_usage_perc", 0.0)

                model_stats_list.append(
                    {
                        "timestamp": now,
                        "model": local_model_name,
                        "gpu_id": gid,
                        "running": running,
                        "waiting": waiting,
                        "kv_cache_usage": kv_cache,
                    }
                )

            time.sleep(interval)

    t = threading.Thread(target=monitor, daemon=True)
    t.start()
    return stop_event


# =========================================================
# Build lookup from evaluation CSV:
# (prompt_source_id, model_hf) -> (quality, predicted_length)
# =========================================================
def build_eval_lookup(eval_csv_path):
    df = pd.read_csv(eval_csv_path)

    required = [
        "prompt_source_id",
        "model_hf",
        "carrot_predicted_quality",
        "carrot_predicted_length",
        "umr_quality_score"
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"eval_csv missing column: {c}")

    df["prompt_source_id"] = df["prompt_source_id"].astype(str)
    df["model_hf"] = df["model_hf"].astype(str)

    lookup = {}
    for _, row in df.iterrows():
        key = (row["prompt_source_id"], row["model_hf"])
        lookup[key] = (
            float(row["carrot_predicted_quality"]),
            float(row["carrot_predicted_length"]),
            float(row["umr_quality_score"]),
        )

    return lookup


# =========================================================
# Main evaluation logic for a single (router, arrival_rate)
# =========================================================
def run_eval(
    config,
    prompt_path,
    eval_csv_path,
    router_type,  # "carrot" or "hw"
    output_dir,
    arrival_rate,
    concurrency,
    interval,
    pattern_type,
    num_prompts=None,
):
    os.makedirs(output_dir, exist_ok=True)

    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(
        f"[Eval] run_id={run_id}, router={router_type}, "
        f"score_lambda={SCORE_LAMBDA}, arrival_rate={arrival_rate}, pattern={pattern_type}"
    )

    # ----------------------------
    # Load GPU ↔ model map (YAML)
    # ----------------------------
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    clients = {}
    model_to_gpu = {}
    model_url_map = {}

    for g, models in cfg["gpus"].items():
        g_id = int(g)
        for m in models:
            local_name = m["name"]
            base_root = m["base_url"].rstrip("/")
            openai_base = f"{base_root}/v1"

            clients[local_name] = OpenAI(base_url=openai_base, api_key="EMPTY")
            model_to_gpu[local_name] = g_id
            model_url_map[local_name] = f"{base_root}/metrics"

    model_names = list(clients.keys())
    print(f"[Eval] Models: {model_names}")

    # ----------------------------
    # Start background metrics watcher
    # ----------------------------
    print("[Eval] Starting metrics watcher...")
    start_metrics_watcher(model_url_map, interval=interval)

    # GPU/model monitor
    model_stats = []
    gpu_monitor_stop = start_gpu_monitor(model_to_gpu, model_stats, interval=1.0)

    # ----------------------------
    # Load prompts (eval set)
    # ----------------------------
    df_prompts = pd.read_parquet(prompt_path)

    # Must contain 'id' and 'prompt'
    if "id" not in df_prompts.columns or "prompt" not in df_prompts.columns:
        raise ValueError("Prompt parquet must contain 'id' and 'prompt' columns.")

    # Map 'id' to prompt_source_id used in eval CSV
    df_prompts["prompt_source_id"] = df_prompts["id"].astype(str)

    if num_prompts is not None:
        df_prompts = df_prompts.head(num_prompts)

    prompts = [
        (row["prompt_source_id"], row["prompt"])
        for _, row in df_prompts.iterrows()
    ]
    print(f"[Eval] Loaded {len(prompts)} prompts from {prompt_path}")

    # ----------------------------
    # Load evaluation lookup (quality, length)
    # ----------------------------
    print(f"[Eval] Loading eval CSV: {eval_csv_path}")
    eval_lookup = build_eval_lookup(eval_csv_path)
    print(f"[Eval] Eval lookup size: {len(eval_lookup)} entries")

    # ----------------------------
    # HW cost predictor (only for hw-router)
    # ----------------------------
    cost_predictor = None
    if router_type == "hw":
        cost_predictor = HardwareCostPredictor(
            "checkpoints/hardware_cost_model/model.pt",
            "checkpoints/hardware_cost_model/preproc.joblib",
        )
        print("[Eval] HW cost predictor loaded.")

    # ----------------------------
    # Aggregation structures (thread-safe)
    # ----------------------------
    dispatch_model_counter = Counter()  # counts per HF model
    dispatch_gpu_counter = Counter()    # counts per gpu_id

    agg_stats = {
        "ttft_sum": 0.0,
        "tpot_sum": 0.0,
        "lat_sum": 0.0,
        "p_tokens_sum": 0.0,
        "d_tokens_sum": 0.0,
        "count": 0,
    }

    stats_lock = threading.Lock()
    dispatch_lock = threading.Lock()

    # ----------------------------
    # Arrival pattern
    # ----------------------------
    pattern = RequestPattern(pattern_type, arrival_rate)

    # ----------------------------
    # Worker per request
    # ----------------------------
    def worker(prompt_source_id, prompt):
        p_tokens = len(prompt.split())

        best_score = None
        best_choice = None

        # Model selection
        for local_model_name in model_names:
            gpu_id = model_to_gpu[local_model_name]
            hf_name = get_model_hugging_face_name(local_model_name)
            key = (prompt_source_id, hf_name)

            if key not in eval_lookup:
                raise KeyError(
                    f"Missing eval row for (prompt_source_id={prompt_source_id}, model={hf_name})"
                )

            quality, pred_len, quality_umr = eval_lookup[key]

            # static CARROT cost
            price = MODEL_PRICES.get(hf_name)
            if price is None:
                raise KeyError(f"Missing price for model {hf_name}")
            static_cost = pred_len * price
            static_cost_norm = min(static_cost / STATIC_P95, 1.0)

            static_cost_irt = price
            static_cost_norm_irt = min(static_cost_irt / STATIC_P95_IRT, 1.0)
            static_cost_norm_umr = min(static_cost_irt / STATIC_P95_IRT, 1.0)


            # HW snapshot
            snap = model_metrics.get(local_model_name, {})
            running = snap.get("num_requests_running", 0)
            waiting = snap.get("num_requests_waiting", 0)
            kv_usage = snap.get("kv_cache_usage_perc", 0.0)

            # cost term: CARROT vs HW
            if router_type == "carrot":
                cost_term = static_cost_norm
            elif router_type == "irt":
                cost_term = static_cost_norm_irt
            elif router_type == "umr":
                cost_term = static_cost_norm_umr
                quality = quality_umr
            else:
                model_id_int = get_model_id(local_model_name)
                feat = {
                    "p_tokens": p_tokens,
                    "running_req_count": running,
                    "waiting_req_count": waiting,
                    "kv_cache_usage_perc": kv_usage,
                    "ttft_avg": snap.get("ttft_avg", 0.0),
                    "itl_avg": snap.get("itl_avg", 0.0),
                    "e2e_avg": snap.get("e2e_avg", 0.0),
                    "model_id": model_id_int,
                    "gpu_id": str(gpu_id),
                }

                ttft_hat, tpot_hat = cost_predictor(model_id_int, feat)
                raw_lat = ttft_hat + pred_len * tpot_hat
                raw_lat = max(raw_lat, 1e-6)

                # log-normalized latency cost
                lat_norm = np.log1p(raw_lat) / LAT_P95_LOG
                cost_term = min(lat_norm, 1.0)

            # Score with fixed lambda = 0.5
            score = SCORE_LAMBDA * quality - (1.0 - SCORE_LAMBDA) * cost_term

            if best_score is None or score > best_score:
                best_score = score
                best_choice = {
                    "local_model": local_model_name,
                    "hf_model": hf_name,
                    "gpu_id": gpu_id,
                    "quality": quality,
                    "pred_len": pred_len,
                }

        if best_choice is None:
            return

        # Record dispatch counts
        with dispatch_lock:
            dispatch_model_counter[best_choice["hf_model"]] += 1
            dispatch_gpu_counter[best_choice["gpu_id"]] += 1

        # Real inference
        client = clients[best_choice["local_model"]]
        ttft, tpot, latency, d_tokens = send_request_and_measure(
            client, best_choice["local_model"], prompt
        )

        # Aggregate stats
        with stats_lock:
            agg_stats["ttft_sum"] += ttft
            agg_stats["tpot_sum"] += tpot
            agg_stats["lat_sum"] += latency
            agg_stats["p_tokens_sum"] += p_tokens
            agg_stats["d_tokens_sum"] += d_tokens
            agg_stats["count"] += 1

    # ----------------------------
    # Launch threads with pattern
    # ----------------------------
    threads = []
    for prompt_source_id, prompt in prompts:
        # Concurrency control
        while len([t for t in threads if t.is_alive()]) >= concurrency:
            time.sleep(0.01)

        t = threading.Thread(target=worker, args=(prompt_source_id, prompt))
        t.start()
        threads.append(t)

        time.sleep(pattern.next_delay())

    for t in threads:
        t.join()

    # Stop GPU/model monitor
    gpu_monitor_stop.set()
    time.sleep(0.1)

    # ----------------------------
    # Aggregate summary statistics
    # ----------------------------
    summary = {
        "run_id": run_id,
        "router": router_type,
        "lambda_score": SCORE_LAMBDA,
        "arrival_rate": arrival_rate,
        "pattern_type": pattern_type,
        "num_prompts": len(prompts),
        "num_completed": agg_stats["count"],
    }

    if agg_stats["count"] > 0:
        cnt = agg_stats["count"]
        summary.update(
            {
                "avg_ttft_real": agg_stats["ttft_sum"] / cnt,
                "avg_tpot_real": agg_stats["tpot_sum"] / cnt,
                "avg_latency_real": agg_stats["lat_sum"] / cnt,
                "avg_p_tokens": agg_stats["p_tokens_sum"] / cnt,
                "avg_d_tokens": agg_stats["d_tokens_sum"] / cnt,
            }
        )
    else:
        summary.update(
            {
                "avg_ttft_real": 0.0,
                "avg_tpot_real": 0.0,
                "avg_latency_real": 0.0,
                "avg_p_tokens": 0.0,
                "avg_d_tokens": 0.0,
            }
        )

    # Model-level queue stats from monitor
    if len(model_stats) > 0:
        ms_df = pd.DataFrame(model_stats)

        # Per-model averages (use HF names)
        for model_name in ms_df["model"].unique():
            sub = ms_df[ms_df["model"] == model_name]
            hf_name = get_model_hugging_face_name(model_name)
            key_prefix = f"model_{hf_name}"
            summary[f"{key_prefix}_avg_running"] = float(sub["running"].mean())
            summary[f"{key_prefix}_avg_waiting"] = float(sub["waiting"].mean())
            summary[f"{key_prefix}_avg_kv_cache"] = float(
                sub["kv_cache_usage"].mean()
            )

        # Per-GPU averages
        for gid in ms_df["gpu_id"].unique():
            sub = ms_df[ms_df["gpu_id"] == gid]
            key_prefix = f"gpu_{gid}"
            summary[f"{key_prefix}_avg_running"] = float(sub["running"].mean())
            summary[f"{key_prefix}_avg_waiting"] = float(sub["waiting"].mean())
            summary[f"{key_prefix}_avg_kv_cache"] = float(
                sub["kv_cache_usage"].mean()
            )

        # Overall averages
        summary["overall_avg_running"] = float(ms_df["running"].mean())
        summary["overall_avg_waiting"] = float(ms_df["waiting"].mean())
        summary["overall_avg_kv_cache"] = float(
            ms_df["kv_cache_usage"].mean()
        )
    else:
        summary["overall_avg_running"] = 0.0
        summary["overall_avg_waiting"] = 0.0
        summary["overall_avg_kv_cache"] = 0.0

    # Prompt distribution among models (HF names)
    for m_name, c in dispatch_model_counter.items():
        summary[f"dispatch_count_model_{m_name}"] = int(c)

    # Prompt distribution among GPUs
    for gid, c in dispatch_gpu_counter.items():
        summary[f"dispatch_count_gpu_{gid}"] = int(c)

    print(
        f"[Eval] Done. Completed={summary['num_completed']}, "
        f"avg_latency={summary['avg_latency_real']:.3f}s, "
        f"overall_avg_waiting={summary['overall_avg_waiting']:.3f}"
    )

    return summary


# =========================================================
# CLI: sweep over routers (outer) and arrival_rates (inner)
# Append to CSV after each run
# =========================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/gpu_model_map_h100.yaml",
        help="YAML GPU→model map (same as training/eval).",
    )
    parser.add_argument(
        "--prompt_path",
        default="data/prompts/mixed_prompts_eval.parquet",
        help="Parquet with eval prompts (must have columns 'id' and 'prompt').",
    )
    parser.add_argument(
        "--eval_csv",
        default="data/evaluation_dataset_processed_full.csv",
        help="evaluation_dataset_processed_full.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="data/router_eval_runs",
        help="Directory to write summary CSV.",
    )
    parser.add_argument(
        "--arrival_rates",
        default="15,18,21",
        help="Comma-separated arrival rates (e.g., '5,10,15,20').",
    )
    parser.add_argument(
        "--pattern_type",
        default="sustained",
        help="Request pattern type: e.g., 'poisson', 'sustained', 'microburst'.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=40,
        help="Max concurrent threads (requests).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.2,
        help="Metrics watcher poll interval (seconds).",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=1685,
        help="Optional: cap number of prompts (for testing).",
    )
    parser.add_argument(
        "--router",
        choices=["carrot", "hw", "irt", "umr"],
        default="hw",
        help="Router type to evaluate.",
    )

    args = parser.parse_args()

    arrival_values = parse_float_list(args.arrival_rates)

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, f"eval_summary_{args.router}.csv")

    # Define full schema once
    template_cols = [
        "run_id","router","lambda_score","arrival_rate","pattern_type",
        "num_prompts","num_completed",
        "avg_ttft_real","avg_tpot_real","avg_latency_real",
        "avg_p_tokens","avg_d_tokens",
        "overall_avg_running","overall_avg_waiting","overall_avg_kv_cache",
        "model_Qwen2.5-14B-Instruct_avg_running",
        "model_Qwen2.5-14B-Instruct_avg_waiting",
        "model_Qwen2.5-14B-Instruct_avg_kv_cache",
        "model_Phi-3-mini-128k-instruct_avg_running",
        "model_Phi-3-mini-128k-instruct_avg_waiting",
        "model_Phi-3-mini-128k-instruct_avg_kv_cache",
        "model_Llama-3.1-8B-Instruct_avg_running",
        "model_Llama-3.1-8B-Instruct_avg_waiting",
        "model_Llama-3.1-8B-Instruct_avg_kv_cache",
        "model_Qwen2.5-3B-Instruct_avg_running",
        "model_Qwen2.5-3B-Instruct_avg_waiting",
        "model_Qwen2.5-3B-Instruct_avg_kv_cache",
        "model_Mistral-7B-Instruct-v0.3_avg_running",
        "model_Mistral-7B-Instruct-v0.3_avg_waiting",
        "model_Mistral-7B-Instruct-v0.3_avg_kv_cache",
        "gpu_0_avg_running","gpu_0_avg_waiting","gpu_0_avg_kv_cache",
        "gpu_1_avg_running","gpu_1_avg_waiting","gpu_1_avg_kv_cache",
        "dispatch_count_model_Qwen2.5-3B-Instruct",
        "dispatch_count_model_Llama-3.1-8B-Instruct",
        "dispatch_count_model_Mistral-7B-Instruct-v0.3",
        "dispatch_count_gpu_1",
        "dispatch_count_gpu_0"
    ]

    df_template = pd.DataFrame(columns=template_cols)

    # Create CSV with header once
    if not os.path.exists(summary_path):
        df_template.to_csv(summary_path, index=False)

    # Main loop
    for router in [args.router]:
        print(f"\n===== ROUTER: {router} =====")
        for arr in arrival_values:
            print(f"\n--- arrival_rate={arr} ---")
            summary = run_eval(
                config=args.config,
                prompt_path=args.prompt_path,
                eval_csv_path=args.eval_csv,
                router_type=router,
                output_dir=args.output_dir,
                arrival_rate=arr,
                concurrency=args.concurrency,
                interval=args.interval,
                pattern_type=args.pattern_type,
                num_prompts=args.num_prompts,
            )

            df_row = pd.DataFrame([summary])

            # Fill missing fields with 0
            for col in template_cols:
                if col not in df_row.columns:
                    df_row[col] = 0

            df_row = df_row[template_cols]

            # Append row
            df_row.to_csv(summary_path, mode="a", header=False, index=False)

    print(f"\n[Eval] All summaries appended to {summary_path}")
