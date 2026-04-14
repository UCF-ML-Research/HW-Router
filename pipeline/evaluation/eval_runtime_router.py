import os
import time
import threading
import datetime
import yaml
import pandas as pd
import numpy as np
from openai import OpenAI

from hw_router.hardware_monitor import start_metrics_watcher, model_metrics
from hw_router.load_patterns import RequestPattern
from hw_router.cost_predictor import HardwareCostPredictor
from hw_router.model_registry import get_model_id, get_model_hugging_face_name
from hw_router.constants import (
    MODEL_PRICES,
    LAT_P95_LOG,
    STATIC_COST_P95,
    DEFAULT_LAMBDA,
)

# =========================================================
# Normalization constants (from hw_router.constants)
# =========================================================
STATIC_P95 = STATIC_COST_P95

# Scoring lambda
LAMBDA = DEFAULT_LAMBDA


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
# GPU-level monitor (aggregated running / waiting / kv-cache)
# =========================================================
def start_gpu_monitor(model_to_gpu, gpu_stats_list, interval=1.0):
    def monitor():
        while True:
            now = time.time()
            for gid in sorted(set(model_to_gpu.values())):
                running = 0
                waiting = 0
                kv_cache = 0.0

                for model, mg in model_to_gpu.items():
                    if mg != gid:
                        continue
                    snap = model_metrics.get(model, {})
                    running += snap.get("num_requests_running", 0)
                    waiting += snap.get("num_requests_waiting", 0)
                    kv_cache = max(kv_cache, snap.get("kv_cache_usage_perc", 0.0))

                gpu_stats_list.append({
                    "timestamp": now,
                    "gpu_id": gid,
                    "running": running,
                    "waiting": waiting,
                    "kv_cache_usage": kv_cache,
                })

            time.sleep(interval)

    t = threading.Thread(target=monitor, daemon=True)
    t.start()


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
        )

    return lookup


# =========================================================
# Main evaluation logic
# =========================================================
def run_eval(
    config,
    prompt_path,
    eval_csv_path,
    router_type,      # "carrot" or "hw"
    output_dir,
    arrival_rate,
    concurrency,
    interval,
    num_prompts=None,
):
    os.makedirs(output_dir, exist_ok=True)

    RUN_ID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"[Eval] run_id={RUN_ID}, router={router_type}, λ={LAMBDA}")

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

    # GPU monitor
    gpu_stats = []
    start_gpu_monitor(model_to_gpu, gpu_stats, interval=1.0)

    # ----------------------------
    # Load prompts (eval set)
    # ----------------------------
    df_prompts = pd.read_parquet(prompt_path)

    # Your file has columns: ['id', 'source', 'prompt', 'p_tokens']
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
    # Output paths
    # ----------------------------
    out_router = os.path.join(output_dir, f"{RUN_ID}_router_results.csv")
    out_gpu = os.path.join(output_dir, f"{RUN_ID}_gpu_monitor.csv")
    writer_lock = threading.Lock()

    # ----------------------------
    # Poisson arrival pattern
    # ----------------------------
    pattern = RequestPattern("poisson", arrival_rate)

    # ----------------------------
    # Worker per request
    # ----------------------------
    def worker(prompt_source_id, prompt):
        p_tokens = len(prompt.split())

        best_score = None
        best_choice = None

        for local_model_name in model_names:
            gpu_id = model_to_gpu[local_model_name]
            hf_name = get_model_hugging_face_name(local_model_name)
            key = (prompt_source_id, hf_name)

            if key not in eval_lookup:
                raise KeyError(f"Missing eval row for (prompt_source_id={prompt_source_id}, model={hf_name})")

            quality, pred_len = eval_lookup[key]

            # static CARROT cost
            price = MODEL_PRICES.get(hf_name)
            if price is None:
                raise KeyError(f"Missing price for model {hf_name}")
            static_cost = pred_len * price
            static_cost_norm = min(static_cost / STATIC_P95, 1.0)

            # HW snapshot
            snap = model_metrics.get(local_model_name, {})
            running = snap.get("num_requests_running", 0)
            waiting = snap.get("num_requests_waiting", 0)
            kv_usage = snap.get("kv_cache_usage_perc", 0.0)

            # cost term: CARROT vs HW
            if router_type == "carrot":
                cost_term = static_cost_norm
                ttft_hat = 0.0
                tpot_hat = 0.0
                pred_total = 0.0
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

                pred_total = raw_lat
                lat_norm = np.log1p(raw_lat) / LAT_P95_LOG
                cost_term = min(lat_norm, 1.0)

            score = LAMBDA * quality - (1.0 - LAMBDA) * cost_term

            if best_score is None or score > best_score:
                best_score = score
                best_choice = {
                    "local_model": local_model_name,
                    "hf_model": hf_name,
                    "gpu_id": gpu_id,
                    "quality": quality,
                    "pred_len": pred_len,
                    "static_cost_norm": static_cost_norm,
                    "cost_term": cost_term,
                    "score": score,
                    "running": running,
                    "waiting": waiting,
                    "kv": kv_usage,
                    "ttft_hat": ttft_hat if router_type == "hw" else 0.0,
                    "tpot_hat": tpot_hat if router_type == "hw" else 0.0,
                    "pred_total": pred_total if router_type == "hw" else 0.0,
                }

        if best_choice is None:
            return

        client = clients[best_choice["local_model"]]
        ttft, tpot, latency, d_tokens = send_request_and_measure(
            client, best_choice["local_model"], prompt
        )

        row = {
            "run_id": RUN_ID,
            "router": router_type,
            "prompt_source_id": prompt_source_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "model_hf": best_choice["hf_model"],
            "gpu_id": best_choice["gpu_id"],
            "quality_pred": best_choice["quality"],
            "pred_length": best_choice["pred_len"],
            "static_cost_norm": best_choice["static_cost_norm"],
            "cost_term_used": best_choice["cost_term"],
            "score": best_choice["score"],
            "hw_running": best_choice["running"],
            "hw_waiting": best_choice["waiting"],
            "hw_kv_cache": best_choice["kv"],
            "ttft_hat": best_choice["ttft_hat"],
            "tpot_hat": best_choice["tpot_hat"],
            "pred_total_latency": best_choice["pred_total"],
            "p_tokens": p_tokens,
            "d_tokens": d_tokens,
            "ttft_real": ttft,
            "tpot_real": tpot,
            "latency_real": latency,
        }

        with writer_lock:
            file_exists = os.path.exists(out_router)
            with open(out_router, "a") as f:
                if not file_exists:
                    f.write(",".join(row.keys()) + "\n")
                f.write(",".join(str(v) for v in row.values()) + "\n")

    # ----------------------------
    # Launch threads with Poisson arrival
    # ----------------------------
    threads = []
    for prompt_source_id, prompt in prompts:
        while len([t for t in threads if t.is_alive()]) >= concurrency:
            time.sleep(0.01)

        t = threading.Thread(target=worker, args=(prompt_source_id, prompt))
        t.start()
        threads.append(t)

        time.sleep(pattern.next_delay())

    for t in threads:
        t.join()

    pd.DataFrame(gpu_stats).to_csv(out_gpu, index=False)

    print(f"[Eval] Router results → {out_router}")
    print(f"[Eval] GPU monitor    → {out_gpu}")


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="YAML GPU→model map (same as training/eval).")
    parser.add_argument("--prompt_path", required=True,
                        help="Parquet with eval prompts (must have columns 'id' and 'prompt').")
    parser.add_argument("--eval_csv", required=True,
                        help="evaluation_dataset_processed_full_with_umr_irt.csv")
    parser.add_argument("--router", choices=["carrot", "hw"], required=True,
                        help="Router type: 'carrot' (static cost) or 'hw' (hardware-aware).")
    parser.add_argument("--output_dir", default="data/router_eval_runs",
                        help="Directory to write results CSVs.")
    parser.add_argument("--arrival_rate", type=float, default=18.0,
                        help="Poisson arrival rate λ.")
    parser.add_argument("--concurrency", type=int, default=80,
                        help="Max concurrent threads (requests).")
    parser.add_argument("--interval", type=float, default=0.3,
                        help="Metrics watcher poll interval (seconds).")
    parser.add_argument("--num_prompts", type=int, default=None,
                        help="Optional: cap number of prompts (for testing).")

    args = parser.parse_args()

    run_eval(
        config=args.config,
        prompt_path=args.prompt_path,
        eval_csv_path=args.eval_csv,
        router_type=args.router,
        output_dir=args.output_dir,
        arrival_rate=args.arrival_rate,
        concurrency=args.concurrency,
        interval=args.interval,
        num_prompts=args.num_prompts,
    )
