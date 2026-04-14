"""
build_hardware_cost_dataset.py
Collects per-request latency + hardware metrics from vLLM for cost-model training.
Now loads prompts from src/data/mixed_prompts_final.parquet and supports
realistic arrival patterns using RequestPattern.
"""

import argparse, csv, os, random, time, uuid, yaml, datetime, torch, threading
from openai import OpenAI
import pandas as pd
from hw_router.hardware_monitor import start_metrics_watcher, model_metrics
from hw_router.load_patterns import RequestPattern


# ---------------------- CONFIG ----------------------

CSV_FIELDS = [
    "request_id", "timestamp", "prompt_id", "model_id", "gpu_id",
    "p_tokens",
    "running_req_count", "waiting_req_count", "kv_cache_usage_perc",
    "ttft_avg", "itl_avg", "e2e_avg",
    "ttft_s", "tpot_s_per_token", "latency_s",
    "d_tokens",
    "pattern_type", "arrival_rate",
]


# ---------------------- PROMPTS ----------------------

def load_local_prompts(parquet_path: str, n: int = 10):
    """
    Load n prompts from the combined local Parquet dataset.
    Randomly samples to ensure diverse lengths and sources.
    """
    print(f"📂 Loading mixed prompts from {parquet_path} ...")
    df = pd.read_parquet(parquet_path)
    print(f"   Total available: {len(df)}")

    # random sample of prompts
    n = min(n, len(df))
    sampled = df.sample(n=n, random_state=42).reset_index(drop=True)

    prompts = [(str(i), row["prompt"]) for i, row in sampled.iterrows()]
    lengths = sampled["p_tokens"]
    print(
        f"✅ Loaded {len(prompts)} prompts "
        f"(avg length ≈ {lengths.mean():.0f} tokens, "
        f"min={lengths.min()}, max={lengths.max()})"
    )
    return prompts


# ---------------------- REQUEST HANDLER ----------------------

def send_request_and_measure(openai_client, model_name, prompt):
    """Send a single completion request and record TTFT, TPOT, latency."""
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
    ttft = (first_token_time - start) if first_token_time else 0
    latency = end - start
    tpot = (latency - ttft) / max(total_tokens, 1) if ttft > 0 else 0

    return {
        "ttft_s": ttft,
        "tpot_s_per_token": tpot,
        "latency_s": latency,
        "d_tokens": total_tokens,
    }


# ---------------------- WORKER FUNCTION ----------------------

def handle_request(
    prompt_id,
    prompt,
    model_name,
    gpu_id,
    client,
    args,
    writer_lock,
    pattern_type,
    arrival_rate,
):
    """Single threaded worker that sends one request and logs result."""

    p_tokens = len(prompt.split())

    hw = model_metrics.get(model_name, {})
    metrics_snapshot = {
        "running_req_count": hw.get("num_requests_running", 0),
        "waiting_req_count": hw.get("num_requests_waiting", 0),
        "kv_cache_usage_perc": hw.get("kv_cache_usage_perc", 0),
        "ttft_avg": hw.get("ttft_avg", 0),
        "itl_avg": hw.get("itl_avg", 0),
        "e2e_avg": hw.get("e2e_avg", 0),
    }

    try:
        latency_info = send_request_and_measure(client, model_name, prompt)
    except Exception as e:
        print(f"[ERROR] {model_name} on GPU {gpu_id}: {repr(e)}")
        return

    row = {
        "request_id": str(uuid.uuid4()),
        "timestamp": datetime.datetime.now().isoformat(),
        "prompt_id": prompt_id,
        "model_id": model_name,
        "gpu_id": gpu_id,
        "p_tokens": p_tokens,
        "d_tokens": latency_info["d_tokens"],
        **metrics_snapshot,
        **latency_info,
        "pattern_type": pattern_type,
        "arrival_rate": arrival_rate,
    }

    with writer_lock:
        file_exists = os.path.exists(args.output)
        with open(args.output, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"[{prompt_id}] {model_name} latency={latency_info['latency_s']:.3f}s")


# ---------------------- MAIN ----------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect hardware-aware latency dataset from vLLM."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to GPU-Model YAML map.",
    )
    parser.add_argument(
        "--output",
        default="data/h100_full_sweep.csv",
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=20,
        help="Number of prompts to sample and send.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.3,
        help="Metric watcher polling interval (seconds).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Maximum concurrent requests (threads).",
    )
    parser.add_argument(
        "--prompt_path",
        default="data/prompts/mixed_prompts_train.parquet",
        help="Path to combined prompt dataset (Parquet).",
    )
    parser.add_argument(
        "--pattern",
        default="poisson",
        choices=["poisson", "microburst", "sustained"],
        help="Request arrival pattern type.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=5.0,
        help="Base arrival rate (requests per second).",
    )
    args = parser.parse_args()

    # --- Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    clients, model_to_gpu, model_url_map = {}, {}, {}
    for g, models in cfg["gpus"].items():
        g_id = int(g)
        for m in models:
            name = m["name"]
            base_root = m["base_url"].rstrip("/")
            openai_base = f"{base_root}/v1"
            clients[name] = OpenAI(base_url=openai_base, api_key="EMPTY")
            model_to_gpu[name] = g_id
            model_url_map[name] = f"{base_root}/metrics"

    print("Starting metrics watcher...")
    start_metrics_watcher(model_url_map, interval=args.interval)

    print(f"Initialized {len(clients)} OpenAI clients:")
    for name, client in clients.items():
        base = getattr(client._client, "_base_url", "<custom>")
        print(f"  - {name} → {base}")

    # --- Load prompts from local parquet
    prompts = load_local_prompts(args.prompt_path, args.num_prompts)

    # --- Threaded execution
    threads, writer_lock = [], threading.Lock()
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    model_names = list(clients.keys())
    pattern = RequestPattern(args.pattern, args.rate)

    for prompt_id, prompt in prompts:
        model_name = random.choice(model_names)
        client = clients[model_name]
        gpu_id = model_to_gpu[model_name]

        # Wait until concurrency limit allows a new request
        while len([th for th in threads if th.is_alive()]) >= args.concurrency:
            time.sleep(0.05)

        # Launch request thread
        t = threading.Thread(
            target=handle_request,
            args=(
                prompt_id,
                prompt,
                model_name,
                gpu_id,
                client,
                args,
                writer_lock,
                args.pattern,
                args.rate,
            ),
        )
        t.start()
        threads.append(t)

        # Delay until next request arrival (based on pattern)
        time.sleep(pattern.next_delay())

    # Wait for all threads to finish
    for th in threads:
        th.join()

    print(f"\n✅ Data collection complete → {args.output}")


if __name__ == "__main__":
    main()
