import pandas as pd
import numpy as np

from hw_router.constants import MODEL_PRICES
from hw_router.model_registry import get_model_hugging_face_name

# =========================================================
#   Local Model → HF Mapping (from your model_maps.py)
# =========================================================

LOCAL_MODEL_TO_HUGGINGFACE_NAME = {
    local: get_model_hugging_face_name(local)
    for local in [
        "qwen14b", "phi3-mini", "llama3-8b", "qwen3b", "mistral7b",
    ]
}

DATA_PATH = "data/h100_full_sweep.csv"

# =========================================================
#   Load Dataset
# =========================================================

print(f"Loading training data: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

required_cols = ["latency_s", "d_tokens", "model_id"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column '{col}'")

# =========================================================
#   Compute latency percentile constant (log-scaling)
# =========================================================

df["latency_s"] = df["latency_s"].astype(float).clip(lower=1e-6)
df["log_latency"] = np.log1p(df["latency_s"])

latency_p95_log = float(np.percentile(df["log_latency"], 95))

print(f"\nlatency_p95_log = {latency_p95_log:.6f}")

# =========================================================
#   Compute static CARROT cost percentile constant
# =========================================================

def compute_static_cost(row):
    model_local = row["model_id"]
    d = float(row["d_tokens"])

    hf = get_model_hugging_face_name(model_local)

    if hf not in MODEL_PRICES:
        raise ValueError(f"Missing price for model: {hf}")

    return d * MODEL_PRICES[hf]

df["static_cost"] = df.apply(compute_static_cost, axis=1)

static_cost_p95 = float(np.percentile(df["static_cost"], 95))

print(f"static_cost_p95 = {static_cost_p95:.12f}")



# =========================================================
#   Compute static IRT cost percentile constant
# =========================================================

def compute_static_cost_irt(row):
    model_local = row["model_id"]

    hf = get_model_hugging_face_name(model_local)

    if hf not in MODEL_PRICES:
        raise ValueError(f"Missing price for model: {hf}")

    return MODEL_PRICES[hf]

df["static_cost_irt"] = df.apply(compute_static_cost_irt, axis=1)

static_cost_p95_irt = float(np.percentile(df["static_cost_irt"], 95))

print(f"static_cost_p95_irt = {static_cost_p95_irt:.12f}")

# =========================================================
#   Summary
# =========================================================

print("\n=== FINAL NORMALIZATION CONSTANTS ===")
print(f"latency_p95_log = {latency_p95_log:.6f}")
print(f"static_cost_p95 = {static_cost_p95:.12f}")
print(f"static_cost_p95_irt = {static_cost_p95_irt:.12f}")
print("=====================================\n")
