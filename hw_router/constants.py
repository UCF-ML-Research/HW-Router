"""
constants.py — Single source of truth for all shared constants.

Consolidates MODEL_PRICES, MODEL_QUALITY, and normalization constants
that were previously duplicated across routers.py, umr_router.py,
eval_runtime_router.py, eval_lambda_sweep.py, and others.
"""

# =====================================================================
# Full HuggingFace model names
# =====================================================================
HF_MODEL_NAMES = [
    "Qwen/Qwen2.5-14B-Instruct",
    "microsoft/Phi-3-mini-128k-instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

# Short names (without org prefix), used in eval CSVs and baselines
HF_SHORT_NAMES = [n.split("/")[-1] for n in HF_MODEL_NAMES]

# =====================================================================
# Price per output token (USD)
# =====================================================================
MODEL_PRICES = {
    "Qwen2.5-14B-Instruct":          0.22 / 1_000_000,
    "Phi-3-mini-128k-instruct":      0.10 / 1_000_000,
    "Llama-3.1-8B-Instruct":         0.03 / 1_000_000,
    "Qwen2.5-3B-Instruct":           0.05 / 1_000_000,
    "Mistral-7B-Instruct-v0.3":      0.20 / 1_000_000,
}

# Also index by full HF name (for IRTRouter / CarrotRouter compatibility)
MODEL_PRICES_FULL = {
    full: MODEL_PRICES[short]
    for full, short in zip(HF_MODEL_NAMES, HF_SHORT_NAMES)
}
MODEL_PRICES.update(MODEL_PRICES_FULL)

# =====================================================================
# Static quality proxies (for BaselineRouter only)
# =====================================================================
MODEL_QUALITY = {
    "Qwen/Qwen2.5-14B-Instruct":              0.14,
    "microsoft/Phi-3-mini-128k-instruct":      0.30,
    "meta-llama/Llama-3.1-8B-Instruct":        0.80,
    "Qwen/Qwen2.5-3B-Instruct":               0.30,
    "mistralai/Mistral-7B-Instruct-v0.3":      0.70,
}

# =====================================================================
# Cost normalization constants (computed from h100_full_sweep training data)
# =====================================================================
LAT_P95_LOG = 4.556350           # 95th percentile of log(latency)
STATIC_COST_P95 = 0.00010604    # 95th percentile of static cost (tokens * price)
STATIC_COST_P95_IRT = 0.000000220000
STATIC_COST_P95_UMR = 0.000000220000

# =====================================================================
# Default scoring parameter
# =====================================================================
DEFAULT_LAMBDA = 0.5  # quality-cost trade-off weight
