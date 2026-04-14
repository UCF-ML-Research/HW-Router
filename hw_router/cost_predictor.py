"""
cost_predictor.py — Hardware Cost Predictor (Paper Section 3.2).

Predicts TTFT (Time-to-First-Token) and TPOT (Time-Per-Output-Token)
using a lightweight neural network that takes query features, model/GPU
identifiers, and real-time hardware metrics as input.

Contains:
    - HardwareCostNet: The neural network architecture
    - HardwareCostPredictor: Inference wrapper (loads weights + preprocessor)
    - load_cost_model: Convenience loader
    - predict_ttft_tpot: Single-request inference function
"""

import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from joblib import load


# =====================================================================
# Neural Network Architecture
# =====================================================================

class HardwareCostNet(nn.Module):
    """
    3-layer shared MLP with two output heads:
        Input → Linear(→128) → GELU → Linear(→64) → GELU
        Heads: TTFT (64→1), TPOT (64→1)

    Targets are in log-space (log(TTFT), log(TPOT)).
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
        )
        self.ttft_head = nn.Linear(64, 1)
        self.tpot_head = nn.Linear(64, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.ttft_head(h), self.tpot_head(h)


# =====================================================================
# Inference Wrapper
# =====================================================================

class HardwareCostPredictor:
    """
    Hardware cost model predictor.

    Training features:
        Numerical:
            - p_tokens
            - running_req_count
            - waiting_req_count
            - kv_cache_usage_perc
            - ttft_avg
            - itl_avg

        Categorical:
            - model_gpu = "<model_id_int>_<gpu_id_str>"

    During prediction, always call:
        predictor(model_id_int, feat_dict)

    Where feat_dict contains:
        {
            "p_tokens": int,
            "running_req_count": int,
            "waiting_req_count": int,
            "kv_cache_usage_perc": float,
            "ttft_avg": float,
            "itl_avg": float,
            "model_id": int,     # must be int 0..4
            "gpu_id": str,       # "0", "1", ...
        }
    """

    def __init__(self, model_path: str, preproc_path: str):
        # Load the ColumnTransformer used during training
        self.preproc = load(preproc_path)

        # Build neural network with correct input dimension
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = len(self.preproc.get_feature_names_out())

        self.model = HardwareCostNet(input_dim).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def __call__(self, model_id: int, feat: dict):
        """
        Predict TTFT and TPOT (original scale, seconds).
        """
        df = self._prepare_df(model_id, feat)
        X = self.preproc.transform(df)

        with torch.no_grad():
            Xv = torch.tensor(X, dtype=torch.float32).to(self.device)
            ttft_log, tpot_log = self.model(Xv)

        # Back-transform from log-space
        ttft = float(np.exp(ttft_log.cpu().numpy().squeeze()))
        tpot = float(np.exp(tpot_log.cpu().numpy().squeeze()))

        return ttft, tpot

    def _prepare_df(self, model_id: int, feat: dict) -> pd.DataFrame:
        """Build single-row DataFrame with exact training columns."""
        data = {
            "p_tokens": feat["p_tokens"],
            "running_req_count": feat["running_req_count"],
            "waiting_req_count": feat["waiting_req_count"],
            "kv_cache_usage_perc": feat["kv_cache_usage_perc"],
            "ttft_avg": feat["ttft_avg"],
            "itl_avg": feat["itl_avg"],
            "model_gpu": f"{model_id}_{feat['gpu_id']}",
        }
        return pd.DataFrame([data])


# =====================================================================
# Convenience Functions
# =====================================================================

def load_cost_model(ckpt_dir="checkpoints/hardware_cost_model"):
    """Load trained cost model and its preprocessor for runtime inference."""
    preproc_path = os.path.join(ckpt_dir, "preproc.joblib")
    model_path = os.path.join(ckpt_dir, "model.pt")

    preproc = load(preproc_path)

    input_dim = len(preproc.get_feature_names_out())
    model = HardwareCostNet(input_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    return preproc, model, device


def predict_ttft_tpot(preproc, model, features_dict, device):
    """Predict TTFT & TPOT given a single request's hardware + prompt features."""
    model_gpu = f"{features_dict['model_id']}_{features_dict['gpu_id']}"
    df_in = pd.DataFrame([{
        "p_tokens": features_dict["p_tokens"],
        "running_req_count": features_dict["running_req_count"],
        "waiting_req_count": features_dict["waiting_req_count"],
        "kv_cache_usage_perc": features_dict["kv_cache_usage_perc"],
        "ttft_avg": features_dict["ttft_avg"],
        "itl_avg": features_dict["itl_avg"],
        "model_gpu": model_gpu,
    }])
    x = preproc.transform(df_in)
    x_t = torch.tensor(x, dtype=torch.float32).to(device)
    with torch.no_grad():
        ttft_hat, tpot_hat = model(x_t)
    return float(torch.exp(ttft_hat)), float(torch.exp(tpot_hat))
