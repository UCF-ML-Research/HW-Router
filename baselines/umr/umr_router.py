#!/usr/bin/env python3
"""
UMR Router (Final Version, No d_tokens)
--------------------------------------

Training CSV format (produced by build_umr_training_csv.py):

    prompt_id, prompt, p_tokens,
    Llama-3.1-8B-Instruct_score,
    Mistral-7B-Instruct-v0.3_score,
    Phi-3-mini-128k-instruct_score,
    Qwen2.5-3B-Instruct_score,
    Qwen2.5-14B-Instruct_score

UMR TRAINING:
    - Threshold model scores at 0.5 -> binary labels.
    - Cluster prompts (KMeans).
    - Build per-cluster per-model error vectors.

INFERENCE:
    quality = 1 - cluster_error(model, cluster)
    cost    = price_per_token(model)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances_argmin

import torch
import torch.nn.functional as F

from hw_router.constants import MODEL_PRICES

# =====================================================================
# CONFIG — EDIT THESE
# =====================================================================

WORK_DIR = "checkpoints/umr"

TRAIN_CSV = "data/UMR_router_training_data.csv"

PROMPT_COL = "prompt"
P_TOK_COL = "p_tokens"

THRESHOLD = 0.65
NUMBER_OF_PROMPTS = 6000
K_CLUSTERS = 20

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cpu"

# =====================================================================
# ENCODER (final version)
# =====================================================================

from sentence_transformers import SentenceTransformer
import numpy as np

class QueryEncoder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2",
                 device="cpu"):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts, project=False):
        if isinstance(texts, str):
            texts = [texts]
        emb = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
        return emb.astype(np.float32)



# =====================================================================
# COST = price per token (no d_tokens)
# =====================================================================

def compute_cost(model_name: str) -> float:
    if model_name not in MODEL_PRICES:
        raise ValueError(f"Model {model_name} not found in MODEL_PRICES.")
    return MODEL_PRICES[model_name]


# =====================================================================
# BUILDER (TRAINING)
# =====================================================================

class UMRBuilder:
    def __init__(self, embed_model=EMBED_MODEL, device=DEVICE):
        self.encoder = QueryEncoder(embed_model, device=device)
        self.embed_model = embed_model
        self.device = device

    def _embed_batch(self, texts, batch_size=32) -> np.ndarray:
        chunks = []
        for i in range(0, len(texts), batch_size):
            chunks.append(self.encoder.encode(texts[i:i+batch_size]))
        return np.vstack(chunks)

    def build(self, csv_path: str, out_dir: str, k: int = K_CLUSTERS):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(csv_path)

        # Extract model score columns
        model_score_cols = [c for c in df.columns if c.endswith("_score")]
        model_names = [c.replace("_score", "") for c in model_score_cols]

        # Subsample if needed
        if NUMBER_OF_PROMPTS and len(df) > NUMBER_OF_PROMPTS:
            df = df.sample(NUMBER_OF_PROMPTS, random_state=42)

        # =============================
        # Split full rows, not just prompts
        # =============================
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

        train_prompts = train_df[PROMPT_COL].tolist()
        val_prompts   = val_df[PROMPT_COL].tolist()

        # Extract model names and their validation labels
        model_score_cols = [c for c in df.columns if c.endswith("_score")]
        model_names = [c.replace("_score", "") for c in model_score_cols]

        # Convert validation model scores -> binary using threshold
        val_labels = {
            model: (val_df[f"{model}_score"].values >= THRESHOLD).astype(float)
            for model in model_names
        }


        # 1. Cluster
        print("[UMR] Embedding training prompts...")
        train_emb = self._embed_batch(train_prompts)

        print("[UMR] Running KMeans clustering...")
        km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(train_emb)
        centroids = km.cluster_centers_

        json.dump({
            "centroids": centroids.tolist(),
            "embed_model": self.embed_model,
            "model_names": model_names
        }, open(out / "clusters.json", "w"))

        # 2. Assign val prompts to clusters
        print("[UMR] Embedding validation prompts...")
        val_emb = self._embed_batch(val_prompts)
        assign = pairwise_distances_argmin(val_emb, centroids, metric="euclidean")

        # 3. Build error table
        print("[UMR] Computing per-cluster model errors...")
        errors = {}

        assign = np.array(assign)
        for model in model_names:
            lbls = np.array(val_labels[model])
            err_vec = []
            for c in range(k):
                mask = (assign == c)
                if mask.any():
                    err_vec.append(float(1 - lbls[mask].mean()))
                else:
                    err_vec.append(0.5)  # neutral
            errors[model] = err_vec

        json.dump(errors, open(out / "errors.json", "w"))

        print(f"[UMR] Training completed. Saved artifacts to: {out.resolve()}")


# =====================================================================
# ROUTER (INFERENCE)
# =====================================================================

class UMRRouter:
    def __init__(self, work_dir=WORK_DIR, device=DEVICE):
        self.work_dir = Path(work_dir)
        self.device = device
        self._load()

    def _load(self):
        clusters = json.load(open(self.work_dir / "clusters.json"))
        centroids = np.asarray(clusters["centroids"], dtype=np.float32)
        self.centroids = torch.tensor(centroids, device=self.device)
        self.centroids = F.normalize(self.centroids, dim=-1)

        self.model_names = clusters["model_names"]
        self.encoder = QueryEncoder(clusters["embed_model"], device=self.device)

        self.errors = json.load(open(self.work_dir / "errors.json"))

    @torch.no_grad()
    def _cluster(self, prompt: str) -> int:
        emb = self.encoder.encode(prompt)
        v = torch.tensor(emb[0], dtype=torch.float32, device=self.device)
        v = F.normalize(v, dim=-1)
        sims = v @ self.centroids.T
        return int(torch.argmax(sims).item())

    def score(self, prompt: str, model_name: str) -> Tuple[float, float]:
        if model_name not in self.errors:
            raise ValueError(f"Model {model_name} was not trained.")

        cluster_id = self._cluster(prompt)
        err = self.errors[model_name][cluster_id]

        quality = 1.0 - err
        cost = compute_cost(model_name)

        return float(quality), float(cost)


# =====================================================================
# PUBLIC API
# =====================================================================

_UMR_CACHE = None

def umr_score(prompt: str, model_name: str) -> Tuple[float, float]:
    """
    Returns:
        quality : float  (0-1, higher = better)
        cost    : float  (price_per_token)
    """
    global _UMR_CACHE
    if _UMR_CACHE is None:
        _UMR_CACHE = UMRRouter()
    return _UMR_CACHE.score(prompt, model_name)


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", default=WORK_DIR,
                        help="Directory for UMR artifacts (default: %(default)s)")
    parser.add_argument("--train_csv", default=TRAIN_CSV,
                        help="Path to training CSV (default: %(default)s)")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("train", help="Train UMR from multi-model score CSV")

    p_score = sub.add_parser("score", help="Score a prompt-model pair")
    p_score.add_argument("--prompt", required=True)
    p_score.add_argument("--model", required=True)

    args = parser.parse_args()

    if args.cmd == "train":
        builder = UMRBuilder()
        builder.build(args.train_csv, args.work_dir)

    elif args.cmd == "score":
        router = UMRRouter(work_dir=args.work_dir)
        q, c = router.score(args.prompt, args.model)
        print({"quality": q, "cost": c})

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
