"""
load_longbench_balanced.py
Builds a LongBench dataset with mixed prompt lengths (short→long)
using all English subsets only — larger sample set.
"""

import os
import random
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np

# ---------------- CONFIG -----------------
OUT_PATH = "data/longbench_prompts.parquet"
ALL_SUBSETS = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa",
    "2wikimqa", "musique", "gov_report", "qmsum",
    "multi_news", "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en", "lcc", "repobench-p"
]
MAX_TOKENS = 20000
TARGET_SAMPLES = 8000       # ⬅️ doubled target
TOKENIZER_NAME = "microsoft/Phi-3-mini-128k-instruct"
RANDOM_SEED = 42
# ------------------------------------------

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def get_token_len(text: str) -> int:
    """Return token length without truncation."""
    toks = tokenizer.encode(text, add_special_tokens=False)
    return len(toks)


def make_prompt(row):
    inp = (row.get("input") or "").strip()
    ctx = (row.get("context") or "").strip()
    return (inp + "\n\n" + ctx).strip()


def load_longbench_balanced():
    all_prompts = []
    for subset in ALL_SUBSETS:
        print(f"📂 Loading subset: {subset}")
        try:
            ds = load_dataset("THUDM/LongBench", subset, split="test")
        except Exception as e:
            print(f"⚠️ Skipping {subset}: {e}")
            continue

        for row in tqdm(ds, desc=subset):
            lang = (row.get("language") or "").lower()
            if "en" not in lang:  # skip non-English
                continue
            text = make_prompt(row)
            if not text:
                continue
            tok_len = get_token_len(text)
            if tok_len == 0:
                continue
            all_prompts.append({
                "id": f"{subset}-{row.get('_id', len(all_prompts))}",
                "source": subset,
                "prompt": text,
                "p_tokens": tok_len
            })

    df = pd.DataFrame(all_prompts)
    print(f"Total collected: {len(df)} samples before balancing")

    # -------- Balanced sampling by length bins (looser) --------
    bins = [0, 200, 1000, 3000, 6000, 10000, MAX_TOKENS]
    df["len_bin"] = pd.cut(df["p_tokens"], bins=bins, labels=False, include_lowest=True)

    samples_per_bin = (TARGET_SAMPLES // len(bins)) * 2  # ⬅️ allow double-per-bin
    balanced_rows = []
    for b in sorted(df["len_bin"].dropna().unique()):
        subset_bin = df[df["len_bin"] == b]
        take = min(samples_per_bin, len(subset_bin))
        if take > 0:
            balanced_rows.append(subset_bin.sample(take, random_state=RANDOM_SEED))
        print(f"Bin {b}: {len(subset_bin)} available → taking {take}")

    df_balanced = pd.concat(balanced_rows, ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # -------- Save --------
    df_balanced.to_parquet(OUT_PATH, index=False)
    print(f"✅ Saved {len(df_balanced)} balanced prompts → {OUT_PATH}")
    print(df_balanced["p_tokens"].describe(percentiles=[0.1,0.25,0.5,0.75,0.9]))
    return df_balanced


if __name__ == "__main__":
    df = load_longbench_balanced()
