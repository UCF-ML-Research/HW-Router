"""
combine_datasets.py
Combine MixInstruct and LongBench datasets,
shuffle them, and save balanced train/eval splits.
"""

import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# ---------------- CONFIG ----------------
MIXINSTRUCT_PATH = "data/mixinstruct_prompts.parquet"
LONGBENCH_PATH = "data/longbench_prompts.parquet"
OUT_DIR = "data/prompts"
TRAIN_SPLIT_PATH = os.path.join(OUT_DIR, "mixed_prompts_train.parquet")
EVAL_SPLIT_PATH = os.path.join(OUT_DIR, "mixed_prompts_eval.parquet")
RANDOM_SEED = 42
TEST_SIZE = 0.2
# ----------------------------------------

def combine_and_split():
    print("📂 Loading MixInstruct dataset ...")
    mix_df = pd.read_parquet(MIXINSTRUCT_PATH)
    print(f"   ✅ Loaded {len(mix_df)} MixInstruct samples.")

    print("📂 Loading LongBench dataset ...")
    lb_df = pd.read_parquet(LONGBENCH_PATH)
    print(f"   ✅ Loaded {len(lb_df)} LongBench samples.")

    # --- Standardize columns ---
    required_cols = ["id", "source", "prompt", "p_tokens"]
    mix_df = mix_df[required_cols]
    lb_df = lb_df[required_cols]

    # --- Combine + shuffle ---
    combined_df = pd.concat([mix_df, lb_df], ignore_index=True)
    combined_df = shuffle(combined_df, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"🧩 Total combined samples: {len(combined_df)}")
    print(combined_df["source"].value_counts())

    # --- Create bins for stratification based on prompt length ---
    combined_df["len_bin"] = pd.qcut(
        np.log1p(combined_df["p_tokens"]),
        q=10,
        labels=False,
        duplicates="drop"
    )

    # --- Stratified train/eval split ---
    train_df, eval_df = train_test_split(
        combined_df,
        test_size=TEST_SIZE,
        stratify=combined_df["len_bin"],
        random_state=RANDOM_SEED,
    )

    # Drop helper column
    train_df = train_df.drop(columns=["len_bin"])
    eval_df = eval_df.drop(columns=["len_bin"])

    # --- Save splits ---
    os.makedirs(OUT_DIR, exist_ok=True)
    train_df.to_parquet(TRAIN_SPLIT_PATH, index=False)
    eval_df.to_parquet(EVAL_SPLIT_PATH, index=False)

    print(f"\n✅ Saved balanced splits:")
    print(f"   Train → {TRAIN_SPLIT_PATH} ({len(train_df)})")
    print(f"   Eval  → {EVAL_SPLIT_PATH} ({len(eval_df)})")
    print("\n📊 Composition check:")
    print("Train source distribution:")
    print(train_df["source"].value_counts(normalize=True).round(3))
    print("\nEval source distribution:")
    print(eval_df["source"].value_counts(normalize=True).round(3))

    return train_df, eval_df


if __name__ == "__main__":
    train_df, eval_df = combine_and_split()
    print("\n🧠 Train preview:")
    print(train_df.head())
