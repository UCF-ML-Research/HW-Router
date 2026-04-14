#!/usr/bin/env python3
"""
Build UMR training data CSV.

Inputs:
    1. mixed_prompts_train.parquet
    2. Each model's judge_score CSV

Output:
    UMR_router_training_data.csv

Columns:
    prompt_id, prompt, p_tokens, modelA_score, modelB_score, ...
"""

import pandas as pd
from pathlib import Path

# =====================================================================
# CONFIG — EDIT THESE PATHS AS NEEDED
# =====================================================================

PROMPTS_PARQUET = "data/prompts/mixed_prompts_train.parquet"

MODEL_SCORE_PATHS = {
    "Llama-3.1-8B-Instruct":        "data/data_quality/Llama-3.1-8B-Instruct.csv",
    "Mistral-7B-Instruct-v0.3":     "data/data_quality/Mistral-7B-Instruct-v0.3.csv",
    "Phi-3-mini-128k-instruct":     "data/data_quality/Phi-3-mini-128k-instruct.csv",
    "Qwen2.5-3B-Instruct":          "data/data_quality/Qwen2.5-3B-Instruct.csv",
    "Qwen2.5-14B-Instruct":         "data/data_quality/Qwen2.5-14B-Instruct.csv",
}

OUTPUT_CSV = "data/UMR_router_training_data.csv"


# =====================================================================
# MAIN LOGIC
# =====================================================================

def main():
    print("[LOAD] Loading prompt parquet...")
    df_prompts = pd.read_parquet(PROMPTS_PARQUET)

    # Keep only necessary columns
    df = df_prompts[["id", "prompt", "p_tokens"]].copy()
    df.rename(columns={"id": "prompt_id"}, inplace=True)

    # Merge judge_score for every model
    for model_name, csv_path in MODEL_SCORE_PATHS.items():
        print(f"[LOAD] Loading scores for {model_name} ...")
        df_score = pd.read_csv(csv_path)

        df_score = df_score[["id", "judge_score"]].rename(
            columns={
                "id": "prompt_id",
                "judge_score": f"{model_name}_score"
            }
        )

        df = df.merge(df_score, on="prompt_id", how="inner")

    print(f"[INFO] Rows after merging all model scores = {len(df)}")

    # Save final multi-model score CSV
    print("[SAVE] Saving:", OUTPUT_CSV)
    df.to_csv(OUTPUT_CSV, index=False)

    print("[DONE] UMR training data built successfully")


if __name__ == "__main__":
    main()
