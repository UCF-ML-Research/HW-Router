"""
Precompute CARROT embeddings for all prompts in the eval dataset.

Output:
    mixed_prompts_eval_prompt_embeddings.parquet

This file contains ONLY:
    - prompt_id
    - carrot_emb (list of floats)

Use this as a lookup table to avoid re-embedding prompts during evaluation.
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse
import torch

ENCODER_MODEL = "all-MiniLM-L6-v2"   # CARROT encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/prompts/mixed_prompts_eval.parquet")
    parser.add_argument("--output", default="data/prompts/mixed_prompts_eval_prompt_embeddings.parquet")
    args = parser.parse_args()

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------
    print(f"📂 Loading prompts: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"   Loaded {len(df)} prompts")

    # Must contain "prompt" and "prompt_id"
    if "prompt_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "prompt_id"})

    # -----------------------------------------------------
    # Load encoder
    # -----------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔌 Loading encoder: {ENCODER_MODEL} (device={device})")

    encoder = SentenceTransformer(ENCODER_MODEL, device=device)

    # -----------------------------------------------------
    # Encode prompts
    # -----------------------------------------------------
    print("⚡ Encoding full prompts (no truncation) ...")

    embeddings = []
    for p in tqdm(df["prompt"], desc="Encoding", ncols=80):
        emb = encoder.encode(p, convert_to_numpy=True)
        embeddings.append(emb.tolist())

    # -----------------------------------------------------
    # Build OUTPUT LOOKUP table
    # ONLY keep (prompt_id, carrot_emb)
    # -----------------------------------------------------
    out_df = pd.DataFrame({
        "prompt_id": df["prompt_id"].astype(str),
        "carrot_emb": embeddings
    })

    # -----------------------------------------------------
    # Save output
    # -----------------------------------------------------
    print(f"💾 Saving lookup table → {args.output}")
    out_df.to_parquet(args.output, index=False)

    # -----------------------------------------------------
    # Show sample
    # -----------------------------------------------------
    print("\n================ SAMPLE SAVED ROWS (5) ================")
    sample = out_df.head(5).copy()
    sample["carrot_emb"] = sample["carrot_emb"].apply(lambda x: f"[len={len(x)}]")
    print(sample.to_string(index=False))
    print("=======================================================\n")

    print("✅ Finished computing + saving embeddings.")


if __name__ == "__main__":
    main()
