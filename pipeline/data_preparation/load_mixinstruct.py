"""
load_mixinstruct.py
Load MixInstruct prompts and save as Parquet.
"""

import os
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

OUT_PATH = "data/mixinstruct_prompts.parquet"
TOKENIZER_NAME = "microsoft/Phi-3-mini-128k-instruct"
MAX_TOKENS = 10_000
SAMPLE_COUNT = 6000

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def truncate_prompt(text):
    toks = tokenizer.encode(text, add_special_tokens=False)
    if len(toks) > MAX_TOKENS:
        toks = toks[:MAX_TOKENS]
    return tokenizer.decode(toks, skip_special_tokens=True), len(toks)


def load_mixinstruct_prompts(n=SAMPLE_COUNT):
    print("Loading Mix-Instruct (llm-blender/mix-instruct, split='train')...")
    ds = load_dataset("llm-blender/mix-instruct", split="train")

    def make_prompt(x):
        inp = (x["input"] or "").strip()
        instr = (x["instruction"] or "").strip()
        return {"prompt": f"{instr} {inp}".strip()}

    ds = ds.map(make_prompt)
    sampled = ds.select(range(min(n, len(ds))))

    prompts = []
    for row in tqdm(sampled, desc="Processing MixInstruct"):
        text, tok_len = truncate_prompt(row["prompt"])
        prompts.append({
            "id": str(row["id"]),
            "source": "mix_instruct",
            "prompt": text,
            "p_tokens": tok_len,
        })

    df = pd.DataFrame(prompts)
    df.to_parquet(OUT_PATH, index=False)
    print(f"✅ Saved {len(df)} MixInstruct prompts → {OUT_PATH}")
    return df


if __name__ == "__main__":
    df = load_mixinstruct_prompts()
    print(df.head())
