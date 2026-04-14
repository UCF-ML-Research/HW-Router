import pandas as pd

from baselines.umr.umr_router import umr_score

# ================================================================
# Paths
# ================================================================
EVAL_PARQUET = "data/prompts/mixed_prompts_eval.parquet"
EVAL_CSV = "data/evaluation_dataset_processed_full.csv"
OUTPUT_CSV = "data/evaluation_dataset_processed_full_with_umr.csv"

# ================================================================
# Load data
# ================================================================
print("[LOAD] Loading eval parquet...")
df_prompts = pd.read_parquet(EVAL_PARQUET)   # has columns: id, prompt, p_tokens

print("[LOAD] Loading eval CSV...")
df_eval = pd.read_csv(EVAL_CSV)

# Map from prompt_id_string → prompt text
prompt_lookup = dict(zip(df_prompts["id"], df_prompts["prompt"]))

# ================================================================
# Prepare new columns
# ================================================================
umr_quality = []
umr_cost = []

print("[INFO] Running UMR inference on each row...")

for idx, row in df_eval.iterrows():
    prompt_id = row["prompt_source_id"]   # example: unified_chip2/21055
    model_name = row["model_hf"]

    if prompt_id not in prompt_lookup:
        print(f"[WARN] prompt id {prompt_id} not found in eval parquet. Using empty string.")
        prompt_text = ""
    else:
        prompt_text = prompt_lookup[prompt_id]

    # UMR inference
    q, c = umr_score(prompt_text, model_name)

    umr_quality.append(q)
    umr_cost.append(c)

    if idx % 100 == 0:
        print(f"[INFO] processed {idx} rows...")

# ================================================================
# Add columns and save
# ================================================================
df_eval["umr_quality_score"] = umr_quality
df_eval["umr_cost_score"] = umr_cost

print("[SAVE] Writing:", OUTPUT_CSV)
df_eval.to_csv(OUTPUT_CSV, index=False)

print("[DONE] Updated eval CSV with UMR scores.")
