import pandas as pd

from hw_router.routers import irt_score

# ================================================================
# Paths
# ================================================================
EVAL_PARQUET = "data/prompts/mixed_prompts_eval.parquet"
EVAL_CSV = "data/evaluation_dataset_processed_full_with_umr.csv"
OUTPUT_CSV = "data/evaluation_dataset_processed_full_with_umr_irt.csv"

# ================================================================
# Model name mapping (dataset name -> HF name used by IRT)
# ================================================================
MODEL_NAME_TO_HF = {
    "Qwen2.5-14B-Instruct": "Qwen/Qwen2.5-14B-Instruct",
    "Phi-3-mini-128k-instruct": "microsoft/Phi-3-mini-128k-instruct",
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
    "Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
}

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
irt_quality = []
irt_cost = []

print("[INFO] Running IRT inference on each row...")

for idx, row in df_eval.iterrows():
    prompt_id = row["prompt_source_id"]   # example: unified_chip2/21055
    model_name = row["model_hf"]
    model_name = MODEL_NAME_TO_HF.get(model_name, model_name)

    if prompt_id not in prompt_lookup:
        print(f"[WARN] prompt id {prompt_id} not found in eval parquet. Using empty string.")
        prompt_text = ""
    else:
        prompt_text = prompt_lookup[prompt_id]

    # IRT inference
    q, c = irt_score(prompt_text, model_name)

    irt_quality.append(q)
    irt_cost.append(c)

    if idx % 100 == 0:
        print(f"[INFO] processed {idx} rows...")

# ================================================================
# Add columns and save
# ================================================================
df_eval["irt_quality_score"] = irt_quality
df_eval["irt_cost_score"] = irt_cost

print("[SAVE] Writing:", OUTPUT_CSV)
df_eval.to_csv(OUTPUT_CSV, index=False)

print("[DONE] Updated eval CSV with IRT scores.")
