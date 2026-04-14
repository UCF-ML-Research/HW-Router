"""
process_eval_dataset.py
- Deduplicate (prompt_id, model_id)
- Compute:
    carrot_predicted_quality
    carrot_predicted_cost
    carrot_predicted_length
    predicted_ttft
    predicted_tpot
"""

import pandas as pd
import argparse

from baselines.carrot import load_carrot_router
from hw_router.routers import CarrotRouter
from hw_router.cost_predictor import HardwareCostPredictor
from hw_router.model_registry import (
    get_model_id,
    get_model_hugging_face_name
)

def process_csv(input_csv, output_csv, emb_parquet,
                carrot_ckpt="checkpoints/carrot",
                hw_model_path="checkpoints/hardware_cost_model/model.pt",
                hw_preproc_path="checkpoints/hardware_cost_model/preproc.joblib"):

    print(f"[Process] Loading raw CSV: {input_csv}")
    df = pd.read_csv(input_csv)

    # -----------------------------
    # 1. Deduplicate
    # -----------------------------
    df = df.sample(frac=1, random_state=42)
    df = df.drop_duplicates(
        subset=["prompt_id", "model_id"],
        keep="first"
    )
    print(f"[Process] After dedup: {len(df)} rows")

    # -----------------------------
    # 2. Load precomputed embeddings
    # -----------------------------
    emb_df = pd.read_parquet(emb_parquet)
    emb_map = {str(r["prompt_id"]): r["carrot_emb"] for _, r in emb_df.iterrows()}

    # -----------------------------
    # 3. Map model_id → HF name
    # -----------------------------
    df["model_hf"] = df["model_id"].map(get_model_hugging_face_name)

    # -----------------------------
    # 4. Load Carrot Router
    # -----------------------------
    carrot_raw = load_carrot_router(carrot_ckpt, model_type="linear")
    carrot_router = CarrotRouter(carrot_raw)

    # -----------------------------
    # 5. Load Hardware Cost Predictor
    # -----------------------------
    hw_predictor = HardwareCostPredictor(hw_model_path, hw_preproc_path)

    # Lists to accumulate results
    carrot_q_list = []
    carrot_c_list = []
    carrot_len_list = []
    pred_ttft_list = []
    pred_tpot_list = []

    # -----------------------------
    # 6. Main compute loop
    # -----------------------------
    for _, row in df.iterrows():
        pid = str(row["prompt_id"])

        # Retrieve embedding
        if pid not in emb_map:
            raise ValueError(f"Missing embedding for prompt_id={pid}")

        emb = emb_map[pid]

        hf_name = row["model_hf"]
        model_id_int = get_model_id(row["model_id"])

        # ---------------------------------------------
        # Carrot (embedding version)
        # ---------------------------------------------

        if emb is None:
            raise ValueError(f"Missing embedding")

        q, c = carrot_router.compute_from_embedding(hf_name, emb)
        pred_len = int(carrot_router.length_predictor(model_name=hf_name, emb=emb))


        # ---------------------------------------------
        # Hardware features
        # ---------------------------------------------
        feat = {
            "p_tokens": row["p_tokens"],
            "running_req_count": row["running_req_count"],
            "waiting_req_count": row["waiting_req_count"],
            "kv_cache_usage_perc": row["kv_cache_usage_perc"],
            "ttft_avg": row["ttft_avg"],
            "itl_avg": row["itl_avg"],
            "model_id": model_id_int,
            "gpu_id": str(row["gpu_id"]),
        }

        # ---------------------------------------------
        # Hardware cost predictor
        # ---------------------------------------------
        pred_ttft, pred_tpot = hw_predictor(model_id_int, feat)

        carrot_q_list.append(q)
        carrot_c_list.append(c)
        carrot_len_list.append(pred_len)
        pred_ttft_list.append(pred_ttft)
        pred_tpot_list.append(pred_tpot)

    # -----------------------------
    # 7. Save new columns
    # -----------------------------
    df["carrot_predicted_quality"] = carrot_q_list
    df["carrot_predicted_cost"] = carrot_c_list
    df["carrot_predicted_length"] = carrot_len_list
    df["predicted_ttft"] = pred_ttft_list
    df["predicted_tpot"] = pred_tpot_list



    # -----------------------------------------------------------
    # 6. Add judge-based actual quality scores for each (prompt, model)
    # -----------------------------------------------------------

    print("[Process] Loading prompt parquet for source IDs...")
    prompt_df = pd.read_parquet("data/prompts/mixed_prompts_eval.parquet")
    prompt_df = prompt_df.reset_index().rename(columns={"index": "pid"})
    prompt_id_map = {str(row["pid"]): row["id"] for _, row in prompt_df.iterrows()}

    df["prompt_source_id"] = df["prompt_id"].astype(str).map(prompt_id_map)

    # -----------------------------------------------------------
    # Model → scoring CSV mapping
    # -----------------------------------------------------------
    MODEL_SCORE_FILES = {
        "Llama-3.1-8B-Instruct": "data_quality/eval/Llama-3.1-8B-Instruct_eval_scored.csv",
        "Mistral-7B-Instruct-v0.3": "data_quality/eval/Mistral-7B-Instruct-v0.3_eval_scored.csv",
        "Phi-3-mini-128k-instruct": "data_quality/eval/Phi-3-mini-128k-instruct_eval_scored.csv",
        "Qwen2.5-3B-Instruct": "data_quality/eval/Qwen2.5-3B-Instruct_eval_scored.csv",
        "Qwen2.5-14B-Instruct": "data_quality/eval/Qwen2.5-14B-Instruct_eval_scored.csv",
    }

    print("[Process] Loading model score CSVs...")
    score_maps = {}

    for model_name, score_file in MODEL_SCORE_FILES.items():
        score_df = pd.read_csv(score_file)
        # id → judge_score
        score_maps[model_name] = {
            str(row["id"]): row["judge_score"]
            for _, row in score_df.iterrows()
        }

    # -----------------------------------------------------------
    # Lookup judge score for each row
    # -----------------------------------------------------------
    actual_scores = []

    for _, row in df.iterrows():
        model = row["model_hf"]
        source_id = row["prompt_source_id"]

        model_scores = score_maps.get(model, {})
        score = model_scores.get(str(source_id), None)  # may be missing

        actual_scores.append(score)

    df["actual_quality_score"] = actual_scores



    # Sort (prompt_id, model_id)
    df["model_id_int"] = df["model_id"].map(get_model_id)
    df = df.sort_values(by=["prompt_id", "model_id_int"]).reset_index(drop=True)

    # Final save
    df.to_csv(output_csv, index=False)
    print(f"[Process] Saved → {output_csv}")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", default="data/evaluation_dataset_full.csv")
    parser.add_argument("--output", default="data/evaluation_dataset_processed_full.csv")
    parser.add_argument("--emb", default="data/prompts/mixed_prompts_eval_prompt_embeddings.parquet")

    parser.add_argument("--carrot_ckpt", default="checkpoints/carrot")
    parser.add_argument("--hw_model", default="checkpoints/hardware_cost_model/model.pt")
    parser.add_argument("--hw_preproc", default="checkpoints/hardware_cost_model/preproc.joblib")

    args = parser.parse_args()

    process_csv(
        args.input,
        args.output,
        args.emb,
        carrot_ckpt=args.carrot_ckpt,
        hw_model_path=args.hw_model,
        hw_preproc_path=args.hw_preproc,
    )
