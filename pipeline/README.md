# HW-Router Pipeline — Detailed Walkthrough

This directory contains every stage needed to train HW-Router on your own models and hardware, and to evaluate it under live or simulated traffic. The main [README](../README.md) has a compact overview — this document is the long form with full commands, inputs/outputs, and tips for each stage.

```
pipeline/
├── data_preparation/   # Build a prompt dataset from public sources (or plug in your own)
├── data_collection/    # Collect hardware telemetry from your vLLM servers
├── training/           # Train the MLP cost predictor
├── eval_processing/    # Attach quality + cost predictions to an eval set
└── evaluation/         # Offline λ-sweep and online routing experiments
```

## Prerequisites

- **Python 3.9+** with dependencies installed: `pip install -e ".[all]"` from the repository root.
- **vLLM servers** for the models you want to route across — required for hardware telemetry collection and online evaluation.
- **GPUs** — we trained and evaluated on 2× NVIDIA H100, but any GPU pool works as long as vLLM can serve the models. See [../docs/CUSTOM_HARDWARE_GUIDE.md](../docs/CUSTOM_HARDWARE_GUIDE.md) for adapting to a different setup.
- **Model pool config** — a YAML mapping GPUs to models. See [../configs/gpu_model_map_h100.yaml](../configs/gpu_model_map_h100.yaml) for an example.

Data preparation, cost-model training, and the offline λ-sweep all run **CPU-only**. Only hardware telemetry collection and online evaluation need live GPUs.

---

## 1. Prepare a Prompt Dataset

You can either build the mix we use in the paper (balanced MixInstruct + LongBench) or drop in your own prompts as a parquet with columns `id`, `source`, `prompt`, `p_tokens`.

```bash
python pipeline/data_preparation/load_mixinstruct.py
python pipeline/data_preparation/load_longbench.py
python pipeline/data_preparation/combine_datasets.py

# Cache prompt embeddings once (used by CARROT and eval processing)
python pipeline/data_preparation/save_prompt_embeddings.py \
    --input  data/prompts/mixed_prompts_train.parquet \
    --output data/prompts/mixed_prompts_train_with_prompt_embeddings.parquet
python pipeline/data_preparation/save_prompt_embeddings.py \
    --input  data/prompts/mixed_prompts_eval.parquet \
    --output data/prompts/mixed_prompts_eval_with_prompt_embeddings.parquet
```

**Output:** `data/prompts/mixed_prompts_{train,eval}.parquet` and their embedding variants.

**Notes:**
- `combine_datasets.py` produces a stratified train/eval split (default 80/20) balanced across prompt-length bins.
- Embeddings use the encoder specified in [hw_router/constants.py](../hw_router/constants.py); change it there if you want a different backbone.

---

## 2. Collect Hardware Telemetry

Launch the vLLM servers defined in your config, then drive realistic traffic through them while recording live hardware state (queue depths, KV-cache utilization, observed TTFT/TPOT). This is the training signal for the cost predictor.

```bash
python pipeline/data_collection/build_hardware_cost_dataset.py \
    --config configs/gpu_model_map_h100.yaml \
    --output data/h100_full_sweep.csv \
    --pattern poisson --rate 5.0

# Normalization constants the router uses at inference time
python pipeline/data_collection/compute_normalization.py
```

**Output:** `data/h100_full_sweep.csv` — one row per dispatched request with features `(p_tokens, model_id, gpu_id, running_req_count, waiting_req_count, kv_cache_usage_perc, ttft_avg, itl_avg, e2e_avg, …)` and labels `(ttft_s, tpot_s_per_token, latency_s)`.

**Tips:**
- Mix arrival patterns (`poisson`, `bursty`, `sustained`) to avoid overfitting the predictor to a single load regime.
- A few thousand rows per model is enough to train well. The shipped `h100_full_sweep.csv` has ~50k rows across 5 models.
- `compute_normalization.py` writes scaling constants back into [hw_router/constants.py](../hw_router/constants.py) — rerun it any time your hardware pool or model set changes.

---

## 3. Train the Cost Predictor

A lightweight MLP that maps `(prompt tokens, live GPU state) → (TTFT, TPOT)` per model. Runs in ~20 seconds on CPU.

```bash
python -m pipeline.training.train_cost_model
# Force CPU: python -m pipeline.training.train_cost_model --cpu
# Custom training CSV: python -m pipeline.training.train_cost_model --data path/to/your.csv
```

**Output:** `checkpoints/hardware_cost_model/{model.pt, preproc.joblib}` — loaded at inference time by `hw_router.cost_predictor`.

---

## 4. Build an Evaluation Set (Optional)

If you want to compare routers offline before deploying, attach each router's predictions to an eval CSV so you can score different λ values without re-running traffic.

```bash
# Collect ground-truth latency for the eval prompts
python pipeline/data_collection/build_eval_dataset.py \
    --config configs/gpu_model_map_h100.yaml \
    --output data/evaluation_dataset.csv

# Attach CARROT predictions + hardware cost predictions
python pipeline/eval_processing/process_eval_dataset.py \
    --input  data/evaluation_dataset.csv \
    --output data/evaluation_dataset_processed_full.csv

# Attach UMR and IRT quality scores
python pipeline/eval_processing/update_eval_with_umr.py
python pipeline/eval_processing/update_eval_with_irt.py
```

**Output:** `data/evaluation_dataset_processed_full_with_umr_irt.csv` — one row per `(prompt, candidate model)` pair with every router's predicted quality and predicted cost, plus the ground-truth latency and judge score. This is the format the offline λ-sweep consumes.

> A pre-computed version using our 5-model H100 pool is shipped at [../data/evaluation_dataset_processed_full_with_umr_irt.csv](../data/evaluation_dataset_processed_full_with_umr_irt.csv). You can use it directly to sanity-check the offline sweep without running Steps 1–2 yourself.

---

## 5. Evaluate Routers

### Offline λ-sweep (CPU only)

Score each router at a range of λ values (the quality–cost trade-off knob) against the eval CSV. Useful for picking a λ that hits your SLO target.

```bash
python pipeline/evaluation/eval_lambda_sweep.py \
    --input data/evaluation_dataset_processed_full_with_umr_irt.csv
```

### Online evaluation (requires live vLLM servers)

Dispatch real requests through a live router and measure end-to-end latency, SLO attainment, and GPU load skew.

```bash
# Single router at a fixed arrival rate
python pipeline/evaluation/eval_runtime_router.py \
    --config      configs/gpu_model_map_h100.yaml \
    --prompt_path data/prompts/mixed_prompts_eval.parquet \
    --eval_csv    data/evaluation_dataset_processed_full_with_umr_irt.csv \
    --router      hw

# Sweep over arrival rates
python pipeline/evaluation/eval_realtime_sweep.py \
    --router hw \
    --arrival_rates "15,18,21" \
    --pattern_type sustained
```

`--router` accepts `carrot` (static cost) or `hw` (hardware-aware). For `baseline`, `irt`, `umr`, or your own subclass, extend `eval_runtime_router.py`.

---

## Retraining the Baselines (Optional)

Pre-trained IRT and UMR artifacts ship under [../baselines/](../baselines/). If you change your model pool or want to retrain from scratch, use the judge scores under [../data/data_quality/](../data/data_quality/):

```bash
python pipeline/data_preparation/build_umr_training_csv.py

python baselines/irt/train_irt.py train \
    --data-path data/UMR_router_training_data.csv \
    --checkpoint baselines/irt/mirt_hw.snapshot

python baselines/umr/umr_router.py train \
    --train_csv data/UMR_router_training_data.csv \
    --work_dir  checkpoints/umr
```
