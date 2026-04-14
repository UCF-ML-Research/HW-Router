# Running HW-Router on Your Own Hardware

This guide walks you through running HW-Router end-to-end on your own GPU(s) and LLM pool.

**What you need:** One or more NVIDIA GPUs and a set of HuggingFace models you want to serve.

---

## Step 1: Write Your LLM Pool Config

Create a YAML file that maps your GPUs to models. Each model gets a unique port.

```yaml
# configs/my_setup.yaml
gpus:
  "0":                                    # GPU 0
    - name: my-model-a                    # Short name (you pick this)
      base_url: http://localhost:8010
    - name: my-model-b
      base_url: http://localhost:8011
  "1":                                    # GPU 1
    - name: my-model-c
      base_url: http://localhost:8012
```

The `name` field is a short identifier you choose. It must match the folder name you use when launching vLLM (see Step 2). For example, if you download a model to `~/models/my-model-a`, the name should be `my-model-a`.

**Important:** You also need to register your models in `hw_router/model_registry.py`:

```python
_KNOWN_MODELS = {
    "my-model-a":  (0, "ModelA-7B-Instruct"),    # (integer_id, HuggingFace display name)
    "my-model-b":  (1, "ModelB-3B-Chat"),
    "my-model-c":  (2, "ModelC-14B-Instruct"),
}
```

And add pricing/quality entries in `hw_router/constants.py` for each model's HuggingFace display name.

---

## Step 2: Launch Your LLM Pool with vLLM

Install vLLM and download your models:

```bash
pip install vllm
# Download models (example)
huggingface-cli download Qwen/Qwen2.5-14B-Instruct --local-dir ~/models/my-model-a
```

Launch each model as a vLLM server. The ports must match your config YAML:

```bash
mkdir -p logs

# GPU 0 — Model A
CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model ~/models/my-model-a \
  --port 8010 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 32768 \
  > logs/model_a.log 2>&1 &

sleep 120  # Wait for model to load before starting the next one

# GPU 0 — Model B (sharing GPU 0)
CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model ~/models/my-model-b \
  --port 8011 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.3 \
  --max-model-len 32768 \
  > logs/model_b.log 2>&1 &

sleep 120

# GPU 1 — Model C
CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
  --model ~/models/my-model-c \
  --port 8012 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 32768 \
  > logs/model_c.log 2>&1 &
```

**Key flags:**
- `--gpu-memory-utilization`: Fraction of GPU memory for this model. When sharing a GPU, the fractions must sum to < 1.0.
- `--max-model-len`: Maximum context length. Lower this if you run out of memory.
- The sleep between launches gives each model time to load weights before the next one starts.

**Verify all servers are running:**

```bash
# Check each server responds
curl http://localhost:8010/v1/models
curl http://localhost:8011/v1/models
curl http://localhost:8012/v1/models

# Check GPU usage
nvidia-smi
```

---

## Step 3: Collect Hardware Data

First, prepare the prompt dataset (downloads from public HuggingFace datasets):

```bash
python pipeline/data_preparation/load_mixinstruct.py
python pipeline/data_preparation/load_longbench.py
python pipeline/data_preparation/combine_datasets.py
```

Then collect hardware metrics + latency data under different load patterns:

```bash
# Collect with Poisson arrivals
python pipeline/data_collection/build_hardware_cost_dataset.py \
    --config configs/my_setup.yaml \
    --output data/my_hardware_data.csv \
    --pattern poisson --rate 5.0 \
    --num_prompts 500

# Collect with microburst arrivals
python pipeline/data_collection/build_hardware_cost_dataset.py \
    --config configs/my_setup.yaml \
    --output data/my_hardware_data.csv \
    --pattern microburst --rate 5.0 \
    --num_prompts 500

# Collect with sustained load
python pipeline/data_collection/build_hardware_cost_dataset.py \
    --config configs/my_setup.yaml \
    --output data/my_hardware_data.csv \
    --pattern sustained --rate 5.0 \
    --num_prompts 500
```

This sends prompts to your vLLM servers and records per-request TTFT, TPOT, and real-time hardware metrics (running requests, waiting requests, KV-cache usage). The output CSV is appended across runs, so you end up with data across all load patterns.

**Tip:** More data = better cost predictor. We collected ~15,000 samples across all patterns.

---

## Step 4: Train the Cost Predictor

```bash
python -m pipeline.training.train_cost_model \
    --data data/my_hardware_data.csv \
    --output-dir checkpoints/hardware_cost_model \
    --epochs 50
```

This trains a lightweight MLP (~20 seconds on CPU) that predicts TTFT and TPOT from hardware state. Output: `checkpoints/hardware_cost_model/model.pt` and `preproc.joblib`.

---

## Step 5: Run Inference — With vs Without Cost Predictor

Now compare routing **with** the hardware cost predictor (HW-Router) vs **without** it (CARROT baseline, which uses static dollar cost):

```bash
# First, build the evaluation dataset
python pipeline/data_collection/build_eval_dataset.py \
    --config configs/my_setup.yaml \
    --output data/my_eval_dataset.csv

python pipeline/eval_processing/process_eval_dataset.py \
    --input data/my_eval_dataset.csv \
    --output data/my_eval_processed.csv

# Run with CARROT (static cost — no cost predictor)
python pipeline/evaluation/eval_runtime_router.py \
    --config configs/my_setup.yaml \
    --prompt_path data/prompts/mixed_prompts_eval.parquet \
    --eval_csv data/my_eval_processed.csv \
    --router carrot \
    --arrival_rate 18.0

# Run with HW-Router (hardware-aware cost predictor)
python pipeline/evaluation/eval_runtime_router.py \
    --config configs/my_setup.yaml \
    --prompt_path data/prompts/mixed_prompts_eval.parquet \
    --eval_csv data/my_eval_processed.csv \
    --router hw \
    --arrival_rate 18.0
```

Results are saved to `data/router_eval_runs/`. Compare the two runs to see HW-Router's latency improvement and SLO attainment gains on your hardware.

---

## GPU Memory Guide

When sharing a GPU between models, the `--gpu-memory-utilization` fractions must sum to less than 1.0. Here are rough guidelines:

| Model Size | Suggested `--gpu-memory-utilization` |
|------------|--------------------------------------|
| 3B params  | 0.15 - 0.25                          |
| 7-8B params | 0.25 - 0.40                         |
| 14B params | 0.45 - 0.60                          |

If a model fails to load, check `logs/*.log` for OOM errors and reduce `--gpu-memory-utilization` or `--max-model-len`.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `curl` to a port returns "Connection refused" | Model is still loading. Check `logs/*.log` and wait. |
| OOM during vLLM launch | Lower `--gpu-memory-utilization` or `--max-model-len`. |
| `KeyError: Unknown model name` | Register your model names in `hw_router/model_registry.py`. |
| Empty or missing metrics | Ensure vLLM servers expose `/metrics` endpoint (default behavior). |
