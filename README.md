# HW-Router: Hardware-Aware Routing for Scalable Multi-LLM Serving

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![DAC 2026](https://img.shields.io/badge/DAC-2026-green.svg)](https://dac.com)

> **Accepted at the 63rd Design Automation Conference (DAC), 2026**

## Overview

<p align="center">
  <img src="assets/overview.png" alt="HW-Router Overview" width="600">
</p>

HW-Router is a hardware-aware routing framework for multi-LLM serving that dynamically selects the best model for each incoming request based on both predicted response quality and real-time hardware conditions.

Unlike static routing approaches that ignore server load, HW-Router integrates a lightweight neural cost predictor that estimates per-request latency (TTFT and TPOT) from live hardware metrics (queue depths, KV-cache utilization, GPU load). Combined with an IRT-based quality predictor, this enables quality-cost trade-off decisions that respect Service Level Objectives (SLOs).

Across diverse workloads, HW-Router achieves **3.4–3.9× lower end-to-end latency**, **46–48 percentage points higher SLO attainment**, **6–8× lower GPU load skew**, and a **3.1–3.4× reduction in waiting-queue fraction** over state-of-the-art baselines (CARROT and IRT) — with only **~200 μs** of additional routing overhead and no loss in output quality.

## Architecture

![HW-Router Methodology](assets/methodology.png)

**Components:**

| Component | Module | Description |
|-----------|--------|-------------|
| Hardware Monitor | `hw_router.hardware_monitor` | Polls vLLM Prometheus metrics in real-time |
| Cost Predictor | `hw_router.cost_predictor` | Lightweight MLP predicting TTFT and TPOT — **plug-in component** |
| Quality Predictor | `hw_router.routers` | Any quality predictor: CARROT, IRT, UMR, or custom |
| Decision Maker | `pipeline/evaluation/` | Scores each model: S = (1−λ)·Q − λ·C, picks argmax |

### Modular Design

The hardware cost predictor is a **plug-in** — it works with any quality predictor by replacing the static price/token cost term with real-time hardware-aware latency predictions:

```
Quality-only router:   S = (1−λ) · Q(x)   − λ · static_price/token
Hardware-aware (+HW):  S = (1−λ) · Q(x)   − λ · C(x, h)       ← same Q, HW cost swapped in
                                                      ↑
                                             MLP predicts TTFT + TPOT
                                             from live hardware state h
```

This means **any quality predictor can be made hardware-aware** simply by pairing it with the cost predictor. In the paper we use IRT as the quality component because it yields the best results, but CARROT and UMR benefit equally from hardware cost awareness:

| Quality Predictor | Without HW Cost | With HW Cost (+) | SLO lift |
|-------------------|-----------------|------------------|----------|
| CARROT | 44.7% SLO, 43.9s | 96.1% SLO, 14.4s | +51pp |
| IRT | 42.2% SLO, 45.3s | 97.9% SLO, 12.9s | +56pp ⭐ |
| UMR | 37.3% SLO, 48.4s | 91.5% SLO, 16.7s | +54pp |

*Evaluated at λ = 0.5. IRT+HW is the configuration reported as "HW-Router" in the paper.*

## Quick Start

### Installation

```bash
git clone https://github.com/UCF-ML-Research/HW-Router.git
cd HW-Router
pip install -e .
```

To install with specific components only:

```bash
pip install -e ".[irt]"       # Core + IRT quality predictor
pip install -e ".[carrot]"    # Core + CARROT baseline
pip install -e ".[serving]"   # Core + vLLM serving stack
pip install -e ".[all]"       # Everything
```

### Using the Routers

All routers share the same interface — `compute(model_name, prompt) -> (quality, cost)`:

```python
from hw_router import BaselineRouter, IRTRouter, CarrotRouter, UMRRouter
from hw_router.constants import DEFAULT_LAMBDA

# Choose a quality predictor
router = BaselineRouter()  # Static quality lookup (no dependencies)
# router = IRTRouter()     # IRT-based quality (requires transformers)
# router = UMRRouter()     # Cluster-based quality (requires sentence-transformers)

# Score each candidate model
prompt = "Explain quantum computing in simple terms."
models = ["Qwen2.5-14B-Instruct", "Llama-3.1-8B-Instruct", "Qwen2.5-3B-Instruct"]

for model in models:
    quality, cost = router.compute(model, prompt)
    score = DEFAULT_LAMBDA * quality - (1 - DEFAULT_LAMBDA) * cost
    print(f"{model}: quality={quality:.3f}, score={score:.4f}")
```

### Running on Your Own Hardware

Want to run HW-Router with your own GPUs and models? See **[docs/CUSTOM_HARDWARE_GUIDE.md](docs/CUSTOM_HARDWARE_GUIDE.md)** for a step-by-step walkthrough covering LLM pool config, vLLM setup, data collection, training, and evaluation.

### Training HW-Router on Your Own Models

HW-Router is designed to be adapted to any vLLM serving stack. At a glance, the workflow is:

| # | Stage | Command | Runs on |
|---|---|---|---|
| 1 | Prepare a prompt dataset | `python pipeline/data_preparation/combine_datasets.py` | CPU |
| 2 | Collect hardware telemetry from your live vLLM servers | `python pipeline/data_collection/build_hardware_cost_dataset.py --config <your-config>` | GPU |
| 3 | Train the MLP cost predictor (~20s) | `python -m pipeline.training.train_cost_model` | CPU |
| 4 | Build an offline evaluation set (optional) | `python pipeline/eval_processing/process_eval_dataset.py …` | CPU |
| 5 | Offline λ-sweep or online routing eval | `python pipeline/evaluation/eval_lambda_sweep.py …` | CPU / GPU |

Once trained, the cost predictor plugs into any quality predictor (CARROT, IRT, UMR, or your own) to form a hardware-aware router.

Full commands, inputs/outputs, and tips for each stage are in **[pipeline/README.md](pipeline/README.md)**. If you're swapping in different GPUs or a different model pool, also see **[docs/CUSTOM_HARDWARE_GUIDE.md](docs/CUSTOM_HARDWARE_GUIDE.md)**.

## Models

Evaluated with 5 LLMs across 2 NVIDIA H100 GPUs:

- **GPU 0:** Qwen2.5-14B-Instruct, Phi-3-mini-128k-instruct (3.8B)
- **GPU 1:** Llama-3.1-8B-Instruct, Qwen2.5-3B-Instruct, Mistral-7B-Instruct-v0.3

## Adding Your Own Router

Subclass `BaseRouter` and implement the `compute()` method:

```python
from hw_router import BaseRouter

class MyRouter(BaseRouter):
    def compute(self, model_name: str, prompt: str):
        quality = your_quality_function(model_name, prompt)
        cost = your_cost_function(model_name, prompt)
        return quality, cost
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full details.

## Citation

```bibtex
@inproceedings{kabir2026hwrouter,
  title     = {{HW-Router}: Hardware-Aware Routing for Scalable Multi-{LLM} Serving},
  author    = {Kabir, Ahasan and Xue, Jiaqi and Zheng, Mengxin and Lou, Qian},
  booktitle = {Proceedings of the 63rd Design Automation Conference (DAC)},
  year      = {2026}
}
```

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.
