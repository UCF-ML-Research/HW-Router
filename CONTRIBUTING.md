# Contributing to HW-Router

Thank you for your interest in contributing to HW-Router.

## Adding a New Router

All routers share the `BaseRouter` interface. To add your own:

1. Subclass `BaseRouter` and implement `compute()`:

```python
from hw_router import BaseRouter

class MyRouter(BaseRouter):
    def compute(self, model_name: str, prompt: str):
        """
        Args:
            model_name: HuggingFace model name (e.g., "Qwen2.5-14B-Instruct")
            prompt: The user's input text

        Returns:
            quality: float in [0, 1], higher is better
            cost: float, lower is better (static price, latency, or custom metric)
        """
        quality = your_quality_function(model_name, prompt)
        cost = your_cost_function(model_name, prompt)
        return quality, cost
```

2. Register it in `hw_router/routers.py` and export from `hw_router/__init__.py`.

3. Add it to the evaluation pipeline in `pipeline/evaluation/eval_lambda_sweep.py`.

See `examples/add_custom_router.py` for a complete working example.

## Adding a New GPU Configuration

Create a YAML file in `configs/`. Each model gets a short `name` (your choice) and a `base_url` pointing to its vLLM server:

```yaml
# configs/my_gpu_setup.yaml
gpus:
  "0":                                   # GPU 0
    - name: my-model-a                   # Short identifier (must match model_registry.py)
      base_url: http://localhost:8010
    - name: my-model-b
      base_url: http://localhost:8011
  "1":                                   # GPU 1
    - name: my-model-c
      base_url: http://localhost:8012
```

You also need to register the model names in `hw_router/model_registry.py`. See [docs/CUSTOM_HARDWARE_GUIDE.md](docs/CUSTOM_HARDWARE_GUIDE.md) for the full walkthrough.

Then use it with pipeline scripts:

```bash
python pipeline/data_collection/build_hardware_cost_dataset.py \
    --config configs/my_gpu_setup.yaml
```

`configs/test_small_models.yaml` is a minimal 2-model config useful for testing the pipeline without H100s (uses small public models).

## Development Setup

```bash
git clone https://github.com/UCF-ML-Research/HW-Router.git
cd HW-Router
pip install -e ".[dev]"
```

## Code Style

We use [ruff](https://github.com/astral-sh/ruff) for linting:

```bash
ruff check hw_router/ baselines/
# Or: make lint
```

## Pull Requests

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run linter: `ruff check hw_router/ baselines/`
5. Submit a pull request
