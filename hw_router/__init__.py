"""
HW-Router: Hardware-Aware Routing for Scalable Multi-LLM Serving.

Accepted at DAC 2026.

Core components (Paper Section 3):
    - hardware_monitor: Real-time vLLM metrics polling
    - cost_predictor: Neural network latency predictor (TTFT, TPOT)
    - routers: Quality predictors (IRT, CARROT, baselines)
    - model_registry: Config-driven model name/ID mappings
    - constants: Shared constants (prices, normalization, quality proxies)
    - load_patterns: Request arrival pattern generators

Quick start::

    from hw_router import IRTRouter, BaselineRouter, HardwareCostPredictor
    from hw_router.constants import DEFAULT_LAMBDA

    router = BaselineRouter()
    quality, cost = router.compute("Qwen/Qwen2.5-14B-Instruct", "Explain quantum computing.")
    score = DEFAULT_LAMBDA * quality - (1 - DEFAULT_LAMBDA) * cost
"""

__version__ = "0.1.0"

from hw_router.routers import (
    BaseRouter,
    BaselineRouter,
    RandomRouter,
    RoundRobinRouter,
    CarrotRouter,
    IRTRouter,
    UMRRouter,
)
from hw_router.cost_predictor import HardwareCostPredictor, HardwareCostNet
from hw_router.hardware_monitor import start_metrics_watcher
from hw_router.constants import MODEL_PRICES, MODEL_QUALITY, DEFAULT_LAMBDA
from hw_router.model_registry import get_model_id, get_model_hugging_face_name

__all__ = [
    "BaseRouter",
    "BaselineRouter",
    "RandomRouter",
    "RoundRobinRouter",
    "CarrotRouter",
    "IRTRouter",
    "UMRRouter",
    "HardwareCostPredictor",
    "HardwareCostNet",
    "start_metrics_watcher",
    "MODEL_PRICES",
    "MODEL_QUALITY",
    "DEFAULT_LAMBDA",
    "get_model_id",
    "get_model_hugging_face_name",
]
