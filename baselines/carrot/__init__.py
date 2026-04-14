"""
CARROT (Cost-Aware Router with Regressor-based Optimization Techniques)

This package provides LLM routing with quality and cost prediction.
"""

from .carrot import (
    CarrotRouter,
    load_carrot_router,
    CarrotKNNBaseline,
    CarrotLinearBaseline,
    load_and_align_data,
    filter_predictions_to_models,
    route_baseline
)

__all__ = [
    'CarrotRouter',
    'load_carrot_router',
    'CarrotKNNBaseline',
    'CarrotLinearBaseline',
    'load_and_align_data',
    'filter_predictions_to_models',
    'route_baseline',
]

__version__ = '0.1.0'
