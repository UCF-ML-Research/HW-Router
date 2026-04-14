"""
UMR Router (Unsupervised Model Router)

This package provides LLM routing using cluster-based error estimation
with quality and cost scoring.
"""

from .umr_router import UMRRouter, umr_score

__all__ = [
    'UMRRouter',
    'umr_score',
]

__version__ = '0.1.0'
