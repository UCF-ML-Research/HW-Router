#!/usr/bin/env python3
"""
Quickstart: Use HW-Router's quality predictors to score models.

This example uses the BaselineRouter (static quality lookup) which
requires no model downloads or GPU. For IRT-based quality prediction,
replace BaselineRouter with IRTRouter().

Usage:
    pip install -e .
    python examples/quickstart.py
"""

from hw_router import BaselineRouter
from hw_router.constants import DEFAULT_LAMBDA, MODEL_QUALITY

# All routers share the same interface: compute(model_name, prompt) -> (quality, cost)
router = BaselineRouter()

prompt = "Explain quantum computing in simple terms."

# Score each candidate model
models = list(MODEL_QUALITY.keys())
print(f"Prompt: {prompt!r}\n")
print(f"{'Model':<35} {'Quality':>8} {'Cost':>8} {'Score':>8}")
print("-" * 65)

best_model, best_score = None, float("-inf")
for model in models:
    quality, cost = router.compute(model, prompt)
    score = DEFAULT_LAMBDA * quality - (1 - DEFAULT_LAMBDA) * cost
    print(f"{model:<35} {quality:>8.3f} {cost:>8.4f} {score:>8.4f}")

    if score > best_score:
        best_score = score
        best_model = model

print(f"\nSelected model: {best_model} (score={best_score:.4f})")
