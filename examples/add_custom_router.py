#!/usr/bin/env python3
"""
Example: How to add your own router as a baseline.

Subclass BaseRouter and implement the compute() method.
This lets you plug into the HW-Router evaluation pipeline
and compare against CARROT, IRT, UMR, and HW-Router.

Usage:
    pip install -e .
    python examples/add_custom_router.py
"""

from hw_router import BaseRouter
from hw_router.constants import MODEL_QUALITY, MODEL_PRICES, DEFAULT_LAMBDA


class LengthAwareRouter(BaseRouter):
    """
    Example custom router that estimates quality based on
    prompt length (shorter prompts → higher quality for smaller models).
    """

    def compute(self, model_name: str, prompt: str):
        # Your quality prediction logic here
        base_quality = MODEL_QUALITY.get(model_name, 0.5)

        # Example: penalize large models for short prompts
        # (they are overkill and waste resources)
        prompt_length = len(prompt.split())
        if prompt_length < 20:
            quality = base_quality * 0.8  # short prompt, prefer smaller models
        else:
            quality = base_quality

        # Cost: use static price per token
        cost = MODEL_PRICES.get(model_name, 1e-7)

        return quality, cost


# --- Demo ---
if __name__ == "__main__":
    router = LengthAwareRouter()

    prompts = [
        "What is 2+2?",
        "Explain the theory of general relativity and its implications for modern physics, "
        "including gravitational waves, black holes, and the expansion of the universe.",
    ]

    models = list(MODEL_QUALITY.keys())

    for prompt in prompts:
        print(f"\nPrompt ({len(prompt.split())} words): {prompt[:80]}...")
        print(f"  {'Model':<35} {'Quality':>8} {'Score':>8}")

        best_model, best_score = None, float("-inf")
        for model in models:
            quality, cost = router.compute(model, prompt)
            score = DEFAULT_LAMBDA * quality - (1 - DEFAULT_LAMBDA) * cost
            print(f"  {model:<35} {quality:>8.3f} {score:>8.4f}")
            if score > best_score:
                best_score = score
                best_model = model

        print(f"  -> Selected: {best_model}")
