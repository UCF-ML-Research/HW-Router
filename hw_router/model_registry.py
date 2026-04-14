"""
model_registry.py — Config-driven model name/ID mappings.

Uses basename matching so that any local model path resolves
correctly regardless of the system.

Usage:
    from hw_router.model_registry import get_model_id, get_model_hugging_face_name

    get_model_id("/any/path/to/qwen14b")        → 0
    get_model_id("qwen14b")                      → 0
    get_model_hugging_face_name("qwen14b")        → "Qwen2.5-14B-Instruct"
"""

# Basename → (integer_id, HuggingFace short name)
_KNOWN_MODELS = {
    "qwen14b":    (0, "Qwen2.5-14B-Instruct"),
    "phi3-mini":  (1, "Phi-3-mini-128k-instruct"),
    "llama3-8b":  (2, "Llama-3.1-8B-Instruct"),
    "qwen3b":     (3, "Qwen2.5-3B-Instruct"),
    "mistral7b":  (4, "Mistral-7B-Instruct-v0.3"),
}


def _basename(path: str) -> str:
    """Extract the basename from a model path."""
    return path.rstrip("/").split("/")[-1]


def get_model_id(name: str) -> int:
    """
    Return integer model ID (0-4) given a local path or basename.

    Examples:
        get_model_id("/home/user/models/qwen14b")  → 0
        get_model_id("qwen14b")                     → 0
    """
    key = _basename(name)
    if key in _KNOWN_MODELS:
        return _KNOWN_MODELS[key][0]
    raise KeyError(f"Unknown model name: {name} (basename: {key})")


def get_model_hugging_face_name(name: str) -> str:
    """
    Return HuggingFace short name given a local path or basename.

    Examples:
        get_model_hugging_face_name("/home/user/models/qwen14b")  → "Qwen2.5-14B-Instruct"
        get_model_hugging_face_name("qwen14b")                     → "Qwen2.5-14B-Instruct"
    """
    key = _basename(name)
    if key in _KNOWN_MODELS:
        return _KNOWN_MODELS[key][1]
    raise KeyError(f"Unknown model name: {name} (basename: {key})")


def get_all_models():
    """Return list of (basename, model_id, hf_name) for all known models."""
    return [
        (basename, mid, hf_name)
        for basename, (mid, hf_name) in sorted(_KNOWN_MODELS.items(), key=lambda x: x[1][0])
    ]
