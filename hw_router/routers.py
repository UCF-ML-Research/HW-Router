# routers.py — Quality Predictors (Paper Section 3.3)
#
# All router classes that estimate response quality Q_i(x) for each model.
# The hardware cost predictor (cost_predictor.py) is a plug-in component:
# it can be combined with ANY quality predictor here to make it hardware-aware.
# In the paper we pair IRT with the hardware cost predictor (best results),
# but CARROT and UMR benefit equally — see README for the full comparison.

from abc import ABC, abstractmethod
import json
import random

from hw_router.constants import MODEL_QUALITY, MODEL_PRICES

# ============================================================
# Base Router Interface
# ============================================================
class BaseRouter(ABC):
    @abstractmethod
    def compute(self, model_name: str, prompt: str):
        """
        Return:
            quality_score (float)
            cost_score (float)  # static or router-defined cost
        """
        pass


# ============================================================
# Baseline Router (quality-only OR static cost = 0)
# ============================================================
class BaselineRouter(BaseRouter):
    def compute(self, model_name, prompt):
        quality = MODEL_QUALITY.get(model_name, 1.0)
        cost = 0.0
        return quality, cost


# ============================================================
# Random Router (for sanity)
# ============================================================
class RandomRouter(BaseRouter):
    def compute(self, model_name, prompt):
        quality = random.random()
        cost = random.random()
        return quality, cost


# ============================================================
# Round Robin with dummy scoring
# ============================================================
class RoundRobinRouter(BaseRouter):
    def __init__(self):
        self.counter = 0

    def compute(self, model_name, prompt):
        quality = MODEL_QUALITY.get(model_name, 1.0)
        cost = 0.0
        return quality, cost


# ============================================================
# CARROT Router (static cost + CARROT quality)
# ============================================================
class CarrotRouter(BaseRouter):
    def __init__(self, carrot_model):
        """
        carrot_model: object returned by load_carrot_router(...)
        """
        self.carrot = carrot_model

    # ---------------------------------------------------------
    # OLD: text-based compute (kept for compatibility)
    # ---------------------------------------------------------
    def compute(self, model_name, prompt):
        emb = self.carrot.encode(prompt)
        return self.compute_from_embedding(model_name, emb)

    # ---------------------------------------------------------
    # NEW FAST PATH: embedding-based compute
    # ---------------------------------------------------------
    def compute_from_embedding(self, model_name, emb):
        """
        emb: numpy array (precomputed embedding)
        """
        # CARROT predicted quality
        q = self.carrot.get_quality(emb, model_name)

        # CARROT predicted token count (static cost)
        static_cost = self.carrot.get_cost(emb, model_name)

        # Convert cost = tokens * USD/token
        static_cost = static_cost * MODEL_PRICES.get(model_name, 1e-7)

        return q, static_cost

    # ---------------------------------------------------------
    # Length predictor (updated to use either embedding or text)
    # ---------------------------------------------------------
    def length_predictor(self, model_name, prompt=None, emb=None):
        """
        Either provide prompt (slow) OR emb (fast).
        """
        if emb is None:
            emb = self.carrot.encode(prompt)
        return self.carrot.get_cost(emb, model_name)


# ============================================================
# IRT Router (quality from MIRT + static price)
# ============================================================
IRT_CHECKPOINT = "baselines/irt/mirt_hw.snapshot"
IRT_META_PATH = "baselines/irt/mirt_llm.meta.json"


class _IRTEmbedder:
    def __init__(self, model_name: str, device: str):
        from transformers import AutoModel, AutoTokenizer
        import torch

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(device)
        self._model.eval()
        self._device = device

    def _mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = self._torch.sum(last_hidden_state * mask, dim=1)
        counts = self._torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def encode(self, texts, batch_size=16, max_length=512):
        import numpy as np

        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self._device) for k, v in encoded.items()}
            with self._torch.no_grad():
                outputs = self._model(**encoded)
                pooled = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            all_embeddings.append(pooled.cpu().numpy())

        return np.vstack(all_embeddings)


class IRTRouter(BaseRouter):
    def __init__(
        self,
        checkpoint: str = IRT_CHECKPOINT,
        meta_path: str = IRT_META_PATH,
        device: str = "auto",
        embed_batch_size: int = 16,
        max_length: int = 512,
    ):
        self.checkpoint = checkpoint
        self.meta_path = meta_path
        self.embed_batch_size = embed_batch_size
        self.max_length = max_length
        self._device = self._resolve_device(device)
        self._load()

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_llm_profiles(self, llm_profile_path: str):
        import pandas as pd

        llm_df = pd.read_csv(llm_profile_path)
        if "name" not in llm_df.columns or "profile" not in llm_df.columns:
            raise ValueError("llm profile file must include columns: name, profile")
        return dict(zip(llm_df["name"], llm_df["profile"]))

    def _extract_hf_name(self, profile: str):
        marker = " is released"
        if marker in profile:
            return profile.split(marker, 1)[0].strip()
        return None

    def _load(self):
        from baselines.irt import MIRT
        import torch

        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        llm_profiles = self._load_llm_profiles(meta["llm_profile_path"])
        used_llms = meta["used_llms"]

        internal_to_hf = {}
        for internal_name in used_llms:
            profile = llm_profiles.get(internal_name, "")
            hf_name = self._extract_hf_name(profile)
            if hf_name:
                internal_to_hf[internal_name] = hf_name

        self._internal_to_hf = internal_to_hf
        self._hf_to_internal = {v: k for k, v in internal_to_hf.items()}
        self._used_llms = used_llms

        self._embedder = _IRTEmbedder(meta["embed_model"], self._device)
        llm_texts = [llm_profiles[name] for name in used_llms]
        llm_embeddings = self._embedder.encode(
            llm_texts,
            batch_size=self.embed_batch_size,
            max_length=self.max_length,
        )
        self._llm_emb_map = dict(zip(used_llms, llm_embeddings))

        self._mirt = MIRT.MIRT(meta["llm_dim"], meta["query_dim"], meta["latent_dim"])
        state = torch.load(self.checkpoint, map_location=self._device)
        self._mirt.irt_net.load_state_dict(state)
        self._mirt.irt_net.eval()
        self._torch = torch

    def _resolve_model_name(self, model_name: str) -> str:
        if model_name in self._used_llms:
            return model_name
        if model_name in self._hf_to_internal:
            return self._hf_to_internal[model_name]
        raise ValueError(f"Unknown model_name: {model_name}")

    def _resolve_cost(self, model_name: str, internal_name: str) -> float:
        if model_name in MODEL_PRICES:
            return MODEL_PRICES[model_name]
        hf_name = self._internal_to_hf.get(internal_name)
        if hf_name and hf_name in MODEL_PRICES:
            return MODEL_PRICES[hf_name]
        raise ValueError(f"Model {model_name} not found in MODEL_PRICES.")

    def compute(self, model_name, prompt):
        internal_name = self._resolve_model_name(model_name)
        if internal_name not in self._llm_emb_map:
            raise ValueError(f"Model {model_name} was not trained.")

        prompt_embedding = self._embedder.encode(
            [prompt],
            batch_size=1,
            max_length=self.max_length,
        )[0]

        llm_vec = self._torch.tensor(
            self._llm_emb_map[internal_name],
            dtype=self._torch.float32,
        ).unsqueeze(0)
        prompt_vec = self._torch.tensor(
            prompt_embedding,
            dtype=self._torch.float32,
        ).unsqueeze(0)

        with self._torch.no_grad():
            quality = self._mirt.generate(llm_vec, prompt_vec, device=self._device)[0]

        cost = self._resolve_cost(model_name, internal_name)
        return float(quality), float(cost)


# ============================================================
# UMR Router (wrapper around baselines.umr)
# ============================================================
class UMRRouter(BaseRouter):
    """
    Unified Model Router quality predictor.

    Uses KMeans clustering of prompt embeddings and per-cluster
    error vectors to predict quality. Cost is static (price per token).

    Args:
        work_dir: Path to UMR checkpoint directory (clusters.json, errors.json).
        device: Device for embeddings ("cpu" or "cuda").
    """
    def __init__(self, work_dir="checkpoints/umr", device="cpu"):
        from baselines.umr.umr_router import UMRRouter as _UMRRouter
        self._umr = _UMRRouter(work_dir=work_dir, device=device)

    def compute(self, model_name: str, prompt: str):
        quality, cost = self._umr.score(prompt, model_name)
        return quality, cost


_IRT_CACHE = None


def irt_score(
    prompt: str,
    model_name: str,
    checkpoint: str = IRT_CHECKPOINT,
    meta_path: str = IRT_META_PATH,
    device: str = "auto",
    embed_batch_size: int = 16,
    max_length: int = 512,
):
    """
    Returns:
        quality : float  (0-1, higher = better)
        cost    : float  (price_per_token from MODEL_PRICES)
    """
    global _IRT_CACHE
    if _IRT_CACHE is None:
        _IRT_CACHE = IRTRouter(
            checkpoint=checkpoint,
            meta_path=meta_path,
            device=device,
            embed_batch_size=embed_batch_size,
            max_length=max_length,
        )
    return _IRT_CACHE.compute(model_name, prompt)
