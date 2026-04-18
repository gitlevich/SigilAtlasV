"""Model adapter registry — declares what each embedding model can do.

Each model is a strategy that knows its own capabilities:
  - Can it encode text? (CLIP yes, DINOv2 no)
  - What dimension are its embeddings?
  - How does it resolve a text query into its own embedding space?

Models without a text encoder bridge through one that does:
CLIP finds seed images matching the text, then the target model
averages its own embeddings for those seeds to produce a proxy
vector in its native space.

Generic code asks the adapter — never hardcodes model IDs.
"""

import logging
from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

# Avoid circular imports — type-check only
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sigil_atlas.embedding_provider import EmbeddingProvider


class ModelAdapter(ABC):
    """What a model can do. Each model declares its capabilities."""

    model_id: str
    dimension: int
    supports_text: bool
    bridge_model_id: str | None = None

    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text directly. Raises if supports_text is False."""
        ...

    def resolve_text_vector(
        self,
        text: str,
        provider: "EmbeddingProvider",
        image_ids: list[str],
    ) -> np.ndarray:
        """Resolve a text query into this model's embedding space.

        Models with a text encoder use it directly.
        Models without one bridge through their declared bridge model.
        """
        if self.supports_text:
            return self.encode_text(text)
        return self._bridge_text(text, provider, image_ids)

    def _bridge_text(
        self,
        text: str,
        provider: "EmbeddingProvider",
        image_ids: list[str],
    ) -> np.ndarray:
        """Bridge a text query through another model's text encoder.

        1. Encode text in bridge model's space
        2. Score all images in bridge space → find top-K seeds
        3. Average this model's embeddings for those seeds → proxy vector
        """
        if self.bridge_model_id is None:
            raise ValueError(f"{self.model_id} has no text encoder and no bridge model")

        bridge = get_adapter(self.bridge_model_id)
        text_vec = bridge.encode_text(text)

        # Score images in bridge model's space
        bridge_matrix = provider.fetch_matrix(image_ids, bridge.model_id)
        scores = bridge_matrix @ text_vec

        # Top-K seeds
        k = min(50, len(image_ids))
        top_indices = np.argsort(-scores)[:k]
        seed_ids = [image_ids[i] for i in top_indices]

        # Average this model's embeddings for the seeds
        seed_matrix = provider.fetch_matrix(seed_ids, self.model_id)
        proxy = seed_matrix.mean(axis=0)
        norm = np.linalg.norm(proxy)
        if norm > 1e-8:
            proxy = proxy / norm

        logger.info(
            "Bridged '%s' via %s → %d seeds → proxy in %s space",
            text[:40], bridge.model_id, k, self.model_id,
        )
        return proxy


# ── Text encoding (cached, lazy-loaded) ──

_text_cache: dict[tuple[str, str], np.ndarray] = {}
_clip_models: dict[str, tuple] = {}


def _get_clip_model(arch: str):
    """Lazy-load a CLIP model for text encoding. Cached per architecture."""
    if arch not in _clip_models:
        import open_clip
        import torch

        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        model, _, _ = open_clip.create_model_and_transforms(arch, pretrained="openai")
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer(arch)
        _clip_models[arch] = (model, tokenizer, device)
        logger.info("Loaded %s on %s for text encoding", arch, device)

    return _clip_models[arch]


def _encode_clip_text(arch: str, text: str) -> np.ndarray:
    """Encode text with a specific CLIP architecture. Cached."""
    key = (arch, text)
    if key in _text_cache:
        return _text_cache[key]

    import torch

    model, tokenizer, device = _get_clip_model(arch)
    with torch.no_grad():
        tokens = tokenizer([text]).to(device)
        features = model.encode_text(tokens)
        vec = features.cpu().numpy().astype(np.float32)[0]
        vec = vec / max(np.linalg.norm(vec), 1e-8)

    _text_cache[key] = vec
    return vec


# ── Concrete adapters ──


class CLIPAdapter(ModelAdapter):
    """CLIP ViT-B/32 — 512-dim, has text encoder."""

    model_id = "clip-vit-b-32"
    dimension = 512
    supports_text = True
    bridge_model_id = None

    def encode_text(self, text: str) -> np.ndarray:
        return _encode_clip_text("ViT-B-32", text)


class CLIPLargeAdapter(ModelAdapter):
    """CLIP ViT-L/14 — 768-dim, has text encoder."""

    model_id = "clip-vit-l-14"
    dimension = 768
    supports_text = True
    bridge_model_id = None

    def encode_text(self, text: str) -> np.ndarray:
        return _encode_clip_text("ViT-L-14", text)


class DINOv2Adapter(ModelAdapter):
    """DINOv2 ViT-B/14 — 768-dim, visual only, bridges through CLIP B-32.

    B-32 is the default bridge because (a) it's smaller and faster to
    encode and (b) workspaces tend to have full B-32 coverage by default,
    while L-14 is opt-in and often partial. Bridge precision is bounded by
    the seed-image step, not by which CLIP variant runs — the difference
    between B-32 and L-14 as a bridge is below the noise floor here.
    """

    model_id = "dinov2-vitb14"
    dimension = 768
    supports_text = False
    bridge_model_id = "clip-vit-b-32"

    def encode_text(self, text: str) -> np.ndarray:
        raise ValueError("DINOv2 has no text encoder — use resolve_text_vector()")


# ── Registry ──

_ADAPTERS: dict[str, ModelAdapter] = {}


def _register(adapter: ModelAdapter) -> None:
    _ADAPTERS[adapter.model_id] = adapter


_register(CLIPAdapter())
_register(CLIPLargeAdapter())
_register(DINOv2Adapter())


def get_adapter(model_id: str) -> ModelAdapter:
    """Look up a model adapter by ID. Raises ValueError if unknown."""
    adapter = _ADAPTERS.get(model_id)
    if adapter is None:
        available = ", ".join(sorted(_ADAPTERS.keys()))
        raise ValueError(f"Unknown model: {model_id!r}. Available: {available}")
    return adapter


def available_adapters() -> list[ModelAdapter]:
    """Return all registered adapters."""
    return list(_ADAPTERS.values())
