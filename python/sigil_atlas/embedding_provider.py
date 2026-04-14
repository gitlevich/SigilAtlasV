"""Provider-agnostic embedding interface.

The layout and slice modules depend only on EmbeddingProvider, not on CorpusDB
directly. When ported to TS, a different provider fetches from a web API.
"""

import struct
from typing import Protocol

import numpy as np

from sigil_atlas.db import CorpusDB


class EmbeddingProvider(Protocol):
    """Abstract interface for fetching embeddings and encoding text."""

    def fetch_matrix(self, image_ids: list[str], model: str) -> np.ndarray:
        """Return (N, D) float32 matrix of embeddings for the given image IDs.

        Rows correspond 1:1 to image_ids. Missing embeddings raise ValueError.
        """
        ...

    def encode_text(self, text: str, model: str) -> np.ndarray:
        """Encode a text query into the same embedding space. Returns (D,) vector."""
        ...

    def available_models(self) -> list[str]:
        """Return list of model identifiers with stored embeddings."""
        ...


class SqliteEmbeddingProvider:
    """Reads pre-computed embeddings from CorpusDB."""

    def __init__(self, db: CorpusDB) -> None:
        self._db = db

    def fetch_matrix(self, image_ids: list[str], model: str) -> np.ndarray:
        if not image_ids:
            return np.empty((0, 0), dtype=np.float32)

        rows = self._db._conn.execute(
            f"SELECT image_id, vector FROM embeddings "
            f"WHERE model_identifier = ? AND image_id IN ({','.join('?' * len(image_ids))})",
            [model] + image_ids,
        ).fetchall()

        vectors_by_id: dict[str, np.ndarray] = {}
        for row in rows:
            blob = row[1]
            count = len(blob) // 4
            vec = np.array(struct.unpack(f"<{count}f", blob), dtype=np.float32)
            vectors_by_id[row[0]] = vec

        missing = [iid for iid in image_ids if iid not in vectors_by_id]
        if missing:
            raise ValueError(f"Missing embeddings for {len(missing)} images with model {model}")

        return np.stack([vectors_by_id[iid] for iid in image_ids])

    def encode_text(self, text: str, model: str) -> np.ndarray:
        """Encode text via CLIP. Only supports clip-vit-b-32."""
        if "clip" not in model:
            raise ValueError(f"Text encoding only supported for CLIP models, got {model}")

        import open_clip
        import torch

        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        clip_model = clip_model.to(device).eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-32")

        with torch.no_grad():
            tokens = tokenizer([text]).to(device)
            features = clip_model.encode_text(tokens)
            vec = features.cpu().numpy().astype(np.float32)[0]
            vec = vec / max(np.linalg.norm(vec), 1e-8)

        del clip_model
        return vec

    def available_models(self) -> list[str]:
        rows = self._db._conn.execute(
            "SELECT DISTINCT model_identifier FROM embeddings"
        ).fetchall()
        return [r[0] for r in rows]
