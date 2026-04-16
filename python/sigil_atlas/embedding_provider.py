"""Provider-agnostic embedding interface.

The layout and slice modules depend only on EmbeddingProvider, not on CorpusDB
directly. When ported to TS, a different provider fetches from a web API.
"""

import logging
import struct
import time
from typing import Protocol

import numpy as np

from sigil_atlas.db import CorpusDB

logger = logging.getLogger(__name__)


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
    """Reads pre-computed embeddings from CorpusDB.

    Caches the full embedding matrix per model in RAM so that repeated
    queries (attract, contrast, layout) are instant numpy dot products
    instead of 70k SQLite blob reads each time.
    """

    def __init__(self, db: CorpusDB) -> None:
        self._db = db
        # model -> (ordered_ids, matrix)
        self._matrix_cache: dict[str, tuple[list[str], dict[str, int], np.ndarray]] = {}

    def invalidate_cache(self, model: str | None = None) -> None:
        """Drop cached matrices. Call after ingest adds new embeddings."""
        if model:
            self._matrix_cache.pop(model, None)
        else:
            self._matrix_cache.clear()

    def _ensure_cached(self, model: str) -> tuple[list[str], dict[str, int], np.ndarray]:
        """Load the full embedding matrix for a model into RAM, if not already cached."""
        if model in self._matrix_cache:
            return self._matrix_cache[model]

        t0 = time.monotonic()
        rows = self._db._conn.execute(
            "SELECT image_id, vector FROM embeddings WHERE model_identifier = ?",
            (model,),
        ).fetchall()

        if not rows:
            empty = ([], {}, np.empty((0, 0), dtype=np.float32))
            self._matrix_cache[model] = empty
            return empty

        # Unpack all vectors at once
        first_blob = rows[0][1]
        dim = len(first_blob) // 4
        n = len(rows)

        ids: list[str] = []
        id_to_idx: dict[str, int] = {}
        matrix = np.empty((n, dim), dtype=np.float32)

        for i, row in enumerate(rows):
            iid = row[0]
            ids.append(iid)
            id_to_idx[iid] = i
            blob = row[1]
            matrix[i] = np.frombuffer(blob, dtype=np.float32)

        dt = time.monotonic() - t0
        mb = matrix.nbytes / (1024 * 1024)
        logger.info(
            "Cached %d embeddings for %s (%.0f MB, %.1fs)",
            n, model, mb, dt,
        )

        result = (ids, id_to_idx, matrix)
        self._matrix_cache[model] = result
        return result

    def fetch_matrix(self, image_ids: list[str], model: str) -> np.ndarray:
        if not image_ids:
            return np.empty((0, 0), dtype=np.float32)

        _, id_to_idx, full_matrix = self._ensure_cached(model)

        # Gather rows by index — no per-row SQL or struct.unpack
        indices = []
        missing = []
        for iid in image_ids:
            idx = id_to_idx.get(iid)
            if idx is not None:
                indices.append(idx)
            else:
                missing.append(iid)

        if missing:
            raise ValueError(f"Missing embeddings for {len(missing)} images with model {model}")

        return full_matrix[indices]

    _text_model_cache: dict[str, tuple] = {}

    def encode_text(self, text: str, model: str) -> np.ndarray:
        """Encode text via CLIP. Supports clip-vit-b-32 and clip-vit-l-14."""
        if "clip" not in model:
            raise ValueError(f"Text encoding only supported for CLIP models, got {model}")

        import open_clip
        import torch

        arch = "ViT-L-14" if "l-14" in model else "ViT-B-32"

        if arch not in self._text_model_cache:
            clip_model, _, _ = open_clip.create_model_and_transforms(arch, pretrained="openai")
            device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
            clip_model = clip_model.to(device).eval()
            tokenizer = open_clip.get_tokenizer(arch)
            self._text_model_cache[arch] = (clip_model, tokenizer, device)

        clip_model, tokenizer, device = self._text_model_cache[arch]

        with torch.no_grad():
            tokens = tokenizer([text]).to(device)
            features = clip_model.encode_text(tokens)
            vec = features.cpu().numpy().astype(np.float32)[0]
            vec = vec / max(np.linalg.norm(vec), 1e-8)

        return vec

    def available_models(self) -> list[str]:
        rows = self._db._conn.execute(
            "SELECT DISTINCT model_identifier FROM embeddings"
        ).fetchall()
        return [r[0] for r in rows]
