"""Embedding stages — CLIP ViT-B/32 and DINOv2 ViT-B/14."""

import logging
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from torchvision import transforms

from sigil_atlas.cancel import CancellationToken
from sigil_atlas.db import CorpusDB
from sigil_atlas.progress import StageProgress

logger = logging.getLogger(__name__)


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return vectors / norms


# ── CLIP ViT-B/32 ──


class CLIPEmbedder:
    """CLIP ViT-B/32 via open_clip. Produces 512-dim L2-normalized embeddings."""

    MODEL_ID = "clip-vit-b-32"
    DIMENSION = 512

    def __init__(self) -> None:
        self.device = _select_device()
        self.model = None
        self.preprocess = None

    def load(self) -> None:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        model = model.to(self.device).eval()
        self.model = model
        self.preprocess = preprocess
        logger.info("CLIP ViT-B/32 loaded on %s", self.device)

    def unload(self) -> None:
        self.model = None
        self.preprocess = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("CLIP model unloaded")

    @torch.no_grad()
    def embed_batch(self, images: list[Image.Image]) -> list[list[float]]:
        """Embed a batch of PIL images. Returns L2-normalized 512-dim vectors."""
        tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        features = self.model.encode_image(tensors)
        vectors = features.cpu().numpy().astype(np.float32)
        vectors = _l2_normalize(vectors)
        return [v.tolist() for v in vectors]


# ── DINOv2 ViT-B/14 ──


class DINOv2Embedder:
    """DINOv2 ViT-B/14 via torch.hub. Produces 768-dim L2-normalized embeddings."""

    MODEL_ID = "dinov2-vitb14"
    DIMENSION = 768

    def __init__(self) -> None:
        self.device = _select_device()
        self.model = None
        self._transform = None

    def load(self) -> None:
        self.model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True
        )
        self.model = self.model.to(self.device).eval()

        self._transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        logger.info("DINOv2 ViT-B/14 loaded on %s", self.device)

    def unload(self) -> None:
        self.model = None
        self._transform = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("DINOv2 model unloaded")

    @torch.no_grad()
    def embed_batch(self, images: list[Image.Image]) -> list[list[float]]:
        """Embed a batch of PIL images. Returns L2-normalized 768-dim vectors."""
        tensors = torch.stack([self._transform(img) for img in images]).to(self.device)
        features = self.model(tensors)
        vectors = features.cpu().numpy().astype(np.float32)
        vectors = _l2_normalize(vectors)
        return [v.tolist() for v in vectors]


# ── Embedding stage runner ──


def run_embedding_stage(
    db: CorpusDB,
    thumbnails_dir: Path,
    embedder: CLIPEmbedder | DINOv2Embedder,
    progress: StageProgress,
    token: CancellationToken,
    batch_size: int = 32,
) -> None:
    """Run an embedding stage: load model, process unembedded images, store results.

    Resumable: queries DB for images that have thumbnails but no embedding for this model.
    Cancellable: checks token between batches.
    Streaming: processes batches as they come, doesn't require all thumbnails upfront.
    """
    embedder.load()

    try:
        unembedded = db.fetch_unembedded_image_ids(embedder.MODEL_ID)
        progress.set_total(len(unembedded))

        if not unembedded:
            logger.info("All images already embedded with %s", embedder.MODEL_ID)
            return

        logger.info(
            "Embedding %d images with %s", len(unembedded), embedder.MODEL_ID
        )

        for i in range(0, len(unembedded), batch_size):
            if token.is_cancelled:
                logger.info("Embedding %s cancelled", embedder.MODEL_ID)
                return

            batch_ids = unembedded[i : i + batch_size]
            images: list[Image.Image] = []
            valid_ids: list[str] = []

            for image_id in batch_ids:
                thumb_path = thumbnails_dir / f"{image_id}.jpg"
                if not thumb_path.exists():
                    logger.warning("Thumbnail missing for %s, skipping", image_id)
                    continue
                try:
                    img = Image.open(thumb_path).convert("RGB")
                    images.append(img)
                    valid_ids.append(image_id)
                except Exception:
                    logger.warning(
                        "Failed to load thumbnail %s", thumb_path, exc_info=True
                    )

            if not images:
                progress.advance(len(batch_ids))
                continue

            vectors = embedder.embed_batch(images)

            pairs = [
                (img_id, embedder.MODEL_ID, vec)
                for img_id, vec in zip(valid_ids, vectors)
            ]
            db.insert_embeddings_batch(pairs)

            # Close PIL images
            for img in images:
                img.close()

            progress.advance(len(batch_ids))

    finally:
        embedder.unload()
