"""Image wrapping — characterize by descending the ontology tree.

For each image, start at root. At each node, compute CLIP similarity
to all children's prompts. Pick the best match. Descend. Repeat
until leaf. The path is the image's characterization.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import open_clip
import torch

from sigil_atlas.cancel import CancellationToken
from sigil_atlas.db import CorpusDB
from sigil_atlas.ontology import OntologyNode, all_prompts, get_ontology
from sigil_atlas.progress import StageProgress

logger = logging.getLogger(__name__)


@dataclass
class ImageCharacterization:
    """Result of wrapping a single image."""

    image_id: str
    path: list[str] = field(default_factory=list)

    @property
    def invariant_labels(self) -> frozenset[str]:
        """Each prefix of the path is an invariant.

        path = [subject, organism, animal, bird, songbird]
        produces: {subject, subject/organism, subject/organism/animal, ...}
        """
        labels = set()
        for i in range(len(self.path)):
            labels.add("/".join(self.path[: i + 1]))
        return frozenset(labels)


class OntologyIndex:
    """Pre-embedded ontology prompts for CLIP zero-shot classification."""

    def __init__(self, model, tokenizer, device: torch.device) -> None:
        self.device = device
        prompts_list = all_prompts()
        self._embeddings: dict[str, np.ndarray] = {}
        self._build(model, tokenizer, prompts_list)

    @torch.no_grad()
    def _build(self, model, tokenizer, prompts_list: list[tuple[str, str]]) -> None:
        prompts = [prompt for _, prompt in prompts_list]
        names = [name for name, _ in prompts_list]
        tokens = tokenizer(prompts).to(self.device)
        text_features = model.encode_text(tokens)
        text_features = text_features.cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(text_features, axis=1, keepdims=True)
        normalized = text_features / np.maximum(norms, 1e-8)

        for i, name in enumerate(names):
            self._embeddings[name] = normalized[i]

        logger.info("Ontology index built: %d prompts embedded", len(prompts))

    def similarity(self, image_embedding: np.ndarray, node_name: str) -> float:
        emb = self._embeddings.get(node_name)
        if emb is None:
            return 0.0
        img_norm = image_embedding / max(np.linalg.norm(image_embedding), 1e-8)
        return float(np.dot(img_norm, emb))


def characterize_image(
    image_embedding: np.ndarray,
    index: OntologyIndex,
    root: OntologyNode | None = None,
) -> list[str]:
    """Descend the ontology tree, picking best-matching child at each level."""
    root = root or get_ontology()
    path: list[str] = []

    node = root
    while node.children:
        best_child = None
        best_sim = -float("inf")
        for child in node.children:
            sim = index.similarity(image_embedding, child.name)
            if sim > best_sim:
                best_sim = sim
                best_child = child
        path.append(best_child.name)
        node = best_child

    return path


def run_wrapping_stage(
    db: CorpusDB,
    progress: StageProgress,
    token: CancellationToken,
    batch_size: int = 64,
) -> None:
    """Characterize all uncharacterized images by descending the ontology tree."""
    uncharacterized = db.fetch_uncharacterized_image_ids()
    progress.set_total(len(uncharacterized))

    if not uncharacterized:
        logger.info("All images already characterized")
        return

    logger.info("Characterizing %d images", len(uncharacterized))

    device = _select_device()
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    try:
        index = OntologyIndex(model, tokenizer, device)
        root = get_ontology()

        for i in range(0, len(uncharacterized), batch_size):
            if token.is_cancelled:
                logger.info("Wrapping cancelled")
                return

            batch_ids = uncharacterized[i : i + batch_size]
            db_rows = []

            for image_id in batch_ids:
                embedding = db.fetch_embedding(image_id, "clip-vit-b-32")
                if embedding is None:
                    logger.warning("No CLIP embedding for %s, skipping", image_id)
                    continue

                image_vec = np.array(embedding, dtype=np.float32)
                path = characterize_image(image_vec, index, root)

                for depth, node_name in enumerate(path):
                    prefix = "/".join(path[: depth + 1])
                    db_rows.append((image_id, prefix, "enum", node_name, None))

            if db_rows:
                db.insert_characterizations_batch(db_rows)

            progress.advance(len(batch_ids))

    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("Wrapping stage complete")


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
