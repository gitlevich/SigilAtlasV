"""Image wrapping — characterize by descending taxonomy trees.

For each image, walk both taxonomy sigils (semantic, visual).
At each node, compute CLIP similarity to all children's prompts.
Pick the best match. Descend. The path is the characterization.

Two paths per image:
  semantic/person/portrait/child_portrait
  visual/light/bright/sunlit

Uses CLIP B-32 adapter for zero-shot classification — declared
via model_registry, not hardcoded.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from sigil_atlas.cancel import CancellationToken
from sigil_atlas.db import CorpusDB
from sigil_atlas.model_registry import get_adapter
from sigil_atlas.ontology import OntologyNode, all_prompts
from sigil_atlas.progress import StageProgress
from sigil_atlas.taxonomy import get_taxonomy

logger = logging.getLogger(__name__)

# Wrapping uses B-32 for taxonomy classification.
# Declared here so it's visible and auditable — not buried in code.
WRAPPING_MODEL = "clip-vit-b-32"


@dataclass
class ImageCharacterization:
    """Result of wrapping a single image."""

    image_id: str
    path: list[str] = field(default_factory=list)

    @property
    def invariant_labels(self) -> frozenset[str]:
        labels = set()
        for i in range(len(self.path)):
            labels.add("/".join(self.path[: i + 1]))
        return frozenset(labels)


class TaxonomyIndex:
    """Pre-embedded taxonomy prompts for zero-shot classification.

    Uses the wrapping adapter's text encoder to embed all taxonomy
    prompts at construction time.
    """

    def __init__(self) -> None:
        self._embeddings: dict[str, np.ndarray] = {}
        adapter = get_adapter(WRAPPING_MODEL)

        taxonomy = get_taxonomy()
        all_prompt_pairs = []
        for root in taxonomy.values():
            all_prompt_pairs.extend(all_prompts(root))

        for name, prompt in all_prompt_pairs:
            vec = adapter.encode_text(prompt)
            self._embeddings[name] = vec

        logger.info(
            "Taxonomy index built: %d prompts embedded via %s",
            len(all_prompt_pairs), adapter.model_id,
        )

    def similarity(self, image_embedding: np.ndarray, node_name: str) -> float:
        emb = self._embeddings.get(node_name)
        if emb is None:
            return 0.0
        img_norm = image_embedding / max(np.linalg.norm(image_embedding), 1e-8)
        return float(np.dot(img_norm, emb))


OntologyIndex = TaxonomyIndex


def characterize_image(
    image_embedding: np.ndarray,
    index: TaxonomyIndex,
    root: OntologyNode | None = None,
) -> list[str]:
    """Descend one taxonomy tree, picking best-matching child at each level."""
    if root is None:
        from sigil_atlas.ontology import get_ontology
        root = get_ontology()
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
    """Characterize all uncharacterized images along both taxonomy sigils.

    Uses CLIP B-32 for taxonomy classification. Image embeddings are
    fetched for the wrapping model from the database.
    """
    uncharacterized = db.fetch_uncharacterized_image_ids()
    progress.set_total(len(uncharacterized))

    if not uncharacterized:
        logger.info("All images already characterized")
        return

    logger.info("Characterizing %d images via %s", len(uncharacterized), WRAPPING_MODEL)

    index = TaxonomyIndex()
    taxonomy = get_taxonomy()

    for i in range(0, len(uncharacterized), batch_size):
        if token.is_cancelled:
            logger.info("Wrapping cancelled")
            return

        batch_ids = uncharacterized[i : i + batch_size]
        db_rows = []

        for image_id in batch_ids:
            embedding = db.fetch_embedding(image_id, WRAPPING_MODEL)
            if embedding is None:
                logger.warning("No %s embedding for %s, skipping", WRAPPING_MODEL, image_id)
                continue

            image_vec = np.array(embedding, dtype=np.float32)

            for sigil_name, root in taxonomy.items():
                path = characterize_image(image_vec, index, root)
                for depth, node_name in enumerate(path):
                    prefix = sigil_name + "/" + "/".join(path[: depth + 1])
                    db_rows.append((image_id, prefix, "enum", node_name, None))

        if db_rows:
            db.insert_characterizations_batch(db_rows)

        progress.advance(len(batch_ids))

    logger.info("Wrapping stage complete")
