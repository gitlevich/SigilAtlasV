"""SliceMode — controls which images are present in the current view.

Invariant: Slice mode controls which images are present.
It never affects how they are arranged on the torus.
"""

import logging
from dataclasses import dataclass

import numpy as np

from sigil_atlas.db import CorpusDB
from sigil_atlas.embedding_provider import EmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass
class RangeFilter:
    """A filter on a characterization dimension."""
    dimension: str
    min_value: float
    max_value: float


@dataclass
class ProximityFilter:
    """A filter by semantic proximity to a text description."""
    text: str
    weight: float = 1.0


def filter_by_range(db: CorpusDB, filters: list[RangeFilter]) -> set[str]:
    """Apply range filters on characterizations. Multiple filters AND together.

    Returns the set of image IDs that pass all range filters.
    """
    if not filters:
        return set(db.fetch_image_ids())

    # Start with all images, intersect each filter
    result: set[str] | None = None
    for f in filters:
        rows = db._conn.execute(
            "SELECT image_id FROM characterizations "
            "WHERE proximity_name = ? AND value_type = 'range' "
            "AND value_range >= ? AND value_range <= ?",
            (f.dimension, f.min_value, f.max_value),
        ).fetchall()
        ids = {r[0] for r in rows}
        result = ids if result is None else result & ids

    return result or set()


def filter_by_proximity(
    provider: EmbeddingProvider,
    candidate_ids: list[str],
    filters: list[ProximityFilter],
    model: str = "clip-vit-b-32",
    threshold: float = 0.15,
) -> list[str]:
    """Score images by proximity to text descriptions. Exclude those below threshold.

    Multiple proximity filters combine into a weighted composite score.
    Returns image IDs sorted by descending composite score.
    """
    if not filters or not candidate_ids:
        return candidate_ids

    # Encode all text queries
    text_vectors = []
    weights = []
    for f in filters:
        vec = provider.encode_text(f.text, model)
        text_vectors.append(vec * f.weight)
        weights.append(abs(f.weight))

    # Composite direction: weighted sum of text embeddings
    composite = np.sum(text_vectors, axis=0)
    norm = np.linalg.norm(composite)
    if norm < 1e-8:
        return candidate_ids
    composite = composite / norm

    # Score each image
    matrix = provider.fetch_matrix(candidate_ids, model)
    scores = matrix @ composite  # cosine similarity (both L2-normalized)

    # Filter and sort
    passing = [(candidate_ids[i], float(scores[i])) for i in range(len(candidate_ids)) if scores[i] >= threshold]
    passing.sort(key=lambda x: x[1], reverse=True)
    return [iid for iid, _ in passing]


def compute_slice(
    db: CorpusDB,
    provider: EmbeddingProvider,
    range_filters: list[RangeFilter] | None = None,
    proximity_filters: list[ProximityFilter] | None = None,
    model: str = "clip-vit-b-32",
) -> list[str]:
    """Compute the slice: the subset of corpus images matching all filters.

    1. Apply range filters (AND) to get candidate set
    2. Apply proximity filters to score and threshold candidates
    3. Return ordered list of image IDs
    """
    range_filters = range_filters or []
    proximity_filters = proximity_filters or []

    # Step 1: range filtering
    if range_filters:
        candidates = filter_by_range(db, range_filters)
        candidate_list = list(candidates)
    else:
        candidate_list = db.fetch_image_ids()

    logger.info("Range filters: %d -> %d candidates", len(range_filters), len(candidate_list))

    # Step 2: proximity filtering
    if proximity_filters:
        result = filter_by_proximity(provider, candidate_list, proximity_filters, model)
        logger.info("Proximity filters: %d -> %d images", len(proximity_filters), len(result))
        return result

    return candidate_list
