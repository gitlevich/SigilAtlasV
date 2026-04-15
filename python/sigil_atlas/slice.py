"""Slice — which images are present and how they're scored.

The slice is the input to the torus layout engine.
SigilControls define ContrastControls. The active set IS the query.

RelevanceFilter composition (from spec):
  1. Start with all corpus images
  2. Apply metadata range filters (AND)
  3. Apply all filter-role ContrastControls as bandpass (AND)
  4. Score remaining images against attract-role ContrastControls
  5. Slice = filtered images, ranked by composite score
  6. Order-role ContrastControl determines strip layout ordering
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
    """Attract toward a text concept. Soft ranking — doesn't exclude."""
    text: str
    weight: float = 1.0


@dataclass
class ContrastControl:
    """A bandpass along a tension between two poles.

    direction = normalize(pole_a_embedding - pole_b_embedding)
    Each image projected onto this direction via dot product.

    Roles:
      filter — images outside [band_min, band_max] are excluded
      attract — images scored by proximity to direction, soft ranking
      order — determines strip ordering (one at a time)
    """
    pole_a: str  # text concept for pole A
    pole_b: str  # text concept for pole B
    role: str = "filter"  # "filter", "attract", or "order"
    band_min: float = -1.0  # bandpass range (normalized projection)
    band_max: float = 1.0


@dataclass
class SliceResult:
    """The output of compute_slice."""
    image_ids: list[str]
    scores: dict[str, float]  # image_id -> composite attract score
    order_projections: dict[str, float] | None  # image_id -> order axis value
    capture_dates: dict[str, float]  # image_id -> unix timestamp


def _encode_cached(provider: EmbeddingProvider, text: str, model: str) -> np.ndarray:
    """Encode text with caching via things module."""
    from sigil_atlas.things import _encode_prompt
    # Use the taxonomy prompt if available, otherwise raw text
    from sigil_atlas.things import _find_node
    node = _find_node(text)
    prompt = node.prompt if node else text
    return _encode_prompt(prompt)


def _compute_contrast_direction(
    provider: EmbeddingProvider,
    pole_a: str,
    pole_b: str,
    model: str,
) -> np.ndarray:
    """Compute contrast direction = normalize(pole_a - pole_b)."""
    vec_a = _encode_cached(provider, pole_a, model)
    vec_b = _encode_cached(provider, pole_b, model)
    direction = vec_a - vec_b
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return direction
    return direction / norm


def _project_images(
    provider: EmbeddingProvider,
    image_ids: list[str],
    direction: np.ndarray,
    model: str,
) -> np.ndarray:
    """Project images onto a contrast direction. Returns (N,) array of projections."""
    if not image_ids:
        return np.array([], dtype=np.float32)
    matrix = provider.fetch_matrix(image_ids, model)
    return matrix @ direction


def filter_by_range(db: CorpusDB, filters: list[RangeFilter]) -> set[str]:
    """Apply range filters on characterizations. Multiple filters AND together."""
    if not filters:
        return set(db.fetch_image_ids())

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


def compute_slice(
    db: CorpusDB,
    provider: EmbeddingProvider,
    range_filters: list[RangeFilter] | None = None,
    proximity_filters: list[ProximityFilter] | None = None,
    contrast_controls: list[ContrastControl] | None = None,
    model: str = "clip-vit-b-32",
) -> SliceResult:
    """Compute the slice per the spec's RelevanceFilter composition.

    1. Start with all corpus images
    2. Apply metadata range filters (AND)
    3. Apply all filter-role ContrastControls as bandpass (AND)
    4. Score remaining images against attract-role contrasts + proximity filters
    5. Slice = filtered images, ranked by composite score
    6. Order-role contrast determines strip ordering
    """
    range_filters = range_filters or []
    proximity_filters = proximity_filters or []
    contrast_controls = contrast_controls or []

    # Step 1+2: range filtering
    if range_filters:
        candidates = list(filter_by_range(db, range_filters))
    else:
        candidates = db.fetch_image_ids()
    logger.info("Range filters: %d -> %d candidates", len(range_filters), len(candidates))

    # Step 3: filter-role ContrastControls as bandpass
    filter_controls = [c for c in contrast_controls if c.role == "filter"]
    for cc in filter_controls:
        if not candidates:
            break
        direction = _compute_contrast_direction(provider, cc.pole_a, cc.pole_b, model)
        projections = _project_images(provider, candidates, direction, model)
        # Normalize projections to [-1, 1] range based on corpus distribution
        p_min, p_max = projections.min(), projections.max()
        span = p_max - p_min
        if span > 1e-8:
            normalized = 2.0 * (projections - p_min) / span - 1.0
        else:
            normalized = np.zeros_like(projections)
        # Apply bandpass
        mask = (normalized >= cc.band_min) & (normalized <= cc.band_max)
        candidates = [candidates[i] for i in range(len(candidates)) if mask[i]]
        logger.info(
            "Bandpass '%s' vs '%s' [%.1f, %.1f]: %d -> %d",
            cc.pole_a, cc.pole_b, cc.band_min, cc.band_max,
            mask.shape[0], sum(mask),
        )

    if not candidates:
        return SliceResult([], {}, None)

    # Step 4: score against attract-role contrasts + proximity filters
    attract_controls = [c for c in contrast_controls if c.role == "attract"]
    composite_scores = np.zeros(len(candidates), dtype=np.float32)

    # Attract-role contrasts: project onto direction, use as score
    for cc in attract_controls:
        direction = _compute_contrast_direction(provider, cc.pole_a, cc.pole_b, model)
        projections = _project_images(provider, candidates, direction, model)
        composite_scores += projections

    # Proximity filters: score by cosine similarity to text
    for pf in proximity_filters:
        vec = _encode_cached(provider, pf.text, model)
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        matrix = provider.fetch_matrix(candidates, model)
        scores = matrix @ vec
        composite_scores += scores * pf.weight

    # Step 5: sort by composite score (descending)
    scores_dict = {candidates[i]: float(composite_scores[i]) for i in range(len(candidates))}

    if attract_controls or proximity_filters:
        order = np.argsort(-composite_scores)
        sorted_ids = [candidates[i] for i in order]
    else:
        sorted_ids = candidates

    # Step 6: order-role contrast (for strip ordering)
    order_controls = [c for c in contrast_controls if c.role == "order"]
    order_projections = None
    if order_controls:
        oc = order_controls[0]  # single order axis invariant
        direction = _compute_contrast_direction(provider, oc.pole_a, oc.pole_b, model)
        projs = _project_images(provider, sorted_ids, direction, model)
        order_projections = {sorted_ids[i]: float(projs[i]) for i in range(len(sorted_ids))}

    # Fetch capture dates for default time ordering
    capture_dates = db.fetch_capture_dates(sorted_ids)

    logger.info("Slice: %d images, %d attract, %d filter, %d order controls",
                len(sorted_ids), len(attract_controls), len(filter_controls), len(order_controls))

    return SliceResult(sorted_ids, scores_dict, order_projections, capture_dates)
