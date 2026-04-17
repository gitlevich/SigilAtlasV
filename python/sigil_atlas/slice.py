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

Text operations use the adapter for the active model. Models with
a text encoder use it directly. Models without one (DINOv2) bridge
through a CLIP model to find seed images, then search in their own
embedding space.
"""

import logging
from dataclasses import dataclass

import numpy as np

from sigil_atlas.db import CorpusDB
from sigil_atlas.embedding_provider import EmbeddingProvider
from sigil_atlas.model_registry import ModelAdapter, get_adapter

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
    pole_a: str
    pole_b: str
    role: str = "filter"
    band_min: float = -1.0
    band_max: float = 1.0


@dataclass
class SliceResult:
    """The output of compute_slice."""
    image_ids: list[str]
    scores: dict[str, float]
    order_projections: dict[str, float] | None
    capture_dates: dict[str, float]


def _resolve_text(
    adapter: ModelAdapter,
    text: str,
    provider: EmbeddingProvider,
    image_ids: list[str],
) -> np.ndarray:
    """Resolve a text query into the adapter's embedding space."""
    return adapter.resolve_text_vector(text, provider, image_ids)


def _score_category(
    adapter: ModelAdapter,
    provider: EmbeddingProvider,
    image_ids: list[str],
    text: str,
) -> np.ndarray:
    """Score images against a text concept. Normalized to [0, 1]."""
    if not image_ids:
        return np.array([], dtype=np.float32)
    vec = _resolve_text(adapter, text, provider, image_ids)
    matrix = provider.fetch_matrix(image_ids, adapter.model_id)
    raw = matrix @ vec
    lo, hi = float(raw.min()), float(raw.max())
    if hi - lo < 1e-9:
        return np.full(len(raw), 0.5, dtype=np.float32)
    return ((raw - lo) / (hi - lo)).astype(np.float32)


def _score_contrast(
    adapter: ModelAdapter,
    provider: EmbeddingProvider,
    image_ids: list[str],
    pole_a: str,
    pole_b: str,
) -> np.ndarray:
    """Score images along a contrast axis. Normalized to [-1, 1]."""
    if not image_ids:
        return np.array([], dtype=np.float32)
    vec_a = _resolve_text(adapter, pole_a, provider, image_ids)
    vec_b = _resolve_text(adapter, pole_b, provider, image_ids)
    matrix = provider.fetch_matrix(image_ids, adapter.model_id)
    raw = matrix @ vec_a - matrix @ vec_b
    lo, hi = float(raw.min()), float(raw.max())
    if hi - lo < 1e-9:
        return np.zeros(len(raw), dtype=np.float32)
    return ((raw - lo) / (hi - lo) * 2.0 - 1.0).astype(np.float32)


# Corpus size threshold: below this, knee detection works well on the
# smooth falloff curve. Above this, scores compress into a narrow band
# and the knee becomes too sharp — statistical thresholding scales better.
_SMALL_CORPUS = 1000


def _select_relevant(
    scores: np.ndarray,
    candidates: list[str],
    selected: set[str],
    feathering: float = 0.5,
) -> int:
    """Select images relevant to a text query.

    Automatically picks the right strategy:
      Small corpus (<1000): knee detection on the descending score curve.
        Works well when the falloff is smooth and distinct.
      Large corpus (>=1000): sigma thresholding above the mean.
        CLIP scores compress into a narrow range at scale; statistical
        selection scales correctly.

    Feathering controls selectivity in both strategies:
      0.0 (loose) → more results
      0.5 (default) → balanced
      1.0 (tight) → fewer, more specific results
    """
    n = len(scores)
    if n == 0:
        return 0

    if n < _SMALL_CORPUS:
        return _select_by_knee(scores, candidates, selected, feathering)
    return _select_by_sigma(scores, candidates, selected, feathering)


def _select_by_knee(
    scores: np.ndarray,
    candidates: list[str],
    selected: set[str],
    feathering: float = 0.5,
) -> int:
    """Knee detection on the descending score curve. Good for small corpora."""
    n = min(200, len(scores))
    if n < 3:
        for i in range(n):
            selected.add(candidates[int(np.argmax(scores))])
        return n

    order = np.argsort(-scores)
    sorted_scores = scores[order[:n]]

    x = np.arange(n, dtype=float)
    y = sorted_scores
    x0, y0 = 0.0, float(y[0])
    x1, y1 = float(n - 1), float(y[n - 1])
    dx, dy = x1 - x0, y1 - y0
    length = np.sqrt(dx * dx + dy * dy)
    if length < 1e-12:
        return 0

    distances = np.abs(dy * x - dx * y + x1 * y0 - x0 * y1) / length
    knee = int(np.argmax(distances))

    # 0 = strict (knee/2), 1 = permissive (2x knee)
    scale = 0.5 + 1.5 * feathering
    cutoff = max(5, int(knee * scale))
    cutoff = min(cutoff, n)

    count = 0
    for i in range(cutoff):
        selected.add(candidates[order[i]])
        count += 1
    return count


def _select_by_sigma(
    scores: np.ndarray,
    candidates: list[str],
    selected: set[str],
    feathering: float = 0.5,
) -> int:
    """Sigma thresholding above the mean. Good for large corpora.

    Tightness maps to sigma:
      0.0 → mean + 1.0 sigma (~16% of corpus)
      0.5 → mean + 2.0 sigma (~2-5%)
      1.0 → mean + 3.0 sigma (<1%)
    """
    mean = float(scores.mean())
    std = float(scores.std())
    if std < 1e-9:
        return 0

    # Slider: 0 = strict (few results), 1 = permissive (many results)
    # Sigma: high = strict, low = permissive
    sigma = 3.0 - 2.0 * feathering
    threshold = mean + sigma * std

    count = 0
    for i in range(len(scores)):
        if scores[i] >= threshold:
            selected.add(candidates[i])
            count += 1
    return count


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
    model: str = "clip-vit-l-14",
    feathering: float = 0.5,
) -> SliceResult:
    """Compute the slice per the spec's RelevanceFilter composition.

    The model parameter selects which adapter handles all operations —
    text encoding, image scoring, and layout. Models without a text
    encoder bridge through their declared bridge model.
    """
    adapter = get_adapter(model)

    range_filters = range_filters or []
    proximity_filters = proximity_filters or []
    contrast_controls = contrast_controls or []

    candidates = db.fetch_image_ids()

    # Step 1: attract selects matching images per term (union)
    # Raw scores are preserved for ordering — re-scoring the filtered
    # subset would destroy ranking (min-max on a narrow band = noise).
    attract_raw_scores: dict[str, float] = {}
    if proximity_filters:
        selected = set()
        for pf in proximity_filters:
            vec = _resolve_text(adapter, pf.text, provider, candidates)
            matrix = provider.fetch_matrix(candidates, adapter.model_id)
            raw = matrix @ vec
            count = _select_relevant(raw, candidates, selected, feathering=feathering)
            # Preserve raw scores for ordering
            for i, iid in enumerate(candidates):
                if iid in selected:
                    prev = attract_raw_scores.get(iid, 0.0)
                    attract_raw_scores[iid] = prev + float(raw[i]) * pf.weight
            logger.info("Attract '%s' [%s]: %d images", pf.text, adapter.model_id, count)
        candidates = [c for c in candidates if c in selected]
        logger.info("Attract total: %d images", len(candidates))

    if not candidates:
        return SliceResult([], {}, None, {})

    # Step 2: contrast bandpass (AND)
    filter_controls = [c for c in contrast_controls if c.role == "filter"]
    for cc in filter_controls:
        if not candidates:
            break
        normalized = _score_contrast(adapter, provider, candidates, cc.pole_a, cc.pole_b)
        mask = (normalized >= cc.band_min) & (normalized <= cc.band_max)
        candidates = [candidates[i] for i in range(len(candidates)) if mask[i]]
        logger.info("Bandpass [%s]: %d remain", adapter.model_id, len(candidates))

    # Step 3: range filters (color/tone)
    if range_filters:
        range_pass = filter_by_range(db, range_filters)
        candidates = [c for c in candidates if c in range_pass]
        logger.info("Range filters: %d remain", len(candidates))

    # Score: composite of all attract signals
    attract_controls = [c for c in contrast_controls if c.role == "attract"]
    composite_scores = np.zeros(len(candidates), dtype=np.float32)

    for cc in attract_controls:
        composite_scores += _score_contrast(adapter, provider, candidates, cc.pole_a, cc.pole_b)

    # Use raw scores from the selection pass — not re-scored against the
    # filtered subset (min-max normalization on a narrow band = noise).
    if attract_raw_scores:
        for i, iid in enumerate(candidates):
            composite_scores[i] += attract_raw_scores.get(iid, 0.0)

    scores_dict = {candidates[i]: float(composite_scores[i]) for i in range(len(candidates))}

    if attract_controls or proximity_filters:
        order = np.argsort(-composite_scores)
        sorted_ids = [candidates[i] for i in order]
    else:
        sorted_ids = candidates

    # Order axis
    order_controls = [c for c in contrast_controls if c.role == "order"]
    order_projections = None
    if order_controls:
        oc = order_controls[0]
        projs = _score_contrast(adapter, provider, sorted_ids, oc.pole_a, oc.pole_b)
        order_projections = {sorted_ids[i]: float(projs[i]) for i in range(len(sorted_ids))}

    capture_dates = db.fetch_capture_dates(sorted_ids)

    logger.info("Slice [%s]: %d images", adapter.model_id, len(sorted_ids))

    return SliceResult(sorted_ids, scores_dict, order_projections, capture_dates)
