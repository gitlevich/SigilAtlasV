"""Slice — the image_ids surviving the RelevanceFilter, plus an optional
ordering projection for TimeLike.

The slice is the result of evaluating the @RelevanceFilter expression against
the @corpus. Its distinctness — what is present and what is not — is entirely
a function of that expression, per the spec's
`Neighborhood/language.md`:

    "Distinctness of neighborhoods comes from the @slice's @relevanceFilter,
     not from spatial separation."

TimeLike ordering is a separate concern and enters through `order_contrast`
(a `Contrast` atom used only for projection, not for gating).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from sigil_atlas.db import CorpusDB
from sigil_atlas.embedding_provider import EmbeddingProvider
from sigil_atlas.model_registry import get_adapter
from sigil_atlas.relevance_filter import Context, Contrast, Expression, evaluate

logger = logging.getLogger(__name__)


@dataclass
class SliceResult:
    image_ids: list[str]
    order_projections: dict[str, float] | None
    capture_dates: dict[str, float]


def compute_slice(
    db: CorpusDB,
    provider: EmbeddingProvider,
    filter_expr: Expression | None,
    model: str,
    relevance: float = 0.5,
    order_contrast: Contrast | None = None,
) -> SliceResult:
    """Evaluate the RelevanceFilter and optionally project survivors onto an
    ordering axis.
    """
    corpus_ids = db.fetch_image_ids()
    ctx = Context(
        db=db,
        provider=provider,
        model=model,
        relevance=relevance,
        corpus_ids=corpus_ids,
    )

    survivors = evaluate(filter_expr, ctx)
    # Preserve the corpus enumeration order for stability across recomputes.
    image_ids = [iid for iid in corpus_ids if iid in survivors]

    order_projections = None
    if order_contrast is not None and image_ids:
        order_projections = _project_onto_contrast(
            provider, model, image_ids, order_contrast
        )

    capture_dates = db.fetch_capture_dates(image_ids)

    logger.info("Slice [%s]: %d images", model, len(image_ids))
    return SliceResult(
        image_ids=image_ids,
        order_projections=order_projections,
        capture_dates=capture_dates,
    )


def _project_onto_contrast(
    provider: EmbeddingProvider,
    model: str,
    image_ids: list[str],
    contrast: Contrast,
) -> dict[str, float]:
    adapter = get_adapter(model)
    vec_a = adapter.resolve_text_vector(contrast.pole_a, provider, image_ids)
    vec_b = adapter.resolve_text_vector(contrast.pole_b, provider, image_ids)
    matrix = provider.fetch_matrix(image_ids, model)
    raw = matrix @ vec_a - matrix @ vec_b
    lo, hi = float(raw.min()), float(raw.max())
    if hi - lo < 1e-9:
        return {iid: 0.0 for iid in image_ids}
    projs = ((raw - lo) / (hi - lo) * 2.0 - 1.0).astype(np.float32)
    return {image_ids[i]: float(projs[i]) for i in range(len(image_ids))}
