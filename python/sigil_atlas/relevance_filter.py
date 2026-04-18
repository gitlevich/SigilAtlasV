"""RelevanceFilter — the membrane between intent and slice.

A SigilML expression: Boolean composition over atoms. One shape, two faces.
As a query, evaluating it yields the slice — the image_ids satisfying it.
As an invariant, it declares what must hold inside the slice.

Atoms:
    Thing(name)                    — semantic: images depicting the named thing
    TargetImage(id)                — semantic: images near a specific image
    Contrast(pole_a, pole_b, band) — bandpass along a named contrast axis
    Range(dimension, min, max)     — numeric range on a characterization

Composites:
    And(children), Or(children), Not(child)

The Relevance scalar in [0, 1] controls how strictly semantic atoms gate:
    0.0 = loose  (top ~50% by score)
    1.0 = strict (top ~1% by score)

Corpus-size invariant by design — percentile, not knee/sigma.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Union

import numpy as np

from sigil_atlas.db import CorpusDB
from sigil_atlas.embedding_provider import EmbeddingProvider
from sigil_atlas.model_registry import get_adapter

logger = logging.getLogger(__name__)


# --- Atoms -----------------------------------------------------------------

@dataclass(frozen=True)
class Thing:
    name: str


@dataclass(frozen=True)
class TargetImage:
    image_id: str


@dataclass(frozen=True)
class Contrast:
    pole_a: str
    pole_b: str
    band_min: float = -1.0
    band_max: float = 1.0


@dataclass(frozen=True)
class Range:
    dimension: str
    min_value: float
    max_value: float


# --- Composites ------------------------------------------------------------

@dataclass(frozen=True)
class And:
    children: tuple["Expression", ...]


@dataclass(frozen=True)
class Or:
    children: tuple["Expression", ...]


@dataclass(frozen=True)
class Not:
    child: "Expression"


Atom = Union[Thing, TargetImage, Contrast, Range]
Expression = Union[Atom, And, Or, Not]


# --- Parsing from wire format ---------------------------------------------

def parse(node: dict | None) -> Expression | None:
    """Parse a JSON object into an Expression. None or empty And means 'all'."""
    if node is None:
        return None
    t = node.get("type")
    if t == "thing":
        return Thing(name=node["name"])
    if t == "target_image":
        return TargetImage(image_id=node["image_id"])
    if t == "contrast":
        return Contrast(
            pole_a=node["pole_a"],
            pole_b=node["pole_b"],
            band_min=float(node.get("band_min", -1.0)),
            band_max=float(node.get("band_max", 1.0)),
        )
    if t == "range":
        return Range(
            dimension=node["dimension"],
            min_value=float(node["min"]),
            max_value=float(node["max"]),
        )
    if t == "and":
        return And(tuple(parse(c) for c in node.get("children", []) if c is not None))
    if t == "or":
        return Or(tuple(parse(c) for c in node.get("children", []) if c is not None))
    if t == "not":
        child = parse(node.get("child"))
        if child is None:
            return None
        return Not(child)
    raise ValueError(f"Unknown expression type: {t!r}")


# --- Atom extraction (for downstream consumers like SpaceLike) ------------

def thing_atoms(expr: Expression | None) -> list[Thing]:
    """Collect all Thing atoms from the expression, in lexical order."""
    return [a for a in _walk(expr) if isinstance(a, Thing)]


def target_image_atoms(expr: Expression | None) -> list[TargetImage]:
    """Collect all TargetImage atoms."""
    return [a for a in _walk(expr) if isinstance(a, TargetImage)]


def walk(expr: Expression | None):
    """Yield every atom in the expression in depth-first lexical order."""
    if expr is None:
        return
    if isinstance(expr, (Thing, TargetImage, Contrast, Range)):
        yield expr
    elif isinstance(expr, (And, Or)):
        for c in expr.children:
            yield from walk(c)
    elif isinstance(expr, Not):
        yield from walk(expr.child)


_walk = walk  # backwards-compatible alias for internal callers


# --- Evaluation context ----------------------------------------------------

@dataclass
class Context:
    db: CorpusDB
    provider: EmbeddingProvider
    model: str
    relevance: float  # [0, 1]; 0 = loose, 1 = strict
    corpus_ids: list[str] = field(default_factory=list)
    # Cached scores per atom identity to avoid recomputing inside And/Or/Not
    _score_cache: dict[int, np.ndarray] = field(default_factory=dict)


def _relevance_to_keep_fraction(relevance: float) -> float:
    """Map r in [0, 1] to fraction of corpus to keep for a semantic atom.

    Log-linear from 50% (loose) to 1% (strict). Corpus-size invariant by design.
    """
    r = max(0.0, min(1.0, float(relevance)))
    return math.exp(math.log(0.5) + r * (math.log(0.01) - math.log(0.5)))


# --- Atom evaluators -------------------------------------------------------

def _score_thing(ctx: Context, atom: Thing) -> np.ndarray:
    """Cosine scores (N,) for every corpus image against the text prompt.

    Uses the ontology's canonical prompt for the name when available, else a
    free-text prompt — useful during exploration before the taxonomy catches up.
    """
    cached = ctx._score_cache.get(id(atom))
    if cached is not None:
        return cached

    from sigil_atlas.things import _find_node
    node = _find_node(atom.name)
    prompt = node.prompt if node is not None else f"a photograph of {atom.name}"

    adapter = get_adapter(ctx.model)
    vec = adapter.resolve_text_vector(prompt, ctx.provider, ctx.corpus_ids)
    matrix = ctx.provider.fetch_matrix(ctx.corpus_ids, ctx.model)
    scores = (matrix @ vec).astype(np.float32)
    ctx._score_cache[id(atom)] = scores
    return scores


def _score_target_image(ctx: Context, atom: TargetImage) -> np.ndarray:
    cached = ctx._score_cache.get(id(atom))
    if cached is not None:
        return cached

    matrix = ctx.provider.fetch_matrix(ctx.corpus_ids, ctx.model)
    target = ctx.provider.fetch_matrix([atom.image_id], ctx.model)[0]
    norm = float(np.linalg.norm(target))
    if norm > 1e-8:
        target = target / norm
    scores = (matrix @ target).astype(np.float32)
    ctx._score_cache[id(atom)] = scores
    return scores


def _semantic_gate(scores: np.ndarray, ctx: Context) -> np.ndarray:
    """Boolean mask (N,) keeping the top fraction by relevance."""
    n = len(scores)
    if n == 0:
        return np.zeros(0, dtype=bool)
    keep = max(1, int(round(n * _relevance_to_keep_fraction(ctx.relevance))))
    if keep >= n:
        return np.ones(n, dtype=bool)
    threshold = np.partition(scores, n - keep)[n - keep]
    return scores >= threshold


def _eval_thing(ctx: Context, atom: Thing) -> set[str]:
    scores = _score_thing(ctx, atom)
    mask = _semantic_gate(scores, ctx)
    result = {ctx.corpus_ids[i] for i in range(len(mask)) if mask[i]}
    logger.info("Thing '%s': %d/%d pass relevance=%.2f",
                atom.name, len(result), len(ctx.corpus_ids), ctx.relevance)
    return result


def _eval_target_image(ctx: Context, atom: TargetImage) -> set[str]:
    """TargetImage is purely a layout directive, not a slice gate.

    Per spec `Attractor/TargetImage/affordance-point-at`:
      "designate an @image in the current @slice as the @TargetImage.
       A single @neighborhood forms around it."

    The slice already exists (defined by other atoms or the whole corpus);
    the TargetImage just picks the focal point for arrangement. So as a
    filter atom it is a no-op — return the full corpus, which automatically
    satisfies `!in-slice` for the target itself.
    """
    return set(ctx.corpus_ids)


def _eval_contrast(ctx: Context, atom: Contrast) -> set[str]:
    adapter = get_adapter(ctx.model)
    vec_a = adapter.resolve_text_vector(atom.pole_a, ctx.provider, ctx.corpus_ids)
    vec_b = adapter.resolve_text_vector(atom.pole_b, ctx.provider, ctx.corpus_ids)
    matrix = ctx.provider.fetch_matrix(ctx.corpus_ids, ctx.model)
    raw = matrix @ vec_a - matrix @ vec_b
    lo, hi = float(raw.min()), float(raw.max())
    if hi - lo < 1e-9:
        return set(ctx.corpus_ids)
    normalized = ((raw - lo) / (hi - lo) * 2.0 - 1.0).astype(np.float32)
    mask = (normalized >= atom.band_min) & (normalized <= atom.band_max)
    result = {ctx.corpus_ids[i] for i in range(len(mask)) if mask[i]}
    logger.info("Contrast '%s' vs '%s' [%g, %g]: %d/%d pass",
                atom.pole_a, atom.pole_b, atom.band_min, atom.band_max,
                len(result), len(ctx.corpus_ids))
    return result


def _eval_range(ctx: Context, atom: Range) -> set[str]:
    rows = ctx.db._conn.execute(
        "SELECT image_id FROM characterizations "
        "WHERE proximity_name = ? AND value_type = 'range' "
        "AND value_range >= ? AND value_range <= ?",
        (atom.dimension, atom.min_value, atom.max_value),
    ).fetchall()
    result = {r[0] for r in rows}
    corpus = set(ctx.corpus_ids)
    result &= corpus  # never return images outside the visible corpus
    logger.info("Range '%s' [%g, %g]: %d/%d pass",
                atom.dimension, atom.min_value, atom.max_value,
                len(result), len(corpus))
    return result


# --- Top-level evaluation --------------------------------------------------

def evaluate(expr: Expression | None, ctx: Context) -> set[str]:
    """Evaluate the expression to the set of image_ids satisfying it.

    None expression (or an empty And) means 'all of the corpus'.
    """
    if expr is None:
        return set(ctx.corpus_ids)

    if isinstance(expr, Thing):
        return _eval_thing(ctx, expr)
    if isinstance(expr, TargetImage):
        return _eval_target_image(ctx, expr)
    if isinstance(expr, Contrast):
        return _eval_contrast(ctx, expr)
    if isinstance(expr, Range):
        return _eval_range(ctx, expr)

    if isinstance(expr, And):
        if not expr.children:
            return set(ctx.corpus_ids)
        result: set[str] | None = None
        for child in expr.children:
            part = evaluate(child, ctx)
            result = part if result is None else result & part
            if not result:
                return set()
        return result or set()

    if isinstance(expr, Or):
        if not expr.children:
            return set()
        result: set[str] = set()
        for child in expr.children:
            result |= evaluate(child, ctx)
        return result

    if isinstance(expr, Not):
        universe = set(ctx.corpus_ids)
        return universe - evaluate(expr.child, ctx)

    raise TypeError(f"Not an Expression: {type(expr).__name__}")
