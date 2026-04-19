"""SpaceLike — contiguous square-cell tiling by gravitational attractor field.

Each image is a square. Cells tile the torus without gaps. Each image's target
position is the equilibrium of gravitational pulls from active attractors;
with no attractors, the vision model's native embedding distance projected to
2D via UMAP is the default metric.

Targets are assigned to a uniform grid of square cells via recursive median
split (partition-at-count along alternating axes). O(N log N), gapless, and
proximity-preserving without the seam artifacts of Hilbert-curve matching.

Invariants:
  - !no-gaps: every cell in the rows x cols grid is filled
  - !square-crop: all cells are the same square size (crop handled in shader)
  - !preserves-proximity: images close in contrast-space land close on the grid
"""

import logging
import math
import sys
from dataclasses import dataclass
from typing import Literal

import numpy as np

from sigil_atlas.db import CorpusDB
from sigil_atlas.embedding_provider import EmbeddingProvider
from sigil_atlas.model_registry import get_adapter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Attractor:
    kind: Literal["thing", "target_image"]
    ref: str  # taxonomy term for "thing"; image_id for "target_image"


@dataclass(frozen=True)
class ContrastAxis:
    """A user-named axis of similarity: pole_a minus pole_b in embedding space.

    When contrasts are present, similarity between images is measured in the
    subspace spanned by their directions — making "similar" mean only what the
    user has chosen to care about. Per spec: "unnamed contrasts are in
    superposition and do not contribute."
    """
    pole_a: str
    pole_b: str


@dataclass
class CellPosition:
    id: str
    col: int
    row: int
    elevation: float  # [0, 1]; density in continuous-target space at this image's target


@dataclass
class AttractorPosition:
    kind: Literal["thing", "target_image"]
    ref: str
    col: int
    row: int


@dataclass
class SpaceLikeLayout:
    positions: list[CellPosition]
    cell_size: float
    cols: int
    rows: int
    torus_width: float
    torus_height: float
    attractor_positions: list[AttractorPosition]


# ---------------------------------------------------------------------------
# Feathering: softmax temperature on proximity-to-attractor
# ---------------------------------------------------------------------------

def _feathering_to_temperature(feathering: float) -> float:
    """Map feathering in [0, 1] to softmax temperature over cosine similarity.

    feathering=0 -> sharp (winner-takes-all locally), T ~ 0.05
    feathering=1 -> soft (broad blend), T ~ 1.5
    """
    f = max(0.0, min(1.0, float(feathering)))
    return float(math.exp(math.log(0.05) + f * (math.log(1.5) - math.log(0.05))))


# ---------------------------------------------------------------------------
# Continuous targets
# ---------------------------------------------------------------------------

def _ensure_umap(
    provider: EmbeddingProvider,
    db: CorpusDB,
    image_ids: list[str],
    model: str,
) -> np.ndarray:
    """Return (N, 2) UMAP positions normalized to [0, 1]^2 for the slice."""
    cached = db.fetch_umap_positions(model, image_ids)
    missing = [iid for iid in image_ids if iid not in cached]

    if missing:
        logger.info(
            "Computing UMAP for %d images (model=%s, %d missing)...",
            len(image_ids), model, len(missing),
        )
        import umap as umap_lib
        matrix = provider.fetch_matrix(image_ids, model)
        n = matrix.shape[0]
        nn = min(15, max(2, n - 1))
        reducer = umap_lib.UMAP(
            n_components=2, n_neighbors=nn, min_dist=0.1,
            metric="cosine", random_state=42,
        )
        coords = reducer.fit_transform(matrix).astype(np.float32)
        batch = [(image_ids[i], float(coords[i, 0]), float(coords[i, 1])) for i in range(n)]
        db.insert_umap_batch(model, batch)
        cached = {iid: (float(coords[i, 0]), float(coords[i, 1])) for i, iid in enumerate(image_ids)}

    positions = np.array([cached[iid] for iid in image_ids], dtype=np.float32)
    for axis in range(2):
        lo, hi = float(positions[:, axis].min()), float(positions[:, axis].max())
        span = hi - lo
        if span > 1e-8:
            positions[:, axis] = (positions[:, axis] - lo) / span
        else:
            positions[:, axis] = 0.5
    return positions


def _resolve_attractor_vectors(
    attractors: list[Attractor],
    provider: EmbeddingProvider,
    model: str,
    image_ids: list[str],
) -> tuple[list[Attractor], np.ndarray]:
    """Resolve each attractor to a unit-norm vector in the model's embedding space.

    Returns (resolved_attractors, matrix) where the two lists/rows are aligned.
    Invalid attractors (unknown terms, missing images) are dropped silently.
    """
    if not attractors:
        return [], np.empty((0, 0), dtype=np.float32)

    adapter = get_adapter(model)
    resolved: list[Attractor] = []
    vecs: list[np.ndarray] = []

    for att in attractors:
        if att.kind == "thing":
            from sigil_atlas.things import _find_node
            node = _find_node(att.ref)
            if node is not None:
                prompt = node.prompt
            else:
                # Free-form text fallback — treat the ref as a prompt directly.
                # Useful during exploration before the taxonomy catches up.
                prompt = f"a photograph of {att.ref}"
                logger.info("Attractor '%s' not in taxonomy; using free-text prompt", att.ref)
            vec = adapter.resolve_text_vector(prompt, provider, image_ids)
            resolved.append(att)
            vecs.append(np.asarray(vec, dtype=np.float32))

            # Rich taxonomy: the dropped node has children. Articulate one
            # level down as peer attractors so the field terraces by
            # sub-category. Each child contributes its own root-to-child
            # narrative to the @SpaceLike superposition per the
            # @Arrangement duality. Crude sigils (leaves, no children) fall
            # through unchanged as a single attractor.
            if node is not None and node.children:
                for child in node.children:
                    child_vec = adapter.resolve_text_vector(
                        child.prompt, provider, image_ids,
                    )
                    resolved.append(Attractor(kind="thing", ref=child.name))
                    vecs.append(np.asarray(child_vec, dtype=np.float32))
                logger.info(
                    "Rich taxonomy '%s' articulated with %d children",
                    att.ref, len(node.children),
                )
            continue
        elif att.kind == "target_image":
            matrix = provider.fetch_matrix([att.ref], model)
            vec = matrix[0]
            n = float(np.linalg.norm(vec))
            if n > 1e-8:
                vec = vec / n
        else:
            logger.warning("Unknown attractor kind: %s", att.kind)
            continue
        resolved.append(att)
        vecs.append(np.asarray(vec, dtype=np.float32))

    if not vecs:
        return [], np.empty((0, 0), dtype=np.float32)
    return resolved, np.stack(vecs)


def _attractor_anchor_positions(attractor_vecs: np.ndarray) -> np.ndarray:
    """Anchor (K, 2) positions in [0.15, 0.85]^2, via PCA of attractor vectors.

    One attractor: centered. Two: along a horizontal axis. 3+: 2D PCA.
    The inset keeps attractors off the torus boundary so their neighborhoods
    have room to spread.
    """
    k = attractor_vecs.shape[0]
    if k == 0:
        return np.empty((0, 2), dtype=np.float32)
    if k == 1:
        return np.array([[0.5, 0.5]], dtype=np.float32)

    centered = attractor_vecs - attractor_vecs.mean(axis=0, keepdims=True)

    if k == 2:
        diff = centered[1] - centered[0]
        proj = centered @ diff
        lo, hi = float(proj.min()), float(proj.max())
        t = (proj - lo) / (hi - lo) if hi - lo > 1e-8 else np.array([0.0, 1.0], dtype=np.float32)
        xs = 0.15 + t * 0.70
        ys = np.full(k, 0.5, dtype=np.float32)
        return np.column_stack([xs, ys]).astype(np.float32)

    U, S, _ = np.linalg.svd(centered, full_matrices=False)
    coords = (U[:, :2] * S[:2]).astype(np.float32)
    for dim in range(2):
        lo, hi = float(coords[:, dim].min()), float(coords[:, dim].max())
        span = hi - lo
        if span > 1e-8:
            coords[:, dim] = 0.15 + (coords[:, dim] - lo) / span * 0.70
        else:
            coords[:, dim] = 0.5
    return coords


def _gravity_targets(
    provider: EmbeddingProvider,
    image_ids: list[str],
    model: str,
    attractor_vecs: np.ndarray,
    attractor_anchors: np.ndarray,
    umap_fallback: np.ndarray,
    feathering: float,
    contrast_directions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (targets[N,2], peak_indices[K]).

    targets: softmax-weighted mean of anchors, blended with UMAP by confidence.
    peak_indices: per-attractor, the image index with the highest similarity
                  (the "true" peak of that attractor's neighbourhood).

    When contrast_directions is non-empty, similarity is computed in the
    named-contrast subspace rather than the raw embedding — per the spec,
    "unnamed contrasts are in superposition and do not contribute."
    """
    n = len(image_ids)
    if attractor_vecs.shape[0] == 0:
        return umap_fallback.copy(), np.empty(0, dtype=np.int64)

    matrix = provider.fetch_matrix(image_ids, model)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-8)

    if contrast_directions is not None and contrast_directions.shape[0] > 0:
        # Similarity as negative Euclidean distance in the K-dim contrast
        # subspace. Higher = more similar; equivalent to cosine ordering in
        # the default path because all vectors are unit-normalised, but in
        # contrast space magnitudes matter.
        image_proj = _project(matrix, contrast_directions)
        attractor_proj = _project(attractor_vecs, contrast_directions)
        # (N, A) distance matrix
        diffs = image_proj[:, None, :] - attractor_proj[None, :, :]
        sims = -np.linalg.norm(diffs, axis=2)
    else:
        sims = matrix @ attractor_vecs.T  # cosine on unit-normalised raw embeddings

    T = _feathering_to_temperature(feathering)
    sims_shifted = sims - sims.max(axis=1, keepdims=True)
    exp = np.exp(sims_shifted / T)
    weights = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)

    gravity_pos = weights @ attractor_anchors  # (N, 2)

    max_sim = sims.max(axis=1)
    lo = float(np.percentile(max_sim, 10))
    hi = float(np.percentile(max_sim, 90))
    if hi - lo > 1e-8:
        confidence = np.clip((max_sim - lo) / (hi - lo), 0.0, 1.0)
    else:
        confidence = np.full(n, 0.5, dtype=np.float32)
    # Cap below 1.0 so the most-similar cohort doesn't collapse to a single
    # anchor point. Lower values preserve more UMAP-residual at the peak,
    # which softens the kd-tree partition seams visible at content-cluster
    # boundaries (sky/crowd/cityscape bands in 2-attractor Axis mode). 0.6
    # gives a noticeably smoother field while keeping the polar tension legible.
    PEAK_MAX_BIAS = 0.60
    confidence = (confidence * PEAK_MAX_BIAS).reshape(-1, 1)

    targets = confidence * gravity_pos + (1.0 - confidence) * umap_fallback

    # Per-attractor peak: image with the highest raw cosine similarity to it.
    peak_indices = np.argmax(sims, axis=0)
    return targets.astype(np.float32), peak_indices.astype(np.int64)


# ---------------------------------------------------------------------------
# Contrast space: user-named axes of similarity
# ---------------------------------------------------------------------------

def _contrast_directions(
    contrasts: list[ContrastAxis],
    provider: EmbeddingProvider,
    model: str,
    image_ids: list[str],
) -> np.ndarray:
    """Resolve each contrast to a unit-norm direction in the model's embedding
    space. Direction = (pole_a_vec - pole_b_vec) / ||.||. Invalid poles skipped.

    Returns (K, D) matrix, K <= len(contrasts).
    """
    if not contrasts:
        return np.empty((0, 0), dtype=np.float32)
    adapter = get_adapter(model)
    dirs: list[np.ndarray] = []
    for c in contrasts:
        pole_a = (c.pole_a or "").strip()
        pole_b = (c.pole_b or "").strip()
        if not pole_a or not pole_b:
            continue
        try:
            va = adapter.resolve_text_vector(pole_a, provider, image_ids)
            vb = adapter.resolve_text_vector(pole_b, provider, image_ids)
        except Exception as exc:
            logger.warning("Contrast resolution failed (%s vs %s): %s", pole_a, pole_b, exc)
            continue
        d = va - vb
        n = float(np.linalg.norm(d))
        if n < 1e-8:
            continue
        dirs.append((d / n).astype(np.float32))
    if not dirs:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack(dirs)


def _project(matrix: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Project (N, D) vectors onto (K, D) directions -> (N, K).

    Per spec @Proximity: each image becomes a vector of scalars, one per
    named contrast. Similarity between two images is Euclidean distance in
    this K-dimensional contrast space — magnitudes carry meaning, not just
    direction, so an image strongly aligned to a contrast reads differently
    from a neutral one.
    """
    return (matrix @ directions.T).astype(np.float32)


# ---------------------------------------------------------------------------
# Radial layout (single target_image): rank by similarity, place in rings
# ---------------------------------------------------------------------------

_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


def _radial_cells(cols: int, rows: int, cc: int, rc: int) -> list[tuple[int, int]]:
    """All (col, row) cells in Chebyshev-ring order from (cc, rc), wrap-aware.

    Ring 0 is the centre cell; ring r is the 8r cells at Chebyshev distance r.
    Within each ring, cells are sorted by angle around the centre with a
    per-ring golden-angle rotation offset — this prevents the "most-similar
    fills the top-left first" corner bias that an axis-aligned walk creates,
    so the residual ordering inside a ring distributes rotationally rather
    than smearing into a quadrant gradient. Torus wrap is honoured.
    """
    cc = cc % cols
    rc = rc % rows
    seen: set[tuple[int, int]] = {(cc, rc)}
    order: list[tuple[int, int]] = [(cc, rc)]
    max_r = max(cols, rows)
    total = cols * rows

    for r in range(1, max_r + 1):
        if len(order) >= total:
            break
        # Logical (un-wrapped) offsets covering the ring's perimeter.
        ring: list[tuple[int, int, int, int]] = []
        for d in range(-r, r + 1):
            ring.append((d, -r, (cc + d) % cols, (rc - r) % rows))
            ring.append((d,  r, (cc + d) % cols, (rc + r) % rows))
        for d in range(-r + 1, r):
            ring.append((-r, d, (cc - r) % cols, (rc + d) % rows))
            ring.append(( r, d, (cc + r) % cols, (rc + d) % rows))

        # Angular sort with a per-ring golden-angle rotation. Successive rings
        # start at different angles so any residual "starting cell" bias does
        # not stack into a visible spoke across rings.
        offset = r * _GOLDEN_ANGLE
        ring.sort(key=lambda t: (math.atan2(t[1], t[0]) - offset) % (2 * math.pi))

        for _, _, col, row in ring:
            cell = (col, row)
            if cell in seen:
                continue
            seen.add(cell)
            order.append(cell)
            if len(order) >= total:
                break
    return order


# ---------------------------------------------------------------------------
# Elevation: density of continuous targets reveals where the field compresses
# ---------------------------------------------------------------------------

def _local_density(targets: np.ndarray, radius: float = 0.04) -> np.ndarray:
    """Per-image density in continuous target space [0, 1]^2.

    Fixed-radius neighbor count via scipy cKDTree. O(N log N) overall.
    Normalized to [0, 1]; zero = isolated, one = densest point in the slice.
    """
    n = targets.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    if n == 1:
        return np.ones(1, dtype=np.float32)

    from scipy.spatial import cKDTree
    tree = cKDTree(targets)
    counts = tree.query_ball_point(targets, r=radius, return_length=True)
    counts = np.asarray(counts, dtype=np.float32) - 1.0  # exclude self

    lo = float(counts.min())
    hi = float(counts.max())
    if hi - lo > 1e-8:
        return ((counts - lo) / (hi - lo)).astype(np.float32)
    return np.zeros(n, dtype=np.float32)


# ---------------------------------------------------------------------------
# Grid sizing and recursive median split
# ---------------------------------------------------------------------------

def _pick_grid(n: int) -> tuple[int, int]:
    """(rows, cols) with rows*cols >= n, minimising extras and aspect skew."""
    if n <= 0:
        return (0, 0)
    side = max(1, int(math.ceil(math.sqrt(n))))
    best = (side, side)
    best_score = (side * side - n, abs(side - side))
    for rows in range(max(1, side - 2), side + 3):
        for cols in range(max(1, rows - 2), rows + 3):
            if rows * cols < n:
                continue
            score = (rows * cols - n, abs(rows - cols))
            if score < best_score:
                best = (rows, cols)
                best_score = score
    return best


def _pick_tight_grid(n: int) -> tuple[int, int]:
    """(rows, cols) with rows*cols <= n, near-square, minimising dropped images.

    Used by the tight @field expansion mode: every cell carries a unique
    image, no padding cycle. May drop the least-similar 0-3 images so the
    grid stays close to square. For n with good divisors (perfect squares,
    near-squares) the grid is exact and nothing is dropped.
    """
    if n <= 0:
        return (0, 0)
    side = max(1, int(math.sqrt(n)))
    best: tuple[tuple[int, int], tuple[int, int]] | None = None  # (score, (r,c))
    for rows in range(max(1, side - 3), side + 4):
        for cols in range(rows, rows + 4):
            cells = rows * cols
            if cells <= 0 or cells > n:
                continue
            dropped = n - cells
            score = (dropped, abs(rows - cols))
            if best is None or score < best[0]:
                best = (score, (rows, cols))
    return best[1] if best else (1, 1)


def _recursive_split(
    indices: list[int],
    targets: np.ndarray,
    col_start: int,
    row_start: int,
    cols: int,
    rows: int,
    out: list[tuple[int, int, int]],
) -> None:
    """Assign image indices to cells by partition-at-count along the longer axis.

    Pads with cycled duplicates when the subregion has more cells than points,
    keeping the field gapless at the torus boundary.
    """
    cells = rows * cols
    if cells == 0 or not indices:
        return

    if cells > len(indices):
        pad = cells - len(indices)
        reps = (pad + len(indices) - 1) // len(indices)
        indices = indices + (indices * reps)[:pad]

    if cells == 1:
        out.append((indices[0], col_start, row_start))
        return

    if cols >= rows:
        mid = cols // 2
        left_count = mid * rows
        sorted_idx = sorted(indices, key=lambda i: targets[i, 0])
        _recursive_split(sorted_idx[:left_count], targets, col_start, row_start, mid, rows, out)
        _recursive_split(sorted_idx[left_count:], targets, col_start + mid, row_start, cols - mid, rows, out)
    else:
        mid = rows // 2
        top_count = cols * mid
        sorted_idx = sorted(indices, key=lambda i: targets[i, 1])
        _recursive_split(sorted_idx[:top_count], targets, col_start, row_start, cols, mid, out)
        _recursive_split(sorted_idx[top_count:], targets, col_start, row_start + mid, cols, rows - mid, out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def compute_wireframe_edges(
    provider: EmbeddingProvider,
    image_ids: list[str],
    model: str,
    k: int = 6,
) -> list[tuple[str, str]]:
    """Per-image k-nearest neighbors in embedding space.

    Uses pynndescent (approximate NN) — scales to hundreds of thousands in
    seconds where sklearn brute-force takes minutes. Approximate graph is
    fine for visualization: we want topology, not exactness.
    """
    n = len(image_ids)
    if n <= 1:
        return []

    import time
    from pynndescent import NNDescent

    t0 = time.monotonic()
    matrix = provider.fetch_matrix(image_ids, model)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-8)

    k_actual = min(k + 1, n)  # +1 because a point is its own nearest neighbor
    index = NNDescent(matrix, n_neighbors=k_actual, metric="cosine", n_jobs=-1, random_state=42)
    indices, _ = index.neighbor_graph

    seen: set[tuple[int, int]] = set()
    edges: list[tuple[str, str]] = []
    for i in range(n):
        for j in indices[i, 1:]:  # skip self
            j_int = int(j)
            if j_int == i:
                continue
            a, b = (i, j_int) if i < j_int else (j_int, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            edges.append((image_ids[a], image_ids[b]))

    logger.info("Wireframe k-NN: %d edges over %d images in %.1fs", len(edges), n, time.monotonic() - t0)
    return edges


def compute_spacelike(
    provider: EmbeddingProvider,
    db: CorpusDB,
    image_ids: list[str],
    attractors: list[Attractor] | None = None,
    contrasts: list[ContrastAxis] | None = None,
    model: str = "clip-vit-l-14",
    feathering: float = 0.5,
    cell_size: float = 100.0,
    field_expansion: str = "echo",
    arrangement: str = "rings",
) -> SpaceLikeLayout:
    n = len(image_ids)
    if n == 0:
        return SpaceLikeLayout([], cell_size, 0, 0, 0.0, 0.0, [])

    attractors = attractors or []
    contrasts = contrasts or []
    get_adapter(model)

    logger.info(
        "SpaceLike: n=%d attractors=%d contrasts=%d feathering=%.2f model=%s",
        n, len(attractors), len(contrasts), feathering, model,
    )

    contrast_directions = _contrast_directions(contrasts, provider, model, image_ids)

    # Field expansion modes:
    #   "echo"  — grid >= slice; surplus cells cycle the most-similar prefix.
    #             Creates the moire/repeat-ring visual that emerges from
    #             radial layouts on small slices.
    #   "tight" — grid <= slice; every cell unique. The downstream layouts
    #             naturally drop the overflow: the radial path takes the
    #             top-N most-similar; the gravity-field path drops via the
    #             recursive median split.
    if field_expansion == "tight":
        rows, cols = _pick_tight_grid(n)
        dropped = max(0, n - rows * cols)
        logger.info("Grid (tight): %d x %d = %d cells (drops %d overflow)",
                    rows, cols, rows * cols, dropped)
    else:
        rows, cols = _pick_grid(n)
        logger.info("Grid (echo): %d x %d = %d cells (%d padding duplicates)",
                    rows, cols, rows * cols, rows * cols - n)

    resolved_attractors: list[Attractor] = []
    attractor_vecs: np.ndarray = np.empty((0, 0), dtype=np.float32)
    if attractors:
        resolved_attractors, attractor_vecs = _resolve_attractor_vectors(attractors, provider, model, image_ids)

    # Single-attractor arrangement choice (per `arrangement` parameter):
    #   "rings" — radial layout: target at centre, concentric Chebyshev rings
    #             outward by similarity-to-attractor. Sharp boundaries between
    #             rings; no mutual-similarity within a ring. Good for "what
    #             ranks where" intuition; produces the moiré-on-wrap visuals.
    #   "field" — biased-UMAP deformation: UMAP topology (mutual proximity)
    #             is preserved everywhere; the attractor pulls similar images
    #             toward its anchor proportional to similarity. Per spec:
    #             "continuous field, no disjoint patches, no hard boundaries."
    # Both are spec-coherent reads of @Neighborhood; they're different lenses.
    # Multi-attractor always uses the field path — rings only makes sense for
    # one focal point.
    target_indices = [
        i for i, a in enumerate(resolved_attractors) if a.kind == "target_image"
    ]

    # Axis arrangement (TargetImage only): the target sits at one pole; the
    # opposite pole is the *real corpus image* whose embedding is closest to
    # -target_vec — i.e. the actual most-cosine-opposite picture in the slice.
    # Both poles are real images with real cells. Each image in between is
    # pulled toward whichever pole it's more similar to; the field settles
    # into a smooth gradient along that axis. PCA in
    # _attractor_anchor_positions places the two diametric vectors on
    # opposite sides of the torus.
    if (
        arrangement == "axis"
        and len(target_indices) == 1
        and len(resolved_attractors) == 1
    ):
        target = resolved_attractors[0]
        target_vec = attractor_vecs[0]
        # Find the image whose embedding is closest to -target_vec.
        matrix = provider.fetch_matrix(image_ids, model)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix_n = matrix / np.maximum(norms, 1e-8)
        anti_sims = matrix_n @ (-target_vec).astype(np.float32)
        anti_idx = int(np.argmax(anti_sims))
        anti_id = image_ids[anti_idx]
        anti_vec = matrix_n[anti_idx].astype(np.float32)
        logger.info("Axis: target=%s, antipode=%s (cos=%.3f)",
                    target.ref[:20], anti_id[:20], float(anti_sims[anti_idx]))
        resolved_attractors = [
            target,
            Attractor(kind="target_image", ref=anti_id),
        ]
        attractor_vecs = np.stack([target_vec, anti_vec])
        # Fall through to the gravity-field path with the two real attractors.
    else:
        use_rings = arrangement == "rings" and (
            bool(target_indices) or len(resolved_attractors) == 1
        )
        if use_rings and target_indices:
            idx = target_indices[0]
            return _radial_attractor_layout(
                provider, image_ids, model, cell_size, rows, cols,
                resolved_attractors[idx], attractor_vecs[idx],
                contrast_directions,
            )
        if use_rings and len(resolved_attractors) == 1:
            return _radial_attractor_layout(
                provider, image_ids, model, cell_size, rows, cols,
                resolved_attractors[0], attractor_vecs[0],
                contrast_directions,
            )

    # Gravity-field (biased-UMAP) layout. For single attractor + Field this
    # is a single-pole deformation. For Axis it's a two-pole tension.
    # For multi-attractor it's the only path.
    umap_fallback = _ensure_umap(provider, db, image_ids, model)
    peak_indices: np.ndarray | None = None
    anchors: np.ndarray | None = None
    if resolved_attractors and attractor_vecs.shape[0] > 0:
        anchors = _attractor_anchor_positions(attractor_vecs)
        targets, peak_indices = _gravity_targets(
            provider, image_ids, model,
            attractor_vecs, anchors, umap_fallback, feathering,
            contrast_directions=contrast_directions,
        )
    else:
        targets = umap_fallback

    sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))
    assignments: list[tuple[int, int, int]] = []
    _recursive_split(list(range(n)), targets, 0, 0, cols, rows, assignments)

    elevation = _local_density(targets)
    positions = [
        CellPosition(id=image_ids[idx], col=col, row=row, elevation=float(elevation[idx]))
        for idx, col, row in assignments
    ]

    attractor_positions: list[AttractorPosition] = []
    if resolved_attractors and peak_indices is not None and peak_indices.size > 0:
        idx_to_cell: dict[int, tuple[int, int]] = {img_idx: (col, row) for img_idx, col, row in assignments}
        for att, peak_idx in zip(resolved_attractors, peak_indices.tolist()):
            cell = idx_to_cell.get(int(peak_idx))
            if cell is None:
                continue
            col, row = cell
            attractor_positions.append(AttractorPosition(kind=att.kind, ref=att.ref, col=col, row=row))

    return SpaceLikeLayout(
        positions=positions,
        cell_size=cell_size,
        cols=cols,
        rows=rows,
        torus_width=cols * cell_size,
        torus_height=rows * cell_size,
        attractor_positions=attractor_positions,
    )


def _radial_attractor_layout(
    provider: EmbeddingProvider,
    image_ids: list[str],
    model: str,
    cell_size: float,
    rows: int,
    cols: int,
    attractor: Attractor,
    attractor_vec: np.ndarray,
    contrast_directions: np.ndarray,
) -> SpaceLikeLayout:
    """Place the most-representative image at grid centre, then rank all other
    images by similarity (in contrast subspace if named) and place them in
    concentric Chebyshev rings outward. Works for both `thing` (centre = the
    image with highest cosine to the resolved text vector) and `target_image`
    (centre = the image itself, top of the cosine ranking by construction).
    """
    n = len(image_ids)
    matrix = provider.fetch_matrix(image_ids, model)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-8)

    if contrast_directions.shape[0] > 0:
        image_proj = _project(matrix, contrast_directions)
        target_proj = (contrast_directions @ attractor_vec).astype(np.float32)  # (K,)
        sims = -np.linalg.norm(image_proj - target_proj, axis=1)
    else:
        sims = matrix @ attractor_vec

    # Rank: most similar first.
    order_idx = np.argsort(-sims)

    cc, rc = cols // 2, rows // 2
    cells = _radial_cells(cols, rows, cc, rc)

    # Fill padding cells (if any) by repeating the most similar images last —
    # keeps the field gapless without polluting the near-centre rings.
    if len(order_idx) < len(cells):
        pad = len(cells) - len(order_idx)
        order_idx = np.concatenate([order_idx, order_idx[:pad]])

    assignments: list[tuple[int, int, int]] = []
    for i, idx in enumerate(order_idx[: len(cells)]):
        col, row = cells[i]
        assignments.append((int(idx), col, row))

    # Elevation by rank — target at elev 1.0, falling to 0 at the outermost.
    rank = np.empty(n, dtype=np.float32)
    for i, idx in enumerate(np.argsort(-sims)[:n]):
        rank[int(idx)] = float(i)
    elevation = 1.0 - rank / max(1, n - 1)

    positions = [
        CellPosition(id=image_ids[idx], col=col, row=row, elevation=float(elevation[idx]))
        for idx, col, row in assignments
    ]

    attractor_positions = [
        AttractorPosition(kind=attractor.kind, ref=attractor.ref, col=cc, row=rc)
    ]

    logger.info(
        "Radial layout: %s='%s' at (%d, %d); contrasts=%d",
        attractor.kind, attractor.ref[:20], cc, rc, contrast_directions.shape[0],
    )

    return SpaceLikeLayout(
        positions=positions,
        cell_size=cell_size,
        cols=cols,
        rows=rows,
        torus_width=cols * cell_size,
        torus_height=rows * cell_size,
        attractor_positions=attractor_positions,
    )


