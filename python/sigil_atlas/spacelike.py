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
) -> tuple[np.ndarray, np.ndarray]:
    """Return (targets[N,2], peak_indices[K]).

    targets: softmax-weighted mean of anchors, blended with UMAP by confidence.
    peak_indices: per-attractor, the image index with the highest similarity
                  (the "true" peak of that attractor's neighbourhood).
    """
    n = len(image_ids)
    if attractor_vecs.shape[0] == 0:
        return umap_fallback.copy(), np.empty(0, dtype=np.int64)

    matrix = provider.fetch_matrix(image_ids, model)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-8)

    sims = matrix @ attractor_vecs.T  # (N, K)

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
    confidence = confidence.reshape(-1, 1)

    targets = confidence * gravity_pos + (1.0 - confidence) * umap_fallback

    # Per-attractor peak: image with the highest raw cosine similarity to it.
    peak_indices = np.argmax(sims, axis=0)
    return targets.astype(np.float32), peak_indices.astype(np.int64)


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
    model: str = "clip-vit-l-14",
    feathering: float = 0.5,
    cell_size: float = 100.0,
) -> SpaceLikeLayout:
    n = len(image_ids)
    if n == 0:
        return SpaceLikeLayout([], cell_size, 0, 0, 0.0, 0.0, [])

    attractors = attractors or []
    get_adapter(model)

    logger.info(
        "SpaceLike: n=%d attractors=%d feathering=%.2f model=%s",
        n, len(attractors), feathering, model,
    )

    umap_fallback = _ensure_umap(provider, db, image_ids, model)

    anchors: np.ndarray | None = None
    resolved_attractors: list[Attractor] = []
    peak_indices: np.ndarray | None = None
    if attractors:
        resolved_attractors, attractor_vecs = _resolve_attractor_vectors(attractors, provider, model, image_ids)
        if attractor_vecs.shape[0] > 0:
            anchors = _attractor_anchor_positions(attractor_vecs)
            targets, peak_indices = _gravity_targets(
                provider, image_ids, model,
                attractor_vecs, anchors, umap_fallback, feathering,
            )
        else:
            targets = umap_fallback
    else:
        targets = umap_fallback

    rows, cols = _pick_grid(n)
    logger.info(
        "Grid: %d x %d = %d cells (%d padding duplicates)",
        rows, cols, rows * cols, rows * cols - n,
    )

    sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))
    assignments: list[tuple[int, int, int]] = []
    _recursive_split(list(range(n)), targets, 0, 0, cols, rows, assignments)

    # Per-image elevation from local density in continuous target space.
    # Packed cells are equidistant, but the original 2D targets encode where
    # the field compresses; that compression becomes terrain relief.
    elevation = _local_density(targets)

    positions = [
        CellPosition(id=image_ids[idx], col=col, row=row, elevation=float(elevation[idx]))
        for idx, col, row in assignments
    ]

    # Per-attractor framing: use the grid cell of the peak-similarity image
    # (the real centre of the neighbourhood) rather than the anchor's
    # abstract PCA coordinate, which may land a few cells off the density.
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
