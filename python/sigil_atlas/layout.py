"""Layout — arranges images on a rectangle.

Two modes:
  SpaceLike (strips): UMAP → Hilbert → strip packing. For CLIP models.
  Treemap: hierarchical KMeans clusters → squarified rectangles. For DINOv2.

Invariants:
  - Similar images cluster locally as patches (!local-neighborhoods)
  - No gaps (!gapless) — each strip's images tile edge-to-edge
  - No overlaps (!no-overlaps)
  - Aspect ratio preserved (!respect-image-aspect-ratio, !no-crop)
  - Height uniform within each strip; slight variation across strips (!fixed-height)
"""

import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from sigil_atlas.db import CorpusDB
from sigil_atlas.embedding_provider import EmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass
class ImagePosition:
    id: str
    x: float
    width: float
    thumbnail_path: str | None = None


@dataclass
class Strip:
    y: float
    height: float
    images: list[ImagePosition] = field(default_factory=list)


@dataclass
class StripLayout:
    strips: list[Strip]
    torus_width: float
    torus_height: float
    strip_height: float  # nominal height (actual per-strip height may vary slightly)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_image_dimensions(db: CorpusDB, image_ids: list[str]) -> dict[str, tuple[int, int, str | None]]:
    rows = db._query_in_batches(
        "SELECT id, pixel_width, pixel_height, thumbnail_path FROM images WHERE id IN ({placeholders})",
        image_ids,
    )
    return {r[0]: (r[1] or 512, r[2] or 512, r[3]) for r in rows}


def _build_embedding_matrix(
    provider: EmbeddingProvider, image_ids: list[str], model: str,
    axes: list[str] | None, db: CorpusDB | None,
) -> np.ndarray:
    matrix = provider.fetch_matrix(image_ids, model)
    if not axes or db is None:
        return matrix

    extra = []
    if "time" in axes:
        rows = db._query_in_batches(
            "SELECT id, capture_date FROM images WHERE id IN ({placeholders})",
            image_ids,
        )
        col = np.array([dict(rows).get(iid, 0.0) or 0.0 for iid in image_ids], dtype=np.float32)
        if col.std() > 0:
            col = (col - col.mean()) / col.std()
        extra.append(col.reshape(-1, 1))

    if "location" in axes:
        rows = db._query_in_batches(
            "SELECT id, gps_latitude, gps_longitude FROM images WHERE id IN ({placeholders})",
            image_ids,
        )
        lat = np.array([dict([(r[0], r[1] or 0.0) for r in rows]).get(iid, 0.0) for iid in image_ids], dtype=np.float32)
        lon = np.array([dict([(r[0], r[2] or 0.0) for r in rows]).get(iid, 0.0) for iid in image_ids], dtype=np.float32)
        for c in [lat, lon]:
            if c.std() > 0:
                c[:] = (c - c.mean()) / c.std()
        extra.append(np.column_stack([lat, lon]))

    if extra:
        scale = np.std(matrix)
        matrix = np.hstack([matrix, np.hstack(extra) * scale])
    return matrix


# ---------------------------------------------------------------------------
# Stage 1: UMAP projection
# ---------------------------------------------------------------------------

def _umap_positions(matrix: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    """Project embedding matrix to 2D positions in [0, 1]^2."""
    import umap as umap_lib
    n = matrix.shape[0]
    nn = min(n_neighbors, max(2, n - 1))
    reducer = umap_lib.UMAP(
        n_components=2, n_neighbors=nn, min_dist=min_dist,
        metric="cosine", random_state=42,
    )
    pos = reducer.fit_transform(matrix)
    # Normalise to [0, 1]
    for axis in range(2):
        lo, hi = pos[:, axis].min(), pos[:, axis].max()
        span = hi - lo
        if span > 1e-8:
            pos[:, axis] = (pos[:, axis] - lo) / span
        else:
            pos[:, axis] = 0.5
    return pos


# ---------------------------------------------------------------------------
# Stage 2: Hilbert curve linearisation
# ---------------------------------------------------------------------------

def _xy_to_hilbert(n: int, x: int, y: int) -> int:
    """Convert (x, y) grid coordinates to Hilbert curve index. n must be a power of 2."""
    d = 0
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s //= 2
    return d


def _hilbert_order(positions: np.ndarray, order: int = 10) -> np.ndarray:
    """Return indices that sort images by Hilbert curve traversal of their 2D positions."""
    n = 2 ** order
    xi = np.clip((positions[:, 0] * (n - 1)).astype(int), 0, n - 1)
    yi = np.clip((positions[:, 1] * (n - 1)).astype(int), 0, n - 1)
    hilbert_indices = np.array([
        _xy_to_hilbert(n, int(xi[i]), int(yi[i]))
        for i in range(len(positions))
    ])
    return np.argsort(hilbert_indices)


# ---------------------------------------------------------------------------
# Stage 3: Greedy strip packing
# ---------------------------------------------------------------------------

def _greedy_pack_strips(
    sorted_ids: list[str],
    natural_widths: dict[str, float],
    thumbnails: dict[str, str | None],
    nominal_height: float,
    target_width: float,
) -> list[Strip]:
    """Pack sorted images into strips with proportional scaling.

    Greedy packing along the Hilbert-sorted sequence: accumulate images
    until adding the next would exceed target_width, then choose whichever
    break point (before or after) is closer to target. Each strip is then
    uniformly scaled so its total width equals target_width exactly.

    The per-strip height adjusts proportionally: h_strip = nominal * scale.
    """
    # Build the sequence of (id, natural_width) pairs
    items = [(iid, natural_widths[iid]) for iid in sorted_ids]
    n = len(items)
    if n == 0:
        return []

    # Find optimal break points using dynamic programming on cumulative widths.
    # For large N this is O(N) with the greedy heuristic: split whenever the
    # cumulative width crosses the nearest multiple of target_width.
    cum = np.zeros(n + 1)
    for i, (_, w) in enumerate(items):
        cum[i + 1] = cum[i] + w

    total = cum[n]
    n_strips_est = max(1, round(total / target_width))

    # Find break points: for each strip k, find the image index where
    # cumulative width is closest to k * target_width
    breaks = [0]
    for k in range(1, n_strips_est):
        boundary = k * target_width
        # Binary search for closest index
        idx = int(np.searchsorted(cum, boundary))
        idx = max(breaks[-1] + 1, min(n, idx))
        # Check if idx-1 or idx is closer
        if idx > 0 and idx <= n:
            before = abs(cum[idx - 1] - boundary)
            after = abs(cum[idx] - boundary) if idx < n else float('inf')
            if before < after and idx - 1 > breaks[-1]:
                idx = idx - 1
        breaks.append(idx)
    breaks.append(n)

    # Remove duplicate or empty breaks
    breaks = sorted(set(breaks))

    # Build strips
    strips: list[Strip] = []
    for b in range(len(breaks) - 1):
        start, end = breaks[b], breaks[b + 1]
        if start >= end:
            continue
        group = items[start:end]
        width_sum = sum(w for _, w in group)
        scale = target_width / width_sum if width_sum > 0 else 1.0
        strip_h = nominal_height * scale
        y = sum(s.height for s in strips)
        images = []
        x = 0.0
        for iid, nw in group:
            w = nw * scale
            images.append(ImagePosition(id=iid, x=x, width=w, thumbnail_path=thumbnails.get(iid)))
            x += w
        strips.append(Strip(y=y, height=strip_h, images=images))

    return strips


# ---------------------------------------------------------------------------
# Treemap layout — hierarchical KMeans → squarified rectangles
# ---------------------------------------------------------------------------

TREEMAP_K_LEVELS = [10, 50, 200, 500, 1000]
TREEMAP_GAPS = [0.5, 0.25, 0.1, 0.0, 0.0]  # gap as fraction of strip_height per level


@dataclass
class _TreeNode:
    """Node in the cluster hierarchy."""
    cluster_id: int
    level: int
    image_ids: list[str] = field(default_factory=list)
    children: list["_TreeNode"] = field(default_factory=list)


def _build_hierarchy(
    db: CorpusDB,
    image_ids: list[str],
    model: str,
    k_levels: list[int],
) -> _TreeNode:
    """Build a cluster tree from precomputed KMeans at multiple k-levels.

    For each pair of adjacent levels (coarse, fine), assign each fine cluster
    to whichever coarse cluster contains the majority of its members.
    """
    # Filter to k-levels that actually exist in DB
    available = [k for k in k_levels if db.has_kmeans(model, k)]
    if not available:
        # No clustering — single root with all images
        return _TreeNode(cluster_id=0, level=0, image_ids=image_ids)

    # Fetch assignments at all levels
    assignments: dict[int, dict[str, int]] = {}
    for k in available:
        assignments[k] = db.fetch_kmeans_assignments_for_ids(model, k, image_ids)

    # Build leaf nodes from finest level
    finest_k = available[-1]
    finest = assignments[finest_k]
    leaf_groups: dict[int, list[str]] = defaultdict(list)
    for iid in image_ids:
        cid = finest.get(iid, 0)
        leaf_groups[cid].append(iid)

    leaves = {
        cid: _TreeNode(cluster_id=cid, level=len(available) - 1, image_ids=ids)
        for cid, ids in leaf_groups.items()
    }

    # Build upward: for each level pair, group fine clusters into coarse ones
    current_nodes = leaves
    for level_idx in range(len(available) - 2, -1, -1):
        coarse_k = available[level_idx]
        coarse_assignments = assignments[coarse_k]

        # For each current node, vote on which coarse cluster it belongs to
        parent_groups: dict[int, list[_TreeNode]] = defaultdict(list)
        for node in current_nodes.values():
            votes: Counter = Counter()
            for iid in node.image_ids:
                votes[coarse_assignments.get(iid, 0)] += 1
            parent_cid = votes.most_common(1)[0][0]
            parent_groups[parent_cid].append(node)

        # Create parent nodes
        current_nodes = {}
        for cid, children in parent_groups.items():
            all_ids = []
            for child in children:
                all_ids.extend(child.image_ids)
            parent = _TreeNode(
                cluster_id=cid, level=level_idx,
                image_ids=all_ids, children=children,
            )
            current_nodes[cid] = parent

    # Root
    root = _TreeNode(
        cluster_id=-1, level=-1,
        image_ids=image_ids,
        children=list(current_nodes.values()),
    )
    return root


def _squarify(
    children: list[tuple[float, any]],
    x: float, y: float, w: float, h: float,
) -> list[tuple[any, float, float, float, float]]:
    """Squarified treemap layout (Bruls et al. 2000).

    Input: [(area, data), ...] sorted by area descending.
    Output: [(data, x, y, w, h), ...]
    """
    if not children:
        return []

    total_area = sum(a for a, _ in children)
    if total_area <= 0:
        return [(d, x, y, w / len(children), h) for _, d in children]

    # Normalize areas to fill the rectangle
    scale = (w * h) / total_area
    items = [(a * scale, d) for a, d in children]

    result = []
    _squarify_recursive(items, x, y, w, h, result)
    return result


def _squarify_recursive(
    items: list[tuple[float, any]],
    x: float, y: float, w: float, h: float,
    result: list,
) -> None:
    if not items:
        return
    if len(items) == 1:
        result.append((items[0][1], x, y, w, h))
        return

    # Lay out along the shorter side
    if w >= h:
        _layout_row(items, x, y, w, h, True, result)
    else:
        _layout_row(items, x, y, w, h, False, result)


def _layout_row(
    items: list[tuple[float, any]],
    x: float, y: float, w: float, h: float,
    horizontal: bool,
    result: list,
) -> None:
    """Lay items in a row along the shorter side, optimizing aspect ratios."""
    side = h if horizontal else w

    row = [items[0]]
    row_area = items[0][0]
    best_worst = _worst_ratio(row, row_area, side)

    for i in range(1, len(items)):
        new_row = row + [items[i]]
        new_area = row_area + items[i][0]
        new_worst = _worst_ratio(new_row, new_area, side)

        if new_worst <= best_worst:
            row = new_row
            row_area = new_area
            best_worst = new_worst
        else:
            break

    # Place this row
    consumed = len(row)
    row_span = row_area / side if side > 0 else 0

    pos = 0.0
    for area, data in row:
        frac = area / row_area if row_area > 0 else 1.0 / len(row)
        if horizontal:
            result.append((data, x, y + pos * side, row_span, frac * side))
            pos += frac
        else:
            result.append((data, x + pos * side, y, frac * side, row_span))
            pos += frac

    # Recurse on remaining items in the leftover rectangle
    remaining = items[consumed:]
    if remaining:
        if horizontal:
            _squarify_recursive(remaining, x + row_span, y, w - row_span, h, result)
        else:
            _squarify_recursive(remaining, x, y + row_span, w, h - row_span, result)


def _worst_ratio(row: list[tuple[float, any]], total: float, side: float) -> float:
    """Worst aspect ratio in a row. Lower is better (closer to square)."""
    if side <= 0 or total <= 0:
        return float("inf")
    s2 = side * side
    rmax = max(a for a, _ in row)
    rmin = min(a for a, _ in row)
    return max(s2 * rmax / (total * total), total * total / (s2 * rmin))


def _treemap_to_strips(
    node: _TreeNode,
    rx: float, ry: float, rw: float, rh: float,
    strip_height: float,
    natural_widths: dict[str, float],
    thumbnails: dict[str, str | None],
    gap_fractions: list[float],
    strips: list[Strip],
) -> None:
    """Recursively lay out tree nodes as squarified rectangles, packing strips at leaves."""
    if not node.children:
        # Leaf: pack images into strips within this rectangle
        if not node.image_ids or rw <= 0 or rh <= 0:
            return
        leaf_strips = _greedy_pack_strips(
            node.image_ids, natural_widths, thumbnails,
            strip_height, rw,
        )
        # Position strips within the rectangle
        for s in leaf_strips:
            s.y += ry
            for img in s.images:
                img.x += rx
            strips.append(s)
        return

    # Interior node: subdivide rectangle among children
    level = max(0, node.level + 1)
    gap = strip_height * (gap_fractions[level] if level < len(gap_fractions) else 0.0)

    # Sort children by image count descending (squarify works best this way)
    sorted_children = sorted(node.children, key=lambda c: len(c.image_ids), reverse=True)
    child_areas = [(float(len(c.image_ids)), c) for c in sorted_children]

    placements = _squarify(child_areas, rx, ry, rw, rh)

    for child, cx, cy, cw, ch in placements:
        # Inset by gap
        inset_x = cx + gap
        inset_y = cy + gap
        inset_w = cw - 2 * gap
        inset_h = ch - 2 * gap
        if inset_w <= 0 or inset_h <= 0:
            inset_x, inset_y, inset_w, inset_h = cx, cy, cw, ch
        _treemap_to_strips(
            child, inset_x, inset_y, inset_w, inset_h,
            strip_height, natural_widths, thumbnails, gap_fractions, strips,
        )


def compute_treemap_layout(
    provider: EmbeddingProvider, db: CorpusDB, image_ids: list[str],
    model: str = "dinov2-vitb14", strip_height: float = 100.0,
    k_levels: list[int] | None = None,
) -> StripLayout:
    """Treemap layout: hierarchical KMeans → squarified rectangles → strip-packed leaves."""
    k_levels = k_levels or TREEMAP_K_LEVELS
    n = len(image_ids)
    if n == 0:
        return StripLayout([], 0, 0, strip_height)

    dims = _fetch_image_dimensions(db, image_ids)
    natural_widths = {iid: strip_height * (dims[iid][0] / dims[iid][1]) for iid in image_ids}
    thumbnails = {iid: dims[iid][2] for iid in image_ids}

    # Total area at nominal strip height
    total_width = sum(natural_widths.values())
    total_area = total_width * strip_height
    side = math.sqrt(total_area)
    # Aim for a roughly square torus
    torus_width = side * 1.2  # slightly wider than tall
    torus_height = total_area / torus_width

    logger.info("Treemap: %d images, %d k-levels, surface %.0f x %.0f", n, len(k_levels), torus_width, torus_height)

    # Build hierarchy
    root = _build_hierarchy(db, image_ids, model, k_levels)

    # Lay out into strips
    strips: list[Strip] = []
    _treemap_to_strips(
        root, 0, 0, torus_width, torus_height,
        strip_height, natural_widths, thumbnails, TREEMAP_GAPS, strips,
    )

    # Sort strips by y for binary search in the renderer
    strips.sort(key=lambda s: s.y)

    actual_height = max((s.y + s.height for s in strips), default=0)
    logger.info("Treemap: %d strips, %.0f x %.0f", len(strips), torus_width, actual_height)

    return StripLayout(strips, torus_width, actual_height, strip_height)


# ---------------------------------------------------------------------------
# UMAP + Hilbert layout (for CLIP models)
# ---------------------------------------------------------------------------

def _ensure_umap_cached(
    provider: EmbeddingProvider, db: CorpusDB,
    image_ids: list[str], model: str,
) -> dict[str, tuple[float, float]]:
    """Return cached UMAP positions, computing if missing."""
    cached = db.fetch_umap_positions(model, image_ids)
    missing = [iid for iid in image_ids if iid not in cached]

    if not missing:
        return cached

    # Compute UMAP for all images (not just missing — positions are relative)
    logger.info("Computing UMAP for %d images (model=%s)...", len(image_ids), model)
    matrix = provider.fetch_matrix(image_ids, model)
    positions = _umap_positions(matrix, n_neighbors=min(15, max(2, len(image_ids) - 1)))

    batch = [
        (image_ids[i], float(positions[i, 0]), float(positions[i, 1]))
        for i in range(len(image_ids))
    ]
    db.insert_umap_batch(model, batch)
    logger.info("UMAP positions cached for %d images", len(image_ids))

    return {iid: (x, y) for iid, x, y in batch}


def compute_layout(
    provider: EmbeddingProvider, db: CorpusDB, image_ids: list[str],
    axes: list[str] | None = None, tightness: float = 0.5,
    model: str = "clip-vit-l-14", strip_height: float = 100.0,
    preserve_order: bool = False,
    order_values: dict[str, float] | None = None,
    layout_mode: str = "auto",
) -> StripLayout:
    # Auto-select treemap for models without a text encoder (DINOv2)
    use_treemap = layout_mode == "treemap"
    if layout_mode == "auto":
        from sigil_atlas.model_registry import get_adapter
        try:
            adapter = get_adapter(model)
            use_treemap = not adapter.supports_text
        except ValueError:
            pass

    n = len(image_ids)
    if n == 0:
        return StripLayout([], 0, 0, strip_height)

    if use_treemap and n > 1 and not preserve_order and not order_values:
        return compute_treemap_layout(provider, db, image_ids, model, strip_height)

    dims = _fetch_image_dimensions(db, image_ids)

    if n == 1:
        iid = image_ids[0]
        pw, ph, thumb = dims.get(iid, (512, 512, None))
        w = strip_height * (pw / ph)
        strip = Strip(y=0, height=strip_height, images=[ImagePosition(iid, 0, w, thumb)])
        return StripLayout([strip], w, strip_height, strip_height)

    # Natural widths at nominal strip height
    natural_widths = {iid: strip_height * (dims[iid][0] / dims[iid][1]) for iid in image_ids}
    thumbnails = {iid: dims[iid][2] for iid in image_ids}
    total_width = sum(natural_widths.values())

    # Target: roughly square rectangle
    n_strips = max(1, round(np.sqrt(total_width / strip_height)))
    target_width = total_width / n_strips

    mode = "ordered" if order_values else ("scored" if preserve_order else "spacelike")
    logger.info("Layout: %d images, %s", n, mode)

    if order_values:
        # Sort by order values (capture date or contrast projection)
        sorted_ids = sorted(image_ids, key=lambda iid: order_values.get(iid, 0.0))
    elif preserve_order:
        # Slice already ordered by score — use that order directly
        sorted_ids = image_ids
    else:
        # Stage 1: cached UMAP positions
        pos_dict = _ensure_umap_cached(provider, db, image_ids, model)
        positions = np.array([pos_dict[iid] for iid in image_ids], dtype=np.float32)
        # Stage 2: Hilbert sort
        order = _hilbert_order(positions)
        sorted_ids = [image_ids[i] for i in order]

    # Stage 3: Greedy strip pack
    strips = _greedy_pack_strips(sorted_ids, natural_widths, thumbnails, strip_height, target_width)

    torus_width = target_width
    torus_height = sum(s.height for s in strips)
    logger.info("Torus: %d strips, %.0f x %.0f", len(strips), torus_width, torus_height)

    return StripLayout(strips, torus_width, torus_height, strip_height)
