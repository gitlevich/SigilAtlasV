"""NeighborhoodMode — controls how images are arranged on the torus.

Invariant: Neighborhood mode controls how images are arranged on the torus.
It never changes which images are in the slice.

Core invariant: Similar images cluster locally on the surface.
Neighborhoods are patches, not strips or axes.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import umap

from sigil_atlas.db import CorpusDB
from sigil_atlas.embedding_provider import EmbeddingProvider

logger = logging.getLogger(__name__)


def _xy_to_hilbert(x: int, y: int, order: int) -> int:
    """Convert (x, y) grid coordinates to a Hilbert curve index.

    Maps a 2D position on a 2^order x 2^order grid to its position along
    the Hilbert curve. Nearby points in 2D map to nearby indices on the curve.
    """
    d = 0
    s = 2 ** (order - 1)
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        # Rotate quadrant
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s //= 2
    return d


@dataclass
class ImagePosition:
    """Position of a single image within a strip."""
    id: str
    x: float
    width: float
    thumbnail_path: str | None = None


@dataclass
class Strip:
    """A horizontal row of images on the torus."""
    y: float
    images: list[ImagePosition] = field(default_factory=list)


@dataclass
class StripLayout:
    """Complete layout: all strips forming the torus surface."""
    strips: list[Strip]
    torus_width: float
    torus_height: float
    strip_height: float


def _fetch_image_dimensions(db: CorpusDB, image_ids: list[str]) -> dict[str, tuple[int, int, str | None]]:
    """Fetch (pixel_width, pixel_height, thumbnail_path) for images."""
    placeholders = ",".join("?" * len(image_ids))
    rows = db._conn.execute(
        f"SELECT id, pixel_width, pixel_height, thumbnail_path FROM images WHERE id IN ({placeholders})",
        image_ids,
    ).fetchall()
    return {r[0]: (r[1] or 512, r[2] or 512, r[3]) for r in rows}


def _build_distance_matrix(
    provider: EmbeddingProvider,
    image_ids: list[str],
    model: str,
    axes: list[str] | None,
    db: CorpusDB | None = None,
) -> np.ndarray:
    """Build the embedding matrix, optionally augmenting with metadata axes."""
    matrix = provider.fetch_matrix(image_ids, model)

    if not axes:
        return matrix

    # Augment with metadata axes if requested
    extra_cols = []

    if db is not None:
        if "time" in axes:
            rows = db._conn.execute(
                f"SELECT id, capture_date FROM images WHERE id IN ({','.join('?' * len(image_ids))})",
                image_ids,
            ).fetchall()
            dates = {r[0]: r[1] or 0.0 for r in rows}
            col = np.array([dates.get(iid, 0.0) for iid in image_ids], dtype=np.float32)
            if col.std() > 0:
                col = (col - col.mean()) / col.std()
            extra_cols.append(col.reshape(-1, 1))

        if "location" in axes:
            rows = db._conn.execute(
                f"SELECT id, gps_latitude, gps_longitude FROM images WHERE id IN ({','.join('?' * len(image_ids))})",
                image_ids,
            ).fetchall()
            lats = {r[0]: r[1] or 0.0 for r in rows}
            lons = {r[0]: r[2] or 0.0 for r in rows}
            lat_col = np.array([lats.get(iid, 0.0) for iid in image_ids], dtype=np.float32)
            lon_col = np.array([lons.get(iid, 0.0) for iid in image_ids], dtype=np.float32)
            for col in [lat_col, lon_col]:
                if col.std() > 0:
                    col[:] = (col - col.mean()) / col.std()
            extra_cols.append(np.column_stack([lat_col, lon_col]))

    if extra_cols:
        # Weight metadata axes to be comparable to embedding dimensions
        embedding_scale = np.std(matrix)
        extras = np.hstack(extra_cols) * embedding_scale
        matrix = np.hstack([matrix, extras])

    return matrix


def _pack_strip(
    items: list[tuple[str, float]],
    strip_idx: int,
    strip_height: float,
    torus_width: float,
    dims: dict[str, tuple[int, int, str | None]],
) -> Strip:
    """Pack images into a strip, scaling widths to fill torus_width exactly."""
    natural_total = sum(w for _, w in items)
    scale = torus_width / natural_total if natural_total > 0 else 1.0

    images = []
    cx = 0.0
    for iid, w in items:
        scaled_w = w * scale
        thumb = dims.get(iid, (512, 512, None))[2]
        images.append(ImagePosition(id=iid, x=cx, width=scaled_w, thumbnail_path=thumb))
        cx += scaled_w
    return Strip(y=strip_idx * strip_height, images=images)


def compute_layout(
    provider: EmbeddingProvider,
    db: CorpusDB,
    image_ids: list[str],
    axes: list[str] | None = None,
    tightness: float = 0.5,
    model: str = "clip-vit-b-32",
    strip_height: float = 100.0,
) -> StripLayout:
    """Compute torus layout using UMAP for 2D projection, then strip assignment.

    Torus dimensions are derived from the images (!gapless) — not arbitrary.
    torus_width = max strip length, torus_height = n_strips * strip_height.

    Args:
        provider: Embedding source
        db: Database for image metadata
        image_ids: Images to lay out (the current slice)
        axes: Which dimensions define similarity (None = raw embedding distance)
        tightness: 0.0 (tight clusters) to 1.0 (loose, bleeding neighborhoods)
        model: Vision model identifier
        strip_height: Height of each horizontal strip
    """
    n = len(image_ids)
    if n == 0:
        return StripLayout(strips=[], torus_width=0, torus_height=0, strip_height=strip_height)

    dims = _fetch_image_dimensions(db, image_ids)

    if n == 1:
        iid = image_ids[0]
        pw, ph, thumb = dims.get(iid, (512, 512, None))
        w = strip_height * (pw / ph)
        strip = Strip(y=0.0, images=[ImagePosition(id=iid, x=0.0, width=w, thumbnail_path=thumb)])
        return StripLayout(strips=[strip], torus_width=w, torus_height=strip_height, strip_height=strip_height)

    # 1. Compute image display widths
    img_widths = {}
    for iid in image_ids:
        pw, ph, _ = dims.get(iid, (512, 512, None))
        img_widths[iid] = strip_height * (pw / ph)

    total_width = sum(img_widths.values())

    # 2. Determine number of strips for a roughly square torus
    n_strips = max(1, round(np.sqrt(total_width / strip_height)))
    torus_width = total_width / n_strips

    # 3. UMAP to 2D — preserves neighborhoods as patches (!local-neighborhoods)
    matrix = _build_distance_matrix(provider, image_ids, model, axes, db)
    min_dist = max(0.0, min(0.99, tightness))
    n_neighbors = min(15, max(2, n - 1))

    logger.info("UMAP 2D: %d images, min_dist=%.2f, n_neighbors=%d",
                n, min_dist, n_neighbors)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=42,
    )
    points_2d = reducer.fit_transform(matrix)

    # 4. Hilbert curve ordering: convert 2D UMAP positions into a single
    #    continuous path that preserves 2D locality. Then fill strips
    #    sequentially from this path — the strips are just where the
    #    path wraps around the torus.
    #
    #    This is the correct way to linearize a 2D surface while preserving
    #    neighborhood structure (!local-neighborhoods: patches, not strips).

    # Normalize UMAP to [0, 1]
    x_vals = points_2d[:, 0]
    y_vals = points_2d[:, 1]
    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()
    x_range = max(x_max - x_min, 1e-8)
    y_range = max(y_max - y_min, 1e-8)
    norm_x = (x_vals - x_min) / x_range
    norm_y = (y_vals - y_min) / y_range

    # Map to Hilbert curve grid and compute indices
    hilbert_order = 8  # 256x256 grid — sufficient resolution for 5K+ images
    grid_size = 2 ** hilbert_order
    gx = np.clip((norm_x * (grid_size - 1)).astype(int), 0, grid_size - 1)
    gy = np.clip((norm_y * (grid_size - 1)).astype(int), 0, grid_size - 1)
    hilbert_indices = np.array([_xy_to_hilbert(int(gx[i]), int(gy[i]), hilbert_order) for i in range(n)])

    # Sort by Hilbert index — this is the continuous path through 2D space
    order = np.argsort(hilbert_indices)

    # 5. Fill strips sequentially from the Hilbert path (!gapless).
    #    Each strip is scaled so it fills exactly torus_width.
    strips: list[Strip] = []
    current_items: list[tuple[str, float]] = []  # (id, width)
    current_width = 0.0

    for idx in order:
        iid = image_ids[idx]
        w = img_widths[iid]
        current_items.append((iid, w))
        current_width += w

        if current_width >= torus_width:
            strips.append(_pack_strip(current_items, len(strips), strip_height, torus_width, dims))
            current_items = []
            current_width = 0.0

    # Last strip
    if current_items:
        if strips and current_width < torus_width * 0.5:
            # Merge into previous
            prev_items = [(img.id, img.width) for img in strips[-1].images]
            prev_items.extend(current_items)
            strips[-1] = _pack_strip(prev_items, len(strips) - 1, strip_height, torus_width, dims)
        else:
            strips.append(_pack_strip(current_items, len(strips), strip_height, torus_width, dims))

    torus_height = len(strips) * strip_height

    logger.info("Layout: %d images -> %d strips, torus=%.0fx%.0f",
                n, len(strips), torus_width, torus_height)

    return StripLayout(
        strips=strips,
        torus_width=torus_width,
        torus_height=torus_height,
        strip_height=strip_height,
    )
