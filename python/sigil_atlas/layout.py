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

    # 4. Map UMAP 2D positions directly onto the torus surface.
    #    Both axes of the UMAP output map to torus coordinates, preserving
    #    2D neighborhood structure (!local-neighborhoods: patches, not strips).
    #
    #    Normalize UMAP x -> [0, torus_width], UMAP y -> strip index.
    #    Each image's torus x comes from its UMAP x position proportionally.
    #    This keeps horizontal neighbors in UMAP space as horizontal neighbors
    #    on the torus, not just sorted-then-packed.

    # Normalize UMAP coordinates to [0, 1]
    x_vals = points_2d[:, 0]
    y_vals = points_2d[:, 1]
    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()
    x_range = max(x_max - x_min, 1e-8)
    y_range = max(y_max - y_min, 1e-8)

    norm_x = (x_vals - x_min) / x_range  # 0..1
    norm_y = (y_vals - y_min) / y_range  # 0..1

    # Assign to strips by UMAP y, filling greedily to balance widths
    order = np.argsort(norm_y)

    strip_bins: list[list[tuple[float, str, float]]] = []  # [(norm_x, id, width)]
    current_bin: list[tuple[float, str, float]] = []
    current_width = 0.0

    for idx in order:
        iid = image_ids[idx]
        w = img_widths[iid]
        current_bin.append((float(norm_x[idx]), iid, w))
        current_width += w

        if current_width >= torus_width:
            strip_bins.append(current_bin)
            current_bin = []
            current_width = 0.0

    # Last bin
    if current_bin:
        if strip_bins and current_width < torus_width * 0.5:
            strip_bins[-1].extend(current_bin)
        else:
            strip_bins.append(current_bin)

    # 5. Within each strip, place images at their UMAP x position on the torus.
    #    Sort by UMAP x, then position proportionally across torus_width.
    #    Each image is placed at its proportional x, and widths are scaled
    #    so images collectively fill the strip exactly (!gapless).
    strips: list[Strip] = []
    for s_idx, bin_items in enumerate(strip_bins):
        bin_items.sort(key=lambda t: t[0])  # sort by UMAP x

        # Total natural width of images in this strip
        natural_total = sum(w for _, _, w in bin_items)
        # Scale factor so strip fills exactly torus_width
        scale = torus_width / natural_total if natural_total > 0 else 1.0

        images = []
        cx = 0.0
        for _, iid, w in bin_items:
            scaled_w = w * scale
            thumb = dims.get(iid, (512, 512, None))[2]
            images.append(ImagePosition(id=iid, x=cx, width=scaled_w, thumbnail_path=thumb))
            cx += scaled_w

        strips.append(Strip(y=s_idx * strip_height, images=images))

    torus_height = len(strips) * strip_height

    logger.info("Layout: %d images -> %d strips, torus=%.0fx%.0f",
                n, len(strips), torus_width, torus_height)

    return StripLayout(
        strips=strips,
        torus_width=torus_width,
        torus_height=torus_height,
        strip_height=strip_height,
    )
