"""NeighborhoodMode — arranges images on the torus.

Invariants:
- Similar images cluster locally as patches, not strips or axes.
- Never changes which images are in the slice.
- Strips: no gaps, no overlaps, fixed height, width = torus_width.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import umap
from sklearn.cluster import KMeans

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
    images: list[ImagePosition] = field(default_factory=list)


@dataclass
class StripLayout:
    strips: list[Strip]
    torus_width: float
    torus_height: float
    strip_height: float


def _fetch_image_dimensions(db: CorpusDB, image_ids: list[str]) -> dict[str, tuple[int, int, str | None]]:
    placeholders = ",".join("?" * len(image_ids))
    rows = db._conn.execute(
        f"SELECT id, pixel_width, pixel_height, thumbnail_path FROM images WHERE id IN ({placeholders})",
        image_ids,
    ).fetchall()
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
        rows = db._conn.execute(
            f"SELECT id, capture_date FROM images WHERE id IN ({','.join('?' * len(image_ids))})",
            image_ids,
        ).fetchall()
        col = np.array([dict(rows).get(iid, 0.0) or 0.0 for iid in image_ids], dtype=np.float32)
        if col.std() > 0:
            col = (col - col.mean()) / col.std()
        extra.append(col.reshape(-1, 1))

    if "location" in axes:
        rows = db._conn.execute(
            f"SELECT id, gps_latitude, gps_longitude FROM images WHERE id IN ({','.join('?' * len(image_ids))})",
            image_ids,
        ).fetchall()
        lat = np.array([dict([(r[0], r[1] or 0.0) for r in rows]).get(iid, 0.0) for iid in image_ids], dtype=np.float32)
        lon = np.array([dict([(r[0], r[2] or 0.0) for r in rows]).get(iid, 0.0) for iid in image_ids], dtype=np.float32)
        for c in [lat, lon]:
            if c.std() > 0: c[:] = (c - c.mean()) / c.std()
        extra.append(np.column_stack([lat, lon]))

    if extra:
        scale = np.std(matrix)
        matrix = np.hstack([matrix, np.hstack(extra) * scale])
    return matrix


def _pack_strip(items: list[tuple[str, float, str | None]], strip_idx: int,
                strip_height: float, torus_width: float) -> Strip:
    """Pack images into a strip. Scale widths uniformly so total = torus_width."""
    total = sum(w for _, w, _ in items)
    scale = torus_width / total if total > 0 else 1.0
    images = []
    x = 0.0
    for iid, w, thumb in items:
        sw = w * scale
        images.append(ImagePosition(id=iid, x=x, width=sw, thumbnail_path=thumb))
        x += sw
    return Strip(y=strip_idx * strip_height, images=images)


def compute_layout(
    provider: EmbeddingProvider, db: CorpusDB, image_ids: list[str],
    axes: list[str] | None = None, tightness: float = 0.5,
    model: str = "clip-vit-b-32", strip_height: float = 100.0,
) -> StripLayout:
    n = len(image_ids)
    if n == 0:
        return StripLayout([], 0, 0, strip_height)

    dims = _fetch_image_dimensions(db, image_ids)

    if n == 1:
        iid = image_ids[0]
        pw, ph, thumb = dims.get(iid, (512, 512, None))
        w = strip_height * (pw / ph)
        return StripLayout([Strip(0, [ImagePosition(iid, 0, w, thumb)])], w, strip_height, strip_height)

    # Image display widths
    widths = {iid: strip_height * (dims[iid][0] / dims[iid][1]) for iid in image_ids}
    total_width = sum(widths.values())
    n_strips = max(1, round(np.sqrt(total_width / strip_height)))
    torus_width = total_width / n_strips

    # Cluster, place attractors on 2D surface, assign images to torus positions
    matrix = _build_embedding_matrix(provider, image_ids, model, axes, db)
    n_attractors = max(2, min(int(np.sqrt(n)), 100))
    min_dist = max(0.0, min(0.99, tightness))

    logger.info("Layout: %d images, %d attractors, %d strips", n, n_attractors, n_strips)

    kmeans = KMeans(n_clusters=n_attractors, random_state=42, n_init=3)
    labels = kmeans.fit_predict(matrix)

    reducer = umap.UMAP(
        n_components=2, n_neighbors=min(15, max(2, n_attractors - 1)),
        min_dist=min_dist, metric="cosine", random_state=42,
    )
    apos = reducer.fit_transform(kmeans.cluster_centers_)
    # Normalize to [0, n_strips) for y and [0, torus_width) for x
    ax = (apos[:, 0] - apos[:, 0].min()) / max(apos[:, 0].ptp(), 1e-8) * torus_width
    ay = (apos[:, 1] - apos[:, 1].min()) / max(apos[:, 1].ptp(), 1e-8) * (n_strips - 1)

    # Assign each image a target (strip_index, x_position) via its attractor
    assignments: list[list[tuple[float, str, float, str | None]]] = [[] for _ in range(n_strips)]

    for k in range(n_attractors):
        members = np.where(labels == k)[0]
        if len(members) == 0:
            continue

        home_strip = int(round(ay[k]))
        home_x = float(ax[k])
        dists = np.linalg.norm(matrix[members] - kmeans.cluster_centers_[k], axis=1)
        order = np.argsort(dists)

        # Spiral distribution: closest to centroid placed at home, outer images radiate out
        max_ring = max(1, int(np.sqrt(len(members) * strip_height / torus_width) * 0.5))

        for rank, li in enumerate(order):
            idx = members[li]
            iid = image_ids[idx]
            w = widths[iid]
            thumb = dims[iid][2]

            angle = rank * 2.3998628  # golden angle
            r = np.sqrt((rank + 1) / len(members))
            dy = int(round(r * max_ring * np.sin(angle)))
            dx = r * torus_width * 0.12 * np.cos(angle)

            s = (home_strip + dy) % n_strips
            x = (home_x + dx) % torus_width
            assignments[s].append((x, iid, w, thumb))

    # Build strips — each strip sorts by requested x, packs to torus_width
    strips = []
    for i in range(n_strips):
        if not assignments[i]:
            continue
        assignments[i].sort(key=lambda t: t[0])
        items = [(iid, w, thumb) for _, iid, w, thumb in assignments[i]]
        strips.append(_pack_strip(items, len(strips), strip_height, torus_width))

    torus_height = len(strips) * strip_height
    logger.info("Layout: %d strips, torus=%.0fx%.0f", len(strips), torus_width, torus_height)

    return StripLayout(strips, torus_width, torus_height, strip_height)
