"""Neighborhood layout — clusters images around attractors on the torus surface.

Pipeline:
  1. Fetch precomputed KMeans assignments (tightness selects k level)
  2. PCA of active cluster centroids to 2D surface positions
  3. Sort images within each cluster by similarity (PCA + band sort)
  4. Pack each cluster into a near-square rectangle via local strip packing
  5. Place neighborhoods at PCA positions, resolve overlaps with minimal displacement

Invariants:
  - Each neighborhood internally gapless (!no-gaps within neighborhood)
  - No overlaps (!no-overlaps)
  - Aspect ratio preserved (!respect-image-aspect-ratio, !no-crop)
  - Neighborhoods are patches, not strips (!local-neighborhoods)
"""

import logging
import math
import struct
from dataclasses import dataclass, field

import numpy as np

from sigil_atlas.db import CorpusDB
from sigil_atlas.embedding_provider import EmbeddingProvider
from sigil_atlas.ingest.cluster import KMEANS_K_LEVELS
from sigil_atlas.layout import _fetch_image_dimensions, _greedy_pack_strips

logger = logging.getLogger(__name__)


@dataclass
class NeighborhoodStrip:
    y: float
    height: float
    images: list


@dataclass
class Neighborhood:
    cluster_id: int
    x: float
    y: float
    width: float
    height: float
    strips: list[NeighborhoodStrip] = field(default_factory=list)


@dataclass
class NeighborhoodLayout:
    neighborhoods: list[Neighborhood]
    torus_width: float
    torus_height: float
    strip_height: float


def _tightness_to_k(tightness: float) -> int:
    idx = round(tightness * (len(KMEANS_K_LEVELS) - 1))
    idx = max(0, min(len(KMEANS_K_LEVELS) - 1, idx))
    return KMEANS_K_LEVELS[idx]


def _unpack_centroid(blob: bytes) -> np.ndarray:
    count = len(blob) // 4
    return np.array(struct.unpack(f"<{count}f", blob), dtype=np.float32)


def _pca_2d(vectors: np.ndarray) -> np.ndarray:
    """Project vectors to 2D via PCA. Returns positions in [0, 1]^2."""
    if len(vectors) <= 1:
        return np.array([[0.5, 0.5]])
    centered = vectors - vectors.mean(axis=0)
    U, S, _ = np.linalg.svd(centered, full_matrices=False)
    coords = U[:, :2] * S[:2]
    for dim in range(2):
        lo, hi = coords[:, dim].min(), coords[:, dim].max()
        span = hi - lo
        if span > 1e-8:
            coords[:, dim] = (coords[:, dim] - lo) / span
        else:
            coords[:, dim] = 0.5
    return coords


def _similarity_order(
    member_ids: list[str],
    embeddings: np.ndarray,
) -> list[str]:
    """Sort images within a cluster by PCA + band sort + snake fill.

    Same approach as v2 similarity.py — preserves local neighborhoods
    within the cluster rectangle.
    """
    n = len(member_ids)
    if n <= 1:
        return list(member_ids)

    centered = embeddings - embeddings.mean(axis=0)
    U, S, _ = np.linalg.svd(centered, full_matrices=False)
    coords = U[:, :2] * S[:2]

    for dim in range(2):
        lo, hi = coords[:, dim].min(), coords[:, dim].max()
        if hi > lo:
            coords[:, dim] = (coords[:, dim] - lo) / (hi - lo)

    # Estimate rows for a near-square rectangle
    row_count = max(1, round(math.sqrt(n)))
    row_size = max(1, math.ceil(n / row_count))

    # Sort by y into bands
    indices = list(range(n))
    indices.sort(key=lambda i: coords[i, 1])

    # Within each band, sort by x with snake fill
    ordered: list[str] = []
    for band_start in range(0, n, row_size):
        band = indices[band_start:band_start + row_size]
        band.sort(key=lambda i: coords[i, 0])
        if (band_start // row_size) % 2 == 1:
            band.reverse()
        ordered.extend(member_ids[i] for i in band)

    return ordered


def _resolve_overlaps(
    rects: list[tuple[int, float, float, float, float]],
    torus_w: float,
    torus_h: float,
    iterations: int = 200,
) -> list[tuple[int, float, float, float, float]]:
    """Push overlapping rectangles apart with minimal displacement.

    Each rect is (cid, x, y, w, h) where x,y is the center.
    Returns adjusted rects with overlaps resolved.
    """
    if len(rects) <= 1:
        return rects

    centers = np.array([[r[1], r[2]] for r in rects], dtype=np.float64)
    sizes = np.array([[r[3], r[4]] for r in rects], dtype=np.float64)

    for _ in range(iterations):
        moved = False
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]

                # Minimum separation to avoid overlap (plus thin gap)
                min_sep_x = (sizes[i, 0] + sizes[j, 0]) * 0.5 + sizes[i, 0] * 0.05
                min_sep_y = (sizes[i, 1] + sizes[j, 1]) * 0.5 + sizes[i, 1] * 0.05

                overlap_x = min_sep_x - abs(dx)
                overlap_y = min_sep_y - abs(dy)

                if overlap_x > 0 and overlap_y > 0:
                    # Push apart along the axis with less overlap
                    if overlap_x < overlap_y:
                        shift = overlap_x * 0.5 * (1 if dx >= 0 else -1)
                        centers[i, 0] -= shift
                        centers[j, 0] += shift
                    else:
                        shift = overlap_y * 0.5 * (1 if dy >= 0 else -1)
                        centers[i, 1] -= shift
                        centers[j, 1] += shift
                    moved = True

        if not moved:
            break

    return [(rects[i][0], centers[i, 0], centers[i, 1], rects[i][3], rects[i][4])
            for i in range(len(rects))]


def compute_neighborhood_layout(
    provider: EmbeddingProvider,
    db: CorpusDB,
    image_ids: list[str],
    tightness: float = 0.5,
    model: str = "clip-vit-l-14",
    strip_height: float = 100.0,
) -> NeighborhoodLayout:
    n = len(image_ids)
    if n == 0:
        return NeighborhoodLayout([], 0, 0, strip_height)

    # Step 1: Get cluster assignments
    k = _tightness_to_k(tightness)
    assignments = db.fetch_kmeans_assignments_for_ids(model, k, image_ids)

    unassigned = [iid for iid in image_ids if iid not in assignments]
    for iid in unassigned:
        assignments[iid] = -1

    clusters: dict[int, list[str]] = {}
    for iid, cid in assignments.items():
        clusters.setdefault(cid, []).append(iid)

    # Step 2: PCA of centroids to 2D
    centroid_blobs = db.fetch_kmeans_centroids(model, k)
    active_cids = sorted(cid for cid in clusters if cid >= 0 and cid in centroid_blobs)

    if active_cids:
        centroid_matrix = np.stack([_unpack_centroid(centroid_blobs[cid]) for cid in active_cids])
        positions = _pca_2d(centroid_matrix)
        cid_to_pos = {cid: positions[i] for i, cid in enumerate(active_cids)}
    else:
        cid_to_pos = {}

    if -1 in clusters:
        cid_to_pos[-1] = np.array([0.5, 0.5])

    # Step 3: Fetch image dimensions and compute natural widths
    dims = _fetch_image_dimensions(db, image_ids)
    natural_widths = {iid: strip_height * (dims[iid][0] / dims[iid][1]) for iid in image_ids}
    thumbnails = {iid: dims[iid][2] for iid in image_ids}

    # Step 4: Load embeddings for within-cluster ordering
    embedding_matrix = provider.fetch_matrix(image_ids, model)
    id_to_idx = {iid: i for i, iid in enumerate(image_ids)}

    # Step 5: Size and pack each neighborhood
    packed_neighborhoods: list[tuple[int, float, float, np.ndarray, list]] = []

    for cid in sorted(clusters.keys()):
        members = clusters[cid]

        # Sort within cluster by similarity
        if len(members) > 1:
            member_embeddings = embedding_matrix[[id_to_idx[m] for m in members]]
            sorted_members = _similarity_order(members, member_embeddings)
        else:
            sorted_members = members

        total_natural_width = sum(natural_widths[m] for m in members)

        # Target near-square: area = total_width * strip_height, side = sqrt(area)
        area = total_natural_width * strip_height
        target_side = math.sqrt(area)
        rect_width = max(target_side, strip_height * 2)

        local_strips = _greedy_pack_strips(
            sorted_members, natural_widths, thumbnails, strip_height, rect_width,
        )
        rect_height = sum(s.height for s in local_strips)
        if rect_height == 0:
            rect_height = strip_height

        pos = cid_to_pos.get(cid, np.array([0.5, 0.5]))
        packed_neighborhoods.append((cid, rect_width, rect_height, pos, local_strips))

    # Step 6: Place neighborhoods at PCA positions
    # Compute surface size from total area with breathing room
    total_area = sum(rw * rh for _, rw, rh, _, _ in packed_neighborhoods)
    surface_side = math.sqrt(total_area) * 1.15  # 15% breathing room

    # Map PCA [0,1] positions to surface coordinates (as centers)
    placed_rects: list[tuple[int, float, float, float, float]] = []
    for cid, rw, rh, pos, _ in packed_neighborhoods:
        # Leave margin so rectangles don't start at the very edge
        margin_x = rw * 0.5
        margin_y = rh * 0.5
        usable_w = surface_side - rw
        usable_h = surface_side - rh
        cx = margin_x + pos[0] * max(0, usable_w)
        cy = margin_y + pos[1] * max(0, usable_h)
        placed_rects.append((cid, cx, cy, rw, rh))

    # Resolve overlaps
    placed_rects = _resolve_overlaps(placed_rects, surface_side, surface_side)

    # Build final neighborhoods (convert center to top-left)
    cid_to_strips = {cid: strips for cid, _, _, _, strips in packed_neighborhoods}

    neighborhoods: list[Neighborhood] = []
    min_x = min(cx - w * 0.5 for _, cx, cy, w, h in placed_rects) if placed_rects else 0
    min_y = min(cy - h * 0.5 for _, cx, cy, w, h in placed_rects) if placed_rects else 0

    for cid, cx, cy, rw, rh in placed_rects:
        local_strips = cid_to_strips[cid]
        nbr_strips = [NeighborhoodStrip(y=s.y, height=s.height, images=s.images)
                      for s in local_strips]

        neighborhoods.append(Neighborhood(
            cluster_id=cid,
            x=cx - rw * 0.5 - min_x,
            y=cy - rh * 0.5 - min_y,
            width=rw,
            height=rh,
            strips=nbr_strips,
        ))

    max_x = max(nb.x + nb.width for nb in neighborhoods) if neighborhoods else 0
    max_y = max(nb.y + nb.height for nb in neighborhoods) if neighborhoods else 0
    torus_width = max_x * 1.02
    torus_height = max_y * 1.02

    logger.info(
        "Neighborhood layout: %d neighborhoods, %d images, surface=%.0fx%.0f",
        len(neighborhoods), n, torus_width, torus_height,
    )

    return NeighborhoodLayout(neighborhoods, torus_width, torus_height, strip_height)
