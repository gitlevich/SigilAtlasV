"""Overview atlas — one precomputed texture with every image as a tiny tile.

Baked once per corpus and cached to disk. Lets the frontend render a coherent
field of all ~N images the instant the canvas loads, regardless of how many
full-size thumbnails have fetched yet. A single PNG + a JSON index.

Tile size is chosen so the packed atlas fits in a 4096x4096 WebGL texture.
For 74k images: 273 cols x 272 rows x 15px tile = 4095 x 4080 PNG, ~6-8 MB
after compression.
"""

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from sigil_atlas.db import CorpusDB
from sigil_atlas.workspace import Workspace

logger = logging.getLogger(__name__)

MAX_ATLAS_DIM = 4096  # WebGL 1 minimum max texture size


@dataclass
class OverviewIndex:
    tile_size: int
    cols: int
    rows: int
    atlas_width: int
    atlas_height: int
    # image_id -> [col, row]
    mapping: dict[str, list[int]]


def overview_paths(workspace: Workspace) -> tuple[Path, Path]:
    return workspace.cache_dir / "overview.png", workspace.cache_dir / "overview.json"


def mid_atlas_index_path(workspace: Workspace) -> Path:
    return workspace.cache_dir / "mid-atlas.json"


def mid_atlas_page_path(workspace: Workspace, page: int) -> Path:
    return workspace.cache_dir / f"mid-atlas-{page}.png"


@dataclass
class MidAtlasIndex:
    tile_size: int
    cols_per_page: int
    rows_per_page: int
    atlas_width: int
    atlas_height: int
    pages: int
    # image_id -> [page, col, row]
    mapping: dict[str, list[int]]


def generate_mid_atlas(
    workspace: Workspace,
    db: CorpusDB,
    tile_size: int = 32,
    page_dim: int = MAX_ATLAS_DIM,
) -> MidAtlasIndex:
    """Multi-page atlas at a higher resolution than the overview.

    Fills the zoom gap between 15px overview tiles (coarse preview) and
    streamed 96px per-image thumbnails (slow under HTTP/1.1 connection caps).
    Each page is a single PNG; download is a fixed one-time cost per model
    cached to disk.
    """
    image_ids = db.fetch_image_ids()
    n = len(image_ids)
    if n == 0:
        raise ValueError("No images in corpus")

    cols_per_page = page_dim // tile_size
    rows_per_page = page_dim // tile_size
    tiles_per_page = cols_per_page * rows_per_page
    pages_needed = (n + tiles_per_page - 1) // tiles_per_page
    atlas_w = cols_per_page * tile_size
    atlas_h = rows_per_page * tile_size

    idx_path = mid_atlas_index_path(workspace)
    if idx_path.exists():
        try:
            cached = json.loads(idx_path.read_text())
            if (
                cached.get("tile_size") == tile_size
                and cached.get("pages") == pages_needed
                and len(cached.get("mapping", {})) == n
                and all(mid_atlas_page_path(workspace, p).exists() for p in range(pages_needed))
            ):
                logger.info("Mid-atlas cache hit (%d tiles across %d pages)", n, pages_needed)
                return MidAtlasIndex(
                    tile_size=tile_size,
                    cols_per_page=cols_per_page,
                    rows_per_page=rows_per_page,
                    atlas_width=atlas_w,
                    atlas_height=atlas_h,
                    pages=pages_needed,
                    mapping=cached["mapping"],
                )
        except Exception as exc:
            logger.info("Mid-atlas cache invalid, regenerating: %s", exc)

    workspace.cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Generating mid-atlas: %d tiles, %dx%dpx tile, %d pages of %dx%d",
        n, tile_size, tile_size, pages_needed, atlas_w, atlas_h,
    )
    t0 = time.monotonic()

    mapping: dict[str, list[int]] = {}

    for page in range(pages_needed):
        atlas = Image.new("RGB", (atlas_w, atlas_h), (0, 0, 0))
        start = page * tiles_per_page
        end = min(start + tiles_per_page, n)
        for local_idx in range(end - start):
            global_idx = start + local_idx
            iid = image_ids[global_idx]
            col = local_idx % cols_per_page
            row = local_idx // cols_per_page
            mapping[iid] = [page, col, row]

            thumb_path = workspace.thumbnails_dir / f"{iid}.jpg"
            if not thumb_path.exists():
                continue
            try:
                img = Image.open(thumb_path).convert("RGB")
                w, h = img.size
                s = min(w, h)
                img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
                img = img.resize((tile_size, tile_size), Image.BILINEAR)
                atlas.paste(img, (col * tile_size, row * tile_size))
            except Exception as exc:
                logger.warning("Mid-atlas: failed to tile %s: %s", iid, exc)

        page_path = mid_atlas_page_path(workspace, page)
        atlas.save(page_path, "PNG", optimize=False)
        logger.info("Mid-atlas page %d/%d written (%.1f MB)", page + 1, pages_needed, page_path.stat().st_size / (1024 * 1024))

    idx_path.write_text(json.dumps({
        "tile_size": tile_size,
        "cols_per_page": cols_per_page,
        "rows_per_page": rows_per_page,
        "atlas_width": atlas_w,
        "atlas_height": atlas_h,
        "pages": pages_needed,
        "mapping": mapping,
    }))

    logger.info("Mid-atlas complete in %.1fs", time.monotonic() - t0)
    return MidAtlasIndex(
        tile_size=tile_size,
        cols_per_page=cols_per_page,
        rows_per_page=rows_per_page,
        atlas_width=atlas_w,
        atlas_height=atlas_h,
        pages=pages_needed,
        mapping=mapping,
    )


def _pick_tile_size(n: int) -> tuple[int, int, int]:
    """(tile_size, cols, rows) that packs n tiles into a near-square grid
    whose largest side fits in MAX_ATLAS_DIM pixels.
    """
    cols = max(1, int(math.ceil(math.sqrt(n))))
    rows = max(1, int(math.ceil(n / cols)))
    longest = max(cols, rows)
    tile_size = max(1, MAX_ATLAS_DIM // longest)
    return tile_size, cols, rows


def generate_overview(workspace: Workspace, db: CorpusDB) -> OverviewIndex:
    """Build the overview atlas and cache it under workspace/cache/.

    Idempotent: existing cache is reused if it matches the current image set.
    """
    png_path, idx_path = overview_paths(workspace)
    image_ids = db.fetch_image_ids()
    n = len(image_ids)
    if n == 0:
        raise ValueError("No images in corpus")

    tile_size, cols, rows = _pick_tile_size(n)
    atlas_w, atlas_h = cols * tile_size, rows * tile_size

    # Cache hit?
    if png_path.exists() and idx_path.exists():
        try:
            cached = json.loads(idx_path.read_text())
            if (
                cached.get("tile_size") == tile_size
                and cached.get("cols") == cols
                and cached.get("rows") == rows
                and len(cached.get("mapping", {})) == n
            ):
                logger.info("Overview atlas cache hit (%d tiles)", n)
                return OverviewIndex(
                    tile_size=tile_size, cols=cols, rows=rows,
                    atlas_width=atlas_w, atlas_height=atlas_h,
                    mapping=cached["mapping"],
                )
        except Exception as exc:
            logger.info("Overview cache invalid, regenerating: %s", exc)

    workspace.cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Generating overview atlas: %d tiles, %dx%dpx tile, %dx%d atlas",
        n, tile_size, tile_size, atlas_w, atlas_h,
    )
    t0 = time.monotonic()

    atlas = Image.new("RGB", (atlas_w, atlas_h), (0, 0, 0))
    mapping: dict[str, list[int]] = {}

    for idx, iid in enumerate(image_ids):
        col = idx % cols
        row = idx // cols
        mapping[iid] = [col, row]

        thumb_path = workspace.thumbnails_dir / f"{iid}.jpg"
        if not thumb_path.exists():
            continue

        try:
            img = Image.open(thumb_path).convert("RGB")
            w, h = img.size
            s = min(w, h)
            # Centre-crop to square, then resize to tile_size.
            img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
            img = img.resize((tile_size, tile_size), Image.BILINEAR)
            atlas.paste(img, (col * tile_size, row * tile_size))
        except Exception as exc:
            logger.warning("Overview: failed to tile %s: %s", iid, exc)

        if (idx + 1) % 10000 == 0:
            logger.info("Overview: %d / %d tiles written", idx + 1, n)

    atlas.save(png_path, "PNG", optimize=False)
    idx_path.write_text(json.dumps({
        "tile_size": tile_size,
        "cols": cols,
        "rows": rows,
        "atlas_width": atlas_w,
        "atlas_height": atlas_h,
        "mapping": mapping,
    }))

    logger.info(
        "Overview atlas written: %s (%.1f MB, %.1fs)",
        png_path, png_path.stat().st_size / (1024 * 1024), time.monotonic() - t0,
    )
    return OverviewIndex(
        tile_size=tile_size, cols=cols, rows=rows,
        atlas_width=atlas_w, atlas_height=atlas_h,
        mapping=mapping,
    )
