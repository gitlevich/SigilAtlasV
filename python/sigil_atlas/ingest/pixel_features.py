"""Pixel-derived visual features — color, brightness, saturation.

Computed from thumbnails during ingest. Stored as range characterizations.
These replace CLIP for perceptual qualities CLIP can't handle.

Features per image:
  - brightness: mean luminance [0, 1]
  - saturation: mean saturation [0, 1]
  - contrast: std of luminance [0, 0.5]
  - hue_dominant: dominant hue angle [0, 360]
  - hue_spread: std of hue [0, 180]
  - color_temperature: warm vs cool [-1, 1]
  - r_fraction, g_fraction, b_fraction: channel dominance [0, 1]
"""

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from sigil_atlas.cancel import CancellationToken
from sigil_atlas.db import CorpusDB
from sigil_atlas.progress import StageProgress

logger = logging.getLogger(__name__)


def extract_pixel_features(image_path: Path) -> dict[str, float]:
    """Extract color/brightness/saturation features from an image."""
    img = Image.open(image_path).convert("RGB")
    # Downsample for speed — 64px is enough for color stats
    img = img.resize((64, 64), Image.LANCZOS)

    rgb = np.array(img, dtype=np.float32) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    # Brightness (perceived luminance)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    brightness = float(luminance.mean())
    contrast = float(luminance.std())

    # Convert to HSV for hue/saturation
    hsv = _rgb_to_hsv(rgb)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    saturation = float(s.mean())

    # Dominant hue — circular mean of hue weighted by saturation
    # (desaturated pixels shouldn't influence hue)
    weights = s.flatten()
    hue_rad = h.flatten() * 2 * np.pi
    if weights.sum() > 0.01:
        sin_mean = np.average(np.sin(hue_rad), weights=weights)
        cos_mean = np.average(np.cos(hue_rad), weights=weights)
        hue_dominant = float(np.degrees(np.arctan2(sin_mean, cos_mean)) % 360)
        # Hue spread (circular std)
        r_len = np.sqrt(sin_mean**2 + cos_mean**2)
        hue_spread = float(np.degrees(np.sqrt(-2 * np.log(max(r_len, 1e-8))))) if r_len < 1 else 0.0
    else:
        hue_dominant = 0.0
        hue_spread = 180.0  # achromatic = maximum spread

    # Channel fractions
    total = r + g + b + 1e-8
    r_frac = float((r / total).mean())
    g_frac = float((g / total).mean())
    b_frac = float((b / total).mean())

    # Color temperature: warm (red/yellow) vs cool (blue)
    # Simple: (r_frac - b_frac) normalized
    color_temp = float(np.clip((r_frac - b_frac) * 3, -1, 1))

    return {
        "brightness": brightness,
        "saturation": saturation,
        "contrast": contrast,
        "hue_dominant": hue_dominant,
        "hue_spread": hue_spread,
        "color_temperature": color_temp,
        "r_fraction": r_frac,
        "g_fraction": g_frac,
        "b_fraction": b_frac,
    }


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorized RGB to HSV. Input/output [0,1] except H in [0,1] (fraction of 360)."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    diff = maxc - minc

    # Hue
    h = np.zeros_like(maxc)
    mask = diff > 1e-8
    rm = mask & (maxc == r)
    gm = mask & (maxc == g) & ~rm
    bm = mask & (maxc == b) & ~rm & ~gm
    h[rm] = ((g[rm] - b[rm]) / diff[rm]) % 6
    h[gm] = ((b[gm] - r[gm]) / diff[gm]) + 2
    h[bm] = ((r[bm] - g[bm]) / diff[bm]) + 4
    h = h / 6.0  # normalize to [0, 1]

    # Saturation
    s = np.where(maxc > 1e-8, diff / maxc, 0.0)

    return np.stack([h, s, maxc], axis=2)


def run_pixel_features_stage(
    db: CorpusDB,
    thumbnails_dir: Path,
    progress: StageProgress,
    token: CancellationToken,
) -> None:
    """Extract pixel features for all images that don't have them yet."""
    all_ids = db.fetch_image_ids()

    # Check which already have pixel features
    existing = set()
    rows = db._conn.execute(
        "SELECT DISTINCT image_id FROM characterizations WHERE proximity_name = 'brightness'"
    ).fetchall()
    existing = {r[0] for r in rows}

    pending = [iid for iid in all_ids if iid not in existing]
    progress.set_total(len(pending))

    if not pending:
        logger.info("All images already have pixel features")
        return

    logger.info("Extracting pixel features for %d images", len(pending))

    # Get thumbnail paths
    thumb_rows = db._query_in_batches(
        "SELECT id, thumbnail_path FROM images WHERE id IN ({placeholders})",
        pending,
    )
    thumb_map = {r[0]: r[1] for r in thumb_rows}

    batch_rows = []
    for i, iid in enumerate(pending):
        if token.is_cancelled:
            logger.info("Pixel features cancelled")
            break

        thumb_name = thumb_map.get(iid)
        if not thumb_name:
            progress.advance(1)
            continue

        thumb_path = thumbnails_dir / thumb_name
        if not thumb_path.exists():
            progress.advance(1)
            continue

        try:
            features = extract_pixel_features(thumb_path)
            for name, value in features.items():
                batch_rows.append((iid, name, "range", None, value))
        except Exception as e:
            logger.warning("Failed to extract pixel features for %s: %s", iid, e)

        if len(batch_rows) >= 1000:
            db.insert_characterizations_batch(batch_rows)
            batch_rows = []

        progress.advance(1)

    if batch_rows:
        db.insert_characterizations_batch(batch_rows)

    logger.info("Pixel features extraction complete")
