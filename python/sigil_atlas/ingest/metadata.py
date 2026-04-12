"""Metadata extraction stage — reads EXIF from source images."""

import logging
import time
from pathlib import Path

import exifread

from sigil_atlas.cancel import CancellationToken
from sigil_atlas.db import CorpusDB
from sigil_atlas.ingest.source import content_hash
from sigil_atlas.progress import StageProgress

logger = logging.getLogger(__name__)


def _parse_gps_coord(tag_value, tag_ref) -> float | None:
    """Convert EXIF GPS DMS to decimal degrees."""
    try:
        values = tag_value.values
        d = float(values[0])
        m = float(values[1])
        s = float(values[2])
        result = d + m / 60.0 + s / 3600.0
        if tag_ref and str(tag_ref) in ("S", "W"):
            result = -result
        return result
    except (IndexError, ValueError, TypeError):
        return None


def _parse_rational(tag) -> float | None:
    """Convert an EXIF rational value to float."""
    try:
        val = tag.values[0]
        return float(val)
    except (IndexError, ValueError, TypeError):
        return None


def _parse_capture_date(tags: dict) -> float | None:
    """Parse EXIF DateTimeOriginal to unix timestamp."""
    tag = tags.get("EXIF DateTimeOriginal") or tags.get("Image DateTime")
    if tag is None:
        return None
    try:
        from datetime import datetime
        dt = datetime.strptime(str(tag), "%Y:%m:%d %H:%M:%S")
        return dt.timestamp()
    except (ValueError, TypeError):
        return None


def extract_metadata_batch(
    db: CorpusDB,
    items: list[tuple[str, str]],
    progress: StageProgress,
    token: CancellationToken,
    batch_size: int = 32,
) -> None:
    """Extract EXIF metadata for a batch of (image_id, source_path) pairs."""
    for i in range(0, len(items), batch_size):
        if token.is_cancelled:
            logger.info("Metadata extraction cancelled")
            return

        batch = items[i : i + batch_size]
        for image_id, source_path in batch:
            _extract_one(db, image_id, Path(source_path))
        progress.advance(len(batch))


def _extract_one(db: CorpusDB, image_id: str, path: Path) -> None:
    """Extract metadata from a single image and update the database."""
    try:
        tags = {}
        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=False)

        updates: dict = {"metadata_extracted_at": time.time()}

        # Content hash
        updates["content_hash"] = content_hash(path)

        # Dimensions from EXIF (fallback to Pillow if needed)
        width_tag = tags.get("EXIF ExifImageWidth") or tags.get("Image ImageWidth")
        height_tag = tags.get("EXIF ExifImageLength") or tags.get("Image ImageLength")
        if width_tag:
            try:
                updates["pixel_width"] = int(str(width_tag))
            except ValueError:
                pass
        if height_tag:
            try:
                updates["pixel_height"] = int(str(height_tag))
            except ValueError:
                pass

        # If dimensions not in EXIF, get from Pillow
        if "pixel_width" not in updates or "pixel_height" not in updates:
            try:
                from PIL import Image
                with Image.open(path) as img:
                    updates["pixel_width"] = img.width
                    updates["pixel_height"] = img.height
            except Exception:
                pass

        # Capture date
        capture_date = _parse_capture_date(tags)
        if capture_date is not None:
            updates["capture_date"] = capture_date

        # GPS
        lat_tag = tags.get("GPS GPSLatitude")
        lat_ref = tags.get("GPS GPSLatitudeRef")
        lon_tag = tags.get("GPS GPSLongitude")
        lon_ref = tags.get("GPS GPSLongitudeRef")
        if lat_tag and lon_tag:
            lat = _parse_gps_coord(lat_tag, lat_ref)
            lon = _parse_gps_coord(lon_tag, lon_ref)
            if lat is not None:
                updates["gps_latitude"] = lat
            if lon is not None:
                updates["gps_longitude"] = lon

        # Camera
        model_tag = tags.get("Image Model")
        if model_tag:
            updates["camera_model"] = str(model_tag).strip()

        lens_tag = tags.get("EXIF LensModel")
        if lens_tag:
            updates["lens_model"] = str(lens_tag).strip()

        # Exposure
        focal = tags.get("EXIF FocalLength")
        if focal:
            updates["focal_length"] = _parse_rational(focal)

        aperture = tags.get("EXIF FNumber")
        if aperture:
            updates["aperture"] = _parse_rational(aperture)

        speed = tags.get("EXIF ExposureTime")
        if speed:
            updates["shutter_speed"] = _parse_rational(speed)

        iso_tag = tags.get("EXIF ISOSpeedRatings")
        if iso_tag:
            try:
                updates["iso"] = int(str(iso_tag))
            except ValueError:
                pass

        db.update_metadata(image_id, **updates)

    except Exception:
        logger.warning("Failed to extract metadata for %s", path, exc_info=True)
        db.update_metadata(image_id, metadata_extracted_at=time.time())
