"""Thumbnail generation stage — creates preview images for embedding and display."""

import logging
from pathlib import Path

from PIL import Image

from sigil_atlas.cancel import CancellationToken
from sigil_atlas.db import CorpusDB
from sigil_atlas.progress import StageProgress

logger = logging.getLogger(__name__)

THUMB_MAX_SIZE = 512
PREVIEW_MAX_SIZE = 1024
JPEG_QUALITY = 85


def generate_thumbnails_batch(
    db: CorpusDB,
    items: list[tuple[str, str]],
    thumbnails_dir: Path,
    progress: StageProgress,
    token: CancellationToken,
    batch_size: int = 32,
) -> None:
    """Generate thumbnails and previews for a batch of (image_id, source_path) pairs."""
    previews_dir = thumbnails_dir.parent / "previews"
    previews_dir.mkdir(exist_ok=True)

    for i in range(0, len(items), batch_size):
        if token.is_cancelled:
            logger.info("Thumbnail generation cancelled")
            return

        batch = items[i : i + batch_size]
        for image_id, source_path in batch:
            _generate_one(db, image_id, Path(source_path), thumbnails_dir, previews_dir)
        progress.advance(len(batch))


def _generate_one(
    db: CorpusDB, image_id: str, source_path: Path,
    thumbnails_dir: Path, previews_dir: Path,
) -> None:
    """Generate thumbnail (512px) and preview (1024px) for a single image."""
    thumb_path = thumbnails_dir / f"{image_id}.jpg"
    preview_path = previews_dir / f"{image_id}.jpg"

    if thumb_path.exists() and preview_path.exists():
        db.update_thumbnail(image_id, f"{image_id}.jpg")
        return

    try:
        with Image.open(source_path) as img:
            img = img.convert("RGB")

            if not preview_path.exists():
                preview = img.copy()
                preview.thumbnail((PREVIEW_MAX_SIZE, PREVIEW_MAX_SIZE), Image.LANCZOS)
                preview.save(preview_path, "JPEG", quality=JPEG_QUALITY)

            if not thumb_path.exists():
                img.thumbnail((THUMB_MAX_SIZE, THUMB_MAX_SIZE), Image.LANCZOS)
                img.save(thumb_path, "JPEG", quality=JPEG_QUALITY)

        db.update_thumbnail(image_id, f"{image_id}.jpg")

    except Exception:
        logger.warning("Failed to generate thumbnail for %s", source_path, exc_info=True)
