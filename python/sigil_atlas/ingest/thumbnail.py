"""Thumbnail generation stage — creates preview images for embedding and display."""

import logging
from pathlib import Path

from PIL import Image

from sigil_atlas.cancel import CancellationToken
from sigil_atlas.db import CorpusDB
from sigil_atlas.progress import StageProgress

logger = logging.getLogger(__name__)

THUMBNAIL_MAX_SIZE = 512
THUMBNAIL_QUALITY = 85


def generate_thumbnails_batch(
    db: CorpusDB,
    items: list[tuple[str, str]],
    thumbnails_dir: Path,
    progress: StageProgress,
    token: CancellationToken,
    batch_size: int = 32,
) -> None:
    """Generate thumbnails for a batch of (image_id, source_path) pairs."""
    for i in range(0, len(items), batch_size):
        if token.is_cancelled:
            logger.info("Thumbnail generation cancelled")
            return

        batch = items[i : i + batch_size]
        for image_id, source_path in batch:
            _generate_one(db, image_id, Path(source_path), thumbnails_dir)
        progress.advance(len(batch))


def _generate_one(
    db: CorpusDB, image_id: str, source_path: Path, thumbnails_dir: Path
) -> None:
    """Generate a single thumbnail and update the database."""
    output_path = thumbnails_dir / f"{image_id}.jpg"

    if output_path.exists():
        # Already generated (e.g., from a previous partial run)
        db.update_thumbnail(image_id, f"{image_id}.jpg")
        return

    try:
        with Image.open(source_path) as img:
            img = img.convert("RGB")
            img.thumbnail((THUMBNAIL_MAX_SIZE, THUMBNAIL_MAX_SIZE), Image.LANCZOS)
            img.save(output_path, "JPEG", quality=THUMBNAIL_QUALITY)

        db.update_thumbnail(image_id, f"{image_id}.jpg")

    except Exception:
        logger.warning("Failed to generate thumbnail for %s", source_path, exc_info=True)
