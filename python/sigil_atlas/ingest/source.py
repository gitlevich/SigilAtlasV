"""Source — where images come from."""

import hashlib
import logging
import uuid
from pathlib import Path

from sigil_atlas.db import CorpusDB, ImageRecord

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}


class FolderSource:
    """Scans a filesystem folder for images and registers them in the corpus.

    Uniquely identified by its absolute path. Disconnectable (folder may not
    always be mounted). Movable (path can be updated).
    """

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        if not self.path.is_dir():
            raise ValueError(f"Source folder does not exist: {self.path}")

    @property
    def location(self) -> str:
        return str(self.path)

    def scan(self) -> list[Path]:
        """Enumerate all image files in the folder recursively."""
        files = []
        for f in sorted(self.path.rglob("*")):
            if f.suffix.lower() in IMAGE_EXTENSIONS and f.is_file():
                files.append(f)
        logger.info("Scanned %d images from %s", len(files), self.path)
        return files

    def register_images(self, db: CorpusDB, files: list[Path], batch_size: int = 500) -> int:
        """Register image files in the database. Deduplicates by content hash.

        Computes SHA-256 of each file. Images whose hash already exists in the
        corpus are silently skipped — the same photo from a different path or
        source won't produce a duplicate.

        Returns count of newly registered images.
        """
        known_hashes = db.fetch_content_hashes()
        registered = 0
        skipped = 0
        batch: list[ImageRecord] = []

        for f in files:
            h = content_hash(f)
            if h in known_hashes:
                skipped += 1
                continue
            known_hashes.add(h)

            batch.append(ImageRecord(
                id=str(uuid.uuid4()),
                source_path=str(f),
                content_hash=h,
            ))

            if len(batch) >= batch_size:
                db.insert_images_batch(batch)
                registered += len(batch)
                batch.clear()

        if batch:
            db.insert_images_batch(batch)
            registered += len(batch)

        if skipped:
            logger.info("Skipped %d duplicate images (by content hash)", skipped)
        logger.info("Registered %d new images", registered)
        return registered


def content_hash(path: Path) -> str:
    """Derive a stable content hash from file bytes (SHA-256)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
