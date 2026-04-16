"""Workspace — directory structure and initialization for a SigilAtlas corpus."""

import logging
from pathlib import Path

from sigil_atlas.db import CorpusDB

logger = logging.getLogger(__name__)


class Workspace:
    """A workspace directory containing all application state.

    Structure:
        workspace/
        ├── datastore/
        │   └── corpus.db
        └── image_cache/
            └── thumbnails/
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.datastore_dir = root / "datastore"
        self.image_cache_dir = root / "image_cache"
        self.thumbnails_dir = self.image_cache_dir / "thumbnails"
        self.previews_dir = self.image_cache_dir / "previews"
        self.db_path = self.datastore_dir / "corpus.db"

    def initialize(self) -> "Workspace":
        """Create directory structure and initialize the database."""
        self.datastore_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)
        self.previews_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Workspace initialized at %s", self.root)
        self.open_db().close()
        return self

    def open_db(self) -> CorpusDB:
        db = CorpusDB(self.db_path)
        db.initialize_schema()
        self._verify_thumbnails(db)
        # Promote any images that were fully processed before a crash/restart.
        completed = db.mark_completed()
        if completed:
            logger.info("Recovered %d images from interrupted import", completed)
        return db

    def _verify_thumbnails(self, db: CorpusDB) -> None:
        """Demote images whose thumbnail file is missing on disk.

        This handles the case where the app was killed after the DB was
        updated but the file wasn't flushed, or the cache was cleared.
        """
        rows = db._conn.execute(
            "SELECT id, thumbnail_path FROM images "
            "WHERE thumbnail_generated_at IS NOT NULL AND thumbnail_path IS NOT NULL"
        ).fetchall()
        missing = []
        for image_id, thumb_path in rows:
            if not (self.thumbnails_dir / thumb_path).exists():
                missing.append(image_id)
        if not missing:
            return
        logger.warning(
            "%d images have missing thumbnails on disk, resetting for re-generation",
            len(missing),
        )
        for image_id in missing:
            db._conn.execute(
                "UPDATE images SET thumbnail_path = NULL, thumbnail_generated_at = NULL, "
                "completed_at = NULL WHERE id = ?",
                (image_id,),
            )
        db._conn.commit()
