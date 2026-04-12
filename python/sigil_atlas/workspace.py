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
        self.db_path = self.datastore_dir / "corpus.db"

    def initialize(self) -> "Workspace":
        """Create directory structure and initialize the database."""
        self.datastore_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Workspace initialized at %s", self.root)
        db = self.open_db()
        db.initialize_schema()
        db.close()
        return self

    def open_db(self) -> CorpusDB:
        return CorpusDB(self.db_path)
