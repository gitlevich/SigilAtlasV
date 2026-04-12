"""SQLite repository for images and embeddings."""

import logging
import sqlite3
import struct
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id TEXT PRIMARY KEY,
    source_path TEXT UNIQUE NOT NULL,
    content_hash TEXT,
    capture_date REAL,
    pixel_width INTEGER,
    pixel_height INTEGER,
    created_at REAL NOT NULL,
    gps_latitude REAL,
    gps_longitude REAL,
    camera_model TEXT,
    lens_model TEXT,
    focal_length REAL,
    aperture REAL,
    shutter_speed REAL,
    iso INTEGER,
    thumbnail_path TEXT,
    metadata_extracted_at REAL,
    thumbnail_generated_at REAL
);

CREATE TABLE IF NOT EXISTS embeddings (
    image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    model_identifier TEXT NOT NULL,
    vector BLOB NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (image_id, model_identifier)
);

CREATE INDEX IF NOT EXISTS idx_embeddings_image_id ON embeddings(image_id);
CREATE INDEX IF NOT EXISTS idx_images_content_hash ON images(content_hash);

CREATE TABLE IF NOT EXISTS characterizations (
    image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    proximity_name TEXT NOT NULL,
    value_type TEXT NOT NULL,
    value_enum TEXT,
    value_range REAL,
    created_at REAL NOT NULL,
    PRIMARY KEY (image_id, proximity_name)
);
"""


@dataclass
class ImageRecord:
    id: str
    source_path: str
    content_hash: str | None = None
    capture_date: float | None = None
    pixel_width: int | None = None
    pixel_height: int | None = None
    created_at: float | None = None
    gps_latitude: float | None = None
    gps_longitude: float | None = None
    camera_model: str | None = None
    lens_model: str | None = None
    focal_length: float | None = None
    aperture: float | None = None
    shutter_speed: float | None = None
    iso: int | None = None
    thumbnail_path: str | None = None
    metadata_extracted_at: float | None = None
    thumbnail_generated_at: float | None = None


class CorpusDB:
    """Thread-safe SQLite repository for corpus data."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.row_factory = sqlite3.Row

    def initialize_schema(self) -> None:
        self._conn.executescript(SCHEMA)
        logger.info("Database schema initialized at %s", self.path)

    def close(self) -> None:
        self._conn.close()

    # ── Images ──

    def insert_image(self, rec: ImageRecord) -> None:
        if rec.created_at is None:
            rec.created_at = time.time()
        self._conn.execute(
            """INSERT OR IGNORE INTO images
               (id, source_path, content_hash, capture_date, pixel_width, pixel_height,
                created_at, gps_latitude, gps_longitude, camera_model, lens_model,
                focal_length, aperture, shutter_speed, iso, thumbnail_path,
                metadata_extracted_at, thumbnail_generated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rec.id, rec.source_path, rec.content_hash, rec.capture_date,
                rec.pixel_width, rec.pixel_height, rec.created_at,
                rec.gps_latitude, rec.gps_longitude, rec.camera_model, rec.lens_model,
                rec.focal_length, rec.aperture, rec.shutter_speed, rec.iso,
                rec.thumbnail_path, rec.metadata_extracted_at, rec.thumbnail_generated_at,
            ),
        )
        self._conn.commit()

    def insert_images_batch(self, records: list[ImageRecord]) -> None:
        now = time.time()
        rows = []
        for rec in records:
            if rec.created_at is None:
                rec.created_at = now
            rows.append((
                rec.id, rec.source_path, rec.content_hash, rec.capture_date,
                rec.pixel_width, rec.pixel_height, rec.created_at,
                rec.gps_latitude, rec.gps_longitude, rec.camera_model, rec.lens_model,
                rec.focal_length, rec.aperture, rec.shutter_speed, rec.iso,
                rec.thumbnail_path, rec.metadata_extracted_at, rec.thumbnail_generated_at,
            ))
        self._conn.executemany(
            """INSERT OR IGNORE INTO images
               (id, source_path, content_hash, capture_date, pixel_width, pixel_height,
                created_at, gps_latitude, gps_longitude, camera_model, lens_model,
                focal_length, aperture, shutter_speed, iso, thumbnail_path,
                metadata_extracted_at, thumbnail_generated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()

    def update_metadata(self, image_id: str, **kwargs) -> None:
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [image_id]
        self._conn.execute(f"UPDATE images SET {sets} WHERE id = ?", vals)
        self._conn.commit()

    def update_thumbnail(self, image_id: str, thumbnail_path: str) -> None:
        self._conn.execute(
            "UPDATE images SET thumbnail_path = ?, thumbnail_generated_at = ? WHERE id = ?",
            (thumbnail_path, time.time(), image_id),
        )
        self._conn.commit()

    def image_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM images").fetchone()
        return row[0]

    def fetch_image_ids(self) -> list[str]:
        rows = self._conn.execute("SELECT id FROM images").fetchall()
        return [r[0] for r in rows]

    def fetch_images_without_thumbnails(self) -> list[tuple[str, str]]:
        """Return (id, source_path) for images missing thumbnails."""
        rows = self._conn.execute(
            "SELECT id, source_path FROM images WHERE thumbnail_generated_at IS NULL"
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def fetch_images_without_metadata(self) -> list[tuple[str, str]]:
        """Return (id, source_path) for images missing metadata extraction."""
        rows = self._conn.execute(
            "SELECT id, source_path FROM images WHERE metadata_extracted_at IS NULL"
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def fetch_image_source_path(self, image_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT source_path FROM images WHERE id = ?", (image_id,)
        ).fetchone()
        return row[0] if row else None

    # ── Embeddings ──

    def insert_embeddings_batch(
        self, pairs: list[tuple[str, str, list[float]]]
    ) -> None:
        """Insert batch of (image_id, model_identifier, vector)."""
        now = time.time()
        rows = []
        for image_id, model_id, vector in pairs:
            blob = struct.pack(f"<{len(vector)}f", *vector)
            rows.append((image_id, model_id, blob, now))
        self._conn.executemany(
            """INSERT OR REPLACE INTO embeddings
               (image_id, model_identifier, vector, created_at)
               VALUES (?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()

    def fetch_embedded_image_ids(self, model_identifier: str) -> set[str]:
        rows = self._conn.execute(
            "SELECT image_id FROM embeddings WHERE model_identifier = ?",
            (model_identifier,),
        ).fetchall()
        return {r[0] for r in rows}

    def fetch_unembedded_image_ids(self, model_identifier: str) -> list[str]:
        """Return image IDs that have thumbnails but no embedding for this model."""
        rows = self._conn.execute(
            """SELECT i.id FROM images i
               WHERE i.thumbnail_generated_at IS NOT NULL
               AND i.id NOT IN (
                   SELECT image_id FROM embeddings WHERE model_identifier = ?
               )""",
            (model_identifier,),
        ).fetchall()
        return [r[0] for r in rows]

    def embedding_count(self, model_identifier: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE model_identifier = ?",
            (model_identifier,),
        ).fetchone()
        return row[0]

    def fetch_embedding(self, image_id: str, model_identifier: str) -> list[float] | None:
        row = self._conn.execute(
            "SELECT vector FROM embeddings WHERE image_id = ? AND model_identifier = ?",
            (image_id, model_identifier),
        ).fetchone()
        if row is None:
            return None
        blob = row[0]
        count = len(blob) // 4
        return list(struct.unpack(f"<{count}f", blob))

    # ── Characterizations ──

    def insert_characterizations_batch(
        self, rows: list[tuple[str, str, str, str | None, float | None]]
    ) -> None:
        """Insert batch of (image_id, proximity_name, value_type, value_enum, value_range)."""
        now = time.time()
        self._conn.executemany(
            """INSERT OR REPLACE INTO characterizations
               (image_id, proximity_name, value_type, value_enum, value_range, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [(r[0], r[1], r[2], r[3], r[4], now) for r in rows],
        )
        self._conn.commit()

    def fetch_characterizations(self, image_id: str) -> dict[str, str | float]:
        """Return {proximity_name: value} for an image. Enum returns str, range returns float."""
        rows = self._conn.execute(
            "SELECT proximity_name, value_type, value_enum, value_range FROM characterizations WHERE image_id = ?",
            (image_id,),
        ).fetchall()
        result = {}
        for r in rows:
            if r[1] == "enum":
                result[r[0]] = r[2]
            else:
                result[r[0]] = r[3]
        return result

    def fetch_all_characterizations(self) -> dict[str, dict[str, str | float]]:
        """Return {image_id: {proximity_name: value}} for all characterized images."""
        rows = self._conn.execute(
            "SELECT image_id, proximity_name, value_type, value_enum, value_range FROM characterizations"
        ).fetchall()
        result: dict[str, dict[str, str | float]] = {}
        for r in rows:
            if r[0] not in result:
                result[r[0]] = {}
            if r[2] == "enum":
                result[r[0]][r[1]] = r[3]
            else:
                result[r[0]][r[1]] = r[4]
        return result

    def fetch_uncharacterized_image_ids(self) -> list[str]:
        """Return image IDs that have CLIP embeddings but no characterizations."""
        rows = self._conn.execute(
            """SELECT DISTINCT e.image_id FROM embeddings e
               WHERE e.model_identifier = 'clip-vit-b-32'
               AND e.image_id NOT IN (
                   SELECT DISTINCT image_id FROM characterizations
               )"""
        ).fetchall()
        return [r[0] for r in rows]
