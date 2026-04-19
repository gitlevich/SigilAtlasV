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
    thumbnail_generated_at REAL,
    completed_at REAL
);

CREATE TABLE IF NOT EXISTS embeddings (
    image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    model_identifier TEXT NOT NULL,
    vector BLOB NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (image_id, model_identifier)
);

CREATE INDEX IF NOT EXISTS idx_embeddings_image_id ON embeddings(image_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_images_content_hash ON images(content_hash);

CREATE TABLE IF NOT EXISTS kmeans_clusters (
    model_identifier TEXT NOT NULL,
    k INTEGER NOT NULL,
    image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    cluster_id INTEGER NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (model_identifier, k, image_id)
);
CREATE INDEX IF NOT EXISTS idx_kmeans_model_k ON kmeans_clusters(model_identifier, k);

CREATE TABLE IF NOT EXISTS kmeans_centroids (
    model_identifier TEXT NOT NULL,
    k INTEGER NOT NULL,
    cluster_id INTEGER NOT NULL,
    centroid BLOB NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (model_identifier, k, cluster_id)
);

CREATE TABLE IF NOT EXISTS umap_positions (
    image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    model_identifier TEXT NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (image_id, model_identifier)
);
CREATE INDEX IF NOT EXISTS idx_umap_model ON umap_positions(model_identifier);

CREATE TABLE IF NOT EXISTS characterizations (
    image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    proximity_name TEXT NOT NULL,
    value_type TEXT NOT NULL,
    value_enum TEXT,
    value_range REAL,
    created_at REAL NOT NULL,
    PRIMARY KEY (image_id, proximity_name)
);
CREATE INDEX IF NOT EXISTS idx_characterizations_name ON characterizations(proximity_name, value_type);

CREATE TABLE IF NOT EXISTS things_library (
    name TEXT PRIMARY KEY,
    created_at REAL NOT NULL
);

-- workspace_state is a key/value store for the persisted @Explore state
-- (POV, mode, arrangement, layers, relevance, feathering, ...). A single
-- row with key='ui' holds a JSON blob; see Explore/invariant-persistent-state.
CREATE TABLE IF NOT EXISTS workspace_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    modified_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS collages (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at REAL NOT NULL,
    modified_at REAL NOT NULL,
    expression_json TEXT NOT NULL,
    pov_json TEXT NOT NULL,
    mode TEXT NOT NULL,
    model TEXT NOT NULL,
    relevance REAL NOT NULL,
    feathering REAL NOT NULL,
    cell_size REAL NOT NULL,
    thumbnail_blob BLOB
);
CREATE INDEX IF NOT EXISTS idx_collages_modified ON collages(modified_at DESC);
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
        self._migrate()
        logger.info("Database schema initialized at %s", self.path)

    def _migrate(self) -> None:
        """Run forward-only migrations on existing databases."""
        # Upgrade content_hash index from non-unique to unique.
        # The old index has the same name but isn't unique; drop and recreate.
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name='idx_images_content_hash'"
        ).fetchone()
        if row and "UNIQUE" not in (row[0] or ""):
            self._conn.execute("DROP INDEX idx_images_content_hash")
            self._conn.execute(
                "CREATE UNIQUE INDEX idx_images_content_hash ON images(content_hash)"
            )
            self._conn.commit()
            logger.info("Migrated content_hash index to UNIQUE")

        # Add completed_at column if missing.
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(images)").fetchall()}
        if "completed_at" not in cols:
            self._conn.execute("ALTER TABLE images ADD COLUMN completed_at REAL")
            # Backfill: mark images that have all artifacts as complete.
            self._conn.execute("""
                UPDATE images SET completed_at = metadata_extracted_at
                WHERE completed_at IS NULL
                  AND metadata_extracted_at IS NOT NULL
                  AND thumbnail_generated_at IS NOT NULL
                  AND id IN (SELECT image_id FROM embeddings)
            """)
            self._conn.commit()
            count = self._conn.execute(
                "SELECT COUNT(*) FROM images WHERE completed_at IS NOT NULL"
            ).fetchone()[0]
            logger.info("Added completed_at column, backfilled %d images", count)

    def nuke(self) -> None:
        """Purge all corpus state. Irreversible."""
        self._conn.executescript("""
            DELETE FROM characterizations;
            DELETE FROM kmeans_centroids;
            DELETE FROM kmeans_clusters;
            DELETE FROM umap_positions;
            DELETE FROM embeddings;
            DELETE FROM images;
        """)
        logger.info("Corpus nuked")

    def close(self) -> None:
        self._conn.close()

    # ── Batched IN queries ──

    _BATCH_SIZE = 900  # SQLite variable limit is 999; leave headroom

    def _query_in_batches(
        self, sql_template: str, ids: list[str], extra_params: list | None = None,
    ) -> list[sqlite3.Row]:
        """Execute a query with IN(...) clause, batching to stay under SQLite's variable limit.

        sql_template must contain {placeholders} where the IN list goes.
        extra_params are prepended to each batch's parameter list.
        """
        extra = extra_params or []
        results: list[sqlite3.Row] = []
        for i in range(0, len(ids), self._BATCH_SIZE):
            chunk = ids[i : i + self._BATCH_SIZE]
            placeholders = ",".join("?" * len(chunk))
            sql = sql_template.format(placeholders=placeholders)
            results.extend(self._conn.execute(sql, extra + chunk).fetchall())
        return results

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

    def mark_completed(self) -> int:
        """Mark images as complete when all artifacts are in place.

        An image's unit of work is done when it has metadata, thumbnail,
        and at least one embedding. Returns count of newly completed images.
        """
        cursor = self._conn.execute("""
            UPDATE images SET completed_at = ?
            WHERE completed_at IS NULL
              AND metadata_extracted_at IS NOT NULL
              AND thumbnail_generated_at IS NOT NULL
              AND id IN (SELECT image_id FROM embeddings)
        """, (time.time(),))
        self._conn.commit()
        return cursor.rowcount

    def fetch_content_hashes(self) -> set[str]:
        """Return all known content hashes for dedup."""
        rows = self._conn.execute(
            "SELECT content_hash FROM images WHERE content_hash IS NOT NULL"
        ).fetchall()
        return {r[0] for r in rows}

    def image_count(self) -> int:
        """Count of completed (visible) images."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM images WHERE completed_at IS NOT NULL"
        ).fetchone()
        return row[0]

    def fetch_image_ids(self) -> list[str]:
        """Return IDs of completed images only — the visible corpus."""
        rows = self._conn.execute(
            "SELECT id FROM images WHERE completed_at IS NOT NULL"
        ).fetchall()
        return [r[0] for r in rows]

    def fetch_capture_dates(self, image_ids: list[str]) -> dict[str, float]:
        """Return {image_id: capture_date} for images that have dates."""
        rows = self._query_in_batches(
            "SELECT id, capture_date FROM images WHERE id IN ({placeholders}) AND capture_date IS NOT NULL",
            image_ids,
        )
        return {r[0]: r[1] for r in rows}

    def fetch_images_without_thumbnails(self) -> list[tuple[str, str]]:
        """Return (id, source_path) for images missing thumbnails."""
        rows = self._conn.execute(
            "SELECT id, source_path FROM images WHERE thumbnail_generated_at IS NULL"
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def fetch_completed_images_with_paths(self) -> list[tuple[str, str]]:
        """Return (id, source_path) for every completed image. Used by the
        preview-regeneration sweep — touches every image, not just ones
        missing artifacts."""
        rows = self._conn.execute(
            "SELECT id, source_path FROM images WHERE completed_at IS NOT NULL"
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

    def fetch_image_metadata(self, image_id: str) -> dict | None:
        """Return all metadata columns for one image, or None if unknown.

        Powers the Lightbox metadata overlay — everything the image knows about
        itself that lives in the DB. Source path and EXIF-derived fields.
        """
        row = self._conn.execute(
            """SELECT id, source_path, capture_date, pixel_width, pixel_height,
                      gps_latitude, gps_longitude, camera_model, lens_model,
                      focal_length, aperture, shutter_speed, iso
               FROM images WHERE id = ?""",
            (image_id,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)

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

    # ── UMAP positions ──

    def has_umap(self, model: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM umap_positions WHERE model_identifier = ? LIMIT 1",
            (model,),
        ).fetchone()
        return row is not None

    def umap_count(self, model: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM umap_positions WHERE model_identifier = ?",
            (model,),
        ).fetchone()
        return row[0]

    def insert_umap_batch(
        self, model: str, positions: list[tuple[str, float, float]]
    ) -> None:
        """Insert batch of (image_id, x, y) UMAP positions."""
        now = time.time()
        self._conn.executemany(
            """INSERT OR REPLACE INTO umap_positions
               (image_id, model_identifier, x, y, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            [(iid, model, x, y, now) for iid, x, y in positions],
        )
        self._conn.commit()

    def fetch_umap_positions(
        self, model: str, image_ids: list[str]
    ) -> dict[str, tuple[float, float]]:
        """Return {image_id: (x, y)} for the given IDs."""
        rows = self._query_in_batches(
            "SELECT image_id, x, y FROM umap_positions "
            "WHERE model_identifier = ? AND image_id IN ({placeholders})",
            image_ids, extra_params=[model],
        )
        return {r[0]: (r[1], r[2]) for r in rows}

    # ── KMeans clusters ──

    def has_kmeans(self, model: str, k: int) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM kmeans_centroids WHERE model_identifier = ? AND k = ? LIMIT 1",
            (model, k),
        ).fetchone()
        return row is not None

    def insert_kmeans_batch(
        self, model: str, k: int, assignments: list[tuple[str, int]]
    ) -> None:
        now = time.time()
        self._conn.executemany(
            """INSERT OR REPLACE INTO kmeans_clusters
               (model_identifier, k, image_id, cluster_id, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            [(model, k, image_id, cluster_id, now) for image_id, cluster_id in assignments],
        )
        self._conn.commit()

    def insert_kmeans_centroids(
        self, model: str, k: int, centroids: dict[int, bytes]
    ) -> None:
        now = time.time()
        self._conn.executemany(
            """INSERT OR REPLACE INTO kmeans_centroids
               (model_identifier, k, cluster_id, centroid, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            [(model, k, cid, blob, now) for cid, blob in centroids.items()],
        )
        self._conn.commit()

    def fetch_kmeans_assignments_for_ids(
        self, model: str, k: int, image_ids: list[str]
    ) -> dict[str, int]:
        if not image_ids:
            return {}
        rows = self._query_in_batches(
            "SELECT image_id, cluster_id FROM kmeans_clusters "
            "WHERE model_identifier = ? AND k = ? AND image_id IN ({placeholders})",
            image_ids, extra_params=[model, k],
        )
        return {r[0]: r[1] for r in rows}

    def fetch_kmeans_centroids(self, model: str, k: int) -> dict[int, bytes]:
        rows = self._conn.execute(
            "SELECT cluster_id, centroid FROM kmeans_centroids WHERE model_identifier = ? AND k = ?",
            (model, k),
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    # ── Things library ──

    def list_things_library(self) -> list[str]:
        """Return all thing names, sorted by creation time."""
        rows = self._conn.execute(
            "SELECT name FROM things_library ORDER BY created_at ASC"
        ).fetchall()
        return [r[0] for r in rows]

    def add_thing_to_library(self, name: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO things_library (name, created_at) VALUES (?, ?)",
            (name, time.time()),
        )
        self._conn.commit()

    def remove_thing_from_library(self, name: str) -> None:
        self._conn.execute("DELETE FROM things_library WHERE name = ?", (name,))
        self._conn.commit()

    # ── Workspace state (key/value) ──

    def get_workspace_state(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM workspace_state WHERE key = ?", (key,),
        ).fetchone()
        return row[0] if row else None

    def set_workspace_state(self, key: str, value_json: str) -> None:
        self._conn.execute(
            "INSERT INTO workspace_state (key, value, modified_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, modified_at = excluded.modified_at",
            (key, value_json, time.time()),
        )
        self._conn.commit()

    # ── Collages ──

    def list_collages(self) -> list[dict]:
        """Return collage metadata (no thumbnail blob), most-recent first."""
        rows = self._conn.execute(
            "SELECT id, name, created_at, modified_at, mode, model, "
            "relevance, feathering, cell_size, "
            "(thumbnail_blob IS NOT NULL) AS has_thumbnail "
            "FROM collages ORDER BY modified_at DESC"
        ).fetchall()
        return [
            {
                "id": r[0], "name": r[1],
                "created_at": r[2], "modified_at": r[3],
                "mode": r[4], "model": r[5],
                "relevance": r[6], "feathering": r[7], "cell_size": r[8],
                "has_thumbnail": bool(r[9]),
            }
            for r in rows
        ]

    def fetch_collage(self, collage_id: str) -> dict | None:
        """Return a single collage with its full expression and pov, no thumbnail."""
        row = self._conn.execute(
            "SELECT id, name, created_at, modified_at, expression_json, pov_json, "
            "mode, model, relevance, feathering, cell_size "
            "FROM collages WHERE id = ?",
            (collage_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0], "name": row[1],
            "created_at": row[2], "modified_at": row[3],
            "expression_json": row[4], "pov_json": row[5],
            "mode": row[6], "model": row[7],
            "relevance": row[8], "feathering": row[9], "cell_size": row[10],
        }

    def fetch_collage_thumbnail(self, collage_id: str) -> bytes | None:
        row = self._conn.execute(
            "SELECT thumbnail_blob FROM collages WHERE id = ?", (collage_id,),
        ).fetchone()
        return row[0] if row and row[0] else None

    def insert_collage(
        self, collage_id: str, name: str,
        expression_json: str, pov_json: str,
        mode: str, model: str,
        relevance: float, feathering: float, cell_size: float,
        thumbnail_blob: bytes | None,
    ) -> None:
        now = time.time()
        self._conn.execute(
            "INSERT INTO collages (id, name, created_at, modified_at, "
            "expression_json, pov_json, mode, model, relevance, feathering, "
            "cell_size, thumbnail_blob) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (collage_id, name, now, now, expression_json, pov_json,
             mode, model, relevance, feathering, cell_size, thumbnail_blob),
        )
        self._conn.commit()

    def rename_collage(self, collage_id: str, new_name: str) -> bool:
        cur = self._conn.execute(
            "UPDATE collages SET name = ?, modified_at = ? WHERE id = ?",
            (new_name, time.time(), collage_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def delete_collage(self, collage_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM collages WHERE id = ?", (collage_id,))
        self._conn.commit()
        return cur.rowcount > 0

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
