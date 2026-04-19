"""Source — Apple Photos library via the sa-photos Swift helper.

The helper runs inside the Tauri app and streams one NDJSON record per still
image to the Rust side, which POSTs batches to `/sources/photos/ingest`. This
module turns those records into `ImageRecord` rows.

The records already carry the metadata the pipeline would otherwise extract
from EXIF (capture date, pixel dimensions, GPS), so we pre-fill those columns
and stamp `metadata_extracted_at` — the metadata stage then skips these rows.
Thumbnails are not present at register time; they arrive in a second pass
driven by the Tauri `photos_thumb` command.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass

from sigil_atlas.db import CorpusDB, ImageRecord

logger = logging.getLogger(__name__)


@dataclass
class PhotosRecord:
    """One record as emitted by `sa-photos enumerate`."""
    id: str                       # PHAsset.localIdentifier
    capture_date: float | None
    w: int | None
    h: int | None
    lat: float | None
    lon: float | None
    is_live: bool
    is_screenshot: bool
    favorite: bool

    @classmethod
    def from_json(cls, data: dict) -> "PhotosRecord":
        return cls(
            id=data["id"],
            capture_date=data.get("capture_date"),
            w=data.get("w"),
            h=data.get("h"),
            lat=data.get("lat"),
            lon=data.get("lon"),
            is_live=bool(data.get("is_live", False)),
            is_screenshot=bool(data.get("is_screenshot", False)),
            favorite=bool(data.get("favorite", False)),
        )


class PhotosSource:
    """Receives photo records streamed from the sa-photos helper.

    Uniquely identified by the label "photos". Not path-movable — the System
    Photos library is a global singleton.
    """

    SCHEME = "photos://"

    @property
    def location(self) -> str:
        return "photos://system"

    def register_batch(
        self,
        db: CorpusDB,
        records: list[PhotosRecord],
    ) -> tuple[list[tuple[str, str]], int]:
        """Insert new photo records into the corpus.

        Returns ([(image_id, local_identifier)], skipped). The returned list
        lets the caller drive thumbnail generation (Tauri invokes
        `sa-photos thumb` per entry and writes to `<thumbnails_dir>/<image_id>.jpg`).

        Dedup key: content_hash derived from the stable-identity fingerprint
        (local_id, capture_date, w, h). Re-enumerations of the same library
        therefore collapse to the same hash.
        """
        if not records:
            return [], 0

        known_hashes = db.fetch_content_hashes()
        batch: list[ImageRecord] = []
        assigned: list[tuple[str, str]] = []
        skipped = 0
        now = time.time()

        for rec in records:
            h = _identity_hash(rec)
            if h in known_hashes:
                skipped += 1
                continue
            known_hashes.add(h)

            image_id = str(uuid.uuid4())
            batch.append(ImageRecord(
                id=image_id,
                source_path=f"{self.SCHEME}{rec.id}",
                content_hash=h,
                capture_date=rec.capture_date,
                pixel_width=rec.w,
                pixel_height=rec.h,
                gps_latitude=rec.lat,
                gps_longitude=rec.lon,
                metadata_extracted_at=now,
                created_at=now,
            ))
            assigned.append((image_id, rec.id))

        if batch:
            db.insert_images_batch(batch)
        logger.info(
            "Photos source: registered %d new, skipped %d duplicates",
            len(batch), skipped,
        )
        return assigned, skipped


def _identity_hash(rec: PhotosRecord) -> str:
    """Stable fingerprint of a photo asset. SHA-256 hex over a canonical
    string representation. Does not read pixel bytes, so iCloud-only assets
    hash without triggering a download.
    """
    parts = [
        rec.id,
        f"{rec.capture_date:.3f}" if rec.capture_date is not None else "",
        str(rec.w or ""),
        str(rec.h or ""),
    ]
    payload = "|".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
