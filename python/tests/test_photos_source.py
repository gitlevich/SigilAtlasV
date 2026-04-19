"""Tests for PhotosSource — the receiver of sa-photos NDJSON records."""

import tempfile
from pathlib import Path

from sigil_atlas.db import CorpusDB
from sigil_atlas.ingest.photos_source import PhotosRecord, PhotosSource, _identity_hash


def _rec(local_id: str = "ABC-123/L0/001", **overrides) -> PhotosRecord:
    defaults = dict(
        id=local_id,
        capture_date=1680000000.0,
        w=4032,
        h=3024,
        lat=37.7,
        lon=-122.4,
        is_live=False,
        is_screenshot=False,
        favorite=False,
    )
    defaults.update(overrides)
    return PhotosRecord(**defaults)


def test_from_json_populates_fields():
    data = {
        "id": "XYZ",
        "capture_date": 1.0,
        "w": 100,
        "h": 200,
        "lat": 1.5,
        "lon": -1.5,
        "is_live": True,
        "is_screenshot": False,
        "favorite": True,
    }
    rec = PhotosRecord.from_json(data)
    assert rec.id == "XYZ"
    assert rec.is_live is True
    assert rec.favorite is True
    assert rec.w == 100


def test_identity_hash_is_stable():
    h1 = _identity_hash(_rec())
    h2 = _identity_hash(_rec())
    assert h1 == h2
    assert len(h1) == 64


def test_identity_hash_differs_by_local_id():
    h1 = _identity_hash(_rec(local_id="A"))
    h2 = _identity_hash(_rec(local_id="B"))
    assert h1 != h2


def test_register_batch_inserts_with_prefilled_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        db = CorpusDB(Path(tmp) / "test.db")
        db.initialize_schema()

        source = PhotosSource()
        assigned, skipped = source.register_batch(db, [_rec()])

        assert len(assigned) == 1
        assert skipped == 0
        image_id, local_id = assigned[0]
        assert local_id == "ABC-123/L0/001"

        row = db._conn.execute(
            "SELECT source_path, capture_date, pixel_width, gps_latitude, metadata_extracted_at "
            "FROM images WHERE id = ?",
            (image_id,),
        ).fetchone()
        assert row["source_path"] == "photos://ABC-123/L0/001"
        assert row["capture_date"] == 1680000000.0
        assert row["pixel_width"] == 4032
        assert row["gps_latitude"] == 37.7
        # metadata stage must skip this row
        assert row["metadata_extracted_at"] is not None
        db.close()


def test_register_batch_dedups_by_identity_hash():
    with tempfile.TemporaryDirectory() as tmp:
        db = CorpusDB(Path(tmp) / "test.db")
        db.initialize_schema()

        source = PhotosSource()
        source.register_batch(db, [_rec()])
        assigned, skipped = source.register_batch(db, [_rec()])

        assert assigned == []
        assert skipped == 1
        total = db._conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        assert total == 1
        db.close()


def test_register_batch_handles_missing_fields():
    with tempfile.TemporaryDirectory() as tmp:
        db = CorpusDB(Path(tmp) / "test.db")
        db.initialize_schema()

        rec = _rec(capture_date=None, lat=None, lon=None)
        source = PhotosSource()
        assigned, _ = source.register_batch(db, [rec])
        assert len(assigned) == 1

        row = db._conn.execute(
            "SELECT capture_date, gps_latitude, gps_longitude FROM images WHERE id = ?",
            (assigned[0][0],),
        ).fetchone()
        assert row["capture_date"] is None
        assert row["gps_latitude"] is None
        db.close()
