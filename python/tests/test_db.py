"""Tests for the corpus database."""

import tempfile
import time
from pathlib import Path

from sigil_atlas.db import CorpusDB, ImageRecord


def _make_completed(db: CorpusDB, image_id: str) -> None:
    """Mark an image as completed (metadata + thumbnail + embedding done)."""
    now = time.time()
    db._conn.execute(
        "UPDATE images SET metadata_extracted_at=?, thumbnail_generated_at=?, completed_at=? WHERE id=?",
        (now, now, now, image_id),
    )
    db._conn.commit()


def test_schema_initialization():
    with tempfile.TemporaryDirectory() as tmp:
        db = CorpusDB(Path(tmp) / "test.db")
        db.initialize_schema()
        assert db.image_count() == 0
        db.close()


def test_insert_and_count():
    with tempfile.TemporaryDirectory() as tmp:
        db = CorpusDB(Path(tmp) / "test.db")
        db.initialize_schema()

        rec = ImageRecord(id="test-1", source_path="/tmp/img.jpg")
        db.insert_image(rec)
        _make_completed(db, "test-1")
        assert db.image_count() == 1
        db.close()


def test_insert_batch():
    with tempfile.TemporaryDirectory() as tmp:
        db = CorpusDB(Path(tmp) / "test.db")
        db.initialize_schema()

        records = [
            ImageRecord(id=f"img-{i}", source_path=f"/tmp/img{i}.jpg")
            for i in range(100)
        ]
        db.insert_images_batch(records)
        for i in range(100):
            _make_completed(db, f"img-{i}")
        assert db.image_count() == 100
        db.close()


def test_insert_duplicate_ignored():
    with tempfile.TemporaryDirectory() as tmp:
        db = CorpusDB(Path(tmp) / "test.db")
        db.initialize_schema()

        rec = ImageRecord(id="test-1", source_path="/tmp/img.jpg")
        db.insert_image(rec)
        db.insert_image(rec)
        _make_completed(db, "test-1")
        assert db.image_count() == 1
        db.close()


def test_uncompleted_images_not_counted():
    """image_count only counts completed images."""
    with tempfile.TemporaryDirectory() as tmp:
        db = CorpusDB(Path(tmp) / "test.db")
        db.initialize_schema()

        db.insert_image(ImageRecord(id="done", source_path="/tmp/done.jpg"))
        db.insert_image(ImageRecord(id="pending", source_path="/tmp/pending.jpg"))
        _make_completed(db, "done")

        assert db.image_count() == 1
        assert "done" in db.fetch_image_ids()
        assert "pending" not in db.fetch_image_ids()
        db.close()


def test_embeddings_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        db = CorpusDB(Path(tmp) / "test.db")
        db.initialize_schema()

        rec = ImageRecord(id="img-1", source_path="/tmp/img.jpg")
        db.insert_image(rec)

        vector = [0.1 * i for i in range(512)]
        db.insert_embeddings_batch([("img-1", "clip-vit-b-32", vector)])

        retrieved = db.fetch_embedding("img-1", "clip-vit-b-32")
        assert retrieved is not None
        assert len(retrieved) == 512
        for a, b in zip(vector, retrieved):
            assert abs(a - b) < 1e-5

        db.close()


def test_fetch_unembedded():
    with tempfile.TemporaryDirectory() as tmp:
        db = CorpusDB(Path(tmp) / "test.db")
        db.initialize_schema()

        for i in range(5):
            rec = ImageRecord(id=f"img-{i}", source_path=f"/tmp/img{i}.jpg")
            db.insert_image(rec)

        db.update_thumbnail("img-0", "img-0.jpg")
        db.update_thumbnail("img-1", "img-1.jpg")
        db.update_thumbnail("img-2", "img-2.jpg")

        db.insert_embeddings_batch([("img-0", "clip-vit-b-32", [0.0] * 512)])

        unembedded = db.fetch_unembedded_image_ids("clip-vit-b-32")
        assert set(unembedded) == {"img-1", "img-2"}

        db.close()


def test_mark_completed():
    """mark_completed transitions images with all artifacts."""
    with tempfile.TemporaryDirectory() as tmp:
        db = CorpusDB(Path(tmp) / "test.db")
        db.initialize_schema()

        db.insert_image(ImageRecord(id="img-1", source_path="/tmp/img.jpg"))
        db.update_metadata("img-1", metadata_extracted_at=time.time())
        db.update_thumbnail("img-1", "img-1.jpg")
        db.insert_embeddings_batch([("img-1", "clip-vit-b-32", [0.0] * 512)])

        count = db.mark_completed()
        assert count == 1
        assert db.image_count() == 1

        # Second call should be idempotent
        count = db.mark_completed()
        assert count == 0
        db.close()


def test_update_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        db = CorpusDB(Path(tmp) / "test.db")
        db.initialize_schema()

        rec = ImageRecord(id="img-1", source_path="/tmp/img.jpg")
        db.insert_image(rec)

        db.update_metadata(
            "img-1",
            pixel_width=1024,
            pixel_height=768,
            camera_model="X-T5",
        )

        row = db._conn.execute(
            "SELECT pixel_width, pixel_height, camera_model FROM images WHERE id = ?",
            ("img-1",),
        ).fetchone()
        assert row[0] == 1024
        assert row[1] == 768
        assert row[2] == "X-T5"

        db.close()
