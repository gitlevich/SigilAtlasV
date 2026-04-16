"""Tests for folder source scanning."""

import tempfile
from pathlib import Path

from PIL import Image

from sigil_atlas.db import CorpusDB
from sigil_atlas.ingest.source import FolderSource, content_hash


def _create_test_image(path: Path, size: tuple[int, int] = (100, 100)) -> None:
    img = Image.new("RGB", size, color="red")
    img.save(path, "JPEG")
    img.close()


def test_scan_finds_images():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _create_test_image(root / "a.jpg")
        _create_test_image(root / "b.png")
        (root / "c.txt").write_text("not an image")

        source = FolderSource(root)
        files = source.scan()

        assert len(files) == 2
        names = {f.name for f in files}
        assert "a.jpg" in names
        assert "b.png" in names


def test_scan_recursive():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sub = root / "subdir"
        sub.mkdir()
        _create_test_image(root / "a.jpg")
        _create_test_image(sub / "b.jpg")

        source = FolderSource(root)
        files = source.scan()
        assert len(files) == 2


def test_register_images():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        img_dir = root / "images"
        img_dir.mkdir()
        _create_test_image(img_dir / "a.jpg", size=(100, 100))
        _create_test_image(img_dir / "b.jpg", size=(200, 100))

        db = CorpusDB(root / "test.db")
        db.initialize_schema()

        source = FolderSource(img_dir)
        files = source.scan()
        count = source.register_images(db, files)

        assert count == 2
        # image_count() returns only completed images; registered but
        # not-yet-processed images are invisible by design (unit of work).
        total = db._conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        assert total == 2
        db.close()


def test_content_hash_deterministic():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.jpg"
        _create_test_image(path)

        h1 = content_hash(path)
        h2 = content_hash(path)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex
