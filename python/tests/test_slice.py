"""Tests for SliceMode filtering."""

import numpy as np
import pytest

from sigil_atlas.db import CorpusDB, ImageRecord
from sigil_atlas.slice import RangeFilter, ProximityFilter, filter_by_range, compute_slice


class FakeEmbeddingProvider:
    """In-memory embedding provider for testing."""

    def __init__(self, embeddings: dict[str, np.ndarray], text_embeddings: dict[str, np.ndarray] | None = None):
        self._embeddings = embeddings
        self._text_embeddings = text_embeddings or {}

    def fetch_matrix(self, image_ids: list[str], model: str) -> np.ndarray:
        return np.stack([self._embeddings[iid] for iid in image_ids])

    def encode_text(self, text: str, model: str) -> np.ndarray:
        if text in self._text_embeddings:
            return self._text_embeddings[text]
        vec = np.random.randn(next(iter(self._embeddings.values())).shape[0]).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def available_models(self) -> list[str]:
        return ["test-model"]


@pytest.fixture
def db(tmp_path):
    db = CorpusDB(tmp_path / "test.db")
    db.initialize_schema()
    return db


@pytest.fixture
def seeded_db(db):
    """DB with 5 images and characterizations."""
    import time
    for i in range(5):
        rec = ImageRecord(
            id=f"img_{i}",
            source_path=f"/test/img_{i}.jpg",
            created_at=time.time(),
        )
        db.insert_image(rec)
        # Mark as completed so fetch_image_ids returns them
        db._conn.execute(
            "UPDATE images SET completed_at=?, metadata_extracted_at=?, thumbnail_generated_at=? WHERE id=?",
            (time.time(), time.time(), time.time(), f"img_{i}"),
        )
    db._conn.commit()

    # Add range characterizations: "brightness" ranges from 0.1 to 0.9
    rows = []
    for i in range(5):
        rows.append((f"img_{i}", "brightness", "range", None, 0.1 + i * 0.2))
    db.insert_characterizations_batch(rows)
    return db


class TestFilterByRange:
    def test_no_filters_returns_all(self, seeded_db):
        result = filter_by_range(seeded_db, [])
        assert len(result) == 5

    def test_single_range_filter(self, seeded_db):
        filters = [RangeFilter(dimension="brightness", min_value=0.25, max_value=0.75)]
        result = filter_by_range(seeded_db, filters)
        assert len(result) == 3
        assert "img_1" in result
        assert "img_2" in result
        assert "img_3" in result

    def test_tight_range_filter(self, seeded_db):
        filters = [RangeFilter(dimension="brightness", min_value=0.45, max_value=0.55)]
        result = filter_by_range(seeded_db, filters)
        assert result == {"img_2"}

    def test_no_matches(self, seeded_db):
        filters = [RangeFilter(dimension="brightness", min_value=2.0, max_value=3.0)]
        result = filter_by_range(seeded_db, filters)
        assert len(result) == 0

    def test_nonexistent_dimension(self, seeded_db):
        filters = [RangeFilter(dimension="contrast", min_value=0.0, max_value=1.0)]
        result = filter_by_range(seeded_db, filters)
        assert len(result) == 0


class TestComputeSlice:
    def test_no_filters_returns_all(self, seeded_db):
        provider = FakeEmbeddingProvider({})
        result = compute_slice(seeded_db, provider)
        assert len(result.image_ids) == 5

    def test_range_only(self, seeded_db):
        provider = FakeEmbeddingProvider({})
        result = compute_slice(
            seeded_db, provider,
            range_filters=[RangeFilter("brightness", 0.0, 0.2)],
        )
        assert len(result.image_ids) == 1
        assert result.image_ids[0] == "img_0"

    def test_slice_is_subset(self, seeded_db):
        """Filtered slice is a strict subset of unfiltered."""
        provider = FakeEmbeddingProvider({})
        full = compute_slice(seeded_db, provider)
        filtered = compute_slice(
            seeded_db, provider,
            range_filters=[RangeFilter("brightness", 0.0, 0.5)],
        )
        assert set(filtered.image_ids).issubset(set(full.image_ids))
        assert len(filtered.image_ids) < len(full.image_ids)
