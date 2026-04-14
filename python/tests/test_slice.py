"""Tests for SliceMode filtering."""

import numpy as np
import pytest

from sigil_atlas.db import CorpusDB, ImageRecord
from sigil_atlas.slice import RangeFilter, ProximityFilter, filter_by_range, filter_by_proximity, compute_slice


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
        # Return a random unit vector
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
        db.insert_image(ImageRecord(
            id=f"img_{i}",
            source_path=f"/test/img_{i}.jpg",
            created_at=time.time(),
        ))
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
        # brightness values: 0.1, 0.3, 0.5, 0.7, 0.9 -> 0.3, 0.5, 0.7 pass
        assert len(result) == 3
        assert "img_1" in result  # 0.3
        assert "img_2" in result  # 0.5
        assert "img_3" in result  # 0.7

    def test_tight_range_filter(self, seeded_db):
        filters = [RangeFilter(dimension="brightness", min_value=0.45, max_value=0.55)]
        result = filter_by_range(seeded_db, filters)
        assert result == {"img_2"}  # only 0.5

    def test_no_matches(self, seeded_db):
        filters = [RangeFilter(dimension="brightness", min_value=2.0, max_value=3.0)]
        result = filter_by_range(seeded_db, filters)
        assert len(result) == 0

    def test_nonexistent_dimension(self, seeded_db):
        filters = [RangeFilter(dimension="contrast", min_value=0.0, max_value=1.0)]
        result = filter_by_range(seeded_db, filters)
        assert len(result) == 0


class TestFilterByProximity:
    def test_no_filters_returns_all(self):
        provider = FakeEmbeddingProvider({"a": np.ones(4, dtype=np.float32)})
        result = filter_by_proximity(provider, ["a"], [], "test")
        assert result == ["a"]

    def test_scores_by_similarity(self):
        # Create embeddings where img_0 is close to text, img_1 is far
        dim = 8
        text_vec = np.zeros(dim, dtype=np.float32)
        text_vec[0] = 1.0  # unit vector along dim 0

        close_vec = np.zeros(dim, dtype=np.float32)
        close_vec[0] = 0.9
        close_vec[1] = 0.1
        close_vec = close_vec / np.linalg.norm(close_vec)

        far_vec = np.zeros(dim, dtype=np.float32)
        far_vec[3] = 1.0  # orthogonal

        provider = FakeEmbeddingProvider(
            {"close": close_vec, "far": far_vec},
            {"sunset": text_vec},
        )
        result = filter_by_proximity(
            provider, ["close", "far"],
            [ProximityFilter(text="sunset")],
            model="test",
            threshold=0.1,
        )
        assert result[0] == "close"
        assert "far" not in result  # orthogonal = score ~0, below threshold


class TestComputeSlice:
    def test_no_filters_returns_all(self, seeded_db):
        provider = FakeEmbeddingProvider({})
        result = compute_slice(seeded_db, provider)
        assert len(result) == 5

    def test_range_only(self, seeded_db):
        provider = FakeEmbeddingProvider({})
        result = compute_slice(
            seeded_db, provider,
            range_filters=[RangeFilter("brightness", 0.0, 0.2)],
        )
        assert len(result) == 1
        assert result[0] == "img_0"

    def test_invariant_slice_never_affects_arrangement(self, seeded_db):
        """Verify the !content-only invariant: slice only controls membership."""
        provider = FakeEmbeddingProvider({})
        full = compute_slice(seeded_db, provider)
        filtered = compute_slice(
            seeded_db, provider,
            range_filters=[RangeFilter("brightness", 0.0, 0.5)],
        )
        # filtered is a strict subset of full
        assert set(filtered).issubset(set(full))
        assert len(filtered) < len(full)
