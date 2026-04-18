"""Tests for the RelevanceFilter membrane and the resulting slice."""

import numpy as np
import pytest

from sigil_atlas.db import CorpusDB, ImageRecord
from sigil_atlas.relevance_filter import And, Or, Not, Range, Thing, evaluate, parse, Context
from sigil_atlas.slice import compute_slice


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
        db._conn.execute(
            "UPDATE images SET completed_at=?, metadata_extracted_at=?, thumbnail_generated_at=? WHERE id=?",
            (time.time(), time.time(), time.time(), f"img_{i}"),
        )
    db._conn.commit()

    rows = []
    for i in range(5):
        rows.append((f"img_{i}", "brightness", "range", None, 0.1 + i * 0.2))
    db.insert_characterizations_batch(rows)
    return db


def _ctx(db, provider, model="test-model", relevance=0.5):
    return Context(
        db=db, provider=provider, model=model, relevance=relevance,
        corpus_ids=db.fetch_image_ids(),
    )


class TestRangeAtom:
    def test_single_range(self, seeded_db):
        provider = FakeEmbeddingProvider({})
        ctx = _ctx(seeded_db, provider)
        result = evaluate(Range("brightness", 0.25, 0.75), ctx)
        assert result == {"img_1", "img_2", "img_3"}

    def test_tight_range(self, seeded_db):
        provider = FakeEmbeddingProvider({})
        ctx = _ctx(seeded_db, provider)
        result = evaluate(Range("brightness", 0.45, 0.55), ctx)
        assert result == {"img_2"}

    def test_no_matches(self, seeded_db):
        provider = FakeEmbeddingProvider({})
        ctx = _ctx(seeded_db, provider)
        result = evaluate(Range("brightness", 2.0, 3.0), ctx)
        assert result == set()


class TestComposition:
    def test_and_intersects(self, seeded_db):
        provider = FakeEmbeddingProvider({})
        ctx = _ctx(seeded_db, provider)
        expr = And((
            Range("brightness", 0.0, 0.75),  # img_0..img_3
            Range("brightness", 0.25, 1.0),  # img_1..img_4
        ))
        assert evaluate(expr, ctx) == {"img_1", "img_2", "img_3"}

    def test_or_unions(self, seeded_db):
        provider = FakeEmbeddingProvider({})
        ctx = _ctx(seeded_db, provider)
        expr = Or((
            Range("brightness", 0.0, 0.25),  # img_0
            Range("brightness", 0.75, 1.0),  # img_4
        ))
        assert evaluate(expr, ctx) == {"img_0", "img_4"}

    def test_not_complements(self, seeded_db):
        provider = FakeEmbeddingProvider({})
        ctx = _ctx(seeded_db, provider)
        expr = Not(Range("brightness", 0.25, 0.75))
        assert evaluate(expr, ctx) == {"img_0", "img_4"}

    def test_empty_and_is_all(self, seeded_db):
        provider = FakeEmbeddingProvider({})
        ctx = _ctx(seeded_db, provider)
        assert evaluate(And(()), ctx) == {"img_0", "img_1", "img_2", "img_3", "img_4"}


class TestComputeSlice:
    def test_none_filter_returns_all(self, seeded_db):
        provider = FakeEmbeddingProvider({})
        result = compute_slice(seeded_db, provider, None, model="test-model")
        assert len(result.image_ids) == 5

    def test_range_atom(self, seeded_db):
        provider = FakeEmbeddingProvider({})
        result = compute_slice(
            seeded_db, provider,
            Range("brightness", 0.0, 0.2),
            model="test-model",
        )
        assert result.image_ids == ["img_0"]

    def test_slice_is_subset(self, seeded_db):
        """Filtered slice is a strict subset of unfiltered."""
        provider = FakeEmbeddingProvider({})
        full = compute_slice(seeded_db, provider, None, model="test-model")
        filtered = compute_slice(
            seeded_db, provider,
            Range("brightness", 0.0, 0.5),
            model="test-model",
        )
        assert set(filtered.image_ids).issubset(set(full.image_ids))
        assert len(filtered.image_ids) < len(full.image_ids)


class TestParse:
    def test_thing(self):
        assert parse({"type": "thing", "name": "bird"}) == Thing("bird")

    def test_range(self):
        assert parse({"type": "range", "dimension": "brightness", "min": 0.2, "max": 0.8}) == Range("brightness", 0.2, 0.8)

    def test_and_of_things(self):
        node = {
            "type": "and",
            "children": [
                {"type": "thing", "name": "bird"},
                {"type": "thing", "name": "bee"},
            ],
        }
        expr = parse(node)
        assert isinstance(expr, And)
        assert expr.children == (Thing("bird"), Thing("bee"))

    def test_nested(self):
        node = {
            "type": "and",
            "children": [
                {"type": "thing", "name": "bird"},
                {"type": "not", "child": {"type": "range", "dimension": "brightness", "min": 0, "max": 0.2}},
            ],
        }
        expr = parse(node)
        assert isinstance(expr, And)
        assert isinstance(expr.children[1], Not)
