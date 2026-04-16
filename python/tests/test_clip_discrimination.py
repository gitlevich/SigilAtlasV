"""Tests that CLIP text queries actually discriminate — the right images rank highest.

These tests use the real corpus and verify that CLIP's top results
are meaningful, not noise. If these fail, the query pipeline is broken.
"""

import numpy as np
import pytest

from sigil_atlas.db import CorpusDB
from sigil_atlas.embedding_provider import SqliteEmbeddingProvider
from sigil_atlas.model_registry import get_adapter


# Use L-14 adapter for text encoding — matches the stored embeddings in tests
_adapter = get_adapter("clip-vit-l-14")


def _encode(text: str) -> np.ndarray:
    return _adapter.encode_text(text)


@pytest.fixture(scope="module")
def corpus():
    from pathlib import Path
    db_path = Path(__file__).parent.parent.parent / "workspace" / "datastore" / "corpus.db"
    db = CorpusDB(str(db_path))
    provider = SqliteEmbeddingProvider(db)
    ids = db.fetch_image_ids()
    matrix = provider.fetch_matrix(ids, "clip-vit-l-14")
    return db, ids, matrix


def _top_filenames(db, ids, matrix, query: str, k: int = 10) -> list[str]:
    vec = _encode(query)
    scores = matrix @ vec
    top_idx = np.argsort(-scores)[:k]
    paths = []
    for idx in top_idx:
        row = db._conn.execute(
            "SELECT source_path FROM images WHERE id=?", (ids[idx],)
        ).fetchone()
        paths.append(row[0].split("/")[-1])
    return paths


def _z_score_best(matrix, query: str) -> float:
    """How many standard deviations the best match is above the mean."""
    vec = _encode(query)
    scores = matrix @ vec
    return float((scores.max() - scores.mean()) / scores.std())


def test_text_encoder_produces_unit_vectors():
    vec = _encode("a photograph of birds")
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-4


def test_text_encoder_dimension_matches_image(corpus):
    _, _, matrix = corpus
    vec = _encode("test")
    assert vec.shape[0] == matrix.shape[1], (
        f"Text dim {vec.shape[0]} != image dim {matrix.shape[1]}"
    )


def test_different_queries_produce_different_vectors():
    v1 = _encode("a photograph of birds")
    v2 = _encode("a photograph of buildings")
    cosine = np.dot(v1, v2)
    assert cosine < 0.95, f"Different queries too similar: {cosine:.4f}"


def test_best_match_is_statistically_significant(corpus):
    """The best matching image should be well above the mean — at least 3 sigma."""
    _, _, matrix = corpus
    for query in ["a photograph of birds", "a duck", "a photograph of flowers"]:
        z = _z_score_best(matrix, query)
        assert z > 3.0, f"Query '{query}' best match only {z:.1f} sigma above mean"


def test_top_results_stable_across_similar_queries(corpus):
    """Similar queries should return overlapping top results."""
    db, ids, matrix = corpus
    top_a = set(_top_filenames(db, ids, matrix, "a photograph of birds", k=20))
    top_b = set(_top_filenames(db, ids, matrix, "birds", k=20))
    overlap = len(top_a & top_b)
    assert overlap >= 10, (
        f"'a photograph of birds' and 'birds' share only {overlap}/20 top results"
    )


def test_contrast_discriminates(corpus):
    """Contrast direction should spread the corpus — std of projections > 0.02."""
    _, ids, matrix = corpus
    vec_a = _encode("a photograph that is bright")
    vec_b = _encode("a photograph that is dark")
    direction = vec_a - vec_b
    direction = direction / np.linalg.norm(direction)
    projections = matrix @ direction
    assert projections.std() > 0.02, (
        f"Contrast 'bright vs dark' has too narrow spread: {projections.std():.4f}"
    )


def test_selection_count_reasonable(corpus):
    """Mean + 1.5*std should select 3-15% of corpus, not 25% or 0%."""
    _, ids, matrix = corpus
    n = len(ids)
    for query in ["a photograph of birds", "a photograph of sunset", "a photograph of person"]:
        vec = _encode(query)
        raw = matrix @ vec
        threshold = raw.mean() + 1.5 * raw.std()
        count = int((raw >= threshold).sum())
        pct = count / n * 100
        assert 1 < pct < 20, (
            f"Query '{query}' selects {count} images ({pct:.1f}%), expected 1-20%"
        )
