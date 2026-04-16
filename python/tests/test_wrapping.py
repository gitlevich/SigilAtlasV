"""Tests for image wrapping — tree-walk characterization via CLIP."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from sigil_atlas.db import CorpusDB, ImageRecord
from sigil_atlas.ontology import get_ontology, tree_depth
from sigil_atlas.wrapping import (
    ImageCharacterization,
    OntologyIndex,
    WRAPPING_MODEL,
    characterize_image,
)
from sigil_atlas.model_registry import get_adapter


@pytest.fixture(scope="module")
def ontology_index():
    return OntologyIndex()


def test_ontology_index_embeddings_exist(ontology_index):
    adapter = get_adapter(WRAPPING_MODEL)
    assert len(ontology_index._embeddings) > 0
    for name, emb in ontology_index._embeddings.items():
        assert emb.shape == (adapter.dimension,), f"{name} has wrong shape: {emb.shape}"


def test_ontology_index_normalized(ontology_index):
    for name, emb in ontology_index._embeddings.items():
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-4, f"{name} not normalized: {norm}"


def test_similarity_returns_float(ontology_index):
    adapter = get_adapter(WRAPPING_MODEL)
    fake_emb = np.random.randn(adapter.dimension).astype(np.float32)
    sim = ontology_index.similarity(fake_emb, "dark")
    assert isinstance(sim, float)
    assert -1.0 <= sim <= 1.0


def test_characterize_produces_path(ontology_index):
    adapter = get_adapter(WRAPPING_MODEL)
    fake_emb = np.random.randn(adapter.dimension).astype(np.float32)
    path = characterize_image(fake_emb, ontology_index)
    assert len(path) >= 2, f"Path too short: {path}"
    root = get_ontology()
    valid_first = {c.name for c in root.children}
    assert path[0] in valid_first, f"First choice {path[0]} not in {valid_first}"


def test_path_length_bounded_by_depth(ontology_index):
    adapter = get_adapter(WRAPPING_MODEL)
    depth = tree_depth()
    fake_emb = np.random.randn(adapter.dimension).astype(np.float32)
    path = characterize_image(fake_emb, ontology_index)
    assert len(path) <= depth, f"Path length {len(path)} exceeds tree depth {depth}"


def test_invariant_labels_are_prefixes(ontology_index):
    adapter = get_adapter(WRAPPING_MODEL)
    fake_emb = np.random.randn(adapter.dimension).astype(np.float32)
    path = characterize_image(fake_emb, ontology_index)
    char = ImageCharacterization(image_id="test", path=path)
    labels = char.invariant_labels
    for label in labels:
        parts = label.split("/")
        assert parts == path[: len(parts)]


def test_different_embeddings_can_differ(ontology_index):
    adapter = get_adapter(WRAPPING_MODEL)
    paths = set()
    for _ in range(20):
        fake_emb = np.random.randn(adapter.dimension).astype(np.float32)
        path = tuple(characterize_image(fake_emb, ontology_index))
        paths.add(path)
    assert len(paths) > 1, "All random embeddings produced identical paths"


def test_characterizations_db_roundtrip(ontology_index):
    adapter = get_adapter(WRAPPING_MODEL)
    fake_emb = np.random.randn(adapter.dimension).astype(np.float32)
    path = characterize_image(fake_emb, ontology_index)

    with tempfile.TemporaryDirectory() as tmp:
        db = CorpusDB(Path(tmp) / "test.db")
        db.initialize_schema()

        image_id = "test-img-1"
        db.insert_image(ImageRecord(id=image_id, source_path="/tmp/test.jpg"))

        db.insert_embeddings_batch([(image_id, WRAPPING_MODEL, [0.0] * adapter.dimension)])

        rows = []
        for depth, pole_name in enumerate(path):
            prefix = "/".join(path[: depth + 1])
            rows.append((image_id, prefix, "enum", pole_name, None))

        db.insert_characterizations_batch(rows)

        retrieved = db.fetch_characterizations(image_id)
        assert len(retrieved) == len(path)

        db.close()
