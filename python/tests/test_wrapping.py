"""Tests for image wrapping — tree-walk characterization via CLIP."""

import tempfile
from pathlib import Path

import numpy as np
import open_clip
import pytest
import torch

from sigil_atlas.db import CorpusDB, ImageRecord
from sigil_atlas.ontology import get_ontology, tree_depth
from sigil_atlas.wrapping import (
    ImageCharacterization,
    OntologyIndex,
    characterize_image,
)


@pytest.fixture(scope="module")
def clip_model():
    device = torch.device("cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, preprocess, tokenizer, device


@pytest.fixture(scope="module")
def ontology_index(clip_model):
    model, _, tokenizer, device = clip_model
    return OntologyIndex(model, tokenizer, device)


def test_ontology_index_embeddings_exist(ontology_index):
    assert len(ontology_index._embeddings) > 0
    for name, emb in ontology_index._embeddings.items():
        assert emb.shape == (512,), f"{name} has wrong shape: {emb.shape}"


def test_ontology_index_normalized(ontology_index):
    for name, emb in ontology_index._embeddings.items():
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-4, f"{name} not normalized: {norm}"


def test_similarity_returns_float(ontology_index):
    fake_emb = np.random.randn(512).astype(np.float32)
    sim = ontology_index.similarity(fake_emb, "dark")
    assert isinstance(sim, float)
    assert -1.0 <= sim <= 1.0


def test_characterize_produces_path(ontology_index):
    fake_emb = np.random.randn(512).astype(np.float32)
    path = characterize_image(fake_emb, ontology_index)
    assert len(path) >= 2, f"Path too short: {path}"
    # First element should be a child of root (subject or environment)
    root = get_ontology()
    valid_first = {c.name for c in root.children}
    assert path[0] in valid_first, f"First choice {path[0]} not in {valid_first}"


def test_path_length_bounded_by_depth(ontology_index):
    depth = tree_depth()
    fake_emb = np.random.randn(512).astype(np.float32)
    path = characterize_image(fake_emb, ontology_index)
    assert len(path) <= depth, f"Path length {len(path)} exceeds tree depth {depth}"


def test_invariant_labels_are_prefixes(ontology_index):
    fake_emb = np.random.randn(512).astype(np.float32)
    path = characterize_image(fake_emb, ontology_index)
    char = ImageCharacterization(image_id="test", path=path)
    labels = char.invariant_labels
    # Each label should be a slash-separated prefix of the path
    for label in labels:
        parts = label.split("/")
        assert parts == path[: len(parts)]


def test_different_embeddings_can_differ(ontology_index):
    paths = set()
    for _ in range(20):
        fake_emb = np.random.randn(512).astype(np.float32)
        path = tuple(characterize_image(fake_emb, ontology_index))
        paths.add(path)
    # With random embeddings, we should get at least a few different paths
    assert len(paths) > 1, "All random embeddings produced identical paths"


def test_characterizations_db_roundtrip(ontology_index):
    fake_emb = np.random.randn(512).astype(np.float32)
    path = characterize_image(fake_emb, ontology_index)

    with tempfile.TemporaryDirectory() as tmp:
        db = CorpusDB(Path(tmp) / "test.db")
        db.initialize_schema()

        image_id = "test-img-1"
        db.insert_image(ImageRecord(id=image_id, source_path="/tmp/test.jpg"))

        # Store embedding so it shows as "embedded"
        db.insert_embeddings_batch([(image_id, "clip-vit-b-32", [0.0] * 512)])

        rows = []
        for depth, pole_name in enumerate(path):
            prefix = "/".join(path[: depth + 1])
            rows.append((image_id, prefix, "enum", pole_name, None))

        db.insert_characterizations_batch(rows)

        retrieved = db.fetch_characterizations(image_id)
        assert len(retrieved) == len(path)

        db.close()
