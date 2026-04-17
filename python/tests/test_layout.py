"""Tests for NeighborhoodMode layout."""

import time

import numpy as np
import pytest

from sigil_atlas.db import CorpusDB, ImageRecord
from sigil_atlas.layout import compute_layout, StripLayout


class FakeEmbeddingProvider:
    """In-memory embedding provider for testing."""

    def __init__(self, embeddings: dict[str, np.ndarray]):
        self._embeddings = embeddings

    def fetch_matrix(self, image_ids: list[str], model: str) -> np.ndarray:
        return np.stack([self._embeddings[iid] for iid in image_ids])

    def encode_text(self, text: str, model: str) -> np.ndarray:
        raise NotImplementedError

    def available_models(self) -> list[str]:
        return ["test-model"]


@pytest.fixture
def db(tmp_path):
    db = CorpusDB(tmp_path / "test.db")
    db.initialize_schema()
    return db


def _seed_images(db: CorpusDB, n: int, embeddings: dict[str, np.ndarray] | None = None):
    """Insert n images with pixel dimensions."""
    now = time.time()
    for i in range(n):
        db.insert_image(ImageRecord(
            id=f"img_{i}",
            source_path=f"/test/img_{i}.jpg",
            created_at=now,
            pixel_width=800,
            pixel_height=600,
        ))


class TestComputeLayout:
    def test_empty_input(self, db):
        provider = FakeEmbeddingProvider({})
        result = compute_layout(provider, db, [])
        assert isinstance(result, StripLayout)
        assert len(result.strips) == 0

    def test_single_image(self, db):
        _seed_images(db, 1)
        provider = FakeEmbeddingProvider({"img_0": np.random.randn(8).astype(np.float32)})
        result = compute_layout(provider, db, ["img_0"], strip_height=100.0)
        assert len(result.strips) == 1
        assert len(result.strips[0].images) == 1
        assert result.strips[0].images[0].id == "img_0"
        # Width should be 100 * (800/600) = 133.33
        assert abs(result.strips[0].images[0].width - 133.33) < 1.0

    def test_multiple_images_produce_strips(self, db):
        n = 30
        _seed_images(db, n)
        # Create clustered embeddings: two groups
        embeddings = {}
        for i in range(n):
            vec = np.random.randn(16).astype(np.float32)
            if i < 15:
                vec[0] += 5.0  # cluster A
            else:
                vec[0] -= 5.0  # cluster B
            embeddings[f"img_{i}"] = vec / np.linalg.norm(vec)

        provider = FakeEmbeddingProvider(embeddings)
        result = compute_layout(
            provider, db, list(embeddings.keys()),
            strip_height=100.0,
        )

        assert len(result.strips) > 0
        total_images = sum(len(s.images) for s in result.strips)
        assert total_images == n
        # Torus dimensions derived from content
        assert result.torus_width > 0
        assert result.torus_height > 0

    def test_invariant_gapless(self, db):
        """Verify !gapless: every strip fills exactly torus_width, no gaps between images."""
        n = 50
        _seed_images(db, n)
        embeddings = {f"img_{i}": np.random.randn(16).astype(np.float32) for i in range(n)}
        provider = FakeEmbeddingProvider(embeddings)
        result = compute_layout(provider, db, list(embeddings.keys()), strip_height=100.0)

        assert result.torus_width > 0
        assert result.torus_height > 0

        for i, strip in enumerate(result.strips):
            # Strip width must equal torus_width
            strip_width = sum(img.width for img in strip.images)
            assert abs(strip_width - result.torus_width) < 1.0, (
                f"Strip {i}: width {strip_width:.1f} != torus_width {result.torus_width:.1f}"
            )

            # No gaps between adjacent images
            for j in range(len(strip.images) - 1):
                end = strip.images[j].x + strip.images[j].width
                start = strip.images[j + 1].x
                gap = abs(start - end)
                assert gap < 0.01, f"Strip {i}, images {j}-{j+1}: gap={gap:.4f}"

        # Strips tile vertically with no gaps
        expected_height = sum(s.height for s in result.strips)
        assert abs(result.torus_height - expected_height) < 0.01, (
            f"torus_height {result.torus_height:.1f} != sum of strip heights {expected_height:.1f}"
        )
        # Strips stack without vertical gaps
        for i in range(len(result.strips) - 1):
            strip_end = result.strips[i].y + result.strips[i].height
            next_start = result.strips[i + 1].y
            assert abs(strip_end - next_start) < 0.01, (
                f"Vertical gap between strips {i} and {i+1}: {abs(strip_end - next_start):.4f}"
            )

    def test_invariant_arrangement_never_changes_slice(self, db):
        """Verify !arrangement-only: layout doesn't change which images are present."""
        n = 20
        _seed_images(db, n)
        embeddings = {f"img_{i}": np.random.randn(8).astype(np.float32) for i in range(n)}
        provider = FakeEmbeddingProvider(embeddings)
        ids = list(embeddings.keys())

        layout_a = compute_layout(provider, db, ids)
        layout_b = compute_layout(provider, db, ids)

        ids_a = {img.id for s in layout_a.strips for img in s.images}
        ids_b = {img.id for s in layout_b.strips for img in s.images}

        # Layout is deterministic and preserves all images
        assert ids_a == ids_b == set(ids)

    def test_invariant_local_neighborhoods(self, db):
        """Verify !local-neighborhoods: similar images cluster locally as patches.

        Since strip packing replaces UMAP x-coordinates with sequential packing,
        we verify locality by checking that same-cluster images share strips
        (y-bands) more than they share with the other cluster.
        """
        n = 40
        _seed_images(db, n)

        # Create two clearly separated clusters
        embeddings = {}
        for i in range(n):
            vec = np.zeros(16, dtype=np.float32)
            if i < 20:
                vec[0] = 1.0
                vec[1] = np.random.randn() * 0.1
            else:
                vec[0] = -1.0
                vec[1] = np.random.randn() * 0.1
            embeddings[f"img_{i}"] = vec / np.linalg.norm(vec)

        provider = FakeEmbeddingProvider(embeddings)
        result = compute_layout(
            provider, db, list(embeddings.keys()),
            strip_height=100.0,
                    )

        # Collect strip indices per cluster
        cluster_a_strips = set()
        cluster_b_strips = set()
        for i, strip in enumerate(result.strips):
            for img in strip.images:
                idx = int(img.id.split("_")[1])
                if idx < 20:
                    cluster_a_strips.add(i)
                else:
                    cluster_b_strips.add(i)

        # Clusters should not perfectly overlap in strips
        # At minimum, each cluster should have some strips the other doesn't
        only_a = cluster_a_strips - cluster_b_strips
        only_b = cluster_b_strips - cluster_a_strips
        exclusive = len(only_a) + len(only_b)
        total = len(cluster_a_strips | cluster_b_strips)

        # At least some strip separation between clusters
        assert exclusive > 0 or total <= 2, (
            f"No strip separation between clusters: A={cluster_a_strips}, B={cluster_b_strips}"
        )

    def test_invariant_bounded_height_variation(self, db):
        """Per-strip height should stay within a few percent of nominal strip_height."""
        n = 200
        _seed_images(db, n)
        embeddings = {f"img_{i}": np.random.randn(16).astype(np.float32) for i in range(n)}
        provider = FakeEmbeddingProvider(embeddings)
        result = compute_layout(provider, db, list(embeddings.keys()), strip_height=100.0)

        for i, strip in enumerate(result.strips):
            ratio = strip.height / result.strip_height
            assert 0.90 < ratio < 1.10, (
                f"Strip {i}: height {strip.height:.1f} deviates >10% from nominal {result.strip_height:.1f}"
            )

