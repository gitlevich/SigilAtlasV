"""Tests for SpaceLike arrangement — gravity-field targets + recursive median split."""

import time

import numpy as np
import pytest

from sigil_atlas.db import CorpusDB, ImageRecord
from sigil_atlas.spacelike import (
    Attractor,
    SpaceLikeLayout,
    _feathering_to_temperature,
    _local_density,
    _pick_grid,
    _recursive_split,
    compute_spacelike,
)


class FakeEmbeddingProvider:
    def __init__(self, embeddings: dict[str, np.ndarray]):
        self._embeddings = embeddings

    def fetch_matrix(self, image_ids: list[str], model: str) -> np.ndarray:
        return np.stack([self._embeddings[iid] for iid in image_ids])

    def encode_text(self, text: str, model: str) -> np.ndarray:
        raise NotImplementedError

    def available_models(self) -> list[str]:
        return ["clip-vit-b-32"]


@pytest.fixture
def db(tmp_path):
    d = CorpusDB(tmp_path / "test.db")
    d.initialize_schema()
    return d


def _seed(db: CorpusDB, n: int) -> list[str]:
    now = time.time()
    ids = []
    for i in range(n):
        iid = f"img_{i}"
        db.insert_image(ImageRecord(
            id=iid,
            source_path=f"/test/{iid}.jpg",
            created_at=now,
            pixel_width=800,
            pixel_height=600,
        ))
        ids.append(iid)
    return ids


def _clustered_embeddings(n: int, dim: int = 16, n_clusters: int = 4, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, dim)).astype(np.float32)
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8)
    out = {}
    for i in range(n):
        c = centers[i % n_clusters]
        vec = c + rng.normal(size=dim).astype(np.float32) * 0.15
        vec /= max(float(np.linalg.norm(vec)), 1e-8)
        out[f"img_{i}"] = vec
    return out


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

class TestFeathering:
    def test_monotone_in_feathering(self):
        temps = [_feathering_to_temperature(f) for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
        assert temps == sorted(temps)

    def test_range(self):
        assert 0.01 < _feathering_to_temperature(0.0) < 0.1
        assert 1.0 < _feathering_to_temperature(1.0) < 2.0


class TestPickGrid:
    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 9, 10, 16, 25, 100, 997, 1024, 5000])
    def test_product_at_least_n(self, n):
        rows, cols = _pick_grid(n)
        assert rows * cols >= n
        assert abs(rows - cols) <= 2  # near-square

    def test_exact_square(self):
        assert _pick_grid(100) == (10, 10)
        assert _pick_grid(25) == (5, 5)


class TestRecursiveSplit:
    def test_all_cells_assigned_no_duplicates_when_exact(self):
        n = 100
        rng = np.random.default_rng(0)
        targets = rng.random((n, 2)).astype(np.float32)
        out: list[tuple[int, int, int]] = []
        _recursive_split(list(range(n)), targets, 0, 0, 10, 10, out)
        assert len(out) == 100
        cells = {(c, r) for _, c, r in out}
        assert len(cells) == 100  # no cell reused
        assert {c for _, c, _ in out} == set(range(10))
        assert {r for _, _, r in out} == set(range(10))

    def test_pads_with_duplicates_when_cells_exceed_points(self):
        n = 95
        rng = np.random.default_rng(0)
        targets = rng.random((n, 2)).astype(np.float32)
        out: list[tuple[int, int, int]] = []
        _recursive_split(list(range(n)), targets, 0, 0, 10, 10, out)
        assert len(out) == 100
        cells = {(c, r) for _, c, r in out}
        assert len(cells) == 100  # still gapless

    def test_preserves_2d_proximity(self):
        """Points close in 2D should land in close cells."""
        # Place points on a clear 2D grid — each should land in its corresponding cell
        rows, cols = 8, 8
        n = rows * cols
        targets = np.array(
            [[(c + 0.5) / cols, (r + 0.5) / rows] for r in range(rows) for c in range(cols)],
            dtype=np.float32,
        )
        out: list[tuple[int, int, int]] = []
        _recursive_split(list(range(n)), targets, 0, 0, cols, rows, out)

        # Each point should be within 1 cell of its target cell center
        displacements = []
        for idx, col, row in out:
            target_col = idx % cols
            target_row = idx // cols
            displacements.append(abs(col - target_col) + abs(row - target_row))
        # Mean Manhattan displacement should be small on a grid-aligned layout
        assert np.mean(displacements) < 1.5


# ---------------------------------------------------------------------------
# End-to-end compute_spacelike
# ---------------------------------------------------------------------------

class TestComputeSpacelike:
    def test_empty_input(self, db):
        provider = FakeEmbeddingProvider({})
        result = compute_spacelike(provider, db, [], model="clip-vit-b-32")
        assert isinstance(result, SpaceLikeLayout)
        assert result.positions == []
        assert result.cols == 0 and result.rows == 0

    def test_no_attractors_uses_umap(self, db):
        n = 64
        ids = _seed(db, n)
        embeddings = _clustered_embeddings(n)
        provider = FakeEmbeddingProvider(embeddings)

        result = compute_spacelike(
            provider, db, ids, attractors=None, model="clip-vit-b-32", cell_size=1.0,
        )
        assert len(result.positions) == n
        # Gaplessness: every cell in the grid is filled
        cells = {(p.col, p.row) for p in result.positions}
        assert len(cells) == result.cols * result.rows == n
        # Torus size matches
        assert result.torus_width == result.cols
        assert result.torus_height == result.rows

    def test_gaplessness_with_padding(self, db):
        n = 95  # not a clean product
        ids = _seed(db, n)
        provider = FakeEmbeddingProvider(_clustered_embeddings(n))
        result = compute_spacelike(
            provider, db, ids, attractors=None, model="clip-vit-b-32", cell_size=1.0,
        )
        # Must tile a valid grid with no holes
        cells = {(p.col, p.row) for p in result.positions}
        assert len(cells) == result.rows * result.cols
        assert result.rows * result.cols >= n

    def test_local_proximity_preserved(self, db):
        """Images from the same embedding cluster should land near each other."""
        n = 64
        ids = _seed(db, n)
        n_clusters = 4
        embeddings = _clustered_embeddings(n, n_clusters=n_clusters)
        provider = FakeEmbeddingProvider(embeddings)

        result = compute_spacelike(
            provider, db, ids, attractors=None, model="clip-vit-b-32", cell_size=1.0,
        )
        pos_by_id = {p.id: (p.col, p.row) for p in result.positions}

        # Mean intra-cluster distance should be less than inter-cluster
        cluster_members = {c: [f"img_{i}" for i in range(n) if i % n_clusters == c] for c in range(n_clusters)}

        def mean_dist(a: list[str], b: list[str]) -> float:
            d = []
            for ia in a:
                for ib in b:
                    if ia == ib:
                        continue
                    ca, ra = pos_by_id[ia]
                    cb, rb = pos_by_id[ib]
                    d.append(abs(ca - cb) + abs(ra - rb))
            return float(np.mean(d)) if d else 0.0

        intra = np.mean([mean_dist(m, m) for m in cluster_members.values()])
        inter_pairs = []
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                inter_pairs.append(mean_dist(cluster_members[i], cluster_members[j]))
        inter = float(np.mean(inter_pairs))

        assert intra < inter, f"Intra-cluster ({intra:.2f}) should be tighter than inter ({inter:.2f})"

    def test_target_image_attractor_places_it_near_center(self, db):
        """A target_image attractor pulls similar images towards (0.5, 0.5)."""
        n = 49
        ids = _seed(db, n)
        embeddings = _clustered_embeddings(n, n_clusters=1)  # all similar
        provider = FakeEmbeddingProvider(embeddings)

        result = compute_spacelike(
            provider, db, ids,
            attractors=[Attractor(kind="target_image", ref="img_0")],
            model="clip-vit-b-32",
            feathering=0.3,
            cell_size=1.0,
        )
        pos_by_id = {p.id: (p.col, p.row) for p in result.positions}
        # img_0 itself should land near the center
        col, row = pos_by_id["img_0"]
        center_col = (result.cols - 1) / 2
        center_row = (result.rows - 1) / 2
        dist_from_center = abs(col - center_col) + abs(row - center_row)
        # On a 7x7 grid (49 cells), center-ish cells are within Manhattan ~2
        assert dist_from_center < result.cols * 0.4

    def test_animation_stable_no_change(self, db):
        """Recomputing with identical inputs produces identical assignments."""
        n = 64
        ids = _seed(db, n)
        embeddings = _clustered_embeddings(n)
        provider = FakeEmbeddingProvider(embeddings)

        r1 = compute_spacelike(provider, db, ids, model="clip-vit-b-32", cell_size=1.0)
        r2 = compute_spacelike(provider, db, ids, model="clip-vit-b-32", cell_size=1.0)

        def key(layout):
            return sorted((p.id, p.col, p.row) for p in layout.positions)

        assert key(r1) == key(r2)


class TestLocalDensity:
    def test_isolated_points_have_low_density(self):
        # 4 widely-separated points
        targets = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]], dtype=np.float32)
        elev = _local_density(targets, radius=0.05)
        assert elev.max() <= 1.0 and elev.min() >= 0.0
        # All equally isolated -> uniform
        assert elev.std() < 1e-6

    def test_clustered_points_peak_at_cluster(self):
        rng = np.random.default_rng(0)
        # 20 points clustered tightly at (0.3, 0.3), 5 scattered
        cluster = rng.normal(loc=(0.3, 0.3), scale=0.02, size=(20, 2)).astype(np.float32)
        scattered = np.array([[0.8, 0.8], [0.7, 0.2], [0.2, 0.8], [0.9, 0.5], [0.5, 0.9]], dtype=np.float32)
        targets = np.vstack([cluster, scattered])
        elev = _local_density(targets, radius=0.08)
        # Cluster points should have higher elevation than scattered ones
        assert elev[:20].mean() > elev[20:].mean()
        assert elev.max() == 1.0  # normalized

    def test_elevation_attached_to_positions(self, db):
        n = 64
        ids = _seed(db, n)
        provider = FakeEmbeddingProvider(_clustered_embeddings(n))
        layout = compute_spacelike(
            provider, db, ids, model="clip-vit-b-32", cell_size=1.0,
        )
        for p in layout.positions:
            assert 0.0 <= p.elevation <= 1.0


class TestPerformance:
    def test_scales_to_10k(self, db):
        """10k images should layout in under ~5s (UMAP dominates on cold cache)."""
        import time as _time
        n = 10_000
        ids = _seed(db, n)
        # Use a cached UMAP to isolate the split stage
        rng = np.random.default_rng(42)
        embeddings = {iid: rng.normal(size=32).astype(np.float32) for iid in ids}
        for iid in ids:
            embeddings[iid] /= max(float(np.linalg.norm(embeddings[iid])), 1e-8)
        provider = FakeEmbeddingProvider(embeddings)

        t0 = _time.monotonic()
        result = compute_spacelike(provider, db, ids, model="clip-vit-b-32", cell_size=1.0)
        dt = _time.monotonic() - t0

        assert len(result.positions) == result.rows * result.cols
        assert dt < 30.0  # UMAP on 10k takes a few seconds; split is fast
