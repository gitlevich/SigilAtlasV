"""Contrast Space — the ground state from which sigils are excited.

The transaction matrix holds all potential. Sigils don't exist until
observed. The frame excites a region of contrast space and collapses
the superposition into visible sigils.

No precomputed hierarchy. No stored neighborhoods. Just the matrix
and a function that takes a viewpoint and returns what's there.
"""

import logging
import struct

import numpy as np
import torch

from sigil_atlas.db import CorpusDB

logger = logging.getLogger(__name__)


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ContrastSpace:
    """The ground state. Holds all potential sigils as a transaction matrix.

    Each row is an image. Each column is a (contrast_dim, bin) item.
    A True means this image has this contrast at this value.
    """

    def __init__(
        self,
        image_ids: list[str],
        transactions: np.ndarray,  # (n_images, n_items) bool
        item_labels: list[str],    # name for each column
        n_pca: int,
        n_bins: int,
    ):
        self.image_ids = np.array(image_ids)
        self.transactions = transactions
        self.item_labels = item_labels
        self.n_images = len(image_ids)
        self.n_items = len(item_labels)
        self.n_pca = n_pca
        self.n_bins = n_bins

    def observe(self, constraints: dict[int, int]) -> np.ndarray:
        """Collapse: given a set of (dim -> bin) constraints, return which images match.

        Returns boolean mask over images.
        """
        mask = np.ones(self.n_images, dtype=bool)
        for dim, bin_val in constraints.items():
            col_idx = dim * self.n_bins + bin_val
            mask &= self.transactions[:, col_idx]
        return mask

    def excite(self, constraints: dict[int, int], depth: int) -> "Sigil":
        """Excite a region of contrast space into a sigil.

        The sigil has children formed by adding one more constraint
        (trying each unconstrained dim and each bin value).
        Recurses up to `depth` levels.
        """
        member_mask = self.observe(constraints)
        member_count = int(member_mask.sum())

        if member_count == 0:
            return None

        children = []
        if depth > 0 and member_count > 1:
            # Which dims are not yet constrained?
            constrained_dims = set(constraints.keys())
            free_dims = [d for d in range(self.n_pca) if d not in constrained_dims]

            for dim in free_dims:
                # For this dim, which bins actually have members?
                for b in range(self.n_bins):
                    child_constraints = {**constraints, dim: b}
                    col_idx = dim * self.n_bins + b
                    # Quick check: how many current members have this item?
                    child_mask = member_mask & self.transactions[:, col_idx]
                    child_count = int(child_mask.sum())

                    if child_count >= 2 and child_count < member_count:
                        # This is a real subdivision — recurse
                        child = self.excite(child_constraints, depth - 1)
                        if child is not None:
                            children.append(child)

        member_ids = self.image_ids[member_mask].tolist()

        return Sigil(
            constraints=constraints,
            member_ids=member_ids,
            children=children,
        )


class Sigil:
    """A collapsed region of contrast space. Exists because someone looked."""

    def __init__(
        self,
        constraints: dict[int, int],
        member_ids: list[str],
        children: list["Sigil"],
    ):
        self.constraints = constraints
        self.member_ids = member_ids
        self.children = children
        self.scale = len(constraints)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def member_count(self) -> int:
        return len(self.member_ids)


def build_contrast_space(
    db: CorpusDB,
    model: str = "dinov2-vitb14",
    n_pca: int = 24,
    n_bins: int = 4,
    progress_reporter=None,
) -> ContrastSpace:
    """Prepare contrast space from embeddings. No hierarchy, just the matrix."""

    logger.info("Loading embeddings for %s", model)
    rows = db._conn.execute(
        "SELECT image_id, vector FROM embeddings WHERE model_identifier = ?",
        (model,),
    ).fetchall()

    image_ids = []
    vectors = []
    for row in rows:
        image_ids.append(row[0])
        blob = row[1]
        count = len(blob) // 4
        vectors.append(struct.unpack(f"<{count}f", blob))

    matrix = np.array(vectors, dtype=np.float32)
    n_images = len(image_ids)
    logger.info("Loaded %d embeddings of dim %d", n_images, matrix.shape[1])

    # PCA
    device = _select_device()
    t_matrix = torch.tensor(matrix, device=device)
    mean = t_matrix.mean(dim=0)
    centered = t_matrix - mean
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    projections = (centered @ Vh[:n_pca].T).cpu().numpy()
    explained = ((S[:n_pca] ** 2) / (S ** 2).sum()).cpu().numpy()
    logger.info("PCA: %d components explain %.1f%%", n_pca, explained.sum() * 100)
    del t_matrix, centered, U, S, Vh

    if progress_reporter:
        progress_reporter.emit_event("contrast_space_pca_done", n_pca=n_pca)

    # Per image: filter low amplitude, keep top 40%
    min_amplitude = 0.01
    keep_ratio = 0.4
    abs_proj = np.abs(projections)
    above_min = abs_proj > min_amplitude
    counts_above = above_min.sum(axis=1)
    n_keep = np.maximum(1, (counts_above * keep_ratio).astype(int))

    # Quantize
    bin_edges = []
    for j in range(n_pca):
        col = projections[:, j]
        edges = np.linspace(col.min() - 1e-8, col.max() + 1e-8, n_bins + 1)
        bin_edges.append(edges)

    bin_assignments = np.zeros((n_images, n_pca), dtype=np.int32)
    for j in range(n_pca):
        bin_assignments[:, j] = np.digitize(projections[:, j], bin_edges[j][1:-1])

    # Build transaction matrix
    n_items = n_pca * n_bins
    item_labels = [f"d{d}_b{b}" for d in range(n_pca) for b in range(n_bins)]

    transactions = np.zeros((n_images, n_items), dtype=bool)
    for i in range(n_images):
        top_k = int(n_keep[i])
        top_dims = np.argsort(abs_proj[i])[-top_k:]
        for d in top_dims:
            b = bin_assignments[i, d]
            transactions[i, d * n_bins + b] = True

    logger.info(
        "Contrast space ready: %d images, %d items, %.0f contrasts/image avg",
        n_images, n_items, transactions.sum(axis=1).mean(),
    )

    if progress_reporter:
        progress_reporter.emit_event("contrast_space_ready", n_images=n_images, n_items=n_items)

    return ContrastSpace(image_ids, transactions, item_labels, n_pca, n_bins)
