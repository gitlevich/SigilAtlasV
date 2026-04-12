"""Neighborhood building — FCA-style concept lattice from image characterizations.

Builds ImageNeighborhoodSigils by iteratively dropping invariants from
ImageSigils, deduplicating via frozenset keys. Matches the spec's
NeighborhoodBuildingAlgorithm exactly.

Performance: bitmap-based set intersections, with three backends:
- GPU (MPS/CUDA): batch AND + popcount via PyTorch int64 tensors
- CPU parallel: multiprocessing across cores (numpy uint64 bitmaps)
- CPU serial: fallback for small workloads
"""

import logging
import multiprocessing as mp
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class ImageSigil:
    """An image wrapped with its invariant labels."""

    image_id: str
    invariants: frozenset[str]


@dataclass
class ImageNeighborhoodSigil:
    """A neighborhood in the concept lattice.

    Contains images (or sub-neighborhoods) that share all invariants
    in its constraint set. Built by dropping invariants from tighter
    neighborhoods.
    """

    invariants: frozenset[str]
    member_ids: frozenset[str]
    children: list["ImageNeighborhoodSigil"] = field(default_factory=list)
    parents: list["ImageNeighborhoodSigil"] = field(default_factory=list)

    @property
    def scale(self) -> int:
        return len(self.invariants)

    @property
    def is_root(self) -> bool:
        return len(self.invariants) == 0

    @property
    def member_count(self) -> int:
        return len(self.member_ids)

    def __hash__(self) -> int:
        return hash(self.invariants)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImageNeighborhoodSigil):
            return NotImplemented
        return self.invariants == other.invariants


class BitmapIndex:
    """Inverted index using bitmaps for fast set intersection.

    Each invariant label maps to a numpy uint64 array where bit i
    is set if image i has that invariant. Intersection = bitwise AND.
    For n=5000: 79 uint64s per bitmap. AND = 79 ops instead of 5000.
    For n=74000: ~1157 uint64s. Still sub-microsecond.
    """

    def __init__(self, image_sigils: list[ImageSigil]) -> None:
        self.image_ids = [s.image_id for s in image_sigils]
        self.n_images = len(self.image_ids)
        self.n_words = (self.n_images + 63) // 64
        self._id_to_idx = {img_id: i for i, img_id in enumerate(self.image_ids)}

        # Build bitmap per invariant label
        self.bitmaps: dict[str, np.ndarray] = {}
        for sigil in image_sigils:
            idx = self._id_to_idx[sigil.image_id]
            word = idx // 64
            bit = np.uint64(1) << np.uint64(idx % 64)
            for label in sigil.invariants:
                if label not in self.bitmaps:
                    self.bitmaps[label] = np.zeros(self.n_words, dtype=np.uint64)
                self.bitmaps[label][word] |= bit

        # All-ones bitmap for empty invariant sets
        self._all = np.full(self.n_words, np.uint64(0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
        # Mask out unused bits in the last word
        remainder = self.n_images % 64
        if remainder > 0:
            self._all[self.n_words - 1] = np.uint64((1 << remainder) - 1)

    def intersect(self, invariants: frozenset[str]) -> np.ndarray:
        """AND all bitmaps for the given invariants. Returns bitmap."""
        if not invariants:
            return self._all.copy()
        it = iter(invariants)
        first = next(it)
        bm = self.bitmaps.get(first)
        if bm is None:
            return np.zeros(self.n_words, dtype=np.uint64)
        result = bm.copy()
        for label in it:
            bm = self.bitmaps.get(label)
            if bm is None:
                return np.zeros(self.n_words, dtype=np.uint64)
            np.bitwise_and(result, bm, out=result)
        return result

    def popcount(self, bitmap: np.ndarray) -> int:
        """Count set bits in bitmap."""
        # Unpack to bytes, use lookup or sum of bit counts
        return int(np.unpackbits(bitmap.view(np.uint8)).sum())

    def bitmap_to_ids(self, bitmap: np.ndarray) -> frozenset[str]:
        """Convert bitmap back to a frozenset of image IDs."""
        ids = []
        for word_idx in range(self.n_words):
            word = int(bitmap[word_idx])
            if word == 0:
                continue
            base = word_idx * 64
            while word:
                bit_pos = (word & -word).bit_length() - 1
                img_idx = base + bit_pos
                if img_idx < self.n_images:
                    ids.append(self.image_ids[img_idx])
                word &= word - 1  # clear lowest set bit
        return frozenset(ids)


class GpuBitmapEngine:
    """Batch bitmap intersection on GPU via PyTorch.

    Stores all invariant bitmaps as a (n_labels, n_words) int64 tensor on GPU.
    Given a batch of candidate invariant sets, computes all their intersections
    and popcounts in one GPU kernel launch.
    """

    def __init__(self, bm_index: BitmapIndex) -> None:
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Map label → row index
        self.label_to_row: dict[str, int] = {}
        labels = sorted(bm_index.bitmaps.keys())
        for i, label in enumerate(labels):
            self.label_to_row[label] = i

        # Stack all bitmaps into a (n_labels, n_words) tensor
        # Use int64 since PyTorch doesn't have uint64
        n_labels = len(labels)
        n_words = bm_index.n_words
        bitmap_matrix = np.zeros((n_labels, n_words), dtype=np.int64)
        for label in labels:
            row = self.label_to_row[label]
            bitmap_matrix[row] = bm_index.bitmaps[label].view(np.int64)

        self.bitmap_matrix = torch.tensor(bitmap_matrix, device=self.device)
        self.n_words = n_words
        self.bm_index = bm_index

        logger.info(
            "GPU bitmap engine ready on %s: %d labels x %d words",
            self.device, n_labels, n_words,
        )

    def batch_intersect_and_popcount(
        self,
        candidates: list[tuple[frozenset[str], list[int]]],
    ) -> list[tuple[frozenset[str], int]]:
        """Compute intersections and popcounts for a batch of candidates.

        Each candidate is (invariant_set, list_of_row_indices).
        Returns (invariant_set, popcount) pairs.

        Groups candidates by number of invariants (same-size AND reductions)
        for efficient batched GPU computation.
        """
        if not candidates:
            return []

        # Group by number of invariants for uniform reduction
        by_size: dict[int, list[tuple[int, frozenset[str], list[int]]]] = {}
        for idx, (inv_set, rows) in enumerate(candidates):
            n = len(rows)
            by_size.setdefault(n, []).append((idx, inv_set, rows))

        results = [None] * len(candidates)

        for n_inv, group in by_size.items():
            batch_size = len(group)

            # Gather row indices into a (batch_size, n_inv) tensor
            row_indices = torch.tensor(
                [rows for _, _, rows in group],
                dtype=torch.long,
                device=self.device,
            )

            # Index into bitmap_matrix: (batch_size, n_inv, n_words)
            gathered = self.bitmap_matrix[row_indices]

            # AND-reduce along dim=1: (batch_size, n_words)
            # For MPS: use iterative AND since prod may overflow
            intersected = gathered[:, 0, :]
            for j in range(1, n_inv):
                intersected = intersected & gathered[:, j, :]

            # Popcount: count set bits
            # Convert to bytes, sum bits per row
            # PyTorch doesn't have native popcount, so we transfer to CPU for this
            intersected_cpu = intersected.cpu().numpy().view(np.uint8)
            # unpackbits along last axis, sum per row
            # Shape: (batch_size, n_words * 8) -> sum per row
            popcounts = np.unpackbits(intersected_cpu, axis=1).sum(axis=1)

            for i, (orig_idx, inv_set, _) in enumerate(group):
                results[orig_idx] = (inv_set, int(popcounts[i]))

        return results

    def invariants_to_rows(self, invariants: frozenset[str]) -> list[int] | None:
        """Convert invariant set to row indices. Returns None if any label missing."""
        rows = []
        for label in invariants:
            row = self.label_to_row.get(label)
            if row is None:
                return None
            rows.append(row)
        return rows


def _process_level_chunk(
    chunk: list[frozenset[str]],
    bitmap_data: dict[str, np.ndarray],
    n_words: int,
    n_images: int,
    all_bitmap: np.ndarray,
    min_members: int,
    existing_keys: set[frozenset[str]],
) -> list[tuple[frozenset[str], np.ndarray, int]]:
    """Process a chunk of parent keys: drop each invariant, compute membership.

    Runs in a worker process. Returns list of (invariants, bitmap, popcount)
    for new neighborhoods that meet min_members.
    """
    results = []
    seen_in_chunk: set[frozenset[str]] = set()

    for parent_key in chunk:
        if len(parent_key) <= 1:
            continue
        for label in parent_key:
            child_key = parent_key - {label}
            if not child_key:
                continue
            if child_key in existing_keys or child_key in seen_in_chunk:
                continue

            # Intersect bitmaps
            it = iter(child_key)
            first = next(it)
            bm = bitmap_data.get(first)
            if bm is None:
                continue
            result = bm.copy()
            for lbl in it:
                bm2 = bitmap_data.get(lbl)
                if bm2 is None:
                    result = np.zeros(n_words, dtype=np.uint64)
                    break
                np.bitwise_and(result, bm2, out=result)

            count = int(np.unpackbits(result.view(np.uint8)).sum())
            if count >= min_members:
                results.append((child_key, result, count))
                seen_in_chunk.add(child_key)

    return results


def build_inverted_index(
    image_sigils: list[ImageSigil],
) -> dict[str, set[str]]:
    """Build {invariant_label: set of image_ids} for fast membership lookup."""
    index: dict[str, set[str]] = {}
    for sigil in image_sigils:
        for label in sigil.invariants:
            index.setdefault(label, set()).add(sigil.image_id)
    return index


def build_lattice(
    image_sigils: list[ImageSigil],
    min_members: int = 2,
    n_workers: int | None = None,
) -> dict[frozenset[str], ImageNeighborhoodSigil]:
    """Build the concept lattice from image sigils using bitmap intersections.

    Algorithm (per spec):
    1. For each ImageSigil with invariant set S, for each invariant i in S:
       create candidate neighborhood with invariants S - {i},
       members = all images matching S - {i}.
    2. Dedup via frozenset key.
    3. Recurse: for each neighborhood, drop one more invariant.
    4. Stop when invariant set has 1 element.

    Uses bitmap-based set intersection and parallel processing per level.
    Returns dict keyed by frozenset of invariants.
    """
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 16)

    bm_index = BitmapIndex(image_sigils)
    all_image_ids = {s.image_id for s in image_sigils}

    neighborhoods: dict[frozenset[str], ImageNeighborhoodSigil] = {}
    # Cache bitmaps alongside neighborhoods for fast child computation
    bitmaps: dict[frozenset[str], np.ndarray] = {}

    # Seed: each image's full invariant set
    current_level: set[frozenset[str]] = set()
    for sigil in image_sigils:
        if len(sigil.invariants) < 1:
            continue
        key = sigil.invariants
        if key not in neighborhoods:
            bm = bm_index.intersect(key)
            member_ids = bm_index.bitmap_to_ids(bm)
            neighborhoods[key] = ImageNeighborhoodSigil(
                invariants=key,
                member_ids=member_ids,
            )
            bitmaps[key] = bm
            current_level.add(key)

    logger.info("Seeded %d neighborhoods from %d images", len(current_level), len(image_sigils))

    # Choose processing strategy
    use_gpu = (
        len(current_level) > 100
        and (torch.cuda.is_available() or torch.backends.mps.is_available())
    )
    gpu_engine = GpuBitmapEngine(bm_index) if use_gpu else None
    use_parallel = not use_gpu and len(current_level) > 500 and n_workers > 1

    if use_gpu:
        logger.info("Using GPU engine for level processing")
    elif use_parallel:
        logger.info("Using %d CPU workers for level processing", n_workers)

    level = 0
    while current_level:
        level += 1
        parent_keys = [k for k in current_level if len(k) > 1]

        if not parent_keys:
            break

        if use_gpu and gpu_engine is not None:
            new_entries = _process_level_gpu(
                parent_keys, gpu_engine, bm_index, min_members,
                set(neighborhoods.keys()),
            )
        elif use_parallel and len(parent_keys) > 100:
            new_entries = _process_level_parallel(
                parent_keys, bm_index, min_members,
                set(neighborhoods.keys()), n_workers,
            )
        else:
            new_entries = _process_level_serial(
                parent_keys, bm_index, min_members,
                set(neighborhoods.keys()),
            )

        next_level: set[frozenset[str]] = set()
        for child_key, bm, count in new_entries:
            if child_key not in neighborhoods:
                member_ids = bm_index.bitmap_to_ids(bm)
                neighborhoods[child_key] = ImageNeighborhoodSigil(
                    invariants=child_key,
                    member_ids=member_ids,
                )
                bitmaps[child_key] = bm
                next_level.add(child_key)

        # Link parent/child relationships for this level
        for parent_key in parent_keys:
            if parent_key not in neighborhoods:
                continue
            for label in parent_key:
                child_key = parent_key - {label}
                if child_key and child_key in neighborhoods:
                    tighter = neighborhoods[parent_key]
                    looser = neighborhoods[child_key]
                    if tighter not in looser.children:
                        looser.children.append(tighter)
                    if looser not in tighter.parents:
                        tighter.parents.append(looser)

        logger.info(
            "Level %d: %d new neighborhoods (total %d)",
            level, len(next_level), len(neighborhoods),
        )
        current_level = next_level

    # Prune seed neighborhoods that didn't meet min_members
    to_remove = [
        key for key, nbr in neighborhoods.items()
        if len(nbr.member_ids) < min_members and key != frozenset()
    ]
    for key in to_remove:
        nbr = neighborhoods[key]
        for child in nbr.children:
            if nbr in child.parents:
                child.parents.remove(nbr)
        for parent in nbr.parents:
            if nbr in parent.children:
                parent.children.remove(nbr)
        del neighborhoods[key]
        bitmaps.pop(key, None)

    # Add root neighborhood
    root_key = frozenset[str]()
    root = ImageNeighborhoodSigil(
        invariants=root_key,
        member_ids=frozenset(all_image_ids),
    )
    for key, nbr in neighborhoods.items():
        if len(key) == 1 and key != root_key:
            root.children.append(nbr)
            nbr.parents.append(root)
    neighborhoods[root_key] = root

    logger.info("Lattice complete: %d neighborhoods", len(neighborhoods))
    return neighborhoods


def _process_level_serial(
    parent_keys: list[frozenset[str]],
    bm_index: BitmapIndex,
    min_members: int,
    existing_keys: set[frozenset[str]],
) -> list[tuple[frozenset[str], np.ndarray, int]]:
    """Process one level of the lattice serially."""
    results = []
    seen: set[frozenset[str]] = set()

    for parent_key in parent_keys:
        for label in parent_key:
            child_key = parent_key - {label}
            if not child_key or child_key in existing_keys or child_key in seen:
                continue

            bm = bm_index.intersect(child_key)
            count = bm_index.popcount(bm)
            if count >= min_members:
                results.append((child_key, bm, count))
                seen.add(child_key)

    return results


def _process_level_gpu(
    parent_keys: list[frozenset[str]],
    gpu_engine: GpuBitmapEngine,
    bm_index: BitmapIndex,
    min_members: int,
    existing_keys: set[frozenset[str]],
) -> list[tuple[frozenset[str], np.ndarray, int]]:
    """Process one level using GPU batch intersection."""
    # Generate all unique candidate child keys
    seen: set[frozenset[str]] = set()
    candidates: list[tuple[frozenset[str], list[int]]] = []

    for parent_key in parent_keys:
        if len(parent_key) <= 1:
            continue
        for label in parent_key:
            child_key = parent_key - {label}
            if not child_key or child_key in existing_keys or child_key in seen:
                continue
            rows = gpu_engine.invariants_to_rows(child_key)
            if rows is not None:
                candidates.append((child_key, rows))
                seen.add(child_key)

    if not candidates:
        return []

    # Batch process on GPU (in chunks to manage memory)
    gpu_batch_size = 10000
    results = []
    for i in range(0, len(candidates), gpu_batch_size):
        batch = candidates[i : i + gpu_batch_size]
        batch_results = gpu_engine.batch_intersect_and_popcount(batch)
        for inv_set, count in batch_results:
            if count >= min_members:
                bm = bm_index.intersect(inv_set)
                results.append((inv_set, bm, count))

    return results


def _process_level_parallel(
    parent_keys: list[frozenset[str]],
    bm_index: BitmapIndex,
    min_members: int,
    existing_keys: set[frozenset[str]],
    n_workers: int,
) -> list[tuple[frozenset[str], np.ndarray, int]]:
    """Process one level of the lattice in parallel using multiprocessing."""
    chunk_size = max(1, len(parent_keys) // n_workers)
    chunks = [
        parent_keys[i : i + chunk_size]
        for i in range(0, len(parent_keys), chunk_size)
    ]

    worker_fn = partial(
        _process_level_chunk,
        bitmap_data=bm_index.bitmaps,
        n_words=bm_index.n_words,
        n_images=bm_index.n_images,
        all_bitmap=bm_index._all,
        min_members=min_members,
        existing_keys=existing_keys,
    )

    all_results = []
    with mp.Pool(processes=n_workers) as pool:
        for chunk_results in pool.map(worker_fn, chunks):
            all_results.extend(chunk_results)

    # Dedup across chunks
    seen: set[frozenset[str]] = set()
    deduped = []
    for child_key, bm, count in all_results:
        if child_key not in seen and child_key not in existing_keys:
            deduped.append((child_key, bm, count))
            seen.add(child_key)

    return deduped


def build_lattice_from_characterizations(
    characterizations: dict[str, frozenset[str]],
    min_members: int = 2,
    n_workers: int | None = None,
) -> dict[frozenset[str], ImageNeighborhoodSigil]:
    """Convenience: build lattice from {image_id: invariant_labels} dict."""
    sigils = [
        ImageSigil(image_id=image_id, invariants=labels)
        for image_id, labels in characterizations.items()
    ]
    return build_lattice(sigils, min_members=min_members, n_workers=n_workers)
