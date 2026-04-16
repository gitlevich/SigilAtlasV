"""Sigil Space — project taxonomy ontologies into CLIP embedding space
and visualize interference patterns as an interactive 3D scene.

Each taxonomy is a camera angle. Crossing orthogonal taxonomies reveals
standing waves: antinodes where both reinforce, nodes where neither resolves.
The 3D scene encodes many dimensions of contrast via position, color,
size, and opacity.

Usage:
    PYTHONPATH=python python tools/sigil_space.py \
        --workspace workspace \
        --taxonomy-a semantic \
        --taxonomy-b visual \
        --taxonomy-c cinematic \
        --max-points 15000 \
        --output sigil_space.html
"""

import argparse
import colorsys
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from sigil_atlas.embedding_provider import SqliteEmbeddingProvider
from sigil_atlas.model_registry import get_adapter
from sigil_atlas.ontology import OntologyNode, _parse_node
from sigil_atlas.workspace import Workspace

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required: pip install pyyaml")

logger = logging.getLogger(__name__)

TAXONOMY_DIR = Path(__file__).resolve().parent.parent / "python" / "sigil_atlas"
MODEL_ID = "clip-vit-b-32"


# ── Data classes ────────────────────────────────────────────────


@dataclass
class TermInfo:
    taxonomy: str
    name: str
    prompt: str
    depth: int


@dataclass
class SigilSpace:
    """All computed data for a single projection."""

    image_ids: list[str]
    term_infos: list[TermInfo]
    term_vectors: np.ndarray          # (N_terms, dim)
    excitation: np.ndarray            # (N_images, N_terms)
    positions: np.ndarray             # (N_sampled, 3)
    interference_strength: np.ndarray # (N_sampled,)
    sample_indices: np.ndarray        # indices into full arrays
    taxonomy_a: str
    taxonomy_b: str
    taxonomy_c: str | None


# ── Stage 1: Load and encode taxonomies ─────────────────────────


def discover_taxonomy_files() -> list[Path]:
    """Find all taxonomy_*.yaml files in the package directory."""
    files = sorted(TAXONOMY_DIR.glob("taxonomy_*.yaml"))
    logger.info("Discovered %d taxonomy files", len(files))
    return files


def load_all_taxonomies(paths: list[Path] | None = None) -> dict[str, OntologyNode]:
    """Load all taxonomy YAML files. Returns {root_name: OntologyNode}."""
    paths = paths or discover_taxonomy_files()
    roots = {}
    for path in paths:
        with open(path) as f:
            data = yaml.safe_load(f)
        root_key = next(iter(data))
        roots[root_key] = _parse_node(root_key, data[root_key])
        logger.info("  %s: %d nodes, depth %d", root_key,
                     len(roots[root_key].walk()), roots[root_key].depth())
    return roots


def collect_term_infos(taxonomies: dict[str, OntologyNode]) -> list[TermInfo]:
    """Flatten all taxonomy trees into a list of TermInfo."""
    infos = []
    for tax_name, root in taxonomies.items():
        _collect_recursive(root, tax_name, 0, infos)
    return infos


def _collect_recursive(
    node: OntologyNode, taxonomy: str, depth: int, out: list[TermInfo],
) -> None:
    out.append(TermInfo(taxonomy=taxonomy, name=node.name,
                        prompt=node.prompt, depth=depth))
    for child in node.children:
        _collect_recursive(child, taxonomy, depth + 1, out)


def encode_terms(term_infos: list[TermInfo]) -> np.ndarray:
    """Encode all term prompts via CLIP. Returns (N_terms, dim) matrix."""
    adapter = get_adapter(MODEL_ID)
    vectors = []
    t0 = time.monotonic()
    for i, ti in enumerate(term_infos):
        vec = adapter.encode_text(ti.prompt)
        vectors.append(vec)
        if (i + 1) % 200 == 0:
            logger.info("  Encoded %d / %d terms", i + 1, len(term_infos))
    matrix = np.stack(vectors)
    dt = time.monotonic() - t0
    logger.info("Encoded %d terms in %.1fs, shape %s", len(term_infos), dt, matrix.shape)
    return matrix


# ── Stage 2: Excitation matrix ──────────────────────────────────


def compute_excitation(
    image_matrix: np.ndarray,
    term_vectors: np.ndarray,
    term_infos: list[TermInfo],
) -> np.ndarray:
    """Compute excitation: (N_images, N_terms) cosine similarities, z-normalized per taxonomy."""
    t0 = time.monotonic()
    raw = image_matrix @ term_vectors.T  # (N_images, N_terms)

    # Z-normalize per taxonomy block
    taxonomy_names = list(dict.fromkeys(ti.taxonomy for ti in term_infos))
    for tax_name in taxonomy_names:
        cols = [i for i, ti in enumerate(term_infos) if ti.taxonomy == tax_name]
        block = raw[:, cols]
        mu = block.mean()
        sigma = block.std()
        if sigma > 1e-9:
            raw[:, cols] = (block - mu) / sigma

    dt = time.monotonic() - t0
    logger.info("Excitation matrix: %s in %.1fs", raw.shape, dt)
    return raw.astype(np.float32)


# ── Stage 3: Interference ───────────────────────────────────────


def compute_taxonomy_profile(
    excitation: np.ndarray,
    term_infos: list[TermInfo],
    taxonomy_name: str,
    n_components: int = 3,
) -> np.ndarray:
    """PCA-reduced excitation profile for one taxonomy. Shape (N_images, n_components)."""
    cols = [i for i, ti in enumerate(term_infos) if ti.taxonomy == taxonomy_name]
    if len(cols) < n_components:
        n_components = len(cols)
    block = excitation[:, cols]
    pca = PCA(n_components=n_components)
    return pca.fit_transform(block).astype(np.float32)


def compute_interference_strength(
    profile_a: np.ndarray,
    profile_b: np.ndarray,
) -> np.ndarray:
    """Scalar interference: product of excitation norms. High = antinode."""
    norm_a = np.linalg.norm(profile_a, axis=1)
    norm_b = np.linalg.norm(profile_b, axis=1)
    return (norm_a * norm_b).astype(np.float32)


# ── Stage 4: 3D projection ─────────────────────────────────────


def compute_taxonomy_axes(
    term_vectors: np.ndarray,
    term_infos: list[TermInfo],
    taxonomy_a: str,
    taxonomy_b: str,
    taxonomy_c: str | None,
) -> np.ndarray:
    """Compute 3 orthogonal axes from taxonomy term structure. Returns (3, dim)."""
    dim = term_vectors.shape[1]

    # Dictionary mode: A and B are the same, use top-3 PCA of all terms
    if taxonomy_a == taxonomy_b:
        pca = PCA(n_components=3)
        pca.fit(term_vectors)
        axes = pca.components_.astype(np.float32)
        logger.info("PCA axes (dictionary mode): variance explained = %.1f%%, %.1f%%, %.1f%%",
                    pca.explained_variance_ratio_[0] * 100,
                    pca.explained_variance_ratio_[1] * 100,
                    pca.explained_variance_ratio_[2] * 100)
        return axes

    def principal_direction(tax_name: str) -> np.ndarray:
        cols = [i for i, ti in enumerate(term_infos) if ti.taxonomy == tax_name]
        if len(cols) < 2:
            return term_vectors[cols[0]] if cols else np.random.randn(dim).astype(np.float32)
        block = term_vectors[cols]
        pca = PCA(n_components=1)
        pca.fit(block)
        return pca.components_[0].astype(np.float32)

    # Axis X: taxonomy A's principal direction
    axis_x = principal_direction(taxonomy_a)
    axis_x = axis_x / max(np.linalg.norm(axis_x), 1e-8)

    # Axis Y: taxonomy B's principal direction, orthogonalized against X
    axis_y = principal_direction(taxonomy_b)
    axis_y = axis_y - np.dot(axis_y, axis_x) * axis_x  # Gram-Schmidt
    axis_y = axis_y / max(np.linalg.norm(axis_y), 1e-8)

    # Axis Z: taxonomy C's direction or residual
    if taxonomy_c:
        axis_z = principal_direction(taxonomy_c)
    else:
        # Use second component of taxonomy A as residual
        cols_a = [i for i, ti in enumerate(term_infos) if ti.taxonomy == taxonomy_a]
        if len(cols_a) >= 3:
            pca = PCA(n_components=3)
            pca.fit(term_vectors[cols_a])
            axis_z = pca.components_[2].astype(np.float32)
        else:
            axis_z = np.random.randn(dim).astype(np.float32)

    # Orthogonalize Z against X and Y
    axis_z = axis_z - np.dot(axis_z, axis_x) * axis_x
    axis_z = axis_z - np.dot(axis_z, axis_y) * axis_y
    norm_z = np.linalg.norm(axis_z)
    if norm_z < 1e-8:
        axis_z = np.random.randn(dim).astype(np.float32)
        axis_z = axis_z - np.dot(axis_z, axis_x) * axis_x
        axis_z = axis_z - np.dot(axis_z, axis_y) * axis_y
        norm_z = np.linalg.norm(axis_z)
    axis_z = axis_z / max(norm_z, 1e-8)

    axes = np.stack([axis_x, axis_y, axis_z])
    logger.info("Taxonomy axes: orthogonality check x.y=%.4f, x.z=%.4f, y.z=%.4f",
                abs(np.dot(axis_x, axis_y)),
                abs(np.dot(axis_x, axis_z)),
                abs(np.dot(axis_y, axis_z)))
    return axes


def project_images(image_matrix: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """Project images onto 3 taxonomy axes. Returns (N, 3)."""
    return (image_matrix @ axes.T).astype(np.float32)


def refine_with_umap(
    image_matrix: np.ndarray,
    init_positions: np.ndarray,
) -> np.ndarray:
    """Optional UMAP-3D refinement preserving taxonomy-axis initialization."""
    import umap

    logger.info("UMAP refinement on %d points...", len(image_matrix))
    t0 = time.monotonic()
    reducer = umap.UMAP(
        n_components=3,
        metric="cosine",
        n_neighbors=15,
        min_dist=0.1,
        init=init_positions,
    )
    result = reducer.fit_transform(image_matrix).astype(np.float32)
    logger.info("UMAP done in %.1fs", time.monotonic() - t0)
    return result


def subsample(
    n_total: int,
    max_points: int,
    excitation: np.ndarray,
    term_infos: list[TermInfo],
    taxonomy_a: str,
    taxonomy_b: str,
) -> np.ndarray:
    """Stratified subsample preserving density structure. Returns indices."""
    if n_total <= max_points:
        return np.arange(n_total)

    # Assign each image to its top-claiming term in each taxonomy
    cols_a = [i for i, ti in enumerate(term_infos) if ti.taxonomy == taxonomy_a]
    cols_b = [i for i, ti in enumerate(term_infos) if ti.taxonomy == taxonomy_b]

    if not cols_a or not cols_b:
        return np.random.choice(n_total, max_points, replace=False)

    label_a = np.argmax(excitation[:, cols_a], axis=1)
    label_b = np.argmax(excitation[:, cols_b], axis=1)

    # Combine into cell labels
    n_a = len(cols_a)
    cell_label = label_a * len(cols_b) + label_b

    # Proportional sampling per cell
    unique_cells, counts = np.unique(cell_label, return_counts=True)
    fractions = counts / counts.sum()
    budgets = np.maximum(1, (fractions * max_points).astype(int))

    # Adjust to hit exact target
    total = budgets.sum()
    if total > max_points:
        excess = total - max_points
        order = np.argsort(-budgets)
        for idx in order:
            trim = min(budgets[idx] - 1, excess)
            budgets[idx] -= trim
            excess -= trim
            if excess <= 0:
                break

    selected = []
    for cell, budget in zip(unique_cells, budgets):
        mask = cell_label == cell
        indices = np.where(mask)[0]
        k = min(budget, len(indices))
        chosen = np.random.choice(indices, k, replace=False)
        selected.extend(chosen)

    result = np.array(sorted(selected))
    logger.info("Subsampled %d → %d points (%d cells)",
                n_total, len(result), len(unique_cells))
    return result


# ── Stage 5: Visual channels ───────────────────────────────────


def _percentile_normalize(arr: np.ndarray, lo_pct: float = 2, hi_pct: float = 98) -> np.ndarray:
    """Normalize using percentiles to spread the distribution, clipped to [0, 1]."""
    lo = np.percentile(arr, lo_pct)
    hi = np.percentile(arr, hi_pct)
    if hi - lo < 1e-9:
        return np.full_like(arr, 0.5)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def compute_visual_channels(
    excitation: np.ndarray,
    term_infos: list[TermInfo],
    taxonomy_a: str,
    taxonomy_b: str,
    taxonomy_c: str | None,
    interference_strength: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Compute color (RGB strings), sizes, opacities for each point.

    Two modes:
      Taxonomy mode: hue = balance A vs B, saturation = interference, lightness = specificity
      Dictionary mode (A==B): PCA of excitation → 3 independent color axes

    Returns (colors, sizes, opacities).
    """
    n = len(excitation)
    is_dictionary = taxonomy_a == taxonomy_b

    if is_dictionary:
        # ── Dictionary mode: PCA-based coloring ──
        # Top 3 PCA components of excitation → map to HSL
        # This gives colors that reflect what KIND of words describe each image
        pca = PCA(n_components=3)
        coords = pca.fit_transform(excitation).astype(np.float32)
        logger.info("Color PCA variance: %.1f%%, %.1f%%, %.1f%%",
                     pca.explained_variance_ratio_[0] * 100,
                     pca.explained_variance_ratio_[1] * 100,
                     pca.explained_variance_ratio_[2] * 100)

        hue = _percentile_normalize(coords[:, 0])
        sat_raw = _percentile_normalize(coords[:, 1])
        lit_raw = _percentile_normalize(coords[:, 2])

        sat = 0.5 + 0.5 * sat_raw   # [0.5, 1.0]
        lit = 0.20 + 0.35 * lit_raw  # [0.20, 0.55]
    else:
        # ── Taxonomy mode: hue from balance A vs B ──
        cols_a = [i for i, ti in enumerate(term_infos) if ti.taxonomy == taxonomy_a]
        cols_b = [i for i, ti in enumerate(term_infos) if ti.taxonomy == taxonomy_b]
        cols_c = ([i for i, ti in enumerate(term_infos) if ti.taxonomy == taxonomy_c]
                  if taxonomy_c else [])

        exc_a = excitation[:, cols_a].max(axis=1) if cols_a else np.zeros(n)
        exc_b = excitation[:, cols_b].max(axis=1) if cols_b else np.zeros(n)

        rank_a = np.argsort(np.argsort(exc_a)).astype(np.float32) / max(n - 1, 1)
        rank_b = np.argsort(np.argsort(exc_b)).astype(np.float32) / max(n - 1, 1)
        hue_raw = np.arctan2(rank_b - 0.5, rank_a - 0.5)
        hue = (hue_raw + np.pi) / (2 * np.pi)

        sat_norm = _percentile_normalize(interference_strength)
        sat = 0.4 + 0.6 * sat_norm

        # Specificity
        exp_exc = np.exp(excitation - excitation.max(axis=1, keepdims=True))
        probs = exp_exc / exp_exc.sum(axis=1, keepdims=True)
        probs = np.clip(probs, 1e-10, 1.0)
        entropy = -(probs * np.log(probs)).sum(axis=1)
        max_entropy = np.log(excitation.shape[1])
        specificity = 1.0 - entropy / max_entropy
        spec_norm = _percentile_normalize(specificity)
        lit = 0.15 + 0.40 * spec_norm

        if cols_c:
            exc_c = excitation[:, cols_c].max(axis=1)
            c_norm = _percentile_normalize(exc_c)
            lit = lit * (0.7 + 0.6 * c_norm)
            lit = np.clip(lit, 0.10, 0.60)

    # Convert HSL to RGB
    colors = []
    for i in range(n):
        r, g, b = colorsys.hls_to_rgb(float(hue[i]), float(lit[i]), float(sat[i]))
        colors.append(f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})")

    # ── Size: interference (antinodes large, nodes small) ──
    interf_norm = _percentile_normalize(interference_strength)
    sizes = (2.0 + 6.0 * interf_norm).astype(np.float32)

    # ── Opacity: overall max excitation (strongly claimed = solid) ──
    max_exc = excitation.max(axis=1)
    opacity_norm = _percentile_normalize(max_exc)
    opacities = (0.4 + 0.6 * opacity_norm).astype(np.float32)

    return colors, sizes, opacities


def build_hover_texts(
    image_ids: list[str],
    excitation: np.ndarray,
    term_infos: list[TermInfo],
    taxonomy_a: str,
    taxonomy_b: str,
    interference_strength: np.ndarray,
    top_k_words: int = 15,
) -> list[str]:
    """Build hover text for each point.

    In dictionary mode, shows the top-K words that cluster around each image —
    the narrative ingredients. In taxonomy mode, shows the top claiming term
    per taxonomy.
    """
    is_dictionary = taxonomy_a == taxonomy_b and taxonomy_a == "dictionary"

    if is_dictionary:
        # Find top-K words per image from excitation scores
        texts = []
        for i in range(len(image_ids)):
            scores = excitation[i]
            top_idx = np.argpartition(-scores, top_k_words)[:top_k_words]
            top_idx = top_idx[np.argsort(-scores[top_idx])]

            parts = [f"<b>{image_ids[i][:40]}</b>"]
            words_line = ", ".join(
                f"{term_infos[j].name}" for j in top_idx
            )
            parts.append(f"<i>{words_line}</i>")
            parts.append(f"interference: {interference_strength[i]:.1f}")
            texts.append("<br>".join(parts))
        return texts

    cols_a = [i for i, ti in enumerate(term_infos) if ti.taxonomy == taxonomy_a]
    cols_b = [i for i, ti in enumerate(term_infos) if ti.taxonomy == taxonomy_b]

    texts = []
    for i in range(len(image_ids)):
        parts = [f"<b>{image_ids[i][:40]}</b>"]

        if cols_a:
            best_a_idx = cols_a[int(np.argmax(excitation[i, cols_a]))]
            parts.append(f"{taxonomy_a}: {term_infos[best_a_idx].name} "
                        f"({excitation[i, best_a_idx]:.2f})")
        if cols_b:
            best_b_idx = cols_b[int(np.argmax(excitation[i, cols_b]))]
            parts.append(f"{taxonomy_b}: {term_infos[best_b_idx].name} "
                        f"({excitation[i, best_b_idx]:.2f})")

        parts.append(f"interference: {interference_strength[i]:.2f}")
        texts.append("<br>".join(parts))

    return texts


# ── Density surface ─────────────────────────────────────────────


def compute_density_grid(
    positions: np.ndarray,
    grid_size: int = 80,
    sigma: float = 1.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute smoothed 3D density field on a regular grid.

    Uses ALL positions (not subsampled) for the true density.
    Returns (X, Y, Z, density) meshgrids.
    """
    from scipy.ndimage import gaussian_filter

    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    padding = (maxs - mins) * 0.08
    mins -= padding
    maxs += padding

    edges = [np.linspace(mins[d], maxs[d], grid_size + 1) for d in range(3)]
    density, _ = np.histogramdd(positions, bins=edges)
    density = gaussian_filter(density.astype(np.float32), sigma=sigma)

    centers = [(e[:-1] + e[1:]) / 2 for e in edges]
    X, Y, Z = np.meshgrid(centers[0], centers[1], centers[2], indexing="ij")

    return X, Y, Z, density


def compute_hue_grid(
    positions: np.ndarray,
    excitation: np.ndarray,
    term_infos: list[TermInfo],
    taxonomy_a: str,
    taxonomy_b: str,
    grid_size: int = 80,
    sigma: float = 1.8,
) -> np.ndarray:
    """Compute average taxonomy hue per grid cell.

    Returns hue grid same shape as density grid, [0, 1].
    """
    from scipy.ndimage import gaussian_filter

    if taxonomy_a == taxonomy_b:
        # Dictionary mode: use first PCA component of excitation as hue
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(excitation)[:, 0]
        hue_per_image = _percentile_normalize(pc1)
    else:
        cols_a = [i for i, ti in enumerate(term_infos) if ti.taxonomy == taxonomy_a]
        cols_b = [i for i, ti in enumerate(term_infos) if ti.taxonomy == taxonomy_b]

        exc_a = excitation[:, cols_a].max(axis=1) if cols_a else np.zeros(len(positions))
        exc_b = excitation[:, cols_b].max(axis=1) if cols_b else np.zeros(len(positions))

        rank_a = np.argsort(np.argsort(exc_a)).astype(np.float32) / max(len(exc_a) - 1, 1)
        rank_b = np.argsort(np.argsort(exc_b)).astype(np.float32) / max(len(exc_b) - 1, 1)
        hue_per_image = (np.arctan2(rank_b - 0.5, rank_a - 0.5) + np.pi) / (2 * np.pi)

    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    padding = (maxs - mins) * 0.08
    mins -= padding
    maxs += padding

    edges = [np.linspace(mins[d], maxs[d], grid_size + 1) for d in range(3)]

    # Accumulate hue and count per cell
    hue_sum = np.zeros([grid_size] * 3, dtype=np.float64)
    count = np.zeros([grid_size] * 3, dtype=np.float64)

    # Digitize positions
    bin_idx = []
    for d in range(3):
        idx = np.digitize(positions[:, d], edges[d]) - 1
        idx = np.clip(idx, 0, grid_size - 1)
        bin_idx.append(idx)

    for i in range(len(positions)):
        ix, iy, iz = bin_idx[0][i], bin_idx[1][i], bin_idx[2][i]
        hue_sum[ix, iy, iz] += hue_per_image[i]
        count[ix, iy, iz] += 1

    mask = count > 0
    hue_grid = np.zeros_like(hue_sum, dtype=np.float32)
    hue_grid[mask] = (hue_sum[mask] / count[mask]).astype(np.float32)

    hue_grid = gaussian_filter(hue_grid, sigma=sigma)
    return hue_grid


# ── Stage 6: Plotly output ──────────────────────────────────────


def _add_wireframe_surfaces(
    fig,
    density: np.ndarray,
    density_norm: np.ndarray,
    hue_grid: np.ndarray | None,
    mins: np.ndarray,
    maxs: np.ndarray,
    grid_size: int,
    n_levels: int = 6,
) -> None:
    """Extract isosurfaces via marching cubes and render as wireframe meshes."""
    import plotly.graph_objects as go
    from skimage.measure import marching_cubes

    dmax = density_norm.max()
    # Density levels from outer skin to inner core
    levels = np.linspace(0.08, 0.6, n_levels)
    span = maxs - mins

    for k, level in enumerate(levels):
        if level > dmax:
            continue
        try:
            verts, faces, _, _ = marching_cubes(density_norm, level=level)
        except (ValueError, RuntimeError):
            continue

        if len(verts) == 0:
            continue

        # Map from grid coords to data coords
        for d in range(3):
            verts[:, d] = mins[d] + verts[:, d] / grid_size * span[d]

        # Color by hue grid if available, else by depth
        if hue_grid is not None:
            # Sample hue at each vertex (grid coords)
            grid_verts = np.clip(
                ((verts - mins) / span * grid_size).astype(int),
                0, grid_size - 1,
            )
            vertex_hues = hue_grid[grid_verts[:, 0], grid_verts[:, 1], grid_verts[:, 2]]
            vertex_colors = []
            for h in vertex_hues:
                r, g, b = colorsys.hls_to_rgb(float(h), 0.45, 0.85)
                vertex_colors.append(f"rgb({int(r*255)},{int(g*255)},{int(b*255)})")
        else:
            t = k / max(n_levels - 1, 1)
            r, g, b = colorsys.hls_to_rgb(0.6 - 0.4 * t, 0.35 + 0.15 * t, 0.8)
            vertex_colors = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"

        # Collect wireframe edges from triangle faces
        edge_x, edge_y, edge_z = [], [], []
        # Subsample faces for performance — keep every Nth triangle
        step = max(1, len(faces) // 3000)
        for face in faces[::step]:
            for a_idx, b_idx in [(0, 1), (1, 2), (2, 0)]:
                edge_x.extend([verts[face[a_idx], 0], verts[face[b_idx], 0], None])
                edge_y.extend([verts[face[a_idx], 1], verts[face[b_idx], 1], None])
                edge_z.extend([verts[face[a_idx], 2], verts[face[b_idx], 2], None])

        alpha = 0.15 + 0.25 * (k / max(n_levels - 1, 1))  # outer = faint, inner = brighter
        line_color = vertex_colors if isinstance(vertex_colors, str) else f"rgba(140,180,220,{alpha:.2f})"

        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode="lines",
            line=dict(color=line_color, width=1.0),
            opacity=alpha,
            showlegend=False,
            hoverinfo="skip",
            name=f"wireframe {level:.2f}",
        ))

    logger.info("Wireframe: %d density levels rendered", n_levels)


def build_figure(
    positions: np.ndarray,
    colors: list[str],
    sizes: np.ndarray,
    opacities: np.ndarray,
    hover_texts: list[str],
    taxonomy_a: str,
    taxonomy_b: str,
    taxonomy_c: str | None,
    all_positions: np.ndarray | None = None,
    all_excitation: np.ndarray | None = None,
    term_infos: list[TermInfo] | None = None,
    wireframe: bool = False,
):
    """Build Plotly 3D scatter figure with density surface."""
    import plotly.graph_objects as go

    fig = go.Figure()

    # ── Density surface (rendered first, behind scatter) ──
    if all_positions is not None:
        grid_size = 80
        sigma_val = 1.8

        X, Y, Z, density = compute_density_grid(all_positions, grid_size, sigma_val)
        dmax = density.max()
        if dmax > 1e-9:
            density_norm = density / dmax

            # Compute hue grid for taxonomy-colored surface
            hue_grid = None
            if all_excitation is not None and term_infos is not None:
                hue_grid = compute_hue_grid(
                    all_positions, all_excitation, term_infos,
                    taxonomy_a, taxonomy_b, grid_size, sigma_val,
                )

            # ── Solid isosurface (lens 1) ──
            if hue_grid is not None:
                surface_value = hue_grid.copy()
                surface_value[density_norm < 0.02] = -0.1
            else:
                surface_value = density_norm

            hue_colorscale = []
            for t in np.linspace(0, 1, 16):
                r, g, b = colorsys.hls_to_rgb(t, 0.4, 0.9)
                hue_colorscale.append([t, f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"])

            fig.add_trace(go.Isosurface(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=surface_value.flatten(),
                isomin=0.0,
                isomax=1.0,
                surface_count=25,
                colorscale=hue_colorscale,
                opacity=0.12,
                caps=dict(x_show=False, y_show=False, z_show=False),
                showscale=False,
                name="solid surface",
                legendgroup="solid",
                hoverinfo="skip",
                flatshading=True,
                visible=True,
            ))

            # ── Wireframe (lens 2) — marching cubes at multiple density levels ──
            mins = all_positions.min(axis=0) - (all_positions.max(axis=0) - all_positions.min(axis=0)) * 0.08
            maxs = all_positions.max(axis=0) + (all_positions.max(axis=0) - all_positions.min(axis=0)) * 0.08
            _add_wireframe_surfaces(
                fig, density, density_norm, hue_grid,
                mins, maxs, grid_size, n_levels=6,
            )
            # Hide wireframe traces initially (they were just added)
            # The isosurface is trace index 0, wireframe traces follow
            wireframe_start_idx = 1  # after isosurface
            for trace_idx in range(wireframe_start_idx, len(fig.data)):
                fig.data[trace_idx].visible = False
                fig.data[trace_idx].legendgroup = "wireframe"

            logger.info("Density grid: %dx%dx%d, max=%.1f",
                        grid_size, grid_size, grid_size, dmax)

    # ── Scatter points (on top of surface) ──
    rgba_colors = []
    for i in range(len(colors)):
        rgb_part = colors[i][4:-1]
        rgba_colors.append(f"rgba({rgb_part},{opacities[i]:.2f})")

    fig.add_trace(go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode="markers",
        marker=dict(
            size=sizes,
            color=rgba_colors,
            line=dict(width=0),
        ),
        text=hover_texts,
        hoverinfo="text",
        name="images",
    ))

    z_label = f"{taxonomy_c} axis" if taxonomy_c else "residual axis"

    def _axis(label: str) -> dict:
        return dict(
            title=dict(text=label,
                       font=dict(color="rgba(200,200,220,0.8)", size=13)),
            gridcolor="rgba(60,60,80,0.3)",
            zerolinecolor="rgba(80,80,120,0.4)",
            showbackground=False,
            tickfont=dict(color="rgba(180,180,200,0.6)", size=10),
        )

    # ── Build lens toggle buttons ──
    # Trace layout: [isosurface, wireframe_0..wireframe_N, scatter]
    n_traces = len(fig.data)
    scatter_idx = n_traces - 1  # scatter is always last

    # Visibility arrays for each lens mode
    def _vis(show_solid: bool, show_wire: bool, show_points: bool) -> list:
        v = []
        for i in range(n_traces):
            if i == 0:          # isosurface
                v.append(show_solid)
            elif i == scatter_idx:  # scatter
                v.append(show_points)
            else:               # wireframe traces
                v.append(show_wire)
        return v

    buttons = [
        dict(label="Solid + Points", method="update",
             args=[{"visible": _vis(True, False, True)}]),
        dict(label="Wireframe + Points", method="update",
             args=[{"visible": _vis(False, True, True)}]),
        dict(label="Wireframe Only", method="update",
             args=[{"visible": _vis(False, True, False)}]),
        dict(label="Points Only", method="update",
             args=[{"visible": _vis(False, False, True)}]),
        dict(label="Solid Only", method="update",
             args=[{"visible": _vis(True, False, False)}]),
    ]

    fig.update_layout(
        scene=dict(
            xaxis=_axis(f"{taxonomy_a} axis"),
            yaxis=_axis(f"{taxonomy_b} axis"),
            zaxis=_axis(z_label),
            bgcolor="rgb(8,8,12)",
            dragmode="turntable",
        ),
        paper_bgcolor="rgb(8,8,12)",
        font=dict(color="rgba(200,200,220,0.8)", size=12),
        title=dict(
            text=f"Sigil Space: {taxonomy_a} x {taxonomy_b}"
                 + (f" x {taxonomy_c}" if taxonomy_c else ""),
            font=dict(size=16, color="rgba(200,200,220,0.7)"),
        ),
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.5, xanchor="center",
            y=1.02, yanchor="bottom",
            bgcolor="rgba(30,30,40,0.8)",
            bordercolor="rgba(100,100,140,0.4)",
            font=dict(color="rgba(200,200,220,0.9)", size=11),
            buttons=buttons,
        )],
        showlegend=False,
        width=1400,
        height=900,
        margin=dict(l=0, r=0, t=60, b=0),
    )

    return fig


# ── Main ────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Sigil Space — taxonomy photogrammetry in CLIP space",
    )
    parser.add_argument("--workspace", type=Path, default=Path("workspace"),
                        help="Path to SigilAtlas workspace directory")
    parser.add_argument("--taxonomy-a", default="semantic",
                        help="Primary taxonomy (X axis)")
    parser.add_argument("--taxonomy-b", default="visual",
                        help="Secondary taxonomy (Y axis)")
    parser.add_argument("--taxonomy-c", default=None,
                        help="Tertiary taxonomy (Z axis, optional)")
    parser.add_argument("--max-points", type=int, default=15000,
                        help="Max points for visualization")
    parser.add_argument("--dictionary", type=Path, default=None,
                        help="Path to pre-computed dictionary .npz (from precompute_dictionary.py)")
    parser.add_argument("--ontology", type=Path, default=None,
                        help="Path to a custom ontology YAML file (a focused lens)")
    parser.add_argument("--wireframe", action="store_true",
                        help="Render surface as wireframe instead of solid")
    parser.add_argument("--refine-umap", action="store_true",
                        help="Apply UMAP refinement (slower, more local detail)")
    parser.add_argument("--output", type=Path, default=Path("sigil_space.html"),
                        help="Output HTML file path")
    parser.add_argument("--model", default=MODEL_ID,
                        help="Embedding model to use")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    t_start = time.monotonic()

    # ── Initialize workspace ──
    logger.info("Opening workspace: %s", args.workspace)
    ws = Workspace(args.workspace)
    db = ws.open_db()
    provider = SqliteEmbeddingProvider(db)

    # Verify model has embeddings
    available = provider.available_models()
    if args.model not in available:
        logger.error("No embeddings for %s. Available: %s", args.model, available)
        sys.exit(1)

    # ── Stage 1: Load and encode terms ──
    if args.dictionary:
        logger.info("Stage 1: Loading pre-computed dictionary from %s", args.dictionary)
        data = np.load(args.dictionary, allow_pickle=True)
        dict_words = list(data["words"])
        dict_vectors = data["vectors"]
        logger.info("Dictionary: %d words, dim %d", len(dict_words), dict_vectors.shape[1])

        term_infos = [TermInfo(taxonomy="dictionary", name=w, prompt=w, depth=0)
                      for w in dict_words]
        term_vectors = dict_vectors

        args.taxonomy_a = "dictionary"
        args.taxonomy_b = "dictionary"
        args.taxonomy_c = None
        logger.info("Dictionary mode: axes from PCA of %d word vectors", len(dict_words))
    elif args.ontology:
        logger.info("Stage 1: Loading custom ontology from %s", args.ontology)
        with open(args.ontology) as f:
            data = yaml.safe_load(f)
        root_key = next(iter(data))
        root = _parse_node(root_key, data[root_key])
        ontology_name = root_key
        logger.info("  %s: %d nodes, depth %d", ontology_name,
                     len(root.walk()), root.depth())

        # Single ontology → both axes from its PCA
        taxonomies = {ontology_name: root}
        term_infos = collect_term_infos(taxonomies)
        logger.info("Ontology terms: %d", len(term_infos))

        logger.info("Encoding terms via %s...", args.model)
        term_vectors = encode_terms(term_infos)

        args.taxonomy_a = ontology_name
        args.taxonomy_b = ontology_name
        args.taxonomy_c = None
        logger.info("Ontology mode: axes from PCA of %s (%d terms)",
                     ontology_name, len(term_infos))
    else:
        logger.info("Stage 1: Loading taxonomies...")
        taxonomies = load_all_taxonomies()
        available_names = list(taxonomies.keys())
        logger.info("Available taxonomies: %s", available_names)

        for name in [args.taxonomy_a, args.taxonomy_b]:
            if name not in taxonomies:
                logger.error("Taxonomy '%s' not found. Available: %s", name, available_names)
                sys.exit(1)
        if args.taxonomy_c and args.taxonomy_c not in taxonomies:
            logger.error("Taxonomy '%s' not found. Available: %s", args.taxonomy_c, available_names)
            sys.exit(1)

        term_infos = collect_term_infos(taxonomies)
        logger.info("Total terms: %d across %d taxonomies", len(term_infos), len(taxonomies))

        logger.info("Encoding terms via %s...", args.model)
        term_vectors = encode_terms(term_infos)

    # ── Stage 2: Excitation matrix ──
    logger.info("Stage 2: Computing excitation matrix...")
    image_ids = db.fetch_image_ids()
    image_matrix = provider.fetch_matrix(image_ids, args.model)
    logger.info("Image matrix: %s", image_matrix.shape)

    excitation = compute_excitation(image_matrix, term_vectors, term_infos)

    # ── Stage 3: Interference ──
    logger.info("Stage 3: Computing interference...")
    profile_a = compute_taxonomy_profile(excitation, term_infos, args.taxonomy_a)
    profile_b = compute_taxonomy_profile(excitation, term_infos, args.taxonomy_b)
    interference = compute_interference_strength(profile_a, profile_b)
    logger.info("Interference range: [%.3f, %.3f], mean=%.3f",
                interference.min(), interference.max(), interference.mean())

    # ── Stage 4: 3D projection ──
    logger.info("Stage 4: Projecting to 3D...")
    axes = compute_taxonomy_axes(
        term_vectors, term_infos,
        args.taxonomy_a, args.taxonomy_b, args.taxonomy_c,
    )
    positions = project_images(image_matrix, axes)
    logger.info("Position range: x=[%.3f,%.3f] y=[%.3f,%.3f] z=[%.3f,%.3f]",
                positions[:,0].min(), positions[:,0].max(),
                positions[:,1].min(), positions[:,1].max(),
                positions[:,2].min(), positions[:,2].max())

    # Subsample
    sample_idx = subsample(
        len(image_ids), args.max_points,
        excitation, term_infos,
        args.taxonomy_a, args.taxonomy_b,
    )

    pos_sub = positions[sample_idx]
    exc_sub = excitation[sample_idx]
    interf_sub = interference[sample_idx]
    ids_sub = [image_ids[i] for i in sample_idx]

    # Optional UMAP refinement
    if args.refine_umap:
        img_sub = image_matrix[sample_idx]
        # Normalize init positions to reasonable range for UMAP
        pos_norm = pos_sub.copy()
        for d in range(3):
            lo, hi = pos_norm[:, d].min(), pos_norm[:, d].max()
            if hi - lo > 1e-9:
                pos_norm[:, d] = (pos_norm[:, d] - lo) / (hi - lo) * 10
        pos_sub = refine_with_umap(img_sub, pos_norm)

    # ── Stage 5: Visual channels ──
    logger.info("Stage 5: Computing visual channels...")
    colors, sizes, opacities = compute_visual_channels(
        exc_sub, term_infos,
        args.taxonomy_a, args.taxonomy_b, args.taxonomy_c,
        interf_sub,
    )

    hover_texts = build_hover_texts(
        ids_sub, exc_sub, term_infos,
        args.taxonomy_a, args.taxonomy_b, interf_sub,
    )

    # ── Stage 6: Build and save ──
    logger.info("Stage 6: Building Plotly figure with density surface...")
    fig = build_figure(
        pos_sub, colors, sizes, opacities, hover_texts,
        args.taxonomy_a, args.taxonomy_b, args.taxonomy_c,
        all_positions=positions,        # full 74K for density
        all_excitation=excitation,      # full 74K for hue
        term_infos=term_infos,
        wireframe=args.wireframe,
    )
    # Write HTML with color legend
    is_dictionary = args.dictionary is not None
    plotly_html = fig.to_html(
        include_plotlyjs="cdn", full_html=False,
        config={"scrollZoom": True},
    )

    if is_dictionary:
        legend_html = """
        <div style="position:fixed;bottom:20px;left:20px;background:rgba(15,15,20,0.85);
                     padding:16px 20px;border-radius:8px;color:#ccc;font:13px/1.6 monospace;
                     max-width:360px;border:1px solid rgba(100,100,140,0.3);">
          <div style="color:#eee;font-size:14px;margin-bottom:8px;font-weight:bold;">
            Color Legend (Dictionary Mode)</div>
          <div><b>Hue</b> — PC1 of word excitation (what kind of words describe this image)</div>
          <div><b>Saturation</b> — PC2 (secondary word-space axis)</div>
          <div><b>Lightness</b> — PC3 (tertiary word-space axis)</div>
          <div style="margin-top:6px;"><b>Size</b> — interference strength (large = antinode)</div>
          <div><b>Opacity</b> — max word excitation (solid = strongly claimed)</div>
          <div style="margin-top:6px;"><b>Surface</b> — corpus density envelope, colored by
            dominant word character per region</div>
          <div style="margin-top:8px;color:#999;font-size:11px;">
            Hover over points to see the top 15 dictionary words — the narrative ingredients.</div>
        </div>"""
    else:
        legend_html = f"""
        <div style="position:fixed;bottom:20px;left:20px;background:rgba(15,15,20,0.85);
                     padding:16px 20px;border-radius:8px;color:#ccc;font:13px/1.6 monospace;
                     max-width:360px;border:1px solid rgba(100,100,140,0.3);">
          <div style="color:#eee;font-size:14px;margin-bottom:8px;font-weight:bold;">
            Color Legend</div>
          <div><span style="color:#e44;">Red</span> — {args.taxonomy_a} dominates</div>
          <div><span style="color:#4ae;">Cyan</span> — {args.taxonomy_b} dominates</div>
          <div><span style="color:#6c4;">Green</span> — balanced between both</div>
          <div style="margin-top:6px;"><b>Saturation</b> — interference (vivid = antinode, gray = node)</div>
          <div><b>Brightness</b> — specificity (bright = one term claims it, dim = ambiguous)</div>
          <div><b>Size</b> — interference strength (large = antinode)</div>
          <div style="margin-top:6px;"><b>Surface</b> — corpus density envelope, colored by
            taxonomy balance per region</div>
        </div>"""

    full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Sigil Space</title>
<style>body{{margin:0;background:rgb(8,8,12);}}</style>
</head><body>
{plotly_html}
{legend_html}
</body></html>"""

    args.output.write_text(full_html)

    dt = time.monotonic() - t_start
    logger.info("Sigil Space written to %s (%d points, %.1fs total)",
                args.output, len(pos_sub), dt)

    db.close()


if __name__ == "__main__":
    main()
