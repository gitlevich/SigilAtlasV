"""Sigil Multi-Lens — switch between ontology lenses in a single viewer.

Pre-computes all lens projections and embeds them in one HTML file.
A dropdown swaps the entire 3D scene instantly.

Usage:
    PYTHONPATH=python python tools/sigil_multilens.py \
        --workspace workspace \
        --output workspace/sigil_multilens.html
"""

import argparse
import colorsys
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from sigil_atlas.embedding_provider import SqliteEmbeddingProvider
from sigil_atlas.model_registry import get_adapter
from sigil_atlas.ontology import OntologyNode, _parse_node
from sigil_atlas.workspace import Workspace

import yaml

logger = logging.getLogger(__name__)

TOOLS_DIR = Path(__file__).resolve().parent
TAXONOMY_DIR = TOOLS_DIR.parent / "python" / "sigil_atlas"
MODEL_ID = "clip-vit-b-32"
MAX_POINTS = 12000


# ── Reuse core functions from sigil_space ───────────────────────

@dataclass
class TermInfo:
    taxonomy: str
    name: str
    prompt: str
    depth: int


def _percentile_normalize(arr, lo_pct=2, hi_pct=98):
    lo = np.percentile(arr, lo_pct)
    hi = np.percentile(arr, hi_pct)
    if hi - lo < 1e-9:
        return np.full_like(arr, 0.5)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def collect_term_infos(taxonomies):
    infos = []
    for tax_name, root in taxonomies.items():
        _collect(root, tax_name, 0, infos)
    return infos


def _collect(node, taxonomy, depth, out):
    out.append(TermInfo(taxonomy=taxonomy, name=node.name,
                        prompt=node.prompt, depth=depth))
    for child in node.children:
        _collect(child, taxonomy, depth + 1, out)


def encode_terms(term_infos):
    adapter = get_adapter(MODEL_ID)
    vectors = []
    for ti in term_infos:
        vectors.append(adapter.encode_text(ti.prompt))
    return np.stack(vectors)


# ── Lens definition ─────────────────────────────────────────────

@dataclass
class LensConfig:
    name: str
    kind: str  # "ontology", "dictionary", "taxonomy_pair"
    path: Path | None = None
    taxonomy_a: str | None = None
    taxonomy_b: str | None = None


def discover_lenses() -> list[LensConfig]:
    """Auto-discover available lenses."""
    lenses = []

    # Custom ontology YAMLs in tools/
    for f in sorted(TOOLS_DIR.glob("ontology_*.yaml")):
        stem = f.stem.replace("ontology_", "")
        lenses.append(LensConfig(name=stem, kind="ontology", path=f))

    # Dictionary if pre-computed
    dict_path = TOOLS_DIR / "dictionary_clip_b32.npz"
    if dict_path.exists():
        lenses.append(LensConfig(name="dictionary", kind="dictionary", path=dict_path))

    # Taxonomy pairs
    lenses.append(LensConfig(name="semantic x visual", kind="taxonomy_pair",
                              taxonomy_a="semantic", taxonomy_b="visual"))
    lenses.append(LensConfig(name="semantic x cinematic", kind="taxonomy_pair",
                              taxonomy_a="semantic", taxonomy_b="cinematic"))

    return lenses


# ── Compute one lens ────────────────────────────────────────────

def compute_lens(
    lens: LensConfig,
    image_matrix: np.ndarray,
    image_ids: list[str],
    all_taxonomies: dict[str, OntologyNode],
    sample_indices: np.ndarray | None = None,
    canonical_positions: np.ndarray | None = None,
) -> dict:
    """Compute positions, colors, sizes, hover texts for one lens.

    Returns a dict ready for JSON serialization.
    """
    t0 = time.monotonic()

    # ── Load terms ──
    if lens.kind == "dictionary":
        data = np.load(lens.path, allow_pickle=True)
        dict_words = list(data["words"])
        term_vectors = data["vectors"]
        term_infos = [TermInfo("dictionary", w, w, 0) for w in dict_words]
        single_taxonomy = True
        tax_name = "dictionary"
    elif lens.kind == "ontology":
        with open(lens.path) as f:
            raw = yaml.safe_load(f)
        root_key = next(iter(raw))
        root = _parse_node(root_key, raw[root_key])
        taxonomies = {root_key: root}
        term_infos = collect_term_infos(taxonomies)
        term_vectors = encode_terms(term_infos)
        single_taxonomy = True
        tax_name = root_key
    else:  # taxonomy_pair
        tax_a_name = lens.taxonomy_a
        tax_b_name = lens.taxonomy_b
        if tax_a_name not in all_taxonomies or tax_b_name not in all_taxonomies:
            logger.warning("Taxonomy %s or %s not found, skipping", tax_a_name, tax_b_name)
            return None
        taxonomies = {tax_a_name: all_taxonomies[tax_a_name],
                      tax_b_name: all_taxonomies[tax_b_name]}
        term_infos = collect_term_infos(taxonomies)
        term_vectors = encode_terms(term_infos)
        single_taxonomy = False
        tax_name = f"{tax_a_name} x {tax_b_name}"

    # ── Excitation ──
    excitation = image_matrix @ term_vectors.T

    # Z-normalize per taxonomy
    tax_names_set = list(dict.fromkeys(ti.taxonomy for ti in term_infos))
    for tn in tax_names_set:
        cols = [i for i, ti in enumerate(term_infos) if ti.taxonomy == tn]
        block = excitation[:, cols]
        mu, sigma = block.mean(), block.std()
        if sigma > 1e-9:
            excitation[:, cols] = (block - mu) / sigma

    # ── Detect contrast axes in ontology ──
    # Branches with exactly 2 children are contrast poles.
    # Use their directions (pole_a - pole_b) as spatial axes.
    contrast_axes_info = []  # [(label, direction_vector), ...]
    if lens.kind == "ontology":
        with open(lens.path) as f:
            raw = yaml.safe_load(f)
        root_key = next(iter(raw))
        root_node = _parse_node(root_key, raw[root_key])
        for branch in root_node.children:
            if len(branch.children) == 2:
                c0, c1 = branch.children
                # Find their encoded vectors
                idx0 = next((i for i, ti in enumerate(term_infos)
                             if ti.name == c0.name and ti.taxonomy == root_key), None)
                idx1 = next((i for i, ti in enumerate(term_infos)
                             if ti.name == c1.name and ti.taxonomy == root_key), None)
                if idx0 is not None and idx1 is not None:
                    direction = term_vectors[idx0] - term_vectors[idx1]
                    norm = np.linalg.norm(direction)
                    if norm > 1e-8:
                        direction = direction / norm
                        contrast_axes_info.append((
                            f"{c0.name} / {c1.name}",
                            direction.astype(np.float32),
                        ))
        if contrast_axes_info:
            logger.info("  Found %d contrast axes: %s",
                        len(contrast_axes_info),
                        [c[0] for c in contrast_axes_info[:6]])

    # ── 3D projection ──
    axis_labels = ["axis 1", "axis 2", "axis 3"]

    if contrast_axes_info and len(contrast_axes_info) >= 3:
        # Use the first 3 contrast directions, orthogonalized
        ax = contrast_axes_info[0][1].copy()
        ax = ax / max(np.linalg.norm(ax), 1e-8)

        ay = contrast_axes_info[1][1].copy()
        ay = ay - np.dot(ay, ax) * ax
        ay = ay / max(np.linalg.norm(ay), 1e-8)

        az = contrast_axes_info[2][1].copy()
        az = az - np.dot(az, ax) * ax - np.dot(az, ay) * ay
        norm_z = np.linalg.norm(az)
        if norm_z < 1e-8:
            az = contrast_axes_info[3][1].copy() if len(contrast_axes_info) > 3 else np.random.randn(term_vectors.shape[1]).astype(np.float32)
            az = az - np.dot(az, ax) * ax - np.dot(az, ay) * ay
            norm_z = np.linalg.norm(az)
        az = az / max(norm_z, 1e-8)

        axes = np.stack([ax, ay, az])
        axis_labels = [contrast_axes_info[0][0],
                       contrast_axes_info[1][0],
                       contrast_axes_info[2][0]]
    elif single_taxonomy:
        pca = PCA(n_components=3)
        pca.fit(term_vectors)
        axes = pca.components_.astype(np.float32)
    else:
        # Two-taxonomy axis projection
        def pdir(tn):
            cols = [i for i, ti in enumerate(term_infos) if ti.taxonomy == tn]
            p = PCA(n_components=1)
            p.fit(term_vectors[cols])
            return p.components_[0].astype(np.float32)

        ax = pdir(lens.taxonomy_a)
        ax = ax / max(np.linalg.norm(ax), 1e-8)
        ay = pdir(lens.taxonomy_b)
        ay = ay - np.dot(ay, ax) * ax
        ay = ay / max(np.linalg.norm(ay), 1e-8)
        # Residual
        cols_a = [i for i, ti in enumerate(term_infos) if ti.taxonomy == lens.taxonomy_a]
        p3 = PCA(n_components=3)
        p3.fit(term_vectors[cols_a])
        az = p3.components_[2].astype(np.float32)
        az = az - np.dot(az, ax) * ax - np.dot(az, ay) * ay
        az = az / max(np.linalg.norm(az), 1e-8)
        axes = np.stack([ax, ay, az])

    # Use canonical positions — same spatial layout for all lenses
    positions = canonical_positions

    # ── Subsample ──
    if sample_indices is None:
        n = len(image_ids)
        if n > MAX_POINTS:
            sample_indices = np.random.choice(n, MAX_POINTS, replace=False)
            sample_indices.sort()
        else:
            sample_indices = np.arange(n)

    pos = positions[sample_indices]
    exc = excitation[sample_indices]
    ids = [image_ids[i] for i in sample_indices]

    # ── Colors (PCA-based for single, atan2 for pairs) ──
    n = len(pos)
    if single_taxonomy:
        cpca = PCA(n_components=3)
        coords = cpca.fit_transform(exc).astype(np.float32)
        hue = _percentile_normalize(coords[:, 0])
        sat = 0.5 + 0.5 * _percentile_normalize(coords[:, 1])
        lit = 0.20 + 0.35 * _percentile_normalize(coords[:, 2])
    else:
        cols_a = [i for i, ti in enumerate(term_infos) if ti.taxonomy == lens.taxonomy_a]
        cols_b = [i for i, ti in enumerate(term_infos) if ti.taxonomy == lens.taxonomy_b]
        ea = exc[:, cols_a].max(axis=1)
        eb = exc[:, cols_b].max(axis=1)
        ra = np.argsort(np.argsort(ea)).astype(np.float32) / max(n - 1, 1)
        rb = np.argsort(np.argsort(eb)).astype(np.float32) / max(n - 1, 1)
        hue = (np.arctan2(rb - 0.5, ra - 0.5) + np.pi) / (2 * np.pi)
        sat = np.full(n, 0.8, dtype=np.float32)
        lit = 0.20 + 0.30 * _percentile_normalize(exc.max(axis=1))

    colors = []
    for i in range(n):
        r, g, b = colorsys.hls_to_rgb(float(hue[i]), float(lit[i]), float(sat[i]))
        colors.append(f"rgb({int(r*255)},{int(g*255)},{int(b*255)})")

    # ── Sizes ──
    max_exc = exc.max(axis=1)
    sizes = (2.0 + 5.0 * _percentile_normalize(max_exc)).tolist()

    # ── Hover text with contrast scores ──
    hover = []
    top_k = 12
    # If we have contrast axes, show scores along them
    contrast_labels = [c[0] for c in contrast_axes_info] if contrast_axes_info else []
    contrast_dirs = [c[1] for c in contrast_axes_info] if contrast_axes_info else []

    for i in range(n):
        si = sample_indices[i]
        parts = [f"<b>{ids[i][:35]}</b>"]

        # Show contrast axis positions
        if contrast_dirs:
            img_vec = image_matrix[si]
            scores_line = []
            for label, direction in zip(contrast_labels[:6], contrast_dirs[:6]):
                proj = float(np.dot(img_vec, direction))
                poles = label.split(" / ")
                arrow = poles[0] if proj > 0 else poles[1] if len(poles) > 1 else "?"
                scores_line.append(f"{arrow}({proj:+.3f})")
            parts.append(" ".join(scores_line))

        # Top terms
        scores = exc[i]
        top_idx = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        words = ", ".join(term_infos[j].name for j in top_idx)
        parts.append(f"<i>{words}</i>")
        hover.append("<br>".join(parts))

    # ── Excitation-weighted density on canonical positions ──
    # Each lens illuminates different regions of the same space.
    # Weight = max excitation across this lens's terms per image.
    from scipy.ndimage import gaussian_filter
    grid_size = 40
    all_pos = positions  # full 74K, canonical
    p_min = all_pos.min(axis=0)
    p_max = all_pos.max(axis=0)
    pad = (p_max - p_min) * 0.08
    p_min -= pad
    p_max += pad
    span = p_max - p_min
    edges = [np.linspace(p_min[d], p_max[d], grid_size + 1) for d in range(3)]

    # Compute excitation weight per image (how strongly this lens claims it)
    exc_weight = excitation.max(axis=1)
    exc_weight = np.clip(exc_weight, 0, None)  # only positive excitation

    # Bin into grid with excitation weighting
    density_weighted = np.zeros([grid_size] * 3, dtype=np.float32)
    bin_idx = []
    for d in range(3):
        idx = np.digitize(all_pos[:, d], edges[d]) - 1
        idx = np.clip(idx, 0, grid_size - 1)
        bin_idx.append(idx)
    for i in range(len(all_pos)):
        density_weighted[bin_idx[0][i], bin_idx[1][i], bin_idx[2][i]] += exc_weight[i]

    density = gaussian_filter(density_weighted, sigma=1.2)
    centers = [(e[:-1] + e[1:]) / 2 for e in edges]
    gx, gy, gz = np.meshgrid(centers[0], centers[1], centers[2], indexing='ij')
    dmax = density.max()
    dnorm = (density / dmax if dmax > 1e-9 else density).flatten().tolist()

    # ── Wireframe shells via marching cubes ──
    # Same canonical coordinate system for all lenses
    wireframe_shells = []
    try:
        from skimage.measure import marching_cubes
        density_norm_3d = density / dmax if dmax > 1e-9 else density
        levels = [0.08, 0.15, 0.25, 0.40, 0.60]
        for k, level in enumerate(levels):
            if level > density_norm_3d.max():
                continue
            try:
                verts, faces, _, _ = marching_cubes(density_norm_3d, level=level)
            except (ValueError, RuntimeError):
                continue
            if len(verts) == 0:
                continue
            # Map grid coords back to canonical data coords
            for d in range(3):
                verts[:, d] = p_min[d] + verts[:, d] / grid_size * span[d]
            # Subsample edges for manageable JSON size
            step = max(1, len(faces) // 2000)
            ex, ey, ez = [], [], []
            for face in faces[::step]:
                for a_i, b_i in [(0, 1), (1, 2), (2, 0)]:
                    ex.extend([float(verts[face[a_i], 0]), float(verts[face[b_i], 0]), None])
                    ey.extend([float(verts[face[a_i], 1]), float(verts[face[b_i], 1]), None])
                    ez.extend([float(verts[face[a_i], 2]), float(verts[face[b_i], 2]), None])
            alpha = 0.12 + 0.18 * (k / max(len(levels) - 1, 1))
            wireframe_shells.append({
                "x": ex, "y": ey, "z": ez,
                "color": f"rgba(140,170,220,{alpha:.2f})",
                "opacity": alpha,
            })
        logger.info("  Wireframe: %d shells", len(wireframe_shells))
    except ImportError:
        logger.warning("  skimage not available, skipping wireframe")

    dt = time.monotonic() - t0
    logger.info("  Lens '%s': %d terms, %d points, %.1fs",
                lens.name, len(term_infos), n, dt)

    return {
        "name": lens.name,
        "x": pos[:, 0].tolist(),
        "y": pos[:, 1].tolist(),
        "z": pos[:, 2].tolist(),
        "colors": colors,
        "sizes": sizes,
        "hover": hover,
        "axis_labels": axis_labels,
        # Density grid for surface rendering
        "grid_x": gx.flatten().tolist(),
        "grid_y": gy.flatten().tolist(),
        "grid_z": gz.flatten().tolist(),
        "density": dnorm,
        # Wireframe shells
        "wireframe": wireframe_shells,
    }


# ── HTML generation ─────────────────────────────────────────────

def build_multilens_html(lens_data: list[dict], output: Path) -> None:
    """Generate a self-contained HTML with lens switcher."""

    # Serialize lens data as JSON
    lenses_json = json.dumps(lens_data)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Sigil Space — Multi-Lens</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  body {{ margin: 0; background: rgb(8,8,12); font-family: system-ui, sans-serif; }}
  #controls {{
    position: fixed; top: 12px; left: 50%; transform: translateX(-50%);
    z-index: 100; display: flex; gap: 6px; flex-wrap: wrap; justify-content: center;
  }}
  .lens-btn, .mode-btn {{
    background: rgba(30,30,45,0.85); color: rgba(180,180,200,0.8);
    border: 1px solid rgba(80,80,120,0.4); border-radius: 6px;
    padding: 8px 16px; font-size: 13px; cursor: pointer;
    transition: all 0.15s;
  }}
  .lens-btn:hover, .mode-btn:hover {{ background: rgba(50,50,70,0.9); color: #fff; }}
  .lens-btn.active {{
    background: rgba(60,60,100,0.9); color: #fff;
    border-color: rgba(120,120,180,0.6);
  }}
  .mode-btn.active {{
    background: rgba(40,60,50,0.9); color: #afd;
    border-color: rgba(100,160,120,0.6);
  }}
  .separator {{
    width: 1px; background: rgba(80,80,120,0.3); margin: 0 4px;
    align-self: stretch;
  }}
  #plot {{ width: 100vw; height: 100vh; }}
  #slice-bar {{
    position: fixed; bottom: 0; left: 0; right: 0; z-index: 200;
    background: rgba(10,10,15,0.95); padding: 8px 20px 10px;
    border-top: 1px solid rgba(80,80,120,0.3);
  }}
  #slice-track {{
    width: 100%; height: 6px; background: rgba(40,40,60,0.8);
    border-radius: 3px; position: relative; cursor: pointer;
  }}
  #slice-cursor {{
    position: absolute; top: -4px; width: 14px; height: 14px;
    background: rgba(120,160,220,0.9); border-radius: 50%;
    transform: translateX(-7px); pointer-events: none;
    box-shadow: 0 0 8px rgba(100,140,200,0.5);
  }}
  #slice-info {{
    display: flex; justify-content: space-between; align-items: center;
    margin-top: 6px; color: rgba(180,180,200,0.7); font: 12px/1 monospace;
  }}
  .tk {{
    display: inline-block; width: 28px; height: 22px; line-height: 22px;
    text-align: center; background: rgba(30,30,50,0.8);
    border: 1px solid rgba(80,80,120,0.4); border-radius: 4px;
    margin: 0 2px; cursor: pointer; font-size: 11px; font-weight: bold;
    color: rgba(180,180,200,0.7); transition: all 0.1s;
  }}
  .tk.active {{ background: rgba(60,80,120,0.9); color: #fff; border-color: rgba(120,160,220,0.6); }}
  .tk:hover {{ background: rgba(50,50,70,0.9); color: #fff; }}
  #scan-keys {{
    position: fixed; top: 80px; right: 20px; z-index: 210;
    background: rgba(12,12,18,0.92); padding: 12px 16px; border-radius: 8px;
    border: 1px solid rgba(80,80,120,0.3); color: rgba(160,160,190,0.7);
    font: 11px/1.8 monospace; width: 260px;
  }}
  #scan-keys b {{ color: rgba(200,200,220,0.8); }}
  #scan-status {{ color: rgba(120,180,240,0.9); margin-top: 6px; font-size: 12px; }}
  #legend {{
    position: fixed; bottom: 20px; left: 20px;
    background: rgba(15,15,20,0.85); padding: 14px 18px; border-radius: 8px;
    color: #ccc; font: 13px/1.6 monospace; max-width: 360px;
    border: 1px solid rgba(100,100,140,0.3);
  }}
  #legend b {{ color: #eee; }}
  #axes-info {{ color: rgba(160,160,200,0.7); font-size: 11px; margin-top: 6px; }}
</style>
</head><body>

<div id="controls"></div>
<div id="plot"></div>
<div id="plot2d" style="display:none;width:100vw;height:100vh;"></div>
<div id="slice-bar" style="display:none;">
  <div id="slice-track">
    <div id="slice-cursor"></div>
  </div>
  <div id="slice-info">
    <span id="slice-label">0.000</span>
    <span id="slice-count" style="margin-left:8px;">0 pts</span>
    <span id="angle-val" style="margin-left:8px;">0</span>
  </div>
</div>
<div id="scan-keys" style="display:none;">
  <div><b>Scan Controls</b></div>
  <div>L 2x 4x 8x 16x &nbsp; K stop &nbsp; J rev 2x 4x 8x 16x</div>
  <div>Hold K + J/L = frame step</div>
  <div>Left/Right pan &nbsp; Up/Down tilt &nbsp; +/- thickness &nbsp; Space play/stop</div>
  <div id="scan-status"></div>
</div>
<div id="legend">
  <b id="legend-title"></b>
  <div id="axes-info"></div>
  <div style="margin-top:6px;color:#999;font-size:11px;">
    Hover points for contrast scores and top terms.
  </div>
</div>

<script>
var LENSES = {lenses_json};
var currentLens = 0;
var showSurface = false;
var showWireframe = false;
var showAllWireframes = false;
var showPoints = true;
var sliceActive = false;
var sliceAngleH = 0;    // horizontal heading, degrees
var sliceAngleV = 0;    // vertical tilt, degrees (-90 to 90)
var slicePos = 0.5;     // normalized [0, 1] — middle
var sliceThick = 0.02;  // half-thickness as fraction of range
var playDirection = 0;  // -1=J reverse, 0=K stopped, 1=L forward
var playSpeed = 0.002;  // fraction per frame
var playRAF = null;
var panX = 0;           // pan offset in screen-right direction
var panY = 0;           // pan offset in screen-up direction
var panStep = 0.05;     // how far each arrow press moves

// Distinct hues per lens for "all" mode
var LENS_COLORS = [
  [255, 80, 100],   // red-pink
  [80, 200, 255],   // cyan
  [255, 200, 50],   // gold
  [100, 255, 130],  // green
  [200, 120, 255],  // violet
  [255, 140, 60],   // orange
  [60, 255, 220],   // teal
];

var axisBase = {{
  gridcolor: 'rgba(60,60,80,0.3)',
  zerolinecolor: 'rgba(80,80,120,0.3)',
  showbackground: false,
  tickfont: {{ color: 'rgba(180,180,200,0.5)', size: 9 }},
}};

function makeLayout(lens) {{
  var labels = lens.axis_labels || ['axis 1', 'axis 2', 'axis 3'];
  return {{
    scene: {{
      xaxis: Object.assign({{}}, axisBase, {{
        title: {{ text: labels[0], font: {{ color: 'rgba(200,200,220,0.8)', size: 12 }} }}
      }}),
      yaxis: Object.assign({{}}, axisBase, {{
        title: {{ text: labels[1], font: {{ color: 'rgba(200,200,220,0.8)', size: 12 }} }}
      }}),
      zaxis: Object.assign({{}}, axisBase, {{
        title: {{ text: labels[2], font: {{ color: 'rgba(200,200,220,0.8)', size: 12 }} }}
      }}),
      bgcolor: 'rgb(8,8,12)',
      dragmode: 'turntable'
    }},
    paper_bgcolor: 'rgb(8,8,12)',
    font: {{ color: 'rgba(200,200,220,0.7)' }},
    showlegend: false,
    margin: {{ l: 0, r: 0, t: 0, b: 0 }}
  }};
}}

function makeTraces(lens) {{
  var traces = [];

  // Scatter points
  if (showPoints) {{
    traces.push({{
      type: 'scatter3d',
      x: lens.x, y: lens.y, z: lens.z,
      mode: 'markers',
      marker: {{ size: lens.sizes, color: lens.colors, line: {{ width: 0 }} }},
      text: lens.hover,
      hoverinfo: 'text',
      name: 'images'
    }});
  }}

  // Density surface (solid)
  if (showSurface && lens.density) {{
    traces.push({{
      type: 'isosurface',
      x: lens.grid_x, y: lens.grid_y, z: lens.grid_z,
      value: lens.density,
      isomin: 0.06,
      isomax: 0.7,
      surface: {{ count: 12 }},
      colorscale: [
        [0, 'rgba(10,15,40,0.8)'], [0.25, 'rgba(30,50,120,0.8)'],
        [0.5, 'rgba(60,100,180,0.8)'], [0.75, 'rgba(120,160,220,0.8)'],
        [1.0, 'rgba(200,220,255,0.8)']
      ],
      opacity: 0.15,
      caps: {{ x: {{ show: false }}, y: {{ show: false }}, z: {{ show: false }} }},
      showscale: false,
      hoverinfo: 'skip',
      name: 'density'
    }});
  }}

  // Wireframe shells (single lens)
  if (showWireframe && lens.wireframe) {{
    lens.wireframe.forEach(function(shell) {{
      traces.push({{
        type: 'scatter3d',
        x: shell.x, y: shell.y, z: shell.z,
        mode: 'lines',
        line: {{ color: shell.color, width: 1 }},
        opacity: shell.opacity,
        showlegend: false,
        hoverinfo: 'skip',
        name: 'wireframe'
      }});
    }});
  }}

  // All wireframes overlaid — every lens at once, each a different color
  if (showAllWireframes) {{
    LENSES.forEach(function(otherLens, li) {{
      if (!otherLens.wireframe) return;
      var rgb = LENS_COLORS[li % LENS_COLORS.length];
      otherLens.wireframe.forEach(function(shell, si) {{
        var alpha = 0.08 + 0.14 * (si / Math.max(otherLens.wireframe.length - 1, 1));
        traces.push({{
          type: 'scatter3d',
          x: shell.x, y: shell.y, z: shell.z,
          mode: 'lines',
          line: {{ color: 'rgba(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ',' + alpha.toFixed(2) + ')', width: 1 }},
          opacity: alpha,
          showlegend: false,
          hoverinfo: 'skip',
          name: otherLens.name
        }});
      }});
    }});
  }}

  return traces;
}}

function render() {{
  var lens = LENSES[currentLens];
  var traces = makeTraces(lens);
  Plotly.react('plot', traces, makeLayout(lens), {{ scrollZoom: true }});

  document.querySelectorAll('.lens-btn').forEach(function(btn, i) {{
    btn.classList.toggle('active', i === currentLens);
  }});
  document.getElementById('legend-title').textContent =
    showAllWireframes ? 'All Lenses' : lens.name;

  var info = '';
  if (showAllWireframes) {{
    // Color key for each lens
    LENSES.forEach(function(l, i) {{
      var rgb = LENS_COLORS[i % LENS_COLORS.length];
      info += '<span style="color:rgb(' + rgb.join(',') + ')">' + l.name + '</span><br>';
    }});
  }} else {{
    var labels = lens.axis_labels || [];
    if (labels.length > 0) {{
      info = 'X: ' + labels[0] + '<br>Y: ' + labels[1] + '<br>Z: ' + labels[2];
    }}
  }}
  document.getElementById('axes-info').innerHTML = info;
}}

function switchLens(idx) {{ currentLens = idx; render(); }}
function toggleSurface() {{ showSurface = !showSurface; showWireframe = false; showAllWireframes = false; render(); updateModeButtons(); }}
function toggleWireframe() {{ showWireframe = !showWireframe; showSurface = false; showAllWireframes = false; render(); updateModeButtons(); }}
function togglePoints() {{ showPoints = !showPoints; render(); updateModeButtons(); }}
function toggleAllWireframes() {{ showAllWireframes = !showAllWireframes; showWireframe = false; showSurface = false; render(); updateModeButtons(); }}

function updateModeButtons() {{
  document.getElementById('btn-surface').classList.toggle('active', showSurface);
  document.getElementById('btn-wireframe').classList.toggle('active', showWireframe);
  document.getElementById('btn-all').classList.toggle('active', showAllWireframes);
  document.getElementById('btn-points').classList.toggle('active', showPoints);
}}

// Build controls
var controls = document.getElementById('controls');
LENSES.forEach(function(lens, i) {{
  var btn = document.createElement('button');
  btn.className = 'lens-btn';
  btn.textContent = lens.name;
  btn.addEventListener('click', function() {{ switchLens(i); }});
  controls.appendChild(btn);
}});

// Separator + mode toggles
var sep = document.createElement('div');
sep.className = 'separator';
controls.appendChild(sep);

var btnPts = document.createElement('button');
btnPts.id = 'btn-points';
btnPts.className = 'mode-btn active';
btnPts.textContent = 'Points';
btnPts.addEventListener('click', togglePoints);
controls.appendChild(btnPts);

var btnWire = document.createElement('button');
btnWire.id = 'btn-wireframe';
btnWire.className = 'mode-btn';
btnWire.textContent = 'Wireframe';
btnWire.addEventListener('click', toggleWireframe);
controls.appendChild(btnWire);

var btnAll = document.createElement('button');
btnAll.id = 'btn-all';
btnAll.className = 'mode-btn';
btnAll.textContent = 'All Lenses';
btnAll.addEventListener('click', toggleAllWireframes);
controls.appendChild(btnAll);

var btnSlice = document.createElement('button');
btnSlice.id = 'btn-slice';
btnSlice.className = 'mode-btn';
btnSlice.textContent = 'Scan';
btnSlice.addEventListener('click', function() {{
  sliceActive = !sliceActive;
  btnSlice.classList.toggle('active', sliceActive);
  document.getElementById('plot').style.display = sliceActive ? 'none' : '';
  document.getElementById('plot2d').style.display = sliceActive ? '' : 'none';
  document.getElementById('slice-bar').style.display = sliceActive ? '' : 'none';
  document.getElementById('scan-keys').style.display = sliceActive ? '' : 'none';
  if (sliceActive) {{ renderSlice2D(); }} else {{ render(); }}
}});
controls.appendChild(btnSlice);

var btnSurf = document.createElement('button');
btnSurf.id = 'btn-surface';
btnSurf.className = 'mode-btn';
btnSurf.textContent = 'Surface';
btnSurf.addEventListener('click', toggleSurface);
controls.appendChild(btnSurf);

// ── 2D Slice viewer with JKL playback ──

// Precompute stable axis ranges per lens (so view doesn't jump during playback)
var lensRanges = {{}};
function getLensRange(lens) {{
  if (lensRanges[lens.name]) return lensRanges[lens.name];
  var xmn=Infinity, xmx=-Infinity, ymn=Infinity, ymx=-Infinity, zmn=Infinity, zmx=-Infinity;
  for (var i=0; i<lens.x.length; i++) {{
    if (lens.x[i]<xmn) xmn=lens.x[i]; if (lens.x[i]>xmx) xmx=lens.x[i];
    if (lens.y[i]<ymn) ymn=lens.y[i]; if (lens.y[i]>ymx) ymx=lens.y[i];
    if (lens.z[i]<zmn) zmn=lens.z[i]; if (lens.z[i]>zmx) zmx=lens.z[i];
  }}
  var pad = 0.05;
  var r = {{ x:[xmn-(xmx-xmn)*pad, xmx+(xmx-xmn)*pad],
             y:[ymn-(ymx-ymn)*pad, ymx+(ymx-ymn)*pad],
             z:[zmn-(zmx-zmn)*pad, zmx+(zmx-zmn)*pad] }};
  lensRanges[lens.name] = r;
  return r;
}}

function renderSlice2D() {{
  var lens = LENSES[currentLens];
  var n = lens.x.length;

  // Time direction from horizontal + vertical angles
  var hRad = sliceAngleH * Math.PI / 180;
  var vRad = sliceAngleV * Math.PI / 180;
  var cosV = Math.cos(vRad);
  // Direction vector (spherical to cartesian)
  var dirX = Math.cos(hRad) * cosV;
  var dirY = Math.sin(hRad) * cosV;
  var dirZ = Math.sin(vRad);

  // Screen axes via Gram-Schmidt
  // "right" = perpendicular to dir in the horizontal plane
  var rightX = -Math.sin(hRad), rightY = Math.cos(hRad), rightZ = 0;
  // "up" = cross(dir, right)
  var upX = dirY * rightZ - dirZ * rightY;
  var upY = dirZ * rightX - dirX * rightZ;
  var upZ = dirX * rightY - dirY * rightX;

  // Project all points
  var tAll = new Float32Array(n);
  var sxAll = new Float32Array(n);
  var syAll = new Float32Array(n);
  for (var i = 0; i < n; i++) {{
    tAll[i] = lens.x[i]*dirX + lens.y[i]*dirY + lens.z[i]*dirZ;
    sxAll[i] = lens.x[i]*rightX + lens.y[i]*rightY + lens.z[i]*rightZ;
    syAll[i] = lens.x[i]*upX + lens.y[i]*upY + lens.z[i]*upZ;
  }}

  // Time range (stable across playback)
  var tMin = Infinity, tMax = -Infinity;
  for (var i = 0; i < n; i++) {{
    if (tAll[i] < tMin) tMin = tAll[i];
    if (tAll[i] > tMax) tMax = tAll[i];
  }}
  var tRange = tMax - tMin;
  var center = tMin + slicePos * tRange;
  var halfT = sliceThick * tRange;

  // Stable screen ranges (computed once per lens+angle)
  var rng = getLensRange(lens);
  var sxRange = [
    rng.x[0]*rightX + rng.y[0]*rightY,
    rng.x[1]*rightX + rng.y[1]*rightY
  ].sort(function(a,b){{return a-b;}});
  // Widen to cover all possible rotations
  var maxSpread = Math.max(rng.x[1]-rng.x[0], rng.y[1]-rng.y[0]) * 0.75;
  sxRange = [-maxSpread, maxSpread];
  var syRange = [rng.z[0], rng.z[1]];

  // Filter to slab
  var fx=[], fy=[], fc=[], fs=[], fh=[], fLabels=[], fExc=[];
  for (var i = 0; i < n; i++) {{
    if (Math.abs(tAll[i] - center) <= halfT) {{
      fx.push(sxAll[i]);
      fy.push(syAll[i]);
      fc.push(lens.colors[i]);
      fs.push(lens.sizes[i] * 1.8);
      fh.push(lens.hover[i]);
      // Extract first term name from hover (after <i> tag, before comma)
      var m = lens.hover[i].match(/<i>([^,<]+)/);
      fLabels.push(m ? m[1] : '');
      fExc.push(lens.sizes[i]); // size ≈ excitation
    }}
  }}

  // Show labels only on top 25 points by excitation (to avoid clutter)
  var labelTexts = new Array(fLabels.length).fill('');
  if (fLabels.length > 0) {{
    var idxSorted = fExc.map(function(_,i){{return i;}}).sort(function(a,b){{return fExc[b]-fExc[a];}});
    var showCount = Math.min(25, idxSorted.length);
    for (var k = 0; k < showCount; k++) {{
      labelTexts[idxSorted[k]] = fLabels[idxSorted[k]];
    }}
  }}

  var traces = [{{
    type: 'scatter',
    x: fx, y: fy,
    mode: 'markers+text',
    marker: {{ size: fs, color: fc, line: {{ width: 0 }} }},
    text: labelTexts,
    textposition: 'top center',
    textfont: {{ size: 9, color: 'rgba(200,200,220,0.7)' }},
    hovertext: fh,
    hoverinfo: 'text'
  }}];

  var layout2d = {{
    xaxis: {{ range: [sxRange[0]+panX, sxRange[1]+panX], autorange: false,
              gridcolor: 'rgba(40,40,60,0.3)',
              zerolinecolor: 'rgba(60,60,80,0.2)', color: 'rgba(150,150,180,0.4)',
              showticklabels: false }},
    yaxis: {{ range: [syRange[0]+panY, syRange[1]+panY], autorange: false,
              gridcolor: 'rgba(40,40,60,0.3)',
              zerolinecolor: 'rgba(60,60,80,0.2)', color: 'rgba(150,150,180,0.4)',
              showticklabels: false }},
    plot_bgcolor: 'rgb(8,8,12)',
    paper_bgcolor: 'rgb(8,8,12)',
    font: {{ color: 'rgba(180,180,200,0.5)' }},
    showlegend: false,
    margin: {{ l: 20, r: 20, t: 10, b: 30 }}
  }};
  Plotly.react('plot2d', traces, layout2d);

  // Update transport bar
  document.getElementById('slice-cursor').style.left = (slicePos * 100) + '%';
  document.getElementById('slice-label').textContent = center.toFixed(3);
  document.getElementById('slice-count').textContent = fx.length + ' pts';
  var status = document.getElementById('scan-status');
  if (status) {{
    var speedLabel = speedLevel === 0 ? 'STOP' :
      (speedLevel > 0 ? '' : '-') + Math.pow(2, Math.abs(speedLevel)-1) + 'x';
    status.textContent = speedLabel +
      ' pan:' + Math.round(sliceAngleH) +
      ' tilt:' + Math.round(sliceAngleV) +
      ' thick:' + (sliceThick*100).toFixed(1) + '%';
  }}
}}

function stopPlay() {{
  speedLevel = 0;
  setSpeedFromLevel();
}}

function startPlay(dir) {{
  speedLevel = dir;
  setSpeedFromLevel();
}}

// Click on track to seek
document.getElementById('slice-track').addEventListener('click', function(e) {{
  var rect = this.getBoundingClientRect();
  slicePos = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  renderSlice2D();
}});

// JKL transport: J doubles reverse speed, L doubles forward speed, K stops.
// While K is held: J = 1 frame back, L = 1 frame forward.
var speedLevel = 0;     // 0=stopped, positive=forward, negative=reverse
var kHeld = false;
var baseSpeed = 0.002;  // 1x speed

function setSpeedFromLevel() {{
  if (speedLevel === 0) {{
    playDirection = 0;
    playSpeed = 0;
  }} else {{
    playDirection = speedLevel > 0 ? 1 : -1;
    playSpeed = baseSpeed * Math.pow(2, Math.abs(speedLevel) - 1);
  }}
}}

function stepFrame(dir) {{
  slicePos += dir * baseSpeed * 0.5;
  if (slicePos > 1) slicePos -= 1;
  if (slicePos < 0) slicePos += 1;
  renderSlice2D();
}}

document.addEventListener('keydown', function(e) {{
  if (!sliceActive) return;
  var handled = true;
  switch(e.key) {{
    case 'k': case 'K':
      kHeld = true;
      speedLevel = 0;
      setSpeedFromLevel();
      break;
    case 'j': case 'J':
      if (kHeld) {{
        stepFrame(-1);
      }} else {{
        if (speedLevel > 0) speedLevel = 0; // was going forward, stop first
        else speedLevel = Math.max(-4, speedLevel - 1);
        setSpeedFromLevel();
      }}
      break;
    case 'l': case 'L':
      if (kHeld) {{
        stepFrame(1);
      }} else {{
        if (speedLevel < 0) speedLevel = 0; // was going reverse, stop first
        else speedLevel = Math.min(4, speedLevel + 1);
        setSpeedFromLevel();
      }}
      break;
    case ' ':
      if (speedLevel !== 0) {{ speedLevel = 0; }} else {{ speedLevel = 1; }}
      setSpeedFromLevel();
      break;
    case 'ArrowLeft':  sliceAngleH = (sliceAngleH - 5 + 360) % 360; lensRanges={{}}; renderSlice2D(); break;
    case 'ArrowRight': sliceAngleH = (sliceAngleH + 5) % 360; lensRanges={{}}; renderSlice2D(); break;
    case 'ArrowUp':    sliceAngleV = Math.min(90, sliceAngleV + 5); lensRanges={{}}; renderSlice2D(); break;
    case 'ArrowDown':  sliceAngleV = Math.max(-90, sliceAngleV - 5); lensRanges={{}}; renderSlice2D(); break;
    case '=': case '+': sliceThick = Math.min(0.3, sliceThick * 1.3); renderSlice2D(); break;
    case '-': case '_': sliceThick = Math.max(0.003, sliceThick * 0.7); renderSlice2D(); break;
    default: handled = false;
  }}
  if (handled) e.preventDefault();
}});

document.addEventListener('keyup', function(e) {{
  if (e.key === 'k' || e.key === 'K') kHeld = false;
}});

// Flight loop: advances position along current heading. Never touches heading.
setInterval(function() {{
  if (!sliceActive || playDirection === 0) return;
  slicePos += playDirection * playSpeed;
  if (slicePos > 1) slicePos -= 1;
  if (slicePos < 0) slicePos += 1;
  renderSlice2D();
}}, 16);


render();
</script>
</body></html>"""

    output.write_text(html)
    size_mb = output.stat().st_size / (1024 * 1024)
    logger.info("Multi-lens viewer: %s (%.1f MB, %d lenses)", output, size_mb, len(lens_data))


def main():
    parser = argparse.ArgumentParser(description="Sigil Multi-Lens viewer")
    parser.add_argument("--workspace", type=Path, default=Path("workspace"))
    parser.add_argument("--output", type=Path,
                        default=Path("workspace/sigil_multilens.html"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    t_start = time.monotonic()

    # Load corpus
    ws = Workspace(args.workspace)
    db = ws.open_db()
    provider = SqliteEmbeddingProvider(db)
    image_ids = db.fetch_image_ids()
    image_matrix = provider.fetch_matrix(image_ids, MODEL_ID)
    logger.info("Corpus: %d images, dim %d", len(image_ids), image_matrix.shape[1])

    # Load ALL taxonomy files (not just the subset in taxonomy.py's TAXONOMY_FILES)
    all_taxonomies = {}
    for path in sorted(TAXONOMY_DIR.glob("taxonomy_*.yaml")):
        with open(path) as f:
            data = yaml.safe_load(f)
        root_key = next(iter(data))
        all_taxonomies[root_key] = _parse_node(root_key, data[root_key])
    logger.info("Loaded %d taxonomies: %s", len(all_taxonomies), list(all_taxonomies.keys()))

    # ── Canonical positions: PCA of raw CLIP embeddings ──
    # One fixed spatial layout. Lenses don't move images, they illuminate them.
    logger.info("Computing canonical 3D positions (PCA of raw embeddings)...")
    pca_canon = PCA(n_components=3)
    canonical_positions = pca_canon.fit_transform(image_matrix).astype(np.float32)
    logger.info("Canonical PCA variance: %.1f%%, %.1f%%, %.1f%%",
                pca_canon.explained_variance_ratio_[0] * 100,
                pca_canon.explained_variance_ratio_[1] * 100,
                pca_canon.explained_variance_ratio_[2] * 100)

    # Shared subsample for consistent point IDs across lenses
    n = len(image_ids)
    if n > MAX_POINTS:
        sample_indices = np.random.choice(n, MAX_POINTS, replace=False)
        sample_indices.sort()
    else:
        sample_indices = np.arange(n)

    # Discover and compute all lenses
    lenses = discover_lenses()
    logger.info("Discovered %d lenses: %s", len(lenses), [l.name for l in lenses])

    lens_data = []
    for lens in lenses:
        result = compute_lens(lens, image_matrix, image_ids,
                              all_taxonomies, sample_indices,
                              canonical_positions)
        if result:
            lens_data.append(result)

    build_multilens_html(lens_data, args.output)

    dt = time.monotonic() - t_start
    logger.info("Total: %.1fs", dt)
    db.close()


if __name__ == "__main__":
    main()
