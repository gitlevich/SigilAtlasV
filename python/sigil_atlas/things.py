"""Things — named sigils on the torus surface.

A Thing is a taxonomy term that manifests as a neighborhood.
Its boundary is the similarity score threshold: images above it
satisfy the thing's invariant, images below don't.

A Contrast is two sibling things — opposite poles under the same parent.
The contrast slider places images on a spectrum between the two poles.
"""

import logging
import struct
from dataclasses import dataclass, field

import numpy as np

from sigil_atlas.db import CorpusDB
from sigil_atlas.embedding_provider import EmbeddingProvider
from sigil_atlas.layout import _fetch_image_dimensions, _greedy_pack_strips, Strip, StripLayout
from sigil_atlas.ontology import OntologyNode
from sigil_atlas.taxonomy import get_taxonomy

logger = logging.getLogger(__name__)


@dataclass
class Thing:
    """A named sigil with its matching images and scores."""
    term: str
    prompt: str
    image_ids: list[str]
    scores: list[float]


@dataclass
class ThingNeighborhood:
    """A Thing laid out as a neighborhood rectangle."""
    term: str
    prompt: str
    x: float
    y: float
    width: float
    height: float
    strips: list[Strip]
    image_count: int


@dataclass
class ThingsLayout:
    """Multiple thing-neighborhoods placed on a surface."""
    neighborhoods: list[ThingNeighborhood]
    torus_width: float
    torus_height: float
    strip_height: float


# ── Cached text embeddings ──

_text_cache: dict[str, np.ndarray] = {}
_clip_model = None
_clip_tokenizer = None


def _get_clip():
    global _clip_model, _clip_tokenizer
    if _clip_model is None:
        import open_clip
        import torch
        device = _select_device()
        model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        _clip_model = model.to(device).eval()
        _clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")
    return _clip_model, _clip_tokenizer


def _encode_prompt(prompt: str) -> np.ndarray:
    """Encode a text prompt, with caching."""
    if prompt in _text_cache:
        return _text_cache[prompt]
    import torch
    model, tokenizer = _get_clip()
    device = next(model.parameters()).device
    with torch.no_grad():
        tokens = tokenizer([prompt]).to(device)
        features = model.encode_text(tokens)
        vec = features.cpu().numpy().astype(np.float32)[0]
        vec = vec / max(np.linalg.norm(vec), 1e-8)
    _text_cache[prompt] = vec
    return vec


def _select_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Taxonomy navigation ──

def _find_node(name: str) -> OntologyNode | None:
    """Find a node by name across all taxonomy trees."""
    taxonomy = get_taxonomy()
    for root in taxonomy.values():
        for node in root.walk():
            if node.name == name:
                return node
    return None


def _find_node_by_path(path: str) -> OntologyNode | None:
    """Find a node by its full path (e.g. 'semantic/built/infrastructure')."""
    parts = path.strip("/").split("/")
    taxonomy = get_taxonomy()
    if not parts:
        return None
    root = taxonomy.get(parts[0])
    if root is None:
        return None
    node = root
    for part in parts[1:]:
        found = None
        for child in node.children:
            if child.name == part:
                found = child
                break
        if found is None:
            return None
        node = found
    return node


def _find_parent(target_name: str) -> OntologyNode | None:
    """Find the parent node of a given node name."""
    taxonomy = get_taxonomy()
    for root in taxonomy.values():
        result = _find_parent_recursive(root, target_name)
        if result is not None:
            return result
    return None


def _find_parent_recursive(node: OntologyNode, target_name: str) -> OntologyNode | None:
    for child in node.children:
        if child.name == target_name:
            return node
        result = _find_parent_recursive(child, target_name)
        if result is not None:
            return result
    return None


def siblings(term: str) -> list[dict]:
    """Return sibling terms for contrast autocomplete.

    Siblings share the same parent in the taxonomy tree.
    The invariant: a contrast's two poles must be siblings.
    """
    parent = _find_parent(term)
    if parent is None:
        return []
    return [
        {"name": child.name, "prompt": child.prompt}
        for child in parent.children
        if child.name != term
    ]


# ── Scoring ──

def score_images(
    provider: EmbeddingProvider,
    image_ids: list[str],
    prompt: str,
    model: str = "clip-vit-l-14",
) -> np.ndarray:
    """Score all images against a text prompt. Returns (N,) similarity array."""
    if not image_ids:
        return np.array([], dtype=np.float32)
    matrix = provider.fetch_matrix(image_ids, model)
    # Normalize image embeddings
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-8)
    text_vec = _encode_prompt(prompt)
    return matrix @ text_vec


def compute_thing(
    provider: EmbeddingProvider,
    db: CorpusDB,
    image_ids: list[str],
    term: str,
    model: str = "clip-vit-l-14",
    threshold: float | None = None,
) -> Thing:
    """Score all images against a term, filter by threshold."""
    node = _find_node(term)
    if node is None:
        raise ValueError(f"Unknown taxonomy term: {term}")

    scores = score_images(provider, image_ids, node.prompt, model)

    if threshold is not None:
        mask = scores >= threshold
        filtered_ids = [iid for iid, keep in zip(image_ids, mask) if keep]
        filtered_scores = scores[mask].tolist()
    else:
        # Return all, sorted by score descending
        order = np.argsort(-scores)
        filtered_ids = [image_ids[i] for i in order]
        filtered_scores = scores[order].tolist()

    return Thing(
        term=term,
        prompt=node.prompt,
        image_ids=filtered_ids,
        scores=filtered_scores,
    )


# ── Layout ──

def _select_by_threshold(
    scores: np.ndarray,
    image_ids: list[str],
    min_count: int = 20,
    max_count: int = 300,
    sigma: float = 2.0,
) -> tuple[list[str], list[float]]:
    """Select images above mean + sigma*std, with min/max bounds.

    CLIP scores for any single prompt have narrow spread.
    Threshold-based selection picks the statistically distinctive
    images rather than an arbitrary top-k.
    """
    mean = scores.mean()
    std = scores.std()
    threshold = mean + sigma * std

    order = np.argsort(-scores)
    selected_ids = []
    selected_scores = []

    for idx in order:
        if len(selected_ids) >= max_count:
            break
        if scores[idx] >= threshold or len(selected_ids) < min_count:
            selected_ids.append(image_ids[idx])
            selected_scores.append(float(scores[idx]))

    return selected_ids, selected_scores


def compute_things_layout(
    provider: EmbeddingProvider,
    db: CorpusDB,
    image_ids: list[str],
    terms: list[str],
    model: str = "clip-vit-l-14",
    strip_height: float = 100.0,
    top_k: int = 200,
) -> ThingsLayout:
    """Lay out multiple things as neighborhood rectangles.

    Each thing gets images that are statistically above-average
    for its prompt, packed into a near-square rectangle.
    """
    if not terms or not image_ids:
        return ThingsLayout([], 0, 0, strip_height)

    dims = _fetch_image_dimensions(db, image_ids)
    natural_widths = {iid: strip_height * (dims[iid][0] / dims[iid][1]) for iid in image_ids}
    thumbnails = {iid: dims[iid][2] for iid in image_ids}

    neighborhoods: list[ThingNeighborhood] = []

    for term in terms:
        node = _find_node(term)
        if node is None:
            continue
        scores = score_images(provider, image_ids, node.prompt, model)
        selected, selected_scores = _select_by_threshold(scores, image_ids)
        logger.info(
            "Thing '%s': %d images selected (threshold at mean+2std, scores %.3f-%.3f)",
            term, len(selected),
            selected_scores[-1] if selected_scores else 0,
            selected_scores[0] if selected_scores else 0,
        )
        if not selected:
            continue

        # Pack into near-square
        import math
        total_w = sum(natural_widths.get(iid, strip_height) for iid in selected)
        area = total_w * strip_height
        target_side = max(math.sqrt(area), strip_height * 2)

        strips = _greedy_pack_strips(
            selected, natural_widths, thumbnails, strip_height, target_side,
        )
        rect_h = sum(s.height for s in strips) if strips else strip_height

        neighborhoods.append(ThingNeighborhood(
            term=term,
            prompt=node.prompt,
            x=0, y=0,  # placed below
            width=target_side,
            height=rect_h,
            strips=strips,
            image_count=len(selected),
        ))

    # Place neighborhoods in a row with gaps
    gap = strip_height * 0.5
    x = 0.0
    for nb in neighborhoods:
        nb.x = x
        nb.y = 0
        x += nb.width + gap

    torus_width = x - gap if neighborhoods else 0
    torus_height = max((nb.height for nb in neighborhoods), default=0)

    logger.info(
        "Things layout: %d neighborhoods, surface=%.0fx%.0f",
        len(neighborhoods), torus_width, torus_height,
    )

    return ThingsLayout(neighborhoods, torus_width, torus_height, strip_height)
