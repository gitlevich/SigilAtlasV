"""Taxonomy — two orthogonal sigils loaded from separate YAML files.

Semantic: what is depicted (time-like direction).
Visual: how it appears (space-like direction).

Each is a tree of CLIP text poles. Reuses OntologyNode for the tree structure.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from sigil_atlas.ontology import OntologyNode, _parse_node

logger = logging.getLogger(__name__)

TAXONOMY_DIR = Path(__file__).parent
TAXONOMY_FILES = [
    TAXONOMY_DIR / "taxonomy_semantic.yaml",
    TAXONOMY_DIR / "taxonomy_semantic_environments.yaml",
    TAXONOMY_DIR / "taxonomy_semantic_animals.yaml",
    TAXONOMY_DIR / "taxonomy_semantic_insects.yaml",
    TAXONOMY_DIR / "taxonomy_semantic_plants.yaml",
    TAXONOMY_DIR / "taxonomy_semantic_clothes.yaml",
    TAXONOMY_DIR / "taxonomy_visual_arts.yaml",
    TAXONOMY_DIR / "taxonomy_visual_artists.yaml",
    TAXONOMY_DIR / "taxonomy_cinematic.yaml",
    TAXONOMY_DIR / "taxonomy_photographic.yaml",
    TAXONOMY_DIR / "taxonomy_composition.yaml",
]

_TAXONOMY: dict[str, OntologyNode] | None = None


def load_taxonomy() -> dict[str, OntologyNode]:
    """Load all taxonomy sigils. Returns {sigil_name: root} for each file."""
    roots = {}
    for path in TAXONOMY_FILES:
        with open(path) as f:
            data = yaml.safe_load(f)
        root_key = next(iter(data))
        roots[root_key] = _parse_node(root_key, data[root_key])
    return roots


def get_taxonomy() -> dict[str, OntologyNode]:
    global _TAXONOMY
    if _TAXONOMY is None:
        _TAXONOMY = load_taxonomy()
    return _TAXONOMY


def vocabulary() -> dict[str, list[str]]:
    """Return all node names per sigil, for frontend autocomplete.

    Returns {"semantic": ["person", "portrait", ...], "visual": ["light", "bright", ...]}.
    """
    taxonomy = get_taxonomy()
    result = {}
    for sigil_name, root in taxonomy.items():
        names = []
        for node in root.walk():
            if node.name != sigil_name:
                names.append(node.name)
        result[sigil_name] = names
    return result


def vocabulary_tree() -> dict[str, list[dict]]:
    """Return the full tree structure per sigil for rich UI."""
    taxonomy = get_taxonomy()
    result = {}
    for sigil_name, root in taxonomy.items():
        result[sigil_name] = [_node_to_dict(c) for c in root.children]
    return result


def _node_to_dict(node: OntologyNode) -> dict:
    d = {"name": node.name, "prompt": node.prompt}
    if node.children:
        d["children"] = [_node_to_dict(c) for c in node.children]
    return d
