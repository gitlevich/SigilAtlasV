"""Ontology — progressive refinement taxonomy loaded from YAML.

Each node is a type with a CLIP prompt. Children are subtypes.
Walking the tree for an image: at each node, pick the child whose
prompt best matches. The path from root to leaf is the image's
description at increasing precision. Depth = resolution of attention.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

ONTOLOGY_PATH = Path(__file__).parent / "ontology.yaml"


@dataclass
class OntologyNode:
    """A node in the specialization tree."""

    name: str
    prompt: str
    children: list[OntologyNode] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def walk(self) -> list[OntologyNode]:
        """All nodes depth-first."""
        result = [self]
        for child in self.children:
            result.extend(child.walk())
        return result

    def leaf_count(self) -> int:
        if not self.children:
            return 1
        return sum(c.leaf_count() for c in self.children)


def load_ontology(path: Path | None = None) -> OntologyNode:
    """Load the ontology tree from YAML. Returns the root node."""
    path = path or ONTOLOGY_PATH
    with open(path) as f:
        data = yaml.safe_load(f)
    # Top-level has one key: the root node name
    root_name = next(iter(data))
    return _parse_node(root_name, data[root_name])


def _parse_node(name: str, data: dict) -> OntologyNode:
    """Parse a node and its children recursively."""
    prompt = data.get("prompt", f"a photograph of {name}")
    children = []
    if "children" in data and data["children"]:
        for child_name, child_data in data["children"].items():
            if isinstance(child_data, dict):
                children.append(_parse_node(child_name, child_data))
    return OntologyNode(name=name, prompt=prompt, children=children)


# ── Lazy singleton ──

_ONTOLOGY: OntologyNode | None = None


def get_ontology() -> OntologyNode:
    """Get the ontology tree (loaded once, cached)."""
    global _ONTOLOGY
    if _ONTOLOGY is None:
        _ONTOLOGY = load_ontology()
    return _ONTOLOGY


# ── Traversal utilities ──

def all_nodes(root: OntologyNode | None = None) -> list[OntologyNode]:
    """All nodes depth-first."""
    return (root or get_ontology()).walk()


def all_prompts(root: OntologyNode | None = None) -> list[tuple[str, str]]:
    """All (name, prompt) pairs for CLIP embedding."""
    return [(n.name, n.prompt) for n in all_nodes(root)]


def tree_depth(root: OntologyNode | None = None) -> int:
    return (root or get_ontology()).depth()


def leaf_count(root: OntologyNode | None = None) -> int:
    return (root or get_ontology()).leaf_count()


def max_branching_factor(root: OntologyNode | None = None) -> int:
    nodes = all_nodes(root)
    return max((len(n.children) for n in nodes if n.children), default=0)
