"""Tests for the ontology specialization tree."""

from sigil_atlas.ontology import (
    all_nodes,
    all_prompts,
    get_ontology,
    leaf_count,
    max_branching_factor,
    tree_depth,
)


def test_root_is_photograph():
    root = get_ontology()
    assert root.name == "photograph"


def test_root_has_children():
    root = get_ontology()
    assert len(root.children) >= 2
    child_names = {c.name for c in root.children}
    assert "subject" in child_names
    assert "environment" in child_names


def test_all_nodes_non_empty():
    nodes = all_nodes()
    assert len(nodes) >= 20


def test_all_prompts_non_empty():
    prompts = all_prompts()
    for name, prompt in prompts:
        assert name, "Empty node name"
        assert prompt, f"Empty prompt for {name}"


def test_all_names_unique():
    prompts = all_prompts()
    names = [name for name, _ in prompts]
    dupes = [n for n in names if names.count(n) > 1]
    assert len(names) == len(set(names)), f"Duplicate names: {set(dupes)}"


def test_tree_depth_reasonable():
    depth = tree_depth()
    assert 3 <= depth <= 10, f"Depth {depth} outside range"


def test_branching_factor_small():
    bf = max_branching_factor()
    assert bf <= 6, f"Branching factor {bf} too large"


def test_leaf_count():
    count = leaf_count()
    assert count >= 10
    assert count <= 500


def test_specialization_path_exists():
    """Can find a path from root to a specific leaf."""
    root = get_ontology()
    # Find bird
    def find(node, target):
        if node.name == target:
            return [node.name]
        for child in node.children:
            path = find(child, target)
            if path:
                return [node.name] + path
        return None

    path = find(root, "bird")
    assert path is not None, "Cannot find 'bird' in tree"
    assert path[0] == "photograph"
    assert path[-1] == "bird"
