"""Tests for neighborhood building — FCA lattice construction.

Pure logic tests, no CLIP model needed. Uses synthetic invariant sets
structured as path prefixes (matching the tree-walk ontology).
"""

from sigil_atlas.neighborhood import (
    ImageNeighborhoodSigil,
    ImageSigil,
    BitmapIndex,
    build_inverted_index,
    build_lattice,
    build_lattice_from_characterizations,
)


def test_inverted_index():
    sigils = [
        ImageSigil("img-1", frozenset({"dark", "dark/warm", "dark/warm/vivid"})),
        ImageSigil("img-2", frozenset({"dark", "dark/warm", "dark/warm/muted"})),
        ImageSigil("img-3", frozenset({"bright", "bright/cool"})),
    ]
    index = build_inverted_index(sigils)
    assert index["dark"] == {"img-1", "img-2"}
    assert index["dark/warm"] == {"img-1", "img-2"}
    assert index["bright"] == {"img-3"}


def test_bitmap_index():
    sigils = [
        ImageSigil("img-1", frozenset({"dark", "dark/warm"})),
        ImageSigil("img-2", frozenset({"dark", "dark/cool"})),
        ImageSigil("img-3", frozenset({"bright"})),
    ]
    bm = BitmapIndex(sigils)
    assert bm.n_images == 3

    # Intersect: dark only
    result = bm.intersect(frozenset({"dark"}))
    ids = bm.bitmap_to_ids(result)
    assert ids == frozenset({"img-1", "img-2"})

    # Intersect: dark/warm
    result = bm.intersect(frozenset({"dark", "dark/warm"}))
    ids = bm.bitmap_to_ids(result)
    assert ids == frozenset({"img-1"})


def test_single_neighborhood():
    sigils = [
        ImageSigil("img-1", frozenset({"dark", "dark/warm"})),
        ImageSigil("img-2", frozenset({"dark", "dark/warm"})),
    ]
    lattice = build_lattice(sigils)
    key = frozenset({"dark", "dark/warm"})
    assert key in lattice
    assert lattice[key].member_ids == frozenset({"img-1", "img-2"})


def test_shared_prefix_forms_neighborhood():
    """Images sharing a prefix but diverging deeper share a neighborhood."""
    sigils = [
        ImageSigil("img-1", frozenset({"dark", "dark/warm", "dark/warm/vivid"})),
        ImageSigil("img-2", frozenset({"dark", "dark/warm", "dark/warm/muted"})),
    ]
    lattice = build_lattice(sigils)

    # Both share dark and dark/warm
    assert frozenset({"dark"}) in lattice
    assert lattice[frozenset({"dark"})].member_ids == frozenset({"img-1", "img-2"})

    assert frozenset({"dark", "dark/warm"}) in lattice
    assert lattice[frozenset({"dark", "dark/warm"})].member_ids == frozenset({"img-1", "img-2"})


def test_deduplication():
    sigils = [
        ImageSigil("img-1", frozenset({"dark", "dark/warm"})),
        ImageSigil("img-2", frozenset({"dark", "dark/warm"})),
        ImageSigil("img-3", frozenset({"dark", "dark/warm"})),
    ]
    lattice = build_lattice(sigils)
    key = frozenset({"dark", "dark/warm"})
    assert key in lattice
    assert lattice[key].member_count == 3


def test_nested_neighborhoods():
    """Tighter neighborhoods are children of looser ones."""
    sigils = [
        ImageSigil("img-1", frozenset({"dark", "dark/warm", "dark/warm/vivid"})),
        ImageSigil("img-2", frozenset({"dark", "dark/warm", "dark/warm/vivid"})),
    ]
    lattice = build_lattice(sigils)

    parent_key = frozenset({"dark", "dark/warm"})
    child_key = frozenset({"dark", "dark/warm", "dark/warm/vivid"})

    assert parent_key in lattice
    assert child_key in lattice
    assert lattice[child_key] in lattice[parent_key].children
    assert lattice[parent_key] in lattice[child_key].parents


def test_root_contains_all():
    sigils = [
        ImageSigil("img-1", frozenset({"dark"})),
        ImageSigil("img-2", frozenset({"bright"})),
    ]
    lattice = build_lattice(sigils)
    root = lattice[frozenset()]
    assert root.member_ids == frozenset({"img-1", "img-2"})
    assert root.is_root


def test_scale_equals_invariant_count():
    sigils = [
        ImageSigil("img-1", frozenset({"a", "b", "c"})),
        ImageSigil("img-2", frozenset({"a", "b", "c"})),
    ]
    lattice = build_lattice(sigils)
    for key, nbr in lattice.items():
        assert nbr.scale == len(key)


def test_min_members_filtering():
    sigils = [
        ImageSigil("img-1", frozenset({"dark", "dark/warm"})),
        ImageSigil("img-2", frozenset({"bright", "bright/cool"})),
    ]
    lattice = build_lattice(sigils, min_members=2)
    for key in lattice:
        if len(key) >= 1:
            assert lattice[key].member_count >= 2 or key == frozenset()


def test_build_from_characterizations():
    chars = {
        "img-1": frozenset({"dark", "dark/warm"}),
        "img-2": frozenset({"dark", "dark/warm"}),
        "img-3": frozenset({"dark", "dark/cool"}),
    }
    lattice = build_lattice_from_characterizations(chars)
    assert frozenset({"dark"}) in lattice
    assert lattice[frozenset({"dark"})].member_count == 3


def test_parent_child_consistency():
    sigils = [
        ImageSigil("img-1", frozenset({"a", "b", "c"})),
        ImageSigil("img-2", frozenset({"a", "b", "c"})),
        ImageSigil("img-3", frozenset({"a", "b", "d"})),
    ]
    lattice = build_lattice(sigils)
    for key, nbr in lattice.items():
        for child in nbr.children:
            assert nbr in child.parents
        for parent in nbr.parents:
            assert nbr in parent.children
