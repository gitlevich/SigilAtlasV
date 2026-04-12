"""Render a slice of the neighborhood lattice as a PNG.

Black borders, white cells. Each neighborhood is a cell containing its children.
Border thickness decreases with nesting depth. Area proportional to member count.
"""

import argparse
import logging
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from sigil_atlas.db import CorpusDB
from sigil_atlas.neighborhood import (
    ImageNeighborhoodSigil,
    build_lattice_from_characterizations,
)

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

CANVAS_SIZE = 4096


def _squarify(weights, x, y, w, h):
    """Recursive binary split treemap layout."""
    n = len(weights)
    if n == 0:
        return []
    if n == 1:
        return [(x, y, w, h)]
    if w < 2 or h < 2:
        return [(x, y, max(w, 1), max(h, 1))] * n

    total = sum(weights)
    if total <= 0:
        weights = [1.0] * n
        total = n

    best_split = 1
    best_diff = float("inf")
    running = 0
    half = total / 2
    for i in range(n - 1):
        running += weights[i]
        diff = abs(running - half)
        if diff < best_diff:
            best_diff = diff
            best_split = i + 1

    left = weights[:best_split]
    right = weights[best_split:]
    lt = sum(left)

    if w >= h:
        lw = max(1, int(w * lt / total))
        return _squarify(left, x, y, lw, h) + _squarify(right, x + lw, y, w - lw, h)
    else:
        lh = max(1, int(h * lt / total))
        return _squarify(left, x, y, w, lh) + _squarify(right, x, y + lh, w, h - lh)


def _render_cell(
    draw: ImageDraw.ImageDraw,
    nbr: ImageNeighborhoodSigil,
    x: int, y: int, w: int, h: int,
    depth: int = 0,
    max_depth: int = 10,
) -> None:
    """Render a neighborhood as a nested rectangle."""
    if w < 2 or h < 2:
        return

    border_w = max(1, 5 - depth)
    draw.rectangle(
        [x, y, x + w - 1, y + h - 1],
        fill=(255, 255, 255),
        outline=(0, 0, 0),
        width=border_w,
    )

    if depth >= max_depth or not nbr.children:
        return

    inner_x = x + border_w
    inner_y = y + border_w
    inner_w = w - 2 * border_w
    inner_h = h - 2 * border_w
    if inner_w < 4 or inner_h < 4:
        return

    # Sort children by member count descending for stable layout
    children = sorted(nbr.children, key=lambda c: c.member_count, reverse=True)
    weights = [max(c.member_count, 1) for c in children]
    rects = _squarify(weights, inner_x, inner_y, inner_w, inner_h)

    for child, (cx, cy, cw, ch) in zip(children, rects):
        _render_cell(draw, child, cx, cy, cw, ch, depth + 1, max_depth)


def render(
    workspace_path: Path,
    output_path: Path,
    max_depth: int = 10,
    canvas_size: int = CANVAS_SIZE,
) -> None:
    """Build lattice from DB and render as nested treemap."""
    db = CorpusDB(workspace_path / "datastore" / "corpus.db")
    db.initialize_schema()

    logger.info("Loading characterizations...")
    all_chars = db.fetch_all_characterizations()
    char_labels = {}
    for image_id, proximities in all_chars.items():
        char_labels[image_id] = frozenset(proximities.keys())

    logger.info("Building lattice from %d images...", len(char_labels))
    lattice = build_lattice_from_characterizations(char_labels)
    root = lattice.get(frozenset())

    if root is None:
        logger.error("No root neighborhood")
        db.close()
        return

    logger.info(
        "Root: %d members, %d children, %d total neighborhoods",
        root.member_count, len(root.children), len(lattice),
    )

    canvas = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    _render_cell(draw, root, 0, 0, canvas_size, canvas_size, max_depth=max_depth)

    canvas.save(str(output_path), "PNG")
    logger.info("Saved to %s (%dx%d)", output_path, canvas_size, canvas_size)
    db.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("slice.png"))
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--canvas-size", type=int, default=CANVAS_SIZE)
    args = parser.parse_args()
    render(args.workspace, args.output, args.depth, args.canvas_size)


if __name__ == "__main__":
    main()
