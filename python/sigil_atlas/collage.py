"""Collage — a saved view, materialised as a sub-@sigil per the spec.

Per spec (root `affordance-create-sigil`): a sigil is "a `.sigil` directory:
structure (nested sub-sigils), narrative (language), and any assets attached."
A saved collage takes that exact form so it can be browsed in Finder, shared
as a self-contained archive, and later refined (per-image sub-sigils, model-
generated narrative, etc.) without changing the storage shape.

Layout of a saved collage:

    <auto-name>.sigil/
      language.md       — narrative + frontmatter metadata (human-readable)
      collage.json      — SigilML expression + POV + arrangement (machine state)
      screenshot.png    — full-resolution canvas snapshot at save time
      preview.png       — small (256px) version of the screenshot
      Icon\r            — invisible; carries the preview as a Finder folder icon
      images/           — (optional, future) one sub-sigil per image in slice

Folder name is auto-generated and slug-safe:
  1. Active Thing pills, if any, joined with hyphens.
  2. Else: the slice's centroid scored against a 30k-word CLIP dictionary,
     top three meaningful words.
  3. Else: a timestamp.
Always made unique by appending -2, -3, ... on collision.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from sigil_atlas.embedding_provider import EmbeddingProvider

logger = logging.getLogger(__name__)


# Module-level cache; ~60 MB, loaded once on first use.
_DICTIONARY: tuple[np.ndarray, list[str]] | None = None


def _dictionary_path() -> Path:
    return Path(__file__).resolve().parents[2] / "tools" / "dictionary_clip_b32.npz"


def _load_dictionary() -> tuple[np.ndarray, list[str]] | None:
    """Return (vectors, words) where vectors is (N, 512) unit-normalized."""
    global _DICTIONARY
    if _DICTIONARY is not None:
        return _DICTIONARY
    path = _dictionary_path()
    if not path.exists():
        logger.warning("CLIP dictionary missing at %s — semantic naming disabled", path)
        return None
    data = np.load(path, allow_pickle=True)
    words = [str(w) for w in data["words"]]
    vectors = np.asarray(data["vectors"], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-8)
    _DICTIONARY = (vectors, words)
    logger.info("Loaded CLIP dictionary: %d words, %d-dim", len(words), vectors.shape[1])
    return _DICTIONARY


# Words that score high but say nothing useful about the picture.
_STOP = {
    "photo", "photograph", "photographic", "photographs", "photography",
    "image", "images", "picture", "pictures", "pic", "pics",
    "shot", "snap", "snapshot", "frame", "view", "scene", "scenes",
    "of", "the", "and", "or", "in", "on", "with", "for", "by", "to",
    "is", "are", "be", "this", "that", "those", "these", "an", "a",
    "from", "into", "very", "much", "more", "most", "also",
    "jpg", "jpeg", "png", "raw", "file", "files",
    "color", "colors", "colour", "colours",
}


def _is_useful(word: str) -> bool:
    if len(word) < 3:
        return False
    if word.lower() in _STOP:
        return False
    if not word.isalpha():
        return False
    return True


def suggest_semantic_words(
    provider: EmbeddingProvider,
    image_ids: list[str],
    top_k: int = 3,
) -> list[str]:
    """Score the slice's centroid against a 30k-word CLIP dictionary,
    return the top_k words after filtering generic ones.

    Uses CLIP B-32 unconditionally so naming works regardless of which
    embedding model the user has selected for similarity. Returns []
    if the dictionary is missing or the slice has no B-32 embeddings.
    """
    if not image_ids:
        return []
    dict_data = _load_dictionary()
    if dict_data is None:
        return []
    vectors, words = dict_data

    try:
        matrix = provider.fetch_matrix(image_ids, "clip-vit-b-32")
    except (ValueError, KeyError) as e:
        logger.info("Semantic naming skipped — no B-32 coverage: %s", e)
        return []
    if matrix.size == 0:
        return []

    # Normalize each image vector, then take the centroid of the unit vectors —
    # cosine-on-mean is roughly mean-of-cosines for normalised inputs.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-8)
    centroid = matrix.mean(axis=0)
    cn = float(np.linalg.norm(centroid))
    if cn < 1e-8:
        return []
    centroid = centroid / cn

    scores = vectors @ centroid  # (30000,)
    order = np.argsort(-scores)

    chosen: list[str] = []
    seen_stems: set[str] = set()
    for idx in order:
        word = words[int(idx)].lower()
        if not _is_useful(word):
            continue
        # De-stem crudely: skip simple plurals if the singular is already in.
        stem = word.rstrip("s") if len(word) > 4 else word
        if stem in seen_stems:
            continue
        seen_stems.add(stem)
        chosen.append(word)
        if len(chosen) >= top_k:
            break
    return chosen


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """Lowercase, hyphenated, no spaces, no punctuation."""
    return _SLUG_RE.sub("-", text.lower()).strip("-") or "untitled"


def derive_folder_name(
    *,
    user_hint: str | None,
    pill_names: list[str],
    image_ids: list[str],
    provider: EmbeddingProvider,
) -> str:
    """Three sources, in order of preference:
    1. User-supplied hint (slugified).
    2. Active Thing pills joined with hyphens.
    3. CLIP-centroid top-3 dictionary words.
    Falls back to a timestamp if all three are empty.
    """
    if user_hint and user_hint.strip():
        return slugify(user_hint)
    if pill_names:
        return slugify("-".join(pill_names))
    semantic = suggest_semantic_words(provider, image_ids, top_k=3)
    if semantic:
        return slugify("-".join(semantic))
    return time.strftime("collage-%Y-%m-%d-%H%M%S")


def unique_sigil_folder(parent: Path, base_name: str) -> Path:
    """Return parent/<base_name>.sigil with -2, -3, ... appended on collision."""
    suffix = ".sigil"
    candidate = parent / f"{base_name}{suffix}"
    if not candidate.exists():
        return candidate
    n = 2
    while True:
        candidate = parent / f"{base_name}-{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def _summarise_filter(expression) -> str:
    """One-line human description of a SigilML expression, for the narrative."""
    if expression is None:
        return "unconstrained — entire @corpus"

    def walk(node) -> list[str]:
        if not isinstance(node, dict):
            return []
        t = node.get("type")
        if t == "thing":
            return [str(node.get("name"))]
        if t == "target_image":
            return ["a target image"]
        if t == "contrast":
            return [f"{node.get('pole_b')} \u2194 {node.get('pole_a')}"]
        if t == "range":
            return [f"{node.get('dimension')} \u2208 [{node.get('min')}, {node.get('max')}]"]
        if t in ("and", "or"):
            children = [w for c in node.get("children", []) for w in walk(c)]
            joiner = " AND " if t == "and" else " OR "
            return [joiner.join(children)] if children else []
        if t == "not":
            inner = walk(node.get("child"))
            return [f"NOT ({inner[0]})"] if inner else []
        return []

    parts = walk(expression)
    return parts[0] if parts else "unconstrained — entire @corpus"


def _render_language_md(
    *,
    name: str,
    expression,
    pov: dict,
    mode: str,
    model: str,
    relevance: float,
    feathering: float,
    image_count: int,
) -> str:
    iso = time.strftime("%Y-%m-%dT%H:%M:%S")
    summary = _summarise_filter(expression)
    return (
        f"---\n"
        f"status: saved\n"
        f"saved_at: {iso}\n"
        f"mode: {mode}\n"
        f"model: {model}\n"
        f"relevance: {relevance}\n"
        f"feathering: {feathering}\n"
        f"image_count: {image_count}\n"
        f"---\n\n"
        f"# {name}\n\n"
        f"A saved view of the @corpus. The @RelevanceFilter compresses the "
        f"@corpus to **{image_count}** @images.\n\n"
        f"**Filter:** {summary}\n\n"
        f"**Camera:** x={pov.get('x'):.1f}, y={pov.get('y'):.1f}, "
        f"z={pov.get('z'):.1f}\n\n"
        f"---\n\n"
        f"_Narrative pending — a future model will describe what these "
        f"@images say together._\n"
    )


def write_collage(
    folder: Path,
    *,
    name: str,
    expression,  # JSON-serialisable Expression | None
    pov: dict,
    mode: str,
    model: str,
    relevance: float,
    feathering: float,
    cell_size: float,
    image_ids: list[str],
    screenshot_base64: str | None,
    field_expansion: str = "echo",
    arrangement: str = "rings",
    time_direction: str = "capture_date",
    strip_height: float = 100.0,
    torus_width: float = 0.0,
    torus_height: float = 0.0,
) -> None:
    """Materialise a collage as a `.sigil` directory with the spec's shape:

      language.md    — narrative + frontmatter (human)
      collage.json   — SigilML expression + POV + arrangement (machine)
      screenshot.png — full-resolution canvas snapshot
    """
    folder.mkdir(parents=True, exist_ok=False)

    manifest = {
        "version": 2,
        "name": name,
        "saved_at": time.time(),
        "expression": expression,
        "pov": pov,
        "mode": mode,
        "model": model,
        "relevance": relevance,
        "feathering": feathering,
        "cell_size": cell_size,
        "field_expansion": field_expansion,
        "arrangement": arrangement,
        "time_direction": time_direction,
        "strip_height": strip_height,
        "torus_width": torus_width,
        "torus_height": torus_height,
        "image_ids": image_ids,
    }
    (folder / "collage.json").write_text(json.dumps(manifest, indent=2))

    (folder / "language.md").write_text(_render_language_md(
        name=name, expression=expression, pov=pov,
        mode=mode, model=model, relevance=relevance, feathering=feathering,
        image_count=len(image_ids),
    ))

    if screenshot_base64:
        try:
            png = base64.b64decode(screenshot_base64)
            screenshot_path = folder / "screenshot.png"
            screenshot_path.write_bytes(png)
            # Derive a small browse-friendly preview from the full-res shot.
            preview_path = folder / "preview.png"
            _write_preview(png, preview_path, max_side=256)
            # Decorate the folder so Finder shows the preview as the icon.
            _set_folder_icon(folder, preview_path)
            # Per-folder OpenWith hint: prefer Sigil Atlas without making it
            # the system default for `.sigil`.
            _claim_folder_for_app(folder)
        except Exception as e:
            logger.warning("Failed to write screenshot/preview/icon: %s", e)


_PREVIEW_FILTER = Image.Resampling.LANCZOS


def _write_preview(source_png_bytes: bytes, out_path: Path, max_side: int) -> None:
    """Downscale the screenshot to max_side and write as PNG."""
    with Image.open(io.BytesIO(source_png_bytes)) as img:
        img.thumbnail((max_side, max_side), _PREVIEW_FILTER)
        img.save(out_path, "PNG", optimize=True)


_OPEN_WITH_BUNDLE_ID = "com.sigilatlas.desktop"


def _claim_folder_for_app(folder: Path, bundle_id: str = _OPEN_WITH_BUNDLE_ID) -> None:
    """Set `com.apple.LaunchServices.OpenWith` xattr so Finder prefers our app
    for *this* folder, without changing the system default for the .sigil type.
    Other registered openers stay available via right-click \u2192 Open With.

    Best-effort: failure logs and returns. Decoration only; the app still
    opens correctly via Cmd+O even when the xattr is missing.
    """
    import plistlib
    payload = plistlib.dumps(
        {"version": 1, "bundleidentifier": bundle_id},
        fmt=plistlib.FMT_BINARY,
    )
    try:
        subprocess.run(
            [
                "xattr", "-wx",
                "com.apple.LaunchServices.OpenWith",
                payload.hex(),
                str(folder),
            ],
            check=True, capture_output=True, timeout=5,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.info("OpenWith xattr skipped: %s", e)


def _set_folder_icon(folder: Path, source_png: Path) -> None:
    """Decorate `folder` so Finder shows `source_png` as its icon.

    Uses NSWorkspace.setIcon:forFile: via JavaScript-for-Automation —
    osascript ships with every macOS, no extra dependencies. Cocoa builds
    the resource fork on `Icon\\r` and sets the folder's FinderInfo flag
    in one call, which is what `fileicon` and similar tools do under the
    hood. Best-effort: failure logs a warning and returns.
    """
    # Escape backslashes and double quotes for embedding into the JS string.
    folder_s = str(folder).replace("\\", "\\\\").replace('"', '\\"')
    png_s = str(source_png).replace("\\", "\\\\").replace('"', '\\"')
    script = (
        'ObjC.import("AppKit");\n'
        'const ws = $.NSWorkspace.sharedWorkspace;\n'
        f'const img = $.NSImage.alloc.initWithContentsOfFile("{png_s}");\n'
        f'ws.setIconForFileOptions(img, "{folder_s}", 0);\n'
    )
    try:
        subprocess.run(
            ["osascript", "-l", "JavaScript", "-e", script],
            check=True, capture_output=True, timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.info("Folder icon decoration skipped: %s", e)


def read_collage(folder: Path) -> dict:
    """Read collage.json from a `.sigil` folder. Raises FileNotFoundError."""
    manifest_path = folder / "collage.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No collage.json in {folder}")
    return json.loads(manifest_path.read_text())
