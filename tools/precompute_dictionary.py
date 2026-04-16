"""Pre-compute CLIP embeddings for the English dictionary.

Encodes words from /usr/share/dict/words via CLIP ViT-B/32 and saves
as a .npz file that ships with the app. One-time cost.

Output: tools/dictionary_clip_b32.npz containing:
  - words: array of word strings
  - vectors: (N, 512) float32 matrix of L2-normalized CLIP embeddings

Usage:
    PYTHONPATH=python python tools/precompute_dictionary.py \
        --max-words 30000 \
        --output tools/dictionary_clip_b32.npz
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

DICT_PATH = Path("/usr/share/dict/words")


def load_dictionary(max_words: int | None = None) -> list[str]:
    """Load and filter English dictionary words."""
    raw = DICT_PATH.read_text().splitlines()
    logger.info("Raw dictionary: %d entries", len(raw))

    # Filter: lowercase, alphabetic, 3-15 chars, no duplicates
    seen = set()
    words = []
    for w in raw:
        w_lower = w.strip().lower()
        if (w_lower.isalpha()
                and 3 <= len(w_lower) <= 15
                and w_lower not in seen):
            seen.add(w_lower)
            words.append(w_lower)

    logger.info("After filtering: %d unique words", len(words))

    if max_words and len(words) > max_words:
        # Deterministic subsample — evenly spaced through alphabetical order
        step = len(words) / max_words
        indices = [int(i * step) for i in range(max_words)]
        words = [words[i] for i in indices]
        logger.info("Subsampled to %d words", len(words))

    return words


def batch_encode_clip(
    words: list[str],
    batch_size: int = 256,
) -> np.ndarray:
    """Batch-encode words via CLIP ViT-B/32. Returns (N, 512) L2-normalized."""
    import open_clip

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    logger.info("Loading CLIP ViT-B-32...")
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    n = len(words)
    dim = 512
    vectors = np.empty((n, dim), dtype=np.float32)

    t0 = time.monotonic()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = words[start:end]

        with torch.no_grad():
            tokens = tokenizer(batch).to(device)
            features = model.encode_text(tokens)
            vecs = features.cpu().numpy().astype(np.float32)

        # L2 normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        vecs = vecs / norms

        vectors[start:end] = vecs

        if (end % 5000 < batch_size) or end == n:
            elapsed = time.monotonic() - t0
            rate = end / elapsed
            eta = (n - end) / rate if rate > 0 else 0
            logger.info("  Encoded %d / %d words (%.0f/s, ETA %.0fs)",
                        end, n, rate, eta)

    dt = time.monotonic() - t0
    logger.info("Encoded %d words in %.1fs (%.0f words/sec)", n, dt, n / dt)
    return vectors


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute CLIP embeddings for the English dictionary",
    )
    parser.add_argument("--max-words", type=int, default=30000,
                        help="Max words to encode (default 30000)")
    parser.add_argument("--output", type=Path,
                        default=Path("tools/dictionary_clip_b32.npz"),
                        help="Output .npz file path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    words = load_dictionary(args.max_words)
    vectors = batch_encode_clip(words)

    np.savez_compressed(
        args.output,
        words=np.array(words, dtype=object),
        vectors=vectors,
    )

    size_mb = args.output.stat().st_size / (1024 * 1024)
    logger.info("Saved %s: %d words, %.1f MB", args.output, len(words), size_mb)


if __name__ == "__main__":
    main()
