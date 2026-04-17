"""CLI entry point for the SigilAtlas Python sidecar."""

import argparse
import logging
import signal
import sys
from pathlib import Path

from sigil_atlas.cancel import CancellationToken
from sigil_atlas.ingest.pipeline import IngestPipeline
from sigil_atlas.ingest.source import FolderSource
from sigil_atlas.progress import ProgressReporter
from sigil_atlas.workspace import Workspace

# Logging goes to stderr so stdout is reserved for JSON progress lines
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="SigilAtlas Python sidecar")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_parser = sub.add_parser("ingest", help="Ingest images from a source folder")
    ingest_parser.add_argument(
        "--workspace", type=Path, required=True, help="Workspace directory"
    )
    ingest_parser.add_argument(
        "--source", type=Path, required=True, help="Source folder containing images"
    )

    backfill_parser = sub.add_parser("backfill-hashes", help="Compute content hashes for images missing them")
    backfill_parser.add_argument(
        "--workspace", type=Path, required=True, help="Workspace directory"
    )

    embed_parser = sub.add_parser("embed-missing", help="Embed images missing from any model (detached from the app)")
    embed_parser.add_argument(
        "--workspace", type=Path, required=True, help="Workspace directory"
    )
    embed_parser.add_argument(
        "--model", choices=["clip-vit-b-32", "clip-vit-l-14", "dinov2-vitb14"],
        help="Run just this model; omit for all models with gaps",
    )

    args = parser.parse_args()

    if args.command == "ingest":
        _run_ingest(args.workspace, args.source)
    elif args.command == "backfill-hashes":
        _run_backfill_hashes(args.workspace)
    elif args.command == "embed-missing":
        _run_embed_missing(args.workspace, args.model)


def _run_ingest(workspace_path: Path, source_path: Path) -> None:
    token = CancellationToken()

    # Wire up SIGINT/SIGTERM for graceful cancellation
    def _handle_signal(signum, frame):
        logger.info("Received signal %d, cancelling pipeline...", signum)
        token.cancel()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    workspace = Workspace(workspace_path)
    workspace.initialize()

    source = FolderSource(source_path)
    reporter = ProgressReporter()

    pipeline = IngestPipeline(
        workspace=workspace,
        source=source,
        token=token,
        reporter=reporter,
    )

    try:
        pipeline.run()
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted")
    except Exception:
        logger.error("Pipeline failed", exc_info=True)
        sys.exit(1)


def _run_backfill_hashes(workspace_path: Path) -> None:
    from sigil_atlas.ingest.source import content_hash

    workspace = Workspace(workspace_path)
    db = workspace.open_db()

    rows = db._conn.execute(
        "SELECT id, source_path FROM images WHERE content_hash IS NULL"
    ).fetchall()

    if not rows:
        logger.info("All images already have content hashes")
        db.close()
        return

    logger.info("Backfilling hashes for %d images", len(rows))
    updated = 0
    for row in rows:
        image_id, source_path = row[0], row[1]
        path = Path(source_path)
        if not path.is_file():
            logger.warning("Source file missing, skipping: %s", source_path)
            continue
        h = content_hash(path)
        db._conn.execute(
            "UPDATE images SET content_hash = ? WHERE id = ?", (h, image_id)
        )
        updated += 1
        if updated % 500 == 0:
            db._conn.commit()
            logger.info("  %d / %d", updated, len(rows))

    db._conn.commit()
    logger.info("Backfilled %d content hashes", updated)
    db.close()


def _run_embed_missing(workspace_path: Path, only_model: str | None) -> None:
    """Detached-from-app embedding. Streams unembedded images through each
    missing model and writes results to the workspace DB.
    """
    from sigil_atlas.ingest.embed import (
        CLIPEmbedder, CLIPLargeEmbedder, DINOv2Embedder, run_embedding_stage,
    )
    from sigil_atlas.progress import ProgressReporter

    token = CancellationToken()

    def _handle_signal(signum, frame):
        logger.info("Received signal %d, cancelling embedding...", signum)
        token.cancel()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    workspace = Workspace(workspace_path)
    workspace.initialize()
    db = workspace.open_db()
    reporter = ProgressReporter()

    embedders: list = []
    if only_model is None or only_model == CLIPEmbedder.MODEL_ID:
        embedders.append(CLIPEmbedder())
    if only_model is None or only_model == CLIPLargeEmbedder.MODEL_ID:
        embedders.append(CLIPLargeEmbedder())
    if only_model is None or only_model == DINOv2Embedder.MODEL_ID:
        embedders.append(DINOv2Embedder())

    for embedder in embedders:
        if token.is_cancelled:
            break
        unembedded = db.fetch_unembedded_image_ids(embedder.MODEL_ID)
        if not unembedded:
            logger.info("%s: already complete", embedder.MODEL_ID)
            continue
        progress = reporter.create_stage(embedder.MODEL_ID, len(unembedded))
        logger.info("%s: embedding %d missing images", embedder.MODEL_ID, len(unembedded))
        try:
            run_embedding_stage(db, workspace.thumbnails_dir, embedder, progress, token)
        except Exception:
            logger.error("Failed to embed %s", embedder.MODEL_ID, exc_info=True)
    db.close()
    logger.info("Embed-missing complete")


if __name__ == "__main__":
    main()
