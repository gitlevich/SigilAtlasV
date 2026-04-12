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

    args = parser.parse_args()

    if args.command == "ingest":
        _run_ingest(args.workspace, args.source)


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


if __name__ == "__main__":
    main()
