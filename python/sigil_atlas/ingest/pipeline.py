"""Pipeline — orchestrates ingest stages concurrently.

Invariants from the spec:
- streaming: images flow through stages as they become ready
- parallel: independent stages run concurrently
- resumable: each stage picks up where it left off
- cancellable: graceful stop between batches
- reports progress: per-stage, real-time
"""

import logging
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

from sigil_atlas.cancel import CancellationToken
from sigil_atlas.db import CorpusDB
from sigil_atlas.ingest.embed import CLIPEmbedder, DINOv2Embedder, run_embedding_stage
from sigil_atlas.ingest.metadata import extract_metadata_batch
from sigil_atlas.ingest.source import FolderSource
from sigil_atlas.ingest.thumbnail import generate_thumbnails_batch
from sigil_atlas.progress import ProgressReporter
from sigil_atlas.workspace import Workspace

logger = logging.getLogger(__name__)


class IngestPipeline:
    """Orchestrates the full ingest pipeline.

    Stage graph:
        scan → register
             ├─► metadata extraction  (parallel with thumbnails)
             └─► thumbnail generation
                      ├─► CLIP embedding   (parallel with DINOv2)
                      └─► DINOv2 embedding
                               └─► wrapping (characterize via ontology)
                                        └─► neighborhood building
    """

    def __init__(
        self,
        workspace: Workspace,
        source: FolderSource,
        token: CancellationToken | None = None,
        reporter: ProgressReporter | None = None,
    ) -> None:
        self.workspace = workspace
        self.source = source
        self.token = token or CancellationToken()
        self.reporter = reporter or ProgressReporter()

    def run(self) -> None:
        """Run the full ingest pipeline."""
        self.reporter.emit_event("pipeline_started", source=self.source.location)
        db = self.workspace.open_db()

        try:
            self._run_scan_and_register(db)

            if self.token.is_cancelled:
                return

            self._run_preparation_stages(db)

            if self.token.is_cancelled:
                return

            self._run_embedding_stages(db)

            if self.token.is_cancelled:
                return

            self._run_wrapping(db)

            if self.token.is_cancelled:
                return

            self._run_neighborhood_building(db)

            self.reporter.emit_event("pipeline_completed")

        except Exception:
            logger.error("Pipeline failed", exc_info=True)
            self.reporter.emit_event("pipeline_error")
            raise
        finally:
            db.close()

    def _run_scan_and_register(self, db: CorpusDB) -> None:
        """Scan source folder and register new images."""
        scan_progress = self.reporter.create_stage("scan", 0)

        files = self.source.scan()
        scan_progress.set_total(len(files))

        registered = self.source.register_images(db, files)
        scan_progress.advance(len(files))

        logger.info(
            "Scan complete: %d files found, %d newly registered, %d total in corpus",
            len(files), registered, db.image_count(),
        )

    def _run_preparation_stages(self, db: CorpusDB) -> None:
        """Run metadata extraction and thumbnail generation in parallel."""
        needs_metadata = db.fetch_images_without_metadata()
        needs_thumbnails = db.fetch_images_without_thumbnails()

        metadata_progress = self.reporter.create_stage("metadata", len(needs_metadata))
        thumbnail_progress = self.reporter.create_stage("thumbnails", len(needs_thumbnails))

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="prep") as pool:
            futures: list[Future] = []

            if needs_metadata:
                futures.append(pool.submit(
                    extract_metadata_batch,
                    db, needs_metadata, metadata_progress, self.token,
                ))

            if needs_thumbnails:
                futures.append(pool.submit(
                    generate_thumbnails_batch,
                    db, needs_thumbnails, self.workspace.thumbnails_dir,
                    thumbnail_progress, self.token,
                ))

            # Wait for both to complete
            for f in futures:
                f.result()

    def _run_embedding_stages(self, db: CorpusDB) -> None:
        """Run CLIP and DINOv2 embedding in parallel."""
        clip_embedder = CLIPEmbedder()
        dino_embedder = DINOv2Embedder()

        # Count unembedded for progress
        clip_unembedded = len(db.fetch_unembedded_image_ids(CLIPEmbedder.MODEL_ID))
        dino_unembedded = len(db.fetch_unembedded_image_ids(DINOv2Embedder.MODEL_ID))

        clip_progress = self.reporter.create_stage("clip", clip_unembedded)
        dino_progress = self.reporter.create_stage("dinov2", dino_unembedded)

        if clip_unembedded == 0 and dino_unembedded == 0:
            logger.info("All images already embedded with both models")
            return

        # Run both models concurrently.
        # Each loads its own model, processes independently.
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="embed") as pool:
            futures: list[Future] = []

            if clip_unembedded > 0:
                futures.append(pool.submit(
                    run_embedding_stage,
                    db, self.workspace.thumbnails_dir,
                    clip_embedder, clip_progress, self.token,
                ))

            if dino_unembedded > 0:
                futures.append(pool.submit(
                    run_embedding_stage,
                    db, self.workspace.thumbnails_dir,
                    dino_embedder, dino_progress, self.token,
                ))

            for f in futures:
                f.result()

    def _run_wrapping(self, db: CorpusDB) -> None:
        """Characterize images via CLIP zero-shot against the ontology."""
        from sigil_atlas.wrapping import run_wrapping_stage

        wrapping_progress = self.reporter.create_stage(
            "wrapping", len(db.fetch_uncharacterized_image_ids())
        )
        self.reporter.emit_event("wrapping_started")
        run_wrapping_stage(db, wrapping_progress, self.token)
        self.reporter.emit_event("wrapping_completed")

    def _run_neighborhood_building(self, db: CorpusDB) -> None:
        """Build the concept lattice from image characterizations."""
        from sigil_atlas.neighborhood import build_lattice_from_characterizations
        from sigil_atlas.wrapping import ImageCharacterization

        self.reporter.emit_event("neighborhood_building_started")

        all_chars = db.fetch_all_characterizations()
        # Convert to invariant label sets
        from sigil_atlas.wrapping import _range_to_bin
        char_labels: dict[str, frozenset[str]] = {}
        for image_id, proximities in all_chars.items():
            labels = set()
            for name, value in proximities.items():
                if isinstance(value, str):
                    labels.add(f"{name}:{value}")
                else:
                    labels.add(f"{name}:{_range_to_bin(value)}")
            char_labels[image_id] = frozenset(labels)

        lattice = build_lattice_from_characterizations(char_labels)

        self.reporter.emit_event(
            "neighborhood_building_completed",
            total_neighborhoods=len(lattice),
        )
        logger.info("Lattice built: %d neighborhoods", len(lattice))
