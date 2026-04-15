"""Pipeline — orchestrates ingest stages concurrently.

Invariants from the spec:
- streaming: images flow through stages as they become ready
- parallel: independent stages run concurrently
- resumable: each stage picks up where it left off
- cancellable: graceful stop between batches
- reports progress: per-stage, real-time
"""

import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

from sigil_atlas.cancel import CancellationToken
from sigil_atlas.db import CorpusDB
from sigil_atlas.ingest.cluster import KMEANS_K_LEVELS, run_clustering_stage
from sigil_atlas.ingest.embed import CLIPEmbedder, DINOv2Embedder, run_embedding_stage
from sigil_atlas.ingest.metadata import extract_metadata_batch
from sigil_atlas.ingest.source import FolderSource
from sigil_atlas.ingest.thumbnail import generate_thumbnails_batch
from sigil_atlas.progress import ProgressReporter
from sigil_atlas.workspace import Workspace

logger = logging.getLogger(__name__)


class CharacterizationStrategy(ABC):
    """Strategy for wrapping images and building neighborhoods."""

    @abstractmethod
    def run(
        self,
        db: CorpusDB,
        reporter: ProgressReporter,
        token: CancellationToken,
    ) -> None:
        """Run wrapping + neighborhood building."""


class TopDownStrategy(CharacterizationStrategy):
    """Top-down taxonomy-driven characterization.

    Walks a YAML ontology tree using CLIP zero-shot classification.
    Each image descends the tree by picking the best-matching child
    at each level. The path becomes its invariant set.
    Good when you know what distinctions matter. Misses what you
    don't name.
    """

    def run(self, db: CorpusDB, reporter: ProgressReporter, token: CancellationToken) -> None:
        from sigil_atlas.wrapping import run_wrapping_stage
        from sigil_atlas.neighborhood import build_lattice_from_characterizations

        wrapping_progress = reporter.create_stage(
            "wrapping", len(db.fetch_uncharacterized_image_ids())
        )
        reporter.emit_event("wrapping_started")
        run_wrapping_stage(db, wrapping_progress, token)
        reporter.emit_event("wrapping_completed")

        if token.is_cancelled:
            return

        reporter.emit_event("neighborhood_building_started")
        all_chars = db.fetch_all_characterizations()
        char_labels: dict[str, frozenset[str]] = {}
        for image_id, proximities in all_chars.items():
            char_labels[image_id] = frozenset(proximities.keys())

        lattice = build_lattice_from_characterizations(char_labels)
        reporter.emit_event(
            "neighborhood_building_completed",
            total_neighborhoods=len(lattice),
        )
        logger.info("Lattice built: %d neighborhoods", len(lattice))


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
        strategy: CharacterizationStrategy | None = None,
    ) -> None:
        self.workspace = workspace
        self.source = source
        self.token = token or CancellationToken()
        self.reporter = reporter or ProgressReporter()
        self.strategy = strategy or TopDownStrategy()

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

            self._run_clustering_stage(db)

            if self.token.is_cancelled:
                return

            self.strategy.run(db, self.reporter, self.token)

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

    def _run_clustering_stage(self, db: CorpusDB) -> None:
        """Precompute KMeans at multiple k levels for both models."""
        from sigil_atlas.embedding_provider import SqliteEmbeddingProvider

        provider = SqliteEmbeddingProvider(db)
        for model_id in [CLIPEmbedder.MODEL_ID, DINOv2Embedder.MODEL_ID]:
            if self.token.is_cancelled:
                return
            cluster_progress = self.reporter.create_stage(
                f"cluster_{model_id}", len(KMEANS_K_LEVELS)
            )
            run_clustering_stage(db, provider, model_id, KMEANS_K_LEVELS, cluster_progress, self.token)

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

