"""HTTP sidecar server for the Tauri app.

Exposes SliceMode and NeighborhoodMode as JSON endpoints.
Spawns on a random port and prints the port to stdout for Tauri to read.
"""

import argparse
import json
import logging
import mimetypes
import socket
import sys
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from sigil_atlas.db import CorpusDB
from sigil_atlas.embedding_provider import SqliteEmbeddingProvider
from sigil_atlas.layout import compute_layout
from sigil_atlas.model_registry import get_adapter
from sigil_atlas.relevance_filter import Contrast as FilterContrast, parse as parse_filter, thing_atoms, target_image_atoms, walk as walk_filter
from sigil_atlas.slice import compute_slice
from sigil_atlas.overview import (
    generate_mid_atlas,
    generate_overview,
    mid_atlas_page_path,
    overview_paths,
)
from sigil_atlas.spacelike import Attractor, ContrastAxis, compute_spacelike, compute_wireframe_edges
from sigil_atlas.taxonomy import vocabulary, vocabulary_tree
from sigil_atlas.things import siblings, compute_things_layout
from sigil_atlas.workspace import Workspace

logger = logging.getLogger(__name__)


class IngestState:
    """Tracks the current ingest pipeline, if any."""

    def __init__(self) -> None:
        from sigil_atlas.progress import BufferedProgressReporter
        self.thread: threading.Thread | None = None
        self.token = None
        self.reporter = BufferedProgressReporter()
        self.current_source: str | None = None
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, workspace: "Workspace", source_path: str) -> str:
        from sigil_atlas.cancel import CancellationToken
        from sigil_atlas.ingest.pipeline import IngestPipeline
        from sigil_atlas.ingest.source import FolderSource
        from sigil_atlas.progress import BufferedProgressReporter

        with self._lock:
            if self.is_running:
                return "already_running"

            source = FolderSource(Path(source_path))
            self.token = CancellationToken()
            self.reporter = BufferedProgressReporter()
            self.current_source = source_path

            pipeline = IngestPipeline(
                workspace=workspace,
                source=source,
                token=self.token,
                reporter=self.reporter,
            )

            self.thread = threading.Thread(
                target=self._run_pipeline,
                args=(pipeline,),
                daemon=True,
                name="ingest-pipeline",
            )
            self.thread.start()
            return "started"

    def _run_pipeline(self, pipeline) -> None:
        try:
            pipeline.run()
        except Exception:
            logger.error("Ingest pipeline failed", exc_info=True)
        finally:
            # Invalidate cached embedding matrices so queries pick up new data
            if _state is not None:
                _state.provider.invalidate_cache()

    def pause(self) -> str:
        with self._lock:
            if not self.is_running or self.token is None:
                return "not_running"
            self.token.cancel()
            self.reporter.set_status("paused")
            return "paused"

    def progress(self) -> dict:
        return self.reporter.snapshot()


class SidecarState:
    """Shared server state."""

    def __init__(self, workspace_path: Path) -> None:
        self.workspace = Workspace(workspace_path)
        self.db = self.workspace.open_db()
        self.provider = SqliteEmbeddingProvider(self.db)
        self.ingest = IngestState()


_state: SidecarState | None = None


class RequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        logger.debug(format, *args)

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        return json.loads(body) if body else {}

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _send_file(self, file_path: Path):
        if not file_path.is_file():
            # 204 instead of 404 so the browser doesn't spam the console with
            # failed-resource errors for previews/thumbnails that haven't been
            # generated yet. The client just gets an empty response.
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            return
        mime = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok"})
        elif self.path == "/dimensions":
            self._handle_dimensions()
        elif self.path == "/models":
            self._handle_models()
        elif self.path == "/vocabulary":
            self._send_json(vocabulary())
        elif self.path == "/vocabulary/tree":
            self._send_json(vocabulary_tree())
        elif self.path.startswith("/siblings/"):
            term = self.path[len("/siblings/"):]
            self._send_json({"siblings": siblings(term)})
        elif self.path.startswith("/thumbnail/"):
            image_id = self.path[len("/thumbnail/"):]
            thumb_path = _state.workspace.thumbnails_dir / f"{image_id}.jpg"
            self._send_file(thumb_path)
        elif self.path.startswith("/preview/"):
            image_id = self.path[len("/preview/"):]
            preview_path = _state.workspace.previews_dir / f"{image_id}.jpg"
            self._send_file(preview_path)
        elif self.path.startswith("/image/info/"):
            image_id = self.path[len("/image/info/"):]
            meta = _state.db.fetch_image_metadata(image_id)
            if meta is None:
                self._send_json({"error": "unknown image"}, 404)
            else:
                self._send_json(meta)
        elif self.path.startswith("/image/source/"):
            # Serve the original file from @source if it's still on disk.
            # Powers the @Lightbox's full-resolution swap-in; falls back to
            # the preview when the source is disconnected.
            image_id = self.path[len("/image/source/"):]
            src = _state.db.fetch_image_source_path(image_id)
            if src is None:
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
            else:
                self._send_file(Path(src))
        elif self.path == "/ingest/progress":
            self._send_json(_state.ingest.progress())
        elif self.path == "/things/library":
            self._send_json({"names": _state.db.list_things_library()})
        elif self.path == "/collages":
            self._send_json({"collages": _state.db.list_collages()})
        elif self.path.startswith("/collages/") and self.path.endswith("/thumbnail"):
            collage_id = self.path[len("/collages/"):-len("/thumbnail")]
            self._handle_collage_thumbnail(collage_id)
        elif self.path.startswith("/collages/"):
            collage_id = self.path[len("/collages/"):]
            row = _state.db.fetch_collage(collage_id)
            if row is None:
                self._send_json({"error": "not found"}, 404)
            else:
                self._send_json(row)
        elif self.path == "/overview/index":
            self._handle_overview_index()
        elif self.path == "/overview/atlas":
            self._handle_overview_atlas()
        elif self.path == "/midatlas/index":
            self._handle_midatlas_index()
        elif self.path.startswith("/midatlas/page/"):
            self._handle_midatlas_page()
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            if self.path == "/slice":
                self._handle_slice()
            elif self.path == "/layout":
                self._handle_layout()
            elif self.path == "/spacelike":
                self._handle_spacelike()
            elif self.path == "/wireframe":
                self._handle_wireframe()
            elif self.path == "/neighborhoods":
                self._handle_neighborhoods()
            elif self.path == "/things":
                self._handle_things()
            elif self.path == "/things/library/add":
                self._handle_things_library_add()
            elif self.path == "/things/library/remove":
                self._handle_things_library_remove()
            elif self.path == "/collages/save":
                self._handle_collage_save()
            elif self.path == "/collages/rename":
                self._handle_collage_rename()
            elif self.path == "/collages/delete":
                self._handle_collage_delete()
            elif self.path == "/collages/export":
                self._handle_collage_export()
            elif self.path == "/collages/import":
                self._handle_collage_import()
            elif self.path == "/corpus/nuke":
                self._handle_corpus_nuke()
            elif self.path == "/ingest/start":
                self._handle_ingest_start()
            elif self.path == "/ingest/pause":
                self._handle_ingest_pause()
            elif self.path == "/ingest/resume":
                self._handle_ingest_resume()
            elif self.path == "/tools/pixel-features":
                self._handle_pixel_features()
            elif self.path == "/tools/embed-missing":
                self._handle_embed_missing()
            elif self.path == "/tools/regenerate-previews":
                self._handle_regenerate_previews()
            else:
                self._send_json({"error": "not found"}, 404)
        except Exception as e:
            logger.exception("Error handling %s", self.path)
            self._send_json({"error": str(e)}, 500)

    def _handle_corpus_nuke(self):
        if _state.ingest.is_running:
            _state.ingest.pause()
        _state.db.nuke()
        _state.provider.invalidate_cache()
        self._send_json({"status": "nuked"})

    def _handle_ingest_start(self):
        data = self._read_json()
        source = data.get("source")
        if not source:
            self._send_json({"error": "missing 'source' field"}, 400)
            return
        source_path = Path(source)
        if not source_path.is_dir():
            self._send_json({"error": f"source folder does not exist: {source}"}, 400)
            return
        result = _state.ingest.start(_state.workspace, source)
        if result == "already_running":
            self._send_json({"error": "import already running"}, 409)
            return
        self._send_json({"status": result})

    def _handle_ingest_pause(self):
        result = _state.ingest.pause()
        self._send_json({"status": result})

    def _handle_ingest_resume(self):
        source = _state.ingest.current_source
        if not source:
            self._send_json({"error": "no previous import to resume"}, 400)
            return
        result = _state.ingest.start(_state.workspace, source)
        if result == "already_running":
            self._send_json({"error": "import already running"}, 409)
            return
        self._send_json({"status": result})

    def _handle_pixel_features(self):
        """Run pixel feature extraction in the ingest thread."""
        if _state.ingest.is_running:
            self._send_json({"error": "import already running"}, 409)
            return

        import threading
        from sigil_atlas.cancel import CancellationToken
        from sigil_atlas.progress import BufferedProgressReporter
        from sigil_atlas.ingest.pixel_features import run_pixel_features_stage

        _state.ingest.token = CancellationToken()
        _state.ingest.reporter = BufferedProgressReporter()
        _state.ingest.reporter.emit_event("pipeline_started", source="pixel_features")

        def run():
            try:
                db = _state.workspace.open_db()
                progress = _state.ingest.reporter.create_stage("pixel_features", 0)
                run_pixel_features_stage(db, _state.workspace.thumbnails_dir, progress, _state.ingest.token)
                _state.ingest.reporter.emit_event("pipeline_completed")
            except Exception:
                logger.error("Pixel features failed", exc_info=True)
                _state.ingest.reporter.emit_event("pipeline_error")
            finally:
                db.close()

        _state.ingest.thread = threading.Thread(target=run, daemon=True, name="pixel-features")
        _state.ingest.thread.start()
        self._send_json({"status": "started"})

    def _handle_embed_missing(self):
        """Run embedding for all models that have missing images."""
        if _state.ingest.is_running:
            self._send_json({"error": "import already running"}, 409)
            return

        import threading
        from sigil_atlas.cancel import CancellationToken
        from sigil_atlas.progress import BufferedProgressReporter
        from sigil_atlas.ingest.embed import CLIPEmbedder, CLIPLargeEmbedder, DINOv2Embedder, run_embedding_stage

        _state.ingest.token = CancellationToken()
        _state.ingest.reporter = BufferedProgressReporter()
        _state.ingest.reporter.emit_event("pipeline_started", source="embed_missing")

        def run():
            try:
                db = _state.workspace.open_db()
                for embedder_cls in [CLIPEmbedder, CLIPLargeEmbedder, DINOv2Embedder]:
                    if _state.ingest.token.is_cancelled:
                        break
                    embedder = embedder_cls()
                    count = len(db.fetch_unembedded_image_ids(embedder.MODEL_ID))
                    if count == 0:
                        continue
                    progress = _state.ingest.reporter.create_stage(embedder.MODEL_ID, count)
                    run_embedding_stage(db, _state.workspace.thumbnails_dir, embedder, progress, _state.ingest.token)
                _state.ingest.reporter.emit_event("pipeline_completed")
            except Exception:
                logger.error("Embedding failed", exc_info=True)
                _state.ingest.reporter.emit_event("pipeline_error")
            finally:
                _state.provider.invalidate_cache()
                db.close()

        _state.ingest.thread = threading.Thread(target=run, daemon=True, name="embed-missing")
        _state.ingest.thread.start()
        self._send_json({"status": "started"})

    def _handle_regenerate_previews(self):
        """Sweep all completed images and regenerate any thumbnail or
        preview that is missing or below target size. Idempotent — images
        already at the target are no-ops. Upgrades legacy workspaces and
        fills gaps from partial ingests.

        Precheck: sample a handful of source_paths. If most are missing,
        the source disk is likely disconnected and re-running would
        destroy no data but also accomplish nothing except logging
        warnings for every image. Abort with a visible error instead.

        On successful completion, invalidate the baked overview and
        mid-atlas caches so they re-bake from the now-present source
        thumbnails on the next app launch.
        """
        if _state.ingest.is_running:
            self._send_json({"error": "import already running"}, 409)
            return

        import random
        import threading
        from sigil_atlas.cancel import CancellationToken
        from sigil_atlas.progress import BufferedProgressReporter
        from sigil_atlas.ingest.thumbnail import _generate_one
        from sigil_atlas.overview import (
            mid_atlas_index_path,
            mid_atlas_page_path,
            overview_paths,
        )

        items = _state.db.fetch_completed_images_with_paths()
        if not items:
            self._send_json({"status": "started", "count": 0})
            return

        # Sample-probe the source disk. A handful of accessible files is
        # enough to confirm the volume is mounted; many missing means the
        # user's external drive is unplugged.
        SAMPLE = min(20, len(items))
        REQUIRED_FRACTION = 0.5
        sample = random.sample(items, SAMPLE)
        present = sum(1 for _, p in sample if Path(p).exists())
        if present / SAMPLE < REQUIRED_FRACTION:
            self._send_json({
                "error": (
                    f"Source files not accessible: only {present}/{SAMPLE} of a "
                    f"random sample were found on disk. The drive the images "
                    f"live on may be disconnected. Connect it and try again."
                ),
            }, 503)
            return

        _state.ingest.token = CancellationToken()
        _state.ingest.reporter = BufferedProgressReporter()
        _state.ingest.reporter.emit_event("pipeline_started", source="regenerate_previews")

        def run():
            db = None
            try:
                db = _state.workspace.open_db()
                progress = _state.ingest.reporter.create_stage("regenerate_previews", len(items))
                previews_dir = _state.workspace.previews_dir
                thumbnails_dir = _state.workspace.thumbnails_dir
                for image_id, source_path in items:
                    if _state.ingest.token.is_cancelled:
                        break
                    _generate_one(db, image_id, Path(source_path), thumbnails_dir, previews_dir)
                    progress.advance(1)

                # Invalidate baked atlas caches so the next launch re-bakes
                # them from the now-filled thumbnails_dir. Missing files are
                # swallowed — this is best-effort cleanup.
                if not _state.ingest.token.is_cancelled:
                    png_path, idx_path = overview_paths(_state.workspace)
                    for p in (png_path, idx_path):
                        try: p.unlink()
                        except FileNotFoundError: pass
                        except Exception: logger.warning("Could not remove %s", p, exc_info=True)
                    mid_idx = mid_atlas_index_path(_state.workspace)
                    try: mid_idx.unlink()
                    except FileNotFoundError: pass
                    except Exception: logger.warning("Could not remove %s", mid_idx, exc_info=True)
                    # Mid-atlas pages: we don't know the exact count without
                    # the index, so glob for mid-atlas-*.png.
                    for p in _state.workspace.cache_dir.glob("mid-atlas-*.png"):
                        try: p.unlink()
                        except Exception: logger.warning("Could not remove %s", p, exc_info=True)
                    logger.info("Regenerate previews: invalidated baked atlas caches")

                _state.ingest.reporter.emit_event("pipeline_completed")
            except Exception:
                logger.error("Preview regeneration failed", exc_info=True)
                _state.ingest.reporter.emit_event("pipeline_error")
            finally:
                if db is not None:
                    db.close()

        _state.ingest.thread = threading.Thread(target=run, daemon=True, name="regenerate-previews")
        _state.ingest.thread.start()
        self._send_json({"status": "started", "count": len(items)})

    def _handle_dimensions(self):
        """Return range characterization dimensions with their min/max.

        Only range dimensions are returned — they drive the discriminate
        sliders in the Slice panel. Enum dimensions (taxonomy terms)
        are accessed via the vocabulary endpoints instead.
        """
        rows = _state.db._conn.execute(
            "SELECT proximity_name, MIN(value_range), MAX(value_range) "
            "FROM characterizations WHERE value_type = 'range' "
            "GROUP BY proximity_name"
        ).fetchall()

        dimensions = [
            {"name": r[0], "type": "range", "min": r[1], "max": r[2]}
            for r in rows
        ]
        self._send_json({"dimensions": dimensions})

    def _handle_models(self):
        models = _state.provider.available_models()
        total = _state.db.image_count()
        # Count embeddings per model so the UI can disable incomplete ones.
        rows = _state.db._conn.execute(
            "SELECT model_identifier, COUNT(*) FROM embeddings GROUP BY model_identifier"
        ).fetchall()
        counts = {r[0]: r[1] for r in rows}
        self._send_json({
            "models": models,
            "total": total,
            "counts": counts,
        })

    def _handle_slice(self):
        data = self._read_json()
        filter_expr = parse_filter(data.get("filter"))
        relevance = float(data.get("relevance", 0.5))
        model = data.get("model", "clip-vit-l-14")

        order_node = data.get("order_contrast")
        order_contrast = None
        if order_node:
            order_contrast = FilterContrast(
                pole_a=order_node["pole_a"],
                pole_b=order_node["pole_b"],
            )

        try:
            get_adapter(model)
        except ValueError as e:
            self._send_json({"error": str(e)}, 400)
            return

        logger.info("Slice request: model=%s, relevance=%.2f, filter=%s",
                    model, relevance, data.get("filter"))

        result = compute_slice(
            _state.db, _state.provider,
            filter_expr, model,
            relevance=relevance,
            order_contrast=order_contrast,
        )

        if result.order_projections is not None:
            order_values = result.order_projections
        else:
            order_values = result.capture_dates

        self._send_json({
            "image_ids": result.image_ids,
            "count": len(result.image_ids),
            "has_order_axis": result.order_projections is not None,
            "order_values": order_values,
        })

    def _handle_layout(self):
        data = self._read_json()
        image_ids = data.get("image_ids", [])
        axes = data.get("axes")
        model = data.get("model", "clip-vit-l-14")
        strip_height = data.get("strip_height", 100.0)
        order_values = data.get("order_values")

        try:
            get_adapter(model)
        except ValueError as e:
            self._send_json({"error": str(e)}, 400)
            return

        if not image_ids:
            image_ids = _state.db.fetch_image_ids()

        preserve_order = data.get("preserve_order", False)
        layout_mode = data.get("layout_mode", "auto")
        layout = compute_layout(
            _state.provider, _state.db, image_ids,
            axes=axes, model=model,
            strip_height=strip_height, preserve_order=preserve_order,
            order_values=order_values, layout_mode=layout_mode,
        )

        strips_json = []
        for strip in layout.strips:
            images_json = [
                {"id": img.id, "x": img.x, "width": img.width, "thumbnail_path": img.thumbnail_path}
                for img in strip.images
            ]
            strips_json.append({"y": strip.y, "height": strip.height, "images": images_json})

        self._send_json({
            "strips": strips_json,
            "torus_width": layout.torus_width,
            "torus_height": layout.torus_height,
            "strip_height": layout.strip_height,
        })


    def _handle_spacelike(self):
        data = self._read_json()
        model = data.get("model", "clip-vit-l-14")
        feathering = data.get("feathering", 0.5)
        cell_size = data.get("cell_size", 100.0)
        relevance = float(data.get("relevance", 0.5))
        field_expansion = data.get("field_expansion", "echo")
        arrangement = data.get("arrangement", "rings")

        try:
            get_adapter(model)
        except ValueError as e:
            self._send_json({"error": str(e)}, 400)
            return

        # Single expression, three roles: gates image_ids, names attractors,
        # names contrasts that shape proximity. Per spec, one sigil with faces.
        filter_expr = parse_filter(data.get("filter"))
        slice_result = compute_slice(
            _state.db, _state.provider, filter_expr, model,
            relevance=relevance,
        )
        image_ids = slice_result.image_ids

        attractors = [
            Attractor(kind="thing", ref=t.name) for t in thing_atoms(filter_expr)
        ] + [
            Attractor(kind="target_image", ref=ti.image_id)
            for ti in target_image_atoms(filter_expr)
        ]

        contrasts = [
            ContrastAxis(pole_a=c.pole_a, pole_b=c.pole_b)
            for c in walk_filter(filter_expr) if isinstance(c, FilterContrast)
        ]

        layout = compute_spacelike(
            _state.provider, _state.db, image_ids,
            attractors=attractors, contrasts=contrasts, model=model,
            feathering=feathering, cell_size=cell_size,
            field_expansion=field_expansion,
            arrangement=arrangement,
        )

        self._send_json({
            "positions": [
                {"id": p.id, "col": p.col, "row": p.row, "elevation": p.elevation}
                for p in layout.positions
            ],
            "attractor_positions": [
                {"kind": a.kind, "ref": a.ref, "col": a.col, "row": a.row}
                for a in layout.attractor_positions
            ],
            "cell_size": layout.cell_size,
            "cols": layout.cols,
            "rows": layout.rows,
            "torus_width": layout.torus_width,
            "torus_height": layout.torus_height,
        })

    def _handle_overview_index(self):
        idx = generate_overview(_state.workspace, _state.db)
        self._send_json({
            "tile_size": idx.tile_size,
            "cols": idx.cols,
            "rows": idx.rows,
            "atlas_width": idx.atlas_width,
            "atlas_height": idx.atlas_height,
            "mapping": idx.mapping,
        })

    def _handle_overview_atlas(self):
        png_path, _ = overview_paths(_state.workspace)
        if not png_path.exists():
            generate_overview(_state.workspace, _state.db)
        self._send_file(png_path)

    def _handle_midatlas_index(self):
        idx = generate_mid_atlas(_state.workspace, _state.db)
        self._send_json({
            "tile_size": idx.tile_size,
            "cols_per_page": idx.cols_per_page,
            "rows_per_page": idx.rows_per_page,
            "atlas_width": idx.atlas_width,
            "atlas_height": idx.atlas_height,
            "pages": idx.pages,
            "mapping": idx.mapping,
        })

    def _handle_midatlas_page(self):
        # URL: /midatlas/page/<n>
        try:
            page = int(self.path.rsplit("/", 1)[-1])
        except ValueError:
            self._send_json({"error": "bad page number"}, 400)
            return
        page_path = mid_atlas_page_path(_state.workspace, page)
        if not page_path.exists():
            generate_mid_atlas(_state.workspace, _state.db)
        self._send_file(page_path)

    def _handle_wireframe(self):
        data = self._read_json()
        image_ids = data.get("image_ids") or _state.db.fetch_image_ids()
        model = data.get("model", "clip-vit-l-14")
        k = int(data.get("k", 6))

        try:
            get_adapter(model)
        except ValueError as e:
            self._send_json({"error": str(e)}, 400)
            return

        edges = compute_wireframe_edges(_state.provider, image_ids, model, k)
        logger.info("Wireframe: %d edges over %d images (k=%d)", len(edges), len(image_ids), k)
        self._send_json({"edges": edges})

    def _handle_neighborhoods(self):
        """Per-image cluster ids from the precomputed KMeans at the chosen k.

        Used by the Neighborhoods overlay: each image carries its cluster id;
        the frontend renders boundary lines where adjacent cells on the grid
        disagree.
        """
        data = self._read_json()
        image_ids = data.get("image_ids") or _state.db.fetch_image_ids()
        model = data.get("model", "clip-vit-l-14")
        k = int(data.get("k", 50))
        try:
            get_adapter(model)
        except ValueError as e:
            self._send_json({"error": str(e)}, 400)
            return
        cluster_ids = _state.db.fetch_kmeans_assignments_for_ids(model, k, image_ids)
        logger.info("Neighborhoods: k=%d, %d/%d assigned", k, len(cluster_ids), len(image_ids))
        self._send_json({"k": k, "cluster_ids": cluster_ids})

    def _handle_things_library_add(self):
        data = self._read_json()
        name = (data.get("name") or "").strip()
        if not name:
            self._send_json({"error": "missing 'name'"}, 400)
            return
        _state.db.add_thing_to_library(name)
        self._send_json({"names": _state.db.list_things_library()})

    def _handle_things_library_remove(self):
        data = self._read_json()
        name = (data.get("name") or "").strip()
        if not name:
            self._send_json({"error": "missing 'name'"}, 400)
            return
        _state.db.remove_thing_from_library(name)
        self._send_json({"names": _state.db.list_things_library()})

    # ── Collages ───────────────────────────────────────────────────────────

    def _handle_collage_thumbnail(self, collage_id: str):
        blob = _state.db.fetch_collage_thumbnail(collage_id)
        if blob is None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(blob)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        self.wfile.write(blob)

    def _handle_collage_save(self):
        import base64
        import uuid
        data = self._read_json()
        name = (data.get("name") or "").strip()
        if not name:
            self._send_json({"error": "missing 'name'"}, 400)
            return
        expression = data.get("expression")  # may be None for unconstrained
        pov = data.get("pov")
        if pov is None:
            self._send_json({"error": "missing 'pov'"}, 400)
            return
        thumbnail_blob = None
        thumb_b64 = data.get("thumbnail_base64")
        if thumb_b64:
            try:
                thumbnail_blob = base64.b64decode(thumb_b64)
            except Exception:
                logger.warning("Bad thumbnail_base64 for collage save")
        collage_id = str(uuid.uuid4())
        _state.db.insert_collage(
            collage_id=collage_id,
            name=name,
            expression_json=json.dumps(expression),
            pov_json=json.dumps(pov),
            mode=data.get("mode", "spacelike"),
            model=data.get("model", "clip-vit-b-32"),
            relevance=float(data.get("relevance", 0.5)),
            feathering=float(data.get("feathering", 0.5)),
            cell_size=float(data.get("cell_size", 100.0)),
            thumbnail_blob=thumbnail_blob,
        )
        self._send_json({"id": collage_id, "collages": _state.db.list_collages()})

    def _handle_collage_rename(self):
        data = self._read_json()
        collage_id = data.get("id")
        new_name = (data.get("name") or "").strip()
        if not collage_id or not new_name:
            self._send_json({"error": "missing 'id' or 'name'"}, 400)
            return
        ok = _state.db.rename_collage(collage_id, new_name)
        if not ok:
            self._send_json({"error": "not found"}, 404)
            return
        self._send_json({"collages": _state.db.list_collages()})

    def _handle_collage_delete(self):
        data = self._read_json()
        collage_id = data.get("id")
        if not collage_id:
            self._send_json({"error": "missing 'id'"}, 400)
            return
        ok = _state.db.delete_collage(collage_id)
        if not ok:
            self._send_json({"error": "not found"}, 404)
            return
        self._send_json({"collages": _state.db.list_collages()})

    # ── Collage export / import as `.sigil` directories ───────────────────

    def _handle_collage_export(self):
        from sigil_atlas import collage as collage_mod
        data = self._read_json()
        parent = data.get("parent_path")
        if not parent:
            self._send_json({"error": "missing 'parent_path'"}, 400)
            return
        parent_path = Path(parent)
        if not parent_path.is_dir():
            self._send_json({"error": f"not a directory: {parent}"}, 400)
            return

        expression = data.get("expression")  # may be None
        pov = data.get("pov")
        if pov is None:
            self._send_json({"error": "missing 'pov'"}, 400)
            return

        image_ids = data.get("image_ids") or []
        pill_names = [a["ref"] for a in data.get("attractors", []) if a.get("kind") == "thing"]

        base_name = collage_mod.derive_folder_name(
            user_hint=data.get("name_hint"),
            pill_names=pill_names,
            image_ids=image_ids,
            provider=_state.provider,
        )
        folder = collage_mod.unique_sigil_folder(parent_path, base_name)

        try:
            collage_mod.write_collage(
                folder,
                name=base_name,
                expression=expression,
                pov=pov,
                mode=data.get("mode", "spacelike"),
                model=data.get("model", "clip-vit-b-32"),
                relevance=float(data.get("relevance", 0.5)),
                feathering=float(data.get("feathering", 0.5)),
                cell_size=float(data.get("cell_size", 100.0)),
                image_ids=image_ids,
                screenshot_base64=data.get("screenshot_base64"),
                field_expansion=str(data.get("field_expansion", "echo")),
                arrangement=str(data.get("arrangement", "rings")),
                time_direction=str(data.get("time_direction", "capture_date")),
                strip_height=float(data.get("strip_height", 100.0)),
                torus_width=float(data.get("torus_width", 0.0)),
                torus_height=float(data.get("torus_height", 0.0)),
            )
        except Exception as e:
            logger.exception("Collage export failed")
            self._send_json({"error": str(e)}, 500)
            return

        self._send_json({"folder_path": str(folder), "name": base_name})

    def _handle_collage_import(self):
        from sigil_atlas import collage as collage_mod
        data = self._read_json()
        folder = data.get("folder_path")
        if not folder:
            self._send_json({"error": "missing 'folder_path'"}, 400)
            return
        folder_path = Path(folder)
        if not folder_path.is_dir():
            self._send_json({"error": f"not a directory: {folder}"}, 400)
            return
        try:
            manifest = collage_mod.read_collage(folder_path)
        except FileNotFoundError as e:
            self._send_json({"error": str(e)}, 400)
            return
        except json.JSONDecodeError as e:
            self._send_json({"error": f"corrupt collage.json: {e}"}, 400)
            return
        self._send_json({"collage": manifest, "folder_path": str(folder_path)})

    def _handle_things(self):
        data = self._read_json()
        terms = data.get("terms", [])
        model = data.get("model", "clip-vit-l-14")
        strip_height = data.get("strip_height", 100.0)
        top_k = data.get("top_k", 200)

        image_ids = data.get("image_ids")
        if not image_ids:
            image_ids = _state.db.fetch_image_ids()

        layout = compute_things_layout(
            _state.provider, _state.db, image_ids,
            terms=terms, model=model,
            strip_height=strip_height, top_k=top_k,
        )

        neighborhoods_json = []
        for nb in layout.neighborhoods:
            strips_json = [
                {
                    "y": s.y,
                    "height": s.height,
                    "images": [
                        {"id": img.id, "x": img.x, "width": img.width, "thumbnail_path": img.thumbnail_path}
                        for img in s.images
                    ],
                }
                for s in nb.strips
            ]
            neighborhoods_json.append({
                "term": nb.term,
                "prompt": nb.prompt,
                "x": nb.x,
                "y": nb.y,
                "width": nb.width,
                "height": nb.height,
                "strips": strips_json,
                "image_count": nb.image_count,
            })

        self._send_json({
            "neighborhoods": neighborhoods_json,
            "torus_width": layout.torus_width,
            "torus_height": layout.torus_height,
            "strip_height": layout.strip_height,
        })


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description="SigilAtlas sidecar server")
    parser.add_argument("--workspace", required=True, help="Path to workspace directory")
    parser.add_argument("--port", type=int, default=0, help="Port (0 = random)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    global _state
    _state = SidecarState(Path(args.workspace))
    logger.info("Workspace loaded: %s (%d images)", args.workspace, _state.db.image_count())

    # CLIP model loads lazily on first text encode

    port = args.port or _find_free_port()
    # ThreadingHTTPServer handles requests concurrently — a preview fetch
    # (megabytes) must not block small JSON endpoints, and the frontend's
    # 16 parallel preview loads must not serialize behind each other.
    server = ThreadingHTTPServer(("127.0.0.1", port), RequestHandler)

    # Print port on first line for Tauri to read
    print(port, flush=True)
    logger.info("Sidecar server listening on http://127.0.0.1:%d", port)

    # Crash recovery runs after the server is serving — it's O(N) in corpus
    # size and not needed for queries, only for repairing interrupted imports.
    threading.Thread(
        target=_state.workspace.recover, args=(_state.db,), daemon=True, name="recover"
    ).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _state.db.close()
        server.server_close()


if __name__ == "__main__":
    main()
