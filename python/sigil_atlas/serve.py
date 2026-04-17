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
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from sigil_atlas.db import CorpusDB
from sigil_atlas.embedding_provider import SqliteEmbeddingProvider
from sigil_atlas.layout import compute_layout
from sigil_atlas.model_registry import get_adapter
from sigil_atlas.slice import RangeFilter, ProximityFilter, ContrastControl, compute_slice
from sigil_atlas.spacelike import Attractor, compute_spacelike
from sigil_atlas.taxonomy import vocabulary, vocabulary_tree, vocabulary_flat
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
            self._send_json({"error": "not found"}, 404)
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
        elif self.path == "/vocabulary/flat":
            self._send_json(vocabulary_flat())
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
        elif self.path == "/ingest/progress":
            self._send_json(_state.ingest.progress())
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
            elif self.path == "/things":
                self._handle_things()
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

    def _handle_dimensions(self):
        """Return range characterization dimensions with their min/max.

        Only range dimensions are returned — they drive the bandpass
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
        self._send_json({"models": models})

    def _handle_slice(self):
        data = self._read_json()
        range_filters = [
            RangeFilter(dimension=f["dimension"], min_value=f["min"], max_value=f["max"])
            for f in data.get("range_filters", [])
        ]
        proximity_filters = [
            ProximityFilter(text=f["text"], weight=f.get("weight", 1.0))
            for f in data.get("proximity_filters", [])
        ]
        contrast_controls = [
            ContrastControl(
                pole_a=c["pole_a"], pole_b=c["pole_b"],
                role=c.get("role", "filter"),
                band_min=c.get("band_min", -1.0),
                band_max=c.get("band_max", 1.0),
            )
            for c in data.get("contrast_controls", [])
        ]
        model = data.get("model", "clip-vit-l-14")
        feathering = data.get("feathering", 0.5)

        # Validate model via registry — fail loudly if unknown
        try:
            get_adapter(model)
        except ValueError as e:
            self._send_json({"error": str(e)}, 400)
            return

        logger.info("Slice request: model=%s, feathering=%.2f, proximity=%s", model, feathering, [f["text"] for f in data.get("proximity_filters", [])])

        result = compute_slice(
            _state.db, _state.provider,
            range_filters, proximity_filters, contrast_controls, model,
            feathering=feathering,
        )
        # Return order values: contrast projections if order axis active, else capture dates
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
        image_ids = data.get("image_ids") or _state.db.fetch_image_ids()
        model = data.get("model", "clip-vit-l-14")
        feathering = data.get("feathering", 0.5)
        cell_size = data.get("cell_size", 100.0)
        attractors = [
            Attractor(kind=a["kind"], ref=a["ref"])
            for a in data.get("attractors", [])
        ]

        try:
            get_adapter(model)
        except ValueError as e:
            self._send_json({"error": str(e)}, 400)
            return

        layout = compute_spacelike(
            _state.provider, _state.db, image_ids,
            attractors=attractors, model=model,
            feathering=feathering, cell_size=cell_size,
        )

        self._send_json({
            "positions": [
                {"id": p.id, "col": p.col, "row": p.row}
                for p in layout.positions
            ],
            "cell_size": layout.cell_size,
            "cols": layout.cols,
            "rows": layout.rows,
            "torus_width": layout.torus_width,
            "torus_height": layout.torus_height,
        })

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
    server = HTTPServer(("127.0.0.1", port), RequestHandler)

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
