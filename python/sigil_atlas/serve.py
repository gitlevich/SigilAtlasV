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
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from sigil_atlas.db import CorpusDB
from sigil_atlas.embedding_provider import SqliteEmbeddingProvider
from sigil_atlas.layout import compute_layout
from sigil_atlas.slice import RangeFilter, ProximityFilter, compute_slice
from sigil_atlas.workspace import Workspace

logger = logging.getLogger(__name__)


class SidecarState:
    """Shared server state."""

    def __init__(self, workspace_path: Path) -> None:
        self.workspace = Workspace(workspace_path)
        self.db = self.workspace.open_db()
        self.provider = SqliteEmbeddingProvider(self.db)


_state: SidecarState | None = None


class RequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        logger.info(format, *args)

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
        elif self.path.startswith("/thumbnail/"):
            image_id = self.path[len("/thumbnail/"):]
            thumb_path = _state.workspace.thumbnails_dir / f"{image_id}.jpg"
            self._send_file(thumb_path)
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            if self.path == "/slice":
                self._handle_slice()
            elif self.path == "/layout":
                self._handle_layout()
            else:
                self._send_json({"error": "not found"}, 404)
        except Exception as e:
            logger.exception("Error handling %s", self.path)
            self._send_json({"error": str(e)}, 500)

    def _handle_dimensions(self):
        """Return available characterization dimensions with their ranges."""
        rows = _state.db._conn.execute(
            "SELECT DISTINCT proximity_name, value_type FROM characterizations"
        ).fetchall()

        dimensions = []
        for name, vtype in rows:
            if vtype == "range":
                min_max = _state.db._conn.execute(
                    "SELECT MIN(value_range), MAX(value_range) FROM characterizations WHERE proximity_name = ?",
                    (name,),
                ).fetchone()
                dimensions.append({"name": name, "type": "range", "min": min_max[0], "max": min_max[1]})
            else:
                values = _state.db._conn.execute(
                    "SELECT DISTINCT value_enum FROM characterizations WHERE proximity_name = ?",
                    (name,),
                ).fetchall()
                dimensions.append({"name": name, "type": "enum", "values": [v[0] for v in values]})

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
        model = data.get("model", "clip-vit-b-32")

        image_ids = compute_slice(_state.db, _state.provider, range_filters, proximity_filters, model)
        self._send_json({"image_ids": image_ids, "count": len(image_ids)})

    def _handle_layout(self):
        data = self._read_json()
        image_ids = data.get("image_ids", [])
        axes = data.get("axes")
        tightness = data.get("tightness", 0.5)
        model = data.get("model", "clip-vit-b-32")
        strip_height = data.get("strip_height", 100.0)

        if not image_ids:
            image_ids = _state.db.fetch_image_ids()

        layout = compute_layout(
            _state.provider, _state.db, image_ids,
            axes=axes, tightness=tightness, model=model,
            strip_height=strip_height,
        )

        strips_json = []
        for strip in layout.strips:
            images_json = [
                {"id": img.id, "x": img.x, "width": img.width, "thumbnail_path": img.thumbnail_path}
                for img in strip.images
            ]
            strips_json.append({"y": strip.y, "images": images_json})

        self._send_json({
            "strips": strips_json,
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

    port = args.port or _find_free_port()
    server = HTTPServer(("127.0.0.1", port), RequestHandler)

    # Print port on first line for Tauri to read
    print(port, flush=True)
    logger.info("Sidecar server listening on http://127.0.0.1:%d", port)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _state.db.close()
        server.server_close()


if __name__ == "__main__":
    main()
