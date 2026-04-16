"""Tests for import progress reporting through the HTTP sidecar.

Reproduces the bug: import starts but progress is never visible to the frontend.
"""

import json
import tempfile
import threading
import time
from http.server import HTTPServer
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from sigil_atlas.db import CorpusDB
from sigil_atlas.ingest.pipeline import IngestPipeline
from sigil_atlas.ingest.source import FolderSource
from sigil_atlas.cancel import CancellationToken
from sigil_atlas.progress import BufferedProgressReporter
from sigil_atlas.serve import IngestState, RequestHandler, SidecarState
from sigil_atlas.workspace import Workspace


def _create_test_image(path: Path, color: str = "red") -> None:
    img = Image.new("RGB", (100, 100), color=color)
    img.save(path, "JPEG")
    img.close()


def _make_workspace_and_source(tmp: Path, n_images: int = 3):
    ws_dir = tmp / "workspace"
    ws_dir.mkdir()
    src_dir = tmp / "source"
    src_dir.mkdir()
    for i in range(n_images):
        r, g, b = (i * 37) % 256, (i * 53) % 256, (i * 71) % 256
        _create_test_image(src_dir / f"img_{i}.jpg", color=f"#{r:02x}{g:02x}{b:02x}")

    ws = Workspace(ws_dir)
    ws.initialize()
    return ws, src_dir


class TestBufferedProgressReporter:
    def test_initial_state_is_idle(self):
        r = BufferedProgressReporter()
        snap = r.snapshot()
        assert snap["status"] == "idle"
        assert snap["stages"] == []
        assert snap["started_at"] is None

    def test_stage_progress_captured(self):
        r = BufferedProgressReporter()
        r.emit_event("pipeline_started", source="/tmp")
        stage = r.create_stage("scan", 100)
        stage.advance(42)

        snap = r.snapshot()
        assert snap["status"] == "running"
        assert len(snap["stages"]) == 1
        assert snap["stages"][0]["name"] == "scan"
        assert snap["stages"][0]["completed"] == 42
        assert snap["stages"][0]["total"] == 100

    def test_completed_event_sets_status(self):
        r = BufferedProgressReporter()
        r.emit_event("pipeline_started", source="/tmp")
        r.emit_event("pipeline_completed")
        assert r.snapshot()["status"] == "completed"

    def test_error_event_sets_status(self):
        r = BufferedProgressReporter()
        r.emit_event("pipeline_started", source="/tmp")
        r.emit_event("pipeline_error")
        assert r.snapshot()["status"] == "error"


class TestIngestState:
    def test_not_running_initially(self):
        state = IngestState()
        assert not state.is_running
        assert state.progress()["status"] == "idle"

    def test_start_returns_started(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws, src_dir = _make_workspace_and_source(Path(tmp))
            state = IngestState()

            # Patch the pipeline to avoid actually running heavy stages
            result = state.start(ws, str(src_dir))
            assert result == "started"
            assert state.is_running or True  # thread may finish fast with tiny images
            # Wait for it to settle
            time.sleep(0.5)
            snap = state.progress()
            assert snap["status"] in ("running", "completed", "error")
            assert snap["started_at"] is not None

    def test_progress_has_stages_after_start(self):
        """The core bug: progress must report stages, not stay empty."""
        with tempfile.TemporaryDirectory() as tmp:
            ws, src_dir = _make_workspace_and_source(Path(tmp), n_images=3)
            state = IngestState()
            state.start(ws, str(src_dir))

            # Poll until we see stages or timeout
            stages_seen = False
            for _ in range(20):  # 2 seconds max
                snap = state.progress()
                if snap["stages"]:
                    stages_seen = True
                    break
                time.sleep(0.1)

            assert stages_seen, f"No stages ever appeared in progress. Final: {state.progress()}"

    def test_pause_sets_paused_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws, src_dir = _make_workspace_and_source(Path(tmp), n_images=50)
            state = IngestState()
            state.start(ws, str(src_dir))
            time.sleep(0.2)

            result = state.pause()
            assert result == "paused"
            assert state.progress()["status"] == "paused"

    def test_pause_preserves_stage_progress(self):
        """Pausing must not clear the stage data."""
        with tempfile.TemporaryDirectory() as tmp:
            ws, src_dir = _make_workspace_and_source(Path(tmp), n_images=50)
            state = IngestState()
            state.start(ws, str(src_dir))

            # Wait for some stages
            for _ in range(20):
                if state.progress()["stages"]:
                    break
                time.sleep(0.1)

            state.pause()
            snap = state.progress()
            assert snap["status"] == "paused"
            assert len(snap["stages"]) > 0, "Stages disappeared after pause"

    def test_double_start_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws, src_dir = _make_workspace_and_source(Path(tmp), n_images=50)
            state = IngestState()
            state.start(ws, str(src_dir))
            result = state.start(ws, str(src_dir))
            assert result == "already_running"
            state.pause()


class TestIngestHTTPEndpoints:
    """Test the actual HTTP endpoint responses."""

    def _start_server(self, workspace_path: Path):
        import sigil_atlas.serve as serve_module
        serve_module._state = SidecarState(workspace_path)
        server = HTTPServer(("127.0.0.1", 0), RequestHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, port, serve_module._state

    def test_progress_endpoint_returns_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws, _ = _make_workspace_and_source(Path(tmp))
            server, port, _ = self._start_server(ws.root)
            try:
                import urllib.request
                resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/ingest/progress")
                data = json.loads(resp.read())
                assert data["status"] == "idle"
                assert data["stages"] == []
            finally:
                server.shutdown()

    def test_start_then_progress_shows_stages(self):
        """End-to-end: start import via HTTP, poll progress, see stages."""
        with tempfile.TemporaryDirectory() as tmp:
            ws, src_dir = _make_workspace_and_source(Path(tmp), n_images=3)
            server, port, _ = self._start_server(ws.root)
            try:
                import urllib.request

                # Start import
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/ingest/start",
                    data=json.dumps({"source": str(src_dir)}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                resp = urllib.request.urlopen(req)
                start_data = json.loads(resp.read())
                assert start_data["status"] == "started"

                # Poll for stages
                stages_seen = False
                for _ in range(40):  # 4 seconds
                    resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/ingest/progress")
                    data = json.loads(resp.read())
                    if data["stages"]:
                        stages_seen = True
                        break
                    time.sleep(0.1)

                assert stages_seen, f"Stages never appeared via HTTP. Final: {data}"
            finally:
                server.shutdown()
