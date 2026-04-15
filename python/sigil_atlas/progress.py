"""Progress reporting for pipeline stages — JSON lines to stdout."""

import json
import logging
import sys
import threading
import time

logger = logging.getLogger(__name__)


class StageProgress:
    """Tracks progress for a single pipeline stage."""

    def __init__(self, stage_name: str, total: int, reporter: "ProgressReporter") -> None:
        self.stage_name = stage_name
        self.total = total
        self.completed = 0
        self._reporter = reporter
        self._lock = threading.Lock()

    def advance(self, count: int = 1) -> None:
        with self._lock:
            self.completed += count
            self._reporter.emit(self)

    def set_total(self, total: int) -> None:
        with self._lock:
            self.total = total


class ProgressReporter:
    """Emits progress as JSON lines to stdout for Tauri to consume."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def create_stage(self, stage_name: str, total: int = 0) -> StageProgress:
        return StageProgress(stage_name, total, self)

    def emit(self, stage: StageProgress) -> None:
        msg = {
            "type": "progress",
            "stage": stage.stage_name,
            "completed": stage.completed,
            "total": stage.total,
            "timestamp": time.time(),
        }
        line = json.dumps(msg)
        with self._lock:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    def emit_event(self, event: str, **kwargs) -> None:
        msg = {"type": event, "timestamp": time.time(), **kwargs}
        line = json.dumps(msg)
        with self._lock:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()


class BufferedProgressReporter(ProgressReporter):
    """Buffers progress for HTTP polling instead of writing to stdout.

    Thread-safe. The HTTP handler reads `snapshot()` to return current state.
    """

    def __init__(self) -> None:
        super().__init__()
        self._stages: dict[str, dict] = {}
        self._events: list[dict] = []
        self._status = "idle"
        self._started_at: float | None = None

    def emit(self, stage: StageProgress) -> None:
        with self._lock:
            self._stages[stage.stage_name] = {
                "name": stage.stage_name,
                "completed": stage.completed,
                "total": stage.total,
                "timestamp": time.time(),
            }

    def emit_event(self, event: str, **kwargs) -> None:
        msg = {"type": event, "timestamp": time.time(), **kwargs}
        with self._lock:
            self._events.append(msg)
            if event == "pipeline_started":
                self._status = "running"
                self._started_at = time.time()
            elif event == "pipeline_completed":
                self._status = "completed"
            elif event == "pipeline_error":
                self._status = "error"

    def set_status(self, status: str) -> None:
        with self._lock:
            self._status = status

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "status": self._status,
                "stages": list(self._stages.values()),
                "started_at": self._started_at,
            }
