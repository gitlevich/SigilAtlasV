"""Cancellation token for pipeline stages."""

import threading


class CancellationToken:
    """Thread-safe cancellation signal.

    Checked between batches by every pipeline stage.
    When cancelled, the current batch finishes and the stage stops cleanly.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()
