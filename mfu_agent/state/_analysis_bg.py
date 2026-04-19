"""Module-level state for background analysis thread.

Survives Streamlit reruns and browser refreshes because Python
modules persist for the lifetime of the process.
"""

from __future__ import annotations

import threading
import time

thread: threading.Thread | None = None
progress: dict = {}
log_lines: list[str] = []


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    log_lines.append(f"`{ts}` {msg}")


def reset() -> None:
    global thread
    thread = None
    progress.clear()
    log_lines.clear()
