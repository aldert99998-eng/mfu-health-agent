"""Module-level state for deep-analysis background thread.

Separate from `_analysis_bg` so that deep LLM analysis (minutes-long)
does not collide with the main ingestion worker. Survives Streamlit
reruns because Python modules persist for the lifetime of the process.
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
