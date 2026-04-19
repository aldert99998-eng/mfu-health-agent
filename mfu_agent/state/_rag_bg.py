"""Module-level state for background RAG indexing process.

Uses multiprocessing instead of threading because BGE-M3 model
loading holds the GIL and blocks Streamlit's main thread.

Shared state via multiprocessing.Manager survives Streamlit reruns.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from multiprocessing.managers import SyncManager

_manager: SyncManager | None = None
process: mp.Process | None = None
progress: dict = {}
log_lines: list[str] = []
result: dict = {}


def _get_manager() -> SyncManager:
    global _manager
    if _manager is None:
        ctx = mp.get_context("spawn")
        _manager = ctx.Manager()
    return _manager


def make_shared() -> tuple[dict, list, dict]:
    m = _get_manager()
    shared_progress = m.dict()
    shared_log = m.list()
    shared_result = m.dict()
    return shared_progress, shared_log, shared_result


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    log_lines.append(f"`{ts}` {msg}")


def reset() -> None:
    global process, progress, log_lines, result
    process = None
    progress = {}
    log_lines = []
    result = {}
