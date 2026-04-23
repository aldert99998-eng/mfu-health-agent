"""Manage a locally-running llama-server: discover, inspect, restart.

Scope is intentionally narrow: only llama.cpp's ``llama-server`` binary is
supported. We preserve every argv flag of the running instance except ``-m``,
which is swapped for the chosen model path.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

MODELS_DIR = Path(os.environ.get("MFU_LOCAL_MODELS_DIR", "/home/albert/models"))
_LLAMA_SERVER_EXEC = "llama-server"
_SWITCH_TIMEOUT_S = 120
_READY_POLL_INTERVAL_S = 1.0
_VRAM_RELEASE_WAIT_S = 3.0
_PORT_WAIT_S = 5.0
LLAMA_LOG_PATH = Path(
    os.environ.get("MFU_LLAMA_LOG", "/tmp/llama-server.log")  # noqa: S108
)


@dataclass(frozen=True)
class RunningServer:
    pid: int
    exe: str
    argv: list[str]
    model_path: Path


def list_gguf_models() -> list[Path]:
    """Return all .gguf files in MODELS_DIR, sorted by name."""
    if not MODELS_DIR.is_dir():
        return []
    return sorted(p for p in MODELS_DIR.iterdir() if p.suffix == ".gguf" and p.is_file())


def detect_running_server() -> RunningServer | None:
    """Locate a running llama-server by scanning /proc/*/cmdline.

    Returns None if no llama-server is running (e.g. Ollama-only setup,
    or the user runs the server on a different machine).
    """
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return None

    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        cmdline_path = entry / "cmdline"
        try:
            raw = cmdline_path.read_bytes()
        except OSError:
            continue
        if not raw:
            continue
        argv = raw.rstrip(b"\0").split(b"\0")
        if not argv:
            continue
        try:
            decoded_argv = [a.decode("utf-8", errors="replace") for a in argv]
        except Exception:  # noqa: BLE001
            continue
        exe = decoded_argv[0]
        if not exe.endswith(_LLAMA_SERVER_EXEC):
            continue

        model_path: Path | None = None
        for i, arg in enumerate(decoded_argv):
            if arg == "-m" and i + 1 < len(decoded_argv):
                model_path = Path(decoded_argv[i + 1])
                break

        return RunningServer(
            pid=int(entry.name),
            exe=exe,
            argv=decoded_argv,
            model_path=model_path or Path(""),
        )

    return None


def _rebuild_argv(argv: list[str], new_model: Path) -> list[str]:
    """Return a copy of argv with the -m value replaced by new_model."""
    out: list[str] = []
    i = 0
    replaced = False
    while i < len(argv):
        out.append(argv[i])
        if argv[i] == "-m" and i + 1 < len(argv):
            out.append(str(new_model))
            replaced = True
            i += 2
            continue
        i += 1
    if not replaced:
        out.extend(["-m", str(new_model)])
    return out


def _wait_for_ready(base_url: str, timeout_s: int) -> bool:
    """Poll /v1/models until it answers 200 or the timeout expires."""
    deadline = time.monotonic() + timeout_s
    url = base_url.rstrip("/") + "/models"
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(_READY_POLL_INTERVAL_S)
    return False


def _terminate(pid: int, grace_s: float = 5.0) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.2)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _port_free(host: str, port: int, timeout_s: float) -> bool:
    """Wait until TCP ``host:port`` refuses connections.

    Returns True when connect() raises ConnectionRefusedError — i.e. the
    previous server released the port. Returns False if someone is still
    listening after ``timeout_s``.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                pass
        except (ConnectionRefusedError, OSError):
            return True
        time.sleep(0.2)
    return False


def _reap_zombies() -> None:
    """Reap any child processes in <defunct> state.

    llama-server dying while we're its parent (Streamlit spawned it)
    turns into a zombie entry that detect_running_server() skips.
    Clearing it keeps ps output honest and frees the PID slot.
    """
    try:
        while True:
            pid, _status = os.waitpid(-1, os.WNOHANG)
            if pid == 0:
                break
    except ChildProcessError:
        pass


def _spawn(argv: list[str], env: dict[str, str]) -> subprocess.Popen:
    """Start llama-server, redirecting its output to LLAMA_LOG_PATH.

    stderr is merged into stdout so diagnostics from OOM, missing CUDA
    libs, or bad flags land in one file that the UI can tail on failure.
    """
    LLAMA_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Append so that previous attempts stay in the log for debugging.
    log = LLAMA_LOG_PATH.open("ab")
    return subprocess.Popen(
        argv,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _tail_log(n_lines: int = 40) -> str:
    """Return the last ``n_lines`` lines of LLAMA_LOG_PATH, or ''."""
    if not LLAMA_LOG_PATH.is_file():
        return ""
    try:
        data = LLAMA_LOG_PATH.read_bytes()
    except OSError:
        return ""
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    return "\n".join(lines[-n_lines:])


def _host_port(base_url: str) -> tuple[str, int]:
    parsed = urlparse(base_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return host, port


def switch_model(target: Path, base_url: str) -> tuple[bool, str]:
    """Stop the current llama-server and start a new one for ``target``.

    On failure to bring up the new model, rolls back to the previously
    running model so the user is never left without any LLM. Returns
    ``(ok, message)``. Must NOT be called while the app is doing active
    work against the LLM — the caller gates this.
    """
    current = detect_running_server()
    if current is None:
        return False, "llama-server не обнаружен — нечем управлять"

    if not target.is_file():
        return False, f"Файл модели не найден: {target}"

    if current.model_path.resolve() == target.resolve():
        return True, f"Модель {target.name} уже загружена"

    saved_argv = list(current.argv)
    saved_model = current.model_path
    exe_dir = str(Path(current.exe).parent)
    env = {**os.environ, "LD_LIBRARY_PATH": exe_dir}
    host, port = _host_port(base_url)

    # 1. Stop the old server and wait for the port + VRAM to be released.
    logger.info("Stopping llama-server PID=%d", current.pid)
    _terminate(current.pid)
    _reap_zombies()
    if not _port_free(host, port, _PORT_WAIT_S):
        return False, (
            f"Порт {port} всё ещё занят после остановки старого сервера — "
            "переключение отменено."
        )
    time.sleep(_VRAM_RELEASE_WAIT_S)

    # 2. Start the new server.
    new_argv = _rebuild_argv(saved_argv, target)
    logger.info("Starting llama-server with model=%s", target.name)
    try:
        proc = _spawn(new_argv, env)
    except OSError as exc:
        new_fail_msg = f"Не удалось запустить llama-server: {exc}"
        return _rollback(saved_argv, saved_model, env, base_url, host, port, new_fail_msg)

    # 3. Wait for /v1/models to answer.
    if _wait_for_ready(base_url, _SWITCH_TIMEOUT_S):
        _reap_zombies()
        return True, f"Загружена модель {target.name}"

    # 4. New model failed to come up — gather diagnostics, kill remnants,
    #    then roll back to the old model.
    rc = proc.poll()
    tail = _tail_log()
    _terminate(proc.pid)
    _reap_zombies()

    head = (
        f"Не удалось загрузить {target.name}"
        + (f" (rc={rc})" if rc is not None else " (процесс не завершился сам)")
    )
    return _rollback(saved_argv, saved_model, env, base_url, host, port, head, tail)


def _rollback(
    saved_argv: list[str],
    saved_model: Path,
    env: dict[str, str],
    base_url: str,
    host: str,
    port: int,
    head: str,
    tail: str = "",
) -> tuple[bool, str]:
    """Try to restart the previously running model after a failed switch."""
    if not _port_free(host, port, _PORT_WAIT_S):
        return False, f"{head}. Порт {port} не освободился — откат невозможен."
    time.sleep(_VRAM_RELEASE_WAIT_S)

    rollback_argv = _rebuild_argv(saved_argv, saved_model)
    logger.info("Rolling back to previous model=%s", saved_model.name)
    try:
        _spawn(rollback_argv, env)
    except OSError as exc:
        return False, (
            f"{head}. Откат на {saved_model.name} тоже не удался: {exc}. "
            f"Лог: {LLAMA_LOG_PATH}"
        )

    suffix = f"\n\nЛог (последние строки):\n{tail}" if tail else ""
    if _wait_for_ready(base_url, _SWITCH_TIMEOUT_S):
        _reap_zombies()
        return False, f"{head}. Откатились на {saved_model.name}.{suffix}"

    return False, (
        f"{head}, и предыдущая модель {saved_model.name} тоже не поднялась. "
        f"Запустите llama-server вручную. Лог: {LLAMA_LOG_PATH}{suffix}"
    )
