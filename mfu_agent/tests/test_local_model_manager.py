"""Unit tests for ui.local_model_manager.switch_model.

Covers the happy path and the rollback paths that matter when the new
model fails to come up (OOM on Qwen-32B was the real trigger for adding
these safeguards).
"""

from __future__ import annotations

import socket
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ui import local_model_manager as lmm
from ui.local_model_manager import RunningServer

# ── Pure helpers ─────────────────────────────────────────────────────────────


def test_rebuild_argv_replaces_m() -> None:
    argv = ["llama-server", "-m", "/models/old.gguf", "--port", "8000", "-ngl", "99"]
    out = lmm._rebuild_argv(argv, Path("/models/new.gguf"))
    assert out == [
        "llama-server", "-m", "/models/new.gguf", "--port", "8000", "-ngl", "99",
    ]


def test_rebuild_argv_adds_m_when_missing() -> None:
    argv = ["llama-server", "--port", "8000"]
    out = lmm._rebuild_argv(argv, Path("/models/new.gguf"))
    assert out == ["llama-server", "--port", "8000", "-m", "/models/new.gguf"]


def test_port_free_true_when_nobody_listening() -> None:
    # Pick a port nobody should be on.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]
    # Port is released now.
    assert lmm._port_free("127.0.0.1", free_port, timeout_s=1.0) is True


def test_port_free_false_when_occupied(monkeypatch) -> None:
    """When connect() keeps succeeding, _port_free gives up and returns False."""
    class _Stub:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(lmm.socket, "create_connection", lambda *a, **kw: _Stub())
    monkeypatch.setattr(lmm.time, "sleep", lambda _s: None)
    assert lmm._port_free("127.0.0.1", 9999, timeout_s=0.5) is False


def test_tail_log_reads_last_n_lines(monkeypatch, tmp_path: Path) -> None:
    log = tmp_path / "llama.log"
    log.write_text("A\nB\nC\nD\nE\n", encoding="utf-8")
    monkeypatch.setattr(lmm, "LLAMA_LOG_PATH", log)
    assert lmm._tail_log(3) == "C\nD\nE"


def test_tail_log_missing_file_returns_empty(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(lmm, "LLAMA_LOG_PATH", tmp_path / "nope.log")
    assert lmm._tail_log() == ""


def test_host_port_parsing() -> None:
    assert lmm._host_port("http://localhost:8000/v1") == ("localhost", 8000)
    assert lmm._host_port("http://127.0.0.1:9999") == ("127.0.0.1", 9999)
    assert lmm._host_port("https://example.com/v1")[1] == 443


# ── switch_model guard clauses ───────────────────────────────────────────────


def test_switch_model_no_server(monkeypatch) -> None:
    monkeypatch.setattr(lmm, "detect_running_server", lambda: None)
    ok, msg = lmm.switch_model(Path("/models/x.gguf"), "http://localhost:8000/v1")
    assert ok is False
    assert "не обнаружен" in msg


def test_switch_model_missing_file(monkeypatch, tmp_path: Path) -> None:
    rs = RunningServer(
        pid=1, exe="/opt/llama-server",
        argv=["llama-server", "-m", "/models/old.gguf"],
        model_path=tmp_path / "old.gguf",
    )
    (tmp_path / "old.gguf").write_bytes(b"x")
    monkeypatch.setattr(lmm, "detect_running_server", lambda: rs)
    ok, msg = lmm.switch_model(tmp_path / "missing.gguf", "http://localhost:8000/v1")
    assert ok is False
    assert "не найден" in msg


def test_switch_model_same_model(monkeypatch, tmp_path: Path) -> None:
    model = tmp_path / "same.gguf"
    model.write_bytes(b"x")
    rs = RunningServer(
        pid=1, exe="/opt/llama-server",
        argv=["llama-server", "-m", str(model)],
        model_path=model,
    )
    monkeypatch.setattr(lmm, "detect_running_server", lambda: rs)
    ok, msg = lmm.switch_model(model, "http://localhost:8000/v1")
    assert ok is True
    assert "уже загружена" in msg


# ── switch_model success and failure paths ──────────────────────────────────


@pytest.fixture
def running_server(tmp_path: Path) -> tuple[RunningServer, Path, Path]:
    old = tmp_path / "old.gguf"
    new = tmp_path / "new.gguf"
    old.write_bytes(b"x")
    new.write_bytes(b"x")
    rs = RunningServer(
        pid=42,
        exe="/opt/llama/llama-server",
        argv=["llama-server", "-m", str(old), "--port", "8000"],
        model_path=old,
    )
    return rs, old, new


def _make_popen(pid: int = 9999, returncode: int | None = None) -> MagicMock:
    proc = MagicMock()
    proc.pid = pid
    proc.poll = MagicMock(return_value=returncode)
    return proc


def _install_common_mocks(monkeypatch, rs: RunningServer, *, ready_sequence: list[bool]) -> list:
    """Install mocks shared by success/failure tests.

    Returns the list of Popen calls so tests can assert argv.
    """
    monkeypatch.setattr(lmm, "detect_running_server", lambda: rs)
    monkeypatch.setattr(lmm, "_terminate", lambda pid, grace_s=5.0: None)
    monkeypatch.setattr(lmm, "_reap_zombies", lambda: None)
    monkeypatch.setattr(lmm, "_port_free", lambda host, port, timeout_s: True)
    monkeypatch.setattr(lmm.time, "sleep", lambda _s: None)

    ready_iter = iter(ready_sequence)
    monkeypatch.setattr(lmm, "_wait_for_ready", lambda url, t: next(ready_iter))

    popen_calls: list = []

    def _fake_spawn(argv, env):
        popen_calls.append(argv)
        return _make_popen(pid=90000 + len(popen_calls))

    monkeypatch.setattr(lmm, "_spawn", _fake_spawn)
    return popen_calls


def test_switch_model_happy_path(monkeypatch, running_server) -> None:
    rs, _old, new = running_server
    calls = _install_common_mocks(monkeypatch, rs, ready_sequence=[True])

    ok, msg = lmm.switch_model(new, "http://localhost:8000/v1")
    assert ok is True
    assert "Загружена модель" in msg
    assert new.name in msg
    # Started exactly once with the new model.
    assert len(calls) == 1
    assert str(new) in calls[0]


def test_switch_model_new_fails_rollback_ok(monkeypatch, running_server, tmp_path: Path) -> None:
    rs, old, new = running_server
    # Seed a fake log so _tail_log returns something visible.
    log = tmp_path / "llama.log"
    log.write_text("loading model...\nCUDA OOM: unable to alloc 19 GiB\n", encoding="utf-8")
    monkeypatch.setattr(lmm, "LLAMA_LOG_PATH", log)

    # Ready: first attempt (new model) → False; rollback (old model) → True.
    calls = _install_common_mocks(monkeypatch, rs, ready_sequence=[False, True])

    ok, msg = lmm.switch_model(new, "http://localhost:8000/v1")
    assert ok is False
    assert "Откатились" in msg
    assert old.name in msg
    assert "CUDA OOM" in msg  # tail of the log made it into the message
    # Two spawn calls: new model, then rollback with old.
    assert len(calls) == 2
    assert str(new) in calls[0]
    assert str(old) in calls[1]


def test_switch_model_both_fail(monkeypatch, running_server) -> None:
    rs, old, new = running_server
    _install_common_mocks(monkeypatch, rs, ready_sequence=[False, False])

    ok, msg = lmm.switch_model(new, "http://localhost:8000/v1")
    assert ok is False
    assert old.name in msg
    assert "не поднялась" in msg


def test_switch_model_port_stuck(monkeypatch, running_server) -> None:
    rs, _old, new = running_server
    monkeypatch.setattr(lmm, "detect_running_server", lambda: rs)
    monkeypatch.setattr(lmm, "_terminate", lambda pid, grace_s=5.0: None)
    monkeypatch.setattr(lmm, "_reap_zombies", lambda: None)
    monkeypatch.setattr(lmm, "_port_free", lambda host, port, timeout_s: False)
    monkeypatch.setattr(lmm.time, "sleep", lambda _s: None)
    # _spawn / _wait_for_ready should never be reached.
    monkeypatch.setattr(
        lmm, "_spawn",
        lambda *a, **kw: pytest.fail("_spawn must not be called when port is stuck"),
    )

    ok, msg = lmm.switch_model(new, "http://localhost:8000/v1")
    assert ok is False
    assert "занят" in msg or "не освободился" in msg
