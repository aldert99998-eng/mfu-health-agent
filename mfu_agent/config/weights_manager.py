"""Weights profile management — CRUD, diffing, and audit history."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from config.loader import WEIGHTS_DIR, ConfigValidationError
from data_io.models import WeightsProfile

PROFILES_DIR = WEIGHTS_DIR / "profiles"
HISTORY_LOG = WEIGHTS_DIR / "history.log"
DEFAULT_YAML = WEIGHTS_DIR / "default.yaml"


# ── Models ───────────────────────────────────────────────────────────────────


class ProfileMeta(BaseModel):
    """Metadata for a single weights profile."""

    model_config = ConfigDict(frozen=True)

    name: str
    created_at: datetime
    modified_at: datetime
    author: str | None = None
    params_hash: str


class Diff(BaseModel):
    """Single parameter difference between two profiles."""

    model_config = ConfigDict(frozen=True)

    path: str
    value_a: Any = None
    value_b: Any = None


class HistoryEntry(BaseModel):
    """Single append-only record in history.log (JSONL)."""

    timestamp: datetime
    author: str
    profile_name: str
    params_hash: str
    action: str = "save"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _canonical_hash(profile: WeightsProfile) -> str:
    blob = json.dumps(
        profile.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigValidationError(path, f"File not found: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigValidationError(path, f"Invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigValidationError(path, "Expected a YAML mapping")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _flat_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dict into dot-separated keys."""
    items: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flat_dict(v, key))
        else:
            items[key] = v
    return items


# ── WeightsManager ───────────────────────────────────────────────────────────


class WeightsManager:
    """CRUD for weights profiles with audit trail and file locking."""

    def __init__(
        self,
        profiles_dir: Path = PROFILES_DIR,
        history_path: Path = HISTORY_LOG,
        default_path: Path = DEFAULT_YAML,
    ) -> None:
        self._profiles_dir = profiles_dir
        self._history_path = history_path
        self._default_path = default_path

    # ── name validation ────────────────────────────────────────────────

    @staticmethod
    def _validate_name(name: str) -> None:
        if not name or "/" in name or "\\" in name or ".." in name or "\x00" in name:
            raise ValueError(
                f"Invalid profile name '{name}': "
                "must not contain '/', '\\', '..', or null bytes"
            )
        if name != name.strip() or name.startswith("."):
            raise ValueError(
                f"Invalid profile name '{name}': "
                "must not start with '.' or have leading/trailing whitespace"
            )

    # ── 1. list_profiles ─────────────────────────────────────────────────

    def list_profiles(self) -> list[ProfileMeta]:
        self._profiles_dir.mkdir(parents=True, exist_ok=True)
        result: list[ProfileMeta] = []
        for path in sorted(self._profiles_dir.glob("*.yaml")):
            try:
                meta = self._read_meta(path)
                result.append(meta)
            except Exception:
                continue
        return result

    # ── 2. load_profile ──────────────────────────────────────────────────

    def load_profile(self, name: str) -> WeightsProfile:
        self._validate_name(name)
        path = self._profiles_dir / f"{name}.yaml"
        data = _read_yaml(path)
        try:
            return WeightsProfile.model_validate(data)
        except Exception as exc:
            raise ConfigValidationError(path, str(exc)) from exc

    # ── 3. save_profile ──────────────────────────────────────────────────

    def save_profile(self, profile: WeightsProfile, author: str) -> Path:
        self._validate_name(profile.profile_name)
        self._profiles_dir.mkdir(parents=True, exist_ok=True)
        path = self._profiles_dir / f"{profile.profile_name}.yaml"

        data = profile.model_dump(mode="json")
        params_hash = _canonical_hash(profile)
        now = datetime.now(UTC)

        _write_yaml(path, data)

        entry = HistoryEntry(
            timestamp=now,
            author=author,
            profile_name=profile.profile_name,
            params_hash=params_hash,
        )
        self._append_history(entry)

        return path

    # ── 4. compare_profiles ──────────────────────────────────────────────

    def compare_profiles(self, name_a: str, name_b: str) -> list[Diff]:
        a = self.load_profile(name_a)
        b = self.load_profile(name_b)

        flat_a = _flat_dict(a.model_dump(mode="json"))
        flat_b = _flat_dict(b.model_dump(mode="json"))

        all_keys = sorted(set(flat_a) | set(flat_b))
        diffs: list[Diff] = []
        for key in all_keys:
            va = flat_a.get(key)
            vb = flat_b.get(key)
            if va != vb:
                diffs.append(Diff(path=key, value_a=va, value_b=vb))
        return diffs

    # ── 5. reset_to_default ──────────────────────────────────────────────

    def reset_to_default(self) -> WeightsProfile:
        data = _read_yaml(self._default_path)
        try:
            return WeightsProfile.model_validate(data)
        except Exception as exc:
            raise ConfigValidationError(self._default_path, str(exc)) from exc

    # ── internals ────────────────────────────────────────────────────────

    def _read_meta(self, path: Path) -> ProfileMeta:
        stat = path.stat()
        data = _read_yaml(path)
        profile = WeightsProfile.model_validate(data)
        return ProfileMeta(
            name=path.stem,
            created_at=datetime.fromtimestamp(stat.st_ctime, tz=UTC),
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
            author=self._find_last_author(path.stem),
            params_hash=_canonical_hash(profile),
        )

    def _find_last_author(self, profile_name: str) -> str | None:
        if not self._history_path.exists():
            return None
        last_author: str | None = None
        with open(self._history_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if record.get("profile_name") == profile_name:
                        last_author = record.get("author")
                except json.JSONDecodeError:
                    continue
        return last_author

    def _append_history(self, entry: HistoryEntry) -> None:
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        line = entry.model_dump_json() + "\n"

        fd = os.open(
            str(self._history_path),
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            os.write(fd, line.encode("utf-8"))
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    # ── dunder ───────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"WeightsManager(profiles_dir={self._profiles_dir!r}, "
            f"history={self._history_path!r})"
        )
