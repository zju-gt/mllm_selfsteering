"""Atomic JSON/JSONL/tensor persistence and content hashes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, TextIO
import uuid

import torch
from safetensors.torch import save_file


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{number}")
            yield value


def repair_truncated_jsonl(path: Path | str) -> bool:
    """Repair only an unterminated final JSONL record after an abrupt shutdown.

    A syntactically complete final value merely receives its missing newline so the
    normal reader can still validate its object type. An invalid trailing fragment is
    discarded. Corruption in any newline-terminated record remains an error and is
    deliberately not hidden.
    """

    source = Path(path)
    if not source.is_file() or source.stat().st_size == 0:
        return False
    with source.open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) == b"\n":
            return False
        handle.seek(0)
        payload = handle.read()
        boundary = payload.rfind(b"\n") + 1
        fragment = payload[boundary:]
        try:
            json.loads(fragment.decode("utf-8"))
            valid_json = True
        except (UnicodeDecodeError, json.JSONDecodeError):
            valid_json = False
        if valid_json:
            handle.seek(0, os.SEEK_END)
            handle.write(b"\n")
        else:
            handle.truncate(boundary)
        handle.flush()
        os.fsync(handle.fileno())
    return True


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def write_jsonl(path: Path | str, rows: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(destination)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path: Path | str, row: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class JsonlCheckpointWriter:
    """Append buffered JSONL and fsync only at explicit checkpoint intervals."""

    def __init__(self, path: Path | str, *, sync_every: int = 100) -> None:
        if sync_every < 1:
            raise ValueError("sync_every must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: TextIO = self.path.open("a", encoding="utf-8", newline="\n")
        self._sync_every = int(sync_every)
        self._pending = 0
        self._closed = False

    def write(self, row: Mapping[str, Any]) -> None:
        if self._closed:
            raise ValueError("cannot write to a closed JsonlCheckpointWriter")
        self._handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        self._pending += 1
        if self._pending >= self._sync_every:
            self.checkpoint()

    def checkpoint(self) -> None:
        if self._closed or self._pending == 0:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._pending = 0

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.checkpoint()
        finally:
            self._handle.close()
            self._closed = True

    def __enter__(self) -> "JsonlCheckpointWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def atomic_save_json(path: Path | str, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(destination)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_save_tensors(path: Path | str, tensors: Mapping[str, torch.Tensor]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(destination)
    try:
        save_file(
            {key: value.detach().cpu().contiguous() for key, value in tensors.items()},
            str(temporary),
            metadata={"format": "mllm-self-steering-v1"},
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
