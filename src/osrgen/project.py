from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Any, Iterable


def video_output_dir(video_path: str | Path, output_root: str | Path) -> Path:
    src = Path(video_path)
    return Path(output_root) / src.stem


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def file_fingerprint(path: str | Path) -> dict[str, int]:
    stat = Path(path).stat()
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def file_signature(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_manifest_signature(root: str | Path, paths: Iterable[str | Path]) -> str:
    base = Path(root).resolve()
    digest = hashlib.sha256()
    resolved_paths = [Path(path).resolve() for path in paths]
    for resolved in sorted(resolved_paths, key=lambda item: item.as_posix().lower()):
        relative = resolved.relative_to(base).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_signature(resolved).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()
