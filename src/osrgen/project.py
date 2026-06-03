from __future__ import annotations

from pathlib import Path
import json
from typing import Any


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
