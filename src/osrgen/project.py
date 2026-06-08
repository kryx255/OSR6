from __future__ import annotations

from pathlib import Path
import json
from typing import Any


WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
WINDOWS_INVALID_NAME_CHARS = frozenset('<>:"/\\|?*')


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_path_name(name: str, *, fallback: str = "output") -> str:
    cleaned = "".join(
        "_" if character in WINDOWS_INVALID_NAME_CHARS or ord(character) < 32 else character
        for character in name
    )
    tail_start = len(cleaned)
    while tail_start > 0 and cleaned[tail_start - 1] in {" ", "."}:
        tail_start -= 1
    if tail_start < len(cleaned):
        cleaned = f"{cleaned[:tail_start]}{'_' * (len(cleaned) - tail_start)}"
    if not cleaned or cleaned in {".", ".."}:
        cleaned = fallback

    reserved_check = cleaned.split(".", 1)[0].upper()
    if reserved_check in WINDOWS_RESERVED_NAMES:
        cleaned = f"_{cleaned}"
    return cleaned


def file_fingerprint(path: str | Path) -> dict[str, int]:
    stat = Path(path).stat()
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
