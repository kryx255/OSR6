from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Action:
    at: int
    pos: int

    def to_json(self) -> dict[str, int]:
        return {"at": int(self.at), "pos": clamp_pos(self.pos)}


@dataclass
class Funscript:
    actions: list[Action] = field(default_factory=list)
    version: str = "1.0"
    inverted: bool = False
    range: int = 90
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_actions(
        cls,
        actions: Iterable[Action | dict[str, Any] | tuple[int, int]],
        *,
        version: str = "1.0",
        inverted: bool = False,
        range: int = 90,
    ) -> "Funscript":
        parsed: list[Action] = []
        for action in actions:
            if isinstance(action, Action):
                parsed.append(Action(int(action.at), clamp_pos(action.pos)))
            elif isinstance(action, dict):
                parsed.append(Action(int(action["at"]), clamp_pos(action["pos"])))
            else:
                at, pos = action
                parsed.append(Action(int(at), clamp_pos(pos)))
        parsed.sort(key=lambda item: item.at)
        return cls(actions=dedupe_actions(parsed), version=version, inverted=inverted, range=range)

    @classmethod
    def load(cls, path: str | Path) -> "Funscript":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        actions = [Action(int(item["at"]), clamp_pos(item["pos"])) for item in data.get("actions", [])]
        known = {"actions", "version", "inverted", "range"}
        metadata = {key: value for key, value in data.items() if key not in known}
        return cls(
            actions=dedupe_actions(sorted(actions, key=lambda item: item.at)),
            version=str(data.get("version", "1.0")),
            inverted=bool(data.get("inverted", False)),
            range=int(data.get("range", 90)),
            metadata=metadata,
        )

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "version": self.version,
            "inverted": self.inverted,
            "range": self.range,
            "actions": [action.to_json() for action in self.actions],
        }
        data.update(self.metadata)
        return data

    @property
    def duration_ms(self) -> int:
        if not self.actions:
            return 0
        return max(action.at for action in self.actions)

    def sample_at(self, time_ms: int | float) -> float:
        if not self.actions:
            return 50.0
        t = float(time_ms)
        if t <= self.actions[0].at:
            return float(self.actions[0].pos)
        if t >= self.actions[-1].at:
            return float(self.actions[-1].pos)

        lo = 0
        hi = len(self.actions) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.actions[mid].at < t:
                lo = mid + 1
            else:
                hi = mid - 1

        right = self.actions[lo]
        left = self.actions[lo - 1]
        span = right.at - left.at
        if span <= 0:
            return float(right.pos)
        alpha = (t - left.at) / span
        return float(left.pos + (right.pos - left.pos) * alpha)


def clamp_pos(value: int | float) -> int:
    return max(0, min(100, int(round(float(value)))))


def dedupe_actions(actions: list[Action]) -> list[Action]:
    if not actions:
        return []
    by_time: dict[int, Action] = {}
    for action in actions:
        by_time[int(action.at)] = Action(int(action.at), clamp_pos(action.pos))
    return [by_time[key] for key in sorted(by_time)]
