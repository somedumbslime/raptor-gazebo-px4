from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class EventLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")

    def log(self, ts: float, event: str, **payload: Any) -> None:
        row = {"ts": float(ts), "event": event}
        row.update(payload)
        self._fh.write(json.dumps(row, ensure_ascii=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()
