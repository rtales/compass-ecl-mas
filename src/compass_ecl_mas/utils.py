from __future__ import annotations
from pathlib import Path
import json, hashlib, time
from dataclasses import dataclass

def ensure_dir(p: str | Path) -> Path:
    pp = Path(p)
    pp.mkdir(parents=True, exist_ok=True)
    return pp

def now_tag() -> str:
    import datetime as dt
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def sha1_of_dict(d: dict) -> str:
    b = json.dumps(d, sort_keys=True).encode("utf-8")
    return hashlib.sha1(b).hexdigest()

@dataclass
class Timer:
    t0: float = 0.0
    dt: float = 0.0
    def __enter__(self):
        self.t0 = time.time()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.dt = time.time() - self.t0
