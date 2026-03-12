"""LLM API usage tracking (file-backed)."""

import json
import threading
import time
from pathlib import Path

_USAGE_FILE = Path(__file__).resolve().parent.parent / "llm_usage.json"
_lock = threading.Lock()


def _load() -> dict:
    if _USAGE_FILE.exists():
        try:
            with open(_USAGE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"total": {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}, "daily": {}}


def _save(data: dict) -> None:
    with open(_USAGE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def record_usage(prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
    """Record one LLM request usage. Thread-safe."""
    if prompt_tokens == 0 and completion_tokens == 0:
        return
    day = time.strftime("%Y-%m-%d", time.gmtime())
    with _lock:
        data = _load()
        data["total"]["prompt_tokens"] = data["total"].get("prompt_tokens", 0) + prompt_tokens
        data["total"]["completion_tokens"] = data["total"].get("completion_tokens", 0) + completion_tokens
        data["total"]["requests"] = data["total"].get("requests", 0) + 1
        if day not in data["daily"]:
            data["daily"][day] = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}
        data["daily"][day]["prompt_tokens"] = data["daily"][day].get("prompt_tokens", 0) + prompt_tokens
        data["daily"][day]["completion_tokens"] = data["daily"][day].get("completion_tokens", 0) + completion_tokens
        data["daily"][day]["requests"] = data["daily"][day].get("requests", 0) + 1
        _save(data)


def get_usage(days: int = 30) -> dict:
    """Return total usage and daily breakdown for the last `days` days."""
    with _lock:
        data = _load()
    total = data.get("total", {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0})
    daily = data.get("daily", {})
    # Sort days descending and take last `days`
    sorted_days = sorted(daily.keys(), reverse=True)[:days]
    daily_list = [{"date": d, **daily[d]} for d in sorted_days]
    return {"total": total, "daily": daily_list}
