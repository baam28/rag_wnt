"""LLM API usage tracking (PostgreSQL-backed)."""

import time

from pg_client import get_llm_usage_collection


def record_usage(prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
    """Record one LLM request usage in PostgreSQL."""
    if prompt_tokens == 0 and completion_tokens == 0:
        return
    day = time.strftime("%Y-%m-%d", time.gmtime())
    coll = get_llm_usage_collection()
    coll.update_one(
        {"date": day},
        {
            "$setOnInsert": {"date": day},
            "$inc": {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "requests": 1,
            },
        },
        upsert=True,
    )


def get_usage(days: int = 30) -> dict:
    """Return total usage and daily breakdown for the last `days` days."""
    coll = get_llm_usage_collection()

    docs = coll.find({})

    daily_list = [
        {
            "date": d.get("date", ""),
            "prompt_tokens": int(d.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(d.get("completion_tokens", 0) or 0),
            "requests": int(d.get("requests", 0) or 0),
        }
        for d in docs
    ]

    agg = coll.aggregate([])
    if agg:
        total = {
            "prompt_tokens": int(agg[0].get("prompt_tokens", 0) or 0),
            "completion_tokens": int(agg[0].get("completion_tokens", 0) or 0),
            "requests": int(agg[0].get("requests", 0) or 0),
        }
    else:
        total = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}

    return {"total": total, "daily": daily_list}
