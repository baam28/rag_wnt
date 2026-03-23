"""Reporting helpers for RAGAS evaluation runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def ensure_output_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _numeric_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                acc.setdefault(key, []).append(float(value))
    return {key: (sum(vals) / len(vals)) for key, vals in acc.items() if vals}


def serialize_result(result: Any) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Normalize ragas evaluate() result into rows + aggregate means."""
    rows: list[dict[str, Any]] = []
    if hasattr(result, "to_pandas"):
        dataframe = result.to_pandas()
        rows = dataframe.to_dict(orient="records")
    elif isinstance(result, dict):
        rows = [result]
    elif isinstance(result, list):
        rows = [r for r in result if isinstance(r, dict)]

    summary = _numeric_summary(rows)
    return rows, summary


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _to_jsonable(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, payload: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as file_obj:
        json.dump(_to_jsonable(payload), file_obj, ensure_ascii=False, indent=2)


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        with Path(path).open("w", encoding="utf-8") as file_obj:
            file_obj.write("")
        return

    headers = sorted({key for row in rows for key in row.keys()})
    with Path(path).open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
