"""JSON export helpers for planned pipelines."""

from __future__ import annotations

from dataclasses import asdict
from enum import Enum
import json
from pathlib import Path

from MAPS.pipeline.pipeline import Pipeline


def _to_jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_to_jsonable(item) for item in sorted(value, key=lambda item: repr(item))]
    return str(value)




def write_pipeline_json(pipeline: Pipeline, output_path: str | Path) -> Path:
    """Write one pipeline object to JSON and return its path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_jsonable(asdict(pipeline))
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
