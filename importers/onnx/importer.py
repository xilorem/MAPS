"""ONNX importer entry points."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from MAPS.core.graph import Graph

from .graph_parser import parse_graph

if TYPE_CHECKING:
    from onnx import ModelProto


def load_onnx_model(path: str | Path) -> "ModelProto":
    """Load and validate one ONNX model from disk."""

    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "The optional 'onnx' package is required to load ONNX models"
        ) from exc

    model_path = Path(path)
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    return model


def import_onnx_graph(path: str | Path) -> Graph:
    """Import one ONNX model directly into the shared scheduler graph IR."""

    model = load_onnx_model(path)
    return parse_graph(model.graph, graph_name=model.graph.name or Path(path).stem)
