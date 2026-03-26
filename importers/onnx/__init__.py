"""ONNX frontend for the scheduler IR."""

from .importer import import_onnx_graph, load_onnx_model

__all__ = [
    "import_onnx_graph",
    "load_onnx_model",
]
