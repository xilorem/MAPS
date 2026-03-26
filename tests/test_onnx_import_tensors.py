"""Tests for ONNX tensor import."""

from MAPS.importers.onnx.tensor_parser import onnx_dtype_elem_bytes


def test_onnx_dtype_elem_bytes_maps_common_float32() -> None:
    assert onnx_dtype_elem_bytes(1) == 4
