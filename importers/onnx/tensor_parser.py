"""ONNX tensor parsing helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from MAPS.core.tensor import Tensor

if TYPE_CHECKING:
    from onnx import ValueInfoProto, TensorProto, GraphProto
    


_ONNX_DTYPE_ELEM_BYTES: dict[int, int] = {
    1: 4,   # FLOAT
    2: 1,   # UINT8
    3: 1,   # INT8
    4: 2,   # UINT16
    5: 2,   # INT16
    6: 4,   # INT32
    7: 8,   # INT64
    9: 1,   # BOOL
    10: 2,  # FLOAT16
    11: 8,  # DOUBLE
    12: 4,  # UINT32
    13: 8,  # UINT64
    14: 8,  # COMPLEX64
    15: 16, # COMPLEX128
    16: 2,  # BFLOAT16
}


def onnx_dtype_elem_bytes(dtype: int) -> int | None:
    """Return the element size in bytes for one ONNX tensor dtype."""

    return _ONNX_DTYPE_ELEM_BYTES.get(dtype)


def parse_value_shape(value: "ValueInfoProto") -> tuple[int, ...]:
    """Extract a concrete shape from ONNX value info when available.

    If any dimension is symbolic or unknown, return an empty shape for now.
    """

    tensor_type = value.type.tensor_type
    if not tensor_type.HasField("shape"):
        return ()

    dims: list[int] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value") and dim.dim_value > 0:
            dims.append(dim.dim_value)
            continue
        return ()
    return tuple(dims)


def parse_value_tensor(value: "ValueInfoProto",
                       *,
                       is_input: bool = False,
                       is_output: bool = False) -> tuple[str, tuple[int, ...], int | None]:
    """Extract tensor metadata from one ONNX value-info entry."""

    tensor_type = value.type.tensor_type
    elem_type = tensor_type.elem_type if tensor_type.HasField("elem_type") else 0
    _ = is_input, is_output
    return value.name, parse_value_shape(value), onnx_dtype_elem_bytes(elem_type)


def parse_initializer_tensor(initializer: "TensorProto") -> tuple[str, tuple[int, ...], int | None]:
    """Extract tensor metadata from one ONNX initializer."""

    return (
        initializer.name,
        tuple(int(dim) for dim in initializer.dims),
        onnx_dtype_elem_bytes(initializer.data_type),
    )


def _merge_tensor_metadata(
    metadata: dict[str, dict[str, object]],
    name: str,
    shape: tuple[int, ...],
    elem_bytes: int | None,
) -> None:
    """Merge one shape / dtype observation into the graph tensor metadata table."""

    record = metadata.setdefault(name, {"shape": (), "elem_bytes": None})
    if not record["shape"] and shape:
        record["shape"] = shape
    if record["elem_bytes"] is None and elem_bytes is not None:
        record["elem_bytes"] = elem_bytes


def collect_scheduler_tensors(graph: "GraphProto") -> dict[str, Tensor]:
    """Collect scheduler-side logical tensors from one ONNX graph."""

    metadata: dict[str, dict[str, object]] = {}

    for value in graph.input:
        name, shape, elem_bytes = parse_value_tensor(value, is_input=True)
        _merge_tensor_metadata(metadata, name, shape, elem_bytes)

    for value in graph.output:
        name, shape, elem_bytes = parse_value_tensor(value, is_output=True)
        _merge_tensor_metadata(metadata, name, shape, elem_bytes)

    for value in graph.value_info:
        name, shape, elem_bytes = parse_value_tensor(value)
        _merge_tensor_metadata(metadata, name, shape, elem_bytes)

    for initializer in graph.initializer:
        name, shape, elem_bytes = parse_initializer_tensor(initializer)
        metadata[name] = {
            "shape": shape,
            "elem_bytes": elem_bytes,
        }

    tensors: dict[str, Tensor] = {}
    for name, record in metadata.items():
        shape = record["shape"]
        elem_bytes = record["elem_bytes"]
        if not shape or elem_bytes is None:
            continue
        tensors[name] = Tensor(
            name=name,
            rank=len(shape),
            dims=shape,
            elem_bytes=elem_bytes,
        )

    return tensors
