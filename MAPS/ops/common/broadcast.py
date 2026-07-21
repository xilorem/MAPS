"""ONNX-compatible multidirectional broadcasting helpers."""

from __future__ import annotations

from MAPS.core.layout import TensorRange, TensorSlice
from MAPS.core.tensor import Tensor


def broadcast_shape(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    """Return the exact multidirectional broadcast result for ``shapes``."""

    if not shapes:
        raise ValueError("at least one shape is required for broadcasting")

    result: list[int] = []
    max_rank = max(len(shape) for shape in shapes)
    padded_shapes = ((1,) * (max_rank - len(shape)) + shape for shape in shapes)
    for dimensions in zip(*padded_shapes):
        non_unit = {dimension for dimension in dimensions if dimension != 1}
        if len(non_unit) > 1:
            raise ValueError(f"shapes are not broadcast-compatible: {shapes}")
        result.append(next(iter(non_unit), 1))
    return tuple(result)


def validate_broadcast_output(
    inputs: tuple[Tensor, ...],
    output: Tensor,
    op_name: str,
) -> None:
    """Require ``output`` to have exactly the broadcast result shape."""

    expected = broadcast_shape(*(tensor.dims for tensor in inputs))
    if output.dims != expected:
        raise ValueError(
            f"{op_name} output shape must be the broadcast result {expected}, "
            f"got {output.dims}"
        )


def validate_broadcastable_to(
    input_tensor: Tensor,
    output: Tensor,
    op_name: str,
) -> None:
    """Require one input to be unidirectionally broadcastable to ``output``."""

    if input_tensor.rank > output.rank:
        raise ValueError(f"{op_name} input rank cannot exceed output rank")
    padded = (1,) * (output.rank - input_tensor.rank) + input_tensor.dims
    if any(
        input_dim not in (1, output_dim)
        for input_dim, output_dim in zip(padded, output.dims)
    ):
        raise ValueError(f"{op_name} input shape is not broadcastable to output")


def broadcast_input_slice(
    input_tensor: Tensor,
    output: Tensor,
    output_slice: TensorSlice,
    op_name: str,
) -> TensorSlice:
    """Map an output slice back to the required slice of one broadcast input."""

    validate_broadcastable_to(input_tensor, output, op_name)
    rank_offset = output.rank - input_tensor.rank
    dims = tuple(
        TensorRange(start=0, length=1)
        if input_dim == 1
        else output_slice.dims[input_axis + rank_offset]
        for input_axis, input_dim in enumerate(input_tensor.dims)
    )
    return TensorSlice(rank=input_tensor.rank, dims=dims)
