"""ONNX MatMul lowering helpers."""

from __future__ import annotations

from MAPS.core.graph import OpKind
from MAPS.core.tensor import Tensor
from MAPS.ops.gemm import GemmLayerOp

from .types import OnnxLoweringFn


def lower_matmul_node(
    node_name: str,
    inputs: tuple[Tensor, ...],
    outputs: tuple[Tensor, ...],
) -> tuple[OpKind, object]:
    """Lower one ONNX MatMul node into scheduler-side GEMM semantics."""

    if len(inputs) != 2:
        raise ValueError(f"MatMul node '{node_name}' must have exactly 2 inputs")
    if len(outputs) != 1:
        raise ValueError(f"MatMul node '{node_name}' must have exactly 1 output")

    x_tensor, w_tensor = inputs
    output_tensor = outputs[0]
    return (
        OpKind.GEMM,
        GemmLayerOp(
            x=x_tensor,
            w=w_tensor,
            y=None,
            output=output_tensor,
        ),
    )


ONNX_OP_LOWERERS: dict[str, OnnxLoweringFn] = {
    "MatMul": lower_matmul_node,
}
