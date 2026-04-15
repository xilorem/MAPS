"""ONNX Gemm lowering helpers."""

from __future__ import annotations

from MAPS.core.graph import OpKind
from MAPS.core.tensor import Tensor
from MAPS.ops.gemm import GemmLayerOp

from .types import OnnxLoweringFn


def lower_gemm_node(
    node_name: str,
    inputs: tuple[Tensor, ...],
    outputs: tuple[Tensor, ...],
    attributes: dict[str, object],
) -> tuple[OpKind, object]:
    """Lower one ONNX Gemm node into scheduler-side GEMM semantics."""

    del attributes
    if len(inputs) not in (2, 3):
        raise ValueError(f"Gemm node '{node_name}' must have 2 or 3 inputs")
    if len(outputs) != 1:
        raise ValueError(f"Gemm node '{node_name}' must have exactly 1 output")

    x_tensor = inputs[0]
    w_tensor = inputs[1]
    y_tensor = inputs[2] if len(inputs) == 3 else None
    output_tensor = outputs[0]
    return (
        OpKind.GEMM,
        GemmLayerOp(
            x=x_tensor,
            w=w_tensor,
            y=y_tensor,
            output=output_tensor,
        ),
    )


ONNX_OP_LOWERERS: dict[str, OnnxLoweringFn] = {
    "Gemm": lower_gemm_node,
}
