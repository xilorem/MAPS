"""Exp lowering helper."""

from __future__ import annotations

from MAPS.core.graph import OpKind
from MAPS.core.tensor import Tensor
from MAPS.ops.exp import ExpLayerOp

from .types import OnnxLoweringFn


def lower_exp_node(
    node_name: str,
    inputs: tuple[Tensor, ...],
    outputs: tuple[Tensor, ...],
    attributes: dict[str, object],
) -> tuple[OpKind, object]:
    del attributes
    if len(inputs) != 1:
        raise ValueError(f"Exp node '{node_name}' must have exactly 1 input")
    if len(outputs) != 1:
        raise ValueError(f"Exp node '{node_name}' must have exactly 1 output")

    return (
        OpKind.EXP,
        ExpLayerOp(
            x=inputs[0],
            output=outputs[0],
        ),
    )


ONNX_OP_LOWERERS: dict[str, OnnxLoweringFn] = {
    "Exp": lower_exp_node,
}
