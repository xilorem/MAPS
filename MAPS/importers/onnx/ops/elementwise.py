"""Table-driven ONNX elementwise lowerers."""

from __future__ import annotations

from MAPS.arch import WorkKind
from MAPS.core.graph import OpKind
from MAPS.core.tensor import Tensor
from MAPS.ops.elementwise import BinaryElementwiseOp, UnaryElementwiseOp

from .types import OnnxLoweringFn

UNARY_ELEMENTWISE_OPS: dict[str, WorkKind] = {
    "Abs": WorkKind.ELEMENTWISE,
    "Exp": WorkKind.EXP,
    "Neg": WorkKind.ELEMENTWISE,
    "Sqrt": WorkKind.ELEMENTWISE,
}

BINARY_ELEMENTWISE_OPS: dict[str, WorkKind] = {
    "Add": WorkKind.ELEMENTWISE,
    "Div": WorkKind.ELEMENTWISE,
    "Mul": WorkKind.ELEMENTWISE,
    "Pow": WorkKind.ELEMENTWISE,
    "Sub": WorkKind.ELEMENTWISE,
}


def _lower_unary_elementwise_node(
    op_name: str,
    node_name: str,
    inputs: tuple[Tensor, ...],
    outputs: tuple[Tensor, ...],
    attributes: dict[str, object],
) -> tuple[OpKind, object]:
    del attributes
    if len(inputs) != 1:
        raise ValueError(f"{op_name} node '{node_name}' must have exactly 1 input")
    if len(outputs) != 1:
        raise ValueError(f"{op_name} node '{node_name}' must have exactly 1 output")
    return (
        OpKind.ELEMENTWISE,
        UnaryElementwiseOp(
            op_name=op_name,
            x=inputs[0],
            output=outputs[0],
            work_kind=UNARY_ELEMENTWISE_OPS[op_name],
        ),
    )


def _lower_binary_elementwise_node(
    op_name: str,
    node_name: str,
    inputs: tuple[Tensor, ...],
    outputs: tuple[Tensor, ...],
    attributes: dict[str, object],
) -> tuple[OpKind, object]:
    del attributes
    if len(inputs) != 2:
        raise ValueError(f"{op_name} node '{node_name}' must have exactly 2 inputs")
    if len(outputs) != 1:
        raise ValueError(f"{op_name} node '{node_name}' must have exactly 1 output")
    return (
        OpKind.ELEMENTWISE,
        BinaryElementwiseOp(
            op_name=op_name,
            lhs=inputs[0],
            rhs=inputs[1],
            output=outputs[0],
            work_kind=BINARY_ELEMENTWISE_OPS[op_name],
        ),
    )


def _make_unary_lowerer(op_name: str) -> OnnxLoweringFn:
    def lowerer(
        node_name: str,
        inputs: tuple[Tensor, ...],
        outputs: tuple[Tensor, ...],
        attributes: dict[str, object],
    ) -> tuple[OpKind, object]:
        return _lower_unary_elementwise_node(op_name, node_name, inputs, outputs, attributes)

    return lowerer


def _make_binary_lowerer(op_name: str) -> OnnxLoweringFn:
    def lowerer(
        node_name: str,
        inputs: tuple[Tensor, ...],
        outputs: tuple[Tensor, ...],
        attributes: dict[str, object],
    ) -> tuple[OpKind, object]:
        return _lower_binary_elementwise_node(op_name, node_name, inputs, outputs, attributes)

    return lowerer


ONNX_OP_LOWERERS: dict[str, OnnxLoweringFn] = {
    **{
        op_name: _make_unary_lowerer(op_name)
        for op_name in UNARY_ELEMENTWISE_OPS
    },
    **{
        op_name: _make_binary_lowerer(op_name)
        for op_name in BINARY_ELEMENTWISE_OPS
    },
}
