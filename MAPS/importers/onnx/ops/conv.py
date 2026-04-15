"""Conv lowering helper."""

from __future__ import annotations

from MAPS.core.graph import OpKind
from MAPS.core.tensor import Tensor
from MAPS.ops.conv import ConvLayerOp

from .types import OnnxLoweringFn


def lower_conv_node(
    node_name: str,
    inputs: tuple[Tensor, ...],
    outputs: tuple[Tensor, ...],
    attributes: dict[str, object],
) -> tuple[OpKind, object]:

    if len(inputs) not in (2, 3):
        raise ValueError(f"Conv node '{node_name}' must have 2 or 3 inputs")
    if len(outputs) != 1:
        raise ValueError(f"Conv node '{node_name}' must have exactly 1 output")
    if "auto_pad" in attributes and attributes["auto_pad"] != "NOTSET":
        raise NotImplementedError("Conv auto_pad is not implemented")

    x_tensor = inputs[0]
    w_tensor = inputs[1]
    b_tensor = inputs[2] if len(inputs) == 3 else None
    output_tensor = outputs[0]
    return (
        OpKind.CONV,
        ConvLayerOp(
            x=x_tensor,
            w=w_tensor,
            b=b_tensor,
            output=output_tensor,
            strides=tuple(attributes.get("strides", (1, 1))),
            pads=tuple(attributes.get("pads", (0, 0, 0, 0))),
            dilations=tuple(attributes.get("dilations", (1, 1))),
            group=int(attributes.get("group", 1)),
        ),
    )


ONNX_OP_LOWERERS: dict[str, OnnxLoweringFn] = {
    "Conv": lower_conv_node,
}
