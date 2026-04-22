"""ONNX softmax graph expansion."""

from __future__ import annotations

from MAPS.core.graph import Node, OpKind
from MAPS.core.tensor import Tensor
from MAPS.ops import AllReduceOp, BinaryElementwiseOp, ReduceOp, UnaryElementwiseOp
from MAPS.arch import WorkKind

from ..node_parser import (
    node_inputs,
    node_name,
    node_outputs,
    parse_node_attributes,
    resolve_node_tensors,
)


def lower_softmax_node(node, node_idx: int, tensors: dict[str, Tensor]) -> tuple[tuple[Tensor, ...], tuple[Node, ...]]:
    """Lower one ONNX Softmax node into grouped planner nodes."""

    node_name_value = node_name(node, node_idx)
    input_names = node_inputs(node)
    output_names = node_outputs(node)
    input_tensors, output_tensors = resolve_node_tensors(
        node_name_value,
        input_names,
        output_names,
        tensors,
    )
    if len(input_tensors) != 1:
        raise ValueError(f"Softmax node '{node_name_value}' must have exactly 1 input")
    if len(output_tensors) != 1:
        raise ValueError(f"Softmax node '{node_name_value}' must have exactly 1 output")

    attributes = parse_node_attributes(node)
    x = input_tensors[0]
    output = output_tensors[0]
    axis = int(attributes.get("axis", -1))
    if axis < 0:
        axis += x.rank
    if axis < 0 or axis >= x.rank:
        raise ValueError(f"Softmax node '{node_name_value}' axis {axis} is out of range")
    if x.rank != output.rank or x.dims != output.dims:
        raise ValueError(f"Softmax node '{node_name_value}' input and output shapes must match")
    if x.elem_bytes != output.elem_bytes:
        raise ValueError(f"Softmax node '{node_name_value}' input and output element sizes must match")

    stage_group_id = f"{node_name_value}::softmax"
    shared_attributes = dict(attributes)
    shared_attributes["stage_group_id"] = stage_group_id

    max_local = _reduced_tensor(f"{node_name_value}__max_local", x, axis)
    collective_axis = _collective_axis_for_softmax(x, axis)
    max_value = max_local
    new_tensors: list[Tensor] = [max_local]
    nodes: list[Node] = [
        Node(
            name=f"{node_name_value}__reduce_max",
            kind=OpKind.REDUCTION,
            inputs=(x,),
            outputs=(max_local,),
            payload=ReduceOp(
                op_name="ReduceMax",
                x=x,
                output=max_local,
                axis=axis,
                work_kind=WorkKind.REDUCE_MAX,
            ),
            attributes={**shared_attributes, "softmax_step": "reduce_max"},
        )
    ]

    if collective_axis is not None:
        max_global = _same_shape_tensor(f"{node_name_value}__max_global", max_local)
        new_tensors.append(max_global)
        nodes.append(
            Node(
                name=f"{node_name_value}__allreduce_max",
                kind=OpKind.CUSTOM,
                inputs=(max_local,),
                outputs=(max_global,),
                payload=AllReduceOp(
                    op_name="AllReduceMax",
                    x=max_local,
                    output=max_global,
                    reduction="max",
                    collective_axis=collective_axis,
                ),
                attributes={**shared_attributes, "softmax_step": "allreduce_max"},
            )
        )
        max_value = max_global

    shifted = _same_shape_tensor(f"{node_name_value}__shifted", x)
    exp = _same_shape_tensor(f"{node_name_value}__exp", x)
    sum_local = _reduced_tensor(f"{node_name_value}__sum_local", x, axis)
    new_tensors.extend((shifted, exp, sum_local))
    nodes.extend(
        (
            Node(
                name=f"{node_name_value}__sub",
                kind=OpKind.ELEMENTWISE,
                inputs=(x, max_value),
                outputs=(shifted,),
                payload=BinaryElementwiseOp(
                    op_name="Sub",
                    lhs=x,
                    rhs=max_value,
                    output=shifted,
                    work_kind=WorkKind.ELEMENTWISE,
                ),
                attributes={**shared_attributes, "softmax_step": "sub"},
            ),
            Node(
                name=f"{node_name_value}__exp",
                kind=OpKind.ELEMENTWISE,
                inputs=(shifted,),
                outputs=(exp,),
                payload=UnaryElementwiseOp(
                    op_name="Exp",
                    x=shifted,
                    output=exp,
                    work_kind=WorkKind.EXP,
                ),
                attributes={**shared_attributes, "softmax_step": "exp"},
            ),
            Node(
                name=f"{node_name_value}__reduce_sum",
                kind=OpKind.REDUCTION,
                inputs=(exp,),
                outputs=(sum_local,),
                payload=ReduceOp(
                    op_name="ReduceSum",
                    x=exp,
                    output=sum_local,
                    axis=axis,
                    work_kind=WorkKind.REDUCE_SUM,
                ),
                attributes={**shared_attributes, "softmax_step": "reduce_sum"},
            ),
        )
    )

    sum_value = sum_local
    if collective_axis is not None:
        sum_global = _same_shape_tensor(f"{node_name_value}__sum_global", sum_local)
        new_tensors.append(sum_global)
        nodes.append(
            Node(
                name=f"{node_name_value}__allreduce_sum",
                kind=OpKind.CUSTOM,
                inputs=(sum_local,),
                outputs=(sum_global,),
                payload=AllReduceOp(
                    op_name="AllReduceSum",
                    x=sum_local,
                    output=sum_global,
                    reduction="sum",
                    collective_axis=collective_axis,
                ),
                attributes={**shared_attributes, "softmax_step": "allreduce_sum"},
            )
        )
        sum_value = sum_global

    nodes.append(
        Node(
            name=f"{node_name_value}__div",
            kind=OpKind.ELEMENTWISE,
            inputs=(exp, sum_value),
            outputs=(output,),
            payload=BinaryElementwiseOp(
                op_name="Div",
                lhs=exp,
                rhs=sum_value,
                output=output,
                work_kind=WorkKind.ELEMENTWISE,
            ),
            attributes={**shared_attributes, "softmax_step": "div"},
        )
    )

    return tuple(new_tensors), tuple(nodes)


def _same_shape_tensor(name: str, reference: Tensor) -> Tensor:
    return Tensor(
        name=name,
        rank=reference.rank,
        dims=reference.dims,
        elem_bytes=reference.elem_bytes,
    )


def _reduced_tensor(name: str, reference: Tensor, axis: int) -> Tensor:
    dims = list(reference.dims)
    dims[axis] = 1
    return Tensor(
        name=name,
        rank=reference.rank,
        dims=tuple(dims),
        elem_bytes=reference.elem_bytes,
    )


def _collective_axis_for_softmax(tensor: Tensor, axis: int) -> str | None:
    if axis == tensor.rank - 1:
        return "x"
    if tensor.rank >= 2 and axis == tensor.rank - 2:
        return "y"
    return None
