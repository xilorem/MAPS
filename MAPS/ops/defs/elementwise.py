"""Reusable elementwise op payloads."""

from __future__ import annotations

from dataclasses import dataclass
from MAPS.arch import Tile, WorkKind
from MAPS.core.graph import OpKind
from MAPS.core.layout import TensorLayout, TensorRange, TensorSlice, TensorSliceRef, tile_tensor_slice
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.ops.common.payload import OpPayload, sharded_layout
from MAPS.ops.common.tile_work import TileWork
from MAPS.ops.registry import register_op
from MAPS.ops.spec import OpSpec


@dataclass(frozen=True)
class ElementwiseTileWork(TileWork):
    """Concrete elementwise slices associated with one tile."""

    work_kind: WorkKind
    output: Tensor
    output_slice: TensorSlice
    inputs: tuple[Tensor, ...]
    input_tile_slices: tuple[TensorSlice, ...]

    @property
    def input_slices(self) -> tuple[TensorSliceRef, ...]:
        return tuple(
            TensorSliceRef(tensor=tensor, tensor_slice=tensor_slice)
            for tensor, tensor_slice in zip(self.inputs, self.input_tile_slices)
        )

    @property
    def output_slices(self) -> tuple[TensorSliceRef, ...]:
        return (TensorSliceRef(tensor=self.output, tensor_slice=self.output_slice),)

    @property
    def l1_bytes(self) -> int:
        return sum(ref.num_bytes for ref in self.input_slices + self.output_slices)

    def fits_l1(self, tile: Tile) -> bool:
        return self.l1_bytes <= tile.memory.size

    def operation_count(self) -> int:
        return self.output_slice.num_elements


@dataclass(frozen=True)
class UnaryElementwisePayload(OpPayload):
    """Configured unary elementwise operation."""

    op_name: str
    x: Tensor
    output: Tensor
    work_kind: WorkKind = WorkKind.ELEMENTWISE

    def __post_init__(self) -> None:
        self.validate_shapes()

    @property
    def cost_model(self) -> object:
        from MAPS.ops.costs.elementwise_cost import ElementwiseCostModel

        return ElementwiseCostModel(work_kind=self.work_kind)

    def output_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        return (sharded_layout(self.output, submesh, logical_shape),)

    def required_input_slices(self, output_slice: TensorSlice) -> tuple[TensorSlice, ...]:
        if output_slice.rank != self.x.rank:
            raise ValueError(f"{self.op_name} output slice rank must match input rank")
        return (output_slice,)

    def build_tile_work(
        self,
        output_layouts: tuple[TensorLayout, ...],
        tile: Tile,
    ) -> ElementwiseTileWork:
        output_slice = tile_tensor_slice(self.output, output_layouts[0], tile)
        return ElementwiseTileWork(
            work_kind=self.work_kind,
            output=self.output,
            output_slice=output_slice,
            inputs=(self.x,),
            input_tile_slices=self.required_input_slices(output_slice),
        )

    def validate_shapes(self) -> None:
        if self.x.rank != self.output.rank or self.x.dims != self.output.dims:
            raise ValueError(f"{self.op_name} input and output shapes must match")
        if self.x.elem_bytes != self.output.elem_bytes:
            raise ValueError(f"{self.op_name} input and output element sizes must match")


@dataclass(frozen=True)
class BinaryElementwisePayload(OpPayload):
    """Configured binary elementwise operation with ONNX-style broadcasting."""

    op_name: str
    lhs: Tensor
    rhs: Tensor
    output: Tensor
    work_kind: WorkKind = WorkKind.ELEMENTWISE

    def __post_init__(self) -> None:
        self.validate_shapes()

    @property
    def cost_model(self) -> object:
        from MAPS.ops.costs.elementwise_cost import ElementwiseCostModel

        return ElementwiseCostModel(work_kind=self.work_kind)

    def output_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        return (sharded_layout(self.output, submesh, logical_shape),)

    def required_input_slices(self, output_slice: TensorSlice) -> tuple[TensorSlice, ...]:
        return (
            _broadcast_input_slice(self.lhs, self.output, output_slice),
            _broadcast_input_slice(self.rhs, self.output, output_slice),
        )

    def build_tile_work(
        self,
        output_layouts: tuple[TensorLayout, ...],
        tile: Tile,
    ) -> ElementwiseTileWork:
        output_slice = tile_tensor_slice(self.output, output_layouts[0], tile)
        return ElementwiseTileWork(
            work_kind=self.work_kind,
            output=self.output,
            output_slice=output_slice,
            inputs=(self.lhs, self.rhs),
            input_tile_slices=self.required_input_slices(output_slice),
        )

    def validate_shapes(self) -> None:
        _validate_broadcastable(self.lhs, self.output, self.op_name)
        _validate_broadcastable(self.rhs, self.output, self.op_name)
        if self.lhs.elem_bytes != self.output.elem_bytes or self.rhs.elem_bytes != self.output.elem_bytes:
            raise ValueError(f"{self.op_name} input and output element sizes must match")


def _validate_broadcastable(input_tensor: Tensor, output: Tensor, op_name: str) -> None:
    if input_tensor.rank > output.rank:
        raise ValueError(f"{op_name} input rank cannot exceed output rank")
    padded_input_dims = (1,) * (output.rank - input_tensor.rank) + input_tensor.dims
    for input_dim, output_dim in zip(padded_input_dims, output.dims):
        if input_dim not in (1, output_dim):
            raise ValueError(f"{op_name} input shape is not broadcastable to output")


def _broadcast_input_slice(
    input_tensor: Tensor,
    output: Tensor,
    output_slice: TensorSlice,
) -> TensorSlice:
    _validate_broadcastable(input_tensor, output, "elementwise")
    rank_offset = output.rank - input_tensor.rank
    dims: list[TensorRange] = []
    for input_axis, input_dim in enumerate(input_tensor.dims):
        output_axis = input_axis + rank_offset
        if input_dim == 1:
            dims.append(TensorRange(start=0, length=1))
        else:
            dims.append(output_slice.dims[output_axis])
    return TensorSlice(rank=input_tensor.rank, dims=tuple(dims))


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
        UnaryElementwisePayload(
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
        BinaryElementwisePayload(
            op_name=op_name,
            lhs=inputs[0],
            rhs=inputs[1],
            output=outputs[0],
            work_kind=BINARY_ELEMENTWISE_OPS[op_name],
        ),
    )


def _make_unary_lowerer(op_name: str):
    def lowerer(
        node_name: str,
        inputs: tuple[Tensor, ...],
        outputs: tuple[Tensor, ...],
        attributes: dict[str, object],
    ) -> tuple[OpKind, object]:
        return _lower_unary_elementwise_node(op_name, node_name, inputs, outputs, attributes)

    return lowerer


def _make_binary_lowerer(op_name: str):
    def lowerer(
        node_name: str,
        inputs: tuple[Tensor, ...],
        outputs: tuple[Tensor, ...],
        attributes: dict[str, object],
    ) -> tuple[OpKind, object]:
        return _lower_binary_elementwise_node(op_name, node_name, inputs, outputs, attributes)

    return lowerer


for _op_name, _work_kind in UNARY_ELEMENTWISE_OPS.items():
    register_op(
        OpSpec(
            name=_op_name.lower(),
            onnx_names=(_op_name,),
            lower_onnx=_make_unary_lowerer(_op_name),
            work_kinds=(_work_kind,),
        )
    )

for _op_name, _work_kind in BINARY_ELEMENTWISE_OPS.items():
    register_op(
        OpSpec(
            name=_op_name.lower(),
            onnx_names=(_op_name,),
            lower_onnx=_make_binary_lowerer(_op_name),
            work_kinds=(_work_kind,),
        )
    )
