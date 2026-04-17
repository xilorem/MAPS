"""Reusable elementwise op payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from MAPS.arch import Tile, WorkKind
from MAPS.core.layout import TensorLayout, TensorRange, TensorSlice
from MAPS.core.ownership import tile_tensor_slice
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.ops.base import TensorSliceRef, default_sharded_layout

if TYPE_CHECKING:
    from MAPS.core.layer import LayerInput, LayerOutput


@dataclass(frozen=True)
class ElementwiseTileWork:
    """Concrete elementwise slices associated with one tile."""

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


@dataclass(frozen=True)
class UnaryElementwiseOp:
    """Configured unary elementwise operation."""

    op_name: str
    x: Tensor
    output: Tensor
    work_kind: WorkKind = WorkKind.ELEMENTWISE

    @property
    def cost_model(self) -> object:
        from MAPS.cost_models.elementwise_cost import ElementwiseCostModel

        return ElementwiseCostModel(work_kind=self.work_kind)

    def default_input_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        return (default_sharded_layout(self.x, submesh, logical_shape),)

    def default_output_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        return (default_sharded_layout(self.output, submesh, logical_shape),)

    def required_input_slices(self, output_slice: TensorSlice) -> tuple[TensorSlice, ...]:
        if output_slice.rank != self.x.rank:
            raise ValueError(f"{self.op_name} output slice rank must match input rank")
        return (output_slice,)

    def build_tile_work(
        self,
        input_layouts: tuple[TensorLayout, ...],
        output_layouts: tuple[TensorLayout, ...],
        tile: Tile,
    ) -> ElementwiseTileWork:
        del input_layouts
        output_slice = tile_tensor_slice(self.output, output_layouts[0], tile)
        return ElementwiseTileWork(
            output=self.output,
            output_slice=output_slice,
            inputs=(self.x,),
            input_tile_slices=self.required_input_slices(output_slice),
        )

    def validate_tensors(
        self,
        inputs: tuple[LayerInput, ...],
        outputs: tuple[LayerOutput, ...],
        tensors: tuple[Tensor, ...],
    ) -> None:
        if len(inputs) != 1:
            raise ValueError(f"{self.op_name} stages require exactly one input")
        if len(outputs) != 1:
            raise ValueError(f"{self.op_name} stages require exactly one output")
        bound_inputs = tuple(tensors[binding.tensor_id] for binding in inputs)
        bound_outputs = tuple(tensors[binding.tensor_id] for binding in outputs)
        if self.x not in bound_inputs:
            raise ValueError(f"{self.op_name} input tensor is not present in stage inputs")
        if self.output not in bound_outputs:
            raise ValueError(f"{self.op_name} output tensor is not present in stage outputs")
        self.validate_shapes()

    def validate_shapes(self) -> None:
        if self.x.rank != self.output.rank or self.x.dims != self.output.dims:
            raise ValueError(f"{self.op_name} input and output shapes must match")
        if self.x.elem_bytes != self.output.elem_bytes:
            raise ValueError(f"{self.op_name} input and output element sizes must match")


@dataclass(frozen=True)
class BinaryElementwiseOp:
    """Configured binary elementwise operation with ONNX-style broadcasting."""

    op_name: str
    lhs: Tensor
    rhs: Tensor
    output: Tensor
    work_kind: WorkKind = WorkKind.ELEMENTWISE

    @property
    def cost_model(self) -> object:
        from MAPS.cost_models.elementwise_cost import ElementwiseCostModel

        return ElementwiseCostModel(work_kind=self.work_kind)

    def default_input_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        return (
            default_sharded_layout(self.lhs, submesh, logical_shape),
            default_sharded_layout(self.rhs, submesh, logical_shape),
        )

    def default_output_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        return (default_sharded_layout(self.output, submesh, logical_shape),)

    def required_input_slices(self, output_slice: TensorSlice) -> tuple[TensorSlice, ...]:
        return (
            _broadcast_input_slice(self.lhs, self.output, output_slice),
            _broadcast_input_slice(self.rhs, self.output, output_slice),
        )

    def build_tile_work(
        self,
        input_layouts: tuple[TensorLayout, ...],
        output_layouts: tuple[TensorLayout, ...],
        tile: Tile,
    ) -> ElementwiseTileWork:
        del input_layouts
        output_slice = tile_tensor_slice(self.output, output_layouts[0], tile)
        return ElementwiseTileWork(
            output=self.output,
            output_slice=output_slice,
            inputs=(self.lhs, self.rhs),
            input_tile_slices=self.required_input_slices(output_slice),
        )

    def validate_tensors(
        self,
        inputs: tuple[LayerInput, ...],
        outputs: tuple[LayerOutput, ...],
        tensors: tuple[Tensor, ...],
    ) -> None:
        if len(inputs) != 2:
            raise ValueError(f"{self.op_name} stages require exactly two inputs")
        if len(outputs) != 1:
            raise ValueError(f"{self.op_name} stages require exactly one output")
        bound_inputs = tuple(tensors[binding.tensor_id] for binding in inputs)
        bound_outputs = tuple(tensors[binding.tensor_id] for binding in outputs)
        if self.lhs not in bound_inputs:
            raise ValueError(f"{self.op_name} lhs tensor is not present in stage inputs")
        if self.rhs not in bound_inputs:
            raise ValueError(f"{self.op_name} rhs tensor is not present in stage inputs")
        if self.output not in bound_outputs:
            raise ValueError(f"{self.op_name} output tensor is not present in stage outputs")
        self.validate_shapes()

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
