"""Reduction op payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from MAPS.arch import Tile, WorkKind
from MAPS.core.layout import LayoutAxis, LayoutAxisMode, TensorLayout, TensorSlice
from MAPS.core.ownership import tile_tensor_slice
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.ops.base import TensorSliceRef, default_sharded_layout

if TYPE_CHECKING:
    from MAPS.core.layer import LayerInput, LayerOutput


@dataclass(frozen=True)
class ReductionTileWork:
    """Concrete reduction slices associated with one tile."""

    x: Tensor
    output: Tensor
    input_slice: TensorSlice
    output_slice: TensorSlice

    @property
    def input_slices(self) -> tuple[TensorSliceRef, ...]:
        return (TensorSliceRef(tensor=self.x, tensor_slice=self.input_slice),)

    @property
    def output_slices(self) -> tuple[TensorSliceRef, ...]:
        return (TensorSliceRef(tensor=self.output, tensor_slice=self.output_slice),)

    @property
    def l1_bytes(self) -> int:
        return sum(ref.num_bytes for ref in self.input_slices + self.output_slices)

    def fits_l1(self, tile: Tile) -> bool:
        return self.l1_bytes <= tile.memory.size


@dataclass(frozen=True)
class ReduceOp:
    """Configured tile-local reduction operation."""

    op_name: str
    x: Tensor
    output: Tensor
    axis: int
    work_kind: WorkKind

    def __post_init__(self) -> None:
        if self.work_kind not in (WorkKind.REDUCE_SUM, WorkKind.REDUCE_MAX):
            raise ValueError("ReduceOp work_kind must be REDUCE_SUM or REDUCE_MAX")
        if self.axis < 0 or self.axis >= self.x.rank:
            raise ValueError("ReduceOp axis must be in input tensor rank")

    @property
    def cost_model(self) -> object:
        from MAPS.cost_models.reduction_cost import ReductionCostModel

        return ReductionCostModel(work_kind=self.work_kind)

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
        input_layout = default_sharded_layout(self.x, submesh, logical_shape)
        mesh_x = input_layout.mesh_x
        mesh_y = input_layout.mesh_y
        if mesh_x.tensor_axis == self.axis:
            mesh_x = LayoutAxis(mode=LayoutAxisMode.REPLICATE)
        if mesh_y.tensor_axis == self.axis:
            mesh_y = LayoutAxis(mode=LayoutAxisMode.REPLICATE)
        return (
            TensorLayout(
                submesh=submesh,
                mesh_x=mesh_x,
                mesh_y=mesh_y,
                logical_width=input_layout.logical_width,
                logical_height=input_layout.logical_height,
            ),
        )

    def build_tile_work(
        self,
        input_layouts: tuple[TensorLayout, ...],
        output_layouts: tuple[TensorLayout, ...],
        tile: Tile,
    ) -> ReductionTileWork:
        return ReductionTileWork(
            x=self.x,
            output=self.output,
            input_slice=tile_tensor_slice(self.x, input_layouts[0], tile),
            output_slice=tile_tensor_slice(self.output, output_layouts[0], tile),
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
        if self.x.rank != self.output.rank:
            raise ValueError(f"{self.op_name} input and output ranks must match")
        if self.x.elem_bytes != self.output.elem_bytes:
            raise ValueError(f"{self.op_name} input and output element sizes must match")
        for axis, (input_dim, output_dim) in enumerate(zip(self.x.dims, self.output.dims)):
            expected_output_dim = 1 if axis == self.axis else input_dim
            if output_dim != expected_output_dim:
                raise ValueError(
                    f"{self.op_name} output dim {axis} must be {expected_output_dim}, got {output_dim}"
                )
