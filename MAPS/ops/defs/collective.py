"""Collective communication op payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from MAPS.arch import Tile
from MAPS.core.layout import LayoutAxis, LayoutAxisMode, TensorLayout, TensorSlice
from MAPS.core.ownership import tile_tensor_slice
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.ops.common.base import TensorSliceRef, default_sharded_layout

if TYPE_CHECKING:
    from MAPS.core.layer import LayerInput, LayerOutput


@dataclass(frozen=True)
class CollectiveTileWork:
    """Concrete collective slices associated with one tile."""

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
class AllReduceOp:
    """Configured intra-stage allreduce collective."""

    op_name: str
    x: Tensor
    output: Tensor
    reduction: str
    collective_axis: str = "x"

    def __post_init__(self) -> None:
        if self.reduction not in {"sum", "max"}:
            raise ValueError("AllReduceOp reduction must be 'sum' or 'max'")
        if self.collective_axis not in {"x", "y"}:
            raise ValueError("AllReduceOp collective_axis must be 'x' or 'y'")

    @property
    def cost_model(self) -> object:
        from MAPS.ops.costs.collective_cost import AllReduceCostModel

        return AllReduceCostModel(
            reduction=self.reduction,
            collective_axis=self.collective_axis,
        )

    def default_input_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        return (self._collective_layout(self.x, submesh, logical_shape),)

    def default_output_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        return (self._collective_layout(self.output, submesh, logical_shape),)

    def _collective_layout(
        self,
        tensor: Tensor,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None,
    ) -> TensorLayout:
        layout = default_sharded_layout(tensor, submesh, logical_shape)
        mesh_x = layout.mesh_x
        mesh_y = layout.mesh_y
        if self.collective_axis == "x":
            mesh_x = LayoutAxis(mode=LayoutAxisMode.REPLICATE)
        else:
            mesh_y = LayoutAxis(mode=LayoutAxisMode.REPLICATE)
        return TensorLayout(
            submesh=submesh,
            mesh_x=mesh_x,
            mesh_y=mesh_y,
            logical_width=layout.logical_width,
            logical_height=layout.logical_height,
        )

    def build_tile_work(
        self,
        input_layouts: tuple[TensorLayout, ...],
        output_layouts: tuple[TensorLayout, ...],
        tile: Tile,
    ) -> CollectiveTileWork:
        return CollectiveTileWork(
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
        if self.x.rank != self.output.rank or self.x.dims != self.output.dims:
            raise ValueError(f"{self.op_name} input and output shapes must match")
        if self.x.elem_bytes != self.output.elem_bytes:
            raise ValueError(f"{self.op_name} input and output element sizes must match")
