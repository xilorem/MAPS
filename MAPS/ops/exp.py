"""Exp payload IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from MAPS.arch import Tile
from MAPS.core.layout import LayoutAxis, LayoutAxisMode, TensorLayout, TensorSlice
from MAPS.core.ownership import tile_tensor_slice
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor

if TYPE_CHECKING:
    from MAPS.core.stage import StageInputBinding, StageOutputBinding


@dataclass(frozen=True)
class ExpTileWork:
    """Concrete Exp slices associated with one tile."""

    output_slice: TensorSlice
    x_slice: TensorSlice


@dataclass(frozen=True)
class ExpLayerOp:
    """Unary elementwise exponential payload."""

    x: Tensor
    output: Tensor

    @property
    def cost_model(self) -> object:
        from MAPS.cost_models.exp_cost import ExpCostModel

        return ExpCostModel()

    def default_input_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        return (self._default_tensor_layout(self.x, submesh, logical_shape),)

    def default_output_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        return (self._default_tensor_layout(self.output, submesh, logical_shape),)

    def _default_tensor_layout(
        self,
        tensor: Tensor,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None,
    ) -> TensorLayout:
        logical_width = None
        logical_height = None
        if logical_shape is not None:
            logical_width, logical_height = logical_shape

        mesh_y = LayoutAxis(mode=LayoutAxisMode.REPLICATE)
        if tensor.rank >= 2:
            mesh_y = LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=tensor.rank - 2)

        return TensorLayout(
            submesh=submesh,
            mesh_x=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=tensor.rank - 1),
            mesh_y=mesh_y,
            logical_width=logical_width,
            logical_height=logical_height,
        )

    def required_x_slice(self, output_slice: TensorSlice) -> TensorSlice:
        if output_slice.rank != self.x.rank:
            raise ValueError("output slice rank must match Exp input rank")
        return output_slice

    def build_tile_work(
        self,
        input_layouts: tuple[TensorLayout, ...],
        output_layouts: tuple[TensorLayout, ...],
        tile: Tile,
    ) -> ExpTileWork:
        del input_layouts
        output_slice = tile_tensor_slice(
            tensor=self.output,
            layout=output_layouts[0],
            tile=tile,
        )
        return ExpTileWork(
            output_slice=output_slice,
            x_slice=self.required_x_slice(output_slice),
        )

    def validate_tensors(
        self,
        inputs: tuple[StageInputBinding, ...],
        outputs: tuple[StageOutputBinding, ...],
        tensors: tuple[Tensor, ...],
    ) -> None:
        if len(inputs) != 1:
            raise ValueError("Exp stages require exactly one input")
        if len(outputs) != 1:
            raise ValueError("Exp stages require exactly one output")

        bound_inputs = tuple(tensors[binding.tensor_id] for binding in inputs)
        bound_outputs = tuple(tensors[binding.tensor_id] for binding in outputs)
        if self.x not in bound_inputs:
            raise ValueError("Exp X tensor is not present in stage inputs")
        if self.output not in bound_outputs:
            raise ValueError("Exp output tensor is not present in stage outputs")
        self.validate_shapes()

    def validate_shapes(self) -> None:
        if self.x.rank != self.output.rank or self.x.dims != self.output.dims:
            raise ValueError("Exp input and output shapes must match")
        if self.x.elem_bytes != self.output.elem_bytes:
            raise ValueError("Exp input and output element sizes must match")
