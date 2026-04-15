"""GEMM payload IR matching the runtime-side `gemm_layer_op_t`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from MAPS.arch import Tile
from MAPS.core.layout import LayoutAxis, LayoutAxisMode, TensorLayout, TensorRange, TensorSlice
from MAPS.core.ownership import tile_tensor_slice
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.ops.base import TensorSliceRef

if TYPE_CHECKING:
    from MAPS.core.stage import StageInputBinding, StageOutputBinding


@dataclass(frozen=True)
class GemmTileWork:
    """Concrete GEMM slices associated with one tile."""

    output_slice: TensorSlice
    x_slice: TensorSlice
    w_slice: TensorSlice
    y_slice: TensorSlice | None
    x: Tensor | None = None
    w: Tensor | None = None
    y: Tensor | None = None
    output: Tensor | None = None

    @property
    def input_slices(self) -> tuple[TensorSliceRef, ...]:
        refs = []
        if self.x is not None:
            refs.append(TensorSliceRef(tensor=self.x, tensor_slice=self.x_slice))
        if self.w is not None:
            refs.append(TensorSliceRef(tensor=self.w, tensor_slice=self.w_slice))
        if self.y is not None and self.y_slice is not None:
            refs.append(TensorSliceRef(tensor=self.y, tensor_slice=self.y_slice))
        return tuple(refs)

    @property
    def output_slices(self) -> tuple[TensorSliceRef, ...]:
        if self.output is None:
            return ()
        return (TensorSliceRef(tensor=self.output, tensor_slice=self.output_slice),)


def _full_range(dim: int) -> TensorRange:
    return TensorRange(start=0, length=dim)


@dataclass(frozen=True)
class GemmLayerOp:
    """GEMM-specific operation payload.

    The planner-side GEMM convention is:
    - ``x`` has shape ``[..., M, K]``
    - ``w`` has shape ``[..., K, N]``
    - ``output`` has shape ``[..., M, N]``
    - optional ``y`` must match ``output`` exactly
    """

    x: Tensor
    w: Tensor
    y: Tensor | None
    output: Tensor

    @property
    def cost_model(self) -> object:
        from MAPS.cost_models.gemm_cost import GemmCostModel

        return GemmCostModel()

    def default_input_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        return tuple(
            self._default_tensor_layout(tensor, submesh, logical_shape)
            for tensor in (self.x, self.w)
        ) + (
            (self._default_tensor_layout(self.y, submesh, logical_shape),)
            if self.y is not None
            else ()
        )

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
        if tensor.rank < 2:
            raise ValueError("default GEMM layout requires rank >= 2")

        logical_width = None
        logical_height = None
        if logical_shape is not None:
            logical_width, logical_height = logical_shape

        return TensorLayout(
            submesh=submesh,
            mesh_x=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=tensor.rank - 1),
            mesh_y=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=tensor.rank - 2),
            logical_width=logical_width,
            logical_height=logical_height,
        )

    def required_x_slice(self, output_slice: TensorSlice) -> TensorSlice:
        if output_slice.rank != self.x.rank:
            raise ValueError("output slice rank must match X tensor rank")
        dims = list(output_slice.dims[:-2])
        dims.append(output_slice.dims[-2])
        dims.append(_full_range(self.x.dims[-1]))
        return TensorSlice(rank=self.x.rank, dims=tuple(dims))

    def required_w_slice(self, output_slice: TensorSlice) -> TensorSlice:
        if output_slice.rank != self.w.rank:
            raise ValueError("output slice rank must match W tensor rank")
        dims = list(output_slice.dims[:-2])
        dims.append(_full_range(self.w.dims[-2]))
        dims.append(output_slice.dims[-1])
        return TensorSlice(rank=self.w.rank, dims=tuple(dims))

    def required_y_slice(self, output_slice: TensorSlice) -> TensorSlice | None:
        if self.y is None:
            return None
        if output_slice.rank != self.y.rank:
            raise ValueError("output slice rank must match Y tensor rank")
        return output_slice

    def build_tile_work(
        self,
        input_layouts: tuple[TensorLayout, ...],
        output_layouts: tuple[TensorLayout, ...],
        tile: Tile,
    ) -> GemmTileWork:
        del input_layouts
        output_slice = tile_tensor_slice(
            tensor=self.output,
            layout=output_layouts[0],
            tile=tile,
        )
        return GemmTileWork(
            output_slice=output_slice,
            x_slice=self.required_x_slice(output_slice),
            w_slice=self.required_w_slice(output_slice),
            y_slice=self.required_y_slice(output_slice),
            x=self.x,
            w=self.w,
            y=self.y,
            output=self.output,
        )

    def validate_tensors(
        self,
        inputs: tuple[StageInputBinding, ...],
        outputs: tuple[StageOutputBinding, ...],
        tensors: tuple[Tensor, ...],
    ) -> None:
        """Validate tensor availability against one GEMM stage instance."""

        if len(inputs) < 2:
            raise ValueError("GEMM stages require at least two inputs")
        if len(outputs) == 0:
            raise ValueError("GEMM stages require at least one output")

        bound_inputs = tuple(tensors[binding.tensor_id] for binding in inputs)
        bound_outputs = tuple(tensors[binding.tensor_id] for binding in outputs)

        if self.x not in bound_inputs:
            raise ValueError("GEMM X tensor is not present in stage inputs")
        if self.w not in bound_inputs:
            raise ValueError("GEMM W tensor is not present in stage inputs")
        if self.y is not None and self.y not in bound_inputs:
            raise ValueError("GEMM Y tensor is not present in stage inputs")
        if self.output not in bound_outputs:
            raise ValueError("GEMM output tensor is not present in stage outputs")

        for tensor_name, tensor in (
            ("X", self.x),
            ("W", self.w),
            ("output", self.output),
        ):
            if tensor.rank < 2:
                raise ValueError(f"{tensor_name} tensor rank must be >= 2 for GEMM")

        if self.x.elem_bytes != self.w.elem_bytes or self.x.elem_bytes != self.output.elem_bytes:
            raise ValueError("GEMM tensors must agree on element size")

        if self.x.dims[-1] != self.w.dims[-2]:
            raise ValueError("GEMM tensors must agree on K dimension")
        if self.x.dims[-2] != self.output.dims[-2]:
            raise ValueError("GEMM X and output must agree on M dimension")
        if self.w.dims[-1] != self.output.dims[-1]:
            raise ValueError("GEMM W and output must agree on N dimension")

        if self.y is not None:
            if self.y.rank != self.output.rank or self.y.dims != self.output.dims:
                raise ValueError("Y input must match output tensor shape exactly")
            if self.y.elem_bytes != self.output.elem_bytes:
                raise ValueError("Y input element size must match output tensor")
