"""GEMM payload IR matching the runtime-side `gemm_layer_op_t`."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch import Tile, WorkKind
from MAPS.core.graph import OpKind
from MAPS.core.layout import LayoutAxis, LayoutAxisMode, TensorLayout, TensorRange, TensorSlice, TensorSliceRef, tile_tensor_slice
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.ops.common.payload import OpPayload
from MAPS.ops.common.tile_work import TileWork
from MAPS.ops.registry import register_op
from MAPS.ops.spec import OpSpec


@dataclass(frozen=True)
class GemmTileWork(TileWork):
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
    def work_kind(self) -> WorkKind:
        return WorkKind.GEMM

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

    @property
    def l1_bytes(self) -> int:
        return sum(ref.num_bytes for ref in self.input_slices + self.output_slices)

    def fits_l1(self, tile: Tile) -> bool:
        return self.l1_bytes <= tile.memory.size

    def operation_count(self) -> int:
        return self.output_slice.num_elements * self.x_slice.dims[-1].length

    def dimensions(self) -> tuple[int, int, int, int]:
        batch_volume = 1
        for dim in self.output_slice.dims[:-2]:
            batch_volume *= dim.length
        m_size = self.output_slice.dims[-2].length
        n_size = self.output_slice.dims[-1].length
        k_size = self.x_slice.dims[-1].length
        return batch_volume, m_size, n_size, k_size


def _full_range(dim: int) -> TensorRange:
    return TensorRange(start=0, length=dim)


@dataclass(frozen=True)
class GemmPayload(OpPayload):
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

    def __post_init__(self) -> None:
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

    @property
    def cost_model(self) -> object:
        from MAPS.ops.costs.gemm_cost import GemmCostModel

        return GemmCostModel()

    def input_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        return tuple(
            self._tensor_layout(tensor, submesh, logical_shape)
            for tensor in (self.x, self.w)
        ) + (
            (self._tensor_layout(self.y, submesh, logical_shape),)
            if self.y is not None
            else ()
        )

    def output_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        return (self._tensor_layout(self.output, submesh, logical_shape),)

    def _tensor_layout(
        self,
        tensor: Tensor,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None,
    ) -> TensorLayout:
        if tensor.rank < 2:
            raise ValueError("GEMM layout requires rank >= 2")

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

def lower_gemm_node(
    node_name: str,
    inputs: tuple[Tensor, ...],
    outputs: tuple[Tensor, ...],
    attributes: dict[str, object],
) -> tuple[OpKind, object]:
    """Lower one ONNX Gemm node into scheduler-side GEMM semantics."""

    del attributes
    if len(inputs) not in (2, 3):
        raise ValueError(f"Gemm node '{node_name}' must have 2 or 3 inputs")
    if len(outputs) != 1:
        raise ValueError(f"Gemm node '{node_name}' must have exactly 1 output")

    return (
        OpKind.GEMM,
        GemmPayload(
            x=inputs[0],
            w=inputs[1],
            y=inputs[2] if len(inputs) == 3 else None,
            output=outputs[0],
        ),
    )


def lower_matmul_node(
    node_name: str,
    inputs: tuple[Tensor, ...],
    outputs: tuple[Tensor, ...],
    attributes: dict[str, object],
) -> tuple[OpKind, object]:
    """Lower one ONNX MatMul node into scheduler-side GEMM semantics."""

    del attributes
    if len(inputs) != 2:
        raise ValueError(f"MatMul node '{node_name}' must have exactly 2 inputs")
    if len(outputs) != 1:
        raise ValueError(f"MatMul node '{node_name}' must have exactly 1 output")

    return (
        OpKind.GEMM,
        GemmPayload(
            x=inputs[0],
            w=inputs[1],
            y=None,
            output=outputs[0],
        ),
    )


register_op(
    OpSpec(
        name="gemm",
        onnx_names=("Gemm",),
        lower_onnx=lower_gemm_node,
        work_kinds=(WorkKind.GEMM,),
    )
)

register_op(
    OpSpec(
        name="matmul",
        onnx_names=("MatMul",),
        lower_onnx=lower_matmul_node,
        work_kinds=(WorkKind.GEMM,),
    )
)
