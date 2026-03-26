"""Helpers to build concrete GEMM tile work from layouts and op semantics."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.core.layout import TensorLayout, TensorRange, TensorSlice
from MAPS.core.mesh import Tile
from MAPS.core.ownership import tile_tensor_slice
from MAPS.core.tensor import Tensor
from MAPS.ops.gemm import GemmLayerOp


@dataclass(frozen=True)
class GemmTileWork:
    """Concrete GEMM slices associated with one tile."""

    output_slice: TensorSlice
    x_slice: TensorSlice
    w_slice: TensorSlice
    y_slice: TensorSlice | None


def _full_range(dim: int) -> TensorRange:
    return TensorRange(start=0, length=dim)


def required_x_slice(output_slice: TensorSlice, x: Tensor) -> TensorSlice:
    """Derive the X slice required to produce one output slice."""

    if output_slice.rank != x.rank:
        raise ValueError("output slice rank must match X tensor rank")
    dims = list(output_slice.dims[:-2])
    dims.append(output_slice.dims[-2])
    dims.append(_full_range(x.dims[-1]))
    return TensorSlice(rank=x.rank, dims=tuple(dims))


def required_w_slice(output_slice: TensorSlice, w: Tensor) -> TensorSlice:
    """Derive the W slice required to produce one output slice."""

    if output_slice.rank != w.rank:
        raise ValueError("output slice rank must match W tensor rank")
    dims = list(output_slice.dims[:-2])
    dims.append(_full_range(w.dims[-2]))
    dims.append(output_slice.dims[-1])
    return TensorSlice(rank=w.rank, dims=tuple(dims))


def required_y_slice(output_slice: TensorSlice, y: Tensor | None) -> TensorSlice | None:
    """Derive the optional Y slice required to produce one output slice."""

    if y is None:
        return None
    if output_slice.rank != y.rank:
        raise ValueError("output slice rank must match Y tensor rank")
    return output_slice


def build_gemm_tile_work(op: GemmLayerOp,
                         output_layout: TensorLayout,
                         x_layout: TensorLayout,
                         w_layout: TensorLayout,
                         tile: Tile,
                         microbatch_idx: int) -> GemmTileWork:
    """Build the slices one tile must consume/produce for one GEMM."""

    output_slice = tile_tensor_slice(op.output, output_layout, tile, microbatch_idx)
    return GemmTileWork(
        output_slice=output_slice,
        x_slice=required_x_slice(output_slice, op.x),
        w_slice=required_w_slice(output_slice, op.w),
        y_slice=required_y_slice(output_slice, op.y),
    )
