"""Helpers to turn abstract layouts into concrete per-tile tensor slices."""

from MAPS.core.layout import LayoutAxis, LayoutAxisMode, TensorLayout, TensorRange, TensorSlice
from MAPS.arch import Tile
from MAPS.core.tensor import Tensor


def partition_range(total_length: int,
                    num_parts: int,
                    part_idx: int) -> TensorRange:
    """Return the balanced partition owned by one partition index.

    The axis is split as evenly as possible across ``num_parts``. When the
    division is not exact, the first ``total_length % num_parts`` partitions get
    one extra element (right and low edge cases).
    """
    if total_length < 0:
        raise ValueError("total_length must be non-negative")
    if num_parts <= 0:
        raise ValueError("num_parts must be positive")
    if part_idx < 0 or part_idx >= num_parts:
        raise ValueError("part_idx must be in [0, num_parts)")

    base = total_length // num_parts
    remainder = total_length % num_parts

    start = part_idx * base + min(part_idx, remainder)
    length = base + (1 if part_idx < remainder else 0)
    return TensorRange(start=start, length=length)


def microbatch_range(total_length: int,
                     num_microbatches: int,
                     microbatch_idx: int) -> TensorRange:
    """Return the active range of one axis for one microbatch."""
    return partition_range(total_length, num_microbatches, microbatch_idx)


def _apply_layout_axis(current_range: TensorRange,
                       axis: LayoutAxis,
                       num_parts: int,
                       part_idx: int) -> TensorRange:
    """Apply one mesh-axis policy to one tensor-axis range."""
    if axis.mode in (LayoutAxisMode.NONE, LayoutAxisMode.REPLICATE):
        return current_range
    if axis.mode is LayoutAxisMode.PARTIAL:
        raise NotImplementedError("PARTIAL ownership is not implemented yet")
    if axis.mode is not LayoutAxisMode.SHARD:
        raise ValueError(f"unsupported layout axis mode: {axis.mode}")

    local_range = partition_range(current_range.length, num_parts, part_idx)
    return TensorRange(
        start=current_range.start + local_range.start,
        length=local_range.length,
    )


def tile_tensor_slice(tensor: Tensor, layout: TensorLayout,
                      tile: Tile, microbatch_idx: int) -> TensorSlice:
    """Return the concrete tensor slice owned by one tile."""
    layout.validate_for(tensor)
    if not layout.submesh.contains_tile_id(tile.tile_id):
        raise ValueError(
            f"tile {tile.tile_id} is not inside submesh {layout.submesh.submesh_id}"
        )

    local_x, local_y = layout.submesh.local_coords(tile.tile_id)
    dims = [TensorRange(start=0, length=dim) for dim in tensor.dims]

    if layout.microbatch_axis is not None:
        dims[layout.microbatch_axis] = microbatch_range(
            tensor.dims[layout.microbatch_axis],
            layout.num_microbatches,
            microbatch_idx,
        )

    if layout.mesh_x.tensor_axis is not None:
        axis = layout.mesh_x.tensor_axis
        dims[axis] = _apply_layout_axis(
            dims[axis], layout.mesh_x, layout.submesh.width, local_x
        )

    if layout.mesh_y.tensor_axis is not None:
        axis = layout.mesh_y.tensor_axis
        dims[axis] = _apply_layout_axis(
            dims[axis], layout.mesh_y, layout.submesh.height, local_y
        )

    return TensorSlice(rank=tensor.rank, dims=tuple(dims))
