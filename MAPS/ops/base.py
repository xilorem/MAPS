"""Shared op payload helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from MAPS.core.layout import LayoutAxis, LayoutAxisMode, TensorLayout, TensorSlice
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor


@dataclass(frozen=True)
class TensorSliceRef:
    """A concrete slice tied to its logical tensor."""

    tensor: Tensor
    tensor_slice: TensorSlice


class TileWork(Protocol):
    """Planner-facing tile-work interface."""

    @property
    def input_slices(self) -> tuple[TensorSliceRef, ...]: ...

    @property
    def output_slices(self) -> tuple[TensorSliceRef, ...]: ...


def default_sharded_layout(
    tensor: Tensor,
    submesh: Submesh,
    logical_shape: tuple[int, int] | None,
) -> TensorLayout:
    """Shard the last tensor axis over mesh X and the previous axis over mesh Y."""

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


def tensor_slice_num_elements(tensor_slice: TensorSlice) -> int:
    total = 1
    for dim in tensor_slice.dims:
        total *= dim.length
    return total
