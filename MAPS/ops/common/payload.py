"""Shared payload contract and layout helpers for operation definitions.

An operation payload is the planner-facing semantic object stored in
``Node.payload``. It describes what an operation is, which layouts it expects,
which cost model should evaluate it, and how to lower that high-level
operation into concrete per-tile work once a placement has been chosen.

Payloads stay at the operation level; they do not describe one tile's exact
reads and writes directly. That lower-level description belongs to
``TileWork``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from MAPS.core.layout import LayoutAxis, LayoutAxisMode, TensorLayout
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.ops.common.tile_work import TileWork

if TYPE_CHECKING:
    from MAPS.arch import Tile


class OpPayload(ABC):
    """Planner-facing operation contract attached to one graph node.

    A payload owns operation semantics at the MAPS IR level. It chooses the
    canonical layouts for the op's inputs and outputs, exposes the cost model
    family used to estimate execution, and can lower itself into tile-local
    work for one concrete tile once placement is fixed.
    """

    @property
    @abstractmethod
    def cost_model(self) -> object: ...

    @abstractmethod
    def input_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]: ...

    @abstractmethod
    def output_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]: ...

    @abstractmethod
    def build_tile_work(
        self,
        input_layouts: tuple[TensorLayout, ...],
        output_layouts: tuple[TensorLayout, ...],
        tile: "Tile",
    ) -> TileWork: ...


def sharded_layout(
    tensor: Tensor,
    submesh: Submesh,
    logical_shape: tuple[int, int] | None,
    *,
    mesh_x_axis: int | None = None,
    mesh_y_axis: int | None = None,
) -> TensorLayout:
    """Build the canonical sharded layout for one tensor on one submesh.

    By default the last tensor axis is sharded on mesh X and the previous axis
    is sharded on mesh Y. Callers can override those tensor axes explicitly for
    ops whose natural tiling differs from the generic trailing-dimension rule.
    """

    logical_width = None
    logical_height = None
    if logical_shape is not None:
        logical_width, logical_height = logical_shape

    if mesh_x_axis is None:
        mesh_x_axis = tensor.rank - 1
    if mesh_y_axis is None and tensor.rank >= 2:
        mesh_y_axis = tensor.rank - 2

    if mesh_x_axis < 0 or mesh_x_axis >= tensor.rank:
        raise ValueError("mesh_x_axis must be within tensor rank")
    if mesh_y_axis is not None and (mesh_y_axis < 0 or mesh_y_axis >= tensor.rank):
        raise ValueError("mesh_y_axis must be within tensor rank")
    if mesh_y_axis is not None and mesh_x_axis == mesh_y_axis:
        raise ValueError("mesh_x_axis and mesh_y_axis must be different when both shard")

    mesh_y = LayoutAxis(mode=LayoutAxisMode.REPLICATE)
    if mesh_y_axis is not None:
        mesh_y = LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=mesh_y_axis)

    return TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=mesh_x_axis),
        mesh_y=mesh_y,
        logical_width=logical_width,
        logical_height=logical_height,
    )
