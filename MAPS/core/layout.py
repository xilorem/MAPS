"""Tensor layout IR matching the runtime-side layout structures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .submesh import Submesh
from .tensor import TENSOR_MAX_DIMS, Tensor

TENSOR_AXIS_NONE: int | None = None


class LayoutAxisMode(IntEnum):
    NONE = 0
    SHARD = 1
    PARTIAL = 2
    REPLICATE = 3


@dataclass(frozen=True)
class LayoutAxis:
    """One mesh-axis policy applied to one tensor axis."""

    mode: LayoutAxisMode
    tensor_axis: int | None = TENSOR_AXIS_NONE

    def validate_for(self, tensor: Tensor) -> None:
        if self.mode in (LayoutAxisMode.NONE, LayoutAxisMode.REPLICATE):
            return
        if self.tensor_axis is None:
            raise ValueError("tensor_axis must be set for shard/partial modes")
        if self.tensor_axis < 0 or self.tensor_axis >= tensor.rank:
            raise ValueError("tensor_axis out of range for tensor rank")


@dataclass(frozen=True)
class TensorLayout:
    """Distribution policy for one tensor on one submesh."""

    submesh: Submesh
    mesh_x: LayoutAxis
    mesh_y: LayoutAxis
    microbatch_axis: int | None
    num_microbatches: int
    logical_width: int | None = None
    logical_height: int | None = None

    @property
    def effective_logical_width(self) -> int:
        return self.logical_width if self.logical_width is not None else self.submesh.width

    @property
    def effective_logical_height(self) -> int:
        return self.logical_height if self.logical_height is not None else self.submesh.height

    def validate_for(self, tensor: Tensor) -> None:
        self.mesh_x.validate_for(tensor)
        self.mesh_y.validate_for(tensor)
        logical_width = self.effective_logical_width
        logical_height = self.effective_logical_height
        if logical_width <= 0:
            raise ValueError("logical_width must be > 0")
        if logical_height <= 0:
            raise ValueError("logical_height must be > 0")
        if logical_width * logical_height != self.submesh.num_tiles:
            raise ValueError("logical shape area must match submesh tile count")
        if self.microbatch_axis is not None and (
            self.microbatch_axis < 0 or self.microbatch_axis >= tensor.rank
        ):
            raise ValueError("microbatch_axis out of range for tensor rank")
        if self.num_microbatches <= 0:
            raise ValueError("num_microbatches must be > 0")


@dataclass(frozen=True)
class TensorRange:
    """One 1D interval on a tensor axis."""

    start: int
    length: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("start must be >= 0")
        if self.length < 0:
            raise ValueError("length must be >= 0")


@dataclass(frozen=True)
class TensorSlice:
    """One concrete multi-dimensional slice of a tensor."""

    rank: int
    dims: tuple[TensorRange, ...]

    def __post_init__(self) -> None:
        if self.rank < 0 or self.rank > TENSOR_MAX_DIMS:
            raise ValueError(f"rank must be in [0, {TENSOR_MAX_DIMS}]")
        if len(self.dims) != self.rank:
            raise ValueError("dims length must match rank")
