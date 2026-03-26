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

    def validate_for(self, tensor: Tensor) -> None:
        self.mesh_x.validate_for(tensor)
        self.mesh_y.validate_for(tensor)
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
