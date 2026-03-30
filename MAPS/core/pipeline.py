from __future__ import annotations

from dataclasses import dataclass, field

from MAPS.arch import Mesh
from .stage import Stage
from .tensor import Tensor
from .transition import Transition


@dataclass(frozen=True)
class Pipeline:
    """One generated pipeline plan."""

    name: str
    mesh: Mesh
    num_microbatches: int
    tensors: tuple[Tensor, ...] = field(default_factory=tuple)
    stages: tuple[Stage, ...] = field(default_factory=tuple)
    transitions: tuple[Transition, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("pipeline name must not be empty")
        if self.num_microbatches <= 0:
            raise ValueError("num_microbatches must be > 0")
