"""Stage IR for scheduled execution units on the mesh."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from MAPS.core.submesh import Submesh
from MAPS.pipeline.layer import Layer

if TYPE_CHECKING:
    from MAPS.core.tensor import Tensor


@dataclass(frozen=True)
class Stage:
    """One scheduled pipeline stage on a placed submesh."""

    name: str
    submesh: Submesh
    layers: tuple[Layer, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("stage name must not be empty")
        if not self.layers:
            raise ValueError("stages must contain at least one layer")

    def validate_tensors(self, tensors: tuple["Tensor", ...]) -> None:
        """Validate layer tensor ids and output layout compatibility."""

        for layer in self.layers:
            layer.validate_tensors(tensors)
