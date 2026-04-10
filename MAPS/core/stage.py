"""Stage IR for scheduled execution units on the mesh."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

from .graph import OpKind
from .layout import TensorLayout
from .submesh import Submesh

if TYPE_CHECKING:
    from .graph import Node
    from .tensor import Tensor


class InputSourceKind(IntEnum):
    EXTERNAL = 0
    TRANSITION = 1
    LOCAL = 2


@dataclass(frozen=True)
class StageOutputRef:
    """Reference to a specific output of another stage."""

    stage_id: int
    output_idx: int

    def __post_init__(self) -> None:
        if self.stage_id < 0 or self.output_idx < 0:
            raise ValueError("stage_id and output_idx must be >= 0")


@dataclass(frozen=True)
class InputSource:
    """How one stage input is wired."""

    kind: InputSourceKind
    external_base_addr: int | None = None
    transition_id: int | None = None
    local_output: StageOutputRef | None = None

    def __post_init__(self) -> None:
        if self.kind is InputSourceKind.EXTERNAL:
            if self.external_base_addr is None or self.external_base_addr <= 0:
                raise ValueError("external inputs require external_base_addr > 0")
        elif self.kind is InputSourceKind.TRANSITION:
            if self.transition_id is None or self.transition_id < 0:
                raise ValueError("transition inputs require transition_id >= 0")
        elif self.kind is InputSourceKind.LOCAL:
            if self.local_output is None:
                raise ValueError("local inputs require a local_output reference")


@dataclass(frozen=True)
class StageInputBinding:
    """One input binding of a stage."""

    tensor_id: int
    source: InputSource

    def __post_init__(self) -> None:
        if self.tensor_id < 0:
            raise ValueError("tensor_id must be >= 0")


@dataclass(frozen=True)
class StageOutputBinding:
    """One output binding of a stage."""

    tensor_id: int
    layout: TensorLayout

    def __post_init__(self) -> None:
        if self.tensor_id < 0:
            raise ValueError("tensor_id must be >= 0")


@dataclass(frozen=True)
class Stage:
    """One scheduled execution stage on a placed submesh."""

    name: str
    submesh: Submesh
    nodes: tuple["Node", ...] = field(default_factory=tuple)
    inputs: tuple[StageInputBinding, ...] = field(default_factory=tuple)
    outputs: tuple[StageOutputBinding, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("stage name must not be empty")
        if not self.nodes:
            raise ValueError("stages must contain at least one node")

    def validate_tensors(self, tensors: tuple["Tensor", ...]) -> None:
        """Validate bound tensor ids and output layout compatibility."""

        for binding in self.inputs:
            if binding.tensor_id >= len(tensors):
                raise ValueError(f"input tensor_id out of range: {binding.tensor_id}")
        for binding in self.outputs:
            if binding.tensor_id >= len(tensors):
                raise ValueError(f"output tensor_id out of range: {binding.tensor_id}")
            binding.layout.validate_for(tensors[binding.tensor_id])
