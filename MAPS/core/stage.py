"""Stage IR for scheduled execution units on the mesh."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .layout import TensorLayout
from .submesh import Submesh

if TYPE_CHECKING:
    from .graph import Node
    from .tensor import Tensor


@dataclass(frozen=True)
class ExternalInput:
    """Stage input read from an external base address."""

    base_addr: int

    def __post_init__(self) -> None:
        if self.base_addr <= 0:
            raise ValueError("external inputs require base_addr > 0")


@dataclass(frozen=True)
class TransitionInput:
    """Stage input produced by an explicit transition."""

    transition_id: int

    def __post_init__(self) -> None:
        if self.transition_id < 0:
            raise ValueError("transition inputs require transition_id >= 0")


@dataclass(frozen=True)
class LocalInput:
    """Stage input read from another output in the same pipeline."""

    stage_id: int
    output_idx: int

    def __post_init__(self) -> None:
        if self.stage_id < 0 or self.output_idx < 0:
            raise ValueError("stage_id and output_idx must be >= 0")


InputSource = ExternalInput | TransitionInput | LocalInput


@dataclass(frozen=True)
class StageInput:
    """One input of a stage."""

    tensor_id: int
    source: InputSource

    def __post_init__(self) -> None:
        if self.tensor_id < 0:
            raise ValueError("tensor_id must be >= 0")

    @classmethod
    def external(cls, tensor_id: int, base_addr: int) -> "StageInput":
        return cls(tensor_id=tensor_id, source=ExternalInput(base_addr=base_addr))

    @classmethod
    def transition(cls, tensor_id: int, transition_id: int) -> "StageInput":
        return cls(
            tensor_id=tensor_id,
            source=TransitionInput(transition_id=transition_id),
        )

    @classmethod
    def local(cls, tensor_id: int, stage_id: int, output_idx: int) -> "StageInput":
        return cls(
            tensor_id=tensor_id,
            source=LocalInput(stage_id=stage_id, output_idx=output_idx),
        )


@dataclass(frozen=True)
class StageOutput:
    """One output of a stage."""

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
    inputs: tuple[StageInput, ...] = field(default_factory=tuple)
    outputs: tuple[StageOutput, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("stage name must not be empty")
        if not self.nodes:
            raise ValueError("stages must contain at least one node")

    def validate_tensors(self, tensors: tuple["Tensor", ...]) -> None:
        """Validate bound tensor ids and output layout compatibility."""

        for stage_input in self.inputs:
            if stage_input.tensor_id >= len(tensors):
                raise ValueError(f"input tensor_id out of range: {stage_input.tensor_id}")
        for stage_output in self.outputs:
            if stage_output.tensor_id >= len(tensors):
                raise ValueError(f"output tensor_id out of range: {stage_output.tensor_id}")
            stage_output.layout.validate_for(tensors[stage_output.tensor_id])
