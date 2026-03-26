"""Layer IR matching the runtime-side `layer_t` and related structs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

from .layout import TensorLayout
from .submesh import Submesh

if TYPE_CHECKING:
    from .tensor import Tensor


class InputSourceKind(IntEnum):
    EXTERNAL = 0
    TRANSITION = 1
    LOCAL = 2


@dataclass(frozen=True)
class LayerOutputRef:
    """Reference to a specific output of another layer."""

    layer_id: int
    output_idx: int

    def __post_init__(self) -> None:
        if self.layer_id < 0 or self.output_idx < 0:
            raise ValueError("layer_id and output_idx must be >= 0")


@dataclass(frozen=True)
class InputSource:
    """How one layer input is wired."""

    kind: InputSourceKind
    external_base_addr: int | None = None
    transition_id: int | None = None
    local_output: LayerOutputRef | None = None

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

        print("ok")

class LayerOpKind(IntEnum):
    GEMM = 0
    ELEMENTWISE = 1
    REDUCTION = 2
    CUSTOM = 255


@dataclass(frozen=True)
class LayerInputBinding:
    """One input binding of a layer."""

    tensor_id: int
    source: InputSource

    def __post_init__(self) -> None:
        if self.tensor_id < 0:
            raise ValueError("tensor_id must be >= 0")


@dataclass(frozen=True)
class LayerOutputBinding:
    """One output binding of a layer."""

    tensor_id: int
    layout: TensorLayout

    def __post_init__(self) -> None:
        if self.tensor_id < 0:
            raise ValueError("tensor_id must be >= 0")


@dataclass(frozen=True)
class Layer:
    """One logical compute stage."""

    name: str
    submesh: Submesh
    kind: LayerOpKind
    inputs: tuple[LayerInputBinding, ...] = field(default_factory=tuple)
    outputs: tuple[LayerOutputBinding, ...] = field(default_factory=tuple)
    payload: object | None = None # contains the operations specific descriptor (ex, which input is x, which is w for matmul)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("layer name must not be empty")
        self.validate_payload()

    def validate_payload(self) -> None:
        """Validate that the payload matches the declared op kind."""

        from MAPS.ops.gemm import GemmLayerOp

        if self.kind is LayerOpKind.GEMM:
            if not isinstance(self.payload, GemmLayerOp):
                raise ValueError("GEMM layers require a GemmLayerOp payload")
            self.payload.validate_bindings(self.inputs, self.outputs)
            return

        if isinstance(self.payload, GemmLayerOp):
            raise ValueError("GemmLayerOp payloads are only valid for GEMM layers")

    def validate_tensors(self, tensors: tuple["Tensor", ...]) -> None:
        """Validate bound tensor ids and op-specific tensor compatibility."""

        for binding in self.inputs:
            if binding.tensor_id >= len(tensors):
                raise ValueError(f"input tensor_id out of range: {binding.tensor_id}")
        for binding in self.outputs:
            if binding.tensor_id >= len(tensors):
                raise ValueError(f"output tensor_id out of range: {binding.tensor_id}")
            binding.layout.validate_for(tensors[binding.tensor_id])

        if self.kind is LayerOpKind.GEMM:
            assert self.payload is not None
            self.payload.validate_tensors(self.inputs, self.outputs, tensors)
