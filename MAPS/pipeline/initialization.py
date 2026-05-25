"""Initialization transfers from external storage into pipeline tiles."""

from __future__ import annotations

from dataclasses import dataclass, field

from MAPS.core.layout import TensorSlice


@dataclass(frozen=True)
class InitializationFragment:
    """One initial external-to-tile tensor transfer."""

    src_hartid: int
    dst_hartid: int
    src_slice: TensorSlice
    dst_slice: TensorSlice

    def __post_init__(self) -> None:
        if self.src_hartid != -1:
            raise ValueError("initialization source hart id must be -1")
        if self.dst_hartid < 0:
            raise ValueError("initialization destination hart id must be >= 0")


@dataclass(frozen=True)
class Initialization:
    """Populate one externally bound layer input before it is consumed."""

    name: str
    tensor_id: int
    dst_layer_id: int
    dst_input_idx: int
    fragments: tuple[InitializationFragment, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("initialization name must not be empty")
        if self.tensor_id < 0 or self.dst_layer_id < 0 or self.dst_input_idx < 0:
            raise ValueError("initialization ids and indices must be >= 0")
