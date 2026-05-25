"""Final output transfers from pipeline tiles into external storage."""

from __future__ import annotations

from dataclasses import dataclass, field

from MAPS.core.layout import TensorSlice


@dataclass(frozen=True)
class FinalizationFragment:
    """One terminal tile-to-external tensor transfer."""

    src_hartid: int
    dst_hartid: int
    src_slice: TensorSlice
    dst_slice: TensorSlice

    def __post_init__(self) -> None:
        if self.src_hartid < 0:
            raise ValueError("finalization source hart id must be >= 0")
        if self.dst_hartid != -1:
            raise ValueError("finalization destination hart id must be -1")


@dataclass(frozen=True)
class Finalization:
    """Write one terminal layer output back to external storage."""

    name: str
    tensor_id: int
    src_layer_id: int
    src_output_idx: int
    fragments: tuple[FinalizationFragment, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("finalization name must not be empty")
        if self.tensor_id < 0 or self.src_layer_id < 0 or self.src_output_idx < 0:
            raise ValueError("finalization ids and indices must be >= 0")
