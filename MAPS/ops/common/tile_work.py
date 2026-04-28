"""Shared tile-work contract for planner and cost-model code.

Tile work is the concrete, tile-local view of one operation after layouts and
placement have been chosen. A payload answers "what should this op do on this
submesh?"; a tile-work object answers "what exactly does one tile read, write,
and compute?".

Cost models, L1-capacity checks, and memory estimation all operate on this
lowered tile-local description rather than on high-level payloads.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from MAPS.core.layout import TensorSliceRef

if TYPE_CHECKING:
    from MAPS.arch import Tile


class TileWork(ABC):
    """Concrete per-tile work contract used after layout selection.

    Implementations expose the exact tensor slices a tile consumes and
    produces, plus the aggregate L1 footprint of keeping those slices resident
    while the tile executes the operation.
    """

    @property
    @abstractmethod
    def input_slices(self) -> tuple[TensorSliceRef, ...]: ...

    @property
    @abstractmethod
    def output_slices(self) -> tuple[TensorSliceRef, ...]: ...

    @property
    def l1_bytes(self) -> int:
        return sum(ref.num_bytes for ref in self.input_slices + self.output_slices)

    def fits_l1(self, tile: "Tile") -> bool:
        return self.l1_bytes <= tile.memory.size
