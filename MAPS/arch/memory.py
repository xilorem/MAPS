
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class L1Memory:
    """Tile-local memory metadata."""

    size: int
    bandwidth: int = 1

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("L1 memory size must be > 0")
        if self.bandwidth <= 0:
            raise ValueError("L1 memory bandwidth must be > 0")


@dataclass(frozen=True)
class L2Memory:
    """Shared mesh-level memory metadata."""

    size: int
    access_points: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    bandwidth: int = 1

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("L2 memory size must be > 0")
        if self.bandwidth <= 0:
            raise ValueError("L2 memory bandwidth must be > 0")
        for x, y in self.access_points:
            if x < 0 or y < 0:
                raise ValueError("L2 access point coordinates must be >= 0")
