"""Primitive transport-cost model for one transfer leg."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from MAPS.core.mesh import Tile


class TransferKind(Enum):
    """Atomic transfer kinds supported by the current cost model."""

    L1_TO_L2 = auto()
    L2_TO_L1 = auto()
    L1_TO_L1 = auto()


@dataclass(frozen=True)
class TransferLeg:
    """
    One concrete transfer operation.

    A leg is the atomic costed movement used to build a full transition cost.
    For example:
    - one producer tile writing a fragment to L2
    - one L2 read into one consumer tile
    - one direct L1-to-L1 fragment transfer between two tiles
    """

    kind: TransferKind
    bytes: int
    src_tile: Tile | None = None
    dst_tile: Tile | None = None

    def __post_init__(self) -> None:
        if self.bytes <= 0:
            raise ValueError("transfer leg bytes must be > 0")

        if self.kind is TransferKind.L1_TO_L2:
            if self.src_tile is None:
                raise ValueError("L1_TO_L2 legs require src_tile")
        elif self.kind is TransferKind.L2_TO_L1:
            if self.dst_tile is None:
                raise ValueError("L2_TO_L1 legs require dst_tile")
        elif self.kind is TransferKind.L1_TO_L1:
            if self.src_tile is None or self.dst_tile is None:
                raise ValueError("L1_TO_L1 legs require src_tile and dst_tile")
        else:
            raise ValueError(f"unsupported transfer kind: {self.kind}")


class TransportCostModel:
    """Primitive latency model for one transfer leg."""

    def l1_to_l2(self, src: Tile, bytes_: int) -> float:
        return 88.0 + (1.5 + 0.5 * src.x) * bytes_

    def l2_to_l1(self, dst: Tile, bytes_: int) -> float:
        return 75.0 + (1.5 + 0.5 * dst.x) * bytes_

    def l1_to_l1(self, src: Tile, dst: Tile, bytes_: int) -> float:
        return 75.0 + (
            1.75
            + 0.5 * abs(src.x - dst.x)
            + 0.5 * abs(src.y - dst.y)
        ) * bytes_

    def cost(self, leg: TransferLeg) -> float:
        if leg.kind is TransferKind.L1_TO_L2:
            assert leg.src_tile is not None
            return self.l1_to_l2(leg.src_tile, leg.bytes)
        if leg.kind is TransferKind.L2_TO_L1:
            assert leg.dst_tile is not None
            return self.l2_to_l1(leg.dst_tile, leg.bytes)
        if leg.kind is TransferKind.L1_TO_L1:
            assert leg.src_tile is not None and leg.dst_tile is not None
            return self.l1_to_l1(leg.src_tile, leg.dst_tile, leg.bytes)
        raise ValueError(f"unsupported transfer kind: {leg.kind}")


