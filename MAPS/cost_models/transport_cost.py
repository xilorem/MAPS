"""Primitive transport-cost model for one transfer leg."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from MAPS.arch import Mesh, Tile


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


@dataclass(frozen=True)
class TransportCostModel:
    """Primitive latency model for one transfer leg."""

    mesh: Mesh | None = None
    l1_to_l2_startup_cycles: float = 88.0
    l2_to_l1_startup_cycles: float = 75.0
    l1_to_l1_startup_cycles: float = 75.0
    l2_access_hop_cycles: float = 0.5
    l1_to_l1_hop_cycles: float = 0.5

    def l1_to_l2(self, src: Tile, bytes_: int) -> float:
        bandwidth = self._effective_l2_bandwidth(src)
        return (
            self.l1_to_l2_startup_cycles
            + bytes_ / bandwidth
            + self.l2_access_hop_cycles * self._nearest_l2_access_distance(src)
        )

    def l2_to_l1(self, dst: Tile, bytes_: int) -> float:
        bandwidth = self._effective_l2_bandwidth(dst)
        return (
            self.l2_to_l1_startup_cycles
            + bytes_ / bandwidth
            + self.l2_access_hop_cycles * self._nearest_l2_access_distance(dst)
        )

    def l1_to_l1(self, src: Tile, dst: Tile, bytes_: int) -> float:
        bandwidth = min(src.memory.bandwidth, dst.memory.bandwidth)
        return (
            self.l1_to_l1_startup_cycles
            + bytes_ / bandwidth
            + self.l1_to_l1_hop_cycles * src.manhattan_distance(dst)
        )

    def cost(self, leg: TransferLeg) -> float:
        if leg.kind is TransferKind.L1_TO_L2:
            return self.l1_to_l2(leg.src_tile, leg.bytes)
        if leg.kind is TransferKind.L2_TO_L1:
            return self.l2_to_l1(leg.dst_tile, leg.bytes)
        if leg.kind is TransferKind.L1_TO_L1:
            return self.l1_to_l1(leg.src_tile, leg.dst_tile, leg.bytes)
        raise ValueError(f"unsupported transfer kind: {leg.kind}")

    def _effective_l2_bandwidth(self, tile: Tile) -> int:
        if self.mesh is None:
            return tile.memory.bandwidth
        return min(tile.memory.bandwidth, self.mesh.l2_memory.bandwidth)

    def _nearest_l2_access_distance(self, tile: Tile) -> int:
        if self.mesh is None or not self.mesh.l2_memory.access_points:
            return 0

        return min(
            abs(tile.x - access_x) + abs(tile.y - access_y)
            for access_x, access_y in self.mesh.l2_memory.access_points
        )
