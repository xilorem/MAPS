"""Primitive transport-cost model for one transfer leg."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from MAPS.arch import EndpointKind, Mesh, NoCChannel, NoCRoute, Tile, TrafficKind


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
        if self.mesh is not None and self.mesh.has_noc and self.mesh.noc.endpoints_of_kind(EndpointKind.L2):
            return self._noc_l1_to_l2(src, bytes_)

        bandwidth = self._effective_l2_bandwidth(src)
        return (
            self.l1_to_l2_startup_cycles
            + bytes_ / bandwidth
            + self.l2_access_hop_cycles * self._nearest_l2_access_distance(src)
        )

    def l2_to_l1(self, dst: Tile, bytes_: int) -> float:
        if self.mesh is not None and self.mesh.has_noc and self.mesh.noc.endpoints_of_kind(EndpointKind.L2):
            return self._noc_l2_to_l1(dst, bytes_)

        bandwidth = self._effective_l2_bandwidth(dst)
        return (
            self.l2_to_l1_startup_cycles
            + bytes_ / bandwidth
            + self.l2_access_hop_cycles * self._nearest_l2_access_distance(dst)
        )

    def l1_to_l1(self, src: Tile, dst: Tile, bytes_: int) -> float:
        if self.mesh is not None and self.mesh.has_noc:
            return self._noc_l1_to_l1(src, dst, bytes_)

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

    def _noc_l1_to_l1(self, src: Tile, dst: Tile, bytes_: int) -> float:
        src_endpoint = self.mesh.noc.endpoint_for_tile(src.tile_id, EndpointKind.L1)
        dst_endpoint = self.mesh.noc.endpoint_for_tile(dst.tile_id, EndpointKind.L1)
        return self._route_transfer_cost(
            src_endpoint_id=src_endpoint.endpoint_id,
            dst_endpoint_id=dst_endpoint.endpoint_id,
            bytes_=bytes_,
            startup_cycles=self.l1_to_l1_startup_cycles,
            bandwidth_limit=min(src.memory.bandwidth, dst.memory.bandwidth),
        )

    def _noc_l1_to_l2(self, src: Tile, bytes_: int) -> float:
        src_endpoint = self.mesh.noc.endpoint_for_tile(src.tile_id, EndpointKind.L1)
        return min(
            self._route_transfer_cost(
                src_endpoint_id=src_endpoint.endpoint_id,
                dst_endpoint_id=l2_endpoint.endpoint_id,
                bytes_=bytes_,
                startup_cycles=self.l1_to_l2_startup_cycles,
                bandwidth_limit=min(src.memory.bandwidth, self.mesh.l2_memory.bandwidth),
            )
            for l2_endpoint in self.mesh.noc.endpoints_of_kind(EndpointKind.L2)
        )

    def _noc_l2_to_l1(self, dst: Tile, bytes_: int) -> float:
        dst_endpoint = self.mesh.noc.endpoint_for_tile(dst.tile_id, EndpointKind.L1)
        return min(
            self._route_transfer_cost(
                src_endpoint_id=l2_endpoint.endpoint_id,
                dst_endpoint_id=dst_endpoint.endpoint_id,
                bytes_=bytes_,
                startup_cycles=self.l2_to_l1_startup_cycles,
                bandwidth_limit=min(self.mesh.l2_memory.bandwidth, dst.memory.bandwidth),
            )
            for l2_endpoint in self.mesh.noc.endpoints_of_kind(EndpointKind.L2)
        )

    def _route_transfer_cost(
        self,
        src_endpoint_id: int,
        dst_endpoint_id: int,
        bytes_: int,
        startup_cycles: float,
        bandwidth_limit: float,
    ) -> float:
        route = self.mesh.noc.route_endpoints(src_endpoint_id, dst_endpoint_id)
        route_channels = self._route_channels(route, TrafficKind.TRANSFER)
        route_bandwidth = min(
            (channel.width_bytes for channel in route_channels),
            default=float("inf"),
        )
        bandwidth = min(bandwidth_limit, route_bandwidth)
        return startup_cycles + bytes_ / bandwidth + sum(
            channel.hop_latency_cycles for channel in route_channels
        )

    def _route_channels(self, route: NoCRoute, traffic_kind: TrafficKind) -> tuple[NoCChannel, ...]:
        allowed_channel_ids = ()
        if self.mesh.noc.traffic_policy is not None:
            allowed_channel_ids = self.mesh.noc.traffic_policy.allowed_channel_ids(traffic_kind)

        selected_channels = []
        for link_id in route.link_ids:
            link = self.mesh.noc.link_by_id(link_id)
            candidates = tuple(
                channel
                for channel in link.channels
                if channel.supports(traffic_kind)
                and (not allowed_channel_ids or channel.channel_id in allowed_channel_ids)
            )
            if not candidates:
                raise ValueError(
                    f"no channel available for {traffic_kind.name} on link {link.link_id}"
                )
            selected_channels.append(
                max(candidates, key=lambda channel: (channel.width_bytes, -channel.hop_latency_cycles))
            )

        return tuple(selected_channels)
