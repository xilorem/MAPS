"""Primitive transport-cost model for one transfer leg."""

from __future__ import annotations

from dataclasses import dataclass, field
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
class TransferCostEstimate:
    """Cost plus shared-resource loads for one transfer leg."""

    total_cost: float
    resource_loads: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class _NoCFlow:
    """One protocol flow routed over the NoC."""

    src_endpoint_id: int
    dst_endpoint_id: int
    bytes: int
    traffic_kind: TrafficKind
    bandwidth_limit: float = float("inf")


@dataclass(frozen=True)
class TransportCostModel:
    """Primitive latency model for one transfer leg."""

    mesh: Mesh | None = None
    l1_to_l2_startup_cycles: float = 88.0
    l2_to_l1_startup_cycles: float = 75.0
    l1_to_l1_startup_cycles: float = 75.0
    account_noc_contention: bool = False
    read_request_bytes: int = 1
    write_request_bytes: int = 1
    write_response_bytes: int = 1

    def l1_to_l2(self, src: Tile, bytes_: int) -> float:
        return self._noc_l1_to_l2(src, bytes_)

    def l2_to_l1(self, dst: Tile, bytes_: int) -> float:
        return self._noc_l2_to_l1(dst, bytes_)

    def l1_to_l1(self, src: Tile, dst: Tile, bytes_: int) -> float:
        return self._noc_l1_to_l1(src, dst, bytes_)

    def estimate(self, leg: TransferLeg) -> TransferCostEstimate:
        if leg.kind is TransferKind.L1_TO_L2:
            return self._estimate_l1_to_l2(leg.src_tile, leg.bytes)
        if leg.kind is TransferKind.L2_TO_L1:
            return self._estimate_l2_to_l1(leg.dst_tile, leg.bytes)
        if leg.kind is TransferKind.L1_TO_L1:
            return self._estimate_l1_to_l1(leg.src_tile, leg.dst_tile, leg.bytes)
        raise ValueError(f"unsupported transfer kind: {leg.kind}")

    def cost(self, leg: TransferLeg) -> float:
        return self.estimate(leg).total_cost

    def resource_loads(self, leg: TransferLeg) -> dict[str, float]:
        return self.estimate(leg).resource_loads

    def _estimate_l1_to_l2(self, src: Tile, bytes_: int) -> TransferCostEstimate:
        return self._estimate_noc_l1_to_l2(src, bytes_)

    def _estimate_l2_to_l1(self, dst: Tile, bytes_: int) -> TransferCostEstimate:
        return self._estimate_noc_l2_to_l1(dst, bytes_)

    def _estimate_l1_to_l1(self, src: Tile, dst: Tile, bytes_: int) -> TransferCostEstimate:
        return self._estimate_noc_l1_to_l1(src, dst, bytes_)

    def _noc_l1_to_l1(self, src: Tile, dst: Tile, bytes_: int) -> float:
        return self._estimate_noc_l1_to_l1(src, dst, bytes_).total_cost

    def _estimate_noc_l1_to_l1(self, src: Tile, dst: Tile, bytes_: int) -> TransferCostEstimate:
        self._require_noc()
        src_endpoint = self.mesh.noc.endpoint_for_tile(src.tile_id, EndpointKind.L1)
        dst_endpoint = self.mesh.noc.endpoint_for_tile(dst.tile_id, EndpointKind.L1)
        return self._route_protocol_cost(
            flows=(
                _NoCFlow(
                    src_endpoint_id=dst_endpoint.endpoint_id,
                    dst_endpoint_id=src_endpoint.endpoint_id,
                    bytes=self.read_request_bytes,
                    traffic_kind=TrafficKind.READ_REQ,
                ),
                _NoCFlow(
                    src_endpoint_id=src_endpoint.endpoint_id,
                    dst_endpoint_id=dst_endpoint.endpoint_id,
                    bytes=bytes_,
                    traffic_kind=TrafficKind.READ_RSP,
                    bandwidth_limit=min(src.memory.bandwidth, dst.memory.bandwidth),
                ),
            ),
            startup_cycles=self.l1_to_l1_startup_cycles,
        )

    def _noc_l1_to_l2(self, src: Tile, bytes_: int) -> float:
        return self._estimate_noc_l1_to_l2(src, bytes_).total_cost

    def _estimate_noc_l1_to_l2(self, src: Tile, bytes_: int) -> TransferCostEstimate:
        self._require_noc_with_l2()
        src_endpoint = self.mesh.noc.endpoint_for_tile(src.tile_id, EndpointKind.L1)
        return min(
            (
                self._route_protocol_cost(
                    flows=(
                        _NoCFlow(
                            src_endpoint_id=src_endpoint.endpoint_id,
                            dst_endpoint_id=l2_endpoint.endpoint_id,
                            bytes=self.write_request_bytes,
                            traffic_kind=TrafficKind.WRITE_REQ,
                        ),
                        _NoCFlow(
                            src_endpoint_id=src_endpoint.endpoint_id,
                            dst_endpoint_id=l2_endpoint.endpoint_id,
                            bytes=bytes_,
                            traffic_kind=TrafficKind.WRITE_DATA,
                            bandwidth_limit=min(src.memory.bandwidth, self.mesh.l2_memory.bandwidth),
                        ),
                        _NoCFlow(
                            src_endpoint_id=l2_endpoint.endpoint_id,
                            dst_endpoint_id=src_endpoint.endpoint_id,
                            bytes=self.write_response_bytes,
                            traffic_kind=TrafficKind.WRITE_RSP,
                        ),
                    ),
                    startup_cycles=self.l1_to_l2_startup_cycles,
                )
                for l2_endpoint in self.mesh.noc.endpoints_of_kind(EndpointKind.L2)
            ),
            key=lambda estimate: estimate.total_cost,
        )

    def _noc_l2_to_l1(self, dst: Tile, bytes_: int) -> float:
        return self._estimate_noc_l2_to_l1(dst, bytes_).total_cost

    def _estimate_noc_l2_to_l1(self, dst: Tile, bytes_: int) -> TransferCostEstimate:
        self._require_noc_with_l2()
        dst_endpoint = self.mesh.noc.endpoint_for_tile(dst.tile_id, EndpointKind.L1)
        return min(
            (
                self._route_protocol_cost(
                    flows=(
                        _NoCFlow(
                            src_endpoint_id=dst_endpoint.endpoint_id,
                            dst_endpoint_id=l2_endpoint.endpoint_id,
                            bytes=self.read_request_bytes,
                            traffic_kind=TrafficKind.READ_REQ,
                        ),
                        _NoCFlow(
                            src_endpoint_id=l2_endpoint.endpoint_id,
                            dst_endpoint_id=dst_endpoint.endpoint_id,
                            bytes=bytes_,
                            traffic_kind=TrafficKind.READ_RSP,
                            bandwidth_limit=min(self.mesh.l2_memory.bandwidth, dst.memory.bandwidth),
                        ),
                    ),
                    startup_cycles=self.l2_to_l1_startup_cycles,
                )
                for l2_endpoint in self.mesh.noc.endpoints_of_kind(EndpointKind.L2)
            ),
            key=lambda estimate: estimate.total_cost,
        )

    def _route_protocol_cost(
        self,
        flows: tuple[_NoCFlow, ...],
        startup_cycles: float,
    ) -> TransferCostEstimate:
        total_cost = startup_cycles
        resource_loads: dict[str, float] = {}

        for flow in flows:
            estimate = self._route_flow_cost(flow)
            total_cost += estimate.total_cost
            for resource_id, load in estimate.resource_loads.items():
                resource_loads[resource_id] = resource_loads.get(resource_id, 0.0) + load

        return TransferCostEstimate(total_cost=total_cost, resource_loads=resource_loads)

    def _route_flow_cost(
        self,
        flow: _NoCFlow,
    ) -> TransferCostEstimate:
        src_endpoint = self.mesh.noc.endpoint_by_id(flow.src_endpoint_id)
        dst_endpoint = self.mesh.noc.endpoint_by_id(flow.dst_endpoint_id)
        route = self.mesh.noc.route_endpoints(flow.src_endpoint_id, flow.dst_endpoint_id)
        src_attachment_channel = self._endpoint_attachment_channel(
            src_endpoint.egress_channels,
            flow.traffic_kind,
            f"endpoint {src_endpoint.endpoint_id} egress attachment",
        )
        dst_attachment_channel = self._endpoint_attachment_channel(
            dst_endpoint.ingress_channels,
            flow.traffic_kind,
            f"endpoint {dst_endpoint.endpoint_id} ingress attachment",
        )
        route_channels = self._route_channels(route, flow.traffic_kind)
        src_endpoint_bandwidth = (
            src_endpoint.egress_bandwidth_bytes
            if src_endpoint.egress_bandwidth_bytes is not None
            else float("inf")
        )
        dst_endpoint_bandwidth = (
            dst_endpoint.ingress_bandwidth_bytes
            if dst_endpoint.ingress_bandwidth_bytes is not None
            else float("inf")
        )
        route_bandwidth = min(
            (channel.width_bytes for channel in route_channels),
            default=float("inf"),
        )
        bandwidth = min(
            flow.bandwidth_limit,
            src_endpoint_bandwidth,
            dst_endpoint_bandwidth,
            src_attachment_channel.width_bytes if src_attachment_channel is not None else float("inf"),
            route_bandwidth,
            dst_attachment_channel.width_bytes if dst_attachment_channel is not None else float("inf"),
        )
        total_cost = (
            src_endpoint.egress_latency_cycles
            + (src_attachment_channel.hop_latency_cycles if src_attachment_channel is not None else 0.0)
            + dst_endpoint.ingress_latency_cycles
            + flow.bytes / bandwidth
            + sum(channel.hop_latency_cycles for channel in route_channels)
            + (dst_attachment_channel.hop_latency_cycles if dst_attachment_channel is not None else 0.0)
        )
        resource_loads = {}
        if self.account_noc_contention:
            resource_loads = {
                self._route_resource_id(link_id, channel.channel_id): flow.bytes / channel.width_bytes
                for link_id, channel in zip(route.link_ids, route_channels)
            }
            if src_attachment_channel is not None:
                resource_loads[
                    self._endpoint_attachment_resource_id(
                        src_endpoint.endpoint_id,
                        "egress",
                        src_attachment_channel.channel_id,
                    )
                ] = flow.bytes / src_attachment_channel.width_bytes
            if dst_attachment_channel is not None:
                resource_loads[
                    self._endpoint_attachment_resource_id(
                        dst_endpoint.endpoint_id,
                        "ingress",
                        dst_attachment_channel.channel_id,
                    )
                ] = flow.bytes / dst_attachment_channel.width_bytes
            if src_endpoint.egress_bandwidth_bytes is not None:
                resource_loads[self._endpoint_resource_id(src_endpoint.endpoint_id, "egress")] = (
                    flow.bytes / src_endpoint.egress_bandwidth_bytes
                )
            if dst_endpoint.ingress_bandwidth_bytes is not None:
                resource_loads[self._endpoint_resource_id(dst_endpoint.endpoint_id, "ingress")] = (
                    flow.bytes / dst_endpoint.ingress_bandwidth_bytes
                )
        return TransferCostEstimate(total_cost=total_cost, resource_loads=resource_loads)

    def _require_noc(self) -> None:
        if self.mesh is None or not self.mesh.has_noc:
            raise ValueError("transport cost model requires a mesh with a NoC")

    def _require_noc_with_l2(self) -> None:
        self._require_noc()
        if not self.mesh.noc.endpoints_of_kind(EndpointKind.L2):
            raise ValueError("transport cost model requires at least one NoC L2 endpoint")

    def _route_channels(self, route: NoCRoute, traffic_kind: TrafficKind) -> tuple[NoCChannel, ...]:
        selected_channels = []
        for link_id in route.link_ids:
            link = self.mesh.noc.link_by_id(link_id)
            selected_channels.append(
                self._select_channel(
                    link.channels,
                    traffic_kind,
                    f"link {link.link_id}",
                )
            )

        return tuple(selected_channels)

    def _endpoint_attachment_channel(
        self,
        channels: tuple[NoCChannel, ...],
        traffic_kind: TrafficKind,
        resource_name: str,
    ) -> NoCChannel | None:
        if not channels:
            return None
        return self._select_channel(channels, traffic_kind, resource_name)

    def _select_channel(
        self,
        channels: tuple[NoCChannel, ...],
        traffic_kind: TrafficKind,
        resource_name: str,
    ) -> NoCChannel:
        allowed_channel_ids = ()
        if self.mesh.noc.traffic_policy is not None:
            allowed_channel_ids = self.mesh.noc.traffic_policy.allowed_channel_ids(traffic_kind)

        candidates = tuple(
            channel
            for channel in channels
            if channel.supports(traffic_kind)
            and (not allowed_channel_ids or channel.channel_id in allowed_channel_ids)
        )
        if not candidates:
            raise ValueError(f"no channel available for {traffic_kind.name} on {resource_name}")

        return max(candidates, key=lambda channel: (channel.width_bytes, -channel.hop_latency_cycles))

    @staticmethod
    def _route_resource_id(link_id: int, channel_id: int) -> str:
        return f"noc_link:{link_id}:channel:{channel_id}"

    @staticmethod
    def _endpoint_resource_id(endpoint_id: int, direction: str) -> str:
        return f"noc_endpoint:{endpoint_id}:{direction}"

    @staticmethod
    def _endpoint_attachment_resource_id(endpoint_id: int, direction: str, channel_id: int) -> str:
        return f"noc_endpoint_attachment:{endpoint_id}:{direction}:channel:{channel_id}"
