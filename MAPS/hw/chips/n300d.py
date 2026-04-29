
"""Tenstorrent Wormhole n300d chip description.

This module models one Wormhole ASIC as:
- one logical 8x8 compute mesh used by the planner
- one larger physical 10x12 NoC used by transport costs

The DRAM endpoint coordinates follow the physical Wormhole NoC grid. The
logical 8x8 compute tiles are attached to a regular subset of NoC nodes so the
current MAPS planner can still operate on a dense compute mesh.
"""

from __future__ import annotations

from MAPS.arch import (
    EndpointKind,
    L1Memory,
    L2Memory,
    Mesh,
    NoC,
    NoCChannel,
    NoCEndpoint,
    NoCLink,
    NoCNode,
    RoutingPolicy,
    Tile,
    TrafficKind,
    TrafficPolicy,
)
from MAPS.hw.devices import IDMA_READ_DEVICE, IDMA_WRITE_DEVICE, SCALAR_CORE_DEVICE
from MAPS.hw.devices.redmule import REDMULE_DEVICE


# Single-ASIC approximation of one Wormhole die.
N300D_MESH_WIDTH = 8
N300D_MESH_HEIGHT = 8

N300D_L1_SIZE_BYTES = 1464 * 1024
N300D_L1_USABLE_BYTES = 1464 * 1024
N300D_L1_STACK_BYTES = 0
N300D_L1_RESERVED_BYTES = 0
N300D_L1_BANDWIDTH_BYTES = 32
N300D_L2_SIZE_BYTES = 12 * 1024 * 1024 * 1024
N300D_L2_BANDWIDTH_BYTES = 288
N300D_NOC_CHANNEL_WIDTH_BYTES = 32
N300D_NOC_HOP_LATENCY_CYCLES = 9
N300D_NOC_WIDTH = 10
N300D_NOC_HEIGHT = 12

N300D_DRAM_ENDPOINTS_BY_CONTROLLER = {
    "D1": ((0, 0), (0, 1), (0, 11)),
    "D2": ((0, 5), (0, 7)),
    "D3": ((0, 6), (5, 0), (5, 1), (5, 11)),
    "D4": ((5, 2), (5, 9), (5, 10)),
    "D5": ((5, 3), (5, 4), (5, 8)),
    "D6": ((5, 5), (5, 6), (5, 7)),
}

N300D_L2_ENDPOINT_COORDS = (
    (0, 0),
    (0, 1),
    (0, 5),
    (0, 6),
    (0, 7),
    (0, 11),
    (5, 0),
    (5, 1),
    (5, 2),
    (5, 3),
    (5, 4),
    (5, 5),
    (5, 6),
    (5, 7),
    (5, 8),
    (5, 9),
    (5, 10),
    (5, 11),
)

# Logical 8x8 compute tiles are attached to a regular subset of physical NoC
# nodes that avoids the DRAM columns. This is still an approximation, but keeps
# transport costs on the larger Wormhole NoC grid without breaking planner
# assumptions about rectangular compute placement.
N300D_TILE_NOC_COORDS = tuple(
    (x, y)
    for y in range(2, 10)
    for x in (1, 2, 3, 4, 6, 7, 8, 9)
)

N300D_IDMA_READ_DEVICE = IDMA_READ_DEVICE
N300D_IDMA_WRITE_DEVICE = IDMA_WRITE_DEVICE
N300D_CORE_DEVICE = SCALAR_CORE_DEVICE
N300D_REDMULE_DEVICE = REDMULE_DEVICE

N300D_TILE_DEVICES = (
    N300D_IDMA_READ_DEVICE,
    N300D_IDMA_WRITE_DEVICE,
    N300D_CORE_DEVICE,
    N300D_REDMULE_DEVICE,
)


def _n300d_noc_node_id(x: int, y: int, width: int) -> int:
    return y * width + x


def _n300d_noc_channels() -> tuple[NoCChannel, ...]:
    return (
        NoCChannel(
            channel_id=0,
            width_bytes=N300D_NOC_CHANNEL_WIDTH_BYTES,
            hop_latency_cycles=N300D_NOC_HOP_LATENCY_CYCLES,
            tag="req",
            supported_traffic=frozenset(
                {
                    TrafficKind.READ_REQ,
                    TrafficKind.WRITE_REQ,
                    TrafficKind.WRITE_DATA,
                }
            ),
        ),
        NoCChannel(
            channel_id=1,
            width_bytes=N300D_NOC_CHANNEL_WIDTH_BYTES,
            hop_latency_cycles=N300D_NOC_HOP_LATENCY_CYCLES,
            tag="rsp",
            supported_traffic=frozenset(
                {
                    TrafficKind.READ_RSP,
                    TrafficKind.WRITE_RSP,
                }
            ),
        ),
    )


def _n300d_noc(width: int = N300D_NOC_WIDTH, height: int = N300D_NOC_HEIGHT) -> NoC:
    attachment_channels = _n300d_noc_channels()
    nodes = tuple(
        NoCNode(
            node_id=_n300d_noc_node_id(x, y, width),
            x=x,
            y=y,
        )
        for y in range(height)
        for x in range(width)
    )
    link_pairs = tuple(
        (_n300d_noc_node_id(x, y, width), _n300d_noc_node_id(x + 1, y, width))
        for y in range(height)
        for x in range(width - 1)
    ) + tuple(
        (_n300d_noc_node_id(x, y, width), _n300d_noc_node_id(x, y + 1, width))
        for y in range(height - 1)
        for x in range(width)
    )
    links = tuple(
        NoCLink(
            link_id=link_id,
            src_node_id=src_node_id,
            dst_node_id=dst_node_id,
            channels=_n300d_noc_channels(),
            bidirectional=True,
        )
        for link_id, (src_node_id, dst_node_id) in enumerate(link_pairs)
    )
    l1_endpoints = tuple(
        NoCEndpoint(
            endpoint_id=tile_id,
            kind=EndpointKind.L1,
            node_id=_n300d_noc_node_id(x, y, width),
            tile_id=tile_id,
        )
        for tile_id, (x, y) in enumerate(N300D_TILE_NOC_COORDS)
    )
    l2_endpoints = tuple(
        NoCEndpoint(
            endpoint_id=len(N300D_TILE_NOC_COORDS) + endpoint_index,
            kind=EndpointKind.L2,
            node_id=_n300d_noc_node_id(x, y, width),
            name=f"l2_{endpoint_index}",
            ingress_channels=attachment_channels,
            egress_channels=attachment_channels,
        )
        for endpoint_index, (x, y) in enumerate(N300D_L2_ENDPOINT_COORDS)
    )
    return NoC(
        nodes=nodes,
        links=links,
        endpoints=l1_endpoints + l2_endpoints,
        traffic_policy=TrafficPolicy(
            {
                TrafficKind.READ_REQ: (0,),
                TrafficKind.WRITE_REQ: (0,),
                TrafficKind.READ_RSP: (1,),
                TrafficKind.WRITE_RSP: (1,),
                TrafficKind.WRITE_DATA: (0,),
            }
        ),
        routing_policy=RoutingPolicy.XY,
    )


def n300d_mesh(
    width: int = N300D_MESH_WIDTH,
    height: int = N300D_MESH_HEIGHT,
) -> Mesh:
    if (width, height) != (N300D_MESH_WIDTH, N300D_MESH_HEIGHT):
        raise ValueError("n300d_mesh uses a fixed logical 8x8 compute mesh")

    return Mesh(
        width=width,
        height=height,
        l2_memory=L2Memory(size=N300D_L2_SIZE_BYTES, bandwidth=N300D_L2_BANDWIDTH_BYTES),
        noc=_n300d_noc(),
        tiles=tuple(
            Tile(
                tile_id=y * width + x,
                x=x,
                y=y,
                memory=L1Memory(size=N300D_L1_USABLE_BYTES, bandwidth=N300D_L1_BANDWIDTH_BYTES),
                devices=N300D_TILE_DEVICES,
            )
            for y in range(height)
            for x in range(width)
        ),
    )
