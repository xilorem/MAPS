
"""Tenstorrent Wormhole n300d single-ASIC chip description.

This module models one Wormhole ASIC as:
- one logical 8x8 compute mesh used by the planner
- one larger physical 10x12 NoC used by transport costs

The DRAM endpoint coordinates follow the physical Wormhole NoC grid. The
logical 8x8 compute tiles are attached to a regular subset of NoC nodes so the
current MAPS planner can still operate on a dense compute mesh.

This is intentionally a per-ASIC model, not a full dual-ASIC n300d board model.
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
from MAPS.hw.devices import (
    TENSIX_MATRIX_DEVICE,
    TENSIX_READ_CORE,
    TENSIX_SCALAR_DEVICE,
    TENSIX_VECTOR_DEVICE,
    TENSIX_WRITE_CORE,
)
from MAPS.utils.print_mesh import print_mesh


# Single-ASIC approximation of one Wormhole die.
N300D_MESH_WIDTH = 8
N300D_MESH_HEIGHT = 8

N300D_L1_SIZE_BYTES = 1464 * 1024
N300D_L1_USABLE_BYTES = 1464 * 1024
N300D_L1_STACK_BYTES = 0
N300D_L1_RESERVED_BYTES = 0
# MAPS uses tile memory bandwidth as the NoC-facing transfer limit. For
# Wormhole, use the 32 B/cycle NoC-facing port width rather than the aggregate
# internal SRAM bank bandwidth.
N300D_L1_BANDWIDTH_BYTES = 32
N300D_L2_SIZE_BYTES = 12 * 1024 * 1024 * 1024
N300D_L2_BANDWIDTH_BYTES = 288
N300D_NOC_CHANNEL_WIDTH_BYTES = 32
N300D_NOC_HOP_LATENCY_CYCLES = 9
N300D_NIU_LATENCY_CYCLES = 5
N300D_NOC_WIDTH_PADDING = 2
N300D_NOC_HEIGHT_PADDING = 4
N300D_NOC_WIDTH = N300D_MESH_WIDTH + N300D_NOC_WIDTH_PADDING
N300D_NOC_HEIGHT = N300D_MESH_HEIGHT + N300D_NOC_HEIGHT_PADDING
N300D_RESERVED_NOC_ROW_COUNT = 2

_N300D_TEMPLATE_NOC_HEIGHT = 12
_N300D_TEMPLATE_LEFT_L2_ROWS = (0, 1, 5, 6, 7, 11)

N300D_TILE_DEVICES = (
    TENSIX_READ_CORE,
    TENSIX_WRITE_CORE,
    TENSIX_SCALAR_DEVICE,
    TENSIX_VECTOR_DEVICE,
    TENSIX_MATRIX_DEVICE,
)


def _n300d_noc_node_id(x: int, y: int, width: int) -> int:
    return y * width + x


def _scale_template_coord(value: int, template_size: int, target_size: int) -> int:
    if template_size <= 1:
        return 0
    return round(value * (target_size - 1) / (template_size - 1))


def _n300d_noc_dims(mesh_width: int, mesh_height: int) -> tuple[int, int]:
    return mesh_width + N300D_NOC_WIDTH_PADDING, mesh_height + N300D_NOC_HEIGHT_PADDING


def _n300d_reserved_rows(mesh_height: int) -> tuple[int, ...]:
    _, noc_height = _n300d_noc_dims(N300D_MESH_WIDTH, mesh_height)
    start = noc_height // 2
    return tuple(range(start, start + N300D_RESERVED_NOC_ROW_COUNT))


def _select_contiguous_block(available: tuple[int, ...], count: int) -> tuple[int, ...]:
    if count <= 0:
        return ()
    if count > len(available):
        raise ValueError("not enough NoC positions for requested compute mesh size")
    start = (len(available) - count) // 2
    return available[start:start + count]


def _n300d_middle_l2_x(mesh_width: int) -> int:
    left_compute_width = (mesh_width + 1) // 2
    return 1 + left_compute_width


def _n300d_tile_noc_coords(
    mesh_width: int,
    mesh_height: int,
) -> tuple[tuple[int, int], ...]:
    noc_width, noc_height = _n300d_noc_dims(mesh_width, mesh_height)
    middle_l2_x = _n300d_middle_l2_x(mesh_width)
    left_positions = tuple(range(1, middle_l2_x))
    right_positions = tuple(range(middle_l2_x + 1, noc_width))
    x_positions = left_positions + right_positions
    reserved_rows = set(_n300d_reserved_rows(mesh_height))
    y_positions = tuple(y for y in range(2, noc_height) if y not in reserved_rows)
    return tuple((x, y) for y in y_positions for x in x_positions)


def _n300d_l2_endpoint_coords(mesh_width: int, mesh_height: int) -> tuple[tuple[int, int], ...]:
    _, noc_height = _n300d_noc_dims(mesh_width, mesh_height)
    middle_l2_x = _n300d_middle_l2_x(mesh_width)
    left_rows = tuple(dict.fromkeys(
        _scale_template_coord(y, _N300D_TEMPLATE_NOC_HEIGHT, noc_height)
        for y in _N300D_TEMPLATE_LEFT_L2_ROWS
    ))
    middle_rows = tuple(range(noc_height))
    return tuple((0, y) for y in left_rows) + tuple((middle_l2_x, y) for y in middle_rows)


N300D_L2_ENDPOINT_COORDS = _n300d_l2_endpoint_coords(N300D_MESH_WIDTH, N300D_MESH_HEIGHT)
N300D_TILE_NOC_COORDS = _n300d_tile_noc_coords(N300D_MESH_WIDTH, N300D_MESH_HEIGHT)


def _n300d_noc_channels() -> tuple[NoCChannel, ...]:
    all_traffic = frozenset(
        {
            TrafficKind.READ_REQ,
            TrafficKind.WRITE_REQ,
            TrafficKind.READ_RSP,
            TrafficKind.WRITE_RSP,
            TrafficKind.WRITE_DATA,
        }
    )
    return (
        NoCChannel(
            channel_id=0,
            width_bytes=N300D_NOC_CHANNEL_WIDTH_BYTES,
            hop_latency_cycles=N300D_NOC_HOP_LATENCY_CYCLES,
            tag="noc0",
            supported_traffic=all_traffic,
        ),
        NoCChannel(
            channel_id=1,
            width_bytes=N300D_NOC_CHANNEL_WIDTH_BYTES,
            hop_latency_cycles=N300D_NOC_HOP_LATENCY_CYCLES,
            tag="noc1",
            supported_traffic=all_traffic,
        ),
    )


def _n300d_noc(
    mesh_width: int,
    mesh_height: int,
) -> NoC:
    width, height = _n300d_noc_dims(mesh_width, mesh_height)
    attachment_channels = _n300d_noc_channels()
    tile_noc_coords = _n300d_tile_noc_coords(mesh_width, mesh_height)
    l2_endpoint_coords = _n300d_l2_endpoint_coords(mesh_width, mesh_height)
    nodes = tuple(
        NoCNode(
            node_id=_n300d_noc_node_id(x, y, width),
            x=x,
            y=y,
        )
        for y in range(height)
        for x in range(width)
    )
    horizontal_pairs = tuple(
        (
            _n300d_noc_node_id(x, y, width),
            _n300d_noc_node_id((x + 1) % width, y, width),
        )
        for y in range(height)
        for x in range(width)
    )
    vertical_pairs = tuple(
        (
            _n300d_noc_node_id(x, y, width),
            _n300d_noc_node_id(x, (y + 1) % height, width),
        )
        for y in range(height)
        for x in range(width)
    )
    link_pairs = horizontal_pairs + vertical_pairs
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
            ingress_latency_cycles=N300D_NIU_LATENCY_CYCLES,
            egress_latency_cycles=N300D_NIU_LATENCY_CYCLES,
        )
        for tile_id, (x, y) in enumerate(tile_noc_coords)
    )
    l2_endpoints = tuple(
        NoCEndpoint(
            endpoint_id=len(tile_noc_coords) + endpoint_index,
            kind=EndpointKind.L2,
            node_id=_n300d_noc_node_id(x, y, width),
            name=f"l2_{endpoint_index}",
            ingress_latency_cycles=N300D_NIU_LATENCY_CYCLES,
            egress_latency_cycles=N300D_NIU_LATENCY_CYCLES,
            ingress_channels=attachment_channels,
            egress_channels=attachment_channels,
        )
        for endpoint_index, (x, y) in enumerate(l2_endpoint_coords)
    )
    return NoC(
        nodes=nodes,
        links=links,
        endpoints=l1_endpoints + l2_endpoints,
        traffic_policy=TrafficPolicy(
            {
                TrafficKind.READ_REQ: (0, 1),
                TrafficKind.WRITE_REQ: (0, 1),
                TrafficKind.READ_RSP: (0, 1),
                TrafficKind.WRITE_RSP: (0, 1),
                TrafficKind.WRITE_DATA: (0, 1),
            }
        ),
        routing_policy=RoutingPolicy.TORUS_XY,
    )


def wormhole_n300d_mesh(
    width: int = N300D_MESH_WIDTH,
    height: int = N300D_MESH_HEIGHT,
) -> Mesh:
    return Mesh(
        width=width,
        height=height,
        l2_memory=L2Memory(size=N300D_L2_SIZE_BYTES, bandwidth=N300D_L2_BANDWIDTH_BYTES),
        noc=_n300d_noc(width, height),
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


if __name__ == "__main__":
    print("Wormhole n300d")
    print_mesh(wormhole_n300d_mesh())
