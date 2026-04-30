from MAPS.arch import EndpointKind, RoutingPolicy, TrafficKind
from MAPS.hw.chips.n300d import (
    N300D_L1_BANDWIDTH_BYTES,
    N300D_L1_USABLE_BYTES,
    N300D_L2_BANDWIDTH_BYTES,
    N300D_NOC_HEIGHT_PADDING,
    N300D_NOC_WIDTH_PADDING,
    N300D_L2_ENDPOINT_COORDS,
    N300D_MESH_HEIGHT,
    N300D_MESH_WIDTH,
    N300D_NOC_CHANNEL_WIDTH_BYTES,
    N300D_NOC_HEIGHT,
    N300D_NOC_HOP_LATENCY_CYCLES,
    N300D_NIU_LATENCY_CYCLES,
    N300D_NOC_WIDTH,
    N300D_TILE_NOC_COORDS,
    _n300d_reserved_rows,
    _n300d_l2_endpoint_coords,
    _n300d_tile_noc_coords,
    wormhole_n300d_mesh,
)


def test_n300d_mesh_uses_logical_compute_mesh_and_physical_noc() -> None:
    mesh = wormhole_n300d_mesh()

    assert mesh.shape == (N300D_MESH_WIDTH, N300D_MESH_HEIGHT)
    assert mesh.num_tiles == N300D_MESH_WIDTH * N300D_MESH_HEIGHT
    assert mesh.l2_memory.bandwidth == N300D_L2_BANDWIDTH_BYTES
    assert all(tile.memory.size == N300D_L1_USABLE_BYTES for tile in mesh.tiles)
    assert all(tile.memory.bandwidth == N300D_L1_BANDWIDTH_BYTES for tile in mesh.tiles)

    assert len(mesh.noc.nodes) == N300D_NOC_WIDTH * N300D_NOC_HEIGHT
    assert len(mesh.noc.endpoints_of_kind(EndpointKind.L1)) == mesh.num_tiles
    assert len(mesh.noc.endpoints_of_kind(EndpointKind.L2)) == len(N300D_L2_ENDPOINT_COORDS)
    assert mesh.noc.routing_policy is RoutingPolicy.TORUS_XY
    assert mesh.noc.traffic_policy.allowed_channel_ids(TrafficKind.READ_REQ) == (0, 1)
    assert mesh.noc.traffic_policy.allowed_channel_ids(TrafficKind.WRITE_REQ) == (0, 1)
    assert mesh.noc.traffic_policy.allowed_channel_ids(TrafficKind.READ_RSP) == (0, 1)
    assert mesh.noc.traffic_policy.allowed_channel_ids(TrafficKind.WRITE_RSP) == (0, 1)
    assert mesh.noc.traffic_policy.allowed_channel_ids(TrafficKind.WRITE_DATA) == (0, 1)


def test_n300d_tile_and_dram_endpoints_attach_to_expected_noc_coordinates() -> None:
    mesh = wormhole_n300d_mesh()

    l1_endpoints = mesh.noc.endpoints_of_kind(EndpointKind.L1)
    l2_endpoints = mesh.noc.endpoints_of_kind(EndpointKind.L2)

    assert tuple(mesh.noc.node_by_id(endpoint.node_id).coords for endpoint in l1_endpoints) == N300D_TILE_NOC_COORDS
    assert tuple(mesh.noc.node_by_id(endpoint.node_id).coords for endpoint in l2_endpoints) == N300D_L2_ENDPOINT_COORDS

    assert all(link.bidirectional for link in mesh.noc.links)
    assert len(mesh.noc.links) == 2 * N300D_NOC_WIDTH * N300D_NOC_HEIGHT
    assert all(tuple(channel.tag for channel in link.channels) == ("noc0", "noc1") for link in mesh.noc.links)
    assert all(link.channels[0].width_bytes == N300D_NOC_CHANNEL_WIDTH_BYTES for link in mesh.noc.links)
    assert all(link.channels[1].width_bytes == N300D_NOC_CHANNEL_WIDTH_BYTES for link in mesh.noc.links)
    assert all(
        all(channel.hop_latency_cycles == N300D_NOC_HOP_LATENCY_CYCLES for channel in link.channels)
        for link in mesh.noc.links
    )
    assert all(endpoint.ingress_latency_cycles == N300D_NIU_LATENCY_CYCLES for endpoint in l1_endpoints + l2_endpoints)
    assert all(endpoint.egress_latency_cycles == N300D_NIU_LATENCY_CYCLES for endpoint in l1_endpoints + l2_endpoints)

    reserved_rows = set(_n300d_reserved_rows(N300D_MESH_HEIGHT))
    compute_rows = {
        mesh.noc.node_by_id(endpoint.node_id).y
        for endpoint in l1_endpoints
    }
    assert reserved_rows.isdisjoint(compute_rows)


def test_n300d_mesh_scales_noc_with_custom_compute_shape() -> None:
    mesh = wormhole_n300d_mesh(width=9, height=9)

    assert mesh.shape == (9, 9)
    assert mesh.num_tiles == 81
    assert len(mesh.noc.nodes) == (9 + N300D_NOC_WIDTH_PADDING) * (9 + N300D_NOC_HEIGHT_PADDING)
    assert len(mesh.noc.endpoints_of_kind(EndpointKind.L1)) == 81
    assert tuple(
        mesh.noc.node_by_id(endpoint.node_id).coords
        for endpoint in mesh.noc.endpoints_of_kind(EndpointKind.L1)
    ) == _n300d_tile_noc_coords(9, 9)
    assert tuple(
        mesh.noc.node_by_id(endpoint.node_id).coords
        for endpoint in mesh.noc.endpoints_of_kind(EndpointKind.L2)
    ) == _n300d_l2_endpoint_coords(9, 9)
