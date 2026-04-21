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
    Tile,
    TrafficKind,
    TrafficPolicy,
)
from MAPS.cost_models import TransferKind, TransferLeg, TransportCostModel


def test_l2_transfer_cost_uses_l1_and_l2_bandwidth() -> None:
    slow_tile = Tile(
        tile_id=0,
        x=0,
        y=0,
        memory=L1Memory(size=4096, bandwidth=4),
    )
    fast_tile = Tile(
        tile_id=0,
        x=0,
        y=0,
        memory=L1Memory(size=4096, bandwidth=16),
    )
    slow_mesh = Mesh(
        width=1,
        height=1,
        l2_memory=L2Memory(size=4096, bandwidth=4),
        tiles=(slow_tile,),
    )
    fast_mesh = Mesh(
        width=1,
        height=1,
        l2_memory=L2Memory(size=4096, bandwidth=16),
        tiles=(fast_tile,),
    )

    assert TransportCostModel(mesh=fast_mesh).l1_to_l2(fast_tile, 64) < (
        TransportCostModel(mesh=slow_mesh).l1_to_l2(slow_tile, 64)
    )


def test_l2_transfer_cost_uses_nearest_l2_access_point_distance() -> None:
    mesh = Mesh(
        width=4,
        height=1,
        l2_memory=L2Memory(size=4096, access_points=((0, 0),), bandwidth=8),
        tiles=(
            Tile(tile_id=0, x=0, y=0, memory=L1Memory(size=4096, bandwidth=8)),
            Tile(tile_id=1, x=1, y=0, memory=L1Memory(size=4096, bandwidth=8)),
            Tile(tile_id=2, x=2, y=0, memory=L1Memory(size=4096, bandwidth=8)),
            Tile(tile_id=3, x=3, y=0, memory=L1Memory(size=4096, bandwidth=8)),
        ),
    )
    model = TransportCostModel(mesh=mesh)

    assert model.l1_to_l2(mesh.tile(3, 0), 64) > model.l1_to_l2(mesh.tile(0, 0), 64)
    assert model.l2_to_l1(mesh.tile(3, 0), 64) > model.l2_to_l1(mesh.tile(0, 0), 64)


def test_l1_to_l1_transfer_cost_uses_tile_bandwidth_and_distance() -> None:
    src = Tile(tile_id=0, x=0, y=0, memory=L1Memory(size=4096, bandwidth=16))
    fast_dst = Tile(tile_id=1, x=1, y=0, memory=L1Memory(size=4096, bandwidth=16))
    slow_dst = Tile(tile_id=2, x=1, y=0, memory=L1Memory(size=4096, bandwidth=4))
    far_dst = Tile(tile_id=3, x=3, y=0, memory=L1Memory(size=4096, bandwidth=16))
    model = TransportCostModel()

    assert model.l1_to_l1(src, slow_dst, 64) > model.l1_to_l1(src, fast_dst, 64)
    assert model.l1_to_l1(src, far_dst, 64) > model.l1_to_l1(src, fast_dst, 64)


def test_l1_to_l1_transfer_cost_uses_noc_route_hops_when_available() -> None:
    mesh = Mesh(
        width=3,
        height=1,
        l2_memory=L2Memory(size=4096),
        tiles=(
            Tile(tile_id=0, x=0, y=0, memory=L1Memory(size=4096, bandwidth=16)),
            Tile(tile_id=1, x=1, y=0, memory=L1Memory(size=4096, bandwidth=16)),
            Tile(tile_id=2, x=2, y=0, memory=L1Memory(size=4096, bandwidth=16)),
        ),
        noc=NoC(
            nodes=(
                NoCNode(node_id=0, x=0, y=0),
                NoCNode(node_id=1, x=1, y=0),
                NoCNode(node_id=2, x=2, y=0),
            ),
            links=(
                NoCLink(
                    link_id=0,
                    src_node_id=0,
                    dst_node_id=1,
                    channels=(NoCChannel(channel_id=0, width_bytes=8, hop_latency_cycles=2.0),),
                    bidirectional=True,
                ),
                NoCLink(
                    link_id=1,
                    src_node_id=1,
                    dst_node_id=2,
                    channels=(NoCChannel(channel_id=0, width_bytes=8, hop_latency_cycles=2.0),),
                    bidirectional=True,
                ),
            ),
            endpoints=(
                NoCEndpoint(endpoint_id=0, kind=EndpointKind.L1, node_id=0, tile_id=0),
                NoCEndpoint(endpoint_id=1, kind=EndpointKind.L1, node_id=1, tile_id=1),
                NoCEndpoint(endpoint_id=2, kind=EndpointKind.L1, node_id=2, tile_id=2),
            ),
        ),
    )
    model = TransportCostModel(mesh=mesh)

    assert model.l1_to_l1(mesh.tile(0, 0), mesh.tile(2, 0), 64) > model.l1_to_l1(mesh.tile(0, 0), mesh.tile(1, 0), 64)


def test_l1_to_l1_transfer_cost_respects_read_req_and_rsp_traffic_policy_channel_selection() -> None:
    wide_mesh = Mesh(
        width=2,
        height=1,
        l2_memory=L2Memory(size=4096),
        tiles=(
            Tile(tile_id=0, x=0, y=0, memory=L1Memory(size=4096, bandwidth=64)),
            Tile(tile_id=1, x=1, y=0, memory=L1Memory(size=4096, bandwidth=64)),
        ),
        noc=NoC(
            nodes=(
                NoCNode(node_id=0, x=0, y=0),
                NoCNode(node_id=1, x=1, y=0),
            ),
            links=(
                NoCLink(
                    link_id=0,
                    src_node_id=0,
                    dst_node_id=1,
                    channels=(
                        NoCChannel(channel_id=0, width_bytes=4, hop_latency_cycles=1.0),
                        NoCChannel(channel_id=1, width_bytes=16, hop_latency_cycles=1.0),
                    ),
                    bidirectional=True,
                ),
            ),
            endpoints=(
                NoCEndpoint(endpoint_id=0, kind=EndpointKind.L1, node_id=0, tile_id=0),
                NoCEndpoint(endpoint_id=1, kind=EndpointKind.L1, node_id=1, tile_id=1),
            ),
            traffic_policy=TrafficPolicy(
                {
                    TrafficKind.READ_REQ: (1,),
                    TrafficKind.READ_RSP: (1,),
                }
            ),
        ),
    )
    narrow_mesh = Mesh(
        width=2,
        height=1,
        l2_memory=L2Memory(size=4096),
        tiles=wide_mesh.tiles,
        noc=NoC(
            nodes=wide_mesh.noc.nodes,
            links=wide_mesh.noc.links,
            endpoints=wide_mesh.noc.endpoints,
            traffic_policy=TrafficPolicy(
                {
                    TrafficKind.READ_REQ: (0,),
                    TrafficKind.READ_RSP: (0,),
                }
            ),
        ),
    )

    wide_cost = TransportCostModel(mesh=wide_mesh, read_request_bytes=64).l1_to_l1(
        wide_mesh.tile(0, 0),
        wide_mesh.tile(1, 0),
        64,
    )
    narrow_cost = TransportCostModel(mesh=narrow_mesh, read_request_bytes=64).l1_to_l1(
        narrow_mesh.tile(0, 0),
        narrow_mesh.tile(1, 0),
        64,
    )

    assert narrow_cost > wide_cost


def test_l2_transfer_cost_uses_noc_route_to_nearest_l2_endpoint_when_available() -> None:
    mesh = Mesh(
        width=3,
        height=1,
        l2_memory=L2Memory(size=4096, bandwidth=16),
        tiles=(
            Tile(tile_id=0, x=0, y=0, memory=L1Memory(size=4096, bandwidth=16)),
            Tile(tile_id=1, x=1, y=0, memory=L1Memory(size=4096, bandwidth=16)),
            Tile(tile_id=2, x=2, y=0, memory=L1Memory(size=4096, bandwidth=16)),
        ),
        noc=NoC(
            nodes=(
                NoCNode(node_id=0, x=0, y=0),
                NoCNode(node_id=1, x=1, y=0),
                NoCNode(node_id=2, x=2, y=0),
            ),
            links=(
                NoCLink(
                    link_id=0,
                    src_node_id=0,
                    dst_node_id=1,
                    channels=(NoCChannel(channel_id=0, width_bytes=8, hop_latency_cycles=2.0),),
                    bidirectional=True,
                ),
                NoCLink(
                    link_id=1,
                    src_node_id=1,
                    dst_node_id=2,
                    channels=(NoCChannel(channel_id=0, width_bytes=8, hop_latency_cycles=2.0),),
                    bidirectional=True,
                ),
            ),
            endpoints=(
                NoCEndpoint(endpoint_id=0, kind=EndpointKind.L1, node_id=0, tile_id=0),
                NoCEndpoint(endpoint_id=1, kind=EndpointKind.L1, node_id=1, tile_id=1),
                NoCEndpoint(endpoint_id=2, kind=EndpointKind.L1, node_id=2, tile_id=2),
                NoCEndpoint(endpoint_id=3, kind=EndpointKind.L2, node_id=0, name="l2_west"),
                NoCEndpoint(endpoint_id=4, kind=EndpointKind.L2, node_id=2, name="l2_east"),
            ),
        ),
    )
    model = TransportCostModel(mesh=mesh)

    assert model.l1_to_l2(mesh.tile(0, 0), 64) < model.l1_to_l2(mesh.tile(1, 0), 64)
    assert model.l2_to_l1(mesh.tile(2, 0), 64) < model.l2_to_l1(mesh.tile(1, 0), 64)


def test_l2_transfer_cost_respects_directional_traffic_policy_channel_selection() -> None:
    write_wide_read_narrow_mesh = Mesh(
        width=2,
        height=1,
        l2_memory=L2Memory(size=4096, bandwidth=64),
        tiles=(
            Tile(tile_id=0, x=0, y=0, memory=L1Memory(size=4096, bandwidth=64)),
            Tile(tile_id=1, x=1, y=0, memory=L1Memory(size=4096, bandwidth=64)),
        ),
        noc=NoC(
            nodes=(
                NoCNode(node_id=0, x=0, y=0),
                NoCNode(node_id=1, x=1, y=0),
            ),
            links=(
                NoCLink(
                    link_id=0,
                    src_node_id=0,
                    dst_node_id=1,
                    channels=(
                        NoCChannel(channel_id=0, width_bytes=4, hop_latency_cycles=1.0),
                        NoCChannel(channel_id=1, width_bytes=16, hop_latency_cycles=1.0),
                    ),
                    bidirectional=True,
                ),
            ),
            endpoints=(
                NoCEndpoint(endpoint_id=0, kind=EndpointKind.L1, node_id=0, tile_id=0),
                NoCEndpoint(endpoint_id=1, kind=EndpointKind.L1, node_id=1, tile_id=1),
                NoCEndpoint(endpoint_id=2, kind=EndpointKind.L2, node_id=1, name="l2"),
            ),
            traffic_policy=TrafficPolicy(
                {
                    TrafficKind.WRITE_REQ: (0,),
                    TrafficKind.WRITE_DATA: (1,),
                    TrafficKind.READ_REQ: (0,),
                    TrafficKind.READ_RSP: (0,),
                    TrafficKind.WRITE_RSP: (0,),
                }
            ),
        ),
    )
    write_narrow_read_wide_mesh = Mesh(
        width=2,
        height=1,
        l2_memory=L2Memory(size=4096, bandwidth=64),
        tiles=write_wide_read_narrow_mesh.tiles,
        noc=NoC(
            nodes=write_wide_read_narrow_mesh.noc.nodes,
            links=write_wide_read_narrow_mesh.noc.links,
            endpoints=write_wide_read_narrow_mesh.noc.endpoints,
            traffic_policy=TrafficPolicy(
                {
                    TrafficKind.WRITE_REQ: (0,),
                    TrafficKind.WRITE_DATA: (0,),
                    TrafficKind.READ_REQ: (0,),
                    TrafficKind.READ_RSP: (1,),
                    TrafficKind.WRITE_RSP: (1,),
                }
            ),
        ),
    )

    write_wide_read_narrow_model = TransportCostModel(mesh=write_wide_read_narrow_mesh)
    write_narrow_read_wide_model = TransportCostModel(mesh=write_narrow_read_wide_mesh)

    assert (
        write_wide_read_narrow_model.l1_to_l2(write_wide_read_narrow_mesh.tile(0, 0), 64)
        < write_narrow_read_wide_model.l1_to_l2(write_narrow_read_wide_mesh.tile(0, 0), 64)
    )
    assert (
        write_wide_read_narrow_model.l2_to_l1(write_wide_read_narrow_mesh.tile(0, 0), 64)
        > write_narrow_read_wide_model.l2_to_l1(write_narrow_read_wide_mesh.tile(0, 0), 64)
    )


def test_l2_transfer_cost_includes_noc_endpoint_attachment_latency_without_hops() -> None:
    mesh = Mesh(
        width=1,
        height=1,
        l2_memory=L2Memory(size=4096, bandwidth=16),
        tiles=(
            Tile(tile_id=0, x=0, y=0, memory=L1Memory(size=4096, bandwidth=16)),
        ),
        noc=NoC(
            nodes=(
                NoCNode(node_id=0, x=0, y=0),
            ),
            links=(),
            endpoints=(
                NoCEndpoint(
                    endpoint_id=0,
                    kind=EndpointKind.L1,
                    node_id=0,
                    tile_id=0,
                    ingress_latency_cycles=11.0,
                    egress_latency_cycles=3.0,
                ),
                NoCEndpoint(
                    endpoint_id=1,
                    kind=EndpointKind.L2,
                    node_id=0,
                    name="l2",
                    ingress_latency_cycles=5.0,
                    egress_latency_cycles=7.0,
                ),
            ),
        ),
    )
    model = TransportCostModel(mesh=mesh)

    assert model.l1_to_l2(mesh.tile(0, 0), 64) == 126.0
    assert model.l2_to_l1(mesh.tile(0, 0), 64) == 105.0


def test_l2_transfer_cost_uses_noc_endpoint_attachment_bandwidth_without_hops() -> None:
    mesh = Mesh(
        width=1,
        height=1,
        l2_memory=L2Memory(size=4096, bandwidth=64),
        tiles=(
            Tile(tile_id=0, x=0, y=0, memory=L1Memory(size=4096, bandwidth=64)),
        ),
        noc=NoC(
            nodes=(
                NoCNode(node_id=0, x=0, y=0),
            ),
            links=(),
            endpoints=(
                NoCEndpoint(
                    endpoint_id=0,
                    kind=EndpointKind.L1,
                    node_id=0,
                    tile_id=0,
                    ingress_bandwidth_bytes=32.0,
                    egress_bandwidth_bytes=8.0,
                ),
                NoCEndpoint(
                    endpoint_id=1,
                    kind=EndpointKind.L2,
                    node_id=0,
                    name="l2",
                    ingress_bandwidth_bytes=16.0,
                    egress_bandwidth_bytes=4.0,
                ),
            ),
        ),
    )
    model = TransportCostModel(mesh=mesh, account_noc_contention=True)

    l1_to_l2_estimate = model.estimate(
        TransferLeg(
            kind=TransferKind.L1_TO_L2,
            bytes=64,
            src_tile=mesh.tile(0, 0),
        )
    )
    l2_to_l1_estimate = model.estimate(
        TransferLeg(
            kind=TransferKind.L2_TO_L1,
            bytes=64,
            dst_tile=mesh.tile(0, 0),
        )
    )

    assert model.l1_to_l2(mesh.tile(0, 0), 64) == 96.375
    assert model.l2_to_l1(mesh.tile(0, 0), 64) == 91.125
    assert l1_to_l2_estimate.resource_loads == {
        "noc_endpoint:0:egress": 8.125,
        "noc_endpoint:1:ingress": 4.0625,
        "noc_endpoint:1:egress": 0.25,
        "noc_endpoint:0:ingress": 0.03125,
    }
    assert l2_to_l1_estimate.resource_loads == {
        "noc_endpoint:0:egress": 0.125,
        "noc_endpoint:1:ingress": 0.0625,
        "noc_endpoint:1:egress": 16.0,
        "noc_endpoint:0:ingress": 2.0,
    }


def test_l1_to_l2_transfer_cost_respects_write_rsp_traffic_policy_channel_selection() -> None:
    ack_wide_mesh = Mesh(
        width=2,
        height=1,
        l2_memory=L2Memory(size=4096, bandwidth=64),
        tiles=(
            Tile(tile_id=0, x=0, y=0, memory=L1Memory(size=4096, bandwidth=64)),
            Tile(tile_id=1, x=1, y=0, memory=L1Memory(size=4096, bandwidth=64)),
        ),
        noc=NoC(
            nodes=(
                NoCNode(node_id=0, x=0, y=0),
                NoCNode(node_id=1, x=1, y=0),
            ),
            links=(
                NoCLink(
                    link_id=0,
                    src_node_id=0,
                    dst_node_id=1,
                    channels=(
                        NoCChannel(channel_id=0, width_bytes=4, hop_latency_cycles=1.0),
                        NoCChannel(channel_id=1, width_bytes=16, hop_latency_cycles=1.0),
                    ),
                    bidirectional=True,
                ),
            ),
            endpoints=(
                NoCEndpoint(endpoint_id=0, kind=EndpointKind.L1, node_id=0, tile_id=0),
                NoCEndpoint(endpoint_id=1, kind=EndpointKind.L2, node_id=1, name="l2"),
            ),
            traffic_policy=TrafficPolicy(
                {
                    TrafficKind.WRITE_REQ: (0,),
                    TrafficKind.WRITE_DATA: (0,),
                    TrafficKind.WRITE_RSP: (1,),
                }
            ),
        ),
    )
    ack_narrow_mesh = Mesh(
        width=2,
        height=1,
        l2_memory=L2Memory(size=4096, bandwidth=64),
        tiles=ack_wide_mesh.tiles,
        noc=NoC(
            nodes=ack_wide_mesh.noc.nodes,
            links=ack_wide_mesh.noc.links,
            endpoints=ack_wide_mesh.noc.endpoints,
            traffic_policy=TrafficPolicy(
                {
                    TrafficKind.WRITE_REQ: (0,),
                    TrafficKind.WRITE_DATA: (0,),
                    TrafficKind.WRITE_RSP: (0,),
                }
            ),
        ),
    )

    ack_wide_model = TransportCostModel(mesh=ack_wide_mesh, write_response_bytes=64)
    ack_narrow_model = TransportCostModel(mesh=ack_narrow_mesh, write_response_bytes=64)

    assert ack_narrow_model.l1_to_l2(ack_narrow_mesh.tile(0, 0), 64) > (
        ack_wide_model.l1_to_l2(ack_wide_mesh.tile(0, 0), 64)
    )
