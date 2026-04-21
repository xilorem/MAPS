import pytest

from MAPS.arch import (
    EndpointKind,
    NoC,
    NoCChannel,
    NoCEndpoint,
    NoCLink,
    NoCNode,
    NoCRoute,
    TrafficKind,
    TrafficPolicy,
)


def test_noc_preserves_lookup_helpers_and_link_directions() -> None:
    noc = NoC(
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
                    NoCChannel(channel_id=0, width_bytes=4, tag="req"),
                    NoCChannel(channel_id=1, width_bytes=32, tag="data"),
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
                TrafficKind.READ_REQ: (0,),
                TrafficKind.TRANSFER: (1,),
            }
        ),
    )

    assert noc.node_by_id(0).coords == (0, 0)
    assert noc.link_by_id(0).bidirectional is True
    assert noc.endpoint_by_id(1).name == "l2"
    assert noc.traffic_policy.allowed_channel_ids(TrafficKind.READ_REQ) == (0,)
    assert tuple(link.link_id for link in noc.outgoing_links(0)) == (0,)
    assert tuple(link.link_id for link in noc.outgoing_links(1)) == (0,)
    assert tuple(link.link_id for link in noc.incoming_links(0)) == (0,)
    assert tuple(link.link_id for link in noc.incoming_links(1)) == (0,)


def test_noc_rejects_duplicate_node_ids() -> None:
    with pytest.raises(ValueError, match="node ids must be unique"):
        NoC(
            nodes=(
                NoCNode(node_id=0, x=0, y=0),
                NoCNode(node_id=0, x=1, y=0),
            ),
            links=(),
        )


def test_noc_rejects_link_referencing_unknown_node() -> None:
    with pytest.raises(ValueError, match="link references unknown node_id"):
        NoC(
            nodes=(NoCNode(node_id=0, x=0, y=0),),
            links=(
                NoCLink(
                    link_id=0,
                    src_node_id=0,
                    dst_node_id=1,
                    channels=(NoCChannel(channel_id=0, width_bytes=4),),
                ),
            ),
        )


def test_noc_link_rejects_empty_channels() -> None:
    with pytest.raises(ValueError, match="link must have at least one channel"):
        NoCLink(
            link_id=0,
            src_node_id=0,
            dst_node_id=1,
            channels=(),
        )


def test_noc_link_rejects_duplicate_channel_ids() -> None:
    with pytest.raises(ValueError, match="link channel ids must be unique"):
        NoCLink(
            link_id=0,
            src_node_id=0,
            dst_node_id=1,
            channels=(
                NoCChannel(channel_id=0, width_bytes=4),
                NoCChannel(channel_id=0, width_bytes=8),
            ),
        )


def test_noc_rejects_policy_referencing_unknown_channel_ids() -> None:
    with pytest.raises(ValueError, match="references unknown channel ids"):
        NoC(
            nodes=(
                NoCNode(node_id=0, x=0, y=0),
                NoCNode(node_id=1, x=1, y=0),
            ),
            links=(
                NoCLink(
                    link_id=0,
                    src_node_id=0,
                    dst_node_id=1,
                    channels=(NoCChannel(channel_id=0, width_bytes=4),),
                ),
            ),
            traffic_policy=TrafficPolicy({TrafficKind.READ_REQ: (1,)}),
        )


def test_noc_route_requires_one_more_node_than_links() -> None:
    with pytest.raises(ValueError, match="route node_ids length must be link_ids length \\+ 1"):
        NoCRoute(
            src_endpoint_id=0,
            dst_endpoint_id=1,
            node_ids=(0, 1),
            link_ids=(0, 1),
        )
