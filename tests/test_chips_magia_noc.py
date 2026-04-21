from MAPS.arch import EndpointKind
from MAPS.chips.magia import (
    MAGIA_NOC_CHANNEL_WIDTH_BYTES,
    MAGIA_NOC_HOP_LATENCY_CYCLES,
    magia_mesh,
)


def test_magia_mesh_attaches_default_noc() -> None:
    mesh = magia_mesh(width=4, height=3)

    assert mesh.noc is not None
    assert len(mesh.noc.nodes) == 12
    assert len(mesh.noc.links) == 17
    assert len(mesh.noc.endpoints_of_kind(EndpointKind.L1)) == 12
    assert len(mesh.noc.endpoints_of_kind(EndpointKind.L2)) == 3
    assert mesh.noc.endpoint_for_tile(0, EndpointKind.L1).node_id == 0
    assert mesh.noc.endpoint_for_tile(11, EndpointKind.L1).node_id == 11
    assert tuple(endpoint.node_id for endpoint in mesh.noc.endpoints_of_kind(EndpointKind.L2)) == (0, 4, 8)
    assert all(link.bidirectional for link in mesh.noc.links)
    assert all(link.channels[0].width_bytes == MAGIA_NOC_CHANNEL_WIDTH_BYTES for link in mesh.noc.links)
    assert all(
        link.channels[0].hop_latency_cycles == MAGIA_NOC_HOP_LATENCY_CYCLES
        for link in mesh.noc.links
    )
