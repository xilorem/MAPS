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
)
from MAPS.core import LayoutAxis, LayoutAxisMode, Submesh, Tensor, TensorLayout, TensorRange, TensorSlice
from MAPS.transitions.model import Transition, TransitionFragment, TransitionMode
from MAPS.transitions import TransportCostModel, estimate_transition_cost
from tests.noc_utils import rectangular_test_tiles


def _shared_link_remap_case() -> tuple[Mesh, Tensor, Transition]:
    mesh = Mesh(
        width=4,
        height=1,
        l2_memory=L2Memory(size=4096, bandwidth=1),
        tiles=rectangular_test_tiles(4, 1, memory=L1Memory(size=4096, bandwidth=64)),
        noc=NoC(
            nodes=(
                NoCNode(node_id=0, x=0, y=0),
                NoCNode(node_id=1, x=1, y=0),
                NoCNode(node_id=2, x=2, y=0),
                NoCNode(node_id=3, x=3, y=0),
            ),
            links=(
                NoCLink(
                    link_id=0,
                    src_node_id=0,
                    dst_node_id=1,
                    channels=(NoCChannel(channel_id=0, width_bytes=8),),
                    bidirectional=True,
                ),
                NoCLink(
                    link_id=1,
                    src_node_id=1,
                    dst_node_id=2,
                    channels=(NoCChannel(channel_id=0, width_bytes=8),),
                    bidirectional=True,
                ),
                NoCLink(
                    link_id=2,
                    src_node_id=2,
                    dst_node_id=3,
                    channels=(NoCChannel(channel_id=0, width_bytes=8),),
                    bidirectional=True,
                ),
            ),
            endpoints=(
                NoCEndpoint(endpoint_id=0, kind=EndpointKind.L1, node_id=0, tile_id=0),
                NoCEndpoint(endpoint_id=1, kind=EndpointKind.L1, node_id=1, tile_id=1),
                NoCEndpoint(endpoint_id=2, kind=EndpointKind.L1, node_id=2, tile_id=2),
                NoCEndpoint(endpoint_id=3, kind=EndpointKind.L1, node_id=3, tile_id=3),
            ),
        ),
    )
    tensor = Tensor(name="x", rank=1, dims=(16,), elem_bytes=8)
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=4, height=1)
    layout = TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
        logical_width=4,
        logical_height=1,
    )
    transition = Transition(
        name="remap",
        tensor_id=0,
        src_layer_id=0,
        src_output_idx=0,
        dst_layer_id=1,
        dst_input_idx=0,
        mode=TransitionMode.DIRECT_REMAP,
        src_layout=layout,
        dst_layout=layout,
        fragments=(
            TransitionFragment(
                src_hartid=0,
                dst_hartid=2,
                src_slice=TensorSlice(rank=1, dims=(TensorRange(start=0, length=8),)),
                dst_slice=TensorSlice(rank=1, dims=(TensorRange(start=0, length=8),)),
            ),
            TransitionFragment(
                src_hartid=1,
                dst_hartid=3,
                src_slice=TensorSlice(rank=1, dims=(TensorRange(start=8, length=8),)),
                dst_slice=TensorSlice(rank=1, dims=(TensorRange(start=8, length=8),)),
            ),
        ),
    )
    return mesh, tensor, transition


def test_direct_remap_cost_ignores_shared_noc_link_load_when_disabled() -> None:
    mesh, tensor, transition = _shared_link_remap_case()
    model = TransportCostModel(mesh=mesh)

    cost = estimate_transition_cost(transition, tensor, mesh, model)

    assert cost.producer_loads == {0: 9, 1: 9}
    assert cost.consumer_loads == {2: 9, 3: 9}
    assert cost.resource_loads == {
        "tile:2:dma:idma_read": 9,
        "tile:3:dma:idma_read": 9,
    }
    assert cost.total_cost == 9


def test_direct_remap_cost_accounts_for_shared_noc_link_load_when_enabled() -> None:
    mesh, tensor, transition = _shared_link_remap_case()
    model = TransportCostModel(mesh=mesh, account_noc_contention=True)

    cost = estimate_transition_cost(transition, tensor, mesh, model)

    assert cost.producer_loads == {0: 9, 1: 9}
    assert cost.consumer_loads == {2: 9, 3: 9}
    assert cost.resource_loads == {
        "noc_link:0:channel:0": 9,
        "noc_link:1:channel:0": 18,
        "noc_link:2:channel:0": 9,
        "tile:2:dma:idma_read": 9,
        "tile:3:dma:idma_read": 9,
    }
    assert cost.total_cost == 18
