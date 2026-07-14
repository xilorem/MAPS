from MAPS.arch import L2Memory, Mesh
from MAPS.core.graph import Edge, Graph, Node, OpKind
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.ops.defs.gemm import GemmPayload
from MAPS.planner.placement import StagePlacement
import MAPS.planner.spatial_mapping as spatial_mapping
from MAPS.planner.spatial_mapping import build_virtual_traffic, map_spatially
from MAPS.planner.workload_balancing import StagePlan
from tests.noc_utils import rectangular_test_noc, rectangular_test_tiles


def _test_mesh(width: int, height: int) -> Mesh:
    return Mesh(
        width=width,
        height=height,
        l2_memory=L2Memory(size=4096, bandwidth=1),
        noc=rectangular_test_noc(width, height),
        tiles=rectangular_test_tiles(width, height),
    )


def _gemm_node(name: str, x: Tensor | None = None) -> Node:
    input_tensor = x if x is not None else Tensor(name=f"{name}_x", rank=2, dims=(8, 8), elem_bytes=2)
    weight_tensor = Tensor(name=f"{name}_w", rank=2, dims=(8, 8), elem_bytes=2)
    output_tensor = Tensor(name=f"{name}_out", rank=2, dims=(8, 8), elem_bytes=2)
    op = GemmPayload(x=input_tensor, w=weight_tensor, y=None, output=output_tensor)
    return Node(
        name=name,
        kind=OpKind.GEMM,
        inputs=(input_tensor, weight_tensor),
        outputs=(output_tensor,),
        payload=op,
    )


def _single_node_stage_plan(mesh: Mesh, stage_id: int, node: Node, tile_ids: set[int]) -> StagePlan:
    virtual_submesh = Submesh(mesh=mesh, submesh_id=stage_id, tile_ids=frozenset(tile_ids))
    output_layouts = node.payload.output_layouts(virtual_submesh, logical_shape=(len(tile_ids), 1))
    return StagePlan(
        stage_id=stage_id,
        tile_count=len(tile_ids),
        logical_shape=(len(tile_ids), 1),
        output_layouts=output_layouts,
        nodes=(node,),
        node_output_layouts=(output_layouts,),
    )


def _share_boundary(mesh: Mesh, left: set[int], right: set[int]) -> bool:
    for tile_id in left:
        x, y = mesh.coords(tile_id)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx = x + dx
            ny = y + dy
            if mesh.contains_coord(nx, ny) and mesh.tile_id(nx, ny) in right:
                return True
    return False


def test_build_virtual_traffic_tracks_inter_stage_bytes() -> None:
    mesh = _test_mesh(4, 2)
    producer = _gemm_node("producer")
    consumer = _gemm_node("consumer", x=producer.outputs[0])
    graph = Graph(
        name="g",
        nodes=(producer, consumer),
        edges=(Edge(tensor=producer.outputs[0], src=producer, dst=consumer),),
    )
    stage_plans = {
        0: _single_node_stage_plan(mesh, 0, producer, {0, 1}),
        1: _single_node_stage_plan(mesh, 1, consumer, {0, 1}),
    }

    traffic = build_virtual_traffic(
        graph=graph,
        mesh=mesh,
        stage_plans=stage_plans,
        node_stage_ids={id(producer): 0, id(consumer): 1},
    )

    assert traffic.stage_comm[(0, 1)] > 0
    assert sum(traffic.input_weights[1].values()) > 0
    assert sum(traffic.output_weights[0].values()) > 0


def test_map_spatially_returns_connected_adjacent_mapping() -> None:
    mesh = _test_mesh(4, 2)
    producer = _gemm_node("producer")
    consumer = _gemm_node("consumer", x=producer.outputs[0])
    graph = Graph(
        name="g",
        nodes=(producer, consumer),
        edges=(Edge(tensor=producer.outputs[0], src=producer, dst=consumer),),
    )
    stage_plans = {
        0: _single_node_stage_plan(mesh, 0, producer, {0, 1}),
        1: _single_node_stage_plan(mesh, 1, consumer, {0, 1}),
    }

    mapping = map_spatially(
        graph=graph,
        mesh=mesh,
        stage_plans=stage_plans,
        print_mapping=False,
        show_progress=False,
    )

    assert set(mapping) == {0, 1}
    all_tile_ids = set()
    for stage_id, placement in mapping.items():
        assert placement.physical_submesh.num_tiles == stage_plans[stage_id].tile_count
        assert len(placement.virtual_to_physical) == stage_plans[stage_id].tile_count
        assert len(set(placement.virtual_to_physical.values())) == stage_plans[stage_id].tile_count
        all_tile_ids |= set(placement.physical_submesh.tile_ids)

    assert len(all_tile_ids) == 4
    assert _share_boundary(
        mesh,
        set(mapping[0].physical_submesh.tile_ids),
        set(mapping[1].physical_submesh.tile_ids),
    )


def test_mapping_charges_l1_communication_to_the_producer_tile() -> None:
    mesh = _test_mesh(2, 1)
    producer = _gemm_node("producer")
    consumer = _gemm_node("consumer", x=producer.outputs[0])
    graph = Graph(
        name="g",
        nodes=(producer, consumer),
        edges=(Edge(tensor=producer.outputs[0], src=producer, dst=consumer),),
    )
    stage_plans = {
        0: _single_node_stage_plan(mesh, 0, producer, {0}),
        1: _single_node_stage_plan(mesh, 1, consumer, {0}),
    }
    placements = {
        stage_id: StagePlacement(
            stage_id=stage_id,
            virtual_submesh=spatial_mapping._stage_virtual_submesh(plan),
            physical_submesh=Submesh(mesh=mesh, submesh_id=stage_id, tile_ids=frozenset({stage_id})),
            virtual_to_physical={0: stage_id},
        )
        for stage_id, plan in stage_plans.items()
    }

    evaluation = spatial_mapping._evaluate_mapping(
        graph=graph,
        mesh=mesh,
        stage_plans=stage_plans,
        placements=placements,
        node_stage_ids={id(producer): 0, id(consumer): 1},
    )

    producer_score = evaluation.tile_scores[0]
    consumer_score = evaluation.tile_scores[1]
    assert producer_score.tile_to_tile_writes > 0
    assert producer_score.consumer_stage_writes == {1: producer_score.tile_to_tile_writes}
    assert consumer_score.tile_to_tile_writes == 0
    assert evaluation.stage_breakdowns[0].l1_write == producer_score.tile_to_tile_writes
    assert producer_score.score == (
        producer_score.l2_reads + producer_score.l2_writes + producer_score.tile_to_tile_writes
    )


def test_repair_region_skips_an_infeasible_growth_attempt(monkeypatch) -> None:
    mesh = _test_mesh(2, 1)
    submesh = Submesh(mesh=mesh, submesh_id=0, tile_ids=frozenset({0}))
    placement = StagePlacement(
        stage_id=0,
        virtual_submesh=submesh,
        physical_submesh=submesh,
        virtual_to_physical={0: 0},
    )
    traffic = spatial_mapping.VirtualTraffic(
        stage_comm={},
        edge_matrices={},
        input_weights={},
        output_weights={},
        l2_read_weights={},
        l2_write_weights={},
        communication_degree={},
        bottleneck_risk={},
        l2_pressure={},
    )

    def fail_growth(**kwargs) -> set[int]:
        del kwargs
        raise ValueError("infeasible region")

    monkeypatch.setattr(spatial_mapping, "_grow_stage_region", fail_growth)

    assert spatial_mapping._repair_region(
        mesh=mesh,
        stage_plans={0: StagePlan(stage_id=0, tile_count=1, logical_shape=(1, 1), output_layouts=())},
        current_placements={0: placement},
        traffic=traffic,
        affected_stages=frozenset({0}),
        focus_stage_id=0,
        debug=False,
    ) is None
