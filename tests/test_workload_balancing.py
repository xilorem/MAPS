from MAPS.arch import L1Memory, L2Memory, Mesh
from MAPS.core.graph import Edge, Graph, Node, OpKind
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.ops.defs.gemm import GemmPayload
from MAPS.planner import balance_workload
from MAPS.planner.select_stage import select_stages
from MAPS.planner.workload_balancing import (
    _best_stage_plan_for_stage_nodes,
    _estimate_selection_metrics,
    _plan_all_stages_for_tile_counts,
    _virtual_communication_cycles,
    grow_tile_count_for_stage,
)
from tests.noc_utils import rectangular_test_noc, rectangular_test_tiles


def _gemm_node(name: str, m: int, k: int, n: int) -> Node:
    x = Tensor(name=f"{name}_x", rank=2, dims=(m, k), elem_bytes=2)
    w = Tensor(name=f"{name}_w", rank=2, dims=(k, n), elem_bytes=2)
    out = Tensor(name=f"{name}_out", rank=2, dims=(m, n), elem_bytes=2)
    op = GemmPayload(x=x, w=w, y=None, output=out)
    return Node(
        name=name,
        kind=OpKind.GEMM,
        inputs=(x, w),
        outputs=(out,),
        payload=op,
    )


def _batched_gemm_node(name: str, b: int, m: int, k: int, n: int) -> Node:
    x = Tensor(name=f"{name}_x", rank=3, dims=(b, m, k), elem_bytes=2)
    w = Tensor(name=f"{name}_w", rank=3, dims=(b, k, n), elem_bytes=2)
    out = Tensor(name=f"{name}_out", rank=3, dims=(b, m, n), elem_bytes=2)
    op = GemmPayload(x=x, w=w, y=None, output=out)
    return Node(
        name=name,
        kind=OpKind.GEMM,
        inputs=(x, w),
        outputs=(out,),
        payload=op,
    )


def _mesh_with_l1(width: int, height: int, l1_size: int) -> Mesh:
    return Mesh(
        width=width,
        height=height,
        l2_memory=L2Memory(size=4096, bandwidth=1),
        noc=rectangular_test_noc(width, height),
        tiles=rectangular_test_tiles(width, height, memory=L1Memory(size=l1_size, bandwidth=1)),
    )


def test_balance_workload_uses_full_tile_budget() -> None:
    node0 = _gemm_node("gemm0", m=16, k=16, n=16)
    node1 = _gemm_node("gemm1", m=16, k=16, n=16)
    graph = Graph(name="g", nodes=(node0, node1))
    mesh = _mesh_with_l1(4, 4, l1_size=4096)

    allocation = {
        stage_id: plan.tile_count
        for stage_id, plan in balance_workload(graph, mesh).items()
    }

    assert allocation == {0: 8, 1: 8}
    assert sum(allocation.values()) == mesh.num_tiles


def test_balance_workload_gives_more_tiles_to_heavier_gemm() -> None:
    heavy = _gemm_node("heavy", m=64, k=64, n=64)
    light = _gemm_node("light", m=8, k=8, n=8)
    graph = Graph(name="g", nodes=(heavy, light))
    mesh = _mesh_with_l1(3, 2, l1_size=32768)

    allocation = {
        stage_id: plan.tile_count
        for stage_id, plan in balance_workload(graph, mesh).items()
    }

    assert allocation[0] > allocation[1]
    assert sum(allocation.values()) == mesh.num_tiles


def test_virtual_traffic_charges_inter_stage_writes_to_producer_tiles() -> None:
    producer = _gemm_node("producer", m=8, k=8, n=8)
    consumer = _gemm_node("consumer", m=8, k=8, n=8)
    consumer = Node(
        name=consumer.name,
        kind=consumer.kind,
        inputs=(producer.outputs[0], consumer.inputs[1]),
        outputs=consumer.outputs,
        payload=consumer.payload,
    )
    graph = Graph(name="g", nodes=(producer, consumer))
    mesh = _mesh_with_l1(2, 1, l1_size=4096)
    stage_selection = {0: (producer,), 1: (consumer,)}
    plans = _plan_all_stages_for_tile_counts(
        stage_selection,
        mesh,
        tile_counts={0: 1, 1: 1},
        initializer_tensors=frozenset(),
        debug=False,
    )

    communication = _virtual_communication_cycles(graph, mesh, plans, {})

    assert communication[0][0] > communication[1][0]


def test_balance_workload_preserves_layout_decisions() -> None:
    node = _gemm_node("gemm", m=16, k=16, n=16)
    graph = Graph(name="g", nodes=(node,))
    mesh = _mesh_with_l1(4, 1, l1_size=32768)

    plans = balance_workload(graph, mesh)

    assert plans[0].tile_count == 4
    assert plans[0].logical_shape[0] * plans[0].logical_shape[1] == 4
    assert plans[0].output_layouts[0].logical_width == plans[0].logical_shape[0]
    assert plans[0].output_layouts[0].logical_height == plans[0].logical_shape[1]


def test_balance_workload_starts_from_minimum_l1_feasible_tile_count() -> None:
    node = _gemm_node("gemm", m=4, k=4, n=4)
    graph = Graph(name="g", nodes=(node,))
    mesh = _mesh_with_l1(2, 1, l1_size=80)

    plans = balance_workload(graph, mesh)

    assert plans[0].tile_count == 2


def test_balance_workload_accepts_explicit_stage_selection() -> None:
    node0 = _gemm_node("gemm0", m=16, k=16, n=16)
    node1 = _gemm_node("gemm1", m=16, k=16, n=16)
    graph = Graph(name="g", nodes=(node0, node1))
    mesh = _mesh_with_l1(2, 2, l1_size=4096)

    plans = balance_workload(
        graph,
        mesh,
        stage_selection={0: (node0, node1)},
    )

    assert tuple(plans) == (0,)
    assert plans[0].tile_count == mesh.num_tiles
    assert plans[0].nodes == (node0, node1)
    assert len(plans[0].node_output_layouts) == 2


def test_balance_workload_can_use_selected_stage_groups() -> None:
    node0 = _gemm_node("gemm0", m=16, k=16, n=16)
    node1 = _gemm_node(
        "gemm1",
        m=16,
        k=16,
        n=16,
    )
    node1 = Node(
        name=node1.name,
        kind=node1.kind,
        inputs=node1.inputs,
        outputs=node1.outputs,
        payload=node1.payload,
        attributes={"stage_group_id": "group0"},
    )
    node2 = _gemm_node("gemm2", m=16, k=16, n=16)
    node0 = Node(
        name=node0.name,
        kind=node0.kind,
        inputs=node0.inputs,
        outputs=node0.outputs,
        payload=node0.payload,
        attributes={"stage_group_id": "group0"},
    )
    graph = Graph(name="g", nodes=(node0, node1, node2))
    mesh = _mesh_with_l1(3, 2, l1_size=4096)

    plans = balance_workload(
        graph,
        mesh,
        stage_selection=select_stages(graph),
    )

    assert tuple(plans) == (0, 1)
    assert plans[0].nodes == (node0, node1)
    assert plans[1].nodes == (node2,)


def test_best_stage_plan_uses_l1_feasible_logical_shape_for_fixed_tile_count() -> None:
    node = _gemm_node("gemm", m=4, k=16, n=7)
    mesh = _mesh_with_l1(6, 1, l1_size=32768)

    plan = _best_stage_plan_for_stage_nodes(
        stage_nodes=(node,),
        mesh=mesh,
        stage_id=0,
        tile_count=6,
        initializer_tensors=frozenset(),
        debug=False,
    )

    assert plan.tile_count == 6
    assert plan.logical_shape[0] * plan.logical_shape[1] == 6


def test_growth_prefers_tile_count_with_more_physical_shape_options() -> None:
    node = _gemm_node("gemm", m=32, k=32, n=32)
    mesh = _mesh_with_l1(4, 4, l1_size=32768)
    stage_selection = {0: (node,)}
    current_plan = _best_stage_plan_for_stage_nodes(
        stage_nodes=(node,),
        mesh=mesh,
        stage_id=0,
        tile_count=2,
        initializer_tensors=frozenset(),
        debug=False,
    )
    current_metric = _estimate_selection_metrics(
        {0: current_plan},
        stage_selection,
        mesh=mesh,
        alpha=1.0,
        beta=1.0,
        graph_inputs=frozenset(),
        graph_outputs=frozenset(),
        producer_stage_id_by_tensor={},
        initializer_tensors=frozenset(),
    )[0]

    best_growth = grow_tile_count_for_stage(
        stage_id=0,
        stage_selection=stage_selection,
        mesh=mesh,
        tile_counts={0: 2},
        used_tiles=2,
        current_metric=current_metric,
        initializer_tensors=frozenset(),
        debug=False,
    )

    assert best_growth is not None
    assert best_growth > 2


def test_best_stage_plan_rejects_tile_work_that_does_not_fit() -> None:
    node = _batched_gemm_node("batched", b=4, m=4, k=4, n=4)
    mesh = _mesh_with_l1(1, 1, l1_size=64)

    try:
        _best_stage_plan_for_stage_nodes(
            stage_nodes=(node,),
            mesh=mesh,
            stage_id=0,
            tile_count=1,
            initializer_tensors=frozenset(),
            debug=False,
        )
    except ValueError as exc:
        assert "full tile-work slices" in str(exc)
    else:
        raise AssertionError("expected tile-work L1 fit failure")


def test_best_stage_plan_counts_outputs_in_l1_fit() -> None:
    node = _gemm_node("gemm", m=4, k=4, n=4)
    mesh = _mesh_with_l1(1, 1, l1_size=80)

    try:
        _best_stage_plan_for_stage_nodes(
            stage_nodes=(node,),
            mesh=mesh,
            stage_id=0,
            tile_count=1,
            initializer_tensors=frozenset(),
            debug=False,
        )
    except ValueError as exc:
        assert "full tile-work slices" in str(exc)
    else:
        raise AssertionError("expected output slice to be included in L1 fit")
