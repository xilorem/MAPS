from MAPS.arch import L1Memory, L2Memory, Mesh, Tile
from MAPS.core.graph import Edge, Graph, Node, OpKind
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.ops.gemm import GemmLayerOp
from MAPS.planner import balance_workload
from MAPS.planner.select_stage import select_stages
from MAPS.planner.workload_balancing import (
    _best_stage_plan_for_tile_count,
    _has_feasible_submesh_placement,
    _tile_count_options_after_growth,
)


def _gemm_node(name: str, m: int, k: int, n: int) -> Node:
    x = Tensor(name=f"{name}_x", rank=2, dims=(m, k), elem_bytes=2)
    w = Tensor(name=f"{name}_w", rank=2, dims=(k, n), elem_bytes=2)
    out = Tensor(name=f"{name}_out", rank=2, dims=(m, n), elem_bytes=2)
    op = GemmLayerOp(x=x, w=w, y=None, output=out)
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
    op = GemmLayerOp(x=x, w=w, y=None, output=out)
    return Node(
        name=name,
        kind=OpKind.GEMM,
        inputs=(x, w),
        outputs=(out,),
        payload=op,
    )


def _mesh_with_l1(width: int, height: int, l1_size: int) -> Mesh:
    tiles = tuple(
        Tile(tile_id=(y * width + x), x=x, y=y, memory=L1Memory(size=l1_size))
        for y in range(height)
        for x in range(width)
    )
    return Mesh(width, height, l2_memory=L2Memory(size=4096), tiles=tiles)


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
    assert len(plans[0].node_input_layouts) == 2
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


def test_best_stage_plan_selects_best_logical_shape_for_fixed_tile_count() -> None:
    node = _gemm_node("gemm", m=4, k=16, n=7)
    mesh = _mesh_with_l1(6, 1, l1_size=32768)

    plan = _best_stage_plan_for_tile_count(
        node=node,
        mesh=mesh,
        stage_id=0,
        tile_count=6,
        debug=False,
    )

    assert plan.tile_count == 6
    assert plan.logical_shape == (3, 2)


def test_tile_count_growth_skips_counts_without_rectangular_placement() -> None:
    mesh = Mesh(2, 2, l2_memory=L2Memory(size=4096))

    options = _tile_count_options_after_growth(
        current_tile_count=2,
        remaining_tiles=2,
        mesh=mesh,
    )

    assert options == (4,)


def test_submesh_placement_feasibility_requires_global_rectangle_packing() -> None:
    mesh = Mesh(3, 3, l2_memory=L2Memory(size=4096))

    feasible = _has_feasible_submesh_placement(
        {0: 4, 1: 4, 2: 1},
        mesh,
        placement_masks_by_tile_count={},
        feasibility_cache={},
    )

    assert not feasible


def test_best_stage_plan_rejects_tile_work_that_does_not_fit() -> None:
    node = _batched_gemm_node("batched", b=4, m=4, k=4, n=4)
    mesh = _mesh_with_l1(1, 1, l1_size=64)

    try:
        _best_stage_plan_for_tile_count(
            node=node,
            mesh=mesh,
            stage_id=0,
            tile_count=1,
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
        _best_stage_plan_for_tile_count(
            node=node,
            mesh=mesh,
            stage_id=0,
            tile_count=1,
            debug=False,
        )
    except ValueError as exc:
        assert "full tile-work slices" in str(exc)
    else:
        raise AssertionError("expected output slice to be included in L1 fit")
