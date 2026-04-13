from MAPS.arch import Mesh, Tile
from MAPS.core.graph import Edge, Graph, Node, OpKind
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.ops.gemm import GemmLayerOp
from MAPS.planner import balance_stage_plans, balance_workload
from MAPS.planner.workload_balancing import (
    _best_stage_plan_for_tile_count,
    _estimate_local_num_microbatches,
    _estimate_node_num_microbatches,
    _tile_count_options_after_growth,
    _propagate_num_microbatches,
    _topological_stage_ids,
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


def _mesh_with_l1(width: int, height: int, l1_bytes: int) -> Mesh:
    tiles = tuple(
        Tile(tile_id=(y * width + x), x=x, y=y, l1_bytes=l1_bytes)
        for y in range(height)
        for x in range(width)
    )
    return Mesh(width, height, l2_bytes=4096, tiles=tiles)


def test_balance_workload_uses_full_tile_budget() -> None:
    node0 = _gemm_node("gemm0", m=16, k=16, n=16)
    node1 = _gemm_node("gemm1", m=16, k=16, n=16)
    graph = Graph(name="g", nodes=(node0, node1))
    mesh = _mesh_with_l1(4, 4, l1_bytes=4096)

    allocation = balance_workload(graph, mesh)

    assert allocation == {0: 8, 1: 8}
    assert sum(allocation.values()) == mesh.num_tiles


def test_balance_workload_gives_more_tiles_to_heavier_gemm() -> None:
    heavy = _gemm_node("heavy", m=64, k=64, n=64)
    light = _gemm_node("light", m=8, k=8, n=8)
    graph = Graph(name="g", nodes=(heavy, light))
    mesh = _mesh_with_l1(3, 2, l1_bytes=32768)

    allocation = balance_workload(graph, mesh)

    assert allocation[0] > allocation[1]
    assert sum(allocation.values()) == mesh.num_tiles


def test_balance_stage_plans_preserve_layout_decisions() -> None:
    node = _gemm_node("gemm", m=16, k=16, n=16)
    graph = Graph(name="g", nodes=(node,))
    mesh = _mesh_with_l1(4, 1, l1_bytes=32768)

    plans = balance_stage_plans(graph, mesh)

    assert plans[0].tile_count == 4
    assert plans[0].logical_shape[0] * plans[0].logical_shape[1] == 4
    assert plans[0].output_layouts[0].logical_width == plans[0].logical_shape[0]
    assert plans[0].output_layouts[0].logical_height == plans[0].logical_shape[1]


def test_best_stage_plan_selects_best_logical_shape_for_fixed_tile_count() -> None:
    node = _gemm_node("gemm", m=4, k=16, n=7)
    mesh = _mesh_with_l1(6, 1, l1_bytes=32768)

    plan = _best_stage_plan_for_tile_count(
        node=node,
        mesh=mesh,
        stage_id=0,
        tile_count=6,
        num_microbatches=1,
        debug=False,
    )

    assert plan.tile_count == 6
    assert plan.logical_shape == (3, 2)


def test_tile_count_growth_skips_counts_without_rectangular_placement() -> None:
    mesh = Mesh(2, 2, l2_bytes=4096)

    options = _tile_count_options_after_growth(
        current_tile_count=2,
        remaining_tiles=2,
        mesh=mesh,
    )

    assert options == (4,)


def test_estimate_node_num_microbatches_grows_until_inputs_fit() -> None:
    node = _batched_gemm_node("batched", b=4, m=4, k=4, n=4)
    mesh = _mesh_with_l1(1, 1, l1_bytes=64)
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=1, height=1)

    num_microbatches = _estimate_node_num_microbatches(node, submesh)

    assert num_microbatches == 4


def test_propagate_num_microbatches_respects_predecessor_requirement() -> None:
    producer = _batched_gemm_node("producer", b=4, m=4, k=4, n=4)
    consumer_input = producer.outputs[0]
    consumer_w = Tensor(name="consumer_w", rank=3, dims=(4, 4, 4), elem_bytes=2)
    consumer_out = Tensor(name="consumer_out", rank=3, dims=(4, 4, 4), elem_bytes=2)
    consumer_op = GemmLayerOp(x=consumer_input, w=consumer_w, y=None, output=consumer_out)
    consumer = Node(
        name="consumer",
        kind=OpKind.GEMM,
        inputs=(consumer_input, consumer_w),
        outputs=(consumer_out,),
        payload=consumer_op,
    )
    graph = Graph(name="g", nodes=(producer, consumer))
    mesh = _mesh_with_l1(2, 1, l1_bytes=64)
    local_num_microbatches = _estimate_local_num_microbatches(
        graph,
        mesh,
        {0: (1, 1), 1: (1, 1)},
        debug=False,
    )

    num_microbatches = _propagate_num_microbatches(
        graph,
        local_num_microbatches,
        debug=False,
    )

    assert num_microbatches == {0: 4, 1: 4}


def test_estimate_local_num_microbatches_is_node_local_only() -> None:
    producer = _batched_gemm_node("producer", b=4, m=4, k=4, n=4)
    consumer_input = producer.outputs[0]
    consumer_w = Tensor(name="consumer_w", rank=3, dims=(4, 4, 4), elem_bytes=2)
    consumer_out = Tensor(name="consumer_out", rank=3, dims=(4, 4, 4), elem_bytes=2)
    consumer_op = GemmLayerOp(x=consumer_input, w=consumer_w, y=None, output=consumer_out)
    consumer = Node(
        name="consumer",
        kind=OpKind.GEMM,
        inputs=(consumer_input, consumer_w),
        outputs=(consumer_out,),
        payload=consumer_op,
    )
    graph = Graph(name="g", nodes=(producer, consumer))
    mesh = _mesh_with_l1(2, 1, l1_bytes=64)

    local_num_microbatches = _estimate_local_num_microbatches(
        graph,
        mesh,
        {0: (1, 1), 1: (1, 1)},
        debug=False,
    )

    assert local_num_microbatches == {0: 4, 1: 4}


def test_topological_stage_ids_follow_graph_edges_not_node_order() -> None:
    producer = _batched_gemm_node("producer", b=4, m=4, k=4, n=4)
    consumer_input = producer.outputs[0]
    consumer_w = Tensor(name="consumer_w", rank=3, dims=(4, 4, 4), elem_bytes=2)
    consumer_out = Tensor(name="consumer_out", rank=3, dims=(4, 4, 4), elem_bytes=2)
    consumer_op = GemmLayerOp(x=consumer_input, w=consumer_w, y=None, output=consumer_out)
    consumer = Node(
        name="consumer",
        kind=OpKind.GEMM,
        inputs=(consumer_input, consumer_w),
        outputs=(consumer_out,),
        payload=consumer_op,
    )
    graph = Graph(
        name="g",
        nodes=(consumer, producer),
        edges=(Edge(tensor=consumer_input, src=producer, dst=consumer),),
    )

    order = _topological_stage_ids(graph)

    assert order == (1, 0)


def test_propagate_num_microbatches_uses_topological_graph_order() -> None:
    producer = _batched_gemm_node("producer", b=4, m=4, k=4, n=4)
    consumer_input = producer.outputs[0]
    consumer_w = Tensor(name="consumer_w", rank=3, dims=(4, 4, 4), elem_bytes=2)
    consumer_out = Tensor(name="consumer_out", rank=3, dims=(4, 4, 4), elem_bytes=2)
    consumer_op = GemmLayerOp(x=consumer_input, w=consumer_w, y=None, output=consumer_out)
    consumer = Node(
        name="consumer",
        kind=OpKind.GEMM,
        inputs=(consumer_input, consumer_w),
        outputs=(consumer_out,),
        payload=consumer_op,
    )
    graph = Graph(
        name="g",
        nodes=(consumer, producer),
        edges=(Edge(tensor=consumer_input, src=producer, dst=consumer),),
    )
    mesh = _mesh_with_l1(2, 1, l1_bytes=64)
    local_num_microbatches = _estimate_local_num_microbatches(
        graph,
        mesh,
        {0: (1, 1), 1: (1, 1)},
        debug=False,
    )

    num_microbatches = _propagate_num_microbatches(
        graph,
        local_num_microbatches,
        debug=False,
    )

    assert num_microbatches == {1: 4, 0: 4}
