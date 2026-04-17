from MAPS.arch import L2Memory, Mesh
from MAPS.core.graph import Edge, Graph, Node, OpKind
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.ops.gemm import GemmLayerOp
from MAPS.planner.spatial_mapping import (
    _edge_placement_costs,
    _edge_shape_costs,
    _layout_on_submesh,
    _prune_placement_options_losslessly,
    _shape_options,
    _stage_io_costs,
    _stage_io_costs_for_placements,
    map_spatially,
    place_stage_plans,
    print_spatial_mapping_details,
)
from MAPS.planner.workload_balancing import StagePlan


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


def _assert_valid_mapping(
    mapping: dict[int, object],
    tile_counts: dict[int, int],
) -> None:
    assert set(mapping) == set(tile_counts)
    all_tiles = []
    for stage_id, tile_count in tile_counts.items():
        assert mapping[stage_id].num_tiles == tile_count
        all_tiles.extend(tile.tile_id for tile in mapping[stage_id].tiles)
    assert len(all_tiles) == len(set(all_tiles))

def test_shape_options_for_area_two_on_2x2_mesh() -> None:
    mesh = Mesh(2, 2, l2_memory=L2Memory(size=4096))

    shapes = _shape_options(2, mesh)

    assert set(shapes) == {(2, 1), (1, 2)}


def test_shape_options_raise_when_tile_count_cannot_fit_mesh() -> None:
    mesh = Mesh(2, 2, l2_memory=L2Memory(size=4096))

    try:
        _shape_options(3, mesh)
    except ValueError as exc:
        assert "no rectangular shape fits" in str(exc)
    else:
        raise AssertionError("expected _shape_options to raise for infeasible tile count")


def test_lossless_pruning_removes_placements_without_global_support() -> None:
    mesh = Mesh(4, 2, l2_memory=L2Memory(size=4096))
    placement_options = {
        0: (
            Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=1),
            Submesh(mesh=mesh, submesh_id=0, x0=0, y0=1, width=2, height=1),
        ),
        1: (
            Submesh(mesh=mesh, submesh_id=1, x0=0, y0=0, width=2, height=1),
        ),
        2: (
            Submesh(mesh=mesh, submesh_id=2, x0=2, y0=0, width=2, height=1),
        ),
    }

    pruned = _prune_placement_options_losslessly(placement_options)

    assert pruned[0] == (
        Submesh(mesh=mesh, submesh_id=0, x0=0, y0=1, width=2, height=1),
    )
    assert pruned[1] == placement_options[1]
    assert pruned[2] == placement_options[2]


def test_layout_on_submesh_preserves_logical_shape() -> None:
    mesh = Mesh(6, 2, l2_memory=L2Memory(size=4096))
    planning_submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=6, height=1)
    placed_submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=1, width=6, height=1)
    node = _gemm_node("node", 8, 8, 8)
    layout = node.payload.default_output_layouts(
        planning_submesh,
        logical_shape=(3, 2),
    )[0]
    plan = StagePlan(
        stage_id=0,
        tile_count=6,
        logical_shape=(3, 2),
        input_layouts=(),
        output_layouts=(layout,),
    )

    placed_layout = _layout_on_submesh(plan.output_layouts[0], placed_submesh)

    assert placed_layout.submesh == placed_submesh
    assert placed_layout.logical_width == 3
    assert placed_layout.logical_height == 2


def test_place_stage_plans_reattaches_layouts_to_mapping() -> None:
    mesh = Mesh(6, 2, l2_memory=L2Memory(size=4096))
    planning_submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=6, height=1)
    placed_submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=1, width=6, height=1)
    node = _gemm_node("node", 8, 8, 8)
    input_layouts = node.payload.default_input_layouts(
        planning_submesh,
        logical_shape=(3, 2),
    )
    output_layouts = node.payload.default_output_layouts(
        planning_submesh,
        logical_shape=(3, 2),
    )
    plan = StagePlan(
        stage_id=0,
        tile_count=6,
        logical_shape=(3, 2),
        input_layouts=input_layouts,
        output_layouts=output_layouts,
    )

    placed_plans = place_stage_plans({0: plan}, {0: placed_submesh})

    assert placed_plans[0].output_layouts[0].submesh == placed_submesh
    assert placed_plans[0].output_layouts[0].logical_width == 3
    assert placed_plans[0].output_layouts[0].logical_height == 2


def test_map_spatially_returns_non_overlapping_submeshes_on_4x4_mesh() -> None:
    try:
        import pulp  # noqa: F401
    except ImportError:
        return

    producer = _gemm_node("producer", 8, 8, 8)
    consumer_input = producer.outputs[0]
    consumer_w = Tensor(name="consumer_w", rank=2, dims=(8, 8), elem_bytes=2)
    consumer_out = Tensor(name="consumer_out", rank=2, dims=(8, 8), elem_bytes=2)
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
        nodes=(producer, consumer),
        edges=(Edge(tensor=consumer_input, src=producer, dst=consumer),),
    )
    mesh = Mesh(4, 4, l2_memory=L2Memory(size=4096))

    mapping = map_spatially(
        graph,
        mesh,
        tile_counts={0: 4, 1: 4},
        print_mapping=False,
    )

    assert set(mapping) == {0, 1}
    assert mapping[0].num_tiles == 4
    assert mapping[1].num_tiles == 4
    tiles0 = {tile.tile_id for tile in mapping[0].tiles}
    tiles1 = {tile.tile_id for tile in mapping[1].tiles}
    assert not (tiles0 & tiles1)


def test_map_spatially_prunes_placement_candidates() -> None:
    try:
        import pulp  # noqa: F401
    except ImportError:
        return

    producer = _gemm_node("producer", 8, 8, 8)
    consumer_input = producer.outputs[0]
    consumer_w = Tensor(name="consumer_w", rank=2, dims=(8, 8), elem_bytes=2)
    consumer_out = Tensor(name="consumer_out", rank=2, dims=(8, 8), elem_bytes=2)
    consumer = Node(
        name="consumer",
        kind=OpKind.GEMM,
        inputs=(consumer_input, consumer_w),
        outputs=(consumer_out,),
        payload=GemmLayerOp(x=consumer_input, w=consumer_w, y=None, output=consumer_out),
    )
    graph = Graph(
        name="g",
        nodes=(producer, consumer),
        edges=(Edge(tensor=consumer_input, src=producer, dst=consumer),),
    )
    mesh = Mesh(4, 4, l2_memory=L2Memory(size=4096))
    tile_counts = {0: 4, 1: 4}

    mapping = map_spatially(
        graph,
        mesh,
        tile_counts=tile_counts,
        max_placements_per_stage=4,
        print_mapping=False,
    )

    _assert_valid_mapping(mapping, tile_counts)


def test_map_spatially_can_require_l2_access_point_for_input_stage() -> None:
    try:
        import pulp  # noqa: F401
    except ImportError:
        return

    node = _gemm_node("node", 8, 8, 8)
    graph = Graph(
        name="g",
        tensors=(*node.inputs, *node.outputs),
        nodes=(node,),
        edges=(
            Edge(tensor=node.inputs[0], src=None, dst=node),
            Edge(tensor=node.inputs[1], src=None, dst=node),
            Edge(tensor=node.outputs[0], src=node, dst=None),
        ),
        inputs=node.inputs,
        outputs=node.outputs,
        initializers=(node.inputs[1],),
    )
    mesh = Mesh(2, 1, l2_memory=L2Memory(size=4096, access_points=((1, 0),)))

    mapping = map_spatially(
        graph,
        mesh,
        tile_counts={0: 1},
        require_l2_input_access_point=True,
        print_mapping=False,
    )

    assert mapping[0].contains_tile_id(mesh.tile_id(1, 0))


def test_map_spatially_can_require_l2_access_point_for_output_stage() -> None:
    try:
        import pulp  # noqa: F401
    except ImportError:
        return

    node = _gemm_node("node", 8, 8, 8)
    graph = Graph(
        name="g",
        tensors=(*node.inputs, *node.outputs),
        nodes=(node,),
        edges=(
            Edge(tensor=node.inputs[0], src=None, dst=node),
            Edge(tensor=node.inputs[1], src=None, dst=node),
            Edge(tensor=node.outputs[0], src=node, dst=None),
        ),
        inputs=node.inputs,
        outputs=node.outputs,
        initializers=node.inputs,
    )
    mesh = Mesh(2, 1, l2_memory=L2Memory(size=4096, access_points=((1, 0),)))

    mapping = map_spatially(
        graph,
        mesh,
        tile_counts={0: 1},
        require_l2_output_access_point=True,
        print_mapping=False,
    )

    assert mapping[0].contains_tile_id(mesh.tile_id(1, 0))


def test_map_spatially_solves_four_node_graph_on_6x6_mesh(capsys) -> None:
    try:
        import pulp  # noqa: F401
    except ImportError:
        return

    node0 = _gemm_node("node0", 8, 8, 8)
    node1_input = node0.outputs[0]
    node1_w = Tensor(name="node1_w", rank=2, dims=(8, 8), elem_bytes=2)
    node1_out = Tensor(name="node1_out", rank=2, dims=(8, 8), elem_bytes=2)
    node1 = Node(
        name="node1",
        kind=OpKind.GEMM,
        inputs=(node1_input, node1_w),
        outputs=(node1_out,),
        payload=GemmLayerOp(x=node1_input, w=node1_w, y=None, output=node1_out),
    )

    node2_input = node1.outputs[0]
    node2_w = Tensor(name="node2_w", rank=2, dims=(8, 8), elem_bytes=2)
    node2_out = Tensor(name="node2_out", rank=2, dims=(8, 8), elem_bytes=2)
    node2 = Node(
        name="node2",
        kind=OpKind.GEMM,
        inputs=(node2_input, node2_w),
        outputs=(node2_out,),
        payload=GemmLayerOp(x=node2_input, w=node2_w, y=None, output=node2_out),
    )

    node3_input = node2.outputs[0]
    node3_w = Tensor(name="node3_w", rank=2, dims=(8, 8), elem_bytes=2)
    node3_out = Tensor(name="node3_out", rank=2, dims=(8, 8), elem_bytes=2)
    node3 = Node(
        name="node3",
        kind=OpKind.GEMM,
        inputs=(node3_input, node3_w),
        outputs=(node3_out,),
        payload=GemmLayerOp(x=node3_input, w=node3_w, y=None, output=node3_out),
    )

    graph = Graph(
        name="g",
        nodes=(node0, node1, node2, node3),
        edges=(
            Edge(tensor=node1_input, src=node0, dst=node1),
            Edge(tensor=node2_input, src=node1, dst=node2),
            Edge(tensor=node3_input, src=node2, dst=node3),
        ),
    )
    mesh = Mesh(6, 6, l2_memory=L2Memory(size=4096))
    tile_counts = {0: 3, 1: 4, 2: 5, 3: 6}

    mapping = map_spatially(
        graph,
        mesh,
        tile_counts=tile_counts,
        objective="max",
        print_mapping=False,
    )
    with capsys.disabled():
        print_spatial_mapping_details(graph, mesh, mapping, label="max")

    _assert_valid_mapping(mapping, tile_counts)

    sum_mapping = map_spatially(
        graph,
        mesh,
        tile_counts=tile_counts,
        objective="sum",
        print_mapping=False,
    )
    with capsys.disabled():
        print_spatial_mapping_details(graph, mesh, sum_mapping, label="sum")

    _assert_valid_mapping(sum_mapping, tile_counts)


def test_edge_shape_costs_include_l2_data_movement() -> None:
    producer = _gemm_node("producer", 8, 8, 8)
    consumer_input = producer.outputs[0]
    consumer_w = Tensor(name="consumer_w", rank=2, dims=(8, 8), elem_bytes=2)
    consumer_out = Tensor(name="consumer_out", rank=2, dims=(8, 8), elem_bytes=2)
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
        nodes=(producer, consumer),
        edges=(Edge(tensor=consumer_input, src=producer, dst=consumer),),
    )

    costs = _edge_shape_costs(
        graph,
        shape_options={0: ((2, 1),), 1: ((2, 1),)},
    )

    edge_cost = costs[(0, 0, 1, 0, 0)]
    assert edge_cost["l2"] > 0
    assert edge_cost["l2"] >= edge_cost["l1"]


def test_stage_io_costs_skip_initializer_reads() -> None:
    x = Tensor(name="x", rank=2, dims=(8, 8), elem_bytes=2)
    w = Tensor(name="w", rank=2, dims=(8, 16), elem_bytes=2)
    out = Tensor(name="out", rank=2, dims=(8, 16), elem_bytes=2)
    node = Node(
        name="matmul_0",
        kind=OpKind.GEMM,
        inputs=(x, w),
        outputs=(out,),
        payload=GemmLayerOp(x=x, w=w, y=None, output=out),
    )
    graph = Graph(
        name="g",
        tensors=(x, w, out),
        nodes=(node,),
        edges=(
            Edge(tensor=x, src=None, dst=node),
            Edge(tensor=w, src=None, dst=node),
            Edge(tensor=out, src=node, dst=None),
        ),
        inputs=(x, w),
        outputs=(out,),
        initializers=(w,),
    )
    mesh = Mesh(2, 1, l2_memory=L2Memory(size=4096))
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=1)

    costs = _stage_io_costs_for_placements(
        graph,
        placement_options={0: (submesh,)},
    )
    costs_with_repeated_weight_read = _stage_io_costs_for_placements(
        Graph(
            name="g",
            tensors=(x, w, out),
            nodes=(node,),
            edges=graph.edges,
            inputs=(x, w),
            outputs=(out,),
            initializers=(),
        ),
        placement_options={0: (submesh,)},
    )

    assert costs[0][0]["read"] < costs_with_repeated_weight_read[0][0]["read"]
    assert costs[0][0]["write"] == costs_with_repeated_weight_read[0][0]["write"]
