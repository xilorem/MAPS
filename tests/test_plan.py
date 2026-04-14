from pathlib import Path
from tempfile import TemporaryDirectory

from MAPS.arch import L1Memory, L2Memory, Mesh, Tile
from MAPS.core.graph import Edge, Graph, Node, OpKind
from MAPS.core.ownership import tile_tensor_slice
from MAPS.core.stage import InputSourceKind
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.core.transition import TransitionMode
from MAPS.ops.gemm import GemmLayerOp
from MAPS.planner import PlannerConstraints, validate_constraints
from MAPS.planner.plan import _build_pipeline_from_graph, build_pipeline
from MAPS.planner.workload_balancing import StagePlan


def _mesh_with_l1(width: int, height: int, l1_size: int) -> Mesh:
    return Mesh(
        width,
        height,
        l2_memory=L2Memory(size=4096),
        tiles=tuple(
            Tile(tile_id=(y * width + x), x=x, y=y, memory=L1Memory(size=l1_size))
            for y in range(height)
            for x in range(width)
        ),
    )


def test_build_pipeline_from_graph_assembles_stages_transitions_and_bindings() -> None:
    mesh = _mesh_with_l1(2, 2, l1_size=4096)
    src_submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=1)
    dst_submesh = Submesh(mesh=mesh, submesh_id=1, x0=0, y0=1, width=2, height=1)

    x = Tensor(name="x", rank=2, dims=(4, 4), elem_bytes=2)
    w0 = Tensor(name="w0", rank=2, dims=(4, 8), elem_bytes=2)
    y = Tensor(name="y", rank=2, dims=(4, 8), elem_bytes=2)
    w1 = Tensor(name="w1", rank=2, dims=(8, 6), elem_bytes=2)
    z = Tensor(name="z", rank=2, dims=(4, 6), elem_bytes=2)

    gemm0 = GemmLayerOp(x=x, w=w0, y=None, output=y)
    gemm1 = GemmLayerOp(x=y, w=w1, y=None, output=z)
    node0 = Node(
        name="gemm_0",
        kind=OpKind.GEMM,
        inputs=(x, w0),
        outputs=(y,),
        payload=gemm0,
    )
    node1 = Node(
        name="gemm_1",
        kind=OpKind.GEMM,
        inputs=(y, w1),
        outputs=(z,),
        payload=gemm1,
    )
    graph = Graph(
        name="direct_two_gemms",
        tensors=(x, w0, y, w1, z),
        nodes=(node0, node1),
        edges=(
            Edge(tensor=x, src=None, dst=node0),
            Edge(tensor=w0, src=None, dst=node0),
            Edge(tensor=y, src=node0, dst=node1),
            Edge(tensor=w1, src=None, dst=node1),
            Edge(tensor=z, src=node1, dst=None),
        ),
        inputs=(x,),
        outputs=(z,),
        initializers=(w0, w1),
    )
    plan0 = StagePlan(
        stage_id=0,
        tile_count=2,
        logical_shape=(2, 1),
        input_layouts=gemm0.default_input_layouts(src_submesh, logical_shape=(2, 1)),
        output_layouts=gemm0.default_output_layouts(src_submesh, logical_shape=(2, 1)),
    )
    plan1 = StagePlan(
        stage_id=1,
        tile_count=2,
        logical_shape=(2, 1),
        input_layouts=gemm1.default_input_layouts(dst_submesh, logical_shape=(2, 1)),
        output_layouts=gemm1.default_output_layouts(dst_submesh, logical_shape=(2, 1)),
    )

    pipeline = _build_pipeline_from_graph(graph, mesh, {0: plan0, 1: plan1})

    assert pipeline.name == "direct_two_gemms"
    assert len(pipeline.stages) == 2
    assert len(pipeline.transitions) == 1
    assert pipeline.stages[0].nodes == (node0,)
    assert pipeline.stages[1].nodes == (node1,)
    assert pipeline.stages[0].inputs[0].source.kind is InputSourceKind.EXTERNAL
    assert pipeline.stages[1].inputs[0].source.kind is InputSourceKind.TRANSITION
    assert pipeline.stages[1].inputs[0].source.transition_id == 0

    transition = pipeline.transitions[0]
    assert transition.mode is TransitionMode.DIRECT_REMAP
    assert transition.tensor_id == 2
    assert transition.src_layer_id == 0
    assert transition.dst_layer_id == 1
    assert transition.src_layout == pipeline.stages[0].outputs[0].layout
    assert transition.dst_layout == plan1.input_layouts[0]
    assert len(transition.fragments) == 2
    assert {
        (fragment.src_hartid, fragment.dst_hartid)
        for fragment in transition.fragments
    } == {(0, 2), (1, 3)}
    assert {
        fragment.src_slice
        for fragment in transition.fragments
    } == {
        tile_tensor_slice(y, transition.src_layout, tile)
        for tile in transition.src_layout.submesh.tiles
    }
    assert {
        fragment.dst_slice
        for fragment in transition.fragments
    } == {
        tile_tensor_slice(y, transition.dst_layout, tile)
        for tile in transition.dst_layout.submesh.tiles
    }

    report = validate_constraints(pipeline, PlannerConstraints())
    assert report.is_valid, report.violations


def test_build_pipeline_parses_balances_maps_and_builds_transitions() -> None:
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        return

    with TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "two_matmuls.onnx"
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
        w0 = helper.make_tensor_value_info("w0", TensorProto.FLOAT, [3, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])
        w1 = helper.make_tensor_value_info("w1", TensorProto.FLOAT, [4, 5])
        z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [2, 5])
        node0 = helper.make_node("MatMul", inputs=["x", "w0"], outputs=["y"], name="matmul_0")
        node1 = helper.make_node("MatMul", inputs=["y", "w1"], outputs=["z"], name="matmul_1")
        graph = helper.make_graph(
            [node0, node1],
            "two_matmuls",
            [x, w0, w1],
            [z],
            value_info=[y],
        )
        model = helper.make_model(graph)
        onnx.save(model, model_path)

        pipeline = build_pipeline(model_path, _mesh_with_l1(2, 2, l1_size=4096))

    assert pipeline.name == "two_matmuls"
    assert len(pipeline.stages) == 2
    assert len(pipeline.transitions) == 1
    transition = pipeline.transitions[0]
    assert transition.src_layer_id == 0
    assert transition.dst_layer_id == 1
    assert transition.src_layout == pipeline.stages[0].outputs[0].layout
    assert transition.dst_layout == pipeline.stages[1].nodes[0].payload.default_input_layouts(
        pipeline.stages[1].submesh,
        logical_shape=(
            transition.dst_layout.effective_logical_width,
            transition.dst_layout.effective_logical_height,
        ),
    )[0]
    if transition.mode is TransitionMode.DIRECT_REMAP:
        assert transition.fragments
        for fragment in transition.fragments:
            assert fragment.src_slice.rank == pipeline.tensors[transition.tensor_id].rank
            assert fragment.dst_slice.rank == pipeline.tensors[transition.tensor_id].rank
            for src_dim, dst_dim in zip(fragment.src_slice.dims, fragment.dst_slice.dims):
                assert src_dim.length > 0
                assert dst_dim.length > 0
    else:
        assert transition.mode is TransitionMode.LOCAL_REUSE
        assert transition.fragments == ()
    assert pipeline.stages[1].inputs[0].source.kind is InputSourceKind.TRANSITION
    assert pipeline.stages[1].inputs[0].source.transition_id == 0
    assert pipeline.stages[0].inputs[0].source.kind is InputSourceKind.EXTERNAL

    report = validate_constraints(pipeline, PlannerConstraints())
    assert report.is_valid, report.violations
