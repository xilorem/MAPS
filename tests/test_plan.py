from pathlib import Path
from tempfile import TemporaryDirectory

from MAPS.arch import L1Memory, L2Memory, Mesh
from MAPS.chips import magia_mesh
from MAPS.core.graph import Edge, Graph, Node, OpKind
from MAPS.core.layer import ExternalInput, LocalInput, TransitionInput
from MAPS.core.ownership import tile_tensor_slice
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.core.transition import TransitionMode
from MAPS.ops.defs.gemm import GemmPayload
from MAPS.planner import PlannerConstraints, validate_constraints
import MAPS.planner.plan as plan_module
from MAPS.planner.plan import _build_pipeline_from_graph, build_pipeline
from MAPS.planner.workload_balancing import StagePlan
from tests.noc_utils import rectangular_test_noc, rectangular_test_tiles


def _mesh_with_l1(width: int, height: int, l1_size: int) -> Mesh:
    return Mesh(
        width=width,
        height=height,
        l2_memory=L2Memory(size=4096, bandwidth=1),
        noc=rectangular_test_noc(width, height),
        tiles=rectangular_test_tiles(width, height, memory=L1Memory(size=l1_size, bandwidth=1)),
    )


def test_build_pipeline_from_graph_assembles_stages_transitions_and_bindings() -> None:
    mesh = magia_mesh()
    src_submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=1)
    dst_submesh = Submesh(mesh=mesh, submesh_id=1, x0=0, y0=1, width=2, height=1)

    x = Tensor(name="x", rank=2, dims=(4, 4), elem_bytes=2)
    w0 = Tensor(name="w0", rank=2, dims=(4, 8), elem_bytes=2)
    y = Tensor(name="y", rank=2, dims=(4, 8), elem_bytes=2)
    w1 = Tensor(name="w1", rank=2, dims=(8, 6), elem_bytes=2)
    z = Tensor(name="z", rank=2, dims=(4, 6), elem_bytes=2)

    gemm0 = GemmPayload(x=x, w=w0, y=None, output=y)
    gemm1 = GemmPayload(x=y, w=w1, y=None, output=z)
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
        input_layouts=gemm0.input_layouts(src_submesh, logical_shape=(2, 1)),
        output_layouts=gemm0.output_layouts(src_submesh, logical_shape=(2, 1)),
    )
    plan1 = StagePlan(
        stage_id=1,
        tile_count=2,
        logical_shape=(2, 1),
        input_layouts=gemm1.input_layouts(dst_submesh, logical_shape=(2, 1)),
        output_layouts=gemm1.output_layouts(dst_submesh, logical_shape=(2, 1)),
    )

    pipeline = _build_pipeline_from_graph(graph, mesh, {0: plan0, 1: plan1})

    assert pipeline.name == "direct_two_gemms"
    assert len(pipeline.stages) == 2
    assert len(pipeline.transitions) == 1
    assert pipeline.stages[0].layers[0].node == node0
    assert pipeline.stages[1].layers[0].node == node1
    assert isinstance(pipeline.stages[0].layers[0].inputs[0].source, ExternalInput)
    assert isinstance(pipeline.stages[1].layers[0].inputs[0].source, TransitionInput)
    assert pipeline.stages[1].layers[0].inputs[0].source.transition_id == 0

    transition = pipeline.transitions[0]
    assert transition.mode is TransitionMode.DIRECT_REMAP
    assert transition.tensor_id == 2
    assert transition.src_layer_id == 0
    assert transition.dst_layer_id == 1
    assert transition.src_layout == pipeline.stages[0].layers[-1].outputs[0].layout
    assert transition.dst_layout == plan1.input_layouts[0]
    assert len(transition.fragments) == 2
    assert {
        (fragment.src_hartid, fragment.dst_hartid)
        for fragment in transition.fragments
    } == {(0, 8), (1, 9)}
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


def test_build_pipeline_from_graph_builds_local_inputs_for_grouped_stage_nodes() -> None:
    mesh = magia_mesh()
    stage0_submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=1)
    stage1_submesh = Submesh(mesh=mesh, submesh_id=1, x0=0, y0=1, width=2, height=1)

    x = Tensor(name="x", rank=2, dims=(4, 4), elem_bytes=2)
    w0 = Tensor(name="w0", rank=2, dims=(4, 8), elem_bytes=2)
    y0 = Tensor(name="y0", rank=2, dims=(4, 8), elem_bytes=2)
    w1 = Tensor(name="w1", rank=2, dims=(8, 6), elem_bytes=2)
    y1 = Tensor(name="y1", rank=2, dims=(4, 6), elem_bytes=2)
    w2 = Tensor(name="w2", rank=2, dims=(6, 5), elem_bytes=2)
    z = Tensor(name="z", rank=2, dims=(4, 5), elem_bytes=2)

    gemm0 = GemmPayload(x=x, w=w0, y=None, output=y0)
    gemm1 = GemmPayload(x=y0, w=w1, y=None, output=y1)
    gemm2 = GemmPayload(x=y1, w=w2, y=None, output=z)
    node0 = Node(
        name="gemm_0",
        kind=OpKind.GEMM,
        inputs=(x, w0),
        outputs=(y0,),
        payload=gemm0,
    )
    node1 = Node(
        name="gemm_1",
        kind=OpKind.GEMM,
        inputs=(y0, w1),
        outputs=(y1,),
        payload=gemm1,
    )
    node2 = Node(
        name="gemm_2",
        kind=OpKind.GEMM,
        inputs=(y1, w2),
        outputs=(z,),
        payload=gemm2,
    )
    graph = Graph(
        name="grouped_two_gemms",
        tensors=(x, w0, y0, w1, y1, w2, z),
        nodes=(node0, node1, node2),
        edges=(
            Edge(tensor=x, src=None, dst=node0),
            Edge(tensor=w0, src=None, dst=node0),
            Edge(tensor=y0, src=node0, dst=node1),
            Edge(tensor=w1, src=None, dst=node1),
            Edge(tensor=y1, src=node1, dst=node2),
            Edge(tensor=w2, src=None, dst=node2),
            Edge(tensor=z, src=node2, dst=None),
        ),
        inputs=(x,),
        outputs=(z,),
        initializers=(w0, w1, w2),
    )
    plan0 = StagePlan(
        stage_id=0,
        tile_count=2,
        logical_shape=(2, 1),
        input_layouts=gemm0.input_layouts(stage0_submesh, logical_shape=(2, 1)),
        output_layouts=gemm1.output_layouts(stage0_submesh, logical_shape=(2, 1)),
        nodes=(node0, node1),
        node_input_layouts=(
            gemm0.input_layouts(stage0_submesh, logical_shape=(2, 1)),
            gemm1.input_layouts(stage0_submesh, logical_shape=(2, 1)),
        ),
        node_output_layouts=(
            gemm0.output_layouts(stage0_submesh, logical_shape=(2, 1)),
            gemm1.output_layouts(stage0_submesh, logical_shape=(2, 1)),
        ),
    )
    plan1 = StagePlan(
        stage_id=1,
        tile_count=2,
        logical_shape=(2, 1),
        input_layouts=gemm2.input_layouts(stage1_submesh, logical_shape=(2, 1)),
        output_layouts=gemm2.output_layouts(stage1_submesh, logical_shape=(2, 1)),
        nodes=(node2,),
        node_input_layouts=(gemm2.input_layouts(stage1_submesh, logical_shape=(2, 1)),),
        node_output_layouts=(gemm2.output_layouts(stage1_submesh, logical_shape=(2, 1)),),
    )

    pipeline = _build_pipeline_from_graph(graph, mesh, {0: plan0, 1: plan1})

    assert len(pipeline.stages) == 2
    assert len(pipeline.stages[0].layers) == 2
    assert pipeline.stages[0].layers[0].node == node0
    assert pipeline.stages[0].layers[1].node == node1
    assert isinstance(pipeline.stages[0].layers[1].inputs[0].source, LocalInput)
    assert pipeline.stages[0].layers[1].inputs[0].source.layer_idx == 0
    assert len(pipeline.transitions) == 1
    assert pipeline.transitions[0].src_layer_id == 0
    assert pipeline.transitions[0].dst_layer_id == 1
    assert isinstance(pipeline.stages[1].layers[0].inputs[0].source, TransitionInput)
    assert pipeline.stages[1].layers[0].inputs[0].source.transition_id == 0

    report = validate_constraints(pipeline, PlannerConstraints(max_stage_nodes=2))
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
    if pipeline.transitions:
        assert len(pipeline.transitions) == 1
        transition = pipeline.transitions[0]
        assert transition.src_layer_id == 0
        assert transition.dst_layer_id == 1
        assert transition.src_layout == pipeline.stages[0].layers[-1].outputs[0].layout
        assert transition.dst_layout == pipeline.stages[1].layers[0].node.payload.input_layouts(
            pipeline.stages[1].submesh,
            logical_shape=(
                transition.dst_layout.effective_logical_width,
                transition.dst_layout.effective_logical_height,
            ),
        )[0]
        assert transition.mode is TransitionMode.DIRECT_REMAP
        assert transition.fragments
        for fragment in transition.fragments:
            assert fragment.src_slice.rank == pipeline.tensors[transition.tensor_id].rank
            assert fragment.dst_slice.rank == pipeline.tensors[transition.tensor_id].rank
            for src_dim, dst_dim in zip(fragment.src_slice.dims, fragment.dst_slice.dims):
                assert src_dim.length > 0
                assert dst_dim.length > 0
        assert isinstance(pipeline.stages[1].layers[0].inputs[0].source, TransitionInput)
        assert pipeline.stages[1].layers[0].inputs[0].source.transition_id == 0
    assert isinstance(pipeline.stages[0].layers[0].inputs[0].source, ExternalInput)

    report = validate_constraints(pipeline, PlannerConstraints())
    assert report.is_valid, report.violations


def test_build_pipeline_lowers_softmax_into_one_grouped_stage() -> None:
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        return

    with TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "softmax.onnx"
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 8])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 8])
        node = helper.make_node("Softmax", inputs=["x"], outputs=["y"], name="softmax_0", axis=-1)
        graph = helper.make_graph([node], "tiny_softmax", [x], [y])
        model = helper.make_model(graph)
        onnx.save(model, model_path)

        pipeline = build_pipeline(model_path, _mesh_with_l1(2, 2, l1_size=4096))

    assert pipeline.name == "tiny_softmax"
    assert len(pipeline.stages) == 1
    assert len(pipeline.transitions) == 0
    assert tuple(layer.node.name for layer in pipeline.stages[0].layers) == (
        "softmax_0__reduce_max",
        "softmax_0__allreduce_max",
        "softmax_0__sub",
        "softmax_0__exp",
        "softmax_0__reduce_sum",
        "softmax_0__allreduce_sum",
        "softmax_0__div",
    )
    assert isinstance(pipeline.stages[0].layers[1].inputs[0].source, LocalInput)
    assert pipeline.stages[0].layers[1].inputs[0].source.layer_idx == 0
    assert isinstance(pipeline.stages[0].layers[2].inputs[1].source, LocalInput)
    assert pipeline.stages[0].layers[2].inputs[1].source.layer_idx == 1
    assert isinstance(pipeline.stages[0].layers[6].inputs[1].source, LocalInput)
    assert pipeline.stages[0].layers[6].inputs[1].source.layer_idx == 5

    report = validate_constraints(pipeline, PlannerConstraints(max_stage_nodes=7))
    assert report.is_valid, report.violations


def test_build_pipeline_disables_spatial_mapping_pruning_by_default(monkeypatch) -> None:
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        return

    seen = {}

    def fake_map_spatially(graph, mesh, stage_plans, **kwargs):
        seen["enable_lossless_pruning"] = kwargs["enable_lossless_pruning"]
        seen["max_placements_per_stage"] = kwargs["max_placements_per_stage"]
        seen["show_progress"] = kwargs["show_progress"]
        return {
            0: Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=1),
            1: Submesh(mesh=mesh, submesh_id=1, x0=0, y0=1, width=2, height=1),
        }

    monkeypatch.setattr(plan_module, "map_spatially", fake_map_spatially)

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

        build_pipeline(model_path, _mesh_with_l1(2, 2, l1_size=4096))

    assert seen["enable_lossless_pruning"] is False
    assert seen["max_placements_per_stage"] is None
    assert seen["show_progress"] is False


def test_build_pipeline_can_enable_spatial_mapping_pruning(monkeypatch) -> None:
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        return

    seen = {}

    def fake_map_spatially(graph, mesh, stage_plans, **kwargs):
        seen["enable_lossless_pruning"] = kwargs["enable_lossless_pruning"]
        seen["max_placements_per_stage"] = kwargs["max_placements_per_stage"]
        seen["show_progress"] = kwargs["show_progress"]
        return {
            0: Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=1),
            1: Submesh(mesh=mesh, submesh_id=1, x0=0, y0=1, width=2, height=1),
        }

    monkeypatch.setattr(plan_module, "map_spatially", fake_map_spatially)

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

        build_pipeline(
            model_path,
            _mesh_with_l1(2, 2, l1_size=4096),
            print_spatial_mapping_progress=True,
            enable_lossless_spatial_mapping_pruning=True,
            enable_lossy_spatial_mapping_pruning=True,
        )

    assert seen["enable_lossless_pruning"] is True
    assert seen["max_placements_per_stage"] == 16
    assert seen["show_progress"] is True
