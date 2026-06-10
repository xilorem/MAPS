from pathlib import Path

from MAPS.arch import L1Memory, L2Memory, Mesh
from MAPS.core.graph import Edge, Graph, Node, OpKind
from MAPS.core.layout import tile_tensor_slice
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.hw.chips import magia_mesh
from MAPS.ops.defs.gemm import GemmPayload
from MAPS.pipeline.layer import ExternalInput, TransitionInput
import MAPS.planner.plan_v2 as plan_v2_module
from MAPS.planner.plan_v2 import _build_pipeline_from_graph, build_pipeline
from MAPS.planner.workload_balancing_v2 import StagePlan
from MAPS.transitions.model import TransitionMode
from tests.noc_utils import rectangular_test_noc, rectangular_test_tiles


def _mesh_with_l1(width: int, height: int, l1_size: int) -> Mesh:
    return Mesh(
        width=width,
        height=height,
        l2_memory=L2Memory(size=4096, bandwidth=1),
        noc=rectangular_test_noc(width, height),
        tiles=rectangular_test_tiles(width, height, memory=L1Memory(size=l1_size, bandwidth=1)),
    )


def _rectangle_tile_ids(mesh: Mesh, x0: int, y0: int, width: int, height: int) -> frozenset[int]:
    return frozenset(
        mesh.tile_id(x, y)
        for y in range(y0, y0 + height)
        for x in range(x0, x0 + width)
    )


def test_build_pipeline_from_graph_v2_assembles_stages_transitions_and_bindings() -> None:
    mesh = magia_mesh()
    src_submesh = Submesh(
        mesh=mesh,
        submesh_id=0,
        tile_ids=_rectangle_tile_ids(mesh, 0, 0, 2, 1),
    )
    dst_submesh = Submesh(
        mesh=mesh,
        submesh_id=1,
        tile_ids=_rectangle_tile_ids(mesh, 0, 1, 1, 2),
    )

    x = Tensor(name="x", rank=2, dims=(4, 4), elem_bytes=2)
    w0 = Tensor(name="w0", rank=2, dims=(4, 8), elem_bytes=2)
    y = Tensor(name="y", rank=2, dims=(4, 8), elem_bytes=2)
    w1 = Tensor(name="w1", rank=2, dims=(8, 6), elem_bytes=2)
    z = Tensor(name="z", rank=2, dims=(4, 6), elem_bytes=2)

    gemm0 = GemmPayload(x=x, w=w0, y=None, output=y)
    gemm1 = GemmPayload(x=y, w=w1, y=None, output=z)
    node0 = Node(name="gemm_0", kind=OpKind.GEMM, inputs=(x, w0), outputs=(y,), payload=gemm0)
    node1 = Node(name="gemm_1", kind=OpKind.GEMM, inputs=(y, w1), outputs=(z,), payload=gemm1)
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
        output_layouts=gemm0.output_layouts(src_submesh, logical_shape=(2, 1)),
    )
    plan1 = StagePlan(
        stage_id=1,
        tile_count=2,
        logical_shape=(2, 1),
        output_layouts=gemm1.output_layouts(dst_submesh, logical_shape=(2, 1)),
    )

    pipeline = _build_pipeline_from_graph(graph, mesh, {0: plan0, 1: plan1})

    assert len(pipeline.stages) == 2
    assert len(pipeline.transitions) == 1
    assert tuple(initialization.name for initialization in pipeline.initializations) == (
        "init_x",
        "init_w0",
        "init_w1",
    )
    assert tuple(finalization.name for finalization in pipeline.finalizations) == ("output_z",)
    assert isinstance(pipeline.stages[0].layers[0].inputs[0].source, ExternalInput)
    assert isinstance(pipeline.stages[1].layers[0].inputs[0].source, TransitionInput)
    assert pipeline.stages[1].layers[0].inputs[0].source.transition_id == 0

    transition = pipeline.transitions[0]
    assert transition.mode is TransitionMode.DIRECT_REMAP
    assert transition.src_layout == pipeline.stages[0].layers[-1].outputs[0].layout
    assert transition.dst_layout == plan1.output_layouts[0]
    assert {
        fragment.src_slice
        for fragment in pipeline.finalizations[0].fragments
    } == {
        tile_tensor_slice(z, plan1.output_layouts[0], tile)
        for tile in plan1.output_layouts[0].submesh.tiles
    }


def test_build_pipeline_v2_disables_spatial_mapping_pruning_by_default(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}
    built_pipeline = object()

    def fake_import(model_path: str | Path) -> Graph:
        captured["model_path"] = Path(model_path)
        return Graph(
            name="g",
            tensors=(),
            nodes=(),
            edges=(),
            inputs=(),
            outputs=(),
            initializers=(),
        )

    def fake_balance(graph: Graph, mesh: Mesh, debug: bool, stage_selection):
        captured["balance"] = (graph.name, mesh.shape, debug, tuple(stage_selection))
        return {}

    def fake_map(graph: Graph, mesh: Mesh, tile_counts, **kwargs):
        captured["map"] = (graph.name, mesh.shape, tile_counts, kwargs)
        return {}

    def fake_build(graph: Graph, mesh: Mesh, stage_plans):
        captured["build"] = (graph.name, mesh.shape, stage_plans)
        return built_pipeline

    monkeypatch.setattr(plan_v2_module, "import_onnx_graph", fake_import)
    monkeypatch.setattr(plan_v2_module, "select_stages", lambda graph: {})
    monkeypatch.setattr(plan_v2_module, "balance_workload", fake_balance)
    monkeypatch.setattr(plan_v2_module, "map_spatially", fake_map)
    monkeypatch.setattr(plan_v2_module, "place_stage_plans", lambda stage_plans, mapping: stage_plans)
    monkeypatch.setattr(plan_v2_module, "_build_pipeline_from_graph", fake_build)
    monkeypatch.setattr(plan_v2_module, "_print_pipeline_stage_cost", lambda graph, mesh, plans: None)

    pipeline = build_pipeline(tmp_path / "model.onnx", _mesh_with_l1(2, 2, l1_size=4096))

    assert pipeline is built_pipeline
    assert captured["model_path"] == tmp_path / "model.onnx"
    assert captured["map"][3]["enable_lossless_pruning"] is False
    assert captured["map"][3]["max_placements_per_stage"] is None
