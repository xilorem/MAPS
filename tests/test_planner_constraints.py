from MAPS.arch import L1Memory, L2Memory, Mesh
from MAPS.chips import magia_mesh
from MAPS.core.graph import Node, OpKind
from MAPS.core.layer import Layer, LayerInput, LayerOutput
from MAPS.core.layout import LayoutAxis, LayoutAxisMode, TensorLayout
from MAPS.core.pipeline import Pipeline
from MAPS.core.stage import Stage
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.ops.gemm import GemmLayerOp
from MAPS.planner import PlannerConstraints, validate_constraints
from tests.noc_utils import rectangular_test_noc, rectangular_test_tiles


def _make_layout(submesh: Submesh) -> TensorLayout:
    return TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
    )


def _make_mesh(width: int, height: int, l1_size: int, l2_memory: L2Memory) -> Mesh:
    return Mesh(
        width=width,
        height=height,
        l2_memory=l2_memory,
        noc=rectangular_test_noc(width, height),
        tiles=rectangular_test_tiles(width, height, memory=L1Memory(size=l1_size, bandwidth=1)),
    )


def test_validate_constraints_accepts_consistent_single_stage_pipeline() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=2)
    layout = _make_layout(submesh)
    tensors = (
        Tensor(name="x", rank=2, dims=(4, 8), elem_bytes=2),
        Tensor(name="w", rank=2, dims=(8, 16), elem_bytes=2),
        Tensor(name="out", rank=2, dims=(4, 16), elem_bytes=2),
    )
    node = Node(
        name="gemm0",
        kind=OpKind.GEMM,
        inputs=(tensors[0], tensors[1]),
        outputs=(tensors[2],),
        payload=GemmLayerOp(x=tensors[0], w=tensors[1], y=None, output=tensors[2]),
    )
    stage = Stage(
        name="stage0",
        submesh=submesh,
        layers=(
            Layer(
                node=node,
                inputs=(
                    LayerInput.external(tensor_id=0, base_addr=1),
                    LayerInput.external(tensor_id=1, base_addr=2),
                ),
                outputs=(LayerOutput(tensor_id=2, layout=layout),),
            ),
        ),
    )
    pipeline = Pipeline(
        name="p0",
        mesh=mesh,
        tensors=tensors,
        stages=(stage,),
        transitions=(),
    )

    report = validate_constraints(pipeline, PlannerConstraints(max_stage_nodes=2))

    assert report.is_valid
    assert report.violations == ()


def test_validate_constraints_counts_external_inputs_against_l2() -> None:
    mesh = _make_mesh(1, 1, l1_size=4096, l2_memory=L2Memory(size=32, bandwidth=1))
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=1, height=1)
    layout = _make_layout(submesh)
    tensors = (
        Tensor(name="x", rank=2, dims=(4, 8), elem_bytes=2),
        Tensor(name="out", rank=2, dims=(4, 8), elem_bytes=2),
    )
    node = Node(
        name="copy0",
        kind=OpKind.CUSTOM,
        inputs=(tensors[0],),
        outputs=(tensors[1],),
    )
    stage = Stage(
        name="stage0",
        submesh=submesh,
        layers=(
            Layer(
                node=node,
                inputs=(
                    LayerInput.external(tensor_id=0, base_addr=1),
                ),
                outputs=(LayerOutput(tensor_id=1, layout=layout),),
            ),
        ),
    )
    pipeline = Pipeline(
        name="p0",
        mesh=mesh,
        tensors=tensors,
        stages=(stage,),
        transitions=(),
    )

    report = validate_constraints(pipeline, PlannerConstraints(max_stage_nodes=2))

    assert not report.is_valid
    assert any(
        violation.kind == "mesh_l2_capacity_exceeded"
        for violation in report.violations
    )


def test_validate_constraints_does_not_count_local_layer_inputs_against_l2() -> None:
    mesh = _make_mesh(1, 1, l1_size=64, l2_memory=L2Memory(size=1, bandwidth=1))
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=1, height=1)
    layout = _make_layout(submesh)
    tensors = (
        Tensor(name="x", rank=2, dims=(1, 1), elem_bytes=1),
        Tensor(name="y", rank=2, dims=(1, 1), elem_bytes=1),
    )
    producer = Node(
        name="producer",
        kind=OpKind.CUSTOM,
        inputs=(),
        outputs=(tensors[0],),
    )
    consumer = Node(
        name="consumer",
        kind=OpKind.CUSTOM,
        inputs=(tensors[0],),
        outputs=(tensors[1],),
    )
    stage = Stage(
        name="stage0",
        submesh=submesh,
        layers=(
            Layer(
                node=producer,
                outputs=(LayerOutput(tensor_id=0, layout=layout),),
            ),
            Layer(
                node=consumer,
                inputs=(
                    LayerInput.local(tensor_id=0, layer_idx=0),
                ),
                outputs=(LayerOutput(tensor_id=1, layout=layout),),
            ),
        ),
    )
    pipeline = Pipeline(
        name="p0",
        mesh=mesh,
        tensors=tensors,
        stages=(stage,),
        transitions=(),
    )

    report = validate_constraints(pipeline, PlannerConstraints(max_stage_nodes=2))

    assert report.is_valid
