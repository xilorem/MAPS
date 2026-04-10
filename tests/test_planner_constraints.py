from MAPS.arch import Mesh, Tile
from MAPS.core.graph import Node, OpKind
from MAPS.core.layout import LayoutAxis, LayoutAxisMode, TensorLayout
from MAPS.core.pipeline import Pipeline
from MAPS.core.stage import (
    InputSource,
    InputSourceKind,
    Stage,
    StageInputBinding,
    StageOutputBinding,
    StageOutputRef,
)
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.ops.gemm import GemmLayerOp
from MAPS.planner import PlannerConstraints, validate_constraints


def _make_layout(submesh: Submesh) -> TensorLayout:
    return TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
        microbatch_axis=None,
        num_microbatches=1,
    )


def _make_mesh(width: int, height: int, l1_bytes: int, l2_bytes: int) -> Mesh:
    return Mesh(
        width=width,
        height=height,
        l2_bytes=l2_bytes,
        tiles=tuple(
            Tile(tile_id=(y * width + x), x=x, y=y, l1_bytes=l1_bytes)
            for y in range(height)
            for x in range(width)
        ),
    )


def test_validate_constraints_accepts_consistent_single_stage_pipeline() -> None:
    mesh = _make_mesh(2, 2, l1_bytes=4096, l2_bytes=4096)
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
        nodes=(node,),
        inputs=(
            StageInputBinding(
                tensor_id=0,
                source=InputSource(
                    kind=InputSourceKind.EXTERNAL,
                    external_base_addr=1,
                ),
            ),
            StageInputBinding(
                tensor_id=1,
                source=InputSource(
                    kind=InputSourceKind.EXTERNAL,
                    external_base_addr=2,
                ),
            ),
        ),
        outputs=(StageOutputBinding(tensor_id=2, layout=layout),),
    )
    pipeline = Pipeline(
        name="p0",
        mesh=mesh,
        num_microbatches=1,
        tensors=tensors,
        stages=(stage,),
        transitions=(),
    )

    report = validate_constraints(pipeline, PlannerConstraints())

    assert report.is_valid
    assert report.violations == ()


def test_validate_constraints_rejects_microbatch_range_violation() -> None:
    mesh = _make_mesh(1, 1, l1_bytes=64, l2_bytes=4096)
    pipeline = Pipeline(name="p0", mesh=mesh, num_microbatches=2)

    report = validate_constraints(
        pipeline,
        PlannerConstraints(min_num_microbatches=1, max_num_microbatches=1),
    )

    assert not report.is_valid
    assert any(
        violation.kind == "num_microbatches_above_max"
        for violation in report.violations
    )


def test_validate_constraints_counts_external_inputs_against_l2() -> None:
    mesh = _make_mesh(1, 1, l1_bytes=4096, l2_bytes=32)
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
        nodes=(node,),
        inputs=(
            StageInputBinding(
                tensor_id=0,
                source=InputSource(
                    kind=InputSourceKind.EXTERNAL,
                    external_base_addr=1,
                ),
            ),
        ),
        outputs=(StageOutputBinding(tensor_id=1, layout=layout),),
    )
    pipeline = Pipeline(
        name="p0",
        mesh=mesh,
        num_microbatches=1,
        tensors=tensors,
        stages=(stage,),
        transitions=(),
    )

    report = validate_constraints(pipeline, PlannerConstraints())

    assert not report.is_valid
    assert any(
        violation.kind == "mesh_l2_capacity_exceeded"
        for violation in report.violations
    )


def test_validate_constraints_does_not_count_local_inputs_against_l2() -> None:
    mesh = _make_mesh(1, 1, l1_bytes=64, l2_bytes=1)
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
    stage0 = Stage(
        name="stage0",
        submesh=submesh,
        nodes=(producer,),
        inputs=(),
        outputs=(StageOutputBinding(tensor_id=0, layout=layout),),
    )
    stage1 = Stage(
        name="stage1",
        submesh=submesh,
        nodes=(consumer,),
        inputs=(
            StageInputBinding(
                tensor_id=0,
                source=InputSource(
                    kind=InputSourceKind.LOCAL,
                    local_output=StageOutputRef(stage_id=0, output_idx=0),
                ),
            ),
        ),
        outputs=(StageOutputBinding(tensor_id=1, layout=layout),),
    )
    pipeline = Pipeline(
        name="p0",
        mesh=mesh,
        num_microbatches=1,
        tensors=tensors,
        stages=(stage0, stage1),
        transitions=(),
    )

    report = validate_constraints(pipeline, PlannerConstraints())

    assert report.is_valid
