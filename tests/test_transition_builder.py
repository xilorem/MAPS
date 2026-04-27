import pytest

from MAPS.transitions import build_transition
from MAPS.core.layout import (
    LayoutAxis,
    LayoutAxisMode,
    TensorLayout,
    TensorRange,
    TensorSlice,
)
from MAPS.hw.chips import magia_mesh
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.transitions.model import TransitionMode


def test_build_transition_rejects_identical_layouts() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=2)
    tensor = Tensor(name="x", rank=2, dims=(8, 8), elem_bytes=2)
    layout = TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=1),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=0),
    )

    with pytest.raises(ValueError, match="identical layouts"):
        build_transition(
            name="reuse",
            tensor=tensor,
            tensor_id=0,
            src_layer_id=0,
            src_output_idx=0,
            dst_layer_id=1,
            dst_input_idx=0,
            src_layout=layout,
            dst_layout=layout,
        )


def test_build_transition_builds_direct_remap_fragments() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=2)
    tensor = Tensor(name="x", rank=2, dims=(8, 8), elem_bytes=2)
    src_layout = TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=1),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=0),
    )
    dst_layout = TensorLayout(
        submesh=submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=0),
    )

    transition = build_transition(
        name="remap",
        tensor=tensor,
        tensor_id=0,
        src_layer_id=0,
        src_output_idx=0,
        dst_layer_id=1,
        dst_input_idx=0,
        src_layout=src_layout,
        dst_layout=dst_layout,
    )

    assert transition.mode is TransitionMode.DIRECT_REMAP
    # Destination mesh_x is replicated, so both tiles in each destination row
    # need the same row slice. That duplicates the top-row and bottom-row
    # transfers across two consumers each.
    assert len(transition.fragments) == 8
    assert {(fragment.src_hartid, fragment.dst_hartid) for fragment in transition.fragments} == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (8, 8),
        (8, 9),
        (9, 8),
        (9, 9),
    }
    assert {fragment.src_slice for fragment in transition.fragments} == {
        TensorSlice(
            rank=2,
            dims=(
                TensorRange(start=0, length=4),
                TensorRange(start=0, length=4),
            ),
        ),
        TensorSlice(
            rank=2,
            dims=(
                TensorRange(start=0, length=4),
                TensorRange(start=4, length=4),
            ),
        ),
        TensorSlice(
            rank=2,
            dims=(
                TensorRange(start=4, length=4),
                TensorRange(start=0, length=4),
            ),
        ),
        TensorSlice(
            rank=2,
            dims=(
                TensorRange(start=4, length=4),
                TensorRange(start=4, length=4),
            ),
        ),
    }


def test_build_transition_builds_direct_remap_between_different_submeshes() -> None:
    mesh = magia_mesh()
    src_submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=2)
    dst_submesh = Submesh(mesh=mesh, submesh_id=1, x0=2, y0=2, width=2, height=2)
    tensor = Tensor(name="x", rank=2, dims=(8, 8), elem_bytes=2)
    src_layout = TensorLayout(
        submesh=src_submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=1),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=0),
    )
    dst_layout = TensorLayout(
        submesh=dst_submesh,
        mesh_x=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=1),
        mesh_y=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=0),
    )

    transition = build_transition(
        name="cross_submesh_remap",
        tensor=tensor,
        tensor_id=0,
        src_layer_id=0,
        src_output_idx=0,
        dst_layer_id=1,
        dst_input_idx=0,
        src_layout=src_layout,
        dst_layout=dst_layout,
    )

    assert transition.mode is TransitionMode.DIRECT_REMAP
    assert len(transition.fragments) == 4
    assert {(fragment.src_hartid, fragment.dst_hartid) for fragment in transition.fragments} == {
        (0, 18),
        (1, 19),
        (8, 26),
        (9, 27),
    }
    assert {fragment.src_slice for fragment in transition.fragments} == {
        TensorSlice(
            rank=2,
            dims=(
                TensorRange(start=0, length=4),
                TensorRange(start=0, length=4),
            ),
        ),
        TensorSlice(
            rank=2,
            dims=(
                TensorRange(start=0, length=4),
                TensorRange(start=4, length=4),
            ),
        ),
        TensorSlice(
            rank=2,
            dims=(
                TensorRange(start=4, length=4),
                TensorRange(start=0, length=4),
            ),
        ),
        TensorSlice(
            rank=2,
            dims=(
                TensorRange(start=4, length=4),
                TensorRange(start=4, length=4),
            ),
        ),
    }
