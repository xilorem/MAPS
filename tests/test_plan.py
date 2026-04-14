from pathlib import Path
from tempfile import TemporaryDirectory

from MAPS.arch import Mesh, Tile
from MAPS.core.stage import InputSourceKind
from MAPS.planner.plan import build_pipeline


def _mesh_with_l1(width: int, height: int, l1_bytes: int) -> Mesh:
    return Mesh(
        width,
        height,
        l2_bytes=4096,
        tiles=tuple(
            Tile(tile_id=(y * width + x), x=x, y=y, l1_bytes=l1_bytes)
            for y in range(height)
            for x in range(width)
        ),
    )


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

        pipeline = build_pipeline(model_path, _mesh_with_l1(2, 2, l1_bytes=4096))

    assert pipeline.name == "two_matmuls"
    assert len(pipeline.stages) == 2
    assert len(pipeline.transitions) == 1
    assert pipeline.transitions[0].src_layer_id == 0
    assert pipeline.transitions[0].dst_layer_id == 1
    assert pipeline.stages[1].inputs[0].source.kind is InputSourceKind.TRANSITION
    assert pipeline.stages[1].inputs[0].source.transition_id == 0
    assert pipeline.stages[0].inputs[0].source.kind is InputSourceKind.EXTERNAL
