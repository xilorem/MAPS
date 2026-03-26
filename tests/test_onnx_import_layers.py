"""Tests for ONNX lowering into the shared graph IR."""

from MAPS.core.graph import OpKind
from MAPS.importers.onnx.graph_parser import parse_graph
from MAPS.importers.onnx.utils import build_tensor_producer_table
from MAPS.ops.gemm import GemmLayerOp


def _make_tiny_matmul_graph():
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [8, 16])
    w = helper.make_tensor_value_info("w", TensorProto.FLOAT, [16, 12])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [8, 12])
    node = helper.make_node("MatMul", inputs=["x", "w"], outputs=["y"], name="matmul_0")
    return helper.make_graph([node], "tiny_matmul", [x, w], [y])


def test_parse_graph_lowers_matmul_to_graph_node() -> None:
    try:
        graph = _make_tiny_matmul_graph()
    except ImportError:
        return

    lowered_graph = parse_graph(graph)

    assert lowered_graph.name == "tiny_matmul"
    assert len(lowered_graph.nodes) == 1
    assert lowered_graph.nodes[0].kind is OpKind.GEMM
    assert isinstance(lowered_graph.nodes[0].payload, GemmLayerOp)
    assert lowered_graph.nodes[0].inputs[0].name == "x"
    assert lowered_graph.nodes[0].inputs[1].name == "w"
    assert lowered_graph.nodes[0].outputs[0].name == "y"
    assert len(lowered_graph.edges) == 3
    input_edges = [edge for edge in lowered_graph.edges if edge.dst is not None]
    assert len(input_edges) == 2
    assert all(edge.src is None for edge in input_edges)


def test_parse_graph_supports_gemm_bias() -> None:
    try:
        from onnx import TensorProto, helper
    except ImportError:
        return

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [8, 16])
    w = helper.make_tensor_value_info("w", TensorProto.FLOAT, [16, 12])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [8, 12])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [8, 12])
    node = helper.make_node("Gemm", inputs=["x", "w", "b"], outputs=["y"], name="gemm_0")
    graph = helper.make_graph([node], "tiny_gemm", [x, w, b], [y])

    lowered_graph = parse_graph(graph)

    assert len(lowered_graph.nodes) == 1
    assert isinstance(lowered_graph.nodes[0].payload, GemmLayerOp)
    assert lowered_graph.nodes[0].payload.y is not None
    assert lowered_graph.nodes[0].payload.y.name == "b"


def test_build_tensor_producer_table_tracks_outputs() -> None:
    try:
        from onnx import TensorProto, helper
    except ImportError:
        return

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [8, 16])
    w0 = helper.make_tensor_value_info("w0", TensorProto.FLOAT, [16, 12])
    y0 = helper.make_tensor_value_info("y0", TensorProto.FLOAT, [8, 12])
    w1 = helper.make_tensor_value_info("w1", TensorProto.FLOAT, [12, 10])
    y1 = helper.make_tensor_value_info("y1", TensorProto.FLOAT, [8, 10])
    node0 = helper.make_node("MatMul", inputs=["x", "w0"], outputs=["y0"], name="matmul_0")
    node1 = helper.make_node("MatMul", inputs=["y0", "w1"], outputs=["y1"], name="matmul_1")
    graph = helper.make_graph([node0, node1], "toy", [x, w0, w1], [y1], value_info=[y0])

    producers = build_tensor_producer_table(graph)

    assert producers["y0"] == ("matmul_0", 0)
    assert producers["y1"] == ("matmul_1", 0)


def test_parse_graph_builds_node_to_node_and_initializer_edges() -> None:
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        return

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [8, 16])
    y1 = helper.make_tensor_value_info("y1", TensorProto.FLOAT, [8, 10])
    y0 = helper.make_tensor_value_info("y0", TensorProto.FLOAT, [8, 12])
    w0 = helper.make_tensor(
        "w0",
        TensorProto.FLOAT,
        [16, 12],
        [0.0] * (16 * 12),
    )
    w1 = helper.make_tensor(
        "w1",
        TensorProto.FLOAT,
        [12, 10],
        [0.0] * (12 * 10),
    )
    node0 = helper.make_node("MatMul", inputs=["x", "w0"], outputs=["y0"], name="matmul_0")
    node1 = helper.make_node("MatMul", inputs=["y0", "w1"], outputs=["y1"], name="matmul_1")
    graph = helper.make_graph(
        [node0, node1],
        "toy",
        [x],
        [y1],
        initializer=[w0, w1],
        value_info=[y0],
    )

    lowered_graph = parse_graph(graph)

    first_node = lowered_graph.nodes[0]
    second_node = lowered_graph.nodes[1]
    incoming_first = [edge for edge in lowered_graph.edges if edge.dst == first_node]
    incoming_second = [edge for edge in lowered_graph.edges if edge.dst == second_node]
    output_edges = [edge for edge in lowered_graph.edges if edge.dst is None]

    assert any(edge.tensor.name == "x" and edge.src is None for edge in incoming_first)
    assert any(edge.tensor.name == "w0" and edge.src is None for edge in incoming_first)
    assert any(edge.tensor.name == "y0" and edge.src == first_node for edge in incoming_second)
    assert any(edge.tensor.name == "w1" and edge.src is None for edge in incoming_second)
    assert any(edge.tensor.name == "y1" and edge.src == second_node for edge in output_edges)
    assert tuple(tensor.name for tensor in lowered_graph.initializers) == ("w0", "w1")
