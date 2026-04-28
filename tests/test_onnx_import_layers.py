"""Tests for ONNX lowering into the shared graph IR."""

from dataclasses import dataclass

from MAPS.core.graph import OpKind
from MAPS.importers.onnx.graph_parser import parse_graph
from MAPS.importers.onnx.tensor_parser import onnx_dtype_elem_bytes
from MAPS.importers.onnx.utils import build_tensor_producer_table
from MAPS.ops import SoftmaxPayload
from MAPS.ops.defs.collective import AllReducePayload
from MAPS.ops.defs.conv import ConvPayload
from MAPS.ops.defs.elementwise import BinaryElementwisePayload, UnaryElementwisePayload
from MAPS.ops.defs.gemm import GemmPayload
from MAPS.ops.registry import get_onnx_lowerer, register_op, registered_ops
from MAPS.ops.defs.reduction import ReductionPayload
from MAPS.ops.spec import OpSpec
from MAPS.transforms import decompose_graph


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
    assert isinstance(lowered_graph.nodes[0].payload, GemmPayload)
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
    assert isinstance(lowered_graph.nodes[0].payload, GemmPayload)
    assert lowered_graph.nodes[0].payload.y is not None
    assert lowered_graph.nodes[0].payload.y.name == "b"


def test_parse_graph_lowers_conv_to_graph_node() -> None:
    try:
        from onnx import TensorProto, helper
    except ImportError:
        return

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])
    w = helper.make_tensor_value_info("w", TensorProto.FLOAT, [8, 3, 3, 3])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 4, 4])
    node = helper.make_node(
        "Conv",
        inputs=["x", "w", "b"],
        outputs=["y"],
        name="conv_0",
        strides=[2, 2],
        pads=[1, 1, 1, 1],
    )
    graph = helper.make_graph([node], "tiny_conv", [x, w, b], [y])

    lowered_graph = parse_graph(graph)

    assert len(lowered_graph.nodes) == 1
    assert lowered_graph.nodes[0].kind is OpKind.CONV
    assert isinstance(lowered_graph.nodes[0].payload, ConvPayload)
    assert lowered_graph.nodes[0].payload.strides == (2, 2)
    assert lowered_graph.nodes[0].payload.pads == (1, 1, 1, 1)
    assert lowered_graph.nodes[0].payload.b is not None
    assert lowered_graph.nodes[0].attributes["strides"] == (2, 2)


def test_parse_graph_lowers_exp_to_graph_node() -> None:
    try:
        from onnx import TensorProto, helper
    except ImportError:
        return

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 8])
    node = helper.make_node("Exp", inputs=["x"], outputs=["y"], name="exp_0")
    graph = helper.make_graph([node], "tiny_exp", [x], [y])

    lowered_graph = parse_graph(graph)

    assert len(lowered_graph.nodes) == 1
    assert lowered_graph.nodes[0].kind is OpKind.ELEMENTWISE
    assert isinstance(lowered_graph.nodes[0].payload, UnaryElementwisePayload)
    assert lowered_graph.nodes[0].payload.op_name == "Exp"
    assert lowered_graph.nodes[0].payload.x.name == "x"
    assert lowered_graph.nodes[0].payload.output.name == "y"


def test_parse_graph_lowers_binary_elementwise_to_graph_node() -> None:
    try:
        from onnx import TensorProto, helper
    except ImportError:
        return

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 8])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 8])
    node = helper.make_node("Add", inputs=["x", "b"], outputs=["y"], name="add_0")
    graph = helper.make_graph([node], "tiny_add", [x, b], [y])

    lowered_graph = parse_graph(graph)

    assert len(lowered_graph.nodes) == 1
    assert lowered_graph.nodes[0].kind is OpKind.ELEMENTWISE
    assert isinstance(lowered_graph.nodes[0].payload, BinaryElementwisePayload)
    assert lowered_graph.nodes[0].payload.op_name == "Add"


def test_parse_graph_keeps_softmax_as_high_level_node() -> None:
    try:
        from onnx import TensorProto, helper
    except ImportError:
        return

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 8])
    node = helper.make_node("Softmax", inputs=["x"], outputs=["y"], name="softmax_0", axis=-1)
    graph = helper.make_graph([node], "tiny_softmax", [x], [y])

    lowered_graph = parse_graph(graph)

    assert len(lowered_graph.nodes) == 1
    assert lowered_graph.nodes[0].name == "softmax_0"
    assert lowered_graph.nodes[0].kind is OpKind.CUSTOM
    assert isinstance(lowered_graph.nodes[0].payload, SoftmaxPayload)
    assert lowered_graph.nodes[0].payload.axis == 1


def test_decompose_graph_lowers_softmax_to_grouped_internal_nodes() -> None:
    try:
        from onnx import TensorProto, helper
    except ImportError:
        return

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 8])
    node = helper.make_node("Softmax", inputs=["x"], outputs=["y"], name="softmax_0", axis=-1)
    graph = helper.make_graph([node], "tiny_softmax", [x], [y])

    lowered_graph = decompose_graph(parse_graph(graph))

    assert tuple(node.name for node in lowered_graph.nodes) == (
        "softmax_0__reduce_max",
        "softmax_0__allreduce_max",
        "softmax_0__sub",
        "softmax_0__exp",
        "softmax_0__reduce_sum",
        "softmax_0__allreduce_sum",
        "softmax_0__div",
    )
    assert isinstance(lowered_graph.nodes[0].payload, ReductionPayload)
    assert isinstance(lowered_graph.nodes[1].payload, AllReducePayload)
    assert isinstance(lowered_graph.nodes[2].payload, BinaryElementwisePayload)
    assert isinstance(lowered_graph.nodes[3].payload, UnaryElementwisePayload)
    assert isinstance(lowered_graph.nodes[4].payload, ReductionPayload)
    assert isinstance(lowered_graph.nodes[5].payload, AllReducePayload)
    assert isinstance(lowered_graph.nodes[6].payload, BinaryElementwisePayload)
    assert all(
        node.attributes["stage_group_id"] == "softmax_0::softmax"
        for node in lowered_graph.nodes
    )
    assert tuple(tensor.name for tensor in lowered_graph.outputs) == ("y",)
    assert {
        "softmax_0__max_local",
        "softmax_0__max_global",
        "softmax_0__shifted",
        "softmax_0__exp",
        "softmax_0__sum_local",
        "softmax_0__sum_global",
    }.issubset({tensor.name for tensor in lowered_graph.tensors})

    edges_by_dst = {
        node.name: {edge.tensor.name for edge in lowered_graph.edges if edge.dst == node}
        for node in lowered_graph.nodes
    }
    assert edges_by_dst["softmax_0__reduce_max"] == {"x"}
    assert edges_by_dst["softmax_0__allreduce_max"] == {"softmax_0__max_local"}
    assert edges_by_dst["softmax_0__sub"] == {"x", "softmax_0__max_global"}
    assert edges_by_dst["softmax_0__exp"] == {"softmax_0__shifted"}
    assert edges_by_dst["softmax_0__reduce_sum"] == {"softmax_0__exp"}
    assert edges_by_dst["softmax_0__allreduce_sum"] == {"softmax_0__sum_local"}
    assert edges_by_dst["softmax_0__div"] == {"softmax_0__exp", "softmax_0__sum_global"}

    output_edges = [edge for edge in lowered_graph.edges if edge.dst is None]
    assert len(output_edges) == 1
    assert output_edges[0].src == lowered_graph.nodes[-1]
    assert output_edges[0].tensor.name == "y"


def test_decompose_graph_lowers_softmax_without_collectives_outside_default_mesh_axes() -> None:
    try:
        from onnx import TensorProto, helper
    except ImportError:
        return

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4, 8])
    node = helper.make_node("Softmax", inputs=["x"], outputs=["y"], name="softmax_0", axis=0)
    graph = helper.make_graph([node], "tiny_softmax_no_collective", [x], [y])

    lowered_graph = decompose_graph(parse_graph(graph))

    assert tuple(node.name for node in lowered_graph.nodes) == (
        "softmax_0__reduce_max",
        "softmax_0__sub",
        "softmax_0__exp",
        "softmax_0__reduce_sum",
        "softmax_0__div",
    )
    assert all(not isinstance(node.payload, AllReducePayload) for node in lowered_graph.nodes)


def test_op_registry_reports_supported_onnx_ops() -> None:
    assert get_onnx_lowerer("MatMul") is not None
    assert get_onnx_lowerer("Gemm") is not None
    assert get_onnx_lowerer("Conv") is not None
    assert get_onnx_lowerer("Exp") is not None
    assert get_onnx_lowerer("Softmax") is not None
    assert {spec.name for spec in registered_ops()} >= {"matmul", "gemm", "conv", "softmax"}


@dataclass(frozen=True)
class _FakePayload:
    x: object
    output: object


def _lower_fake_identity(
    node_name: str,
    inputs: tuple[object, ...],
    outputs: tuple[object, ...],
    attributes: dict[str, object],
) -> tuple[OpKind, object]:
    del node_name, attributes
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("FakeIdentityTestOp expects exactly one input and one output")
    return OpKind.CUSTOM, _FakePayload(x=inputs[0], output=outputs[0])


def test_parse_graph_uses_registry_for_new_test_op() -> None:
    try:
        from onnx import TensorProto, helper
    except ImportError:
        return

    register_op(
        OpSpec(
            name="fake_identity_test",
            onnx_names=("FakeIdentityTestOp",),
            lower_onnx=_lower_fake_identity,
            payload_type=_FakePayload,
        )
    )

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 8])
    node = helper.make_node("FakeIdentityTestOp", inputs=["x"], outputs=["y"], name="fake_0")
    graph = helper.make_graph([node], "tiny_fake_identity", [x], [y])

    lowered_graph = parse_graph(graph)

    assert len(lowered_graph.nodes) == 1
    assert lowered_graph.nodes[0].name == "fake_0"
    assert lowered_graph.nodes[0].kind is OpKind.CUSTOM
    assert isinstance(lowered_graph.nodes[0].payload, _FakePayload)


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


def test_onnx_dtype_elem_bytes_maps_common_float32() -> None:
    assert onnx_dtype_elem_bytes(1) == 4
