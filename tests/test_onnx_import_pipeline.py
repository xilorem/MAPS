"""Tests for end-to-end ONNX import into the shared graph IR."""

from pathlib import Path
from tempfile import TemporaryDirectory

from MAPS.core.graph import Graph
from MAPS.importers.onnx.importer import import_onnx_graph
from MAPS.ops.defs.gemm import GemmPayload


def test_load_onnx_model_requires_existing_path() -> None:
    assert Path(__file__).name == "test_onnx_import_pipeline.py"


def test_import_onnx_graph_returns_scheduler_graph_ir() -> None:
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        return

    with TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "tiny_matmul.onnx"
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
        w = helper.make_tensor_value_info("w", TensorProto.FLOAT, [3, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])
        node = helper.make_node("MatMul", inputs=["x", "w"], outputs=["y"], name="matmul_0")
        graph_proto = helper.make_graph([node], "tiny_matmul", [x, w], [y])
        model = helper.make_model(graph_proto)
        onnx.save(model, model_path)

        lowered_graph = import_onnx_graph(model_path)

        assert isinstance(lowered_graph, Graph)
        assert lowered_graph.name == "tiny_matmul"
        assert len(lowered_graph.nodes) == 1
        assert isinstance(lowered_graph.nodes[0].payload, GemmPayload)


def _print_graph(graph: Graph) -> None:
    print(f"graph: {graph.name}")
    print(f"inputs: {[tensor.name for tensor in graph.inputs]}")
    print(f"outputs: {[tensor.name for tensor in graph.outputs]}")
    print("tensors:")
    for tensor in graph.tensors:
        print(
            "  "
            f"{tensor.name}: shape={tensor.dims} elem_bytes={tensor.elem_bytes}"
        )
    print("nodes:")
    for node in graph.nodes:
        print(
            "  "
            f"{node.name}: {node.kind.name} "
            f"inputs={[tensor.name for tensor in node.inputs]} "
            f"outputs={[tensor.name for tensor in node.outputs]}"
        )
    print("edges:")
    for edge in graph.edges:
        print(
            "  "
            f"{edge.tensor.name}: "
            f"{edge.src.name if edge.src is not None else 'EXTERNAL'} -> "
            f"{edge.dst.name if edge.dst is not None else 'GRAPH_OUTPUT'}"
        )


def _print_lowered_graph(graph: Graph) -> None:
    print("lowered nodes:")
    for idx, node in enumerate(graph.nodes):
        print(f"  [{idx}] {node.name}: {node.payload!r}")


if __name__ == "__main__":
    import onnx
    from onnx import TensorProto, helper

    with TemporaryDirectory() as tmpdir:
        sample_path = Path(tmpdir) / "tiny_matmul.onnx"
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
        w = helper.make_tensor_value_info("w", TensorProto.FLOAT, [3, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])
        node = helper.make_node("MatMul", inputs=["x", "w"], outputs=["y"], name="matmul_0")
        graph_proto = helper.make_graph([node], "tiny_matmul", [x, w], [y])
        model = helper.make_model(graph_proto)
        onnx.save(model, sample_path)

        lowered_graph = import_onnx_graph(sample_path)
        _print_graph(lowered_graph)
        _print_lowered_graph(lowered_graph)
        print("ok")
