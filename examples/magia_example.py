"""Run a small ONNX network through the MAGIA planning flow."""

from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MAPS.chips import magia_mesh
from MAPS.planner import PlannerConstraints, validate_constraints
from MAPS.planner.plan import build_pipeline


def _make_initializer(name: str, dims: tuple[int, ...]):
    from onnx import TensorProto, helper

    size = 1
    for dim in dims:
        size *= dim
    values = [0.01] * size
    return helper.make_tensor(name, TensorProto.FLOAT, dims, values)


def _write_example_model(model_path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper

    image = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 1, 64, 64])
    mat_input = helper.make_tensor_value_info("mat_input", TensorProto.FLOAT, [256, 256])

    conv_out = helper.make_tensor_value_info("conv_out", TensorProto.FLOAT, [1, 16, 64, 64])
    exp_out = helper.make_tensor_value_info("exp_out", TensorProto.FLOAT, [1, 16, 64, 64])
    add_out = helper.make_tensor_value_info("add_out", TensorProto.FLOAT, [1, 16, 64, 64])
    matmul_out = helper.make_tensor_value_info("matmul_out", TensorProto.FLOAT, [256, 256])
    gemm_out = helper.make_tensor_value_info("gemm_out", TensorProto.FLOAT, [256, 128])

    nodes = [
        helper.make_node(
            "Conv",
            inputs=["image", "conv_w", "conv_b"],
            outputs=["conv_out"],
            name="conv_0",
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        ),
        helper.make_node("Exp", inputs=["conv_out"], outputs=["exp_out"], name="exp_0"),
        helper.make_node("Add", inputs=["exp_out", "add_bias"], outputs=["add_out"], name="add_0"),
        helper.make_node("MatMul", inputs=["mat_input", "matmul_w"], outputs=["matmul_out"], name="matmul_0"),
        helper.make_node("Gemm", inputs=["matmul_out", "gemm_w", "gemm_y"], outputs=["gemm_out"], name="gemm_0"),
    ]

    graph = helper.make_graph(
        nodes,
        "magia_example",
        [image, mat_input],
        [add_out, gemm_out],
        initializer=[
            _make_initializer("conv_w", (16, 1, 3, 3)),
            _make_initializer("conv_b", (16,)),
            _make_initializer("add_bias", (1, 16, 64, 64)),
            _make_initializer("matmul_w", (256, 256)),
            _make_initializer("gemm_w", (256, 128)),
            _make_initializer("gemm_y", (256, 128)),
        ],
        value_info=[conv_out, exp_out, matmul_out],
    )
    model = helper.make_model(graph)
    onnx.save(model, model_path)


def main():
    mesh = magia_mesh(width=16, height=16)
    with TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "magia_example.onnx"
        _write_example_model(model_path)

        pipeline = build_pipeline(
            model_path,
            mesh,
            print_workload_balancing=True,
            print_spatial_mapping=True,
            print_spatial_mapping_progress=True,
            require_l2_input_access_point=False,
            require_l2_output_access_point=False,
            enable_lossless_spatial_mapping_pruning=True,
            enable_lossy_spatial_mapping_pruning=False,
        )
        report = validate_constraints(pipeline, PlannerConstraints())

    print(f"Model: {pipeline.name}")
    print(f"Mesh: {mesh.width}x{mesh.height}")
    print(f"Stages: {len(pipeline.stages)}")
    print(f"Transitions: {len(pipeline.transitions)}")
    print(f"Constraint valid: {report.is_valid}")
    if report.violations:
        print("Constraint violations:")
        for violation in report.violations:
            print(f"  {violation.kind}: {violation.message}")


if __name__ == "__main__":
    main()
