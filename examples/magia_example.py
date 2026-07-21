"""Run a small ONNX network through the MAGIA planning flow."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MAPS.hw.chips import magia_mesh
from MAPS.planner.passes.validation import validate_constraints
from MAPS.planner.validation.contracts import PlannerConstraints
from MAPS.planner.plan import build_pipeline
from MAPS.utils.pipeline_json import write_pipeline_json
from MAPS.utils.print_submeshes import print_submeshes

DEFAULT_MODEL_PATH = PROJECT_ROOT / "examples" / "simple_three_stage.onnx"


def main():
    mesh = magia_mesh(width=4, height=4)
    output_path = PROJECT_ROOT / "generated" / "magia_example.pipeline.json"
    pipeline = build_pipeline(
        DEFAULT_MODEL_PATH,
        mesh,
        print_workload_balancing=True,
        print_spatial_mapping=True,
        print_spatial_mapping_progress=True,
    )
    report = validate_constraints(pipeline, PlannerConstraints())

    print(f"Model: {pipeline.name}")
    print(f"Mesh: {mesh.width}x{mesh.height}")
    print(f"Stages: {len(pipeline.stages)}")
    print(f"Transitions: {len(pipeline.transitions)}")
    print(f"Constraint valid: {report.is_valid}")
    print_submeshes(pipeline)
    if report.violations:
        print("Constraint violations:")
        for violation in report.violations:
            print(f"  {violation.kind}: {violation.message}")
    print(f"Pipeline JSON: {write_pipeline_json(pipeline, output_path)}")


if __name__ == "__main__":
    main()
