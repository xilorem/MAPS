"""Diagnostics for complete planner results."""

from __future__ import annotations

from MAPS.arch import Mesh
from MAPS.core.graph import Graph
from MAPS.planner.contracts.stages import StagePlacement, StagePlan, virtual_submesh
from MAPS.planner.spatial.evaluation import evaluate_mapping
from MAPS.planner.workload.metrics import worst_tile_compute_workload


def print_pipeline_stage_cost(
    graph: Graph,
    mesh: Mesh,
    stage_plans: dict[int, StagePlan],
    placements: dict[int, StagePlacement],
) -> None:
    """Print the combined worst-stage compute and physical IO estimate.

    Compute is evaluated from the final virtual layouts. Physical IO is
    evaluated from the separate spatial placements and ownership maps. The
    displayed total is the sum of the greatest stage compute and IO bottlenecks.
    """

    worst_stage_compute = max(
        (
            worst_tile_compute_workload(
                stage_nodes=plan.nodes,
                node_output_layouts=plan.node_output_layouts,
                submesh=virtual_submesh(plan),
            )
            for plan in stage_plans.values()
        ),
        default=0,
    )
    node_stage_ids = {
        id(node): stage_id
        for stage_id, plan in stage_plans.items()
        for node in plan.nodes
    }
    evaluation = evaluate_mapping(
        graph,
        mesh,
        stage_plans,
        placements,
        node_stage_ids,
    )
    worst_stage_io = max(
        (breakdown.total for breakdown in evaluation.stage_breakdowns.values()),
        default=0,
    )
    print(
        "[planner] pipeline_stage_cost="
        f"{worst_stage_compute + worst_stage_io} "
        f"(worst_compute={worst_stage_compute} worst_io={worst_stage_io})"
    )
