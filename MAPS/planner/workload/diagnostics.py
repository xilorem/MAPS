"""Human-readable diagnostics for workload-balancing results."""

from __future__ import annotations

from MAPS.arch import Mesh
from MAPS.core.graph import Node
from MAPS.planner.contracts.stages import StagePlan, StageSelection
from MAPS.planner.workload.submesh import representative_connected_submesh
from MAPS.planner.workload.metrics import (
    worst_tile_compute_workload,
    worst_tile_l2_transfer_workload,
)


def print_stage_metric_breakdown(
    enabled: bool,
    plans: dict[int, StagePlan],
    stage_selection: StageSelection,
    mesh: Mesh,
    graph_inputs: frozenset,
    graph_outputs: frozenset,
    producer_stage_id_by_tensor: dict[object, int],
    initializer_tensors: frozenset,
) -> None:
    """Print final compute and boundary-IO bottlenecks for every stage."""

    if not enabled:
        return
    print("[workload_balancing] final_stage_metric_breakdown:")
    for stage_id, stage_nodes in stage_selection.items():
        plan = plans[stage_id]
        submesh = representative_connected_submesh(mesh, plan.stage_id, plan.tile_count)
        compute = worst_tile_compute_workload(
            stage_nodes=stage_nodes,
            node_output_layouts=plan.node_output_layouts,
            submesh=submesh,
        )
        io = worst_tile_l2_transfer_workload(
            stage_id=stage_id,
            stage_nodes=stage_nodes,
            node_output_layouts=plan.node_output_layouts,
            submesh=submesh,
            mesh=mesh,
            graph_inputs=graph_inputs,
            graph_outputs=graph_outputs,
            producer_stage_id_by_tensor=producer_stage_id_by_tensor,
            initializer_tensors=initializer_tensors,
        )
        print(
            f"  stage={stage_id} nodes={_stage_label(stage_nodes)} "
            f"compute={compute} worst_tile_io={io}"
        )


def _stage_label(stage_nodes: tuple[Node, ...]) -> str:
    """Return a compact stage label for diagnostics."""

    return "+".join(node.name for node in stage_nodes)
