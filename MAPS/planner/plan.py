"""Top-level planner flow.

This module intentionally contains no planning algorithms.  Reading
``plan_graph`` shows the complete order of planner passes and the data passed
between them.  Detailed behavior lives behind the corresponding pass facade.
"""

from __future__ import annotations

from pathlib import Path

from MAPS.arch import Mesh
from MAPS.core.graph import Graph
from MAPS.importers.onnx.importer import import_onnx_graph
from MAPS.pipeline.pipeline import Pipeline
from MAPS.planner.contracts.options import (
    PlannerOptions,
    SpatialMappingOptions,
    WorkloadBalancingOptions,
)
from MAPS.planner.passes.pipeline_lowering import lower_pipeline
from MAPS.planner.passes.spatial_mapping import map_spatially
from MAPS.planner.passes.stage_selection import select_stages
from MAPS.planner.passes.workload_balancing import balance_workload
from MAPS.planner.reporting.pipeline import print_pipeline_stage_cost
from MAPS.utils.pipeline_json import write_pipeline_json


def plan_graph(
    graph: Graph,
    mesh: Mesh,
    options: PlannerOptions,
) -> Pipeline:
    """Plan an imported graph for a homogeneous multi-tile mesh.

    Contract:
        ``graph`` must be in planner-supported Graph IR and ``mesh`` must fully
        describe its tiles, memories, and NoC.  This function performs no model
        import and writes no files.  All configurable search and diagnostic
        behavior is supplied through ``PlannerOptions``.

    Pass order:
        1. Select graph nodes that execute together as stages.
        2. Allocate virtual tiles and choose stage-local tensor layouts.
        3. Map those virtual stages onto disjoint connected physical regions.
        4. Lower the decisions into executable Pipeline IR.

    Returns:
        A complete physical ``Pipeline``.  A failure to find a legal decision is
        reported as ``ValueError`` by the pass that discovered the infeasibility.
    """


    stage_selection = select_stages(graph)

    stage_plans = balance_workload(
        graph,
        mesh,
        stage_selection=stage_selection,
        debug=options.workload.print_progress,
        compute_weight=options.workload.compute_weight,
        communication_weight=options.workload.communication_weight,
    )

    placements = map_spatially(
        graph,
        mesh,
        stage_plans,
        show_progress=options.spatial_mapping.print_progress,
        print_mapping=options.spatial_mapping.print_mapping,
        print_costs=options.spatial_mapping.print_costs,
    )

    pipeline = lower_pipeline(graph, mesh, stage_plans, placements)

    if options.print_pipeline_cost:
        print_pipeline_stage_cost(graph, mesh, stage_plans, placements)

    return pipeline


def build_pipeline(
    model_path: str | Path,
    mesh: Mesh,
    print_workload_balancing: bool = False,
    print_spatial_mapping: bool = False,
    print_spatial_mapping_progress: bool = False,
    output_json_path: str | Path | None = None,
) -> Pipeline:
    """Main planning entry point"""

    graph = import_onnx_graph(model_path)
    pipeline = plan_graph(
        graph,
        mesh,
        PlannerOptions(
            workload=WorkloadBalancingOptions(
                compute_weight=1.0,
                communication_weight=10.0,
                print_progress=print_workload_balancing,
            ),
            spatial_mapping=SpatialMappingOptions(
                print_progress=print_spatial_mapping_progress,
                print_mapping=not print_spatial_mapping,
                print_costs=print_spatial_mapping,
            ),
        ),
    )
    if output_json_path is not None:
        write_pipeline_json(pipeline, output_json_path)
    return pipeline


__all__ = ["build_pipeline", "plan_graph"]
