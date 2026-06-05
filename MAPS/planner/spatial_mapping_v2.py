"""Planner v2 spatial mapping over connected tile sets."""

from __future__ import annotations

from contextlib import contextmanager
import importlib
from types import ModuleType
from typing import Iterator

_base = importlib.import_module("MAPS.planner.spatial_mapping")

from MAPS.arch import Mesh
from MAPS.core.graph import Graph
from MAPS.core.submesh import Submesh
from MAPS.planner.connected_submesh import connected_submesh_placements
from MAPS.planner.workload_balancing import StagePlan

place_stage_plans = _base.place_stage_plans


def map_spatially(
    graph: Graph,
    mesh: Mesh,
    tile_counts: dict[int, int] | dict[int, StagePlan],
    objective: str = "max",
    enable_lossless_pruning: bool = False,
    max_placements_per_stage: int | None = 16,
    solver_msg: bool = False,
    show_progress: bool = False,
    print_mapping: bool = True,
    print_costs: bool = False,
    require_l2_input_access_point: bool = False,
    require_l2_output_access_point: bool = False,
) -> dict[int, Submesh]:
    """Map stages onto connected non-rectangular tile sets."""
    with _patched_placement_generator(_base):
        return _base.map_spatially(
            graph=graph,
            mesh=mesh,
            tile_counts=tile_counts,
            objective=objective,
            enable_lossless_pruning=enable_lossless_pruning,
            max_placements_per_stage=max_placements_per_stage,
            solver_msg=solver_msg,
            show_progress=show_progress,
            print_mapping=print_mapping,
            print_costs=print_costs,
            require_l2_input_access_point=require_l2_input_access_point,
            require_l2_output_access_point=require_l2_output_access_point,
        )


@contextmanager
def _patched_placement_generator(module: ModuleType) -> Iterator[None]:
    original_placement_options = module._placement_options
    try:
        module._placement_options = _placement_options_v2
        yield
    finally:
        module._placement_options = original_placement_options


def _placement_options_v2(stage_id: int, tile_count: int, mesh: Mesh):
    return connected_submesh_placements(
        tile_count=tile_count,
        mesh=mesh,
        submesh_id=stage_id,
    )
