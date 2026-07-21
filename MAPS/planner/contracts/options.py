"""Configuration contracts for the planner and its individual passes."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WorkloadBalancingOptions:
    """Weights and diagnostics used while allocating virtual stage tiles.

    ``compute_weight`` and ``communication_weight`` scale the two components of
    the bottleneck objective. They do not change legality: every returned plan
    must fit in tile L1 memory and the total allocation must fit on the mesh.
    """

    compute_weight: float = 1.0
    communication_weight: float = 10.0
    print_progress: bool = False


@dataclass(frozen=True)
class SpatialMappingOptions:
    """Control physical-placement diagnostics.

    Progress and result printing affect diagnostics only. The current heuristic
    mapper has no additional public search knobs.
    """

    print_progress: bool = False
    print_mapping: bool = True
    print_costs: bool = False


@dataclass(frozen=True)
class PlannerOptions:
    """Complete configuration for planning an already imported graph.

    Options are grouped by pass so adding a pass-specific setting does not make
    the outer planner signature grow.
    """

    workload: WorkloadBalancingOptions = field(default_factory=WorkloadBalancingOptions)
    spatial_mapping: SpatialMappingOptions = field(default_factory=SpatialMappingOptions)
    print_pipeline_cost: bool = True
