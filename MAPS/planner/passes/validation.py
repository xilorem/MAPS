"""Planner-side Pipeline validation facade."""

from __future__ import annotations

from MAPS.pipeline.pipeline import Pipeline
from MAPS.planner.validation.contracts import (
    ConstraintReport,
    ConstraintViolation,
    PlannerConstraints,
    append_violation,
)
from MAPS.planner.validation.memory import estimate_stage_l2_memory
from MAPS.planner.validation.stages import validate_stage
from MAPS.planner.validation.transitions import validate_transition


def validate_constraints(
    pipeline: Pipeline,
    constraints: PlannerConstraints,
) -> ConstraintReport:
    """Validate a completed Pipeline against planner legality constraints.

    Contract:
        Validation is read-only and exhaustive where references remain safe to
        inspect. Invalid pipelines are reported through ``ConstraintReport``;
        ordinary constraint violations are not raised as exceptions.

    Behavior:
        Every stage is checked for mesh consistency, node limits, tensor
        bindings, local/transition input correctness, and optional L1 capacity.
        External-input residency is accumulated for the mesh-wide L2 check, then
        every transition is checked independently.

    Returns:
        A deterministic report containing all violations in stage, mesh-L2, then
        transition order. ``report.is_valid`` is true exactly when it is empty.
    """

    violations: list[ConstraintViolation] = []
    required_l2_memory = 0
    for stage_id, stage in enumerate(pipeline.stages):
        violations.extend(
            validate_stage(stage, stage_id, pipeline, constraints).violations
        )
        if constraints.enforce_l2_capacity:
            required_l2_memory += estimate_stage_l2_memory(stage, pipeline)

    if (
        constraints.enforce_l2_capacity
        and required_l2_memory > pipeline.mesh.l2_memory.size
    ):
        append_violation(
            violations,
            "mesh_l2_capacity_exceeded",
            f"pipeline requires {required_l2_memory} L2 memory but mesh only "
            f"provides {pipeline.mesh.l2_memory.size}",
        )

    for transition_id, transition in enumerate(pipeline.transitions):
        violations.extend(
            validate_transition(
                transition,
                transition_id,
                pipeline,
                constraints,
            ).violations
        )
    return ConstraintReport(tuple(violations))
