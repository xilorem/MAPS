"""Validation of inter-stage transition references and fragments."""

from __future__ import annotations

from MAPS.pipeline.pipeline import Pipeline
from MAPS.planner.validation.contracts import (
    ConstraintReport,
    ConstraintViolation,
    PlannerConstraints,
    append_violation,
)
from MAPS.transitions.model import Transition


def validate_transition(
    transition: Transition,
    transition_id: int,
    pipeline: Pipeline,
    constraints: PlannerConstraints,
) -> ConstraintReport:
    """Validate one transition's references, layouts, bindings, and endpoints.

    Structural range checks run first because later checks need valid tensors and
    stages.  Once those references are safe, the transition is validated against
    its tensor, source output, destination input, cross-submesh policy, and the
    mesh membership of every physical fragment endpoint.
    """

    violations: list[ConstraintViolation] = []
    if transition.tensor_id >= len(pipeline.tensors):
        append_violation(
            violations,
            "transition_tensor_out_of_range",
            f"transition {transition_id} references missing tensor "
            f"{transition.tensor_id}",
        )
        return ConstraintReport(tuple(violations))

    for attribute_name, stage_id in (
        ("src_layer_id", transition.src_layer_id),
        ("dst_layer_id", transition.dst_layer_id),
    ):
        if stage_id >= len(pipeline.stages):
            append_violation(
                violations,
                "transition_stage_out_of_range",
                f"transition {transition_id} {attribute_name} references "
                f"missing stage {stage_id}",
            )
    if violations:
        return ConstraintReport(tuple(violations))

    tensor = pipeline.tensors[transition.tensor_id]
    try:
        transition.validate_for(tensor)
    except ValueError as exc:
        append_violation(
            violations,
            "transition_layout_invalid",
            f"transition {transition_id}: {exc}",
        )

    source_stage = pipeline.stages[transition.src_layer_id]
    destination_stage = pipeline.stages[transition.dst_layer_id]
    source_outputs = source_stage.layers[-1].outputs
    destination_inputs = destination_stage.layers[0].inputs
    if transition.src_output_idx >= len(source_outputs):
        append_violation(
            violations,
            "transition_src_output_out_of_range",
            f"transition {transition_id} references missing source output "
            f"{transition.src_output_idx} on stage {transition.src_layer_id}",
        )
    elif source_outputs[transition.src_output_idx].tensor_id != transition.tensor_id:
        append_violation(
            violations,
            "transition_src_tensor_mismatch",
            f"transition {transition_id} tensor_id does not match its "
            "source stage output binding",
        )

    if transition.dst_input_idx >= len(destination_inputs):
        append_violation(
            violations,
            "transition_dst_input_out_of_range",
            f"transition {transition_id} references missing destination input "
            f"{transition.dst_input_idx} on stage {transition.dst_layer_id}",
        )
    elif destination_inputs[transition.dst_input_idx].tensor_id != transition.tensor_id:
        append_violation(
            violations,
            "transition_dst_tensor_mismatch",
            f"transition {transition_id} tensor_id does not match its "
            "destination stage input binding",
        )

    if (
        not constraints.allow_cross_submesh_remap
        and transition.src_layout.submesh != transition.dst_layout.submesh
    ):
        append_violation(
            violations,
            "cross_submesh_remap_disallowed",
            f"transition {transition_id} remaps across different submeshes "
            "while planner constraints disallow it",
        )

    for fragment_idx, fragment in enumerate(transition.fragments):
        if not pipeline.mesh.contains_tile_id(fragment.src_hartid):
            append_violation(
                violations,
                "transition_fragment_src_out_of_mesh",
                f"transition {transition_id} fragment {fragment_idx} references "
                f"source hartid {fragment.src_hartid} outside mesh",
            )
        if not pipeline.mesh.contains_tile_id(fragment.dst_hartid):
            append_violation(
                violations,
                "transition_fragment_dst_out_of_mesh",
                f"transition {transition_id} fragment {fragment_idx} references "
                f"destination hartid {fragment.dst_hartid} outside mesh",
            )
    return ConstraintReport(tuple(violations))
