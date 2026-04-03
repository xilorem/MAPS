"""Planner constraints and planner-side validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

from MAPS.core.layout import TensorRange, TensorSlice
from MAPS.core.ownership import tile_tensor_slice
from MAPS.core.pipeline import Pipeline
from MAPS.core.stage import InputSourceKind, Stage
from MAPS.core.tensor import Tensor
from MAPS.core.transition import Transition
from MAPS.ops.gemm import GemmLayerOp


@dataclass(frozen=True)
class PlannerConstraints:
    """Hard planner-side limits and policy switches."""

    max_stage_nodes: int = 1
    min_num_microbatches: int = 1
    max_num_microbatches: int = 1
    random_seed: int | None = None
    allow_cross_submesh_remap: bool = True
    enforce_l1_capacity: bool = True
    enforce_l2_capacity: bool = True

    def __post_init__(self) -> None:
        if self.max_stage_nodes <= 0:
            raise ValueError("max_stage_nodes must be > 0")
        if self.min_num_microbatches <= 0:
            raise ValueError("min_num_microbatches must be > 0")
        if self.max_num_microbatches < self.min_num_microbatches:
            raise ValueError(
                "max_num_microbatches must be >= min_num_microbatches"
            )


@dataclass(frozen=True)
class ConstraintViolation:
    """One planner-side legality violation."""

    kind: str
    message: str

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("constraint violation kind must not be empty")
        if not self.message:
            raise ValueError("constraint violation message must not be empty")


@dataclass(frozen=True)
class ConstraintReport:
    """Collection of planner-side legality violations."""

    violations: tuple[ConstraintViolation, ...] = field(default_factory=tuple)

    @property
    def is_valid(self) -> bool:
        return not self.violations


def _append_violation(
    violations: list[ConstraintViolation],
    kind: str,
    message: str,
) -> None:
    violations.append(ConstraintViolation(kind=kind, message=message))


def _tensor_slice_num_bytes(tensor: Tensor, tensor_slice: TensorSlice) -> int:
    num_elements = 1
    for dim in tensor_slice.dims:
        num_elements *= dim.length
    return num_elements * tensor.elem_bytes


def _default_tensor_slice(tensor: Tensor) -> TensorSlice:
    return TensorSlice(
        rank=tensor.rank,
        dims=tuple(
            TensorRange(
                start=0,
                length=dim,
            )
            for dim in tensor.dims
        ),
    )


def _infer_input_slice_for_tile(
    stage: Stage,
    binding_idx: int,
    pipeline: Pipeline,
    tile,
    microbatch_idx: int,
) -> TensorSlice:
    tensor = pipeline.tensors[stage.inputs[binding_idx].tensor_id]
    node = stage.nodes[0] if len(stage.nodes) == 1 else None

    if node is not None and isinstance(node.payload, GemmLayerOp):
        op = node.payload
        if len(stage.outputs) == 0:
            return _default_tensor_slice(tensor)

        output_binding = stage.outputs[0]
        output_tensor = pipeline.tensors[output_binding.tensor_id]
        output_slice = tile_tensor_slice(
            output_tensor,
            output_binding.layout,
            tile,
            microbatch_idx,
        )

        if tensor == op.x:
            return op.required_x_slice(output_slice)
        if tensor == op.w:
            return op.required_w_slice(output_slice)
        if op.y is not None and tensor == op.y:
            y_slice = op.required_y_slice(output_slice)
            assert y_slice is not None
            return y_slice

    # Fallback for unsupported payloads: treat the whole tensor as required.
    return _default_tensor_slice(tensor)


def _estimate_stage_l1_bytes_for_tile(
    stage: Stage,
    pipeline: Pipeline,
    tile,
    microbatch_idx: int,
) -> int:
    """Estimate L1-resident bytes required by one stage on one tile."""

    l1_bytes = 0

    for binding in stage.outputs:
        tensor = pipeline.tensors[binding.tensor_id]
        tensor_slice = tile_tensor_slice(
            tensor,
            binding.layout,
            tile,
            microbatch_idx,
        )
        l1_bytes += _tensor_slice_num_bytes(tensor, tensor_slice)

    for binding_idx, binding in enumerate(stage.inputs):
        if binding.source.kind is InputSourceKind.LOCAL:
            continue
        tensor = pipeline.tensors[binding.tensor_id]
        tensor_slice = _infer_input_slice_for_tile(
            stage,
            binding_idx,
            pipeline,
            tile,
            microbatch_idx,
        )
        l1_bytes += _tensor_slice_num_bytes(tensor, tensor_slice)

    return l1_bytes


def _estimate_stage_l2_bytes(stage: Stage, pipeline: Pipeline) -> int:
    """Estimate mesh-level L2 bytes required by one stage.

    Current policy:
    - EXTERNAL inputs are assumed to be sourced from shared L2
    - LOCAL inputs do not consume additional L2
    - TRANSITION inputs do not consume L2 for the currently supported remap modes
    """

    l2_bytes = 0
    for binding_idx, binding in enumerate(stage.inputs):
        if binding.source.kind is not InputSourceKind.EXTERNAL:
            continue
        tensor = pipeline.tensors[binding.tensor_id]
        max_binding_bytes = 0
        for tile in stage.submesh.tiles:
            # Use the largest per-microbatch slice to approximate peak shared-L2 demand.
            for microbatch_idx in range(pipeline.num_microbatches):
                tensor_slice = _infer_input_slice_for_tile(
                    stage,
                    binding_idx,
                    pipeline,
                    tile,
                    microbatch_idx=microbatch_idx,
                )
                max_binding_bytes = max(
                    max_binding_bytes,
                    _tensor_slice_num_bytes(tensor, tensor_slice),
                )
        l2_bytes += max_binding_bytes
    return l2_bytes


def _estimate_stage_required_l1_bytes(
    stage: Stage,
    pipeline: Pipeline,
    tile,
) -> int:
    """Estimate peak L1 demand for one tile across all microbatches."""

    required_l1_bytes = 0
    for microbatch_idx in range(pipeline.num_microbatches):
        required_l1_bytes = max(
            required_l1_bytes,
            _estimate_stage_l1_bytes_for_tile(
                stage,
                pipeline,
                tile,
                microbatch_idx=microbatch_idx,
            ),
        )
    return required_l1_bytes


def validate_num_microbatches(
    chosen_num_microbatches: int,
    constraints: PlannerConstraints,
) -> ConstraintReport:
    """Validate one microbatching choice against planner limits."""

    violations: list[ConstraintViolation] = []
    # Enforce the planner-selected microbatch count range.
    if chosen_num_microbatches < constraints.min_num_microbatches:
        violations.append(
            ConstraintViolation(
                kind="num_microbatches_below_min",
                message=(
                    "chosen num_microbatches is below the minimum allowed by "
                    "planner constraints"
                ),
            )
        )
    if chosen_num_microbatches > constraints.max_num_microbatches:
        violations.append(
            ConstraintViolation(
                kind="num_microbatches_above_max",
                message=(
                    "chosen num_microbatches exceeds the maximum allowed by "
                    "planner constraints"
                ),
            )
        )
    return ConstraintReport(violations=tuple(violations))


def _validate_stage(
    stage: Stage,
    stage_id: int,
    pipeline: Pipeline,
    constraints: PlannerConstraints,
) -> ConstraintReport:
    violations: list[ConstraintViolation] = []

    # Check the stage object is structurally well-formed.
    try:
        stage.__post_init__()
    except ValueError as exc:
        _append_violation(
            violations,
            kind="stage_invalid",
            message=f"stage {stage_id}: {exc}",
        )
        return ConstraintReport(violations=tuple(violations))

    # Ensure the stage placement belongs to the same hardware mesh as the pipeline.
    if stage.submesh.mesh != pipeline.mesh:
        _append_violation(
            violations,
            kind="stage_mesh_mismatch",
            message=f"stage {stage_id} submesh belongs to a different mesh",
        )

    # Enforce the current planner limit on how many nodes may share a stage.
    if len(stage.nodes) > constraints.max_stage_nodes:
        _append_violation(
            violations,
            kind="stage_node_limit_exceeded",
            message=(
                f"stage {stage_id} has {len(stage.nodes)} nodes, exceeding "
                f"max_stage_nodes={constraints.max_stage_nodes}"
            ),
        )

    # Validate tensor ids and output layouts against the pipeline tensor table.
    try:
        stage.validate_tensors(pipeline.tensors)
    except ValueError as exc:
        _append_violation(
            violations,
            kind="stage_tensor_binding_invalid",
            message=f"stage {stage_id}: {exc}",
        )

    for node in stage.nodes:
        validator = getattr(node.payload, "validate_tensors", None)
        if validator is None:
            continue
        # Validate op-specific tensor semantics using the node payload contract.
        try:
            validator(stage.inputs, stage.outputs, pipeline.tensors)
        except ValueError as exc:
            _append_violation(
                violations,
                kind="stage_payload_invalid",
                message=f"stage {stage_id} node '{node.name}': {exc}",
            )

    for binding_idx, binding in enumerate(stage.inputs):
        source = binding.source
        # Validate the input-source descriptor itself before chasing references.
        try:
            source.__post_init__()
        except ValueError as exc:
            _append_violation(
                violations,
                kind="stage_input_source_invalid",
                message=f"stage {stage_id} input {binding_idx}: {exc}",
            )
            continue

        if source.kind is InputSourceKind.TRANSITION:
            assert source.transition_id is not None
            # Ensure transition-backed inputs point to a matching transition entry.
            if source.transition_id >= len(pipeline.transitions):
                _append_violation(
                    violations,
                    kind="transition_reference_out_of_range",
                    message=(
                        f"stage {stage_id} input {binding_idx} references "
                        f"missing transition {source.transition_id}"
                    ),
                )
            else:
                transition = pipeline.transitions[source.transition_id]
                if transition.dst_layer_id != stage_id:
                    _append_violation(
                        violations,
                        kind="transition_destination_mismatch",
                        message=(
                            f"stage {stage_id} input {binding_idx} references "
                            f"transition {source.transition_id} targeting "
                            f"stage {transition.dst_layer_id}"
                        ),
                    )
                if transition.tensor_id != binding.tensor_id:
                    _append_violation(
                        violations,
                        kind="transition_tensor_mismatch",
                        message=(
                            f"stage {stage_id} input {binding_idx} tensor_id "
                            f"{binding.tensor_id} does not match transition "
                            f"tensor_id {transition.tensor_id}"
                        ),
                    )

        if source.kind is InputSourceKind.LOCAL:
            assert source.local_output is not None
            # Ensure local-input references point to an existing stage output.
            if source.local_output.stage_id >= len(pipeline.stages):
                _append_violation(
                    violations,
                    kind="local_output_stage_out_of_range",
                    message=(
                        f"stage {stage_id} input {binding_idx} references "
                        f"missing local stage {source.local_output.stage_id}"
                    ),
                )
            else:
                local_stage = pipeline.stages[source.local_output.stage_id]
                if source.local_output.output_idx >= len(local_stage.outputs):
                    _append_violation(
                        violations,
                        kind="local_output_index_out_of_range",
                        message=(
                            f"stage {stage_id} input {binding_idx} references "
                            f"missing output {source.local_output.output_idx} "
                            f"on stage {source.local_output.stage_id}"
                        ),
                    )
                else:
                    local_output = local_stage.outputs[source.local_output.output_idx]
                    if local_output.tensor_id != binding.tensor_id:
                        _append_violation(
                            violations,
                            kind="local_output_tensor_mismatch",
                            message=(
                                f"stage {stage_id} input {binding_idx} tensor_id "
                                f"{binding.tensor_id} does not match referenced "
                                f"local output tensor_id {local_output.tensor_id}"
                            ),
                        )

    if constraints.enforce_l1_capacity:
        # Require every tile used by the stage to advertise a positive L1 capacity.
        for tile in stage.submesh.tiles:
            if tile.l1_bytes <= 0:
                _append_violation(
                    violations,
                    kind="tile_l1_invalid",
                    message=(
                        f"stage {stage_id} submesh references tile {tile.tile_id} "
                        "with non-positive l1_bytes"
                    ),
                )
        # Check stage-local L1 demand using source-kind-aware residency rules.
        for tile in stage.submesh.tiles:
            required_l1_bytes = _estimate_stage_required_l1_bytes(
                stage,
                pipeline,
                tile,
            )
            if required_l1_bytes > tile.l1_bytes:
                _append_violation(
                    violations,
                    kind="tile_l1_capacity_exceeded",
                    message=(
                        f"stage {stage_id} requires {required_l1_bytes} L1 bytes "
                        f"but tile {tile.tile_id} only provides {tile.l1_bytes}"
                    ),
                )

    return ConstraintReport(violations=tuple(violations))


def _validate_transition(
    transition: Transition,
    transition_id: int,
    pipeline: Pipeline,
    constraints: PlannerConstraints,
) -> ConstraintReport:
    violations: list[ConstraintViolation] = []

    # Check the transition object is structurally well-formed.
    try:
        transition.__post_init__()
    except ValueError as exc:
        _append_violation(
            violations,
            kind="transition_invalid",
            message=f"transition {transition_id}: {exc}",
        )
        return ConstraintReport(violations=tuple(violations))

    # Ensure the transition points to a tensor that exists in the pipeline.
    if transition.tensor_id >= len(pipeline.tensors):
        _append_violation(
            violations,
            kind="transition_tensor_out_of_range",
            message=(
                f"transition {transition_id} references missing tensor "
                f"{transition.tensor_id}"
            ),
        )
        return ConstraintReport(violations=tuple(violations))

    # Ensure producer and consumer stage references are in range.
    for attr_name, stage_id in (
        ("src_layer_id", transition.src_layer_id),
        ("dst_layer_id", transition.dst_layer_id),
    ):
        if stage_id >= len(pipeline.stages):
            _append_violation(
                violations,
                kind="transition_stage_out_of_range",
                message=(
                    f"transition {transition_id} {attr_name} references "
                    f"missing stage {stage_id}"
                ),
            )

    if violations:
        return ConstraintReport(violations=tuple(violations))

    tensor = pipeline.tensors[transition.tensor_id]
    # Validate source and destination layouts for the carried tensor.
    try:
        transition.validate_for(tensor)
    except ValueError as exc:
        _append_violation(
            violations,
            kind="transition_layout_invalid",
            message=f"transition {transition_id}: {exc}",
        )

    src_stage = pipeline.stages[transition.src_layer_id]
    dst_stage = pipeline.stages[transition.dst_layer_id]

    # Ensure the transition source output exists and carries the right tensor.
    if transition.src_output_idx >= len(src_stage.outputs):
        _append_violation(
            violations,
            kind="transition_src_output_out_of_range",
            message=(
                f"transition {transition_id} references missing source output "
                f"{transition.src_output_idx} on stage {transition.src_layer_id}"
            ),
        )
    else:
        src_output = src_stage.outputs[transition.src_output_idx]
        if src_output.tensor_id != transition.tensor_id:
            _append_violation(
                violations,
                kind="transition_src_tensor_mismatch",
                message=(
                    f"transition {transition_id} tensor_id does not match its "
                    "source stage output binding"
                ),
            )

    # Ensure the transition destination input exists and expects the right tensor.
    if transition.dst_input_idx >= len(dst_stage.inputs):
        _append_violation(
            violations,
            kind="transition_dst_input_out_of_range",
            message=(
                f"transition {transition_id} references missing destination input "
                f"{transition.dst_input_idx} on stage {transition.dst_layer_id}"
            ),
        )
    else:
        dst_input = dst_stage.inputs[transition.dst_input_idx]
        if dst_input.tensor_id != transition.tensor_id:
            _append_violation(
                violations,
                kind="transition_dst_tensor_mismatch",
                message=(
                    f"transition {transition_id} tensor_id does not match its "
                    "destination stage input binding"
                ),
            )

    if (
        not constraints.allow_cross_submesh_remap
        and transition.src_layout.submesh != transition.dst_layout.submesh
    ):
        # Enforce the planner policy on remaps across different submeshes.
        _append_violation(
            violations,
            kind="cross_submesh_remap_disallowed",
            message=(
                f"transition {transition_id} remaps across different submeshes "
                "while planner constraints disallow it"
            ),
        )

    for fragment_idx, fragment in enumerate(transition.fragments):
        # Validate fragment descriptors and ensure their hart ids are on the mesh.
        try:
            fragment.__post_init__()
        except ValueError as exc:
            _append_violation(
                violations,
                kind="transition_fragment_invalid",
                message=f"transition {transition_id} fragment {fragment_idx}: {exc}",
            )
            continue

        if not pipeline.mesh.contains_tile_id(fragment.src_hartid):
            _append_violation(
                violations,
                kind="transition_fragment_src_out_of_mesh",
                message=(
                    f"transition {transition_id} fragment {fragment_idx} "
                    f"references source hartid {fragment.src_hartid} outside mesh"
                ),
            )
        if not pipeline.mesh.contains_tile_id(fragment.dst_hartid):
            _append_violation(
                violations,
                kind="transition_fragment_dst_out_of_mesh",
                message=(
                    f"transition {transition_id} fragment {fragment_idx} "
                    f"references destination hartid {fragment.dst_hartid} "
                    "outside mesh"
                ),
            )

    return ConstraintReport(violations=tuple(violations))


def validate_constraints(
    pipeline: Pipeline,
    constraints: PlannerConstraints,
) -> ConstraintReport:
    """Validate one pipeline against planner-side hard constraints."""

    violations: list[ConstraintViolation] = []

    # Check the pipeline object is structurally well-formed.
    try:
        pipeline.__post_init__()
    except ValueError as exc:
        _append_violation(
            violations,
            kind="pipeline_invalid",
            message=str(exc),
        )
        return ConstraintReport(violations=tuple(violations))

    # Enforce planner limits on the chosen number of microbatches.
    try:
        num_microbatch_report = validate_num_microbatches(
            pipeline.num_microbatches,
            constraints,
        )
        violations.extend(num_microbatch_report.violations)
    except ValueError as exc:
        _append_violation(
            violations,
            kind="num_microbatches_invalid",
            message=str(exc),
        )

    if constraints.enforce_l2_capacity and pipeline.mesh.l2_bytes <= 0:
        # Require the target mesh to advertise a positive shared L2 capacity.
        _append_violation(
            violations,
            kind="mesh_l2_invalid",
            message="pipeline mesh has non-positive l2_bytes",
        )

    # Validate every scheduled stage against structure and policy constraints.
    required_l2_bytes = 0
    for stage_id, stage in enumerate(pipeline.stages):
        violations.extend(
            _validate_stage(stage, stage_id, pipeline, constraints).violations
        )
        if constraints.enforce_l2_capacity:
            # Accumulate source-kind-aware L2 demand from externally sourced inputs.
            required_l2_bytes += _estimate_stage_l2_bytes(stage, pipeline)

    if constraints.enforce_l2_capacity and required_l2_bytes > pipeline.mesh.l2_bytes:
        _append_violation(
            violations,
            kind="mesh_l2_capacity_exceeded",
            message=(
                f"pipeline requires {required_l2_bytes} L2 bytes but mesh only "
                f"provides {pipeline.mesh.l2_bytes}"
            ),
        )

    # Validate every transition against tensor, stage, and mesh consistency.
    for transition_id, transition in enumerate(pipeline.transitions):
        violations.extend(
            _validate_transition(
                transition,
                transition_id,
                pipeline,
                constraints,
            ).violations
        )

    return ConstraintReport(violations=tuple(violations))
