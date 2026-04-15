"""Planner constraints and planner-side validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

from MAPS.core.layout import TensorRange, TensorSlice
from MAPS.core.ownership import tile_tensor_slice
from MAPS.core.pipeline import Pipeline
from MAPS.core.stage import InputSourceKind, Stage
from MAPS.core.tensor import Tensor
from MAPS.core.transition import Transition
from MAPS.ops.conv import ConvLayerOp
from MAPS.ops.exp import ExpLayerOp
from MAPS.ops.gemm import GemmLayerOp


@dataclass(frozen=True)
class PlannerConstraints:
    """Hard planner-side limits and policy switches."""

    max_stage_nodes: int = 1
    random_seed: int | None = None
    allow_cross_submesh_remap: bool = True
    enforce_l1_capacity: bool = True
    enforce_l2_capacity: bool = True

    def __post_init__(self) -> None:
        if self.max_stage_nodes <= 0:
            raise ValueError("max_stage_nodes must be > 0")


@dataclass(frozen=True)
class ConstraintViolation:
    """One planner-side legality violation."""

    kind: str
    message: str


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
        )

        if tensor == op.x:
            return op.required_x_slice(output_slice)
        if tensor == op.w:
            return op.required_w_slice(output_slice)
        if op.y is not None and tensor == op.y:
            return output_slice

    if node is not None and isinstance(node.payload, ConvLayerOp):
        op = node.payload
        if len(stage.outputs) == 0:
            return _default_tensor_slice(tensor)

        output_binding = stage.outputs[0]
        output_tensor = pipeline.tensors[output_binding.tensor_id]
        output_slice = tile_tensor_slice(
            output_tensor,
            output_binding.layout,
            tile,
        )

        if tensor == op.x:
            return op.required_x_slice(output_slice)
        if tensor == op.w:
            return op.required_w_slice(output_slice)
        if op.b is not None and tensor == op.b:
            bias_slice = op.required_b_slice(output_slice)
            if bias_slice is not None:
                return bias_slice

    if node is not None and isinstance(node.payload, ExpLayerOp):
        op = node.payload
        if len(stage.outputs) == 0:
            return _default_tensor_slice(tensor)

        output_binding = stage.outputs[0]
        output_tensor = pipeline.tensors[output_binding.tensor_id]
        output_slice = tile_tensor_slice(
            output_tensor,
            output_binding.layout,
            tile,
        )

        if tensor == op.x:
            return op.required_x_slice(output_slice)

    return _default_tensor_slice(tensor)


def _estimate_stage_l1_memory_for_tile(
    stage: Stage,
    pipeline: Pipeline,
    tile,
) -> int:
    """Estimate L1-resident bytes required by one stage on one tile."""

    l1_memory = 0

    for binding in stage.outputs:
        tensor = pipeline.tensors[binding.tensor_id]
        tensor_slice = tile_tensor_slice(
            tensor,
            binding.layout,
            tile,
        )
        l1_memory += _tensor_slice_num_bytes(tensor, tensor_slice)

    for binding_idx, binding in enumerate(stage.inputs):
        if binding.source.kind is InputSourceKind.LOCAL:
            continue
        tensor = pipeline.tensors[binding.tensor_id]
        tensor_slice = _infer_input_slice_for_tile(
            stage,
            binding_idx,
            pipeline,
            tile,
        )
        l1_memory += _tensor_slice_num_bytes(tensor, tensor_slice)

    return l1_memory


def _estimate_stage_l2_memory(stage: Stage, pipeline: Pipeline) -> int:
    l2_memory = 0
    for binding_idx, binding in enumerate(stage.inputs):
        if binding.source.kind is not InputSourceKind.EXTERNAL:
            continue
        tensor = pipeline.tensors[binding.tensor_id]
        max_binding_bytes = 0
        for tile in stage.submesh.tiles:
            tensor_slice = _infer_input_slice_for_tile(
                stage,
                binding_idx,
                pipeline,
                tile,
            )
            max_binding_bytes = max(
                max_binding_bytes,
                _tensor_slice_num_bytes(tensor, tensor_slice),
            )
        l2_memory += max_binding_bytes
    return l2_memory


def _validate_transition_input(
    violations: list[ConstraintViolation],
    stage_id: int,
    binding_idx: int,
    binding,
    pipeline: Pipeline,
) -> None:
    transition_id = binding.source.transition_id
    if transition_id is None or transition_id >= len(pipeline.transitions):
        _append_violation(
            violations,
            kind="transition_reference_out_of_range",
            message=(
                f"stage {stage_id} input {binding_idx} references "
                f"missing transition {transition_id}"
            ),
        )
        return

    transition = pipeline.transitions[transition_id]
    if transition.dst_layer_id != stage_id:
        _append_violation(
            violations,
            kind="transition_destination_mismatch",
            message=(
                f"stage {stage_id} input {binding_idx} references transition "
                f"{transition_id} targeting stage {transition.dst_layer_id}"
            ),
        )
    if transition.tensor_id != binding.tensor_id:
        _append_violation(
            violations,
            kind="transition_tensor_mismatch",
            message=(
                f"stage {stage_id} input {binding_idx} tensor_id {binding.tensor_id} "
                f"does not match transition tensor_id {transition.tensor_id}"
            ),
        )


def _validate_local_input(
    violations: list[ConstraintViolation],
    stage_id: int,
    binding_idx: int,
    binding,
    pipeline: Pipeline,
) -> None:
    local_output = binding.source.local_output
    if local_output is None or local_output.stage_id >= len(pipeline.stages):
        _append_violation(
            violations,
            kind="local_output_stage_out_of_range",
            message=(
                f"stage {stage_id} input {binding_idx} references "
                f"missing local stage {None if local_output is None else local_output.stage_id}"
            ),
        )
        return

    local_stage = pipeline.stages[local_output.stage_id]
    if local_output.output_idx >= len(local_stage.outputs):
        _append_violation(
            violations,
            kind="local_output_index_out_of_range",
            message=(
                f"stage {stage_id} input {binding_idx} references missing output "
                f"{local_output.output_idx} on stage {local_output.stage_id}"
            ),
        )
        return

    local_tensor_id = local_stage.outputs[local_output.output_idx].tensor_id
    if local_tensor_id != binding.tensor_id:
        _append_violation(
            violations,
            kind="local_output_tensor_mismatch",
            message=(
                f"stage {stage_id} input {binding_idx} tensor_id {binding.tensor_id} "
                f"does not match referenced local output tensor_id {local_tensor_id}"
            ),
        )


def _validate_stage(
    stage: Stage,
    stage_id: int,
    pipeline: Pipeline,
    constraints: PlannerConstraints,
) -> ConstraintReport:
    violations: list[ConstraintViolation] = []

    if stage.submesh.mesh != pipeline.mesh:
        _append_violation(
            violations,
            kind="stage_mesh_mismatch",
            message=f"stage {stage_id} submesh belongs to a different mesh",
        )

    if len(stage.nodes) > constraints.max_stage_nodes:
        _append_violation(
            violations,
            kind="stage_node_limit_exceeded",
            message=(
                f"stage {stage_id} has {len(stage.nodes)} nodes, exceeding "
                f"max_stage_nodes={constraints.max_stage_nodes}"
            ),
        )

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
        try:
            validator(stage.inputs, stage.outputs, pipeline.tensors)
        except ValueError as exc:
            _append_violation(
                violations,
                kind="stage_payload_invalid",
                message=f"stage {stage_id} node '{node.name}': {exc}",
            )

    for binding_idx, binding in enumerate(stage.inputs):
        if binding.source.kind is InputSourceKind.TRANSITION:
            _validate_transition_input(violations, stage_id, binding_idx, binding, pipeline)
        elif binding.source.kind is InputSourceKind.LOCAL:
            _validate_local_input(violations, stage_id, binding_idx, binding, pipeline)

    if constraints.enforce_l1_capacity:
        for tile in stage.submesh.tiles:
            required_l1_memory = _estimate_stage_l1_memory_for_tile(
                stage,
                pipeline,
                tile,
            )
            if required_l1_memory > tile.memory.size:
                _append_violation(
                    violations,
                    kind="tile_l1_capacity_exceeded",
                    message=(
                        f"stage {stage_id} requires {required_l1_memory} L1 memory "
                        f"but tile {tile.tile_id} only provides {tile.memory.size}"
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
        _append_violation(
            violations,
            kind="cross_submesh_remap_disallowed",
            message=(
                f"transition {transition_id} remaps across different submeshes "
                "while planner constraints disallow it"
            ),
        )

    for fragment_idx, fragment in enumerate(transition.fragments):
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

    required_l2_memory = 0
    for stage_id, stage in enumerate(pipeline.stages):
        violations.extend(
            _validate_stage(stage, stage_id, pipeline, constraints).violations
        )
        if constraints.enforce_l2_capacity:
            required_l2_memory += _estimate_stage_l2_memory(stage, pipeline)

    if constraints.enforce_l2_capacity and required_l2_memory > pipeline.mesh.l2_memory.size:
        _append_violation(
            violations,
            kind="mesh_l2_capacity_exceeded",
            message=(
                f"pipeline requires {required_l2_memory} L2 memory but mesh only "
                f"provides {pipeline.mesh.l2_memory.size}"
            ),
        )

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
