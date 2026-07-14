from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from MAPS.arch import EndpointKind, Mesh, Tile
from MAPS.core.graph import Graph, Node
from MAPS.core.layout import TensorLayout, TensorSlice, tile_tensor_slice
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.planner.placement import StagePlacement
from MAPS.planner.workload_balancing import StagePlan
from MAPS.transitions import build_direct_remap_fragments
from MAPS.transitions.transport import TransportCostModel


@dataclass(frozen=True)
class VirtualTraffic:
    """Virtual communication summary between already balanced stages."""

    stage_comm: dict[tuple[int, int], int]
    edge_matrices: dict[tuple[int, int], dict[tuple[int, int], int]]
    input_weights: dict[int, dict[int, int]]
    output_weights: dict[int, dict[int, int]]
    l2_read_weights: dict[int, dict[int, int]]
    l2_write_weights: dict[int, dict[int, int]]
    communication_degree: dict[int, int]
    bottleneck_risk: dict[int, int]
    l2_pressure: dict[int, int]


@dataclass(frozen=True)
class TileIOScore:
    """Exact physical IO accounting for one tile."""

    tile_id: int
    stage_id: int | None
    tile_to_tile_writes: int
    l2_reads: int
    l2_writes: int
    consumer_stage_writes: dict[int, int]

    @property
    def score(self) -> int:
        """Return the additive physical IO score for one tile."""

        return self.tile_to_tile_writes + self.l2_reads + self.l2_writes


@dataclass(frozen=True)
class StageIOBreakdown:
    """Worst physical tile IO components for one placed stage."""

    physical_tile_id: int | None
    l2_read: int
    l2_write: int
    l1_write: int

    @property
    def total(self) -> int:
        """Return the additive physical IO score of the worst tile."""

        return self.l1_write + self.l2_read + self.l2_write


@dataclass(frozen=True)
class MappingEvaluation:
    """Exact score for a complete ownership-aware spatial mapping."""

    placements: dict[int, StagePlacement]
    tile_scores: dict[int, TileIOScore]
    stage_breakdowns: dict[int, StageIOBreakdown]
    objective: tuple[int, int, int, int]
    worst_tile_id: int | None


@dataclass(frozen=True)
class RepairCandidate:
    """A local region that may improve the current bottleneck."""

    stages: frozenset[int]
    priority: float
    reason: str


def map_spatially(
    graph: Graph,
    mesh: Mesh,
    stage_plans: dict[int, StagePlan],
    objective: str = "max",
    max_placements_per_stage: int | None = 16,
    solver_msg: bool = False,
    show_progress: bool = False,
    print_mapping: bool = True,
    print_costs: bool = False,
    require_l2_input_access_point: bool = False,
    require_l2_output_access_point: bool = False,
) -> dict[int, StagePlacement]:
    """Place connected stage regions, assign ownership, then locally repair IO bottlenecks."""

    del objective
    del max_placements_per_stage
    del solver_msg
    del require_l2_input_access_point
    del require_l2_output_access_point

    stage_selection = {stage_id: plan.nodes for stage_id, plan in stage_plans.items()}
    node_stage_ids = {id(node): stage_id for stage_id, nodes in stage_selection.items() for node in nodes}
    tile_counts = {stage_id: plan.tile_count for stage_id, plan in stage_plans.items()}

    if sum(tile_counts.values()) > mesh.num_tiles:
        raise ValueError("requested stage tiles exceed available mesh tiles")

    traffic = build_virtual_traffic(
        graph=graph,
        mesh=mesh,
        stage_plans=stage_plans,
        node_stage_ids=node_stage_ids,
    )
    _debug(show_progress, "[spatial_mapping] phase=virtual_analysis")
    _debug(
        show_progress,
        "[spatial_mapping] "
        f"stage_order={_stage_order(tile_counts, traffic)} "
        f"communication_degree={traffic.communication_degree} "
        f"bottleneck_risk={traffic.bottleneck_risk} "
        f"l2_pressure={traffic.l2_pressure}",
    )

    placements = _initial_stage_placements(
        mesh=mesh,
        stage_plans=stage_plans,
        tile_counts=tile_counts,
        traffic=traffic,
        debug=show_progress,
    )
    placements = _assign_stage_ownerships(
        mesh=mesh,
        stage_plans=stage_plans,
        placements=placements,
        traffic=traffic,
    )
    evaluation = _evaluate_mapping(
        graph=graph,
        mesh=mesh,
        stage_plans=stage_plans,
        placements=placements,
        node_stage_ids=node_stage_ids,
    )
    _debug(
        show_progress,
        "[spatial_mapping] "
        f"phase=initial_mapping objective={evaluation.objective} worst_tile={evaluation.worst_tile_id}",
    )

    improved = _improve_spatial_mapping(
        graph=graph,
        mesh=mesh,
        stage_plans=stage_plans,
        placements=placements,
        traffic=traffic,
        node_stage_ids=node_stage_ids,
        initial_evaluation=evaluation,
        debug=show_progress,
    )

    if print_costs:
        print_spatial_mapping_details(
            graph=graph,
            mesh=mesh,
            stage_plans=stage_plans,
            placements=improved,
            node_stage_ids=node_stage_ids,
            label="ownership_aware",
        )
    elif print_mapping:
        _print_placement_grid(mesh, improved)

    return improved


def build_virtual_traffic(
    graph: Graph,
    mesh: Mesh,
    stage_plans: dict[int, StagePlan],
    node_stage_ids: dict[int, int],
) -> VirtualTraffic:
    """Build virtual communication matrices before physical placement."""

    del mesh

    producer_by_tensor = {tensor: node for node in graph.nodes for tensor in node.outputs}
    initializer_tensors = frozenset(graph.initializers)
    graph_inputs = frozenset(graph.inputs) - initializer_tensors
    graph_outputs = frozenset(graph.outputs)

    stage_ids = tuple(stage_plans)
    stage_comm: dict[tuple[int, int], int] = {}
    edge_matrices: dict[tuple[int, int], dict[tuple[int, int], int]] = {}
    input_weights = {stage_id: _zero_by_virtual_tile(_stage_virtual_submesh(stage_plans[stage_id])) for stage_id in stage_ids}
    output_weights = {stage_id: _zero_by_virtual_tile(_stage_virtual_submesh(stage_plans[stage_id])) for stage_id in stage_ids}
    l2_read_weights = {stage_id: _zero_by_virtual_tile(_stage_virtual_submesh(stage_plans[stage_id])) for stage_id in stage_ids}
    l2_write_weights = {stage_id: _zero_by_virtual_tile(_stage_virtual_submesh(stage_plans[stage_id])) for stage_id in stage_ids}

    for dst_stage_id, dst_plan in stage_plans.items():
        for dst_node, dst_output_layouts in zip(dst_plan.nodes, dst_plan.node_output_layouts):
            for tensor in dst_node.inputs:
                if tensor in initializer_tensors:
                    continue
                required_slices = _transition_required_slices(
                    tensor=tensor,
                    dst_node=dst_node,
                    dst_output_layouts=dst_output_layouts,
                )
                src_node = producer_by_tensor.get(tensor)
                if tensor in graph_inputs or src_node is None:
                    for dst_tile, dst_tensor_slice in required_slices:
                        bytes_ = tensor.slice_num_bytes(dst_tensor_slice)
                        input_weights[dst_stage_id][dst_tile.tile_id] += bytes_
                        l2_read_weights[dst_stage_id][dst_tile.tile_id] += bytes_
                    continue

                src_stage_id = node_stage_ids[id(src_node)]
                if src_stage_id == dst_stage_id:
                    continue

                src_layout = _stage_node_output_layout(stage_plans[src_stage_id], src_node)[
                    _node_output_index(src_node, tensor)
                ]
                fragments = build_direct_remap_fragments(
                    tensor=tensor,
                    src_layout=src_layout,
                    dst_required_slices=required_slices,
                )
                matrix = edge_matrices.setdefault((src_stage_id, dst_stage_id), {})
                for fragment in fragments:
                    bytes_ = fragment.src_subslice.num_elements * tensor.elem_bytes
                    key = (fragment.src_hartid, fragment.dst_hartid)
                    matrix[key] = matrix.get(key, 0) + bytes_
                    stage_comm[(src_stage_id, dst_stage_id)] = stage_comm.get((src_stage_id, dst_stage_id), 0) + bytes_
                    output_weights[src_stage_id][fragment.src_hartid] += bytes_
                    input_weights[dst_stage_id][fragment.dst_hartid] += bytes_

            for output_idx, tensor in enumerate(dst_node.outputs):
                if tensor not in graph_outputs:
                    continue
                output_layout = dst_output_layouts[output_idx]
                for virtual_tile in output_layout.submesh.tiles:
                    bytes_ = tensor.slice_num_bytes(tile_tensor_slice(tensor, output_layout, virtual_tile))
                    output_weights[dst_stage_id][virtual_tile.tile_id] += bytes_
                    l2_write_weights[dst_stage_id][virtual_tile.tile_id] += bytes_

    communication_degree = {
        stage_id: sum(
            weight
            for (src_stage_id, dst_stage_id), weight in stage_comm.items()
            if src_stage_id == stage_id or dst_stage_id == stage_id
        )
        for stage_id in stage_ids
    }
    bottleneck_risk = {
        stage_id: max(input_weights[stage_id].values(), default=0)
        for stage_id in stage_ids
    }
    l2_pressure = {
        stage_id: sum(l2_read_weights[stage_id].values()) + sum(l2_write_weights[stage_id].values())
        for stage_id in stage_ids
    }
    return VirtualTraffic(
        stage_comm=stage_comm,
        edge_matrices=edge_matrices,
        input_weights=input_weights,
        output_weights=output_weights,
        l2_read_weights=l2_read_weights,
        l2_write_weights=l2_write_weights,
        communication_degree=communication_degree,
        bottleneck_risk=bottleneck_risk,
        l2_pressure=l2_pressure,
    )


def _initial_stage_placements(
    mesh: Mesh,
    stage_plans: dict[int, StagePlan],
    tile_counts: dict[int, int],
    traffic: VirtualTraffic,
    debug: bool,
) -> dict[int, StagePlacement]:
    """Build the first feasible communication-aware placement."""

    free_tile_ids = set(range(mesh.num_tiles))
    placed_regions: dict[int, set[int]] = {}
    stage_order = _stage_order(tile_counts, traffic)
    _debug(debug, f"[spatial_mapping] phase=initial_seeding stage_order={stage_order}")

    for stage_idx, stage_id in enumerate(stage_order):
        remaining_tile_counts = {
            other_stage_id: tile_counts[other_stage_id]
            for other_stage_id in stage_order[stage_idx + 1:]
        }
        target = _stage_target_point(
            stage_id=stage_id,
            mesh=mesh,
            placed_regions=placed_regions,
            traffic=traffic,
        )
        region = _grow_stage_region(
            stage_id=stage_id,
            mesh=mesh,
            allowed_tile_ids=free_tile_ids,
            tile_count=tile_counts[stage_id],
            target=target,
            traffic=traffic,
            placed_regions=placed_regions,
            remaining_tile_counts=remaining_tile_counts,
        )
        placed_regions[stage_id] = region
        free_tile_ids -= region
        _debug(
            debug,
            "[spatial_mapping] "
            f"seeded stage={stage_id} target=({target[0]:.2f},{target[1]:.2f}) "
            f"tiles={sorted(region)}",
        )

    placements = _placements_from_regions(mesh, stage_plans, placed_regions)
    return placements


def _improve_spatial_mapping(
    graph: Graph,
    mesh: Mesh,
    stage_plans: dict[int, StagePlan],
    placements: dict[int, StagePlacement],
    traffic: VirtualTraffic,
    node_stage_ids: dict[int, int],
    initial_evaluation: MappingEvaluation,
    debug: bool,
    max_iters: int = 32,
    max_repair_regions: int = 5,
) -> dict[int, StagePlacement]:
    """Apply ownership-aware local repairs until the exact objective stalls."""

    current_placements = placements
    current_evaluation = initial_evaluation
    tile_counts = {stage_id: plan.tile_count for stage_id, plan in stage_plans.items()}
    tabu: deque[frozenset[int]] = deque(maxlen=10)

    for iteration in range(max_iters):
        if current_evaluation.worst_tile_id is None:
            break

        worst_tile = current_evaluation.tile_scores[current_evaluation.worst_tile_id]
        if worst_tile.stage_id is None:
            break

        candidates = _choose_repair_regions(
            mesh=mesh,
            placements=current_placements,
            traffic=traffic,
            evaluation=current_evaluation,
            worst_tile=worst_tile,
        )
        _debug(
            debug,
            "[spatial_mapping] "
            f"iter={iteration} objective={current_evaluation.objective} "
            f"worst_tile={worst_tile.tile_id} worst_stage={worst_tile.stage_id} "
            f"repair_candidates={[ (sorted(candidate.stages), candidate.reason) for candidate in candidates[:max_repair_regions] ]}",
        )

        best_trial: MappingEvaluation | None = None
        best_trial_placements: dict[int, StagePlacement] | None = None
        best_candidate: RepairCandidate | None = None

        for candidate in candidates[:max_repair_regions]:
            if candidate.stages in tabu:
                continue

            trial_placements = _repair_region(
                mesh=mesh,
                stage_plans=stage_plans,
                current_placements=current_placements,
                traffic=traffic,
                affected_stages=candidate.stages,
                focus_stage_id=worst_tile.stage_id,
                debug=debug,
            )
            if trial_placements is None:
                continue

            trial_placements = _assign_stage_ownerships(
                mesh=mesh,
                stage_plans=stage_plans,
                placements=trial_placements,
                traffic=traffic,
            )
            trial_evaluation = _evaluate_mapping(
                graph=graph,
                mesh=mesh,
                stage_plans=stage_plans,
                placements=trial_placements,
                node_stage_ids=node_stage_ids,
            )
            _debug(
                debug,
                "[spatial_mapping] "
                f"iter={iteration} region={sorted(candidate.stages)} reason={candidate.reason} "
                f"trial_objective={trial_evaluation.objective}",
            )
            if trial_evaluation.objective < current_evaluation.objective and (
                best_trial is None or trial_evaluation.objective < best_trial.objective
            ):
                best_trial = trial_evaluation
                best_trial_placements = trial_placements
                best_candidate = candidate

        if best_trial is None or best_trial_placements is None or best_candidate is None:
            _debug(debug, f"[spatial_mapping] iter={iteration} no_improving_repair_found")
            break

        current_placements = best_trial_placements
        current_evaluation = best_trial
        tabu.append(best_candidate.stages)
        _debug(
            debug,
            "[spatial_mapping] "
            f"iter={iteration} accepted_region={sorted(best_candidate.stages)} "
            f"reason={best_candidate.reason} objective={current_evaluation.objective}",
        )

    return current_placements


def _repair_region(
    mesh: Mesh,
    stage_plans: dict[int, StagePlan],
    current_placements: dict[int, StagePlacement],
    traffic: VirtualTraffic,
    affected_stages: frozenset[int],
    focus_stage_id: int,
    debug: bool,
) -> dict[int, StagePlacement] | None:
    """Replace one local stage set inside its old union of physical tiles."""

    affected_tile_ids = {
        tile_id
        for stage_id in affected_stages
        for tile_id in current_placements[stage_id].physical_submesh.tile_ids
    }
    fixed_regions = {
        stage_id: set(placement.physical_submesh.tile_ids)
        for stage_id, placement in current_placements.items()
        if stage_id not in affected_stages
    }
    local_tile_counts = {stage_id: stage_plans[stage_id].tile_count for stage_id in affected_stages}
    local_order = _local_stage_order(
        affected_stages=affected_stages,
        tile_counts=local_tile_counts,
        traffic=traffic,
        focus_stage_id=focus_stage_id,
    )

    best_regions: dict[int, set[int]] | None = None
    best_key: tuple[float, tuple[int, ...], tuple[int, ...]] | None = None
    restart_count = max(1, min(4, len(affected_tile_ids)))
    for restart_idx in range(restart_count):
        free_tile_ids = set(affected_tile_ids)
        placed_regions = dict(fixed_regions)
        local_regions: dict[int, set[int]] = {}
        feasible = True
        for order_idx, stage_id in enumerate(local_order):
            remaining_tile_counts = {
                other_stage_id: local_tile_counts[other_stage_id]
                for other_stage_id in local_order[order_idx + 1:]
            }
            target = _stage_target_point(
                stage_id=stage_id,
                mesh=mesh,
                placed_regions=placed_regions,
                traffic=traffic,
            )
            seed_candidates = _sorted_candidate_tiles(
                mesh=mesh,
                candidate_tile_ids=free_tile_ids,
                target=target,
                stage_id=stage_id,
                traffic=traffic,
                placed_regions=placed_regions,
            )
            if not seed_candidates:
                feasible = False
                break
            preferred_seed = seed_candidates[min(restart_idx, len(seed_candidates) - 1)]
            try:
                region = _grow_stage_region(
                    stage_id=stage_id,
                    mesh=mesh,
                    allowed_tile_ids=free_tile_ids,
                    tile_count=local_tile_counts[stage_id],
                    target=target,
                    traffic=traffic,
                    placed_regions=placed_regions,
                    remaining_tile_counts=remaining_tile_counts,
                    preferred_seed=preferred_seed,
                )
            except ValueError:
                feasible = False
                break
            local_regions[stage_id] = region
            placed_regions[stage_id] = region
            free_tile_ids -= region

        if not feasible or set(local_regions) != set(affected_stages):
            continue

        focus_region = tuple(sorted(local_regions[focus_stage_id]))
        key = (
            _region_anchor_cost(
                stage_id=focus_stage_id,
                mesh=mesh,
                region=local_regions[focus_stage_id],
                traffic=traffic,
                placed_regions=placed_regions,
            ),
            focus_region,
            tuple(stage_id for stage_id in local_order),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_regions = local_regions

    if best_regions is None:
        _debug(debug, f"[spatial_mapping] repair_failed stages={sorted(affected_stages)}")
        return None

    merged_regions = {
        stage_id: set(placement.physical_submesh.tile_ids)
        for stage_id, placement in current_placements.items()
    }
    merged_regions.update(best_regions)
    _debug(debug, f"[spatial_mapping] repair_regions stages={sorted(affected_stages)}")
    return _placements_from_regions(mesh, stage_plans, merged_regions)


def _choose_repair_regions(
    mesh: Mesh,
    placements: dict[int, StagePlacement],
    traffic: VirtualTraffic,
    evaluation: MappingEvaluation,
    worst_tile: TileIOScore,
) -> list[RepairCandidate]:
    """Rank local repair regions using bottleneck blame and physical blockers."""

    if worst_tile.stage_id is None:
        return []

    bottleneck_stage_id = worst_tile.stage_id
    candidates: dict[frozenset[int], RepairCandidate] = {}
    consumer_blames = sorted(
        worst_tile.consumer_stage_writes.items(),
        key=lambda item: (-item[1], item[0]),
    )

    for consumer_stage_id, blame in consumer_blames[:2]:
        direct_region = frozenset({bottleneck_stage_id, consumer_stage_id})
        _record_candidate(
            candidates,
            direct_region,
            priority=float(blame),
            reason=f"direct_{bottleneck_stage_id}_to_{consumer_stage_id}",
        )
        blocker_stage_id = _first_blocker_on_path(
            mesh=mesh,
            placements=placements,
            src_stage_id=bottleneck_stage_id,
            dst_stage_id=consumer_stage_id,
        )
        if blocker_stage_id is not None:
            _record_candidate(
                candidates,
                frozenset({bottleneck_stage_id, blocker_stage_id}),
                priority=float(blame) * 0.9,
                reason=f"blocker_{blocker_stage_id}_toward_{consumer_stage_id}",
            )

    l2_blocker = _first_blocker_to_l2(mesh, placements, bottleneck_stage_id)
    if l2_blocker is not None:
        _record_candidate(
            candidates,
            frozenset({bottleneck_stage_id, l2_blocker}),
            priority=float(worst_tile.l2_reads + worst_tile.l2_writes),
            reason="l2_blocker",
        )

    for neighbor_stage_id in _neighbor_stage_ids(mesh, placements, bottleneck_stage_id):
        interface_bonus = _shared_boundary_length(
            mesh,
            placements[bottleneck_stage_id].physical_submesh.tile_ids,
            placements[neighbor_stage_id].physical_submesh.tile_ids,
        )
        _record_candidate(
            candidates,
            frozenset({bottleneck_stage_id, neighbor_stage_id}),
            priority=float(interface_bonus),
            reason=f"physical_neighbor_{neighbor_stage_id}",
        )

    if len(consumer_blames) >= 2:
        left_stage_id, left_blame = consumer_blames[0]
        right_stage_id, right_blame = consumer_blames[1]
        stronger = max(left_blame, right_blame)
        weaker = min(left_blame, right_blame)
        if weaker > 0 and stronger <= int(1.5 * weaker):
            left_blocker = _first_blocker_on_path(
                mesh=mesh,
                placements=placements,
                src_stage_id=bottleneck_stage_id,
                dst_stage_id=left_stage_id,
            )
            right_blocker = _first_blocker_on_path(
                mesh=mesh,
                placements=placements,
                src_stage_id=bottleneck_stage_id,
                dst_stage_id=right_stage_id,
            )
            multi_region = {bottleneck_stage_id}
            if left_blocker is not None:
                multi_region.add(left_blocker)
            if right_blocker is not None:
                multi_region.add(right_blocker)
            if len(multi_region) >= 2:
                _record_candidate(
                    candidates,
                    frozenset(multi_region),
                    priority=float(left_blame + right_blame),
                    reason="balanced_multi_source",
                )

    return sorted(
        candidates.values(),
        key=lambda candidate: (-candidate.priority, len(candidate.stages), tuple(sorted(candidate.stages))),
    )


def _assign_stage_ownerships(
    mesh: Mesh,
    stage_plans: dict[int, StagePlan],
    placements: dict[int, StagePlacement],
    traffic: VirtualTraffic,
) -> dict[int, StagePlacement]:
    """Assign virtual tiles to physical tiles using virtual communication pressure."""

    stage_order = _stage_order(
        {stage_id: plan.tile_count for stage_id, plan in stage_plans.items()},
        traffic,
    )
    stage_centers = {
        stage_id: _tile_set_center(mesh, placement.physical_submesh.tile_ids)
        for stage_id, placement in placements.items()
    }
    assigned: dict[int, StagePlacement] = {}

    for stage_id in stage_order:
        placement = placements[stage_id]
        virtual_tile_ids = tuple(tile.tile_id for tile in placement.virtual_submesh.tiles)
        physical_tile_ids = tuple(sorted(placement.physical_submesh.tile_ids))
        stage_owner_by_virtual: dict[int, int] = {}
        free_physical_tile_ids = set(physical_tile_ids)
        virtual_priority = sorted(
            virtual_tile_ids,
            key=lambda virtual_tile_id: (
                -_virtual_priority(stage_id, virtual_tile_id, traffic),
                virtual_tile_id,
            ),
        )

        for virtual_tile_id in virtual_priority:
            best_physical_tile_id = min(
                free_physical_tile_ids,
                key=lambda physical_tile_id: (
                    _virtual_assignment_cost(
                        mesh=mesh,
                        stage_id=stage_id,
                        virtual_tile_id=virtual_tile_id,
                        physical_tile_id=physical_tile_id,
                        placements=placements,
                        assigned=assigned,
                        stage_centers=stage_centers,
                        traffic=traffic,
                    ),
                    mesh.tile_by_id(physical_tile_id).y,
                    mesh.tile_by_id(physical_tile_id).x,
                    physical_tile_id,
                ),
            )
            stage_owner_by_virtual[virtual_tile_id] = best_physical_tile_id
            free_physical_tile_ids.remove(best_physical_tile_id)

        assigned[stage_id] = StagePlacement(
            stage_id=placement.stage_id,
            virtual_submesh=placement.virtual_submesh,
            physical_submesh=placement.physical_submesh,
            virtual_to_physical=stage_owner_by_virtual,
        )

    return assigned


def _evaluate_mapping(
    graph: Graph,
    mesh: Mesh,
    stage_plans: dict[int, StagePlan],
    placements: dict[int, StagePlacement],
    node_stage_ids: dict[int, int],
) -> MappingEvaluation:
    """Compute exact tile IO scores after placement and ownership are known."""

    model = TransportCostModel(mesh=mesh)
    producer_by_tensor = {tensor: node for node in graph.nodes for tensor in node.outputs}
    initializer_tensors = frozenset(graph.initializers)
    graph_inputs = frozenset(graph.inputs) - initializer_tensors
    graph_outputs = frozenset(graph.outputs)
    stage_of_tile = {
        tile_id: stage_id
        for stage_id, placement in placements.items()
        for tile_id in placement.physical_submesh.tile_ids
    }
    tile_writes = {tile_id: 0 for tile_id in stage_of_tile}
    tile_l2_reads = {tile_id: 0 for tile_id in stage_of_tile}
    tile_l2_writes = {tile_id: 0 for tile_id in stage_of_tile}
    consumer_stage_writes = {tile_id: {} for tile_id in stage_of_tile}

    for dst_stage_id, dst_plan in stage_plans.items():
        dst_placement = placements[dst_stage_id]
        for dst_node, dst_output_layouts in zip(dst_plan.nodes, dst_plan.node_output_layouts):
            for tensor in dst_node.inputs:
                if tensor in initializer_tensors:
                    continue
                required_slices = _transition_required_slices(
                    tensor=tensor,
                    dst_node=dst_node,
                    dst_output_layouts=dst_output_layouts,
                )
                src_node = producer_by_tensor.get(tensor)
                if tensor in graph_inputs or src_node is None:
                    for virtual_tile, tensor_slice in required_slices:
                        dst_tile_id = dst_placement.physical_tile_id(virtual_tile.tile_id)
                        tile_l2_reads[dst_tile_id] += model.l2_to_l1(
                            mesh.tile_by_id(dst_tile_id),
                            tensor.slice_num_bytes(tensor_slice),
                        )
                    continue

                src_stage_id = node_stage_ids[id(src_node)]
                if src_stage_id == dst_stage_id:
                    continue

                src_layout = _stage_node_output_layout(stage_plans[src_stage_id], src_node)[
                    _node_output_index(src_node, tensor)
                ]
                fragments = build_direct_remap_fragments(
                    tensor=tensor,
                    src_layout=src_layout,
                    dst_required_slices=required_slices,
                )
                src_placement = placements[src_stage_id]
                for fragment in fragments:
                    bytes_ = fragment.src_subslice.num_elements * tensor.elem_bytes
                    src_tile = mesh.tile_by_id(src_placement.physical_tile_id(fragment.src_hartid))
                    dst_tile = mesh.tile_by_id(dst_placement.physical_tile_id(fragment.dst_hartid))
                    transfer_cost = model.l1_to_l1(src_tile, dst_tile, bytes_)
                    src_tile_id = src_tile.tile_id
                    tile_writes[src_tile_id] += transfer_cost
                    consumer_stage_writes[src_tile_id][dst_stage_id] = (
                        consumer_stage_writes[src_tile_id].get(dst_stage_id, 0) + transfer_cost
                    )

            for output_idx, tensor in enumerate(dst_node.outputs):
                if tensor not in graph_outputs:
                    continue
                output_layout = dst_output_layouts[output_idx]
                for virtual_tile in output_layout.submesh.tiles:
                    dst_tile_id = dst_placement.physical_tile_id(virtual_tile.tile_id)
                    tile_l2_writes[dst_tile_id] += model.l1_to_l2(
                        mesh.tile_by_id(dst_tile_id),
                        tensor.slice_num_bytes(tile_tensor_slice(tensor, output_layout, virtual_tile)),
                    )

    tile_scores = {
        tile_id: TileIOScore(
            tile_id=tile_id,
            stage_id=stage_of_tile.get(tile_id),
            tile_to_tile_writes=tile_writes.get(tile_id, 0),
            l2_reads=tile_l2_reads.get(tile_id, 0),
            l2_writes=tile_l2_writes.get(tile_id, 0),
            consumer_stage_writes=dict(sorted(consumer_stage_writes.get(tile_id, {}).items())),
        )
        for tile_id in stage_of_tile
    }
    objective = _tile_score_objective(tile_scores)
    worst_tile_id = max(
        tile_scores,
        key=lambda tile_id: (tile_scores[tile_id].score, -tile_id),
        default=None,
    )
    stage_breakdowns = {}
    for stage_id, placement in placements.items():
        worst_stage_tile = max(
            placement.physical_submesh.tile_ids,
            key=lambda tile_id: (tile_scores[tile_id].score, -tile_id),
            default=None,
        )
        if worst_stage_tile is None:
            stage_breakdowns[stage_id] = StageIOBreakdown(None, 0, 0, 0)
            continue
        tile_score = tile_scores[worst_stage_tile]
        stage_breakdowns[stage_id] = StageIOBreakdown(
            physical_tile_id=worst_stage_tile,
            l2_read=tile_score.l2_reads,
            l2_write=tile_score.l2_writes,
            l1_write=tile_score.tile_to_tile_writes,
        )

    return MappingEvaluation(
        placements=placements,
        tile_scores=tile_scores,
        stage_breakdowns=stage_breakdowns,
        objective=objective,
        worst_tile_id=worst_tile_id,
    )


def print_spatial_mapping_details(
    graph: Graph,
    mesh: Mesh,
    stage_plans: dict[int, StagePlan],
    placements: dict[int, StagePlacement],
    node_stage_ids: dict[int, int],
    label: str = "mapping",
) -> None:
    """Print final physical placements and exact physical IO costs."""

    evaluation = _evaluate_mapping(
        graph=graph,
        mesh=mesh,
        stage_plans=stage_plans,
        placements=placements,
        node_stage_ids=node_stage_ids,
    )

    print(f"\n[spatial_mapping] chosen physical submeshes for {label}:")
    for stage_id in stage_plans:
        placement = placements[stage_id]
        submesh = placement.physical_submesh
        print(
            f"  stage={stage_id} name={_stage_name(stage_plans[stage_id].nodes)} "
            f"bbox=({submesh.x0},{submesh.y0},{submesh.width},{submesh.height}) "
            f"tiles={sorted(submesh.tile_ids)} "
            f"virtual_to_physical={dict(sorted(placement.virtual_to_physical.items()))}"
        )
    _print_placement_grid(mesh, placements)

    print(f"[spatial_mapping] stage worst physical-tile IO costs for {label}:")
    for stage_id in stage_plans:
        io_cost = evaluation.stage_breakdowns[stage_id]
        print(
            f"  stage={stage_id} name={_stage_name(stage_plans[stage_id].nodes)} "
            f"tile={io_cost.physical_tile_id} "
            f"l2_read={io_cost.l2_read} "
            f"l2_write={io_cost.l2_write} "
            f"l1_write={io_cost.l1_write} "
            f"total={io_cost.total}"
        )
    print(
        f"[spatial_mapping] bottleneck for {label} "
        f"worst_stage_io={max((io_cost.total for io_cost in evaluation.stage_breakdowns.values()), default=0)} "
        f"objective={evaluation.objective}"
    )


def evaluate_stage_plans_io(
    graph: Graph,
    mesh: Mesh,
    stage_plans: dict[int, StagePlan],
) -> MappingEvaluation:
    """Evaluate placed stage plans with the ownership-aware IO metric."""

    placements = {
        stage_id: StagePlacement(
            stage_id=stage_id,
            virtual_submesh=_stage_virtual_submesh(plan),
            physical_submesh=plan.physical_submesh,
            virtual_to_physical=plan.virtual_to_physical,
        )
        for stage_id, plan in stage_plans.items()
    }
    node_stage_ids = {
        id(node): stage_id
        for stage_id, plan in stage_plans.items()
        for node in plan.nodes
    }
    return _evaluate_mapping(
        graph=graph,
        mesh=mesh,
        stage_plans=stage_plans,
        placements=placements,
        node_stage_ids=node_stage_ids,
    )


def place_stage_plans(
    stage_plans: dict[int, StagePlan],
    placements: dict[int, StagePlacement],
) -> dict[int, StagePlan]:
    """Attach physical placement bindings while preserving virtual layouts."""

    return {
        stage_id: StagePlan(
            stage_id=plan.stage_id,
            tile_count=plan.tile_count,
            logical_shape=plan.logical_shape,
            output_layouts=plan.output_layouts,
            nodes=plan.nodes,
            node_output_layouts=plan.node_output_layouts,
            physical_submesh=placements[stage_id].physical_submesh,
            virtual_to_physical=placements[stage_id].virtual_to_physical,
        )
        for stage_id, plan in stage_plans.items()
    }


def _stage_order(
    tile_counts: dict[int, int],
    traffic: VirtualTraffic,
) -> tuple[int, ...]:
    """Sort stages by communication-aware placement priority."""

    return tuple(
        sorted(
            tile_counts,
            key=lambda stage_id: (
                -tile_counts[stage_id],
                -traffic.communication_degree.get(stage_id, 0),
                -traffic.bottleneck_risk.get(stage_id, 0),
                -traffic.l2_pressure.get(stage_id, 0),
                stage_id,
            ),
        )
    )


def _local_stage_order(
    affected_stages: frozenset[int],
    tile_counts: dict[int, int],
    traffic: VirtualTraffic,
    focus_stage_id: int,
) -> tuple[int, ...]:
    """Bias local repair around the current bottleneck stage."""

    return tuple(
        sorted(
            affected_stages,
            key=lambda stage_id: (
                0 if stage_id == focus_stage_id else 1,
                -traffic.communication_degree.get(stage_id, 0),
                -traffic.bottleneck_risk.get(stage_id, 0),
                -tile_counts[stage_id],
                stage_id,
            ),
        )
    )


def _stage_target_point(
    stage_id: int,
    mesh: Mesh,
    placed_regions: dict[int, set[int]],
    traffic: VirtualTraffic,
) -> tuple[float, float]:
    """Return the weighted placement target for one stage."""

    weighted_points: list[tuple[float, float, float]] = []
    for (src_stage_id, dst_stage_id), weight in traffic.stage_comm.items():
        if weight <= 0:
            continue
        if dst_stage_id == stage_id and src_stage_id in placed_regions:
            x, y = _tile_set_center(mesh, placed_regions[src_stage_id])
            weighted_points.append((x, y, float(weight)))
        elif src_stage_id == stage_id and dst_stage_id in placed_regions:
            x, y = _tile_set_center(mesh, placed_regions[dst_stage_id])
            weighted_points.append((x, y, float(weight)))

    l2_access_points = tuple(sorted(_l2_access_point_tile_ids(mesh)))
    if l2_access_points and traffic.l2_pressure.get(stage_id, 0) > 0:
        l2_center = _tile_set_center(mesh, set(l2_access_points))
        weighted_points.append((l2_center[0], l2_center[1], float(traffic.l2_pressure[stage_id])))

    if not weighted_points:
        return ((mesh.width - 1) / 2.0, (mesh.height - 1) / 2.0)

    total_weight = sum(weight for _, _, weight in weighted_points)
    return (
        sum(x * weight for x, _, weight in weighted_points) / total_weight,
        sum(y * weight for _, y, weight in weighted_points) / total_weight,
    )


def _grow_stage_region(
    stage_id: int,
    mesh: Mesh,
    allowed_tile_ids: set[int],
    tile_count: int,
    target: tuple[float, float],
    traffic: VirtualTraffic,
    placed_regions: dict[int, set[int]],
    remaining_tile_counts: dict[int, int],
    preferred_seed: int | None = None,
) -> set[int]:
    """Grow one connected stage region while protecting future feasibility."""

    seed_candidates = _sorted_candidate_tiles(
        mesh=mesh,
        candidate_tile_ids=allowed_tile_ids,
        target=target,
        stage_id=stage_id,
        traffic=traffic,
        placed_regions=placed_regions,
    )
    if preferred_seed is not None and preferred_seed in allowed_tile_ids:
        seed_candidates = [preferred_seed] + [tile_id for tile_id in seed_candidates if tile_id != preferred_seed]
    if not seed_candidates:
        raise ValueError(f"cannot seed stage {stage_id} from an empty free region")

    failures: list[str] = []
    for seed_tile_id in seed_candidates[: min(len(seed_candidates), 16)]:
        try:
            return _greedy_connected_region(
                stage_id=stage_id,
                mesh=mesh,
                seed_tile_id=seed_tile_id,
                allowed_tile_ids=allowed_tile_ids,
                tile_count=tile_count,
                target=target,
                traffic=traffic,
                placed_regions=placed_regions,
                remaining_tile_counts=remaining_tile_counts,
            )
        except ValueError as exc:
            failures.append(str(exc))

    region = _beam_connected_region(
        stage_id=stage_id,
        mesh=mesh,
        allowed_tile_ids=allowed_tile_ids,
        tile_count=tile_count,
        target=target,
        traffic=traffic,
        placed_regions=placed_regions,
        remaining_tile_counts=remaining_tile_counts,
    )
    if region is None:
        raise ValueError("; ".join(failures) if failures else f"cannot grow region for stage {stage_id}")
    return region


def _greedy_connected_region(
    stage_id: int,
    mesh: Mesh,
    seed_tile_id: int,
    allowed_tile_ids: set[int],
    tile_count: int,
    target: tuple[float, float],
    traffic: VirtualTraffic,
    placed_regions: dict[int, set[int]],
    remaining_tile_counts: dict[int, int],
) -> set[int]:
    """Greedily grow a connected stage region from one seed tile."""

    chosen = {seed_tile_id}
    frontier = (_neighbor_ids(mesh, seed_tile_id) & allowed_tile_ids) - chosen

    while len(chosen) < tile_count:
        if not frontier:
            raise ValueError(f"stage {stage_id} cannot grow a connected region")
        next_tile_id = None
        candidates = sorted(
            frontier,
            key=lambda tile_id: _growth_candidate_score(
                stage_id=stage_id,
                mesh=mesh,
                tile_id=tile_id,
                chosen=chosen,
                target=target,
                traffic=traffic,
                placed_regions=placed_regions,
                allowed_tile_ids=allowed_tile_ids,
                remaining_tile_counts=remaining_tile_counts,
            ),
        )
        for candidate_tile_id in candidates:
            candidate_region = chosen | {candidate_tile_id}
            if _future_feasible_after_choice(
                mesh=mesh,
                allowed_tile_ids=allowed_tile_ids,
                chosen_tile_ids=candidate_region,
                remaining_tile_counts=remaining_tile_counts,
                current_stage_remaining_tiles=tile_count - len(candidate_region),
            ):
                next_tile_id = candidate_tile_id
                break
        if next_tile_id is None:
            raise ValueError(f"stage {stage_id} fragments the remaining free region")
        chosen.add(next_tile_id)
        frontier.remove(next_tile_id)
        frontier |= ((_neighbor_ids(mesh, next_tile_id) & allowed_tile_ids) - chosen)

    return chosen


def _beam_connected_region(
    stage_id: int,
    mesh: Mesh,
    allowed_tile_ids: set[int],
    tile_count: int,
    target: tuple[float, float],
    traffic: VirtualTraffic,
    placed_regions: dict[int, set[int]],
    remaining_tile_counts: dict[int, int],
) -> set[int] | None:
    """Search a wider set of connected regions when greedy growth gets boxed in."""

    beam_width = 256
    regions = {frozenset({tile_id}) for tile_id in allowed_tile_ids}
    for _ in range(1, tile_count):
        next_regions: set[frozenset[int]] = set()
        for region in regions:
            frontier = set()
            for tile_id in region:
                frontier |= ((_neighbor_ids(mesh, tile_id) & allowed_tile_ids) - set(region))
            for tile_id in frontier:
                next_regions.add(frozenset((*region, tile_id)))
        if not next_regions:
            return None
        scored_regions = sorted(
            next_regions,
            key=lambda region: _region_score(
                stage_id=stage_id,
                mesh=mesh,
                region=set(region),
                target=target,
                traffic=traffic,
                placed_regions=placed_regions,
                allowed_tile_ids=allowed_tile_ids,
                remaining_tile_counts=remaining_tile_counts,
            ),
        )
        regions = set(scored_regions[:beam_width])

    feasible_regions = [
        set(region)
        for region in regions
        if _future_feasible_after_choice(
            mesh=mesh,
            allowed_tile_ids=allowed_tile_ids,
            chosen_tile_ids=set(region),
            remaining_tile_counts=remaining_tile_counts,
            current_stage_remaining_tiles=0,
        )
    ]
    if not feasible_regions:
        return None
    return min(
        feasible_regions,
        key=lambda region: _region_score(
            stage_id=stage_id,
            mesh=mesh,
            region=region,
            target=target,
            traffic=traffic,
            placed_regions=placed_regions,
            allowed_tile_ids=allowed_tile_ids,
            remaining_tile_counts=remaining_tile_counts,
        ),
    )


def _sorted_candidate_tiles(
    mesh: Mesh,
    candidate_tile_ids: Iterable[int],
    target: tuple[float, float],
    stage_id: int,
    traffic: VirtualTraffic,
    placed_regions: dict[int, set[int]],
) -> list[int]:
    """Return candidate seed tiles ordered by communication-aware target score."""

    return sorted(
        candidate_tile_ids,
        key=lambda tile_id: (
            _seed_tile_score(
                stage_id=stage_id,
                mesh=mesh,
                tile=mesh.tile_by_id(tile_id),
                target=target,
                traffic=traffic,
                placed_regions=placed_regions,
            ),
            mesh.tile_by_id(tile_id).y,
            mesh.tile_by_id(tile_id).x,
            tile_id,
        ),
    )


def _seed_tile_score(
    stage_id: int,
    mesh: Mesh,
    tile: Tile,
    target: tuple[float, float],
    traffic: VirtualTraffic,
    placed_regions: dict[int, set[int]],
) -> float:
    """Score one seed tile by communication and L2 access."""

    score = abs(tile.x - target[0]) + abs(tile.y - target[1])
    score += _stage_anchor_cost(
        mesh=mesh,
        stage_id=stage_id,
        tile=tile,
        traffic=traffic,
        placed_regions=placed_regions,
    )
    return score


def _growth_candidate_score(
    stage_id: int,
    mesh: Mesh,
    tile_id: int,
    chosen: set[int],
    target: tuple[float, float],
    traffic: VirtualTraffic,
    placed_regions: dict[int, set[int]],
    allowed_tile_ids: set[int],
    remaining_tile_counts: dict[int, int],
) -> tuple[float, float, float, int]:
    """Score one frontier tile for region growth."""

    tile = mesh.tile_by_id(tile_id)
    candidate_region = chosen | {tile_id}
    target_cost = abs(tile.x - target[0]) + abs(tile.y - target[1])
    compactness_cost = _region_compactness(mesh, candidate_region)
    anchor_cost = _stage_anchor_cost(
        mesh=mesh,
        stage_id=stage_id,
        tile=tile,
        traffic=traffic,
        placed_regions=placed_regions,
    )
    future_penalty = _future_space_penalty(
        mesh=mesh,
        free_tile_ids=allowed_tile_ids - candidate_region,
        remaining_tile_counts=_remaining_counts_tuple(remaining_tile_counts),
    )
    return (target_cost + anchor_cost + compactness_cost + future_penalty, future_penalty, compactness_cost, tile_id)


def _region_score(
    stage_id: int,
    mesh: Mesh,
    region: set[int],
    target: tuple[float, float],
    traffic: VirtualTraffic,
    placed_regions: dict[int, set[int]],
    allowed_tile_ids: set[int],
    remaining_tile_counts: dict[int, int],
) -> tuple[float, float, float, tuple[int, ...]]:
    """Score a complete connected region."""

    center = _tile_set_center(mesh, region)
    target_cost = abs(center[0] - target[0]) + abs(center[1] - target[1])
    anchor_cost = _region_anchor_cost(
        stage_id=stage_id,
        mesh=mesh,
        region=region,
        traffic=traffic,
        placed_regions=placed_regions,
    )
    compactness_cost = _region_compactness(mesh, region)
    future_penalty = _future_space_penalty(
        mesh=mesh,
        free_tile_ids=allowed_tile_ids - region,
        remaining_tile_counts=_remaining_counts_tuple(remaining_tile_counts),
    )
    return (
        target_cost + anchor_cost + compactness_cost + future_penalty,
        future_penalty,
        compactness_cost,
        tuple(sorted(region)),
    )


def _stage_anchor_cost(
    mesh: Mesh,
    stage_id: int,
    tile: Tile,
    traffic: VirtualTraffic,
    placed_regions: dict[int, set[int]],
) -> float:
    """Estimate how expensive one physical tile is relative to already placed communication anchors."""

    score = 0.0
    for (src_stage_id, dst_stage_id), weight in traffic.stage_comm.items():
        if weight <= 0:
            continue
        if src_stage_id == stage_id and dst_stage_id in placed_regions:
            center = _tile_set_center(mesh, placed_regions[dst_stage_id])
            score += weight * _tile_to_point_distance(tile, center)
        elif dst_stage_id == stage_id and src_stage_id in placed_regions:
            center = _tile_set_center(mesh, placed_regions[src_stage_id])
            score += weight * _tile_to_point_distance(tile, center)

    l2_weight = traffic.l2_pressure.get(stage_id, 0)
    if l2_weight > 0:
        l2_access_points = tuple(
            (mesh.tile_by_id(tile_id).x, mesh.tile_by_id(tile_id).y)
            for tile_id in _l2_access_point_tile_ids(mesh)
        )
        if l2_access_points:
            min_distance = min(
                abs(tile.x - x) + abs(tile.y - y)
                for x, y in l2_access_points
            )
            score += l2_weight * min_distance
    return score / max(1, len(traffic.stage_comm))


def _region_anchor_cost(
    stage_id: int,
    mesh: Mesh,
    region: set[int],
    traffic: VirtualTraffic,
    placed_regions: dict[int, set[int]],
) -> float:
    """Estimate how well a whole region lines up with already placed neighbors."""

    return sum(
        _stage_anchor_cost(
            mesh=mesh,
            stage_id=stage_id,
            tile=mesh.tile_by_id(tile_id),
            traffic=traffic,
            placed_regions=placed_regions,
        )
        for tile_id in region
    )


def _future_feasible_after_choice(
    mesh: Mesh,
    allowed_tile_ids: set[int],
    chosen_tile_ids: set[int],
    remaining_tile_counts: dict[int, int],
    current_stage_remaining_tiles: int,
) -> bool:
    """Reject region choices that obviously strand future stages."""

    free_after_choice = allowed_tile_ids - chosen_tile_ids
    future_counts = list(remaining_tile_counts.values())
    if current_stage_remaining_tiles > 0:
        future_counts.append(current_stage_remaining_tiles)
    return _remaining_counts_fit_free_components(
        mesh=mesh,
        free_tile_ids=free_after_choice,
        remaining_tile_counts=tuple(sorted(future_counts, reverse=True)),
    )


def _remaining_counts_fit_free_components(
    mesh: Mesh,
    free_tile_ids: set[int],
    remaining_tile_counts: tuple[int, ...],
) -> bool:
    """Check that the remaining free region can still host the unfixed stages."""

    if not remaining_tile_counts:
        return True
    component_sizes = sorted(_free_component_sizes(mesh, free_tile_ids), reverse=True)
    requested_sizes = sorted(remaining_tile_counts, reverse=True)
    if sum(component_sizes) < sum(requested_sizes):
        return False
    if not component_sizes or requested_sizes[0] > component_sizes[0]:
        return False
    if len(requested_sizes) <= 3 and sum(requested_sizes) <= 20:
        return _can_partition_connected_regions(
            mesh=mesh,
            free_tile_ids=frozenset(free_tile_ids),
            remaining_tile_counts=requested_sizes,
            memo={},
        )
    return True


def _can_partition_connected_regions(
    mesh: Mesh,
    free_tile_ids: frozenset[int],
    remaining_tile_counts: list[int],
    memo: dict[tuple[frozenset[int], tuple[int, ...]], bool],
) -> bool:
    """Return whether the free region can be split into connected stage regions."""

    if not remaining_tile_counts:
        return True
    key = (free_tile_ids, tuple(remaining_tile_counts))
    cached = memo.get(key)
    if cached is not None:
        return cached

    tile_count = remaining_tile_counts[0]
    for region in _iter_connected_subsets_of_size(mesh, set(free_tile_ids), tile_count):
        if _can_partition_connected_regions(
            mesh=mesh,
            free_tile_ids=free_tile_ids - region,
            remaining_tile_counts=remaining_tile_counts[1:],
            memo=memo,
        ):
            memo[key] = True
            return True
    memo[key] = False
    return False


def _iter_connected_subsets_of_size(
    mesh: Mesh,
    tile_ids: set[int],
    tile_count: int,
) -> Iterator[frozenset[int]]:
    """Yield connected subsets without enumerating the full powerset."""

    if tile_count <= 0:
        yield frozenset()
        return

    emitted: set[frozenset[int]] = set()
    for seed_tile_id in sorted(tile_ids):
        regions = {frozenset({seed_tile_id})}
        for _ in range(1, tile_count):
            next_regions: set[frozenset[int]] = set()
            for region in regions:
                frontier = set()
                for tile_id in region:
                    frontier |= ((_neighbor_ids(mesh, tile_id) & tile_ids) - set(region))
                for tile_id in frontier:
                    next_regions.add(frozenset((*region, tile_id)))
            regions = next_regions
            if not regions:
                break
        for region in sorted(regions, key=lambda subset: tuple(sorted(subset))):
            if len(region) == tile_count and region not in emitted:
                emitted.add(region)
                yield region


def _assignable_reference_point(
    stage_id: int,
    virtual_tile_id: int,
    placements: dict[int, StagePlacement],
    assigned: dict[int, StagePlacement],
    stage_centers: dict[int, tuple[float, float]],
    traffic: VirtualTraffic,
    is_dst: bool,
) -> list[tuple[float, float, int]]:
    """Collect weighted reference points for one virtual tile."""

    points: list[tuple[float, float, int]] = []
    for (src_stage_id, dst_stage_id), matrix in traffic.edge_matrices.items():
        if is_dst:
            if dst_stage_id != stage_id:
                continue
            for (src_virtual_tile_id, dst_virtual_tile_candidate), bytes_ in matrix.items():
                if dst_virtual_tile_candidate != virtual_tile_id or bytes_ <= 0:
                    continue
                if src_stage_id in assigned:
                    src_tile_id = assigned[src_stage_id].physical_tile_id(src_virtual_tile_id)
                    src_tile = placements[src_stage_id].physical_submesh.mesh.tile_by_id(src_tile_id)
                    points.append((src_tile.x, src_tile.y, bytes_))
                else:
                    x, y = stage_centers[src_stage_id]
                    points.append((x, y, bytes_))
        else:
            if src_stage_id != stage_id:
                continue
            for (src_virtual_tile_candidate, dst_virtual_tile_id), bytes_ in matrix.items():
                if src_virtual_tile_candidate != virtual_tile_id or bytes_ <= 0:
                    continue
                if dst_stage_id in assigned:
                    dst_tile_id = assigned[dst_stage_id].physical_tile_id(dst_virtual_tile_id)
                    dst_tile = placements[dst_stage_id].physical_submesh.mesh.tile_by_id(dst_tile_id)
                    points.append((dst_tile.x, dst_tile.y, bytes_))
                else:
                    x, y = stage_centers[dst_stage_id]
                    points.append((x, y, bytes_))
    return points


def _virtual_assignment_cost(
    mesh: Mesh,
    stage_id: int,
    virtual_tile_id: int,
    physical_tile_id: int,
    placements: dict[int, StagePlacement],
    assigned: dict[int, StagePlacement],
    stage_centers: dict[int, tuple[float, float]],
    traffic: VirtualTraffic,
) -> float:
    """Score one virtual-to-physical ownership choice."""

    tile = mesh.tile_by_id(physical_tile_id)
    score = 0.0

    for x, y, bytes_ in _assignable_reference_point(
        stage_id=stage_id,
        virtual_tile_id=virtual_tile_id,
        placements=placements,
        assigned=assigned,
        stage_centers=stage_centers,
        traffic=traffic,
        is_dst=True,
    ):
        score += bytes_ * (abs(tile.x - x) + abs(tile.y - y))

    for x, y, bytes_ in _assignable_reference_point(
        stage_id=stage_id,
        virtual_tile_id=virtual_tile_id,
        placements=placements,
        assigned=assigned,
        stage_centers=stage_centers,
        traffic=traffic,
        is_dst=False,
    ):
        score += bytes_ * (abs(tile.x - x) + abs(tile.y - y))

    l2_access_points = tuple((mesh.tile_by_id(tile_id).x, mesh.tile_by_id(tile_id).y) for tile_id in _l2_access_point_tile_ids(mesh))
    if l2_access_points:
        l2_distance = min(abs(tile.x - x) + abs(tile.y - y) for x, y in l2_access_points)
        score += traffic.l2_read_weights.get(stage_id, {}).get(virtual_tile_id, 0) * l2_distance
        score += traffic.l2_write_weights.get(stage_id, {}).get(virtual_tile_id, 0) * l2_distance

    center_x, center_y = stage_centers[stage_id]
    score += 0.1 * (abs(tile.x - center_x) + abs(tile.y - center_y))
    return score


def _virtual_priority(
    stage_id: int,
    virtual_tile_id: int,
    traffic: VirtualTraffic,
) -> int:
    """Prioritize high-pressure virtual tiles first."""

    return max(
        traffic.input_weights.get(stage_id, {}).get(virtual_tile_id, 0),
        traffic.output_weights.get(stage_id, {}).get(virtual_tile_id, 0),
        traffic.l2_read_weights.get(stage_id, {}).get(virtual_tile_id, 0)
        + traffic.l2_write_weights.get(stage_id, {}).get(virtual_tile_id, 0),
    )


def _placements_from_regions(
    mesh: Mesh,
    stage_plans: dict[int, StagePlan],
    regions: dict[int, set[int]],
) -> dict[int, StagePlacement]:
    """Create stage placements from physical regions with identity ownership placeholders."""

    placements = {}
    for stage_id, region in regions.items():
        virtual_submesh = _stage_virtual_submesh(stage_plans[stage_id])
        physical_submesh = Submesh(mesh=mesh, submesh_id=stage_id, tile_ids=frozenset(region))
        placements[stage_id] = StagePlacement(
            stage_id=stage_id,
            virtual_submesh=virtual_submesh,
            physical_submesh=physical_submesh,
            virtual_to_physical=_identity_virtual_mapping(virtual_submesh, physical_submesh),
        )
    return placements


def _identity_virtual_mapping(
    virtual_submesh,
    physical_submesh: Submesh,
) -> dict[int, int]:
    """Build a stable one-to-one placeholder mapping before ownership refinement."""

    virtual_tile_ids = tuple(tile.tile_id for tile in virtual_submesh.tiles)
    physical_tile_ids = tuple(tile.tile_id for tile in physical_submesh.tiles)
    return dict(zip(virtual_tile_ids, physical_tile_ids))


def _zero_by_virtual_tile(virtual_submesh) -> dict[int, int]:
    """Build a zero-initialized per-virtual-tile counter."""

    return {tile.tile_id: 0 for tile in virtual_submesh.tiles}


def _tile_score_objective(
    tile_scores: dict[int, TileIOScore],
    k: int = 5,
) -> tuple[int, int, int, int]:
    """Return the lexicographic objective used during local repair."""

    sorted_scores = sorted((tile_score.score for tile_score in tile_scores.values()), reverse=True)
    return (
        sorted_scores[0] if sorted_scores else 0,
        sorted_scores[1] if len(sorted_scores) > 1 else 0,
        sum(sorted_scores[:k]),
        sum(sorted_scores),
    )


def _record_candidate(
    candidates: dict[frozenset[int], RepairCandidate],
    region: frozenset[int],
    priority: float,
    reason: str,
) -> None:
    """Keep the best reason/priority for one repair region."""

    if len(region) < 2:
        return
    existing = candidates.get(region)
    if existing is None or priority > existing.priority:
        candidates[region] = RepairCandidate(stages=region, priority=priority, reason=reason)


def _neighbor_stage_ids(
    mesh: Mesh,
    placements: dict[int, StagePlacement],
    stage_id: int,
) -> set[int]:
    """Return stages sharing at least one physical boundary edge with stage_id."""

    region = placements[stage_id].physical_submesh.tile_ids
    return {
        other_stage_id
        for other_stage_id, other_placement in placements.items()
        if other_stage_id != stage_id
        and _shared_boundary_length(mesh, region, other_placement.physical_submesh.tile_ids) > 0
    }


def _first_blocker_on_path(
    mesh: Mesh,
    placements: dict[int, StagePlacement],
    src_stage_id: int,
    dst_stage_id: int,
) -> int | None:
    """Return the first foreign stage on a short physical path between two stages."""

    owner_by_tile_id = _owner_by_tile_id(placements)
    src_tiles = placements[src_stage_id].physical_submesh.tile_ids
    dst_tiles = placements[dst_stage_id].physical_submesh.tile_ids
    path = _shortest_path_between_regions(mesh, src_tiles, dst_tiles)
    for tile_id in path[1:]:
        owner = owner_by_tile_id.get(tile_id)
        if owner is not None and owner not in {src_stage_id, dst_stage_id}:
            return owner
    return None


def _first_blocker_to_l2(
    mesh: Mesh,
    placements: dict[int, StagePlacement],
    stage_id: int,
) -> int | None:
    """Return the first foreign stage between one stage and the nearest L2 access point."""

    l2_points = _l2_access_point_tile_ids(mesh)
    if not l2_points:
        return None
    owner_by_tile_id = _owner_by_tile_id(placements)
    stage_tiles = placements[stage_id].physical_submesh.tile_ids
    path = _shortest_path_between_regions(mesh, stage_tiles, l2_points)
    for tile_id in path[1:]:
        owner = owner_by_tile_id.get(tile_id)
        if owner is not None and owner != stage_id:
            return owner
    return None


def _shortest_path_between_regions(
    mesh: Mesh,
    src_tile_ids: Iterable[int],
    dst_tile_ids: Iterable[int],
) -> tuple[int, ...]:
    """Return one shortest 4-neighbor path between two tile sets."""

    src_set = set(src_tile_ids)
    dst_set = set(dst_tile_ids)
    if src_set & dst_set:
        shared = min(src_set & dst_set)
        return (shared,)

    queue = deque(sorted(src_set))
    parent: dict[int, int | None] = {tile_id: None for tile_id in src_set}
    while queue:
        tile_id = queue.popleft()
        if tile_id in dst_set:
            break
        for neighbor_id in sorted(_neighbor_ids(mesh, tile_id)):
            if neighbor_id in parent:
                continue
            parent[neighbor_id] = tile_id
            queue.append(neighbor_id)

    reached = next((tile_id for tile_id in parent if tile_id in dst_set), None)
    if reached is None:
        return ()

    path = []
    cursor: int | None = reached
    while cursor is not None:
        path.append(cursor)
        cursor = parent[cursor]
    return tuple(reversed(path))


def _owner_by_tile_id(placements: dict[int, StagePlacement]) -> dict[int, int]:
    """Map each occupied tile to its owning stage."""

    return {
        tile_id: stage_id
        for stage_id, placement in placements.items()
        for tile_id in placement.physical_submesh.tile_ids
    }


def _shared_boundary_length(
    mesh: Mesh,
    left_tile_ids: Iterable[int],
    right_tile_ids: Iterable[int],
) -> int:
    """Count physical boundary contacts between two tile sets."""

    left = set(left_tile_ids)
    right = set(right_tile_ids)
    if not left or not right:
        return 0
    return sum(
        1
        for tile_id in left
        for neighbor_id in _neighbor_ids(mesh, tile_id)
        if neighbor_id in right
    )


def _tile_set_center(mesh: Mesh, tile_ids: Iterable[int]) -> tuple[float, float]:
    """Return the geometric center of one tile set."""

    tiles = [mesh.tile_by_id(tile_id) for tile_id in tile_ids]
    if not tiles:
        return (0.0, 0.0)
    return (
        sum(tile.x for tile in tiles) / len(tiles),
        sum(tile.y for tile in tiles) / len(tiles),
    )


def _region_compactness(mesh: Mesh, tile_ids: Iterable[int]) -> float:
    """Penalize stretched regions without forcing rectangles."""

    tiles = [mesh.tile_by_id(tile_id) for tile_id in tile_ids]
    if not tiles:
        return 0.0
    xs = [tile.x for tile in tiles]
    ys = [tile.y for tile in tiles]
    bbox_area = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
    return float(bbox_area - len(tiles))


def _future_space_penalty(
    mesh: Mesh,
    free_tile_ids: set[int],
    remaining_tile_counts: tuple[int, ...],
) -> float:
    """Softly penalize fragmented free space."""

    if not remaining_tile_counts:
        return 0.0
    component_sizes = sorted(_free_component_sizes(mesh, free_tile_ids), reverse=True)
    if not component_sizes:
        return 1_000_000.0
    penalty = 0.0
    if sum(component_sizes) < sum(remaining_tile_counts):
        penalty += 1_000_000.0
    if remaining_tile_counts[0] > component_sizes[0]:
        penalty += 1_000_000.0
    penalty += 10.0 * max(0, len(component_sizes) - 1)
    return penalty


def _free_component_sizes(mesh: Mesh, free_tile_ids: set[int]) -> tuple[int, ...]:
    """Return sizes of the connected components inside free_tile_ids."""

    seen: set[int] = set()
    sizes: list[int] = []
    for start in sorted(free_tile_ids):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            tile_id = stack.pop()
            size += 1
            for neighbor_id in _neighbor_ids(mesh, tile_id):
                if neighbor_id in free_tile_ids and neighbor_id not in seen:
                    seen.add(neighbor_id)
                    stack.append(neighbor_id)
        sizes.append(size)
    return tuple(sizes)


def _neighbor_ids(mesh: Mesh, tile_id: int) -> set[int]:
    """Return 4-neighbor tile ids."""

    tile = mesh.tile_by_id(tile_id)
    neighbors: set[int] = set()
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        x = tile.x + dx
        y = tile.y + dy
        if mesh.contains_coord(x, y):
            neighbors.add(mesh.tile_id(x, y))
    return neighbors


def _l2_access_point_tile_ids(mesh: Mesh) -> set[int]:
    """Return tiles that share a NoC node with an L2 endpoint."""

    l1_endpoints = tuple(
        endpoint
        for endpoint in mesh.noc.endpoints
        if endpoint.kind is EndpointKind.L1 and endpoint.tile_id is not None
    )
    return {
        endpoint.tile_id
        for l2_endpoint in mesh.noc.endpoints_of_kind(EndpointKind.L2)
        for endpoint in l1_endpoints
        if endpoint.node_id == l2_endpoint.node_id
    }


def _stage_virtual_submesh(plan: StagePlan):
    """Return the rectangular virtual submesh that drives stage computation."""

    if plan.node_output_layouts:
        for layouts in plan.node_output_layouts:
            if layouts:
                return layouts[0].submesh
    if plan.output_layouts:
        return plan.output_layouts[0].submesh
    raise ValueError(f"stage {plan.stage_id} has no virtual layouts")


def _stage_node_output_layout(
    plan: StagePlan,
    node: Node,
) -> tuple[TensorLayout, ...]:
    """Return output layouts for one node inside a stage plan."""

    if plan.node_output_layouts:
        return plan.node_output_layouts[_plan_node_index(plan, node)]
    return plan.output_layouts


def _plan_node_index(plan: StagePlan, node: Node) -> int:
    """Return the position of one node inside a stage plan."""

    for node_idx, candidate in enumerate(plan.nodes):
        if candidate is node:
            return node_idx
    raise ValueError(f"node {node.name} is not present in stage plan {plan.stage_id}")


def _transition_required_slices(
    tensor: Tensor,
    dst_node: Node,
    dst_output_layouts: tuple[TensorLayout, ...],
) -> tuple[tuple[Tile, TensorSlice], ...]:
    """Return the consumer slices requested by each destination virtual tile."""

    required_slices = []
    for tile in dst_output_layouts[0].submesh.tiles:
        tile_work = dst_node.payload.build_tile_work(output_layouts=dst_output_layouts, tile=tile)
        for ref in tile_work.input_slices:
            if ref.tensor is tensor:
                required_slices.append((tile, ref.tensor_slice))
                break
    return tuple(required_slices)


def _node_output_index(node: Node, tensor: Tensor) -> int:
    """Return the output index for tensor inside node.outputs."""

    for output_idx, candidate in enumerate(node.outputs):
        if candidate == tensor:
            return output_idx
    raise ValueError(f"tensor {tensor.name} is not an output of node {node.name}")


def _print_placement_grid(mesh: Mesh, placements: dict[int, StagePlacement]) -> None:
    """Print a compact grid showing physical stage ownership."""

    owner_by_tile_id = _owner_by_tile_id(placements)
    cell_width = max(1, *(len(str(stage_id)) for stage_id in placements))
    print("Spatial mapping mesh:")
    for y in range(mesh.height):
        cells = []
        for x in range(mesh.width):
            tile_id = mesh.tile_id(x, y)
            owner = owner_by_tile_id.get(tile_id)
            cell = "." if owner is None else str(owner)
            cells.append(cell.rjust(cell_width))
        print(" ".join(cells))


def _stage_name(stage_nodes: tuple[Node, ...]) -> str:
    """Return a compact display name for one selected stage."""

    return "+".join(node.name for node in stage_nodes)


def _remaining_counts_tuple(remaining_tile_counts: dict[int, int]) -> tuple[int, ...]:
    """Normalize remaining tile counts for feasibility helpers."""

    return tuple(sorted(remaining_tile_counts.values(), reverse=True))


def _tile_to_point_distance(tile: Tile, point: tuple[float, float]) -> float:
    """Return Manhattan distance from a tile to a floating-point target."""

    return abs(tile.x - point[0]) + abs(tile.y - point[1])


def _debug(enabled: bool, message: str) -> None:
    """Print a debug message when requested."""

    if enabled:
        print(message)
