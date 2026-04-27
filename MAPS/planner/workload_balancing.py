"""Workload-balancing layer for stage-to-tile-count allocation."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.planner.cost import cost_estimator
from MAPS.arch import Mesh
from MAPS.core.graph import Graph, Node
from MAPS.core.submesh import Submesh
from MAPS.planner.select_stage import StageSelection


@dataclass(frozen=True)
class StagePlan:
    """Compute-layout decisions chosen before physical placement."""

    stage_id: int
    tile_count: int
    logical_shape: tuple[int, int]
    output_layouts: tuple
    nodes: tuple[Node, ...] = ()
    node_output_layouts: tuple[tuple, ...] = ()


def balance_workload(
    graph: Graph,
    mesh: Mesh,
    debug: bool = False,
    stage_selection: StageSelection | None = None,
) -> dict[int, StagePlan]:
    """Greedily assign tile counts to selected stages by internal rectangle growth.

    Each selected stage searches physically placeable tile counts. For one tile
    count, the stage uses one canonical logical shape matching the planning
    submesh shape; later spatial mapping still chooses the concrete physical
    rectangle placement.
    """
    resolved_stage_selection = _resolve_stage_selection(graph, stage_selection)
    stage_ids = tuple(resolved_stage_selection)
    if len(stage_ids) > mesh.num_tiles:
        raise ValueError("graph has more selected stages than available tiles")

    iteration = 0
    placement_masks_by_tile_count: dict[int, tuple[int, ...]] = {}
    placement_feasibility_cache: dict[tuple[tuple[int, int], ...], bool] = {}
    decision_timeline: list[tuple[int, int | None, dict[int, int], dict[int, int]]] = []
    growth_horizon_by_stage: dict[int, int] = {}

    # Seed each stage with the smallest tile count whose tile work fits in L1,
    # then spend any remaining tiles greedily on workload reduction.
    tile_counts: dict[int, int] = {}
    _debug(debug, "[workload_balancing] phase=initial_l1_seeding")
    for stage_id in stage_ids:
        stage_nodes = resolved_stage_selection[stage_id]
        _debug(
            debug,
            f"[workload_balancing] seed stage={stage_id} nodes={_stage_label(stage_nodes)}",
        )
        for tile_count in range(1, mesh.num_tiles + 1):

            # skip tile counts that cannot be represented as a rectangle
            if not _physical_shape_options(tile_count, mesh):
                _debug(
                    debug,
                    "[workload_balancing] "
                    f"seed stage={stage_id} tile_count={tile_count} "
                    "skip=no_physical_rectangle",
                )
                continue

            try:
                plan = _best_stage_plan_for_stage_nodes(
                    stage_nodes,
                    mesh,
                    stage_id,
                    tile_count,
                    debug=False,
                )
            except ValueError as exc:
                _debug(
                    debug,
                    "[workload_balancing] "
                    f"seed stage={stage_id} tile_count={tile_count} "
                    f"skip={exc}",
                )
                continue
            tile_counts[stage_id] = tile_count
            growth_horizon_by_stage[stage_id] = 1
            _debug(
                debug,
                "[workload_balancing] "
                f"seed stage={stage_id} choose tile_count={tile_count} "
                f"logical_shape={plan.logical_shape}",
            )
            break

        if stage_id not in tile_counts:
            raise ValueError(
                "stage "
                f"{stage_id} ({_stage_label(stage_nodes)}) has no L1-feasible tile count "
                f"on mesh {mesh.shape}"
            )
    used_tiles = sum(tile_counts.values())

    if used_tiles > mesh.num_tiles:
        raise ValueError("minimum L1-feasible tile counts exceed available tiles")
        
    _debug(debug, "[workload_balancing] phase=initial_placement_check")
    if not _has_feasible_submesh_placement(
        tile_counts,
        mesh,
        placement_masks_by_tile_count=placement_masks_by_tile_count,
        feasibility_cache=placement_feasibility_cache,
    ):
        raise ValueError("minimum L1-feasible tile counts cannot be placed without overlap")

    _debug(debug, f"[workload_balancing] start used_tiles={used_tiles}/{mesh.num_tiles}")
    _debug(debug, f"[workload_balancing] initial_tile_counts={tile_counts}")
    initial_plans = _plan_all_stages_for_tile_counts(
        resolved_stage_selection,
        mesh,
        tile_counts,
        debug=debug,
    )
    initial_workloads = _estimate_workloads(
        initial_plans,
        resolved_stage_selection,
        debug=debug,
    )
    decision_timeline.append((0, None, dict(tile_counts), dict(initial_workloads)))

    _debug(debug, "[workload_balancing] phase=greedy_growth")
    while used_tiles < mesh.num_tiles:
        iteration += 1
        worst_stage_id: int | None = None
        worst_stage_tile_count: int | None = None
        worst_stage_improvement = 0

        # Rebuild stage plans for the current allocation so candidate growth is
        # compared against the current canonical logical shape at each tile count.
        current_plans = _plan_all_stages_for_tile_counts(
            resolved_stage_selection,
            mesh,
            tile_counts,
            debug=False,
        )
        current_workloads = _estimate_workloads(
            current_plans,
            resolved_stage_selection,
            debug=debug,
        )

        _debug(debug, f"[workload_balancing] iteration={iteration} used_tiles={used_tiles}/{mesh.num_tiles}")
        _debug(debug, f"[workload_balancing] current_workloads={_format_debug_workloads(current_workloads)}")

        # Try the current bottleneck first. Fall through to lower-workload
        # stages only when worse stages have no feasible improving growth.
        stage_order = tuple(
            sorted(
                stage_ids,
                key=lambda stage_id: (-current_workloads[stage_id], stage_id),
            )
        )
        _debug(debug, f"[workload_balancing] stage_order_by_workload={stage_order}")

        for stage_id in stage_order:
            stage_nodes = resolved_stage_selection[stage_id]
            current_tile_count = tile_counts[stage_id]
            current_workload = current_workloads[stage_id]

            _debug(
                debug,
                "[workload_balancing] "
                f"try_stage={stage_id} nodes={_stage_label(stage_nodes)} "
                f"current_tile_count={current_tile_count} "
                f"current_logical_shape={current_plans[stage_id].logical_shape} "
                f"current_workload={_format_debug_cost(current_workload)}",
            )

            candidate_tile_count_options = _tile_count_options_after_growth(
                current_tile_count,
                mesh.num_tiles - used_tiles,
                mesh,
                max_added_tiles=growth_horizon_by_stage[stage_id],
            )
            _debug(
                debug,
                "[workload_balancing] "
                f"stage={stage_id} candidate_tile_counts={candidate_tile_count_options}",
            )
            best_growth = _best_growth_tile_count_for_stage(
                stage_id=stage_id,
                stage_selection=resolved_stage_selection,
                mesh=mesh,
                tile_counts=tile_counts,
                used_tiles=used_tiles,
                current_workload=current_workload,
                placement_masks_by_tile_count=placement_masks_by_tile_count,
                placement_feasibility_cache=placement_feasibility_cache,
                max_added_tiles=growth_horizon_by_stage[stage_id],
                debug=debug,
            )
            growth_horizon_by_stage[stage_id] += 1
            if best_growth is not None:
                worst_stage_id = stage_id
                worst_stage_tile_count, worst_stage_improvement = best_growth

            if worst_stage_id is None:
                _debug(debug, f"[workload_balancing] stage={stage_id} no_valid_growth")
                continue

            break

        if worst_stage_id is None or worst_stage_tile_count is None:
            _debug(debug, "[workload_balancing] no_global_improvement_available")
            break

        # Commit one growth step, then recompute all plans/workloads next round.
        _debug(
            debug,
            "[workload_balancing] "
            f"choose worst_stage={worst_stage_id} "
            f"new_tile_count={worst_stage_tile_count} "
            f"improvement={_format_debug_cost(worst_stage_improvement)}",
        )
        used_tiles += worst_stage_tile_count - tile_counts[worst_stage_id]
        tile_counts[worst_stage_id] = worst_stage_tile_count
        updated_plans = _plan_all_stages_for_tile_counts(
            resolved_stage_selection,
            mesh,
            tile_counts,
            debug=False,
        )
        updated_workloads = _estimate_workloads(
            updated_plans,
            resolved_stage_selection,
            debug=debug,
        )
        decision_timeline.append((iteration, worst_stage_id, dict(tile_counts), dict(updated_workloads)))
        _debug(debug, f"[workload_balancing] updated_tile_counts={tile_counts}")

    # Materialize final StagePlan objects, including the chosen logical layouts.
    _debug(debug, "[workload_balancing] phase=finalize")
    plans = _plan_all_stages_for_tile_counts(
        resolved_stage_selection,
        mesh,
        tile_counts,
        debug=False,
    )
    _debug(debug, f"[workload_balancing] final_tile_counts={tile_counts}")
    _debug(debug, f"[workload_balancing] final_logical_shapes={ {stage_id: plan.logical_shape for stage_id, plan in plans.items()} }")
    _debug(debug, f"[workload_balancing] final_allocation={ {stage_id: plan.tile_count for stage_id, plan in plans.items()} }")
    _debug_decision_timeline(debug, resolved_stage_selection, decision_timeline)
    _debug_final_workloads(debug, resolved_stage_selection, plans)
    return plans


def _resolve_stage_selection(
    graph: Graph,
    stage_selection: StageSelection | None,
) -> StageSelection:
    """Validate and normalize selected stage groups."""

    if stage_selection is None:
        return {
            stage_id: (node,)
            for stage_id, node in enumerate(graph.nodes)
        }

    graph_nodes_by_identity = {
        id(node): node
        for node in graph.nodes
    }
    seen_node_ids: set[int] = set()
    resolved: StageSelection = {}

    for stage_id, stage_nodes in stage_selection.items():
        if not stage_nodes:
            raise ValueError(f"stage {stage_id} must contain at least one node")
        for node in stage_nodes:
            node_identity = id(node)
            if node_identity not in graph_nodes_by_identity:
                raise ValueError(
                    f"stage {stage_id} contains node {node.name} not present in graph {graph.name}"
                )
            if node_identity in seen_node_ids:
                raise ValueError(
                    f"node {node.name} appears in more than one selected stage"
                )
            seen_node_ids.add(node_identity)
        resolved[stage_id] = tuple(stage_nodes)

    if len(seen_node_ids) != len(graph.nodes):
        missing = tuple(
            node.name
            for node in graph.nodes
            if id(node) not in seen_node_ids
        )
        raise ValueError(
            f"selected stages do not cover all graph nodes, missing={missing}"
        )

    return resolved


def _tile_count_options_after_growth(
    current_tile_count: int,
    remaining_tiles: int,
    mesh: Mesh,
    max_added_tiles: int,
) -> tuple[int, ...]:
    """Return growth candidates that fit the mesh and the current growth horizon."""
    if max_added_tiles <= 0:
        raise ValueError("max_added_tiles must be > 0")
    max_tile_count = min(
        current_tile_count + max_added_tiles,
        current_tile_count + remaining_tiles,
    )
    options = []
    for tile_count in range(current_tile_count + 1, max_tile_count + 1):
        if not _physical_shape_options(tile_count, mesh):
            continue
        options.append(tile_count)
    return tuple(options)


def _best_growth_tile_count_for_stage(
    stage_id: int,
    stage_selection: StageSelection,
    mesh: Mesh,
    tile_counts: dict[int, int],
    used_tiles: int,
    current_workload: int,
    placement_masks_by_tile_count: dict[int, tuple[int, ...]],
    placement_feasibility_cache: dict[tuple[tuple[int, int], ...], bool],
    max_added_tiles: int,
    debug: bool,
) -> tuple[int, int] | None:
    """Return the best feasible growth tile count for one stage.

    Growth candidates must still improve workload and preserve global rectangle
    packability. Among valid candidates, prefer the tile count with the most
    possible physical rectangle shapes; break ties by larger workload
    improvement, then smaller tile count.
    """

    current_tile_count = tile_counts[stage_id]
    candidate_tile_count_options = _tile_count_options_after_growth(
        current_tile_count,
        mesh.num_tiles - used_tiles,
        mesh,
        max_added_tiles=max_added_tiles,
    )

    best_candidate_tile_count: int | None = None
    best_candidate_improvement = 0
    best_candidate_shape_count = -1

    for candidate_tile_count in candidate_tile_count_options:
        added_tiles = candidate_tile_count - current_tile_count
        if used_tiles + added_tiles > mesh.num_tiles:
            _debug(
                debug,
                "[workload_balancing] "
                f"stage={stage_id} candidate_tile_count={candidate_tile_count} "
                "skip=tile_budget_exceeded",
            )
            continue

        candidate_tile_counts = dict(tile_counts)
        candidate_tile_counts[stage_id] = candidate_tile_count

        if not _has_feasible_submesh_placement(
            candidate_tile_counts,
            mesh,
            placement_masks_by_tile_count=placement_masks_by_tile_count,
            feasibility_cache=placement_feasibility_cache,
        ):
            _debug(
                debug,
                "[workload_balancing] "
                f"stage={stage_id} candidate_tile_count={candidate_tile_count} "
                "no_feasible_global_placement",
            )
            continue

        candidate_plans = _plan_all_stages_for_tile_counts(
            stage_selection,
            mesh,
            candidate_tile_counts,
            debug=False,
        )
        candidate_workloads = _estimate_workloads(
            candidate_plans,
            stage_selection,
            debug=debug,
        )
        candidate_workload = candidate_workloads[stage_id]
        improvement = current_workload - candidate_workload
        if improvement <= 0:
            _debug(
                debug,
                "[workload_balancing] "
                f"stage={stage_id} candidate_tile_count={candidate_tile_count} "
                f"skip=no_workload_improvement "
                f"candidate_workload={_format_debug_cost(candidate_workload)} "
                f"current_workload={_format_debug_cost(current_workload)}",
            )
            continue

        shape_count = _physical_shape_configuration_count(candidate_tile_count, mesh)
        _debug(
            debug,
            "[workload_balancing] "
            f"stage={stage_id} candidate_tile_count={candidate_tile_count} "
            f"accepted_improvement={_format_debug_cost(improvement)} "
            f"candidate_workload={_format_debug_cost(candidate_workload)} "
            f"shape_count={shape_count}",
        )
        if (
            best_candidate_tile_count is None
            or shape_count > best_candidate_shape_count
            or (
                shape_count == best_candidate_shape_count
                and improvement > best_candidate_improvement
            )
            or (
                shape_count == best_candidate_shape_count
                and improvement == best_candidate_improvement
                and candidate_tile_count < best_candidate_tile_count
            )
        ):
            best_candidate_tile_count = candidate_tile_count
            best_candidate_improvement = improvement
            best_candidate_shape_count = shape_count

    if best_candidate_tile_count is None:
        return None
    return best_candidate_tile_count, best_candidate_improvement


def _physical_shape_options(tile_count: int, mesh: Mesh) -> tuple[tuple[int, int], ...]:
    """Return rectangular physical shapes with tile_count area that fit the mesh."""
    options = []
    for height in range(1, mesh.height + 1):
        if tile_count % height != 0:
            continue
        width = tile_count // height
        if 0 < width <= mesh.width:
            options.append((width, height))
    return tuple(options)


def _physical_shape_configuration_count(tile_count: int, mesh: Mesh) -> int:
    """Return how many distinct physical rectangle shapes fit one tile count."""
    return len(_physical_shape_options(tile_count, mesh))


def _has_feasible_submesh_placement(
    tile_counts: dict[int, int],
    mesh: Mesh,
    placement_masks_by_tile_count: dict[int, tuple[int, ...]],
    feasibility_cache: dict[tuple[tuple[int, int], ...], bool],
) -> bool:
    """Return whether all stage tile counts can be packed as disjoint rectangles."""
    key = tuple(sorted(tile_counts.items()))
    cached = feasibility_cache.get(key)
    if cached is not None:
        return cached

    if sum(tile_counts.values()) > mesh.num_tiles:
        feasibility_cache[key] = False
        return False

    placement_masks = {
        stage_id: _placement_masks_for_tile_count(
            tile_count,
            mesh,
            placement_masks_by_tile_count,
        )
        for stage_id, tile_count in tile_counts.items()
    }
    if any(not masks for masks in placement_masks.values()):
        feasibility_cache[key] = False
        return False

    stage_order = tuple(
        sorted(
            tile_counts,
            key=lambda stage_id: (
                len(placement_masks[stage_id]),
                -tile_counts[stage_id],
                stage_id,
            ),
        )
    )
    feasible = _can_pack_stage_masks(stage_order, placement_masks, index=0, occupied=0)
    feasibility_cache[key] = feasible
    return feasible


def _placement_masks_for_tile_count(
    tile_count: int,
    mesh: Mesh,
    placement_masks_by_tile_count: dict[int, tuple[int, ...]],
) -> tuple[int, ...]:
    """Return cached tile masks for every rectangular placement of one tile count."""
    cached = placement_masks_by_tile_count.get(tile_count)
    if cached is not None:
        return cached

    masks = []
    for width, height in _physical_shape_options(tile_count, mesh):
        for x0 in range(mesh.width - width + 1):
            for y0 in range(mesh.height - height + 1):
                masks.append(
                    Submesh(
                        mesh=mesh,
                        submesh_id=0,
                        x0=x0,
                        y0=y0,
                        width=width,
                        height=height,
                    ).tile_mask
                )

    unique_masks = tuple(dict.fromkeys(masks))
    placement_masks_by_tile_count[tile_count] = unique_masks
    return unique_masks


def _can_pack_stage_masks(
    stage_order: tuple[int, ...],
    placement_masks: dict[int, tuple[int, ...]],
    index: int,
    occupied: int,
) -> bool:
    """Backtrack over placement masks to find one non-overlapping assignment."""
    if index == len(stage_order):
        return True

    stage_id = stage_order[index]
    for mask in placement_masks[stage_id]:
        if occupied & mask:
            continue
        if _can_pack_stage_masks(
            stage_order,
            placement_masks,
            index=index + 1,
            occupied=occupied | mask,
        ):
            return True
    return False


def _planning_submesh(mesh: Mesh, stage_id: int, tile_count: int) -> Submesh:
    """Build a representative submesh used only to derive candidate layouts."""
    physical_options = _physical_shape_options(tile_count, mesh)
    if not physical_options:
        raise ValueError(f"no rectangular shape fits tile_count={tile_count} on mesh {mesh.shape}")
    width, height = physical_options[0]
    return Submesh(
        mesh=mesh,
        submesh_id=stage_id,
        x0=0,
        y0=0,
        width=width,
        height=height,
    )


def _layouts_for_node(
    node: Node,
    submesh: Submesh,
    logical_shape: tuple[int, int],
) -> tuple:
    """Return the op output layouts for one logical shape."""
    return node.payload.output_layouts(
            submesh,
            logical_shape=logical_shape,
    )


def _plan_all_stages_for_tile_counts(
    stage_selection: StageSelection,
    mesh: Mesh,
    tile_counts: dict[int, int],
    debug: bool,
) -> dict[int, StagePlan]:
    """Build the best layout-preserving plan for each current stage tile count."""
    plans: dict[int, StagePlan] = {}
    for stage_id, stage_nodes in stage_selection.items():
        plans[stage_id] = _best_stage_plan_for_stage_nodes(
            stage_nodes,
            mesh,
            stage_id,
            tile_counts[stage_id],
            debug=debug,
        )
    return plans


def _layouts_for_stage_nodes(
    stage_nodes: tuple[Node, ...],
    submesh: Submesh,
    logical_shape: tuple[int, int],
) -> tuple[tuple, ...]:
    """Return output layouts for every node inside one selected stage."""

    node_output_layouts = []
    for node in stage_nodes:
        output_layouts = _layouts_for_node(
            node,
            submesh,
            logical_shape,
        )
        node_output_layouts.append(output_layouts)
    return tuple(node_output_layouts)


def _best_stage_plan_for_stage_nodes(
    stage_nodes: tuple[Node, ...],
    mesh: Mesh,
    stage_id: int,
    tile_count: int,
    debug: bool,
) -> StagePlan:
    """Build the stage plan for one selected multi-node stage."""

    submesh = _planning_submesh(mesh, stage_id, tile_count)
    logical_shape = (submesh.width, submesh.height)
    node_output_layouts = _layouts_for_stage_nodes(
        stage_nodes,
        submesh,
        logical_shape,
    )
    fits_l1 = True
    peak_l1_bytes = 0
    min_l1_capacity = None
    for tile in submesh.tiles:
        tile_peak_l1_bytes = 0
        for node, output_layouts in zip(
            stage_nodes,
            node_output_layouts,
        ):
            tile_work = node.payload.build_tile_work(
                output_layouts=output_layouts,
                tile=tile,
            )
            tile_peak_l1_bytes = max(tile_peak_l1_bytes, tile_work.l1_bytes)
            if not tile_work.fits_l1(tile):
                fits_l1 = False
                break
        peak_l1_bytes = max(peak_l1_bytes, tile_peak_l1_bytes)
        min_l1_capacity = (
            tile.memory.size
            if min_l1_capacity is None
            else min(min_l1_capacity, tile.memory.size)
        )
        if not fits_l1:
            break

    if not fits_l1:
        _debug(
            debug,
            "[workload_balancing] "
            f"stage={stage_id} tile_count={tile_count} "
            f"logical_shape={logical_shape} skip=l1_exceeded "
            f"peak_l1_bytes={peak_l1_bytes} min_l1_capacity={min_l1_capacity}",
        )
        raise ValueError(
            f"stage {_stage_label(stage_nodes)} has no valid logical shape for tile_count={tile_count} "
            "using full tile-work slices"
        )

    plan = StagePlan(
        stage_id=stage_id,
        tile_count=tile_count,
        logical_shape=logical_shape,
        output_layouts=node_output_layouts[-1],
        nodes=stage_nodes,
        node_output_layouts=node_output_layouts,
    )
    workload = _estimate_stage_group_workload(stage_nodes, plan)
    _debug(
        debug,
        "[workload_balancing] "
        f"stage={stage_id} tile_count={tile_count} "
        f"logical_shape={logical_shape} workload={_format_debug_cost(workload)}",
    )
    return plan


def _best_stage_plan_for_tile_count(
    node: Node,
    mesh: Mesh,
    stage_id: int,
    tile_count: int,
    debug: bool,
) -> StagePlan:
    """Choose the lowest-cost logical layout that fits per-tile L1 memory."""
    return _best_stage_plan_for_stage_nodes(
        (node,),
        mesh,
        stage_id,
        tile_count,
        debug,
    )


def _estimate_workloads(
    plans: dict[int, StagePlan],
    stage_selection: StageSelection,
    debug: bool,
) -> dict[int, int]:
    """Estimate per-stage workload for the current stage plans."""
    del debug
    workloads: dict[int, int] = {}
    for stage_id, stage_nodes in stage_selection.items():
        workloads[stage_id] = _estimate_stage_group_workload(
            stage_nodes,
            plans[stage_id],
        )
    return workloads


def _estimate_stage_workload(node: Node, plan: StagePlan) -> int:
    """Estimate compute cost for one node under one stage plan."""
    return _estimate_stage_group_workload((node,), plan)


def _estimate_stage_group_workload(
    stage_nodes: tuple[Node, ...],
    plan: StagePlan,
) -> int:
    """Estimate one selected stage workload as the sum of member node costs."""

    if plan.node_output_layouts:
        node_output_layouts = plan.node_output_layouts
    else:
        node_output_layouts = (plan.output_layouts,)

    total_cost = 0
    for node, output_layouts in zip(
        stage_nodes,
        node_output_layouts,
    ):
        total_cost += cost_estimator(
            node=node,
            output_layouts=output_layouts,
        )
    return total_cost


def _debug(enabled: bool, message: str) -> None:
    """Print a workload-balancing debug message when enabled."""
    if enabled:
        print(message)


def _format_debug_cost(value: int) -> str:
    """Format one reported cycle cost."""
    return str(value)


def _format_debug_workloads(workloads: dict[int, int]) -> str:
    """Format a stage-to-workload dictionary for debug printing."""
    return "{" + ", ".join(
        f"{stage_id}: {_format_debug_cost(workload)}"
        for stage_id, workload in workloads.items()
    ) + "}"


def _debug_final_workloads(
    enabled: bool,
    stage_selection: StageSelection,
    plans: dict[int, StagePlan],
) -> None:
    """Print the final per-stage workload table when debug output is enabled."""
    if not enabled:
        return

    print("[workload_balancing] final_stage_workloads:")
    for stage_id, stage_nodes in stage_selection.items():
        plan = plans[stage_id]
        workload = _estimate_stage_group_workload(stage_nodes, plan)
        print(
            "  "
            f"stage={stage_id} nodes={_stage_label(stage_nodes)} "
            f"tile_count={plan.tile_count} "
            f"logical_shape={plan.logical_shape} "
            f"workload={_format_debug_cost(workload)}"
        )


def _debug_decision_timeline(
    enabled: bool,
    stage_selection: StageSelection,
    decision_timeline: list[tuple[int, int | None, dict[int, int], dict[int, int]]],
) -> None:
    """Print tile counts and workloads after each committed greedy decision."""
    if not enabled:
        return

    stage_headers = " ".join(
        f"stage{stage_id}:{_stage_label(stage_nodes)}"
        for stage_id, stage_nodes in stage_selection.items()
    )
    print("[workload_balancing] decision_timeline:")
    print(f"  stages {stage_headers}")
    for iteration, changed_stage_id, tile_counts, workloads in decision_timeline:
        changed = "initial" if changed_stage_id is None else f"stage={changed_stage_id}"
        stage_state = " ".join(
            (
                f"stage={stage_id} "
                f"tile_count={tile_counts[stage_id]} "
                f"workload={_format_debug_cost(workloads[stage_id])}"
            )
            for stage_id in stage_selection
        )
        print(f"  iter={iteration} changed={changed} {stage_state}")


def _stage_label(stage_nodes: tuple[Node, ...]) -> str:
    """Return a compact debug label for one selected stage."""

    return "+".join(node.name for node in stage_nodes)
