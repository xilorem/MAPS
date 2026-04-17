"""Workload-balancing layer for stage-to-tile-count allocation."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.cost_models import cost_estimator
from MAPS.arch import Mesh
from MAPS.core.graph import Graph, Node, OpKind
from MAPS.core.submesh import Submesh


@dataclass(frozen=True)
class StagePlan:
    """Compute-layout decisions chosen before physical placement."""

    stage_id: int
    tile_count: int
    logical_shape: tuple[int, int]
    input_layouts: tuple
    output_layouts: tuple


def balance_workload(graph: Graph, mesh: Mesh, debug: bool = False) -> dict[int, StagePlan]:
    """Greedily assign tile counts to nodes by internal rectangle growth.

    Each node searches physically placeable tile counts and all logical factor
    pairs for that count. The returned plan keeps the chosen logical compute
    shape and layouts; later spatial mapping still chooses physical rectangles.
    """
    if len(graph.nodes) > mesh.num_tiles:
        raise ValueError("graph has more nodes than available tiles")

    iteration = 0
    placement_masks_by_tile_count: dict[int, tuple[int, ...]] = {}
    placement_feasibility_cache: dict[tuple[tuple[int, int], ...], bool] = {}
    decision_timeline: list[tuple[int, int | None, dict[int, int], dict[int, float]]] = []

    # Seed each stage with the smallest tile count whose tile work fits in L1,
    # then spend any remaining tiles greedily on workload reduction.
    tile_counts: dict[int, int] = {}
    _debug(debug, "[workload_balancing] phase=initial_l1_seeding")
    for stage_id, node in enumerate(graph.nodes):
        _debug(debug, f"[workload_balancing] seed stage={stage_id} node={node.name}")
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
                plan = _best_stage_plan_for_tile_count(
                    node,
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
            _debug(
                debug,
                "[workload_balancing] "
                f"seed stage={stage_id} choose tile_count={tile_count} "
                f"logical_shape={plan.logical_shape}",
            )
            break

        if stage_id not in tile_counts:
            raise ValueError(f"node {node.name} has no L1-feasible tile count on mesh {mesh.shape}")
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
    initial_plans = _plan_all_stages_for_tile_counts(graph, mesh, tile_counts, debug=debug)
    initial_workloads = _estimate_workloads(initial_plans, graph, debug=debug)
    decision_timeline.append((0, None, dict(tile_counts), dict(initial_workloads)))

    _debug(debug, "[workload_balancing] phase=greedy_growth")
    while used_tiles < mesh.num_tiles:
        iteration += 1
        worst_stage_id: int | None = None
        worst_stage_tile_count: int | None = None
        worst_stage_improvement = 0.0

        # Rebuild stage plans for the current allocation so candidate growth is
        # compared against the best logical shape available at each tile count.
        current_plans = _plan_all_stages_for_tile_counts(graph, mesh, tile_counts, debug=False)
        current_workloads = _estimate_workloads(current_plans, graph, debug=debug)

        _debug(debug, f"[workload_balancing] iteration={iteration} used_tiles={used_tiles}/{mesh.num_tiles}")
        _debug(debug, f"[workload_balancing] current_workloads={current_workloads}")

        # Try the current bottleneck first. Fall through to lower-workload
        # stages only when worse stages have no feasible improving growth.
        stage_order = tuple(
            sorted(
                range(len(graph.nodes)),
                key=lambda stage_id: (-current_workloads[stage_id], stage_id),
            )
        )
        _debug(debug, f"[workload_balancing] stage_order_by_workload={stage_order}")

        for stage_id in stage_order:
            node = graph.nodes[stage_id]
            current_tile_count = tile_counts[stage_id]
            current_workload = current_workloads[stage_id]

            _debug(
                debug,
                "[workload_balancing] "
                f"try_stage={stage_id} node={node.name} "
                f"current_tile_count={current_tile_count} "
                f"current_logical_shape={current_plans[stage_id].logical_shape} "
                f"current_workload={current_workload}",
            )

            candidate_tile_count_options = _tile_count_options_after_growth(
                current_tile_count,
                mesh.num_tiles - used_tiles,
                mesh,
            )
            _debug(
                debug,
                "[workload_balancing] "
                f"stage={stage_id} candidate_tile_counts={candidate_tile_count_options}",
            )

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

                # Reject local improvements that would make the final stage
                # rectangles impossible to place without overlap.
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
                    graph,
                    mesh,
                    candidate_tile_counts,
                    debug=False,
                )
                candidate_plan = candidate_plans[stage_id]
                candidate_workloads = _estimate_workloads(candidate_plans, graph, debug=debug)
                candidate_workload = candidate_workloads[stage_id]
                improvement = current_workload - candidate_workload
                if improvement <= 0:
                    _debug(
                        debug,
                        "[workload_balancing] "
                        f"stage={stage_id} candidate_tile_count={candidate_tile_count} "
                        f"skip=no_workload_improvement "
                        f"candidate_workload={candidate_workload} "
                        f"current_workload={current_workload}",
                    )
                    continue
                _debug(
                    debug,
                    "[workload_balancing] "
                    f"stage={stage_id} candidate_tile_count={candidate_tile_count} "
                    f"accepted_improvement={improvement} "
                    f"candidate_workload={candidate_workload}",
                )
                worst_stage_id = stage_id
                worst_stage_tile_count = candidate_tile_count
                worst_stage_improvement = improvement
                break

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
            f"improvement={worst_stage_improvement}",
        )
        used_tiles += worst_stage_tile_count - tile_counts[worst_stage_id]
        tile_counts[worst_stage_id] = worst_stage_tile_count
        updated_plans = _plan_all_stages_for_tile_counts(graph, mesh, tile_counts, debug=False)
        updated_workloads = _estimate_workloads(updated_plans, graph, debug=debug)
        decision_timeline.append((iteration, worst_stage_id, dict(tile_counts), dict(updated_workloads)))
        _debug(debug, f"[workload_balancing] updated_tile_counts={tile_counts}")

    # Materialize final StagePlan objects, including the chosen logical layouts.
    _debug(debug, "[workload_balancing] phase=finalize")
    plans = _plan_all_stages_for_tile_counts(graph, mesh, tile_counts, debug=False)
    _debug(debug, f"[workload_balancing] final_tile_counts={tile_counts}")
    _debug(debug, f"[workload_balancing] final_logical_shapes={ {stage_id: plan.logical_shape for stage_id, plan in plans.items()} }")
    _debug(debug, f"[workload_balancing] final_allocation={ {stage_id: plan.tile_count for stage_id, plan in plans.items()} }")
    _debug_decision_timeline(debug, graph, decision_timeline)
    _debug_final_workloads(debug, graph, plans)
    return plans


def _tile_count_options_after_growth(
    current_tile_count: int,
    remaining_tiles: int,
    mesh: Mesh,
) -> tuple[int, ...]:
    """Return growth candidates that fit the mesh and bounded greedy step size."""
    max_tile_count = min(
        current_tile_count * 2,
        current_tile_count + remaining_tiles,
    )
    options = []
    for tile_count in range(current_tile_count + 1, max_tile_count + 1):
        if not _physical_shape_options(tile_count, mesh):
            continue
        options.append(tile_count)
    return tuple(options)


def _logical_shape_options(tile_count: int) -> tuple[tuple[int, int], ...]:
    """Return logical factor pairs whose area equals the stage tile count."""
    options = []
    for height in range(1, tile_count + 1):
        if tile_count % height == 0:
            options.append((tile_count // height, height))
    return tuple(options)


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


def _default_layouts_for_node(
    node: Node,
    submesh: Submesh,
    logical_shape: tuple[int, int],
) -> tuple[tuple, tuple]:
    """Return the op default input and output layouts for one logical shape."""
    return (
        node.payload.default_input_layouts(
            submesh,
            logical_shape=logical_shape,
        ),
        node.payload.default_output_layouts(
            submesh,
            logical_shape=logical_shape,
        ),
    )


def _plan_all_stages_for_tile_counts(
    graph: Graph,
    mesh: Mesh,
    tile_counts: dict[int, int],
    debug: bool,
) -> dict[int, StagePlan]:
    """Build the best layout-preserving plan for each current stage tile count."""
    plans: dict[int, StagePlan] = {}
    for stage_id, node in enumerate(graph.nodes):
        plans[stage_id] = _best_stage_plan_for_tile_count(
            node,
            mesh,
            stage_id,
            tile_counts[stage_id],
            debug=debug,
        )
    return plans


def _best_stage_plan_for_tile_count(
    node: Node,
    mesh: Mesh,
    stage_id: int,
    tile_count: int,
    debug: bool,
) -> StagePlan:
    """Choose the lowest-cost logical layout that fits per-tile L1 memory."""
    submesh = _planning_submesh(mesh, stage_id, tile_count)
    best_plan: StagePlan | None = None
    best_workload: float | None = None

    for logical_shape in _logical_shape_options(tile_count):
        input_layouts, output_layouts = _default_layouts_for_node(
            node,
            submesh,
            logical_shape,
        )
        fits_l1 = True
        max_l1_bytes = 0
        min_l1_capacity = None
        for tile in submesh.tiles:
            tile_work = node.payload.build_tile_work(
                input_layouts=input_layouts,
                output_layouts=output_layouts,
                tile=tile,
            )
            max_l1_bytes = max(max_l1_bytes, tile_work.l1_bytes)
            min_l1_capacity = (
                tile.memory.size
                if min_l1_capacity is None
                else min(min_l1_capacity, tile.memory.size)
            )
            if not tile_work.fits_l1(tile):
                fits_l1 = False
                break
        if not fits_l1:
            _debug(
                debug,
                "[workload_balancing] "
                f"stage={stage_id} tile_count={tile_count} "
                f"logical_shape={logical_shape} skip=l1_exceeded "
                f"max_l1_bytes={max_l1_bytes} min_l1_capacity={min_l1_capacity}",
            )
            continue

        plan = StagePlan(
            stage_id=stage_id,
            tile_count=tile_count,
            logical_shape=logical_shape,
            input_layouts=input_layouts,
            output_layouts=output_layouts,
        )
        workload = _estimate_stage_workload(node, plan)
        _debug(
            debug,
            "[workload_balancing] "
            f"stage={stage_id} tile_count={tile_count} "
            f"logical_shape={logical_shape} workload={workload}",
        )
        if best_workload is None or workload < best_workload:
            best_plan = plan
            best_workload = workload

    if best_plan is None:
        raise ValueError(
            f"node {node.name} has no valid logical shape for tile_count={tile_count} "
            "using full tile-work slices"
        )
    _debug(
        debug,
        "[workload_balancing] "
        f"stage={stage_id} tile_count={tile_count} "
        f"choose_logical_shape={best_plan.logical_shape} "
        f"workload={best_workload}",
    )
    return best_plan


def _estimate_workloads(
    plans: dict[int, StagePlan],
    graph: Graph,
    debug: bool,
) -> dict[int, float]:
    """Estimate per-stage workload for the current stage plans."""
    workloads: dict[int, float] = {}
    for stage_id, node in enumerate(graph.nodes):
        workloads[stage_id] = _estimate_stage_workload(node, plans[stage_id])
    return workloads


def _estimate_stage_workload(node: Node, plan: StagePlan) -> float:
    """Estimate compute cost for one node under one stage plan."""
    step_cost = cost_estimator(
        node=node,
        input_layouts=plan.input_layouts,
        output_layouts=plan.output_layouts,
    )
    return step_cost


def _debug(enabled: bool, message: str) -> None:
    """Print a workload-balancing debug message when enabled."""
    if enabled:
        print(message)


def _debug_final_workloads(
    enabled: bool,
    graph: Graph,
    plans: dict[int, StagePlan],
) -> None:
    """Print the final per-stage workload table when debug output is enabled."""
    if not enabled:
        return

    print("[workload_balancing] final_stage_workloads:")
    for stage_id, node in enumerate(graph.nodes):
        plan = plans[stage_id]
        workload = _estimate_stage_workload(node, plan)
        print(
            "  "
            f"stage={stage_id} node={node.name} "
            f"tile_count={plan.tile_count} "
            f"logical_shape={plan.logical_shape} "
            f"workload={workload}"
        )


def _debug_decision_timeline(
    enabled: bool,
    graph: Graph,
    decision_timeline: list[tuple[int, int | None, dict[int, int], dict[int, float]]],
) -> None:
    """Print tile counts and workloads after each committed greedy decision."""
    if not enabled:
        return

    stage_headers = " ".join(
        f"stage{stage_id}:{node.name}"
        for stage_id, node in enumerate(graph.nodes)
    )
    print("[workload_balancing] decision_timeline:")
    print(f"  stages {stage_headers}")
    for iteration, changed_stage_id, tile_counts, workloads in decision_timeline:
        changed = "initial" if changed_stage_id is None else f"stage={changed_stage_id}"
        stage_state = " ".join(
            (
                f"stage={stage_id} "
                f"tile_count={tile_counts[stage_id]} "
                f"workload={workloads[stage_id]}"
            )
            for stage_id in range(len(graph.nodes))
        )
        print(f"  iter={iteration} changed={changed} {stage_state}")
