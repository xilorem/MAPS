"""Planner v2 spatial mapping over connected non-rectangular tile sets."""

from __future__ import annotations

from MAPS.arch import (
    EndpointKind,
    L1Memory,
    L2Memory,
    Mesh,
    NoC,
    NoCChannel,
    NoCEndpoint,
    NoCLink,
    NoCNode,
    Tile,
)
from MAPS.core.layout import TensorSlice
from MAPS.core.layout import TensorLayout, tile_tensor_slice
from MAPS.core.graph import Graph, Node
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.hw.devices.generic import GENERIC_SCALAR_DEVICE
from MAPS.planner.cost import placement_cost_estimator
from MAPS.planner.select_stage import select_stages
from MAPS.planner.workload_balancing_v2 import StagePlan
from MAPS.transitions import TransportCostModel, build_transition, estimate_transition_cost

_TRANSPORT_MODELS_BY_MESH_ID: dict[int, TransportCostModel] = {}
_DEFAULT_COMMUNICATION_NOCS_BY_SHAPE: dict[tuple[int, int], NoC] = {}


def _placement_surrogate_score(
    submesh: Submesh,
    target: tuple[float, float],
    mesh_center: tuple[float, float],
    neighbor_count: int,
) -> float:
    """Cheap geometry score for limiting candidate placements."""
    center = _submesh_center(submesh)
    target_distance = abs(center[0] - target[0]) + abs(center[1] - target[1])
    center_distance = abs(center[0] - mesh_center[0]) + abs(center[1] - mesh_center[1])
    compactness = _submesh_compactness_penalty(submesh)
    return target_distance + (0.25 * center_distance * neighbor_count) + compactness


def _future_space_penalty(
    mesh: Mesh,
    occupied_tile_ids: set[int],
    remaining_tile_counts: tuple[int, ...],
) -> float:
    """Penalize choices that leave too little or too fragmented free space."""
    if not remaining_tile_counts:
        return 0.0

    component_sizes = sorted(_free_component_sizes(mesh, occupied_tile_ids), reverse=True)
    requested_sizes = sorted(remaining_tile_counts, reverse=True)

    if not component_sizes:
        return 1_000_000.0 * sum(requested_sizes)

    free_total = sum(component_sizes)
    requested_total = sum(requested_sizes)
    largest_component = component_sizes[0]
    largest_request = requested_sizes[0]

    penalty = 0.0
    if free_total < requested_total:
        penalty += 1_000_000.0 * (requested_total - free_total)
    if largest_request > largest_component:
        penalty += 1_000_000.0 * (largest_request - largest_component)

    # Soft fragmentation penalty only. Do not require one component per future
    # stage: multiple stages can still be placed in the same component.
    penalty += 10.0 * max(0, len(component_sizes) - 1)
    return penalty


def _free_component_sizes(mesh: Mesh, occupied_tile_ids: set[int]) -> tuple[int, ...]:
    """Return sizes of 4-neighbor connected free-space components."""
    free_tile_ids = _all_mesh_tile_ids(mesh) - set(occupied_tile_ids)
    seen: set[int] = set()
    sizes: list[int] = []

    for start in sorted(free_tile_ids):
        if start in seen:
            continue

        size = 0
        stack = [start]
        seen.add(start)

        while stack:
            tile_id = stack.pop()
            size += 1
            for neighbor in _neighbor_tile_ids(mesh, tile_id):
                if neighbor in free_tile_ids and neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)

        sizes.append(size)

    return tuple(sizes)


def _default_tiles(width: int, height: int) -> tuple[Tile, ...]:
    return tuple(
        Tile(
            tile_id=y * width + x,
            x=x,
            y=y,
            memory=L1Memory(size=1, bandwidth=1),
            devices=(GENERIC_SCALAR_DEVICE,),
        )
        for y in range(height)
        for x in range(width)
    )


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
    """Heuristically place stages as connected non-rectangular submeshes.

    This is a deterministic floorplanning heuristic with bounded backtracking.
    Each candidate submesh contains exactly the requested number of tiles and is
    4-neighbor connected.  If a greedy choice fragments the mesh, the search
    backs up and tries the next-best candidate instead of failing immediately.
    """

    if objective not in {"max", "sum"}:
        raise ValueError(f"unsupported spatial mapping objective: {objective}")

    if solver_msg and show_progress:
        print("[spatial_mapping] solver_msg ignored by heuristic placer")
    if enable_lossless_pruning and show_progress:
        print("[spatial_mapping] enable_lossless_pruning ignored by heuristic placer")

    stage_plans = _stage_plans_from_tile_counts(tile_counts)
    stage_selection = _resolve_stage_selection(graph, stage_plans)
    node_stage_ids = _node_stage_ids(stage_selection)
    stage_ids = tuple(stage_selection)
    resolved_tile_counts = {
        stage_id: _stage_tile_count(tile_counts[stage_id])
        for stage_id in stage_ids
    }

    total_requested_tiles = sum(resolved_tile_counts.values())
    total_mesh_tiles = len(_all_mesh_tile_ids(mesh))
    if total_requested_tiles > total_mesh_tiles:
        raise ValueError(
            f"cannot place {total_requested_tiles} tiles on mesh with only "
            f"{total_mesh_tiles} tiles"
        )

    placement_order = _heuristic_stage_order(
        graph=graph,
        stage_selection=stage_selection,
        node_stage_ids=node_stage_ids,
        tile_counts=resolved_tile_counts,
    )

    # Candidate generation is lossy because enumerating all connected polyominoes
    # is too expensive.  Use more candidates than the user-facing pruning value,
    # then branch over the best candidates with backtracking.
    candidate_limit = (
        None
        if max_placements_per_stage is None
        else max(32, max_placements_per_stage * 4)
    )
    branch_limit = (
        None
        if max_placements_per_stage is None
        else max(4, min(max_placements_per_stage, 8))
    )

    progress = _ProgressBar("placing connected submeshes", len(placement_order), show_progress)
    failure: dict[str, object] = {}
    mapping = _search_connected_mapping(
        graph=graph,
        mesh=mesh,
        placement_order=placement_order,
        tile_counts=resolved_tile_counts,
        stage_selection=stage_selection,
        node_stage_ids=node_stage_ids,
        stage_plans=stage_plans,
        objective=objective,
        require_l2_input_access_point=require_l2_input_access_point,
        require_l2_output_access_point=require_l2_output_access_point,
        candidate_limit=candidate_limit,
        branch_limit=branch_limit,
        index=0,
        occupied_tile_ids=set(),
        mapping={},
        progress=progress,
        failure=failure,
    )
    progress.close()

    if mapping is None:
        # Retry once with a wider branch factor.  This keeps the default fast but
        # avoids false failures caused by over-aggressive pruning.
        if show_progress:
            print("[spatial_mapping] retrying with wider heuristic search...")
        failure = {}
        mapping = _search_connected_mapping(
            graph=graph,
            mesh=mesh,
            placement_order=placement_order,
            tile_counts=resolved_tile_counts,
            stage_selection=stage_selection,
            node_stage_ids=node_stage_ids,
            stage_plans=stage_plans,
            objective=objective,
            require_l2_input_access_point=require_l2_input_access_point,
            require_l2_output_access_point=require_l2_output_access_point,
            candidate_limit=None if max_placements_per_stage is None else max(128, max_placements_per_stage * 8),
            branch_limit=None if max_placements_per_stage is None else max(16, max_placements_per_stage),
            index=0,
            occupied_tile_ids=set(),
            mapping={},
            progress=None,
            failure=failure,
        )

    if mapping is None:
        stage_id = failure.get("stage_id", "unknown")
        occupied = set(failure.get("occupied_tile_ids", set()))
        free_components = _free_component_sizes(mesh, occupied)
        requested = failure.get("requested_tile_count", "unknown")
        raise ValueError(
            f"stage {stage_id} has no feasible connected placement. "
            f"requested_tiles={requested}, free_tiles={len(_all_mesh_tile_ids(mesh) - occupied)}, "
            f"free_component_sizes={sorted(free_components, reverse=True)}. "
            "This usually means an earlier heuristic choice fragmented the mesh, "
            "or the candidate/branch limit is still too small."
        )

    mapping = {stage_id: mapping[stage_id] for stage_id in stage_ids}

    if print_costs:
        print_spatial_mapping_details(
            graph,
            mesh,
            mapping,
            stage_plans=stage_plans,
            stage_selection=stage_selection,
            label=f"{objective}_heuristic",
        )
    elif print_mapping:
        _print_mapping_grid(mesh, mapping)
    return mapping


def _search_connected_mapping(
    *,
    graph: Graph,
    mesh: Mesh,
    placement_order: tuple[int, ...],
    tile_counts: dict[int, int],
    stage_selection: dict[int, tuple[Node, ...]],
    node_stage_ids: dict[int, int],
    stage_plans: dict[int, StagePlan] | None,
    objective: str,
    require_l2_input_access_point: bool,
    require_l2_output_access_point: bool,
    candidate_limit: int | None,
    branch_limit: int | None,
    index: int,
    occupied_tile_ids: set[int],
    mapping: dict[int, Submesh],
    progress: _ProgressBar | None,
    failure: dict[str, object],
) -> dict[int, Submesh] | None:
    """Backtracking search over connected submesh candidates."""

    if index == len(placement_order):
        return dict(mapping)

    stage_id = placement_order[index]
    requested_tile_count = tile_counts[stage_id]
    remaining_stage_ids = tuple(placement_order[index + 1 :])
    remaining_tile_counts = tuple(tile_counts[remaining_stage_id] for remaining_stage_id in remaining_stage_ids)

    candidates = _placement_options(
        stage_id=stage_id,
        tile_count=requested_tile_count,
        mesh=mesh,
        forbidden_tile_ids=occupied_tile_ids,
        max_candidates=candidate_limit,
    )

    try:
        candidates = _filter_l2_access_point_placements(
            graph=graph,
            stage_selection=stage_selection,
            mesh=mesh,
            placement_options={stage_id: candidates},
            require_l2_input_access_point=require_l2_input_access_point,
            require_l2_output_access_point=require_l2_output_access_point,
        )[stage_id]
    except ValueError:
        candidates = tuple()

    if not candidates:
        _record_mapping_failure(
            failure,
            stage_id=stage_id,
            requested_tile_count=requested_tile_count,
            occupied_tile_ids=occupied_tile_ids,
        )
        return None

    scored_candidates = sorted(
        candidates,
        key=lambda candidate: _heuristic_placement_score(
            graph=graph,
            stage_id=stage_id,
            candidate=candidate,
            mapping=mapping,
            occupied_tile_ids=occupied_tile_ids,
            remaining_tile_counts=remaining_tile_counts,
            stage_selection=stage_selection,
            node_stage_ids=node_stage_ids,
            stage_plans=stage_plans,
            objective=objective,
        ),
    )
    if branch_limit is not None:
        scored_candidates = scored_candidates[:branch_limit]

    for candidate in scored_candidates:
        candidate_tile_ids = set(_submesh_tile_ids(candidate))
        new_occupied = occupied_tile_ids | candidate_tile_ids

        if not _remaining_counts_fit_free_components(
            mesh=mesh,
            occupied_tile_ids=new_occupied,
            remaining_tile_counts=remaining_tile_counts,
        ):
            continue

        mapping[stage_id] = candidate
        result = _search_connected_mapping(
            graph=graph,
            mesh=mesh,
            placement_order=placement_order,
            tile_counts=tile_counts,
            stage_selection=stage_selection,
            node_stage_ids=node_stage_ids,
            stage_plans=stage_plans,
            objective=objective,
            require_l2_input_access_point=require_l2_input_access_point,
            require_l2_output_access_point=require_l2_output_access_point,
            candidate_limit=candidate_limit,
            branch_limit=branch_limit,
            index=index + 1,
            occupied_tile_ids=new_occupied,
            mapping=mapping,
            progress=progress,
            failure=failure,
        )
        if result is not None:
            if progress is not None:
                progress.advance()
            return result
        del mapping[stage_id]

    _record_mapping_failure(
        failure,
        stage_id=stage_id,
        requested_tile_count=requested_tile_count,
        occupied_tile_ids=occupied_tile_ids,
    )
    return None


def _record_mapping_failure(
    failure: dict[str, object],
    *,
    stage_id: int,
    requested_tile_count: int,
    occupied_tile_ids: set[int],
) -> None:
    """Keep the deepest/most constrained failed stage for diagnostics."""
    old_occupied = failure.get("occupied_tile_ids")
    if old_occupied is None or len(occupied_tile_ids) >= len(old_occupied):  # type: ignore[arg-type]
        failure["stage_id"] = stage_id
        failure["requested_tile_count"] = requested_tile_count
        failure["occupied_tile_ids"] = set(occupied_tile_ids)


def _remaining_counts_fit_free_components(
    *,
    mesh: Mesh,
    occupied_tile_ids: set[int],
    remaining_tile_counts: tuple[int, ...],
) -> bool:
    """Return whether remaining stages are not obviously impossible to pack.

    This must be a *necessary* test only.  Multiple future stages may be placed
    in the same free connected component, so it is wrong to require at least one
    free component per remaining stage.
    """
    if not remaining_tile_counts:
        return True

    component_sizes = sorted(_free_component_sizes(mesh, occupied_tile_ids), reverse=True)
    requested_sizes = sorted(remaining_tile_counts, reverse=True)

    if not component_sizes:
        return False

    if sum(component_sizes) < sum(requested_sizes):
        return False

    # Each individual connected submesh must fit inside at least one currently
    # free connected component.  That is the only safe component-size rejection.
    return max(requested_sizes) <= max(component_sizes)


def place_stage_plans(
    stage_plans: dict[int, StagePlan],
    mapping: dict[int, Submesh],
) -> dict[int, StagePlan]:
    """Attach mapped physical submeshes to preselected logical layouts."""

    return {
        stage_id: StagePlan(
            stage_id=plan.stage_id,
            tile_count=plan.tile_count,
            logical_shape=plan.logical_shape,
            output_layouts=_layouts_on_submesh(plan.output_layouts, mapping[stage_id]),
            nodes=plan.nodes,
            node_output_layouts=tuple(
                _layouts_on_submesh(layouts, mapping[stage_id])
                for layouts in plan.node_output_layouts
            ),
        )
        for stage_id, plan in stage_plans.items()
    }


def _stage_plans_from_tile_counts(
    tile_counts: dict[int, int] | dict[int, StagePlan],
) -> dict[int, StagePlan] | None:
    """Return stage plans when the caller supplied plans instead of bare counts."""
    if not tile_counts:
        return None
    if all(isinstance(value, StagePlan) for value in tile_counts.values()):
        return tile_counts  # type: ignore[return-value]
    if any(isinstance(value, StagePlan) for value in tile_counts.values()):
        raise ValueError("tile_counts must contain either all ints or all StagePlan values")
    return None


def _stage_tile_count(tile_count_or_plan: int | StagePlan) -> int:
    """Return the integer tile count from either supported mapping input type."""
    if isinstance(tile_count_or_plan, StagePlan):
        return tile_count_or_plan.tile_count
    return tile_count_or_plan


def _resolve_stage_selection(
    graph: Graph,
    stage_plans: dict[int, StagePlan] | None,
) -> dict[int, tuple[Node, ...]]:
    """Return stage groups from plans when available, otherwise select them."""

    if stage_plans is not None and all(plan.nodes for plan in stage_plans.values()):
        return {
            stage_id: plan.nodes
            for stage_id, plan in stage_plans.items()
        }
    return select_stages(graph)


def _node_stage_ids(stage_selection: dict[int, tuple[Node, ...]]) -> dict[int, int]:
    """Return stage ids keyed by node object identity."""

    return {
        id(node): stage_id
        for stage_id, stage_nodes in stage_selection.items()
        for node in stage_nodes
    }


def _layout_on_submesh(layout: TensorLayout, submesh: Submesh) -> TensorLayout:
    """Copy a tensor layout policy onto a concrete candidate submesh."""
    return TensorLayout(
        submesh=submesh,
        mesh_x=layout.mesh_x,
        mesh_y=layout.mesh_y,
        logical_width=layout.logical_width,
        logical_height=layout.logical_height,
    )


def _layouts_on_submesh(layouts: tuple[TensorLayout, ...], submesh: Submesh) -> tuple[TensorLayout, ...]:
    """Copy a tuple of tensor layout policies onto one candidate submesh."""
    return tuple(_layout_on_submesh(layout, submesh) for layout in layouts)


def _min_pair_comm_cost(costs: dict[str, int]) -> int:
    """Return the cheaper communication mode cost for one placement pair."""
    return min(costs["l1"], costs["l2"])


def _shape_options(tile_count: int, mesh: Mesh) -> tuple[tuple[int, int], ...]:
    """Return rectangular shapes with tile_count area that fit inside the mesh.

    Kept for callers that still use abstract rectangular shape costing.  The
    actual concrete placement heuristic below is non-rectangular.
    """
    options = []
    for height in range(1, mesh.height + 1):
        if tile_count % height != 0:
            continue
        width = tile_count // height
        if width <= 0 or width > mesh.width:
            continue
        options.append((width, height))
    if not options:
        raise ValueError(f"no rectangular shape fits tile_count={tile_count} on mesh {mesh.shape}")
    return tuple(options)


def _heuristic_stage_order(
    graph: Graph,
    stage_selection: dict[int, tuple[Node, ...]],
    node_stage_ids: dict[int, int],
    tile_counts: dict[int, int],
) -> tuple[int, ...]:
    """Place larger and more connected stages first."""
    neighbors = _stage_neighbors(graph, stage_selection, node_stage_ids)
    return tuple(
        sorted(
            stage_selection,
            key=lambda stage_id: (
                -tile_counts[stage_id],
                -len(neighbors[stage_id]),
                stage_id,
            ),
        )
    )


def _placement_options(
    stage_id: int,
    tile_count: int,
    mesh: Mesh,
    forbidden_tile_ids: set[int] | None = None,
    max_candidates: int | None = 128,
) -> tuple[Submesh, ...]:
    """Generate a small, fast set of connected non-rectangular candidates.

    This deliberately does *not* enumerate all polyominoes. Exhaustive
    connected-shape generation is the reason the previous version was slow.
    Instead, generate compact connected blobs from several anchors plus
    rectangle-prefix shapes. All returned candidates have exactly tile_count
    tiles and are 4-neighbor connected.
    """

    if tile_count <= 0:
        raise ValueError(f"tile_count must be > 0, got {tile_count}")

    forbidden = set() if forbidden_tile_ids is None else set(forbidden_tile_ids)
    free_tile_ids = _all_mesh_tile_ids(mesh) - forbidden

    if tile_count > len(free_tile_ids):
        return tuple()

    if tile_count == 1:
        candidates = [frozenset({tile_id}) for tile_id in sorted(free_tile_ids)]
    else:
        candidates_set: set[frozenset[int]] = set()

        # 1) Compact flood-fill blobs from every free anchor.
        for anchor in sorted(free_tile_ids):
            for mode in ("center", "row", "col", "perimeter"):
                tile_set = _grow_connected_tile_set(
                    mesh=mesh,
                    free_tile_ids=free_tile_ids,
                    anchor_tile_id=anchor,
                    tile_count=tile_count,
                    mode=mode,
                )
                if tile_set is not None:
                    candidates_set.add(tile_set)

        # 2) Rectangle-prefix shapes. These include true rectangles when the
        # area matches tile_count, and compact non-rectangular rectangles with
        # the last row/column partially filled otherwise.
        candidates_set.update(
            _rectangle_prefix_candidates(
                mesh=mesh,
                free_tile_ids=free_tile_ids,
                tile_count=tile_count,
            )
        )

        candidates = [
            tile_ids
            for tile_ids in candidates_set
            if _valid_connected_tile_set(mesh, tile_ids)
        ]

    candidates = sorted(
        set(candidates),
        key=lambda tile_ids: _tile_set_surrogate_score(mesh, tile_ids),
    )
    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    return tuple(
        Submesh(
            mesh=mesh,
            submesh_id=stage_id,
            tile_ids=set(tile_ids),
        )
        for tile_ids in candidates
    )


def _grow_connected_tile_set(
    *,
    mesh: Mesh,
    free_tile_ids: set[int],
    anchor_tile_id: int,
    tile_count: int,
    mode: str,
) -> frozenset[int] | None:
    """Greedily grow one connected tile set from an anchor."""
    if anchor_tile_id not in free_tile_ids:
        return None

    chosen: set[int] = {anchor_tile_id}
    frontier = (_neighbor_tile_ids(mesh, anchor_tile_id) & free_tile_ids) - chosen

    while len(chosen) < tile_count:
        if not frontier:
            return None

        next_tile = min(
            frontier,
            key=lambda tile_id: _growth_priority(
                mesh=mesh,
                tile_id=tile_id,
                chosen=chosen,
                mode=mode,
            ),
        )
        chosen.add(next_tile)
        frontier.remove(next_tile)
        frontier |= (_neighbor_tile_ids(mesh, next_tile) & free_tile_ids) - chosen

    return frozenset(chosen)


def _growth_priority(
    *,
    mesh: Mesh,
    tile_id: int,
    chosen: set[int],
    mode: str,
) -> tuple[float, float, int, int]:
    """Priority used by greedy connected-shape growth."""
    candidate = frozenset(chosen | {tile_id})
    tile = _mesh_tile(mesh, tile_id)
    _, _, width, height = _tile_set_bbox(mesh, candidate)
    bbox_area = width * height
    slack = bbox_area - len(candidate)
    perimeter = _tile_set_perimeter(mesh, candidate)
    center = _tile_set_center(mesh, candidate)
    mesh_center = ((mesh.width - 1) / 2, (mesh.height - 1) / 2)
    center_distance = abs(center[0] - mesh_center[0]) + abs(center[1] - mesh_center[1])

    if mode == "row":
        directional = tile.y * mesh.width + tile.x
    elif mode == "col":
        directional = tile.x * mesh.height + tile.y
    elif mode == "perimeter":
        directional = -sum(1 for n in _neighbor_tile_ids(mesh, tile_id) if n in chosen)
    else:
        directional = center_distance

    return (
        slack + 0.25 * perimeter,
        directional,
        tile.y,
        tile.x,
    )


def _rectangle_prefix_candidates(
    *,
    mesh: Mesh,
    free_tile_ids: set[int],
    tile_count: int,
) -> set[frozenset[int]]:
    """Return compact connected candidates obtained by truncating rectangles."""
    candidates: set[frozenset[int]] = set()

    for height in range(1, mesh.height + 1):
        for width in range(1, mesh.width + 1):
            area = width * height
            if area < tile_count:
                continue
            if area > tile_count + max(mesh.width, mesh.height):
                continue

            for y0 in range(mesh.height - height + 1):
                for x0 in range(mesh.width - width + 1):
                    rect_ids = [
                        mesh.tile_id(x, y)
                        for y in range(y0, y0 + height)
                        for x in range(x0, x0 + width)
                    ]
                    if any(tile_id not in free_tile_ids for tile_id in rect_ids):
                        continue

                    candidates.add(frozenset(rect_ids[:tile_count]))

                    col_major = [
                        mesh.tile_id(x, y)
                        for x in range(x0, x0 + width)
                        for y in range(y0, y0 + height)
                    ]
                    candidates.add(frozenset(col_major[:tile_count]))

    return candidates


def _mesh_tile(mesh: Mesh, tile_id: int) -> Tile:
    """Return a tile by id for Mesh implementations where mesh.tiles is a tuple."""
    return mesh.tile_by_id(tile_id)


def _all_mesh_tile_ids(mesh: Mesh) -> set[int]:
    """Return every tile id in row-major mesh order."""
    return {
        mesh.tile_id(x, y)
        for y in range(mesh.height)
        for x in range(mesh.width)
    }


def _rectangle_tile_ids(mesh: Mesh, x0: int, y0: int, width: int, height: int) -> set[int]:
    """Return tile ids for one rectangular region."""
    return {
        mesh.tile_id(x, y)
        for y in range(y0, y0 + height)
        for x in range(x0, x0 + width)
    }


def _neighbor_tile_ids(mesh: Mesh, tile_id: int) -> set[int]:
    """Return 4-neighbor tile ids inside the mesh."""
    tile = _mesh_tile(mesh, tile_id)
    neighbors: set[int] = set()
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        x = tile.x + dx
        y = tile.y + dy
        if 0 <= x < mesh.width and 0 <= y < mesh.height:
            neighbors.add(mesh.tile_id(x, y))
    return neighbors


def _valid_connected_tile_set(mesh: Mesh, tile_ids: frozenset[int]) -> bool:
    """Return whether tile_ids is one valid 4-neighbor connected component."""
    if not tile_ids:
        return False
    if len(tile_ids) == 1:
        return True

    for tile_id in tile_ids:
        if not (_neighbor_tile_ids(mesh, tile_id) & tile_ids):
            return False

    start = next(iter(tile_ids))
    seen = {start}
    stack = [start]
    while stack:
        tile_id = stack.pop()
        for neighbor in _neighbor_tile_ids(mesh, tile_id):
            if neighbor in tile_ids and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen) == len(tile_ids)


def _tile_set_surrogate_score(mesh: Mesh, tile_ids: frozenset[int]) -> tuple[float, int, int, tuple[int, ...]]:
    """Prefer compact, low-perimeter, center-ish connected shapes."""
    x0, y0, width, height = _tile_set_bbox(mesh, tile_ids)
    bbox_area = width * height
    slack = bbox_area - len(tile_ids)
    perimeter = _tile_set_perimeter(mesh, tile_ids)
    center = _tile_set_center(mesh, tile_ids)
    mesh_center = ((mesh.width - 1) / 2, (mesh.height - 1) / 2)
    center_distance = abs(center[0] - mesh_center[0]) + abs(center[1] - mesh_center[1])
    return (
        slack + 0.25 * perimeter + 0.10 * center_distance,
        y0,
        x0,
        tuple(sorted(tile_ids)),
    )


def _tile_set_bbox(mesh: Mesh, tile_ids: frozenset[int]) -> tuple[int, int, int, int]:
    """Return x0, y0, width, height for a set of tile ids."""
    tiles = [_mesh_tile(mesh, tile_id) for tile_id in tile_ids]
    xs = [tile.x for tile in tiles]
    ys = [tile.y for tile in tiles]
    x0 = min(xs)
    y0 = min(ys)
    return x0, y0, max(xs) - x0 + 1, max(ys) - y0 + 1


def _tile_set_center(mesh: Mesh, tile_ids: frozenset[int]) -> tuple[float, float]:
    """Return centroid of a set of tile ids."""
    tiles = [_mesh_tile(mesh, tile_id) for tile_id in tile_ids]
    return (
        sum(tile.x for tile in tiles) / len(tiles),
        sum(tile.y for tile in tiles) / len(tiles),
    )


def _tile_set_perimeter(mesh: Mesh, tile_ids: frozenset[int]) -> int:
    """Return exposed 4-neighbor perimeter of a tile set."""
    return sum(4 - len(_neighbor_tile_ids(mesh, tile_id) & tile_ids) for tile_id in tile_ids)


def _submesh_tile_ids(submesh: Submesh) -> frozenset[int]:
    """Return a submesh's tile ids as a frozenset."""
    return frozenset(tile.tile_id for tile in submesh.tiles)


def _submesh_tile_mask(submesh: Submesh) -> int:
    """Return a bit mask for fast overlap tests."""
    mask = 0
    for tile_id in _submesh_tile_ids(submesh):
        mask |= 1 << tile_id
    return mask


def _submesh_bbox(submesh: Submesh) -> tuple[int, int, int, int]:
    """Return x0, y0, width, height for a possibly non-rectangular submesh."""
    return _tile_set_bbox(submesh.mesh, _submesh_tile_ids(submesh))


def _submesh_bbox_area(submesh: Submesh) -> int:
    """Return the area of the submesh bounding box."""
    _, _, width, height = _submesh_bbox(submesh)
    return width * height


def _default_noc_node_id(x: int, y: int, width: int) -> int:
    return y * width + x


def _default_communication_noc(width: int, height: int) -> NoC:
    cached = _DEFAULT_COMMUNICATION_NOCS_BY_SHAPE.get((width, height))
    if cached is not None:
        return cached

    nodes = tuple(
        NoCNode(node_id=_default_noc_node_id(x, y, width), x=x, y=y)
        for y in range(height)
        for x in range(width)
    )
    link_pairs = tuple(
        (_default_noc_node_id(x, y, width), _default_noc_node_id(x + 1, y, width))
        for y in range(height)
        for x in range(width - 1)
    ) + tuple(
        (_default_noc_node_id(x, y, width), _default_noc_node_id(x, y + 1, width))
        for y in range(height - 1)
        for x in range(width)
    )
    links = tuple(
        NoCLink(
            link_id=link_id,
            src_node_id=src_node_id,
            dst_node_id=dst_node_id,
            channels=(NoCChannel(channel_id=0, width_bytes=1, hop_latency_cycles=1),),
            bidirectional=True,
        )
        for link_id, (src_node_id, dst_node_id) in enumerate(link_pairs)
    )
    l1_endpoints = tuple(
        NoCEndpoint(
            endpoint_id=tile_id,
            kind=EndpointKind.L1,
            node_id=tile_id,
            tile_id=tile_id,
        )
        for tile_id in range(width * height)
    )
    l2_endpoints = tuple(
        NoCEndpoint(
            endpoint_id=width * height + y,
            kind=EndpointKind.L2,
            node_id=_default_noc_node_id(0, y, width),
            name=f"l2_{y}",
        )
        for y in range(height)
    )
    noc = NoC(nodes=nodes, links=links, endpoints=l1_endpoints + l2_endpoints)
    _DEFAULT_COMMUNICATION_NOCS_BY_SHAPE[(width, height)] = noc
    return noc


def _transport_model(mesh: Mesh) -> TransportCostModel:
    cached = _TRANSPORT_MODELS_BY_MESH_ID.get(id(mesh))
    if cached is not None:
        return cached
    model = TransportCostModel(mesh=mesh)
    _TRANSPORT_MODELS_BY_MESH_ID[id(mesh)] = model
    return model


def _filter_l2_access_point_placements(
    graph: Graph,
    stage_selection: dict[int, tuple[Node, ...]],
    mesh: Mesh,
    placement_options: dict[int, tuple[Submesh, ...]],
    require_l2_input_access_point: bool,
    require_l2_output_access_point: bool,
) -> dict[int, tuple[Submesh, ...]]:
    """Keep only placements with L2 access when graph boundary policy requires it."""
    if not require_l2_input_access_point and not require_l2_output_access_point:
        return placement_options

    access_point_tile_ids = _l2_access_point_tile_ids(mesh)
    graph_inputs = set(graph.inputs) - set(graph.initializers)
    graph_outputs = set(graph.outputs)

    filtered: dict[int, tuple[Submesh, ...]] = {}
    for stage_id, placements in placement_options.items():
        needs_input_access = (
            require_l2_input_access_point
            and any(
                tensor in graph_inputs
                for node in stage_selection[stage_id]
                for tensor in node.inputs
            )
        )
        needs_output_access = (
            require_l2_output_access_point
            and any(
                tensor in graph_outputs
                for node in stage_selection[stage_id]
                for tensor in node.outputs
            )
        )
        if not needs_input_access and not needs_output_access:
            filtered[stage_id] = placements
            continue

        filtered_placements = tuple(
            submesh
            for submesh in placements
            if submesh.intersects_tile_ids(access_point_tile_ids)
        )
        if not filtered_placements:
            raise ValueError(
                f"stage {stage_id} has no placement containing an L2 access point"
            )
        filtered[stage_id] = filtered_placements

    return filtered


def _l2_access_point_tile_ids(mesh: Mesh) -> set[int]:
    """Return tiles that should count as L2 access points for boundary placement."""
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


def _edge_placement_pair_count(
    graph: Graph,
    node_stage_ids: dict[int, int],
    placement_options: dict[int, tuple[Submesh, ...]],
) -> int:
    """Count producer-consumer placement pairs across all graph edges."""
    total = 0
    for edge in graph.edges:
        if edge.src is None or edge.dst is None:
            continue
        src_stage = node_stage_ids[id(edge.src)]
        dst_stage = node_stage_ids[id(edge.dst)]
        if src_stage == dst_stage:
            continue
        total += len(placement_options[src_stage]) * len(placement_options[dst_stage])
    return total


class _ProgressBar:
    """Small stdout progress indicator for expensive spatial mapping phases."""

    def __init__(self, label: str, total: int, enabled: bool) -> None:
        self.label = label
        self.total = total
        self.enabled = enabled
        self.current = 0
        self.next_update = 0
        self.update_step = max(1, total // 40) if total else 1
        if self.enabled:
            self._render()

    def advance(self, amount: int = 1) -> None:
        if not self.enabled:
            return
        self.current += amount
        if self.current >= self.next_update or self.current >= self.total:
            self._render()
            self.next_update = self.current + self.update_step

    def close(self) -> None:
        if not self.enabled:
            return
        self.current = self.total
        self._render()
        print()

    def _render(self) -> None:
        width = 28
        filled = width if self.total == 0 else min(width, int(width * self.current / self.total))
        bar = "#" * filled + "-" * (width - filled)
        print(f"\r[spatial_mapping] {self.label} [{bar}] {self.current}/{self.total}", end="", flush=True)


def _prune_placement_options_losslessly(
    placement_options: dict[int, tuple[Submesh, ...]],
    show_progress: bool = False,
) -> dict[int, tuple[Submesh, ...]]:
    """Drop placements that cannot participate in any global non-overlap packing."""
    placement_masks = {
        stage_id: tuple(_submesh_tile_mask(submesh) for submesh in placements)
        for stage_id, placements in placement_options.items()
    }
    pruned: dict[int, tuple[Submesh, ...]] = {}
    progress = _ProgressBar(
        "applying lossless pruning",
        sum(len(placements) for placements in placement_options.values()),
        show_progress,
    )

    for stage_id, placements in placement_options.items():
        supported = []
        remaining_stage_ids = tuple(
            sorted(
                (other_stage_id for other_stage_id in placement_options if other_stage_id != stage_id),
                key=lambda other_stage_id: (
                    len(placement_options[other_stage_id]),
                    -_submesh_bbox_area(placement_options[other_stage_id][0]),
                    other_stage_id,
                ),
            )
        )
        memo: dict[tuple[int, int], bool] = {}
        for placement_idx, submesh in enumerate(placements):
            if _can_pack_remaining_placements(
                remaining_stage_ids,
                placement_masks,
                index=0,
                occupied=placement_masks[stage_id][placement_idx],
                memo=memo,
            ):
                supported.append(submesh)
            progress.advance()
        if not supported:
            progress.close()
            raise ValueError(f"stage {stage_id} has no placement with global packing support")
        pruned[stage_id] = tuple(supported)

    progress.close()
    return pruned


def _can_pack_remaining_placements(
    stage_order: tuple[int, ...],
    placement_masks: dict[int, tuple[int, ...]],
    index: int,
    occupied: int,
    memo: dict[tuple[int, int], bool],
) -> bool:
    """Return whether remaining stages can be packed around an occupied mask."""
    key = (index, occupied)
    cached = memo.get(key)
    if cached is not None:
        return cached

    if index == len(stage_order):
        memo[key] = True
        return True

    stage_id = stage_order[index]
    for mask in placement_masks[stage_id]:
        if occupied & mask:
            continue
        if _can_pack_remaining_placements(
            stage_order,
            placement_masks,
            index=index + 1,
            occupied=occupied | mask,
            memo=memo,
        ):
            memo[key] = True
            return True

    memo[key] = False
    return False


def _prune_placement_options(
    graph: Graph,
    stage_selection: dict[int, tuple[Node, ...]],
    node_stage_ids: dict[int, int],
    mesh: Mesh,
    placement_options: dict[int, tuple[Submesh, ...]],
    max_placements_per_stage: int,
) -> dict[int, tuple[Submesh, ...]]:
    """Limit candidate placements with a deterministic geometry-based heuristic."""
    if max_placements_per_stage <= 0:
        raise ValueError("max_placements_per_stage must be > 0")

    mesh_center = ((mesh.width - 1) / 2, (mesh.height - 1) / 2)
    stage_neighbors = _stage_neighbors(graph, stage_selection, node_stage_ids)
    pruned: dict[int, tuple[Submesh, ...]] = {}
    for stage_id, placements in placement_options.items():
        if len(placements) <= max_placements_per_stage:
            pruned[stage_id] = placements
            continue

        target = _stage_target_center(
            stage_id=stage_id,
            stage_count=len(stage_selection),
            mesh_center=mesh_center,
        )
        scored = sorted(
            placements,
            key=lambda submesh: (
                _placement_surrogate_score(
                    submesh=submesh,
                    target=target,
                    mesh_center=mesh_center,
                    neighbor_count=len(stage_neighbors[stage_id]),
                ),
                tuple(sorted(_submesh_tile_ids(submesh))),
            ),
        )
        pruned[stage_id] = tuple(scored[:max_placements_per_stage])
    return pruned


def _stage_neighbors(
    graph: Graph,
    stage_selection: dict[int, tuple[Node, ...]],
    node_stage_ids: dict[int, int],
) -> dict[int, set[int]]:
    """Return undirected producer-consumer neighbors for each selected stage."""

    neighbors = {stage_id: set() for stage_id in stage_selection}
    for edge in graph.edges:
        if edge.src is None or edge.dst is None:
            continue
        src_stage = node_stage_ids[id(edge.src)]
        dst_stage = node_stage_ids[id(edge.dst)]
        if src_stage == dst_stage:
            continue
        neighbors[src_stage].add(dst_stage)
        neighbors[dst_stage].add(src_stage)
    return neighbors


def _stage_target_center(
    stage_id: int,
    stage_count: int,
    mesh_center: tuple[float, float],
) -> tuple[float, float]:
    """Return a coarse horizontal target position used by placement pruning."""
    if stage_count <= 1:
        return mesh_center
    x = (stage_id / (stage_count - 1)) * (mesh_center[0] * 2)
    return x, mesh_center[1]


def _heuristic_placement_score(
    graph: Graph,
    stage_id: int,
    candidate: Submesh,
    mapping: dict[int, Submesh],
    occupied_tile_ids: set[int],
    remaining_tile_counts: tuple[int, ...],
    stage_selection: dict[int, tuple[Node, ...]],
    node_stage_ids: dict[int, int],
    stage_plans: dict[int, StagePlan] | None,
    objective: str,
) -> tuple[float, float, float, tuple[int, ...]]:
    """Cheap score used during search.

    Do not call placement_cost_estimator, build_tile_work, or transition-cost
    estimation here. Those are accurate but much too expensive inside the
    recursive search. Detailed costs are still available after a final mapping
    is chosen via print_spatial_mapping_details(...).
    """

    new_occupied = occupied_tile_ids | set(_submesh_tile_ids(candidate))
    fragmentation_penalty = _future_space_penalty(
        mesh=candidate.mesh,
        occupied_tile_ids=new_occupied,
        remaining_tile_counts=remaining_tile_counts,
    )

    compactness = _submesh_compactness_penalty(candidate)
    neighbor_distance = _placed_neighbor_distance_score(
        graph=graph,
        stage_id=stage_id,
        candidate=candidate,
        mapping=mapping,
        node_stage_ids=node_stage_ids,
    )

    center = _submesh_center(candidate)
    mesh_center = ((candidate.mesh.width - 1) / 2, (candidate.mesh.height - 1) / 2)
    center_distance = abs(center[0] - mesh_center[0]) + abs(center[1] - mesh_center[1])

    return (
        fragmentation_penalty,
        neighbor_distance,
        compactness + 0.05 * center_distance,
        tuple(sorted(_submesh_tile_ids(candidate))),
    )


def _placed_neighbor_distance_score(
    *,
    graph: Graph,
    stage_id: int,
    candidate: Submesh,
    mapping: dict[int, Submesh],
    node_stage_ids: dict[int, int],
) -> float:
    """Prefer candidates near already placed producer/consumer stages."""
    if not mapping:
        return 0.0

    candidate_center = _submesh_center(candidate)
    score = 0.0
    edge_count = 0

    for edge in graph.edges:
        if edge.src is None or edge.dst is None:
            continue
        src_stage = node_stage_ids[id(edge.src)]
        dst_stage = node_stage_ids[id(edge.dst)]
        if src_stage == dst_stage:
            continue

        other_stage: int | None = None
        if src_stage == stage_id and dst_stage in mapping:
            other_stage = dst_stage
        elif dst_stage == stage_id and src_stage in mapping:
            other_stage = src_stage

        if other_stage is None:
            continue

        other_center = _submesh_center(mapping[other_stage])
        score += abs(candidate_center[0] - other_center[0]) + abs(candidate_center[1] - other_center[1])
        edge_count += 1

    if edge_count == 0:
        return 0.0
    return score / edge_count


def _submesh_center(submesh: Submesh) -> tuple[float, float]:
    """Return the geometric center of a possibly non-rectangular submesh."""
    return _tile_set_center(submesh.mesh, _submesh_tile_ids(submesh))


def _submesh_compactness_penalty(submesh: Submesh) -> float:
    """Return a compactness penalty based on bounding-box slack and perimeter."""
    tile_ids = _submesh_tile_ids(submesh)
    _, _, width, height = _tile_set_bbox(submesh.mesh, tile_ids)
    slack = width * height - len(tile_ids)
    perimeter = _tile_set_perimeter(submesh.mesh, tile_ids)
    return slack + 0.25 * perimeter


def _print_mapping_grid(mesh: Mesh, mapping: dict[int, Submesh]) -> None:
    """Print a compact grid showing which stage owns each mesh tile."""
    owner_by_tile = {}
    for stage_id, submesh in mapping.items():
        for tile in submesh.tiles:
            owner_by_tile[tile.tile_id] = stage_id

    cell_width = max(1, *(len(str(stage_id)) for stage_id in mapping))
    print("Spatial mapping mesh:")
    for y in range(mesh.height):
        cells = []
        for x in range(mesh.width):
            tile_id = mesh.tile_id(x, y)
            owner = owner_by_tile.get(tile_id)
            cell = "." if owner is None else str(owner)
            cells.append(cell.rjust(cell_width))
        print(" ".join(cells))


def _format_cycle_cost(value: int) -> str:
    """Format one reported cycle cost."""
    return str(value)


def _stage_name(stage_nodes: tuple[Node, ...]) -> str:
    """Return a compact display name for one selected stage."""

    return "+".join(node.name for node in stage_nodes)


def print_spatial_mapping_details(
    graph: Graph,
    mesh: Mesh,
    mapping: dict[int, Submesh],
    *,
    stage_plans: dict[int, StagePlan] | None = None,
    stage_selection: dict[int, tuple[Node, ...]] | None = None,
    label: str = "mapping",
) -> None:
    """Print selected submeshes, boundary I/O, and edge communication costs."""

    resolved_stage_selection = (
        _resolve_stage_selection(graph, stage_plans)
        if stage_selection is None
        else stage_selection
    )
    node_stage_ids = _node_stage_ids(resolved_stage_selection)
    placement_options = {
        stage_id: (mapping[stage_id],)
        for stage_id in mapping
    }
    chosen_costs = _edge_placement_costs(
        graph,
        placement_options=placement_options,
        node_stage_ids=node_stage_ids,
        stage_plans=stage_plans,
    )
    stage_io_costs = _stage_io_costs_for_placements(
        graph,
        placement_options=placement_options,
        stage_selection=resolved_stage_selection,
        stage_plans=stage_plans,
    )
    stage_internal_costs = _stage_internal_costs_for_placements(
        graph,
        placement_options=placement_options,
        stage_selection=resolved_stage_selection,
        stage_plans=stage_plans,
    )

    print(f"\n[spatial_mapping] chosen submeshes for {label}:")
    for stage_id in resolved_stage_selection:
        submesh = mapping[stage_id]
        x0, y0, width, height = _submesh_bbox(submesh)
        print(
            f"  stage={stage_id} name={_stage_name(resolved_stage_selection[stage_id])} "
            f"bbox=({x0},{y0},{width},{height}) "
            f"tiles={[tile.tile_id for tile in submesh.tiles]}"
        )
    _print_mapping_grid(mesh, mapping)

    print(f"[spatial_mapping] stage L2 boundary costs for {label}:")
    total_stage_io = 0
    for stage_id in resolved_stage_selection:
        io_cost = stage_io_costs[stage_id][0]
        total_stage_io += io_cost["total"]
        print(
            f"  stage={stage_id} name={_stage_name(resolved_stage_selection[stage_id])} "
            f"l2_read={_format_cycle_cost(io_cost['read'])} "
            f"l2_write={_format_cycle_cost(io_cost['write'])} "
            f"l2_total={_format_cycle_cost(io_cost['total'])}"
        )

    print(f"[spatial_mapping] stage internal costs for {label}:")
    total_stage_internal = 0
    for stage_id in resolved_stage_selection:
        internal_cost = stage_internal_costs[stage_id][0]
        total_stage_internal += internal_cost
        print(
            f"  stage={stage_id} name={_stage_name(resolved_stage_selection[stage_id])} "
            f"internal={_format_cycle_cost(internal_cost)}"
        )

    print(f"[spatial_mapping] edge modes for {label}:")
    bottleneck = None
    total_edge_cost = 0
    for edge_idx, edge in enumerate(graph.edges):
        if edge.src is None or edge.dst is None:
            continue
        src_stage = node_stage_ids[id(edge.src)]
        dst_stage = node_stage_ids[id(edge.dst)]
        if src_stage == dst_stage:
            continue
        edge_cost = chosen_costs[(edge_idx, src_stage, dst_stage, 0, 0)]
        mode = "L1" if edge_cost["l1"] <= edge_cost["l2"] else "L2"
        cost = edge_cost["l1"] if mode == "L1" else edge_cost["l2"]
        total_edge_cost += cost
        if bottleneck is None or cost > bottleneck[1]:
            bottleneck = (f"{src_stage}->{dst_stage}:{edge.tensor.name}", cost, mode)
        print(
            f"  edge={edge.tensor.name} src={src_stage}->{dst_stage} "
            f"mode={mode} l1_cost={_format_cycle_cost(edge_cost['l1'])} "
            f"l2_cost={_format_cycle_cost(edge_cost['l2'])} chosen_cost={_format_cycle_cost(cost)}"
        )

    if bottleneck is not None:
        print(
            f"[spatial_mapping] bottleneck for {label} "
            f"edge={bottleneck[0]} mode={bottleneck[2]} cost={_format_cycle_cost(bottleneck[1])}"
        )
    print(
        f"[spatial_mapping] total for {label} "
        f"stage_io={_format_cycle_cost(total_stage_io)} "
        f"stage_internal={_format_cycle_cost(total_stage_internal)} "
        f"edge_comm={_format_cycle_cost(total_edge_cost)} "
        f"total={_format_cycle_cost(total_stage_io + total_stage_internal + total_edge_cost)}"
    )


def _edge_shape_costs(
    graph: Graph,
    shape_options: dict[int, tuple[tuple[int, int], ...]],
    node_stage_ids: dict[int, int] | None = None,
) -> dict[tuple[int, int, int, int, int], dict[str, int]]:
    """Estimate communication costs for every edge and shape-pair combination."""
    resolved_node_stage_ids = (
        _node_stage_ids(select_stages(graph))
        if node_stage_ids is None
        else node_stage_ids
    )
    costs: dict[tuple[int, int, int, int, int], dict[str, int]] = {}
    for edge_idx, edge in enumerate(graph.edges):
        if edge.src is None or edge.dst is None:
            continue
        src_stage = resolved_node_stage_ids[id(edge.src)]
        dst_stage = resolved_node_stage_ids[id(edge.dst)]
        if src_stage == dst_stage:
            continue
        src_node = edge.src
        dst_node = edge.dst
        for src_shape_idx, src_shape in enumerate(shape_options[src_stage]):
            for dst_shape_idx, dst_shape in enumerate(shape_options[dst_stage]):
                costs[(edge_idx, src_stage, dst_stage, src_shape_idx, dst_shape_idx)] = _edge_shape_mode_costs(
                    edge.tensor,
                    src_node,
                    src_shape,
                    dst_node,
                    dst_shape,
                )
    return costs


def _edge_placement_costs(
    graph: Graph,
    placement_options: dict[int, tuple[Submesh, ...]],
    node_stage_ids: dict[int, int] | None = None,
    stage_plans: dict[int, StagePlan] | None = None,
    show_progress: bool = False,
) -> dict[tuple[int, int, int, int, int], dict[str, int]]:
    """Estimate communication costs for every edge and placement-pair combination."""
    resolved_node_stage_ids = (
        _node_stage_ids(_resolve_stage_selection(graph, stage_plans))
        if node_stage_ids is None
        else node_stage_ids
    )
    costs: dict[tuple[int, int, int, int, int], dict[str, int]] = {}
    progress = _ProgressBar(
        "estimating edge costs",
        _edge_placement_pair_count(graph, resolved_node_stage_ids, placement_options),
        show_progress,
    )
    for edge_idx, edge in enumerate(graph.edges):
        if edge.src is None or edge.dst is None:
            continue
        src_stage = resolved_node_stage_ids[id(edge.src)]
        dst_stage = resolved_node_stage_ids[id(edge.dst)]
        if src_stage == dst_stage:
            continue
        src_node = edge.src
        dst_node = edge.dst
        for src_placement_idx, src_submesh in enumerate(placement_options[src_stage]):
            for dst_placement_idx, dst_submesh in enumerate(placement_options[dst_stage]):
                costs[(edge_idx, src_stage, dst_stage, src_placement_idx, dst_placement_idx)] = _edge_placement_mode_costs(
                    edge.tensor,
                    src_node,
                    src_submesh,
                    dst_node,
                    dst_submesh,
                    src_plan=None if stage_plans is None else stage_plans[src_stage],
                    dst_plan=None if stage_plans is None else stage_plans[dst_stage],
                )
                progress.advance()
    progress.close()
    return costs


def _stage_io_costs(
    graph: Graph,
    shape_options: dict[int, tuple[tuple[int, int], ...]],
    stage_selection: dict[int, tuple[Node, ...]] | None = None,
) -> dict[int, dict[int, dict[str, int]]]:
    """Estimate graph-boundary L2 costs for abstract shape candidates."""
    resolved_stage_selection = select_stages(graph) if stage_selection is None else stage_selection
    graph_inputs = set(graph.inputs) - set(graph.initializers)
    graph_outputs = set(graph.outputs)

    io_costs: dict[int, dict[int, dict[str, int]]] = {}
    for stage_id, stage_nodes in resolved_stage_selection.items():
        io_costs[stage_id] = {}
        for shape_idx, shape in enumerate(shape_options[stage_id]):
            mesh = Mesh(
                shape[0],
                shape[1],
                noc=_default_communication_noc(shape[0], shape[1]),
                tiles=_default_tiles(shape[0], shape[1]),
                l2_memory=L2Memory(size=1, bandwidth=1),
            )
            submesh = Submesh(mesh=mesh, submesh_id=stage_id, tile_ids=_rectangle_tile_ids(mesh, 0, 0, shape[0], shape[1]))
            model = TransportCostModel(mesh=mesh)

            read_cost = max(
                (
                    _stage_l2_read_cost(
                        node,
                        node.payload.output_layouts(submesh),
                        graph_inputs,
                        model,
                    )
                    for node in stage_nodes
                ),
                default=0,
            )
            write_cost = max(
                (
                    _stage_l2_write_cost(
                        node.outputs,
                        node.payload.output_layouts(submesh),
                        graph_outputs,
                        model,
                    )
                    for node in stage_nodes
                ),
                default=0,
            )

            io_costs[stage_id][shape_idx] = {
                "read": read_cost,
                "write": write_cost,
                "total": read_cost + write_cost,
            }

    return io_costs


def _stage_io_costs_for_placements(
    graph: Graph,
    placement_options: dict[int, tuple[Submesh, ...]],
    stage_selection: dict[int, tuple[Node, ...]] | None = None,
    stage_plans: dict[int, StagePlan] | None = None,
    show_progress: bool = False,
) -> dict[int, dict[int, dict[str, int]]]:
    """Estimate graph-boundary L2 costs for concrete placement candidates."""
    resolved_stage_selection = (
        _resolve_stage_selection(graph, stage_plans)
        if stage_selection is None
        else stage_selection
    )
    graph_inputs = set(graph.inputs) - set(graph.initializers)
    graph_outputs = set(graph.outputs)

    io_costs: dict[int, dict[int, dict[str, int]]] = {}
    progress = _ProgressBar(
        "estimating stage I/O costs",
        sum(len(placements) for placements in placement_options.values()),
        show_progress,
    )
    for stage_id, stage_nodes in resolved_stage_selection.items():
        io_costs[stage_id] = {}
        for placement_idx, submesh in enumerate(placement_options[stage_id]):
            model = _transport_model(submesh.mesh)

            plan = None if stage_plans is None else stage_plans[stage_id]
            read_cost = max(
                (
                    _stage_l2_read_cost(
                        node,
                        _node_output_layouts(node, submesh, plan),
                        graph_inputs,
                        model,
                    )
                    for node in stage_nodes
                ),
                default=0,
            )

            write_cost = max(
                (
                    _stage_l2_write_cost(
                        node.outputs,
                        _node_output_layouts(node, submesh, plan),
                        graph_outputs,
                        model,
                    )
                    for node in stage_nodes
                ),
                default=0,
            )

            io_costs[stage_id][placement_idx] = {
                "read": read_cost,
                "write": write_cost,
                "total": read_cost + write_cost,
            }
            progress.advance()

    progress.close()
    return io_costs


def _stage_internal_costs_for_placements(
    graph: Graph,
    placement_options: dict[int, tuple[Submesh, ...]],
    stage_selection: dict[int, tuple[Node, ...]] | None = None,
    stage_plans: dict[int, StagePlan] | None = None,
    show_progress: bool = False,
) -> dict[int, dict[int, int]]:
    """Estimate stage-internal execution cost for concrete placement candidates."""

    resolved_stage_selection = (
        _resolve_stage_selection(graph, stage_plans)
        if stage_selection is None
        else stage_selection
    )
    internal_costs: dict[int, dict[int, int]] = {}
    progress = _ProgressBar(
        "estimating stage internal costs",
        sum(len(placements) for placements in placement_options.values()),
        show_progress,
    )
    for stage_id, stage_nodes in resolved_stage_selection.items():
        internal_costs[stage_id] = {}
        for placement_idx, submesh in enumerate(placement_options[stage_id]):
            plan = None if stage_plans is None else stage_plans[stage_id]
            internal_costs[stage_id][placement_idx] = sum(
                placement_cost_estimator(
                    node=node,
                    output_layouts=_node_output_layouts(node, submesh, plan),
                )
                for node in stage_nodes
            )
            progress.advance()

    progress.close()
    return internal_costs


def _stage_l2_read_cost(
    node: Node,
    output_layouts: tuple[TensorLayout, ...],
    graph_inputs: set[Tensor],
    model: TransportCostModel,
) -> int:
    """Return aggregate L2 read cost for external graph inputs."""
    return sum(
        (
            _max_l2_read_cost_from_required_slices(
                tensor,
                _required_input_slices_for_tensor(node, tensor, output_layouts),
                model,
            )
            for tensor in node.inputs
            if tensor in graph_inputs
        )
    )


def _stage_l2_write_cost(
    tensors: tuple[Tensor, ...],
    layouts: tuple[TensorLayout, ...],
    graph_outputs: set[Tensor],
    model: TransportCostModel,
) -> int:
    """Return worst per-tensor L2 write cost for graph outputs."""
    return max(
        (
            _max_l2_write_cost(tensor, layout, model)
            for tensor, layout in zip(tensors, layouts)
            if tensor in graph_outputs
        ),
        default=0,
    )


def _plan_node_index(plan: StagePlan, node: Node) -> int:
    """Return one node's index inside a grouped stage plan."""

    for node_idx, candidate in enumerate(plan.nodes):
        if candidate is node:
            return node_idx
    raise ValueError(f"node {node.name} is not present in stage plan {plan.stage_id}")


def _node_output_layouts(
    node: Node,
    submesh: Submesh,
    plan: StagePlan | None,
) -> tuple[TensorLayout, ...]:
    """Return one node's output layouts on a concrete submesh."""

    if plan is None or not plan.nodes:
        return node.payload.output_layouts(submesh)
    return _layouts_on_submesh(plan.node_output_layouts[_plan_node_index(plan, node)], submesh)


def _node_output_layout(
    node: Node,
    tensor: Tensor,
    submesh: Submesh,
    plan: StagePlan | None,
) -> TensorLayout:
    """Return the output layout producing one tensor on a concrete submesh."""

    output_layouts = _node_output_layouts(node, submesh, plan)
    return output_layouts[_node_output_index(node, tensor)]


def _edge_shape_mode_costs(
    tensor: Tensor,
    src_node: Node,
    src_shape: tuple[int, int],
    dst_node: Node,
    dst_shape: tuple[int, int],
) -> dict[str, int]:
    """Estimate L1-remap and L2-roundtrip costs for one abstract shape pair."""
    mesh = Mesh(
        max(src_shape[0], dst_shape[0]),
        max(src_shape[1], dst_shape[1]),
        noc=_default_communication_noc(max(src_shape[0], dst_shape[0]), max(src_shape[1], dst_shape[1])),
        tiles=_default_tiles(max(src_shape[0], dst_shape[0]), max(src_shape[1], dst_shape[1])),
        l2_memory=L2Memory(size=1, bandwidth=1),
    )
    src_submesh = Submesh(mesh=mesh, submesh_id=0, tile_ids=_rectangle_tile_ids(mesh, 0, 0, src_shape[0], src_shape[1]))
    dst_submesh = Submesh(mesh=mesh, submesh_id=1, tile_ids=_rectangle_tile_ids(mesh, 0, 0, dst_shape[0], dst_shape[1]))

    src_output_layout = src_node.payload.output_layouts(src_submesh)[
        _node_output_index(src_node, tensor)
    ]
    dst_output_layouts = dst_node.payload.output_layouts(dst_submesh)
    dst_input_idx = _node_input_index(dst_node, tensor)
    dst_required_slices = _required_input_slices_for_tensor(
        dst_node,
        tensor,
        dst_output_layouts,
    )

    model = _transport_model(mesh)
    l1_cost = 0
    if _requires_direct_remap(tensor, src_output_layout, dst_required_slices):
        transition = build_transition(
            name=f"spatial_map_{src_node.name}_to_{dst_node.name}_{tensor.name}",
            tensor=tensor,
            tensor_id=0,
            src_layer_id=0,
            src_output_idx=0,
            dst_layer_id=1,
            dst_input_idx=dst_input_idx,
            src_layout=src_output_layout,
            dst_layout=dst_output_layouts[0],
            dst_required_slices=dst_required_slices,
        )
        l1_cost = estimate_transition_cost(
            transition=transition,
            tensor=tensor,
            mesh=mesh,
            model=model,
        ).total_cost

    max_tile_to_l2_cost = 0
    for tile in src_submesh.tiles:
        tensor_slice = tile_tensor_slice(tensor, src_output_layout, tile)
        max_tile_to_l2_cost = max(
            max_tile_to_l2_cost,
            model.l1_to_l2(tile, tensor.slice_num_bytes(tensor_slice)),
        )

    max_l2_to_tile_cost = 0
    for tile, required_slice in dst_required_slices:
        max_l2_to_tile_cost = max(
            max_l2_to_tile_cost,
            model.l2_to_l1(tile, tensor.slice_num_bytes(required_slice)),
        )

    return {
        "l1": l1_cost,
        "l2": max_tile_to_l2_cost + max_l2_to_tile_cost,
    }


def _edge_placement_mode_costs(
    tensor: Tensor,
    src_node: Node,
    src_submesh: Submesh,
    dst_node: Node,
    dst_submesh: Submesh,
    src_plan: StagePlan | None = None,
    dst_plan: StagePlan | None = None,
) -> dict[str, int]:
    """Estimate L1-remap and L2-roundtrip costs for one concrete placement pair."""
    src_output_layout = _node_output_layout(
        src_node,
        tensor,
        src_submesh,
        src_plan,
    )
    dst_input_idx = _node_input_index(dst_node, tensor)
    dst_output_layouts = _node_output_layouts(
        dst_node,
        dst_submesh,
        dst_plan,
    )
    dst_required_slices = _required_input_slices_for_tensor(
        dst_node,
        tensor,
        dst_output_layouts,
    )

    model = _transport_model(src_submesh.mesh)
    l1_cost = 0
    if _requires_direct_remap(tensor, src_output_layout, dst_required_slices):
        transition = build_transition(
            name=f"spatial_map_{src_node.name}_to_{dst_node.name}_{tensor.name}",
            tensor=tensor,
            tensor_id=0,
            src_layer_id=0,
            src_output_idx=0,
            dst_layer_id=1,
            dst_input_idx=dst_input_idx,
            src_layout=src_output_layout,
            dst_layout=dst_output_layouts[0],
            dst_required_slices=dst_required_slices,
        )
        l1_cost = estimate_transition_cost(
            transition=transition,
            tensor=tensor,
            mesh=src_submesh.mesh,
            model=model,
        ).total_cost

    max_tile_to_l2_cost = 0
    for tile in src_submesh.tiles:
        tensor_slice = tile_tensor_slice(tensor, src_output_layout, tile)
        max_tile_to_l2_cost = max(
            max_tile_to_l2_cost,
            model.l1_to_l2(tile, tensor.slice_num_bytes(tensor_slice)),
        )

    max_l2_to_tile_cost = 0
    for tile, required_slice in dst_required_slices:
        max_l2_to_tile_cost = max(
            max_l2_to_tile_cost,
            model.l2_to_l1(tile, tensor.slice_num_bytes(required_slice)),
        )

    return {
        "l1": l1_cost,
        "l2": max_tile_to_l2_cost + max_l2_to_tile_cost,
    }


def _required_input_slice_for_tensor(tensor: Tensor, tile_work: object) -> TensorSlice | None:
    """Return the input slice a tile work item needs for a specific tensor."""
    for ref in tile_work.input_slices:
        if ref.tensor == tensor:
            return ref.tensor_slice
    return None


def _required_input_slices_for_tensor(
    node: Node,
    tensor: Tensor,
    output_layouts: tuple[TensorLayout, ...],
) -> tuple[tuple[object, TensorSlice], ...]:
    """Return every tile-local input slice a node needs for one tensor."""
    required_slices = []
    for tile in output_layouts[0].submesh.tiles:
        tile_work = node.payload.build_tile_work(
            output_layouts=output_layouts,
            tile=tile,
        )
        required_slice = _required_input_slice_for_tensor(tensor, tile_work)
        if required_slice is not None:
            required_slices.append((tile, required_slice))
    return tuple(required_slices)


def _requires_direct_remap(
    tensor: Tensor,
    src_layout: TensorLayout,
    dst_required_slices: tuple[tuple[object, TensorSlice], ...],
) -> bool:
    """Return whether consumer demand requires explicit inter-tile transfer."""
    src_owned_by_tile = {
        tile.tile_id: tile_tensor_slice(tensor, src_layout, tile)
        for tile in src_layout.submesh.tiles
    }
    for tile, required_slice in dst_required_slices:
        if src_owned_by_tile.get(tile.tile_id) != required_slice:
            return True
    return False


def _max_slice_bytes(
    tensor: Tensor,
    layout,
) -> int:
    """Return the largest per-tile slice size for one tensor layout."""
    return max(
        (
            tensor.slice_num_bytes(tile_tensor_slice(tensor, layout, tile))
            for tile in layout.submesh.tiles
        ),
        default=0,
    )


def _max_l2_write_cost(
    tensor: Tensor,
    layout,
    model: TransportCostModel,
) -> int:
    """Return the slowest tile-to-L2 write cost for one tensor layout."""
    return max(
        (
            model.l1_to_l2(
                tile,
                tensor.slice_num_bytes(tile_tensor_slice(tensor, layout, tile)),
            )
            for tile in layout.submesh.tiles
        ),
        default=0,
    )


def _max_l2_read_cost_from_required_slices(
    tensor: Tensor,
    required_slices: tuple[tuple[object, TensorSlice], ...],
    model: TransportCostModel,
) -> int:
    """Return the slowest L2-to-tile read cost for one tensor demand."""
    return max(
        (
            model.l2_to_l1(
                tile,
                tensor.slice_num_bytes(tensor_slice),
            )
            for tile, tensor_slice in required_slices
        ),
        default=0,
    )


def _node_input_index(node: Node, tensor: Tensor) -> int:
    """Return a tensor's input index in a node, or fail if it is not an input."""
    for idx, candidate in enumerate(node.inputs):
        if candidate == tensor:
            return idx
    raise ValueError(f"tensor {tensor.name} is not an input of node {node.name}")


def _node_output_index(node: Node, tensor: Tensor) -> int:
    """Return a tensor's output index in a node, or fail if it is not an output."""

    for idx, candidate in enumerate(node.outputs):
        if candidate == tensor:
            return idx
    raise ValueError(f"tensor {tensor.name} is not an output of node {node.name}")
