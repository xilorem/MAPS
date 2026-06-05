"""Planner v2 workload balancing without rectangular-shape constraints."""

from __future__ import annotations

from MAPS.arch import Mesh
from MAPS.core.graph import Graph, Node
from MAPS.core.layout import tile_tensor_slice
from MAPS.planner.connected_submesh import representative_connected_submesh
from MAPS.planner.cost import cost_estimator
from MAPS.planner.select_stage import StageSelection
from MAPS.planner.workload_balancing import StagePlan, _resolve_stage_selection


def balance_workload(
    graph: Graph,
    mesh: Mesh,
    stage_selection: StageSelection | None = None,
    debug: bool = False,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> dict[int, StagePlan]:
    """Balance workload using connected tile sets and one-tile growth steps."""
    resolved_stage_selection = _resolve_stage_selection(graph, stage_selection)
    stage_ids = tuple(resolved_stage_selection)
    initializer_tensors = frozenset(graph.initializers)
    graph_inputs = frozenset(graph.inputs)
    graph_outputs = frozenset(graph.outputs)
    producer_stage_id_by_tensor = _producer_stage_id_by_tensor(resolved_stage_selection)
    tile_counts: dict[int, int] = {}

    if len(stage_ids) > mesh.num_tiles:
        raise ValueError("graph has more selected stages than available tiles")

    _debug(debug, "[workload_balancing] phase=initial_l1_seeding")
    for stage_id in stage_ids:
        tile_counts[stage_id] = initial_tile_count_for_stage(
            mesh=mesh,
            stage_id=stage_id,
            stage_selection=resolved_stage_selection,
            initializer_tensors=initializer_tensors,
            debug=debug,
        )

    used_tiles = sum(tile_counts.values())
    if used_tiles > mesh.num_tiles:
        raise ValueError("minimum L1-feasible tile counts exceed available tiles")

    _debug(debug, f"[workload_balancing] start used_tiles={used_tiles}/{mesh.num_tiles}")
    _debug(debug, f"[workload_balancing] initial_tile_counts={tile_counts}")
    _debug(debug, "[workload_balancing] phase=greedy_growth")
    while used_tiles < mesh.num_tiles:
        current_plans = _plan_all_stages_for_tile_counts(
            resolved_stage_selection,
            mesh,
            tile_counts,
            initializer_tensors=initializer_tensors,
            debug=False,
        )
        current_selection_metrics = _estimate_selection_metrics(
            current_plans,
            resolved_stage_selection,
            mesh=mesh,
            alpha=alpha,
            beta=beta,
            graph_inputs=graph_inputs,
            graph_outputs=graph_outputs,
            producer_stage_id_by_tensor=producer_stage_id_by_tensor,
            initializer_tensors=initializer_tensors,
        )
        stage_order = tuple(
            sorted(
                stage_ids,
                key=lambda stage_id: (-current_selection_metrics[stage_id], stage_id),
            )
        )
        _debug(debug, f"[workload_balancing] used_tiles={used_tiles}/{mesh.num_tiles}")
        _debug(
            debug,
            f"[workload_balancing] current_selection_metrics={_format_debug_workloads(current_selection_metrics)}",
        )
        _debug(debug, f"[workload_balancing] stage_order_by_workload={stage_order}")

        chosen_stage_id: int | None = None
        chosen_tile_count: int | None = None

        for stage_id in stage_order:
            _debug(
                debug,
                "[workload_balancing] "
                f"try_stage={stage_id} nodes={_stage_label(resolved_stage_selection[stage_id])} "
                f"current_tile_count={tile_counts[stage_id]} "
                f"current_logical_shape={current_plans[stage_id].logical_shape} "
                f"current_metric={_format_debug_cost(current_selection_metrics[stage_id])}",
            )
            grown_tile_count = grow_tile_count_for_stage(
                stage_id=stage_id,
                stage_selection=resolved_stage_selection,
                mesh=mesh,
                tile_counts=tile_counts,
                used_tiles=used_tiles,
                current_metric=current_selection_metrics[stage_id],
                initializer_tensors=initializer_tensors,
                debug=debug,
                alpha=alpha,
                beta=beta,
                graph_inputs=graph_inputs,
                graph_outputs=graph_outputs,
                producer_stage_id_by_tensor=producer_stage_id_by_tensor,
            )
            if grown_tile_count is None:
                _debug(debug, f"[workload_balancing] stage={stage_id} no_valid_growth")
                continue
            chosen_stage_id = stage_id
            chosen_tile_count = grown_tile_count
            break

        if chosen_stage_id is None or chosen_tile_count is None:
            _debug(debug, "[workload_balancing] no_global_improvement_available")
            break

        chosen_metric = current_selection_metrics[chosen_stage_id]
        chosen_candidate_tile_counts = dict(tile_counts)
        chosen_candidate_tile_counts[chosen_stage_id] = chosen_tile_count
        chosen_candidate_plans = _plan_all_stages_for_tile_counts(
            resolved_stage_selection,
            mesh,
            chosen_candidate_tile_counts,
            initializer_tensors=initializer_tensors,
            debug=False,
        )
        chosen_candidate_metric = _estimate_selection_metrics(
            chosen_candidate_plans,
            resolved_stage_selection,
            mesh=mesh,
            alpha=alpha,
            beta=beta,
            graph_inputs=graph_inputs,
            graph_outputs=graph_outputs,
            producer_stage_id_by_tensor=producer_stage_id_by_tensor,
            initializer_tensors=initializer_tensors,
        )[chosen_stage_id]
        improvement = chosen_metric - chosen_candidate_metric
        _debug(
            debug,
            "[workload_balancing] "
            f"choose worst_stage={chosen_stage_id} "
            f"new_tile_count={chosen_tile_count} "
            f"improvement={_format_debug_cost(improvement)}",
        )
        used_tiles += chosen_tile_count - tile_counts[chosen_stage_id]
        tile_counts[chosen_stage_id] = chosen_tile_count
        _debug(debug, f"[workload_balancing] updated_tile_counts={tile_counts}")

    plans = _plan_all_stages_for_tile_counts(
        resolved_stage_selection,
        mesh,
        tile_counts,
        initializer_tensors=initializer_tensors,
        debug=False,
    )
    _debug(debug, f"[workload_balancing] final_tile_counts={tile_counts}")
    _debug(
        debug,
        "[workload_balancing] "
        f"final_logical_shapes={ {stage_id: plan.logical_shape for stage_id, plan in plans.items()} }",
    )
    _debug_stage_metric_breakdown(
        debug=debug,
        plans=plans,
        stage_selection=resolved_stage_selection,
        mesh=mesh,
        graph_inputs=graph_inputs,
        graph_outputs=graph_outputs,
        producer_stage_id_by_tensor=producer_stage_id_by_tensor,
        initializer_tensors=initializer_tensors,
    )
    return plans


def initial_tile_count_for_stage(
    mesh: Mesh,
    stage_id: int,
    stage_selection: StageSelection,
    initializer_tensors: frozenset,
    debug: bool = False,
) -> int:
    stage_nodes = stage_selection[stage_id]
    for tile_count in range(1, mesh.num_tiles + 1):
        try:
            plan = _best_stage_plan_for_stage_nodes(
                stage_nodes=stage_nodes,
                mesh=mesh,
                stage_id=stage_id,
                tile_count=tile_count,
                initializer_tensors=initializer_tensors,
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
        _debug(
            debug,
            "[workload_balancing] "
            f"seed stage={stage_id} choose tile_count={tile_count} "
            f"logical_shape={plan.logical_shape}",
        )
        return tile_count
    raise ValueError(
        f"stage {stage_id} ({_stage_label(stage_nodes)}) "
        f"has no L1-feasible tile count on mesh {mesh.shape}"
    )


def grow_tile_count_for_stage(
    stage_id: int,
    stage_selection: StageSelection,
    mesh: Mesh,
    tile_counts: dict[int, int],
    used_tiles: int,
    current_metric: float,
    initializer_tensors: frozenset,
    debug: bool = False,
    alpha: float = 1.0,
    beta: float = 1.0,
    graph_inputs: frozenset = frozenset(),
    graph_outputs: frozenset = frozenset(),
    producer_stage_id_by_tensor: dict[object, int] | None = None,
) -> int | None:
    current_tile_count = tile_counts[stage_id]
    remaining_tiles = mesh.num_tiles - used_tiles
    candidate_tile_count_options = tuple(
        current_tile_count + added_tiles
        for added_tiles in range(1, remaining_tiles + 1)
    )
    _debug(
        debug,
        "[workload_balancing] "
        f"stage={stage_id} candidate_tile_counts={candidate_tile_count_options}",
    )

    for candidate_tile_count in candidate_tile_count_options:
        candidate_tile_counts = dict(tile_counts)
        candidate_tile_counts[stage_id] = candidate_tile_count
        candidate_plans = _plan_all_stages_for_tile_counts(
            stage_selection,
            mesh,
            candidate_tile_counts,
            initializer_tensors=initializer_tensors,
            debug=False,
        )
        candidate_metrics = _estimate_selection_metrics(
            candidate_plans,
            stage_selection,
            mesh=mesh,
            alpha=alpha,
            beta=beta,
            graph_inputs=graph_inputs,
            graph_outputs=graph_outputs,
            producer_stage_id_by_tensor=producer_stage_id_by_tensor or {},
            initializer_tensors=initializer_tensors,
        )
        candidate_metric = candidate_metrics[stage_id]
        improvement = current_metric - candidate_metric
        if improvement <= 0:
            _debug(
                debug,
                "[workload_balancing] "
                f"stage={stage_id} candidate_tile_count={candidate_tile_count} "
                f"skip=no_metric_improvement "
                f"candidate_metric={_format_debug_cost(candidate_metric)} "
                f"current_metric={_format_debug_cost(current_metric)}",
            )
            continue

        _debug(
            debug,
            "[workload_balancing] "
            f"stage={stage_id} candidate_tile_count={candidate_tile_count} "
            f"accepted_improvement={_format_debug_cost(improvement)} "
            f"candidate_metric={_format_debug_cost(candidate_metric)} "
            "shape_count=1",
        )
        return candidate_tile_count

    return None


def _plan_all_stages_for_tile_counts(
    stage_selection: StageSelection,
    mesh: Mesh,
    tile_counts: dict[int, int],
    initializer_tensors: frozenset,
    debug: bool,
) -> dict[int, StagePlan]:
    return {
        stage_id: _best_stage_plan_for_stage_nodes(
            stage_nodes=stage_nodes,
            mesh=mesh,
            stage_id=stage_id,
            tile_count=tile_counts[stage_id],
            initializer_tensors=initializer_tensors,
            debug=debug,
        )
        for stage_id, stage_nodes in stage_selection.items()
    }


def _best_stage_plan_for_stage_nodes(
    stage_nodes: tuple[Node, ...],
    mesh: Mesh,
    stage_id: int,
    tile_count: int,
    initializer_tensors: frozenset,
    debug: bool = False,
) -> StagePlan:
    submesh = representative_connected_submesh(mesh, stage_id, tile_count)
    best_plan: StagePlan | None = None
    best_workload: int | None = None

    for logical_shape in _logical_shape_options(tile_count):
        node_output_layouts = _layouts_for_stage_nodes(
            stage_nodes,
            submesh,
            logical_shape,
        )
        peak_l1_bytes = peak_l1_occupancy_for_stage(
            stage_nodes=stage_nodes,
            node_output_layouts=node_output_layouts,
            submesh=submesh,
            initializer_tensors=initializer_tensors,
        )
        min_l1_capacity = min(tile.memory.size for tile in submesh.tiles)
        if peak_l1_bytes > min_l1_capacity:
            continue

        plan = StagePlan(
            stage_id=stage_id,
            tile_count=tile_count,
            logical_shape=logical_shape,
            output_layouts=node_output_layouts[-1],
            nodes=stage_nodes,
            node_output_layouts=node_output_layouts,
        )
        workload = _estimate_stage_group_workload(stage_nodes, plan)
        if (
            best_plan is None
            or workload < best_workload
            or (workload == best_workload and logical_shape[1] < best_plan.logical_shape[1])
        ):
            best_plan = plan
            best_workload = workload

    if best_plan is None:
        raise ValueError(
            f"stage {_stage_label(stage_nodes)} has no valid logical shape for tile_count={tile_count} "
            "using full tile-work slices"
        )

    _debug(
        debug,
        "[workload_balancing] "
        f"stage={stage_id} tile_count={tile_count} "
        f"logical_shape={best_plan.logical_shape} workload={_format_debug_cost(best_workload)}",
    )
    return best_plan


def _logical_shape_options(tile_count: int) -> tuple[tuple[int, int], ...]:
    options = []
    for height in range(1, tile_count + 1):
        if tile_count % height != 0:
            continue
        options.append((tile_count // height, height))
    return tuple(options)


def _layouts_for_stage_nodes(
    stage_nodes: tuple[Node, ...],
    submesh,
    logical_shape: tuple[int, int],
) -> tuple[tuple, ...]:
    return tuple(
        node.payload.output_layouts(submesh, logical_shape=logical_shape)
        for node in stage_nodes
    )


def peak_l1_occupancy_for_stage(
    stage_nodes: tuple[Node, ...],
    node_output_layouts: tuple[tuple, ...],
    submesh,
    initializer_tensors: frozenset,
) -> int:
    return max(
        (
            peak_l1_occupancy_for_tile(
                stage_nodes=stage_nodes,
                node_output_layouts=node_output_layouts,
                tile=tile,
                initializer_tensors=initializer_tensors,
            )
            for tile in submesh.tiles
        ),
        default=0,
    )


def peak_l1_occupancy_for_tile(
    stage_nodes: tuple[Node, ...],
    node_output_layouts: tuple[tuple, ...],
    tile,
    initializer_tensors: frozenset,
) -> int:
    initializer_memory = 0
    max_node_dynamic_memory = 0
    seen_initializer_slices = set()

    for node, output_layouts in zip(stage_nodes, node_output_layouts):
        tile_work = node.payload.build_tile_work(output_layouts=output_layouts, tile=tile)
        initializer_bytes, node_dynamic_memory = peak_l1_occupancy_for_node(
            tile_work.input_slices,
            tile_work.output_slices,
            initializer_tensors=initializer_tensors,
            seen_initializer_slices=seen_initializer_slices,
        )
        initializer_memory += initializer_bytes
        max_node_dynamic_memory = max(max_node_dynamic_memory, node_dynamic_memory)

    return initializer_memory + max_node_dynamic_memory


def peak_l1_occupancy_for_node(
    input_slices: tuple,
    output_slices: tuple,
    initializer_tensors: frozenset,
    seen_initializer_slices: set[tuple[int, object]],
) -> tuple[int, int]:
    initializer_memory = 0
    node_dynamic_memory = 0
    for ref in input_slices:
        if ref.tensor in initializer_tensors or getattr(ref.tensor, "is_initializer", False):
            key = (id(ref.tensor), ref.tensor_slice)
            if key not in seen_initializer_slices:
                seen_initializer_slices.add(key)
                initializer_memory += ref.num_bytes
        else:
            node_dynamic_memory += ref.num_bytes
    for ref in output_slices:
        node_dynamic_memory += ref.num_bytes
    return initializer_memory, node_dynamic_memory


def _estimate_selection_metrics(
    plans: dict[int, StagePlan],
    stage_selection: StageSelection,
    mesh: Mesh,
    alpha: float,
    beta: float,
    graph_inputs: frozenset,
    graph_outputs: frozenset,
    producer_stage_id_by_tensor: dict[object, int],
    initializer_tensors: frozenset,
) -> dict[int, float]:
    return {
        stage_id: _selection_metric_for_stage(
            stage_nodes=stage_nodes,
            plan=plans[stage_id],
            mesh=mesh,
            alpha=alpha,
            beta=beta,
            graph_inputs=graph_inputs,
            graph_outputs=graph_outputs,
            producer_stage_id_by_tensor=producer_stage_id_by_tensor,
            initializer_tensors=initializer_tensors,
        )
        for stage_id, stage_nodes in stage_selection.items()
    }


def _selection_metric_for_stage(
    stage_nodes: tuple[Node, ...],
    plan: StagePlan,
    mesh: Mesh,
    alpha: float,
    beta: float,
    graph_inputs: frozenset,
    graph_outputs: frozenset,
    producer_stage_id_by_tensor: dict[object, int],
    initializer_tensors: frozenset,
) -> float:
    submesh = representative_connected_submesh(mesh, plan.stage_id, plan.tile_count)
    worst_tile_compute = _worst_tile_compute_workload_for_stage(
        stage_nodes=stage_nodes,
        node_output_layouts=plan.node_output_layouts,
        submesh=submesh,
    )
    worst_tile_io = _worst_tile_l2_transfer_workload_for_stage(
        stage_id=plan.stage_id,
        stage_nodes=stage_nodes,
        node_output_layouts=plan.node_output_layouts,
        submesh=submesh,
        mesh=mesh,
        graph_inputs=graph_inputs,
        graph_outputs=graph_outputs,
        producer_stage_id_by_tensor=producer_stage_id_by_tensor,
        initializer_tensors=initializer_tensors,
    )
    return alpha * worst_tile_compute + beta * worst_tile_io


def _worst_tile_compute_workload_for_stage(
    stage_nodes: tuple[Node, ...],
    node_output_layouts: tuple[tuple, ...],
    submesh,
) -> int:
    return max(
        (
            sum(
                _node_compute_workload_for_tile(node, output_layouts, tile)
                for node, output_layouts in zip(stage_nodes, node_output_layouts)
            )
            for tile in submesh.tiles
        ),
        default=0,
    )


def _node_compute_workload_for_tile(node: Node, output_layouts: tuple, tile) -> int:
    tile_work = node.payload.build_tile_work(output_layouts=output_layouts, tile=tile)
    placement_cost = getattr(node.payload.cost_model, "placement_cost", None)
    if placement_cost is not None:
        return int(placement_cost(node=node, output_layouts=output_layouts))
    return int(node.payload.cost_model.cost(tile_work, tile))


def _worst_tile_l2_transfer_workload_for_stage(
    stage_id: int,
    stage_nodes: tuple[Node, ...],
    node_output_layouts: tuple[tuple, ...],
    submesh,
    mesh: Mesh,
    graph_inputs: frozenset,
    graph_outputs: frozenset,
    producer_stage_id_by_tensor: dict[object, int],
    initializer_tensors: frozenset,
) -> int:
    return max(
        (
            sum(
                _node_l2_transfer_workload_for_tile(
                    stage_id=stage_id,
                    node=node,
                    output_layouts=output_layouts,
                    tile=tile,
                    mesh=mesh,
                    graph_inputs=graph_inputs,
                    graph_outputs=graph_outputs,
                    producer_stage_id_by_tensor=producer_stage_id_by_tensor,
                    initializer_tensors=initializer_tensors,
                )
                for node, output_layouts in zip(stage_nodes, node_output_layouts)
            )
            for tile in submesh.tiles
        ),
        default=0,
    )


def _node_l2_transfer_workload_for_tile(
    stage_id: int,
    node: Node,
    output_layouts: tuple,
    tile,
    mesh: Mesh,
    graph_inputs: frozenset,
    graph_outputs: frozenset,
    producer_stage_id_by_tensor: dict[object, int],
    initializer_tensors: frozenset,
) -> int:
    tile_work = node.payload.build_tile_work(output_layouts=output_layouts, tile=tile)
    l2_read_cost = 0
    other_stage_read_cost = 0
    l2_write_cost = 0

    for ref in tile_work.input_slices:
        producer_stage_id = producer_stage_id_by_tensor.get(ref.tensor)
        if ref.tensor in initializer_tensors or ref.tensor in graph_inputs or producer_stage_id is None:
            l2_read_cost += _one_hop_l2_transfer_cost(
                ref.num_bytes,
                tile.memory.bandwidth,
                mesh.l2_memory.bandwidth,
            )
        elif producer_stage_id != stage_id:
            other_stage_read_cost += _one_hop_peer_transfer_cost(
                ref.num_bytes,
                tile.memory.bandwidth,
            )

    for tensor, layout in zip(node.outputs, output_layouts):
        if tensor not in graph_outputs:
            continue
        l2_write_cost += _one_hop_l2_transfer_cost(
            tensor.slice_num_bytes(tile_tensor_slice(tensor, layout, tile)),
            tile.memory.bandwidth,
            mesh.l2_memory.bandwidth,
        )

    return l2_read_cost + other_stage_read_cost + l2_write_cost


def _one_hop_l2_transfer_cost(bytes_: int, l1_bandwidth: int, l2_bandwidth: int) -> int:
    return _ceil_div(bytes_, min(l1_bandwidth, l2_bandwidth))


def _one_hop_peer_transfer_cost(bytes_: int, l1_bandwidth: int) -> int:
    return _ceil_div(bytes_, l1_bandwidth)


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be > 0")
    return (numerator + denominator - 1) // denominator


def _estimate_stage_group_workload(
    stage_nodes: tuple[Node, ...],
    plan: StagePlan,
) -> int:
    node_output_layouts = plan.node_output_layouts or (plan.output_layouts,)
    return sum(
        cost_estimator(node=node, output_layouts=output_layouts)
        for node, output_layouts in zip(stage_nodes, node_output_layouts)
    )


def _producer_stage_id_by_tensor(stage_selection: StageSelection) -> dict[object, int]:
    return {
        tensor: stage_id
        for stage_id, stage_nodes in stage_selection.items()
        for node in stage_nodes
        for tensor in node.outputs
    }


def _debug_stage_metric_breakdown(
    debug: bool,
    plans: dict[int, StagePlan],
    stage_selection: StageSelection,
    mesh: Mesh,
    graph_inputs: frozenset,
    graph_outputs: frozenset,
    producer_stage_id_by_tensor: dict[object, int],
    initializer_tensors: frozenset,
) -> None:
    if not debug:
        return
    print("[workload_balancing] final_stage_metric_breakdown:")
    for stage_id, stage_nodes in stage_selection.items():
        plan = plans[stage_id]
        submesh = representative_connected_submesh(mesh, plan.stage_id, plan.tile_count)
        worst_tile_compute = _worst_tile_compute_workload_for_stage(
            stage_nodes=stage_nodes,
            node_output_layouts=plan.node_output_layouts,
            submesh=submesh,
        )
        worst_tile_io = _worst_tile_l2_transfer_workload_for_stage(
            stage_id=stage_id,
            stage_nodes=stage_nodes,
            node_output_layouts=plan.node_output_layouts,
            submesh=submesh,
            mesh=mesh,
            graph_inputs=graph_inputs,
            graph_outputs=graph_outputs,
            producer_stage_id_by_tensor=producer_stage_id_by_tensor,
            initializer_tensors=initializer_tensors,
        )
        print(
            "  "
            f"stage={stage_id} nodes={_stage_label(stage_nodes)} "
            f"compute={_format_debug_cost(worst_tile_compute)} "
            f"worst_tile_io={_format_debug_cost(worst_tile_io)}"
        )


def _stage_label(stage_nodes: tuple[Node, ...]) -> str:
    return "+".join(node.name for node in stage_nodes)


def _debug(enabled: bool, message: str) -> None:
    if enabled:
        print(message)


def _format_debug_cost(value: int | float) -> str:
    return str(value)


def _format_debug_workloads(workloads: dict[int, int | float]) -> str:
    return "{" + ", ".join(
        f"{stage_id}: {_format_debug_cost(workload)}"
        for stage_id, workload in workloads.items()
    ) + "}"
