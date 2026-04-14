"""Spatial mapping from stage tile budgets to concrete non-overlapping submeshes."""

from __future__ import annotations

from MAPS.arch import L2Memory, Mesh
from MAPS.builders.transition_builder import build_transition
from MAPS.cost_models.transition_cost import estimate_transition_cost
from MAPS.cost_models.transport_cost import TransportCostModel
from MAPS.core.layout import TensorSlice
from MAPS.core.layout import TensorLayout
from MAPS.core.ownership import tile_tensor_slice
from MAPS.core.graph import Graph, Node
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.planner.workload_balancing import StagePlan


def map_spatially(
    graph: Graph,
    mesh: Mesh,
    tile_counts: dict[int, int] | dict[int, StagePlan],
    objective: str = "max",
) -> dict[int, Submesh]:
    """Solve stage placement with a placement-candidate MILP."""

    import pulp

    if objective not in {"max", "sum"}:
        raise ValueError(f"unsupported spatial mapping objective: {objective}")

    stage_ids = tuple(range(len(graph.nodes)))
    stage_plans = _stage_plans_from_tile_counts(tile_counts)
    resolved_tile_counts = {
        stage_id: _stage_tile_count(tile_counts[stage_id])
        for stage_id in stage_ids
    }

    # enumerate the possible placements of a stage given its counts
    placement_options = {
        stage_id: _placement_options(stage_id, resolved_tile_counts[stage_id], mesh)
        for stage_id in stage_ids
    }
    edge_placement_costs = _edge_placement_costs(
        graph,
        placement_options,
        stage_plans=stage_plans,
    )
    stage_io_costs = _stage_io_costs_for_placements(
        graph,
        placement_options,
        stage_plans=stage_plans,
    )

    model = pulp.LpProblem("maps_spatial_mapping", pulp.LpMinimize)

    placement_selected = {
        (stage_id, placement_idx): pulp.LpVariable(f"place_{stage_id}_{placement_idx}", cat="Binary")
        for stage_id in stage_ids
        for placement_idx in range(len(placement_options[stage_id]))
    }
    edge_comm_cost = {}
    stage_io_cost = {}
    pair_selected = {}
    for edge_idx, edge in enumerate(graph.edges):
        if edge.src is None or edge.dst is None:
            continue
        src_stage = _node_stage_id(graph, edge.src)
        dst_stage = _node_stage_id(graph, edge.dst)
        key = (edge_idx, src_stage, dst_stage)
        edge_comm_cost[key] = pulp.LpVariable(f"edge_comm_cost_{edge_idx}_{src_stage}_{dst_stage}", lowBound=0)
        for src_placement_idx in range(len(placement_options[src_stage])):
            for dst_placement_idx in range(len(placement_options[dst_stage])):
                pair_selected[(edge_idx, src_stage, dst_stage, src_placement_idx, dst_placement_idx)] = pulp.LpVariable(
                    f"pair_selected_{edge_idx}_{src_stage}_{dst_stage}_{src_placement_idx}_{dst_placement_idx}",
                    cat="Binary",
                )
    for stage_id in stage_ids:
        stage_io_cost[stage_id] = pulp.LpVariable(f"stage_io_cost_{stage_id}", lowBound=0)

    max_comm_cost = None
    if objective == "max":
        max_comm_cost = pulp.LpVariable("max_comm_cost", lowBound=0)
        model += max_comm_cost
    else:
        model += (
            pulp.lpSum(edge_comm_cost.values())
            + pulp.lpSum(stage_io_cost.values())
        )

    for stage_id in stage_ids:
        model += pulp.lpSum(
            placement_selected[(stage_id, placement_idx)]
            for placement_idx in range(len(placement_options[stage_id]))
        ) == 1

        model += stage_io_cost[stage_id] == pulp.lpSum(
            stage_io_costs[stage_id][placement_idx]["total"] * placement_selected[(stage_id, placement_idx)]
            for placement_idx in range(len(placement_options[stage_id]))
        )
        if max_comm_cost is not None:
            model += max_comm_cost >= stage_io_cost[stage_id]

    for left_stage in stage_ids:
        for right_stage in stage_ids:
            if left_stage >= right_stage:
                continue
            for left_placement_idx, left_submesh in enumerate(placement_options[left_stage]):
                left_tiles = {tile.tile_id for tile in left_submesh.tiles}
                for right_placement_idx, right_submesh in enumerate(placement_options[right_stage]):
                    right_tiles = {tile.tile_id for tile in right_submesh.tiles}
                    if left_tiles & right_tiles:
                        model += (
                            placement_selected[(left_stage, left_placement_idx)]
                            + placement_selected[(right_stage, right_placement_idx)]
                            <= 1
                        )

    for edge_idx, edge in enumerate(graph.edges):
        if edge.src is None or edge.dst is None:
            continue
        src_stage = _node_stage_id(graph, edge.src)
        dst_stage = _node_stage_id(graph, edge.dst)
        key = (edge_idx, src_stage, dst_stage)

        pair_terms = []
        for src_placement_idx in range(len(placement_options[src_stage])):
            for dst_placement_idx in range(len(placement_options[dst_stage])):
                pair_var = pair_selected[(edge_idx, src_stage, dst_stage, src_placement_idx, dst_placement_idx)]
                model += pair_var <= placement_selected[(src_stage, src_placement_idx)]
                model += pair_var <= placement_selected[(dst_stage, dst_placement_idx)]
                model += pair_var >= (
                    placement_selected[(src_stage, src_placement_idx)]
                    + placement_selected[(dst_stage, dst_placement_idx)]
                    - 1
                )

                costs = edge_placement_costs[(edge_idx, src_stage, dst_stage, src_placement_idx, dst_placement_idx)]
                pair_terms.append(_min_pair_comm_cost(costs) * pair_var)

        model += edge_comm_cost[key] == pulp.lpSum(pair_terms)
        if max_comm_cost is not None:
            model += max_comm_cost >= edge_comm_cost[key]

    status = model.solve(pulp.PULP_CBC_CMD(msg=True))
    if status != pulp.LpStatusOptimal:
        raise ValueError(f"spatial mapping solver did not find an optimal solution: {pulp.LpStatus[status]}")

    mapping = {}
    for stage_id in stage_ids:
        for placement_idx, submesh in enumerate(placement_options[stage_id]):
            if pulp.value(placement_selected[(stage_id, placement_idx)]) >= 0.50000000001:
                mapping[stage_id] = submesh
                break
    return mapping


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
            input_layouts=_layouts_on_submesh(plan.input_layouts, mapping[stage_id]),
            output_layouts=_layouts_on_submesh(plan.output_layouts, mapping[stage_id]),
        )
        for stage_id, plan in stage_plans.items()
    }


def _stage_plans_from_tile_counts(
    tile_counts: dict[int, int] | dict[int, StagePlan],
) -> dict[int, StagePlan] | None:
    if not tile_counts:
        return None
    if all(isinstance(value, StagePlan) for value in tile_counts.values()):
        return tile_counts  # type: ignore[return-value]
    if any(isinstance(value, StagePlan) for value in tile_counts.values()):
        raise ValueError("tile_counts must contain either all ints or all StagePlan values")
    return None


def _stage_tile_count(tile_count_or_plan: int | StagePlan) -> int:
    if isinstance(tile_count_or_plan, StagePlan):
        return tile_count_or_plan.tile_count
    return tile_count_or_plan


def _layout_on_submesh(layout: TensorLayout, submesh: Submesh) -> TensorLayout:
    return TensorLayout(
        submesh=submesh,
        mesh_x=layout.mesh_x,
        mesh_y=layout.mesh_y,
        logical_width=layout.logical_width,
        logical_height=layout.logical_height,
    )


def _layouts_on_submesh(layouts: tuple[TensorLayout, ...], submesh: Submesh) -> tuple[TensorLayout, ...]:
    return tuple(_layout_on_submesh(layout, submesh) for layout in layouts)


def _min_pair_comm_cost(costs: dict[str, float]) -> float:
    return min(costs["l1"], costs["l2"])


def _shape_options(tile_count: int, mesh: Mesh) -> tuple[tuple[int, int], ...]:
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


def _placement_options(stage_id: int, tile_count: int, mesh: Mesh) -> tuple[Submesh, ...]:
    placements = []
    for width, height in _shape_options(tile_count, mesh):
        for x0 in range(mesh.width - width + 1):
            for y0 in range(mesh.height - height + 1):
                placements.append(
                    Submesh(
                        mesh=mesh,
                        submesh_id=stage_id,
                        x0=x0,
                        y0=y0,
                        width=width,
                        height=height,
                    )
                )
    return tuple(placements)


def _node_stage_id(graph: Graph, target: object) -> int:
    for stage_id, node in enumerate(graph.nodes):
        if node is target:
            return stage_id
    raise ValueError("node is not present in graph.nodes")


def _edge_shape_costs(
    graph: Graph,
    shape_options: dict[int, tuple[tuple[int, int], ...]],
) -> dict[tuple[int, int, int, int, int], dict[str, float]]:
    costs: dict[tuple[int, int, int, int, int], dict[str, float]] = {}
    for edge_idx, edge in enumerate(graph.edges):
        if edge.src is None or edge.dst is None:
            continue
        src_stage = _node_stage_id(graph, edge.src)
        dst_stage = _node_stage_id(graph, edge.dst)
        src_node = graph.nodes[src_stage]
        dst_node = graph.nodes[dst_stage]
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
    stage_plans: dict[int, StagePlan] | None = None,
) -> dict[tuple[int, int, int, int, int], dict[str, float]]:
    costs: dict[tuple[int, int, int, int, int], dict[str, float]] = {}
    for edge_idx, edge in enumerate(graph.edges):
        if edge.src is None or edge.dst is None:
            continue
        src_stage = _node_stage_id(graph, edge.src)
        dst_stage = _node_stage_id(graph, edge.dst)
        src_node = graph.nodes[src_stage]
        dst_node = graph.nodes[dst_stage]
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
    return costs


def _stage_io_costs(
    graph: Graph,
    shape_options: dict[int, tuple[tuple[int, int], ...]],
) -> dict[int, dict[int, dict[str, float]]]:
    producer_stage_ids = {_node_stage_id(graph, edge.src) for edge in graph.edges if edge.src is not None and edge.dst is not None}
    consumer_stage_ids = {_node_stage_id(graph, edge.dst) for edge in graph.edges if edge.src is not None and edge.dst is not None}

    io_costs: dict[int, dict[int, dict[str, float]]] = {}
    for stage_id, node in enumerate(graph.nodes):
        io_costs[stage_id] = {}
        for shape_idx, shape in enumerate(shape_options[stage_id]):
            mesh = Mesh(shape[0], shape[1], l2_memory=L2Memory(size=1))
            submesh = Submesh(mesh=mesh, submesh_id=stage_id, x0=0, y0=0, width=shape[0], height=shape[1])
            model = TransportCostModel(mesh=mesh)

            read_cost = 0.0
            if stage_id not in consumer_stage_ids:
                input_layouts = node.payload.default_input_layouts(submesh)
                read_cost = max(
                    (
                        _max_l2_read_cost(tensor, layout, model)
                        for tensor, layout in zip(node.inputs, input_layouts)
                    ),
                    default=0.0,
                )

            write_cost = 0.0
            if stage_id not in producer_stage_ids:
                output_layouts = node.payload.default_output_layouts(submesh)
                write_cost = max(
                    (
                        _max_l2_write_cost(tensor, layout, model)
                        for tensor, layout in zip(node.outputs, output_layouts)
                    ),
                    default=0.0,
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
    stage_plans: dict[int, StagePlan] | None = None,
) -> dict[int, dict[int, dict[str, float]]]:
    producer_stage_ids = {_node_stage_id(graph, edge.src) for edge in graph.edges if edge.src is not None and edge.dst is not None}
    consumer_stage_ids = {_node_stage_id(graph, edge.dst) for edge in graph.edges if edge.src is not None and edge.dst is not None}

    io_costs: dict[int, dict[int, dict[str, float]]] = {}
    for stage_id, node in enumerate(graph.nodes):
        io_costs[stage_id] = {}
        for placement_idx, submesh in enumerate(placement_options[stage_id]):
            model = TransportCostModel(mesh=submesh.mesh)

            read_cost = 0.0
            if stage_id not in consumer_stage_ids:
                plan = None if stage_plans is None else stage_plans[stage_id]
                input_layouts = (
                    node.payload.default_input_layouts(submesh)
                    if plan is None
                    else _layouts_on_submesh(plan.input_layouts, submesh)
                )
                read_cost = max(
                    (
                        _max_l2_read_cost(tensor, layout, model)
                        for tensor, layout in zip(node.inputs, input_layouts)
                    ),
                    default=0.0,
                )

            write_cost = 0.0
            if stage_id not in producer_stage_ids:
                plan = None if stage_plans is None else stage_plans[stage_id]
                output_layouts = (
                    node.payload.default_output_layouts(submesh)
                    if plan is None
                    else _layouts_on_submesh(plan.output_layouts, submesh)
                )
                write_cost = max(
                    (
                        _max_l2_write_cost(tensor, layout, model)
                        for tensor, layout in zip(node.outputs, output_layouts)
                    ),
                    default=0.0,
                )

            io_costs[stage_id][placement_idx] = {
                "read": read_cost,
                "write": write_cost,
                "total": read_cost + write_cost,
            }

    return io_costs


def _edge_shape_mode_costs(
    tensor: Tensor,
    src_node: Node,
    src_shape: tuple[int, int],
    dst_node: Node,
    dst_shape: tuple[int, int],
) -> dict[str, float]:
    mesh = Mesh(max(src_shape[0], dst_shape[0]), max(src_shape[1], dst_shape[1]), l2_memory=L2Memory(size=1))
    src_submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=src_shape[0], height=src_shape[1])
    dst_submesh = Submesh(mesh=mesh, submesh_id=1, x0=0, y0=0, width=dst_shape[0], height=dst_shape[1])

    src_output_layout = src_node.payload.default_output_layouts(src_submesh)[0]
    dst_input_layouts = dst_node.payload.default_input_layouts(dst_submesh)
    dst_output_layouts = dst_node.payload.default_output_layouts(dst_submesh)
    dst_input_idx = _node_input_index(dst_node, tensor)
    dst_input_layout = dst_input_layouts[dst_input_idx]

    transition = build_transition(
        name=f"spatial_map_{src_node.name}_to_{dst_node.name}_{tensor.name}",
        tensor=tensor,
        tensor_id=0,
        src_layer_id=0,
        src_output_idx=0,
        dst_layer_id=1,
        dst_input_idx=dst_input_idx,
        src_layout=src_output_layout,
        dst_layout=dst_input_layout,
    )
    transition_cost = estimate_transition_cost(
        transition=transition,
        tensor=tensor,
        mesh=mesh,
        model=TransportCostModel(mesh=mesh),
    )
    model = TransportCostModel(mesh=mesh)

    max_tile_to_l2_cost = 0.0
    for tile in src_submesh.tiles:
        tensor_slice = tile_tensor_slice(tensor, src_output_layout, tile)
        max_tile_to_l2_cost = max(
            max_tile_to_l2_cost,
            model.l1_to_l2(tile, _tensor_slice_num_bytes(tensor, tensor_slice)),
        )

    max_l2_to_tile_cost = 0.0
    for tile in dst_submesh.tiles:
        tile_work = dst_node.payload.build_tile_work(
            input_layouts=dst_input_layouts,
            output_layouts=dst_output_layouts,
            tile=tile,
        )
        required_slice = _required_input_slice_for_tensor(dst_node, tensor, tile_work)
        if required_slice is None:
            continue
        max_l2_to_tile_cost = max(
            max_l2_to_tile_cost,
            model.l2_to_l1(tile, _tensor_slice_num_bytes(tensor, required_slice)),
        )

    return {
        "l1": float(transition_cost.total_cost),
        "l2": float(max_tile_to_l2_cost + max_l2_to_tile_cost),
    }


def _edge_placement_mode_costs(
    tensor: Tensor,
    src_node: Node,
    src_submesh: Submesh,
    dst_node: Node,
    dst_submesh: Submesh,
    src_plan: StagePlan | None = None,
    dst_plan: StagePlan | None = None,
) -> dict[str, float]:
    src_output_layout = (
        src_node.payload.default_output_layouts(src_submesh)[0]
        if src_plan is None
        else _layout_on_submesh(src_plan.output_layouts[0], src_submesh)
    )
    dst_input_layouts = (
        dst_node.payload.default_input_layouts(dst_submesh)
        if dst_plan is None
        else _layouts_on_submesh(dst_plan.input_layouts, dst_submesh)
    )
    dst_input_idx = _node_input_index(dst_node, tensor)
    dst_input_layout = dst_input_layouts[dst_input_idx]

    transition = build_transition(
        name=f"spatial_map_{src_node.name}_to_{dst_node.name}_{tensor.name}",
        tensor=tensor,
        tensor_id=0,
        src_layer_id=0,
        src_output_idx=0,
        dst_layer_id=1,
        dst_input_idx=dst_input_idx,
        src_layout=src_output_layout,
        dst_layout=dst_input_layout,
    )
    model = TransportCostModel(mesh=src_submesh.mesh)
    transition_cost = estimate_transition_cost(
        transition=transition,
        tensor=tensor,
        mesh=src_submesh.mesh,
        model=model,
    )

    max_tile_to_l2_cost = 0.0
    for tile in src_submesh.tiles:
        tensor_slice = tile_tensor_slice(tensor, src_output_layout, tile)
        max_tile_to_l2_cost = max(
            max_tile_to_l2_cost,
            model.l1_to_l2(tile, _tensor_slice_num_bytes(tensor, tensor_slice)),
        )

    max_l2_to_tile_cost = 0.0
    dst_output_layouts = (
        dst_node.payload.default_output_layouts(dst_submesh)
        if dst_plan is None
        else _layouts_on_submesh(dst_plan.output_layouts, dst_submesh)
    )
    for tile in dst_submesh.tiles:
        tile_work = dst_node.payload.build_tile_work(
            input_layouts=dst_input_layouts,
            output_layouts=dst_output_layouts,
            tile=tile,
        )
        required_slice = _required_input_slice_for_tensor(dst_node, tensor, tile_work)
        if required_slice is None:
            continue
        max_l2_to_tile_cost = max(
            max_l2_to_tile_cost,
            model.l2_to_l1(tile, _tensor_slice_num_bytes(tensor, required_slice)),
        )

    return {
        "l1": float(transition_cost.total_cost),
        "l2": float(max_tile_to_l2_cost + max_l2_to_tile_cost),
    }


def _required_input_slice_for_tensor(node: Node, tensor: Tensor, tile_work: object) -> TensorSlice | None:
    op = node.payload
    if tensor == getattr(op, "x", None):
        return tile_work.x_slice
    if tensor == getattr(op, "w", None):
        return tile_work.w_slice
    if tensor == getattr(op, "y", None):
        return tile_work.y_slice
    return None


def _tensor_slice_num_bytes(tensor: Tensor, tensor_slice: TensorSlice) -> int:
    num_elements = 1
    for dim in tensor_slice.dims:
        num_elements *= dim.length
    return num_elements * tensor.elem_bytes


def _max_slice_bytes(
    tensor: Tensor,
    layout,
) -> int:
    return max(
        (
            _tensor_slice_num_bytes(
                tensor,
                tile_tensor_slice(tensor, layout, tile),
            )
            for tile in layout.submesh.tiles
        ),
        default=0,
    )


def _max_l2_write_cost(
    tensor: Tensor,
    layout,
    model: TransportCostModel,
) -> float:
    return max(
        (
            model.l1_to_l2(
                tile,
                _tensor_slice_num_bytes(
                    tensor,
                    tile_tensor_slice(tensor, layout, tile),
                ),
            )
            for tile in layout.submesh.tiles
        ),
        default=0.0,
    )


def _max_l2_read_cost(
    tensor: Tensor,
    layout,
    model: TransportCostModel,
) -> float:
    return max(
        (
            model.l2_to_l1(
                tile,
                _tensor_slice_num_bytes(
                    tensor,
                    tile_tensor_slice(tensor, layout, tile),
                ),
            )
            for tile in layout.submesh.tiles
        ),
        default=0.0,
    )


def _node_input_index(node: Node, tensor: Tensor) -> int:
    for idx, candidate in enumerate(node.inputs):
        if candidate == tensor:
            return idx
    raise ValueError(f"tensor {tensor.name} is not an input of node {node.name}")
