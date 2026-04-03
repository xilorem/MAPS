"""Workload-balancing layer for stage-to-tile-count allocation."""

from __future__ import annotations

from MAPS.cost_models import cost_estimator
from MAPS.arch import Mesh
from MAPS.core.graph import Graph, Node, OpKind
from MAPS.core.layout import TensorSlice
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor


def balance_workload(graph: Graph, mesh: Mesh, debug: bool = False) -> dict[int, int]:
    """Greedily assign tile counts to nodes by internal rectangle growth.

    Internally, each node grows through rectangle candidates so the cost
    estimator remains shape-aware. The returned result keeps only
    ``stage_id -> tile_count`` so later spatial mapping still chooses the final
    concrete shapes.
    """

    if len(graph.nodes) > mesh.num_tiles:
        raise ValueError("graph has more nodes than available tiles")

    shapes = {stage_id: (1, 1) for stage_id, _ in enumerate(graph.nodes)}
    used_tiles = len(graph.nodes)
    iteration = 0

    _debug(debug, f"[balance_workload] start used_tiles={used_tiles}/{mesh.num_tiles}")
    _debug(debug, f"[balance_workload] initial_shapes={shapes}")

    while used_tiles < mesh.num_tiles:
        iteration += 1
        best_stage_id: int | None = None
        best_shape: tuple[int, int] | None = None
        best_improvement = 0.0
        current_workloads = _estimate_workloads(graph, mesh, shapes, debug=debug)
        best_current_workload = 0.0

        _debug(debug, f"[balance_workload] iteration={iteration} used_tiles={used_tiles}/{mesh.num_tiles}")
        _debug(debug, f"[balance_workload] current_workloads={current_workloads}")

        for stage_id, node in enumerate(graph.nodes):
            current_shape = shapes[stage_id]
            current_workload = current_workloads[stage_id]
            stage_best_shape: tuple[int, int] | None = None
            stage_best_improvement = 0.0

            _debug(
                debug,
                "[balance_workload] "
                f"stage={stage_id} node={node.name} current_shape={current_shape} "
                f"current_workload={current_workload}",
            )

            for candidate_shape in _grow_shape_options(current_shape, mesh):
                added_tiles = _shape_area(candidate_shape) - _shape_area(current_shape)
                if used_tiles + added_tiles > mesh.num_tiles:
                    continue

                candidate_shapes = dict(shapes)
                candidate_shapes[stage_id] = candidate_shape
                candidate_workloads = _estimate_workloads(
                    graph,
                    mesh,
                    candidate_shapes,
                    debug=debug,
                )
                candidate_workload = candidate_workloads[stage_id]
                improvement = current_workload - candidate_workload
                _debug(
                    debug,
                    "[balance_workload] "
                    f"stage={stage_id} candidate_shape={candidate_shape} "
                    f"candidate_workload={candidate_workload} improvement={improvement}",
                )
                if improvement > stage_best_improvement:
                    stage_best_shape = candidate_shape
                    stage_best_improvement = improvement

            if stage_best_shape is None:
                _debug(debug, f"[balance_workload] stage={stage_id} no_valid_growth")
                continue

            _debug(
                debug,
                "[balance_workload] "
                f"stage={stage_id} best_stage_shape={stage_best_shape} "
                f"best_stage_improvement={stage_best_improvement}",
            )

            if stage_best_improvement > best_improvement or (
                stage_best_improvement == best_improvement
                and current_workload > best_current_workload
            ):
                best_stage_id = stage_id
                best_shape = stage_best_shape
                best_improvement = stage_best_improvement
                best_current_workload = current_workload

        if best_stage_id is None or best_shape is None:
            _debug(debug, "[balance_workload] no_global_improvement_available")
            break

        _debug(
            debug,
            "[balance_workload] "
            f"choose stage={best_stage_id} new_shape={best_shape} "
            f"improvement={best_improvement}",
        )
        used_tiles += _shape_area(best_shape) - _shape_area(shapes[best_stage_id])
        shapes[best_stage_id] = best_shape

    allocation = {stage_id: _shape_area(shape) for stage_id, shape in shapes.items()}
    _debug(debug, f"[balance_workload] final_shapes={shapes}")
    _debug(debug, f"[balance_workload] final_allocation={allocation}")
    return allocation


def _shape_area(shape: tuple[int, int]) -> int:
    return shape[0] * shape[1]


def _grow_shape_options(shape: tuple[int, int], mesh: Mesh) -> tuple[tuple[int, int], ...]:
    width, height = shape
    options = []
    if width < mesh.width:
        options.append((width + 1, height))
    if height < mesh.height:
        options.append((width, height + 1))
    return tuple(options)


def _default_layouts_for_node(
    node: Node,
    submesh: Submesh,
    num_microbatches: int,
) -> tuple[tuple, tuple]:
    return (
        node.payload.default_input_layouts(submesh, num_microbatches=num_microbatches),
        node.payload.default_output_layouts(submesh, num_microbatches=num_microbatches),
    )


def _estimate_workloads(
    graph: Graph,
    mesh: Mesh,
    shapes: dict[int, tuple[int, int]],
    debug: bool,
) -> dict[int, float]:
    local_num_microbatches = _estimate_local_num_microbatches(graph, mesh, shapes, debug=debug)
    num_microbatches = _propagate_num_microbatches(
        graph,
        local_num_microbatches,
        debug=debug,
    )
    workloads: dict[int, float] = {}
    for stage_id, node in enumerate(graph.nodes):
        shape = shapes[stage_id]
        submesh = Submesh(
            mesh=mesh,
            submesh_id=0,
            x0=0,
            y0=0,
            width=shape[0],
            height=shape[1],
        )
        input_layouts, output_layouts = _default_layouts_for_node(
            node,
            submesh,
            num_microbatches[stage_id],
        )
        step_cost = cost_estimator(
            node=node,
            input_layouts=input_layouts,
            output_layouts=output_layouts,
            microbatch_idx=0,
        )
        workloads[stage_id] = num_microbatches[stage_id] * step_cost
    return workloads


def _propagate_num_microbatches(
    graph: Graph,
    local_num_microbatches: dict[int, int],
    debug: bool,
) -> dict[int, int]:
    producer_by_tensor = {tensor: stage_id for stage_id, node in enumerate(graph.nodes) for tensor in node.outputs}
    num_microbatches: dict[int, int] = {}

    for stage_id in _topological_stage_ids(graph):
        node = graph.nodes[stage_id]
        local_min = local_num_microbatches[stage_id]
        predecessor_min = max(
            (num_microbatches[producer_by_tensor[tensor]] for tensor in node.inputs if tensor in producer_by_tensor),
            default=1,
        )
        num_microbatches[stage_id] = max(local_min, predecessor_min)
        _debug(
            debug,
            "[balance_workload] "
            f"stage={stage_id} node={node.name} "
            f"local_num_microbatches={local_min} "
            f"predecessor_num_microbatches={predecessor_min} "
            f"propagated_num_microbatches={num_microbatches[stage_id]}",
        )

    return num_microbatches


def _estimate_local_num_microbatches(
    graph: Graph,
    mesh: Mesh,
    shapes: dict[int, tuple[int, int]],
    debug: bool,
) -> dict[int, int]:
    local_num_microbatches: dict[int, int] = {}

    for stage_id, node in enumerate(graph.nodes):
        shape = shapes[stage_id]
        submesh = Submesh(
            mesh=mesh,
            submesh_id=0,
            x0=0,
            y0=0,
            width=shape[0],
            height=shape[1],
        )
        local_num_microbatches[stage_id] = _estimate_node_num_microbatches(
            node,
            submesh,
            debug=debug,
        )

    return local_num_microbatches


def _estimate_node_num_microbatches(node: Node, submesh: Submesh, debug: bool = False) -> int:
    max_num_microbatches = _max_supported_num_microbatches(node)
    num_microbatches = 1
    while num_microbatches <= max_num_microbatches:
        input_layouts, output_layouts = _default_layouts_for_node(
            node,
            submesh,
            num_microbatches,
        )
        if _inputs_fit_in_tile_buffers(node, input_layouts, output_layouts):
            _debug(
                debug,
                "[balance_workload] "
                f"node={node.name} shape={(submesh.width, submesh.height)} "
                f"num_microbatches={num_microbatches} fits_in_l1=True",
            )
            return num_microbatches
        _debug(
            debug,
            "[balance_workload] "
            f"node={node.name} shape={(submesh.width, submesh.height)} "
            f"num_microbatches={num_microbatches} fits_in_l1=False",
        )
        num_microbatches += 1
    raise ValueError(f"node {node.name} inputs do not fit tile buffers for any supported num_microbatches")


def _inputs_fit_in_tile_buffers(
    node: Node,
    input_layouts: tuple,
    output_layouts: tuple,
) -> bool:
    submesh = output_layouts[0].submesh
    for tile in submesh.tiles:
        tile_work = node.payload.build_tile_work(
            input_layouts=input_layouts,
            output_layouts=output_layouts,
            tile=tile,
            microbatch_idx=0,
        )
        if _input_tile_work_bytes(node, tile_work) > tile.l1_bytes:
            return False
    return True


def _input_tile_work_bytes(node: Node, tile_work: object) -> int:
    op = node.payload
    total = 0
    for tensor_name in ("x", "w", "y"):
        tensor = getattr(op, tensor_name, None)
        tensor_slice = getattr(tile_work, f"{tensor_name}_slice", None)
        if tensor is not None and tensor_slice is not None:
            total += _tensor_slice_num_bytes(tensor, tensor_slice)
    return total


def _tensor_slice_num_bytes(tensor: Tensor, tensor_slice: TensorSlice) -> int:
    num_elements = 1
    for dim in tensor_slice.dims:
        num_elements *= dim.length
    return num_elements * tensor.elem_bytes


def _max_supported_num_microbatches(node: Node) -> int:
    candidate_dims = [tensor.dims[0] for tensor in (*node.inputs, *node.outputs) if tensor.rank > 2]
    return min(candidate_dims) if candidate_dims else 1


def _debug(enabled: bool, message: str) -> None:
    if enabled:
        print(message)


def _topological_stage_ids(graph: Graph) -> tuple[int, ...]:
    node_ids = {id(node): stage_id for stage_id, node in enumerate(graph.nodes)}
    indegree = {stage_id: 0 for stage_id in range(len(graph.nodes))}
    adjacency = {stage_id: set() for stage_id in range(len(graph.nodes))}

    for edge in graph.edges:
        if edge.src is None or edge.dst is None:
            continue
        src_id = node_ids[id(edge.src)]
        dst_id = node_ids[id(edge.dst)]
        if dst_id not in adjacency[src_id]:
            adjacency[src_id].add(dst_id)
            indegree[dst_id] += 1

    ready = [stage_id for stage_id in range(len(graph.nodes)) if indegree[stage_id] == 0]
    order: list[int] = []

    while ready:
        stage_id = ready.pop(0)
        order.append(stage_id)
        for dst_id in sorted(adjacency[stage_id]):
            indegree[dst_id] -= 1
            if indegree[dst_id] == 0:
                ready.append(dst_id)

    if len(order) != len(graph.nodes):
        raise ValueError("graph contains a cycle or dangling node references")

    return tuple(order)
