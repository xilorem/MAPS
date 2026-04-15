"""Workload-balancing layer for stage-to-tile-count allocation."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.cost_models import cost_estimator
from MAPS.arch import Mesh
from MAPS.core.graph import Graph, Node, OpKind
from MAPS.core.layout import TensorSlice
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor


@dataclass(frozen=True)
class StagePlan:
    """Compute-layout decisions chosen before physical placement."""

    stage_id: int
    tile_count: int
    logical_shape: tuple[int, int]
    input_layouts: tuple
    output_layouts: tuple


def balance_workload(graph: Graph, mesh: Mesh, debug: bool = False) -> dict[int, int]:
    plans = balance_stage_plans(graph, mesh, debug=debug)
    return {stage_id: plan.tile_count for stage_id, plan in plans.items()}


def balance_stage_plans(graph: Graph, mesh: Mesh, debug: bool = False) -> dict[int, StagePlan]:
    """Greedily assign tile counts to nodes by internal rectangle growth.

    Each node searches physically placeable tile counts and all logical factor
    pairs for that count. The returned plan keeps the chosen logical compute
    shape and layouts; later spatial mapping still chooses physical rectangles.
    """

    if len(graph.nodes) > mesh.num_tiles:
        raise ValueError("graph has more nodes than available tiles")

    tile_counts = {stage_id: 1 for stage_id, _ in enumerate(graph.nodes)}
    used_tiles = len(graph.nodes)
    iteration = 0

    _debug(debug, f"[balance_workload] start used_tiles={used_tiles}/{mesh.num_tiles}")
    _debug(debug, f"[balance_workload] initial_tile_counts={tile_counts}")

    while used_tiles < mesh.num_tiles:
        iteration += 1
        best_stage_id: int | None = None
        best_tile_count: int | None = None
        best_improvement = 0.0
        current_plans = _build_stage_plans(graph, mesh, tile_counts, debug=debug)
        current_workloads = _estimate_workloads(current_plans, graph, debug=debug)
        best_current_workload = 0.0

        _debug(debug, f"[balance_workload] iteration={iteration} used_tiles={used_tiles}/{mesh.num_tiles}")
        _debug(debug, f"[balance_workload] current_workloads={current_workloads}")

        for stage_id, node in enumerate(graph.nodes):
            current_tile_count = tile_counts[stage_id]
            current_workload = current_workloads[stage_id]
            stage_best_tile_count: int | None = None
            stage_best_improvement = 0.0

            _debug(
                debug,
                "[balance_workload] "
                f"stage={stage_id} node={node.name} current_tile_count={current_tile_count} "
                f"current_logical_shape={current_plans[stage_id].logical_shape} "
                f"current_workload={current_workload}",
            )

            for candidate_tile_count in _tile_count_options_after_growth(
                current_tile_count,
                mesh.num_tiles - used_tiles,
                mesh,
            ):
                added_tiles = candidate_tile_count - current_tile_count
                if used_tiles + added_tiles > mesh.num_tiles:
                    continue

                candidate_tile_counts = dict(tile_counts)
                candidate_tile_counts[stage_id] = candidate_tile_count
                candidate_plans = _build_stage_plans(
                    graph,
                    mesh,
                    candidate_tile_counts,
                    debug=debug,
                )
                candidate_plan = candidate_plans[stage_id]
                candidate_workloads = _estimate_workloads(candidate_plans, graph, debug=debug)
                candidate_workload = candidate_workloads[stage_id]
                improvement = current_workload - candidate_workload
                _debug(
                    debug,
                    "[balance_workload] "
                    f"stage={stage_id} candidate_tile_count={candidate_tile_count} "
                    f"candidate_logical_shape={candidate_plan.logical_shape} "
                    f"candidate_workload={candidate_workload} improvement={improvement}",
                )
                if improvement > stage_best_improvement:
                    stage_best_tile_count = candidate_tile_count
                    stage_best_improvement = improvement

            if stage_best_tile_count is None:
                _debug(debug, f"[balance_workload] stage={stage_id} no_valid_growth")
                continue

            _debug(
                debug,
                "[balance_workload] "
                f"stage={stage_id} best_stage_tile_count={stage_best_tile_count} "
                f"best_stage_improvement={stage_best_improvement}",
            )

            if stage_best_improvement > best_improvement or (
                stage_best_improvement == best_improvement
                and current_workload > best_current_workload
            ):
                best_stage_id = stage_id
                best_tile_count = stage_best_tile_count
                best_improvement = stage_best_improvement
                best_current_workload = current_workload

        if best_stage_id is None or best_tile_count is None:
            _debug(debug, "[balance_workload] no_global_improvement_available")
            break

        _debug(
            debug,
            "[balance_workload] "
            f"choose stage={best_stage_id} new_tile_count={best_tile_count} "
            f"improvement={best_improvement}",
        )
        used_tiles += best_tile_count - tile_counts[best_stage_id]
        tile_counts[best_stage_id] = best_tile_count

    plans = _build_stage_plans(graph, mesh, tile_counts, debug=debug)
    _debug(debug, f"[balance_workload] final_tile_counts={tile_counts}")
    _debug(debug, f"[balance_workload] final_logical_shapes={ {stage_id: plan.logical_shape for stage_id, plan in plans.items()} }")
    _debug(debug, f"[balance_workload] final_allocation={ {stage_id: plan.tile_count for stage_id, plan in plans.items()} }")
    return plans


def _shape_area(shape: tuple[int, int]) -> int:
    return shape[0] * shape[1]


def _tile_count_options_after_growth(
    current_tile_count: int,
    remaining_tiles: int,
    mesh: Mesh,
) -> tuple[int, ...]:
    for tile_count in range(current_tile_count + 1, current_tile_count + remaining_tiles + 1):
        if not _physical_shape_options(tile_count, mesh):
            continue
        return (tile_count,)
    return ()


def _logical_shape_options(tile_count: int) -> tuple[tuple[int, int], ...]:
    options = []
    for height in range(1, tile_count + 1):
        if tile_count % height == 0:
            options.append((tile_count // height, height))
    return tuple(options)


def _physical_shape_options(tile_count: int, mesh: Mesh) -> tuple[tuple[int, int], ...]:
    options = []
    for height in range(1, mesh.height + 1):
        if tile_count % height != 0:
            continue
        width = tile_count // height
        if 0 < width <= mesh.width:
            options.append((width, height))
    return tuple(options)


def _planning_submesh(mesh: Mesh, stage_id: int, tile_count: int) -> Submesh:
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


def _build_stage_plans(
    graph: Graph,
    mesh: Mesh,
    tile_counts: dict[int, int],
    debug: bool,
) -> dict[int, StagePlan]:
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
    submesh = _planning_submesh(mesh, stage_id, tile_count)
    best_plan: StagePlan | None = None
    best_workload: float | None = None

    for logical_shape in _logical_shape_options(tile_count):
        input_layouts, output_layouts = _default_layouts_for_node(
            node,
            submesh,
            logical_shape,
        )
        if not _inputs_fit_in_tile_buffers(node, input_layouts, output_layouts):
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
            "[balance_workload] "
            f"stage={stage_id} tile_count={tile_count} "
            f"logical_shape={logical_shape} workload={workload}",
        )
        if best_workload is None or workload < best_workload:
            best_plan = plan
            best_workload = workload

    if best_plan is None:
        raise ValueError(
            f"node {node.name} has no valid logical shape for tile_count={tile_count} "
            "using full input slices"
        )
    return best_plan


def _estimate_workloads(
    plans: dict[int, StagePlan],
    graph: Graph,
    debug: bool,
) -> dict[int, float]:
    workloads: dict[int, float] = {}
    for stage_id, node in enumerate(graph.nodes):
        workloads[stage_id] = _estimate_stage_workload(node, plans[stage_id])
    return workloads


def _estimate_stage_workload(node: Node, plan: StagePlan) -> float:
    step_cost = cost_estimator(
        node=node,
        input_layouts=plan.input_layouts,
        output_layouts=plan.output_layouts,
    )
    return step_cost


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
        )
        if _input_tile_work_bytes(node, tile_work) > tile.memory.size:
            return False
    return True


def _input_tile_work_bytes(node: Node, tile_work: object) -> int:
    input_slices = getattr(tile_work, "input_slices", None)
    if input_slices is not None:
        return sum(
            _tensor_slice_num_bytes(ref.tensor, ref.tensor_slice)
            for ref in input_slices
        )

    op = node.payload
    total = 0
    for tensor_name in ("x", "w", "y", "b"):
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
