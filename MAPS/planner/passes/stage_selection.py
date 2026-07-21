"""Stage-selection pass facade."""

from __future__ import annotations

from MAPS.core.graph import Graph, Node
from MAPS.planner.contracts.stages import StageSelection

STAGE_GROUP_ID_ATTR = "stage_group_id"


def select_stages(graph: Graph) -> StageSelection:
    """Partition graph nodes into ordered planner stages.

    Contract:
        ``graph.nodes`` must be in topological execution order.  An optional
        ``stage_group_id`` node attribute may explicitly request that multiple
        nodes execute in one stage.

    Behavior:
        Nodes sharing the same explicit group id are grouped together.  Nodes
        without a group id form singleton stages.  Stage ids are dense and
        assigned by first appearance, preserving graph order.

    Returns:
        A stage-id mapping whose node tuples cover every graph node exactly once.
    """

    grouped_nodes: dict[int, list[Node]] = {}
    stage_id_by_group_key: dict[object, int] = {}
    next_stage_id = 0
    for node in graph.nodes:
        group_key = _explicit_stage_group_key(node)
        if group_key is None:
            grouped_nodes[next_stage_id] = [node]
            next_stage_id += 1
            continue
        stage_id = stage_id_by_group_key.get(group_key)
        if stage_id is None:
            stage_id = next_stage_id
            stage_id_by_group_key[group_key] = stage_id
            grouped_nodes[stage_id] = []
            next_stage_id += 1
        grouped_nodes[stage_id].append(node)
    return {
        stage_id: tuple(nodes)
        for stage_id, nodes in grouped_nodes.items()
    }


def _explicit_stage_group_key(node: Node) -> object | None:
    """Return one node's validated explicit grouping key, when present."""

    if STAGE_GROUP_ID_ATTR not in node.attributes:
        return None
    group_key = node.attributes[STAGE_GROUP_ID_ATTR]
    try:
        hash(group_key)
    except TypeError as exc:
        raise ValueError(
            f"node {node.name} has an unhashable "
            f"{STAGE_GROUP_ID_ATTR}: {group_key!r}"
        ) from exc
    return group_key


__all__ = ["select_stages"]
