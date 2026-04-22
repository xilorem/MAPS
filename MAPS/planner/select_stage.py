"""Stage-group selection helpers."""

from __future__ import annotations

from MAPS.core.graph import Graph, Node

STAGE_GROUP_ID_ATTR = "stage_group_id"

StageSelection = dict[int, tuple[Node, ...]]


def select_stages(graph: Graph) -> StageSelection:
    """Select stage-local node groups from a graph.

    Nodes are grouped conservatively:
    - nodes with the same explicit ``stage_group_id`` attribute share one stage
    - nodes without that attribute form singleton stages

    Returned stage ids are dense and assigned in first-appearance order so the
    grouping preserves the graph's topological node order.
    """

    grouped_nodes: dict[int, list[Node]] = {}
    stage_id_by_group_key: dict[object, int] = {}
    next_stage_id = 0

    for node_idx, node in enumerate(graph.nodes):
        explicit_group_key = _explicit_stage_group_key(node)
        if explicit_group_key is None:
            grouped_nodes[next_stage_id] = [node]
            next_stage_id += 1
            continue

        stage_id = stage_id_by_group_key.get(explicit_group_key)
        if stage_id is None:
            stage_id = next_stage_id
            stage_id_by_group_key[explicit_group_key] = stage_id
            grouped_nodes[stage_id] = []
            next_stage_id += 1

        grouped_nodes[stage_id].append(node)

    return {
        stage_id: tuple(nodes)
        for stage_id, nodes in grouped_nodes.items()
    }


def _explicit_stage_group_key(node: Node) -> object | None:
    """Return one node's explicit stage-group key, when present."""

    if STAGE_GROUP_ID_ATTR not in node.attributes:
        return None

    group_key = node.attributes[STAGE_GROUP_ID_ATTR]
    try:
        hash(group_key)
    except TypeError as exc:
        raise ValueError(
            f"node {node.name} has an unhashable {STAGE_GROUP_ID_ATTR}: {group_key!r}"
        ) from exc
    return group_key
