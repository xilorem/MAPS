from MAPS.core.graph import Graph, Node, OpKind
from MAPS.planner.select_stage import select_stages


def test_select_stages_defaults_to_singleton_groups() -> None:
    node0 = Node(name="n0", kind=OpKind.CUSTOM)
    node1 = Node(name="n1", kind=OpKind.CUSTOM)
    node2 = Node(name="n2", kind=OpKind.CUSTOM)
    graph = Graph(
        name="singleton_groups",
        nodes=(node0, node1, node2),
    )

    groups = select_stages(graph)

    assert groups == {
        0: (node0,),
        1: (node1,),
        2: (node2,),
    }


def test_select_stages_groups_nodes_with_same_explicit_stage_group_id() -> None:
    node0 = Node(
        name="reduce_max",
        kind=OpKind.REDUCTION,
        attributes={"stage_group_id": "softmax_0"},
    )
    node1 = Node(
        name="allreduce_max",
        kind=OpKind.CUSTOM,
        attributes={"stage_group_id": "softmax_0"},
    )
    node2 = Node(name="next_stage", kind=OpKind.CUSTOM)
    graph = Graph(
        name="grouped_nodes",
        nodes=(node0, node1, node2),
    )

    groups = select_stages(graph)

    assert groups == {
        0: (node0, node1),
        1: (node2,),
    }


def test_select_stages_rejects_unhashable_explicit_group_keys() -> None:
    graph = Graph(
        name="bad_group_key",
        nodes=(
            Node(
                name="softmax_step",
                kind=OpKind.CUSTOM,
                attributes={"stage_group_id": {"bad": "key"}},
            ),
        ),
    )

    try:
        select_stages(graph)
    except ValueError as exc:
        assert "unhashable stage_group_id" in str(exc)
    else:
        raise AssertionError("expected invalid stage_group_id to fail")
