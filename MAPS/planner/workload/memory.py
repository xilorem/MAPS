"""L1 footprint estimation for virtual stage candidates."""

from __future__ import annotations


def peak_l1_occupancy_for_stage(
    stage_nodes: tuple,
    node_output_layouts: tuple[tuple, ...],
    submesh,
    initializer_tensors: frozenset,
) -> int:
    """Return the greatest estimated L1 footprint on any virtual stage tile.

    Initializer slices persist across nodes and are counted once per tile.  For
    non-initializer data, nodes execute sequentially, so the stage footprint uses
    the largest node-local input/output footprint rather than their sum.
    """

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
    stage_nodes: tuple,
    node_output_layouts: tuple[tuple, ...],
    tile,
    initializer_tensors: frozenset,
) -> int:
    """Combine persistent initializer bytes and peak dynamic bytes on one tile."""

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
    """Split one node's referenced slices into new initializer and dynamic bytes."""

    initializer_memory = 0
    node_dynamic_memory = 0
    for reference in input_slices:
        if reference.tensor in initializer_tensors or getattr(
            reference.tensor,
            "is_initializer",
            False,
        ):
            key = (id(reference.tensor), reference.tensor_slice)
            if key not in seen_initializer_slices:
                seen_initializer_slices.add(key)
                initializer_memory += reference.num_bytes
        else:
            node_dynamic_memory += reference.num_bytes
    for reference in output_slices:
        node_dynamic_memory += reference.num_bytes
    return initializer_memory, node_dynamic_memory
