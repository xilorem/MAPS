"""Collective communication cost models."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch import EndpointKind, Mesh, NoC, NoCChannel, NoCEndpoint, NoCLink, NoCNode
from MAPS.cost_models.transport_cost import TransportCostModel
from MAPS.core.graph import Node
from MAPS.core.layout import TensorLayout
from MAPS.core.ownership import tile_tensor_slice


@dataclass(frozen=True)
class AllReduceCostModel:
    """Placement-sensitive allreduce latency model."""

    reduction: str
    collective_axis: str = "x"

    def __post_init__(self) -> None:
        if self.reduction not in {"sum", "max"}:
            raise ValueError("AllReduceCostModel reduction must be 'sum' or 'max'")
        if self.collective_axis not in {"x", "y"}:
            raise ValueError("AllReduceCostModel collective_axis must be 'x' or 'y'")

    def cost(self, tile_work: object, tile: object) -> float:
        del tile_work, tile
        return 0.0

    def placement_cost(
        self,
        *,
        node: Node,
        input_layouts: tuple[TensorLayout, ...],
        output_layouts: tuple[TensorLayout, ...],
    ) -> float:
        if len(output_layouts) != 1:
            raise ValueError("AllReduceCostModel expects exactly one output layout")

        output_layout = output_layouts[0]
        output_tensor = node.outputs[0]
        comm_mesh = _communication_mesh(output_layout.submesh.mesh)
        model = TransportCostModel(mesh=comm_mesh)
        groups = _logical_collective_groups(output_layout, self.collective_axis)
        group_costs = []
        for group_tiles in groups:
            payload_tiles = tuple(
                tile
                for tile in group_tiles
                if output_tensor.slice_num_bytes(tile_tensor_slice(output_tensor, output_layout, tile)) > 0
            )
            if len(payload_tiles) <= 1:
                group_costs.append(0.0)
                continue

            root_tile = payload_tiles[0]
            reduce_phase = max(
                (
                    model.l1_to_l1(
                        tile,
                        root_tile,
                        output_tensor.slice_num_bytes(tile_tensor_slice(output_tensor, output_layout, tile)),
                    )
                    for tile in payload_tiles[1:]
                ),
                default=0.0,
            )
            broadcast_phase = max(
                (
                    model.l1_to_l1(
                        root_tile,
                        tile,
                        output_tensor.slice_num_bytes(tile_tensor_slice(output_tensor, output_layout, tile)),
                    )
                    for tile in payload_tiles[1:]
                ),
                default=0.0,
            )
            group_costs.append(reduce_phase + broadcast_phase)
        return max(group_costs, default=0.0)


def _logical_collective_groups(
    layout: TensorLayout,
    collective_axis: str,
) -> tuple[tuple[object, ...], ...]:
    logical_width = layout.effective_logical_width
    logical_height = layout.effective_logical_height
    groups: dict[int, list[object]] = {}
    for tile_ordinal, tile in enumerate(layout.submesh.tiles):
        logical_x = tile_ordinal % logical_width
        logical_y = tile_ordinal // logical_width
        group_key = logical_y if collective_axis == "x" else logical_x
        groups.setdefault(group_key, []).append(tile)

    return tuple(
        tuple(group_tiles)
        for _, group_tiles in sorted(groups.items())
    )


def _default_noc_node_id(x: int, y: int, width: int) -> int:
    return y * width + x


def _default_communication_noc(width: int, height: int) -> NoC:
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
            channels=(NoCChannel(channel_id=0, width_bytes=1, hop_latency_cycles=0.5),),
            bidirectional=True,
        )
        for link_id, (src_node_id, dst_node_id) in enumerate(link_pairs)
    )
    endpoints = tuple(
        NoCEndpoint(
            endpoint_id=tile_id,
            kind=EndpointKind.L1,
            node_id=tile_id,
            tile_id=tile_id,
        )
        for tile_id in range(width * height)
    )
    return NoC(nodes=nodes, links=links, endpoints=endpoints)


def _communication_mesh(mesh: Mesh) -> Mesh:
    if mesh.has_noc:
        return mesh
    return Mesh(
        width=mesh.width,
        height=mesh.height,
        l2_memory=mesh.l2_memory,
        tiles=mesh.tiles,
        noc=_default_communication_noc(mesh.width, mesh.height),
    )
