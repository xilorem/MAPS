"""Transition-level aggregation built on top of primitive transfer legs."""

from __future__ import annotations

from dataclasses import dataclass, field

from MAPS.arch import Mesh
from MAPS.core.layout import TensorSlice
from MAPS.core.tensor import Tensor
from MAPS.core.transition import Transition, TransitionFragment, TransitionMode
from MAPS.cost_models.transport_cost import TransferKind, TransferLeg, TransportCostModel


@dataclass(frozen=True)
class TransitionCost:
    """Aggregated cost of one transition."""

    mode: TransitionMode
    total_bytes: int
    legs: tuple[TransferLeg, ...] = field(default_factory=tuple)
    producer_loads: dict[int, int] = field(default_factory=dict)
    consumer_loads: dict[int, int] = field(default_factory=dict)
    resource_loads: dict[str, int] = field(default_factory=dict)
    total_cost: int = 0


def _tensor_slice_num_elements(tensor_slice: TensorSlice) -> int:
    total = 1
    for dim in tensor_slice.dims:
        total *= dim.length
    return total


def _transition_fragment_num_bytes(fragment: TransitionFragment,
                                   tensor: Tensor) -> int:
    return _tensor_slice_num_elements(fragment.src_slice) * tensor.elem_bytes


def _aggregate_transition(legs: tuple[TransferLeg, ...],
                          model: TransportCostModel) -> TransitionCost:
    
    """ Computes the time necessary for a remap transition. This is
    obtained by computing the maximum between the maximum fragment transfer
    costs of producer and consumer tiles.
    """
    producer_loads: dict[int, int] = {}
    consumer_loads: dict[int, int] = {}
    resource_loads: dict[str, int] = {}

    for leg in legs:
        estimate = model.estimate(leg)
        cost = estimate.total_cost

        if leg.src_tile is not None:
            producer_loads[leg.src_tile.tile_id] = (
                producer_loads.get(leg.src_tile.tile_id, 0) + cost
            )
        if leg.dst_tile is not None:
            consumer_loads[leg.dst_tile.tile_id] = (
                consumer_loads.get(leg.dst_tile.tile_id, 0) + cost
            )
        for resource_id, load in estimate.resource_loads.items():
            resource_loads[resource_id] = resource_loads.get(resource_id, 0) + load

    total_cost = max(
        max(producer_loads.values(), default=0),
        max(consumer_loads.values(), default=0),
        max(resource_loads.values(), default=0),
    )

    return TransitionCost(
        mode=TransitionMode.DIRECT_REMAP,
        total_bytes=sum(leg.bytes for leg in legs),
        legs=legs,
        producer_loads=producer_loads,
        consumer_loads=consumer_loads,
        resource_loads=resource_loads,
        total_cost=total_cost,
    )


def _build_direct_remap_legs(transition: Transition,
                             tensor: Tensor,
                             mesh: Mesh) -> tuple[TransferLeg, ...]:
    
    """Builds the TransferLeg (fragment movements) considering the tensor
    """
    legs: list[TransferLeg] = []

    for fragment in transition.fragments:
        legs.append(
            TransferLeg(
                kind=TransferKind.L1_TO_L1,
                bytes=_transition_fragment_num_bytes(fragment, tensor),
                src_tile=mesh.tile_by_id(fragment.src_hartid),
                dst_tile=mesh.tile_by_id(fragment.dst_hartid),
            )
        )

    return tuple(legs)


def estimate_transition_cost(transition: Transition,
                             tensor: Tensor,
                             mesh: Mesh,
                             model: TransportCostModel) -> TransitionCost:
    """
    Estimate the cost of one transition.

    DIRECT_REMAP:
    - one round of L1-to-L1 fragment transfers
    """

    transition.validate_for(tensor)

    if transition.mode is TransitionMode.DIRECT_REMAP:
        legs = _build_direct_remap_legs(transition, tensor, mesh)
        return _aggregate_transition(legs, model)

    raise ValueError(f"unsupported transition mode: {transition.mode}")
