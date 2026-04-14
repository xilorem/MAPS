"""Helpers to build transition objects from layout ownership."""

from MAPS.core.transition import Transition, TransitionMode
from MAPS.core.layout import TensorLayout
from MAPS.core.tensor import Tensor

from MAPS.builders.remap_builder import build_direct_remap_fragments


def build_transition(
    name: str,
    tensor: Tensor,
    tensor_id: int,
    src_layer_id: int,
    src_output_idx: int,
    dst_layer_id: int,
    dst_input_idx: int,
    src_layout: TensorLayout,
    dst_layout: TensorLayout,
) -> Transition:
    """Build one concrete transition instance."""

    src_layout.validate_for(tensor)
    dst_layout.validate_for(tensor)

    if src_layout == dst_layout:
        return Transition(
            name=name,
            tensor_id=tensor_id,
            src_layer_id=src_layer_id,
            src_output_idx=src_output_idx,
            dst_layer_id=dst_layer_id,
            dst_input_idx=dst_input_idx,
            mode=TransitionMode.LOCAL_REUSE,
            src_layout=src_layout,
            dst_layout=dst_layout,
            fragments=(),
        )

    fragments = build_direct_remap_fragments(
        tensor=tensor,
        src_layout=src_layout,
        dst_layout=dst_layout,
    )
    return Transition(
        name=name,
        tensor_id=tensor_id,
        src_layer_id=src_layer_id,
        src_output_idx=src_output_idx,
        dst_layer_id=dst_layer_id,
        dst_input_idx=dst_input_idx,
        mode=TransitionMode.DIRECT_REMAP,
        src_layout=src_layout,
        dst_layout=dst_layout,
        fragments=fragments,
    )
