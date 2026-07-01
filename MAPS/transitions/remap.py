"""Helpers to build direct-remap fragments from producer ownership and consumer demand."""

from MAPS.arch import Tile
from MAPS.core.layout import TensorLayout, TensorRange, TensorSlice, TensorSubSlice, tile_tensor_slice
from MAPS.core.tensor import Tensor
from MAPS.transitions.model import TransitionFragment


def _intersect_range(a: TensorRange, b: TensorRange) -> TensorRange | None:
    """Return the overlap between two 1D tensor ranges, if any."""

    start = max(a.start, b.start)
    end = min(a.start + a.length, b.start + b.length)
    if start >= end:
        return None
    return TensorRange(start=start, length=end - start)


def _intersect_slice(a: TensorSlice, b: TensorSlice) -> TensorSlice | None:
    """Return the overlap between two tensor slices, if any."""

    if a.rank != b.rank:
        raise ValueError("cannot intersect slices with different ranks")

    dims: list[TensorRange] = []
    for a_dim, b_dim in zip(a.dims, b.dims):
        overlap = _intersect_range(a_dim, b_dim)
        if overlap is None:
            return None
        dims.append(overlap)

    return TensorSlice(rank=a.rank, dims=tuple(dims))


def _relative_subslice(parent: TensorSlice, child: TensorSlice) -> TensorSubSlice:
    """Return child as a subslice with coordinates relative to parent."""

    if parent.rank != child.rank:
        raise ValueError("cannot build subslice from slices with different ranks")

    dims: list[TensorRange] = []
    for parent_dim, child_dim in zip(parent.dims, child.dims):
        parent_end = parent_dim.start + parent_dim.length
        child_end = child_dim.start + child_dim.length
        if child_dim.start < parent_dim.start or child_end > parent_end:
            raise ValueError("child slice must fit inside parent slice")
        dims.append(
            TensorRange(
                start=child_dim.start - parent_dim.start,
                length=child_dim.length,
            )
        )

    return TensorSubSlice(parent=parent, dims=tuple(dims))


def tile_owned_slices(tensor: Tensor, layout: TensorLayout) -> tuple[tuple[Tile, TensorSlice], ...]:
    """Return the concrete slice owned by each tile in one submesh."""

    owned: list[tuple[Tile, TensorSlice]] = []
    for tile in layout.submesh.tiles:
        owned.append(
            (
                tile,
                tile_tensor_slice(
                    tensor=tensor,
                    layout=layout,
                    tile=tile,
                ),
            )
        )
    return tuple(owned)


def build_direct_remap_fragments(
    tensor: Tensor,
    src_layout: TensorLayout,
    dst_required_slices: tuple[tuple[Tile, TensorSlice], ...],
) -> tuple[TransitionFragment, ...]:
    """Build direct-remap fragments for one tensor."""

    src_owned = tile_owned_slices(tensor, src_layout)

    fragments: list[TransitionFragment] = []
    for src_tile, src_slice in src_owned:
        for dst_tile, dst_slice in dst_required_slices:
            overlap = _intersect_slice(src_slice, dst_slice)
            if overlap is None:
                continue
            fragments.append(
                TransitionFragment(
                    src_hartid=src_tile.tile_id,
                    dst_hartid=dst_tile.tile_id,
                    src_subslice=_relative_subslice(src_slice, overlap),
                    dst_subslice=_relative_subslice(dst_slice, overlap),
                )
            )

    return tuple(fragments)
