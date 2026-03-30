"""Helpers to build direct-remap fragments from concrete tile ownership."""

from MAPS.arch import Tile
from MAPS.core.layout import TensorRange, TensorSlice
from MAPS.core.transition import TransitionFragment
from MAPS.core.ownership import tile_tensor_slice
from MAPS.core.tensor import Tensor
from MAPS.core.layout import TensorLayout


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


def tile_owned_slices(tensor: Tensor,
                      layout: TensorLayout,
                      microbatch_idx: int) -> tuple[tuple[Tile, TensorSlice], ...]:
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
                    microbatch_idx=microbatch_idx,
                ),
            )
        )
    return tuple(owned)


def build_direct_remap_fragments(
    tensor: Tensor,
    src_layout: TensorLayout,
    dst_layout: TensorLayout,
    microbatch_idx: int,
) -> tuple[TransitionFragment, ...]:
    """Build direct-remap fragments for one tensor and one microbatch."""

    src_owned = tile_owned_slices(tensor, src_layout, microbatch_idx)
    dst_owned = tile_owned_slices(tensor, dst_layout, microbatch_idx)

    fragments: list[TransitionFragment] = []
    for src_tile, src_slice in src_owned:
        for dst_tile, dst_slice in dst_owned:
            overlap = _intersect_slice(src_slice, dst_slice)
            if overlap is None:
                continue
            # NOTE: this simple overlap-based builder emits one fragment per
            # overlapping (src_tile, dst_tile) pair. That is correct for
            # destination-side replication, where multiple consumers truly need
            # the same logical slice. It over-generates when the source layout
            # contains replicas, because multiple equivalent source tiles can
            # overlap the same destination need. We will need a replica-
            # selection policy to choose one provider in those cases.
            fragments.append(
                TransitionFragment(
                    src_hartid=src_tile.tile_id,
                    dst_hartid=dst_tile.tile_id,
                    src_slice=overlap,
                    dst_slice=overlap,
                )
            )

    return tuple(fragments)
