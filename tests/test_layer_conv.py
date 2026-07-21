from MAPS.arch import WorkKind
from MAPS.hw.chips import magia_mesh
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.ops.defs.conv_transforms import (
    Im2ColPayload,
    OutputReformatPayload,
    WeightPackPayload,
)

def test_decomposed_conv_transforms_replicate_patches_and_shard_channels() -> None:
    mesh = magia_mesh()
    submesh = Submesh(mesh=mesh, submesh_id=0, x0=0, y0=0, width=2, height=1)
    x = Tensor(name="x", rank=4, dims=(1, 3, 5, 5), elem_bytes=2)
    patches = Tensor(name="patches", rank=2, dims=(9, 27), elem_bytes=2)
    w = Tensor(name="w", rank=4, dims=(8, 3, 3, 3), elem_bytes=2)
    packed_w = Tensor(name="packed_w", rank=2, dims=(27, 8), elem_bytes=2)
    matrix = Tensor(name="matrix", rank=2, dims=(9, 8), elem_bytes=2)
    output = Tensor(name="out", rank=4, dims=(1, 8, 3, 3), elem_bytes=2)

    im2col = Im2ColPayload(x=x, output=patches, kernel_shape=(3, 3))
    pack = WeightPackPayload(w=w, output=packed_w)
    reshape = OutputReformatPayload(x=matrix, output=output)
    second_tile = submesh.tiles[1]

    patch_work = im2col.build_tile_work(im2col.output_layouts(submesh), second_tile)
    pack_work = pack.build_tile_work(pack.output_layouts(submesh), second_tile)
    reshape_work = reshape.build_tile_work(reshape.output_layouts(submesh), second_tile)

    assert patch_work.work_kind is WorkKind.IM2COL
    assert pack_work.work_kind is WorkKind.WEIGHT_PACK
    assert reshape_work.work_kind is WorkKind.OUTPUT_REFORMAT

    assert tuple((dim.start, dim.length) for dim in patch_work.output_slice.dims) == (
        (0, 9),
        (0, 27),
    )
    assert tuple((dim.start, dim.length) for dim in pack_work.output_slice.dims) == (
        (0, 27),
        (4, 4),
    )
    assert tuple((dim.start, dim.length) for dim in pack_work.input_tile_slices[0].dims) == (
        (4, 4),
        (0, 3),
        (0, 3),
        (0, 3),
    )
    assert tuple((dim.start, dim.length) for dim in reshape_work.output_slice.dims) == (
        (0, 1),
        (4, 4),
        (0, 3),
        (0, 3),
    )
    assert tuple((dim.start, dim.length) for dim in reshape_work.input_tile_slices[0].dims) == (
        (0, 9),
        (4, 4),
    )
