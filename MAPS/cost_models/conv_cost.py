"""Conv cost model using im2col GEMM work."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.arch import DeviceKind, Tile, WorkKind
from MAPS.core.layout import TensorRange, TensorSlice
from MAPS.ops.conv import ConvTileWork
from MAPS.ops.gemm import GemmTileWork


@dataclass(frozen=True)
class ConvCostModel:
    """Compute-only Conv cycle model backed by GEMM-capable tile devices."""

    preferred_device_kind: DeviceKind = DeviceKind.SYSTOLIC

    def cost(self, tile_work: ConvTileWork, tile: Tile) -> int:
        gemm_work = _conv_tile_work_as_im2col_gemm(tile_work)
        devices = tuple(device for device in tile.devices if device.supports(WorkKind.GEMM))
        preferred = tuple(
            device for device in devices if device.kind is self.preferred_device_kind
        )
        candidates = preferred or devices
        if not candidates:
            raise ValueError(f"tile {tile.tile_id} has no device for Conv work")
        return min(device.cycles(gemm_work) for device in candidates)


def _conv_tile_work_as_im2col_gemm(tile_work: ConvTileWork) -> GemmTileWork:
    output_slice = tile_work.output_slice
    batch = output_slice.dims[0].length
    out_channels = output_slice.dims[1].length
    output_h = output_slice.dims[2].length
    output_w = output_slice.dims[3].length
    im2col_rows = batch * output_h * output_w
    im2col_cols = (
        tile_work.w_slice.dims[1].length
        * tile_work.w_slice.dims[2].length
        * tile_work.w_slice.dims[3].length
    )
    return GemmTileWork(
        output_slice=TensorSlice(
            rank=2,
            dims=(
                TensorRange(start=0, length=im2col_rows),
                TensorRange(start=0, length=out_channels),
            ),
        ),
        x_slice=TensorSlice(
            rank=2,
            dims=(
                TensorRange(start=0, length=im2col_rows),
                TensorRange(start=0, length=im2col_cols),
            ),
        ),
        w_slice=TensorSlice(
            rank=2,
            dims=(
                TensorRange(start=0, length=im2col_cols),
                TensorRange(start=0, length=out_channels),
            ),
        ),
        y_slice=None,
    )
