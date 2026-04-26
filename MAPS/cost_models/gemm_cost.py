"""GEMM-level aggregation built on top of concrete per-tile GEMM work."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from MAPS.arch import Device, DeviceKind, SystolicDevice, Tile, WorkKind
from MAPS.ops.gemm import GemmTileWork
from MAPS.core.layout import TensorSlice


@dataclass(frozen=True)
class GemmCostModel:
    """Compute-only GEMM cycle model backed by tile devices."""

    preferred_device_kind: DeviceKind = DeviceKind.SYSTOLIC

    def cost(self, tile_work: GemmTileWork, tile: Tile) -> int:
        devices = tuple(device for device in tile.devices if device.supports(WorkKind.GEMM))
        preferred = tuple(
            device for device in devices if device.kind is self.preferred_device_kind
        )
        candidates = preferred or devices
        if not candidates:
            raise ValueError(f"tile {tile.tile_id} has no device for GEMM work")
        return min(_gemm_cycles_on_device(device, tile_work) for device in candidates)


def _tensor_slice_num_elements(tensor_slice: TensorSlice) -> int:
    total = 1
    for dim in tensor_slice.dims:
        total *= dim.length
    return total


def _gemm_tile_num_ops(tile_work: GemmTileWork) -> int:
    """Return a simple GEMM work count for one tile.

    This is a placeholder compute-only estimate:
    - output elements = owned M x owned N x batch volume
    - reduction depth = full K inferred from the required X slice
    - total ops = output elements x K
    """

    output_elements = _tensor_slice_num_elements(tile_work.output_slice)
    k_depth = tile_work.x_slice.dims[-1].length
    return output_elements * k_depth


def _gemm_cycles_on_device(device: Device, tile_work: GemmTileWork) -> int:
    if device.kind is DeviceKind.SYSTOLIC:
        if not isinstance(device, SystolicDevice):
            raise ValueError("systolic GEMM estimation requires SystolicDevice")
        return device.startup_cycles + _systolic_gemm_cycles(device, tile_work)

    amount = _gemm_tile_num_ops(tile_work)
    return device.cycles(WorkKind.GEMM, amount)


def _systolic_gemm_cycles(device: SystolicDevice, tile_work: GemmTileWork) -> int:
    batch_volume = 1
    for dim in tile_work.output_slice.dims[:-2]:
        batch_volume *= dim.length

    m_size = tile_work.output_slice.dims[-2].length
    n_size = tile_work.output_slice.dims[-1].length
    k_size = tile_work.x_slice.dims[-1].length
    m_blocks = ceil(m_size / device.array_height)
    n_blocks = ceil(n_size / device.array_width)
    fill_and_drain_cycles = device.array_height + device.array_width - 2

    return batch_volume * m_blocks * n_blocks * (k_size + fill_and_drain_cycles)
