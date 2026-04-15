"""RedMulE device model."""

from __future__ import annotations

from math import ceil

from MAPS.arch import Device, DeviceKind, WorkKind
from MAPS.ops.gemm import GemmTileWork

REDMULE_ARRAY_WIDTH = 24
REDMULE_ARRAY_HEIGHT = 8


def _batch_volume(tile_work: GemmTileWork) -> int:
    total = 1
    for dim in tile_work.output_slice.dims[:-2]:
        total *= dim.length
    return total


def _redmule_gemm_cycles(
    device: Device,
    work_kind: WorkKind,
    amount: float,
    work: object,
) -> float:
    if work_kind is not WorkKind.GEMM:
        raise ValueError("redmule only estimates GEMM work")
    if not isinstance(work, GemmTileWork):
        raise ValueError("redmule GEMM estimation requires GemmTileWork")

    m_size = work.output_slice.dims[-2].length
    n_size = work.output_slice.dims[-1].length
    k_size = work.x_slice.dims[-1].length
    m_blocks = ceil(m_size / REDMULE_ARRAY_HEIGHT)
    n_blocks = ceil(n_size / REDMULE_ARRAY_WIDTH)
    fill_and_drain_cycles = REDMULE_ARRAY_HEIGHT + REDMULE_ARRAY_WIDTH - 2

    return (
        _batch_volume(work)
        * m_blocks
        * n_blocks
        * (k_size + fill_and_drain_cycles)
    )


REDMULE_DEVICE = Device(
    name="redmule",
    kind=DeviceKind.SYSTOLIC,
    throughput={WorkKind.GEMM: REDMULE_ARRAY_WIDTH * REDMULE_ARRAY_HEIGHT},
    cycle_estimator=_redmule_gemm_cycles,
)
