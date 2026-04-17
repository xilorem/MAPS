"""Generic tile-local devices."""

from __future__ import annotations

from MAPS.arch import CoreDevice, DMADevice, DeviceKind, WorkKind

IDMA_DEVICE = DMADevice(
    name="idma",
    kind=DeviceKind.DMA,
    throughput={WorkKind.DMA: 1.0},
)

SCALAR_CORE_DEVICE = CoreDevice(
    name="core",
    kind=DeviceKind.SCALAR,
    throughput={
        WorkKind.ELEMENTWISE: 1.0,
        WorkKind.REDUCE_SUM: 1.0,
        WorkKind.REDUCE_MAX: 1.0,
        WorkKind.EXP: 1.0,
    },
)

GENERIC_CORE_DEVICE = CoreDevice(
    name="core",
    kind=DeviceKind.SCALAR,
    throughput={
        WorkKind.GEMM: 1.0,
        WorkKind.ELEMENTWISE: 1.0,
        WorkKind.REDUCE_SUM: 1.0,
        WorkKind.REDUCE_MAX: 1.0,
        WorkKind.EXP: 1.0,
    },
)
