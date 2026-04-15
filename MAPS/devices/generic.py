"""Generic tile-local devices."""

from __future__ import annotations

from MAPS.arch import Device, DeviceKind, WorkKind, throughput_cycle_estimator

IDMA_DEVICE = Device(
    name="idma",
    kind=DeviceKind.DMA,
    throughput={WorkKind.DMA: 1.0},
    cycle_estimator=throughput_cycle_estimator,
)

SCALAR_CORE_DEVICE = Device(
    name="core",
    kind=DeviceKind.SCALAR,
    throughput={
        WorkKind.ELEMENTWISE: 1.0,
        WorkKind.REDUCE_SUM: 1.0,
        WorkKind.REDUCE_MAX: 1.0,
        WorkKind.EXP: 1.0,
    },
    cycle_estimator=throughput_cycle_estimator,
)

GENERIC_CORE_DEVICE = Device(
    name="core",
    kind=DeviceKind.SCALAR,
    throughput={
        WorkKind.GEMM: 1.0,
        WorkKind.ELEMENTWISE: 1.0,
        WorkKind.REDUCE_SUM: 1.0,
        WorkKind.REDUCE_MAX: 1.0,
        WorkKind.EXP: 1.0,
    },
    cycle_estimator=throughput_cycle_estimator,
)
