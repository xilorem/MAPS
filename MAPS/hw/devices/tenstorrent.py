"""Tenstorrent tile-local placeholder device models."""

from __future__ import annotations

from MAPS.arch import CoreDevice, DeviceKind, WorkKind

TENSIX_SCALAR_DEVICE = CoreDevice(
    name="tensix_scalar",
    kind=DeviceKind.SCALAR,
    throughput={
        WorkKind.ELEMENTWISE: 1,
        WorkKind.REDUCE_SUM: 1,
        WorkKind.REDUCE_MAX: 1,
        WorkKind.EXP: 1,
    },
)

# Placeholder GEMM engine until a more faithful Tensix matrix model is added.
TENSIX_MATRIX_DEVICE = CoreDevice(
    name="tensix_matrix",
    kind=DeviceKind.SCALAR,
    throughput={
        WorkKind.GEMM: 1,
    },
)
