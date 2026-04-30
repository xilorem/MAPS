"""Tenstorrent tile-local placeholder device models."""

from __future__ import annotations

from MAPS.arch import DeviceKind, MatrixDevice, ScalarDevice, VectorDevice, WorkKind

TENSIX_SCALAR_DEVICE = ScalarDevice(
    name="tensix_scalar",
    kind=DeviceKind.SCALAR,
    throughput={
        WorkKind.ELEMENTWISE: 1,
        WorkKind.REDUCE_SUM: 1,
        WorkKind.REDUCE_MAX: 1,
        WorkKind.EXP: 1,
    },
)

TENSIX_VECTOR_DEVICE = VectorDevice(
    name="tensix_vector",
    kind=DeviceKind.VECTOR,
    throughput={
        WorkKind.ELEMENTWISE: 1,
        WorkKind.REDUCE_SUM: 1,
        WorkKind.REDUCE_MAX: 1,
        WorkKind.EXP: 1,
    },
    vector_length=32,
)

TENSIX_MATRIX_DEVICE = MatrixDevice(
    name="tensix_matrix",
    kind=DeviceKind.MATRIX,
    throughput={
        WorkKind.GEMM: 1,
    },
    srcA_width=16,
    srcA_height=8,
    srcB_width=16,
    srcB_height=16,
    math_fidelity=1,
)
