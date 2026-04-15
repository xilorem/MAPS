"""Tile-local compute device capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import Callable, Mapping


class DeviceKind(IntEnum):
    SCALAR = 0
    VECTOR = 1
    SYSTOLIC = 2
    DMA = 3


class WorkKind(IntEnum):
    GEMM = 0
    ELEMENTWISE = 1
    REDUCE_SUM = 2
    REDUCE_MAX = 3
    EXP = 4
    DMA = 5


CycleEstimator = Callable[["Device", WorkKind, float, object], float]


def throughput_cycle_estimator(
    device: "Device",
    work_kind: WorkKind,
    amount: float,
    work: object,
) -> float:
    return amount / device.throughput[work_kind]


@dataclass(frozen=True)
class Device:
    """One compute or data-movement engine available on a tile."""

    name: str
    kind: DeviceKind
    throughput: Mapping[WorkKind, float]
    cycle_estimator: CycleEstimator
    startup_cycles: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("device name must not be empty")
        if self.startup_cycles < 0:
            raise ValueError("device startup_cycles must be >= 0")
        if not self.throughput:
            raise ValueError("device throughput must not be empty")
        if any(value <= 0 for value in self.throughput.values()):
            raise ValueError("device throughput values must be > 0")
        if not callable(self.cycle_estimator):
            raise ValueError("device cycle_estimator must be callable")
        object.__setattr__(self, "throughput", MappingProxyType(dict(self.throughput)))

    def supports(self, work_kind: WorkKind) -> bool:
        return work_kind in self.throughput

    def cycles(self, work_kind: WorkKind, amount: float, work: object) -> float:
        if amount < 0:
            raise ValueError("device work amount must be >= 0")
        compute_cycles = self.cycle_estimator(self, work_kind, amount, work)
        if compute_cycles < 0:
            raise ValueError("device cycle estimator must return >= 0")
        return self.startup_cycles + compute_cycles
