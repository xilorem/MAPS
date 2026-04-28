"""Tile-local compute device capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from math import ceil


class DeviceKind(Enum):
    SCALAR = auto()
    VECTOR = auto()
    SYSTOLIC = auto()
    DMA = auto()


class WorkKind(Enum):
    GEMM = auto()
    ELEMENTWISE = auto()
    REDUCE_SUM = auto()
    REDUCE_MAX = auto()
    EXP = auto()
    DMA = auto()


class DMAJob(Enum):
    READJOB = auto()
    WRITEJOB = auto()


@dataclass(frozen=True)
class Device:
    """Base class for tile-local device models."""

    name: str
    kind: DeviceKind
    throughput: dict[WorkKind, int]
    startup_cycles: int = 0

    def __post_init__(self) -> None:
        if type(self) is Device:
            raise TypeError("Device must be instantiated through a concrete device type")
        if not self.name:
            raise ValueError("device name must not be empty")
        if self.startup_cycles < 0:
            raise ValueError("device startup_cycles must be >= 0")
        if not self.throughput:
            raise ValueError("device throughput must not be empty")

        # check for invalid throughput value
        if any(value <= 0 for value in self.throughput.values()):
            raise ValueError("device throughput values must be > 0")
        object.__setattr__(self, "throughput", dict(self.throughput))

    def supports(self, work_kind: WorkKind) -> bool:
        return work_kind in self.throughput

    def cycles(self, work: object) -> int:
        raise NotImplementedError

    def _throughput_cycles(self, work_kind: WorkKind, amount: int) -> int:
        if amount < 0:
            raise ValueError("device work amount must be >= 0")
        if not self.supports(work_kind):
            raise ValueError(f"device {self.name} does not support {work_kind.name} work")
        compute_cycles = ceil(amount / self.throughput[work_kind])
        if compute_cycles < 0:
            raise ValueError("device cycle estimator must return >= 0")
        return self.startup_cycles + compute_cycles


@dataclass(frozen=True)
class CoreDevice(Device):
    """Scalar/core device model using throughput-based timing."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind is not DeviceKind.SCALAR:
            raise ValueError("CoreDevice must use DeviceKind.SCALAR")

    def cycles(self, work: object) -> int:
        work_kind = work.work_kind
        amount = work.operation_count()
        return self._throughput_cycles(work_kind, amount)


@dataclass(frozen=True)
class DMADevice(Device):
    """DMA device model using throughput-based timing."""

    job: DMAJob = DMAJob.READJOB

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind is not DeviceKind.DMA:
            raise ValueError("DMADevice must use DeviceKind.DMA")
        if not isinstance(self.job, DMAJob):
            raise ValueError("Bad DMADevice job description, must be a DMAJob type")

    def cycles(self, work: object) -> int:
        raise ValueError("DMA device cannot perform compute operations")




@dataclass(frozen=True)
class SystolicDevice(Device):
    """Systolic-array device model with GEMM-specific timing."""

    array_width: int = 1
    array_height: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind is not DeviceKind.SYSTOLIC:
            raise ValueError("SystolicDevice must use DeviceKind.SYSTOLIC")
        if self.array_width <= 0 or self.array_height <= 0:
            raise ValueError("systolic array dimensions must be > 0")

    def cycles(self, work: object) -> int:
        batch_volume, m_size, n_size, k_size = work.dimensions()
        m_blocks = ceil(m_size / self.array_height)
        n_blocks = ceil(n_size / self.array_width)
        fill_and_drain_cycles = self.array_height + self.array_width - 2
        compute_cycles = batch_volume * m_blocks * n_blocks * (k_size + fill_and_drain_cycles)
        return self.startup_cycles + compute_cycles
