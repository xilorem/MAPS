"""Reusable tile-local device definitions."""

from .generic import IDMA_DEVICE, SCALAR_CORE_DEVICE
from .redmule import (
    REDMULE_ARRAY_HEIGHT,
    REDMULE_ARRAY_WIDTH,
    REDMULE_DEVICE,
)

__all__ = [
    "IDMA_DEVICE",
    "REDMULE_ARRAY_HEIGHT",
    "REDMULE_ARRAY_WIDTH",
    "REDMULE_DEVICE",
    "SCALAR_CORE_DEVICE",
]
