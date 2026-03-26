"""Shared typing for ONNX op lowerers."""

from __future__ import annotations

from typing import Callable

from MAPS.core.layer import LayerOpKind
from MAPS.core.tensor import Tensor

OnnxLoweringFn = Callable[
    [str, tuple[Tensor, ...], tuple[Tensor, ...]],
    tuple[LayerOpKind, object],
]
