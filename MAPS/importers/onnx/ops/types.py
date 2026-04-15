"""Shared typing for ONNX op lowerers."""

from __future__ import annotations

from typing import Callable

from MAPS.core.graph import OpKind
from MAPS.core.tensor import Tensor

OnnxLoweringFn = Callable[
    [str, tuple[Tensor, ...], tuple[Tensor, ...], dict[str, object]],
    tuple[OpKind, object],
]
