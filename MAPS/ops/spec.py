"""Operation metadata used by frontends and legalization passes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from MAPS.arch import WorkKind
from MAPS.core.graph import Node, OpKind
from MAPS.core.tensor import Tensor

OnnxLoweringFn = Callable[
    [str, tuple[Tensor, ...], tuple[Tensor, ...], dict[str, object]],
    tuple[OpKind, object],
]
DecomposeFn = Callable[[Node], tuple[tuple[Tensor, ...], tuple[Node, ...]]]


@dataclass(frozen=True)
class OpSpec:
    """One operation entry in the lightweight op registry."""

    name: str
    onnx_names: tuple[str, ...] = ()
    lower_onnx: OnnxLoweringFn | None = None
    decompose: DecomposeFn | None = None
    payload_type: type | None = None
    work_kinds: tuple[WorkKind, ...] = ()
