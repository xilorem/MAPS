"""GEMM payload IR matching the runtime-side `gemm_layer_op_t`."""

from __future__ import annotations

from dataclasses import dataclass

from MAPS.core.layer import LayerInputBinding, LayerOutputBinding
from MAPS.core.tensor import Tensor


@dataclass(frozen=True)
class GemmLayerOp:
    """GEMM-specific layer payload.

    The planner-side GEMM convention is:
    - ``x`` has shape ``[..., M, K]``
    - ``w`` has shape ``[..., K, N]``
    - ``output`` has shape ``[..., M, N]``
    - optional ``y`` must match ``output`` exactly
    """

    x: Tensor
    w: Tensor
    y: Tensor | None
    output: Tensor

    def validate_bindings(self,
                          inputs: tuple[LayerInputBinding, ...],
                          outputs: tuple[LayerOutputBinding, ...]) -> None:
        """Validate the minimum binding structure required by GEMM."""

        if len(inputs) < 2:
            raise ValueError("GEMM layers require at least two inputs")
        if len(outputs) == 0:
            raise ValueError("GEMM layers require at least one output")

    def validate_tensors(self,
                         inputs: tuple[LayerInputBinding, ...],
                         outputs: tuple[LayerOutputBinding, ...],
                         tensors: tuple[Tensor, ...]) -> None:
        """Validate tensor availability against one GEMM layer instance."""

        self.validate_bindings(inputs, outputs)

        bound_inputs = tuple(tensors[binding.tensor_id] for binding in inputs)
        bound_outputs = tuple(tensors[binding.tensor_id] for binding in outputs)

        if self.x not in bound_inputs:
            raise ValueError("GEMM X tensor is not present in layer inputs")
        if self.w not in bound_inputs:
            raise ValueError("GEMM W tensor is not present in layer inputs")
        if self.y is not None and self.y not in bound_inputs:
            raise ValueError("GEMM Y tensor is not present in layer inputs")
        if self.output not in bound_outputs:
            raise ValueError("GEMM output tensor is not present in layer outputs")

        for tensor_name, tensor in (
            ("X", self.x),
            ("W", self.w),
            ("output", self.output),
        ):
            if tensor.rank < 2:
                raise ValueError(f"{tensor_name} tensor rank must be >= 2 for GEMM")

        if self.x.elem_bytes != self.w.elem_bytes or self.x.elem_bytes != self.output.elem_bytes:
            raise ValueError("GEMM tensors must agree on element size")

        if self.x.dims[-1] != self.w.dims[-2]:
            raise ValueError("GEMM tensors must agree on K dimension")
        if self.x.dims[-2] != self.output.dims[-2]:
            raise ValueError("GEMM X and output must agree on M dimension")
        if self.w.dims[-1] != self.output.dims[-1]:
            raise ValueError("GEMM W and output must agree on N dimension")

        if self.y is not None:
            if self.y.rank != self.output.rank or self.y.dims != self.output.dims:
                raise ValueError("Y input must match output tensor shape exactly")
            if self.y.elem_bytes != self.output.elem_bytes:
                raise ValueError("Y input element size must match output tensor")
