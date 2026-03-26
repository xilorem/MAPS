"""ONNX op-specific lowering registry."""

from .gemm import ONNX_OP_LOWERERS as GEMM_OP_LOWERERS
from .matmul import ONNX_OP_LOWERERS as MATMUL_OP_LOWERERS

ONNX_OP_LOWERERS = {
    **MATMUL_OP_LOWERERS,
    **GEMM_OP_LOWERERS,
}

__all__ = ["ONNX_OP_LOWERERS"]
