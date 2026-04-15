"""ONNX op-specific lowering registry."""

from .conv import ONNX_OP_LOWERERS as CONV_OP_LOWERERS
from .exp import ONNX_OP_LOWERERS as EXP_OP_LOWERERS
from .gemm import ONNX_OP_LOWERERS as GEMM_OP_LOWERERS
from .matmul import ONNX_OP_LOWERERS as MATMUL_OP_LOWERERS

ONNX_OP_LOWERERS = {
    **MATMUL_OP_LOWERERS,
    **GEMM_OP_LOWERERS,
    **CONV_OP_LOWERERS,
    **EXP_OP_LOWERERS,
}

__all__ = ["ONNX_OP_LOWERERS"]
