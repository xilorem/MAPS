"""Central operation registry."""

from __future__ import annotations

from .spec import OnnxLoweringFn, OpSpec

_OPS_BY_NAME: dict[str, OpSpec] = {}
_OPS_BY_ONNX_NAME: dict[str, OpSpec] = {}
_OPS_BY_PAYLOAD_TYPE: dict[type, OpSpec] = {}
_BUILTINS_LOADED = False


def _ensure_builtins_registered() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return

    from MAPS.ops.defs import conv, elementwise, gemm, softmax  # noqa: F401

    _BUILTINS_LOADED = True


def register_op(spec: OpSpec) -> None:
    """Register one operation spec."""

    existing = _OPS_BY_NAME.get(spec.name)
    if existing is not None:
        raise ValueError(f"duplicate op spec name: {spec.name}")
    _OPS_BY_NAME[spec.name] = spec

    for onnx_name in spec.onnx_names:
        existing = _OPS_BY_ONNX_NAME.get(onnx_name)
        if existing is not None:
            raise ValueError(
                f"duplicate ONNX op mapping for {onnx_name}: {existing.name} vs {spec.name}"
            )
        _OPS_BY_ONNX_NAME[onnx_name] = spec

    if spec.payload_type is not None:
        existing = _OPS_BY_PAYLOAD_TYPE.get(spec.payload_type)
        if existing is not None:
            raise ValueError(
                f"duplicate payload_type mapping for {spec.payload_type.__name__}: "
                f"{existing.name} vs {spec.name}"
            )
        _OPS_BY_PAYLOAD_TYPE[spec.payload_type] = spec


def get_op(name: str) -> OpSpec:
    """Return one registered op spec by canonical name."""

    _ensure_builtins_registered()
    try:
        return _OPS_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"unknown op spec: {name}") from exc


def get_onnx_lowerer(onnx_op_type: str) -> OnnxLoweringFn | None:
    """Return the registered ONNX lowerer for one external op type."""

    _ensure_builtins_registered()
    spec = _OPS_BY_ONNX_NAME.get(onnx_op_type)
    if spec is None:
        return None
    return spec.lower_onnx


def get_op_by_onnx_name(onnx_op_type: str) -> OpSpec | None:
    """Return the op spec that handles one ONNX op type."""

    _ensure_builtins_registered()
    return _OPS_BY_ONNX_NAME.get(onnx_op_type)


def get_op_for_payload(payload: object) -> OpSpec | None:
    """Return the op spec for one payload instance."""

    _ensure_builtins_registered()
    for payload_type in type(payload).__mro__:
        spec = _OPS_BY_PAYLOAD_TYPE.get(payload_type)
        if spec is not None:
            return spec
    return None


def registered_ops() -> tuple[OpSpec, ...]:
    """Return all registered operation specs."""

    _ensure_builtins_registered()
    return tuple(_OPS_BY_NAME.values())


def registered_onnx_lowerers() -> dict[str, OnnxLoweringFn]:
    """Return the registered ONNX lowerers keyed by ONNX op type."""

    _ensure_builtins_registered()
    return {
        onnx_name: spec.lower_onnx
        for onnx_name, spec in _OPS_BY_ONNX_NAME.items()
        if spec.lower_onnx is not None
    }
