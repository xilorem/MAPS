from MAPS.arch import CoreDevice, Device, DeviceKind, L1Memory, SystolicDevice, Tile, WorkKind
from MAPS.chips import magia_mesh
from MAPS.chips.magia import MAGIA_REDMULE_DEVICE
from MAPS.core.layout import TensorRange, TensorSlice
from MAPS.ops.costs.gemm_cost import GemmCostModel
from MAPS.devices.generic import GENERIC_CORE_DEVICE
from MAPS.devices.redmule import REDMULE_ARRAY_HEIGHT, REDMULE_ARRAY_WIDTH
from MAPS.ops.defs.gemm import GemmTileWork


def _tile_work(m_size: int = 4, n_size: int = 8, k_size: int = 16) -> GemmTileWork:
    output_slice = TensorSlice(
        rank=2,
        dims=(
            TensorRange(start=0, length=m_size),
            TensorRange(start=0, length=n_size),
        ),
    )
    return GemmTileWork(
        output_slice=output_slice,
        x_slice=TensorSlice(
            rank=2,
            dims=(
                TensorRange(start=0, length=m_size),
                TensorRange(start=0, length=k_size),
            ),
        ),
        w_slice=TensorSlice(
            rank=2,
            dims=(
                TensorRange(start=0, length=k_size),
                TensorRange(start=0, length=n_size),
            ),
        ),
        y_slice=None,
    )


def test_device_base_class_is_not_directly_instantiable() -> None:
    try:
        Device(
            name="base",
            kind=DeviceKind.SCALAR,
            throughput={WorkKind.ELEMENTWISE: 1},
        )
    except TypeError as exc:
        assert "concrete device type" in str(exc)
    else:
        raise AssertionError("expected Device base class construction to fail")


def test_tile_can_use_generic_core_device() -> None:
    tile = Tile(
        tile_id=0,
        x=0,
        y=0,
        memory=L1Memory(size=4096, bandwidth=1),
        devices=(GENERIC_CORE_DEVICE,),
    )

    assert tile.devices[0].name == "core"
    assert tile.devices[0].kind is DeviceKind.SCALAR
    assert isinstance(tile.devices[0], CoreDevice)
    assert tile.devices[0].supports(WorkKind.GEMM)


def test_tile_rejects_empty_devices() -> None:
    try:
        Tile(tile_id=0, x=0, y=0, memory=L1Memory(size=4096, bandwidth=1), devices=())
    except ValueError as exc:
        assert "tile devices must not be empty" in str(exc)
    else:
        raise AssertionError("expected Tile construction to fail")


def test_redmule_is_a_named_systolic_device() -> None:
    device = MAGIA_REDMULE_DEVICE

    assert device.name == "redmule"
    assert device.kind is DeviceKind.SYSTOLIC
    assert isinstance(device, SystolicDevice)
    assert device.array_width == REDMULE_ARRAY_WIDTH
    assert device.array_height == REDMULE_ARRAY_HEIGHT
    assert device.supports(WorkKind.GEMM)
    assert device.throughput[WorkKind.GEMM] == REDMULE_ARRAY_WIDTH * REDMULE_ARRAY_HEIGHT


def test_gemm_cost_uses_systolic_device_when_available() -> None:
    scalar_tile = Tile(
        tile_id=0,
        x=0,
        y=0,
        memory=L1Memory(size=4096, bandwidth=1),
        devices=(GENERIC_CORE_DEVICE,),
    )
    redmule_tile = magia_mesh().tile(0, 0)
    model = GemmCostModel()
    tile_work = _tile_work()

    assert model.cost(tile_work, redmule_tile) < model.cost(tile_work, scalar_tile)


def test_redmule_gemm_cost_accounts_for_array_shape() -> None:
    redmule_tile = magia_mesh().tile(0, 0)
    model = GemmCostModel()

    compact_work = _tile_work(m_size=4, n_size=8, k_size=16)
    wide_work = _tile_work(m_size=1, n_size=32, k_size=16)

    assert model.cost(compact_work, redmule_tile) == 46
    assert model.cost(wide_work, redmule_tile) == 92
