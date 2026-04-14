from MAPS.arch import DeviceKind, Tile, WorkKind
from MAPS.chips import magia_mesh
from MAPS.chips.magia import MAGIA_REDMULE_DEVICE
from MAPS.core.layout import TensorRange, TensorSlice
from MAPS.cost_models.gemm_cost import GemmCostModel
from MAPS.ops.gemm import GemmTileWork


def _tile_work() -> GemmTileWork:
    output_slice = TensorSlice(
        rank=2,
        dims=(
            TensorRange(start=0, length=4),
            TensorRange(start=0, length=8),
        ),
    )
    return GemmTileWork(
        output_slice=output_slice,
        x_slice=TensorSlice(
            rank=2,
            dims=(
                TensorRange(start=0, length=4),
                TensorRange(start=0, length=16),
            ),
        ),
        w_slice=TensorSlice(
            rank=2,
            dims=(
                TensorRange(start=0, length=16),
                TensorRange(start=0, length=8),
            ),
        ),
        y_slice=None,
    )


def test_default_tile_has_core_gemm_device() -> None:
    tile = Tile(tile_id=0, x=0, y=0)

    assert tile.devices[0].name == "core"
    assert tile.devices[0].kind is DeviceKind.SCALAR
    assert tile.devices[0].supports(WorkKind.GEMM)


def test_redmule_is_a_named_systolic_device() -> None:
    device = MAGIA_REDMULE_DEVICE

    assert device.name == "redmule"
    assert device.kind is DeviceKind.SYSTOLIC
    assert device.supports(WorkKind.GEMM)


def test_gemm_cost_uses_systolic_device_when_available() -> None:
    scalar_tile = Tile(tile_id=0, x=0, y=0)
    redmule_tile = magia_mesh().tile(0, 0)
    model = GemmCostModel()
    tile_work = _tile_work()

    assert model.cost(tile_work, redmule_tile) < model.cost(tile_work, scalar_tile)
