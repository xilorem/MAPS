from MAPS.arch import DeviceKind, Tile, WorkKind
from MAPS.chips import magia_mesh
from MAPS.chips.magia import MAGIA_REDMULE_DEVICE
from MAPS.core.layout import TensorRange, TensorSlice
from MAPS.cost_models.gemm_cost import GemmCostModel
from MAPS.devices.redmule import REDMULE_ARRAY_HEIGHT, REDMULE_ARRAY_WIDTH
from MAPS.ops.gemm import GemmTileWork


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
    assert device.throughput[WorkKind.GEMM] == REDMULE_ARRAY_WIDTH * REDMULE_ARRAY_HEIGHT


def test_gemm_cost_uses_systolic_device_when_available() -> None:
    scalar_tile = Tile(tile_id=0, x=0, y=0)
    redmule_tile = magia_mesh().tile(0, 0)
    model = GemmCostModel()
    tile_work = _tile_work()

    assert model.cost(tile_work, redmule_tile) < model.cost(tile_work, scalar_tile)


def test_redmule_gemm_cost_accounts_for_array_shape() -> None:
    redmule_tile = magia_mesh().tile(0, 0)
    model = GemmCostModel()

    compact_work = _tile_work(m_size=4, n_size=8, k_size=16)
    wide_work = _tile_work(m_size=1, n_size=32, k_size=16)

    assert model.cost(compact_work, redmule_tile) == 42
    assert model.cost(wide_work, redmule_tile) == 84
