# MAPS

MAPS is a standalone planner for distributed execution on homogeneous multi-tile architectures.


## Install

Create a virtual environment and install the package with development
dependencies:

```bash
python -m venv .venv
pip install -e '.[dev]'
```


## Project Layout

```text
MAPS/
  arch/          Mesh, tile, and memory-system descriptions
  devices/       Reusable tile-local device definitions
  chips/         Concrete chip descriptions
  core/          Logical graph IR and scheduled pipeline IR
  ops/           Operation payloads and shared tile-work helpers
  importers/     ONNX frontend
  builders/      Transition and remap builders
  cost_models/   Compute, transport, and transition cost models
  planner/       Workload balancing, spatial mapping, and validation
tests/           Unit and planner integration tests
```

## Core Concepts

The code is split into two IR levels.

Graph IR describes logical computation:

- `Tensor`: tensor metadata
- `Node`: one logical operation
- `Edge`: one tensor dependency
- `Graph`: model-level logical graph

Scheduled IR describes placed execution:

- `Stage`: one scheduled layer on a physical submesh
- `Pipeline`: scheduled stages, tensors, and transitions
- `Transition`: movement of one tensor between producer and consumer stages
- `ExternalInput`, `TransitionInput`, and `LocalInput`: typed stage input sources

Tensor layout is separate from physical placement. A physical submesh can be
interpreted through a different logical shape, as long as the logical area
matches the number of physical tiles.

Operation behavior lives in `Node.payload`. Payloads implement the planner-facing
methods for default layouts, tile work, tensor validation, and cost-model
selection. Simple unary and binary elementwise operations share reusable payloads;
specialized operations such as GEMM and Conv keep dedicated payloads.

## Hardware Model

The architecture layer currently models:

- rectangular meshes
- physical tiles
- per-tile L1 memory
- mesh-level L2 memory
- optional L2 access points
- tile-local compute devices

Device models are concrete architecture types. `CoreDevice` and `DMADevice`
currently use throughput-based timing, while `SystolicDevice` owns systolic array
dimensions and provides GEMM-specific timing. RedMulE is modeled as a
`SystolicDevice`; IDMA is modeled as a `DMADevice`; scalar cores are modeled as
`CoreDevice`.

Tiles get a generic scalar `CoreDevice` by default when no device list is
provided. For MAGIA, use the chip helper:

```python
from MAPS.chips import magia_mesh

mesh = magia_mesh()
```

Custom meshes can provide explicit devices:

```python
from MAPS.arch import L1Memory, L2Memory, Mesh, Tile
from MAPS.devices import GENERIC_CORE_DEVICE

devices = (GENERIC_CORE_DEVICE,)
mesh = Mesh(
    width=2,
    height=2,
    l2_memory=L2Memory(size=16384, bandwidth=128, access_points=((0, 0),)),
    tiles=(
        Tile(
            tile_id=0,
            x=0,
            y=0,
            memory=L1Memory(size=4096, bandwidth=64),
            devices=devices,
        ),
        Tile(
            tile_id=1,
            x=1,
            y=0,
            memory=L1Memory(size=4096, bandwidth=64),
            devices=devices,
        ),
        Tile(
            tile_id=2,
            x=0,
            y=1,
            memory=L1Memory(size=4096, bandwidth=64),
            devices=devices,
        ),
        Tile(
            tile_id=3,
            x=1,
            y=1,
            memory=L1Memory(size=4096, bandwidth=64),
            devices=devices,
        ),
    ),
)
```



## Run Tests

```bash
./.venv/bin/python -m pytest -q
```



## Current Planner Flow

The high-level ONNX entry point is:

```python
from MAPS.planner.plan import build_pipeline

pipeline = build_pipeline("model.onnx", mesh)
```

Internally this:

1. imports the ONNX graph
2. seeds each stage with the smallest L1-feasible tile count
3. greedily grows the current bottleneck stage while preserving rectangular placement feasibility
4. chooses logical layouts for the selected stage tile counts
5. maps stages to physical submeshes
6. builds inter-stage transitions
7. returns a scheduled `Pipeline`

Workload balancing returns `StagePlan` objects, not just tile counts. Each plan
keeps the chosen tile count, logical mesh shape, and input/output layouts. The
planner debug output can be enabled with `print_workload_balancing=True` to show
L1 seeding, candidate rejection reasons, committed growth decisions, and the
final tile-count/workload timeline.
