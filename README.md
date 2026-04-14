# MAPS

MAPS is a standalone planner for distributed execution on homogeneous multi-tile architectures.

## Project Layout

```text
MAPS/
  arch/          Mesh, tile, and memory-system descriptions
  core/          Logical graph IR and scheduled pipeline IR
  ops/           Operation payloads, currently focused on GEMM
  importers/     ONNX frontend
  builders/      Transition and remap builders
  cost_models/   GEMM, transport, and transition cost models
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

Tensor layout is separate from physical placement. A physical submesh can be
interpreted through a different logical shape, as long as the logical area
matches the number of physical tiles.

## Hardware Model

The architecture layer currently models:

- rectangular meshes
- physical tiles
- per-tile L1 memory
- mesh-level L2 memory
- optional L2 access points
- tile-local compute devices

Example:

```python
from MAPS.arch import Device, DeviceKind, L1Memory, L2Memory, Mesh, Tile, WorkKind

devices = (
    Device(name="idma", kind=DeviceKind.DMA, throughput={WorkKind.DMA: 1.0}),
    Device(
        name="core",
        kind=DeviceKind.SCALAR,
        throughput={WorkKind.ELEMENTWISE: 1.0},
    ),
    Device(
        name="redmule",
        kind=DeviceKind.SYSTOLIC,
        throughput={WorkKind.GEMM: 192.0},
    ),
)

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

Transport cost uses L1 bandwidth, L2 bandwidth, Manhattan distance, and the
nearest L2 access point.

Compute cost uses device capabilities. `redmule` is the device name; the device
kind is `SYSTOLIC`.

The concrete MAGIA chip definition is available as:

```python
from MAPS.chips import magia_mesh

mesh = magia_mesh()
```

## Install

Create a virtual environment and install the package with development
dependencies:

```bash
python -m venv .venv
./.venv/bin/python -m pip install -e '.[dev]'
```

## Run Tests

```bash
./.venv/bin/python -m pytest -q
```

Some spatial mapping tests perform exhaustive placement searches and may be
slower than the rest of the suite.

## Current Planner Flow

The high-level ONNX entry point is:

```python
from MAPS.planner.plan import build_pipeline

pipeline = build_pipeline("model.onnx", mesh)
```

Internally this:

1. imports the ONNX graph
2. balances stage tile counts
3. searches logical layouts
4. maps stages to physical submeshes
5. builds inter-stage transitions
6. returns a scheduled `Pipeline`
