# MAPS

MAPS is a planner for distributed execution on homogeneous multi-tile architectures. It estimates compute and transport costs, and assembles a scheduled pipeline over a physical mesh.

## Directory layout

```text
MAPS/
  arch/        Hardware and NoC abstractions
  core/        Graph, layout, tensor, and submesh IR
  hw/          Concrete device and chip descriptions
  importers/   ONNX import path
  ops/         Operation payloads, tile work, and cost models
  planner/     Workload balancing, spatial mapping, constraints, and pipeline build
  transitions/ Inter-stage transition building and transport costing
examples/      Runnable examples
tests/         Unit and integration tests
tutorials/     Short development guides
```

## Download

Clone the repository and install it in a local virtual environment:

```bash
git clone <repo-url>
cd MAPS
python -m venv .venv
./.venv/bin/python -m pip install -e '.[dev]'
```

## Run tests

Run the full test suite with:

```bash
./.venv/bin/python -m pytest -q
```

## Run the MAGIA example

MAPS includes a runnable MAGIA planning example in [examples/magia_example.py](/home/ivan/repos/MAPS/examples/magia_example.py).

Run it with:

```bash
./.venv/bin/python examples/magia_example.py
```