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
  pipeline/    Scheduled pipeline IR, layers, stages, and JSON export
  planner/     Workload balancing, spatial mapping, constraints, and pipeline build
  transforms/  Graph decomposition and graph utility transforms
  transitions/ Inter-stage transition building and transport costing
  utils/       Pipeline JSON and mesh/submesh printing helpers
examples/      Runnable examples
tests/         Unit and integration tests
tutorials/     Short development guides
```

## Download

Clone the repository and install it in a local virtual environment:

```bash
git clone --recursive https://github.com/xilorem/MAPS
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

MAPS includes a runnable MAGIA planning example in
`./examples/magia_example.py`.

Run only the Python planner and JSON export with:

```bash
./.venv/bin/python examples/magia_example.py
```

This writes:

```text
generated/magia_example.pipeline.json
```

## Run the full MAGIA flow

The repository root `Makefile` runs the MAPS-side example first, then delegates
the MLIR translation and MAGIA header/data generation to `maps-ir/Makefile`.

Configure `maps-ir` once before running the full flow:

```bash
cmake -S maps-ir -B maps-ir/build \
  -DMLIR_DIR=/path/to/llvm-install/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-install/lib/cmake/llvm
```

Then run:

```bash
make magia-example
```

This builds `maps-translate`, generates the pipeline JSON, converts it
to MAPS MLIR, and emits the MAGIA header and data files:

```text
generated/magia_example.pipeline.json
generated/magia_example.pipeline.mlir
generated/magia_example.h
generated/magia_example_data.c
```

You can also run individual root targets:

```bash
make pipeline-json
make maps-translate
make maps-mlir
make magia-header
make magia-data
```

Clean generated MLIR/header/data artifacts with:

```bash
make clean-generated
```

