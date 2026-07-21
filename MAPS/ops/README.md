# MAPS operation model

An operation enters MAPS as either a **primitive** or a **composite** payload.
The distinction is intentional:

- A primitive `OpPayload` can choose output layouts, build exact per-tile work,
  and expose an `OpCostModel`.
- A composite `CompositeOpPayload` only validates high-level semantics and
  decomposes itself into primitive nodes before planning.

The planner never needs the frontend registry to execute an operation.
`OpSpec` only connects external names, currently ONNX op types, to lowering
functions and provides discoverable metadata.

## Primitive operation checklist

1. Define an immutable payload in `defs/` and validate its complete semantic
   contract in `__post_init__`.
2. Implement `output_layouts`. Current primitives have one output and should
   call `single_output_layout` when consuming the result.
3. Define an immutable `TileWork` containing the exact input and output slices
   required by one tile.
4. Implement `build_tile_work` without architecture-specific constants.
5. Implement an `OpCostModel`. Local execution belongs in `cost`; costs that
   require the complete placement belong in `placement_cost`.
6. Add focused tests for invalid semantics, layouts, tile slices, L1 footprint,
   and costs.
7. If the operation is imported, add an `OpSpec` and a lowerer that either
   normalizes external semantics exactly or rejects unsupported cases clearly.

## Composite operation checklist

1. Define an immutable `CompositeOpPayload` and validate the high-level shape
   and attribute contract.
2. Implement `decompose(node)` and return new tensors plus primitive nodes.
3. Do not add layout, tile-work, or cost behavior to the composite. Those
   belong to the primitive operations produced by decomposition.
4. Preserve dependencies explicitly and group nodes only when they must share
   one stage.

## Identity and responsibility

- `OpSpec.name` identifies a frontend registration.
- `OpKind` is a broad graph/runtime category.
- The payload class defines planner semantics.
- `WorkKind` selects a hardware-device capability and throughput.

These values serve different layers. Payload constructors must derive or
validate `WorkKind` so that callers cannot create contradictory operation and
device identities.

Shared multidirectional broadcasting and slice projection live in
`common/broadcast.py`. Shared execution and cost contracts live in
`common/payload.py`, `common/tile_work.py`, and `common/cost.py`.
