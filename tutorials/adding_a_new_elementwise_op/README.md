## Adding a new elementwise operation

The following steps are a small guide to define a custom elementwise operation in MAPS.

For most elementwise operations, it is not necessary to create a new file in `./MAPS/ops/defs`. The file `./MAPS/ops/defs/elementwise.py` already contains the generic logic for:

- unary elementwise payloads
- binary elementwise payloads
- output layouts
- tile work construction
- broadcasting rules
- ONNX lowering

So, in most cases, a new elementwise operation only needs to be added to the existing registration tables.

### 1. Decide if the operation is unary or binary

A unary elementwise op has:

- one input
- one output
- same input and output shape

A binary elementwise op has:

- two inputs
- one output
- standard ONNX-style broadcasting

If the operation follows one of these two patterns, it can reuse the existing implementation in `./MAPS/ops/defs/elementwise.py`.

### 2. Pick the execution category

The file already maps ONNX op names to a `WorkKind`.

For unary ops, the table is:

```python
UNARY_ELEMENTWISE_OPS: dict[str, WorkKind] = {
    "Abs": WorkKind.ELEMENTWISE,
    "Exp": WorkKind.EXP,
    "Neg": WorkKind.ELEMENTWISE,
    "Sqrt": WorkKind.ELEMENTWISE,
}
```

For binary ops, the table is:

```python
BINARY_ELEMENTWISE_OPS: dict[str, WorkKind] = {
    "Add": WorkKind.ELEMENTWISE,
    "Div": WorkKind.ELEMENTWISE,
    "Mul": WorkKind.ELEMENTWISE,
    "Pow": WorkKind.ELEMENTWISE,
    "Sub": WorkKind.ELEMENTWISE,
}
```

If the new operation behaves like a standard elementwise op, it can reuse `WorkKind.ELEMENTWISE`.

If the hardware and cost model should treat it differently, it can use another already existing work kind such as `WorkKind.EXP`.

### 3. Add the ONNX op name to the proper table

For a unary operation, add it to `UNARY_ELEMENTWISE_OPS`.

For example:

```python
UNARY_ELEMENTWISE_OPS: dict[str, WorkKind] = {
    "Abs": WorkKind.ELEMENTWISE,
    "Exp": WorkKind.EXP,
    "Neg": WorkKind.ELEMENTWISE,
    "Sqrt": WorkKind.ELEMENTWISE,
    "Log": WorkKind.ELEMENTWISE,
}
```

For a binary operation, add it to `BINARY_ELEMENTWISE_OPS`.

For example:

```python
BINARY_ELEMENTWISE_OPS: dict[str, WorkKind] = {
    "Add": WorkKind.ELEMENTWISE,
    "Div": WorkKind.ELEMENTWISE,
    "Mul": WorkKind.ELEMENTWISE,
    "Pow": WorkKind.ELEMENTWISE,
    "Sub": WorkKind.ELEMENTWISE,
    "Max": WorkKind.ELEMENTWISE,
}
```

### 4. Reuse the existing lowerers

The file already has generic lowerers:

- `_lower_unary_elementwise_node`
- `_lower_binary_elementwise_node`

So once the new ONNX name is inserted in the proper table, the registration path can reuse the same payload, layout, tile work, and cost model logic.


