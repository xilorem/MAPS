## Adding a new operation

The following steps are a small guide to define a custom operation in MAPS.

### 1. Create the operation file

The high level file should be created inside `./MAPS/ops/defs`. For this tutorial let's create a file named `tutorial.py`.

### 2. Create the Payload object

The payload object should inherit from the `OpPayload` class. The attributes of the object should be the tensors involved in the computation. For this tutorial let's make a class with 2 inputs and an output.

```python
@dataclass(frozen=True)
class TutorialPayload(OpPayload):
    input_1: Tensor
    input_2: Tensor
    output: Tensor
```

The payload also needs a `cost_model` property. If the new operation can reuse an already existing cost model, it can simply return it.

```python
@property
def cost_model(self) -> object:
    return GemmCostModel()
```
A custom cost model can be defined in a .py file in `./MAPS/ops/costs` following another cost template.

If the operation needs shape validation, it should be added inside `__post_init__`.

### 3. Create the output TensorLayout

A `TensorLayout` defines how a `Tensor` object should be partitioned on a `Submesh`. For example, an output could be sharded across both axes of the `Submesh`. In this case the implementation would look like this:

```python
def output_layouts(
    self,
    submesh: Submesh,
    logical_shape: tuple[int, int] | None = None,
) -> tuple[TensorLayout, ...]:

    logical_width = None
    logical_height = None
    if logical_shape is not None:
        logical_width, logical_height = logical_shape

    return (
        TensorLayout(
            submesh=submesh,
            mesh_x=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=self.output.rank - 1),
            mesh_y=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=self.output.rank - 2),
            logical_width=logical_width,
            logical_height=logical_height,
        ),
    )
```

### 4. Create the Tile Work

The `Payload` object needs a method to be attached named `build_tile_work`. This method should take as inputs an output layouts tuple and a tile, and return the operation specific `TileWork` object.

```python
def build_tile_work(
    self,
    output_layouts: tuple[TensorLayout, ...],
    tile: Tile,
) -> TutorialTileWork:
```

The `TileWork` represents the tensor slices needed by the tile to compute the output slice required by the output layout.

For this example, the `TutorialTileWork` will look like a matmul.

```python
@dataclass(frozen=True)
class TutorialTileWork(TileWork):
    output_slice: TensorSlice
    input_1_slice: TensorSlice
    input_2_slice: TensorSlice
    input_1: Tensor
    input_2: Tensor
    output: Tensor
```

The `TileWork` object still needs to expose the `input_slices` and `output_slices` properties.

```python
@property
def input_slices(self) -> tuple[TensorSliceRef, ...]:
    return (
        TensorSliceRef(tensor=self.input_1, tensor_slice=self.input_1_slice),
        TensorSliceRef(tensor=self.input_2, tensor_slice=self.input_2_slice),
    )

@property
def output_slices(self) -> tuple[TensorSliceRef, ...]:
    return (
        TensorSliceRef(tensor=self.output, tensor_slice=self.output_slice),
    )
```

The `build_tile_work` method can then use `tile_tensor_slice` to derive the output slice owned by one tile and derive the required input slices from it.

```python
def build_tile_work(
    self,
    output_layouts: tuple[TensorLayout, ...],
    tile: Tile,
) -> TutorialTileWork:

    output_slice = tile_tensor_slice(
        tensor=self.output,
        layout=output_layouts[0],
        tile=tile,
    )

    return TutorialTileWork(
        output_slice=output_slice,
        input_1_slice=self.required_input_1_slice(output_slice),
        input_2_slice=self.required_input_2_slice(output_slice),
        input_1=self.input_1,
        input_2=self.input_2,
        output=self.output,
    )
```

The input slices must be implemented through custom logic. In this case they are:

```python
def required_input_1_slice(self, output_slice: TensorSlice) -> TensorSlice:
    dims = list(output_slice.dims[:-2])
    dims.append(output_slice.dims[-2])
    dims.append(_full_range(self.input_1.dims[-1]))
    return TensorSlice(rank=self.input_1.rank, dims=tuple(dims))

def required_input_2_slice(self, output_slice: TensorSlice) -> TensorSlice:
    dims = list(output_slice.dims[:-2])
    dims.append(_full_range(self.input_2.dims[-2]))
    dims.append(output_slice.dims[-1])
    return TensorSlice(rank=self.input_2.rank, dims=tuple(dims))
```

If the cost model needs extra helpers such as `operation_count()` or `dimensions()`, they should be added to the `TileWork` object too.

### 5. Create an ONNX lowerer

An operation needs a function to lower an ONNX node to the MAPS IR.

```python
def lower_tutorial_node(
    node_name: str,
    inputs: tuple[Tensor, ...],
    outputs: tuple[Tensor, ...],
    attributes: dict[str, object],
) -> tuple[OpKind, object]:
    del node_name, attributes
    return (
        OpKind.CUSTOM,
        TutorialPayload(
            input_1=inputs[0],
            input_2=inputs[1],
            output=outputs[0],
        ),
    )
```

For a custom operation the returned `OpKind` should be `OpKind.CUSTOM`, unless the graph IR is extended with a new explicit kind.

### 6. Register the operation

Then, just register the operation:

```python
register_op(
    OpSpec(
        name="tutorial",
        onnx_names=("Tutorial",),
        lower_onnx=lower_tutorial_node,
        payload_type=TutorialPayload,
        work_kinds=(WorkKind.GEMM,),
    )
)
```

The `work_kinds` field should describe which execution category the operation uses. If the operation behaves like a matmul, it can reuse `WorkKind.GEMM`. A new `WorkKind` is only needed if the hardware and cost-model behavior is genuinely different from the existing ones.

### 7. Make sure the operation is imported

If the new operation should be available as part of the builtins, it also needs to be imported from `MAPS/ops/defs/__init__.py`.

