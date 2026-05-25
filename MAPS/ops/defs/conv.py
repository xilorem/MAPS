"""Conv payload IR using an im2col execution model."""

from __future__ import annotations

from dataclasses import dataclass
from MAPS.core.graph import Node, OpKind
from MAPS.core.layout import LayoutAxis, LayoutAxisMode, TensorLayout, TensorRange, TensorSlice, TensorSliceRef, tile_tensor_slice
from MAPS.core.submesh import Submesh
from MAPS.core.tensor import Tensor
from MAPS.ops.common.payload import OpPayload
from MAPS.ops.common.tile_work import TileWork
from MAPS.ops.registry import register_op
from MAPS.ops.spec import OpSpec
from MAPS.ops.defs.conv_transforms import (
    ChannelShardedBiasAddPayload,
    ChannelShardedGemmPayload,
    Im2ColPayload,
    OutputReformatPayload,
    WeightPackPayload,
)


@dataclass(frozen=True)
class ConvTileWork(TileWork):
    """Concrete Conv slices associated with one tile."""

    output_slice: TensorSlice
    x_slice: TensorSlice
    w_slice: TensorSlice
    b_slice: TensorSlice | None
    x: Tensor | None = None
    w: Tensor | None = None
    b: Tensor | None = None
    output: Tensor | None = None

    @property
    def input_slices(self) -> tuple[TensorSliceRef, ...]:
        refs = []
        if self.x is not None:
            refs.append(TensorSliceRef(tensor=self.x, tensor_slice=self.x_slice))
        if self.w is not None:
            refs.append(TensorSliceRef(tensor=self.w, tensor_slice=self.w_slice))
        if self.b is not None and self.b_slice is not None:
            refs.append(TensorSliceRef(tensor=self.b, tensor_slice=self.b_slice))
        return tuple(refs)

    @property
    def output_slices(self) -> tuple[TensorSliceRef, ...]:
        if self.output is None:
            return ()
        return (TensorSliceRef(tensor=self.output, tensor_slice=self.output_slice),)


def _range(start: int, length: int) -> TensorRange:
    return TensorRange(start=start, length=length)


def _full_range(dim: int) -> TensorRange:
    return TensorRange(start=0, length=dim)


@dataclass(frozen=True)
class ConvPayload(OpPayload):
    """2D NCHW Conv payload modeled as im2col plus GEMM.

    The planner-side convention is:
    - ``x`` has shape ``[N, C, H, W]``
    - ``w`` has shape ``[OC, C, KH, KW]`` for ``group == 1``
    - optional ``b`` has shape ``[OC]``
    - ``output`` has shape ``[N, OC, OH, OW]``
    """

    x: Tensor
    w: Tensor
    b: Tensor | None
    output: Tensor
    strides: tuple[int, int] = (1, 1)
    pads: tuple[int, int, int, int] = (0, 0, 0, 0)
    dilations: tuple[int, int] = (1, 1)
    group: int = 1

    def __post_init__(self) -> None:
        if len(self.strides) != 2:
            raise ValueError("Conv strides must have length 2")
        if len(self.pads) != 4:
            raise ValueError("Conv pads must have length 4")
        if len(self.dilations) != 2:
            raise ValueError("Conv dilations must have length 2")
        if any(value <= 0 for value in self.strides):
            raise ValueError("Conv strides must be > 0")
        if any(value < 0 for value in self.pads):
            raise ValueError("Conv pads must be >= 0")
        if any(value <= 0 for value in self.dilations):
            raise ValueError("Conv dilations must be > 0")
        if self.group != 1:
            raise NotImplementedError("grouped Conv is not implemented")
        self.validate_shapes()

    @property
    def cost_model(self) -> object:
        from MAPS.ops.costs.conv_cost import ConvCostModel

        return ConvCostModel()

    def output_layouts(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None = None,
    ) -> tuple[TensorLayout, ...]:
        return (self._output_layout(submesh, logical_shape),)

    def _logical_dims(
        self,
        logical_shape: tuple[int, int] | None,
    ) -> tuple[int | None, int | None]:
        if logical_shape is None:
            return None, None
        return logical_shape

    def _output_layout(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None,
    ) -> TensorLayout:
        logical_width, logical_height = self._logical_dims(logical_shape)
        return TensorLayout(
            submesh=submesh,
            mesh_x=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=1),
            mesh_y=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=3),
            logical_width=logical_width,
            logical_height=logical_height,
        )

    def _x_layout(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None,
    ) -> TensorLayout:
        logical_width, logical_height = self._logical_dims(logical_shape)
        return TensorLayout(
            submesh=submesh,
            mesh_x=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
            mesh_y=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=3),
            logical_width=logical_width,
            logical_height=logical_height,
        )

    def _w_layout(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None,
    ) -> TensorLayout:
        logical_width, logical_height = self._logical_dims(logical_shape)
        return TensorLayout(
            submesh=submesh,
            mesh_x=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=0),
            mesh_y=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
            logical_width=logical_width,
            logical_height=logical_height,
        )

    def _b_layout(
        self,
        submesh: Submesh,
        logical_shape: tuple[int, int] | None,
    ) -> TensorLayout:
        logical_width, logical_height = self._logical_dims(logical_shape)
        return TensorLayout(
            submesh=submesh,
            mesh_x=LayoutAxis(mode=LayoutAxisMode.SHARD, tensor_axis=0),
            mesh_y=LayoutAxis(mode=LayoutAxisMode.REPLICATE),
            logical_width=logical_width,
            logical_height=logical_height,
        )

    def required_x_slice(self, output_slice: TensorSlice) -> TensorSlice:
        if output_slice.rank != self.output.rank:
            raise ValueError("output slice rank must match Conv output rank")

        _, _, input_h, input_w = self.x.dims
        _, _, kernel_h, kernel_w = self.w.dims
        stride_h, stride_w = self.strides
        pad_top, pad_left, _, _ = self.pads
        dilation_h, dilation_w = self.dilations

        out_h = output_slice.dims[2]
        out_w = output_slice.dims[3]
        h_start = out_h.start * stride_h - pad_top
        h_end = (
            (out_h.start + out_h.length - 1) * stride_h
            - pad_top
            + dilation_h * (kernel_h - 1)
            + 1
        )
        w_start = out_w.start * stride_w - pad_left
        w_end = (
            (out_w.start + out_w.length - 1) * stride_w
            - pad_left
            + dilation_w * (kernel_w - 1)
            + 1
        )

        h_start = max(0, h_start)
        w_start = max(0, w_start)
        h_end = min(input_h, h_end)
        w_end = min(input_w, w_end)

        return TensorSlice(
            rank=self.x.rank,
            dims=(
                output_slice.dims[0],
                _full_range(self.x.dims[1]),
                _range(h_start, max(0, h_end - h_start)),
                _range(w_start, max(0, w_end - w_start)),
            ),
        )

    def required_w_slice(self, output_slice: TensorSlice) -> TensorSlice:
        if output_slice.rank != self.output.rank:
            raise ValueError("output slice rank must match Conv output rank")
        return TensorSlice(
            rank=self.w.rank,
            dims=(
                output_slice.dims[1],
                _full_range(self.w.dims[1]),
                _full_range(self.w.dims[2]),
                _full_range(self.w.dims[3]),
            ),
        )

    def required_b_slice(self, output_slice: TensorSlice) -> TensorSlice | None:
        if self.b is None:
            return None
        return TensorSlice(rank=self.b.rank, dims=(output_slice.dims[1],))

    def build_tile_work(
        self,
        output_layouts: tuple[TensorLayout, ...],
        tile: Tile,
    ) -> ConvTileWork:
        output_slice = tile_tensor_slice(
            tensor=self.output,
            layout=output_layouts[0],
            tile=tile,
        )
        return ConvTileWork(
            output_slice=output_slice,
            x_slice=self.required_x_slice(output_slice),
            w_slice=self.required_w_slice(output_slice),
            b_slice=self.required_b_slice(output_slice),
            x=self.x,
            w=self.w,
            b=self.b,
            output=self.output,
        )

    def validate_shapes(self) -> None:
        if self.x.rank != 4:
            raise ValueError("Conv X tensor must be NCHW rank 4")
        if self.w.rank != 4:
            raise ValueError("Conv W tensor must be OIHW rank 4")
        if self.output.rank != 4:
            raise ValueError("Conv output tensor must be NCHW rank 4")
        if self.x.elem_bytes != self.w.elem_bytes or self.x.elem_bytes != self.output.elem_bytes:
            raise ValueError("Conv tensors must agree on element size")
        if self.b is not None:
            if self.b.rank != 1:
                raise ValueError("Conv bias tensor must be rank 1")
            if self.b.elem_bytes != self.output.elem_bytes:
                raise ValueError("Conv bias element size must match output tensor")

        batch, in_channels, input_h, input_w = self.x.dims
        out_channels, weight_channels, kernel_h, kernel_w = self.w.dims
        out_batch, out_channels_actual, output_h, output_w = self.output.dims
        if batch != out_batch:
            raise ValueError("Conv input and output batch dimensions must match")
        if in_channels != weight_channels:
            raise ValueError("Conv input channels must match weight channels")
        if out_channels != out_channels_actual:
            raise ValueError("Conv weight output channels must match output channels")
        if self.b is not None and self.b.dims != (out_channels,):
            raise ValueError("Conv bias shape must match output channels")

        stride_h, stride_w = self.strides
        pad_top, pad_left, pad_bottom, pad_right = self.pads
        dilation_h, dilation_w = self.dilations
        expected_h = (
            input_h
            + pad_top
            + pad_bottom
            - dilation_h * (kernel_h - 1)
            - 1
        ) // stride_h + 1
        expected_w = (
            input_w
            + pad_left
            + pad_right
            - dilation_w * (kernel_w - 1)
            - 1
        ) // stride_w + 1
        if (output_h, output_w) != (expected_h, expected_w):
            raise ValueError("Conv output spatial dimensions do not match parameters")


def lower_conv_node(
    node_name: str,
    inputs: tuple[Tensor, ...],
    outputs: tuple[Tensor, ...],
    attributes: dict[str, object],
) -> tuple[OpKind, object]:
    """Lower one ONNX Conv node into scheduler-side Conv semantics."""

    if len(inputs) not in (2, 3):
        raise ValueError(f"Conv node '{node_name}' must have 2 or 3 inputs")
    if len(outputs) != 1:
        raise ValueError(f"Conv node '{node_name}' must have exactly 1 output")
    if "auto_pad" in attributes and attributes["auto_pad"] != "NOTSET":
        raise NotImplementedError("Conv auto_pad is not implemented")

    return (
        OpKind.CONV,
        ConvPayload(
            x=inputs[0],
            w=inputs[1],
            b=inputs[2] if len(inputs) == 3 else None,
            output=outputs[0],
            strides=tuple(attributes.get("strides", (1, 1))),
            pads=tuple(attributes.get("pads", (0, 0, 0, 0))),
            dilations=tuple(attributes.get("dilations", (1, 1))),
            group=int(attributes.get("group", 1)),
        ),
    )


def decompose_conv_node(node: Node) -> tuple[tuple[Tensor, ...], tuple[Node, ...]]:
    """Lower one NCHW Conv into explicit transforms, GEMM, and optional bias."""

    if not isinstance(node.payload, ConvPayload):
        raise TypeError("decompose_conv_node expects a Node with ConvPayload payload")

    op = node.payload
    n, _, _, _ = op.x.dims
    oc, c, kh, kw = op.w.dims
    _, _, oh, ow = op.output.dims
    matrix_shape = (n * oh * ow, oc)
    patch_shape = (n * oh * ow, c * kh * kw)
    packed_weight_shape = (c * kh * kw, oc)

    patches = Tensor(
        name=f"{node.name}__patches",
        rank=2,
        dims=patch_shape,
        elem_bytes=op.output.elem_bytes,
    )
    packed_weights = Tensor(
        name=f"{node.name}__packed_w",
        rank=2,
        dims=packed_weight_shape,
        elem_bytes=op.output.elem_bytes,
    )
    matmul_output = Tensor(
        name=f"{node.name}__matmul",
        rank=2,
        dims=matrix_shape,
        elem_bytes=op.output.elem_bytes,
    )

    group_attributes = dict(node.attributes)
    group_attributes["stage_group_id"] = f"{node.name}::conv_gemm"

    nodes = [
        Node(
            name=f"{node.name}__im2col",
            kind=OpKind.TRANSFORM,
            inputs=(op.x,),
            outputs=(patches,),
            payload=Im2ColPayload(
                x=op.x,
                output=patches,
                kernel_shape=(kh, kw),
                strides=op.strides,
                pads=op.pads,
                dilations=op.dilations,
            ),
            attributes={**group_attributes, "conv_step": "im2col"},
        ),
        Node(
            name=f"{node.name}__weight_pack",
            kind=OpKind.TRANSFORM,
            inputs=(op.w,),
            outputs=(packed_weights,),
            payload=WeightPackPayload(w=op.w, output=packed_weights),
            attributes={**group_attributes, "conv_step": "weight_pack"},
        ),
        Node(
            name=f"{node.name}__gemm",
            kind=OpKind.GEMM,
            inputs=(patches, packed_weights),
            outputs=(matmul_output,),
            payload=ChannelShardedGemmPayload(
                x=patches,
                w=packed_weights,
                y=None,
                output=matmul_output,
            ),
            attributes={**group_attributes, "conv_step": "gemm"},
        ),
    ]
    new_tensors = [patches, packed_weights, matmul_output]
    reshape_input = matmul_output

    if op.b is not None:
        biased_output = Tensor(
            name=f"{node.name}__biased",
            rank=2,
            dims=matrix_shape,
            elem_bytes=op.output.elem_bytes,
        )
        new_tensors.append(biased_output)
        nodes.append(
            Node(
                name=f"{node.name}__bias_add",
                kind=OpKind.ELEMENTWISE,
                inputs=(matmul_output, op.b),
                outputs=(biased_output,),
                payload=ChannelShardedBiasAddPayload(
                    op_name="Add",
                    lhs=matmul_output,
                    rhs=op.b,
                    output=biased_output,
                ),
                attributes={**group_attributes, "conv_step": "bias_add"},
            )
        )
        reshape_input = biased_output

    nodes.append(
        Node(
            name=f"{node.name}__output_reformat",
            kind=OpKind.TRANSFORM,
            inputs=(reshape_input,),
            outputs=(op.output,),
            payload=OutputReformatPayload(x=reshape_input, output=op.output),
            attributes={**group_attributes, "conv_step": "output_reformat"},
        )
    )
    return tuple(new_tensors), tuple(nodes)


register_op(
    OpSpec(
        name="conv",
        onnx_names=("Conv",),
        lower_onnx=lower_conv_node,
        decompose=decompose_conv_node,
        payload_type=ConvPayload,
    )
)
