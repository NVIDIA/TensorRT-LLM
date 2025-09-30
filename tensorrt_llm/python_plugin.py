# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import pickle  # nosec B403
import typing
from copy import deepcopy
from dataclasses import dataclass
from typing import Sequence, Type, Union

import numpy as np
import tensorrt as trt
import torch

from ._common import default_trtnet
from ._utils import (
    TensorWrapper,
    np_dtype_to_trt,
    str_dtype_to_trt,
    torch_dtype_to_trt,
    trt_dtype_to_torch,
)
from .functional import Tensor, _create_tensor
from .plugin.plugin import TRT_LLM_PLUGIN_NAMESPACE

_plugin_registered = dict()


@dataclass(slots=True, frozen=True)
class PluginInfo:
    trt_plugin_version: int
    plugin_namespace: str
    plugin_name: str
    plugin_version: str
    plugin_num_outputs: int

    def __hash__(self):
        return hash((self.plugin_name, self.plugin_namespace, self.plugin_version))

    def __eq__(self, obj):
        if not isinstance(obj, PluginInfo):
            return False
        return (
            self.plugin_name == obj.plugin_name
            and self.plugin_namespace == obj.plugin_namespace
            and self.plugin_version == obj.plugin_version
        )


def make_expr(
    exprBuilder: Union[trt.IExprBuilder, Type[None]],
    dim: Union["DimensionExpr", trt.IDimensionExpr, int, Type[None]],
) -> Union[trt.IDimensionExpr, Type[None]]:
    """Make a dimension expression.

    Parameters:
        exprBuilder: The trt.exprBuilder object. Using it to check whether dim has the same exprBuilder
            or to create trt.IDimensionExpr if necessary.
        dim: The input dim.

    Returns:
        A trt.IDimensionExpr object.
    """
    if isinstance(dim, DimensionExpr):
        assert exprBuilder == dim.exprBuilder
        return dim.expr
    elif isinstance(dim, int):
        return exprBuilder.constant(dim)
    elif dim is None:
        return None
    elif isinstance(dim, trt.IDimensionExpr):
        return dim
    else:
        raise Exception


def expr_operation(
    a: Union[trt.IDimensionExpr, Type[None]],
    b: Union[trt.IDimensionExpr, Type[None]],
    operation: trt.DimensionOperation,
    exprBuilder: trt.IExprBuilder,
):
    """The function to do expr operation with None support."""
    if exprBuilder is None or a is None or b is None:
        expr = None
    else:
        expr = exprBuilder.operation(operation, a, b)
    return DimensionExpr(expr, exprBuilder)


class DimensionExpr:
    """The class to wrap `trt.IDimensionExpr` to support more pythonic methods."""

    def __init__(
        self,
        expr: Union[trt.IDimensionExpr, int, Type[None]],
        exprBuilder: Union[trt.IExprBuilder, Type[None]],
    ):
        self.exprBuilder = exprBuilder
        self.expr = expr

    @property
    def expr(self):
        return self._expr

    @expr.setter
    def expr(self, expr: Union["DimensionExpr", trt.IDimensionExpr, int, Type[None]]):
        self._expr = make_expr(self.exprBuilder, expr)

    def __add__(self, expr: Union["DimensionExpr", trt.IDimensionExpr, int, Type[None]]):
        expr = make_expr(self.exprBuilder, expr)
        return expr_operation(self.expr, expr, trt.DimensionOperation.SUM, self.exprBuilder)

    def __radd__(self, expr: Union["DimensionExpr", trt.IDimensionExpr, int, Type[None]]):
        return self.__add__(expr)

    def __mul__(self, expr: Union["DimensionExpr", trt.IDimensionExpr, int, Type[None]]):
        expr = make_expr(self.exprBuilder, expr)
        return expr_operation(self.expr, expr, trt.DimensionOperation.PROD, self.exprBuilder)

    def __rmul__(self, expr: Union["DimensionExpr", trt.IDimensionExpr, int, Type[None]]):
        return self.__mul__(expr)

    def __sub__(self, expr: Union["DimensionExpr", trt.IDimensionExpr, int, Type[None]]):
        expr = make_expr(self.exprBuilder, expr)
        return expr_operation(self.expr, expr, trt.DimensionOperation.SUB, self.exprBuilder)

    def __rsub__(self, expr: Union["DimensionExpr", trt.IDimensionExpr, int, Type[None]]):
        expr = make_expr(self.exprBuilder, expr)
        return expr_operation(expr, self.expr, trt.DimensionOperation.SUB, self.exprBuilder)

    def __eq__(self, expr: Union["DimensionExpr", trt.IDimensionExpr, int, Type[None]]):
        expr = make_expr(self.exprBuilder, expr)
        return expr_operation(self.expr, expr, trt.DimensionOperation.EQUAL, self.exprBuilder)

    def __lt__(self, expr: Union["DimensionExpr", trt.IDimensionExpr, int, Type[None]]):
        expr = make_expr(self.exprBuilder, expr)
        return expr_operation(self.expr, expr, trt.DimensionOperation.LESS, self.exprBuilder)

    def __floordiv__(self, expr: Union["DimensionExpr", trt.IDimensionExpr, int, Type[None]]):
        expr = make_expr(self.exprBuilder, expr)
        return expr_operation(self.expr, expr, trt.DimensionOperation.FLOOR_DIV, self.exprBuilder)

    def __rfloordiv__(self, expr: Union["DimensionExpr", trt.IDimensionExpr, int, Type[None]]):
        expr = make_expr(self.exprBuilder, expr)
        return expr_operation(expr, self.expr, trt.DimensionOperation.FLOOR_DIV, self.exprBuilder)

    def __truediv__(self, expr: Union["DimensionExpr", trt.IDimensionExpr, int, Type[None]]):
        expr = make_expr(self.exprBuilder, expr)
        return expr_operation(self.expr, expr, trt.DimensionOperation.CEIL_DIV, self.exprBuilder)

    def __rtruediv__(self, expr: Union["DimensionExpr", trt.IDimensionExpr, int, Type[None]]):
        expr = make_expr(self.exprBuilder, expr)
        return expr_operation(expr, self.expr, trt.DimensionOperation.CEIL_DIV, self.exprBuilder)

    def max(self, expr: Union["DimensionExpr", trt.IDimensionExpr, int, Type[None]]):
        expr = make_expr(self.exprBuilder, expr)
        return expr_operation(self.expr, expr, trt.DimensionOperation.MAX, self.exprBuilder)

    def min(self, expr: Union["DimensionExpr", trt.IDimensionExpr, int, Type[None]]):
        expr = make_expr(self.exprBuilder, expr)
        return expr_operation(self.expr, expr, trt.DimensionOperation.MIN, self.exprBuilder)


class ShapeExpr:
    """The class to Wrap `trt.DimsExprs` to support more pythonic methods."""

    def __init__(
        self,
        dims: Union[Sequence[trt.IDimensionExpr], Sequence[int], Sequence[type[None]]],
        exprBuilder: Union[trt.IExprBuilder, type[None]],
    ):
        self.exprBuilder = exprBuilder
        self.dims = dims

    @property
    def dims(self):
        return self._dims

    @dims.setter
    def dims(
        self,
        dims: Sequence[Union["DimensionExpr", trt.IDimensionExpr, int, Type[None]]],
    ):
        if dims is not None:
            self._dims = [
                DimensionExpr(make_expr(self.exprBuilder, i), self.exprBuilder) for i in dims
            ]
        else:
            self._dims = None

    def __getitem__(self, index: int):
        if self._dims is not None:
            return self._dims[index]
        else:
            return DimensionExpr(None, self.exprBuilder)

    def __setitem__(
        self,
        index: int,
        value: Union["DimensionExpr", trt.IDimensionExpr, int, Type[None]],
    ):
        if self._dims is None:
            return
        assert index < len(self._dims)
        value = DimensionExpr(make_expr(self.exprBuilder, value), self.exprBuilder)
        self._dims[index] = value

    def __len__(self):
        if self._dims is None:
            return 0
        else:
            return len(self._dims)

    def to_trt(self) -> trt.DimsExprs:
        return trt.DimsExprs([i.expr for i in self.dims])


class SymTensor:
    """The class to represent symbolic tensors.

    Only contains dtype and shape information for users to write their own shape/dtype inference function.
    """

    def __init__(
        self,
        dtype: Union[torch.dtype, np.dtype, str, trt.DataType, Type[None]],
        shape: Union[ShapeExpr, Sequence[int]],
    ):
        self.dtype = dtype
        self.shape = shape

    @property
    def shape(self) -> Union[ShapeExpr, Sequence[int]]:
        return self._shape

    @shape.setter
    def shape(self, shape: Union[ShapeExpr, Sequence[int]]):
        assert isinstance(shape, (ShapeExpr, list, tuple))
        if isinstance(shape, (list, tuple)):
            for i in shape:
                assert isinstance(i, int)
        self._shape = shape

    @property
    def dtype(self) -> Union[trt.DataType, Type[None]]:
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: Union[torch.dtype, str, np.dtype, trt.DataType, Type[None]]):
        if isinstance(dtype, torch.dtype):
            self._dtype = torch_dtype_to_trt(dtype)
        elif isinstance(dtype, str):
            self._dtype = str_dtype_to_trt(dtype)
        elif isinstance(dtype, np.dtype):
            self._dtype = np_dtype_to_trt(dtype)
        elif isinstance(dtype, trt.DataType):
            self._dtype = dtype
        elif dtype is None:
            self._dtype = None
        else:
            raise TypeError(f"Unsupported dtype: {dtype}")


def _convert_return_value_to_list(ret):
    if not isinstance(ret, (list, tuple)):
        return [ret]
    assert isinstance(ret, (list, tuple))
    return ret


class PluginBase(
    trt.IPluginV3, trt.IPluginV3OneCore, trt.IPluginV3OneBuild, trt.IPluginV3OneRuntime
):
    """The base class of TRT-LLM plugin.

    All TRT-LLM plugin should inherit this class and at least rewrite `forward` and `shape_dtype_inference`
    function. `forward` defines the plugin's compute flow while `shape_dtype_inference` defines how would
    the output tensor's shape and dtype be inferenced from the input tensor.
    """

    _plugin_creator = None
    _no_serialize_attr = {"_current_stream", "_workspace"}

    def __init__(self):
        cls = type(self)
        # Runtime check for plugin decorator
        assert cls._plugin_creator is not None, (
            "Please make sure the plugin is registered through `@trtllm_plugin`"
        )
        assert cls != PluginBase

        trt.IPluginV3.__init__(self)
        trt.IPluginV3OneCore.__init__(self)
        trt.IPluginV3OneBuild.__init__(self)
        trt.IPluginV3OneRuntime.__init__(self)

        self.plugin_phase = trt.TensorRTPhase.BUILD
        self.num_outputs = self._num_outputs
        self.plugin_namespace = self._plugin_namespace
        self.plugin_name = self._plugin_name
        self.plugin_version = self._plugin_version
        self.current_stream = -1
        self.workspace = 0  # nullptr

    @property
    def current_stream(self):
        if self._current_stream == -1:
            return torch.cuda.current_stream().cuda_stream
        else:
            return self._current_stream

    @current_stream.setter
    def current_stream(self, stream: int):
        assert isinstance(stream, int)
        self._current_stream = stream

    @property
    def workspace(self) -> int:
        buffer = self._workspace
        return buffer if isinstance(buffer, int) else buffer.data_ptr()

    @workspace.setter
    def workspace(self, workspace: Union[int, torch.Tensor]):
        assert isinstance(workspace, (int, torch.Tensor))
        self._workspace = workspace

    def clone(self):
        cls = type(self)
        cloned_plugin = cls.__new__(cls)
        super(cls, cloned_plugin).__init__()
        cloned_plugin.__dict__.update(self._get_dict_to_serialize())
        return cloned_plugin

    def get_capability_interface(self, type):
        return self

    def configure_plugin(self, input_desc, output_desc):
        pass

    def get_output_data_types(self, input_types):
        ret = self.shape_dtype_inference([SymTensor(i, ShapeExpr(None, None)) for i in input_types])

        ret = _convert_return_value_to_list(ret)
        assert len(ret) == self.num_outputs
        for i in ret:
            assert isinstance(i, SymTensor)

        return [i.dtype for i in ret]

    def get_output_shapes(self, inputs, shape_inputs, exprBuilder):
        assert len(shape_inputs) == 0, "Currently we do not support shape inputs"

        ret = self.shape_dtype_inference(
            [SymTensor(None, ShapeExpr(i, exprBuilder)) for i in inputs]
        )

        ret = _convert_return_value_to_list(ret)
        assert len(ret) == self.num_outputs
        for i in ret:
            assert isinstance(i, SymTensor)

        return [i.shape.to_trt() for i in ret]

    def supports_format_combination(self, pos, in_out, num_inputs):
        """By default, TRT-LLM plugin supports all dtype and linear format.

        It is the users responsibility to check the dtype the plugin supported in `forward` function.
        """
        assert pos < len(in_out)

        desc = in_out[pos].desc
        if desc.format != trt.TensorFormat.LINEAR:
            return False

        return True

    def attach_to_context(self, context):
        return self.clone()

    def get_fields_to_serialize(self):
        buffer = pickle.dumps(self._get_dict_to_serialize())
        return trt.PluginFieldCollection(
            [trt.PluginField("__plugin_pickle_obj__", buffer, trt.PluginFieldType.UNKNOWN)]
        )

    def enqueue(self, input_desc, output_desc, inputs, outputs, workspace, stream):
        torch_stream = torch.cuda.ExternalStream(stream_ptr=stream)
        self.workspace = workspace
        self.current_stream = stream

        with torch.cuda.stream(torch_stream):
            self.forward(
                tuple(
                    TensorWrapper.from_trt_desc(input_desc[i], inputs[i])
                    for i in range(len(input_desc))
                ),
                tuple(
                    TensorWrapper.from_trt_desc(output_desc[i], outputs[i])
                    for i in range(len(output_desc))
                ),
            )

        self.current_stream = -1

    def __call__(self, *args: Union[Sequence[TensorWrapper], Sequence[torch.Tensor]]):
        is_trtllm = True
        for i in args:
            is_trtllm &= isinstance(i, Tensor)

        if not is_trtllm:
            for i in args:
                assert isinstance(i, torch.Tensor), (
                    "Plugin inputs must be `tensorrt_llm.Tensor`s or `torch.Tensor`s"
                )
            sym_tensors = self.shape_dtype_inference(
                [SymTensor(i.dtype, [j for j in i.shape]) for i in args]
            )
            sym_tensors = _convert_return_value_to_list(sym_tensors)
            ret = [
                torch.empty(sym_tensor.shape, dtype=trt_dtype_to_torch(sym_tensor.dtype))
                for sym_tensor in sym_tensors
            ]
            self.current_stream = torch.cuda.current_stream().cuda_stream
            self.workspace = torch.empty(self.workspace)
            self.forward(args, ret)
        else:
            args = [i.trt_tensor for i in args]
            layer_plugin = default_trtnet().add_plugin_v3(args, [], self)
            ret = [
                _create_tensor(layer_plugin.get_output(i), layer_plugin)
                for i in range(self.num_outputs)
            ]

        if len(ret) == 1:
            return ret[0]

        return ret

    def on_shape_change(self, input_desc, output_desc):
        pass

    def get_valid_tactics(self):
        return []

    def set_tactic(self, index):
        if index != 0:
            raise RuntimeError(
                "By default TRT should not set tactics since PluginBase do not provide custom tactic."
            )

    def forward(self, inputs: Sequence[TensorWrapper], outputs: Sequence[TensorWrapper]):
        """Expect users to rewrite this function to define the compute flow.

        There are a few special attributes for users to get access to some resources.

        `self.workspace`: The workspace address of TRT managed workspace.
        `self.current_stream`: The CUDA stream this plugin is expected to execute on. By default
        `PluginBase` set the torch.cuda.current_stream() to this stream. This attribute is for the
        toolkit that doesn't work with torch's stream.
        """
        raise NotImplementedError

    def shape_dtype_inference(self, inputs: Sequence[SymTensor]):
        """Expect users to rewrite this function to define the shape dtype inference for output tensors."""
        raise NotImplementedError

    def _get_dict_to_serialize(self):
        ret = {}
        for k, v in self.__dict__.items():
            if k not in self._no_serialize_attr:
                ret[k] = deepcopy(v) if self.deepcopy_clone else v
        return ret


class PluginCreatorBase(trt.IPluginCreatorV3One):
    def __init__(self):
        super().__init__()

    def create_plugin(self, name, fc, phase):
        if len(fc) == 1 and fc[0].name == "__plugin_pickle_obj__":
            data = fc[0].data
            plugin_dict = pickle.loads(data)  # nosec B301
            plugin = self.plugin_cls.__new__(self.plugin_cls)
            super(self.plugin_cls, plugin).__init__()
            plugin.__dict__.update(plugin_dict)
        else:
            raise RuntimeError("Expect to be called by TRT")
        plugin.plugin_phase = phase
        return plugin


def trtllm_plugin(
    plugin_name: str,
    *,
    plugin_version: str = "1",
    plugin_namespace: str = TRT_LLM_PLUGIN_NAMESPACE,
    plugin_num_outputs: Union[int, Type[None]] = None,
    deepcopy_clone: bool = True,
    no_serialize_attr: Sequence[str] = set(),
):
    def plugin_registration(plugin_cls):
        assert issubclass(plugin_cls, PluginBase)
        assert hasattr(plugin_cls, "__dict__"), (
            "Plugin wrapper uses `__dict__` to track plugin states"
        )
        nonlocal plugin_num_outputs

        annotation = inspect.signature(plugin_cls.shape_dtype_inference).return_annotation
        origin_annotation = typing.get_origin(annotation)
        if origin_annotation is tuple or annotation is SymTensor:
            if origin_annotation is tuple:
                element_types = typing.get_args(annotation)
                for ty in element_types:
                    assert ty == SymTensor, (
                        f"Plugin {plugin_name}'s `shape_dtype_inference` return annotation must be SymTensor "
                        "or a tuple of SymTensor"
                    )
                infered_num_outputs = len(element_types)
            else:
                infered_num_outputs = 1
            if plugin_num_outputs is not None:
                assert plugin_num_outputs == infered_num_outputs, (
                    f"Plugin {plugin_name}'s `_num_outputs` and return annotation mismatch, "
                    f"{plugin_cls._num_outputs} != {infered_num_outputs}"
                )
            plugin_num_outputs = infered_num_outputs
        else:
            assert plugin_num_outputs is not None, (
                "Must specify `num_outputs` or valid `shape_dtype_inference` return annotation for "
                f"{plugin_name}. The valid types are SymTensor or a tuple of SymTensor, got {annotation}."
            )

        plugin_info = PluginInfo(
            3, plugin_namespace, plugin_name, plugin_version, plugin_num_outputs
        )
        assert plugin_info not in _plugin_registered, (
            f"Redefine plugin with info: {plugin_info} which is previously defined as "
            f"{_plugin_registered[plugin_info]}"
        )

        _plugin_registered[plugin_info] = plugin_info
        plugin_cls._plugin_name = plugin_name
        plugin_cls._plugin_version = plugin_version
        plugin_cls._plugin_namespace = plugin_namespace
        plugin_cls._num_outputs = plugin_num_outputs
        plugin_cls.deepcopy_clone = deepcopy_clone
        plugin_cls._no_serialize_attr.update(no_serialize_attr)

        plugin_registry = trt.get_plugin_registry()

        plugin_creator = PluginCreatorBase()
        plugin_creator.name = plugin_cls._plugin_name
        plugin_creator.plugin_namespace = plugin_cls._plugin_namespace
        plugin_creator.plugin_version = plugin_cls._plugin_version
        plugin_creator.field_names = trt.PluginFieldCollection([])
        plugin_creator.plugin_cls = plugin_cls

        plugin_cls._plugin_creator = plugin_creator
        ret = plugin_registry.register_creator(plugin_creator, plugin_cls._plugin_namespace)

        assert ret, f"Plugin: {plugin_cls} register failed, please check the error log."

        return plugin_cls

    return plugin_registration
