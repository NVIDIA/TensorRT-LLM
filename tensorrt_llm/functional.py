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
import math
import weakref
from collections import OrderedDict
from enum import IntEnum
from functools import partial
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

# isort: off
import torch
import tensorrt as trt
# isort: on

from . import graph_rewriting as gw
from ._common import default_net, default_trtnet, precision
from ._utils import (bf16_array, dim_resolve_negative, dim_to_trt_axes,
                     fp16_array, fp32_array, int32_array, np_dtype_to_trt,
                     str_dtype_to_np, str_dtype_to_trt, torch_to_numpy,
                     trt_dtype_to_np, trt_dtype_to_torch)
from .logger import logger
from .network import PluginInfo, set_np_weight, set_plugin_info
from .plugin import TRT_LLM_PLUGIN_NAMESPACE, current_all_reduce_helper
from .quantization import QuantMode


class DimRange(object):
    '''
        One DimRange object stores the ranges of all the dimensions of one tensor in one optimization profile.
        For example, tensor has 2 dimensions. Then the data members are:
            self.min = [dim 0 min, dim 1 min]
            self.opt = [dim 0 opt, dim 1 opt]
            self.max = [dim 0 max, dim 1 max]
        For static dimension, it has min==opt==max, thus the \p shape param in the ctor can be an integer
    '''

    def __init__(self, shape: List[Union[int, List[int], Tuple[int, int, int]]],
                 names: List[str]):
        '''
        Parameters:
            shape: a list with length N, each element is an integer or a 3-elements tuple/list of int,
                where N is the number of dimensions for a tensor.
                When one element is an integer, it means that dimension is static.
                Otherwise, when one element is a tuple/list, it means the dimension is dynamic.
                The 3 elements in one tuple/list is ordered by (min, opt, max), and this function asserts
                0 <= min <= opt <= max.

                Example, for a 3 rank tensor, with 1st dimension being static and has value 16, and second dimension being dynamic with
                min/opt/max values being 1/8/32, and 3rd dimension being static and has value 8.
                The shape parameter could be:
                    [16, (1, 8, 32), 8]
                It has same semantics of
                    [(16, 16, 16), (1, 8, 32), (8, 8, 8)]
        '''
        self.min = []
        self.opt = []
        self.max = []
        self.dimension_names = names
        assert len(names) == len(
            shape
        ), "Expecting shape list and name list must have same length, got {shape=}, {name=}"
        for dim in shape:
            if isinstance(dim, (list, tuple)):
                assert len(dim) == 3 and 0 <= dim[0] <= dim[1] <= dim[2], \
                "Each dimension must specify a 3-elements tuple or list in the order of (min,opt,max), got {dim=}"
                self.min.append(dim[0])
                self.opt.append(dim[1])
                self.max.append(dim[2])
            elif isinstance(dim, int):
                self.min.append(dim)
                self.opt.append(dim)
                self.max.append(dim)
            else:
                raise AttributeError(
                    f'Dimension should be [min, opt, max] (dynamic shape) or int (specific value). Got {type(dim)}'
                )

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, DimRange) and \
            self.dimension_names == __value.dimension_names and \
            self.min == __value.min and self.opt == __value.opt and self.max == __value.max

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"{self.dimension_names=} {self.min=}, {self.opt=}, {self.max=})"

    def __hash__(self) -> int:
        return hash(str(self))


class Tensor(object):
    '''
    The class to represent dense tensors.

    A dense tensor is named, has a shape and contains typed elements. Each
    dimension of a tensor can either be static or dynamic. Static dimensions
    are known at engine compilation by TensorRT. Dynamic dimensions can take
    values determined at runtime. The tensor can be located on the host (CPU)
    or the device (GPU).
    '''

    def __init__(self,
                 name=None,
                 dtype=None,
                 shape=None,
                 dim_range=None,
                 is_network_input=True,
                 location=trt.TensorLocation.DEVICE,
                 network=None,
                 trt_tensor=None):
        '''
        Parameters:
            name : str
                The name of the tensor.

            dtype : tensorrt.DataType
                The type of the elements of the tensor. See the TensorRT
                documentation for list of supported data types.

            shape : tensorrt.Dims
                The dimensions of the tensor. In TensorRT-LLM, tensors can have
                static or dynamic dimensions (it is possible to mix static and
                dynamic dimensions).  A static dimension is known when the
                TensorRT engine is built. A dynamic dimension can be set when
                the engine is executed. Use -1 for dynamic dimensions.

            dim_range : OrderedDict
                An ordered dictionary (the positions of the elements matter)
                that associates a name and a range of values to the dimensions.
                For a static dimension, the range must be limited to a single
                value. For a dynamic dimension, the range is defined by three
                values [min, opt, max] where min and max are, respectively, the
                smallest and largest possible values of that dimension.  The
                opt value is used by TensorRT to optimize the engine for the
                most common case.

                Assume there is N optimization profiles, each item dim_range dict is ordered by:
                 (dynamic dimension name : [profile 0 (min, opt, max), profile 1 (min, opt, max), ... profile N(min, opt, max)])
                or it's following when the dimension is static (can think as min==opt==max):
                 (static dimension name : [profile 0 value, profile 1 value, ... profile N value])
                For static dimension the profile 0-N value must be same, (TODO: can it be simplified to be only 1 value?)
                And number of keys is equal to number of optimization profiles.

            is_network_input : bool
                A boolean indicating if that tensor is an input of the network.
                Inputs must be provided by the user to run the engine.

            location : tensorrt.TensorLocation
                A flag to indicate where the tensor will be located. It can be
                on the host (CPU) or the device (GPU).

            network: Network
                A parent Network instance, that helps to fine the users of this tensor.

            trt_tensor: trt.ITensor
                Construct with the ITensor instance directly, and no shape profiles are required.
        '''

        # Layout of self.profiles
        # Opt profile 0: dim 0 (min, opt, max), dim 1 (min, opt, max) ... dim M
        # Opt profile 1: dim 0 (min, opt, max), dim 1 (min, opt, max) ... dim M
        # ...
        # Opt profile N: dim 0 ...                                        dim M

        # So from the dim_range arg to self.profiles conversion, there is a layout transpose
        # dim_range arg is: {M dimension x N profiles}, while self.profiles layout is {N profiles x M dimensions}

        self.profiles = []

        self.is_tensor_wrapper = False  # specially for the graph rewriter

        # work as a wrapper for a trt.ITensor, this is used specially in the graph rewriter
        if trt_tensor is not None:
            self.is_tensor_wrapper = True
            assert network is not None
            self.trt_tensor = trt_tensor
            self._network = weakref.ref(network)
            assert not is_network_input, "is_network_input should be False when trt_tensor is not None"
            return

        # be cautious here, the weakref is critical to avoid circular referencing before Network and Tensor
        # using strong reference will likely cause significant peak memory increase, since Network objects
        # holds the weights data.
        self._network = weakref.ref(default_net())
        if is_network_input:
            if dim_range is not None:
                assert isinstance(dim_range, OrderedDict)
                assert len(
                    dim_range
                ) >= 1, f"Each input tensor shall have at least one dimension, tensor '{name}' found {dim_range=}"

                found_profiles = [
                    len(ranges) for _, ranges in dim_range.items()
                ]
                assert all(
                    [x == found_profiles[0] for x in found_profiles]
                ), f"Expecting all the dimensions in the dim_range has same number of profiles, tensor '{name}' got {dim_range=}"

                num_opt_profile = len(list(dim_range.items())[0][1])
                assert num_opt_profile >= 1
                for i in range(num_opt_profile):
                    range_shape = []
                    dimension_names = []
                    for dim, ranges in dim_range.items():
                        assert isinstance(ranges, (list, tuple))
                        range_shape.append(ranges[i])
                        dimension_names.append(dim)
                    self.profiles.append(DimRange(range_shape, dimension_names))

            default_net()._add_input(self, name, dtype, shape, dim_range)
            self.name = name
            self.dtype = dtype
            self.shape = shape
            self.location = location

    @property
    def network(self):
        return self._network()

    @property
    def name(self):
        '''
        The name of the tensor.
        '''
        return self.trt_tensor.name

    @name.setter
    def name(self, name):
        '''
        Set the name of the tensor.
        '''
        if name is not None:
            self.trt_tensor.name = name

    @property
    def dtype(self):
        '''
        The type of the elements in the tensor.
        '''
        return self.trt_tensor.dtype

    @dtype.setter
    def dtype(self, dtype):
        '''
        Set the type of the elements in the tensor.
        '''
        if dtype is not None:
            self.trt_tensor.dtype = dtype

    @property
    def shape(self):
        '''
        The shape of the tensor.
        '''
        return self.size()

    @shape.setter
    def shape(self, shape):
        '''
        Set the shape of the tensor. See __init__.
        '''
        if shape is not None:
            self.trt_tensor.shape = shape

    @property
    def location(self):
        '''
        The physical location of the tensor (on the host or the device).
        '''
        return self.trt_tensor.location

    @location.setter
    def location(self, location):
        '''
        Set the physical location of the tensor (on the host or the device). See __init__.
        '''
        if location is not None:
            self.trt_tensor.location = location

    def mark_output(self,
                    name: Optional[str] = None,
                    dtype: Optional[Union[str, trt.DataType]] = None):
        '''
        Mark a tensor as a network output.

        When a tensor is marked as an output, its content can be obtained after
        the execution of the TensorRT engine. The user is responsible for
        allocating buffers to store the output tensors when preparing the
        execution of the TensorRT engine.
        '''
        if name is None:
            name = self.name

        if dtype is None:
            dtype = self.dtype
        elif isinstance(dtype, str):
            dtype = str_dtype_to_trt(dtype)
        else:
            assert isinstance(dtype, trt.DataType)

        default_net()._mark_output(self, name, dtype)

    def __add__(self, b):
        '''
        See functional.add.
        '''
        return add(self, b)

    def __radd__(self, b):
        '''
        See functional.add.
        '''
        return add(b, self)

    def __sub__(self, b):
        '''
        See functional.sub.
        '''
        return sub(self, b)

    def __rsub__(self, b):
        '''
        See functional.sub.
        '''
        return sub(b, self)

    def __mul__(self, b):
        '''
        See functional.mul.
        '''
        return mul(self, b)

    def __rmul__(self, b):
        '''
        See functional.mul.
        '''
        return mul(b, self)

    def __truediv__(self, b):
        '''
        See functional.div.
        '''
        return div(self, b)

    def __lt__(self, b):
        '''
        See functional.lt.
        '''
        return lt(self, b)

    def __gt__(self, b):
        '''
        See functional.gt.
        '''
        return gt(self, b)

    def __eq__(self, b):
        '''
        See functional.eq.
        '''
        if self.is_tensor_wrapper:
            # for graph rewriter
            return hash(self) == hash(b)
        else:
            # for creating the network
            return eq(self, b)

    def __ge__(self, b):
        '''
        Maps to functional.gt or functional.eq.
        '''
        return op_or(self.__gt__(b), self.__eq__(b))

    def __le__(self, b):
        '''
        Maps to functional.lt or functional.eq.
        '''
        return op_or(self.__lt__(b), self.__eq__(b))

    def view(self, shape, zero_is_placeholder=True):
        '''
        See functional.view.
        '''
        return view(self, shape, zero_is_placeholder)

    def permute(self, dims):
        '''
        See functional.permute.
        '''
        return permute(self, dims)

    def transpose(self, dim0, dim1):
        '''
        See functional.transpose.
        '''
        return transpose(self, dim0, dim1)

    def mean(self, dim, keepdim=False):
        '''
        See functional.mean.
        '''
        return mean(self, dim, keepdim)

    def max(self, dim, keepdim=False):
        '''
        See functional.max.
        '''
        return max(self, dim, keepdim)

    def abs(self):
        '''
        See functional.abs.
        '''
        return abs(self)

    def sqrt(self):
        '''
        See functional.sqrt.
        '''
        return sqrt(self)

    def cast(self, dtype):
        '''
        See functional.cast.
        '''
        return cast(self, dtype)

    def size(self, dim=None):
        '''
        Returns the shape of the tensor if the dim parameter is None.
        Otherwise, returns a size of the dimension indicated by dim. The
        behavior is undefined if dim is negative or exceeds the rank of the
        tensor.
        '''
        if dim is None:
            return self.trt_tensor.shape

        return self.trt_tensor.shape[dim]

    def rank(self):
        '''
        Returns the rank (i.e. the number of dimensions) of the tensor.
        '''
        return len(self.trt_tensor.shape)

    def ndim(self):
        '''
        Returns the rank (i.e. the number of dimensions) of the tensor.
        '''
        return self.rank()

    def split(self, split_size_or_sections, dim=0):
        '''
        See functional.split.
        '''
        return split(self, split_size_or_sections, dim)

    def is_dynamic(self, dim=None):
        '''
        If the argument 'dim' is None, that function returns a boolean that
        indicates if the tensor contains a dynamic dimension (True) or not
        (False). In that case, the first dimension is excluded (as it usually
        corresponds to the batch size).  If the argument is an integer, that
        functions returns a boolean that indicates if the dimension 'dim' is
        dynamic (True) or not (False).
        '''
        if dim is not None:
            return self.trt_tensor.shape[dim] == -1

        for i, s in enumerate(self.trt_tensor.shape):
            if i != 0 and s == -1:
                return True

        return False

    # graph writer related functions

    def get_parent(self):
        ''' Get the layer that produces this tensor.  '''
        return self.network.get_tensor_parent(self)

    def get_users(self):
        ''' Get the layers that use this tensor as an input.  '''
        return self.network.get_tensor_users(self)

    def replace_all_uses_with(self, new_tensor):
        '''
        Replace all uses of this tensor as an input to consumer layers
        '''

        self.network.is_graph_altered = True
        users = self.get_users()
        for user in users:
            inputs_changed = 0
            for i in range(user.num_inputs):
                if user.get_inputs(i)[0].trt_tensor is self.trt_tensor:
                    inputs_changed += 1
                    user.set_input(i, new_tensor.trt_tensor)
            assert inputs_changed >= 1, "Tensor not found in layer inputs"

            # update the FLayerMetadata as well
            flayer = gw.FLayerInfoMemo.instance().get(user.name)
            flayer and flayer.replace_input_with(self, new_tensor)

    def is_trt_wrapper(self):
        '''
        Check if there is a trt.ITensor member inside, which is required for
        graph rewriter. In order to differentiate usages, it may be necessary
        to have an inheritance hierarchy.
        '''
        if hasattr(self, 'trt_tensor'):
            return True
        else:
            return False

    def __hash__(self):
        if self.is_trt_wrapper():
            return id(self.trt_tensor)
        else:
            return id(None)

    def __repr__(self):
        return f"TensorRT-LLM Tensor: {self.name=} {self.dtype=} {self.shape=}"


def _create_tensor(trt_tensor: trt.ITensor,
                   producer: trt.ILayer = None) -> Tensor:
    '''
    A helper function to create a TensorRT-LLM Tensor object that encapsulates
    the connection between the TensorRT tensor (trt.ITensor) and the layer
    (trt.ILayer) that produces it.

    That function is expected to be used as:

        # Insert a new layer in the network using the TensorRT API:
        layer = default_trtnet().add_<some_layer>(...)
        # Extract the first output of that layer and connect it to the layer.
        return _create_tensor(layer.get_output(0), layer)

    That function also sets the precision of the layer/producer to the default
    precision of the network.

    Parameters:
        trt_tensor : trt.ITensor
            The TensorRT tensor to connect to its producer (the layer).

        producer : trt.ILayer = None
            The producer.

    Returns:
        The TensorRT-LLM tensor (functional.Tensor) that encapsulates the
        TensorRT tensor and the layer that produces it. The former is
        accessible through the attribute 'trt_tensor' and the latter using the
        attribute 'producer'.
    '''
    assert trt_tensor is not None
    tensor = Tensor(name=trt_tensor.name,
                    dtype=trt_tensor.dtype,
                    shape=trt_tensor.shape,
                    is_network_input=False)
    tensor.trt_tensor = trt_tensor
    tensor.producer = producer

    # Set the layer name since this is the only
    # centralized location to pass the name from
    # module space to the TRT IR
    default_net()._set_layer_name(producer)
    if default_net().dtype is not None and not default_net().strongly_typed:
        if producer.type not in [
                trt.LayerType.SHAPE, trt.LayerType.CONSTANT,
                trt.LayerType.GATHER, trt.LayerType.CONCATENATION
        ]:
            producer.precision = default_net().dtype
    assert tensor is not None

    if gw.FLayerInfoMemo.instance().cur_flayer is not None:
        gw.FLayerInfoMemo.instance().cur_flayer.layer_name = producer.name

    return tensor


def _add_plugin_info(layer, plugin_creator: trt.IPluginCreator,
                     plugin_name: str, pfc: trt.PluginFieldCollection) -> None:
    plugin_info = PluginInfo(plugin_creator, plugin_name, pfc)
    set_plugin_info(default_net().trt_network, layer.name, plugin_info)


class RotaryScalingType(IntEnum):
    none = 0
    linear = 1
    dynamic = 2


class PositionEmbeddingType(IntEnum):
    learned_absolute = 0
    rope_gptj = 1
    rope_gpt_neox = 2
    alibi = 3
    alibi_with_scale = 4
    relative = 5
    chatglm = 6

    def is_rope(self) -> bool:
        return self in [self.rope_gptj, self.rope_gpt_neox]

    def is_alibi(self) -> bool:
        return self in [self.alibi, self.alibi_with_scale]

    @staticmethod
    def choices() -> List[str]:
        return [embedding.name for embedding in PositionEmbeddingType]

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return PositionEmbeddingType[s]
        except KeyError:
            raise ValueError(f'Unsupported position embedding type: {s}')


class AttentionMaskType(IntEnum):
    padding = 0
    causal = 1
    bidirectional = 2
    bidirectionalglm = 3  # TODO: merge this mask into bidirectional


class LayerNormType(IntEnum):
    LayerNorm = 0
    RmsNorm = 1
    GroupNorm = 2


class LayerNormPositionType(IntEnum):
    pre_layernorm = 0
    post_layernorm = 1


class MLPType(IntEnum):
    MLP = 0
    GatedMLP = 1
    FusedGatedMLP = 2


def activation(input: Tensor, act_type: trt.ActivationType) -> Tensor:
    '''
    Add an activation function.

    Parameters:
        input : Tensor
            The input tensor on which the activation function is applied.

        act_type : trt.ActivationType
            The type of the activation (RELU, TANH, SIGMOID, ...).

    The following closures are defined in functional.*:

        relu    for op=trt.ActivationType.RELU
        tanh    for op=trt.ActivationType.TANH
        sigmoid for op=trt.ActivationType.SIGMOID

    Returns:
        The tensor produced by the activation layer.
    '''
    layer = default_trtnet().add_activation(input.trt_tensor, act_type)
    return _create_tensor(layer.get_output(0), layer)


def clip(input: Tensor, alpha: float, beta: float) -> Tensor:
    '''
    Add a CLIP operation that sets the range to [alpha, beta].

    Parameters:
        input : Tensor
            The input tensor on which the activation function is applied.

        alpha : float
            The lower bound of the CLIP function.

        beta : float
            The upper bound of the CLIP function.

    Returns:
        The tensor produced by the activation layer.
    '''
    layer = default_trtnet().add_activation(input.trt_tensor,
                                            trt.ActivationType.CLIP)
    layer.alpha = alpha
    layer.beta = beta
    return _create_tensor(layer.get_output(0), layer)


relu = partial(activation, act_type=trt.ActivationType.RELU)
tanh = partial(activation, act_type=trt.ActivationType.TANH)
sigmoid = partial(activation, act_type=trt.ActivationType.SIGMOID)


def silu(input: Tensor) -> Tensor:
    '''
    Add a SiLU (`x * sigmoid(x)`) operation.

    Parameters:
        input : Tensor
            The input tensor on which the activation function is applied.

    Returns:
        The tensor produced by the activation layer.
    '''
    return input * sigmoid(input)


def swiglu(input: Tensor) -> Tensor:
    '''
    Add a SwiGLU (`x * SiLU(gate)`) operation.

    That function takes a tensor, splits it into two halves along the last
    dimension, applies SiLU to the second half and multiply the results. The
    behaviour is undefined if the last dimension is not even.

    Parameters:
        input : Tensor
            The input tensor on which the activation function is applied.

    Returns:
        The tensor produced by the activation layer.
    '''
    x, gate = chunk(input, 2, dim=-1)
    return silu(gate) * x


def squared_relu(x: Tensor) -> Tensor:
    '''
    Add a Squared ReLU operation.

    This function applies ReLU and squares the output.

    Parameters:
        input : Tensor
            The input tensor on which the activation function is applied.

    Returns:
        The tensor produced by the activation layer.
    '''
    return pow(relu(x), 2.0)


def cast(input: Tensor, dtype: Union[str, trt.DataType]):
    '''
    Add a cast operation.

    For an input tensor of type INT8, this function sets the dynamic range of
    the input to [-127, 127] for automatic dequantization. For a cast into
    INT8, that function sets the dynamic range of the output to [-127, 127] for
    automatic quantization.

    Parameters:
        input : Tensor
            The input tensor on which the cast is applied.

        dtype : str or trt.DataType
            The data type of the output tensor after the cast. When 'dtype' is
            provided as a string, it must be a name amongst the valid names.
            See _str_to_trt_dtype_dict in _utils.py for a list of supported
            types and type names.

    Returns:
        The tensor produced by the inserted layer.
    '''
    if isinstance(dtype, str):
        cvt_dtype = str_dtype_to_trt(dtype)
    elif isinstance(dtype, trt.DataType):
        cvt_dtype = dtype
    else:
        raise TypeError("%s is not supported" % type(dtype))

    if input.dtype == cvt_dtype:
        # If input type and cast dtype are the same, do nothing
        return input

    layer = default_trtnet().add_cast(input.trt_tensor, cvt_dtype)
    if not default_net().strongly_typed:
        layer.set_output_type(0, cvt_dtype)
    output = _create_tensor(layer.get_output(0), layer)
    if input.dtype == str_dtype_to_trt('int8'):
        layer.get_input(0).set_dynamic_range(-127, 127)
    if cvt_dtype == str_dtype_to_trt('int8'):
        layer.get_output(0).set_dynamic_range(-127, 127)

    return output


def flip(input: Tensor, dims: Sequence[int]) -> Tensor:
    '''
    Reverses the order of an n-D tensor along given axis in dims.

    That flip operation maps to a TensorRT ISliceLayer. For the dimensions
    listed in dims it copies the elements from the last one to the first one
    (from (N-1) down to 0 with a step of -1). For the dimensions not in 'dims',
    it copies the elements from the first one to the last one (from 0 to N-1
    with a step of 1).

    Parameters:
        input : Tensor
            The input tensor on which the cast is applied.

        dims : list or tuple
            The axes to flip. Negative indices are supported.

    Returns:
        The tensor produced by the inserted layer.
    '''
    assert not input.is_dynamic()

    ndim = input.ndim()

    for index, value in enumerate(dims):
        assert -ndim <= value < ndim
        if -ndim <= value < 0:
            dims[index] += ndim

    assert len(dims) == len(set(dims))

    start_values = [
        input.size()[i] - 1 if i in dims else 0 for i in range(ndim)
    ]
    stride_values = [-1 if i in dims else 1 for i in range(ndim)]

    layer = default_trtnet().add_slice(input.trt_tensor,
                                       start=start_values,
                                       shape=input.size(),
                                       stride=stride_values)

    return _create_tensor(layer.get_output(0), layer)


def interpolate(input: Tensor,
                size: Union[int, List[int]] = None,
                scale_factor: Union[float, List[float]] = None,
                mode: str = 'nearest',
                align_corners: bool = False,
                recompute_scale_factor: bool = False,
                antialias: bool = False) -> Tensor:
    ##
    ## TODO: Document that function!
    ##

    assert not input.is_dynamic()

    input_ndim = input.ndim()

    assert 2 < input_ndim < 6, "Only 3D, 4D and 5D input Tensors supported"
    assert (size is not None) ^ (
        scale_factor
        is not None), "Only one of out_shape or scales should be defined"

    assert mode in ('nearest', 'linear', 'bilinear', 'bicubic', 'trilinear',
                    'nearest-exact')

    if mode == 'trilinear' and input_ndim != 5:
        raise ValueError("trilinear only supports 5D tensor")

    if mode == "bilinear" and input_ndim != 4:
        raise ValueError("bilinear only supports 4D tensor")

    if mode == "linear" and input_ndim != 3:
        raise ValueError("linear only supports 3D tensor")

    layer = default_trtnet().add_resize(input.trt_tensor)

    input_shape = input.size()

    updated_shape = []
    if scale_factor:
        scale_len = 1 if isinstance(scale_factor,
                                    (float, int)) else len(scale_factor)
        if scale_len == 1 and isinstance(scale_factor, (float, int)):
            updated_scale = [scale_factor for _ in range(input_ndim - 2)]

        else:
            updated_scale = scale_factor
        updated_shape = [
            int(math.floor(updated_scale[i - 2] *
                           input_shape[i])) if i > 1 else input_shape[i]
            for i in range(input_ndim)
        ]

    else:
        size_len = 1 if isinstance(size, int) else len(size)
        assert size_len == input_ndim - 2
        if size_len == 1 and isinstance(size, int):
            updated_size = [size for _ in range(input_ndim - 2)]
        else:
            updated_size = size

        updated_shape = [
            input_shape[i] if i < 2 else updated_size[i - 2]
            for i in range(input_ndim)
        ]
    layer.shape = updated_shape

    if mode in ['nearest', 'nearest-exact'] or mode is None:
        layer.resize_mode = trt.InterpolationMode.NEAREST
        layer.coordinate_transformation = trt.ResizeCoordinateTransformation.ASYMMETRIC
    elif mode in ['linear', 'bilinear', 'trilinear']:
        layer.resize_mode = trt.InterpolationMode.LINEAR
        if align_corners:
            layer.coordinate_transformation = trt.ResizeCoordinateTransformation.ALIGN_CORNERS
        else:
            layer.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL
        # TODO, need to confirm the align_corners effect on bilinear mode.
        if mode == 'bilinear':
            layer.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL

    elif mode in ['bicubic']:
        layer.resize_mode = trt.InterpolationMode.CUBIC

        layer.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL

    else:
        layer.resize_mode = trt.InterpolationMode.NEAREST
        layer.coordinate_transformation = trt.ResizeCoordinateTransformation.ASYMMETRIC

    return _create_tensor(layer.get_output(0), layer)


def matmul(input: Tensor,
           mat2: Tensor,
           transa: bool = False,
           transb: bool = False,
           use_fp32_acc: bool = True) -> Tensor:
    '''
    Add a matrix multiplication.

    That operation maps to a tensorrt.IMatrixMultiplyLayer layer. As explained
    in the TensorRT documentation, it computes the inner product between the
    two inputs after applying an optional transposition on the inputs.

    Parameters:
        input : Tensor
            The first tensor (often called A).

        mat2 : Tensor
            The second tensor (often called B).

        transa : bool
            Is the first input transposed? Set to 'True' if you want the first
            input to be transposed, 'False' otherwise.

        transb : bool
            Is the second input transposed? Set to 'True' if you want the
            second input to be transposed, 'False' otherwise.

        use_fp32_acc: bool
            Set to 'True' if for accuracy reason, this fp16 matmul needs to use
            fp32 accumulation. This can be a per model and per matmul decision.
    Returns:
        The tensor produced by the inserted layer.
    '''
    # This option is only supported for fp16, but not bf16 or any other precisions.
    # TODO: fp32 accum has issues with strongly_typed and it will be fixed in TensorRT 10.0
    use_fp32_acc = use_fp32_acc and input.dtype == trt.DataType.HALF and mat2.dtype == trt.DataType.HALF and not default_net(
    ).strongly_typed
    if use_fp32_acc:
        input = cast(input, 'float32')
        mat2 = cast(mat2, 'float32')

    input, mat2 = broadcast_helper(input, mat2)
    op0 = trt.MatrixOperation.TRANSPOSE if transa \
        else trt.MatrixOperation.NONE
    op1 = trt.MatrixOperation.TRANSPOSE if transb \
        else trt.MatrixOperation.NONE
    layer = default_trtnet().add_matrix_multiply(input.trt_tensor, op0,
                                                 mat2.trt_tensor, op1)
    output = _create_tensor(layer.get_output(0), layer)
    if use_fp32_acc:
        output = cast(output, "float16")

    return output


def constant(ndarray: np.ndarray) -> Tensor:
    '''
    Add a constant layer.

    TensorRT graphs encapsulate constant values in the form of constant layers
    (tensorrt.IConstantLayer). That function creates such a layer from a Numpy
    array of values. After compilation of the network by TensorRT, those
    weights are stored in the serialized TensorRT engine.

    Parameters:
        ndarray : numpy.ndarray
            The array of values (weights) encapsulated by this constant layer.

    Returns:
        The tensor produced by the inserted layer.
    '''
    weights = trt.Weights(np_dtype_to_trt(ndarray.dtype), ndarray.ctypes.data,
                          ndarray.size)
    # Prevent underlying numpy array from going out of scope
    default_net().register_ndarray(ndarray)
    layer = default_trtnet().add_constant(trt.Dims(ndarray.shape), weights)
    if not default_net().strongly_typed:
        layer.set_output_type(0, np_dtype_to_trt(ndarray.dtype))
    tensor = _create_tensor(layer.get_output(0), layer)
    # TODO: remove this WAR after https://nvbugs/4359151 fixed.
    set_np_weight(default_trtnet(), layer.name, ndarray)
    return tensor


# TODO: TensorRT uses sizes of the output dimensions.
# DL framework uses ends usually. Will change it to ends.
def slice(input: Tensor,
          starts: Union[Tensor, Sequence[int]],
          sizes: Union[Tensor, Sequence[int]],
          strides: Union[Tensor, Sequence[int]] = None,
          mode: trt.SampleMode = None) -> Tensor:
    '''
    Add an operation to extract a slice from a tensor.

    As described in the TensorRT documentation of the ISliceLayer, the slice
    layer has two variants: Static and dynamic.

    For static slicing, this function takes the starts and sizes values in the
    different dimensions to slice at layer creation time via a sequence of
    integers. For dynamic slicing, it accepts starts and sizes as
    tensorrt.ITensor`s.

    The slice layer selects for each dimension a start location from within the
    input tensor, and copies elements to the output tensor using a stride of 1
    across the input tensor. Start and size tensors must be 1-D int32 shape
    tensors if not specified as a sequence of integers.

    As an example, on input = [[0, 2, 4], [1, 3, 5]], the call to

        slice(input, start=[1, 0], size=[1, 2])

    will produce the tensor [[1, 3]] as output. The slice operator when
    executed by TensorRT will copy one row (because size[0] == 1) starting from
    the 2nd row (because start[0] == 1) and two columns (size[1] == 2) starting
    from the 1st column (because start[1] == 0).

    In pseudo-code the behaviour of that operation can be described as follows
    for a 2D tensor (and easily be extended to more dimensions):

        output = Tensor(shape=sizes)
        for ii in range(sizes[0]):
            for jj in range(sizes[1]):
                output[ii][jj] = input[starts[0]+ii][starts[1]+jj]

    Note that it is common in deep-learning frameworks to use ranges
    [start:end] for similar operations. It can be emulated by setting the sizes
    argument such that in each dimension [start:start+size] == [start:end] i.e.
    size = end-start.

    TensorRT supports different slice modes but that function restricts that
    choice to `mode == tensorrt.SampleMode.STRICT_BOUNDS`.

    Parameters:
        input : Tensor
            The input tensor on which the slicing is performed.

        starts : Union[Tensor, Sequence[int]]
            The starting points, in the input tensor, and each dimension.

        sizes : Union[Tensor, Sequence[int]]
            The number of elements in each dimension of the sliced tensor (output).

        strides : Union[Tensor, Sequence[int]]
            The step be taken from start, in input tensor.

        mode : trt.SampleMode
            The mode that controls how the slice operation handles out of bounds coordinates.

    Returns:
        The tensor produced by the slice layer.
    '''
    input_ndim = input.ndim()

    trt_starts = starts
    if isinstance(starts, Tensor):
        trt_starts = [0 for _ in range(input_ndim)]  # unused dummy value

    trt_sizes = sizes
    if isinstance(sizes, Tensor):
        trt_sizes = [1 for _ in range(input_ndim)]  # unused dummy value

    trt_strides = strides
    if isinstance(strides, Tensor) or strides is None:
        trt_strides = [1 for _ in range(input_ndim)]

    layer = default_trtnet().add_slice(input.trt_tensor,
                                       start=trt_starts,
                                       shape=trt_sizes,
                                       stride=trt_strides)
    if mode is not None:
        layer.mode = mode

    if isinstance(starts, Tensor):
        layer.set_input(1, starts.trt_tensor)

    if isinstance(sizes, Tensor):
        layer.set_input(2, sizes.trt_tensor)

    if isinstance(strides, Tensor):
        layer.set_input(3, strides.trt_tensor)

    return _create_tensor(layer.get_output(0), layer)


# TODO: support step.
def arange(start: Union[Tensor, int], end: Union[Tensor, int],
           dtype: str) -> Tensor:
    '''
    Add an operation to fill a 1D tensor.

    The tensor is filled with the values between start and end with a step of 1
    between the different elements. In pseudo-code, it corresponds to a tensor
    populated with the values:

        output = Tensor([dtype(ii) for ii in range(start, end, 1)])

    For example, a call to arange(3, 6, 'int32') will add an operation to the
    TensorRT graph that will produce [3, 4, 5] when executed. The call to
    arange(2, 5, 'float32') will add a layer to generate [2.0, 3.0, 4.0].

    This operation is implemented using a tensorrt.IFillLayer in
    trt.FillOperation.LINSPACE mode.

    Parameters:
        start : Union[Tensor, int]
            The starting point of the range.

        end : Union[Tensor, int]
            The end point of the range.

        dtype : str
            The type of the elements. See _str_to_trt_dtype_dict in _utils.py
            for a list of supported types and type names.

    Returns:
        The tensor produced by the fill layer. It is a 1D tensor containing
        `end-start` elements of type `dtype`.
    '''
    if isinstance(start, int):
        assert isinstance(end, int)
        start = constant(int32_array(start))
        end = constant(int32_array(end))
    elif isinstance(start, Tensor):
        assert isinstance(end, Tensor)
    else:
        raise TypeError("%s is not supported" % type(start))

    step = constant(int32_array([1]))

    num = end - start
    num = num.view([1])

    layer = default_trtnet().add_fill([0], trt.FillOperation.LINSPACE,
                                      trt.int32)
    layer.set_input(0, num.trt_tensor)  # rank = 1
    layer.set_input(1, start.trt_tensor)  # rank = 0
    layer.set_input(2, step.trt_tensor)  # rank = 1
    tensor = _create_tensor(layer.get_output(0), layer)
    if tensor.dtype != str_dtype_to_trt(dtype):
        tensor = tensor.cast(dtype)
    return tensor


def expand(input: Tensor, expand_shape: Tensor) -> Tensor:
    '''
    Add an operation to expand a tensor.

    The operation expands the input tensor in the singleton dimensions to the
    size indicated by the corresponding dimension in the `expand_shape` tensor.
    In other words, given an input tensor with dimensions of size 1, those
    dimensions will be expanded to the size in `expand_shape`.

    For example, a tensor of shape [4, 3, 1, 3] will be expanded to a tensor of
    shape [4, 3, 2, 3] by the layer created using expand(input, [4, 3, 2, 3]).

    The expansion may either replicate the values or be mapped to a view with a
    stride of 0 in the expanded dimensions. For example, for a tensor [[3, 2]] of
    shape [1, 2],

        expand([[3, 2]], [2, 2])

    can be used to expand the input to [[3, 2], [3, 2]].

    This operation is implemented using a tensorrt.ISliceLayer. The current
    implementation does not verify that non singleton dimensions are not
    shrunk. In other words, for an input of shape [4, 1, 2],

        expand(input, [3, 2, 2])

    will produce a tensor of shape [3, 2, 2]. That behaviour is subject to
    change in the future.

    Parameters:
        input : Tensor
            The input tensor.

        expand_shape : Tensor
            The new shape of the expanded tensor.

    Returns:
        The tensor produced by the expand layer.
    '''
    ndim = input.rank()
    layer = default_trtnet().add_slice(
        input.trt_tensor,
        start=[0 for _ in range(ndim)],
        shape=[1 for _ in range(ndim)],  # unused dummy value
        stride=[1 for _ in range(ndim)]  # unused dummy value
    )

    # The stride is either:
    #   0 for dimensions of size 1 (i.e. shape(input, i) - 1 == 1 - 1 == 0) or,
    #   1 for dimensions of size > 1 since minimum(value >= 1, 1) == 1.
    stride_tensor = concat(
        [minimum((shape(input, i) - 1), 1) for i in range(ndim)])

    layer.set_input(2, expand_shape.trt_tensor)
    layer.set_input(3, stride_tensor.trt_tensor)
    return _create_tensor(layer.get_output(0), layer)


def einsum(einsum_eq: str, inputs: Sequence[Tensor]) -> Tensor:
    '''
    Add an Einsum operation.

    That operation maps to tensorrt.IEinsumLayer. As explained in the TensorRT
    documentation, this layer implements a summation over the elements of the
    inputs along dimensions specified by the equation parameter, based on the
    Einstein summation convention. The layer can have one or more inputs of
    rank >= 0.  All the inputs must be of same data type. This layer supports
    all TensorRT data types except bool. There is one output tensor of the same
    type as the input tensors. The shape of output tensor is determined by the
    equation.

    The equation specifies ASCII lower-case letters for each dimension in the
    inputs in the same order as the dimensions, separated by comma for each
    input. The dimensions labeled with the same subscript must match or be
    broadcastable. Repeated subscript labels in one input take the diagonal.
    Repeating a label across multiple inputs means that those axes will be
    multiplied. Omitting a label from the output means values along those axes
    will be summed. In implicit mode, the indices which appear once in the
    expression will be part of the output in increasing alphabetical order. In
    explicit mode, the output can be controlled by specifying output subscript
    labels by adding an arrow (‘->’) followed by subscripts for the output. For
    example, “ij,jk->ik” is equivalent to “ij,jk”. Ellipsis (‘…’) can be used
    in place of subscripts to broadcast the dimensions. See the TensorRT
    Developer Guide for more details on equation syntax.

    Many common operations can be expressed using the Einsum equation. For
    example:
        Matrix Transpose: ij->ji
        Sum: ij-> Matrix-Matrix
        Multiplication: ik,kj->ij
        Dot Product: i,i->
        Matrix-Vector Multiplication: ik,k->i
        Batch Matrix Multiplication: ijk,ikl->ijl
        Batch Diagonal: …ii->…i

    Note that TensorRT does not support ellipsis or diagonal operations so,
    neither, does TensorRT-LLM.

    Parameters:
        einsum_eq : str
            The Einsum equation.

        inputs: Sequence[Tensor]
            The sequence of inputs consumed by the Einsum operation.

    Returns:
        The tensor produced by the Einsum operation.
    '''
    layer = default_trtnet().add_einsum([i.trt_tensor for i in inputs],
                                        einsum_eq)
    return _create_tensor(layer.get_output(0), layer)


def permute(input: Tensor, dims: Sequence[int]) -> Tensor:
    '''
    Add an operation to permute the dimensions of a tensor.

    The dimensions of the input tensor are permutted according to the sequence
    of dimensions in 'dims'. That operation maps to tensorrt.IShuffleLayer where
    the second transposition is described by the indices in 'dims'.

    Given a tensor of rank N, the result of the permutation is a tensor of rank
    N in which the i-th input dimension maps to the dims[i]-th dimension.

    For example, permute(input, [1, 0]) will transpose a 2D tensor by permuting
    the rows and columns.

    Parameters:
        input : Tensor
            The input tensor to permute.

        dims : Sequence[int]
            The description of the permutation.

    Returns:
        The tensor produced by the permutation layer.
    '''
    dims = dim_resolve_negative(tuple(dims), input.ndim())
    layer = default_trtnet().add_shuffle(input.trt_tensor)
    layer.second_transpose = dims
    return _create_tensor(layer.get_output(0), layer)


def transpose(input: Tensor, dim0: int, dim1: int) -> Tensor:
    '''
    Add an operation to transpose two dimensions of a tensor.

    That operation produces a tensor in which the dimensions 'dim0' and 'dim1'
    are permuted. The other dimensions, if the rank of the tensor is greater
    than 2, remain untouched.

    That function is a helper built on the 'functional.permute' function.

    Parameters:
        input : Tensor
            The input tensor to transpose.

        dim0 : int
            The first dimension to transpose.

        dim1 : int
            The second dimension to transpose.

    Returns:
        The tensor produced by the permutation layer.
    '''
    permutation = list(range(input.ndim()))
    permutation[dim0] = dim1
    permutation[dim1] = dim0

    return permute(input, permutation)


def view(input: Tensor,
         shape: Union[Tensor, Sequence[int]],
         zero_is_placeholder: bool = True) -> Tensor:
    '''
    Add an operation to create a view of a tensor.

    That operation adds a tensorrt.IShuffleLayer to the network. If the 'shape'
    parameter is a Tensor, that view is dynamic. Otherwise, it is a static
    view.

    Note that TensorRT limits the number of inferred dimensions to 1. It means
    that the shape sequence or tensor cannot contain more than one -1. This
    function enforces that constraint and will assert if it is not respected.

    Parameters:
        input : Tensor
            The input tensor to transpose.

        shape : Union[Tensor, Sequence[int]]
            The shape of the new tensor.

        zero_is_placeholder : bool
            When that parameter is True, the 0s in 'shape' are replaced by the
            sizes of the corresponding dimensions from the 'input'. Otherwise,
            the dimensions corresponding to 0s are shrunk.

    Returns:
        The tensor produced by the view/shuffle layer.
    '''

    # TensorRT demands that at most one dimension is permitted to be specified as -1
    def assert_no_more_than_one_inferred_dim(list):
        inferred_dim_list = [i for i in list if i == -1]
        assert len(inferred_dim_list) <= 1

    layer = default_trtnet().add_shuffle(input.trt_tensor)
    layer.zero_is_placeholder = zero_is_placeholder
    if isinstance(shape, Tensor):
        assert_no_more_than_one_inferred_dim(shape.shape)
        layer.set_input(1, shape.trt_tensor)
    elif isinstance(shape, (list, tuple)):
        assert_no_more_than_one_inferred_dim(shape)
        layer.reshape_dims = tuple(shape)
    else:
        raise TypeError("%s is not supported" % type(shape))
    return _create_tensor(layer.get_output(0), layer)


def expand_dims(input: Tensor, dim: Union[int, Sequence[int]]) -> Tensor:
    '''
    Add an operation to expand the tensor shape with singleton dimensions.

    That function adds a tensorrt.IShuffleLayer to the network. Given an 'input'
    of rank N and a sequence of M dimensions, the output tensor produced by
    this operation (when executed by TensorRT) will have a rank of N+M. Singleton
    dimensions will be inserted at the different positions in 'dim'.

    The pseudo-code for that operation is:

        new_shape, ii = [], 0
        for jj in range(input.rank() + len(dim)):
            new_shape.append(1 if jj in dims else input.shape[ii++])

    For example, for a tensor of shape [3, 4, 1, 5]

        expand_dims(input, [0, 2])

    will produce a tensor of shape [1, 3, 1, 4, 1, 5].

    Parameters:
        input : Tensor
            The input tensor to expand.

        dim : Union[int, Sequence[int]]
            The positions in the output tensor where to insert singleton
            dimensions.

    Returns:
        The tensor produced by the shuffle layer.
    '''
    if isinstance(dim, int):
        dim = (dim, )

    out_ndim = len(dim) + input.ndim()

    input_shape = shape(input)
    out_shapes = []
    j = 0
    for i in range(out_ndim):
        if i in dim:
            out_shapes.append(1)
        else:
            out_shapes.append(gather(input_shape, 0, j))
            j = j + 1

    out_shape = concat(out_shapes)

    return view(input, out_shape)


def unsqueeze(input: Tensor, axis: int):
    '''
    Add an operation to insert a singleton dimension to a tensor.

    That functions creates an operation that insert a singleton dimension
    (dimension of size 1) at position 'dim' in the output tensor. It works with
    negative values for the 'axis'.

    For example, for a tensor 'input' of shape [4, 4]:

        unsqueeze(input,  0) will produce an output of shape [1, 4, 4],
        unsqueeze(input,  1) will produce an output of shape [4, 1, 4],
        unsqueeze(input, -1) will produce an output of shape [4, 4, 1],
        unsqueeze(input, -2) will produce an output of shape [4, 1, 4],

    Parameters:
        input : Tensor
            The input tensor to expand with a singleton dimension.

        axis : int
            The index of the singleton dimension in the output tensor.

    Returns:
        The tensor produced by the layer.
    '''
    if axis < 0:
        axis = axis + input.ndim() + 1

    return expand_dims(input, axis)


def stack(inputs: Sequence[Tensor], dim: int = 0) -> Tensor:
    '''
    Add an operation to contact input tensors along a new dimension.

    The function creates an operation that creates a new dim for all the
    input tensors and then concatenates them along that new dim.
.

    All the tensors in 'inputs' must have the same shape.

        for ii in range(inputs[0].rank()):
            assert all(inp.shape[ii] == inputs[0].shape[ii] for inp in inputs)

    The shape of the output tensor is defined as:

        output.rank() = inputs[0].rank() + 1

        output.shape[dim] = len(inputs)

        for ii in range(inputs[0].rank()):
            if ii < dim:
                output.shape[ii] = inputs[0].shape[ii]
            else:
                output.shape[ii+1] = inputs[0].shape[ii]

    For example, given a sequence of two 2D tensors [[0, 1], [2, 3]] and
    [[4, 5], [6, 7]] both of shape [2, 2],

        stack(inputs, 0)

    will produce [[[0, 1], [2, 3]], [[4, 5], [6, 7]]] of shape [2, 2, 2] and

        stack(inputs, 1)

    will produce [[[0, 1], [4, 5]], [[2, 3], [6, 7]]] of shape [2, 2, 2].

    Parameters:
        inputs : Sequence[Tensor]
            The sequence of tensors to stack.

        dim : int
            The dimension in which the stack is performed.

    Returns:
        A tensor that contains the input tensors stacked along a new dimension.
    '''
    return concat([unsqueeze(inp, axis=dim) for inp in inputs], dim=dim)


def expand_dims_like(left: Union[Tensor, int, float], right: Tensor) -> Tensor:
    '''
    Add an operation to expand the first tensor to the same rank as the second
    tensor.

    That function takes a first tensor. It also accepts an integer or a float,
    in which case it creates a constant tensor from it. In both cases, the rank
    of that first tensor is compared to the rank of the second tensor. If they
    are of the same rank, the first tensor is returned. Otherwise, the first
    tensor is expanded on the left to match the rank of the second tensor.

    Note that the shapes do not have to match, only the rank is considered in
    that function.

    For example, for a pair of tensors of shapes [3, 4] and [4, 3, 2], the
    first tensor will be expanded to a tensor of rank 3 and shape [1, 3, 4].

    Parameters:
        left : Union[Tensor, int, float]
            The first tensor to expand. When a scalar value is provided as a
            parameter, that function first creates a tensor before expanding it
            (if needed).

        right : Tensor
            The reference tensor to match.

    Returns:
        The tensor produced by the shuffle layer.
    '''
    if isinstance(left, int):
        left = constant(int32_array([left]))
    elif isinstance(left, float):
        if isinstance(right, Tensor) and right.dtype == trt.DataType.HALF:
            left = constant(fp16_array([left]))
        else:
            left = constant(fp32_array([left]))
    left_ndim = left.ndim()
    right_ndim = right.ndim()
    if right_ndim > left_ndim:
        new_ndim = list(range(right_ndim - left_ndim))
        return expand_dims(left, new_ndim)
    return left


# If dim is None, return a 1-D TensorRT-LLM tensor of the size
# If dim is not None, return a 0-D TensorRT-LLM tensor of the dimension size
def shape(input: Tensor, dim: Optional[int] = None) -> Tensor:
    '''
    Add an operation to create a shape tensor.

    The shape tensor can either be the shape of the input tensor when the
    parameter dim is None or a scalar (tensor of rank 0) that corresponds to
    the size of dim-th dimension.

    Parameters:
        input : Tensor
            The input tensor from which we want to extract the shape or the
            size in one dimension.

        dim : Optional[int]
            The dimension from which to extract the size. If it is None, the
            entire shape of the input tensor is returned.

    Returns:
        A tensor that contains the shape of the input tensor (if 'dim' is None)
        or the size in the dimension 'dim' of the input tensor. If 'dim' is
        'None', that tensor has the same rank as the input tensor, otherwise
        its rank is 0.
    '''
    layer = default_trtnet().add_shape(input.trt_tensor)
    res = _create_tensor(layer.get_output(0), layer)

    if dim is None:
        return res

    return gather(res, dim=0, indices=dim).view([])


def gather(input: Tensor, dim: int, indices: Union[Tensor, int]) -> Tensor:
    '''
    Add an operation to gather elements from a tensor.

    That function implements the GatherElements operator from the ONNX
    specification as described in

        https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherElements

    The input and indices arguments must have the same rank >= 1. The operation
    will produce a tensor with the same shape as the indices tensor. The axis
    is the dimension to gather on.

    As shown in the ONNX description, for a 3D tensor, the output is:

        out[i][j][k] = input[indices[i][j][k]][j][k] if axis = 0,
        out[i][j][k] = input[i][indices[i][j][k]][k] if axis = 1,
        out[i][j][k] = input[i][j][indices[i][j][k]] if axis = 2.

    For example,

        gather([[4, 2], [5, 3]], 0, [[1, 0], [0, 1]])

    will produce [[5, 2], [4, 3]].

        gather([[1, 2, 3], [4, 5, 6], 1, [[1], [0]])

    will produce [[2], [4]]. See the ONNX documentation for more examples.

    That operation maps to the TensorRT IGatherLayer.

    Parameters:
        input : Tensor
            The input tensor to gather elements from.

        dim : int
            The dimension to gather on.

        indices : Union[Tensor, int]
            The positions in the 'dim' dimension to gather from.

    Returns:
        The tensor containing the gathered elements. It has the same shape as
        the indices tensor.
    '''
    if isinstance(indices, int):
        indices = constant(int32_array([indices]))

    # The input and indices tensors must have the same rank.
    assert input.rank() == indices.rank()

    layer = default_trtnet().add_gather_v2(input.trt_tensor,
                                           indices.trt_tensor,
                                           mode=trt.GatherMode.ELEMENT)

    if dim < 0:
        dim = input.ndim() + dim
    layer.axis = dim
    return _create_tensor(layer.get_output(0), layer)


def select(input: Tensor, dim: int, index: Union[Tensor, int]) -> Tensor:
    '''
    Add an operation to select a slice of elements from a tensor.

    Given an input tensor, that function creates an operation that selects the
    index-th slice of elements in the dimension 'dim' to create a new tensor.
    The output tensor has a shape in which the input dimension 'dim' is
    removed.

    The 'index' can either be an integer or a 1D tensor containing a single
    element.

    For example, on input=[[4, 2, 5], [2, 1, 2], [4, 7, 1]], which has a shape
    [3, 3],

        select(input, 0, 1)

    will create a tensor of shape [3] that contains the [2, 1, 2].

    Regarding the shape of the output tensor, the dimension 'dim' is removed.
    It means that for a tensor of shape [4, 2, 6, 3],

        select(input, 2, 4)

    will select the 5th slice (index == 4) from the 3rd dimension (dim == 2)
    and return a tensor of shape [4, 2, 3] (i.e. the 3rd dimension is removed).

    That operation maps to the TensorRT IGatherLayer.

    Parameters:
        input : Tensor
            The input tensor to select from.

        dim : int
            The dimension to select from.

        index : Union[Tensor, int]
            The index of the slice in the 'dim' dimension to select.

    Returns:
        The tensor containing the selected slice.
    '''
    if isinstance(index, int):
        index = constant(int32_array([index]))
    assert index.rank() == 1 and index.size(
        0) == 1, f"index should have rank 1, got {index.rank()}"

    new_shape = []
    for i in range(input.rank()):
        if i != dim:
            new_shape.append(shape(input, i))

    layer = default_trtnet().add_gather(input.trt_tensor, index.trt_tensor, dim)
    return _create_tensor(layer.get_output(0), layer).view(concat(new_shape))


def index_select(input: Tensor, dim: int, index: Tensor) -> Tensor:
    '''
    Add an operation to select slices of elements from a tensor.

    Given an input tensor, that function creates an operation that selects the
    slices of elements in the dimension 'dim' at the indices listed in 'index'
    to create a new tensor.  The output tensor has the same rank as the input
    tensor.

    The 'index' is a tensor of rank 1.

    For example, on input=[[4, 2, 5], [2, 1, 2], [4, 7, 1]], which has a shape
    [3, 3],

        index_select(input, 0, [0, 1])

    will create a tensor of shape [2, 3] that contains the [[4, 2, 5], [2, 1, 2]].

    Regarding the shape of the output tensor, the dimension 'dim' has the same
    size as the 'index' tensor. It means that for a input tensor of shape [4, 2, 6, 3],

        index_select(input, 2, [1, 4])

    will select the 2nd and 5th slices (index == 1 or 4) from the 3rd dimension
    (dim == 2) and return a tensor of shape [4, 2, 2, 3] (i.e. the 3rd
    dimension is shrunk to 2).

    Note that this operation can also be used to expand a tensor in the 'dim'
    dimension, for example, on input [[0, 1], [2, 3]],

        index_select(input, 1, [0, 0, 0])

    will produce a tensor of shape [2, 3] containing [[0, 0, 0], [2, 2, 2]].

    That operation maps to the TensorRT IGatherLayer.

    Parameters:
        input : Tensor
            The input tensor to select from.

        dim : int
            The dimension to select from.

        index : Tensor
            The indices of the slices in the 'dim' dimension to select.

    Returns:
        The tensor containing the selected slices.
    '''
    assert index.rank() == 1, f"index should have rank 1, got {index.rank()}"

    new_shape = []
    for i in range(input.rank()):
        if i != dim:
            new_shape.append(shape(input, i))
        else:
            new_shape.append(shape(index, 0))

    layer = default_trtnet().add_gather(input.trt_tensor, index.trt_tensor, dim)
    return _create_tensor(layer.get_output(0), layer).view(concat(new_shape))


def masked_select(input: Tensor, mask: Tensor) -> Tensor:
    '''
    Add an operation to select elements from a tensor according to a boolean
    mask tensor.

    Given an input tensor, that function creates an operation that selects
    elements at the indices indicated by the boolean mask tensor to create
    a new tensor. The output tensor is a 1-D tensor.

    The input tensor must have rank >= 1. The shapes of the input tensor and
    the mask tensor don’t need to match, but they must be broadcastable.

    For example, on input=[[4, 2, 5], [2, 1, 2], [4, 7, 1]], which has a shape
    [3, 3],

        masked_select(input, [[True, False, True], [False, True, False], [True, False, True]])

    will create a tensor of shape [5] that contains the [4, 5, 1, 4, 1].

        masked_select(input, [[True], [False], [True]])

    will create a tensor of shape [6] that contains the [4, 2, 5, 4, 7, 1].

        masked_select(input, [[False, False, True]])

    will create a tensor of shape [3] that contains the [5, 2, 1].

        masked_select(input, [False])

    will create a tensor of shape [0] which is empty.

    That operation is implemented by NonZero, Shuffle and GatherV2 layers
    in TensorRT.

    Parameters:
        input : Tensor
            The input tensor to select from.

        mask : Tensor
            The boolean mask tensor that indicates elements to select.

    Returns:
        The 1-D tensor containing the selected elements.
    '''
    assert input.rank() >= 1, "input should have rank >= 1"
    input, mask = broadcast_helper(input, mask)
    expanded_mask = expand(mask, shape(input))

    non_zero_layer = default_trtnet().add_non_zero(expanded_mask.trt_tensor)

    shuffle_layer = default_trtnet().add_shuffle(non_zero_layer.get_output(0))
    shuffle_layer.second_transpose = (1, 0)

    gather_layer = default_trtnet().add_gather_v2(input.trt_tensor,
                                                  shuffle_layer.get_output(0),
                                                  mode=trt.GatherMode.ND)
    return _create_tensor(gather_layer.get_output(0), gather_layer)


def cumsum(input: Tensor, dim: int) -> Tensor:
    '''
    Add an operation to calculate inclusive cumulative sum of elements of
    a tensor in a given dimension.

    Given an input tensor, that function creates an operation that calculates
    inclusive cumulative sum of elements in the dimension 'dim' to create
    a new tensor. The output tensor has the same shape as the input tensor.

    The input tensor must have rank >= 1. The 'dim' must be valid, and negative
    value is supported.

    For example, on input=[[4, 2, 5], [2, 1, 2], [4, 7, 1]], which has a shape
    [3, 3],

        cumsum(input, 0)

    will produce [[4, 2, 5], [6, 3, 7], [10, 10, 8]].

        cumsum(input, 1)

    will produce [[4, 6, 11], [2, 3, 5], [4, 11, 12]].

    That operation is implemented by TensorRT ILoopLayer.

    Parameters:
        input : Tensor
            The input tensor to calculate the inclusive cumulative sum.

        dim : int
            The dimension to calculate the inclusive cumulative sum. Negative
            value is supported.

    Returns:
        The tensor containing the inclusive cumulative sum of input.
    '''
    assert input.rank() >= 1, "input should have rank >= 1"
    assert dim < input.rank() and dim >= -input.rank(
    ), f"dim should be in [{-input.rank()}, {input.rank()}) when input have rank {input.rank()}"

    dim = dim_resolve_negative(dim, input.ndim())[0]

    slice_shape = []
    for i in range(input.ndim()):
        if i != dim:
            slice_shape.append(shape(input, i))

    zero_tensor = constant(np.array(0, dtype=trt_dtype_to_np(input.dtype)))
    if len(slice_shape) > 0:
        zero_tensor = expand_dims(zero_tensor,
                                  [i for i in range(len(slice_shape))])
        slice_shape = concat(slice_shape)
        zero_tensor = expand(zero_tensor, slice_shape)

    loop_layer = default_trtnet().add_loop()
    trip_limit = shape(input, dim).trt_tensor
    loop_layer.add_trip_limit(trip_limit, trt.TripLimit.COUNT)

    iterator_layer = loop_layer.add_iterator(input.trt_tensor, dim)
    cur_slice = iterator_layer.get_output(0)

    running_sum_layer = loop_layer.add_recurrence(zero_tensor.trt_tensor)
    running_sum = running_sum_layer.get_output(0)

    cur_sum_layer = default_trtnet().add_elementwise(
        cur_slice, running_sum, trt.ElementWiseOperation.SUM)
    cur_sum = cur_sum_layer.get_output(0)
    running_sum_layer.set_input(1, cur_sum)

    loop_output_layer = loop_layer.add_loop_output(cur_sum,
                                                   trt.LoopOutput.CONCATENATE,
                                                   dim)
    loop_output_layer.set_input(1, trip_limit)
    return _create_tensor(loop_output_layer.get_output(0), loop_output_layer)


def concat(inputs: Sequence[Union[Tensor, int]], dim: int = 0) -> Tensor:
    '''
    Add an operation to concatenate tensors.

    The function creates an operation that concatenates the tensors from the
    sequence 'inputs'. The concatenation is done along the dimension 'dim'.

    All the tensors in 'inputs' must have the same shape expect for the
    dimension 'dim'.

        for ii in range(inputs[0].rank()):
            assert (ii == dim) or all(inp.shape[ii] == inputs[0].shape[ii] for inp in inputs)

    The shape of the output tensor is defined as:

        for ii in range(inputs[0].rank()):
            # Same size as all the inputs in dimension ii != dim.
            output.shape[ii] = inputs[0].shape[ii]

            # Sum of the sizes in the different inputs in dimension 'dim'.
            if ii == dim:
                for jj in range(1, len(inputs)):
                    output.shape[ii] += inputs[jj].shape[ii]

    For example, given a sequence of two 2D tensors [[0, 1], [2, 3]] and
    [[4, 5], [6, 7]] both of shape [2, 2],

        concat(inputs, 0)

    will produce [[0, 1], [2, 3], [4, 5], [6, 7]] of shape [4, 2] and

        concat(inputs, 1)

    will produce [[0, 1, 4, 5], [2, 3, 6, 7]] of shape [2, 4].

    Parameters:
        inputs : Sequence[Union[Tensor, int]]
            The sequence of tensors to concatenate. For integers, that function
            creates constant tensors.

        dim : int
            The dimension in which the concatenation is performed.

    Returns:
        A tensor that contains the concatenation of the tensors.
    '''
    tmp = []
    for i in inputs:
        if isinstance(i, int):
            tmp.append(constant(int32_array([i])))
        elif i.rank() == 0:
            tmp.append(i.view([1]))
        else:
            tmp.append(i)

    layer = default_trtnet().add_concatenation([i.trt_tensor for i in tmp])
    layer.axis = dim
    return _create_tensor(layer.get_output(0), layer)


def softmax(input: Tensor, dim: Optional[int] = None) -> Tensor:
    '''
    Add an operation to compute softmax on a tensor.

    That operation computes the softmax on the input tensor in the dimension
    'dim' if specified. Otherwise, it is applied on the last dimension.

    It inserts a ISoftmaxLayer to the TensorRT graph.

    Parameters:
        input : Tensor
            The input tensor on which to apply softmax.

        dim : Optional[int]
            The dimension used to apply softmax.

    Returns:
        The output tensor of the softmax layer.
    '''
    if dim is None:
        dim = input.ndim() - 1
    if dim < 0:
        dim = input.ndim() + dim
    axes = dim_to_trt_axes(dim)

    layer = default_trtnet().add_softmax(input.trt_tensor)
    layer.axes = axes

    return _create_tensor(layer.get_output(0), layer)


def _lookup_plugin(input: Tensor, weight: Tensor, rank: int) -> Tensor:
    '''
    Add an operation to perform lookup in a tensor.

    That operation performs the lookup needed by embedding layers. Given a
    'weight' tensor of shape [rows, cols], it produces a tensor of shape
    [inputs.size(0), cols] where the ith row corresponds to the input[i] row in
    the weight tensor.

    It inserts a IPluginV2Layer.

    Parameters:
        input : Tensor
            The input tensor contains the indices to perform the lookup.

        weight : Tensor
            The table to gather from.

        rank :  int
            The mpi rank.

    Returns:
        The output tensor of the lookup layer.
    '''
    plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'Lookup', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plg_creator is not None

    p_dtype = default_net().plugin_config.lookup_plugin
    pf_type = trt.PluginField(
        "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
        trt.PluginFieldType.INT32)

    rank = trt.PluginField("rank", np.array([int(rank)], np.int32),
                           trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([pf_type, rank])
    lookup_plug = plg_creator.create_plugin("lookup", pfc)
    plug_inputs = [input.trt_tensor, weight.trt_tensor]
    layer = default_trtnet().add_plugin_v2(plug_inputs, lookup_plug)
    _add_plugin_info(layer, plg_creator, "lookup", pfc)
    return _create_tensor(layer.get_output(0), layer)


def embedding(input: Tensor,
              weight: Tensor,
              tp_size=1,
              tp_group=None,
              sharding_dim=0,
              tp_rank=None) -> Tensor:
    '''
    Add an operation to perform embedding lookup.

    That operation performs the embedding lookup. The 'input' tensor contains
    the identifiers of the rows of 'weight' to gather.

    1. Distribute the embedding lookup table over multiple GPU
    When 'tp_size' is greater than 1 and the 'tp_group' is defined, this
    embedding lookup is distributed among multiple GPUs.

    When 'sharding_dim==0', each GPU stores a subset of the rows of the embedding
    table rows(that number of rows per GPU is given by weights.shape[0] and the offset to
    the 1st row stored on the GPU is given by rank * weights.shape[0]). Each
    parallel rank will query all the indices and set 0s for the weights that
    are not stored on the associated GPU. To compute the final result, a
    parallel all-reduce operation is added to the TensorRT graph. That lookup
    can be performed using either the plugin or the operators TensorRT support.

    When'sharding_dim==1', each GPU stores a subset of the embedding table's columns.
    Each rank can obtain a portion of the embedding results.
    Then the embedding is collected using the  all-gather operation.
    Related transposition operations are also used to obtain the final results.

    2. Store embedding lookup table as a whole
    When 'tp_size' is not greater than 1, the embedding lookup table will not
    be divided. In this case, when the default_net().plugin_config.lookup_plugin is set,
    the operation is implemented using a plugin (without the all-reduce operation).
    Otherwise, this operation is implemented using the standard IGatherLayer in TensorRT.

    Parameters:
        input : Tensor
            The input tensor the contains the indices to perform the lookup.

        weight : Tensor
            The table to gather from.

        tp_size : int
            The number of GPUs collaborating to perform that embedding.

        tg_group : Optional[List[int]]
            The group of world ranks participating in the all-reduce when
            tp_size > 1.

        sharding_dim : int
            sharding_dim = 0 means that we shard the embedding table in vocab dim;
            sharding_dim = 1 means that we shard the embedding table in embedding dim.

        tp_rank : int
            The tensor parallelism rank. Used to calculate offset in TP on vocab dim.

    Returns:
        The tensor produced by the embedding lookup layer.
    '''

    # Distribute embedding lookup table across multiple GPU
    if tp_size > 1 and tp_group is not None:
        if sharding_dim == 0:  # TP on vocab_size dimension
            if tp_rank == None:
                raise ValueError(
                    "Rank cannot be none for tensor parallelism on vocab dim")

            if default_net().plugin_config.lookup_plugin:
                x = _lookup_plugin(input, weight, tp_rank)
                x = allreduce(x, tp_group)
            else:
                shape_weight = shape(weight)
                vocab_size = slice(shape_weight, starts=[0], sizes=[1])
                tmp_input = input - vocab_size * tp_rank

                # Identify the valid indices
                is_qualified = op_and(tmp_input >= 0, tmp_input < vocab_size)
                is_qualified_expand = expand_dims(is_qualified,
                                                  [is_qualified.ndim()])

                # Replace the invalid ones to zero
                placeholder_input = where(is_qualified, tmp_input, 0)

                # Get the temporal results
                layer = default_trtnet().add_gather(
                    weight.trt_tensor, placeholder_input.trt_tensor, 0)
                tmp_output = _create_tensor(layer.get_output(0), layer)

                # Set zero for invalid results
                placeholder_tmp = cast(is_qualified_expand, tmp_output.dtype)
                placeholder = placeholder_tmp - placeholder_tmp
                x = where(is_qualified_expand, tmp_output, placeholder)

                # Use all reduce to collect the results
                x = allreduce(x, tp_group)

        elif sharding_dim == 1:  # TP on hidden dimension
            layer = default_trtnet().add_gather(weight.trt_tensor,
                                                input.trt_tensor, 0)
            x = _create_tensor(layer.get_output(0), layer)

            # [dim0, local_dim] -> [dim0 * tp_size, local_dim] --> [dim0, local_dim * tp_size]
            x = allgather(x, tp_group, gather_dim=-1)

        else:
            raise ValueError(
                'Tensor Parallelism only support splitting Embedding lookup along hidden (sharding_dim==1) and vocab (sharding_dim==0) dimensionis'
            )

    # Store embedding lookup table as a whole
    else:
        if default_net().plugin_config.lookup_plugin:
            x = _lookup_plugin(input, weight, rank=0)
        else:
            layer = default_trtnet().add_gather(weight.trt_tensor,
                                                input.trt_tensor, 0)
            x = _create_tensor(layer.get_output(0), layer)
    return x


def constant_to_tensor_(input: Union[Tensor, int, float],
                        dtype: trt.DataType = trt.float32) -> Tensor:
    if isinstance(input, int):
        return constant(int32_array([input]))
    elif isinstance(input, float):
        assert dtype == trt.float32 or dtype == trt.float16 or dtype == trt.bfloat16
        if dtype == trt.float32:
            return constant(fp32_array([input]))
        elif dtype == trt.bfloat16:
            return constant(bf16_array([input]))
        else:
            return constant(fp16_array([input]))

    return input


def broadcast_helper(left: Union[Tensor, int, float],
                     right: Union[Tensor, int, float]) -> Tuple[Tensor, Tensor]:
    '''
    Helper function to perform a broadcast.

    For each input, that function first creates a constant tensor if the input
    is an integer or a float. Then, if needed, it expands the smaller tensor to
    make sure its rank is the same as the larger one.

    Parameters:
        left : Union[Tensor, int, float]
            The first input. If that input is an integer or a float, the
            function creates a constant tensor.

        right : Union[Tensor, int, float]
            The second input. If that input is an integer or a float, the
            function creates a constant tensor.

    Returns:
        A pair of tensors of same rank.
    '''
    if not default_net().strongly_typed:
        left = constant_to_tensor_(left)
        right = constant_to_tensor_(right)
    else:
        left = constant_to_tensor_(
            left, right.dtype if isinstance(right, Tensor) else trt.float32)
        right = constant_to_tensor_(right, left.dtype)

    if left.rank() == right.rank():
        return (left, right)

    if left.rank() < right.rank():
        left = expand_dims_like(left, right)
        return (left, right)

    if left.rank() > right.rank():
        right = expand_dims_like(right, left)
        return (left, right)


def elementwise_binary(left: Union[Tensor, int,
                                   float], right: Union[Tensor, int, float],
                       op: trt.ElementWiseOperation) -> Tensor:
    '''
    Add an elementwise operation with two inputs.

    For each input, that function first creates a constant tensor if the input
    is an integer or a float. Then, if needed, it expands the smaller tensor to
    make sure its rank is the same as the larger one. Then, it performs the
    elementwise operation 'op'.

    The following closures are defined in functional.*:

        add     for op=trt.ElementWiseOperation.SUM
        sub     for op=trt.ElementWiseOperation.SUB
        mul     for op=trt.ElementWiseOperation.PROD
        div     for op=trt.ElementWiseOperation.DIV
        gt      for op=trt.ElementWiseOperation.GREATER
        lt      for op=trt.ElementWiseOperation.LESS
        op_and  for op=trt.ElementWiseOperation.AND
        op_or   for op=trt.ElementWiseOperation.OR
        eq      for op=trt.ElementWiseOperation.EQUAL
        minimum for op=trt.ElementWiseOperation.MIN
        maximum for op=trt.ElementWiseOperation.MAX
        pow     for op=trt.ElementWiseOperation.POW

    It is implemented using the IElementWiseLayer from TensorRT.

    Parameters:
        left : Union[Tensor, int, float]
            The first input. If that input is an integer or a float, the
            function creates a constant tensor.

        right : Union[Tensor, int, float]
            The second input. If that input is an integer or a float, the
            function creates a constant tensor.

        op : trt.ElementWiseOperation
            The binary operation to perform.

    Returns:
        The tensor produced by this elementwise operation.
    '''
    left, right = broadcast_helper(left, right)
    layer = default_trtnet().add_elementwise(left.trt_tensor, right.trt_tensor,
                                             op)
    return _create_tensor(layer.get_output(0), layer)


add = partial(elementwise_binary, op=trt.ElementWiseOperation.SUM)
sub = partial(elementwise_binary, op=trt.ElementWiseOperation.SUB)
mul = partial(elementwise_binary, op=trt.ElementWiseOperation.PROD)
div = partial(elementwise_binary, op=trt.ElementWiseOperation.DIV)
gt = partial(elementwise_binary, op=trt.ElementWiseOperation.GREATER)
lt = partial(elementwise_binary, op=trt.ElementWiseOperation.LESS)
op_and = partial(elementwise_binary, op=trt.ElementWiseOperation.AND)
op_or = partial(elementwise_binary, op=trt.ElementWiseOperation.OR)
eq = partial(elementwise_binary, op=trt.ElementWiseOperation.EQUAL)
minimum = partial(elementwise_binary, op=trt.ElementWiseOperation.MIN)
maximum = partial(elementwise_binary, op=trt.ElementWiseOperation.MAX)
pow = partial(elementwise_binary, op=trt.ElementWiseOperation.POW)


def where(condition: Union[Tensor, int, float], left: Union[Tensor, int, float],
          right: Union[Tensor, int, float]) -> Tensor:
    '''
    Add a where (aka select or if-then-else) operation.

    Assuming the three input parameters have the same shape, that function creates
    the operation to compute a tensor of the same shape such that:

        for ii in range(mul(condition.shape)):
            output[ii] = left[ii] if condition[ii] else right[ii]

    For each input, that function first creates a constant tensor if the input
    is an integer or a float. Then, if needed, it expands the smaller tensor to
    make sure its rank is the same as the larger one. Then, it performs the
    selection.

    It is implemented using the ISelectLayer from TensorRT.

    Parameters:
        left : Union[Tensor, int, float]
            The condition. If that input is an integer or a float, the function
            creates a constant tensor.

        left : Union[Tensor, int, float]
            The first input. If that input is an integer or a float, the
            function creates a constant tensor.

        right : Union[Tensor, int, float]
            The second input. If that input is an integer or a float, the
            function creates a constant tensor.

        op : trt.ElementWiseOperation
            The binary operation to perform.

    Returns:
        The tensor produced by this select operation.
    '''
    # Convert to tensors.
    condition = constant_to_tensor_(condition)
    left = constant_to_tensor_(left)
    right = constant_to_tensor_(right)

    # Find the tensor with the largest rank of the three.
    largest = condition
    if largest.rank() < left.rank():
        largest = left
    if largest.rank() < right.rank():
        largest = right

    # Expand the tensors to match the largest one.
    if condition is not largest:
        condition = expand_dims_like(condition, largest)
    if left is not largest:
        left = expand_dims_like(left, largest)
    if right is not largest:
        right = expand_dims_like(right, largest)

    # Insert the operation.
    layer = default_trtnet().add_select(condition.trt_tensor, left.trt_tensor,
                                        right.trt_tensor)
    return _create_tensor(layer.get_output(0), layer)


def unary(input: Tensor, op: trt.UnaryOperation) -> Tensor:
    '''
    Add an elementwise operation on a single input.

    The following closures are defined in functional.*:

        round   for op=trt.UnaryOperation.ROUND
        sqrt    for op=trt.UnaryOperation.SQRT
        exp     for op=trt.UnaryOperation.EXP
        sin     for op=trt.UnaryOperation.SIN
        cos     for op=trt.UnaryOperation.COS
        abs     for op=trt.UnaryOperation.ABS

    It is implemented using the IUnaryLayer from TensorRT.

    Parameters:
        input : Tensor
            The input tensor.

        op : trt.UnaryOperation
            The unary operation to perform.

    Returns:
        The tensor produced by this elementwise operation.
    '''
    layer = default_trtnet().add_unary(input.trt_tensor, op)
    return _create_tensor(layer.get_output(0), layer)


round = partial(unary, op=trt.UnaryOperation.ROUND)
sqrt = partial(unary, op=trt.UnaryOperation.SQRT)
exp = partial(unary, op=trt.UnaryOperation.EXP)
sin = partial(unary, op=trt.UnaryOperation.SIN)
cos = partial(unary, op=trt.UnaryOperation.COS)
abs = partial(unary, op=trt.UnaryOperation.ABS)


def mean(input: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    '''
    Add an operation to compute the mean along a dimension.

    Computes the mean along the dimension 'dim' of the input tensor.

    It is implemented using the IReduceLayer from TensorRT.

    Parameters:
        input : Tensor
            The input tensor.

        dim : int
            The dimension along which the mean is computed.

        keepdim : bool
            Is the dimension kept in the reduced tensor? When True the
            dimension is kept, it is removed from the shape otherwise.

    Returns:
        The tensor produced by this reduction operation.
    '''
    dim = dim_resolve_negative(dim, input.ndim())
    axes = dim_to_trt_axes(dim)

    layer = default_trtnet().add_reduce(input.trt_tensor,
                                        trt.ReduceOperation.AVG,
                                        axes,
                                        keep_dims=keepdim)
    return _create_tensor(layer.get_output(0), layer)


def max(input: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    '''
    Add an operation to compute the max along a dimension.

    Computes the max along the dimension 'dim' of the input tensor.

    It is implemented using the IReduceLayer from TensorRT.

    Parameters:
        input : Tensor
            The input tensor.

        dim : int
            The dimension along which the mean is computed.

        keepdim : bool
            Is the dimension kept in the reduced tensor? When True the
            dimension is kept, it is removed from the shape otherwise.

    Returns:
        The tensor produced by this reduction operation.
    '''
    dim = dim_resolve_negative(dim, input.ndim())
    axes = dim_to_trt_axes(dim)

    layer = default_trtnet().add_reduce(input.trt_tensor,
                                        trt.ReduceOperation.MAX,
                                        axes,
                                        keep_dims=keepdim)
    return _create_tensor(layer.get_output(0), layer)


def identity(input: Tensor) -> Tensor:
    '''
    Add an identity operation.

    TODO: Document why it can be done using a plugin!!!

    Parameters:
        input : Tensor
            The input tensor.

    Returns:
        The tensor produced by this identity operation.
    '''
    if not default_net().plugin_config.identity_plugin:
        layer = default_trtnet().add_identity(input.trt_tensor)
    else:
        plg_creator = trt.get_plugin_registry().get_plugin_creator(
            'Identity', '1', TRT_LLM_PLUGIN_NAMESPACE)
        assert plg_creator is not None
        pfc = trt.PluginFieldCollection()
        id_plug = plg_creator.create_plugin("identity", pfc)
        plug_inputs = [input.trt_tensor]
        layer = default_trtnet().add_plugin_v2(plug_inputs, id_plug)
        _add_plugin_info(layer, plg_creator, "identity", pfc)
    return _create_tensor(layer.get_output(0), layer)


def argmax(input: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    '''
    Add an argmax operation.

    As explained in the ONNX documentation,

        https://github.com/onnx/onnx/blob/main/docs/Operators.md#argmax

    that function creates a layer computing the indices of the max elements of
    the input tensor's element along the provided dim. The resulting tensor
    has the same rank as the input if keepdims is True. If keepdims is False,
    then the resulting tensor has the reduced dimension pruned.

    Parameters:
        input : Tensor
            The input tensor.

        dim : int
            The dimension in which to compute the argmax indices.

        keepdim : bool
            Do we keep the dimension along which the reduction is performed?
            Yes, if set to True, no otherwise.

    Returns:
        The tensor produced by this argmax operation.
    '''
    dim = dim_resolve_negative(dim, input.ndim())
    axes = dim_to_trt_axes(dim)

    layer = default_trtnet().add_topk(input.trt_tensor, trt.TopKOperation.MAX,
                                      1, axes)
    output = layer.get_output(1)

    if keepdim:
        return _create_tensor(output, layer)

    a = list(range(len(input.ndim())))
    a.pop(dim)
    indices = constant(int32_array([a]))
    output_shape = shape(output)
    new_shape = gather(output_shape, 0, indices)
    layer = view(output, new_shape)
    return _create_tensor(layer.get_output(0), layer)


def gelu(x: Tensor) -> Tensor:
    '''
    Add a GELU operation.

    Parameters:
        input : Tensor
            The input tensor on which the activation function is applied.

    Returns:
        The tensor produced by the activation layer.
    '''
    if not default_net().strongly_typed:
        return 0.5 * x * (
            tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * pow(x, 3.0))) + 1.0)

    array_fn = {
        trt.float32: fp32_array,
        trt.float16: fp16_array,
        trt.bfloat16: bf16_array,
    }[x.dtype]

    v1 = constant(array_fn([0.5]))
    v2 = constant(array_fn([math.sqrt(2.0 / math.pi)]))
    v3 = constant(array_fn([0.044715]))
    v4 = constant(array_fn([3.0]))
    v5 = constant(array_fn([1.0]))
    return v1 * x * (tanh(v2 * (x + v3 * pow(x, v4))) + v5)


def geglu(x: Tensor) -> Tensor:
    '''
    Add a Gated-GELU operation.

    That function takes a tensor, splits it into two halves along the last
    dimension, applies GELU to the second half and multiply the results. The
    behaviour is undefined if the last dimension is not even.

    Parameters:
        input : Tensor
            The input tensor on which the activation function is applied.

    Returns:
        The tensor produced by the activation layer.
    '''
    a, b = chunk(x, 2, dim=-1)
    return a * gelu(b)


def group_norm(input: Tensor,
               num_groups: int,
               weight: Optional[Tensor] = None,
               bias: Optional[Tensor] = None,
               eps: float = 1e-05):

    ##
    ## TODO: Document that function!
    ##

    assert not input.is_dynamic(1)
    num_channels = input.size()[1]

    ndim = input.ndim()
    old_shape = shape(input)
    new_shape = concat([
        input.size(0),
        num_groups,
        num_channels // num_groups,
    ] + [input.size(i) for i in range(2, ndim)])
    x = input.view(new_shape)

    reduce_dim = tuple(range(2, ndim + 1))
    ux = x.mean(reduce_dim, keepdim=True)
    numerator = x - ux
    varx = numerator * numerator
    varx = varx.mean(reduce_dim, keepdim=True)

    denom = varx + eps
    denom = denom.sqrt()
    y = numerator / denom
    y = y.view(old_shape)

    new_shape = concat([num_channels] + [1 for _ in range(2, ndim)])
    if weight is not None:
        y = y * weight.view(new_shape)
    if bias is not None:
        y = y + bias.view(new_shape)

    return y


def softplus(input: Tensor, beta: float, threshold: float) -> Tensor:
    '''
    Add the softplus activation base on PyTorch definition.

    See https://pytorch.org/docs/stable/generated/torch.nn.functional.softplus.html for a
    description of that function.

    Parameters:
        input : Tensor
            Input TensorRT-LLM Tensor.
        beta : float
            The parameter for softplus computation.
        threshold : float
            The threshold for reverting to the linear function when input * beta > threshold

    Returns:
        The output tensor created by that layer.
    '''
    sf_layer = default_trtnet().add_activation(input.trt_tensor,
                                               trt.ActivationType.SOFTPLUS)
    sf_layer.alpha = 1 / beta
    sf_layer.beta = beta

    prod_tensor = input * beta
    result = prod_tensor > threshold

    return where(result, input, _create_tensor(sf_layer.get_output(0),
                                               sf_layer))


def outer(input: Tensor, vec2: Tensor) -> Tensor:
    '''
    Add an operation to compute the outer product between two tensors.

    That operation creates an Einsum node.

    Parameters:
        input : Tensor
            The first input tensor.

        vec2 : Tensor
            The second input tensor.

    Returns:
        The output tensor produced by this layer.
    '''
    return einsum('i,j->ij', [input, vec2])


def avg_pool2d(input: Tensor,
               kernel_size: Tuple[int],
               stride: Optional[Tuple[int]] = None,
               padding: Optional[Tuple[int]] = (0, 0),
               ceil_mode: bool = False,
               count_include_pad: bool = True) -> Tensor:

    ##
    ## TODO: Document that function!
    ##

    assert not input.is_dynamic()
    ndim = input.ndim()
    if ndim == 3:
        input = expand_dims(input, 0)

    layer = default_trtnet().add_pooling_nd(input.trt_tensor,
                                            trt.PoolingType.AVERAGE,
                                            kernel_size)
    if stride is None:
        stride = kernel_size
    layer.stride_nd = stride

    output = _create_tensor(layer.get_output(0), layer)

    if ndim == 3:
        return output.view(
            concat([output.size(1),
                    output.size(2),
                    output.size(3)]))

    return output


def conv1d(input: Tensor,
           weight: Tensor,
           bias: Optional[Tensor] = None,
           stride: int = 1,
           padding: int = 0,
           dilation: int = 1,
           groups: int = 1) -> Tensor:

    noutput = weight.size()[0]
    kernel_size = weight.size()[-2]
    is_weight_constant = (weight.producer is not None
                          and weight.producer.type == trt.LayerType.CONSTANT)
    weight = weight.producer.weights if is_weight_constant else trt.Weights()

    if bias is not None:
        is_bias_constant = (bias.producer is not None
                            and bias.producer.type == trt.LayerType.CONSTANT)
        bias = bias.producer.weights if is_bias_constant else trt.Weights()

    input_shuffle_layer = default_trtnet().add_shuffle(input.trt_tensor)
    input_shuffle_layer.reshape_dims = trt.Dims([*(input.size()), 1])
    input_shuffled = _create_tensor(input_shuffle_layer.get_output(0),
                                    input_shuffle_layer)

    kernel_size = trt.Dims([kernel_size, 1])

    layer = default_trtnet().add_convolution_nd(input_shuffled.trt_tensor,
                                                noutput, kernel_size, weight,
                                                bias)
    layer.stride_nd = (stride, 2)
    layer.padding_nd = (padding, 0)
    layer.dilation_nd = (dilation, 2)
    layer.num_groups = groups

    if not is_weight_constant:
        layer.set_input(1, weight.trt_tensor)
    if bias is not None and not is_bias_constant:
        layer.set_input(2, bias.trt_tensor)

    output_2d = _create_tensor(layer.get_output(0), layer)
    output_2d_shuffle_layer = default_trtnet().add_shuffle(output_2d.trt_tensor)
    output_2d_shuffle_layer.reshape_dims = trt.Dims(
        [output_2d.size()[0],
         output_2d.size()[1],
         output_2d.size()[2]])
    output_1d = _create_tensor(output_2d_shuffle_layer.get_output(0),
                               output_2d_shuffle_layer)

    return output_1d


def conv2d(input: Tensor,
           weight: Tensor,
           bias: Optional[Tensor] = None,
           stride: Tuple[int, int] = (1, 1),
           padding: Tuple[int, int] = (0, 0),
           dilation: Tuple[int, int] = (1, 1),
           groups: int = 1) -> Tensor:
    ##
    ## TODO: Document that function!
    ##

    ndim = input.ndim()
    if ndim == 3:
        input = expand_dims(input, 0)

    noutput = weight.size()[0]
    kernel_size = (weight.size()[-2], weight.size()[-1])

    is_weight_constant = (weight.producer is not None
                          and weight.producer.type == trt.LayerType.CONSTANT)
    weight = weight.producer.weights if is_weight_constant else trt.Weights()

    if bias is not None:
        is_bias_constant = (bias.producer is not None
                            and bias.producer.type == trt.LayerType.CONSTANT)
        bias = bias.producer.weights if is_bias_constant else trt.Weights()

    layer = default_trtnet().add_convolution_nd(input.trt_tensor, noutput,
                                                kernel_size, weight, bias)
    layer.stride_nd = stride
    layer.padding_nd = padding
    layer.dilation_nd = dilation
    layer.num_groups = groups
    layer.dilation_nd = dilation

    if not is_weight_constant:
        layer.set_input(1, weight.trt_tensor)
    if bias is not None and not is_bias_constant:
        layer.set_input(2, bias.trt_tensor)

    output = _create_tensor(layer.get_output(0), layer)

    if ndim == 3:
        return output.view(
            concat([output.size(1),
                    output.size(2),
                    output.size(3)]))

    return output


def conv_transpose2d(input: Tensor,
                     weight: Tensor,
                     bias: Optional[Tensor] = None,
                     stride: Tuple[int, int] = (1, 1),
                     padding: Tuple[int, int] = (0, 0),
                     output_padding: Tuple[int, int] = (0, 0),
                     dilation: Tuple[int, int] = (1, 1),
                     groups: int = 1) -> Tensor:
    ##
    ## TODO: Document that function!
    ##

    assert not input.is_dynamic()

    ndim = input.ndim()
    if ndim == 3:
        input = expand_dims(input, 0)

    noutput = weight.size()[1]
    kernel_size = (weight.size()[-2], weight.size()[-1])

    is_weight_constant = (weight.producer is not None
                          and weight.producer.type == trt.LayerType.CONSTANT)
    weight = weight.producer.weights if is_weight_constant else trt.Weights()

    if bias is not None:
        is_bias_constant = (bias.producer is not None
                            and bias.producer.type == trt.LayerType.CONSTANT)
        bias = bias.producer.weights if is_bias_constant else trt.Weights()

    layer = default_trtnet().add_deconvolution_nd(input.trt_tensor, noutput,
                                                  kernel_size, weight, bias)
    layer.stride_nd = stride
    layer.padding_nd = padding
    layer.num_groups = groups

    if not is_weight_constant:
        layer.set_input(1, weight.trt_tensor)
    if bias is not None and not is_bias_constant:
        layer.set_input(2, bias.trt_tensor)

    output = _create_tensor(layer.get_output(0), layer)

    if ndim == 3:
        return output.view(
            concat([output.size(1),
                    output.size(2),
                    output.size(3)]))

    return output


def split(tensor: Tensor,
          split_size_or_sections: Union[int, Sequence[int]],
          dim: int = 0) -> Sequence[Tensor]:
    '''
    Add an operation that splits a tensor into sub-tensors.

    This operation creates a list of tensors that are obtained from the input
    tensor by slicing it along the dimension 'dim'. If 'split_size_or_sections'
    is an integer, the tensor is split into 'input.shape[dim] /
    split_size_or_sections' slices. If 'split_size_or_sections' is a list of
    sizes, the tensor is split into 'len(split_size_or_sections)' slices and
    the size of the ith slice is given by 'split_size_or_sections[i]'.

    There are several constraints with the current implementation:

        - The input tensor must be static (no dynamic dimension),
        - If 'split_size_or_sections' is an integer, the number of elements in
          the 'dim' dimension of the input must be a multiple of
          'split_size_or_sections': 'input.shape[dim] % split_size_or_sections == 0'.
        - If 'split_size_or_sections' is a sequence, the sum of the elements in
          'split_size_or_sections' must be equal to the size in the dimension
          'dim': 'input.shape[dim] == sum(ii for ii in split_size_or_sections)'.

    That operation is implemented using a 'slice' operation for each output
    slice.

    Parameters:
        tensor : Tensor
            The input tensor to slice.

        split_size_or_sections : Union[int, Sequence[int]]
            If it is an integer, it encodes the size of each slice. Otherwise,
            if it is a sequence, it is the size of each slice.

        dim : int
            The dimension of the tensor to slice.

    Returns:
        The list of tensors produced by the different operations.
    '''
    assert not tensor.is_dynamic(dim)

    ndim = tensor.ndim()
    if dim < 0:
        dim += ndim
    dim_value = tensor.size()[dim]
    starts = [constant(int32_array([0])) for _ in range(ndim)]
    sizes = [shape(tensor, i) for i in range(ndim)]

    if isinstance(split_size_or_sections, int):
        # TODO: support non-divisible cases
        assert dim_value % split_size_or_sections == 0
        num_sections = dim_value // split_size_or_sections
        sizes[dim] = constant(int32_array([split_size_or_sections]))

        outputs = []
        for i in range(num_sections):
            starts[dim] = constant(int32_array([split_size_or_sections * i]))
            outputs.append(slice(tensor, concat(starts), concat(sizes)))
        return outputs
    else:
        total_size = 0
        for i in split_size_or_sections:
            total_size += i
        assert dim_value == total_size
        num_sections = len(split_size_or_sections)

        outputs = []
        for i in range(num_sections):
            if i > 0:
                starts[dim] = starts[dim] + sizes[dim]
            sizes[dim] = constant(int32_array([split_size_or_sections[i]]))
            outputs.append(slice(tensor, concat(starts), concat(sizes)))
        return outputs


def chunk(tensor: Tensor, chunks: int, dim: int = 0) -> Tensor:
    '''
    Add an operation that splits a tensor into sub-tensors.

    This operation creates a list of tensors that are obtained from the input
    tensor by chunking it along the dimension 'dim'. It produces 'chunks'
    sub-tensors.

    That operation is only defined for static tensors (no dynamic dimension)
    and the size of the tensor in the dimension 'dim' must be a multiple of
    'chunks': 'input.shape[dim] % chunks == 0'.

    It maps to 'split' with 'split_size = input.shape[dim] / chunks'.

    Parameters:
        tensor : Tensor
            The input tensor to slice.

        chunks : int
            The number of slices to split the input tensor into.

        dim : int
            The dimension of the tensor to slice.

    Returns:
        The list of tensors produced by the different operations.
    '''
    assert not tensor.is_dynamic(dim)

    ndim = tensor.ndim()
    if dim < 0:
        dim += ndim
    dim_value = tensor.size()[dim]
    assert dim_value % chunks == 0

    return split(tensor, dim_value // chunks, dim)


class AllReduceStrategy(IntEnum):
    """
    Warning: actual definition is in cpp/tensorrt_llm/kernels/customAllReduceKernels.h
             they must be kept in sync
    """
    RING = 0
    ONESHOT = 1
    TWOSHOT = 2
    AUTO = 3


def allreduce(tensor: Tensor,
              group: List[int],
              strategy: Optional[AllReduceStrategy] = None) -> Tensor:
    '''
    Add an operation that performs a collective all-reduce.

    Let's define 'world_size' as the length of the 'group' list. That functions
    creates a layer to compute the sum of 'world_size' tensors distributed
    amongst the 'world_size' participating ranks (one GPU per rank).

    The list 'group' contains the identifiers of the ranks participating into
    the collective operation.

    The tensors in the different ranks must be 1D tensors (or views) and the output
    tensor will have that same shape. The output tensor will be replicated on
    the 'world_size' ranks.

    That operation is implemented using a plugin that wraps the NCCL all-reduce
    collective operation. See
    https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allreduce
    for details.

    Parameters:
        tensor : Tensor
            The input tensor.

        group : List[int]
            The ranks participating into the all-reduce operation.

        strategy: AllReduceStrategy
            RING delegates all-reduce to NCCL while ONESHOT and TWOSHOT are custom latency-optimal algorithms.
            AUTO chooses amongst the three based on a message-size heuristic.

    Returns:
        The tensor produced by that layer.
    '''

    allreduce_plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'AllReduce', '1', TRT_LLM_PLUGIN_NAMESPACE)

    if strategy is None:
        if default_net().plugin_config.use_custom_all_reduce:
            strategy = AllReduceStrategy.AUTO
        else:
            strategy = AllReduceStrategy.RING

    counter = 0
    workspace = None

    if strategy != AllReduceStrategy.RING:
        counter = current_all_reduce_helper().gen_id()
        workspace = current_all_reduce_helper().workspace

    assert allreduce_plg_creator is not None

    group = trt.PluginField("group", np.array(group, dtype=np.int32),
                            trt.PluginFieldType.INT32)

    p_dtype = default_net().plugin_config.nccl_plugin
    pf_dtype = trt.PluginField(
        "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
        trt.PluginFieldType.INT32)
    pfc = [group, pf_dtype]
    p_strategy = trt.PluginField("strategy", np.array([int(strategy)], np.int8),
                                 trt.PluginFieldType.INT8)
    pfc.append(p_strategy)
    p_counter = trt.PluginField("counter", np.array([counter], np.int32),
                                trt.PluginFieldType.INT32)
    pfc.append(p_counter)

    pfc = trt.PluginFieldCollection(pfc)
    ar_plug = allreduce_plg_creator.create_plugin("allreduce", pfc)
    plug_inputs = [tensor.cast(p_dtype).trt_tensor]
    if strategy != AllReduceStrategy.RING:
        plug_inputs.append(workspace.trt_tensor)

    layer = default_trtnet().add_plugin_v2(plug_inputs, ar_plug)
    _add_plugin_info(layer, allreduce_plg_creator, "allreduce", pfc)
    return _create_tensor(layer.get_output(0), layer).cast(tensor.dtype)


def allgather(tensor: Tensor, group: List[int], gather_dim: int = 0) -> Tensor:
    '''
    Add an operation that performs a collective all-gather.

    Let's define 'group_size' as the length of the 'group' list. That functions
    creates a layer to gather 'group_size' tensors distributed
    amongst the 'group_size' participating ranks (one GPU per rank).

    The list 'group' contains the identifiers of the ranks participating into
    the collective operation.

    Note that 'group' here can be either TP group or PP group, because allgather communication is not limited to a specific split pattern. Therefore 'group_size' does not need to equal MPI 'world_size'.

    The tensors in the different ranks must be 1D tensors (or views) and the
    output tensor will have that same shape.

    Given the 'section_size = input.shape[0] / group_size', each rank
    contributes a section of its input tensor that correspond to
    'rank*section_size:(rank+1)*section_size'.

    That operation is implemented using a plugin that wraps the NCCL all-gather
    collective operation. See
    https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allgather
    for details.

    Parameters:
        tensor : Tensor
            The input tensor.

        group : List[int]
            The ranks participating into the all-gather operation.

        gather_dim: int = 0
            Gather along given dimension. By default 0, i.e. treated as 1D tensor.

    Returns:
        The tensor produced by that layer.
    '''
    allgather_plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'AllGather', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert allgather_plg_creator is not None

    group_size = len(group)
    group = trt.PluginField("group", np.array(group, dtype=np.int32),
                            trt.PluginFieldType.INT32)

    p_dtype = default_net().plugin_config.nccl_plugin
    pf_type = trt.PluginField(
        "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
        trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([group, pf_type])
    allgather = allgather_plg_creator.create_plugin("allgather", pfc)
    plug_inputs = [tensor.cast(p_dtype).trt_tensor]

    layer = default_trtnet().add_plugin_v2(plug_inputs, allgather)
    _add_plugin_info(layer, allgather_plg_creator, "allgather", pfc)

    x = _create_tensor(layer.get_output(0), layer).cast(tensor.dtype)

    # gather along a given dimension other than dim0
    if gather_dim != 0:
        # also support -1 type of dim representation
        if gather_dim < 0:
            gather_dim = x.ndim() + gather_dim

        # plugin above gathers as 1D flattened tensor
        # 1. [dim0, ...dimi, ...dimN] -> [group_size * dim0, ...dimi, ...dimN]

        # now we need to gather-by-dim via split-concat
        # 2. [group_size * dim0, ...dimi, ...dimN] -> [dim0, ...group_size * dimi, ...dimN]
        # 2.1 split
        split_size = shape(x, dim=0) / group_size
        ndim = x.ndim()
        starts = [constant(int32_array([0])) for _ in range(ndim)]
        sizes = [shape(x, dim=d) for d in range(ndim)]
        sizes[0] = split_size
        sections = []
        for i in range(group_size):
            starts[0] = split_size * i
            sections.append(slice(x, concat(starts), concat(sizes)))
        # 2.2 concat
        x = concat(sections, dim=gather_dim)

    return x


def send(tensor: Tensor, tgt: int) -> Tensor:
    '''
    Add an operation that performs a send from a rank to another.

    The send operation sends a tensor from one rank to another. If a rank 'i'
    sends a tensor to a rank 'j', the rank 'j' must have a corresponding 'recv'
    operation from rank 'i'. See 'recv'.

    That operation is implemented using a plugin that wraps the NCCL send
    point-to-point operation. See
    https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#ncclsend
    for details.

    Parameters:
        tensor : Tensor
            The input tensor.

        tgt : int
            The rank that receives the tensor.

    Returns:
        The tensor produced by that layer.
    '''
    send_plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'Send', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert send_plg_creator is not None

    tgt = trt.PluginField("tgt_rank", np.array(tgt, dtype=np.int32),
                          trt.PluginFieldType.INT32)

    p_dtype = default_net().plugin_config.nccl_plugin
    pf_type = trt.PluginField(
        "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
        trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([tgt, pf_type])
    send_plug = send_plg_creator.create_plugin("send", pfc)
    plug_inputs = [tensor.cast(p_dtype).trt_tensor]

    layer = default_trtnet().add_plugin_v2(plug_inputs, send_plug)
    _add_plugin_info(layer, send_plg_creator, "send", pfc)
    return _create_tensor(layer.get_output(0), layer).cast(tensor.dtype)


def recv(tensor: Tensor, src: int) -> Tensor:
    '''
    Add an operation that performs a recv to a rank from another.

    The recv operation receives a tensor from on a rank from another. If a rank 'i'
    receives a tensor from a rank 'j', the rank 'j' must have a corresponding 'send'
    operation to rank 'j'. See 'send'.

    That operation is implemented using a plugin that wraps the NCCL recv
    point-to-point operation. See
    https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#ncclrecv
    for details.

    Parameters:
        tensor : Tensor
            The input tensor.

        src : int
            The rank that sends the tensor to.

    Returns:
        The tensor produced by that layer.
    '''
    recv_plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'Recv', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert recv_plg_creator is not None

    src = trt.PluginField("src_rank", np.array(src, dtype=np.int32),
                          trt.PluginFieldType.INT32)
    p_dtype = default_net().plugin_config.nccl_plugin
    pf_type = trt.PluginField(
        "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
        trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([src, pf_type])
    recv_plug = recv_plg_creator.create_plugin("recv", pfc)
    plug_inputs = [tensor.cast(p_dtype).trt_tensor]

    layer = default_trtnet().add_plugin_v2(plug_inputs, recv_plug)
    _add_plugin_info(layer, recv_plg_creator, "recv", pfc)
    return _create_tensor(layer.get_output(0), layer).cast(tensor.dtype)


def bert_attention(tensor: Tensor,
                   input_lengths: Tensor,
                   num_heads: int,
                   head_size: int,
                   q_scaling: float,
                   relative_attention: bool = False,
                   relative_attention_bias: Tensor = None,
                   max_distance: int = 0,
                   max_input_length: Tensor = None) -> Tuple[Tensor]:
    '''
    Add an operation that performs the multi-head attention in BERT.

    The multihead-attention (MHA) is the sequence of a batched matmul, a
    softmax and a batched matmul as described in
    https://arxiv.org/abs/1706.03762. That function adds an operation that
    performs those computations using a single GPU kernel.

    The input tensor contains the Q, K and V elements. It is a 2D tensor and
    its shape is '[sum_of_tokens, 3*hidden_dim]' where the 'sum_of_tokens' is
    the sum of the sequence lengths in the batch.

    In MHA, the output of the Q*K^T product is scaled by a constant value that
    is computed as:

        1.f / (q_scaling * sqrt(head_size)).

    That 'q_scaling' constant is the last argument of that function.

    That layer is implemented using a plugin (see bertAttentionPlugin).

    Parameters:
        tensor : Tensor
            The QKV input tensor.

        input_lengths : Tensor
            The length of each sequence. It is a 1D tensor of size 'batch_size'.

        num_heads : int
            The number of heads.

        head_size : int
            The size of each head.

        q_scaling : float
            The factor to compute the scaling factor to scale the output of the
            'Q*K^T' product.

        relative_attention: bool = False
            If enable relative attention.

        relative_attention_bias: Tensor = None
            The relative attention bias [num_heads, max_seq_len, max_seq_len], or The relative attention embedding table for implicit mode, [num_heads, num_buckets].

        max_distance: int = 0
            The maximum distance of relative position in attention, for implicit mode.
            Default value is 0, meaning to use the regular mode of relative attention bias.
            Implicit mode is only enabled when passing in non-zero positive max_distance value.
            See relative attention bias in docs/gpt_attention.md

        max_input_length: Tensor = None
            The maximum input sequence length represented by Tensor shape. Requires for remove_input_padding to pre-define plugin workspace size.

    Returns:
        The tensor produced by that layer.
    '''
    attn_plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'BertAttention', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert attn_plg_creator is not None

    nheads = trt.PluginField("num_heads", np.array(num_heads, dtype=np.int32),
                             trt.PluginFieldType.INT32)
    head_size = trt.PluginField("head_size", np.array(head_size,
                                                      dtype=np.int32),
                                trt.PluginFieldType.INT32)
    q_scaling = trt.PluginField("q_scaling",
                                np.array(q_scaling, dtype=np.float32),
                                trt.PluginFieldType.FLOAT32)
    enable_qk_half_accum = trt.PluginField(
        "enable_qk_half_accum",
        np.array(np.int8(
            default_net().plugin_config.attention_qk_half_accumulation),
                 dtype=np.int8), trt.PluginFieldType.INT8)
    context_fmha_type = trt.PluginField(
        "context_fmha_type",
        np.array(np.int8(default_net().plugin_config.context_fmha_type),
                 dtype=np.int8), trt.PluginFieldType.INT8)
    p_dtype = default_net().plugin_config.bert_attention_plugin
    pf_type = trt.PluginField(
        "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
        trt.PluginFieldType.INT32)
    do_relative_attention = trt.PluginField(
        "do_relative_attention",
        np.array(np.int8(relative_attention), dtype=np.int8),
        trt.PluginFieldType.INT8)
    max_distance = trt.PluginField("max_distance",
                                   np.array(max_distance, dtype=np.int32),
                                   trt.PluginFieldType.INT32)
    remove_padding = trt.PluginField(
        "remove_padding",
        np.array(np.int8(default_net().plugin_config.remove_input_padding),
                 dtype=np.int8), trt.PluginFieldType.INT8)
    pfc = trt.PluginFieldCollection([
        nheads, head_size, q_scaling, enable_qk_half_accum, context_fmha_type,
        pf_type, do_relative_attention, max_distance, remove_padding
    ])

    attn_plug = attn_plg_creator.create_plugin("padding_attn", pfc)
    plug_inputs = [tensor, input_lengths]
    if max_input_length is not None:
        # for remove padding mode
        plug_inputs += [max_input_length]
    if relative_attention_bias is not None:
        # for relative attention mode
        plug_inputs += [relative_attention_bias]

    plug_inputs = [i.trt_tensor for i in plug_inputs]

    layer = default_trtnet().add_plugin_v2(plug_inputs, attn_plug)
    _add_plugin_info(layer, attn_plg_creator, "padding_attn", pfc)
    assert layer.num_outputs == 1, \
        f"Plugin outputs number mismatch with expected, got {layer.num_outputs}, expected 1"
    output = _create_tensor(layer.get_output(0), layer)
    assert output is not None
    return output


@gw.record_signature
def gpt_attention(
    qkv: Tensor,
    past_key_value: Tensor,
    sequence_length: Tensor,
    host_past_key_value_lengths: Optional[Tensor],
    host_max_attention_window_sizes: Tensor,
    host_sink_token_length: Tensor,
    context_lengths: Optional[Tensor],
    cache_indirection: Optional[Tensor],
    host_request_types: Tensor,
    num_heads: int,
    num_kv_heads: int,
    hidden_size_per_head: int,
    q_scaling: float,
    rotary_embedding_dim: int = 0,
    rotary_embedding_base: float = 10000.0,
    rotary_embedding_scale_type: RotaryScalingType = RotaryScalingType.none,
    rotary_embedding_scale: float = 1.0,
    rotary_embedding_max_positions: int = 1024,
    position_embedding_type: PositionEmbeddingType = PositionEmbeddingType.
    learned_absolute,
    kv_orig_quant_scale: Optional[Tensor] = None,
    kv_quant_orig_scale: Optional[Tensor] = None,
    kv_cache_quant_mode: QuantMode = QuantMode(0),
    max_context_length: Optional[int] = None,
    mask_type: AttentionMaskType = AttentionMaskType.causal,
    alibi_slopes: Optional[Tensor] = None,
    tp_size: int = 1,
    tp_rank: int = 0,
    kv_cache_block_pointers: Optional[Tensor] = None,
    host_kv_cache_block_pointers: Tensor = None,
    do_cross_attention: bool = False,
    cross_qkv: Optional[Tensor] = None,  # for cross attention
    cross_qkv_length: Optional[Tensor] = None,  # for cross attention
    encoder_input_lengths: Optional[Tensor] = None,  # for cross attention
    relative_attention_bias: Optional[Tensor] = None,  # for relative attention
    max_distance: int = 0,  # for relative attention
    host_context_lengths: Optional[Tensor] = None,  # for pad-free input mode
    enable_pos_shift: Optional[
        bool] = False,  # for position shift attention mode in streamingllm
    dense_context_fmha: Optional[
        bool] = False,  # for dense fmha in context phase
    qkv_bias: Optional[Tensor] = None,
    use_cache: bool = True,
    medusa_position_offsets: Tensor = None,
    medusa_packed_mask: Tensor = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    '''
    Add an operation that performs the multi-head attention in GPT-like models.

    The signature of the function will change in the future release - we are in
    the process of simplifying the API. The current version is still
    work-in-progress! The following API is provided with hints regarding the
    arguments that are likely to be removed or merged with others in the future
    release.

    See docs/gpt_attention.md for the documentation of that function.

    Parameters:
        qkv: Tensor (On GPU)
            The input QKV tensor. Its shape is [batch_beam_size, max_seqlen, qkv_dim] in padded mode and [1, num_tokens, qkv_dim] in
            packed mode. Where qkv_dim depends on using MQA, GQA, or MHA. See QKV Input in docs/gpt_attention.md,

        past_key_value: Tensor (On GPU)
            The tensor that stores KV cache data. Its shape is
            [max_batch_size * max_beam_width, 2, num_kv_heads, max_seqlen, hidden_dim_per_head]
            in contiguous mode and
            [max_blocks, 2, num_kv_heads, num_tokens_per_block, hidden_dim_per_head]
            in paged mode. See KV Cache in docs/gpt_attention.md,

        sequence_lengths: Tensor (On GPU)
            The tensor that stores the length of each sequence. Its shape is
            [batch_size]. See QKV Input in docs/gpt_attention.md,

        host_past_key_value_lengths: Tensor (On CPU)
            An INT32 tensor of shape [batch_size],

        host_max_attention_window_sizes: Tensor (On CPU)
            An INT32 tensor of shape [1].
            by default, the max_attention_window_size is determined by the shape of cache_indir_table.
            And we support independent max_attention_window_size for each layer.
            This controls the sliding-window-attention/cyclic-kv-cache features.

        context_lengths: Tensor (On GPU)
            The tensor that stores the context-phase sequence length of each request. Its shape
            is [batch_size]. See QKV Input in doc/functional.py,

        cache_indirection: Tensor (On GPU)
            The tensor to reconstruct the paths when using beam-search. Its
            shape is [batch_size, beam_width, max_seqlen]. See Beam-Search in
            docs/gpt_attention.md,

        host_request_types: Tensor = None (On CPU)
            The tensor on the host that indicates if a request is in context or
            generation phase. Its shape is [batch_size]. See Inflight Batching
            in docs/gpt_attention.md,

        num_heads: int
            The number of heads,

        num_kv_heads: int
            The number of KV heads, generic to handle MHA/MQA/GQA,

        hidden_size_per_head: int
            The hidden size per head,

        q_scaling: float
            The value used to compute the scaling factor applied to the output
            of the Q*K^T product. See Scaling Factors in docs/gpt_attention.md,

        rotary_embedding_dim: int
            The dimension to compute RoPE. Use 0 when position_embedding_type is not RoPE.

        rotary_embedding_base: float
            The theta value to use for RoPE. Ignored when position_embedding_type is not RoPE.

        rotary_embedding_scale_type: RotaryScalingType
            The scaling type of RoPE. Ignored when position_embedding_type is not RoPE.
            Possible rotary scaling type:
                * RotaryScalingType.none
                * RotaryScalingType.linear
                * RotaryScalingType.dynamic

        rotary_embedding_scale: float
            The scale value to use for linear/dynamic scaling in RoPE.
            Ignored when position_embedding_type is not RoPE.
            Must be set to 1 (default) if rotary_embedding_scale_type is `none`.

        rotary_embedding_max_positions: int
            Needed only for `dynamic` RoPE scaling. Ignored otherwise.

        position_embedding_type: PositionEmbeddingType
            The position embedding type:
                * PositionEmbeddingType.learned_absolute
                * PositionEmbeddingType.relative
                * PositionEmbeddingType.rope_gptj
                * PositionEmbeddingType.rope_gpt_neox
                * PositionEmbeddingType.alibi
                * PositionEmbeddingType.alibi_with_scale

        kv_orig_quant_scale: Tensor
            The tensor to store the scaling factor for quantization to INT8/FP8
            in the KV cache. Its shape is [1]. See INT8/FP8 KV Cache in
            docs/gpt_attention.md,

        kv_quant_orig_scale: Tensor
            The tensor to store the scaling factor for dequantization from
            INT8/FP8 in the KV cache. Its shape is [1]. See INT8/FP8 KV Cache
            in docs/gpt_attention.md,

        kv_cache_quant_mode: QuantMode (int flags)
            Do we enable the INT8 or FP8 KV cache?

        max_context_length: int32_t
            The length of the longest input sequence. See QKV Input in
            docs/gpt_attention.md,

        mask_type: int = 1
            The type of mask:
                * tensorrt_llm.layers.AttentionMaskType.padding for BERT,
                * tensorrt_llm.layers.AttentionMaskType.causal for GPT,
                * tensorrt_llm.layers.AttentionMaskType.bidirectional for ChatGLM-6B,
                * tensorrt_llm.layers.AttentionMaskType.bidirectionalglm for GLM-10B,

        alibi_slopes: Tensor
            The ALiBi slopes. The ALiBi bias is computed on-the-fly in the kernel
            when possible,

        tp_size: int
            The number of processes/GPUs when tensor parallelism is activated,

        tp_rank: int
            The rank of that process (when running tensor parallelism),

        kv_cache_block_pointers:
            The tensor of block pointers for the KV cache. Its shape is
            [max_batch_size, max_beam_width, 2, max_blocks_per_sequence * 2]
            See KV cache section in docs/gpt_attention.md, on gpu

        host_kv_cache_block_pointers:
            The same as kv_cache_block_pointers, but on cpu,

        do_cross_attention: bool = False
            Do we use this as cross attention instead of self attention,

        cross_qkv: Tensor = None
            The QKV tensor of encoder output hidden states. Its shape is [batch_size, max_seqlen, 3
            * hidden_dim] in padded mode and [1, num_tokens, 3 * hidden_dim] in
            packed mode,

        cross_qkv_length: Tensor = None
            The length of the longest encoder output sequence,

        encoder_input_lengths: Tensor
            The tensor that stores the length of each encoder input sequence. Its shape is [batch_size],

        relative_attention_bias: Tensor = None
            The relative attention bias [num_heads, max_seq_len, max_seq_len], or The relative attention embedding table for implicit mode, [num_heads, num_buckets].

        max_distance: int = 0
            The maximum distance of relative position in attention, for implicit mode.
            Default value is 0, meaning to use the regular mode of relative attention bias.
            Implicit mode is only enabled when passing in non-zero positive max_distance value.
            See relative attention bias in docs/gpt_attention.md

        host_context_lengths: Tensor = None (On CPU)
            A host tensor that contains the lengths of the different inputs,

        enable_pos_shift: bool = False
            Do we enable position shift in attention to support streamingllm method,

        dense_context_fmha: bool = False
            Do we use dense fmha in context phase,

        qkv_bias: Tensor = None,
            The qkv bias tensor.

        use_cache: bool = False
            Do we need to store kv cache ? not needed if there is no generation phase.

        medusa_position_offsets: Tensor = None,
            The medusa tokens's position offsets (shared by all sequences).
            Shape: [Num_medusa_tokens + 1].

        medusa_packed_mask: Tensor = None,
            The medusa tokens's attention mask (packed into uint32_t bits).
            Shape: [Num_medusa_tokens + 1, divUp(Num_medusa_tokens + 1, 32)].

    Returns:
        The tensor produced by that layer.
    '''
    assert host_request_types is not None
    assert (alibi_slopes is not None) == (position_embedding_type.is_alibi())
    attn_plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'GPTAttention', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert attn_plg_creator is not None
    assert host_context_lengths is not None or not default_net(
    ).plugin_config.remove_input_padding
    assert isinstance(max_context_length, int)
    assert host_max_attention_window_sizes is not None
    assert host_sink_token_length is not None

    paged_kv_cache_flag = default_net().plugin_config.paged_kv_cache
    if isinstance(qkv, list):
        is_unfuse_qkv_gemm = 1
    else:
        is_unfuse_qkv_gemm = 0
    unfuse_qkv_gemm = trt.PluginField(
        "unfuse_qkv_gemm", np.array(np.int8(is_unfuse_qkv_gemm), dtype=np.int8),
        trt.PluginFieldType.INT8)

    nheads = trt.PluginField("num_heads", np.array(num_heads, dtype=np.int32),
                             trt.PluginFieldType.INT32)
    num_kv_heads = trt.PluginField("num_kv_heads",
                                   np.array(num_kv_heads, dtype=np.int32),
                                   trt.PluginFieldType.INT32)
    head_size = trt.PluginField("head_size",
                                np.array(hidden_size_per_head, dtype=np.int32),
                                trt.PluginFieldType.INT32)
    unidirectional = trt.PluginField("unidirectional",
                                     np.array(1, dtype=np.int32),
                                     trt.PluginFieldType.INT32)
    q_scaling = trt.PluginField("q_scaling",
                                np.array(q_scaling, dtype=np.float32),
                                trt.PluginFieldType.FLOAT32)
    rotary_embedding_dim = trt.PluginField(
        "rotary_embedding_dim", np.array(rotary_embedding_dim, dtype=np.int32),
        trt.PluginFieldType.INT32)
    rotary_embedding_base = trt.PluginField(
        "rotary_embedding_base",
        np.array(rotary_embedding_base, dtype=np.float32),
        trt.PluginFieldType.FLOAT32)
    rotary_embedding_scale_type = trt.PluginField(
        "rotary_embedding_scale_type",
        np.array(rotary_embedding_scale_type, dtype=np.int8),
        trt.PluginFieldType.INT8)
    rotary_embedding_scale = trt.PluginField(
        "rotary_embedding_scale",
        np.array(rotary_embedding_scale, dtype=np.float32),
        trt.PluginFieldType.FLOAT32)
    rotary_embedding_max_positions = trt.PluginField(
        "rotary_embedding_max_positions",
        np.array(rotary_embedding_max_positions, dtype=np.int32),
        trt.PluginFieldType.INT32)
    position_embedding_type = trt.PluginField(
        "position_embedding_type",
        np.array(int(position_embedding_type), dtype=np.int8),
        trt.PluginFieldType.INT8)
    context_fmha_type = trt.PluginField(
        "context_fmha_type",
        np.array(np.int8(default_net().plugin_config.context_fmha_type),
                 dtype=np.int8), trt.PluginFieldType.INT8)
    remove_input_padding = trt.PluginField(
        "remove_input_padding",
        np.array(np.int8(default_net().plugin_config.remove_input_padding),
                 dtype=np.int8), trt.PluginFieldType.INT8)
    is_medusa_enabled = trt.PluginField(
        "is_medusa_enabled",
        np.array(np.int8(medusa_packed_mask is not None), dtype=np.int8),
        trt.PluginFieldType.INT8)
    p_dtype = default_net().plugin_config.gpt_attention_plugin
    pf_type = trt.PluginField(
        "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
        trt.PluginFieldType.INT32)
    mask_type = trt.PluginField("mask_type", np.array([int(mask_type)],
                                                      np.int32),
                                trt.PluginFieldType.INT32)
    multi_block_mode = trt.PluginField(
        "multi_block_mode",
        np.array(np.int8(default_net().plugin_config.multi_block_mode),
                 dtype=np.int8), trt.PluginFieldType.INT8)
    enable_xqa = trt.PluginField(
        "enable_xqa",
        np.array(np.int8(default_net().plugin_config.enable_xqa),
                 dtype=np.int8), trt.PluginFieldType.INT8)
    tp_size = trt.PluginField("tp_size", np.array(tp_size, dtype=np.int32),
                              trt.PluginFieldType.INT32)
    tp_rank = trt.PluginField("tp_rank", np.array(tp_rank, dtype=np.int32),
                              trt.PluginFieldType.INT32)
    kv_cache_quant_mode_field = trt.PluginField(
        "kv_cache_quant_mode",
        np.array(np.int8(kv_cache_quant_mode), dtype=np.int32),
        trt.PluginFieldType.INT32)
    paged_kv_cache = trt.PluginField(
        "paged_kv_cache", np.array(paged_kv_cache_flag, dtype=np.int32),
        trt.PluginFieldType.INT32)
    tokens_per_block = trt.PluginField(
        "tokens_per_block",
        np.array(default_net().plugin_config.tokens_per_block, dtype=np.int32),
        trt.PluginFieldType.INT32)
    max_context_length = trt.PluginField("max_context_length",
                                         np.array(max_context_length, np.int32),
                                         trt.PluginFieldType.INT32)
    pos_shift_enabled = trt.PluginField(
        "pos_shift_enabled", np.array(np.int8(enable_pos_shift), dtype=np.int8),
        trt.PluginFieldType.INT8)
    dense_context_fmha = trt.PluginField(
        "dense_context_fmha",
        np.array(np.int8(dense_context_fmha), dtype=np.int8),
        trt.PluginFieldType.INT8)
    if qkv_bias is None:
        qkv_bias_enabled = trt.PluginField("qkv_bias_enabled",
                                           np.array(0, dtype=np.int8),
                                           trt.PluginFieldType.INT8)
    else:
        qkv_bias_enabled = trt.PluginField("qkv_bias_enabled",
                                           np.array(1, dtype=np.int8),
                                           trt.PluginFieldType.INT8)
    do_cross_attention_field = trt.PluginField(
        "do_cross_attention",
        np.array(np.int8(do_cross_attention), dtype=np.int8),
        trt.PluginFieldType.INT8)
    max_distance = trt.PluginField("max_distance",
                                   np.array(max_distance, dtype=np.int32),
                                   trt.PluginFieldType.INT32)
    use_paged_context_fmha_field = trt.PluginField(
        "use_paged_context_fmha",
        np.array(np.int8(default_net().plugin_config.use_paged_context_fmha),
                 dtype=np.int8), trt.PluginFieldType.INT8)
    use_cache_pf = trt.PluginField("use_cache",
                                   np.array([use_cache], dtype=np.int32),
                                   trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([
        nheads, num_kv_heads, head_size, unidirectional, q_scaling,
        position_embedding_type, rotary_embedding_dim, rotary_embedding_base,
        rotary_embedding_scale_type, rotary_embedding_scale,
        rotary_embedding_max_positions, tp_size, tp_rank, unfuse_qkv_gemm,
        context_fmha_type, multi_block_mode, enable_xqa,
        kv_cache_quant_mode_field, remove_input_padding, mask_type,
        paged_kv_cache, tokens_per_block, pf_type, max_context_length,
        qkv_bias_enabled, do_cross_attention_field, max_distance,
        pos_shift_enabled, dense_context_fmha, use_paged_context_fmha_field,
        use_cache_pf, is_medusa_enabled
    ])

    attn_plug = attn_plg_creator.create_plugin("causal_attn", pfc)
    plug_inputs = [*qkv] if is_unfuse_qkv_gemm else [qkv]
    if use_cache:
        plug_inputs += [
            sequence_length,
            host_past_key_value_lengths,
            host_max_attention_window_sizes,
            host_sink_token_length,
            context_lengths,
            cache_indirection,
            host_request_types,
        ]
    else:
        plug_inputs += [
            host_max_attention_window_sizes,
            host_sink_token_length,
            context_lengths,
            host_request_types,
        ]

    if use_cache:
        if paged_kv_cache_flag:
            plug_inputs += [
                kv_cache_block_pointers, host_kv_cache_block_pointers
            ]
        else:
            plug_inputs += [past_key_value]

    if use_cache and kv_cache_quant_mode.has_kv_cache_quant():
        plug_inputs += [kv_orig_quant_scale, kv_quant_orig_scale]

    if alibi_slopes is not None:
        plug_inputs += [alibi_slopes]

    if relative_attention_bias is not None:
        plug_inputs += [relative_attention_bias]

    if do_cross_attention:
        plug_inputs += [cross_qkv, cross_qkv_length, encoder_input_lengths]

    if default_net().plugin_config.remove_input_padding:
        plug_inputs += [host_context_lengths]

    if qkv_bias is not None:
        plug_inputs += [qkv_bias]

    if medusa_packed_mask is not None:
        # add position_ids as well only if medusa mode
        assert medusa_position_offsets is not None
        plug_inputs += [medusa_packed_mask, medusa_position_offsets]

    plug_inputs = [i.trt_tensor for i in plug_inputs]
    layer = default_trtnet().add_plugin_v2(plug_inputs, attn_plug)
    _add_plugin_info(layer, attn_plg_creator, "causal_attn", pfc)
    output = _create_tensor(layer.get_output(0), layer)
    present_key_value = None
    if use_cache and not paged_kv_cache_flag:
        present_key_value = _create_tensor(layer.get_output(1), layer)
        assert present_key_value is not None
        expected_outputs = 2
    else:
        expected_outputs = 1

    assert layer.num_outputs == expected_outputs, \
        f"Plugin outputs number mismatch with expected, got {layer.num_outputs}, expected {expected_outputs}"

    if kv_cache_quant_mode.has_int8_kv_cache() and not paged_kv_cache_flag:
        # past key value
        layer.get_input(8).set_dynamic_range(-127, 127)
        # present key value
        layer.get_output(1).set_dynamic_range(-127, 127)

    assert output is not None
    return output, present_key_value


def assertion(condition: Tensor, message: str = '') -> None:
    default_trtnet().add_assertion(condition.trt_tensor, message)


def layer_norm(input: Tensor,
               normalized_shape: Union[int, Tuple[int]],
               weight: Optional[Tensor] = None,
               bias: Optional[Tensor] = None,
               eps: float = 1e-05,
               use_diff_of_squares: bool = True) -> Tensor:
    '''
    Add a layer-norm operation on a tensor.

    That operation applies the layer-normalization to its input tensor. In its
    simplest form, for large language models, the 'normalized_shape' should be
    set to the hidden dimension of the activation tensor. Otherwise, it is the
    shape of the normalized fraction of the tensor (starting from the
    right-most dimension).

    The 'weight' tensor corresponds to 'gamma' in the layer-norm formula and
    'bias' is 'beta'. The 'eps' value is added to the variance before computing
    the squared-root.

    This implementation (when using the plugin) supports an additional flag to
    enable/disable the use of a difference of squares ('Var = Mean(X^2) -
    Mean(X)^2').

    Parameters:
        input : Tensor
            The tensor to normalize.

        normalized_shape : Union[int, Tuple[int]]
            The shape of the sub-tensor that is normalized. Use 'hidden_dim' to
            normalize the inner-most dimension of an activation tensor in LLMs.

        weight : Optional[Tensor] = None
            The 'gamma' term in layer-norm. Its shape must be
            'normalized_shape'.

        bias : Optional[Tensor] = None
            The 'beta' term in layer-norm. Its shape must be
            'normalized_shape'.

        eps : float
            The epsilon term to be added to the variance in the squared-root.

        use_diff_of_squares : bool
            Does the plugin use the difference of squares to compute the
            variance?

    Returns:
        The output tensor of that operation.
    '''
    if not default_net().plugin_config.layernorm_plugin:
        input, weight = broadcast_helper(input, weight)
        input, bias = broadcast_helper(input, bias)
        if isinstance(normalized_shape, int):  # FIXME: better way?
            axis = input.ndim() - 1
        else:
            axis = input.ndim() - len(normalized_shape)
        axes_mask = 0
        for i in range(axis, input.ndim()):
            axes_mask |= 1 << i
        layer = default_trtnet().add_normalization(input.trt_tensor,
                                                   weight.trt_tensor,
                                                   bias.trt_tensor, axes_mask)
        layer.epsilon = eps
        return _create_tensor(layer.get_output(0), layer)
    else:
        logger.warning("Layernorm plugin is going to be deprecated, "
                       "disable it for better performance.")
        plg_creator = trt.get_plugin_registry().get_plugin_creator(
            'Layernorm', '1', TRT_LLM_PLUGIN_NAMESPACE)
        assert plg_creator is not None

        eps = trt.PluginField("eps", np.array(eps, dtype=np.float32),
                              trt.PluginFieldType.FLOAT32)
        use_diff_of_squares = trt.PluginField(
            "use_diff_of_squares",
            np.array([int(use_diff_of_squares)], dtype=np.int32),
            trt.PluginFieldType.INT32)
        p_dtype = default_net().plugin_config.layernorm_plugin
        pf_type = trt.PluginField(
            "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
            trt.PluginFieldType.INT32)
        pfc = trt.PluginFieldCollection([eps, use_diff_of_squares, pf_type])
        layernorm_plug = plg_creator.create_plugin("layernorm", pfc)

        normalized_shape = [normalized_shape] if isinstance(
            normalized_shape, int) else normalized_shape
        if weight is None:
            weight = constant(
                np.ones(normalized_shape, dtype=str_dtype_to_np(p_dtype)))
        if bias is None:
            bias = constant(
                np.zeros(normalized_shape, dtype=str_dtype_to_np(p_dtype)))

        plug_inputs = [input.trt_tensor, weight.trt_tensor, bias.trt_tensor]
        layer = default_trtnet().add_plugin_v2(plug_inputs, layernorm_plug)
        _add_plugin_info(layer, plg_creator, "layernorm", pfc)
        return _create_tensor(layer.get_output(0), layer)


def rms_norm(input: Tensor,
             normalized_shape: Union[int, Tuple[int]],
             weight: Optional[Tensor] = None,
             eps: float = 1e-06) -> Tensor:
    '''
    Add a RMS norm operation on a tensor.

    That operation applies the rms-normalization to its input tensor. In its
    simplest form, for large language models, the 'normalized_shape' should be
    set to the hidden dimension of the activation tensor. Otherwise, it is the
    shape of the normalized fraction of the tensor (starting from the
    right-most dimension).

    The 'weight' tensor corresponds to 'gamma' in the rms-norm formula.
    The 'eps' value is added to the variance before computing the squared-root.

    Parameters:
        input: Tensor
            The tensor to normalize.

        normalized_shape : Union[int, Tuple[int]]
            The shape of the sub-tensor that is normalized. Use 'hidden_dim' to
            normalize the inner-most dimension of an activation tensor in LLMs.

        weight : Optional[Tensor] = None
            The 'gamma' term in layer-norm. Its shape must be
            'normalized_shape'.

        eps : float
            The epsilon term to be added to the variance in the squared-root.weig
    Returns:
        The output tensor of that operation.
    '''
    if not default_net().plugin_config.rmsnorm_plugin:
        normalized_shape = [normalized_shape] if isinstance(
            normalized_shape, int) else normalized_shape

        dim = tuple([-i - 1 for i in range(len(normalized_shape))])

        if default_net().strongly_typed:
            input_dtype = input.dtype
            fp32_input = cast(input, "float32")
            varx = pow(fp32_input, 2.0)

            varx = varx.mean(dim, keepdim=True)
            denom = varx + eps
            denom = denom.sqrt()
            fp32_y = fp32_input / denom
            y = cast(fp32_y, input_dtype)
        else:
            with precision("float32"):
                varx = pow(input, 2.0)
                varx = varx.mean(dim, keepdim=True)
                denom = varx + eps
                denom = denom.sqrt()
                y = input / denom

        if weight is not None:
            y = y * weight

        return y
    else:
        logger.warning("RMSnorm plugin is going to be deprecated, "
                       "disable it for better performance.")
        plg_creator = trt.get_plugin_registry().get_plugin_creator(
            'Rmsnorm', '1', TRT_LLM_PLUGIN_NAMESPACE)
        assert plg_creator is not None

        eps = trt.PluginField("eps", np.array(eps, dtype=np.float32),
                              trt.PluginFieldType.FLOAT32)
        p_dtype = default_net().plugin_config.rmsnorm_plugin
        pf_type = trt.PluginField(
            "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
            trt.PluginFieldType.INT32)
        pfc = trt.PluginFieldCollection([eps, pf_type])
        rmsnorm_plug = plg_creator.create_plugin("rmsnorm", pfc)

        normalized_shape = [normalized_shape] if isinstance(
            normalized_shape, int) else normalized_shape
        if weight is None:
            weight = constant(
                np.zeros(normalized_shape, dtype=str_dtype_to_np(p_dtype)))

        plug_inputs = [input.trt_tensor, weight.trt_tensor]
        layer = default_trtnet().add_plugin_v2(plug_inputs, rmsnorm_plug)
        _add_plugin_info(layer, plg_creator, "rmsnorm", pfc)
        return _create_tensor(layer.get_output(0), layer)


def repeat_interleave(tensor: Tensor, repeats: int, dim: int) -> Tensor:
    '''
    Repeats elements of a tensor along an axis.

    Parameters:
        repeats : int
            The number of repetitions along axis specified.
        dim : int
            The dimension along which repetitions are performed.

    Returns:
        A tensor with the same shape as input except for repeated elements along specified dim.

    TODO: Allow repeats to be a list of integers and dim to be unspecified.
    '''
    expanded_tensor = expand_dims(tensor, dim + 1)
    tile_output_size = concat([
        repeats if i == (dim + 1) else shape(expanded_tensor, i)
        for i in range(expanded_tensor.ndim())
    ])
    tile = expand(expanded_tensor, tile_output_size)
    tile_reshape_size = [shape(tensor, i) for i in range(tensor.ndim())]
    tile_reshape_size[dim] = tile_reshape_size[dim] * repeats
    tensor = tile.view(concat(tile_reshape_size))
    return tensor


def generate_alibi_slopes(num_heads: int,
                          dtype: trt.DataType = trt.float32,
                          tp_size: int = 1,
                          tp_rank: int = 0,
                          alibi_scale: float = 1.0) -> Tensor:
    '''
    Compute the ALiBi slopes as described in https://arxiv.org/abs/2211.05100.

    Parameters:
        num_heads : int
            The number of heads.
        dtype : trt.DataType
            The data type of the returned slopes
        tp_size : int
            The tensor parallelism size
        tp_rank : int
            The tensor parallelism rank

    Returns:
        A constant tensor that contains the ALiBi slopes.
    '''
    start_head_id = 0
    end_head_id = num_heads

    if tp_size > 1:
        rank_heads = num_heads // tp_size
        start_head_id = rank_heads * tp_rank
        end_head_id = start_head_id + rank_heads

    closest_power_of_2 = 2**np.floor(np.log2(num_heads))
    # FT's implementation
    # https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/kernels/gen_relative_pos_bias.cu#L248
    slopes_ft = []
    for h_id in range(start_head_id, end_head_id):
        if h_id < closest_power_of_2:
            slopes_ft.append(
                np.power(2**(-(2**-(np.log2(closest_power_of_2) - 3))),
                         h_id + 1))
        else:
            slopes_ft.append(
                np.power(2**(-(2**-(np.log2(closest_power_of_2 * 2) - 3))),
                         (h_id - closest_power_of_2) * 2 + 1))
    slopes = np.asarray(slopes_ft, dtype=np.float32)

    slopes = alibi_scale * slopes
    # Note that for bfloat16, we cannot case numpy tensor from float32 to bfloat16
    # because numpy does not support bfloat16. Even if we use custom type to define
    # the np_bfloat16, the "astype" here would be undefined.
    # So, we must use torch to cast tensor from float32 to bfloat16, and then use torch_to_numpy
    # to cast the tensor back.
    slopes = torch.from_numpy(slopes)
    slopes = slopes.to(trt_dtype_to_torch(dtype))
    slopes = torch_to_numpy(slopes)
    slopes = constant(slopes.reshape(1, (end_head_id - start_head_id), 1, 1))
    return slopes


def generate_alibi_biases(slopes: Tensor, key_length: Tensor) -> Tensor:
    '''
    Compute the ALiBi biases as described in https://arxiv.org/abs/2211.05100.

    The ALiBi biases are added to the result of the Q*K^T product in the
    multihead-attention block.

    Parameters:
        slopes : Tensor
            The slopes.

        key_length : Tensor
            The size of the K vector per head.

    Returns:
        A constant tensor that contains the ALiBi biases.
    '''
    # We don't need to care about the batch size or query length since we can just broadcast
    # across the batch and query dimensions

    trt_0 = constant(int32_array(0))
    arange_shape = concat([1, 1, 1, key_length])

    arange_tensor = arange(trt_0, key_length, "float32").view(arange_shape)
    return slopes * arange_tensor


def expand_mask(mask: Tensor, tgt_len: Optional[Tensor] = None) -> Tensor:
    '''
    Expand an attention mask.

    That function adds the sequence of operations to expand from a tensor of
    shape '[batch_size, src_seq_len]' to a tensor of shape
    '[batch_size, 1, tgt_seq_len, src_seq_len]'. It can be used to create the
    mask applied to the Q*K^T product before the softmax operation in the
    multihead-attention block.

    Parameters:
        mask : Tensor
            The input mask

        tgt_len : Optional[Tensor]
            The dimension of the 3rd dimension in the output tensor. If None,
            the 2nd dimension of the input is used.

    Returns:
        The tensor created by that sequence of operations.
    '''
    bsz = shape(mask, 0)
    src_len = shape(mask, 1)
    tgt_len = tgt_len if tgt_len is not None else src_len

    mask = mask.view(concat([bsz, 1, 1, src_len]))

    mask = expand(mask, concat([bsz, 1, tgt_len, src_len]))
    mask = where(mask == 0, float('-inf'), (1 - mask).cast('float32'))
    return mask


def gather_last_token_logits(hidden_states: Tensor, last_token_ids: Tensor,
                             remove_input_padding: bool) -> Tensor:
    '''
    Extract the logits that correspond to the last token from the hidden states.

    That function adds the operations to extract the logits of the last tokens
    in a batch of sequences.

    Depending on whether 'remove_input_padding' is 'True' or 'False', that
    function assumes inputs of different shapes.

    When 'remove_input_padding' is 'True', the 'hidden_states' tensor is
    assumed to be packed. It has a shape '[num_tokens, hidden_dim]' where
    'num_tokens' is the sum of the lengths of the sequences in the batch and
    'hidden_dim' is the hidden dimension. The 'last_tokens_ids' is a 1D tensor
    that encodes the inclusive prefix-sums of the lengths of the sequences in
    the batch.

    When 'remove_input_padding' is 'False', the 'hidden_states' tensor is
    assumed to be padded. It has a shape '[batch_size, max_seqlen, hidden_dim]'
    where 'max_seqlen' is the length of the longest sequence in the batch and
    'hidden_dim' is the hidden dimension.  The 'last_token_ids' is a 1D tensor
    that encodes the length of each sequence in the batch.

    In both cases, that function produces a tensor of shape '[batch_size,
    hidden_size]' where the row at index 'i' corresponds to the logits of the
    last token from the 'i'-th sequence.

    Parameters:
        hidden_states : Tensor
            The hidden states

        last_token_ids : Tensor
            The inclusive prefix-sum of the lengths or the lengths of the
            sequences in the batch.

        remove_input_padding : bool
            Indicate if the hidden_states are packed ('True') or padded
            ('False').

    Returns:
        The tensor created by that sequence of operations.
    '''
    if last_token_ids is None:
        return hidden_states

    if remove_input_padding:
        hidden_states = index_select(hidden_states, 0,
                                     last_token_ids - 1)  # [seq_len, hidden]

        hidden_states = hidden_states.view(
            concat([shape(last_token_ids, 0),
                    shape(hidden_states, 1)]))
    else:
        ndim = last_token_ids.ndim()
        if ndim == 1:
            # only calculate logits for the last token
            # [batch_size, seqlen, hidden_size] -> [batch_size, hidden_size]
            last_token_ids = last_token_ids.view(
                concat([shape(last_token_ids, 0), 1, 1]))
            last_token_ids = expand(
                last_token_ids,
                concat([shape(last_token_ids, 0), 1,
                        shape(hidden_states, 2)]))
            last_token_ids = last_token_ids - 1
            hidden_states = gather(
                hidden_states, dim=1, indices=last_token_ids).view(
                    concat([shape(hidden_states, 0),
                            shape(hidden_states, 2)]))
        elif ndim == 2:  # speculative decoding needs last few token's logits
            # last_token_ids is of shape [batch_size, num_last_tokens]
            # So [batch_size, seqlen, hidden_size] -> [batch_size, num_last_tokens, hidden_size]
            last_token_ids = last_token_ids.view(
                concat([shape(last_token_ids, 0),
                        shape(last_token_ids, 1), 1]))
            last_token_ids = expand(
                last_token_ids,
                concat([
                    shape(last_token_ids, 0),
                    shape(last_token_ids, 1),
                    shape(hidden_states, 2)
                ]))
            hidden_states = gather(hidden_states, dim=1, indices=last_token_ids)
    return hidden_states


ACT2FN = {
    'relu': relu,
    'tanh': tanh,
    'gelu': gelu,
    'gelu_new': gelu,
    'gelu_fast': gelu,
    'geglu': geglu,
    'silu': silu,
    'softplus': softplus,
    'squared-relu': squared_relu,
    'swiglu': swiglu,
    'fast-swiglu': swiglu,
}

GATED_ACT_2_ACT = {
    'swiglu': 'silu',
    'fast-swiglu': 'silu',
    'geglu': 'gelu',
}


def is_gated_activation(activation):
    '''
    Is a given activation function gated?

    Parameters:
        activation : str
            The name of the activation function.

    Returns:
        True if the function is gated, False otherwise.
    '''
    assert activation in ACT2FN
    return activation in GATED_ACT_2_ACT


def non_gated_version(activation):
    '''
    Given an activation function, get the non-gated version.

    If the activation function is non-gated, it returns the same activation
    function name.

    For example, that function returns 'silu' for 'swiglu' and 'relu' for
    'relu'.

    Parameters:
        activation : str
            The name of the activation function.

    Returns:
        The name of the non-gated activation function.
    '''
    if is_gated_activation(activation):
        return GATED_ACT_2_ACT[activation]
    return activation


def lora_plugin(
    input: Tensor = None,
    in_hidden_size: int = 0,
    out_hidden_sizes: List[int] = [0],
    host_request_types: Tensor = None,
    transa: bool = False,
    transb: bool = False,
    host_context_lengths: Tensor = None,  # for pad-free input mode
    max_context_length: int = 0,
    max_low_rank: int = 0,
    lora_ranks: List[Tensor] = None,
    lora_weights_pointers: List[Tensor] = None,
):
    '''
    Parameters:
        lora_ids : cpu Tensor = None
            A tensor that contains the lora ids of different inputs.

        in_hidden_size/out_hidden_size : int
            the lora computation workflow is
            [M, in_hidden_size] -> [M, low_rank] -> [M, out_hidden_size]

        host_request_types : Tensor = None
            The tensor on the host that indicates if a request is in context or
            generation phase. Its shape is [batch_size]. See Inflight Batching
            in docs/gpt_attention.md,

        transa : bool
            Is the first input transposed? Set to 'True' if you want the first
            input to be transposed, 'False' otherwise.

        transb : bool
            Is the second input transposed? Set to 'True' if you want the
            second input to be transposed, 'False' otherwise.

        host_context_lengths: cpu Tensor = None
            A host tensor that contains the lengths of the different inputs,

        max_context_length : int
            Maximum length during context phase, used to determine the workspace size.

        max_low_rank : int
            Maximum low_rank, used to determine the workspace size.

        lora_ranks : cpu Tensor with shape [batch_size]
            The low_rank of each request

        lora_weights_pointers : cpu int64 Tensor with shape [batch_size, 2]
            The weights pointers of each request. Consist of in_pointer and out_pointer.

    Return:
        The tensor produced by that layer.

    '''
    assert host_context_lengths is not None or not default_net(
    ).plugin_config.remove_input_padding

    trt.get_plugin_registry().plugin_creator_list
    in_hidden_size_field = trt.PluginField(
        "in_hidden_size", np.array(in_hidden_size, dtype=np.int32),
        trt.PluginFieldType.INT32)
    out_hidden_size_field_list = [
        trt.PluginField(f"out_hidden_size_{i}", np.array(o, dtype=np.int32),
                        trt.PluginFieldType.INT32)
        for i, o in enumerate(out_hidden_sizes)
    ]
    transa = 1 if transa else 0
    transa = trt.PluginField("transa", np.array(transa, dtype=np.int32),
                             trt.PluginFieldType.INT32)
    transb = 1 if transb else 0
    transb = trt.PluginField("transb", np.array(transb, dtype=np.int32),
                             trt.PluginFieldType.INT32)

    plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'Lora', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plg_creator is not None

    p_dtype = default_net().plugin_config.lora_plugin
    pf_type = trt.PluginField(
        "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
        trt.PluginFieldType.INT32)
    remove_input_padding = trt.PluginField(
        "remove_input_padding",
        np.array(np.int8(default_net().plugin_config.remove_input_padding),
                 dtype=np.int8), trt.PluginFieldType.INT8)
    max_context_length_field = trt.PluginField(
        "max_context_length", np.array(max_context_length, dtype=np.int32),
        trt.PluginFieldType.INT32)
    max_low_rank_field = trt.PluginField("max_low_rank",
                                         np.array(max_low_rank, dtype=np.int32),
                                         trt.PluginFieldType.INT32)
    num_lora_modules = len(out_hidden_sizes)
    num_lora_modules_field = trt.PluginField(
        "num_lora_modules", np.array(num_lora_modules, dtype=np.int32),
        trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([
        in_hidden_size_field, transa, transb, num_lora_modules_field, pf_type,
        remove_input_padding, max_context_length_field, max_low_rank_field
    ] + out_hidden_size_field_list)
    lora_plug = plg_creator.create_plugin("lora", pfc)

    plug_inputs = [input, host_request_types
                   ] + lora_ranks + lora_weights_pointers

    if default_net().plugin_config.remove_input_padding:
        plug_inputs += [host_context_lengths]

    plug_inputs = [i.trt_tensor for i in plug_inputs]
    layer = default_trtnet().add_plugin_v2(plug_inputs, lora_plug)

    if num_lora_modules == 1:
        return _create_tensor(layer.get_output(0), layer)
    else:
        return [
            _create_tensor(layer.get_output(i), layer)
            for i in range(num_lora_modules)
        ]


def selective_scan(
    input: Tensor,
    state: Tensor,
    delta: Tensor,
    delta_bias: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
    z: Tensor,
    host_request_types: Tensor,
    dim: int,
    dstate: int,
    is_variable_B: bool,
    is_variable_C: bool,
    delta_softplus: bool,
):
    '''
    Parameters:
        input : Tensor (On GPU)
            The input tensor. Its shape is [batch_size, dim, seq_len]

        state : Tensor (On GPU)
            The ssm state tensor. Its shape is [batch_size, dim, dstate]

        delta : Tensor (On GPU)
            The delta tensor. Its shape is [batch_size, dim, seq_len]

        delta_bias : Tensor (On GPU)
            The delta bias tensor. Its shape is [dim]

        A : Tensor (On GPU)
            A matrix. Its shape is [dim, dstate]

        B : Tensor (On GPU)
            B matrix. Its shape is [batch_size, dstate, seq_len]

        C : Tensor (On GPU)
            C matrix. Its shape is [batch_size, dstate, seq_len]

        D : Tensor (On GPU)
            D matrix. Its shape is [dim]

        z : Tensor (On GPU)
            The z tensor. Its shape is [batch_size, dim, seq_len]

        host_request_types : Tensor (On CPU)
            The tensor on the host that indicates if a request is in context or
            generation phase. Its shape is [batch_size]. See Inflight Batching
            in docs/gpt_attention.md,

        dim : int
            The inner dimension of SSM block

        dstate : int
            The state dimension of SSM block

        is_variable_B : bool
            Is the matrix B a variable? Set to 'True' if B is a dynamic matrix
            during inference, 'False' otherwise

        is_variable_C : bool
            Is the matrix C a variable? Set to 'True' if C is a dynamic matrix
            during inference, 'False' otherwise

        delta_softplus : bool
            Do we apply softplus to the delta.
    '''
    assert host_request_types is not None
    selective_scan_plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'SelectiveScan', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert selective_scan_plg_creator is not None

    dim = trt.PluginField("dim", np.array(dim, dtype=np.int32),
                          trt.PluginFieldType.INT32)
    dstate = trt.PluginField("dstate", np.array(dstate, dtype=np.int32),
                             trt.PluginFieldType.INT32)
    is_variable_B = trt.PluginField(
        "is_variable_B", np.array(np.int8(is_variable_B), dtype=np.int8),
        trt.PluginFieldType.INT8)
    is_variable_C = trt.PluginField(
        "is_variable_C", np.array(np.int8(is_variable_C), dtype=np.int8),
        trt.PluginFieldType.INT8)
    delta_softplus = trt.PluginField(
        "delta_softplus", np.array(np.int8(delta_softplus), dtype=np.int8),
        trt.PluginFieldType.INT8)
    p_dtype = default_net().plugin_config.selective_scan_plugin
    pf_type = trt.PluginField(
        "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
        trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection(
        [dim, dstate, is_variable_B, is_variable_C, delta_softplus, pf_type])
    selective_scan_plug = selective_scan_plg_creator.create_plugin(
        "selectives_can", pfc)

    plug_inputs = [
        input, state, delta, delta_bias, A, B, C, D, z, host_request_types
    ]
    plug_inputs = [i.trt_tensor for i in plug_inputs]

    layer = default_trtnet().add_plugin_v2(plug_inputs, selective_scan_plug)
    _add_plugin_info(layer, selective_scan_plg_creator, "selectives_can", pfc)
    output = _create_tensor(layer.get_output(0), layer)
    present_state = _create_tensor(layer.get_output(1), layer)
    return output, present_state
