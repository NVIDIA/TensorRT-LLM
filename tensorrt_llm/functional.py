# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import torch

# isort: off
import tensorrt as trt
# isort: on

from . import graph_rewriting as gw
from ._common import default_net, default_trtnet, precision
from ._utils import (QuantModeWrapper, bf16_array, bool_array,
                     dim_resolve_negative, dim_to_trt_axes, dims_array,
                     fp16_array, fp32_array, get_sm_version, int32_array,
                     int64_array, np_dtype_to_trt, str_dtype_to_trt,
                     trt_dtype_to_np, trt_dtype_to_str)
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
        For static dimension, it has min==opt==max, thus the shape param in the ctor can be an integer
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
        if isinstance(dtype, str):
            dtype = str_dtype_to_trt(dtype)

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
        self.is_network_input = is_network_input
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

        if isinstance(dtype, str):
            dtype = str_dtype_to_trt(dtype)

        assert dtype is None or isinstance(dtype, trt.DataType)
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

    def __floordiv__(self, b):
        '''
        See functional.floordiv.
        '''
        return floordiv(self, b)

    def __mod__(self, b):
        '''
        See functional.floordiv.
        '''
        return modulo(self, b)

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

    def flatten(self, start_dim=0, end_dim=-1):
        '''
        See functional.flatten.
        '''
        return flatten(self, start_dim, end_dim)

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

    def squeeze(self, dim, zero_is_placeholder):
        '''
        See functional.squeeze.
        '''
        return squeeze(self, dim, zero_is_placeholder)

    def unsqueeze(self, dim):
        '''
        See functional.squeeze.
        '''
        return unsqueeze(self, dim)

    def log(self):
        '''
        See functional.log.
        '''
        return log(self)

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

    def select(self, dim, index):
        '''
        See functional.select.
        '''
        return select(self, dim, index)

    def unbind(self, dim=0):
        '''
        See functional.unbind.
        '''
        return unbind(self, dim)

    def repeat(self, sizes):
        '''
        See functional.repeat
        '''
        return repeat(self, sizes)

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
        return f"TensorRT LLM Tensor: {self.name=} {self.dtype=} {self.shape=}"

    def __xor__(self, b):
        '''
        Maps to functional.gt or functional.eq.
        '''
        print(f"self.shape: {self.shape}, b.shape: {b.shape}")
        a, b = broadcast_helper(self, b)
        print(f"a.shape: {a.shape}, b.shape: {b.shape}")
        return op_xor(a, b)


def _create_tensor(trt_tensor: trt.ITensor, producer: trt.ILayer) -> Tensor:
    '''
    A helper function to create a TensorRT LLM Tensor object that encapsulates
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

        producer : trt.ILayer
            The producer.

    Returns:
        The TensorRT LLM tensor (functional.Tensor) that encapsulates the
        TensorRT tensor and the layer that produces it. The former is
        accessible through the attribute 'trt_tensor' and the latter using the
        attribute 'producer'.
    '''
    assert trt_tensor is not None
    assert producer is not None

    # Set the layer name since this is the only
    # centralized location to pass the name from
    # module space to the TRT IR
    default_net()._set_layer_name(producer)

    assert trt_tensor.shape.__len__(
    ) >= 0, f"tensor {trt_tensor.name} has an invalid shape"
    tensor = Tensor(name=trt_tensor.name,
                    dtype=trt_tensor.dtype,
                    shape=trt_tensor.shape,
                    is_network_input=False)
    tensor.trt_tensor = trt_tensor
    tensor.producer = producer

    # tb.print_stack(limit=10) # FOR DEBUGGING: filter producer.name if needed
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
    longrope = 3
    llama3 = 4
    yarn = 5
    mrope = 6

    @staticmethod
    def from_string(s):
        try:
            return RotaryScalingType[s]
        except KeyError:
            raise ValueError(f'Unsupported rotary scaling type: {s}')


class PositionEmbeddingType(IntEnum):
    learned_absolute = 0
    rope_gptj = 1
    rope_gpt_neox = 2
    long_rope = 3
    alibi = 4
    alibi_with_scale = 5
    relative = 6
    chatglm = 7
    yarn = 8
    mrope = 9
    deferred = 10  # Apply customized positional embedding by using an external embedder. K will be cached before embedding.

    def is_rope(self) -> bool:
        return self in [
            self.rope_gptj, self.rope_gpt_neox, self.long_rope, self.mrope
        ]

    def is_mrope(self) -> bool:
        return self in [self.mrope]

    def is_alibi(self) -> bool:
        return self in [self.alibi, self.alibi_with_scale]

    def is_deferred(self) -> bool:
        return self in [self.deferred]

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
    sliding_window_causal = 2
    bidirectional = 3
    bidirectionalglm = 4  # TODO: merge this mask into bidirectional
    blocksparse = 5
    custom_mask = 6


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


class SliceInputType(IntEnum):
    data = 0
    start = 1
    size = 2
    stride = 3
    fill_value = 4
    axes = 5


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


def int_clip(input: Tensor, lower: int, upper: int) -> Tensor:
    assert lower <= upper, f"Lower bound must be less than or equal to upper bound i.e. {lower} <= {upper}"
    res = minimum(input, upper)
    res = maximum(res, lower)
    return res


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
    behavior is undefined if the last dimension is not even.

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
    if not default_net().strongly_typed:
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
    use_fp32_acc = use_fp32_acc and input.dtype == trt.DataType.HALF and mat2.dtype == trt.DataType.HALF

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


def gemm_swiglu(input: Tensor,
                weight: Tensor,
                bias: Optional[Tensor] = None,
                scale_d0: float = 1.0,
                scale_d1: float = 1.0,
                scale_output: float = 1.0) -> Tensor:
    '''
    Add a matrix multiplication, followed by SwiGLU (`x * SiLU(gate)`) operation.

    The second SwiGLU operation takes the preceding tensor, splits it into two halves
    along the last dimension, applies SiLU to the second half and multiply the results. The
    behaviour is undefined if the last dimension is not even.

        Parameters:
        input : Tensor
            The first tensor (often called A).

        weight : Tensor
            The second tensor (often called B).

        bias : Optional[Tensor]
            The per-channel bias. The plugin with fp8 dtype does not support bias yet.

        scale_d0 : float
            The scale for dequantizing x, used for fp8

        scale_d1 : float
            The scale for dequantizing gate, used for fp8

        scale_output : float
            The scale for quantizing output, used for fp8

                Returns:
        The tensor produced by the inserted layer.
    '''
    plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'GemmSwiglu', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plg_creator is not None

    p_dtype = default_net().plugin_config.gemm_swiglu_plugin
    if p_dtype == "fp8":
        assert bias is None, "fp8 gemm_swiglu does not support bias yet"

    pf_type = trt.PluginField(
        "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
        trt.PluginFieldType.INT32)
    pf_has_bias = trt.PluginField(
        "has_bias", np.array(np.int8(0 if bias is None else 1), np.int8),
        trt.PluginFieldType.INT8)
    pf_scale_d0 = trt.PluginField("scale_d0",
                                  np.array(scale_d0, dtype=np.float32),
                                  trt.PluginFieldType.FLOAT32)
    pf_scale_d1 = trt.PluginField("scale_d1",
                                  np.array(scale_d1, dtype=np.float32),
                                  trt.PluginFieldType.FLOAT32)
    pf_scale_output = trt.PluginField("scale_output",
                                      np.array(scale_output, dtype=np.float32),
                                      trt.PluginFieldType.FLOAT32)

    pfc = trt.PluginFieldCollection(
        [pf_type, pf_has_bias, pf_scale_d0, pf_scale_d1, pf_scale_output])
    gemm_swiglu_plug = plg_creator.create_plugin("gemm_swiglu", pfc)

    # TODO(anchengc) pass nullptr when no bias
    if bias is None:
        bias = constant(
            np.zeros([weight.shape[0]], dtype=trt_dtype_to_np(input.dtype)))
    plug_inputs = [input.trt_tensor, weight.trt_tensor, bias.trt_tensor]

    layer = default_trtnet().add_plugin_v2(plug_inputs, gemm_swiglu_plug)

    return _create_tensor(layer.get_output(0), layer)


def constant(ndarray: np.ndarray,
             as_dtype: trt.DataType | None = None,
             as_shape=None) -> Tensor:
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
    trt_dtype = np_dtype_to_trt(ndarray.dtype) if as_dtype is None else as_dtype
    trt_shape = trt.Dims(
        ndarray.shape) if as_shape is None else trt.Dims(as_shape)
    trt_count = 1
    for i in range(len(trt_shape)):
        trt_count *= trt_shape[i]
    weights = trt.Weights(trt_dtype, ndarray.ctypes.data, trt_count)
    # Prevent underlying numpy array from going out of scope
    default_net().register_ndarray(ndarray)
    layer = default_trtnet().add_constant(trt_shape, weights)
    if not default_net().strongly_typed:
        layer.set_output_type(0, trt_dtype)
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
          mode: trt.SampleMode = None,
          fill_value: Union[float, Tensor] = None) -> Tensor:
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

    In pseudo-code the behavior of that operation can be described as follows
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

    if fill_value is not None and isinstance(fill_value, float):
        fill_value = constant(fp32_array(fill_value))

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

    if mode is trt.SampleMode.FILL and isinstance(fill_value, Tensor):
        layer.set_input(4, fill_value.trt_tensor)

    return _create_tensor(layer.get_output(0), layer)


def pad(input: Tensor,
        pad: Union[Sequence[int], Tensor],
        mode: str = 'constant',
        value: Optional[float] = None) -> Tensor:
    '''
    Add a pad layer.

    The padding layer adds zero-padding at the start and end of the input tensor. And the
    padding size by which to pad some dimensions of input are described starting from the
    last dimension and moving forward.

    `[len(pad) / 2]` dimensions of input will be padded. For example, to pad only the last
    dimension of the input tensor, then pad has the form [padding_left, padding_right]; to
    pad the last 2 dimensions of the input tensor, then use [padding_left, padding_right,
    padding_top, padding_bottom]; to pad the last 3 dimensions, use [padding_left,
    padding_right, padding_top, padding_bottom, padding_front, padding_back].

    Parameters:
        input : Tensor
            The input tensor on which the padding_2d is performed.
        pad : sequence of int
            An m-elements tuple for padding, where its length m meets the requirement that
            m <= 2*input dimensions, and m is even.
        mode : str
            Only \'constant\' is supported.
        value : float
            Fill value for 'constant' padding. Default: 0.

    Returns:
        The tensor produced by the inserted layer.
    '''
    assert mode == "constant", "Only `'constant'` is supported now."

    if isinstance(pad, list) or isinstance(pad, tuple):
        assert (
            len(pad) % 2 == 0 and len(pad) <= 2 * input.ndim()
        ), "The length of `pad` should be even and less than 2*input.ndim"
        pad = constant(np.array(pad).astype(np.int32)).view([-1, 2])
    elif isinstance(pad, Tensor):
        pad = pad.flatten()
        assert (
            pad.size(0) % 2 == 0 and pad.size(0) <= 2 * input.ndim()
        ), "The length of `pad` should be even and less than 2*input.ndim"
        pad = pad.cast("int32").view([-1, 2])
    else:
        raise NotImplementedError(f"pad type {type(pad)} not supported")
    if value is None:
        value = 0

    pad = concat([constant(np.zeros((1, 2), dtype=np.int32)),
                  pad])  # pre-padding the indices
    padding_index = [0] * input.ndim()
    padding_index[-(pad.size(0) - 1):] = list(range(pad.size(0) - 1, 0,
                                                    -1))  # reverse the indices
    pad = index_select(pad,
                       dim=0,
                       index=constant(np.array(padding_index, dtype=np.int32)))
    pre_padding, post_padding = chunk(pad, chunks=2, dim=1)
    start = (pre_padding.flatten() * (-1)).cast('int32')
    extend_size = (pre_padding + post_padding).flatten()
    size = (extend_size + shape(input)).cast('int32')
    layer = default_trtnet().add_slice(input.trt_tensor,
                                       start=[0] * input.ndim(),
                                       shape=[0] * input.ndim(),
                                       stride=[1] * input.ndim())
    layer.mode = trt.SampleMode.FILL
    layer.set_input(SliceInputType.start, start.trt_tensor)
    layer.set_input(SliceInputType.size, size.trt_tensor)
    layer.set_input(SliceInputType.fill_value,
                    constant_to_tensor_(value, dtype=input.dtype).trt_tensor)
    output = _create_tensor(layer.get_output(0), layer)
    return output


def rand(shape: Tensor,
         low: float = 0,
         high: float = 1,
         dtype: Union[str, trt.DataType] = 'float32') -> Tensor:
    '''
    This operation adds a fill layer that generates a random (uniform) tensor with the specified shape and data type.

    Parameters:
        shape: Tensor
            The shape of the tensor needed to be generated.
        low: float
            The minimum value (inclusive) of the range used for random.
        high: float
            The maximum value (inclusive) of the range used for random.
        dtype: Union[str, trt.DataType]
            The desired data type for the output tensor.
    Returns:
        The generated random tensor produced by the fill layer.
    '''
    # NOTE: DISABLED FOR NOW UNTIL THE FILL LAYER (RANDOM_UNIFORM) in TRT IS FIXED
    assert False, "The rand() op is temporarily disabled."
    low = constant(fp32_array(low))
    high = constant(fp32_array(high))
    trt_dtype = dtype if isinstance(dtype,
                                    trt.DataType) else str_dtype_to_trt(dtype)

    layer = default_trtnet().add_fill([0], trt.FillOperation.RANDOM_UNIFORM,
                                      trt_dtype)

    layer.set_input(0, shape.trt_tensor)
    layer.set_input(1, low.trt_tensor)
    layer.set_input(2, high.trt_tensor)
    return _create_tensor(layer.get_output(0), layer)


def categorical_sample(probs: Tensor, rand_data: Tensor = None) -> Tensor:
    '''
    This is a sampling operation and an equivalent of torch.distributions.Categorical.sample()
    i.e. given a probability distribution tensor, it samples an index of that tensor.
    See: https://pytorch.org/docs/stable/distributions.html#torch.distributions.categorical.Categorical.sample
    NOTE: This assumes that the given probabilities are **not** normalized.

    Parameters:
        probs: Tensor
            A 1-D floating point tensor representing the probability distributions.
        rand_data: Tensor (optional)
            A random tensor of same shape as `probs` tensor.
            If not provided, this function will add a rand() op to generate it and use for sampling.
    Returns:
        A tensor containing a single index of the `probs` tensor representing the sample.
    '''
    probs = probs / sum(probs, dim=-1, keepdim=True)
    rand_shape = []
    assert probs.ndim() > 0
    for i in range(probs.ndim() - 1):
        rand_shape.append(shape(probs, i))
    rand_shape = concat(rand_shape)
    if rand_data is None:
        rand_data = rand(rand_shape, low=0, high=1, dtype=probs.dtype)
    assert rand_shape == shape(rand_data)
    rand_data = expand(unsqueeze(rand_data, -1), shape(probs))
    cum_probs = cumsum(probs, dim=-1)
    cmp = cast(cum_probs >= rand_data, probs.dtype)
    samples = argmax(cmp, dim=-1)
    return samples


class Conditional:
    '''
    Add an operation to conditionally execute two code paths/subgraphs.

    Usage:
        1. conditional = Conditional(condition)
        2. input_1_ = conditional.add_input(input_1)
           ...
           input_n_ = conditional.add_input(input_n)
        3. Construct the graph to get true_output_value and false_output_value using input_1_, ..., input_n_
        4. output = conditional.add_output(true_output_value, false_output_value)
    '''

    def __init__(self, condition: Tensor):
        self.layer = default_trtnet().add_if_conditional()
        if condition.ndim() > 0:
            condition = view(condition, [])
        self.layer.set_condition(condition.trt_tensor)

    def add_input(self, input: Tensor) -> Tensor:
        in_node = self.layer.add_input(input.trt_tensor)
        return _create_tensor(in_node.get_output(0), in_node)

    def add_output(self, true_value: Tensor, false_value: Tensor) -> Tensor:
        out_node = self.layer.add_output(true_value.trt_tensor,
                                         false_value.trt_tensor)
        return _create_tensor(out_node.get_output(0), out_node)


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
    res_dtype = str_dtype_to_trt(dtype)
    if isinstance(start, int):
        assert isinstance(end, int)
        array_func = int32_array if res_dtype == trt.int32 else int64_array
        start = constant(array_func(start))
        end = constant(array_func(end))
    elif isinstance(start, Tensor):
        assert isinstance(end, Tensor)
        assert start.dtype == trt.int32 or start.dtype == trt.int64
        assert end.dtype == trt.int32 or end.dtype == trt.int64
        if start.dtype != end.dtype:
            if start.dtype == trt.int32:  # end == trt.int64
                if res_dtype == trt.int32:
                    end = cast(end, "int32")
                else:
                    start = cast(start, "int64")
            else:  # start == trt.int64 and end == trt.int32
                if res_dtype == trt.int32:
                    start = cast(start, "int32")
                else:
                    end = cast(end, "int64")
    else:
        raise TypeError("%s is not supported" % type(start))

    assert start.dtype == end.dtype, f"start type ({start.dtype}) != end type ({end.dtype})"
    step = constant_to_tensor_(1, dtype=start.dtype, to_array=True)

    num = end - start
    num = num.view([1]).cast(trt.int64)

    layer = default_trtnet().add_fill([0], trt.FillOperation.LINSPACE,
                                      start.dtype)
    layer.set_input(0, num.trt_tensor)  # rank = 1
    layer.set_input(1, start.trt_tensor)  # rank = 0
    layer.set_input(2, step.trt_tensor)  # rank = 1
    tensor = _create_tensor(layer.get_output(0), layer)
    if tensor.dtype != res_dtype:
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

    will produce a tensor of shape [3, 2, 2]. That behavior is subject to
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
    able to be broadcasted. Repeated subscript labels in one input take the diagonal.
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

    The dimensions of the input tensor are permuted according to the sequence
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


def flatten(input: Tensor, start_dim: int = 0, end_dim: int = -1):
    '''
    Flattens input by reshaping it into a one-dimensional tensor.

    If start_dim or end_dim are passed, only dimensions starting with start_dim and
    ending with end_dim are flattened. The order of elements in input is unchanged.

    Parameters:
        input : Tensor
            The input tensor to flatten.

        start_dim : int
            The first dim to flatten.

        end_dim : int
            The last dim to flatten.

    Returns:
        The tensor produced by the flatten layer.

    '''
    shape = input.shape
    ndim = input.ndim()
    if start_dim < 0: start_dim += ndim
    if end_dim < 0: end_dim += ndim
    new_shape = list()
    for i in range(start_dim):
        new_shape.append(shape[i])
    if end_dim - start_dim >= 0:
        flat_dim = 1
        for i in range(start_dim, end_dim + 1):
            flat_dim *= shape[i]
        new_shape.append(flat_dim)
    for i in range(end_dim + 1, ndim):
        new_shape.append(shape[i])
    return view(input, new_shape)


def expand_dims(input: Tensor,
                dim: Union[int, Sequence[int]],
                shape_cast_dtype=None) -> Tensor:
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

    input_shape = shape(input, cast_to_dtype=shape_cast_dtype)
    out_shapes = []
    j = 0
    for i in range(out_ndim):
        if i in dim:
            out_shapes.append(1)
        else:
            out_shapes.append(gather(input_shape, 0, j))
            j = j + 1

    out_shape = concat(out_shapes)

    return view(input, out_shape, zero_is_placeholder=False)


# NOTE: Jointly added with Apple
def squeeze(input: Tensor,
            dim: Optional[Union[int, Sequence[int]]] = None,
            zero_is_placeholder: bool = False):
    '''
    Add an operation to remove singleton dimensions of a tensor.

    This functions creates an operation that removes singleton dimension
    (dimension of size 1) at positions 'dim' in the input tensor. It works with
    negative values for the 'dim'.

    For example, for a tensor 'input' of shape [1, 4, 1, 4]:

        squeeze(input,  0) will produce an output of shape [4, 1, 4],
        squeeze(input,  2) will produce an output of shape [1, 4, 4],
        squeeze(input, [0, 2]) will produce an output of shape [4, 4],
        squeeze(input, [-2]) will produce an output of shape [1, 4, 4],

    Parameters:
        input : Tensor
            The input tensor for which the singleton dimensions will be removed.

        dim : Union[int, Sequence[int]]
            The index of the singleton dimensions in the input tensor.

    Returns:
        The tensor produced by the layer.
    '''
    if dim is None:
        dim = list(range(input.ndim()))
    if isinstance(dim, int):
        dim = (dim, )
    dim = dim_resolve_negative(dim, input.ndim())

    new_shape = []
    for i, s in enumerate(input.shape):
        if s == 1 and i in dim:
            continue
        new_shape.append(shape(input, i))

    new_shape = concat(new_shape) if len(new_shape) > 0 else []
    input = input.view(new_shape, zero_is_placeholder=zero_is_placeholder)
    return input


def unsqueeze(input: Tensor, axis: int):
    '''
    Add an operation to insert a singleton dimension to a tensor.

    That functions creates an operation that insert a singleton dimension
    (dimension of size 1) at position 'axis' in the output tensor. It works with
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
        left = constant(dims_array([left]))
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


# If dim is None, return a 1-D TensorRT LLM tensor of the size
# If dim is not None, return a 0-D TensorRT LLM tensor of the dimension size
def shape(input: Tensor,
          dim: Optional[int] = None,
          cast_to_dtype: Optional[Union[str, trt.DataType]] = None,
          clip_before_cast: Sequence[int] = None) -> Tensor:
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
    if cast_to_dtype is not None:
        if clip_before_cast is not None and (cast_to_dtype == 'int32'
                                             or cast_to_dtype == trt.int32):
            assert len(
                clip_before_cast
            ) == 2, f"This parameter only expects a tuple of 2 integers (lower, upper) but got {clip_before_cast}"
            res = int_clip(res, clip_before_cast[0], clip_before_cast[1])
        res = cast(res, cast_to_dtype)

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


# NOTE: Jointly added with Apple
def scatter(input: Tensor, dim: int, indices: Tensor,
            updates: Tensor) -> Tensor:
    '''
    This operation adds a layer that creates an output tensor by element-wise
    copying values from the input tensor and then updating values by the given
     `indices` and `updates` tensors.
     For a 2D input tensor, it first copies the input to output,
     then updates the output tensor like the following for each entry in `updates`:
        output[indices[i][j]][j] = updates[i][j] if dim=0
        output[i][indices[i][j]] = updates[i][j] if dim=1
     If the `input` tensor is [[1, 2, 3], [4, 5, 6]],
     the indices tensor is [[1, 2], [0, 1]],
     the updates tensor is [[-1, -2], [-3, -4]], and dim=1
     the output tensor will be [[1, -1, -2], [-3, -4, 6]].
     Parameters:
        input: Tensor
            The input data that needs to be updated.
        dim: int
            The axis on which the scatter is to be performed.
        indices: Tensor
            An integer tensor of the same rank as input that indicates the positions to be updated.
        updates: Tensor
            A data tensor of same shape as the `indices` tensor that contains the update values.
     Returns:
        A tensor created by the element-wise scatter layer.
    '''
    layer = default_trtnet().add_scatter(input.trt_tensor,
                                         indices.trt_tensor,
                                         updates.trt_tensor,
                                         mode=trt.ScatterMode.ELEMENT)
    layer.axis = dim
    return _create_tensor(layer.get_output(0), layer)


def gather_nd(input: Tensor, indices: Tensor, batch_dims: int = 1) -> Tensor:
    '''
    Adds a layer that performs a gather with some element-wise dimensions.
    See: https://onnx.ai/onnx/operators/onnx__GatherND.html
    The gather is performed on dim=batch_dims.

    Parameters:
        input: Tensor
            The tensor on which the gather operation is performed.
        indices: Tensor
            The tensor that indicates which entries to be gathered.
        batch_dims: int
            The number of first dimensions that should be skipped before gather starts.
    Returns:
        A tensor created by the gather layer with GatherMode.ND.
    '''
    gather_layer = default_trtnet().add_gather_v2(input.trt_tensor,
                                                  indices.trt_tensor,
                                                  mode=trt.GatherMode.ND)
    gather_layer.num_elementwise_dims = batch_dims
    return _create_tensor(gather_layer.get_output(0), gather_layer)


def nonzero(input: Tensor) -> Tensor:
    '''
    Adds a layer that finds the indices of non-zero values of the input tensor.

    Parameters:
        input: Tensor
            The input tensor for which we need to find the indices of non-zero values.
    Returns:
        A tensor of shape [D, C] where D is the number of dimensions of `input` and
        C is the number of non-zero values in it.
        Each column of this 2D tensor represents the index tuple for each non-zero value.
    '''
    non_zero_layer = default_trtnet().add_non_zero(input.trt_tensor)
    return _create_tensor(non_zero_layer.get_output(0), non_zero_layer)


def masked_select(input: Tensor, mask: Tensor) -> Tensor:
    '''
    Add an operation to select elements from a tensor according to a boolean
    mask tensor.

    Given an input tensor, that function creates an operation that selects
    elements at the indices indicated by the boolean mask tensor to create
    a new tensor. The output tensor is a 1-D tensor.

    The input tensor must have rank >= 1. The shapes of the input tensor and
    the mask tensor don’t need to match, but they must be able to be broadcasted.

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


def cumsum(input: Tensor, dim: int, prefer_plugin: bool = True) -> Tensor:
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

        prefer_plugin : bool
            Whether to use the cumsumLastDim plugin if dim is last dim.

    Returns:
        The tensor containing the inclusive cumulative sum of input.
    '''
    assert input.rank() >= 1, "input should have rank >= 1"
    assert dim < input.rank() and dim >= -input.rank(
    ), f"dim should be in [{-input.rank()}, {input.rank()}) when input have rank {input.rank()}"

    dim = dim_resolve_negative(dim, input.ndim())[0]

    if dim == input.ndim() - 1:
        if prefer_plugin:
            last_dim = input.size(-1)
            if last_dim == -1:  # dynamic?
                last_dim = shape(input, -1)
            old_shape = shape(input)
            if input.ndim() == 1:
                input_2d = unsqueeze(
                    input, 0)  # special handling of rank-1 dynamic tensor
            elif input.ndim() != 2:
                input_2d = input.view(concat([-1, last_dim]),
                                      zero_is_placeholder=False)
            else:
                input_2d = input
            cumsum_last_dim_plg_creator = trt.get_plugin_registry(
            ).get_plugin_creator('CumsumLastDim', '1', TRT_LLM_PLUGIN_NAMESPACE)
            assert cumsum_last_dim_plg_creator is not None
            input_length = trt.PluginField(
                "input_length", np.array(input_2d.size(-1), dtype=np.int32),
                trt.PluginFieldType.INT32)
            pf_type = trt.PluginField("type_id",
                                      np.array([int(input_2d.dtype)], np.int32),
                                      trt.PluginFieldType.INT32)
            pfc = trt.PluginFieldCollection([input_length, pf_type])
            cumsum_last_dim_plug = cumsum_last_dim_plg_creator.create_plugin(
                "cumsum_last_dim", pfc)
            plug_inputs = [input_2d]
            plug_inputs = [i.trt_tensor for i in plug_inputs]
            layer = default_trtnet().add_plugin_v2(plug_inputs,
                                                   cumsum_last_dim_plug)
            _add_plugin_info(layer, cumsum_last_dim_plg_creator,
                             "cumsum_last_dim", pfc)
            output = _create_tensor(layer.get_output(0), layer)
            output = output.view(old_shape, zero_is_placeholder=False)
            return output
        else:
            # credit to Apple
            reduction_length = shape(input, -1)
            reduction_range = arange(constant_to_tensor_(0,
                                                         dtype='int64',
                                                         to_array=False),
                                     reduction_length,
                                     dtype='int64')
            lower_triangle = cast(unsqueeze(reduction_range, 0)
                                  <= unsqueeze(reduction_range, 1),
                                  dtype=input.dtype)
            output = sum(unsqueeze(input, -2) * lower_triangle, dim=-1)
            return output
    else:
        slice_shape = []
        for i in range(input.ndim()):
            if i != dim:
                slice_shape.append(shape(input, i))

        zero_tensor = constant_to_tensor_(0, input.dtype, False)
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

        loop_output_layer = loop_layer.add_loop_output(
            cur_sum, trt.LoopOutput.CONCATENATE, dim)
        loop_output_layer.set_input(1, trip_limit)
        return _create_tensor(loop_output_layer.get_output(0),
                              loop_output_layer)


def masked_scatter(input: Tensor, mask: Tensor, source: Tensor) -> Tensor:
    '''
    Add the masked_scatter base on PyTorch definition.

    See https://pytorch.org/docs/stable/generated/torch.Tensor.masked_scatter_.html#torch-tensor-masked-scatter for a
    description of that function.

    Parameters:
        input : Tensor
            The input tensor.

        mask : Tensor
            The boolean mask tensor that indicates elements to select.

        source: Tensor
            The tensor to copy from
    Returns:
        The tensor containing the source tensor selected by mask.

    '''
    assert input.rank() >= 1, "input should have rank >= 1"
    input, mask = broadcast_helper(input, mask)
    expanded_mask = expand(mask, shape(input))

    non_zero_layer = default_trtnet().add_non_zero(expanded_mask.trt_tensor)

    shuffle_layer = default_trtnet().add_shuffle(non_zero_layer.get_output(0))
    shuffle_layer.second_transpose = (1, 0)
    source = source.view([-1])

    scatter_layer = default_trtnet().add_scatter(input.trt_tensor,
                                                 shuffle_layer.get_output(0),
                                                 source.trt_tensor,
                                                 mode=trt.ScatterMode.ND)

    return _create_tensor(scatter_layer.get_output(0), scatter_layer)


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
    assert len(
        inputs
    ) > 0, f"Number of inputs ({len(inputs)}) to the concatenation layer must be > 0."
    tmp = []
    inputs = constants_to_tensors_(*inputs)
    for i in inputs:
        if i.rank() == 0:
            tmp.append(i.view([1]))
        else:
            tmp.append(i)

    layer = default_trtnet().add_concatenation([i.trt_tensor for i in tmp])
    layer.axis = dim_resolve_negative(dim, tmp[0].ndim())[0]
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


def _lookup_plugin(input: Tensor, weight: Tensor, rank: int,
                   per_token_scale: Tensor) -> Tensor:
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

    p_dtype = per_token_scale.dtype
    pf_type = trt.PluginField("type_id", np.array([int(p_dtype)], np.int32),
                              trt.PluginFieldType.INT32)

    rank = trt.PluginField("rank", np.array([int(rank)], np.int32),
                           trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([pf_type, rank])
    lookup_plug = plg_creator.create_plugin("lookup", pfc)
    plug_inputs = [input.trt_tensor, weight.trt_tensor]
    if per_token_scale is not None:
        plug_inputs.append(per_token_scale.trt_tensor)
        weight.trt_tensor.set_dynamic_range(-127, 127)
    layer = default_trtnet().add_plugin_v2(plug_inputs, lookup_plug)
    _add_plugin_info(layer, plg_creator, "lookup", pfc)
    return _create_tensor(layer.get_output(0), layer)


def embedding(input: Tensor,
              weight: Tensor,
              tp_size=1,
              tp_group=None,
              sharding_dim=0,
              tp_rank=None,
              per_token_scale=None,
              padding=None) -> Tensor:
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

        padding: Tensor
            Additional padding added to the end of the embedding table before feeding into gather op.

    Returns:
        The tensor produced by the embedding lookup layer.
    '''

    # Per token scale is only supported by lookup plugin so if per_token_scale is not None, we must use lookup plugin
    # Otherwise, we prefer to use ootb
    use_lookup_plugin = per_token_scale is not None

    if padding is not None:
        padded_weight = concat([weight, padding], dim=0)
    else:
        padded_weight = weight

    # Distribute embedding lookup table across multiple GPU
    if tp_size > 1 and tp_group is not None:
        if sharding_dim == 0:  # TP on vocab_size dimension
            if tp_rank is None:
                raise ValueError(
                    "Rank cannot be none for tensor parallelism on vocab dim")

            if use_lookup_plugin:
                x = _lookup_plugin(input, weight, tp_rank, per_token_scale)
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
                    padded_weight.trt_tensor, placeholder_input.trt_tensor, 0)
                tmp_output = _create_tensor(layer.get_output(0), layer)

                # Set zero for invalid results
                placeholder_tmp = cast(is_qualified_expand, tmp_output.dtype)
                placeholder = placeholder_tmp - placeholder_tmp
                x = where(is_qualified_expand, tmp_output, placeholder)

                # Use all reduce to collect the results
                x = allreduce(x, tp_group)

        elif sharding_dim == 1:  # TP on hidden dimension
            layer = default_trtnet().add_gather(padded_weight.trt_tensor,
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
        if use_lookup_plugin:
            x = _lookup_plugin(input,
                               padded_weight,
                               rank=0,
                               per_token_scale=per_token_scale)
        else:
            layer = default_trtnet().add_gather(padded_weight.trt_tensor,
                                                input.trt_tensor, 0)
            x = _create_tensor(layer.get_output(0), layer)
    return x


def constant_to_tensor_(input: Union[Tensor, int, float, bool],
                        dtype: Union[trt.DataType, str] = None,
                        to_array=True) -> Tensor:
    if dtype is None:
        # deduce the type from the given value
        # NOTE: bool is a subtype of int, so bool needs to be checked first
        if isinstance(input, bool):
            dtype = trt.bool
        elif isinstance(input, int):
            dtype = trt.int32
        else:
            dtype = trt.float32

    if not isinstance(input, Tensor):
        if isinstance(dtype, str):
            dtype = str_dtype_to_trt(dtype)
        array_fn_dict = {
            trt.int64: int64_array,
            trt.int32: int32_array,
            trt.float32: fp32_array,
            trt.float16: fp16_array,
            trt.bfloat16: bf16_array,
            trt.bool: bool_array,
        }
        assert dtype in array_fn_dict
        return constant(array_fn_dict[dtype]([input] if to_array else input))

    return input


def constants_to_tensors_(
        *inputs: Union[Tensor, int, float]) -> Tuple[Tensor, ...]:
    '''
    Helper function to create tensors from multiple inputs.

    For each inputs, that function first creates a constant tensor if the input
    is an integer or a float. Then, if any input is int64, it upcasts other
    integer inputs to int64.

    Parameters:
        inputs : Tuple[Union[Tensor, int, float], ...]
            The inputs to create tensors from.

    Returns:
        A tuple of tensors.
    '''
    has_int64: bool = False
    for i in inputs:
        if isinstance(i, int) and (i >= 2**31 or i < -2**31)\
                or isinstance(i, Tensor) and i.dtype == trt.int64:
            has_int64 = True
            break

    if not has_int64:
        return tuple(constant_to_tensor_(i) for i in inputs)

    result = []
    for i in inputs:
        if isinstance(i, int) or isinstance(i, Tensor) and i.dtype == trt.int32:
            result.append(
                constant_to_tensor_(i, trt.int64 if has_int64 else trt.int32))
        else:
            result.append(constant_to_tensor_(i))
    return tuple(result)


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
            left, right.dtype if isinstance(right, Tensor) else None)
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

        add      for op=trt.ElementWiseOperation.SUM
        sub      for op=trt.ElementWiseOperation.SUB
        mul      for op=trt.ElementWiseOperation.PROD
        div      for op=trt.ElementWiseOperation.DIV
        floordiv for op=trt.ElementWiseOperation.FLOOR_DIV
        gt       for op=trt.ElementWiseOperation.GREATER
        lt       for op=trt.ElementWiseOperation.LESS
        op_and   for op=trt.ElementWiseOperation.AND
        op_or    for op=trt.ElementWiseOperation.OR
        eq       for op=trt.ElementWiseOperation.EQUAL
        minimum  for op=trt.ElementWiseOperation.MIN
        maximum  for op=trt.ElementWiseOperation.MAX
        pow      for op=trt.ElementWiseOperation.POW

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
    if left.dtype == trt.int32 and right.dtype == trt.int64:
        left = cast(left, trt.int64)
    if left.dtype == trt.int64 and right.dtype == trt.int32:
        right = cast(right, trt.int64)
    layer = default_trtnet().add_elementwise(left.trt_tensor, right.trt_tensor,
                                             op)
    return _create_tensor(layer.get_output(0), layer)


add = partial(elementwise_binary, op=trt.ElementWiseOperation.SUM)
sub = partial(elementwise_binary, op=trt.ElementWiseOperation.SUB)
mul = partial(elementwise_binary, op=trt.ElementWiseOperation.PROD)
div = partial(elementwise_binary, op=trt.ElementWiseOperation.DIV)
floordiv = partial(elementwise_binary, op=trt.ElementWiseOperation.FLOOR_DIV)
gt = partial(elementwise_binary, op=trt.ElementWiseOperation.GREATER)
lt = partial(elementwise_binary, op=trt.ElementWiseOperation.LESS)
op_and = partial(elementwise_binary, op=trt.ElementWiseOperation.AND)
op_or = partial(elementwise_binary, op=trt.ElementWiseOperation.OR)
eq = partial(elementwise_binary, op=trt.ElementWiseOperation.EQUAL)
minimum = partial(elementwise_binary, op=trt.ElementWiseOperation.MIN)
maximum = partial(elementwise_binary, op=trt.ElementWiseOperation.MAX)
pow = partial(elementwise_binary, op=trt.ElementWiseOperation.POW)
op_xor = partial(elementwise_binary, op=trt.ElementWiseOperation.XOR)


def modulo(x: Tensor, y: Union[Tensor, int]) -> Tensor:
    '''
    This function adds an element-wise modulo (x % y) operation for a given tensor.
    Since there is no TensorRT layer that can directly perform this,
    this function implements it using some of the basic operations.

    Returns:
        A tensor that represents (x % y) modulo operation.
    '''
    return x - (x // y) * y


def where(condition: Union[Tensor, bool], left: Union[Tensor, int, float],
          right: Union[Tensor, int, float]) -> Tensor:
    '''
    Add a where (aka select or if-then-else) operation.

    Assuming the three input parameters have the same shape, that function creates
    the operation to compute a tensor of the same shape such that:

        for ii in range(mul(condition.shape)):
            output[ii] = left[ii] if condition[ii] else right[ii]

    For each input, that function first creates a constant tensor if the
    condition is boolean or the left/right input is an integer or a float.
    Then, if needed, it expands the smaller tensor to make sure its
    rank is the same as the larger one. Then, it performs the selection.

    It is implemented using the ISelectLayer from TensorRT.

    Parameters:
        condition : Union[Tensor, bool]
            The condition. If that input is a boolean, the function
            creates a constant tensor.

        left : Union[Tensor, int, float]
            The first input. If that input is an integer or a float, the
            function creates a constant tensor.

        right : Union[Tensor, int, float]
            The second input. If that input is an integer or a float, the
            function creates a constant tensor.

    Returns:
        The tensor produced by this where operation.
    '''
    # Convert to tensors.
    condition = constant_to_tensor_(condition)
    left, right = constants_to_tensors_(left, right)

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
        log     for op=trt.UnaryOperation.LOG

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
log = partial(unary, op=trt.UnaryOperation.LOG)
not_op = partial(unary, op=trt.UnaryOperation.NOT)


def log_softmax(input: Tensor, dim: int) -> Tensor:
    '''
    This function is equivalent of torch.nn.functional.log_softmax() i.e.
    it performs log(softmax(input)) in a safer and faster way.

    Parameters:
        input: Tensor
            The data tensor on which log_softmax to be computed.
        dim: int
            The dimension of the input tensor along which log_softmax will be computed.
    Returns:
        A tensor of same shape as input with log_softmax computed on the specified dim.
    '''
    x_max = max(input, dim=dim, keepdim=True)
    x = input - x_max
    return x - log(sum(exp(x), dim=dim, keepdim=True))


def reduce(input: Tensor,
           op: trt.ReduceOperation,
           dim: Union[int, Tuple[int]],
           keepdim: bool = False) -> Tensor:
    '''
    Add an reduction operation to do along a dimension.

    It is implemented using the IReduceLayer from TensorRT.

    Parameters:
        input : Tensor
            The input tensor.

        op : trt.ReduceOperation
            The reduction operation to perform.
            Options: SUM, PROD, MAX, MIN, AVG

        dim : int
            The dimension along which the reduction is performed.

        keepdim : bool
            Is the dimension kept in the reduced tensor? When True the
            dimension is kept, it is removed from the shape otherwise.

    Returns:
        The tensor produced by this reduction operation.
    '''
    dim = dim_resolve_negative(dim, input.ndim())
    axes = dim_to_trt_axes(dim)

    layer = default_trtnet().add_reduce(input.trt_tensor,
                                        op,
                                        axes,
                                        keep_dims=keepdim)
    return _create_tensor(layer.get_output(0), layer)


prod = partial(reduce, op=trt.ReduceOperation.PROD)
min = partial(reduce, op=trt.ReduceOperation.MIN)


def mean(input: Tensor,
         dim: Union[int, Tuple[int]],
         keepdim: bool = False) -> Tensor:
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
    return reduce(input, op=trt.ReduceOperation.AVG, dim=dim, keepdim=keepdim)


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
    return reduce(input, op=trt.ReduceOperation.MAX, dim=dim, keepdim=keepdim)


def sum(input: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    '''
    Add an operation to compute the sum along a dimension.

    Computes the sum along the dimension 'dim' of the input tensor.

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
    return reduce(input, op=trt.ReduceOperation.SUM, dim=dim, keepdim=keepdim)


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

    output = _create_tensor(output, layer)
    a = list(range(input.ndim()))
    for d in dim:
        a.pop(d)
    indices = constant(int32_array(a))
    output_shape = shape(output)
    new_shape = gather(output_shape, 0, indices)
    return view(output, new_shape)


def gelu(x: Tensor) -> Tensor:
    '''
    Add a GELU operation.

    Parameters:
        input : Tensor
            The input tensor on which the activation function is applied.

    Returns:
        The tensor produced by the activation layer.
    '''
    return 0.5 * x * (
        tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * pow(x, 3.0))) + 1.0)


def geglu(x: Tensor) -> Tensor:
    '''
    Add a Gated-GELU operation.

    That function takes a tensor, splits it into two halves along the last
    dimension, applies GELU to the second half and multiply the results. The
    behavior is undefined if the last dimension is not even.

    Parameters:
        input : Tensor
            The input tensor on which the activation function is applied.

    Returns:
        The tensor produced by the activation layer.
    '''
    a, b = chunk(x, 2, dim=-1)
    return a * gelu(b)


def quick_gelu(x: Tensor) -> Tensor:
    return x * sigmoid(1.702 * x)


def gegelu(x: Tensor, limit: Optional[float] = None) -> Tensor:
    # a, b = x[..., ::2], x[..., 1::2]
    ndim = x.ndim()
    a_starts = [0 for i in range(ndim)]
    b_starts = [1 if i == (ndim - 1) else 0 for i in range(ndim)]
    shapes = concat([
        shape(x, i) / 2 if i == (ndim - 1) else shape(x, i) for i in range(ndim)
    ])
    strides = [2 if i == (ndim - 1) else 1 for i in range(ndim)]

    a = slice(x, a_starts, shapes, strides)
    b = slice(x, b_starts, shapes, strides)

    if limit is not None:
        a = clip(a, alpha=float(-1e20), beta=limit)
        b = clip(b, alpha=-limit, beta=limit)

    # C = B + 1
    const1 = arange(constant(int32_array(1)), constant(int32_array(2)),
                    trt_dtype_to_str(b.dtype))
    for _ in range(ndim - 1):
        const1 = expand_dims(const1, 0)

    b_shape = concat([shape(b, i) for i in range(ndim)])
    const1_arr = expand(const1, b_shape)

    return quick_gelu(a) * (b + const1_arr)


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

    # instance norm
    w_shape = [1, num_groups] + [1 for i in range(ndim - 1)]
    instance_weight = constant(np.ones(w_shape, dtype=trt_dtype_to_np(x.dtype)))
    instance_bias = constant(np.zeros(w_shape, dtype=trt_dtype_to_np(x.dtype)))
    axes_mask = 0
    for i in range(2, x.ndim()):
        axes_mask |= 1 << i
    layer = default_trtnet().add_normalization(x.trt_tensor,
                                               instance_weight.trt_tensor,
                                               instance_bias.trt_tensor,
                                               axes_mask)
    layer.epsilon = eps
    y = _create_tensor(layer.get_output(0), layer)
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

    See https://pytorch.org/docs/stable/generated/torch.nn.functional.softplus.html#torch-nn-functional-softplus for a
    description of that function.

    Parameters:
        input : Tensor
            Input TensorRT LLM Tensor.
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

    input_shuffled = stack([input], dim=input.ndim())
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
    output_1d = squeeze(output_2d, dim=-1)
    return output_1d


def conv2d(input: Tensor,
           weight: Tensor,
           bias: Optional[Tensor] = None,
           stride: Tuple[int, int] = (1, 1),
           padding: Tuple[int, int] = (0, 0),
           dilation: Tuple[int, int] = (1, 1),
           groups: int = 1,
           pre_padding: Optional[Tuple[int, int]] = None,
           post_padding: Optional[Tuple[int, int]] = None) -> Tensor:
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
    if pre_padding:
        layer.pre_padding = pre_padding
    if post_padding:
        layer.post_padding = post_padding

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


def conv3d(input: Tensor,
           weight: Tensor,
           bias: Optional[Tensor] = None,
           stride: Union[int, Tuple[int, int]] = (1, 1, 1),
           padding: Union[int, Tuple[int, int]] = (0, 0, 0),
           dilation: Union[int, Tuple[int, int]] = (1, 1, 1),
           groups: int = 1) -> Tensor:
    ##
    ## TODO: Document this function!
    ##

    ndim = input.ndim()
    # TRT requires the input of Conv3D layer to be 5-dimentional tensor.
    if ndim == 4:
        input = expand_dims(input, 0)
    assert input.ndim() == 5

    if isinstance(stride, int):
        stride = tuple([stride] * 3)
    if isinstance(padding, int):
        padding = tuple([padding] * 3)
    if isinstance(dilation, int):
        dilation = tuple([dilation] * 3)

    noutput = weight.size()[0]
    kernel_size = (weight.size()[-3], weight.size()[-2], weight.size()[-1])

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
    starts = [constant(dims_array([0])) for _ in range(ndim)]
    sizes = [shape(tensor, i) for i in range(ndim)]

    if isinstance(split_size_or_sections, int):
        # TODO: support non-divisible cases
        assert dim_value % split_size_or_sections == 0
        num_sections = dim_value // split_size_or_sections
        sizes[dim] = constant(dims_array([split_size_or_sections]))

        outputs = []
        for i in range(num_sections):
            starts[dim] = constant(dims_array([split_size_or_sections * i]))
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
            sizes[dim] = constant(dims_array([split_size_or_sections[i]]))
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


def unbind(input: Tensor, dim: int = 0):
    '''
    Removes a tensor dimension.

    Returns a tuple of all slices along a given dimension, already without it.
    '''
    ndim = input.ndim()
    outputs = split(input, 1, dim)
    output_shape = [input.shape[i] for i in range(ndim) if i != dim]
    return [output.view(output_shape) for output in outputs]


class AllReduceStrategy(IntEnum):
    NCCL = 0
    MIN_LATENCY = 1
    UB = 2
    AUTO = 3
    ONESHOT = 4
    TWOSHOT = 5
    LOWPRECISION = 6
    MNNVL = 7
    NCCL_SYMMETRIC = 8


class AllReduceFusionOp(IntEnum):
    NONE = 0
    RESIDUAL_RMS_NORM = 1
    LAST_PROCESS_FOR_UB = 2
    RESIDUAL_RMS_PREPOST_NORM = 3
    RESIDUAL_RMS_NORM_QUANT_FP8 = 4
    RESIDUAL_RMS_NORM_QUANT_NVFP4 = 5
    RESIDUAL_RMS_NORM_OUT_QUANT_FP8 = 6
    RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4 = 7
    MOE_FINALIZE_ALLREDUCE_RESIDUAL_RMS_NORM = 8


class AllReduceParams():

    def __init__(self,
                 strategy: AllReduceStrategy = AllReduceStrategy.AUTO,
                 fusion_op: AllReduceFusionOp = AllReduceFusionOp.NONE,
                 bias: Optional[Tensor] = None,
                 residual: Optional[Tensor] = None,
                 norm_weight: Optional[Tensor] = None,
                 scale: Optional[Tensor] = None,
                 norm_pre_residual_weight: Optional[Tensor] = None,
                 eps: float = 1e-06,
                 enable_allreduce: bool = True,
                 trigger_completion_at_end: bool = True):
        self.strategy = strategy
        self.fusion_op = fusion_op
        self.bias = bias
        self.residual = residual
        self.norm_weight = norm_weight
        self.scale = scale
        self.norm_pre_residual_weight = norm_pre_residual_weight
        self.eps = eps
        # For torch path only, has no effect on TRT path
        self.enable_allreduce = enable_allreduce
        self.trigger_completion_at_end = trigger_completion_at_end
        assert fusion_op == AllReduceFusionOp.NONE.value or (residual
                                                             is not None)

    def has_affine(self):
        return 1 if self.norm_weight is not None else 0

    def has_bias(self):
        return 1 if self.bias is not None else 0

    def has_scale(self):
        return 1 if self.scale is not None else 0

    def update_strategy(self):
        if self.strategy == AllReduceStrategy.AUTO and default_net(
        ).plugin_config.user_buffer:
            self.strategy = AllReduceStrategy.UB


class MoEAllReduceParams(AllReduceParams):

    def __init__(self,
                 device_num_experts: Optional[Tensor] = None,
                 expert_scale_factor: Optional[Tensor] = None,
                 expanded_idx_to_permuted_idx: Optional[Tensor] = None,
                 shared_expert_output: Optional[Tensor] = None,
                 bias: Optional[Tensor] = None,
                 residual: Optional[Tensor] = None,
                 norm_weight: Optional[Tensor] = None,
                 scale: Optional[Tensor] = None,
                 norm_pre_residual_weight: Optional[Tensor] = None,
                 eps: float = 1e-06,
                 enable_allreduce: bool = True,
                 is_cutlass_min_latency: bool = False):
        super().__init__(
            bias=bias,
            residual=residual,
            norm_weight=norm_weight,
            scale=scale,
            norm_pre_residual_weight=norm_pre_residual_weight,
            eps=eps,
            enable_allreduce=enable_allreduce,
        )
        self.device_num_experts = device_num_experts
        self.expert_scale_factor = expert_scale_factor
        self.expanded_idx_to_permuted_idx = expanded_idx_to_permuted_idx
        self.shared_expert_output = shared_expert_output
        self.is_cutlass_min_latency = is_cutlass_min_latency

    def is_valid(self):
        if self.is_cutlass_min_latency:
            return (self.device_num_experts is not None
                    and self.expert_scale_factor is not None
                    and self.shared_expert_output is not None)
        else:
            return (self.expanded_idx_to_permuted_idx is not None)


def create_allreduce_plugin(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    workspace: Optional[trt.ITensor],
    group: np.array,
    dtype: trt.DataType,
    all_reduce_params: AllReduceParams,
):
    allreduce_plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'AllReduce', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert allreduce_plg_creator is not None

    pf_group = trt.PluginField("group", group, trt.PluginFieldType.INT32)
    pf_dtype = trt.PluginField("type_id", np.array([int(dtype)], np.int32),
                               trt.PluginFieldType.INT32)
    pfc = [pf_group, pf_dtype]
    p_strategy = trt.PluginField(
        "strategy", np.array([int(all_reduce_params.strategy)], np.int8),
        trt.PluginFieldType.INT8)
    pfc.append(p_strategy)
    p_fusion_op = trt.PluginField(
        "fusion_op", np.array([int(all_reduce_params.fusion_op)], np.int8),
        trt.PluginFieldType.INT8)
    pfc.append(p_fusion_op)
    p_eps = trt.PluginField(
        "eps", np.array([float(all_reduce_params.eps)], np.float32),
        trt.PluginFieldType.FLOAT32)
    pfc.append(p_eps)
    p_affine = trt.PluginField(
        "affine", np.array([int(all_reduce_params.has_affine())], np.int8),
        trt.PluginFieldType.INT8)
    pfc.append(p_affine)
    p_bias = trt.PluginField(
        "bias", np.array([int(all_reduce_params.has_bias())], np.int8),
        trt.PluginFieldType.INT8)
    pfc.append(p_bias)
    p_scale = trt.PluginField(
        "scale", np.array([int(all_reduce_params.has_scale())], np.int8),
        trt.PluginFieldType.INT8)
    pfc.append(p_scale)

    pfc = trt.PluginFieldCollection(pfc)
    ar_plug = allreduce_plg_creator.create_plugin("allreduce", pfc)
    plug_inputs = [tensor]
    if all_reduce_params.strategy != AllReduceStrategy.NCCL and all_reduce_params.strategy != AllReduceStrategy.UB:
        plug_inputs.append(workspace)
    if all_reduce_params.fusion_op != AllReduceFusionOp.NONE:
        if all_reduce_params.has_bias() == 1:
            plug_inputs.append(all_reduce_params.bias.trt_tensor)
        plug_inputs.append(all_reduce_params.residual.trt_tensor)
        if all_reduce_params.has_affine() == 1:
            plug_inputs.append(all_reduce_params.norm_weight.trt_tensor)
            if all_reduce_params.fusion_op == AllReduceFusionOp.RESIDUAL_RMS_PREPOST_NORM:
                plug_inputs.append(
                    all_reduce_params.norm_pre_residual_weight.trt_tensor)
        if all_reduce_params.has_scale() == 1:
            plug_inputs.append(all_reduce_params.scale.trt_tensor)

    layer = network.add_plugin_v2(plug_inputs, ar_plug)
    return layer, allreduce_plg_creator, pfc


allreduce_ub_counter = 0


def allreduce(
    tensor: Tensor,
    group: List[int],
    all_reduce_params: Optional[AllReduceParams] = AllReduceParams()
) -> Tensor:
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
            NCCL delegates all-reduce to NCCL while ONESHOT and TWOSHOT are custom latency-optimal algorithms.
            AUTO chooses amongst the three based on a message-size heuristic.

    Returns:
        The tensor produced by that layer.
    '''

    global allreduce_ub_counter
    allreduce_ub_counter += 1

    if all_reduce_params is None:
        all_reduce_params = AllReduceParams()
    all_reduce_params.update_strategy()

    # TODO(TRTLLM-996): remove this WAR when custom allreduce is supported
    # for encoder models in C++ runtime.
    workspace = None
    if all_reduce_params.strategy != AllReduceStrategy.NCCL and all_reduce_params.strategy != AllReduceStrategy.UB:
        if current_all_reduce_helper().workspace is None:
            all_reduce_params.strategy = AllReduceStrategy.NCCL
        else:
            workspace = current_all_reduce_helper().workspace.trt_tensor
    if all_reduce_params.strategy == AllReduceStrategy.UB:
        tensor.mark_output("allreduce_ub_0_" + str(allreduce_ub_counter))
    dtype = default_net().plugin_config.nccl_plugin
    layer, allreduce_plg_creator, pfc = create_allreduce_plugin(
        network=default_trtnet(),
        tensor=tensor.cast(dtype).trt_tensor,
        workspace=workspace,
        group=np.array(group, dtype=np.int32),
        dtype=str_dtype_to_trt(dtype),
        all_reduce_params=all_reduce_params,
    )
    _add_plugin_info(layer, allreduce_plg_creator, "allreduce", pfc)
    if all_reduce_params.fusion_op != AllReduceFusionOp.NONE:
        inter_output = _create_tensor(layer.get_output(1),
                                      layer).cast(tensor.dtype)
        if all_reduce_params.strategy == AllReduceStrategy.UB and all_reduce_params.has_scale(
        ) == 1:
            final_output = _create_tensor(layer.get_output(0), layer)
            if all_reduce_params.fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4:
                scale_factor = _create_tensor(layer.get_output(2), layer)
        else:
            final_output = _create_tensor(layer.get_output(0),
                                          layer).cast(tensor.dtype)
        if all_reduce_params.strategy == AllReduceStrategy.UB:
            if all_reduce_params.has_scale() == 1:
                final_output.mark_output("allreduce_ub_1_" +
                                         str(allreduce_ub_counter))
                if all_reduce_params.fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4:
                    scale_factor.mark_output("allreduce_ub_2_" +
                                             str(allreduce_ub_counter))
                    return (final_output, scale_factor), inter_output
            else:
                assert all_reduce_params.fusion_op == AllReduceFusionOp.LAST_PROCESS_FOR_UB
                inter_output.mark_output("allreduce_ub_1_" +
                                         str(allreduce_ub_counter))
        return final_output, inter_output
    else:
        final_output = _create_tensor(layer.get_output(0),
                                      layer).cast(tensor.dtype)
        return final_output


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
        starts = [constant(dims_array([0])) for _ in range(ndim)]
        sizes = [shape(x, dim=d) for d in range(ndim)]
        sizes[0] = split_size
        sections = []
        for i in range(group_size):
            starts[0] = split_size * i
            sections.append(slice(x, concat(starts), concat(sizes)))
        # 2.2 concat
        x = concat(sections, dim=gather_dim)

    return x


def reduce_scatter(tensor: Tensor, group: List[int]) -> Tensor:

    plg_creater = trt.get_plugin_registry().get_plugin_creator(
        'ReduceScatter', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plg_creater is not None

    p_dtype = default_net().plugin_config.nccl_plugin
    pf_type = trt.PluginField(
        "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
        trt.PluginFieldType.INT32)
    group = trt.PluginField("group", np.array(group, dtype=np.int32),
                            trt.PluginFieldType.INT32)
    pfc = trt.PluginFieldCollection([group, pf_type])

    reduce_scatter_plug = plg_creater.create_plugin("reduce_scatter", pfc)
    plug_inputs = [tensor.cast(p_dtype).trt_tensor]

    layer = default_trtnet().add_plugin_v2(plug_inputs, reduce_scatter_plug)
    _add_plugin_info(layer, plg_creater, "reduce_scatter", pfc)

    return _create_tensor(layer.get_output(0), layer).cast(tensor.dtype)


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


def gemm_allreduce(a: Tensor,
                   b: Tensor,
                   group: List[int],
                   transa: bool = False,
                   transb: bool = False,
                   alpha: Optional[Union[np.ndarray, Tensor]] = None,
                   output_dtype: Optional[trt.DataType] = None,
                   fp8_inputs_override: bool = False,
                   a_sf: Optional[Tensor] = None,
                   b_sf: Optional[Tensor] = None):
    '''
    Add an operation that performs fused GEMM+AllReduce.

    Parameters:
        a: Tensor
            Input tensor A
        b: Tensor
            Input tensor B
        a_sf: Optional[Tensor]
            Input tensor for scaling input A
        b_sf: Optional[Tensor]
            Input tensor for scaling input B
        group: List[int]
            Ranks participating in collective
        transa: bool
            Whether or not input tensor A is transposed
        transb: bool
            Whether or not input tensor B is transposed
        alpha: float
            Alpha for GEMM -> beta * C + (alpha * acc)
        output_dtype: trt.DataType
            Output type for plugin. If it is None, we
            will use type set in plugin_config.
        fp8_inputs_override: bool
            TRT graph does not detect FP8 inputs correctly. This
            flag is used to override the derived input tensor
            types so that our plugin knows to issue FP8 MMAs.

    Returns:
        Returns GEMM output tensor which has been reduced across ranks.
    '''

    # Output tensor needs to be bound to externally managed
    # memory so keep track of layer index so we can assign
    # output tensor unique label.
    if not hasattr(gemm_allreduce, 'layer_idx'):
        gemm_allreduce.layer_idx = 0

    # Check inputs
    assert isinstance(a.dtype, trt.DataType)
    assert isinstance(b.dtype, trt.DataType)

    if fp8_inputs_override:
        assert (
            isinstance(alpha, np.ndarray) and alpha.dtype == np.float32
            and alpha.size == 1
        ), "`alpha` must be passed as a float32 ndarray if `fp8_inputs_override` is enabled for gemm_allreduce_plugin"
        assert a.dtype == trt.fp8
        assert b.dtype == trt.fp8

    if output_dtype is None:
        output_dtype = str_dtype_to_trt(
            default_net().plugin_config.gemm_allreduce_plugin)
    assert output_dtype in [trt.float16, trt.bfloat16]

    alpha_is_tensor = isinstance(alpha, Tensor)
    if alpha is None or alpha_is_tensor:
        alpha_value = np.array(1.0, dtype=np.float32)
    else:
        alpha_value = alpha

    plugin_creator = trt.get_plugin_registry().get_plugin_creator(
        'GemmAllReduce', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plugin_creator is not None

    trt_type_a = trt.fp8 if fp8_inputs_override else a.dtype
    trt_type_b = trt.fp8 if fp8_inputs_override else b.dtype

    # create plugin fields
    field_list = []
    field_list.append(
        trt.PluginField('type_a', np.array([int(trt_type_a)], np.int32),
                        trt.PluginFieldType.INT32))
    field_list.append(
        trt.PluginField('type_b', np.array([int(trt_type_b)], np.int32),
                        trt.PluginFieldType.INT32))
    field_list.append(
        trt.PluginField('type_d', np.array([int(output_dtype)], np.int32),
                        trt.PluginFieldType.INT32))
    field_list.append(
        trt.PluginField('transa', np.array(transa, dtype=np.int32),
                        trt.PluginFieldType.INT32))
    field_list.append(
        trt.PluginField('transb', np.array(transb, dtype=np.int32),
                        trt.PluginFieldType.INT32))
    field_list.append(
        trt.PluginField('group', np.array(group, dtype=np.int32),
                        trt.PluginFieldType.INT32))
    field_list.append(
        trt.PluginField('has_sfa', np.array([int(a_sf is not None)], np.int8),
                        trt.PluginFieldType.INT8))
    field_list.append(
        trt.PluginField('has_sfb', np.array([int(b_sf is not None)], np.int8),
                        trt.PluginFieldType.INT8))
    field_list.append(
        trt.PluginField('alpha_is_ptr', np.array([int(alpha_is_tensor)],
                                                 np.int8),
                        trt.PluginFieldType.INT8))
    field_list.append(
        trt.PluginField('alpha', alpha_value.flatten(),
                        trt.PluginFieldType.FLOAT32))

    # create plugin
    fields = trt.PluginFieldCollection(field_list)
    plugin = plugin_creator.create_plugin("gemm_allreduce", fields)
    # define symbolic input tensors.
    # note this does NOT allocate memory.
    inputs = [a.trt_tensor, b.trt_tensor]
    if a_sf is not None:
        inputs += [a_sf.trt_tensor]
    if b_sf is not None:
        inputs += [b_sf.trt_tensor]
    if alpha_is_tensor:
        inputs += [alpha.trt_tensor]

    layer = default_trtnet().add_plugin_v2(inputs, plugin)
    _add_plugin_info(layer, plugin_creator, "gemm_allreduce", fields)
    # define symbolic output tensors
    # both output tensors point to same physical memory but
    # one has unicast address and other has multicast address
    uc_output = _create_tensor(layer.get_output(0), layer)
    mc_output = _create_tensor(layer.get_output(1), layer)
    ipc_output = _create_tensor(layer.get_output(2), layer)
    assert uc_output is not None
    assert mc_output is not None
    assert ipc_output is not None
    # mark outputs so that we can bind our own allocated memory in runtime
    # (see generation.py)
    uc_output.mark_output(f'gemm_allreduce_uc_out_{gemm_allreduce.layer_idx}')
    mc_output.mark_output(f'gemm_allreduce_mc_out_{gemm_allreduce.layer_idx}')
    ipc_output.mark_output(f'gemm_allreduce_ipc_out_{gemm_allreduce.layer_idx}')
    gemm_allreduce.layer_idx += 1

    return uc_output


def bert_attention(tensor: Tensor,
                   input_lengths: Tensor,
                   num_heads: int,
                   head_size: int,
                   q_scaling: float,
                   relative_attention: bool = False,
                   relative_attention_bias: Tensor = None,
                   max_distance: int = 0,
                   max_input_length: Tensor = None,
                   sage_attn: bool = False,
                   sage_attn_q_block_size: int = 0,
                   sage_attn_k_block_size: int = 0,
                   sage_attn_v_block_size: int = 0,
                   cp_group: list[int] = None,
                   cp_size: int = 1,
                   cp_rank: int = 0) -> Tuple[Tensor]:
    '''
    Add an operation that performs the multi-head attention in BERT.

    The multi-head attention (MHA) is the sequence of a batched matmul, a
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
            See relative attention bias in docs/source/advanced/gpt-attention.md

        max_input_length: Tensor = None
            The maximum input sequence length represented by Tensor shape. Requires for remove_input_padding to pre-define plugin workspace size.

        sage_attn: bool = False
            SageAttention is a 8-bit implementation of attention kernel. It's input q, k, v and output datatypes are 16-bit. It performance dynamic quantization for q, k, v
            tensor every time before attention. https://github.com/thu-ml/SageAttention

        sage_attn_q_quant_size: int = 0
            dynamic quant block size along sequence dimension of q tensor. Each quant block will share one scale.

        sage_attn_k_quant_size: int = 0
            dynamic quant block size along sequence dimension of k tensor. Each quant block will share one scale.

        sage_attn_v_quant_size: int = 0
            dynamic quant block size along sequence dimension of v tensor. Each quant block will share one scale.

        cp_group: list[int] = None
            The communication group for context parallel

        cp_size: int = 1
            The communication size for context parallel

        cp_rank: int = 0
            The communication rank for context parallel

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

    sage_attn = trt.PluginField("sage_attn",
                                np.array(np.int8(sage_attn), dtype=np.int8),
                                trt.PluginFieldType.INT8)

    sage_attn_q_block_size = trt.PluginField(
        "sage_attn_q_block_size",
        np.array(sage_attn_q_block_size, dtype=np.int32),
        trt.PluginFieldType.INT32)

    sage_attn_k_block_size = trt.PluginField(
        "sage_attn_k_block_size",
        np.array(sage_attn_k_block_size, dtype=np.int32),
        trt.PluginFieldType.INT32)

    sage_attn_v_block_size = trt.PluginField(
        "sage_attn_v_block_size",
        np.array(sage_attn_v_block_size, dtype=np.int32),
        trt.PluginFieldType.INT32)

    if cp_size > 1:
        # transpose q,k,v inside qkv to make kv contiguous, which is required by ring attention
        # (b, s, 3d)
        query, key, value = chunk(tensor, 3, dim=-1)
        bs = shape(query, 0)
        seq_len = shape(query, 1)
        # (b, s, d) -> (b, s, 2d) -> (2b, s, d)
        kv = concat([key, value],
                    dim=-1).view(concat((2 * bs, seq_len, query.shape[-1])))
        tensor = concat((query, kv),
                        dim=0).view(concat((bs, seq_len, query.shape[-1] * 3)))

    cp_size = trt.PluginField("cp_size", np.array(cp_size, dtype=np.int32),
                              trt.PluginFieldType.INT32)
    cp_rank = trt.PluginField("cp_rank", np.array(cp_rank, dtype=np.int32),
                              trt.PluginFieldType.INT32)
    cp_group = cp_group or [0]
    cp_group = np.array(cp_group, dtype=np.int32)
    cp_group = trt.PluginField("cp_group", cp_group, trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([
        nheads, head_size, q_scaling, context_fmha_type, pf_type,
        do_relative_attention, max_distance, remove_padding, sage_attn,
        sage_attn_q_block_size, sage_attn_k_block_size, sage_attn_v_block_size,
        cp_size, cp_rank, cp_group
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


class RopeEmbeddingUtils:

    @staticmethod
    # ref: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py#L298
    def apply_llama3_scaling(inv_freqs: np.ndarray, rope_scaling_config: dict):

        scale_factor = rope_scaling_config.get("factor", 8.0)
        low_freq_factor = rope_scaling_config.get("low_freq_factor", 1.0)
        high_freq_factor = rope_scaling_config.get("high_freq_factor", 4.0)
        old_context_len = rope_scaling_config.get(
            "original_max_position_embeddings", 8192)

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        new_inv_freqs = []
        for inv_freq in inv_freqs:
            wavelen = 2 * math.pi / inv_freq
            if wavelen < high_freq_wavelen:
                new_inv_freqs.append(inv_freq)
            elif wavelen > low_freq_wavelen:
                new_inv_freqs.append(inv_freq / scale_factor)
            else:
                assert low_freq_wavelen != high_freq_wavelen
                smooth = (old_context_len / wavelen - low_freq_factor) / (
                    high_freq_factor - low_freq_factor)
                new_inv_freqs.append((1 - smooth) * inv_freq / scale_factor +
                                     smooth * inv_freq)
        return np.array(new_inv_freqs, dtype=inv_freqs.dtype)

    @staticmethod
    def create_sinusoidal_positions(num_pos: int,
                                    dim: int,
                                    theta: float = 10000.0,
                                    dtype=np.float32):
        inv_freq = 1.0 / (theta**(np.arange(0, dim, 2) / dim)).astype(dtype)
        sinusoid_inp = np.einsum("i , j -> i j",
                                 np.arange(num_pos, dtype=dtype),
                                 inv_freq,
                                 dtype=dtype)
        concat = np.concatenate((np.sin(sinusoid_inp), np.cos(sinusoid_inp)),
                                axis=1)
        return np.expand_dims(concat, axis=0).astype(dtype)

    @staticmethod
    def create_sinusoidal_positions_for_attention_plugin(
            num_pos: int,
            dim: int,
            theta: float = 10000.0,
            scale: float = 1.0,
            scale_type: RotaryScalingType = RotaryScalingType.none,
            # Other scaling configs that only used by certain scaling types.
            rope_scaling_config: dict = None,
            dtype=np.float32):
        if scale_type == RotaryScalingType.linear:
            scale = 1.0 / scale
        if scale_type == RotaryScalingType.llama3:
            assert rope_scaling_config is not None, "rotary_scaling config must be provided."
            inv_freq = 1.0 / (theta**(np.arange(0, dim, 2) / dim)).astype(dtype)
            inv_freq = RopeEmbeddingUtils.apply_llama3_scaling(
                inv_freq, rope_scaling_config)
        elif scale_type == RotaryScalingType.dynamic:
            # Make sure scaling_alpha exists in rope_scaling
            # Ref: https://huggingface.co/tencent/Hunyuan-A13B-Instruct-FP8/blob/main/modeling_hunyuan.py#L346
            assert rope_scaling_config[
                "alpha"] is not None, "rope_scaling_config.alpha must be provided."
            scaling_alpha = rope_scaling_config["alpha"]
            adjusted_base = theta * (scaling_alpha**(dim / (dim - 2)))
            inv_freq = 1.0 / (adjusted_base**(
                np.arange(0, dim, 2, dtype=dtype) / dim)).astype(dtype)
        else:
            inv_freq = scale / (theta
                                **(np.arange(0, dim, 2) / dim)).astype(dtype)
        sinusoid_inp = np.expand_dims(np.einsum("i , j -> i j",
                                                np.arange(num_pos, dtype=dtype),
                                                inv_freq,
                                                dtype=dtype),
                                      axis=-1)
        # fuse cos/sin into float2 (cos, sin).
        concat = np.concatenate(
            (np.cos(sinusoid_inp), np.sin(sinusoid_inp)),
            axis=-1)  #np.cos(sinusoid_inp).shape = (32768, 64, 1)

        return inv_freq, concat.reshape(1, -1).astype(dtype)

    @staticmethod
    def create_sinusoidal_positions_for_cogvlm_attention_plugin(
            num_pos: int,
            dim: int,
            theta: float = 10000.0,
            scale: float = 1.0,
            scale_type: RotaryScalingType = RotaryScalingType.none,
            vision_start: int = 1,
            vision_length: int = 1225,
            dtype=np.float32):
        if scale_type == RotaryScalingType.linear:
            scale = 1.0 / scale
        inv_freq = scale / (theta**(np.arange(0, dim, 2) / dim)).astype(dtype)
        position_id = np.hstack([
            np.arange(0, vision_start + 1, dtype=dtype),
            np.full(vision_length, vision_start + 1, dtype=dtype),
            np.arange(vision_start + 2,
                      num_pos - (vision_length - 1),
                      dtype=dtype)
        ])
        sinusoid_inp = np.expand_dims(np.einsum("i , j -> i j",
                                                position_id,
                                                inv_freq,
                                                dtype=dtype),
                                      axis=-1)
        # fuse cos/sin into float2 (cos, sin).
        concat = np.concatenate((np.cos(sinusoid_inp), np.sin(sinusoid_inp)),
                                axis=-1)

        return inv_freq, concat.reshape(1, -1).astype(dtype)

    def create_sinusoidal_positions_long_rope_for_attention_plugin(
            num_pos: int,
            num_orig_pos: int,
            dim: int,
            theta: float = 10000.0,
            scaling_short_factors: Tensor = 1.0,
            scaling_long_factors: Tensor = 1.0,
            short_mscale=None,
            long_mscale=None,
            dtype=np.float32):

        def _calc_mscale(scale):
            if scale <= 1.0:
                return 1.0
            return math.sqrt(1 + math.log(scale) / math.log(num_orig_pos))

        if short_mscale is None:
            short_mscale = _calc_mscale(num_pos / num_orig_pos)
            long_mscale = short_mscale

        def _compute_sinusoidal_positions(scale_factors, is_short,
                                          for_attention_plugin):
            inv_freq = 1 / (scale_factors *
                            (theta**(np.arange(0, dim, 2) / dim)).astype(dtype))
            sinusoid_inp = np.einsum("i , j -> i j",
                                     np.arange(num_pos, dtype=dtype),
                                     inv_freq,
                                     dtype=dtype)

            if for_attention_plugin:
                sinusoid_inp = np.expand_dims(sinusoid_inp, axis=-1)
                concat = np.concatenate(
                    (np.cos(sinusoid_inp), np.sin(sinusoid_inp)), axis=-1)
            else:
                concat = np.concatenate(
                    (np.sin(sinusoid_inp), np.cos(sinusoid_inp)), axis=1)
                concat = np.expand_dims(concat, axis=0)

            mscale = short_mscale if is_short else long_mscale
            concat = concat.astype(dtype) * mscale

            # gpt attention plugins also need inv_freq.
            if for_attention_plugin:
                return inv_freq.reshape(1, -1), concat.reshape(1, -1)
            else:
                return concat

        return _compute_sinusoidal_positions(
            scaling_short_factors, True, False), _compute_sinusoidal_positions(
                scaling_long_factors,
                False, False), _compute_sinusoidal_positions(
                    scaling_short_factors, True,
                    True), _compute_sinusoidal_positions(
                        scaling_long_factors, False, True), short_mscale

    @staticmethod
    def create_sinusoidal_positions_long_rope(
            num_pos: int,
            dim: int,
            theta: float,
            original_max_pos: int,
            short_factor: List[float],
            long_factor: List[float],
            dtype=np.float32,
            max_seq_len: Optional[int] = None):
        short_factor = np.array(short_factor, dtype=np.float32)
        long_factor = np.array(long_factor, dtype=np.float32)

        inv_freq = 1.0 / (theta**(np.arange(0, dim, 2, dtype=np.float32) / dim))
        t_pos = np.arange(np.max([num_pos, original_max_pos]), dtype=np.float32)

        # Choose proper freqs based on max_seq_len.
        factor = long_factor if max_seq_len is None or max_seq_len > original_max_pos else short_factor
        inv_freq = inv_freq / factor
        freqs = np.einsum("i,j->ij", t_pos, inv_freq)
        sinusoid_inp = freqs.astype(np.float32)[..., np.newaxis]

        # Apply scaling
        scale = num_pos / original_max_pos
        if scale <= 1.0:
            scaling_factor = 1.0
        else:
            scaling_factor = np.sqrt(1.0 +
                                     np.log(scale) / np.log(original_max_pos))

        # fuse cos/sin into float2 (cos, sin).
        concat = np.concatenate(
            (np.cos(sinusoid_inp) * scaling_factor,
             np.sin(sinusoid_inp) * scaling_factor),
            axis=-1,
        )

        return None, concat.reshape(1, -1).astype(dtype)

    @staticmethod
    def create_fake_weight(dim: int, dtype=np.half):
        return np.random.rand(dim).astype(dtype)

    # Note: When not using deepseek_yarn, make sure to set mscale_all_dim to 0.0.
    @staticmethod
    def create_sinusoidal_positions_yarn(
            num_pos: int,
            dim: int,
            base: int = 10000,
            scaling_factor: float = 1.0,
            original_max_position_embeddings: int = 4096,
            beta_fast: int = 32,
            beta_slow: int = 1,
            mscale: float = 1.0,
            mscale_all_dim: float = 1.0,
            duplicate_data: bool = True,
            dtype=torch.float32):

        # Copy from https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/modeling_deepseek.py
        # Inverse dim formula to find dim based on number of rotations
        def yarn_find_correction_dim(num_rotations, dim, base,
                                     max_position_embeddings):
            return (dim * math.log(max_position_embeddings /
                                   (num_rotations * 2 * math.pi))) / (
                                       2 * math.log(base))

        # Find dim range bounds based on rotations
        def yarn_find_correction_range(low_rot, high_rot, dim, base,
                                       max_position_embeddings):
            low = math.floor(
                yarn_find_correction_dim(low_rot, dim, base,
                                         max_position_embeddings))
            high = math.ceil(
                yarn_find_correction_dim(high_rot, dim, base,
                                         max_position_embeddings))
            if low < 0:
                low = 0
            if high > dim - 1:
                high = dim - 1
            return low, high  # Clamp values just in case

        def yarn_get_mscale(scale, mscale):
            if scale <= 1:
                return 1.0
            return 0.1 * mscale * math.log(scale) + 1.0

        def yarn_linear_ramp_mask(min, max, dim):
            if min == max:
                max += 0.001  # Prevent singularity

            linear_func = (torch.arange(dim, dtype=dtype) - min) / (max - min)
            ramp_func = torch.clamp(linear_func, 0, 1)
            return ramp_func

        pos_freqs = base**(torch.arange(0, dim, 2, dtype=dtype) / dim)
        freq_extra = 1.0 / pos_freqs
        freq_inter = 1.0 / (scaling_factor * pos_freqs)

        low, high = yarn_find_correction_range(
            beta_fast,
            beta_slow,
            dim,
            base,
            original_max_position_embeddings,
        )
        inv_freq_mask = (1 - yarn_linear_ramp_mask(low, high, dim // 2))
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        t = torch.arange(num_pos, dtype=dtype)
        sinusoid_inp = torch.einsum("i,j -> ij", t, inv_freq).unsqueeze(-1)

        _mscale = float(
            yarn_get_mscale(scaling_factor, mscale) /
            yarn_get_mscale(scaling_factor, mscale_all_dim))

        if duplicate_data:
            emb = torch.cat((sinusoid_inp, sinusoid_inp), dim=-2)
        else:
            emb = sinusoid_inp

        concat = torch.cat((torch.cos(emb) * _mscale, torch.sin(emb) * _mscale),
                           dim=-1)
        return inv_freq.numpy(), concat.reshape((1, -1)).to(dtype).numpy()

    @staticmethod
    def rotate_every_two(tensor: Tensor) -> Tensor:
        assert tensor.ndim() == 4

        shape_tensor = concat([
            shape(tensor, i) / 2 if i == (tensor.ndim() -
                                          1) else shape(tensor, i)
            for i in range(tensor.ndim())
        ])
        x1 = slice(tensor, [0, 0, 0, 0], shape_tensor, [1, 1, 1, 2])
        x2 = slice(tensor, [0, 0, 0, 1], shape_tensor, [1, 1, 1, 2])
        x1 = expand_dims(x1, 4)
        x2 = expand_dims(x2, 4)
        zero = constant(
            np.ascontiguousarray(
                np.zeros([1], dtype=trt_dtype_to_np(tensor.dtype))))
        x2 = zero - x2
        x = concat([x2, x1], 4)
        return view(
            x, concat([shape(x, 0),
                       shape(x, 1),
                       shape(x, 2),
                       shape(x, 3) * 2]))

    @staticmethod
    def rotate_half(tensor: Tensor) -> Tensor:
        # [bs, num_attention_kv_heads, seqlen, attention_head_size]
        assert tensor.ndim() == 4
        shape_tensor = concat([
            shape(tensor, i) / 2 if i == (tensor.ndim() -
                                          1) else shape(tensor, i)
            for i in range(tensor.ndim())
        ])
        last_dim = shape(tensor, tensor.ndim() - 1) / 2
        x1 = slice(tensor, [0, 0, 0, 0], shape_tensor, [1, 1, 1, 1])
        x2 = slice(tensor, concat([0, 0, 0, last_dim]), shape_tensor,
                   [1, 1, 1, 1])
        zero = constant(
            np.ascontiguousarray(
                np.zeros([1], dtype=trt_dtype_to_np(tensor.dtype))))
        x2 = zero - x2
        x = concat([x2, x1], 3)
        return x

    @staticmethod
    def apply_rotary_pos_emb(
        tensor: Tensor,
        position_embedding: List[Tensor] = None,
        pos_emb_type: PositionEmbeddingType = PositionEmbeddingType.rope_gptj
    ) -> Tensor:

        rotate_func = None
        if pos_emb_type == PositionEmbeddingType.rope_gpt_neox or pos_emb_type == PositionEmbeddingType.long_rope:
            assert len(position_embedding) == 2
            cos, sin = position_embedding
            sin = expand_dims(sin, 2)
            cos = expand_dims(cos, 2)
            sin = concat([sin, sin], 3)
            cos = concat([cos, cos], 3)
            rotate_func = RopeEmbeddingUtils.rotate_half
        elif pos_emb_type == PositionEmbeddingType.rope_gptj:
            assert len(position_embedding) == 2
            cos, sin = position_embedding
            sin = expand_dims(sin, 2)
            cos = expand_dims(cos, 2)
            sin = repeat_interleave(sin, 2, 3)
            cos = repeat_interleave(cos, 2, 3)
            rotate_func = RopeEmbeddingUtils.rotate_every_two
        elif pos_emb_type == PositionEmbeddingType.chatglm:
            assert len(position_embedding) == 4
            cos0, cos1, sin0, sin1 = position_embedding
            shape_tensor = concat([
                shape(tensor, i) / 2 if i == (tensor.ndim() -
                                              1) else shape(tensor, i)
                for i in range(tensor.ndim())
            ])
            last_dim = shape(tensor, tensor.ndim() - 1) / 2
            x_part0 = slice(tensor, [0, 0, 0, 0], shape_tensor, [1, 1, 1, 1])
            x_part1 = slice(tensor, concat([0, 0, 0, last_dim]), shape_tensor,
                            [1, 1, 1, 1])

            y_part0 = (x_part0 *
                       cos0) + (RopeEmbeddingUtils.rotate_half(x_part0) * sin0)
            y_part1 = (x_part1 *
                       cos1) + (RopeEmbeddingUtils.rotate_half(x_part1) * sin1)

            result = concat([y_part0, y_part1], dim=3)
            return result.view(shape(tensor))

        else:
            raise ValueError('The PositionEmbeddingType is not RoPE')
        return (tensor * cos) + (rotate_func(tensor) * sin)

    @staticmethod
    def apply_rotary_pos_emb_chatglm(qkv, position_embedding,
                                     num_attention_heads, attention_head_size,
                                     max_position_embeddings,
                                     rotary_embedding_scale,
                                     remove_input_padding) -> Tensor:

        half_head_size = attention_head_size // 2
        input = qkv[0] if isinstance(qkv, list) else qkv
        input_shape = shape(input)
        batch_size = 1 if remove_input_padding else shape(input, 0)
        seqlen = shape(input, 0 if remove_input_padding else 1)
        if isinstance(qkv, list):
            query, key, value = qkv
        else:
            qkv = qkv.view(
                concat([
                    batch_size,
                    seqlen,
                    num_attention_heads,
                    3,
                    attention_head_size,
                ]))
            query, key, value = split(qkv, 1, dim=3)
        q_shape = concat([
            batch_size,
            seqlen,
            num_attention_heads,
            attention_head_size,
        ])
        query = query.view(q_shape)
        key = key.view(q_shape)
        value = value.view(q_shape)

        embedding_weight = RopeEmbeddingUtils.create_sinusoidal_positions(
            max_position_embeddings, half_head_size)
        embedding_weight /= rotary_embedding_scale
        embedding_weight = np.split(embedding_weight.squeeze(0), 2, axis=1)
        embedding_weight = np.concatenate(
            [
                embedding_weight[0],
                embedding_weight[0],
                embedding_weight[1],
                embedding_weight[1],
            ],
            axis=1,
        )

        if remove_input_padding:
            position_embedding = unsqueeze(position_embedding, 0)

        embedding_weight = embedding_weight.astype(trt_dtype_to_np(query.dtype))
        embedding_weight = constant(embedding_weight)
        position_embedding = embedding(position_embedding, embedding_weight)
        position_embedding, block_embedding = split(
            position_embedding,
            1,
            dim=1,
        )
        sin0, cos0 = split(position_embedding, half_head_size, dim=3)
        sin1, cos1 = split(block_embedding, half_head_size, dim=3)

        new_shape = concat([
            batch_size,
            seqlen,
            1,
            half_head_size,
        ])
        position_embedding = [
            tensor.view(new_shape) for tensor in [cos0, cos1, sin0, sin1]
        ]

        query = RopeEmbeddingUtils.apply_rotary_pos_emb(
            tensor=query,
            position_embedding=position_embedding,
            pos_emb_type=PositionEmbeddingType.chatglm)
        key = RopeEmbeddingUtils.apply_rotary_pos_emb(
            tensor=key,
            position_embedding=position_embedding,
            pos_emb_type=PositionEmbeddingType.chatglm)

        if isinstance(qkv, list):
            qkv = [
                query.view(input_shape),
                key.view(input_shape),
                value.view(input_shape),
            ]
        else:
            qkv = concat([query, key, value], dim=2)
            qkv = qkv.view(input_shape)

        return qkv

    @staticmethod
    def apply_rotary_pos_emb_cogvlm(qkv, position_embedding,
                                    num_attention_heads, attention_head_size,
                                    max_position_embeddings,
                                    rotary_embedding_scale,
                                    remove_input_padding) -> Tensor:
        input = qkv[0] if isinstance(qkv, list) else qkv
        input_shape = shape(input)
        batch_size = 1 if remove_input_padding else shape(input, 0)
        seqlen = shape(input, 0 if remove_input_padding else 1)
        if isinstance(qkv, list):
            query, key, value = qkv
        else:
            qkv = qkv.view(
                concat([
                    batch_size,
                    seqlen,
                    3,
                    num_attention_heads,
                    attention_head_size,
                ]))
            query, key, value = split(qkv, 1, dim=2)
        q_shape = concat([
            batch_size,
            seqlen,
            num_attention_heads,
            attention_head_size,
        ])
        query = query.view(q_shape)
        key = key.view(q_shape)
        value = value.view(q_shape)

        embedding_weight = RopeEmbeddingUtils.create_sinusoidal_positions(
            max_position_embeddings, attention_head_size).squeeze(0)
        embedding_weight /= rotary_embedding_scale  # [max_position_embeddings, attention_head_size]

        if remove_input_padding:
            position_embedding = unsqueeze(position_embedding, 0)  # [1, seqlen]

        embedding_weight = constant(embedding_weight)  # float32
        position_embedding = embedding(
            position_embedding,
            embedding_weight)  # [1, seqlen, attention_head_size]
        sin, cos = split(position_embedding, attention_head_size // 2,
                         dim=-1)  # [1, seqlen, attention_head_size//2]

        input_dtype = query.dtype
        fp32_query = cast(query, "float32")
        fp32_key = cast(key, "float32")
        fp32_query = RopeEmbeddingUtils.apply_rotary_pos_emb(
            tensor=fp32_query,
            position_embedding=[cos, sin],
            pos_emb_type=PositionEmbeddingType.rope_gpt_neox)
        fp32_key = RopeEmbeddingUtils.apply_rotary_pos_emb(
            tensor=fp32_key,
            position_embedding=[cos, sin],
            pos_emb_type=PositionEmbeddingType.rope_gpt_neox)

        query = cast(fp32_query, input_dtype)
        key = cast(fp32_key, input_dtype)

        if isinstance(qkv, list):
            qkv = [
                query.view(input_shape),
                key.view(input_shape),
                value.view(input_shape),
            ]
        else:
            qkv = concat([query, key, value], dim=2)
            qkv = qkv.view(input_shape)

        return qkv


@gw.record_signature
def gpt_attention(
    *,
    qkv: Tensor,
    past_key_value: Tensor,
    attention_mask: Optional[Tensor] = None,
    attention_packed_mask: Optional[Tensor] = None,
    sequence_length: Tensor,
    host_past_key_value_lengths: Optional[Tensor],
    host_max_attention_window_sizes: Tensor,
    host_sink_token_length: Tensor,
    context_lengths: Optional[Tensor],
    cache_indirection: Optional[Tensor],
    host_request_types: Tensor,
    layer_idx: int,
    num_heads: int,
    num_kv_heads: int,
    hidden_size_per_head: int,
    q_scaling: float,
    attn_logit_softcapping_scale: float = 0.0,
    rotary_embedding_dim: int = 0,
    rotary_embedding_base: float = 10000.0,
    rotary_embedding_scale_type: RotaryScalingType = RotaryScalingType.none,
    rotary_embedding_short_m_scale: float = 1.0,
    rotary_embedding_long_m_scale: float = 1.0,
    rotary_embedding_scale: float = 1.0,
    rotary_embedding_max_positions: int = 1024,
    rotary_embedding_original_max_positions: int = 1024,
    position_embedding_type: PositionEmbeddingType = PositionEmbeddingType.
    learned_absolute,
    rotary_inv_freq: Optional[Tensor] = None,
    rotary_cos_sin: Optional[Tensor] = None,
    kv_orig_quant_scale: Optional[Tensor] = None,
    kv_quant_orig_scale: Optional[Tensor] = None,
    attention_output_orig_quant_scale: Optional[Tensor] = None,
    attention_output_sf_scale: Optional[Tensor] = None,
    kv_cache_quant_mode: Union[QuantModeWrapper, QuantMode] = QuantMode(0),
    max_context_length: Optional[int] = None,
    mask_type: AttentionMaskType = AttentionMaskType.causal,
    block_sparse_block_size: int = 64,
    block_sparse_homo_head_pattern: bool = False,
    block_sparse_num_local_blocks: int = 16,
    block_sparse_vertical_stride: int = 8,
    alibi_slopes: Optional[Tensor] = None,
    tp_size: int = 1,
    tp_rank: int = 0,
    vision_start: int = -1,
    vision_length: int = -1,
    kv_cache_block_offsets: Optional[Tensor] = None,
    host_kv_cache_block_offsets: Tensor = None,
    host_kv_cache_pool_pointers: Tensor = None,
    host_kv_cache_pool_mapping: Tensor = None,
    do_cross_attention: bool = False,
    cross_kv: Optional[Tensor] = None,  # for cross attention
    cross_kv_length: Optional[Tensor] = None,  # for cross attention
    encoder_input_lengths: Optional[Tensor] = None,  # for cross attention
    relative_attention_bias: Optional[Tensor] = None,  # for relative attention
    logn_scaling: Optional[Tensor] = None,  # for logn scaling
    max_distance: int = 0,  # for relative attention
    host_context_lengths: Optional[Tensor] = None,  # for pad-free input mode
    qkv_bias: Optional[Tensor] = None,
    use_cache: bool = True,
    spec_decoding_is_generation_length_variable: bool = False,
    spec_decoding_max_generation_length: int = 0,
    spec_decoding_generation_lengths: Tensor = None,
    spec_decoding_position_offsets: Tensor = None,
    spec_decoding_packed_mask: Tensor = None,
    spec_decoding_use: Tensor = None,
    long_rope_rotary_inv_freq: Optional[Tensor] = None,
    long_rope_rotary_cos_sin: Optional[Tensor] = None,
    mrope_rotary_cos_sin: Tensor = None,
    mrope_position_deltas: Tensor = None,
    host_runtime_perf_knobs: Optional[Tensor] = None,
    host_context_progress: Tensor = None,
    is_mla_enabled_flag: bool = False,
    q_lora_rank: int = 0,
    kv_lora_rank: int = 0,
    qk_nope_head_dim: int = 0,
    qk_rope_head_dim: int = 0,
    v_head_dim: int = 0,
    q_b_proj: Optional[Tensor] = None,
    kv_b_proj: Optional[Tensor] = None,
    k_b_proj_trans: Optional[Tensor] = None,
    skip_attn=None,
    cp_group: List[int] = [0],
    cp_size: int = 1,
    cp_rank: int = 0,
    num_kv_heads_origin: int = -1,
) -> Tuple[Tensor, Optional[Tensor]]:
    '''
    Add an operation that performs the multi-head attention in GPT-like models.

    The signature of the function will change in the future release - we are in
    the process of simplifying the API. The current version is still
    work-in-progress! The following API is provided with hints regarding the
    arguments that are likely to be removed or merged with others in the future
    release.

    See docs/source/advanced/gpt-attention.md for the documentation of that function.

    Parameters:
        qkv: Tensor (On GPU)
            The input QKV tensor. Its shape is [batch_beam_size, max_seqlen, qkv_dim] in padded mode and [num_tokens, qkv_dim] in
            packed mode. Where qkv_dim depends on using MQA, GQA, or MHA. See QKV Input in docs/source/advanced/gpt-attention.md,

        past_key_value: Tensor (On GPU)
            The tensor that stores KV cache data. Its shape is
            [max_batch_size * max_beam_width, 2, num_kv_heads, max_seqlen, hidden_dim_per_head]
            in contiguous mode and
            [max_blocks, 2, num_kv_heads, num_tokens_per_block, hidden_dim_per_head]
            in paged mode. See KV Cache in docs/source/advanced/gpt-attention.md,

        attention_mask: Tensor (On GPU)
            The tensor that stores the attention mask for unfused MHA or MMHA.
            Its shape is [num_tokens, max_kv_seqlen].

        attention_packed_mask: Tensor (On GPU)
            The tensor that stores the packed custom mask for fmha.
            Its shape is [num_tokens, max_kv_seqlen / 32], where each bit represents one mask position.

        sequence_lengths: Tensor (On GPU)
            The tensor that stores the length of each sequence. Its shape is
            [batch_size]. See QKV Input in docs/source/advanced/gpt-attention.md,

        host_past_key_value_lengths: Tensor (On CPU)
            An INT32 tensor of shape [batch_size],

        host_max_attention_window_sizes: Tensor (On CPU)
            An INT32 tensor of shape [1].
            by default, the max_attention_window_size is determined by the shape of cache_indir_table.
            And we support independent max_attention_window_size for each layer.
            This controls the sliding-window-attention kv-cache features.

        context_lengths: Tensor (On GPU)
            The tensor that stores the context-phase sequence length of each request. Its shape
            is [batch_size]. See QKV Input in doc/functional.py,

        cache_indirection: Tensor (On GPU)
            The tensor to reconstruct the paths when using beam-search. Its
            shape is [batch_size, beam_width, max_seqlen]. See Beam-Search in
            docs/source/advanced/gpt-attention.md,

        host_request_types: Tensor = None (On CPU)
            The tensor on the host that indicates if a request is in context or
            generation phase. Its shape is [batch_size]. See Inflight Batching
            in docs/source/advanced/gpt-attention.md,

        layer_idx: int
            The index of this attention layer, used to access kv_cache_block_offsets,

        num_heads: int
            The number of heads,

        num_kv_heads: int
            The number of KV heads, generic to handle MHA/MQA/GQA,

        hidden_size_per_head: int
            The hidden size per head,

        q_scaling: float
            The value used to compute the scaling factor applied to the output
            of the Q*K^T product. See Scaling Factors in docs/source/advanced/gpt-attention.md,

        attn_logit_softcapping_scale: float
            The scale * tanh(value / scale) used to compute the scaling factor applied to the output
            of the Q*K^T product.

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
                * RotaryScalingType.longrope
                * RotaryScalingType.llama3

        rotary_embedding_scale: float
            The scale value to use for linear/dynamic scaling in RoPE.
            Ignored when position_embedding_type is not RoPE.
            Must be set to 1 (default) if rotary_embedding_scale_type is `none`.

        rotary_inv_freq: float Tensor
            The rotary inv freq with shape [head_size / 2].

        rotary_cos_sin: float2(cos/sin) Tensor
            The rotary cos/sin cache, which will be reused among different requests.
            It is taken as constant tensor.

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
            docs/source/advanced/gpt-attention.md,

        kv_quant_orig_scale: Tensor
            The tensor to store the scaling factor for dequantization from
            INT8/FP8 in the KV cache. Its shape is [1]. See INT8/FP8 KV Cache
            in docs/source/advanced/gpt-attention.md,

        attention_output_orig_quant_scale: Tensor
            The tensor to store the scaling factor for quantization to FP8
            in the KV cache. Its shape is [1].

        kv_cache_quant_mode: QuantMode (int flags)
            Do we enable the INT8 or FP8 KV cache?

        max_context_length: int32_t
            The length of the longest input sequence. See QKV Input in
            docs/source/advanced/gpt-attention.md,

        mask_type: int = 1
            The type of mask:
                * tensorrt_llm.layers.AttentionMaskType.padding for BERT,
                * tensorrt_llm.layers.AttentionMaskType.causal for GPT,
                * tensorrt_llm.layers.AttentionMaskType.sliding_window_causal for GPT,
                * tensorrt_llm.layers.AttentionMaskType.bidirectional for ChatGLM-6B,
                * tensorrt_llm.layers.AttentionMaskType.bidirectionalglm for GLM-10B,
                * tensorrt_llm.layers.AttentionMaskType.blocksparse for Phi-3-small,
                * tensorrt_llm.layers.AttentionMaskType.custom_mask for any models.

        block_sparse_block_size: int
            Block size in block sparse attention

        block_sparse_homo_head_pattern: bool
            Do all attention heads share same vertical stride pattern?

        block_sparse_num_local_blocks: int
            Number of active blocks near diagonal

        block_sparse_vertical_stride: int
            Stride of active blocks in vertical dimension

        alibi_slopes: Tensor
            The ALiBi slopes. The ALiBi bias is computed on-the-fly in the kernel
            when possible,

        tp_size: int
            The number of processes/GPUs when tensor parallelism is activated,

        tp_rank: int
            The rank of that process (when running tensor parallelism),

        kv_cache_block_offsets:
            The tensor of block offsets for the KV cache. Its shape is
            [num_layers, max_batch_size, max_beam_width, 2, max_blocks_per_sequence * 2],
            See KV cache section in docs/source/advanced/gpt-attention.md, on gpu,

        host_kv_cache_block_offsets:
            The same as kv_cache_block_offsets, but on cpu,

        host_kv_cache_pool_pointers:
            The tensor of pool pointers for the KV cache. Its shape is [num_layers, 2],
            See KV cache section in docs/source/advanced/gpt-attention.md, on gpu,

        host_kv_cache_pool_mapping:
            The tensor of pool mapping for the different memory pools. Its shape is [num_layers,2] - for each layer, the index of the pool, and the index of the layer within the pool,

        do_cross_attention: bool = False
            Do we use this as cross attention instead of self attention,

        cross_kv: Tensor = None
            The KV tensor of encoder output hidden states. Its shape is [batch_size, max_seqlen, 2 * kvHeadNum * headSize] in padded mode and [1, num_tokens, 2 * kvHeadNum * headSize] in
            packed mode,

        cross_kv_length: Tensor = None
            The length of the longest encoder output sequence,

        encoder_input_lengths: Tensor
            The tensor that stores the length of each encoder input sequence. Its shape is [batch_size],

        logn_scaling: Tensor = None
            The logn scaling tensor [max_position_embedding_len], which is applied to q in order to help extrapolation

        relative_attention_bias: Tensor = None
            The relative attention bias [num_heads, max_seq_len, max_seq_len], or The relative attention embedding table for implicit mode, [num_heads, num_buckets].

        max_distance: int = 0
            The maximum distance of relative position in attention, for implicit mode.
            Default value is 0, meaning to use the regular mode of relative attention bias.
            Implicit mode is only enabled when passing in non-zero positive max_distance value.
            See relative attention bias in docs/source/advanced/gpt-attention.md

        host_context_lengths: Tensor = None (On CPU)
            A host tensor that contains the lengths of the different inputs,

        qkv_bias: Tensor = None,
            The qkv bias tensor.

        use_cache: bool = False
            Do we need to store kv cache ? not needed if there is no generation phase.

        spec_decoding_is_generation_length_variable: bool = False,
            Whether the generation lengths can be different for each sequence in a batch.
            For Medusa, this should be set False.
            For Redrafter, this should be set to True.

        spec_decoding_max_generation_length: int = 1,
            The maximum number of tokens possible in the generation phase per sequence.

        spec_decoding_generation_lengths: Tensor = None,
            The generation phase tokens' lengths for each sequence.
            Shape: [batch_size]

        spec_decoding_position_offsets: Tensor = None,
            The speculative decoding tokens's position offsets (shared by all sequences).
            Shape: [batch_size, num_draft_tokens + 1].

        spec_decoding_packed_mask: Tensor = None,
            The speculative decoding tokens's attention mask (packed into uint32_t bits).
            remove_input_padding is False:
                Shape: [batch_size, num_draft_tokens + 1, divUp(num_draft_tokens + 1, 32)].
            remove_input_padding is True:
                Shape: [sum(spec_decoding_generation_lengths), divUp(num_draft_tokens + 1, 32)].

        long_rope_rotary_inv_freq: float Tensor
            Additional rotary inv freq used for longer sequence lengths. Shape: [head_size / 2]

        long_rope_rotary_cos_sin: float2(cos/sin) Tensor
            Additional rotary cos/sin cache used for longer sequence lengths.

        is_mla_enable: bool = False
            Do we need to enable deepseekv2 mla?

        host_runtime_perf_knobs: Tensor = None,
            The runtime perf knobs bit mask, controls whether to use certain perf knob in the runtime.

        host_context_progress: Tensor = None,
            The structure used to track layer-wise progress in context phase.

        skip_attn: Tensor = None,
            A bool tensor on CPU. If it is true, don't run attention plugin, returning directly.

        num_kv_heads_origin: int
            The origin number of KV heads, without the process of TP

    Returns:
        The tensor produced by that layer.
    '''

    assert host_request_types is not None
    assert (alibi_slopes is not None) == (position_embedding_type.is_alibi())
    assert (mrope_rotary_cos_sin
            is not None) == (position_embedding_type.is_mrope())
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

    default_net().plugin_config.context_fmha_type
    if do_cross_attention and not paged_kv_cache_flag:
        pass
    if logn_scaling is not None:
        use_logn_scaling = 1
    else:
        use_logn_scaling = 0

    if num_kv_heads_origin < 1:
        num_kv_heads_origin = num_kv_heads

    unfuse_qkv_gemm = trt.PluginField(
        "unfuse_qkv_gemm", np.array(np.int8(is_unfuse_qkv_gemm), dtype=np.int8),
        trt.PluginFieldType.INT8)

    layer_idx = trt.PluginField("layer_idx", np.array(layer_idx,
                                                      dtype=np.int32),
                                trt.PluginFieldType.INT32)
    nheads = trt.PluginField("num_heads", np.array(num_heads, dtype=np.int32),
                             trt.PluginFieldType.INT32)
    vision_start = trt.PluginField("vision_start",
                                   np.array(vision_start, dtype=np.int32),
                                   trt.PluginFieldType.INT32)
    vision_length = trt.PluginField("vision_length",
                                    np.array(vision_length, dtype=np.int32),
                                    trt.PluginFieldType.INT32)
    num_kv_heads = trt.PluginField("num_kv_heads",
                                   np.array(num_kv_heads, dtype=np.int32),
                                   trt.PluginFieldType.INT32)
    num_kv_heads_origin = trt.PluginField(
        "num_kv_heads_origin", np.array(num_kv_heads_origin, dtype=np.int32),
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
    attn_logit_softcapping_scale = trt.PluginField(
        "attn_logit_softcapping_scale",
        np.array(attn_logit_softcapping_scale, dtype=np.float32),
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
    rotary_embedding_short_m_scale = trt.PluginField(
        "rotary_embedding_short_m_scale",
        np.array(rotary_embedding_short_m_scale, dtype=np.float32),
        trt.PluginFieldType.FLOAT32)
    rotary_embedding_long_m_scale = trt.PluginField(
        "rotary_embedding_long_m_scale",
        np.array(rotary_embedding_long_m_scale, dtype=np.float32),
        trt.PluginFieldType.FLOAT32)
    rotary_embedding_max_positions = trt.PluginField(
        "rotary_embedding_max_positions",
        np.array(rotary_embedding_max_positions, dtype=np.int32),
        trt.PluginFieldType.INT32)
    rotary_embedding_original_max_positions = trt.PluginField(
        "rotary_embedding_original_max_positions",
        np.array(rotary_embedding_original_max_positions, dtype=np.int32),
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
    is_spec_decoding_enabled = trt.PluginField(
        "is_spec_decoding_enabled",
        np.array(np.int8(spec_decoding_packed_mask is not None), dtype=np.int8),
        trt.PluginFieldType.INT8)
    spec_decoding_is_generation_length_variable = trt.PluginField(
        "spec_decoding_is_generation_length_variable",
        np.array(np.int8(spec_decoding_is_generation_length_variable),
                 dtype=np.int8), trt.PluginFieldType.INT8)
    spec_decoding_max_generation_length = trt.PluginField(
        "spec_decoding_max_generation_length",
        np.array(spec_decoding_max_generation_length, dtype=np.int32),
        trt.PluginFieldType.INT32)
    is_mla_enabled = trt.PluginField(
        "is_mla_enabled", np.array(is_mla_enabled_flag, dtype=np.int8),
        trt.PluginFieldType.INT8)
    q_lora_rank = trt.PluginField("q_lora_rank",
                                  np.array(q_lora_rank, dtype=np.int32),
                                  trt.PluginFieldType.INT32)
    kv_lora_rank = trt.PluginField("kv_lora_rank",
                                   np.array(kv_lora_rank, dtype=np.int32),
                                   trt.PluginFieldType.INT32)
    qk_nope_head_dim = trt.PluginField(
        "qk_nope_head_dim", np.array(qk_nope_head_dim, dtype=np.int32),
        trt.PluginFieldType.INT32)
    qk_rope_head_dim = trt.PluginField(
        "qk_rope_head_dim", np.array(qk_rope_head_dim, dtype=np.int32),
        trt.PluginFieldType.INT32)
    v_head_dim = trt.PluginField("v_head_dim",
                                 np.array(v_head_dim, dtype=np.int32),
                                 trt.PluginFieldType.INT32)
    p_dtype = default_net().plugin_config.gpt_attention_plugin
    pf_type = trt.PluginField(
        "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
        trt.PluginFieldType.INT32)
    # reset mask_type to custom_mask.
    if (attention_mask is not None) or (attention_packed_mask is not None):
        # context fmha needs packed mask.
        assert attention_packed_mask is not None
        if get_sm_version() < 100:
            mask_type = AttentionMaskType.custom_mask

    mask_type_filed = trt.PluginField("mask_type",
                                      np.array([int(mask_type)], np.int32),
                                      trt.PluginFieldType.INT32)
    block_sparse_block_size = trt.PluginField(
        "block_sparse_block_size", np.array([block_sparse_block_size],
                                            np.int32),
        trt.PluginFieldType.INT32)
    block_sparse_homo_head_pattern = trt.PluginField(
        "block_sparse_homo_head_pattern",
        np.array(np.int8(block_sparse_homo_head_pattern), np.int8),
        trt.PluginFieldType.INT8)
    block_sparse_num_local_blocks = trt.PluginField(
        "block_sparse_num_local_blocks",
        np.array([block_sparse_num_local_blocks], np.int32),
        trt.PluginFieldType.INT32)
    block_sparse_vertical_stride = trt.PluginField(
        "block_sparse_vertical_stride",
        np.array([block_sparse_vertical_stride], np.int32),
        trt.PluginFieldType.INT32)
    tp_size = trt.PluginField("tp_size", np.array(tp_size, dtype=np.int32),
                              trt.PluginFieldType.INT32)
    tp_rank = trt.PluginField("tp_rank", np.array(tp_rank, dtype=np.int32),
                              trt.PluginFieldType.INT32)
    if isinstance(kv_cache_quant_mode, QuantModeWrapper):
        # Now in TRT-LLM only use global kv_cache, so it's enough to get the first quant mode from list
        kv_cache_quant_mode = kv_cache_quant_mode[0]
    kv_cache_quant_mode_field = trt.PluginField(
        "kv_cache_quant_mode", np.array(kv_cache_quant_mode, dtype=np.int32),
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
        "pos_shift_enabled",
        np.array(np.int8(default_net().plugin_config.streamingllm),
                 dtype=np.int8), trt.PluginFieldType.INT8)
    dense_context_fmha = trt.PluginField(
        "dense_context_fmha",
        np.array(np.int8(default_net().plugin_config.streamingllm),
                 dtype=np.int8), trt.PluginFieldType.INT8)
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
    use_fp8_context_fmha_field = trt.PluginField(
        "use_fp8_context_fmha",
        np.array(np.int8(default_net().plugin_config.use_fp8_context_fmha),
                 dtype=np.int8), trt.PluginFieldType.INT8)
    has_full_attention_mask_field = trt.PluginField(
        "has_full_attention_mask",
        np.array(np.int8(attention_mask is not None), dtype=np.int8),
        trt.PluginFieldType.INT8)
    use_cache_pf = trt.PluginField("use_cache",
                                   np.array([use_cache], dtype=np.int32),
                                   trt.PluginFieldType.INT32)
    fuse_fp4_quant = default_net().plugin_config.fuse_fp4_quant
    fuse_fp4_quant_pf = trt.PluginField(
        "fuse_fp4_quant", np.array(np.int8(fuse_fp4_quant), dtype=np.int8),
        trt.PluginFieldType.INT8)
    skip_attn_pf = trt.PluginField(
        "skip_attn", np.array([skip_attn is not None], dtype=np.int8),
        trt.PluginFieldType.INT8)
    cp_size = trt.PluginField("cp_size", np.array(cp_size, dtype=np.int32),
                              trt.PluginFieldType.INT32)
    cp_rank = trt.PluginField("cp_rank", np.array(cp_rank, dtype=np.int32),
                              trt.PluginFieldType.INT32)
    cp_group = np.array(cp_group, dtype=np.int32)
    cp_group = trt.PluginField("cp_group", cp_group, trt.PluginFieldType.INT32)
    use_logn_scaling = trt.PluginField(
        "use_logn_scaling", np.array(np.int8(use_logn_scaling), dtype=np.int8),
        trt.PluginFieldType.INT8)

    pfc = trt.PluginFieldCollection([
        layer_idx, nheads, vision_start, vision_length, num_kv_heads,
        num_kv_heads_origin, head_size, unidirectional, q_scaling,
        attn_logit_softcapping_scale, position_embedding_type,
        rotary_embedding_dim, rotary_embedding_base,
        rotary_embedding_scale_type, rotary_embedding_scale,
        rotary_embedding_short_m_scale, rotary_embedding_long_m_scale,
        rotary_embedding_max_positions, rotary_embedding_original_max_positions,
        tp_size, tp_rank, unfuse_qkv_gemm, context_fmha_type,
        kv_cache_quant_mode_field, remove_input_padding, mask_type_filed,
        block_sparse_block_size, block_sparse_homo_head_pattern,
        block_sparse_num_local_blocks, block_sparse_vertical_stride,
        paged_kv_cache, tokens_per_block, pf_type, max_context_length,
        qkv_bias_enabled, do_cross_attention_field, max_distance,
        pos_shift_enabled, dense_context_fmha, use_paged_context_fmha_field,
        use_fp8_context_fmha_field, has_full_attention_mask_field, use_cache_pf,
        is_spec_decoding_enabled, spec_decoding_is_generation_length_variable,
        spec_decoding_max_generation_length, is_mla_enabled, q_lora_rank,
        kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim,
        fuse_fp4_quant_pf, skip_attn_pf, cp_size, cp_rank, cp_group,
        use_logn_scaling
    ])

    attn_plug = attn_plg_creator.create_plugin("causal_attn", pfc)
    assert attn_plug
    plug_inputs = [*qkv] if is_unfuse_qkv_gemm else [qkv]
    if attention_mask is not None and mask_type == AttentionMaskType.custom_mask:
        # useFullCustomMask
        plug_inputs += [attention_mask]
    if attention_packed_mask is not None and get_sm_version() < 100:
        # usePackedCustomMask
        plug_inputs += [attention_packed_mask]
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
            assert kv_cache_block_offsets is not None, "Paged kv cache is enabled, the kv_cache_block_offsets tensor shall not be None"
            assert host_kv_cache_block_offsets is not None, "Paged kv cache is enabled, the host_kv_cache_block_offsets tensor shall not be None"
            assert host_kv_cache_pool_pointers is not None, "Paged kv cache is enabled, the host_kv_cache_pool_pointers tensor shall not be None"
            assert host_kv_cache_pool_mapping is not None, "Paged kv cache is enabled, the host_kv_cache_pool_mapping tensor shall not be None"
            plug_inputs += [
                kv_cache_block_offsets, host_kv_cache_block_offsets,
                host_kv_cache_pool_pointers, host_kv_cache_pool_mapping
            ]
        else:
            plug_inputs += [past_key_value]

    if use_cache and kv_cache_quant_mode.has_kv_cache_quant():
        plug_inputs += [kv_orig_quant_scale, kv_quant_orig_scale]

    if attention_output_orig_quant_scale is not None:
        assert default_net(
        ).plugin_config.use_fp8_context_fmha, "FP8 Context FMHA needs to be enabled"
        plug_inputs += [attention_output_orig_quant_scale]

    if fuse_fp4_quant:
        assert attention_output_sf_scale is not None, "attention_output_sf_scale must be provided when fuse_fp4_quant is enabled."
        plug_inputs += [attention_output_sf_scale]

    if rotary_inv_freq is not None:
        plug_inputs += [rotary_inv_freq]
    if rotary_cos_sin is not None:
        plug_inputs += [rotary_cos_sin]

    if alibi_slopes is not None:
        plug_inputs += [alibi_slopes]

    if relative_attention_bias is not None:
        plug_inputs += [relative_attention_bias]

    if do_cross_attention:
        plug_inputs += [cross_kv, cross_kv_length, encoder_input_lengths]

    if default_net().plugin_config.remove_input_padding:
        plug_inputs += [host_context_lengths]

    if qkv_bias is not None:
        plug_inputs += [qkv_bias]

    if spec_decoding_packed_mask is not None:
        # add position_ids as well only if speculative decoding mode
        assert spec_decoding_position_offsets is not None
        assert spec_decoding_generation_lengths is not None
        assert spec_decoding_use is not None
        plug_inputs += [
            spec_decoding_generation_lengths, spec_decoding_packed_mask,
            spec_decoding_position_offsets, spec_decoding_use
        ]

    if long_rope_rotary_inv_freq is not None:
        assert long_rope_rotary_cos_sin is not None
        plug_inputs += [long_rope_rotary_inv_freq, long_rope_rotary_cos_sin]

    if mrope_rotary_cos_sin is not None:
        assert mrope_position_deltas is not None
        plug_inputs += [
            mrope_rotary_cos_sin,
            mrope_position_deltas,
        ]
    if host_runtime_perf_knobs is not None:
        plug_inputs += [host_runtime_perf_knobs]

    if host_context_progress is not None:
        plug_inputs += [host_context_progress]

    if is_mla_enabled_flag:
        assert q_b_proj is not None
        assert kv_b_proj is not None
        assert k_b_proj_trans is not None
        plug_inputs += [q_b_proj, kv_b_proj, k_b_proj_trans]

    if skip_attn is not None:
        plug_inputs += [skip_attn]

    if logn_scaling is not None:
        plug_inputs += [logn_scaling]

    for idx, i in enumerate(plug_inputs):
        assert i is not None, f"Found None input for {idx} th item in plugin inputs {plug_inputs}"

    plug_inputs = [i.trt_tensor for i in plug_inputs]
    layer = default_trtnet().add_plugin_v2(plug_inputs, attn_plug)
    _add_plugin_info(layer, attn_plg_creator, "causal_attn", pfc)
    output = _create_tensor(layer.get_output(0), layer)
    expected_outputs = 1

    # The output scaling factor tensor.
    output_sf = None
    if fuse_fp4_quant:
        output_sf = _create_tensor(layer.get_output(expected_outputs), layer)
        expected_outputs += 1

    present_key_value = None
    if use_cache and not paged_kv_cache_flag:
        present_key_value = _create_tensor(layer.get_output(expected_outputs),
                                           layer)
        assert present_key_value is not None
        expected_outputs += 1

    assert layer.num_outputs == expected_outputs, \
        f"Plugin outputs number mismatch with expected, got {layer.num_outputs}, expected {expected_outputs}"

    if kv_cache_quant_mode.has_int8_kv_cache(
    ) and not default_net().strongly_typed:
        if not paged_kv_cache_flag:
            # past key value
            layer.get_input(8).set_dynamic_range(-127, 127)
            # present key value
            layer.get_output(expected_outputs - 1).set_dynamic_range(-127, 127)
        else:
            layer.get_input(0).set_dynamic_range(-127, 127)
            layer.get_input(1).set_dynamic_range(-127, 127)
            layer.get_output(expected_outputs - 1).set_dynamic_range(-127, 127)

    assert output is not None
    if fuse_fp4_quant:
        assert output_sf is not None
        return (output, output_sf), present_key_value
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


def rms_norm(input: Tensor,
             normalized_shape: Union[int, Tuple[int]],
             num_groups: int = 1,
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

        num_groups: int = 1
            The group size.

        weight : Optional[Tensor] = None
            The 'gamma' term in layer-norm. Its shape must be
            'normalized_shape'.

        eps : float
            The epsilon term to be added to the variance in the squared-root.weig
    Returns:
        The output tensor of that operation.
    '''
    normalized_shape = [normalized_shape] if isinstance(
        normalized_shape, int) else normalized_shape

    dim = tuple([-i - 1 for i in range(len(normalized_shape))])

    if num_groups > 1:
        assert len(normalized_shape) == 1
        num_channels = input.size()[-1]
        ndim = input.ndim()
        old_shape = shape(input)
        new_shape = concat([input.size(i) for i in range(ndim - 1)] +
                           [num_groups, num_channels // num_groups])
        input = input.view(new_shape)

    with precision("float32"):
        input_dtype = input.dtype
        fp32_input = cast(input, "float32")
        varx = pow(fp32_input, 2.0)

        varx = varx.mean(dim=dim, keepdim=True)
        denom = varx + eps
        denom = denom.sqrt()
        fp32_y = fp32_input / denom
        y = cast(fp32_y, input_dtype)

    if num_groups > 1:
        y = y.view(old_shape)

    if weight is not None:
        y = y * weight

    return y


def rearrange(inputs: Union[Tensor, Sequence[Tensor]], expression: str,
              **kwargs) -> Tensor:
    '''
    Add a rearrange operation on a tensor.

    This operation is a reader-friendly smart element reordering for multidimensional tensors,
    including functionality of transpose (axes permutation), reshape (view), squeeze, unsqueeze,
    stack, concatenate and other operations. Please see: https://einops.rocks/api/rearrange/

    For example, if the shape of input tensor is [32, 30, 40, 3], and run:
        `rearrange(x, 'b (h h1) (w w1) c -> b h w 1 (c h1 w1) 1', h1=2, w1=2)`
    it would produce a tensor with shape as [32, 15, 20, 1, 12, 1].

    Parameters:
        input: Union[Tensor, Sequence[Tensor]]
            If it is a tensor, it will directly operate on it.
            Otherwise, if it is a sequence, it will concat it to a tensor and then
            operates on it.

        expression : str
            The expression about how to reorder the tensor in a reader-friendly way.

        kwargs:
            Keyword arguments to set some identifiers with specific values.

    Returns:
        The output tensor of this operation.
    '''
    import re

    def _init_expression(expr):
        expr_items = expr.split(" ")
        tmp_name_index = 0
        for idx, item in enumerate(expr_items):
            values = re.findall(r'\b\d+\b', item)
            if len(values) > 0:
                prefix = "(" if "(" in item else ""
                subfix = ")" if ")" in item else ""
                expr_items[
                    idx] = f"{prefix}NumericId{tmp_name_index}Val{values[0]}{subfix}"
                tmp_name_index += 1
        return " ".join(expr_items)

    def _get_all_identifier(expr):
        return re.findall(r'\b[a-zA-Z_]+\d*\b', expr)

    def _get_all_symbols(expr):
        return re.findall(r'\b\w+\b', expr)

    def _get_dim_expr(expr):
        return [
            _get_all_symbols(match.group())
            for match in re.finditer(r'\b\w+\b|\(.*?\)', expr)
        ]

    src_shape_expr, _, dst_shape_expr = expression.partition("->")
    unknown_identifiers = re.findall(r'[^a-zA-Z0-9_\(\)]',
                                     src_shape_expr + dst_shape_expr)
    assert len(
        unknown_identifiers) > 0, f"Unknown identifiers: {unknown_identifiers}"
    src_identifiers = _get_all_identifier(src_shape_expr)
    dst_identifiers = _get_all_identifier(dst_shape_expr)
    assert (len(src_identifiers) == len(set(src_identifiers))
            and len(dst_identifiers) == len(set(dst_identifiers))
            ), "Indexing expression contains duplicate dimension."
    assert (set(src_identifiers) == set(dst_identifiers)
            ), "Identifiers only on one side of expression (should be on both)."

    new_expression = _init_expression(expression)
    src_shape_expr, _, dst_shape_expr = new_expression.partition("->")

    # concat if inputs are sequence of tensors
    if isinstance(inputs, Sequence):
        inputs = concat([unsqueeze(t, 0) for t in inputs], dim=0)
    assert (
        inputs.ndim() == len(_get_dim_expr(src_shape_expr))
    ), f"inputs.ndim() is {inputs.ndim()} while indexing expression has {len(_get_dim_expr(src_shape_expr))}"

    src_symbols = _get_all_symbols(src_shape_expr)
    dst_symbols = _get_all_symbols(dst_shape_expr)

    # find all the symbols-values mapping and store them in symbol_map
    symbol_map = {
        symbol: {
            "updated": False,
            "value": None
        }
        for symbol in set(src_symbols + dst_symbols)
    }
    for symbol in symbol_map:
        if "NumericId" in symbol:
            symbol_map[symbol]["value"] = int(symbol.partition("Val")[-1])
            symbol_map[symbol]["updated"] = True
    for symbol, value in kwargs.items():
        symbol_map[symbol]["value"] = value
        symbol_map[symbol]["updated"] = True

    for idx, dim_expr in enumerate(_get_dim_expr(src_shape_expr)):
        if len(dim_expr) == 1:
            symbol = dim_expr[0]
            if not symbol_map[symbol]["updated"]:
                symbol_map[symbol]["value"] = shape(inputs, idx)
                symbol_map[symbol]["updated"] = True
        else:
            divisors = []
            unknown_symbol = None
            for symbol in dim_expr:
                if not symbol_map[symbol]["updated"]:
                    unknown_symbol = symbol
                else:
                    divisors.append(symbol_map[symbol]["value"])
            if unknown_symbol is not None:
                assert len(divisors) > 0
                divisor = prod(cast(concat(divisors), "int64"), dim=-1)
                symbol_map[unknown_symbol]["value"] = shape(inputs,
                                                            idx) / divisor
                symbol_map[unknown_symbol]["updated"] = True

    for symbol, item in symbol_map.items():
        assert (item["updated"]
                ), f"{symbol} cannot be inferred, please set it manually"

    dst_dims = []
    for dim_expr in _get_dim_expr(dst_shape_expr):
        if len(dim_expr) == 1:
            dst_dims.append(symbol_map[dim_expr[0]]["value"])
        else:
            accumulator = prod(cast(
                concat([symbol_map[symbol]["value"] for symbol in dim_expr]),
                "int64"),
                               dim=-1)
            dst_dims.append(accumulator)
    dst_dims = cast(concat(dst_dims, dim=-1), "int64")

    src_indices = {symbol: idx for idx, symbol in enumerate(src_identifiers)}
    permute_dims = [src_indices[symbol] for symbol in dst_identifiers]

    symbol_shape = cast(
        concat([symbol_map[symbol]["value"] for symbol in src_identifiers],
               dim=-1), "int64")
    tensor = inputs.view(symbol_shape)
    tensor = permute(tensor, permute_dims)
    tensor = tensor.view(dst_dims)
    return tensor


def repeat(input: Tensor, sizes: Sequence[int]) -> Tensor:
    '''
    Repeats the tensor along the specified dimensions.

    Parameters:
        input : Tensor
            The tensor to be repeated.
        sizes : Sequence[int]
            The number of times to repeat the tensor along each dimension.

    Returns:
        A tensor except for repeated input tensors along specified dim.

    '''
    assert input.ndim() <= len(sizes), \
        "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor"
    repeated_tensor = input
    for k in range(-1, -len(sizes) - 1, -1):
        repeated_tensor = concat([repeated_tensor] * sizes[k], dim=k)
    return repeated_tensor


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


def meshgrid2d(x: Tensor, y: Tensor) -> Tuple[Tensor]:
    '''
    Creates grids (2D) of coordinates specified by the 1D inputs (only supports `indexing=\'xy\'`).

    Parameters:
        x : Tensor
            The first input (1D) tensor.
        y : Tensor
            The second input (1D) tensor.

    Returns:
        The tuple of two tensors produced.

    TODO: Add full support for torch.meshgrid.
          See https://pytorch.org/docs/stable/generated/torch.meshgrid.html#torch-meshgrid
    '''
    if x.ndim() == 1:
        x = expand_dims(x, 0)
    if y.ndim() == 1:
        y = expand_dims(y, 0)
    grid_x = repeat_interleave(x, shape(y, 1),
                               1).view([x.shape[-1], y.shape[-1]])
    grid_y = repeat(y, (x.shape[-1], 1))
    return (grid_x, grid_y)


def generate_logn_scaling(seq_length: int = 8192,
                          max_position_embeddings: int = 32768) -> np.ndarray:
    '''
    Compute the Log-N scaling vector for Qwen inference extrapolation

    Parameters:
        seq_length : int
            The max seq length in training (default to 8192 in Qwen-1)
        max_position_embeddings : int
            The max position embeddings. (default to 32768 in Qwen-1)

    Returns:
        A constant np.ndarray that contains logn scaling vector
    '''
    logn_list = [
        math.log(i, seq_length) if i > seq_length else 1
        for i in range(1, max_position_embeddings + 1)
    ]
    return np.asarray(logn_list, dtype=np.float32)


def generate_alibi_slopes(num_heads: int,
                          tp_size: int = 1,
                          tp_rank: int = 0,
                          alibi_scale: float = 1.0,
                          alibi_bias_max: int = 8) -> np.ndarray:
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
                np.power(
                    2**(-(2**-(np.log2(closest_power_of_2) -
                               np.log2(alibi_bias_max)))), h_id + 1))
        else:
            slopes_ft.append(
                np.power(
                    2**(-(2**-(np.log2(closest_power_of_2 * 2) -
                               np.log2(alibi_bias_max)))),
                    (h_id - closest_power_of_2) * 2 + 1))
    slopes = np.asarray(slopes_ft, dtype=np.float32)

    slopes = alibi_scale * slopes
    slopes = slopes.reshape(1, (end_head_id - start_head_id), 1, 1)
    return slopes


def generate_alibi_biases(slopes: Tensor, key_length: Tensor) -> Tensor:
    '''
    Compute the ALiBi biases as described in https://arxiv.org/abs/2211.05100.

    The ALiBi biases are added to the result of the Q*K^T product in the
    multi-head attention block.

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
    multi-head attention block.

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
    mask = where(mask == 0, float('-inf'), 0.0)
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
    'gelu_pytorch_tanh': gelu,
    'openai-gelu': gelu,
    'geglu': geglu,
    'gegelu': gegelu,
    'identity': identity,
    'silu': silu,
    'softplus': softplus,
    'relu2': squared_relu,
    'squared-relu': squared_relu,
    'swiglu': swiglu,
    'fast-swiglu': swiglu,
    'sigmoid': sigmoid,
    'quick_gelu': quick_gelu,
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
    max_low_rank: int = 0,
    lora_ranks: List[Tensor] = None,
    lora_weights_pointers: List[Tensor] = None,
    weight_index: int = 0,
):
    '''
    Parameters:
        input : Tensor (On GPU)
            The input tensor. Its shape is [batch_size, seq_len, dim] or [num_tokens, dim] for remove_input_padding

        in_hidden_size/out_hidden_size : int
            the lora computation workflow is
            [M, in_hidden_size] -> [M, low_rank] -> [M, out_hidden_size]

        host_request_types : Tensor = None
            The tensor on the host that indicates if a request is in context or
            generation phase. Its shape is [batch_size]. See Inflight Batching
            in docs/source/advanced/gpt-attention.md,

        transa : bool
            Is the first input transposed? Set to 'True' if you want the first
            input to be transposed, 'False' otherwise.

        transb : bool
            Is the second input transposed? Set to 'True' if you want the
            second input to be transposed, 'False' otherwise.

        host_context_lengths: cpu Tensor = None
            A host tensor that contains the lengths of the different inputs,

        max_low_rank : int
            Maximum low_rank, used to determine the workspace size.

        lora_ranks : cpu Tensor with shape [batch_size]
            The low_rank of each request

        lora_weights_pointers : cpu int64 Tensor with shape [batch_size, 3]
            The weights pointers of each request. Consist of in_pointer, out_pointer and possibly a scales vector pointer.

        weight_index : int
            The index of weight if the weight pointer pointing to multiple weights.

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
    max_low_rank_field = trt.PluginField("max_low_rank",
                                         np.array(max_low_rank, dtype=np.int32),
                                         trt.PluginFieldType.INT32)
    weight_index_field = trt.PluginField("weight_index",
                                         np.array(weight_index, dtype=np.int32),
                                         trt.PluginFieldType.INT32)
    num_lora_modules = len(out_hidden_sizes)
    num_lora_modules_field = trt.PluginField(
        "num_lora_modules", np.array(num_lora_modules, dtype=np.int32),
        trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([
        in_hidden_size_field, transa, transb, num_lora_modules_field, pf_type,
        remove_input_padding, max_low_rank_field, weight_index_field
    ] + out_hidden_size_field_list)
    lora_plug = plg_creator.create_plugin("lora", pfc)

    plug_inputs = [input.cast(p_dtype), host_request_types
                   ] + lora_ranks + lora_weights_pointers

    if default_net().plugin_config.remove_input_padding:
        plug_inputs += [host_context_lengths]

    plug_inputs = [i.trt_tensor for i in plug_inputs]
    layer = default_trtnet().add_plugin_v2(plug_inputs, lora_plug)

    if num_lora_modules == 1:
        return _create_tensor(layer.get_output(0), layer).cast(input.dtype)
    else:
        return [
            _create_tensor(layer.get_output(i), layer).cast(input.dtype)
            for i in range(num_lora_modules)
        ]


def dora_plugin(activations: Tensor,
                out_hidden_sizes: list[int],
                lora_weights_pointers: list[Tensor],
                host_request_types: Tensor,
                host_context_lengths: Tensor | None = None) -> Tensor:
    '''
    The DoRA plugin applies column-wise scaling to the output of a LoRA layer.

    Parameters:
        input : Tensor (On GPU)
            The input tensor. Its shape is [batch_size, seq_len, dim] or [num_tokens, dim] for remove_input_padding

        out_hidden_sizes : list[int]
            The output hidden size of each adapter in the related LoRA module.
            For example, for a qkv projection out_hidden_sizes should be [q_dim, k_dim, v_dim].

        host_request_types : Tensor = None
            The tensor on the host that indicates if a request is in context or
            generation phase. Its shape is [batch_size]. See Inflight Batching
            in docs/source/advanced/gpt-attention.md,

        host_context_lengths: cpu Tensor = None
            A host tensor that contains the lengths of the different inputs,

    Return:
        The tensor produced by that layer.

    '''
    assert host_context_lengths is not None or not default_net(
    ).plugin_config.remove_input_padding

    dora_plg_creator = trt.get_plugin_registry().get_creator(
        'Dora', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert dora_plg_creator is not None

    out_hidden_sizes = trt.PluginField(
        f"out_hidden_sizes", np.array(out_hidden_sizes, dtype=np.int32),
        trt.PluginFieldType.INT32)

    remove_input_padding = trt.PluginField(
        "remove_input_padding",
        np.array(np.int8(default_net().plugin_config.remove_input_padding),
                 dtype=np.int8), trt.PluginFieldType.INT8)

    lora_dtype = default_net().plugin_config.lora_plugin
    type_id = trt.PluginField(
        "type", np.array(int(str_dtype_to_trt(lora_dtype)), np.int32),
        trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection(
        [type_id, remove_input_padding, out_hidden_sizes])

    dora_plug = dora_plg_creator.create_plugin("dora", pfc,
                                               trt.TensorRTPhase.BUILD)

    plug_inputs = [activations.cast(lora_dtype), host_request_types
                   ] + lora_weights_pointers

    if default_net().plugin_config.remove_input_padding:
        plug_inputs += [host_context_lengths]

    plug_inputs = [i.trt_tensor for i in plug_inputs]

    layer = default_trtnet().add_plugin_v3(plug_inputs, [], dora_plug)
    _add_plugin_info(layer, dora_plg_creator, "dora", pfc)
    output = _create_tensor(layer.get_output(0), layer).cast(activations.dtype)
    return output


def mamba_conv1d(input: Tensor,
                 conv_state_or_ptr: Tensor,
                 conv_weight: Tensor,
                 conv_bias: Tensor,
                 host_request_types: Tensor,
                 last_token_ids: Tensor,
                 dim: int,
                 dconv: int,
                 dtype: str,
                 pre_stride: int = 0,
                 post_stride: int = 0,
                 host_context_lengths: Optional[Tensor] = None,
                 slot_mapping: Optional[Tensor] = None,
                 apply_silu: bool = True):
    '''
    Parameters:
        input : Tensor (On GPU)
            The input tensor. Its shape is [batch_size, seq_len, dim] or [num_tokens, dim] for remove_input_padding

        conv_state_or_ptr : Tensor (On GPU or CPU)
            The conv state tensor. Its shape is [batch_size, dconv - 1, dim]
            Or the CPU tensor of shape [1] for the pointer of paged states.

        conv_weight : Tensor (On GPU)
            The weight tensor. Its shape is [1, dconv, dim]

        conv_bias : Tensor (On GPU)
            The bias tensor. Its shape is [dim]

        host_request_types : Tensor (On CPU)
            The tensor on the host that indicates if a request is in context or
            generation phase. Its shape is [batch_size]. See Inflight Batching
            in docs/source/advanced/gpt-attention.md,

        last_token_ids : Tensor (On GPU)
            The inclusive prefix-sum of the lengths or the lengths of the
            sequences in the batch.

        dim : int
            The hidden dimension of conv1d

        dconv : int
            The window size of conv1d

        dtype: str
            data type

        pre_stride : int = 0
            The (pre) stride size of the input tensor.
            The valid values of the input tensor are input[..., pre_stride: dim-post_stride]

        post_stride : int = 0
            The (post) stride size of the input tensor.
            The valid values of the input tensor are input[..., pre_stride: dim-post_stride]

        host_context_lengths: Tensor (On CPU) (Optional)
            A host tensor that contains the lengths of the different inputs,

        slot_mapping: Tensor (On GPU) (Optional)
            Real page index in state. Its shape is [dim], used for paged state, each page shape is [dconv, dim]

        apply_silu: bool
            Is there a SiLU operation after the conv1d? When True apply
            SiLU activation function after the conv1d.
    '''
    assert host_request_types is not None
    mamba_conv1d_plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'MambaConv1d', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert mamba_conv1d_plg_creator is not None

    dim = trt.PluginField("dim", np.array(dim, dtype=np.int32),
                          trt.PluginFieldType.INT32)
    dconv = trt.PluginField("dconv", np.array(dconv, dtype=np.int32),
                            trt.PluginFieldType.INT32)
    pre_stride = trt.PluginField("pre_stride",
                                 np.array(pre_stride, dtype=np.int32),
                                 trt.PluginFieldType.INT32)
    post_stride = trt.PluginField("post_stride",
                                  np.array(post_stride, dtype=np.int32),
                                  trt.PluginFieldType.INT32)
    pf_type = trt.PluginField(
        "type_id", np.array([int(str_dtype_to_trt(dtype))], np.int32),
        trt.PluginFieldType.INT32)
    remove_input_padding = trt.PluginField(
        "remove_input_padding",
        np.array(np.int8(default_net().plugin_config.remove_input_padding),
                 dtype=np.int8), trt.PluginFieldType.INT8)
    paged_state = trt.PluginField(
        "paged_state",
        np.array(np.int8(default_net().plugin_config.paged_state),
                 dtype=np.int8), trt.PluginFieldType.INT8)
    apply_silu = trt.PluginField("apply_silu",
                                 np.array(np.int8(apply_silu), dtype=np.int8),
                                 trt.PluginFieldType.INT8)

    pfc = trt.PluginFieldCollection([
        dim, dconv, pre_stride, post_stride, pf_type, remove_input_padding,
        paged_state, apply_silu
    ])
    mamba_conv1d_plug = mamba_conv1d_plg_creator.create_plugin(
        "mamba_conv1d", pfc)
    plug_inputs = [
        input, conv_state_or_ptr, conv_weight, conv_bias, host_request_types,
        last_token_ids
    ]
    if default_net().plugin_config.remove_input_padding:
        plug_inputs += [host_context_lengths]
    if default_net().plugin_config.paged_state:
        plug_inputs += [slot_mapping]
    plug_inputs = [i.trt_tensor for i in plug_inputs]

    layer = default_trtnet().add_plugin_v2(plug_inputs, mamba_conv1d_plug)
    _add_plugin_info(layer, mamba_conv1d_plg_creator, "mamba_conv1d", pfc)
    output = _create_tensor(layer.get_output(0), layer)
    if default_net().plugin_config.paged_state:
        return output, None
    else:
        present_state = _create_tensor(layer.get_output(1), layer)
        return output, present_state


def selective_scan(input: Tensor,
                   state_or_ptr: Tensor,
                   delta: Tensor,
                   delta_bias: Tensor,
                   A: Tensor,
                   BC: Tensor,
                   D: Tensor,
                   host_request_types: Tensor,
                   last_token_ids: Tensor,
                   dim: int,
                   dstate: int,
                   dt_rank: int,
                   delta_softplus: bool,
                   dtype: str,
                   z: Optional[Tensor] = None,
                   host_context_lengths: Optional[Tensor] = None,
                   slot_mapping: Optional[Tensor] = None,
                   nheads: int = 1,
                   ngroups: int = 1,
                   chunk_size: int = 256,
                   mamba_version: str = 'Mamba1'):
    '''
    Parameters:
        input : Tensor (On GPU)
            The input tensor. Its shape is [batch_size, seq_len, dim]

        state_or_ptr : Tensor (On GPU or CPU)
            The ssm state tensor. Its shape is [batch_size, dstate, dim]
            Or the CPU tensor of shape [1] for the pointer of paged states.

        delta : Tensor (On GPU)
            The delta tensor.
            mamba: Its shape is [batch_size, seq_len, dim] or [num_tokens, dim] for remove_input_padding
            mamba2: Its shape is [batch_size, seq_len, nheads] or [num_tokens, nheads] for remove_input_padding

        delta_bias : Tensor (On GPU)
            The delta bias tensor.
            mamba: Its shape is [dim]
            mamba2: Its shape is [nheads]

        A : Tensor (On GPU)
            A matrix.
            mamba: Its shape is [dstate, dim]
            mamba2: Its shape is [nheads]

        BC : Tensor (On GPU)
            B and C matrix.
            mamba: Its shape is [batch_size, seq_len, dstate * 2] or [num_tokens, dstate * 2] for remove_input_padding
            mamba2: Its shape is [batch_size, seq_len, ngroups * dstate * 2] or [num_tokens, ngroups * dstate * 2] for remove_input_padding

        D : Tensor (On GPU)
            D matrix.
            mamba: Its shape is [dim]
            mamba2: Its shape is [nheads]

        host_request_types : Tensor (On CPU)
            The tensor on the host that indicates if a request is in context or
            generation phase. Its shape is [batch_size]. See Inflight Batching
            in docs/source/advanced/gpt-attention.md

        last_token_ids : Tensor (On GPU)
            The inclusive prefix-sum of the lengths or the lengths of the
            sequences in the batch.

        dim : int
            The inner dimension of SSM block

        dstate : int
            The state dimension of SSM block

        dt_rank: int
            The rank dimension of dt_proj

        delta_softplus : bool
            Do we apply softplus to the delta.

        dtype: str
            data type

        z : Tensor (On GPU) (Optional)
            The z tensor. Its shape is [batch_size, seq_len, dim] or [num_tokens, dim] for remove_input_padding

        host_context_lengths: Tensor (On CPU) (Optional)
            A host tensor that contains the lengths of the different inputs,

        slot_mapping: Tensor (On GPU) (Optional)
            Real page index in state. Its shape is [dim], used for paged state, each page shape is [dstate, dim]

        nheads: int (Optional)
            The number of heads.

        ngroups: int (Optional)
            The number of groups.

        chunk_size: int (Optional)
            The chunk_size is used for the chunk_scan kernel.

        mamba_version: int (Optional)
            Mamba version, support Mamba1 as default.
    '''
    assert host_request_types is not None
    selective_scan_plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'SelectiveScan', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert selective_scan_plg_creator is not None

    dim = trt.PluginField("dim", np.array(dim, dtype=np.int32),
                          trt.PluginFieldType.INT32)
    dstate = trt.PluginField("dstate", np.array(dstate, dtype=np.int32),
                             trt.PluginFieldType.INT32)
    dt_rank = trt.PluginField("dt_rank", np.array(dt_rank, dtype=np.int32),
                              trt.PluginFieldType.INT32)
    nheads = trt.PluginField("nheads", np.array(nheads, dtype=np.int32),
                             trt.PluginFieldType.INT32)
    ngroups = trt.PluginField("ngroups", np.array(ngroups, dtype=np.int32),
                              trt.PluginFieldType.INT32)
    chunk_size = trt.PluginField("chunk_size",
                                 np.array(chunk_size, dtype=np.int32),
                                 trt.PluginFieldType.INT32)
    delta_softplus = trt.PluginField(
        "delta_softplus", np.array(np.int8(delta_softplus), dtype=np.int8),
        trt.PluginFieldType.INT8)
    pf_type = trt.PluginField(
        "type_id", np.array([int(str_dtype_to_trt(dtype))], np.int32),
        trt.PluginFieldType.INT32)
    remove_input_padding = trt.PluginField(
        "remove_input_padding",
        np.array(np.int8(default_net().plugin_config.remove_input_padding),
                 dtype=np.int8), trt.PluginFieldType.INT8)
    paged_state = trt.PluginField(
        "paged_state",
        np.array(np.int8(default_net().plugin_config.paged_state),
                 dtype=np.int8), trt.PluginFieldType.INT8)
    if z is None:
        z_enabled = trt.PluginField("z_enabled", np.array(0, dtype=np.int8),
                                    trt.PluginFieldType.INT8)
    else:
        z_enabled = trt.PluginField("z_enabled", np.array(1, dtype=np.int8),
                                    trt.PluginFieldType.INT8)
    is_mamba2 = trt.PluginField(
        "is_mamba2",
        np.array(1 if mamba_version == 'Mamba2' else 0, dtype=np.int8),
        trt.PluginFieldType.INT8)

    pfc = trt.PluginFieldCollection([
        dim, dstate, dt_rank, nheads, ngroups, chunk_size, delta_softplus,
        pf_type, remove_input_padding, paged_state, z_enabled, is_mamba2
    ])
    selective_scan_plug = selective_scan_plg_creator.create_plugin(
        "selective_scan", pfc)

    plug_inputs = [
        input, state_or_ptr, delta, delta_bias, A, BC, D, host_request_types,
        last_token_ids
    ]
    if default_net().plugin_config.remove_input_padding:
        plug_inputs += [host_context_lengths]
    if default_net().plugin_config.paged_state:
        plug_inputs += [slot_mapping]
    if z is not None:
        plug_inputs += [z]
    plug_inputs = [i.trt_tensor for i in plug_inputs]

    layer = default_trtnet().add_plugin_v2(plug_inputs, selective_scan_plug)
    _add_plugin_info(layer, selective_scan_plg_creator, "selective_scan", pfc)
    output = _create_tensor(layer.get_output(0), layer)
    if default_net().plugin_config.paged_state:
        return output, None
    else:
        present_state = _create_tensor(layer.get_output(1), layer)
        return output, present_state


def rg_lru(input: Tensor,
           A: Tensor,
           state_or_ptr: Tensor,
           host_request_types: Tensor,
           last_token_ids: Tensor,
           dim: int,
           dtype: str,
           block_size: int = 0,
           y: Optional[Tensor] = None,
           y_bias: Optional[Tensor] = None,
           gate: Optional[Tensor] = None,
           gate_bias: Optional[Tensor] = None,
           gate_x: Optional[Tensor] = None,
           gate_x_bias: Optional[Tensor] = None,
           gate_a: Optional[Tensor] = None,
           gate_a_bias: Optional[Tensor] = None,
           slot_mapping: Optional[Tensor] = None):
    '''
    Parameters:
        input : Tensor (On GPU)
            The input tensor. Its shape is [batch_size, seq_len, dim]

        A : Tensor (On GPU)
            A matrix. Its shape is [dim]

        state_or_ptr : Tensor (On GPU or CPU)
            The lru state tensor. Its shape is [batch_size, dstate, dim]
            Or the CPU tensor of shape [1] for the pointer of paged states.

        host_request_types : Tensor (On CPU)
            The tensor on the host that indicates if a request is in context or
            generation phase. Its shape is [batch_size]. See Inflight Batching
            in docs/source/advanced/gpt-attention.md,

        last_token_ids : Tensor (On GPU)
            The inclusive prefix-sum of the lengths or the lengths of the
            sequences in the batch.

        dim : int
            The inner dimension of RG_LRU block

        block_size : int
            The block size of the block diagonal linear layer. It is used to
            support the cases that enable fused gate.

        dtype: str
            data type

        y : Tensor (On GPU) (Optional)
            The y tensor. Its shape is [batch_size, seq_len, dim]

        y_bias : Tensor (On GPU) (Optional)
            The y_bias tensor. Its shape is [dim]. If y_bias is not None, we
            will fuse GELU(y + y_bias) in this function.

        gate : Tensor (On GPU) (Optional)
            The gate tensor. Its shape is [batch_size, seq_len, 2 * dim].
            If gate is not None, we will fuse the gate_x and gate_a, otherwise
            use those two tensors.

        gate_bias : Tensor (On GPU) (Optional)
            The gate_bias tensor. Its shape is [2 * block_num, dim // block_num].
            If gate_bias is not None, we will fuse the bias add in this function.

        gate_x : Tensor (On GPU) (Optional)
            The gate_x tensor. Its shape is [batch_size, seq_len, dim]

        gate_x_bias : Tensor (On GPU) (Optional)
            The gate_x_bias tensor. Its shape is [block_num, dim // block_num].
            If gate_x_bias is not None, we will fuse the bias add in this function.

        gate_a : Tensor (On GPU) (Optional)
            The gate_a tensor. Its shape is [batch_size, seq_len, dim]

        gate_a_bias : Tensor (On GPU) (Optional)
            The gate_a_bias tensor. Its shape is [block_num, dim // block_num].
            If gate_a_bias is not None, we will fuse the bias add in this function.

        slot_mapping: Tensor (On GPU) (Optional)
            Real page index in state. Its shape is [dim], used for paged state, each page shape is [dstate, dim]
    '''
    assert host_request_types is not None
    lru_plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'LRU', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert lru_plg_creator is not None
    assert (gate_x_bias is None) == (gate_a_bias is None)
    enable_fuse_gate = gate is not None
    has_gate_bias = (gate_bias is not None) or (gate_x_bias is not None)
    if enable_fuse_gate:
        assert gate is not None
        assert block_size > 0
        if has_gate_bias:
            assert gate_bias is not None
    else:
        assert gate_x is not None and gate_a is not None
        if has_gate_bias:
            assert gate_x_bias is not None and gate_a_bias is not None

    dim = trt.PluginField("dim", np.array(dim, dtype=np.int32),
                          trt.PluginFieldType.INT32)
    block_size = trt.PluginField("block_size",
                                 np.array(block_size, dtype=np.int32),
                                 trt.PluginFieldType.INT32)
    pf_type = trt.PluginField(
        "type_id", np.array([int(str_dtype_to_trt(dtype))], np.int32),
        trt.PluginFieldType.INT32)
    remove_input_padding = trt.PluginField(
        "remove_input_padding",
        np.array(np.int8(default_net().plugin_config.remove_input_padding),
                 dtype=np.int8), trt.PluginFieldType.INT8)
    paged_state = trt.PluginField(
        "paged_state",
        np.array(np.int8(default_net().plugin_config.paged_state),
                 dtype=np.int8), trt.PluginFieldType.INT8)

    if y is None:
        y_enabled = trt.PluginField("y_enabled", np.array(0, dtype=np.int8),
                                    trt.PluginFieldType.INT8)
    else:
        y_enabled = trt.PluginField("y_enabled", np.array(1, dtype=np.int8),
                                    trt.PluginFieldType.INT8)

    if y_bias is None:
        y_bias_enabled = trt.PluginField("y_bias_enabled",
                                         np.array(0, dtype=np.int8),
                                         trt.PluginFieldType.INT8)
    else:
        y_bias_enabled = trt.PluginField("y_bias_enabled",
                                         np.array(1, dtype=np.int8),
                                         trt.PluginFieldType.INT8)

    if enable_fuse_gate:
        fuse_gate_enabled = trt.PluginField("fuse_gate_enabled",
                                            np.array(1, dtype=np.int8),
                                            trt.PluginFieldType.INT8)
    else:
        fuse_gate_enabled = trt.PluginField("fuse_gate_enabled",
                                            np.array(0, dtype=np.int8),
                                            trt.PluginFieldType.INT8)

    if has_gate_bias:
        gate_bias_enabled = trt.PluginField("gate_bias_enabled",
                                            np.array(1, dtype=np.int8),
                                            trt.PluginFieldType.INT8)
    else:
        gate_bias_enabled = trt.PluginField("gate_bias_enabled",
                                            np.array(0, dtype=np.int8),
                                            trt.PluginFieldType.INT8)

    pfc = trt.PluginFieldCollection([
        dim, block_size, pf_type, remove_input_padding, paged_state, y_enabled,
        y_bias_enabled, fuse_gate_enabled, gate_bias_enabled
    ])
    lru_plug = lru_plg_creator.create_plugin("rg_lru", pfc)

    plug_inputs = [
        input,
        A,
        state_or_ptr,
        host_request_types,
        last_token_ids,
    ]
    if default_net().plugin_config.paged_state:
        plug_inputs += [slot_mapping]
    if y is not None:
        plug_inputs += [y]
        if y_bias is not None:
            plug_inputs += [y_bias]
    if enable_fuse_gate:
        plug_inputs += [gate]
        if has_gate_bias:
            plug_inputs += [gate_bias]
    else:
        plug_inputs += [gate_x, gate_a]
        if has_gate_bias:
            plug_inputs += [gate_x_bias, gate_a_bias]
    plug_inputs = [i.trt_tensor for i in plug_inputs]

    layer = default_trtnet().add_plugin_v2(plug_inputs, lru_plug)
    _add_plugin_info(layer, lru_plg_creator, "rg_lru", pfc)
    output = _create_tensor(layer.get_output(0), layer)
    if default_net().plugin_config.paged_state:
        return output, None
    else:
        present_state = _create_tensor(layer.get_output(1), layer)
        return output, present_state


def topk(input: Tensor,
         k: Union[Tensor, int],
         dim: int,
         largest: bool = True,
         prefer_plugin: bool = True) -> Tuple[Tensor, Tensor]:
    '''
    Add an topk operation.

    As explained in the ONNX documentation,

        https://github.com/onnx/onnx/blob/main/docs/Operators.md#topk

    NOTE: One distinction from the ONNX topk op, the output is always sorted
    with TensorRT layer.

    Retrieve the top-K largest elements along a specified axis.
    Given an input tensor of shape [a_1, a_2, ..., a_n, r]
    and integer argument k, return two outputs:
    Value tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] which contains the values of the top k elements along the specified axis
    Index tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] which contains the indices of the top k elements (original indices from the input tensor).

    Parameters:
        input : Tensor
            The input tensor.

        k : int
            A single positive value corresponding to the number of top elements to retrieve

        dim: int
            The dimension in which to compute the topk indices.

        largest: bool
            Controls whether to return largest or smallest elements

        prefer_plugin : bool
            Whether to use the topkLastDim plugin if dim is last dim and k is static.


    Returns:
        The tensors (values, indices) produced by this topk operation.
    '''
    dim = dim_resolve_negative(dim, input.ndim())[0]
    if prefer_plugin and dim == input.ndim() - 1 and not isinstance(k, Tensor):
        last_dim = input.size(-1)
        if last_dim == -1:  # dynamic?
            last_dim = shape(input, -1)
        # since we might need to flatten the input to 2d tensor,
        # we need to prepare the output shape
        out_shape = []
        for i in range(input.ndim() - 1):
            out_shape.append(shape(input, i))
        out_shape = concat(out_shape + [k])
        if input.ndim() == 1:
            input_2d = unsqueeze(input,
                                 0)  # special handling of rank-1 dynamic tensor
        elif input.ndim() != 2:
            input_2d = input.view(concat([-1, last_dim]),
                                  zero_is_placeholder=False)
        else:
            input_2d = input
        plg_creator = trt.get_plugin_registry().get_plugin_creator(
            "TopkLastDim", "1", TRT_LLM_PLUGIN_NAMESPACE)
        assert plg_creator is not None
        is_largest = trt.PluginField(
            "is_largest", np.array(1 if largest else 0, dtype=np.int32),
            trt.PluginFieldType.INT32)
        k = trt.PluginField("k", np.array(k, dtype=np.int32),
                            trt.PluginFieldType.INT32)
        pf_type = trt.PluginField("type_id",
                                  np.array([int(input_2d.dtype)], np.int32),
                                  trt.PluginFieldType.INT32)
        pfc = trt.PluginFieldCollection([pf_type, k, is_largest])
        topk_last_dim_plug = plg_creator.create_plugin("topk_last_dim", pfc)
        plug_inputs = [input_2d]
        plug_inputs = [i.trt_tensor for i in plug_inputs]
        layer = default_trtnet().add_plugin_v2(plug_inputs, topk_last_dim_plug)
        _add_plugin_info(layer, plg_creator, "topk_last_dim", pfc)
        values = _create_tensor(layer.get_output(0), layer)
        indices = _create_tensor(layer.get_output(1), layer)
        values = values.view(out_shape, zero_is_placeholder=False)
        indices = indices.view(out_shape, zero_is_placeholder=False)
    else:
        # non-plugin path
        axes = dim_to_trt_axes(dim)
        layer = default_trtnet().add_topk(
            input.trt_tensor,
            trt.TopKOperation.MAX if largest else trt.TopKOperation.MIN,
            k=k if not isinstance(k, Tensor) else 1,
            axes=axes)
        if isinstance(k, Tensor):
            if k.ndim() == 1:
                k = squeeze(k, 0)
            layer.set_input(1, k.trt_tensor)
        values = _create_tensor(layer.get_output(0), layer)
        indices = _create_tensor(layer.get_output(1), layer)

    return values, indices


def scatter_nd(input: Tensor, mask: Tensor, source: Tensor) -> Tensor:
    '''
    Scatter_nd is a tensor operation that writes or updates values in a tensor based on indices.

    Parameters:
        input: Tensor
            The input tensor to be updated
        mask: Tensor
            A tensor of indices specifying the locations in data to be updated.
        source: Tensor
            A tensor of values to be written or scattered into data.
    Returns:
        New tensor with the same shape as the input tensor data,
        where the values from the source tensor are scattered or written into the output tensor
        at the locations specified by the mask tensor.
    '''
    scatter_layer = default_trtnet().add_scatter(input.trt_tensor,
                                                 mask.trt_tensor,
                                                 source.trt_tensor,
                                                 mode=trt.ScatterMode.ND)
    return _create_tensor(scatter_layer.get_output(0), scatter_layer)


def low_latency_gemm(input: Tensor,
                     mat2: Tensor,
                     alpha: Optional[np.ndarray] = None,
                     strict_dtype: Optional[trt.DataType] = None) -> Tensor:
    if not default_net().plugin_config.low_latency_gemm_plugin:
        raise RuntimeError("Low Latency GEMM is only support with plugin")
    elif default_net().plugin_config.low_latency_gemm_plugin != "fp8":
        raise RuntimeError("Low Latency GEMM plugin only support fp8")
    else:
        plg_creator = trt.get_plugin_registry().get_plugin_creator(
            "LowLatencyGemm", "1", TRT_LLM_PLUGIN_NAMESPACE)
        assert plg_creator is not None
        if ((input.dtype != trt.fp8) or ((mat2.dtype) != trt.fp8)):
            raise TypeError("Low Latency GEMM only support fp8 input")
        if (alpha):
            assert (isinstance(alpha, np.ndarray) and alpha.dtype == np.float32
                    and alpha.size
                    == 1), "`alpha` must be passed as a float32 ndarray"
        alpha = alpha if alpha else np.array(1.0, dtype=np.float32)
        alpha = trt.PluginField("alpha", alpha.flatten(),
                                trt.PluginFieldType.FLOAT32)

        if strict_dtype is not None:
            assert isinstance(strict_dtype, trt.DataType)
            p_dtype = strict_dtype
            if (p_dtype not in [trt.float32, trt.float16, trt.bfloat16]):
                raise ValueError(
                    "strict_dtype must be float32, float16 or bfloat16 in low latency gemm plugin"
                )
        else:
            raise RuntimeError(
                "need to use strict dtype in  low latency gemm plugin fp8")
        pf_type = trt.PluginField("type_id", np.array([int(p_dtype)], np.int32),
                                  trt.PluginFieldType.INT32)
        pfc = trt.PluginFieldCollection([alpha, pf_type])
        low_latency_gemm_plug = plg_creator.create_plugin(
            "low_latency_gemm", pfc)
        plug_inputs = [input.trt_tensor, mat2.trt_tensor]
        layer = default_trtnet().add_plugin_v2(plug_inputs,
                                               low_latency_gemm_plug)
        _add_plugin_info(layer, plg_creator, "low_latency_gemm", pfc)
        return _create_tensor(layer.get_output(0), layer)


class SideStreamIDType(IntEnum):
    disable = 0
    moe = 1


def low_latency_gemm_swiglu(input: Tensor,
                            weight: Tensor,
                            scale_d0: float = 1.0,
                            scale_d1: float = 1.0,
                            scale_output: float = 1.0) -> Tensor:
    '''
    Add a matrix multiplication, followed by SwiGLU (`x * SiLU(gate)`) operation.

    The second SwiGLU operation takes the preceding tensor, splits it into two halves
    along the last dimension, applies SiLU to the second half and multiply the results. The
    behaviour is undefined if the last dimension is not even.

        Parameters:
        input : Tensor
            The first tensor (often called A).

        weight : Tensor
            The second tensor (often called B).

        scale_d0 : float
            The scale for dequantizing x, used for fp8

        scale_d1 : float
            The scale for dequantizing gate, used for fp8

        scale_output : float
            The scale for quantizing output, used for fp8

                Returns:
        The tensor produced by the inserted layer.
    '''
    plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'LowLatencyGemmSwiglu', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plg_creator is not None

    p_dtype = default_net().plugin_config.low_latency_gemm_swiglu_plugin
    pf_type = trt.PluginField(
        "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
        trt.PluginFieldType.INT32)
    pf_scale_d0 = trt.PluginField("scale_d0",
                                  np.array(scale_d0, dtype=np.float32),
                                  trt.PluginFieldType.FLOAT32)
    pf_scale_d1 = trt.PluginField("scale_d1",
                                  np.array(scale_d1, dtype=np.float32),
                                  trt.PluginFieldType.FLOAT32)
    pf_scale_output = trt.PluginField("scale_output",
                                      np.array(scale_output, dtype=np.float32),
                                      trt.PluginFieldType.FLOAT32)

    pfc = trt.PluginFieldCollection(
        [pf_type, pf_scale_output, pf_scale_d0, pf_scale_d1])
    low_latency_gemm_swiglu_plug = plg_creator.create_plugin(
        "low_latency_gemm_swiglu", pfc)

    plug_inputs = [input.trt_tensor, weight.trt_tensor]

    layer = default_trtnet().add_plugin_v2(plug_inputs,
                                           low_latency_gemm_swiglu_plug)

    return _create_tensor(layer.get_output(0), layer)


def cuda_stream_sync(input_list: List[Tensor],
                     side_stream_id: SideStreamIDType) -> Tensor:
    '''
    Wait for the side stream on the main stream.
    output = input_list[0]

    Parameters:
        input_list : List[Tensor] (On GPU)
            The list of input tensors.
        side_stream_id : int (On CPU)
            The side stream ID.
    '''
    plg_creator = trt.get_plugin_registry().get_plugin_creator(
        "CudaStream", "1", TRT_LLM_PLUGIN_NAMESPACE)
    assert plg_creator is not None

    p_side_stream_id = trt.PluginField("side_stream_id",
                                       np.array(side_stream_id, dtype=np.int32),
                                       trt.PluginFieldType.INT32)
    p_num_inputs = trt.PluginField("num_inputs",
                                   np.array(len(input_list), dtype=np.int32),
                                   trt.PluginFieldType.INT32)
    pf_type = trt.PluginField(
        "type_id", np.array([int(input_list[0].dtype)], dtype=np.int32),
        trt.PluginFieldType.INT32)
    pfc = trt.PluginFieldCollection([p_side_stream_id, p_num_inputs, pf_type])
    plug = plg_creator.create_plugin("cuda_stream", pfc)
    plug_inputs = [input.trt_tensor for input in input_list]

    layer = default_trtnet().add_plugin_v2(plug_inputs, plug)
    _add_plugin_info(layer, plg_creator, "cuda_stream", pfc)
    output = _create_tensor(layer.get_output(0), layer)
    return output


def cp_split_plugin(
    input_ids: Tensor,
    host_request_types: Tensor,
    host_context_lengths: Tensor,  # for pad-free input mode
    cp_size: int = 1,
    cp_rank: int = 0,
) -> Tensor:
    '''
    Add an operation to perform splitting for context parallelism.

    This operation split the input_ids into cp_size chunks, and return the cp_rank-th
    chunk.
    When the seqlen % cp_size != 0, the chunk sizes of each rank would be
    [seqlen // cp_size, seqlen // cp_size, ..., seqlen - (seqlen // cp_size) * cp_size]

    It inserts a IPluginV3Layer.

    Parameters:
        input : Tensor
            The input tensor contains the indices to split.

        host_request_types: Tensor = None (On CPU)
            The tensor on the host that indicates if a request is in context or
            generation phase. Its shape is [batch_size]. See Inflight Batching
            in docs/gpt_attention.md,

        host_context_lengths: Tensor = None (On CPU)
            A host tensor that contains the lengths of the different inputs

    Returns:
        The output split tensor.
        The length of the output split tensor.
        The index for rebuilding the sequence
    '''
    plg_creator = trt.get_plugin_registry().get_creator(
        'CpSplit', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plg_creator is not None

    cp_size = trt.PluginField("cp_size", np.array([int(cp_size)], np.int32),
                              trt.PluginFieldType.INT32)
    cp_rank = trt.PluginField("cp_rank", np.array([int(cp_rank)], np.int32),
                              trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([cp_size, cp_rank])
    cp_split_plug = plg_creator.create_plugin("cp_split", pfc,
                                              trt.TensorRTPhase.BUILD)
    plug_inputs = [
        input_ids.trt_tensor, host_request_types.trt_tensor,
        host_context_lengths.trt_tensor
    ]

    layer = default_trtnet().add_plugin_v3(plug_inputs, [], cp_split_plug)
    _add_plugin_info(layer, plg_creator, "cp_split", pfc)
    return _create_tensor(layer.get_output(0),
                          layer), _create_tensor(layer.get_output(2), layer)
