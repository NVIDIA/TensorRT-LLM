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
import ctypes
from collections import OrderedDict
from pathlib import Path
from typing import List

import numpy as np
import tensorrt as trt

from tensorrt_llm._common import default_trtnet
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.functional import Tensor, _create_tensor
from tensorrt_llm.module import Module

TRT_LLM_PLUGIN_NAMESPACE = 'tensorrt_llm'
LAYER_NAME = 'TritonFlashAttentionLayer'
FMHA_KERNEL_BLOCK_SIZE = 128


def _load_triton_plugin_lib():
    triton_plugin_dir = Path(__file__).parent.absolute()
    plugin_lib = triton_plugin_dir / 'build/libtrt_llm_custom_plugins.so'
    handle = ctypes.CDLL(plugin_lib, mode=ctypes.RTLD_GLOBAL)
    if handle is None:
        raise ImportError('TensorRT-LLM Triton Plugin is unavailable')
    handle.initOpenAiTritonPlugins.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    handle.initOpenAiTritonPlugins.restype = ctypes.c_bool
    assert handle.initOpenAiTritonPlugins(
        None, TRT_LLM_PLUGIN_NAMESPACE.encode('utf-8'))


_load_triton_plugin_lib()


def flash_attention_op(num_heads: int, head_size: int, softmax_scale: float,
                       inputs: List[trt.ITensor]) -> Tensor:
    # Create a plugin instance.
    plugin_creator = trt.get_plugin_registry().get_plugin_creator(
        'TritonFlashAttention', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plugin_creator is not None

    pfc = trt.PluginFieldCollection([
        trt.PluginField("num_heads", np.array([num_heads], np.int32),
                        trt.PluginFieldType.INT32),
        trt.PluginField("head_size", np.array([head_size], np.int32),
                        trt.PluginFieldType.INT32),
        trt.PluginField("softmax_scale", np.array([softmax_scale], np.float32),
                        trt.PluginFieldType.FLOAT32),
        trt.PluginField("type_id", np.array([int(inputs[0].dtype)], np.int32),
                        trt.PluginFieldType.INT32)
    ])
    plugin = plugin_creator.create_plugin("flash_attention", pfc)
    layer = default_trtnet().add_plugin_v2(inputs, plugin)
    return _create_tensor(layer.get_output(0), layer)


class FmhaLayer(Module):

    def __init__(self, num_heads: int, head_size: int, softmax_scale: float,
                 dtype: str):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale
        self.dtype = str_dtype_to_trt(dtype)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor):
        inputs = [Q, K, V]
        out = flash_attention_op(num_heads=self.num_heads,
                                 head_size=self.head_size,
                                 softmax_scale=self.softmax_scale,
                                 inputs=[p.trt_tensor for p in inputs])
        out.mark_output('out', self.dtype)
        return out

    def prepare_inputs(self, max_batch_size: int, max_len: int) -> List[Tensor]:
        '''

        @brief: Prepare inputs Tensors for the model, the given sizes are used to
            determine the ranges of the dimensions of when using TRT dynamic shapes.

        @return: a list contains values which can be fed into the self.forward()
        '''

        bs_range = [1, (max_batch_size + 1) // 2, max_batch_size]
        max_len_range = [1, (max_len + 1) // 2, max_len]

        dynamic_shape = [-1, self.num_heads, -1, self.head_size]
        Q = Tensor(name='Q',
                   dtype=self.dtype,
                   shape=dynamic_shape,
                   dim_range=OrderedDict([
                       ('batch_size', [bs_range]),
                       ('num_heads', [self.num_heads]),
                       ('seq_len', [max_len_range]),
                       ('head_size', [self.head_size]),
                   ]))
        K = Tensor(name='K',
                   dtype=self.dtype,
                   shape=dynamic_shape,
                   dim_range=OrderedDict([
                       ('batch_size', [bs_range]),
                       ('num_heads', [self.num_heads]),
                       ('seq_len', [max_len_range]),
                       ('head_size', [self.head_size]),
                   ]))
        V = Tensor(name='V',
                   dtype=self.dtype,
                   shape=dynamic_shape,
                   dim_range=OrderedDict([
                       ('batch_size', [bs_range]),
                       ('num_heads', [self.num_heads]),
                       ('seq_len', [max_len_range]),
                       ('head_size', [self.head_size]),
                   ]))
        return [Q, K, V]


def get_engine_name(head_size, dtype):
    return f'{LAYER_NAME}_{FMHA_KERNEL_BLOCK_SIZE}_d{head_size}_{dtype}.engine'
