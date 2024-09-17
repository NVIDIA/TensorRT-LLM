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
from abc import ABCMeta, abstractmethod
from typing import Optional

import numpy as np
import tensorrt as trt
import torch

from .._common import default_net, default_trtnet
from .._utils import set_obj_attrs, str_dtype_to_torch, str_dtype_to_trt
from ..functional import (AllReduceFusionOp, AllReduceFusionParams, Tensor,
                          _add_plugin_info, _create_tensor, allgather,
                          allreduce, cast, low_latency_gemm, matmul)
from ..mapping import Mapping
from ..module import Module
from ..parameter import Parameter
from ..plugin import TRT_LLM_PLUGIN_NAMESPACE
from .lora import LoraRuntimeParams


def _gemm_plugin(input: Tensor,
                 mat2: Tensor,
                 transa: bool = False,
                 transb: bool = False,
                 pad_lda: int = 0,
                 pad_ldb: int = 0,
                 use_fp8: bool = False,
                 alpha: Optional[np.ndarray] = None,
                 strict_dtype: Optional[trt.DataType] = None) -> Tensor:
    '''
    output = op(mat2)op(input)

    Parameters:
        input : Tensor (On GPU)
            The input tensor.

        mat2 : Tensor (On GPU)
            The mat2 tensor.

        transa : bool
            Is the input tensor transposed? Set to 'True' if you want the
            input tensor to be transposed, 'False' otherwise.

        transb : bool
            Is the mat2 tensor transposed? Set to 'True' if you want the
            mat2 tensor to be transposed, 'False' otherwise.

        pad_lda: int
            Padding to the lead dimension of input tensor. It is used to
            support the strided GEMM that only uses the sub-tensor for
            computation. The GEMM plugin computation is
            [N, K] x [K, M+pad_lda] -> [N, M] if transa,
            [N, K] x [K+pad_lda, M] -> [N, M] if not transa.

        pad_ldb: int
            Padding to the lead dimension of mat2 tensor. It is used to
            support the strided GEMM that only uses the sub-tensor for
            computation. The GEMM plugin computation is
            [N, K+pad_ldb] x [K, M] -> [N, M] if transb,
            [N+pad_ldb, K] x [K, M] -> [N, M] if not transb.

        use_fp8: bool
            Do we use fp8 GEMM.

        alpha: float
            Alpha for fp8 GEMM.

        strict_dtype: trt.DataType
            Set the data type for the GEMM plugin. If it is None, the data
            type is the gemm_plugin type set in the plugin_config.
    '''
    plg_creator = trt.get_plugin_registry().get_plugin_creator(
        "Gemm", "1", TRT_LLM_PLUGIN_NAMESPACE)
    assert plg_creator is not None

    if use_fp8:
        assert (
            isinstance(alpha, np.ndarray) and alpha.dtype == np.float32
            and alpha.size == 1
        ), "`alpha` must be passed as a float32 ndarray if `use_fp8` is enabled for _gemm_plugin"
        assert input.dtype == trt.fp8
        assert mat2.dtype == trt.fp8

    transa = 1 if transa else 0
    transa = trt.PluginField("transa", np.array(transa, dtype=np.int32),
                             trt.PluginFieldType.INT32)
    transb = 1 if transb else 0
    transb = trt.PluginField("transb", np.array(transb, dtype=np.int32),
                             trt.PluginFieldType.INT32)
    pad_lda = trt.PluginField("pad_lda", np.array(pad_lda, dtype=np.int32),
                              trt.PluginFieldType.INT32)
    pad_ldb = trt.PluginField("pad_ldb", np.array(pad_ldb, dtype=np.int32),
                              trt.PluginFieldType.INT32)
    use_fp8 = 1 if use_fp8 else 0
    use_fp8 = trt.PluginField("use_fp8", np.array(use_fp8, dtype=np.int32),
                              trt.PluginFieldType.INT32)
    alpha = alpha if alpha else np.array(1.0, dtype=np.float32)
    alpha = trt.PluginField("alpha", alpha.flatten(),
                            trt.PluginFieldType.FLOAT32)

    if strict_dtype is not None:
        assert isinstance(strict_dtype, trt.DataType)
        p_dtype = strict_dtype
    else:
        p_dtype = str_dtype_to_trt(default_net().plugin_config.gemm_plugin)
        assert p_dtype != trt.fp8, "need to use strict dtype in gemm plugin fp8"
    pf_type = trt.PluginField("type_id", np.array([int(p_dtype)], np.int32),
                              trt.PluginFieldType.INT32)
    pfc = trt.PluginFieldCollection(
        [transa, transb, pad_lda, pad_ldb, pf_type, use_fp8, alpha])
    gemm_plug = plg_creator.create_plugin("gemm", pfc)
    plug_inputs = [input.trt_tensor, mat2.trt_tensor]

    layer = default_trtnet().add_plugin_v2(plug_inputs, gemm_plug)
    _add_plugin_info(layer, plg_creator, "gemm", pfc)
    return _create_tensor(layer.get_output(0), layer)


class LinearBase(Module, metaclass=ABCMeta):

    def __init__(
        self,
        local_in_features,
        local_out_features,
        bias=True,
        dtype=None,
        tp_group=None,
        tp_size=1,
        share_weight=None,
        strict_dtype=False,
        pad_lda=0,
        prefer_managed_weight=True,
    ):
        super().__init__()
        self.in_features = local_in_features
        self.out_features = local_out_features
        self.dtype = dtype
        self.pad_lda = pad_lda
        self.prefer_managed_weight = prefer_managed_weight

        self.share_weight = share_weight
        if not share_weight:
            self.weight = Parameter(
                shape=(self.out_features, self.in_features),
                dtype=dtype,
                prefer_managed=self.prefer_managed_weight,
            )
            set_obj_attrs(
                self.weight,
                {
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.weight = share_weight

        self.tp_size = tp_size
        self.tp_group = tp_group
        self.strict_dtype = self.dtype if strict_dtype else None

        if bias:
            self.bias = Parameter(shape=(self.out_features, ), dtype=dtype)
        else:
            self.register_parameter("bias", None)

        # see optimize_model's add_lora for LoRA initialization
        self.lora = None

    def weight_loader(self, mapping: Mapping, param: Parameter,
                      loaded_weight: torch.Tensor) -> None:
        tp_rank = mapping.tp_rank
        shard_size = param._shape[self.tp_split_dim()]
        start_idx = tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_split_dim(), start_idx,
                                             shard_size)

        param.value = loaded_weight

    @classmethod
    @abstractmethod
    def tp_split_dim(cls) -> int:
        pass

    def weight_is_kn(self):  # WAR for bug 4641821
        return (default_net().plugin_config.manage_weights
                and self.prefer_managed_weight
                and self.weight.dtype == trt.DataType.HALF)

    def get_weight(self) -> Tensor:
        if default_net(
        ).plugin_config.manage_weights and self.prefer_managed_weight:
            use_gemm_plugin = default_net(
            ).plugin_config.gemm_plugin is not None
            use_low_latency_gemm_plugin = default_net(
            ).plugin_config.low_latency_gemm_plugin == 'fp8'
            return self.weight.get_managed_tensor(
                network=default_net(),
                need_transpose=self.weight_is_kn() and not use_gemm_plugin
                and not use_low_latency_gemm_plugin)
        else:
            return self.weight.get_constant_tensor(network=default_net())

    def multiply_and_lora(
        self,
        x,
        weight,
        gemm_plugin: Optional[str] = None,
        low_latency_gemm_plugin: Optional[str] = None,
        use_fp8: bool = False,
        alpha: Optional[np.ndarray] = None,
        lora_runtime_params: Optional[LoraRuntimeParams] = None,
        lora_hidden_state: Optional[Tensor] = None,
    ):
        hidden_state = x
        if low_latency_gemm_plugin:
            strict_dtype = str_dtype_to_trt(self.dtype) if isinstance(
                self.dtype, str) else self.dtype
            x = low_latency_gemm(x, weight, alpha, strict_dtype)
        elif gemm_plugin:
            if gemm_plugin == 'fp8':
                strict_dtype = str_dtype_to_trt(self.dtype) if isinstance(
                    self.dtype, str) else self.dtype
            else:
                strict_dtype = self.strict_dtype
            x = _gemm_plugin(x,
                             weight,
                             transb=True,
                             pad_lda=self.pad_lda,
                             use_fp8=use_fp8,
                             alpha=alpha,
                             strict_dtype=strict_dtype)
        else:
            x = matmul(x, weight, transb=not self.weight_is_kn())

        if default_net(
        ).plugin_config.lora_plugin and lora_runtime_params is not None:
            x = x + self.lora(
                hidden_state
                if lora_hidden_state is None else lora_hidden_state,
                lora_runtime_params=lora_runtime_params,
            )
        return x

    @abstractmethod
    def collect_and_bias(self, x: Tensor) -> Tensor:
        pass

    def multiply_collect(
            self,
            x,
            weight,
            gemm_plugin: Optional[str] = None,
            low_latency_gemm_plugin: Optional[str] = None,
            use_fp8: bool = False,
            alpha: Optional[np.ndarray] = None,
            lora_runtime_params: Optional[LoraRuntimeParams] = None,
            lora_hidden_state: Optional[Tensor] = None,
            **kwargs):
        x = self.multiply_and_lora(
            x,
            weight,
            gemm_plugin=gemm_plugin,
            low_latency_gemm_plugin=low_latency_gemm_plugin,
            use_fp8=use_fp8,
            alpha=alpha,
            lora_runtime_params=lora_runtime_params,
            lora_hidden_state=lora_hidden_state,
        )
        return self.collect_and_bias(x, **kwargs)

    def forward(self,
                x,
                lora_runtime_params: Optional[LoraRuntimeParams] = None,
                lora_hidden_state: Optional[Tensor] = None,
                **kwargs) -> Tensor:
        return self.multiply_collect(
            x,
            self.get_weight(),
            gemm_plugin=default_net().plugin_config.gemm_plugin,
            use_fp8=False,
            lora_runtime_params=lora_runtime_params,
            lora_hidden_state=lora_hidden_state,
            **kwargs)


class Linear(LinearBase):

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        dtype=None,
        tp_group=None,
        tp_size=1,
        gather_output=True,
        share_weight=None,
        strict_dtype=False,
        pad_lda=0,
        prefer_managed_weight=True,
        is_qkv=False,
    ):
        super().__init__(
            local_in_features=in_features,
            local_out_features=out_features // tp_size,
            bias=bias,
            dtype=dtype,
            tp_group=tp_group,
            tp_size=tp_size,
            share_weight=share_weight,
            strict_dtype=strict_dtype,
            pad_lda=pad_lda,
            prefer_managed_weight=prefer_managed_weight,
        )
        self.gather_output = gather_output
        self.is_qkv = is_qkv
        self.tp_dim = 0
        if bias:
            set_obj_attrs(
                self.bias,
                {
                    "weight_loader": self.weight_loader,
                },
            )

    @classmethod
    def tp_split_dim(cls) -> int:
        return 0

    def collect_and_bias(self, x, **kwargs):

        if self.bias is not None:
            bias = cast(self.bias.value, x.dtype)
            x = x + bias

        if self.gather_output and self.tp_size > 1 and self.tp_group is not None:
            # [dim0, local_dim] -> [dim0 * tp_size, local_dim] --> [dim0, local_dim * tp_size]
            x = allgather(x, self.tp_group, gather_dim=-1)

        return x

    def postprocess(self, tllm_key, weights, **kwargs):
        using_head_as_leading_dim = kwargs.get("using_head_as_leading_dim",
                                               False)
        config = kwargs.get("config", None)
        if self.is_qkv:
            if isinstance(weights, list):
                if hasattr(config, "remove_duplicated_kv_heads"):
                    if config.remove_duplicated_kv_heads:
                        head_size = config.hidden_size // config.num_attention_heads if config.head_size is None else config.head_size
                        k, v = weights[1:]
                        k = k.reshape([
                            k.shape[0] // head_size // 2, 2, head_size,
                            self.in_features
                        ])
                        v = v.reshape([
                            v.shape[0] // head_size // 2, 2, head_size,
                            self.in_features
                        ])
                        assert (k[:, 0] == k[:, 1]).all()
                        assert (v[:, 0] == v[:, 1]).all()
                        k = k[:, 0].reshape([-1, self.in_features])
                        v = v[:, 0].reshape([-1, self.in_features])
                        weights[1] = k
                        weights[2] = v
                weights = torch.cat(weights)
            if using_head_as_leading_dim:
                # Reorder [n_head, 3, head_dim, ...] into [3, n_head, head_dim, ...]
                assert config.num_attention_heads == config.num_key_value_heads, "using_head_as_leading_dim require head_size to be multiple of 3."
                num_heads = config.num_attention_heads
                head_dim = self.out_features // (3 * num_heads)
                w = weights.reshape(num_heads, 3, head_dim, -1)
                w = w.transpose(0, 1)
                if w.shape[-1] > 1:
                    weights = w.reshape(-1, self.in_features)  # Weight
                else:
                    weights = w.reshape(-1)  # Bias
        weights = weights.to(str_dtype_to_torch(self.dtype))
        return {tllm_key: weights}


ColumnLinear = Linear


class RowLinear(LinearBase):

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        dtype=None,
        tp_group=None,
        tp_size=1,
        strict_dtype: bool = False,
        pad_lda=0,
        prefer_managed_weight=True,
        is_expert=False,
    ):
        super().__init__(
            local_in_features=in_features // tp_size,
            local_out_features=out_features,
            bias=bias,
            dtype=dtype,
            tp_group=tp_group,
            tp_size=tp_size,
            strict_dtype=strict_dtype,
            pad_lda=pad_lda,
            prefer_managed_weight=prefer_managed_weight,
        )

        self.tp_dim = 1
        self.tp_size = tp_size
        self.is_expert = is_expert

    @classmethod
    def tp_split_dim(cls) -> int:
        return 1

    def collect_and_bias(self, x, **kwargs):
        reduce_fusion_params: Optional[AllReduceFusionParams] = kwargs.get(
            "reduce_fusion_params", None)
        if self.tp_size > 1 and self.tp_group is not None:
            need_bias = self.bias is not None
            fuse_bias_into_all_reduce = (
                need_bias and (reduce_fusion_params is not None)
                and (reduce_fusion_params.fusion_op
                     == AllReduceFusionOp.RESIDUAL_RMS_NORM))
            if fuse_bias_into_all_reduce:
                reduce_fusion_params.bias = self.bias.value
            if not self.is_expert:
                x = allreduce(x,
                              self.tp_group,
                              reduce_fusion_params=reduce_fusion_params)
                if need_bias and not fuse_bias_into_all_reduce:
                    bias = cast(self.bias.value, x.dtype)
                    x = x + bias
            else:
                if need_bias and not fuse_bias_into_all_reduce:
                    bias = cast(self.bias.value, x.dtype)
                    x = x + bias / self.tp_size
            return x

        if self.bias is not None:
            bias = cast(self.bias.value, x.dtype)
            x = x + bias

        return x
