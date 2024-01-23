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
import platform
from collections import OrderedDict
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import List, Optional, Tuple

import tensorrt as trt

from tensorrt_llm.logger import logger

from .._ipc_utils import IpcMemory
from ..mapping import Mapping

TRT_LLM_PLUGIN_NAMESPACE = 'tensorrt_llm'


def plugin_lib_path() -> str:
    project_dir = Path(__file__).parent.parent.absolute()
    dyn_lib = "libnvinfer_plugin_tensorrt_llm.so" if platform.system(
    ) != "Windows" else "nvinfer_plugin_tensorrt_llm.dll"
    return str(project_dir.joinpath("libs", dyn_lib))


def _load_plugin_lib():
    winmode = 0 if platform.system() == "Windows" else None
    handle = ctypes.CDLL(plugin_lib_path(),
                         mode=ctypes.RTLD_GLOBAL,
                         winmode=winmode)
    try:
        handle.initTrtLlmPlugins.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        handle.initTrtLlmPlugins.restype = ctypes.c_bool
    except AttributeError as err:
        raise ImportError('TensorRT-LLM Plugin is unavailable') from err
    assert handle.initTrtLlmPlugins(None,
                                    TRT_LLM_PLUGIN_NAMESPACE.encode('utf-8'))


class ContextFMHAType(IntEnum):
    disabled = 0
    # FP16 I/O, FP16 Accumulation
    enabled = 1
    # FP16 I/O, FP32 Accumulation
    enabled_with_fp32_acc = 2


@dataclass
class PluginConfig:
    bert_attention_plugin: Optional[str] = None
    gpt_attention_plugin: Optional[str] = None
    multi_block_mode: bool = False
    enable_xqa: bool = False
    identity_plugin: Optional[str] = None
    gemm_plugin: Optional[str] = None
    smooth_quant_gemm_plugin: Optional[str] = None
    layernorm_plugin: Optional[str] = None
    layernorm_quantization_plugin: Optional[str] = None
    rmsnorm_plugin: Optional[str] = None
    rmsnorm_quantization_plugin: Optional[str] = None
    attention_qk_half_accumulation: bool = False
    remove_input_padding: bool = False
    context_fmha_type: ContextFMHAType = ContextFMHAType.disabled
    weight_only_groupwise_quant_matmul_plugin: Optional[str] = None
    weight_only_quant_matmul_plugin: Optional[str] = None
    nccl_plugin: Optional[str] = None
    # TODO[kevin]: smart strategy to choose all reduce plugin
    use_custom_all_reduce: bool = False
    quantize_per_token_plugin: bool = False
    quantize_tensor_plugin: bool = False
    paged_kv_cache: bool = False
    tokens_per_block: int = 0
    lookup_plugin: Optional[str] = None
    lora_plugin: Optional[str] = None
    use_paged_context_fmha: bool = False
    use_context_fmha_for_generation: bool = False
    selective_scan_plugin: bool = False

    def enable_qk_half_accum(self):
        self.attention_qk_half_accumulation = True
        logger.info(f"Attention BMM1(QK) accumulation type is set to FP16")
        return self

    def set_context_fmha(self, context_fmha_type=ContextFMHAType.enabled):
        assert context_fmha_type in \
            [ContextFMHAType.disabled, ContextFMHAType.enabled,
                ContextFMHAType.enabled_with_fp32_acc]
        self.context_fmha_type = context_fmha_type
        if context_fmha_type == ContextFMHAType.enabled:
            logger.info(f"Context FMHA Enabled")
        elif context_fmha_type == ContextFMHAType.enabled_with_fp32_acc:
            logger.info(f"Context FMHA with FP32 Accumulation Enabled")
        elif context_fmha_type == ContextFMHAType.disabled:
            logger.info(f"Context FMHA Disabled")
        return self

    def enable_remove_input_padding(self):
        self.remove_input_padding = True
        logger.info(f"Remove Padding Enabled")
        return self

    def enable_paged_kv_cache(self, tokens_per_block=64):
        self.paged_kv_cache = True
        self.tokens_per_block = tokens_per_block
        logger.info(f"Paged KV Cache Enabled")
        return self

    def set_gpt_attention_plugin(self, dtype='float16'):
        self.gpt_attention_plugin = dtype
        return self

    def enable_mmha_multi_block_mode(self):
        self.multi_block_mode = True
        logger.info(f"Generation Multi Block Mode Enabled")
        return self

    def enable_xqa_optimization(self):
        self.enable_xqa = True
        logger.info(f"Optimized Generation MHA kernels (XQA) Enabled")
        return self

    def set_bert_attention_plugin(self, dtype='float16'):
        self.bert_attention_plugin = dtype
        return self

    def set_identity_plugin(self, dtype='float16'):
        self.identity_plugin = dtype
        return self

    def set_gemm_plugin(self, dtype='float16'):
        self.gemm_plugin = dtype
        return self

    def set_smooth_quant_gemm_plugin(self, dtype='float16'):
        self.smooth_quant_gemm_plugin = dtype
        return self

    def set_layernorm_plugin(self, dtype='float16'):
        self.layernorm_plugin = dtype
        return self

    def set_layernorm_quantization_plugin(self, dtype='float16'):
        self.layernorm_quantization_plugin = dtype
        return self

    def set_rmsnorm_plugin(self, dtype='float16'):
        self.rmsnorm_plugin = dtype
        return self

    def set_rmsnorm_quantization_plugin(self, dtype='float16'):
        self.rmsnorm_quantization_plugin = dtype
        return self

    def set_weight_only_quant_matmul_plugin(self, dtype='float16'):
        self.weight_only_quant_matmul_plugin = dtype
        return self

    def set_weight_only_groupwise_quant_matmul_plugin(self, dtype='float16'):
        self.weight_only_groupwise_quant_matmul_plugin = dtype
        return self

    def set_nccl_plugin(self,
                        dtype='float16',
                        use_custom_all_reduce: bool = False):
        self.use_custom_all_reduce = use_custom_all_reduce
        if use_custom_all_reduce:
            init_all_reduce_helper()
        self.nccl_plugin = dtype
        return self

    def set_quantize_per_token_plugin(self):
        self.quantize_per_token_plugin = True
        return self

    def set_quantize_tensor_plugin(self):
        self.quantize_tensor_plugin = True
        return self

    def set_lookup_plugin(self, dtype='float16'):
        self.lookup_plugin = dtype
        return self

    def set_lora_plugin(self, dtype='float16'):
        self.lora_plugin = dtype
        return self

    def set_paged_context_fmha(self):
        self.use_paged_context_fmha = True
        return self

    def set_context_fmha_for_generation(self):
        self.use_context_fmha_for_generation = True
        return self

    def set_selective_scan_plugin(self, dtype='float16'):
        self.selective_scan_plugin = dtype
        return self


class CustomAllReduceHelper:
    """
        Globally visible class to help usage of custom_all_reduce plugin.
        Provides the following utilities:

        gen_id: int
            Used for synchronization with custom kernels. Plugins instances MUST have the same
            id across GPUs. I.e.: GPU#0's allreduce after MLP at layer i must have the same id as
            GPU#1, GPU#2... Also, ids MUST be unique per model. There should not be two allreduce instances
            in GPU#0 that have the same id.

        workspace: Tensor
            When using CUSTOM or AUTO mode, a tensor containing pointers to memory
            visible to all GPUs. It should be 3 poitners per TP rank -
            ptr to data buffer, ptr to barriers in, ptr to barriers out.
            It must be initialized using IpcMemory class.

        Usage:
            - Use `init_all_reduce_helper` to reset the id counter. This must be done in main model class.
            - Set custom_all_reduce_helper.workspace with the required tensor.
              Then, each instance of allreduce will reference that tensor automatically.
    """
    POINTERS_PER_RANK = 4

    def __init__(self) -> None:
        self.current_id: int = 1
        self.workspace: Optional[Tensor] = None

    def gen_id(self) -> int:
        result = self.current_id
        self.current_id += 1
        return result

    def set_workspace_tensor(self,
                             mapping: Mapping,
                             two_opt_profiles: Optional[bool] = None):
        from ..functional import Tensor
        workspace_size = self.POINTERS_PER_RANK * mapping.tp_size

        dim_range = None
        if two_opt_profiles is not None:
            dim_range = OrderedDict([
                ('all_reduce_size', [workspace_size, workspace_size]
                 if two_opt_profiles else [workspace_size])
            ])

        self.workspace = Tensor(
            name='all_reduce_workspace',
            dtype=trt.int64,
            shape=[workspace_size],
            dim_range=dim_range,
        )

    @staticmethod
    def max_workspace_size_auto(tp_size: int) -> int:
        if tp_size <= 2:
            return 16_000_000
        return 8_000_000

    @staticmethod
    def allocate_workspace(mapping: Mapping,
                           size: int) -> Tuple[List[IpcMemory], "torch.tensor"]:
        import torch
        ipc_buffers_ping = IpcMemory(mapping, size)
        ipc_buffers_pong = IpcMemory(mapping, size)
        ipc_barriers_in = IpcMemory(
            mapping, IpcMemory.IPC_BARRIERS_SIZE_PER_GPU * mapping.tp_size)
        ipc_barriers_out = IpcMemory(
            mapping, IpcMemory.IPC_BARRIERS_SIZE_PER_GPU * mapping.tp_size)
        buffers = [
            ipc_buffers_ping, ipc_buffers_pong, ipc_barriers_in,
            ipc_buffers_ping
        ]

        return buffers, torch.tensor(
            ipc_buffers_ping.serialize() + ipc_buffers_pong.serialize() +
            ipc_barriers_in.serialize() + ipc_barriers_out.serialize(),
            dtype=torch.int64,
            device="cpu")


custom_all_reduce_helper = None


def init_all_reduce_helper():
    global custom_all_reduce_helper
    custom_all_reduce_helper = CustomAllReduceHelper()


def current_all_reduce_helper():
    global custom_all_reduce_helper
    assert custom_all_reduce_helper is not None, "You must call `init_all_reduce_helper` first"
    return custom_all_reduce_helper
