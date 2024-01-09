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
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

from tensorrt_llm.logger import logger

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
    bert_attention_plugin: bool = False
    gpt_attention_plugin: bool = False
    multi_block_mode: bool = False
    identity_plugin: bool = False
    gemm_plugin: bool = False
    smooth_quant_gemm_plugin: bool = False
    layernorm_plugin: bool = False
    layernorm_quantization_plugin: bool = False
    rmsnorm_plugin: bool = False
    rmsnorm_quantization_plugin: bool = False
    attention_qk_half_accumulation: bool = False
    remove_input_padding: bool = False
    context_fmha_type: ContextFMHAType = ContextFMHAType.disabled
    weight_only_groupwise_quant_matmul_plugin: bool = False
    weight_only_quant_matmul_plugin: bool = False
    nccl_plugin: bool = False
    # TODO[kevin]: smart strategy to choose all reduce plugin
    use_custom_all_reduce: bool = False
    quantize_per_token_plugin: bool = False
    quantize_tensor_plugin: bool = False
    paged_kv_cache: bool = False
    tokens_per_block: int = 0
    lookup_plugin: bool = False
    lora_plugin: bool = False
    use_paged_context_fmha: bool = False
    use_context_fmha_for_generation: bool = False

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
