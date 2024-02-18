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
import argparse
import ctypes
import platform
from collections import OrderedDict
from dataclasses import dataclass, fields
from enum import IntEnum
from pathlib import Path
from typing import List, Optional, Tuple, Union

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

    # Plugins
    bert_attention_plugin: str = "float16"
    gpt_attention_plugin: str = "float16"
    gemm_plugin: str = None
    smooth_quant_gemm_plugin: str = None
    identity_plugin: str = None
    layernorm_quantization_plugin: str = None
    rmsnorm_quantization_plugin: str = None
    nccl_plugin: str = "float16"
    lookup_plugin: str = None
    lora_plugin: str = None
    weight_only_groupwise_quant_matmul_plugin: str = None
    weight_only_quant_matmul_plugin: str = None
    quantize_per_token_plugin: bool = False
    quantize_tensor_plugin: bool = False
    moe_plugin: str = "float16"

    # Features
    context_fmha: bool = True
    context_fmha_fp32_acc: bool = False  # will use fp16 if disabled
    paged_kv_cache: bool = True
    remove_input_padding: bool = True
    # TODO[kevin]: smart strategy to choose all reduce plugin
    use_custom_all_reduce: bool = True
    multi_block_mode: bool = False
    enable_xqa: bool = True
    attention_qk_half_accumulation: bool = False
    tokens_per_block: int = 128
    use_paged_context_fmha: bool = False
    use_context_fmha_for_generation: bool = False

    def set_plugin(self, name: str, value: Union[str, bool, int]):
        assert hasattr(self, name), f"Plugin name doesn't exist: {name}"
        if value is not None and getattr(self, name) is not None:
            target_type = type(getattr(self, name))
            assert type(value) == target_type, \
                f"Plugin {name} expects {target_type}, got {type(value)}"
        setattr(self, name, value)
        logger.info(f"Set {name} to {value}.")

    def update_from_dict(self, config: dict):
        for name in config.keys():
            if hasattr(self, name):
                value_to_be_update = config[name]
                if type(getattr(self, name)) == bool:
                    if value_to_be_update is True or \
                            value_to_be_update == "enable":
                        value_to_be_update = True
                    elif value_to_be_update is False or \
                            value_to_be_update == "disable":
                        value_to_be_update = False
                    else:
                        raise RuntimeError(
                            f"Unexpected value ({value_to_be_update}) to be updated for {name}."
                        )
                elif value_to_be_update == "disable":
                    value_to_be_update = None
                self.set_plugin(name, value_to_be_update)

    @classmethod
    def from_dict(cls, config: dict):
        plugin_config = cls()
        plugin_config.update_from_dict(config)
        return plugin_config

    @classmethod
    def from_arguments(cls, args: argparse.Namespace):
        return cls.from_dict(vars(args))

    def to_legacy_setting(self):
        '''Legacy setting means that all of the plugins and features are
        disabled, this needed for the legacy `build.py` script, which will be
        migrated to the centralized building script `tensorrt_llm/commands/build.py`.

        After the migration is done, this function may or may not be deleted.
        '''
        for field in fields(self):
            if field.type == str:
                self.set_plugin(field.name, None)
            elif field.type == bool:
                self.set_plugin(field.name, False)

    @property
    def context_fmha_type(self):
        if self.context_fmha:
            if self.context_fmha_fp32_acc:
                return ContextFMHAType.enabled_with_fp32_acc
            else:
                return ContextFMHAType.enabled
        else:
            return ContextFMHAType.disabled

    @context_fmha_type.setter
    def context_fmha_type(self, value):
        if value == ContextFMHAType.disabled:
            self.set_plugin("context_fmha", False)
        else:
            self.set_plugin("context_fmha", True)
            if value == ContextFMHAType.enabled:
                self.set_plugin("context_fmha_fp32_acc", False)
            elif value == ContextFMHAType.enabled_with_fp32_acc:
                self.set_plugin("context_fmha_fp32_acc", True)

    def set_smooth_quant_plugins(self, dtype: str = "float16"):
        self.set_plugin("smooth_quant_gemm_plugin", dtype)
        self.set_plugin("rmsnorm_quantization_plugin", dtype)
        self.set_plugin("layernorm_quantization_plugin", dtype)
        self.set_plugin("quantize_per_token_plugin", True)
        self.set_plugin("quantize_tensor_plugin", True)

    def enable_qk_half_accum(self):
        self.set_plugin("attention_qk_half_accumulation", True)
        return self

    def set_context_fmha(self, context_fmha_type=ContextFMHAType.enabled):
        assert type(context_fmha_type) == ContextFMHAType
        self.context_fmha_type = context_fmha_type
        return self

    def enable_remove_input_padding(self):
        self.set_plugin("remove_input_padding", True)
        return self

    def enable_paged_kv_cache(self, tokens_per_block=128):
        self.set_plugin("paged_kv_cache", True)
        self.set_plugin("tokens_per_block", tokens_per_block)
        return self

    def set_gpt_attention_plugin(self, dtype='float16'):
        self.set_plugin("gpt_attention_plugin", dtype)
        return self

    def enable_mmha_multi_block_mode(self):
        self.set_plugin("multi_block_mode", True)
        return self

    def enable_xqa_optimization(self):
        self.set_plugin("enable_xqa", True)
        return self

    def set_bert_attention_plugin(self, dtype='float16'):
        self.set_plugin("bert_attention_plugin", dtype)
        return self

    def set_identity_plugin(self, dtype='float16'):
        self.set_plugin("identity_plugin", dtype)
        return self

    def set_gemm_plugin(self, dtype='float16'):
        self.set_plugin("gemm_plugin", dtype)
        return self

    def set_moe_plugin(self, dtype='float16'):
        self.moe_plugin = dtype
        return self

    def set_smooth_quant_gemm_plugin(self, dtype='float16'):
        self.set_plugin("smooth_quant_gemm_plugin", dtype)
        return self

    def set_layernorm_quantization_plugin(self, dtype='float16'):
        self.set_plugin("layernorm_quantization_plugin", dtype)
        return self

    def set_rmsnorm_quantization_plugin(self, dtype='float16'):
        self.set_plugin("rmsnorm_quantization_plugin", dtype)
        return self

    def set_weight_only_quant_matmul_plugin(self, dtype='float16'):
        self.set_plugin("weight_only_quant_matmul_plugin", dtype)
        return self

    def set_weight_only_groupwise_quant_matmul_plugin(self, dtype='float16'):
        self.set_plugin("weight_only_groupwise_quant_matmul_plugin", dtype)
        return self

    def set_nccl_plugin(self,
                        dtype='float16',
                        use_custom_all_reduce: bool = False):
        self.set_plugin("nccl_plugin", dtype)
        self.set_plugin("use_custom_all_reduce", use_custom_all_reduce)
        if use_custom_all_reduce:
            init_all_reduce_helper()
        return self

    def set_quantize_per_token_plugin(self):
        self.set_plugin("quantize_per_token_plugin", True)
        return self

    def set_quantize_tensor_plugin(self):
        self.set_plugin("quantize_tensor_plugin", True)
        return self

    def set_lookup_plugin(self, dtype='float16'):
        self.set_plugin("lookup_plugin", dtype)
        return self

    def set_lora_plugin(self, dtype='float16'):
        self.set_plugin("lora_plugin", dtype)
        return self

    def set_paged_context_fmha(self):
        self.set_plugin("use_paged_context_fmha", True)
        return self

    def set_context_fmha_for_generation(self):
        self.set_plugin("use_context_fmha_for_generation", True)
        return self


cli_plugin_args = [
    # Plugins
    "bert_attention_plugin",
    "gpt_attention_plugin",
    "gemm_plugin",
    "lookup_plugin",
    "lora_plugin",
    "moe_plugin",

    # Features
    "context_fmha",
    "context_fmha_fp32_acc",
    "paged_kv_cache",
    "remove_input_padding",
    "use_custom_all_reduce",
    "multi_block_mode",
    "enable_xqa",
    "attention_qk_half_accumulation",
    "tokens_per_block",
    "use_paged_context_fmha",
    "use_context_fmha_for_generation",
]

plugin_options = ["float16", "float32", "bfloat16", "disable"]


def add_plugin_argument(parser):
    plugin_config = PluginConfig()
    for field in fields(plugin_config):
        if field.name not in cli_plugin_args:
            continue
        if field.type == str:
            parser.add_argument(
                "--" + field.name,
                type=str,
                default=field.default if field.default is not None else None,
                choices=plugin_options)
        elif field.type == bool:
            parser.add_argument(
                "--" + field.name,
                type=str,
                default="enable" if field.default else "disable",
                choices=["enable", "disable"])
        else:
            parser.add_argument("--" + field.name,
                                type=field.type,
                                default=field.default)
    return parser


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
