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
from dataclasses import asdict, dataclass, field, fields
from enum import IntEnum
from pathlib import Path
from typing import List, Optional, Tuple

import tensorrt as trt

from .._ipc_utils import IpcMemory
from ..logger import logger
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


DEFAULT_PLUGIN_DTYPE_OPTIONS = [
    "auto", "float16", "float32", "bfloat16", "int32", None
]
PLUGIN_DTYPE_OPTIONS_MAP = {
    "gemm_swiglu_plugin": ["fp8", None],
    "gemm_plugin":
    ["auto", "float16", "float32", "bfloat16", "int32", "fp8", None]
}


def _make_plugin_property(field_name: str, field_type: type):

    def bind(field_name):
        storage_name = f'_{field_name}'

        @property
        def prop(self):
            field_value = getattr(self, storage_name)
            if field_name != 'dtype' and field_value == 'auto':
                return self.dtype
            else:
                return field_value

        @prop.setter
        def prop(self, value):
            if field_type is bool:
                assert isinstance(value, bool), \
                    f"Plugin {field_name} expects {field_type}, got {type(value)}"
            elif field_type in (str, Optional[str]):
                plugin_dtype_options = DEFAULT_PLUGIN_DTYPE_OPTIONS
                if field_name in PLUGIN_DTYPE_OPTIONS_MAP:
                    plugin_dtype_options = PLUGIN_DTYPE_OPTIONS_MAP[field_name]
                assert value in plugin_dtype_options, \
                    f"Plugin {field_name} expects values in {plugin_dtype_options}, got {value}"
            if field_name == 'dtype':
                assert value not in ['auto', None], \
                    "Plugin dtype cannot be auto or None"
            setattr(self, storage_name, value)
            logger.info(f"Set {field_name} to {value}.")

        return prop

    return bind(field_name)


class PluginConfigMeta(type):

    def __new__(cls, name, bases, attrs):
        for storage_name, field_type in attrs['__annotations__'].items():
            assert storage_name.startswith('_')
            field_name = storage_name.lstrip('_')
            attrs[field_name] = _make_plugin_property(field_name, field_type)
        return super().__new__(cls, name, bases, attrs)


@dataclass(slots=True)
class PluginConfig(metaclass=PluginConfigMeta):
    """The config that manages plugin-related options.

    There are two option categories:
    * Plugin options (typically with xxx_plugin naming). These options can be assigned with:
        * "float16"/"bfloat16"/"float32"/"int32", which means the plugin is enabled with the specified precision; (Some plugins only support limited dtype, i.e., gemm_swiglu_plugin only supports fp8 now)
        * "auto", which means the plugin is enabled with the precision of `dtype` field (the `dtype` field must be same to model dtype, i.e., the one in PretrainedConfig);
        * None, which means the plugin is disabled.
    * Other features. These options can be assigned with boolean:
        * True, which means the plugin is enabled;
        * False, which means the plugin is disabled.

    Note: All the fields should use a prefix "_"; PluginConfigMeta will wrap each field as a property.
    This ensures the fields can only be assigned with allowed values.
    """
    _dtype: str = field(default="float16", init=False)

    # Plugins
    _bert_attention_plugin: Optional[str] = field(default="auto", init=False)
    _gpt_attention_plugin: Optional[str] = field(default="auto", init=False)
    _gemm_plugin: Optional[str] = field(default=None, init=False)
    _gemm_swiglu_plugin: Optional[str] = field(default=None, init=False)
    _smooth_quant_gemm_plugin: Optional[str] = field(default=None, init=False)
    _identity_plugin: Optional[str] = field(default=None, init=False)
    _layernorm_quantization_plugin: Optional[str] = field(default=None,
                                                          init=False)
    _rmsnorm_quantization_plugin: Optional[str] = field(default=None,
                                                        init=False)
    _nccl_plugin: Optional[str] = field(default="auto", init=False)
    _lookup_plugin: Optional[str] = field(default=None, init=False)
    _lora_plugin: Optional[str] = field(default=None, init=False)
    _weight_only_groupwise_quant_matmul_plugin: Optional[str] = field(
        default=None, init=False)
    _weight_only_quant_matmul_plugin: Optional[str] = field(default=None,
                                                            init=False)
    _quantize_per_token_plugin: bool = field(default=False, init=False)
    _quantize_tensor_plugin: bool = field(default=False, init=False)
    _moe_plugin: Optional[str] = field(default="auto", init=False)
    _mamba_conv1d_plugin: Optional[str] = field(default="auto", init=False)

    # Features
    _context_fmha: bool = field(default=True, init=False)
    _context_fmha_fp32_acc: bool = field(
        default=False, init=False)  # will use fp16 if disabled
    _paged_kv_cache: bool = field(default=True, init=False)
    _remove_input_padding: bool = field(default=True, init=False)
    _use_custom_all_reduce: bool = field(default=True, init=False)
    _reduce_fusion: bool = field(default=False, init=False)
    _multi_block_mode: bool = field(default=False, init=False)
    _enable_xqa: bool = field(default=True, init=False)
    _tokens_per_block: int = field(default=64, init=False)
    _use_paged_context_fmha: bool = field(default=False, init=False)
    _use_fp8_context_fmha: bool = field(default=False, init=False)
    _multiple_profiles: bool = field(default=False, init=False)
    _paged_state: bool = field(default=True, init=False)
    _streamingllm: bool = field(default=False, init=False)

    def update_from_dict(self, config: dict):
        for name in config.keys():
            if hasattr(self, name):
                value_to_be_update = config[name]
                if isinstance(getattr(self, name), bool):
                    if value_to_be_update == "enable":
                        value_to_be_update = True
                    elif value_to_be_update == "disable":
                        value_to_be_update = False
                elif value_to_be_update == "disable":
                    value_to_be_update = None
                setattr(self, name, value_to_be_update)

    @classmethod
    def from_dict(cls, config: dict):
        plugin_config = cls()
        plugin_config.update_from_dict(config)
        return plugin_config

    @classmethod
    def from_arguments(cls, args: argparse.Namespace):
        return cls.from_dict(vars(args))

    def to_dict(self):
        config = asdict(self)
        # Remove prefix "_" of the storage name
        config = {key.lstrip('_'): value for key, value in config.items()}
        return config

    def to_legacy_setting(self):
        '''Legacy setting means that all of the plugins and features are
        disabled, this needed for the legacy `build.py` script, which will be
        migrated to the centralized building script `tensorrt_llm/commands/build.py`.

        After the migration is done, this function may or may not be deleted.
        '''
        for field in fields(self):
            # Remove prefix "_" of the storage name
            field_name = field.name.lstrip('_')
            if field_name == 'dtype':
                continue
            if field.type in (str, Optional[str]):
                setattr(self, field_name, None)
            elif field.type == bool:
                setattr(self, field_name, False)

    @property
    def context_fmha_type(self):
        if self.context_fmha_fp32_acc:
            return ContextFMHAType.enabled_with_fp32_acc
        elif self.context_fmha:
            return ContextFMHAType.enabled
        else:
            return ContextFMHAType.disabled

    @context_fmha_type.setter
    def context_fmha_type(self, value):
        if value == ContextFMHAType.disabled:
            self.context_fmha = False
            self.context_fmha_fp32_acc = False
        else:
            self.context_fmha = True
            if value == ContextFMHAType.enabled:
                self.context_fmha_fp32_acc = False
            elif value == ContextFMHAType.enabled_with_fp32_acc:
                self.context_fmha_fp32_acc = True

    def set_smooth_quant_plugins(self, dtype: str = "auto"):
        self.smooth_quant_gemm_plugin = dtype
        self.rmsnorm_quantization_plugin = dtype
        self.layernorm_quantization_plugin = dtype
        self.quantize_per_token_plugin = True
        self.quantize_tensor_plugin = True
        return self

    def set_context_fmha(self, context_fmha_type=ContextFMHAType.enabled):
        assert type(context_fmha_type) == ContextFMHAType
        self.context_fmha_type = context_fmha_type
        return self

    def enable_paged_kv_cache(self, tokens_per_block: int = 64):
        self.paged_kv_cache = True
        self.tokens_per_block = tokens_per_block
        return self

    def set_nccl_plugin(self,
                        dtype: str = "auto",
                        use_custom_all_reduce: bool = True):
        if not use_custom_all_reduce:
            logger.warning(
                "allreduce algorithm is selected automatically during execution now. "
                "use_custom_all_reduce will be deprecated in future releases. ")
        self.nccl_plugin = dtype
        self.use_custom_all_reduce = use_custom_all_reduce
        if use_custom_all_reduce:
            init_all_reduce_helper()
        return self


cli_plugin_args = [
    # Plugins
    "bert_attention_plugin",
    "gpt_attention_plugin",
    "gemm_plugin",
    "gemm_swiglu_plugin",
    "lookup_plugin",
    "lora_plugin",
    "moe_plugin",
    "mamba_conv1d_plugin",
    "nccl_plugin",

    # Features
    "context_fmha",
    "context_fmha_fp32_acc",
    "paged_kv_cache",
    "remove_input_padding",
    "use_custom_all_reduce",
    "multi_block_mode",
    "enable_xqa",
    "tokens_per_block",
    "use_paged_context_fmha",
    "use_fp8_context_fmha",
    "multiple_profiles",
    "paged_state",
    "streamingllm",
    "reduce_fusion"
]


def add_plugin_argument(parser):
    plugin_config = PluginConfig()
    for field in fields(plugin_config):
        # Remove prefix "_" of the storage name
        field_name = field.name.lstrip('_')
        if field_name not in cli_plugin_args:
            continue
        if field.type in (str, Optional[str]):
            plugin_dtype_options = DEFAULT_PLUGIN_DTYPE_OPTIONS
            if field_name in PLUGIN_DTYPE_OPTIONS_MAP:
                plugin_dtype_options = PLUGIN_DTYPE_OPTIONS_MAP[field_name]
            parser.add_argument(
                "--" + field_name,
                type=str,
                default=field.default if field.default else "disable",
                choices=[x if x else "disable" for x in plugin_dtype_options],
                help=f"Whether to enable/disable {field_name} and the dtype.")
        elif field.type == bool:
            parser.add_argument(
                "--" + field_name,
                type=str,
                default="enable" if field.default else "disable",
                choices=["enable", "disable"],
                help=f"Whether to enable/disable {field_name}.")
        else:
            parser.add_argument("--" + field_name,
                                type=field.type,
                                default=field.default,
                                help=f"{field_name}.")
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
            visible to all GPUs. It should be 3 pointers per TP rank -
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
                             num_profiles: Optional[int] = None):
        from ..functional import Tensor
        workspace_size = self.POINTERS_PER_RANK * mapping.tp_size

        dim_range = None
        if num_profiles is not None:
            dim_range = OrderedDict([('all_reduce_size',
                                      [workspace_size] * num_profiles)])

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
        ipc_buffers_ping = IpcMemory(mapping, size * mapping.tp_size)
        ipc_buffers_pong = IpcMemory(mapping, size * mapping.tp_size)
        ipc_barriers_in = IpcMemory(
            mapping, IpcMemory.IPC_BARRIERS_SIZE_PER_GPU * mapping.tp_size * 2)
        ipc_barriers_out = IpcMemory(
            mapping, IpcMemory.IPC_BARRIERS_SIZE_PER_GPU * mapping.tp_size * 2)
        buffers = [
            ipc_buffers_ping,
            ipc_buffers_pong,
            ipc_barriers_in,
            ipc_barriers_out,
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
