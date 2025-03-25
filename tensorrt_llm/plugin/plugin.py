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
import os
import platform
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, fields
from enum import IntEnum
from pathlib import Path
from textwrap import dedent
from typing import List, Optional, Tuple

import tensorrt as trt
import torch

from .._ipc_utils import IpcMemory, can_access_peer
from .._utils import get_sm_version
from ..bindings.internal.runtime import lamport_initialize
from ..logger import logger
from ..mapping import Mapping

TRT_LLM_PLUGIN_NAMESPACE = 'tensorrt_llm'


def plugin_lib_path() -> str:
    project_dir = Path(__file__).parent.parent.absolute()
    dyn_lib = "libnvinfer_plugin_tensorrt_llm.so" if platform.system(
    ) != "Windows" else "nvinfer_plugin_tensorrt_llm.dll"
    return str(project_dir.joinpath("libs", dyn_lib))


def _load_plugin_lib():
    on_windows = platform.system() == "Windows"
    winmode = 0 if on_windows else None
    handle = ctypes.CDLL(plugin_lib_path(),
                         mode=ctypes.RTLD_GLOBAL,
                         winmode=winmode)
    try:
        handle.initTrtLlmPlugins.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        handle.initTrtLlmPlugins.restype = ctypes.c_bool
    except AttributeError as err:
        raise ImportError('TensorRT-LLM Plugin is unavailable') from err

    try:
        assert handle.initTrtLlmPlugins(
            None, TRT_LLM_PLUGIN_NAMESPACE.encode('utf-8'))
    except OSError as e:
        windows_err = """
        The error above may be caused by an outdated Microsoft Visual C++ Redistributable Version.
        Please install the latest MSVC from the link below and re-launch.

        https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#latest-microsoft-visual-c-redistributable-version
        """
        err_msg = dedent(windows_err if on_windows else "Unknown error")
        raise RuntimeError(err_msg) from e
    except Exception as e:
        raise e


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
    ["auto", "float16", "float32", "bfloat16", "int32", "fp8", "nvfp4", None],
    "low_latency_gemm_plugin": ["fp8", None],
    "low_latency_gemm_swiglu_plugin": ["fp8", None],
    "gemm_allreduce_plugin": ["float16", "bfloat16", None]
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
        * "float16"/"bfloat16"/"float32"/"int32", which means the plugin is enabled with the specified precision; (Some plugins only support limited dtype, i.e., gemm_swiglu_plugin and low_latency_gemm_swiglu_plugin only supports fp8 now)
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
    _bert_attention_plugin: Optional[str] = field(
        default="auto",
        init=False,
        metadata={
            "help":
            "The plugin that uses efficient kernels and enables an in-place update of the KV cache for attention layer of BERT-like encoder models."
        })
    _gpt_attention_plugin: Optional[str] = field(
        default="auto",
        init=False,
        metadata={
            "help":
            "The plugin that uses efficient kernels and enables an in-place update of the KV cache for attention layer of GPT-like decoder models."
        })
    _gemm_plugin: Optional[str] = field(
        default=None,
        init=False,
        metadata={
            "help":
            "The GEMM plugin that utilizes NVIDIA cuBLASLt to perform GEMM operations. "
            "Note: it's only affective for non-quantized gemm operations (except FP8)."
            "Note: For FP8, it also requires same calibration in checkpoint."
        })
    _explicitly_disable_gemm_plugin: bool = False
    _gemm_swiglu_plugin: Optional[str] = field(
        default=None,
        init=False,
        metadata={
            "help":
            "The GEMM + SwiGLU fusion in Gated-MLP combines two Matmul operations and "
            "one SwiGLU operation into a single kernel. Currently this is only supported for FP8 precision on Hopper."
        })
    _fp8_rowwise_gemm_plugin: Optional[str] = field(
        default=None,
        init=False,
        metadata={
            "help":
            "The quantized GEMM for fp8, which uses per token dynamic scales for "
            "activation and per channel static scales for weights."
            "Note: It also requires same calibration in checkpoint."
        })
    _qserve_gemm_plugin: Optional[str] = field(
        default=None,
        init=False,
        metadata={
            "help":
            "The quantized GEMM from [QServe](https://arxiv.org/abs/2405.04532), "
            "which employs 4-bit quantization for weights and 8-bit quantization for activations."
        })
    _identity_plugin: Optional[str] = field(
        default=None,
        init=False,
        metadata={
            "help":
            "The identity plugin simply copies inputs to outputs, it's used mostly for debugging purpose."
        })
    _nccl_plugin: Optional[str] = field(
        default="auto",
        init=False,
        metadata={
            "help":
            "The NCCL plugin wraps NCCL operators to support multi-GPU and even multi-nodes."
        })
    _lora_plugin: Optional[str] = field(default=None,
                                        init=False,
                                        metadata={"help": "Enable LoRA."})
    _dora_plugin: bool = field(default=False,
                               init=False,
                               metadata={"help": "Enable DoRA."})
    _weight_only_groupwise_quant_matmul_plugin: Optional[str] = field(
        default=None,
        init=False,
        metadata={
            "help":
            "Enable weight-only groupwise quantization matmul operators."
        })
    _weight_only_quant_matmul_plugin: Optional[str] = field(
        default=None,
        init=False,
        metadata={"help": "Enable weight-only quantization matmul operators."})
    _smooth_quant_plugins: bool = field(
        default=True,
        init=False,
        metadata={
            "help": "Enable a group of plugins to support smooth quantization."
        })
    _smooth_quant_gemm_plugin: Optional[str] = field(
        default=None,
        init=False,
        metadata={
            "help":
            "Enable plugin that supports smooth quantization gemm kernels."
        })
    _layernorm_quantization_plugin: Optional[str] = field(
        default=None,
        init=False,
        metadata={
            "help":
            "Enable plugin that supports layernorm quantization kernels."
        })
    _rmsnorm_quantization_plugin: Optional[str] = field(
        default=None,
        init=False,
        metadata={
            "help": "Enable plugin that supports rmsnorm quantization kernels."
        })
    _quantize_per_token_plugin: bool = field(
        default=False,
        init=False,
        metadata={
            "help": "Enable plugin that supports per-token quantization."
        })
    _quantize_tensor_plugin: bool = field(
        default=False,
        init=False,
        metadata={
            "help": "Enable plugin that supports per-tensor quantization."
        })
    _moe_plugin: Optional[str] = field(
        default="auto",
        init=False,
        metadata={
            "help":
            "Enable some customized kernels to speed up the MoE layer of MoE models."
        })
    _mamba_conv1d_plugin: Optional[str] = field(
        default="auto",
        init=False,
        metadata={
            "help":
            "Enable customized kernels to speed up conv1d operator for Mamba."
        })
    _low_latency_gemm_plugin: Optional[str] = field(
        default=None,
        init=False,
        metadata={
            "help":
            "The GEMM plugin that optimized specially for low latency scenarios."
        })
    _low_latency_gemm_swiglu_plugin: Optional[str] = field(
        default=None,
        init=False,
        metadata={
            "help":
            "The GEMM + SwiGLU fusion plugin that optimized specially for low latency scenarios."
        })

    _gemm_allreduce_plugin: Optional[str] = field(
        default=None,
        init=False,
        metadata={"help": "The GEMM + AllReduce kernel fusion plugin."})

    # Features
    _context_fmha: bool = field(
        default=True,
        init=False,
        metadata={
            "help":
            "Enable the fused multi-head attention during the context phase, "
            "will trigger a kernel that performs the MHA/MQA/GQA block using a single kernel."
        })
    _bert_context_fmha_fp32_acc: bool = field(
        default=False,
        init=False,
        metadata={
            "help":
            "Enable the FP32 accumulator for context FMHA in the bert_attention_plugin. "
            "If disabled, FP16 is used, better performance but potentially worse accuracy is expected."
        })
    _paged_kv_cache: Optional[bool] = field(
        default=None,
        init=False,
        metadata={
            "help":
            "Enable paged KV cache, which helps manage memory for the KV cache more efficiently, "
            "and usually leads to an increase in the batch size and an improved efficiency."
        })
    _remove_input_padding: bool = field(
        default=True,
        init=False,
        metadata={
            "help":
            "Pack different tokens together, which reduces both the amount of computations and memory consumption."
        })
    _norm_quant_fusion: bool = field(
        default=False,
        init=False,
        metadata={
            "help":
            "Fuse the LayerNorm and quantization kernels into a single kernel, "
            "resulting in improved end-to-end performance."
        })
    _reduce_fusion: bool = field(
        default=False,
        init=False,
        metadata={
            "help":
            "Fuse the ResidualAdd and LayerNorm kernels after AllReduce into a single kernel, "
            "resulting in improved end-to-end performance."
        })
    _user_buffer: bool = field(
        default=False,
        init=False,
        metadata={
            "help":
            "Eliminate extra copies from the local buffer to the shared buffer "
            "in the communication kernel, leading to improved end-to-end performance. "
            "This feature must be enabled with `--reduce_fusion enable` and "
            "is currently only supported for the FP8 LLAMA model."
        })
    _tokens_per_block: int = field(
        default=32,
        init=False,
        metadata={
            "help":
            "Define how many tokens are contained in each paged kv cache block."
        })
    _use_paged_context_fmha: bool = field(
        default=True,
        init=False,
        metadata={
            "help":
            "Allow advanced features like KV cache reuse and chunked context."
        })
    _use_fp8_context_fmha: bool = field(
        default=True,
        init=False,
        metadata={
            "help":
            "When FP8 quantization is activated, the attention can be further accelerated by enabling FP8 Context FMHA"
        })
    _fuse_fp4_quant: bool = field(
        default=False,
        init=False,
        metadata={
            "help": "Whether to fuse FP4 quantization into attention kernel."
        })
    _multiple_profiles: bool = field(
        default=False,
        init=False,
        metadata={
            "help":
            "Enables multiple TensorRT optimization profiles in the built engines, "
            "will benefits the performance especially when GEMM plugin is disabled, "
            "because more optimization profiles help TensorRT have more chances to select better kernels. "
            "Note: This feature increases engine build time but no other adverse effects are expected."
        })
    _paged_state: bool = field(
        default=True,
        init=False,
        metadata={
            "help":
            "Enable paged state, which helps manage memory for the RNN state more efficiently."
        })
    _streamingllm: bool = field(
        default=False,
        init=False,
        metadata={
            "help":
            "Enable [StreamingLLM](https://arxiv.org/abs/2309.17453), which uses a window attention to perform efficient and stable LLM on long texts."
        })
    _manage_weights: bool = field(
        default=False,
        init=False,
        metadata={
            "help":
            "Enable TensorRT-LLM managed weights to speed up engine building process."
        })
    _use_fused_mlp: bool = field(
        default=True,
        init=False,
        metadata={
            "help":
            "Enable horizontal fusion in Gated-MLP that combines two Matmul "
            "operations into a single one followed by a separate SwiGLU kernel."
        })
    _pp_reduce_scatter: bool = field(
        default=False,
        init=False,
        metadata={
            "help":
            "Enable a pipeline parallelism optimization with "
            "ReduceScatter + AllGather targeting large MoE models."
        })

    def update_from_dict(self, config: dict):
        for name in config.keys():
            if hasattr(self, name):
                value_to_be_update = config[name]
                if isinstance(getattr(self, name),
                              bool) or name == 'paged_kv_cache':
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
        args = vars(args)
        obj = cls.from_dict(args)

        # We want to know if the user explicitly disabled the gemm_plugin
        # because nvfp4 gemm uses plugin by default currently
        if 'gemm_plugin' in args and args['gemm_plugin'] == 'disable':
            obj._explicitly_disable_gemm_plugin = True

        return obj

    def to_dict(self):
        config = asdict(self)
        # Remove prefix "_" of the storage name
        config = {key.lstrip('_'): value for key, value in config.items()}
        return config

    def to_legacy_setting(self):
        '''Legacy setting means that all of the plugins and features are
        disabled, this is needed for the legacy `build.py` script, which will be
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
            elif field.type == bool or field_name == 'paged_kv_cache':
                setattr(self, field_name, False)

    def validate(self):
        unsupported_plugins = {
            # bert_attention_plugin is handled within BertAttention
            100: [
                'gemm_swiglu_plugin', 'fp8_rowwise_gemm_plugin',
                'low_latency_gemm_plugin', 'low_latency_gemm_swiglu_plugin',
                'bert_context_fmha_fp32_acc'
            ]
        }
        sm = get_sm_version()
        if sm in unsupported_plugins:
            for plugin in unsupported_plugins[sm]:
                val = getattr(self, plugin, None)
                if val is not None and val != False:
                    raise NotImplementedError(
                        f"{plugin}={val} is not supported on SM {sm}.")

    @property
    def context_fmha_type(self):
        if self.bert_context_fmha_fp32_acc:
            return ContextFMHAType.enabled_with_fp32_acc
        elif self.context_fmha:
            return ContextFMHAType.enabled
        else:
            return ContextFMHAType.disabled

    def is_context_fmha_enabled(self):
        return self.context_fmha_type != ContextFMHAType.disabled

    @context_fmha_type.setter
    def context_fmha_type(self, value):
        if value == ContextFMHAType.disabled:
            self.context_fmha = False
            self.bert_context_fmha_fp32_acc = False
        else:
            self.context_fmha = True
            if value == ContextFMHAType.enabled:
                self.bert_context_fmha_fp32_acc = False
            elif value == ContextFMHAType.enabled_with_fp32_acc:
                self.bert_context_fmha_fp32_acc = True

    def set_smooth_quant_plugins(self, dtype: str = "auto"):
        self.smooth_quant_gemm_plugin = dtype
        self.rmsnorm_quantization_plugin = dtype
        self.layernorm_quantization_plugin = dtype
        self.quantize_per_token_plugin = True
        self.quantize_tensor_plugin = True
        return self

    def set_qserve_plugins(self, dtype: str = "auto"):
        self.qserve_gemm_plugin = dtype
        self.rmsnorm_quantization_plugin = dtype
        self.quantize_per_token_plugin = True
        return self

    def set_fp8_rowwise_quant_plugins(self, dtype: str = "auto"):
        self.fp8_rowwise_gemm_plugin = dtype
        self.rmsnorm_quantization_plugin = dtype
        # self.layernorm_quantization_plugin = dtype
        self.quantize_per_token_plugin = True
        self.quantize_tensor_plugin = True
        return self

    def set_context_fmha(self, context_fmha_type=ContextFMHAType.enabled):
        assert type(context_fmha_type) == ContextFMHAType
        self.context_fmha_type = context_fmha_type
        return self

    def enable_paged_kv_cache(self, tokens_per_block: int = 32):
        self.paged_kv_cache = True
        self.tokens_per_block = tokens_per_block
        return self

    def set_nccl_plugin(self, dtype: str = "auto"):
        self.nccl_plugin = dtype
        init_all_reduce_helper()
        return self

    def set_lora_plugin(self, dtype: str = None):
        self.lora_plugin = dtype
        return self

    def set_dora_plugin(self, enable: bool = False):
        self.dora_plugin = enable
        return self


# Only plugin configs in this list will be exposed as `trtllm-build` arguments,
# others are automatically enabled when needed, no need for users to control.
cli_plugin_args = [
    # Plugins
    "bert_attention_plugin",
    "gpt_attention_plugin",
    "gemm_plugin",
    "gemm_swiglu_plugin",
    "fp8_rowwise_gemm_plugin",
    "lora_plugin",
    "dora_plugin",
    "moe_plugin",
    "mamba_conv1d_plugin",
    "nccl_plugin",
    "low_latency_gemm_plugin",
    "low_latency_gemm_swiglu_plugin",
    "gemm_allreduce_plugin",

    # Features
    "context_fmha",
    "bert_context_fmha_fp32_acc",
    "remove_input_padding",
    "tokens_per_block",
    "use_paged_context_fmha",
    "use_fp8_context_fmha",
    "fuse_fp4_quant",
    "multiple_profiles",
    "paged_state",
    "streamingllm",
    "norm_quant_fusion",
    "reduce_fusion",
    "user_buffer",
    "use_fused_mlp",
    "pp_reduce_scatter",
]


def add_plugin_argument(parser: argparse.ArgumentParser):
    plugin_config = PluginConfig()
    for field in fields(plugin_config):
        # Remove prefix "_" of the storage name
        field_name = field.name.lstrip('_')
        if field_name not in cli_plugin_args:
            continue
        if field.metadata and "help" in field.metadata:
            help_message = field.metadata["help"]
        else:
            raise AttributeError(f"Please add help message for {field_name}.")
        if field.type in (str, Optional[str]):
            plugin_dtype_options = DEFAULT_PLUGIN_DTYPE_OPTIONS
            if field_name in PLUGIN_DTYPE_OPTIONS_MAP:
                plugin_dtype_options = PLUGIN_DTYPE_OPTIONS_MAP[field_name]
            if field_name == "gemm_plugin":
                default = field.default
            else:
                default = field.default if field.default else "disable"
            parser.add_argument(
                "--" + field_name,
                type=str,
                default=default,
                choices=[x if x else "disable" for x in plugin_dtype_options],
                help=help_message)
        elif field.type == bool:
            parser.add_argument(
                "--" + field_name,
                type=str,
                default="enable" if field.default else "disable",
                choices=["enable", "disable"],
                help=help_message)
        else:
            parser.add_argument("--" + field_name,
                                type=field.type,
                                default=field.default,
                                help=help_message)
    return parser


def force_all_reduce_deterministic():
    return os.getenv("FORCE_DETERMINISTIC", "0") == "1" or os.getenv(
        "FORCE_ALL_REDUCE_DETERMINISTIC", "0") == "1"


class CustomAllReduceHelper:
    """
        Globally visible class to help usage of custom_all_reduce plugin.
        Provides the following utilities:

        workspace: Tensor
            When using CUSTOM or AUTO mode, a tensor containing pointers to memory
            visible to all GPUs. It should be 3 pointers per TP rank -
            ptr to data buffer, ptr to barriers in, ptr to barriers out.
            It must be initialized using IpcMemory class.

        Usage:
            - Set custom_all_reduce_helper.workspace with the required tensor.
              Then, each instance of allreduce will reference that tensor automatically.
    """
    POINTERS_PER_RANK = 4
    POINTERS_OF_COUNTER = 1

    def __init__(self) -> None:
        self.workspace: Optional[Tensor] = None

    def set_workspace_tensor(self,
                             mapping: Mapping,
                             num_profiles: Optional[int] = None):
        from ..functional import Tensor
        workspace_size = self.POINTERS_PER_RANK * mapping.tp_size + self.POINTERS_OF_COUNTER

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
    def max_workspace_size_auto(tp_size: int,
                                support_deterministic=True) -> int:
        if force_all_reduce_deterministic() and support_deterministic:
            workspace_size = os.getenv("FORCE_ALLREDUCE_KERNEL_WORKSPACE_SIZE",
                                       "1000000000")
            return int(workspace_size)
        if tp_size <= 2:
            return 16_000_000
        return 8_000_000

    @staticmethod
    def allocate_workspace(mapping: Mapping,
                           size: int) -> Tuple[List[IpcMemory], torch.Tensor]:
        is_p2p_supported = can_access_peer(mapping)

        force_deterministic = force_all_reduce_deterministic()

        ipc_buffers_size = 1 if force_deterministic else size * mapping.tp_size
        ipc_buffers_ping = IpcMemory(mapping, ipc_buffers_size,
                                     is_p2p_supported)
        ipc_buffers_pong = IpcMemory(mapping, ipc_buffers_size,
                                     is_p2p_supported)
        ipc_barriers = IpcMemory(
            mapping, IpcMemory.IPC_BARRIERS_SIZE_PER_GPU * mapping.tp_size * 2 *
            mapping.tp_size, is_p2p_supported)
        lamport_buffers_size = size if force_all_reduce_deterministic else size * mapping.tp_size
        lamport_buffers = IpcMemory(mapping, 3 * lamport_buffers_size,
                                    is_p2p_supported)
        if is_p2p_supported:
            lamport_initialize(
                lamport_buffers.local_ptr,
                3 * lamport_buffers_size,
            )
        # [counter, oneshot_flag_value, lamport_flag_value, lamport_comm_size, clear_size]
        flag_buffer = torch.tensor([0, 0, 0, lamport_buffers_size, 0],
                                   dtype=torch.uint32,
                                   device="cuda")
        buffers = [
            ipc_buffers_ping, ipc_buffers_pong, ipc_barriers, lamport_buffers,
            flag_buffer
        ]

        return buffers, torch.tensor(
            ipc_buffers_ping.serialize() + ipc_buffers_pong.serialize() +
            ipc_barriers.serialize() + lamport_buffers.serialize() +
            [flag_buffer.data_ptr()],
            dtype=torch.int64,
            device="cuda")


custom_all_reduce_helper = None


def init_all_reduce_helper():
    global custom_all_reduce_helper
    custom_all_reduce_helper = CustomAllReduceHelper()


def current_all_reduce_helper():
    global custom_all_reduce_helper
    assert custom_all_reduce_helper is not None, "You must call `init_all_reduce_helper` first"
    return custom_all_reduce_helper
