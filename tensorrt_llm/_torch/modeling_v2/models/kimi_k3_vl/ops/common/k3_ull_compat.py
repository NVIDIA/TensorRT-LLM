# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stand-ins for the ``vllm`` names the vendored MR8 decode code imports.

One section per ``vllm`` module the code imports (``_compat_envs`` and ``_compat_custom_ops``
stand in for ``vllm.envs`` and ``vllm._custom_ops``, which the code imports as modules):

* ``vllm.logger``: :func:`init_logger`, over ``tensorrt_llm.logger``.
* ``vllm.distributed``: the TP group the chain runs on; the host sets it with :func:`set_tp_group`
  before it builds any provider.
* ``vllm.platforms``: ``current_platform`` (CUDA only).
* ``vllm.config``: the enums the decode runtime imports and a config holding the one field it
  reads (``parallel_config.cp_kv_cache_interleave_size``).
* ``vllm.forward_context``: no forward context (the chain builds its steps directly).
* ``vllm.utils.math_utils``, ``vllm.triton_utils``.
* ``vllm.model_executor``: the B1 / B2 skinny GEMM (inert: the chain runs it at no M <= 2),
  ``UnfinalizedMoEOutput`` (a type the code tests against, never produced here) and the KDA conv
  state layout (dim first).

Names of stock-vLLM-only paths (``GroupedTopKRouter``, ``moe_runner._unpack``,
``MambaStateDtypeCalculator``) are not defined: importing them raises ``ImportError``.
"""

import enum
from types import SimpleNamespace

import torch
import torch.distributed as dist
import triton  # noqa: F401 (vllm.triton_utils)
import triton.language as tl  # noqa: F401 (vllm.triton_utils)

from tensorrt_llm.logger import logger as _trtllm_logger

# vllm.logger ------------------------------------------------------------------------------------

_seen = set()


class _Logger:
    """A vLLM-style logger: printf-style arguments, ``*_once`` variants, ``scope`` ignored."""

    def __init__(self, name):
        self.name = name

    @staticmethod
    def _emit(severity, msg, args):
        _trtllm_logger.log(severity, msg % args if args else msg)

    def _once(self, severity, msg, *args, **kwargs):
        key = (severity, msg, args)
        if key in _seen:
            return
        _seen.add(key)
        self._emit(severity, msg, args)

    def debug(self, msg, *args, **kwargs):
        self._emit(_trtllm_logger.VERBOSE, msg, args)

    def info(self, msg, *args, **kwargs):
        self._emit(_trtllm_logger.INFO, msg, args)

    def warning(self, msg, *args, **kwargs):
        self._emit(_trtllm_logger.WARNING, msg, args)

    def error(self, msg, *args, **kwargs):
        self._emit(_trtllm_logger.ERROR, msg, args)

    def debug_once(self, msg, *args, **kwargs):
        self._once(_trtllm_logger.VERBOSE, msg, *args)

    def info_once(self, msg, *args, **kwargs):
        self._once(_trtllm_logger.INFO, msg, *args)

    def warning_once(self, msg, *args, **kwargs):
        self._once(_trtllm_logger.WARNING, msg, *args)


def init_logger(name):
    return _Logger(name)


# vllm.distributed -------------------------------------------------------------------------------


class GroupCoordinator:
    def __init__(self, group, device):
        self.device_group = group
        self.world_size = dist.get_world_size(group)
        self.rank_in_group = dist.get_rank(group)
        self.rank = dist.get_rank()
        self.device = device
        self.unique_name = "tp:0"


_TP = None


def set_tp_group(group, device):
    global _TP
    _TP = GroupCoordinator(group, device)
    return _TP


def get_tp_group():
    if _TP is None:
        raise RuntimeError("K3 chain: TP group not set (set_tp_group)")
    return _TP


def get_dcp_group():
    return get_tp_group()


def tensor_model_parallel_all_reduce(x):
    dist.all_reduce(x, group=get_tp_group().device_group)
    return x


# vllm.platforms ---------------------------------------------------------------------------------


class _Cuda:
    device_type = "cuda"
    device_name = "cuda"

    @staticmethod
    def current_device():
        return torch.cuda.current_device()

    @staticmethod
    def is_rocm():
        return False

    @staticmethod
    def is_cuda():
        return True

    @staticmethod
    def get_device_capability(device_id=0):
        return torch.cuda.get_device_capability(device_id)


current_platform = _Cuda()

# vllm.config ------------------------------------------------------------------------------------


class CUDAGraphMode(enum.Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = 3
    FULL_AND_PIECEWISE = 4


class CompilationMode(enum.Enum):
    NONE = 0
    VLLM_COMPILE = 3


class SpeculativeConfig:
    pass


_CONFIG = SimpleNamespace(parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1))


def get_current_vllm_config():
    return _CONFIG


# vllm.forward_context ---------------------------------------------------------------------------


def is_forward_context_available():
    return False


def get_forward_context():
    raise RuntimeError("K3 chain: no forward context")


# vllm.utils.math_utils --------------------------------------------------------------------------


def cdiv(a, b):
    return -(a // -b)


def round_up(x, y):
    return ((x + y - 1) // y) * y


# vllm.triton_utils ------------------------------------------------------------------------------

tldevice = None

# vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm ----------------------------------------


class SkinnyGemmConfig:
    def __init__(self, *args):
        self.args = args


def shape_dynamic_skinny_gemm(x, weight, cfg, bias):
    if x.shape[0] > 2:
        raise RuntimeError("skinny GEMM stand-in called at M > 2")
    return None


# vllm.model_executor.layers.fused_moe.moe_output ------------------------------------------------


class UnfinalizedMoEOutput:
    pass


# vllm.model_executor.layers.mamba.mamba_utils ---------------------------------------------------


def is_conv_state_dim_first():
    return True
