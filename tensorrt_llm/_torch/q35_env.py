# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Qwen3.5 decode fast paths and the role profiles of the remaining TRTLLM_Q35_* switches.

The Qwen3.5 kernels and launch policies are tuned for the Qwen3.5-397B shapes on SM 10.3 (GB300). activate()
turns them on when the model being loaded is Qwen3.5 on SM 10.3; fast_paths_enabled() is the predicate the call
sites read.

Switches whose right value depends on the worker's role stay environment variables, default off.
TRTLLM_Q35_PROFILE selects a set of values for them that activate() applies to a Qwen3.5 model on SM 10.3:

* unset, empty or ``off``: nothing is applied.
* ``gen`` (generation / decode workers): the _GEN table, with or without attention DP.
* ``ctx`` (context / prefill workers): the _CTX table without attention DP; nothing with attention DP.

A switch the environment already sets keeps its value: set it to "0" to turn a path off. After activation,
activate() re-runs every registered reader so that modules which read their switches at import time see the new
values. Such modules wrap those reads in a function decorated with @register.
"""

import os
from typing import Callable, Literal

import torch

from tensorrt_llm.logger import logger

PROFILE_ENV = "TRTLLM_Q35_PROFILE"
_PROFILES = ("gen", "ctx")
_PROFILE_OFF = ("", "off")

# Generation (decode) workers. FUSED_AR_TPC and MOE_FRONT_BIG_C16 / _C16W act only beyond the row counts the
# defaults reach: fused all-reduce calls above one CTA wave (2 rows per SM) and MoE front steps above 256 tokens.
# TEMPERATURE_GRAPHS adds a CUDA graph capture pass so that temperature-only batches (requests without top-k /
# top-p / min-p / seed, e.g. OpenAI requests that set none) sample on the vocabulary-sharded logits.
_GEN = {
    "TRTLLM_Q35_CACHED_STREAM_EVENTS": "1",
    "TRTLLM_Q35_DRAFT_ARGMAX_LOGITS": "1",
    "TRTLLM_Q35_DRAFT_BOOKKEEPING": "1",
    "TRTLLM_Q35_DRAFT_FP8_HEAD": "1",
    "TRTLLM_Q35_DRAFT_TAIL": "1",
    "TRTLLM_Q35_FUSED_ACCEPT": "1",
    "TRTLLM_Q35_FUSED_AR_TPC": "1",
    "TRTLLM_Q35_FUSED_STEP0": "1",
    "TRTLLM_Q35_LL_ARGMAX_EXCHANGE": "1",
    "TRTLLM_Q35_MOE_FRONT_BIG_C16": "1",
    "TRTLLM_Q35_MOE_FRONT_BIG_C16W": "1",
    "TRTLLM_Q35_PDL_EARLY_TRIGGER": "1",
    "TRTLLM_Q35_POSITIONS_HOIST": "1",
    "TRTLLM_Q35_REPLICATED_EMBED": "1",
    "TRTLLM_Q35_SAMPLER_EVENT_RING": "1",
    "TRTLLM_Q35_SAMPLER_HOST_RING": "1",
    "TRTLLM_Q35_SAMPLER_ZEROCOPY": "1",
    "TRTLLM_Q35_SAMPLING_SCAN_CACHE": "1",
    "TRTLLM_Q35_SHARDED_TARGET_SAMPLING": "1",
    "TRTLLM_Q35_SKIP_MROPE_COS_SIN": "1",
    "TRTLLM_Q35_SPEC_PREPARE_CACHE": "1",
    "TRTLLM_Q35_TEMPERATURE_GRAPHS": "1",
}
# Context (prefill) workers without attention DP.
_CTX = {
    "TRTLLM_Q35_CTX_ATTN_PLAN": "1",
    "TRTLLM_Q35_CUSTOM_OP_EAGER": "1",
    "TRTLLM_Q35_GDN_GRAPH_DIRECT": "1",
    "TRTLLM_Q35_GDN_GRAPH_LEAN": "1",
    "TRTLLM_Q35_GDN_PREFILL_GRAPH": "1",
    "TRTLLM_Q35_SKIP_MROPE_COS_SIN": "1",
}

MODEL_TYPES = ("qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text")
SUPPORTED_SM = (10, 3)

_readers: list[Callable[[], None]] = []
_fast_paths = False
_profile_applied = False


def register(reader: Callable[[], None]) -> Callable[[], None]:
    """Decorator for a module's import-time reader of its Qwen3.5 settings: runs it now and on every change."""
    reader()
    _readers.append(reader)
    return reader


def _rerun_readers() -> None:
    for reader in _readers:
        reader()


def fast_paths_enabled() -> bool:
    """Whether the Qwen3.5 SM 10.3 kernels and launch policies are on for the model being loaded."""
    return _fast_paths


def set_fast_paths(enabled: bool) -> None:
    """Turn the Qwen3.5 fast paths on or off and re-run the registered readers (activate() and tests)."""
    global _fast_paths
    if _fast_paths != enabled:
        _fast_paths = enabled
        _rerun_readers()


def is_supported(model_type: str | None, capability: tuple[int, int] | None) -> bool:
    """Whether the fast paths apply to ``model_type`` on a GPU of compute ``capability``."""
    return model_type in MODEL_TYPES and capability == SUPPORTED_SM


def requested_profile() -> Literal["gen", "ctx"] | None:
    """The profile TRTLLM_Q35_PROFILE selects, or None when it is unset, empty or ``off``.

    Raises:
        ValueError: TRTLLM_Q35_PROFILE has any other value.
    """
    value = os.environ.get(PROFILE_ENV, "")
    if value in _PROFILE_OFF:
        return None
    if value not in _PROFILES:
        raise ValueError(
            f"{PROFILE_ENV}={value!r} is not a valid profile; valid values: "
            f"{', '.join(repr(v) for v in _PROFILES + _PROFILE_OFF[1:])} or unset"
        )
    return value


def profile_defaults(profile: Literal["gen", "ctx"], enable_attention_dp: bool) -> dict[str, str]:
    """The switch values ``profile`` gives a Qwen3.5 model on SM 10.3 when the environment does not set them.

    Args:
        profile: ``gen`` or ``ctx``.
        enable_attention_dp: Whether the worker runs with attention DP.
    """
    if profile == "gen":
        return dict(_GEN)
    if profile == "ctx":
        return {} if enable_attention_dp else dict(_CTX)
    raise ValueError(f"unknown profile {profile!r}")


def activate(model_config) -> None:
    """Turn the fast paths on for a Qwen3.5 model on SM 10.3 and apply the TRTLLM_Q35_PROFILE values.

    Call it after the model config is loaded and before the model is built: the profile depends on
    model_config.mapping, and the modules re-read their settings before any layer or kernel uses them.

    Args:
        model_config: The loaded ModelConfig (uses ``pretrained_config.model_type`` and
            ``mapping.enable_attention_dp``).

    Raises:
        ValueError: TRTLLM_Q35_PROFILE has an invalid value.
    """
    global _profile_applied
    profile = requested_profile()
    model_type = getattr(model_config.pretrained_config, "model_type", None)
    capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    supported = is_supported(model_type, capability)
    applied = []
    if profile is not None and not supported:
        logger.warning(
            f"{PROFILE_ENV}={profile} ignored: it applies to Qwen3.5 ({', '.join(MODEL_TYPES)}) on SM "
            f"{SUPPORTED_SM[0]}.{SUPPORTED_SM[1]}, got model_type {model_type!r} on "
            f"{'no GPU' if capability is None else f'SM {capability[0]}.{capability[1]}'}"
        )
    elif profile is not None and not _profile_applied:
        _profile_applied = True
        enable_attention_dp = model_config.mapping.enable_attention_dp
        values = profile_defaults(profile, enable_attention_dp)
        applied = [name for name in values if name not in os.environ]
        for name in applied:
            os.environ[name] = values[name]
        logger.info(
            f"{PROFILE_ENV}={profile} (attention DP: {enable_attention_dp}): {len(applied)} TRTLLM_Q35_* switches "
            f"set, {len(values) - len(applied)} already set in the environment"
        )
    if supported != _fast_paths:
        set_fast_paths(supported)
    elif applied:
        _rerun_readers()
    if supported:
        logger.info("Qwen3.5 SM 10.3 fast paths enabled")
