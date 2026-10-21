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
"""Role profiles for the Qwen3.5 TRTLLM_Q35_* switches.

Every TRTLLM_Q35_* switch defaults to off in the code. TRTLLM_Q35_PROFILE selects a set of values for them that
activate() applies when the model being loaded is Qwen3.5 on SM103 (GB300):

* unset, empty or ``off``: nothing is applied.
* ``gen`` (generation / decode workers): the _GEN and _SM103 tables, with or without attention DP.
* ``ctx`` (context / prefill workers): the _CTX table without attention DP; nothing with attention DP.

A switch the environment already sets keeps its value: set it to "0" to turn a path off. After applying a profile,
activate() re-runs every registered reader so that modules which read their switches at import time see the new values.
Modules that read TRTLLM_Q35_* switches at import time wrap those reads in a function decorated with @register.
"""

import os
from typing import Callable, Literal

import torch

from tensorrt_llm.logger import logger

PROFILE_ENV = "TRTLLM_Q35_PROFILE"
_PROFILES = ("gen", "ctx")
_PROFILE_OFF = ("", "off")

# Generation (decode) workers.
_GEN = {
    "TRTLLM_Q35_SE_AUX": "gate",
    "TRTLLM_Q35_LL_ARGMAX_EXCHANGE": "1",
    "TRTLLM_Q35_AR_FP8_OUT": "1",
    "TRTLLM_Q35_AR_NORM_FUSE": "1",
    "TRTLLM_Q35_REPLICATED_EMBED": "1",
    "TRTLLM_Q35_LOWM_FP8": "1",
    "TRTLLM_Q35_LOWM_GDN_BA": "1",
    "TRTLLM_Q35_LOWM_BF16": "1",
    "TRTLLM_Q35_DRAFT_FP8_HEAD": "1",
    "TRTLLM_Q35_DRAFT_SLICES": "1",
    "TRTLLM_Q35_FUSED_ACCEPT": "1",
    "TRTLLM_Q35_SHARDED_TARGET_SAMPLING": "1",
    "TRTLLM_Q35_GDN_REPLAY_TILE": "1",
    "TRTLLM_Q35_GDN_ARANGE_CACHE": "1",
    "TRTLLM_Q35_GDN_NORM_PDL": "1",
    "TRTLLM_Q35_GDN_PROMOTE_FUSED": "1",
    "TRTLLM_Q35_GDN_CONV_PDL": "1",
    "TRTLLM_Q35_GDN_CONV_BARRIER": "1",
    "TRTLLM_Q35_SKIP_MROPE_COS_SIN": "1",
    "TRTLLM_Q35_ATTN_SEGMENTS": "16",
    "TRTLLM_Q35_ATTN_SEG_MAX_BATCH": "2",
    "TRTLLM_Q35_ATTN_SEG_FUSE_GATE": "1",
    "TRTLLM_Q35_STEADY_PREPARE": "1",
    "TRTLLM_Q35_STEADY_PREPARE_ATTN": "1",
    "TRTLLM_Q35_STEADY_PREPARE_STATIC": "1",
    "TRTLLM_Q35_STEADY_MAMBA_INDICES": "1",
    "TRTLLM_Q35_SAMPLER_HOST_RING": "1",
    "TRTLLM_Q35_FAST_COUNT_BCAST": "1",
    "TRTLLM_Q35_SAMPLING_PARAMS_CACHE": "1",
    "TRTLLM_Q35_LEAN_ITER_LOG": "16",
    "TRTLLM_Q35_MNNVL_PRESIZE": "1",
    "TRTLLM_Q35_FUSED_AR": "all",
    "TRTLLM_Q35_FIN_FOLD": "1",
    "TRTLLM_Q35_EMBED_GATHER": "1",
    "TRTLLM_Q35_LOWM_SE": "1",
    "TRTLLM_Q35_PDL_EARLY_TRIGGER": "1",
    "TRTLLM_Q35_MOE_ROUTER_GEMM": "1",
    "TRTLLM_Q35_ATTN_CP": "1",
    "TRTLLM_Q35_ATTN_CP_DRAFT_EARLY": "1",
    "TRTLLM_Q35_GDN_CONV_HOIST": "1",
    "TRTLLM_Q35_GDN_FUSE": "4",
    "TRTLLM_Q35_GDN_FUSE_OPTS": "w4_s_p_t2_x2",
    "TRTLLM_Q35_FUSED_STEP0": "1",
    "TRTLLM_Q35_DRAFT_BOOKKEEPING": "1",
    "TRTLLM_Q35_DRAFT_TAIL": "1",
    "TRTLLM_Q35_MTP_FUSED_INPUT": "1",
    "TRTLLM_Q35_STEADY_PREPARE_SKIP_BLOCKS": "1",
    "TRTLLM_Q35_SPEC_PREPARE_CACHE": "1",
    "TRTLLM_Q35_STEADY_PREPARE_ZEROCOPY": "1",
    "TRTLLM_Q35_SAMPLER_ZEROCOPY": "1",
    "TRTLLM_Q35_SAMPLER_EVENT_RING": "1",
    "TRTLLM_Q35_CACHED_STREAM_EVENTS": "1",
    "TRTLLM_Q35_SAMPLING_SCAN_CACHE": "1",
    "TRTLLM_Q35_HANG_CHECKPOINT_MIN_S": "1",
    "TRTLLM_Q35_ATTN_FUSED_DECODE": "1",
    "TRTLLM_Q35_ATTN_KV_APPEND": "1",
    "TRTLLM_Q35_MOE_PREROUTE": "1",
    "TRTLLM_Q35_MOE_PREROUTE_LOWM": "1",
    "TRTLLM_Q35_GEMM_AR": "1",
    "TRTLLM_Q35_AUTOTUNE_DUMP": "1",
    "TRTLLM_Q35_AR_NO_NCCL_SYMM": "1",
    "TRTLLM_Q35_GEMM_AR_MIN_ROWS": "1",
    "TRTLLM_Q35_MOE_FRONT_FUSED": "1",
    "TRTLLM_Q35_GDN_CORE": "1",
    "TRTLLM_Q35_MTP_MOE_QUANT": "nvfp4",
    "TRTLLM_Q35_MTP_FC_CUTE_AR": "1",
    "TRTLLM_Q35_AR2": "slm",
    "TRTLLM_Q35_ATTN_FOLD": "1",
    "TRTLLM_Q35_ATTN_CP_FUSED": "1",
    "TRTLLM_Q35_ATTN_CP_MAX_BATCH": "64",
    "TRTLLM_Q35_CP_EXCHANGE_KB": "1024",
    "TRTLLM_Q35_LL_EXCHANGE_MAX_ROWS": "512",
    "TRTLLM_Q35_FUSED_AR_MAX_ROWS": "256",
    "TRTLLM_Q35_FUSED_AR_TWOSHOT_ROWS": "64",
    "TRTLLM_Q35_ATTN_CP_FUSED_MAX_BATCH": "32",
    "TRTLLM_Q35_GDN_CORE_LARGE": "1",
    "TRTLLM_Q35_GDN_FUSE_MAX_BATCH": "64",
    "TRTLLM_Q35_GDN_REPLAY_PARTITION_MIN_BATCH": "65",
    "TRTLLM_Q35_CP_MERGE_GRID": "8",
    "TRTLLM_Q35_CP_GROUP_GRID": "2",
    "TRTLLM_Q35_CP_MERGE_MAX_ROWS": "2048",
    "TRTLLM_Q35_MOE_FRONT_BIG": "1",
    "TRTLLM_Q35_ATTN_CP_BALANCED_SPLIT": "1",
    "TRTLLM_Q35_ATTN_CP_FUSED_MAX_BATCH_Q1": "32",
    "TRTLLM_Q35_GDN_CORE_MIN_BATCH": "5",
    "TRTLLM_Q35_GDN_BA_LATE_JOIN": "1",
    "TRTLLM_Q35_POSITIONS_HOIST": "1",
}
# Context (prefill) workers without attention DP.
_CTX = {
    "TRTLLM_Q35_BRANCH_SNAPSHOT": "1",
    "TRTLLM_Q35_CTX_ATTN_PLAN": "1",
    "TRTLLM_Q35_CTX_LEAN_TOKENS": "1",
    "TRTLLM_Q35_CTX_MTP_GRAPH": "1",
    "TRTLLM_Q35_CUSTOM_OP_EAGER": "1",
    "TRTLLM_Q35_GDN_GRAPH_DIRECT": "1",
    "TRTLLM_Q35_GDN_GRAPH_LEAN": "1",
    "TRTLLM_Q35_GDN_PREFILL_GRAPH": "1",
    "TRTLLM_Q35_AR_NO_NCCL_SYMM": "1",
    "TRTLLM_Q35_SKIP_MROPE_COS_SIN": "1",
}
# Generation workers: MoE tactic pins and fused-front row counts, tuned for the Qwen3.5-397B shapes on SM103.
_SM103 = {
    "TRTLLM_Q35_MOEGEMM_PIN": "|".join(
        (
            "512x64:7=8/70,14=8/70,21=8/70,28=8/70",
            "512x128:3=8/59,5=8/59,6=8/70,7=8/70,8=8/48,9=8/48,10=8/70,11=8/70,12=8/70,13=8/70,14=8/48,15=8/70,"
            "16=8/4,18=8/4,20=8/4,21=8/4,22=8/4,24=8/48,26=8/4,28=8/4,30=8/4,32=8/4,35=8/4,42=8/48,49=8/4,56=8/2,"
            "63=8/70,70=16/20,77=16/20,84=8/48,91=16/20,98=16/20,105=16/20,112=8/70,126=16/20,140=16/20,154=16/20,"
            "168=16/20,182=16/20,196=16/20,210=16/20,224=16/20",
            "256x256:7=8/55,14=8/66,28=8/66",
            "256x128:7=8/59,14=8/44,28=8/66",
        )
    ),
    "TRTLLM_Q35_MOE_FRONT_BIG_MS": ",".join(
        str(rows) for rows in range(35, 225, 7)
    ),  # 35..224, every 7 rows
}

_MODEL_TYPES = ("qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text")
_SUPPORTED_SM = (10, 3)

_readers: list[Callable[[], None]] = []
_active = False


def register(reader: Callable[[], None]) -> Callable[[], None]:
    """Decorator for a module's reader of its import-time TRTLLM_Q35_* values: runs it now and in activate()."""
    reader()
    _readers.append(reader)
    return reader


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
    """The switch values ``profile`` gives a Qwen3.5 model on SM103 when the environment does not set them.

    Args:
        profile: ``gen`` or ``ctx``.
        enable_attention_dp: Whether the worker runs with attention DP.
    """
    if profile == "gen":
        return {**_GEN, **_SM103}
    if profile == "ctx":
        return {} if enable_attention_dp else dict(_CTX)
    raise ValueError(f"unknown profile {profile!r}")


def activate(model_config) -> None:
    """Apply the TRTLLM_Q35_PROFILE values if model_config is a Qwen3.5 model on SM103 (once per process).

    Call it after the model config is loaded and before the model is built: the ctx profile depends on
    model_config.mapping, and the modules re-read their switches before any layer or kernel uses them.

    Args:
        model_config: The loaded ModelConfig (uses ``pretrained_config.model_type`` and
            ``mapping.enable_attention_dp``).

    Raises:
        ValueError: TRTLLM_Q35_PROFILE has an invalid value.
    """
    global _active
    if _active:
        return
    profile = requested_profile()
    if profile is None:
        return
    _active = True
    model_type = getattr(model_config.pretrained_config, "model_type", None)
    capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    if model_type not in _MODEL_TYPES or capability != _SUPPORTED_SM:
        logger.warning(
            f"{PROFILE_ENV}={profile} ignored: it applies to Qwen3.5 ({', '.join(_MODEL_TYPES)}) on SM "
            f"{_SUPPORTED_SM[0]}.{_SUPPORTED_SM[1]}, got model_type {model_type!r} on "
            f"{'no GPU' if capability is None else f'SM {capability[0]}.{capability[1]}'}"
        )
        return
    enable_attention_dp = model_config.mapping.enable_attention_dp
    values = profile_defaults(profile, enable_attention_dp)
    applied = [name for name in values if name not in os.environ]
    for name in applied:
        os.environ[name] = values[name]
    if applied:
        for reader in _readers:
            reader()
    logger.info(
        f"{PROFILE_ENV}={profile} (attention DP: {enable_attention_dp}): {len(applied)} TRTLLM_Q35_* switches set, "
        f"{len(values) - len(applied)} already set in the environment"
    )
