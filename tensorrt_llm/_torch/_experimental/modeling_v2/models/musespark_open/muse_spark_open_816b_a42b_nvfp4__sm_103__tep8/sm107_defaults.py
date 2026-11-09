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
"""The sm_107 (Vera Rubin) defaults of this target's fast-path switches.

Each switch below started as an opt-in environment variable while its fast path was measured on Vera Rubin. On
sm_107 the target now defaults each one to the value the measurements settled on (the AgentX
configuration: TEP4 decode at batch 1-3 and TEP8 decode at batch 8, with TEP4 context workers). Setting the variable
still wins, including to ``""`` or ``"0"`` to turn a switch off. On every other device each switch keeps the
default its reader passes, which is the pre-existing behaviour.

The switches are read through ``env``, at import or install as before, so every rank decides alike (one device
type per deployment).
"""

from __future__ import annotations

import functools
import os
from typing import Optional

#: Switch -> value on sm_107 when the environment does not set it.
SM107_DEFAULTS = {
    # Dense GEMMs (flashinfer-style split-K kernel and its dispatcher tables).
    "MUSESPARK_OPEN_SPLITK_MAILBOX_V4": "1",  # conflict-free split-K mailbox, same bits
    "MUSESPARK_OPEN_SPLITK_RS": "1",  # reduce-scatter split-K epilogue, same bits
    "MUSESPARK_OPEN_SPLITK_RS_ONLY": "8",  # ... only for split-K 8 calls (the pool GEMMs)
    "MUSESPARK_OPEN_SPLITK_OVERSIZED_SMEM": "1",  # deeper split-K stages in oversized shared memory, same bits
    "MUSESPARK_OPEN_POOL_SPLITK8": "1",  # TEP4 pool GEMM at split-K 8
    "MUSESPARK_OPEN_TEP4_WIDE_GEMMS": (
        '{"qkv":[[16,"splitk:64,16,2,16"],[32,"splitk:64,32,2,13"]],'
        '"o_proj":[[16,"splitk:64,16,1,13"],[32,"splitk:64,32,1,13"]],'
        '"pool":[[16,"splitk:64,16,8,10"],[32,"splitk:64,32,8,9"]]}'
    ),
    "MUSESPARK_OPEN_TEP4_NARROW_GEMMS": (
        '{"qkv":[[1,"direct:256,2"],[2,"direct:64,4"],[3,"direct:32,2"],[8,"splitk:64,8,2,18"]],'
        '"o_proj":[[2,"direct:128,4"],[8,"splitk:64,8,1,14"]],'
        '"pool":[[1,"direct:256,2"],[6,"direct:128,2"],[8,"splitk:64,8,4,14"]]}'
    ),
    "MUSESPARK_OPEN_TEP8_WIDE64_SPLITK": "1",  # TP8 33..64-row QKV / o_proj on split-K
    "MUSESPARK_OPEN_TEP8_WIDE64_KERNELS": '{"qkv": "splitk:64,64,4,8"}',
    "MUSESPARK_OPEN_TEP8_PUBLISH_SPLITK": "1",  # TP8 pool / unpool slice publication from split-K
    "MUSESPARK_OPEN_TEP8_DRAFTER_WIDE_GEMMS": "1",  # TP8 drafter projections on split-K above 8 rows
    "MUSESPARK_OPEN_PREFETCH_JOIN": "after",
    "MUSESPARK_OPEN_DROP_WEIGHT_PREFETCH": "qkv",
    "MUSESPARK_OPEN_NEXT_QKV_PREFETCH_SKIP_ROWS": "8",
    # Attention.
    "MUSESPARK_OPEN_SWA_CL8": "all",  # TEP4 one-launch SWA block
    "MUSESPARK_OPEN_TEP8_SWA_CL8": "1",  # TEP8 one-launch SWA block at 5..8 requests
    "MUSESPARK_OPEN_SWA_PREP_SPLIT": "1",
    "MUSESPARK_OPEN_SWA_ONE2": "1",
    "MUSESPARK_OPEN_FA_PREP_FOLD_MULTI": "1",  # full-attention prep folded for 2..4 sequences
    "MUSESPARK_OPEN_FA_STAGES": "20",  # full-attention split-KV TMA ring depth
    "MUSESPARK_OPEN_MR_STAGES": "20",  # multi-request split-KV TMA ring depth
    # MoE.
    "MUSESPARK_OPEN_BF16_HALF_WINDOW": "1",
    "MUSESPARK_OPEN_PREROUTED_META": "1",  # TEP4 prerouted experts (trtllm::fp4_block_scale_moe_prerouted)
    "MUSESPARK_OPEN_TEP8_PREROUTED": "1",  # TEP8 prerouted experts (trtllm::fp4_block_scale_moe_prerouted)
    "MUSESPARK_OPEN_ROUTER_ROWS_TILING": "16:1:2,24:2:2,32:2:2",
    "MUSESPARK_OPEN_TEP8_RING_BUDDY_MAX_ROWS": "64",
    "MUSESPARK_OPEN_TEP8_RB_PDL": "1",
    "MUSESPARK_OPEN_TEP8_TILE_PLACEMENT": "1",
    "MUSESPARK_OPEN_TEP8_TILE_PLACEMENT_MIN_ROWS": "41",
    "MUSESPARK_OPEN_TEP8_ROUTER_PREWAIT": "1",
    "MUSESPARK_OPEN_TEP8_ROUTER_PUSH": "1",
    "MUSESPARK_OPEN_TEP8_ROUTER_PUSH_MIN_ROWS": "33",
    "MUSESPARK_OPEN_TEP8_ROUTER_PUSH_ROUTE": "1",
    "MUSESPARK_OPEN_TEP8_POOL_PROGRAMMATIC_FORK": "1",
    "MUSESPARK_OPEN_TEP8_WIDE_POOL_GATHER": "1",
    "MUSESPARK_OPEN_TEP8_WIDE_UNPOOL_GATHER": "1",
    "MUSESPARK_OPEN_TEP8_FC_SHARD_WIDE": "1",
    "MUSESPARK_OPEN_TEP8_FC_GATHER_PROGRAMS": "128",
    # Communication and step glue.
    "MUSESPARK_OPEN_FINALIZE_TRIGGER": "entry",
    "MUSESPARK_OPEN_TEP8_FINALIZE_TRANSPORT_TOKENS": "64",
    "MUSESPARK_OPEN_HOP_ENTRY_TRIGGER": "1",
    "MUSESPARK_OPEN_TEP8_HOP_DIRECT": "1",
    "MUSESPARK_OPEN_TEP8_HOP_EARLY": "1",
    "MUSESPARK_OPEN_TEP8_HOP_MULTICAST": "1",
    "MUSESPARK_OPEN_TEP8_HOP_PAD_ROWS": "9-40",
    "MUSESPARK_OPEN_TEP8_HOP_R2": "w",
    "MUSESPARK_OPEN_TEP8_HOP_R2_MIN_ROWS": "41",
    "MUSESPARK_OPEN_TEP8_HOP_SMEM_PAD": "177408",
    "MUSESPARK_OPEN_TEP8_JOIN_PDL": "1",
    "MUSESPARK_OPEN_PUBLISH_JOIN_CLUSTER": "1",
    "MUSESPARK_OPEN_NORM_GATE_PDL_MAX_ROWS": "128",
    "MUSESPARK_OPEN_STEP_COMMIT_V2": "1",
    "MUSESPARK_OPEN_CTX_KV_GLUE": "1",
    "MUSESPARK_OPEN_CTX_KV_FORK_EARLY": "1",
    "MUSESPARK_OPEN_VP_MIN_ROWS": "8",
    # K2's wide opt-ins (TRTLLM_MUSESPARK_TEP_WIDE_{DIRECT,ROUTER_ROWS,OPROJ_PUBLISH}) are not listed: on this
    # base they are the levers wide_direct_transports / wide_router_per_row / wide_oproj_publish
    # (MUSESPARK_OPEN_LEVER_WIDE_*), and levers.RETIRED raises on the old names.
    # Context workers.
    "MUSESPARK_OPEN_FC_SHARD_PREWARM": "1",  # create the drafter fc's BLAS handles at install
    "MUSESPARK_OPEN_SP_SM107_PARTS": "ce_gather,router_input",
}


#: Opt-in, off by default: ``MUSESPARK_OPEN_SM103_TEP8_F11=1`` serves the sm_107 TEP8 fast path (F11) on sm_103
#: (GB300): the ``(103, 8)`` composition becomes ``modeling._SM103_TEP8_F11_TARGET`` and the switches of
#: ``SM103_TEP8_DEFAULTS`` default on. Set it on TEP8 decode workers only: the defaults are process-wide (read at
#: import, before the TP size is known), and the TEP4 composition ``(103, 4)`` is unchanged by it. Unset, sm_103
#: behaves exactly as before; sm_107 never reads it.
SM103_TEP8_F11_ENV = "MUSESPARK_OPEN_SM103_TEP8_F11"

#: Switch -> value on sm_103 under ``MUSESPARK_OPEN_SM103_TEP8_F11=1`` when the environment does not set it: the
#: TEP8 entries of ``SM107_DEFAULTS`` plus the step glue they build on, with these sm_103 limits (GB300: 160 SMs,
#: 232448 B of opt-in shared memory per CTA, no oversized shared memory):
#:
#: * no ``SPLITK_OVERSIZED_SMEM`` and no ``TEP8_WIDE64_KERNELS`` override: the module's own wide64 QKV tactic
#:   ``splitk:64,64,4,5`` fits the opt-in, F11's 8-stage QKV does not (245760 B); the stage count changes no bits
#:   (``ops/gemm/splitk_gemm``), QKV keeps sm_103's 8-row ``split_k`` 4;
#: * wide64 on QKV and o_proj (the module's sites, which the direct hop needs): sm_103's 8-row o_proj cell
#:   is cuBLAS, and o_proj's split_k-1 wide64 tactic gives its bits row by row on GB300
#:   (``dense_gemms.TEP8_WIDE64_CUBLAS_CELLS``);
#: * the drafter's wide split-K tactics are sm_103's own (``fixed_low_m_gemm.SM103_TP8_DRAFTER_WIDE_TACTICS``);
#: * no TEP4-only switch (``SWA_CL8=all``, ``TEP4_*``, ``PREROUTED_META``, ``POOL_SPLITK8``, ``ROUTER_ROWS_TILING``,
#:   ``FA_STAGES``, ``MR_STAGES``: sm_103 already runs 20 multi-request stages), no context switch. The shared
#:   attention switches F11's TEP8 shapes need (``SWA_ONE2``, ``SWA_PREP_SPLIT``, ``FA_PREP_FOLD_MULTI``) and
#:   ``BF16_HALF_WINDOW`` are carried; they act on a TEP4 process too, which is why the switch is for TEP8 workers.
SM103_TEP8_DEFAULTS = {
    # Dense GEMMs.
    "MUSESPARK_OPEN_SPLITK_MAILBOX_V4": "1",  # conflict-free split-K mailbox, same bits
    "MUSESPARK_OPEN_TEP8_WIDE64_SPLITK": "1",  # TP8 33..64-row QKV / o_proj on split-K (module tactics fit sm_103)
    "MUSESPARK_OPEN_TEP8_PUBLISH_SPLITK": "1",
    "MUSESPARK_OPEN_TEP8_DRAFTER_WIDE_GEMMS": "1",
    "MUSESPARK_OPEN_PREFETCH_JOIN": "after",
    "MUSESPARK_OPEN_DROP_WEIGHT_PREFETCH": "qkv",
    "MUSESPARK_OPEN_NEXT_QKV_PREFETCH_SKIP_ROWS": "8",
    # Attention (SWA_ONE2 / SWA_PREP_SPLIT / FA_PREP_FOLD_MULTI also serve TEP8 shapes: ONE2 takes 5..8 requests at 2
    # KV heads and CL8-TEP8 replaces it there; the fold takes 2..4 requests).
    "MUSESPARK_OPEN_TEP8_SWA_CL8": "1",  # TEP8 one-launch SWA block at 5..8 requests (~180 KB smem, fits)
    "MUSESPARK_OPEN_SWA_PREP_SPLIT": "1",
    "MUSESPARK_OPEN_SWA_ONE2": "1",
    "MUSESPARK_OPEN_FA_PREP_FOLD_MULTI": "1",
    # MoE.
    "MUSESPARK_OPEN_BF16_HALF_WINDOW": "1",  # BF16 layers 0 / 61 on the TEP8 ring-buddy window (flagged numeric)
    "MUSESPARK_OPEN_TEP8_PREROUTED": "1",
    "MUSESPARK_OPEN_TEP8_RING_BUDDY_MAX_ROWS": "64",
    "MUSESPARK_OPEN_TEP8_RB_PDL": "1",
    "MUSESPARK_OPEN_TEP8_TILE_PLACEMENT": "1",
    "MUSESPARK_OPEN_TEP8_TILE_PLACEMENT_MIN_ROWS": "41",
    "MUSESPARK_OPEN_TEP8_ROUTER_PREWAIT": "1",
    "MUSESPARK_OPEN_TEP8_ROUTER_PUSH": "1",
    "MUSESPARK_OPEN_TEP8_ROUTER_PUSH_MIN_ROWS": "33",
    "MUSESPARK_OPEN_TEP8_ROUTER_PUSH_ROUTE": "1",
    "MUSESPARK_OPEN_TEP8_POOL_PROGRAMMATIC_FORK": "1",
    "MUSESPARK_OPEN_TEP8_WIDE_POOL_GATHER": "1",
    "MUSESPARK_OPEN_TEP8_WIDE_UNPOOL_GATHER": "1",
    "MUSESPARK_OPEN_TEP8_FC_SHARD_WIDE": "1",
    "MUSESPARK_OPEN_TEP8_FC_GATHER_PROGRAMS": "128",
    # Communication and step glue.
    "MUSESPARK_OPEN_FINALIZE_TRIGGER": "entry",
    "MUSESPARK_OPEN_TEP8_FINALIZE_TRANSPORT_TOKENS": "64",
    "MUSESPARK_OPEN_HOP_ENTRY_TRIGGER": "1",
    "MUSESPARK_OPEN_TEP8_HOP_DIRECT": "1",
    "MUSESPARK_OPEN_TEP8_HOP_EARLY": "1",
    "MUSESPARK_OPEN_TEP8_HOP_MULTICAST": "1",
    "MUSESPARK_OPEN_TEP8_HOP_PAD_ROWS": "9-40",
    "MUSESPARK_OPEN_TEP8_HOP_R2": "w",
    "MUSESPARK_OPEN_TEP8_HOP_R2_MIN_ROWS": "41",
    "MUSESPARK_OPEN_TEP8_HOP_SMEM_PAD": "177408",  # + the hop's own smem stays under 232448 B; tuned on VR
    "MUSESPARK_OPEN_TEP8_JOIN_PDL": "1",
    "MUSESPARK_OPEN_PUBLISH_JOIN_CLUSTER": "1",
    "MUSESPARK_OPEN_NORM_GATE_PDL_MAX_ROWS": "128",
    "MUSESPARK_OPEN_STEP_COMMIT_V2": "1",
    "MUSESPARK_OPEN_CTX_KV_GLUE": "1",
    "MUSESPARK_OPEN_CTX_KV_FORK_EARLY": "1",
    "MUSESPARK_OPEN_VP_MIN_ROWS": "8",
}


@functools.lru_cache(maxsize=1)
def _capability() -> Optional[tuple]:
    """This process's CUDA device capability (None without CUDA)."""
    try:
        import torch

        return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001 - no device, no defaults
        return None


def on_sm107() -> bool:
    """Whether this process's CUDA device is sm_107 (False without CUDA)."""
    return _capability() == (10, 7)


def sm103_tep8_f11() -> bool:
    """Whether ``MUSESPARK_OPEN_SM103_TEP8_F11=1`` asks for the F11 TEP8 path on sm_103 (the environment only)."""
    return os.environ.get(SM103_TEP8_F11_ENV, "") == "1"


def on_sm103_tep8_f11() -> bool:
    """Whether this process runs on sm_103 with ``MUSESPARK_OPEN_SM103_TEP8_F11=1``."""
    return sm103_tep8_f11() and _capability() == (10, 3)


def env(name: str, default: Optional[str] = None) -> Optional[str]:
    """``os.environ.get(name, default)``, except that an unset switch reads ``SM107_DEFAULTS[name]`` on sm_107 and
    ``SM103_TEP8_DEFAULTS[name]`` on sm_103 under ``MUSESPARK_OPEN_SM103_TEP8_F11=1``."""
    value = os.environ.get(name)
    if value is not None:
        return value
    if name in SM107_DEFAULTS and on_sm107():
        return SM107_DEFAULTS[name]
    if name in SM103_TEP8_DEFAULTS and on_sm103_tep8_f11():
        return SM103_TEP8_DEFAULTS[name]
    return default
