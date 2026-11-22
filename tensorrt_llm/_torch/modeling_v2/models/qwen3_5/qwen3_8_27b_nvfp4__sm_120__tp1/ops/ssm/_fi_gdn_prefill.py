# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Host dispatch of the SM120 chunked gated delta rule over packed fp32 states: the
context-parallel (CP) kernel, or the non-CP kernel (V-split x2 or fully fused).

The SM120 branch of FlashInfer's ``flashinfer.chunk_gated_delta_rule`` (``gdn_prefill``,
flashinfer-ai/flashinfer commit 69ff11fc4954396d98326656dc85debd2223f637, Apache-2.0)
for ``use_cp="auto"``, packed fp32 initial / final states, no checkpoints, with two
additions: a wave-aware CP gate (CP pays only when the longest sequence splits into at
least ``CP_SM120_MIN_CHUNKS_PER_SEQ`` chunks) and the ``num_long_seqs`` hint the
non-CP launch uses to pick V-split x2. The kernels and the heuristics it builds on are
the installed FlashInfer's, imported unmodified.
"""

from __future__ import annotations

import torch
from flashinfer.gdn_kernels import cp_delta_rule_dsl_sm120
from flashinfer.gdn_kernels.delta_rule_dsl.varlen_helper import (
    _ceil_div,
    choose_cp_chunk_len_host,
    cp_short_workload_ratio_host,
    should_use_cp_host,
)
from flashinfer.utils import get_compute_capability, get_device_name, get_device_sm_count

from ._fi_delta_rule_sm120 import run as _run_non_cp
from ._fi_delta_rule_sm120 import vsplit_pays_off

#: Fewest chunks of the longest sequence at which CP beats the non-CP kernel on SM120.
CP_SM120_MIN_CHUNKS_PER_SEQ = 3


def sm120_cp_pays_off_host(
    max_seqlen: int,
    num_heads: int,
    num_sms: int,
    device_capability: tuple[int, int] | None = None,
    total_seqlen: int | None = None,
    num_seqs: int = 1,
    device_name: str = "",
) -> bool:
    """SM120 wave-aware CP gate for head counts outside the short heuristic.

    With more than ``CP_SM120_SHORT_HEURISTIC_MAX_HEADS`` heads the CP chunk length
    targets one wave of ``num_sms // num_heads`` chunks. When that split leaves the
    longest sequence with fewer than ``CP_SM120_MIN_CHUNKS_PER_SEQ`` chunks, the fixed
    T-precompute / MN / fixup cost is not recovered and the non-CP kernel (one CTA per
    (seq, head), single wave) is faster. Measured on RTX PRO 6000 (188 SMs), 48 heads,
    1 sequence, cold L2: non-CP vs CP 0.070/0.134 ms at 512 tokens, 0.111/0.152 ms at
    1024, 0.197/0.247 ms at 2048 (2 chunks); CP wins from 3 chunks on (0.281/0.273 ms
    at 3072, 0.709/0.656 ms at 8192).
    """
    if device_capability is None or device_capability[0] != 12:
        return True
    if cp_short_workload_ratio_host(device_capability, num_heads) is not None:
        return True
    chunk_len = choose_cp_chunk_len_host(
        max_seqlen,
        num_heads,
        num_sms,
        device_capability=device_capability,
        total_seqlen=total_seqlen,
        num_seqs=num_seqs,
        device_name=device_name,
    )
    return _ceil_div(max_seqlen, chunk_len) >= CP_SM120_MIN_CHUNKS_PER_SEQ


def plan(
    num_seqs: int,
    total_seqlen: int,
    num_long_seqs: int,
    num_sab_heads: int,
    num_sms: int,
    device_name: str,
    device_capability: tuple[int, int],
) -> str:
    """Which kernel a call runs: ``"cp"``, ``"vsplit"`` or ``"fused"``. Pure host logic.

    CP when FlashInfer's parallelism heuristic picks it (``num_seqs * heads`` below
    the card's threshold: a third of the SMs on a GDDR card) and the wave-aware gate
    agrees -- evaluated, as the CP launch itself does, with the total token count as
    the longest sequence. Otherwise the non-CP kernel, V-split x2 when the long
    sequences' doubled grid still fits one wave."""
    use_cp = should_use_cp_host(
        num_seqs * num_sab_heads, num_sms, device_name, device_capability=device_capability
    ) and sm120_cp_pays_off_host(
        total_seqlen,
        num_sab_heads,
        num_sms,
        device_capability=device_capability,
        total_seqlen=total_seqlen,
        num_seqs=num_seqs,
        device_name=device_name,
    )
    if use_cp:
        return "cp"
    if vsplit_pays_off(num_seqs, num_long_seqs, num_sab_heads, num_sms):
        return "vsplit"
    return "fused"


def chunk_gated_delta_rule_sm120(
    o: torch.Tensor,
    final_state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    num_long_seqs: int,
) -> None:
    """Writes ``o`` ``[T, HV, V]`` and ``final_state`` ``[N, HV, V, K]`` fp32 from packed
    contiguous q/k/v ``[T, H, D]``, linear-space ``alpha`` and ``beta`` ``[T, HV]`` fp32,
    ``initial_state`` ``[N, HV, V, K]`` fp32 and int64 ``cu_seqlens`` ``[N + 1]``."""
    device = q.device
    num_seqs = cu_seqlens.shape[0] - 1
    total_seqlen = q.shape[0]
    route = plan(
        num_seqs,
        total_seqlen,
        num_long_seqs,
        max(q.shape[1], v.shape[1]),
        get_device_sm_count(device),
        get_device_name(device),
        get_compute_capability(device),
    )
    if route == "cp":
        cp_delta_rule_dsl_sm120(
            o,
            final_state,
            q,
            k,
            v,
            alpha,
            beta,
            cu_seqlens,
            scale,
            initial_state=initial_state,
            max_seqlen=total_seqlen,
        )
        return
    _run_non_cp(
        o,
        final_state,
        q,
        k,
        v,
        initial_state,
        alpha,
        beta,
        cu_seqlens.to(torch.int64),
        scale,
        num_long_seqs,
    )
