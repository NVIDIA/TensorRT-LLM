# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The engine hooks every Kimi K3.1 target declares (``RuntimeHooks``).

* ``torch_mem_safety_margin``: 4 GiB outside the PyTorch cap. The cap is "free at setup - margin", so PyTorch would
  otherwise refill whatever the KV cache leaves, and long context-server runs exhaust the device in the driver
  allocations made after setup (streams, events and copy launches of the KV cache manager and transfer agents, CUDA
  IPC mappings of peer buffers, kernel attribute and module loads).
* ``kv_balanced_admission_planner``: with ``attention_dp_config.enable_kv_balanced_admission`` the attention-DP
  admission places requests count first (``K31CountFirstAdmission``).
"""

from __future__ import annotations

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.kda_dsa_cache_manager import (
    KimiK3DSAHybridCacheManager,
)
from tensorrt_llm._torch.pyexecutor.runtime_hooks import DFlashRuntimePolicy, RuntimeHooks

from .ops.ssm.unified_draft_cache import MlaDFlashContextPolicy

#: The engine's admission window (requests at the head of the queue to choose among), at least.
_ADMISSION_WINDOW = 32
#: Count-first KV slack: a rank may take a request while its KV stays within this factor of the mean KV per rank.
_COUNT_FIRST_KV_SLACK = 1.1


class K31CountFirstAdmission:
    """Attention-DP KV-balanced admission that places each request on the rank with the fewest requests.

    An attention-DP decode step of Kimi K3.1 follows the busiest rank's request count more than its KV, so each
    admitted request goes to the rank with the fewest requests, then the least KV, among the ranks whose KV after it
    stays within ``_COUNT_FIRST_KV_SLACK`` of the mean KV per rank (else to the least-KV rank). When every candidate
    fits, they are placed largest first; otherwise the first candidates in queue order are, forced ones first. Every
    rank computes the same plan from the same all-gathered inputs.
    """

    def window(self, max_new_requests: int, free_slots: int) -> int:
        return max(_ADMISSION_WINDOW, min(max_new_requests, free_slots))

    def plan(
        self,
        *,
        candidate_tokens: list[int],
        candidate_forced: list[bool],
        rank_tokens: list[int],
        rank_num_requests: list[int],
        max_requests_per_rank: int,
        max_new_requests: int,
        candidate_groups: list[int],
    ) -> list[tuple[int, int]]:
        del candidate_groups  # Queue order already keeps the admission groups in order.
        loads = list(rank_tokens)
        counts = list(rank_num_requests)
        num_ranks = len(counts)
        free = [max(0, max_requests_per_rank - count) for count in counts]
        candidates = list(range(len(candidate_tokens)))
        budget = min(max_new_requests, sum(free), len(candidates))
        if budget <= 0:
            return []
        if budget >= len(candidates):
            order = sorted(candidates, key=lambda i: (-candidate_tokens[i], i))
        else:
            order = sorted(candidates, key=lambda i: (not candidate_forced[i], i))[:budget]
        plan = []
        for idx in order:
            tokens = candidate_tokens[idx]
            ranks = [r for r in range(num_ranks) if free[r] > 0]
            ceiling = _COUNT_FIRST_KV_SLACK * (sum(loads) + tokens) / num_ranks
            fit = [r for r in ranks if loads[r] + tokens <= ceiling]
            if fit:
                rank = min(fit, key=lambda r: (counts[r], loads[r], r))
            else:
                rank = min(ranks, key=lambda r: (loads[r], counts[r], r))
            plan.append((idx, rank))
            loads[rank] += tokens
            counts[rank] += 1
            free[rank] -= 1
        return plan


K31_RUNTIME_HOOKS = RuntimeHooks(
    # KDA recurrent state, MLA latent cache and one DSA indexer K-cache buffer per MLA layer in one
    # KV cache manager V2: a subclass of the hybrid manager the engine resolves for DSA on Kimi linear.
    kv_cache_manager_cls=KimiK3DSAHybridCacheManager,
    dflash_policy=DFlashRuntimePolicy(
        uses_paged_context=MlaDFlashContextPolicy(),
        supports_helix_rejection_sampling=True,
    ),
    torch_mem_safety_margin=4 << 30,
    kv_balanced_admission_planner=K31CountFirstAdmission(),
)
