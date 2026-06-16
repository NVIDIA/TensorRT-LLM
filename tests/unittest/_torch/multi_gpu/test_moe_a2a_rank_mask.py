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
"""Unit tests for the active_rank_mask parameter on the MoE AlltoAll kernels (PR 1a.2).

Two scenarios are exercised:

1. **All-active mask matches no-mask** — passing a mask with every bit set must produce
   bit-identical output to omitting the mask. Regression guard for the kernel mod's
   default behavior.

2. **One rank masked completes without hanging** — with bit K cleared, the surviving
   N-1 ranks must complete dispatch + combine without spinning on the dead rank's
   completion flag, and any token routed to the dead rank's experts must be dropped
   (topk_target_ranks[k] == -1) rather than silently corrupting peer memory.

The dispatch/combine kernels require `MnnvlMemory` (multi-node NVLink, GB200), so these
tests skip on hardware that does not support MNNVL and on nodes with fewer GPUs than
`ep_size`.
"""

import pickle
import sys

import cloudpickle
import pynvml
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm as tllm
from tensorrt_llm._mnnvl_utils import MnnvlMemory
from tensorrt_llm._torch.distributed import MoeAlltoAll
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)


# Must match cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.h
# kRankMaskWords. The full mask is little-endian: word 0 covers ranks 0..63.
EP_MASK_NUM_WORDS = 2


@pytest.fixture(autouse=True)
def setup_test():
    torch.manual_seed(0xA2A)
    tllm.logger.set_level("error")


def _skip_if_mnnvl_unsupported() -> None:
    try:
        MnnvlMemory.initialize()
        supports_mnnvl = MnnvlMemory.supports_mnnvl()
    except (RuntimeError, pynvml.NVMLError) as exc:
        pytest.skip(f"MNNVL not supported on this system: {exc}")
    if not supports_mnnvl:
        pytest.skip("MNNVL not supported on this system")


def _ep_mask_words(ep_size: int, dead_ranks: set[int]) -> torch.Tensor:
    """Build the uint64[EP_MASK_NUM_WORDS] CPU tensor expected by the C++ op."""
    mask_int = ((1 << ep_size) - 1) & ~sum(1 << r for r in dead_ranks)
    word_mask = (1 << 64) - 1
    words = [(mask_int >> (i * 64)) & word_mask for i in range(EP_MASK_NUM_WORDS)]
    return torch.tensor(words, dtype=torch.uint64, device="cpu")


def _generate_token_selected_experts(
    local_num_tokens: int, num_experts: int, top_k: int
) -> torch.Tensor:
    return torch.randint(
        0, num_experts, (local_num_tokens, top_k), dtype=torch.int32, device="cuda"
    )


def _make_payload(local_num_tokens: int, hidden_size: int, rank: int) -> torch.Tensor:
    """Deterministic per-rank payload so we can assert exact bit-for-bit equality."""
    base = torch.arange(local_num_tokens * hidden_size, dtype=torch.bfloat16, device="cuda").view(
        local_num_tokens, hidden_size
    )
    # Encode rank into the payload so cross-rank mismatches are immediately visible.
    return base + (rank * 1000.0)


def _read_topk_target_ranks(moe_a2a: MoeAlltoAll, max_num_tokens: int, top_k: int) -> torch.Tensor:
    """Read the kernel-written topk_target_ranks[max_num_tokens, top_k] from workspace."""
    offset = moe_a2a.metainfo[MoeAlltoAll._METAINFO_INDEX["TOPK_TARGET_RANKS_OFFSET_INDEX"]].item()
    raw = moe_a2a.workspace[
        moe_a2a.ep_rank,
        offset : offset + max_num_tokens * top_k * 4,
    ]
    return raw.view(torch.int32).view(max_num_tokens, top_k).cpu()


def _run_dispatch_combine(
    moe_a2a: MoeAlltoAll,
    token_selected_experts: torch.Tensor,
    payload: torch.Tensor,
    runtime_max_tokens_per_rank: int,
    active_rank_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drive dispatch + combine via the raw C++ ops (so we can pass active_rank_mask).

    Returns ``(combined_output, topk_target_ranks_snapshot)``. ``payload`` doubles as
    both the dispatched payload and the staged combine payload, which is the simplest
    end-to-end exercise of the masking path on both kernels.
    """
    recv_tensors, combine_payload_offset, _ = torch.ops.trtllm.moe_a2a_dispatch(
        token_selected_experts,
        [payload],
        moe_a2a.workspace,
        moe_a2a.metainfo,
        runtime_max_tokens_per_rank,
        moe_a2a.ep_rank,
        moe_a2a.ep_size,
        moe_a2a.top_k,
        moe_a2a.num_experts,
        None,  # eplb_local_stats
        active_rank_mask,
    )

    # Snapshot the kernel-written routing table BEFORE combine (combine reads it but
    # may also reset workspace state on subsequent rounds).
    topk_target_ranks = _read_topk_target_ranks(moe_a2a, runtime_max_tokens_per_rank, moe_a2a.top_k)

    combine_payload = recv_tensors[0]  # [ep_size, max_tokens, hidden_size]
    combined = torch.ops.trtllm.moe_a2a_combine(
        combine_payload,
        token_selected_experts.size(0),
        moe_a2a.workspace,
        moe_a2a.metainfo,
        runtime_max_tokens_per_rank,
        moe_a2a.ep_rank,
        moe_a2a.ep_size,
        moe_a2a.top_k,
        int(combine_payload_offset),
        False,  # payload_in_workspace
        False,  # use_low_precision
        active_rank_mask,
    )
    return combined.cpu(), topk_target_ranks


# ---------------------------------------------------------------------------
# Worker: regression — all-active mask must match no-mask (bit-identical).
# ---------------------------------------------------------------------------


def _worker_all_active_matches_no_mask(
    ep_size: int,
    local_num_tokens: int,
    top_k: int,
    workspace_size_per_rank: int,
    num_experts: int,
    hidden_size: int,
):
    rank = tllm.mpi_rank()
    torch.cuda.set_device(rank)
    mapping = Mapping(rank=rank, tp_size=ep_size, moe_ep_size=ep_size, world_size=ep_size)
    moe_a2a = MoeAlltoAll(
        mapping=mapping,
        max_num_tokens=local_num_tokens,
        top_k=top_k,
        num_slots=num_experts,
        workspace_size_per_rank=workspace_size_per_rank,
    )

    # Same RNG seed across both runs => identical inputs.
    torch.manual_seed(0xA2A + rank)
    token_selected_experts = _generate_token_selected_experts(local_num_tokens, num_experts, top_k)
    payload = _make_payload(local_num_tokens, hidden_size, rank)

    out_no_mask, topk_no_mask = _run_dispatch_combine(
        moe_a2a, token_selected_experts, payload, local_num_tokens, active_rank_mask=None
    )
    out_all_active, topk_all_active = _run_dispatch_combine(
        moe_a2a,
        token_selected_experts,
        payload,
        local_num_tokens,
        active_rank_mask=_ep_mask_words(ep_size, dead_ranks=set()),
    )

    return (
        torch.equal(out_no_mask, out_all_active),
        torch.equal(topk_no_mask, topk_all_active),
    )


# ---------------------------------------------------------------------------
# Worker: one rank masked — surviving ranks complete; dead-targeted slots dropped.
# ---------------------------------------------------------------------------


def _worker_one_rank_masked(
    ep_size: int,
    dead_rank: int,
    local_num_tokens: int,
    top_k: int,
    workspace_size_per_rank: int,
    num_experts: int,
    hidden_size: int,
):
    rank = tllm.mpi_rank()
    torch.cuda.set_device(rank)
    mapping = Mapping(rank=rank, tp_size=ep_size, moe_ep_size=ep_size, world_size=ep_size)
    # Every rank participates in workspace init (it has MPI barriers internally).
    moe_a2a = MoeAlltoAll(
        mapping=mapping,
        max_num_tokens=local_num_tokens,
        top_k=top_k,
        num_slots=num_experts,
        workspace_size_per_rank=workspace_size_per_rank,
    )

    if rank == dead_rank:
        # Simulate a dead rank: do not call dispatch/combine. Wait at a final
        # barrier so the surviving ranks have someone to synchronize with at
        # the end of the test. (The kernel itself never observes us because
        # the surviving ranks pass a mask with our bit cleared.)
        MPI.COMM_WORLD.barrier()
        return ("dead", None, None, None)

    torch.manual_seed(0xA2A + rank)
    token_selected_experts = _generate_token_selected_experts(local_num_tokens, num_experts, top_k)
    payload = _make_payload(local_num_tokens, hidden_size, rank)

    # Build mask with dead_rank's bit cleared.
    mask = _ep_mask_words(ep_size, dead_ranks={dead_rank})

    # Compute the per-token target ranks the way the kernel does so we can
    # cross-check the workspace afterwards.
    num_experts_per_rank = num_experts // ep_size
    expected_target_ranks = (token_selected_experts // num_experts_per_rank).cpu()

    combined, topk_target_ranks = _run_dispatch_combine(
        moe_a2a, token_selected_experts, payload, local_num_tokens, active_rank_mask=mask
    )

    MPI.COMM_WORLD.barrier()
    return (
        "alive",
        combined,
        topk_target_ranks,
        expected_target_ranks,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize(
    "mpi_pool_executor,local_num_tokens,top_k",
    [
        (4, 16, 2),
        (4, 32, 4),
    ],
    indirect=["mpi_pool_executor"],
)
def test_all_active_mask_matches_no_mask(mpi_pool_executor, local_num_tokens, top_k):
    """An all-ones active_rank_mask must produce identical output to omitting it."""
    _skip_if_mnnvl_unsupported()

    ep_size = mpi_pool_executor.num_workers
    if ep_size > torch.cuda.device_count():
        pytest.skip(
            f"Need at least {ep_size} GPUs but only {torch.cuda.device_count()} are available"
        )

    hidden_size = 1024
    num_experts = 32
    workspace_size_per_rank = 256 * 1024 * 1024

    args = (ep_size, local_num_tokens, top_k, workspace_size_per_rank, num_experts, hidden_size)
    results = list(
        mpi_pool_executor.map(
            _worker_all_active_matches_no_mask,
            *zip(*[args] * ep_size, strict=True),
        )
    )

    for rank, (output_eq, topk_eq) in enumerate(results):
        assert output_eq, f"rank {rank}: combine output differs between no-mask and all-active mask"
        assert topk_eq, f"rank {rank}: topk_target_ranks differ between no-mask and all-active mask"


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize(
    "mpi_pool_executor,dead_rank,local_num_tokens,top_k",
    [
        (4, 2, 16, 2),
        (4, 0, 16, 4),  # mask the lowest-numbered rank
        (4, 3, 32, 4),  # mask the highest-numbered rank
    ],
    indirect=["mpi_pool_executor"],
)
def test_one_rank_masked_completes(mpi_pool_executor, dead_rank, local_num_tokens, top_k):
    """With one rank masked dead, surviving ranks complete dispatch+combine.

    Verifies:
      * No hang (the test reaches the assertions).
      * On every surviving rank, any topk slot whose expert mapped to the dead
        rank is dropped (topk_target_ranks == -1).
      * Slots whose expert mapped to a surviving rank are unchanged from what
        the contiguous-partition routing rule predicts.
    """
    _skip_if_mnnvl_unsupported()

    ep_size = mpi_pool_executor.num_workers
    if ep_size > torch.cuda.device_count():
        pytest.skip(
            f"Need at least {ep_size} GPUs but only {torch.cuda.device_count()} are available"
        )
    assert 0 <= dead_rank < ep_size

    hidden_size = 1024
    num_experts = 32
    workspace_size_per_rank = 256 * 1024 * 1024

    args = (
        ep_size,
        dead_rank,
        local_num_tokens,
        top_k,
        workspace_size_per_rank,
        num_experts,
        hidden_size,
    )
    results = list(
        mpi_pool_executor.map(
            _worker_one_rank_masked,
            *zip(*[args] * ep_size, strict=True),
        )
    )

    saw_dead = False
    for rank, (status, combined, topk_target, expected_target) in enumerate(results):
        if status == "dead":
            assert rank == dead_rank
            saw_dead = True
            continue
        assert status == "alive"
        # Combine produced an output of the expected shape on the surviving rank.
        assert combined is not None
        assert combined.shape == (local_num_tokens, hidden_size)

        # Per-token routing assertions, using only the live tokens (the workspace
        # topk arrays are sized [max_num_tokens, top_k] and may have stale rows
        # beyond local_num_tokens; we only care about the live region).
        live_topk = topk_target[:local_num_tokens]
        live_expected = expected_target[:local_num_tokens]

        # Slot-level rules:
        #   - If expected_target_rank == dead_rank, the kernel must have set -1.
        #   - Otherwise, kernel must record the same target rank (or -1 if the
        #     same target was already covered earlier in the same token's top-k
        #     list; the kernel uses -1 as a "duplicate" sentinel).
        for token_idx in range(local_num_tokens):
            seen_ranks: set[int] = set()
            for k in range(top_k):
                exp = int(live_expected[token_idx, k].item())
                got = int(live_topk[token_idx, k].item())
                if exp == dead_rank:
                    assert got == -1, (
                        f"rank {rank} token {token_idx} k={k}: token routed to dead "
                        f"rank {dead_rank} should have been dropped (got={got})"
                    )
                elif exp in seen_ranks:
                    # Duplicate target within this token — kernel sets -1.
                    assert got == -1
                else:
                    assert got == exp, (
                        f"rank {rank} token {token_idx} k={k}: target rank mismatch "
                        f"(expected={exp}, got={got})"
                    )
                    seen_ranks.add(exp)

    assert saw_dead, f"dead rank {dead_rank} did not appear in results"
