# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MoE's per-expert finalize meets ConfigurableMoE's finalize-hook contract.

`MuseSparkOpenMoE` registers its `finalize` as the MoE's
`unfinalized_combine_fn`, so the scheduler applies the weight-first
post-expert norm to each expert output before the top-k sum. These run on CPU
with stub modules: the contract is the registration, the keyword signature the
scheduler calls, and the output shape, none of which needs a device.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models import modeling_musespark_open
from tensorrt_llm._torch.models.modeling_musespark_open import (
    MuseSparkOpenMoE,
    post_expert_norm_combine,
)
from tensorrt_llm._torch.moe.fused_moe.moe_scheduler import ExternalCommMoEScheduler
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

TOKENS, TOP_K, HIDDEN = 3, 2, 16


def _finalizer() -> MuseSparkOpenMoE:
    """A `MuseSparkOpenMoE` with only what `finalize` reads."""
    moe = MuseSparkOpenMoE.__new__(MuseSparkOpenMoE)
    torch.nn.Module.__init__(moe)
    moe.mapping = SimpleNamespace(moe_tp_size=1)
    moe.expert_hidden_size = HIDDEN
    moe.top_k = TOP_K
    moe.post_expert_norm_weight = torch.randn(HIDDEN) * 0.1
    moe.post_expert_norm_eps = 1e-6
    moe.nvtx = lambda _: nullcontext()
    return moe


def _scheduler(moe: MuseSparkOpenMoE) -> ExternalCommMoEScheduler:
    """The scheduler, with only what its finalize call site reads."""
    scheduler = ExternalCommMoEScheduler.__new__(ExternalCommMoEScheduler)
    scheduler.moe = SimpleNamespace(
        unfinalized_combine_fn=moe.finalize, backend=object(), comm=None, hidden_size=HIDDEN
    )
    return scheduler


def test_the_moe_registers_its_finalize_when_it_creates_the_experts(monkeypatch):
    captured = {}

    def create_moe(**kwargs):
        captured.update(kwargs)
        return torch.nn.Identity()

    monkeypatch.setattr(modeling_musespark_open, "create_moe", create_moe)
    monkeypatch.setattr(modeling_musespark_open, "Linear", lambda *a, **k: torch.nn.Identity())
    monkeypatch.setattr(modeling_musespark_open, "AllReduce", lambda *a, **k: torch.nn.Identity())
    config = SimpleNamespace(
        pretrained_config=SimpleNamespace(
            hidden_size=16,
            expert_hidden_size=8,
            num_experts=16,
            moe_topk=8,
            metap_multiplier=1.0,
            torch_dtype=torch.bfloat16,
            rms_norm_eps=1e-6,
            rms_post_norm_eps=1e-6,
            intermediate_size=8,
            router_score_division_eps=1e-15,
        ),
        mapping=Mapping(),
        quant_config=None,
        allreduce_strategy=AllReduceStrategy.AUTO,
    )
    moe = MuseSparkOpenMoE(config, layer_idx=0)
    assert captured["unfinalized_combine_fn"] == moe.finalize
    # The per-expert norm runs before the combine, so this module, not the
    # backend, reduces the partial sums.
    assert captured["reduce_results"] is False


def test_the_scheduler_call_site_reaches_the_finalize():
    """Through the scheduler's own call: hidden-padded rows in, the combine out."""
    torch.manual_seed(0)
    moe = _finalizer()
    # Some kernels write gemm2_output wider than the hidden size; the finalize
    # owns the slice.
    gemm2_output = torch.randn(TOKENS * TOP_K, HIDDEN + 8)
    index = torch.randperm(TOKENS * TOP_K).int()
    weights = torch.rand(TOKENS, TOP_K)

    combined = _scheduler(moe)._finalize_before_combine(
        [gemm2_output, None, index],
        token_selected_slots=None,
        token_final_scales=weights,
        output_dtype=torch.bfloat16,
    )
    expected = post_expert_norm_combine(
        gemm2_output,
        index,
        weights,
        moe.post_expert_norm_weight,
        moe.post_expert_norm_eps,
        num_tokens=TOKENS,
        top_k=TOP_K,
        hidden_size=HIDDEN,
    )
    assert combined.shape == (TOKENS, HIDDEN)
    assert combined.dtype == torch.bfloat16
    torch.testing.assert_close(combined, expected.to(torch.bfloat16))


def test_finalized_partials_combine_to_the_finalized_whole():
    """Finalizing before a combine is sound for this finalize.

    Under expert parallelism each rank finalizes only its local slots (the
    others are `-1`) and the combine sums the results. That equals finalizing
    every slot at once only because each output term depends on one
    `(token, expert)` row and non-local slots contribute exact zeros.
    """
    torch.manual_seed(1)
    moe = _finalizer()
    rows = torch.randn(TOKENS * TOP_K, HIDDEN)
    index = torch.arange(TOKENS * TOP_K, dtype=torch.int32)
    weights = torch.rand(TOKENS, TOP_K)
    local = torch.rand(TOKENS * TOP_K) < 0.5

    def finalize(slots: torch.Tensor) -> torch.Tensor:
        return moe.finalize(
            gemm2_output=rows,
            expanded_idx_to_permuted_idx=slots,
            token_selected_slots=None,
            routing_weights=weights,
            num_tokens=TOKENS,
        )

    whole = finalize(index)
    rank_a = finalize(torch.where(local, index, -1))
    rank_b = finalize(torch.where(local, -1, index))
    torch.testing.assert_close(rank_a + rank_b, whole)
