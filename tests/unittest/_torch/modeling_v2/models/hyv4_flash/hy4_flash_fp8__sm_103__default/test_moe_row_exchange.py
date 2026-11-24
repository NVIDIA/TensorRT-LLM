# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`LatentMoe.forward`'s completion under attention DP (row exchange) and attention TP (one sum).

CPU only: the collectives are recorded and the kernels replaced by stand-ins whose outputs carry
row counts.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default import (  # noqa: E501
    modeling,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.modeling import (  # noqa: E501
    Hyv4FlashCore,
    LatentMoe,
)

HIDDEN = 4
LATENT = 4
SHARED_INTER = 2
MOE_INTER = 2
NUM_EXPERTS = 16
LOCAL_EXPERTS = 4
LAYER = 1


class _Core:
    """The target's row-exchange methods over a stand-in `dims`; `_allreduce` records."""

    _dp_exchange = Hyv4FlashCore._dp_exchange
    _dp_row_sizes = Hyv4FlashCore._dp_row_sizes

    def __init__(
        self,
        *,
        attention_dp: bool,
        group: list[int],
        rank: int,
        moe_ep: int = 1,
        counts: list[int] | None = None,
    ) -> None:
        self.dims = SimpleNamespace(
            attention_dp=attention_dp,
            rank=rank,
            moe_ep=moe_ep,
            mlp_tp=1,
            rms_eps=1e-6,
            num_experts=NUM_EXPERTS,
            top_k=2,
            n_group=1,
            topk_group=1,
            moe_inter=MOE_INTER,
            local_experts=LOCAL_EXPERTS,
            expert_offset=LOCAL_EXPERTS * (rank % max(moe_ep, 1)),
            routed_scale=3.998,
            swiglu_limit=10.0,
            max_num_tokens=8192,
        )
        self._dp_group = list(group)
        self._dp_pg = None
        self.md = (
            SimpleNamespace() if counts is None else SimpleNamespace(all_rank_num_tokens=counts)
        )
        self.allreduced: list[torch.Tensor] = []
        identity = torch.eye(HIDDEN, LATENT)
        # Identity latent path; the shared branch gives 4.0 per element on a row of ones.
        self.w = {
            f"l{LAYER}_pre_expert": identity.clone(),
            f"l{LAYER}_post_expert": identity.clone(),
            f"l{LAYER}_router": torch.zeros(NUM_EXPERTS, HIDDEN),
            f"l{LAYER}_router_bias": torch.zeros(NUM_EXPERTS),
            f"l{LAYER}_experts_gate_up": torch.zeros(1),
            f"l{LAYER}_experts_gate_up_scale": torch.zeros(1),
            f"l{LAYER}_experts_down": torch.zeros(1),
            f"l{LAYER}_experts_down_scale": torch.zeros(1),
            f"l{LAYER}_shared_gate": torch.full((SHARED_INTER, HIDDEN), 0.5),
            f"l{LAYER}_shared_up": torch.zeros(SHARED_INTER, HIDDEN),
            f"l{LAYER}_shared_down": torch.ones(LATENT, SHARED_INTER),
            f"l{LAYER}_combine_output_norm": torch.ones(LATENT),
        }

    def _allreduce(self, x: torch.Tensor) -> torch.Tensor:
        self.allreduced.append(x)
        return x


@pytest.fixture
def collectives(monkeypatch):
    """CPU collectives (rank-order concat; the caller's slice of the sum) filling each block with
    its own height, and a runner returning its row count."""
    calls = SimpleNamespace(gather=[], scatter=[], runner=[], own=0)

    def gather(input, sizes, group, process_group=None):
        calls.gather.append(SimpleNamespace(sizes=sizes, group=list(group)))
        heights = sizes if sizes is not None else [int(input.shape[0])] * len(group)
        return torch.cat([torch.full((h, *input.shape[1:]), float(h)) for h in heights])

    def scatter(input, sizes, group, process_group=None):
        heights = sizes if sizes is not None else [int(input.shape[0]) // len(group)] * len(group)
        start = sum(heights[: calls.own])
        calls.scatter.append(SimpleNamespace(sizes=sizes, rows=int(input.shape[0])))
        return input[start : start + heights[calls.own]]

    def runner(**kwargs):
        calls.runner.append(kwargs)
        rows = int(kwargs["routing_logits"].shape[0])
        return torch.full((rows, LATENT), float(rows))

    monkeypatch.setattr(modeling, "allgather", gather)
    monkeypatch.setattr(modeling, "reducescatter", scatter)
    monkeypatch.setattr(modeling, "cublas_mm", lambda a, b: a @ b)
    monkeypatch.setattr(modeling, "fp8_quantize_1x128", lambda x: (x, x))
    monkeypatch.setattr(modeling, "fp8_block_scale_moe_runner", runner)
    monkeypatch.setattr(modeling, "flashinfer_silu_and_mul", lambda x: x[:, : x.shape[-1] // 2])
    monkeypatch.setattr(modeling, "_rms", lambda x, weight, eps: x)
    return calls


def _step(x: torch.Tensor, core: _Core, calls: SimpleNamespace) -> torch.Tensor:
    calls.own = core._dp_group.index(core.dims.rank)
    return LatentMoe(core, LAYER).forward(x, core.md)


def test_attention_dp_step_exchanges_the_rows(collectives):
    """Ragged counts reach both collectives, the window sees the union, and the shared expert is
    added once, locally."""
    core = _Core(counts=[2, 3, 4, 5], attention_dp=True, group=[0, 1, 2, 3], rank=0, moe_ep=4)
    out = _step(torch.ones(2, HIDDEN), core, collectives)

    assert collectives.gather[0].sizes == [2, 3, 4, 5]
    assert collectives.gather[0].group == [0, 1, 2, 3]
    assert collectives.runner[0]["routing_logits"].shape[0] == 14
    assert collectives.scatter[0].sizes == [2, 3, 4, 5]
    assert collectives.scatter[0].rows == 14
    assert not core.allreduced
    # 14 from the routed block (the union's height) + 4 from the shared expert.
    assert out.shape == (2, HIDDEN)
    assert torch.all(out == 18.0)


def test_attention_tp_step_sums_the_ranks_partials(collectives):
    """One allreduce completes both the routed and the shared partials; no row is exchanged."""
    core = _Core(attention_dp=False, group=[0, 1, 2, 3], rank=1, moe_ep=4)
    out = _step(torch.ones(4, HIDDEN), core, collectives)

    assert not collectives.gather and not collectives.scatter
    assert collectives.runner[0]["routing_logits"].shape[0] == 4
    assert len(core.allreduced) == 1
    # 4 from the window + 4 from the shared slice.
    assert torch.all(core.allreduced[0] == 8.0)
    assert torch.all(out == 8.0)
