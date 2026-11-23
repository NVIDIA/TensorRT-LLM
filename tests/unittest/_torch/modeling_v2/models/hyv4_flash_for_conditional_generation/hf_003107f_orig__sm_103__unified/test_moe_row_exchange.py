# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MoE's row exchange under attention DP.

Under attention DP a rank holds only its own requests' rows while the expert
windows of the group are split over the ranks, so no rank can finish a row's
routed sum alone: the group gathers its rows, each rank runs its window over
the union, and the partial sums come back to the rows' own ranks by
reduce-scatter. Attention TP is the opposite case -- every rank holds every row
-- and there the plain sum over the group's partials completes the routed sum,
which is what the engine builds for `enable_attention_dp=False`.

These tests drive the target's own code (`Hyv4FlashCore._dp_exchange`,
`Hyv4FlashCore._dp_row_sizes`, `LatentMoe.forward`) on a core stand-in and pin
the two halves of that contract: which path one step takes, the row counts it
hands the collectives (the group's own counts in group order, or `None` when the
engine equalized the ranks), the height the expert window then sees, and the
rows that come back. They also pin the shared expert's side of both paths: one
sum over the group completes its slice together with the window's partial under
attention TP, while under attention DP it is whole on every rank and is added
once, on the rows that own it. CPU only -- the collectives are recorded, not
run, and the kernels behind `_routed` are replaced by shape-honest stand-ins.

The round trip's split is the one shape this target cannot get wrong: the counts
are ragged whenever the ranks hold different numbers of tokens, and a collective
whose operand height is read from each rank's own step instead is what wedged
the four ranks of the dep4 engine's first eager context step.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified import (  # noqa: E501
    modeling,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.modeling import (  # noqa: E501
    Hyv4FlashCore,
    LatentMoe,
)

#: Small stand-in geometry: the exchange is about rows, so the widths only have
#: to let the matmuls of the stand-in path carry a row count through.
HIDDEN = 4
LATENT = 4
SHARED_INTER = 2
MOE_INTER = 2
NUM_EXPERTS = 16
LOCAL_EXPERTS = 4
LAYER = 1


class _Core:
    """A core with the row-exchange methods and none of the engine behind them.

    The two methods bound here are the target's own; only the fields they read
    (`dims`, `_dp_group`) and the collectives the forward calls are built here.
    `_allreduce` records its operand instead of running it, so a test can tell
    the attention-TP path from the attention-DP round trip.
    """

    _dp_exchange = Hyv4FlashCore._dp_exchange
    _dp_row_sizes = Hyv4FlashCore._dp_row_sizes

    def __init__(
        self,
        *,
        attention_dp: bool,
        group: list[int],
        rank: int,
        moe_ep: int = 1,
        mlp_tp: int = 1,
        counts: list[int] | None = None,
    ) -> None:
        self.dims = SimpleNamespace(
            attention_dp=attention_dp,
            rank=rank,
            moe_ep=moe_ep,
            mlp_tp=mlp_tp,
            rms_eps=1e-6,
            # The fields `_routed` reads off `dims` and hands the runner.
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
        #: Whether the step carries the group's row counts at all is the step's
        #: own business: `all_rank_num_tokens` is what the runner's padding
        #: writes on an attention-DP step, and it is absent without one.
        self.md = (
            SimpleNamespace() if counts is None else SimpleNamespace(all_rank_num_tokens=counts)
        )
        self.allreduced: list[torch.Tensor] = []
        self.w = _weights()

    def _allreduce(self, x: torch.Tensor) -> torch.Tensor:
        self.allreduced.append(x)
        return x


def _weights() -> dict:
    """Identity on the latent path and zero on the shared one, so the routed half
    is the forward's whole output and the window's height is readable in it."""
    identity = torch.eye(HIDDEN, LATENT)
    return {
        f"l{LAYER}_pre_expert": identity.clone(),
        f"l{LAYER}_post_expert": identity.clone(),
        f"l{LAYER}_router": torch.zeros(NUM_EXPERTS, HIDDEN),
        f"l{LAYER}_router_bias": torch.zeros(NUM_EXPERTS),
        f"l{LAYER}_experts_gate_up": torch.zeros(1),
        f"l{LAYER}_experts_gate_up_scale": torch.zeros(1),
        f"l{LAYER}_experts_down": torch.zeros(1),
        f"l{LAYER}_experts_down_scale": torch.zeros(1),
        f"l{LAYER}_shared_gate": torch.zeros(SHARED_INTER, HIDDEN),
        f"l{LAYER}_shared_up": torch.zeros(SHARED_INTER, HIDDEN),
        f"l{LAYER}_shared_down": torch.zeros(LATENT, SHARED_INTER),
        f"l{LAYER}_combine_output_norm": torch.ones(LATENT),
    }


def _shared_weights(core: _Core) -> None:
    """Make the shared branch visible in the output: a row of ones through a 0.5
    gate gives `0.5 * HIDDEN` per intermediate column, and the all-ones down
    projection adds `SHARED_INTER` of them -- 4.0 at the stand-in's geometry."""
    core.w[f"l{LAYER}_shared_gate"] = torch.full((SHARED_INTER, HIDDEN), 0.5)
    core.w[f"l{LAYER}_shared_up"] = torch.zeros(SHARED_INTER, HIDDEN)
    core.w[f"l{LAYER}_shared_down"] = torch.ones(LATENT, SHARED_INTER)


@pytest.fixture
def collectives(monkeypatch):
    """Record the MoE's collectives and stand in for the kernels `_routed` calls.

    The two wrappers implement the collectives' documented semantics on CPU --
    concatenate the group's blocks in rank order, sum them and keep the calling
    rank's slice -- with every block filled with its own height, and with `own`
    (set by `_step`) telling the slice which block the caller owns. They are
    recorded, not compared: what the forward computes is a row count, and the
    tests read the heights, the `sizes` vector and the output.
    """
    calls = SimpleNamespace(gather=[], scatter=[], runner=[], own=0)

    def gather(input, sizes, group, process_group=None):
        calls.gather.append(
            SimpleNamespace(
                sizes=sizes,
                group=list(group),
                pg=process_group,
                rows=int(input.shape[0]),
            )
        )
        heights = sizes if sizes is not None else [int(input.shape[0])] * len(group)
        return torch.cat([torch.full((h, *input.shape[1:]), float(h)) for h in heights])

    def scatter(input, sizes, group, process_group=None):
        heights = sizes if sizes is not None else [int(input.shape[0]) // len(group)] * len(group)
        start = sum(heights[: calls.own])
        slice_ = input[start : start + heights[calls.own]]
        calls.scatter.append(
            SimpleNamespace(
                sizes=sizes,
                group=list(group),
                pg=process_group,
                rows=int(input.shape[0]),
                values=float(slice_.flatten()[0]),
            )
        )
        return slice_

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
    """One layer's MoE forward on the stand-in, with the caller's rank pointed at."""
    calls.own = core._dp_group.index(core.dims.rank)
    return LatentMoe(core, LAYER).forward(x, core.md)


# -- the exchange's decision and its split ---------------------------------


def test_the_exchange_runs_only_where_the_rows_are_split():
    group = [0, 1, 2, 3]
    assert _Core(attention_dp=True, group=group, rank=0)._dp_exchange()
    # Attention TP: every rank holds every row, and the MoE allreduces instead.
    assert not _Core(attention_dp=False, group=group, rank=0)._dp_exchange()
    # One rank's group has nothing to exchange even under attention DP.
    assert not _Core(attention_dp=True, group=[2], rank=2)._dp_exchange()


def test_equal_row_counts_take_the_splitless_form():
    core = _Core(attention_dp=True, group=[0, 1, 2, 3], rank=2)
    md = SimpleNamespace(all_rank_num_tokens=[4, 4, 4, 4])
    assert core._dp_row_sizes(md, rows=4) is None


def test_ragged_row_counts_are_the_groups_counts_in_group_order():
    core = _Core(attention_dp=True, group=[0, 1, 2, 3], rank=2)
    md = SimpleNamespace(all_rank_num_tokens=[2, 3, 4, 5])
    assert core._dp_row_sizes(md, rows=4) == [2, 3, 4, 5]
    # A group whose ranks are not named 0..n-1: the count read is the group's own
    # position of this rank, not the rank number.
    other = _Core(attention_dp=True, group=[3, 2, 1, 0], rank=2)
    md = SimpleNamespace(all_rank_num_tokens=[5, 3, 4, 4])
    assert other._dp_row_sizes(md, rows=3) == [5, 3, 4, 4]


def test_a_step_without_the_groups_counts_is_refused():
    core = _Core(attention_dp=True, group=[0, 1, 2, 3], rank=2)
    with pytest.raises(AssertionError, match="carries the group's row counts"):
        core._dp_row_sizes(SimpleNamespace(), rows=4)
    with pytest.raises(AssertionError, match="carries the group's row counts"):
        core._dp_row_sizes(SimpleNamespace(all_rank_num_tokens=None), rows=4)


def test_a_count_that_disagrees_with_this_ranks_rows_is_refused():
    core = _Core(attention_dp=True, group=[0, 1, 2, 3], rank=2)
    with pytest.raises(AssertionError, match="would gather the wrong split"):
        core._dp_row_sizes(SimpleNamespace(all_rank_num_tokens=[2, 3, 4, 5]), rows=3)
    # A vector of another length is not this group's split at all.
    with pytest.raises(AssertionError):
        core._dp_row_sizes(SimpleNamespace(all_rank_num_tokens=[2, 3, 4]), rows=4)


# -- the forward's path ----------------------------------------------------


def test_attention_dp_step_exchanges_the_rows(collectives):
    """A ragged step: the group's counts go to both collectives, the window sees
    the union, and what comes back is this rank's own block of the partials."""
    core = _Core(counts=[2, 3, 4, 5], attention_dp=True, group=[0, 1, 2, 3], rank=0, moe_ep=4)
    out = _step(torch.zeros(2, HIDDEN), core, collectives)

    assert len(collectives.gather) == 1 and len(collectives.scatter) == 1
    assert collectives.gather[0].sizes == [2, 3, 4, 5]
    assert collectives.gather[0].group == [0, 1, 2, 3]
    assert collectives.gather[0].pg is None
    # The window ran over the union: 2 + 3 + 4 + 5 rows, not this rank's two.
    assert collectives.runner[0]["routing_logits"].shape[0] == 14
    assert collectives.runner[0]["local_num_experts"] == LOCAL_EXPERTS
    assert collectives.runner[0]["local_expert_offset"] == 0
    # The same split sums the partials back onto the rows that own them.
    assert collectives.scatter[0].sizes == [2, 3, 4, 5]
    assert collectives.scatter[0].rows == 14
    assert out.shape == (2, HIDDEN)
    # The stand-in makes the routed block the union's height, so the output says
    # the window saw the union and the scatter kept this rank's own block of it.
    assert torch.all(out == 14.0)
    # No collective over the whole hidden: the exchange is the completion.
    assert not core.allreduced


def test_equal_step_takes_the_splitless_form(collectives):
    core = _Core(counts=[4, 4, 4, 4], attention_dp=True, group=[0, 1, 2, 3], rank=1, moe_ep=4)
    out = _step(torch.zeros(4, HIDDEN), core, collectives)

    assert collectives.gather[0].sizes is None
    assert collectives.runner[0]["routing_logits"].shape[0] == 16
    assert collectives.scatter[0].sizes is None
    assert collectives.scatter[0].values == 16.0
    assert out.shape == (4, HIDDEN)
    assert not core.allreduced


def test_attention_dp_adds_the_shared_expert_locally(collectives):
    """Under attention DP the shared expert is whole on this rank, so its
    contribution is added on the rows that own it, exactly once: a sum over the
    group would count four copies of it, and the exchange carries only the
    routed partials."""
    core = _Core(counts=[2, 3, 4, 5], attention_dp=True, group=[0, 1, 2, 3], rank=0, moe_ep=4)
    _shared_weights(core)
    out = _step(torch.ones(2, HIDDEN), core, collectives)

    assert not core.allreduced
    # 14 from the routed block (the union's height) + 4 from the shared one.
    assert torch.all(out == 18.0)
    assert out.shape == (2, HIDDEN)


def test_attention_tp_step_sums_the_ranks_partials(collectives):
    """Attention TP: every rank holds every row and its own share of the experts
    and of the shared intermediate, so one sum over the group completes both
    partials -- no row is exchanged."""
    core = _Core(attention_dp=False, group=[0, 1, 2, 3], rank=1, moe_ep=4)
    _shared_weights(core)
    out = _step(torch.ones(4, HIDDEN), core, collectives)

    assert not collectives.gather and not collectives.scatter
    assert collectives.runner[0]["routing_logits"].shape[0] == 4
    assert len(core.allreduced) == 1
    assert core.allreduced[0].shape == (4, HIDDEN)
    # The sum sees both partials: 4 from the window (the row count the stand-in
    # leaves in it) and 4 from this rank's slice of the shared intermediate.
    # A completion that covered the routed half alone would sum 4, not 8.
    assert torch.all(core.allreduced[0] == 8.0)
    assert torch.all(out == 8.0)


def test_a_single_rank_completes_nothing(collectives):
    """One rank holds every expert: no collective at all, either mode."""
    core = _Core(counts=[3], attention_dp=True, group=[1], rank=1, moe_ep=1, mlp_tp=1)
    out = _step(torch.zeros(3, HIDDEN), core, collectives)

    assert not collectives.gather and not collectives.scatter
    assert not core.allreduced
    assert collectives.runner[0]["routing_logits"].shape[0] == 3
    assert torch.all(out == 3.0)


# -- the entries' guards on the vectors the forward hands them -------------


def test_the_ragged_vector_passes_the_entries_guards():
    """The forms the forward sends -- `None` for an equalized step and the
    group's own ragged counts otherwise -- are what the entries accept. A vector
    of another length is what their guards exist to refuse: the op sizes itself
    from `sizes` alone, so a malformed one corrupts or wedges the ranks rather
    than erroring."""
    group = [0, 1, 2, 3]
    ragged = [2, 3, 4, 5]
    gathered = torch.zeros(sum(ragged), HIDDEN)

    modeling.allgather.is_valid(torch.zeros(2, HIDDEN), ragged, group)
    modeling.allgather.is_valid(torch.zeros(4, HIDDEN), None, group)
    modeling.reducescatter.is_valid(gathered, ragged, group)
    modeling.reducescatter.is_valid(torch.zeros(16, HIDDEN), None, group)

    with pytest.raises(AssertionError, match="len\\(sizes\\)"):
        modeling.allgather.is_valid(gathered, [2, 3, 4], group)
    with pytest.raises(AssertionError, match="sum\\(sizes\\)"):
        modeling.reducescatter.is_valid(gathered, [2, 3, 4, 4], group)
