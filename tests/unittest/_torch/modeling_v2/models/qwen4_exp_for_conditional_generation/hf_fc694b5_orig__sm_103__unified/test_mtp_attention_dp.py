# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-model MTP-Eagle under attention DP: the unified target's draft layer at dep4.

Each dep4 rank drafts for its own requests. The draft layer's attention holds every head, as a
dep4 trunk attention layer does, and its FP8 expert window `[128r, 128r + 128)` runs over the
rows the whole group gathers: the trunk's attention-DP round trip (`_expert_sum_op`), this
rank's rows padded by `_dp_pad_rows` over the row counts the spec worker hands each draft step
(`_draft_pad_rows`). The core is built on the meta device from the checkpoint's config values,
and the entries and the collectives are stood in for by arithmetic, so nothing here touches a
device.
"""

from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

__extra_import_path__ = ["."]
from test_core_layout import _DEP4, _TEP4, NUM_LAYERS, WORLD, _core, nn_pad  # noqa: E402

pytestmark = pytest.mark.cpu_only

HIDDEN, NUM_EXPERTS, TOPK = 2560, 512, 10


def _mtp(max_draft_len: int = 3) -> MTPDecodingConfig:
    return MTPDecodingConfig(max_draft_len=max_draft_len)


@pytest.mark.parametrize("rank", range(WORLD))
def test_a_dep4_rank_holds_the_whole_draft_attention_and_one_fp8_window(rank) -> None:
    """The constructor builds MTP under attention DP. The draft layer's attention is unsharded,
    as the trunk's dep4 attention is: 24 gated q heads, both KV heads and the whole o_proj;
    nothing sums its output. Its FP8 experts are this rank's 128-expert window at the whole
    640-row intermediate, which the group's four windows complete."""
    core = _core(rank, _DEP4, _mtp())
    assert core.mtp_enabled and core.max_verify_rows == 4
    assert (core.dp_size, core.dp_rank, core.attn_tp_size) == (WORLD, rank, 1)
    assert core.spec_rollback is not None
    p = f"l{NUM_LAYERS}_mtp_"
    w = core.w
    assert tuple(w[f"{p}q_w"].shape) == (2 * 6144, HIDDEN)
    assert tuple(w[f"{p}k_w"].shape) == tuple(w[f"{p}v_w"].shape) == (512, HIDDEN)
    assert tuple(w[f"{p}o_w"].shape) == (HIDDEN, 6144)
    assert tuple(w[f"{p}idx_w"].shape) == (640, HIDDEN)
    assert tuple(w[f"{p}router"].shape) == (NUM_EXPERTS, HIDDEN)
    assert (core.mtp_local_experts, core.mtp_expert_offset, core.moe_inter) == (
        128,
        128 * rank,
        640,
    )
    assert tuple(w[f"{p}fc1_w"].shape) == (128, 2 * 640, HIDDEN)
    assert w[f"{p}fc1_w"].dtype == torch.float8_e4m3fn
    assert tuple(w[f"{p}fc2_w"].shape) == (128, HIDDEN, 640)
    windows = [_core(r, _DEP4, _mtp(1)) for r in range(WORLD)]
    covered = sorted(
        e
        for c in windows
        for e in range(c.mtp_expert_offset, c.mtp_expert_offset + c.mtp_local_experts)
    )
    assert covered == list(range(NUM_EXPERTS))


def test_a_draft_step_pads_by_the_counts_the_worker_hands_it() -> None:
    """The spec worker hands each draft step the group's row counts -- every rank's trunk
    tokens on the loop's first step, its request count on the later ones -- and does not write
    them into the metadata. A count that disagrees with this rank's rows would gather the wrong
    split, and a missing one cannot pad at all; without attention DP nothing is padded."""
    core = _core(2, _DEP4, _mtp())
    # Draft-extend: rank 2 holds 9 trunk rows, the group's largest is 17.
    assert M._draft_pad_rows(core, [5, 17, 9, 1], 9) == 8
    # A later draft step: one row a request, attention-DP dummy requests included.
    assert M._draft_pad_rows(core, [2, 4, 3, 1], 3) == 1
    # A captured decode step pads every rank to one batch.
    assert M._draft_pad_rows(core, [4, 4, 4, 4], 4) == 0
    with pytest.raises(AssertionError, match="wrong split"):
        M._draft_pad_rows(core, [5, 17, 9, 1], 10)
    with pytest.raises(AssertionError, match="without the group's row counts"):
        M._draft_pad_rows(core, None, 9)
    assert M._draft_pad_rows(_core(0, _TEP4, _mtp()), None, 7) == 0


def _expert_out(x, ids, wts, start, count):
    """A stand-in MoE whose output names the experts a row routes to inside `[start, start +
    count)`: windows of whole experts sum to the all-expert output."""
    keep = ((ids >= start) & (ids < start + count)).to(x.dtype)
    return (wts * keep * (ids + 1).to(x.dtype)).sum(-1, keepdim=True) * x


@pytest.mark.parametrize(
    "counts",
    [[3, 7, 1, 4], [1, 2, 1, 3], [4, 4, 4, 4]],
    ids=["draft-extend", "draft-decode-with-a-dummy-rank", "captured-batch"],
)
def test_every_rank_completes_its_draft_rows_through_the_round_trip(counts, monkeypatch) -> None:
    """At every draft step every rank pads its rows to the group's largest count, gathers the
    group's rows (one allgather), routes, quantizes and runs its window over all of them in
    chunks of at most `_MTP_MOE_MAX_ROWS`, and reduce-scatters (one reduce-scatter): the same
    two collectives at the same shape on every rank, a rank holding one dummy request included.
    Each rank gets back the all-expert MoE of its own rows; nothing goes through the TP sum."""
    chunk_rows = 5
    monkeypatch.setattr(M, "_MTP_MOE_MAX_ROWS", chunk_rows)
    gen = torch.Generator().manual_seed(0)
    dt = torch.float64
    xs = [torch.randn(n, HIDDEN, generator=gen, dtype=dt) for n in counts]
    router = torch.randn(HIDDEN, NUM_EXPERTS, generator=gen, dtype=dt)
    rows = max(counts)
    gathered = torch.cat([nn_pad(x, rows) for x in xs], 0)

    def route(t):
        top_logits, top_ids = torch.topk(t @ router, TOPK, dim=-1)
        return top_ids, torch.softmax(top_logits, dim=-1)

    ids, wts = route(gathered)
    full = _expert_out(gathered, ids, wts, 0, NUM_EXPERTS)

    state = {}

    def fake_cublas_mm(a, b, *args, **kwargs):
        return a @ b

    def fake_routing(lg, topk, output_dtype):
        assert topk == TOPK and lg.shape[1] == NUM_EXPERTS and output_dtype == dt
        top_logits, top_ids = torch.topk(lg, topk, dim=-1)
        return top_ids, torch.softmax(top_logits, dim=-1)

    def fake_runner(**kw):
        core = state["core"]
        assert kw["intermediate_size"] == core.moe_inter == 640
        assert kw["local_expert_offset"] == 128 * core.dp_rank
        assert kw["local_num_experts"] == core.mtp_local_experts == 128
        x = kw["hidden_states"]
        assert 0 < x.shape[0] <= chunk_rows
        state["runner_rows"] += x.shape[0]
        return _expert_out(x, kw["topk_ids"], kw["topk_weights"], kw["local_expert_offset"], 128)

    def fake_allgather(t, sizes, group, process_group):
        assert sizes is None and group == list(range(WORLD)) and process_group is None
        state["collectives"].append(("allgather", tuple(t.shape)))
        torch.testing.assert_close(t, nn_pad(xs[state["rank"]], rows), rtol=0, atol=0)
        return gathered

    def fake_reducescatter(t, sizes, group, process_group):
        assert sizes is None and group == list(range(WORLD)) and process_group is None
        state["collectives"].append(("reducescatter", tuple(t.shape)))
        state["partial"] = t
        r = state["rank"]
        return full[r * rows : (r + 1) * rows]

    monkeypatch.setattr(M, "cublas_mm", fake_cublas_mm)
    monkeypatch.setattr(M, "renorm_moe_routing_op", fake_routing)
    monkeypatch.setattr(M, "fp8_quantize_1x128", lambda t, use_ue8m0: (t, None))
    monkeypatch.setattr(M, "fp8_block_scale_moe_runner", fake_runner)
    monkeypatch.setattr(M, "silu_and_mul_rows", lambda gug, inter: gug[:, :inter])
    # The shared expert and its gate are replicated and checked elsewhere: keep the routed sum.
    monkeypatch.setattr(M, "fused_sigmoid_gate_mul_add", lambda routed, gate, shared: routed)
    monkeypatch.setattr(M, "allgather", fake_allgather)
    monkeypatch.setattr(M, "reducescatter", fake_reducescatter)
    monkeypatch.setattr(M, "mpi_disabled", lambda: False)

    per_rank = []
    for rank in range(WORLD):
        core = _core(rank, _DEP4, _mtp())
        core.expert_sum = core._expert_sum_op()

        def no_tp_sum(t):
            raise AssertionError("the draft MoE under attention DP went through the TP sum")

        core.tp_sum = no_tp_sum
        state.update(core=core, rank=rank, collectives=[], runner_rows=0)
        p = f"l{NUM_LAYERS}_mtp_"
        views = SimpleNamespace(
            router=router,
            fc1_w=core.w[f"{p}fc1_w"],
            fc1_s=core.w[f"{p}fc1_s"],
            fc2_w=core.w[f"{p}fc2_w"],
            fc2_s=core.w[f"{p}fc2_s"],
            sh_gug=torch.zeros(HIDDEN, 2 * 640 + 8, dtype=dt),
            sh_dn=torch.zeros(640, HIDDEN, dtype=dt),
        )
        pad_rows = M._draft_pad_rows(core, counts, counts[rank])
        out = M._mtp_moe_block(core, views, xs[rank], pad_rows)
        # The window ran over every gathered row once, chunk by chunk.
        assert state["runner_rows"] == WORLD * rows
        torch.testing.assert_close(
            state["partial"], _expert_out(gathered, ids, wts, 128 * rank, 128)
        )
        assert tuple(out.shape) == (counts[rank], HIDDEN)
        torch.testing.assert_close(out, _expert_out(xs[rank], *route(xs[rank]), 0, NUM_EXPERTS))
        per_rank.append(state["collectives"])
    assert all(c == per_rank[0] for c in per_rank), per_rank
    assert per_rank[0] == [
        ("allgather", (rows, HIDDEN)),
        ("reducescatter", (WORLD * rows, HIDDEN)),
    ]


def _calls(fn: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]


@pytest.mark.parametrize("target", [M.PrefillTarget, M.DecodeTarget])
def test_both_draft_targets_pad_the_draft_moe_by_the_workers_counts(target) -> None:
    """A draft step runs on the prefill target while the metadata still carries context rows
    and on the decode target otherwise; ranks of one attention-DP group can be on either at the
    same draft step. Both pad the draft MoE by the `all_rank_num_tokens` the worker passes them
    for that step, never by the metadata's, which still holds the trunk step's counts."""
    fn = ast.parse(inspect.getsource(target.mtp_forward).lstrip())
    (pad,) = _calls(fn, "_draft_pad_rows")
    assert [ast.unparse(a) for a in pad.args] == ["core", "all_rank_num_tokens", "rows"]
    (tail,) = _calls(fn, "_mtp_tail")
    assert ast.unparse(tail.args[-1]) == "pad_rows"
    assert "_dp_pad_rows" not in ast.unparse(fn)
