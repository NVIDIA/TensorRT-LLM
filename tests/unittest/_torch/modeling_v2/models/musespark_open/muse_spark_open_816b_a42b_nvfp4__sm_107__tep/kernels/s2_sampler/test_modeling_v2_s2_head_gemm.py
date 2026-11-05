# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The s2 head sampler's LM-head GEMM on one GPU (``kernels/s2_sampler/sampler.py``), at the served TEP8 shape.

One process emulates the eight ranks: each emulated rank has an LM-head stand-in over its contiguous shard of a
``[8 * 25256, 8192]`` BF16 weight (with a low-M GEMM dispatcher, as the served LM head) and an ``S2Head`` over its
interleaved copy of the same weight. As in a served process, flashinfer's split-K module path resolves to this
target's ``ops/gemm/splitk_gemm.py`` under its default settings (``serve_flashinfer_path``, which the dense-GEMM
install calls), so the low-M dispatcher's split-K runner validates and launches with that module's tactic bounds and
shared-memory capacity. The tests pin:

* the served 8-row call takes the low-M dispatcher, ``PINNED_TACTIC`` passes the served module's validation for its
  shape and needs no more than the opt-in shared memory, so every rank's arming verdict is ``ok`` and an ``S2Head``
  arms; the served runner's autotune list for the shape leaves the tactic out on the test device (asserted), so
  arming does not rest on membership in that list;
* ``S2Head.produce``'s logits, gathered rank-major and unpermuted, equal bit for bit the stock head's logits with
  every rank's contiguous-shard GEMM at ``PINNED_TACTIC``, and at each other tactic of ``BIT_EQUIVALENT_TACTICS``.
"""

import importlib
import sys
import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.kernels.s2_sampler import (  # noqa: E501
    sampler as s2,
)
from tensorrt_llm._torch.modules import low_m_gemm as low_m

WORLD = 8
SHARD = 25256
HIDDEN = 8192
ROWS = s2.ROWS


def _supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)


pytestmark = pytest.mark.skipif(not _supported(), reason="the s2 head sampler runs on SM107")


@pytest.fixture
def served_splitk(monkeypatch):
    """``(target module, flashinfer's module)``: this target's split-K module under its default settings, installed
    at flashinfer's module path as in a served process, and flashinfer's own module loaded from that path before.
    The module path and the target module's settings are restored after the test."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.ops.gemm import (  # noqa: E501
        splitk_gemm,
    )

    path = splitk_gemm.FLASHINFER_MODULE
    kernels = importlib.import_module("flashinfer.gemm.kernels")
    # Record the path's current state for the teardown, then load flashinfer's file afresh.
    monkeypatch.setitem(sys.modules, path, None)
    del sys.modules[path]
    monkeypatch.setattr(kernels, "dense_bf16_gemm_sm100_splitk", None, raising=False)
    delattr(kernels, "dense_bf16_gemm_sm100_splitk")
    flashinfer_splitk = importlib.import_module(path)
    assert flashinfer_splitk is not splitk_gemm
    for name in (
        "_MAILBOX_V4",
        "_OVERSIZED_SMEM",
        "_RS_SPLIT_K",
        "_WIDE64",
        "_SMEM_RESOLVED",
        "_SMEM_CAPACITY_BYTES",
        "_MAX_AB_STAGES",
    ):
        monkeypatch.setattr(splitk_gemm, name, getattr(splitk_gemm, name))
    splitk_gemm.configure()
    assert splitk_gemm.serve_flashinfer_path()
    assert sys.modules[path] is splitk_gemm
    assert importlib.import_module(path) is splitk_gemm
    return splitk_gemm, flashinfer_splitk


@pytest.fixture
def ranks(monkeypatch, served_splitk):
    """``(x, [(lm_head, interleaved weight)] per emulated rank)``; the LM-head stand-ins share one dispatcher."""
    monkeypatch.setattr(low_m, "LOW_M_GEMM_ACTIVE", True)
    g = torch.Generator(device="cuda").manual_seed(7)
    weight = torch.randn(
        WORLD * SHARD, HIDDEN, generator=g, device="cuda", dtype=torch.bfloat16
    ).mul_(0.02)
    x = torch.randn(ROWS, HIDDEN, generator=g, device="cuda", dtype=torch.bfloat16)
    dispatcher = low_m.LowMGemmDispatcher()
    per_rank = []
    for rank in range(WORLD):
        lm_head = types.SimpleNamespace(
            weight=weight[rank * SHARD : (rank + 1) * SHARD],
            _low_m_gemm_dispatcher=dispatcher,
            use_custom_cublas_mm=False,
        )
        rows = s2.interleaved_rows(WORLD, rank, SHARD, weight.device)
        per_rank.append((lm_head, weight.index_select(0, rows).contiguous()))
    return x, per_rank


def test_the_served_module_launches_the_pinned_tactic_like_flashinfers(served_splitk):
    served, _ = served_splitk
    tactic = served.SplitKTactic(*s2.PINNED_TACTIC)
    served.validate_tactic(tactic, ROWS, SHARD, HIDDEN)
    # Within the opt-in shared memory: the launch requests the opt-in, as flashinfer's module does, and split_k 1
    # takes neither the mailbox nor the reduce-scatter epilogue.
    assert served._smem_bytes(tactic, tactic.ab_stages) <= served._OPTIN_SMEM_BYTES
    key = served.producer_key(ROWS, tactic)
    assert key[2] == "owner" and key[4] == "opt-in"


def test_every_rank_can_run_the_pinned_tactic_at_the_served_shape(ranks):
    x, per_rank = ranks
    with torch.inference_mode():
        # Precondition: the served module's autotune list for the shape (the window near its oversized stage cap)
        # leaves out the pinned tactic, so arming here cannot come from list membership.
        runner, weight_t = s2._pinned_gemm(per_rank[0][0], per_rank[0][0].weight, x)
        listed = runner.get_valid_tactics([x, weight_t], None)
        assert listed, "the served split-K runner lists no tactic for the M = 8 LM-head shape"
        assert s2.PINNED_TACTIC not in listed, listed
        for rank, (lm_head, _) in enumerate(per_rank):
            assert s2._choice_state(lm_head, x) == "ok", rank
        # An S2Head (one emulated rank as a one-rank group) arms at its first 8-row call outside autotuning.
        lm_head, interleaved = per_rank[0]
        mapping = types.SimpleNamespace(tp_size=1)
        states = types.SimpleNamespace(rs_areas=True)
        head = s2.S2Head(mapping, lm_head, interleaved, states, build_seconds=0.0)
        assert head.takes(x, lm_head)
        assert head.armed is True


def test_produce_gives_every_logit_the_stock_bits_of_the_bit_equivalent_tactics(
    ranks, served_splitk
):
    x, per_rank = ranks
    _, flashinfer_splitk = served_splitk
    mapping = types.SimpleNamespace(tp_size=WORLD)
    states = types.SimpleNamespace(rs_areas=False)
    with torch.inference_mode():
        shards = []
        for lm_head, interleaved in per_rank:
            head = s2.S2Head(mapping, lm_head, interleaved, states, build_seconds=0.0)
            head.produce(lm_head, x)
            shards.append(head.local.clone())
        index = s2.unpermute_index(WORLD, SHARD, x.device)
        got = s2.unpermute(torch.cat(shards, dim=-1), index)
        runner = per_rank[0][0]._low_m_gemm_dispatcher._runner_no_bias
        for tactic in sorted(s2.BIT_EQUIVALENT_TACTICS):
            stock = torch.cat(
                [
                    runner([x, lm_head.weight.t()], tactic=tactic, bias=None)
                    for lm_head, _ in per_rank
                ],
                dim=-1,
            )
            torch.cuda.synchronize()
            assert torch.equal(got.view(torch.int16), stock.view(torch.int16)), tactic
            # flashinfer's own kernel at the tactic: the served module computes the same bits.
            reference = torch.cat(
                [
                    flashinfer_splitk.run_splitk_dense(
                        x,
                        lm_head.weight.t(),
                        None,
                        torch.empty(ROWS, SHARD, dtype=x.dtype, device=x.device),
                        runner.pdl,
                        flashinfer_splitk.SplitKTactic(*tactic),
                    )
                    for lm_head, _ in per_rank
                ],
                dim=-1,
            )
            torch.cuda.synchronize()
            assert torch.equal(got.view(torch.int16), reference.view(torch.int16)), tactic
