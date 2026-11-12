# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Row cap of the CuTe DSL MLA decode reduction prefetch: the gate and its launch arguments."""

import math

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._utils import is_sm_100f

requires_dsl = pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="CuTe DSL is not available")
requires_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or not IS_CUTLASS_DSL_AVAILABLE or not is_sm_100f(),
    reason="CuTe DSL MLA requires an SM100-family CUDA GPU",
)

NUM_SMS = 212  # SM count the gate is evaluated at
HEADS_PER_RANK = 12  # target MLA at TP8
CAP = 288  # the default cap, written out so that a changed default fails the tests
# Query rows per request of a speculative step with K draft tokens: S = K + 1.
DECODE_SHAPES = {"K6": 7, "K5": 6}
# last batch the cap keeps on the prefetching reduction
LAST_ON_BATCH = {"K6": 3, "K5": 4}


def _runner(monkeypatch, *, switch=True, cap=CAP):
    import tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops as ops

    runner = ops.CuteDSLNVMlaDecodeBlackwellRunner
    monkeypatch.setattr(ops, "_get_num_sms", lambda: NUM_SMS)
    monkeypatch.setattr(runner, "_REDUCTION_PREFETCH", switch)
    monkeypatch.setattr(runner, "_REDUCTION_PREFETCH_MAX_ROWS", cap)
    return runner


def _gate(runner, rows):
    """(D bands, prefetched splits) of a launch of ``rows`` query rows."""
    return runner.get_reducer_d_tiles(rows), runner.get_reducer_prefetch_splits(rows)


@requires_dsl
def test_default_cap_and_env(monkeypatch) -> None:
    import tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops as ops

    name = "TLLM_CUTE_DSL_MLA_REDUCTION_PREFETCH_MAX_ROWS"
    read = ops._reduction_prefetch_max_rows_env
    # The class attribute was read from the environment at import.
    assert ops.CuteDSLNVMlaDecodeBlackwellRunner._REDUCTION_PREFETCH_MAX_ROWS == read(name, CAP)
    monkeypatch.delenv(name, raising=False)
    assert read(name, CAP) == CAP
    for raw, expected in (("", CAP), ("  ", CAP), ("0", 0), ("1", 1), (" 400 ", 400)):
        monkeypatch.setenv(name, raw)
        assert read(name, CAP) == expected
    for bad in ("-1", "abc", "1.5", "0x10"):
        monkeypatch.setenv(name, bad)
        with pytest.raises(ValueError):
            read(name, CAP)


@requires_dsl
@pytest.mark.parametrize("kind", ["K6", "K5"])
@pytest.mark.parametrize("batch", range(1, 9))
def test_gate_at_decode_shapes(monkeypatch, kind: str, batch: int) -> None:
    """Rows = B x S_q x heads per rank: both effects on for K6 b<=3 and K5 b<=4, both off above."""
    runner = _runner(monkeypatch)
    rows = batch * DECODE_SHAPES[kind] * HEADS_PER_RANK
    expected = (4, 32) if batch <= LAST_ON_BATCH[kind] else (1, 0)
    assert _gate(runner, rows) == expected


@requires_dsl
def test_gate_around_the_cap(monkeypatch) -> None:
    runner = _runner(monkeypatch)
    assert _gate(runner, CAP - 1) == (4, 32)
    assert _gate(runner, CAP) == (4, 32)
    assert _gate(runner, CAP + 1) == (1, 0)
    # The cap is a parameter, not baked into the rule.
    runner = _runner(monkeypatch, cap=100)
    assert _gate(runner, 100) == (4, 32)
    assert _gate(runner, 101) == (1, 0)


@requires_dsl
def test_cap_zero_is_the_uncapped_rule(monkeypatch) -> None:
    """Cap 0: prefetch always, four bands while 4 * rows fits the wave budget (the branch before the cap)."""
    runner = _runner(monkeypatch, cap=0)
    budget = runner._REDUCTION_D_BAND_MAX_WAVES * NUM_SMS
    for rows in list(range(1, 1200)) + [10**5]:
        assert _gate(runner, rows) == ((4 if 4 * rows <= budget else 1), 32)


@requires_dsl
@pytest.mark.parametrize("cap", [0, 1, CAP, 10**6])
def test_switch_off_is_the_default_reduction(monkeypatch, cap: int) -> None:
    runner = _runner(monkeypatch, switch=False, cap=cap)
    for rows in (1, 12, 252, CAP, CAP + 1, 636, 637, 10**5):
        assert _gate(runner, rows) == (1, 0)


@requires_dsl
def test_wave_budget_still_applies_below_a_large_cap(monkeypatch) -> None:
    runner = _runner(monkeypatch, cap=10**6)
    budget = runner._REDUCTION_D_BAND_MAX_WAVES * NUM_SMS
    assert _gate(runner, budget // 4) == (4, 32)
    assert _gate(runner, budget // 4 + 1) == (1, 32)


# ---------------------------------------------------------------------------------------------
# GPU: the launch of a decode call
# ---------------------------------------------------------------------------------------------


def _decode_call(runner_cls, kernel_cls, batch, seq_len_q, *, kv_len=2048):
    """Run one split-KV decode launch of ``batch`` requests; returns (output, kernel kwargs seen,
    cache keys looked up). A fresh kernel cache, so every call compiles and records its kwargs."""
    import cutlass

    num_heads, latent_dim, rope_dim, page_size = HEADS_PER_RANK, 512, 64, 64
    device = torch.device("cuda")
    torch.manual_seed(11)
    blocks = math.ceil(kv_len / page_size)
    pages = batch * blocks
    q = (torch.randn(batch, seq_len_q, num_heads, latent_dim + rope_dim, device=device) * 0.1).to(
        torch.float8_e4m3fn
    )
    cache = (torch.randn(pages, page_size, latent_dim + rope_dim, device=device) * 0.1).to(
        torch.float8_e4m3fn
    )
    page_table = (
        torch.arange(pages, dtype=torch.int32, device=device).view(batch, blocks).transpose(0, 1)
    )
    cache_seqs = torch.full((batch,), kv_len, dtype=torch.int32, device=device)
    kv_bounds = torch.tensor(
        [kv_len - (seq_len_q - 1 - s) for _ in range(batch) for s in range(seq_len_q)],
        dtype=torch.int32,
        device=device,
    )
    runner = runner_cls(
        in_dtype=cutlass.Float8E4M3FN,
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        page_size=page_size,
        max_batch_size=batch,
        emit_softmax_stats=False,
    )
    workspace = torch.empty(
        runner.get_max_padded_workspace_size(
            num_heads, seq_len_q, latent_dim, batch, cutlass.Float32
        ),
        device=device,
        dtype=torch.uint8,
    )
    out = torch.zeros(batch, seq_len_q, num_heads, latent_dim, device=device, dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(latent_dim + rope_dim)
    inputs = [
        q[..., :latent_dim].permute(2, 3, 1, 0),
        q[..., latent_dim:].permute(2, 3, 1, 0),
        cache[..., :latent_dim].permute(1, 2, 0),
        cache[..., latent_dim:].permute(1, 2, 0),
        page_table,
        cache_seqs,
        out.permute(2, 3, 1, 0),
        workspace,
        None,
        kv_bounds,
        torch.tensor([scale], device=device),
        torch.tensor([1.0], device=device),
    ]
    runner.forward(inputs, tactic=((128, 128), (128, 256), 16, False))
    torch.cuda.synchronize()
    return out.float()


class _Recorder:
    def __init__(self):
        self.kwargs, self.keys = [], []


def _record(monkeypatch, runner_cls):
    """Fresh kernel cache; records the constructor kwargs of every kernel built and every key."""
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
        BlackwellMultiHeadLatentAttentionForwardFP8 as kernel_cls,
    )

    rec = _Recorder()

    class Cache(dict):
        def __contains__(self, key):
            rec.keys.append(key)
            return super().__contains__(key)

    monkeypatch.setattr(runner_cls, "kernel_cache", Cache())
    init = kernel_cls.__init__

    def spy(self, *args, **kwargs):
        rec.kwargs.append((kwargs.get("reducer_d_tiles"), kwargs.get("reducer_prefetch_splits")))
        return init(self, *args, **kwargs)

    monkeypatch.setattr(kernel_cls, "__init__", spy)
    return rec, kernel_cls


@requires_sm100
@pytest.mark.parametrize(
    "kind,batch",
    [("K6", 3), ("K6", 4), ("K6", 8), ("K5", 4), ("K5", 5), ("K5", 8)],
)
def test_launch_arguments_follow_the_cap(monkeypatch, kind: str, batch: int) -> None:
    """The kernel is built with (bands, prefetch splits) = (4, 32) below the cap and (1, 0) above
    it, and the kernel cache key carries the same two fields."""
    import tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops as ops

    runner_cls = ops.CuteDSLNVMlaDecodeBlackwellRunner
    monkeypatch.setattr(runner_cls, "_REDUCTION_PREFETCH", True)
    monkeypatch.setattr(runner_cls, "_REDUCTION_PREFETCH_MAX_ROWS", CAP)
    rec, kernel_cls = _record(monkeypatch, runner_cls)
    seq_len_q = DECODE_SHAPES[kind]
    out_on = _decode_call(runner_cls, kernel_cls, batch, seq_len_q)
    expected = (4, 32) if batch <= LAST_ON_BATCH[kind] else (1, 0)
    assert rec.kwargs == [expected]
    assert rec.keys[-1][-2:] == expected

    # Switch off: the default reduction, bitwise the same output.
    monkeypatch.setattr(runner_cls, "_REDUCTION_PREFETCH", False)
    rec_off, _ = _record(monkeypatch, runner_cls)
    out_off = _decode_call(runner_cls, kernel_cls, batch, seq_len_q)
    assert rec_off.kwargs == [(1, 0)] and rec_off.keys[-1][-2:] == (1, 0)
    assert torch.equal(out_on, out_off)
