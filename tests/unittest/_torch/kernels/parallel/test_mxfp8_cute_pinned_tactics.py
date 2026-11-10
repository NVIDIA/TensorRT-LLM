# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The pinned FlashInfer CuTe DSL MXFP8 GEMM tactics (custom_ops/mxfp8_cute_pinned_tactics): every entry is
non-split-K, each
pinned call equals FlashInfer's default (heuristic) CuTe DSL path bit for bit, the runner takes the pinned path above 32
rows only, and the kill switch restores the default path."""

import hashlib

import pytest
import torch

from tensorrt_llm._torch.custom_ops import mxfp8_cute_pinned_tactics as pin
from tensorrt_llm._utils import get_sm_version

sm107 = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107, reason="SM107 (Rubin) only"
)


@pytest.fixture(autouse=True)
def _armed(monkeypatch):
    """The table as an opted-in model's executor arms it (RuntimeHooks.mxfp8_pinned_tactics)."""
    monkeypatch.setattr(pin, "_ARMED", [True])


def _table_digest() -> str:
    return hashlib.md5(repr(sorted(pin.TACTICS.items())).encode()).hexdigest()[:8]


def test_table_is_non_split_k_and_pinned():
    assert pin.TACTICS, "empty table"
    for (n, k, m), (tile, cluster, swap, prefetch, split_k) in pin.TACTICS.items():
        assert split_k == 1, ((n, k, m), "a split-K tactic changes the K order (not bitwise)")
        assert m >= 64 and m & (m - 1) == 0 and m <= pin.MAX_BUCKET, (n, k, m)
        assert (
            len(tile) == 2
            and len(cluster) == 2
            and isinstance(swap, bool)
            and isinstance(prefetch, bool)
        )
    # Drift guard: a re-tuned table is a new measured artifact (bitwise + timing), not an edit in place.
    assert _table_digest() == "d43d4a0d", _table_digest()


def test_bucket_and_row_gate():
    assert (
        pin._bucket(33) == 64
        and pin._bucket(64) == 64
        and pin._bucket(65) == 128
        and pin._bucket(184) == 256
    )
    assert pin.tactic_for(32, 4992, 6144) is None  # 1..32 rows keep FlashInfer's own low-M path


def _operands(m, n, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(m, k, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.05).to(torch.bfloat16)
    q, s = torch.ops.trtllm.mxfp8_quantize(x, True)
    wq, ws = torch.ops.trtllm.mxfp8_quantize(w, True)
    return q, s, wq, ws


def _default(q, s, wq, ws):
    from flashinfer import autotune, mm_mxfp8

    with autotune(tune_mode=False, skip_ops="mxfp8_gemm"):
        return mm_mxfp8(
            q, wq.t(), s, ws, out_dtype=torch.bfloat16, use_8x4_sf_layout=False, backend="cute-dsl"
        )


@sm107
@pytest.mark.parametrize("key", sorted(pin.TACTICS), ids=lambda k: f"N{k[0]}_K{k[1]}_M{k[2]}")
def test_pinned_equals_default_bitwise(key):
    n, k, bucket = key
    for m in sorted({bucket // 2 + 1 if bucket > 64 else 33, bucket}):  # both ends of the bucket
        q, s, wq, ws = _operands(m, n, k, seed=hash(key) % 1000 + m)
        assert pin.tactic_for(m, n, k) == pin.TACTICS[key]
        out = pin.run(q, s, wq, ws, torch.bfloat16)
        assert out is not None, "the pinned path declined a table entry"
        assert torch.equal(out, _default(q, s, wq, ws)), (key, m)


@sm107
def test_runner_takes_pinned_above_32_rows_and_kill(monkeypatch):
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import FlashInferMXFP8GemmRunner

    runner = FlashInferMXFP8GemmRunner(torch.bfloat16)
    calls = []
    real = pin.run
    monkeypatch.setattr(pin, "run", lambda *a, **kw: calls.append(a[0].shape[0]) or real(*a, **kw))
    for m in (16, 80):
        q, s, wq, ws = _operands(m, 4992, 6144, seed=m)
        out = runner.forward([q, s, wq, ws], tactic=FlashInferMXFP8GemmRunner.CUTE_DSL)
        assert torch.equal(out, _default(q, s, wq, ws)), m
    assert calls == [16, 80]
    monkeypatch.setattr(pin, "ENABLED", False)
    assert pin.tactic_for(80, 4992, 6144) is None
