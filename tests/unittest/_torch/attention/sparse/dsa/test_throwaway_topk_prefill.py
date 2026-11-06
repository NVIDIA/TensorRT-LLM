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
"""Correctness tests for the fused throwaway Top-K used by the DSA prefill indexer.

GPU tests compare ``ThrowawayPrefillTopK.run_chunk`` with a PyTorch reference. Exact Top-K
implementations may pick different indices among tied scores, so a selection is valid when it
has the right size, lies inside the causal window, has no duplicates and every selected score
is at least the k-th largest reference score. The dispatch tests run on CPU with a fake runner.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import (
    Indexer,
    IndexerPrefillChunkMetadata,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.throwaway_topk import (
    THROWAWAY_MAX_CHUNK_ENV,
    ThrowawayPrefillTopK,
)
from tensorrt_llm._utils import get_sm_version

HEAD_DIM = 128
REF_ROW_BLOCK = 64
# Relative slack on the k-th largest score: ~1 ulp of the logits storage dtype.
SCORE_SLACK = {torch.float32: 1e-3, torch.bfloat16: 8e-3, torch.float16: 2e-3}

skip_not_sm100 = pytest.mark.skipif(
    get_sm_version() not in (100, 103),
    reason=f"throwaway Top-K supports SM 100/103 only, got SM {get_sm_version()}",
)

# (num_heads, top_k, compress_ratio) pairs with a kernel policy
CONFIGS = {
    "ds_v32": (64, 2048, 1),
    "glm_5_3": (32, 2048, 1),
    "dsv4_flash": (64, 512, 4),
}

_RUNNERS = {}


def _runner(
    config, max_context_length, max_chunk=None, monkeypatch=None, logits_dtype=torch.float32
):
    """Runners JIT-compile on first use, so reuse them across tests."""
    key = (config, max_context_length, max_chunk, logits_dtype)
    if key not in _RUNNERS:
        heads, top_k, ratio = CONFIGS[config]
        if max_chunk is not None:
            monkeypatch.setenv(THROWAWAY_MAX_CHUNK_ENV, str(max_chunk))
        _RUNNERS[key] = ThrowawayPrefillTopK(
            n_heads=heads,
            index_topk=top_k,
            compress_ratio=ratio,
            max_context_length=max_context_length,
            device=torch.device("cuda"),
            logits_dtype=logits_dtype,
        )
    return _RUNNERS[key]


def _make_inputs(config, num_q, kv_prefix, seed):
    heads, _, ratio = CONFIGS[config]
    num_k = (kv_prefix + num_q) // ratio
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(num_q, heads, HEAD_DIM, device="cuda", generator=gen).to(torch.float8_e4m3fn)
    k = torch.randn(num_k, HEAD_DIM, device="cuda", generator=gen).to(torch.float8_e4m3fn)
    k_scale = (torch.rand(num_k, device="cuda", generator=gen) * 0.5 + 0.5).float()
    weights = torch.rand(num_q, heads, device="cuda", generator=gen).float()
    return q, k, k_scale, weights


def _causal_ends(num_q, kv_prefix, ratio):
    return (kv_prefix + torch.arange(1, num_q + 1, device="cuda")) // ratio


def _reference_logits_block(q, k, k_scale, weights, ends, rows):
    """logits[i, j] = sum_h w[i, h] * relu(q[i, h] . k[j] * k_scale[j]); -inf outside [0, end)."""
    dots = torch.einsum("qhd,kd->qhk", q[rows].float(), k.float()) * k_scale[None, None, :]
    logits = (dots.relu() * weights[rows][:, :, None]).sum(1)
    cols = torch.arange(k.shape[0], device=k.device)[None, :]
    return logits.masked_fill(cols >= ends[rows][:, None], float("-inf"))


def _assert_valid_topk(
    out, q, k, k_scale, weights, ends, top_k, context, logits_dtype=torch.float32
):
    """Tie-aware check; scores are rounded to ``logits_dtype`` like the kernel stores them.

    A score computed with a different FP32 summation order can round to the neighbouring
    representable value, so the k-th-largest comparison allows one unit in the last place.
    """
    num_q = out.shape[0]
    for begin in range(0, num_q, REF_ROW_BLOCK):
        rows = slice(begin, min(begin + REF_ROW_BLOCK, num_q))
        ref = _reference_logits_block(q, k, k_scale, weights, ends, rows)
        if logits_dtype != torch.float32:
            ref = ref.to(logits_dtype).float()
        sel = out[rows].long()
        valid = sel >= 0
        row_ends = ends[rows]
        expected = row_ends.clamp(max=top_k)
        assert torch.equal(valid.sum(1), expected), (
            f"{context}: selected counts {valid.sum(1).tolist()[:8]} != {expected.tolist()[:8]}"
        )
        # Padding must be exactly -1 (not stale buffer contents).
        assert bool(((sel == -1) | valid).all()), f"{context}: padding is not -1"
        assert bool(((sel < row_ends[:, None]) | ~valid).all()), (
            f"{context}: index outside the causal window"
        )
        ordered = sel.masked_fill(~valid, -1).sort(dim=1).values
        assert not bool(((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any()), (
            f"{context}: duplicate indices"
        )
        long_rows = row_ends > top_k
        if bool(long_rows.any()):
            kth = ref.topk(top_k, dim=1).values[:, -1]
            picked = ref.gather(1, sel.clamp(min=0))
            slack = SCORE_SLACK[logits_dtype] * kth.abs().clamp(min=1.0)
            bad = valid & long_rows[:, None] & (picked < (kth - slack)[:, None])
            assert not bool(bad.any()), f"{context}: selected a score below the k-th largest"


def _run(runner, q, weights, k, k_scale, kv_prefix, top_k, plan_cache=None):
    out = torch.full((q.shape[0], top_k), -2, dtype=torch.int32, device="cuda")
    runner.run_chunk(
        q,
        weights,
        k,
        k_scale,
        out,
        kv_prefix_len=kv_prefix,
        plan_cache={} if plan_cache is None else plan_cache,
    )
    return out


@skip_not_sm100
@pytest.mark.parametrize(
    "num_q,kv_prefix",
    [
        (128, 0),  # first chunk: fewer valid keys than top_k for every row
        (64, 1024),  # still fewer valid keys than top_k
        (256, 4096),  # crosses top_k inside the chunk
        (300, 5000),  # sizes that are not tile multiples
        (512, 8192),
    ],
)
def test_run_chunk_matches_reference_ds_v32(num_q, kv_prefix):
    runner = _runner("ds_v32", 16384)
    q, k, k_scale, weights = _make_inputs("ds_v32", num_q, kv_prefix, seed=1)
    out = _run(runner, q, weights, k, k_scale, kv_prefix, 2048)
    ends = _causal_ends(num_q, kv_prefix, 1)
    _assert_valid_topk(out, q, k, k_scale, weights, ends, 2048, f"q={num_q} prefix={kv_prefix}")


@skip_not_sm100
@pytest.mark.parametrize(
    "config,num_q,kv_prefix", [("glm_5_3", 256, 3840), ("dsv4_flash", 256, 4096)]
)
def test_run_chunk_matches_reference_other_configs(config, num_q, kv_prefix):
    heads, top_k, ratio = CONFIGS[config]
    runner = _runner(config, 16384)
    q, k, k_scale, weights = _make_inputs(config, num_q, kv_prefix, seed=2)
    out = _run(runner, q, weights, k, k_scale, kv_prefix, top_k)
    ends = _causal_ends(num_q, kv_prefix, ratio)
    _assert_valid_topk(out, q, k, k_scale, weights, ends, top_k, f"{config} prefix={kv_prefix}")


@skip_not_sm100
@pytest.mark.parametrize("logits_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("config,num_q,kv_prefix", [("ds_v32", 256, 4096), ("glm_5_3", 192, 5000)])
def test_run_chunk_matches_reference_low_precision_logits(logits_dtype, config, num_q, kv_prefix):
    """Selecting over BF16 / FP16 stored scores must still be an exact top-k of those scores."""
    _, top_k, ratio = CONFIGS[config]
    runner = _runner(config, 16384, logits_dtype=logits_dtype)
    assert runner.logits_dtype == logits_dtype
    q, k, k_scale, weights = _make_inputs(config, num_q, kv_prefix, seed=5)
    out = _run(runner, q, weights, k, k_scale, kv_prefix, top_k)
    ends = _causal_ends(num_q, kv_prefix, ratio)
    _assert_valid_topk(
        out, q, k, k_scale, weights, ends, top_k, f"{config} {logits_dtype}", logits_dtype
    )


@skip_not_sm100
def test_query_tiling_matches_reference(monkeypatch):
    """A chunk larger than max_chunk_size is processed in tiles with shifted K prefixes."""
    runner = _runner("ds_v32", 16384, max_chunk=256, monkeypatch=monkeypatch)
    num_q, kv_prefix = 700, 3000
    q, k, k_scale, weights = _make_inputs("ds_v32", num_q, kv_prefix, seed=3)
    out = _run(runner, q, weights, k, k_scale, kv_prefix, 2048)
    ends = _causal_ends(num_q, kv_prefix, 1)
    _assert_valid_topk(out, q, k, k_scale, weights, ends, 2048, "tiled chunk")


@skip_not_sm100
def test_layers_reuse_plan_cache_with_different_keys():
    """Every layer of a step reuses the chunk's plan but brings different K / Q / weights."""
    runner = _runner("ds_v32", 16384)
    num_q, kv_prefix = 256, 4096
    plan_cache = {}
    ends = _causal_ends(num_q, kv_prefix, 1)
    for layer in range(4):
        q, k, k_scale, weights = _make_inputs("ds_v32", num_q, kv_prefix, seed=10 + layer)
        out = _run(runner, q, weights, k, k_scale, kv_prefix, 2048, plan_cache)
        _assert_valid_topk(out, q, k, k_scale, weights, ends, 2048, f"layer {layer}")


@skip_not_sm100
def test_interleaved_chunks_across_layers():
    """The engine runs layer-outer / chunk-inner, so chunks with different plans alternate."""
    runner = _runner("ds_v32", 16384)
    chunks = [(256, 4096), (192, 6000), (320, 0)]
    caches = [{} for _ in chunks]
    for layer in range(3):
        for index, (num_q, kv_prefix) in enumerate(chunks):
            q, k, k_scale, weights = _make_inputs(
                "ds_v32", num_q, kv_prefix, seed=100 + 10 * layer + index
            )
            out = _run(runner, q, weights, k, k_scale, kv_prefix, 2048, caches[index])
            ends = _causal_ends(num_q, kv_prefix, 1)
            _assert_valid_topk(
                out, q, k, k_scale, weights, ends, 2048, f"layer {layer} chunk {index}"
            )


@skip_not_sm100
def test_bf16_weights_are_accepted():
    runner = _runner("ds_v32", 16384)
    num_q, kv_prefix = 128, 2048
    q, k, k_scale, weights = _make_inputs("ds_v32", num_q, kv_prefix, seed=7)
    weights_bf16 = weights.bfloat16()
    out = _run(runner, q, weights_bf16, k, k_scale, kv_prefix, 2048)
    ends = _causal_ends(num_q, kv_prefix, 1)
    _assert_valid_topk(out, q, k, k_scale, weights_bf16.float(), ends, 2048, "bf16 weights")


# ---------------------------------------------------------------------------------------------
# Dispatch (CPU only): Indexer._run_throwaway_chunk with a fake runner.
# ---------------------------------------------------------------------------------------------


class _FakeRunner:
    def __init__(self, can_run=True):
        self._can_run = can_run
        self.calls = []

    def can_run(self, num_k_rows):
        return self._can_run

    def run_chunk(self, q, weights, k, k_scale, out, *, kv_prefix_len, plan_cache):
        self.calls.append((q, weights, k, k_scale, out, kv_prefix_len, plan_cache))


def _fake_indexer(runner, enabled=True, failed=False):
    return SimpleNamespace(
        _use_throwaway_topk=enabled,
        _throwaway_topk_failed=failed,
        _throwaway_topk=runner,
        n_heads=4,
        index_topk=8,
    )


def _chunk(kv_prefix_len, token_start=3):
    return IndexerPrefillChunkMetadata(
        cu_seqlen_ks=None,
        cu_seqlen_ke=None,
        token_start=token_start,
        token_end=token_start + 10,
        k_token_start=0,
        k_token_end=20,
        kv_prefix_len=kv_prefix_len,
    )


def _call(indexer, chunk, q_begin=0, q_end=10):
    q = torch.arange(32 * 4 * 2, dtype=torch.float32).reshape(32, 4, 2)
    weights = torch.arange(32 * 4, dtype=torch.float32).reshape(32, 4)
    k = torch.zeros(20, 2)
    k_scale = torch.ones(20)
    out = torch.zeros(32, 8, dtype=torch.int32)
    handled = Indexer._run_throwaway_chunk(
        indexer, None, chunk, q, weights, k, k_scale, q_begin, q_end, out
    )
    return handled, q, weights, out


@pytest.mark.parametrize(
    "indexer_kwargs,kv_prefix_len",
    [
        ({"enabled": False}, 5),  # not opted in
        ({"failed": True}, 5),  # creation failed earlier
        ({}, None),  # multi-request chunk
    ],
)
def test_dispatch_falls_back(indexer_kwargs, kv_prefix_len):
    runner = _FakeRunner()
    handled, *_ = _call(_fake_indexer(runner, **indexer_kwargs), _chunk(kv_prefix_len))
    assert handled is False
    assert runner.calls == []


def test_dispatch_falls_back_when_keys_do_not_fit():
    runner = _FakeRunner(can_run=False)
    handled, *_ = _call(_fake_indexer(runner), _chunk(5))
    assert handled is False
    assert runner.calls == []


def test_dispatch_slices_rows_and_shifts_prefix():
    runner = _FakeRunner()
    chunk = _chunk(kv_prefix_len=100, token_start=3)
    handled, q, weights, out = _call(_fake_indexer(runner), chunk, q_begin=2, q_end=7)
    assert handled is True
    assert len(runner.calls) == 1
    q_arg, w_arg, _, _, out_arg, prefix, plan_cache = runner.calls[0]
    # chunk rows start at token_start=3, local rows [2, 7) -> global rows [5, 10)
    assert torch.equal(q_arg, q[5:10])
    assert torch.equal(w_arg, weights[5:10])
    assert out_arg.data_ptr() == out[5:10].data_ptr()
    assert prefix == 100 + 2
    assert plan_cache is chunk.throwaway_plans
