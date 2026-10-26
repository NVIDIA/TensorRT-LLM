# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TAIL_BK3: the S4FE position-0 Markov embedding (minimax_m31_markov_prev_embed), the block sampler's per-row
parameters (minimax_m31_expand_block_sampling, behind SpecWorkerBase._block_sampling_params) and the int32 block tokens.
CPU: defaults, the kill switch, the stock fallbacks. GPU (SM107, ops built): bitwise the in-tree torch sequences."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (  # noqa: E501
    dspark_s4 as w1,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.ops.spec import (  # noqa: E501
    m31_markov_epilogue as epi,
)
from tensorrt_llm._torch.models.modeling_speculative import markov_prev_embeddings
from tensorrt_llm._torch.speculative.interface import _RNG_SLOT_SPAN, SpecWorkerBase


def test_on_by_default():
    assert w1.TAIL_BK3
    assert w1.S4WorkerMixin._block_tokens_keep_int32
    assert not SpecWorkerBase._block_tokens_keep_int32


def test_kill_switch():
    code = f"from {w1.__name__} import TAIL_BK3, S4WorkerMixin\nprint(TAIL_BK3, S4WorkerMixin._block_tokens_keep_int32)"
    env = {**os.environ, "TRTLLM_M31_DISABLE_TAIL_BK3": "1"}
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.split()[-2:] == ["False", "False"]


class _Base:
    def _block_sampling_params(self, spec_metadata, num_contexts, batch_size, K):
        return "stock"


class _Worker(w1.S4WorkerMixin, _Base):
    pass


def _meta(n, device="cpu", top_ks_dtype=torch.int32):
    return SimpleNamespace(
        request_temperatures=torch.rand(n, device=device),
        request_top_ks=torch.zeros(n, dtype=top_ks_dtype, device=device),
        request_top_ps=torch.rand(n, device=device),
        request_min_ps=torch.zeros(n, device=device),
        request_seeds=torch.randint(0, 2**40, (n,), device=device),
        request_offsets=torch.randint(0, 2**40, (n,), device=device),
        rng_draft_slot=lambda step: 4 + step,
    )


def test_cpu_buffers_take_the_stock_params():
    assert _Worker()._block_sampling_params(_meta(8), 0, 8, 3) == "stock"


def test_unsupported_dtype_takes_the_stock_params(monkeypatch):
    monkeypatch.setattr(w1, "markov_epilogue_op", lambda: object())
    meta = _meta(8, top_ks_dtype=torch.int64)
    assert _Worker()._block_sampling_params(meta, 0, 8, 3) == "stock"


_GPU = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 7)
    or epi.markov_epilogue_op() is None
    or not hasattr(torch.ops.trtllm, "minimax_m31_expand_block_sampling"),
    reason="the TAIL_BK3 ops run on SM107 with the ops built",
)


def _same(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype == torch.bfloat16:
        return torch.equal(a.view(torch.int16), b.view(torch.int16))
    if a.dtype == torch.float32:
        return torch.equal(a.view(torch.int32), b.view(torch.int32))
    return torch.equal(a, b)


@_GPU
@pytest.mark.parametrize("n", [1, 22, 28, 112])
def test_markov_embedding_bitwise(n):
    v, rank = 200064, 256
    w = (torch.randn(v, rank, device="cuda") * 0.05).to(torch.bfloat16)
    w[7, 3], w[11, :] = float("nan"), -0.0
    prev = torch.randint(0, v, (n,), device="cuda")
    edge = torch.tensor([-1, v, 0, v - 1, 7, 11], device="cuda")
    prev[: min(n, 6)] = edge[: min(n, 6)]
    assert _same(markov_prev_embeddings(prev, w), epi._prev_embeddings(prev, w))


@_GPU
@pytest.mark.parametrize("n", [1, 24, 112])
@pytest.mark.parametrize("k", [3, 7])
def test_sampling_params_bitwise(n, k):
    meta = _meta(n + 2, device="cuda")
    meta.request_temperatures[2] = float("nan")
    meta.request_offsets[-1] = 2**63 - 40
    got = _Worker()._block_sampling_params(meta, 2, n + 2, k)
    slot = meta.rng_draft_slot(0)
    sl = slice(2, n + 2)
    offsets = (meta.request_offsets[sl] + slot * _RNG_SLOT_SPAN).repeat_interleave(
        k
    ) + torch.arange(k, dtype=torch.int64, device="cuda").repeat(n) * _RNG_SLOT_SPAN
    ref = (
        meta.request_temperatures[sl].repeat_interleave(k),
        meta.request_top_ks[sl].repeat_interleave(k),
        meta.request_top_ps[sl].repeat_interleave(k),
        meta.request_min_ps[sl].repeat_interleave(k),
        meta.request_seeds[sl].repeat_interleave(k),
        offsets,
    )
    assert got != "stock" and all(_same(a, b) for a, b in zip(ref, got))
