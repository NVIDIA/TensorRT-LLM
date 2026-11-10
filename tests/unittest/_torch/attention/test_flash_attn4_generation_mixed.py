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
"""Mixed context+generation batches through TrtllmAttention with FlashAttention-4 enabled.

The context requests run on the regular op (its context phase) and the generation rows on
FA4 (``CombinedFmha``); the whole forward is compared with the VanillaAttention dense
reference, and the KV cache is checked to hold every new token (``run_backend``).
"""

import pytest
import torch
from backend_case import BackendCase, _tolerances, generate_inputs, run_backend

from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
from tensorrt_llm._torch.attention.backends.fmha.flash_attn4_generation import (
    FlashAttn4GenerationFmha,
)
from tensorrt_llm._torch.attention.backends.fmha.flash_attn4_kernels import fa4_forward
from tensorrt_llm._torch.attention.backends.interface import AttentionInputType
from tensorrt_llm._utils import get_sm_version

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or get_sm_version() != 90 or fa4_forward() is None,
        reason="FlashAttention-4 generation path is SM90-only",
    ),
]

_ROPE = dict(dim=128, theta=1000000.0, max_positions=8192, is_neox=True)


def _case(num_heads, num_kv_heads, q_len, *, gen_lens=None):
    # Context requests start from an empty cache: this harness runs the context op without the
    # paged context kernel, so a cached prefix is not attended (covered end to end instead).
    ctx_lens = [37, 130, 64]
    ctx_cached = [0, 0, 0]
    gen_cached = [31, 64, 100, 257, 5, 63, 128]
    gen_lens = gen_lens or [q_len] * len(gen_cached)
    return BackendCase(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=128,
        seq_lens=ctx_lens + gen_lens,
        num_cached_tokens=ctx_cached + gen_cached,
        num_contexts=len(ctx_lens),
        dtype="bfloat16",
        rope=_ROPE,
        fused_rope=True,
        page_size=64,
    )


@pytest.fixture
def spies(monkeypatch):
    """Record FA4 generation calls, the input type of every whole-batch regular-op call, and
    every phase the regular op ran."""
    calls = {"fa4_rows": [], "fallback": [], "fallback_phases": []}
    run_generation = FlashAttn4GenerationFmha.run_generation
    fallback_forward = FallbackFmha.forward
    fallback_context = FallbackFmha.run_context
    fallback_generation = FallbackFmha.run_generation

    def fa4_spy(self, params):
        calls["fa4_rows"].append((params.batch_size, params.num_tokens, params.seq_offset))
        return run_generation(self, params)

    def fallback_spy(self, q, k, v, metadata, forward_args):
        calls["fallback"].append(forward_args.attention_input_type)
        return fallback_forward(self, q, k, v, metadata, forward_args)

    def fallback_context_spy(self, params):
        calls["fallback_phases"].append("context")
        return fallback_context(self, params)

    def fallback_generation_spy(self, params):
        calls["fallback_phases"].append("generation")
        return fallback_generation(self, params)

    monkeypatch.setattr(FlashAttn4GenerationFmha, "run_generation", fa4_spy)
    monkeypatch.setattr(FallbackFmha, "forward", fallback_spy)
    monkeypatch.setattr(FallbackFmha, "run_context", fallback_context_spy)
    monkeypatch.setattr(FallbackFmha, "run_generation", fallback_generation_spy)
    return calls


def _run(case, monkeypatch, libs):
    # Short cached lengths keep the reference cheap; serve them with FA4 regardless of length.
    monkeypatch.setenv("TLLM_FLASH_ATTN4_GENERATION_MIXED_MIN_KV", "0")
    if libs is None:
        monkeypatch.delenv("TLLM_FMHA_LIBS", raising=False)
    else:
        monkeypatch.setenv("TLLM_FMHA_LIBS", libs)
    inputs = generate_inputs(case, seed=7)
    golden = run_backend(case, "VANILLA", inputs, kv_dtype=torch.bfloat16, kv_layout="NHD")
    out = run_backend(
        case, "TRTLLM", inputs, kv_dtype=torch.bfloat16, fuse_rope=True, kv_layout="HND"
    )
    atol, rtol = _tolerances(case, torch.bfloat16)
    return golden, out, atol, rtol


@pytest.mark.parametrize("num_heads,num_kv_heads", [(28, 4), (20, 20)], ids=["gqa", "mha"])
@pytest.mark.parametrize("q_len", [1, 4, 9])
@pytest.mark.parametrize(
    "libs",
    ["+flash_attn4_generation", "+sampling_shared_prefix,+flash_attn4_generation"],
    ids=["fa4", "library+fa4"],
)
def test_mixed_batch_generation_rows_on_fa4_match_dense(
    num_heads, num_kv_heads, q_len, libs, spies, monkeypatch
):
    case = _case(num_heads, num_kv_heads, q_len)
    golden, out, atol, rtol = _run(case, monkeypatch, libs)
    num_contexts = case.num_contexts
    num_generations = case.num_seqs - num_contexts
    # FA4 served exactly the generation rows; the regular op ran only its context phase.
    assert spies["fa4_rows"] == [(num_generations, num_generations * q_len, num_contexts)]
    assert spies["fallback"] == [] and spies["fallback_phases"] == ["context"]
    torch.testing.assert_close(out.float(), golden.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("q_len", [1])
def test_mixed_batch_without_fa4_keeps_the_regular_op(q_len, spies, monkeypatch):
    """FA4 not enabled: the regular op serves the whole mixed batch, as before. (Multi-token
    rows need speculative-decoding metadata on that op, which this harness does not build.)"""
    case = _case(28, 4, q_len)
    golden, out, atol, rtol = _run(case, monkeypatch, None)
    assert spies["fa4_rows"] == []
    assert spies["fallback"] == [AttentionInputType.mixed]
    torch.testing.assert_close(out.float(), golden.float(), atol=atol, rtol=rtol)


def test_mixed_batch_with_ragged_generation_rows_stays_on_the_regular_op(spies, monkeypatch):
    """Generation rows of different lengths that still divide evenly are not sliced for FA4.

    Only the dispatch is checked: the regular op needs speculative-decoding metadata for
    multi-token rows, which this harness does not build."""
    import backend_case

    monkeypatch.setattr(backend_case, "_assert_cache_contains_new_tokens", lambda *a, **k: None)
    case = _case(28, 4, 1, gen_lens=[1, 1, 1, 1, 1, 2, 7])
    monkeypatch.setenv("TLLM_FMHA_LIBS", "+flash_attn4_generation")
    monkeypatch.setenv("TLLM_FLASH_ATTN4_GENERATION_MIXED_MIN_KV", "0")
    inputs = generate_inputs(case, seed=7)
    run_backend(case, "TRTLLM", inputs, kv_dtype=torch.bfloat16, fuse_rope=True, kv_layout="HND")
    assert spies["fa4_rows"] == []
    assert spies["fallback"] == [AttentionInputType.mixed]


@pytest.mark.parametrize("min_kv, uses_fa4", [(None, False), ("90", True), ("100", False)])
def test_mixed_batch_kv_length_gate(min_kv, uses_fa4, spies, monkeypatch):
    """Generation rows of a mixed step go to FA4 only when their mean KV length reaches
    TLLM_FLASH_ATTN4_GENERATION_MIXED_MIN_KV (default 2048); the mean here is ~97 tokens."""
    case = _case(28, 4, 4)
    if min_kv is None:
        monkeypatch.delenv("TLLM_FLASH_ATTN4_GENERATION_MIXED_MIN_KV", raising=False)
    else:
        monkeypatch.setenv("TLLM_FLASH_ATTN4_GENERATION_MIXED_MIN_KV", min_kv)
    monkeypatch.setenv("TLLM_FMHA_LIBS", "+flash_attn4_generation")
    inputs = generate_inputs(case, seed=7)
    if uses_fa4:
        golden = run_backend(case, "VANILLA", inputs, kv_dtype=torch.bfloat16, kv_layout="NHD")
    else:
        # The regular op's multi-token rows need speculative-decoding metadata this harness lacks.
        import backend_case

        monkeypatch.setattr(backend_case, "_assert_cache_contains_new_tokens", lambda *a, **k: None)
    out = run_backend(
        case, "TRTLLM", inputs, kv_dtype=torch.bfloat16, fuse_rope=True, kv_layout="HND"
    )
    if uses_fa4:
        assert spies["fa4_rows"] and spies["fallback"] == []
        assert spies["fallback_phases"] == ["context"]
        atol, rtol = _tolerances(case, torch.bfloat16)
        torch.testing.assert_close(out.float(), golden.float(), atol=atol, rtol=rtol)
    else:
        assert spies["fa4_rows"] == [] and spies["fallback"] == [AttentionInputType.mixed]
