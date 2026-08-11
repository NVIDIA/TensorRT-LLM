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
"""Dispatch coverage for the DSv4 MLA prologue fusions.

These fusions are chosen inside `MLA`, above the attention backend, so the
backend-level tests in `_torch/attention/sparse/deepseek_v4/` cannot see them:
they build `fused_q` / `latent_cache` / `q_pe` themselves and call
`DeepseekV4TrtllmAttention.forward` directly.

What is checked here is *which path gets taken*, not the numerics -- every
failure this suite was written for was a silent fallback or a plumbing mismatch
rather than a wrong number:

  * the fusions require an FP8 KV cache, so a bf16 config must report them off
    (a bf16-only test otherwise "passes" while exercising nothing);
  * a mixed batch must produce one launch spec per phase instead of falling back;
  * kv-norm fusion and the Q RoPE fold have to move together, because the
    un-fused RoPE kernels would read the raw latent the KV fusion leaves behind.

The fused kernels' numerics live in `test_deepseek_v4_q_norm_fused_rope.py`;
end-to-end agreement is `TestDeepSeekV4Pro::test_gsm8k_full_accuracy`.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.attention_backend.interface import PositionalEmbeddingParams, RopeParams
from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.module import (
    _fused_q_rope_specs,
    _is_fused_kv_norm_enabled,
    _is_fused_prologue_active,
    _is_fused_q_fp8_quant_enabled,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.mla import MLA
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm.functional import PositionEmbeddingType

# DSv4-Pro latent geometry: 448 nope + 64 rope = a 512-wide head.
KV_LORA_RANK = 448
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 448
HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM


class _FakeAttention(nn.Module):
    """Stands in for the attention backend; only the fusion inputs matter."""

    def __init__(self, has_fp8_kv_cache: bool):
        super().__init__()
        self.has_fp8_kv_cache = has_fp8_kv_cache
        self.rotary_cos_sin = torch.zeros(8, dtype=torch.float32)

    def support_fused_rope(self) -> bool:
        return True

    def update_quant_config(self, _quant_config: object) -> None:
        pass

    def _ensure_rope_table_size(self, _max_seq_len: int) -> None:
        pass


def _make_mla(
    *,
    has_fp8_kv_cache: bool,
    dsv4_geometry: bool = True,
    kv_lora_rank: int = KV_LORA_RANK,
    qk_rope_head_dim: int = QK_ROPE_HEAD_DIM,
) -> MLA:
    config = ModelConfig(skip_create_weights_in_init=True)
    position_embedding = PositionalEmbeddingParams(
        type=PositionEmbeddingType.rope_gpt_neox,
        rope=RopeParams(dim=QK_ROPE_HEAD_DIM, max_positions=8192),
    )
    with patch(
        "tensorrt_llm._torch.modules.mla.create_attention",
        side_effect=lambda *a, **kw: _FakeAttention(has_fp8_kv_cache),
    ):
        mla = MLA(
            hidden_size=64,
            num_attention_heads=2,
            num_key_value_heads=1,
            qk_nope_head_dim=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=kv_lora_rank,
            q_lora_rank=32,
            kv_lora_rank=kv_lora_rank,
            predicted_tokens_per_seq=1,
            max_position_embeddings=8192,
            bias=False,
            pos_embd_params=position_embedding,
            layer_idx=0,
            dtype=torch.bfloat16,
            config=config,
        )
    # DSv4 widens kv_a_layernorm to the whole 512 latent, and
    # `_is_fused_kv_norm_enabled` checks that width. Do NOT fabricate attributes the
    # real module lacks: the predicates live in the DSv4 sparse module, so reaching
    # them already implies DSv4 and only the geometry is actually checked.
    if dsv4_geometry:
        mla.kv_a_layernorm = RMSNorm(
            hidden_size=kv_lora_rank + qk_rope_head_dim, dtype=torch.bfloat16, eps=1e-6
        )
    return mla


def _make_metadata(
    *, num_ctx_tokens: int, num_tokens: int, num_seqs: int, num_contexts: int = 0
) -> SimpleNamespace:
    """Only the five attributes `_fused_q_rope_specs` reads."""
    # Production returns `mla_ctx_cu_q_seqlens[:num_contexts + 1]`, so match that length.
    cu_ctx = torch.zeros(num_contexts + 1, dtype=torch.int32)
    return SimpleNamespace(
        kv_lens_cuda_runtime=torch.arange(num_seqs, dtype=torch.int32),
        num_ctx_tokens=num_ctx_tokens,
        num_tokens=num_tokens,
        max_seq_len=8192,
        mla_prepare_ctx_cu_seqlens=lambda: cu_ctx,
    )


@pytest.mark.parametrize("has_fp8_kv_cache", [True, False])
def test_fusions_require_fp8_kv_cache(has_fp8_kv_cache: bool) -> None:
    """Both predicates hang off the KV-cache dtype.

    A bf16 configuration must report the fusions off. Without this the
    backend-level suites, which build a bf16 cache, look like coverage while
    never entering a fused path.
    """
    mla = _make_mla(has_fp8_kv_cache=has_fp8_kv_cache)
    assert _is_fused_kv_norm_enabled(mla, num_generations=1) is has_fp8_kv_cache
    assert _is_fused_q_fp8_quant_enabled(mla, num_generations=1, num_contexts=0) is has_fp8_kv_cache


def test_kv_norm_fusion_needs_the_full_width_weight() -> None:
    """The KV kernels norm the whole 512 latent, so a 448-wide weight must bail.

    This is the guard against an out-of-bounds read, not a style check: the kernel
    indexes `kv_norm_weight` across `K_DIM + ROPE_DIM` regardless of its length.
    """
    mla = _make_mla(has_fp8_kv_cache=True, dsv4_geometry=False)
    assert mla.kv_a_layernorm.weight.shape[0] == KV_LORA_RANK
    assert _is_fused_kv_norm_enabled(mla, num_generations=1) is False


def test_rope_specs_mixed_batch_splits_by_phase() -> None:
    """The regression this file exists for, and the only per-phase spec test.

    Pure-context and pure-generation cases were dropped: mutation attribution
    showed the generation-only test killed no mutant, and every mutant the
    context-only test killed is also killed here. A mixed batch exercises both
    position rules at once, so it strictly dominates them.

    A mixed batch needs both position rules, so it gets one spec per phase. When
    this returned nothing the fused path silently fell back to
    `applyMLARopeAndAssignQKVKernel*` and no test noticed.
    """
    # 2 context sequences (96 tokens) + 3 generation sequences (3 tokens).
    metadata = _make_metadata(num_ctx_tokens=96, num_tokens=99, num_seqs=5, num_contexts=2)
    mla = _make_mla(has_fp8_kv_cache=True)

    cos_sin, specs = _fused_q_rope_specs(mla, metadata, num_contexts=2, num_generations=3)

    assert cos_sin is not None
    assert len(specs) == 2, "mixed batch must not fall back to a single launch"

    (
        (ctx_rows, ctx_cache_lens, ctx_seq_len, ctx_cu),
        (
            gen_rows,
            gen_cache_lens,
            gen_seq_len,
            gen_cu,
        ),
    ) = specs

    # Context first, generation second, disjoint and covering every row exactly once.
    assert ctx_rows == slice(0, 96)
    assert gen_rows == slice(96, 99)
    assert ctx_rows.stop == gen_rows.start

    assert ctx_seq_len == 0 and ctx_cu is not None
    assert gen_seq_len == 1 and gen_cu is None

    # Each half sees only its own sequences' cache lengths.
    assert ctx_cache_lens.shape[0] == 2
    assert gen_cache_lens.shape[0] == 3


def test_kv_norm_fusion_is_coupled_to_the_q_rope_fold() -> None:
    """The two fusions must move together.

    The KV fusion hands the un-fused RoPE kernels the RAW latent, so their Q
    region would read it un-normalized. That is only safe because the fused Q
    path takes the Q side over entirely -- enabling one without the other is a
    silent correctness bug, not a slower path.
    """
    mla = _make_mla(has_fp8_kv_cache=True)
    metadata = _make_metadata(num_ctx_tokens=96, num_tokens=96, num_seqs=3, num_contexts=3)
    metadata.mla_prepare_ctx_cu_seqlens = None  # forces the Q fold off

    _cos_sin, specs = _fused_q_rope_specs(mla, metadata, num_contexts=3, num_generations=0)
    assert not specs

    # The KV predicate on its own still says yes -- so the coupling, not a shared
    # precondition, is what has to turn the fusion off.
    assert _is_fused_kv_norm_enabled(mla, num_generations=0) is True

    # `forward_impl_with_deepseek_v4` assigns `_fused_kv_norm_active` from exactly
    # this call, so asserting on it here is asserting on the shipped decision.
    assert (
        _is_fused_prologue_active(mla, num_contexts=3, num_generations=0, rope_specs=specs) is False
    ), "kv-norm fusion must not engage when the Q RoPE fold is unavailable"

    # ...and it does engage once the specs exist, so the False above is the coupling
    # talking and not a predicate that is off for some unrelated reason.
    assert (
        _is_fused_prologue_active(
            mla, num_contexts=3, num_generations=0, rope_specs=[("dummy", None, 0, None)]
        )
        is True
    )


@pytest.mark.parametrize(
    "kv_lora_rank,qk_rope_head_dim",
    [(512, 64), (448, 128)],
    ids=["lora512", "rope128"],
)
def test_fusions_require_the_448_64_latent(kv_lora_rank: int, qk_rope_head_dim: int) -> None:
    """The kernels hard-code the latent row in template constants.

    `mlaKvNormRopeQuant*Kernel` is instantiated at K_DIM=448 / ROPE_DIM=64, so a
    model whose latent is shaped differently must not reach it -- the kernel would
    stride the wrong row width. Every other fixture here builds DSv4 geometry, so
    without this case the guard is unreachable and deleting it breaks no test.
    """
    mla = _make_mla(
        has_fp8_kv_cache=True,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
    )
    assert _is_fused_kv_norm_enabled(mla, num_generations=1) is False
