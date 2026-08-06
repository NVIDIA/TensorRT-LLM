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


def _make_mla(*, has_fp8_kv_cache: bool, is_deepseek_v4: bool = True) -> MLA:
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
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            v_head_dim=KV_LORA_RANK,
            q_lora_rank=32,
            kv_lora_rank=KV_LORA_RANK,
            predicted_tokens_per_seq=1,
            max_position_embeddings=8192,
            bias=False,
            pos_embd_params=position_embedding,
            layer_idx=0,
            dtype=torch.bfloat16,
            config=config,
        )
    # `is_deepseek_v4` comes from the sparse-attention config on the real model;
    # set it directly so the test does not depend on that plumbing. The flag also
    # widens kv_a_layernorm to the whole 512 latent inside __init__, so rebuild it
    # here to match -- `_is_fused_kv_norm_enabled` checks that width.
    mla.is_deepseek_v4 = is_deepseek_v4
    if is_deepseek_v4:
        mla.kv_a_layernorm = RMSNorm(
            hidden_size=KV_LORA_RANK + QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, eps=1e-6
        )
    return mla


def _make_metadata(*, num_ctx_tokens: int, num_tokens: int, num_seqs: int):
    """Only the four fields `_fused_q_rope_specs` reads."""
    cu_ctx = torch.zeros(num_seqs + 1, dtype=torch.int32)
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
    assert mla._is_fused_kv_norm_enabled(num_generations=1) is has_fp8_kv_cache
    assert mla._is_fused_q_fp8_quant_enabled(num_generations=1, num_contexts=0) is has_fp8_kv_cache


def test_fusions_off_for_non_dsv4() -> None:
    """v3 / v3.2 keep the upstream kernels; the 448 geometry alone is not enough."""
    mla = _make_mla(has_fp8_kv_cache=True, is_deepseek_v4=False)
    assert mla._is_fused_kv_norm_enabled(num_generations=1) is False
    assert mla._is_fused_q_fp8_quant_enabled(num_generations=1, num_contexts=0) is False


def test_rope_specs_context_only() -> None:
    mla = _make_mla(has_fp8_kv_cache=True)
    metadata = _make_metadata(num_ctx_tokens=96, num_tokens=96, num_seqs=3)

    cos_sin, specs = mla._fused_q_rope_specs(metadata, num_contexts=3, num_generations=0)

    assert cos_sin is not None
    assert len(specs) == 1
    rows, cache_lens, seq_len, cu_q_seqlens = specs[0]
    assert rows == slice(0, 96)
    assert seq_len == 0, "context positions are ragged, not a uniform query length"
    assert cu_q_seqlens is not None
    assert cache_lens.shape[0] == 3


def test_rope_specs_generation_only() -> None:
    mla = _make_mla(has_fp8_kv_cache=True)
    # 4 sequences, 2 query tokens each (MTP1-style), no context rows.
    metadata = _make_metadata(num_ctx_tokens=0, num_tokens=8, num_seqs=4)

    cos_sin, specs = mla._fused_q_rope_specs(metadata, num_contexts=0, num_generations=4)

    assert cos_sin is not None
    assert len(specs) == 1
    rows, cache_lens, seq_len, cu_q_seqlens = specs[0]
    assert rows == slice(0, 8)
    assert seq_len == 2, "generation positions come from a uniform query length"
    assert cu_q_seqlens is None
    assert cache_lens.shape[0] == 4


def test_rope_specs_mixed_batch_splits_by_phase() -> None:
    """The regression this file exists for.

    A mixed batch needs both position rules, so it gets one spec per phase. When
    this returned nothing the fused path silently fell back to
    `applyMLARopeAndAssignQKVKernel*` and no test noticed.
    """
    # 2 context sequences (96 tokens) + 3 generation sequences (3 tokens).
    metadata = _make_metadata(num_ctx_tokens=96, num_tokens=99, num_seqs=5)
    mla = _make_mla(has_fp8_kv_cache=True)

    cos_sin, specs = mla._fused_q_rope_specs(metadata, num_contexts=2, num_generations=3)

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
    metadata = _make_metadata(num_ctx_tokens=96, num_tokens=96, num_seqs=3)
    metadata.mla_prepare_ctx_cu_seqlens = None  # forces the Q fold off

    _cos_sin, specs = mla._fused_q_rope_specs(metadata, num_contexts=3, num_generations=0)
    assert not specs

    # Both ingredients of the decision `forward_impl_with_deepseek_v4` makes.
    assert mla._is_fused_kv_norm_enabled(num_generations=0) is True
    fused_kv_norm_active = (
        mla._is_fused_kv_norm_enabled(num_generations=0)
        and mla._is_fused_q_fp8_quant_enabled(num_generations=0, num_contexts=3)
        and bool(specs)
    )
    assert fused_kv_norm_active is False, (
        "kv-norm fusion must not engage when the Q RoPE fold is unavailable"
    )
