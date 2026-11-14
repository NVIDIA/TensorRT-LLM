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

import math
from dataclasses import replace

import pytest
import torch
from backend_case import (
    BackendCase,
    _build_mla_kv_cache_manager,
    _mla_context_pos_embd_params,
    _prepare_mla_context_inputs,
    _quant_config,
    _tolerances,
    generate_mla_context_inputs,
)
from utils.util import getSMVersion

from tensorrt_llm._torch.attention.backends import TrtllmAttention
from tensorrt_llm._torch.attention.backends.fmha.phased import FmhaParams
from tensorrt_llm._torch.attention.backends.fmha.prims_ts import PrimsTSFmha
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    AttentionRuntimeFeatures,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.attention.backends.utils import create_attention
from tensorrt_llm._torch.metadata import KVCacheParams

pytestmark = pytest.mark.skipif(
    getSMVersion() not in (100, 103, 107),
    reason="PrimsTS attention kernels require SM100, SM103 or SM107",
)


def _ragged_causal_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, case: BackendCase
) -> torch.Tensor:
    """FP64 attention with each query aligned to the suffix of its complete KV sequence."""
    outputs = []
    q_offset = kv_offset = 0
    scale = 1.0 / (math.sqrt(case.head_dim) * case.q_scaling)
    for q_length, kv_length in zip(case.seq_lens, case.token_nums, strict=True):
        q_seq = q[q_offset : q_offset + q_length].view(q_length, case.num_heads, -1)
        k_seq = k[kv_offset : kv_offset + kv_length].view(kv_length, case.num_heads, -1)
        v_seq = v[kv_offset : kv_offset + kv_length].view(kv_length, case.num_heads, -1)
        scores = (q_seq.double().transpose(0, 1) @ k_seq.double().permute(1, 2, 0)) * scale
        q_positions = torch.arange(q_length, device=q.device) + kv_length - q_length
        k_positions = torch.arange(kv_length, device=q.device)
        scores.masked_fill_(k_positions[None, :] > q_positions[:, None], -torch.inf)
        outputs.append((scores.softmax(-1) @ v_seq.double().transpose(0, 1)).transpose(0, 1))
        q_offset += q_length
        kv_offset += kv_length
    return torch.cat(outputs).flatten(1).to(q.dtype)


@pytest.mark.parametrize(
    "qk_head_dim,v_head_dim",
    [(128, 128), (192, 128), (256, 256)],
    ids=["qk128-v128", "qk192-v128", "qk256-v256"],
)
@pytest.mark.parametrize(
    "kv_dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16-kv", "fp8-kv"]
)
def test_prims_ts_mla_cached_and_partitioned_context_geometry(
    monkeypatch: pytest.MonkeyPatch, qk_head_dim: int, v_head_dim: int, kv_dtype: torch.dtype
) -> None:
    """Run cached and merged expanded MLA through real metadata and a TP-local head geometry."""
    monkeypatch.setenv("TLLM_FMHA_LIBS", "prims_ts")
    case = BackendCase(
        num_heads=6,
        num_kv_heads=6,
        head_dim=qk_head_dim,
        v_head_dim=v_head_dim,
        dtype="bfloat16",
        seq_lens=[17, 9, 5],
        num_cached_tokens=[129, 0, 33],
        num_contexts=3,
        is_mla=True,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=qk_head_dim - 64,
        qk_rope_head_dim=64,
        page_size=32,
        rope={"dim": 64, "theta": 10_000.0, "max_positions": 512, "is_neox": False},
    )
    pos_embd_params = _mla_context_pos_embd_params(case)
    full_case = replace(case, seq_lens=case.token_nums, num_cached_tokens=[0] * case.num_seqs)
    full_inputs = generate_mla_context_inputs(full_case, seed=1729)
    full_q, k, v, _, _ = _prepare_mla_context_inputs(
        full_case, full_inputs, pos_embd_params, fuse_rope=False
    )
    q = torch.cat(
        [
            part[cached:]
            for part, cached in zip(
                full_q.split(case.token_nums), case.num_cached_tokens, strict=True
            )
        ]
    )
    historical_k, historical_v, fresh_k, fresh_v = (
        torch.cat(
            [
                part[:cached] if history else part[cached:]
                for part, cached in zip(
                    tensor.split(case.token_nums), case.num_cached_tokens, strict=True
                )
            ]
        )
        for history, tensor in ((True, k), (True, v), (False, k), (False, v))
    )

    def strided_value(value: torch.Tensor) -> torch.Tensor:
        padding = case.num_heads * case.qk_nope_head_dim
        storage = torch.full(
            (value.shape[0], padding + value.shape[1]),
            32.0,
            dtype=value.dtype,
            device=value.device,
        )
        view = storage[:, padding:]
        view.copy_(value)
        return view

    v, historical_v, fresh_v = map(strided_value, (v, historical_v, fresh_v))
    reference = _ragged_causal_reference(q, k, v, case)
    atol, rtol = _tolerances(case, kv_dtype)
    attn = create_attention(
        "TRTLLM",
        layer_idx=0,
        num_heads=case.num_heads,
        num_kv_heads=case.num_kv_heads,
        head_dim=qk_head_dim,
        q_scaling=case.q_scaling,
        pos_embd_params=pos_embd_params,
        quant_config=_quant_config(kv_dtype),
        is_mla_enable=True,
        q_lora_rank=case.q_lora_rank,
        kv_lora_rank=case.kv_lora_rank,
        qk_nope_head_dim=case.qk_nope_head_dim,
        qk_rope_head_dim=case.qk_rope_head_dim,
        v_head_dim=v_head_dim,
    )
    assert isinstance(attn, TrtllmAttention)
    context_calls = []
    run_context = PrimsTSFmha.run_mla_context

    def record_context(fmha: PrimsTSFmha, params: FmhaParams) -> None:
        assert params.fwd.latent_cache is None
        assert params.input_seq_length == max(case.seq_lens)
        context_calls.append(
            (
                params.query_input.shape[0],
                params.key_input.shape[0],
                params.fwd.softmax_stats_tensor is not None,
            )
        )
        run_context(fmha, params)

    monkeypatch.setattr(PrimsTSFmha, "run_mla_context", record_context)
    manager = _build_mla_kv_cache_manager(case, "TRTLLM", kv_dtype=kv_dtype)
    try:
        request_ids = list(range(case.num_seqs))
        manager.add_dummy_requests(request_ids, case.token_nums)
        manager.get_buffers(0).zero_()
        metadata = TrtllmAttention.Metadata(
            num_contexts=case.num_contexts,
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=case.num_cached_tokens
            ),
            seq_lens=torch.tensor(case.seq_lens, dtype=torch.int32),
            prompt_lens=case.seq_lens,
            max_num_requests=case.num_seqs,
            max_num_tokens=case.max_num_tokens,
            kv_cache_manager=manager,
            request_ids=request_ids,
            kv_layout="HND",
            enable_context_mla_with_cached_kv=True,
            runtime_features=AttentionRuntimeFeatures(cache_reuse=True),
        )
        metadata.prepare()

        def run_partition(
            partition_k: torch.Tensor,
            partition_v: torch.Tensor,
            lengths: list[int],
            mask: PredefinedAttentionMask,
            stats: torch.Tensor | None,
        ) -> torch.Tensor:
            # These are the same runtime length views updated by MLA's chunked-context loop.
            metadata.kv_lens_runtime = torch.tensor(lengths, dtype=torch.int32)
            metadata.kv_lens_cuda_runtime = metadata.kv_lens_runtime.to("cuda")
            metadata.host_total_kv_lens[0] = sum(lengths)
            return attn.forward(
                q,
                partition_k,
                partition_v,
                metadata,
                forward_args=AttentionForwardArgs(
                    attention_input_type=AttentionInputType.context_only,
                    latent_cache=None,
                    attention_mask=mask,
                    softmax_stats_tensor=stats,
                ),
            )

        cached = run_partition(k, v, case.token_nums, PredefinedAttentionMask.CAUSAL, None)
        torch.testing.assert_close(cached, reference, atol=atol, rtol=rtol)
        merged = torch.full_like(cached, torch.nan)
        merged_stats = torch.full((case.nnz_q, case.num_heads, 2), torch.nan, device="cuda")
        temp_stats = torch.empty_like(merged_stats)
        historical = run_partition(
            historical_k,
            historical_v,
            case.num_cached_tokens,
            PredefinedAttentionMask.FULL,
            temp_stats,
        )
        attn.merge_attention_for_mla(
            merged,
            historical,
            merged_stats,
            temp_stats,
            torch.tensor([2, 0, 2], dtype=torch.int64, device="cuda"),
            metadata,
        )
        # The empty historical request must remain untouched until the fresh partition copies it.
        empty_rows = slice(case.seq_lens[0], sum(case.seq_lens[:2]))
        assert torch.isnan(merged[empty_rows]).all()
        assert torch.isnan(merged_stats[empty_rows]).all()
        fresh = run_partition(
            fresh_k, fresh_v, case.seq_lens, PredefinedAttentionMask.CAUSAL, temp_stats
        )
        attn.merge_attention_for_mla(
            merged,
            fresh,
            merged_stats,
            temp_stats,
            torch.tensor([1, 2, 1], dtype=torch.int64, device="cuda"),
            metadata,
        )
        assert torch.isfinite(merged_stats).all()
        torch.testing.assert_close(merged, reference, atol=atol, rtol=rtol)
        torch.testing.assert_close(merged[empty_rows], fresh[empty_rows], atol=0, rtol=0)
        assert context_calls == [
            (case.nnz_q, sum(case.token_nums), False),
            (case.nnz_q, sum(case.num_cached_tokens), True),
            (case.nnz_q, case.nnz_q, True),
        ]
    finally:
        manager.shutdown()
