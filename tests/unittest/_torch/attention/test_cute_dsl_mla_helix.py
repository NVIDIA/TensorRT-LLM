# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU correctness tests for CuTe DSL MLA's Helix output contract."""

import math

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not IS_CUTLASS_DSL_AVAILABLE
    or get_sm_version() not in (100, 103, 107),
    reason="CuTe DSL MLA decode runs on SM100/SM103/SM107 only (see CuteDslMlaFmha)",
)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("split_kv", [1, 4])
def test_cute_dsl_mla_helix_stats_and_empty_local_kv(
    input_dtype: torch.dtype, split_kv: int
) -> None:
    import cutlass

    from tensorrt_llm._torch.attention.attention import _helix_sanitize_empty_kv
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    torch.manual_seed(17)
    device = torch.device("cuda")
    batch_size, seq_len_q, num_heads = 2, 1, 96
    latent_dim, rope_dim, page_size, kv_len = 512, 64, 64, 256
    blocks_per_sequence = kv_len // page_size
    num_pages = batch_size * blocks_per_sequence
    softmax_scale = 1.0 / math.sqrt(latent_dim + rope_dim)

    q_storage = (
        torch.randn(
            batch_size,
            seq_len_q,
            num_heads,
            latent_dim + rope_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.1
    ).to(input_dtype)
    cache_storage = (
        torch.randn(
            num_pages,
            page_size,
            latent_dim + rope_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.1
    ).to(input_dtype)
    q_latent = q_storage[..., :latent_dim].permute(2, 3, 1, 0)
    q_rope = q_storage[..., latent_dim:].permute(2, 3, 1, 0)
    c_latent = cache_storage[..., :latent_dim].permute(1, 2, 0)
    c_rope = cache_storage[..., latent_dim:].permute(1, 2, 0)
    page_table = (
        torch.arange(num_pages, dtype=torch.int32, device=device)
        .view(batch_size, blocks_per_sequence)
        .transpose(0, 1)
    )
    cache_seqs = torch.tensor([0, kv_len], dtype=torch.int32, device=device)
    output_storage = torch.empty(
        batch_size,
        seq_len_q,
        num_heads,
        latent_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    output = output_storage.permute(2, 3, 1, 0)
    softmax_stats = torch.empty(
        batch_size * seq_len_q,
        num_heads,
        2,
        device=device,
        dtype=torch.float32,
    )

    cutlass_dtype = cutlass.Float8E4M3FN if input_dtype == torch.float8_e4m3fn else cutlass.BFloat16
    runner = CuteDSLNVMlaDecodeBlackwellRunner(
        in_dtype=cutlass_dtype,
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        page_size=page_size,
        max_batch_size=batch_size,
        emit_softmax_stats=True,
    )
    workspace_size = runner.get_max_padded_workspace_size(
        num_heads, seq_len_q, latent_dim, batch_size, cutlass.Float32
    )
    workspace = torch.empty(workspace_size, device=device, dtype=torch.uint8)
    runner.forward(
        [
            q_latent,
            q_rope,
            c_latent,
            c_rope,
            page_table,
            cache_seqs,
            output,
            workspace,
            softmax_stats,
        ],
        tactic=((128, 128), (128, 256), split_kv, False),
        softmax_scale=softmax_scale,
        output_scale=1.0,
    )

    # The existing Helix sanitizer makes a rank with no local pages the identity
    # contribution even though the MLA kernel may leave an empty row undefined.
    sanitized_output, sanitized_stats = _helix_sanitize_empty_kv(
        output_storage.view(batch_size, -1),
        softmax_stats,
        torch.tensor([True, False], device=device),
    )
    torch.testing.assert_close(sanitized_output[0], torch.zeros_like(sanitized_output[0]))
    assert torch.isneginf(sanitized_stats[0, :, 0]).all()
    torch.testing.assert_close(sanitized_stats[0, :, 1], torch.zeros_like(sanitized_stats[0, :, 1]))

    page_ids = page_table[:, 1].long()
    key_latent = c_latent[:, :, page_ids].permute(2, 0, 1).reshape(kv_len, latent_dim)
    key_rope = c_rope[:, :, page_ids].permute(2, 0, 1).reshape(kv_len, rope_dim)
    query_latent = q_latent[:, :, 0, 1].float()
    query_rope = q_rope[:, :, 0, 1].float()
    scores = (
        query_latent @ key_latent.float().transpose(0, 1)
        + query_rope @ key_rope.float().transpose(0, 1)
    ) * softmax_scale
    expected_output = torch.softmax(scores, dim=-1) @ key_latent.float()
    output_atol = 5e-2 if input_dtype == torch.float8_e4m3fn else 3e-2
    torch.testing.assert_close(
        output_storage[1, 0].float(), expected_output, rtol=output_atol, atol=output_atol
    )

    # CuTe stores an equivalent (max, sum) pair: max + log(sum) is the
    # natural-log partition value consumed by the existing Helix reduction.
    actual_log_partition = softmax_stats[1, :, 0] + softmax_stats[1, :, 1].log()
    expected_log_partition = torch.logsumexp(scores, dim=-1)
    torch.testing.assert_close(actual_log_partition, expected_log_partition, rtol=1e-5, atol=1e-5)


class _FakeKvCacheManager:
    def __init__(self, kv_pool: torch.Tensor):
        self._kv_pool = kv_pool

    def get_buffers(self, layer_idx: int) -> torch.Tensor:
        return self._kv_pool


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_cute_dsl_mla_decode_batch_below_max_num_sequences(
    input_dtype: torch.dtype, batch_size: int
) -> None:
    """A decode batch smaller than the metadata's sequence capacity.

    kv_cache_block_offsets holds one row per sequence slot (max_num_sequences),
    while the kernel derives its batch size from the page table it receives.
    The FMHA wrapper must hand it only this batch's generation rows; any extra
    column launches CTAs for requests past the end of q, the output and the
    sequence lengths. Runs the K3 helix verify-group shape (H=96, 1 + 6 draft
    tokens, per-token kv_bounds) through CuteDslMlaFmha._run_mla_decode.
    """
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.fmha.cute_dsl_mla import CuteDslMlaFmha

    torch.manual_seed(23)
    device = torch.device("cuda")
    max_num_sequences, seq_len_q, num_heads = 32, 7, 96
    latent_dim, rope_dim, qk_nope_head_dim, page_size = 512, 64, 128, 64
    max_blocks = 8
    num_pages = max_num_sequences * max_blocks
    head_dim = latent_dim + rope_dim
    kv_lens = [200, 70, 135][:batch_size]
    softmax_scale = 1.0 / math.sqrt(qk_nope_head_dim + rope_dim)

    kv_pool = (
        torch.randn(num_pages, 1, page_size, 1, head_dim, device=device, dtype=torch.bfloat16) * 0.1
    ).to(input_dtype)
    # One disjoint page range per sequence slot. Slots past the batch are
    # allocated but must never reach the kernel.
    block_offsets = (
        torch.arange(num_pages, dtype=torch.int32, device=device)
        .view(1, max_num_sequences, 1, max_blocks)
        .expand(1, max_num_sequences, 2, max_blocks)
        .contiguous()
    )
    kv_bounds = torch.cat(
        [
            torch.arange(kv_len - seq_len_q + 1, kv_len + 1, dtype=torch.int32, device=device)
            for kv_len in kv_lens
        ]
    )
    meta = SimpleNamespace(
        kv_cache_manager=_FakeKvCacheManager(kv_pool),
        kv_cache_block_offsets=block_offsets,
        host_kv_cache_pool_mapping=torch.zeros(1, 2, dtype=torch.int32),
        num_contexts=0,
        max_num_requests=max_num_sequences,
        helix_position_offsets=torch.zeros(
            batch_size * seq_len_q, dtype=torch.int32, device=device
        ),
        _helix_spec_tokens_valid=True,
        helix_kv_bounds=kv_bounds,
    )
    attn = SimpleNamespace(
        kv_lora_rank=latent_dim,
        qk_rope_head_dim=rope_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        num_heads=num_heads,
        q_scaling=1.0,
        layer_idx=0,
        get_local_layer_idx=lambda _meta: 0,
    )
    num_tokens = batch_size * seq_len_q
    q = torch.randn(num_tokens, num_heads * head_dim, device=device, dtype=torch.bfloat16) * 0.1
    output = torch.empty(num_tokens, num_heads * latent_dim, device=device, dtype=torch.bfloat16)
    softmax_stats = torch.empty(num_tokens, num_heads, 2, device=device, dtype=torch.float32)
    # As in production: one sequence length per generation request.
    sequence_lengths = torch.tensor(kv_lens, dtype=torch.int32, device=device)

    fmha = object.__new__(CuteDslMlaFmha)
    if input_dtype == torch.float8_e4m3fn:
        fmha._cute_dsl_fp8_scale = (softmax_scale, 1.0)
    workspace = torch.empty(
        fmha._required_workspace_size(num_heads, seq_len_q, latent_dim, max_num_sequences),
        device=device,
        dtype=torch.uint8,
    )
    params = SimpleNamespace(
        attn=attn,
        meta=meta,
        num_requests=batch_size,
        tokens_per_block=page_size,
        sequence_lengths=sequence_lengths,
        workspace=workspace,
        fwd=SimpleNamespace(quant_q_buffer=None, softmax_stats_tensor=softmax_stats),
    )

    fmha._run_mla_decode(q, output, params, input_dtype)
    torch.cuda.synchronize()

    q_rows = q.to(input_dtype).float().view(batch_size, seq_len_q, num_heads, head_dim)
    out_rows = output.float().view(batch_size, seq_len_q, num_heads, latent_dim)
    tol = 5e-2 if input_dtype == torch.float8_e4m3fn else 3e-2
    for b, kv_len in enumerate(kv_lens):
        pages = block_offsets[0, b, 0, : -(-kv_len // page_size)].long()
        keys = kv_pool[pages, 0, :, 0].float().reshape(-1, head_dim)[:kv_len]
        for t in range(seq_len_q):
            bound = int(kv_bounds[b * seq_len_q + t])
            scores = q_rows[b, t] @ keys[:bound].transpose(0, 1) * softmax_scale
            expected = torch.softmax(scores, dim=-1) @ keys[:bound, :latent_dim]
            torch.testing.assert_close(out_rows[b, t], expected, rtol=tol, atol=tol)
