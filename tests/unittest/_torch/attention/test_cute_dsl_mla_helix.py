# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU correctness tests for CuTe DSL MLA's Helix output contract."""

import math

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._utils import is_sm_100f

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not IS_CUTLASS_DSL_AVAILABLE or not is_sm_100f(),
    reason="CuTe DSL MLA requires an SM100-family CUDA GPU",
)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_cute_dsl_mla_helix_stats_and_empty_local_kv(input_dtype: torch.dtype) -> None:
    import cutlass

    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner
    from tensorrt_llm._torch.modules.attention import _helix_sanitize_empty_kv

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
        tactic=((128, 128), (128, 256), 4, False),
        softmax_scale=softmax_scale,
        output_scale=1.0,
    )

    # The existing Helix sanitizer makes a rank with no local pages the identity
    # contribution even though the unchanged MLA kernel may leave its output
    # row undefined. The stats wrapper itself emits the canonical empty pair.
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
