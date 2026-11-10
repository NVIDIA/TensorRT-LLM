# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU correctness of CuTe DSL MLA decode on FP8 wide-head verify groups.

CuteDslMlaFmha sends an FP8-KV multi-token verify group of 64 < H < 128 heads
without Helix to the CuTe DSL MLA decode kernel on SM100, SM103 and SM107
(``fp8_wide_head_verify_group``). These tests run that shape (H=96, the
attention-DP verify group of a 96-head MLA model) on the decode runner and
compare it with a causal reference.
"""

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

# Largest error allowed, as a share of the token's largest reference value: FP8 Q/K/V, an FP8
# softmax probability in the PV product and a BF16 output (measured: at most ~2% on SM103 and
# SM107). An absolute bound would not scale with the output, which reaches ~15 on the tokens
# that attend to the scaled-up cache rows.
_TOL = 5e-2


@pytest.mark.parametrize("split_kv", [1, 4])
@pytest.mark.parametrize("seq_len_q", [6, 8])
def test_fp8_wide_head_verify_group_matches_causal_reference(seq_len_q: int, split_kv: int) -> None:
    """Each verify token attends to the cached prefix plus the verify tokens up to itself."""
    import cutlass

    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    # A CPU generator gives the same inputs on every GPU; CUDA randn values depend on the device.
    generator = torch.Generator().manual_seed(29)
    device = torch.device("cuda")
    num_heads = 96
    latent_dim, rope_dim, page_size = 512, 64, 64
    # The second request ends inside a page.
    kv_lens = [256, 77]
    batch_size = len(kv_lens)
    blocks_per_sequence = math.ceil(max(kv_lens) / page_size)
    num_pages = batch_size * blocks_per_sequence
    softmax_scale = 1.0 / math.sqrt(latent_dim + rope_dim)

    # Unit-scale inputs give scores of standard deviation ~1, so the attention is far from uniform.
    q_storage = (
        torch.randn(batch_size, seq_len_q, num_heads, latent_dim + rope_dim, generator=generator)
        .to(device)
        .to(torch.float8_e4m3fn)
    )
    cache_fp32 = torch.randn(num_pages, page_size, latent_dim + rope_dim, generator=generator)
    # Scale up the cache rows of verify tokens 1..seq_len_q-1: each is visible to its own token
    # and the later ones only, so a kernel that lets an earlier token see it moves that token's
    # output far beyond the tolerance (checked below). Request b's block i is page
    # b * blocks_per_sequence + i (see page_table).
    for b, kv_len in enumerate(kv_lens):
        for pos in range(kv_len - seq_len_q + 1, kv_len):
            cache_fp32[b * blocks_per_sequence + pos // page_size, pos % page_size] *= 4.0
    cache_storage = cache_fp32.to(device).to(torch.float8_e4m3fn)
    q_latent = q_storage[..., :latent_dim].permute(2, 3, 1, 0)
    q_rope = q_storage[..., latent_dim:].permute(2, 3, 1, 0)
    c_latent = cache_storage[..., :latent_dim].permute(1, 2, 0)
    c_rope = cache_storage[..., latent_dim:].permute(1, 2, 0)
    page_table = (
        torch.arange(num_pages, dtype=torch.int32, device=device)
        .view(batch_size, blocks_per_sequence)
        .transpose(0, 1)
    )
    cache_seqs = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    output_storage = torch.zeros(
        batch_size, seq_len_q, num_heads, latent_dim, device=device, dtype=torch.bfloat16
    )

    runner = CuteDSLNVMlaDecodeBlackwellRunner(
        in_dtype=cutlass.Float8E4M3FN,
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        page_size=page_size,
        max_batch_size=batch_size,
    )
    workspace = torch.empty(
        runner.get_max_padded_workspace_size(
            num_heads, seq_len_q, latent_dim, batch_size, cutlass.Float32
        ),
        device=device,
        dtype=torch.uint8,
    )
    inputs = [
        q_latent,
        q_rope,
        c_latent,
        c_rope,
        page_table,
        cache_seqs,
        output_storage.permute(2, 3, 1, 0),
        workspace,
        None,  # softmax_stats: not a Helix group
        None,  # kv_bounds: the causal bound
        torch.tensor([softmax_scale], dtype=torch.float32, device=device),
        torch.tensor([1.0], dtype=torch.float32, device=device),
    ]
    runner.forward(inputs, tactic=((128, 128), (128, 256), split_kv, False))
    torch.cuda.synchronize()

    for b, kv_len in enumerate(kv_lens):
        page_ids = page_table[:, b].long()
        key_latent = (
            c_latent[:, :, page_ids].permute(2, 0, 1).reshape(-1, latent_dim)[:kv_len].float()
        )
        key_rope = c_rope[:, :, page_ids].permute(2, 0, 1).reshape(-1, rope_dim)[:kv_len].float()
        for t in range(seq_len_q):
            visible = kv_len - seq_len_q + 1 + t
            query_latent = q_latent[:, :, t, b].float()
            query_rope = q_rope[:, :, t, b].float()
            scores = (
                query_latent @ key_latent.transpose(0, 1) + query_rope @ key_rope.transpose(0, 1)
            ) * softmax_scale
            expected = torch.softmax(scores[:, :visible], dim=-1) @ key_latent[:visible]
            atol = _TOL * expected.abs().max().item()
            if t < seq_len_q - 1:
                # The check separates the causal bound from a leak: attending to the later verify
                # tokens as well would move this token's output by many tolerances.
                unmasked = torch.softmax(scores, dim=-1) @ key_latent
                assert (unmasked - expected).abs().max() > 10 * atol
            torch.testing.assert_close(output_storage[b, t].float(), expected, rtol=0, atol=atol)
