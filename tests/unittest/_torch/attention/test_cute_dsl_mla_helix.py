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
    or get_sm_version() not in (100, 103),
    reason="CuTe DSL MLA decode runs on SM100/SM103 only (see CuteDslMlaFmha)",
)


def _scale_inputs(input_dtype: torch.dtype, softmax_scale: float, device) -> list:
    """FP8 runners read both scales from device tensors at inputs[10:12]."""
    if input_dtype != torch.float8_e4m3fn:
        return []
    return [
        None,  # kv_bounds
        torch.tensor([softmax_scale], dtype=torch.float32, device=device),
        torch.tensor([1.0], dtype=torch.float32, device=device),
    ]


def _scale_kwargs(input_dtype: torch.dtype, softmax_scale: float) -> dict:
    """FP16/BF16 runners keep their scales as host scalars."""
    if input_dtype == torch.float8_e4m3fn:
        return {}
    return {"softmax_scale": softmax_scale, "output_scale": 1.0}


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
    inputs = [
        q_latent,
        q_rope,
        c_latent,
        c_rope,
        page_table,
        cache_seqs,
        output,
        workspace,
        softmax_stats,
    ]
    runner.forward(
        inputs + _scale_inputs(input_dtype, softmax_scale, device),
        tactic=((128, 128), (128, 256), split_kv, False),
        **_scale_kwargs(input_dtype, softmax_scale),
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


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("num_heads", [96, 12])
def test_cute_dsl_mla_helix_per_token_bounds(input_dtype: torch.dtype, num_heads: int) -> None:
    """Verify-group semantics: kv_bounds replaces the causal bound per token.

    num_heads=96 is the helix production shape (fold ratio 1); num_heads=12
    folds seq_len_q into the M tile (fold ratio 4), exercising the
    folded-grid token indexing in both the masking and the reduction
    stats gating.
    """
    import cutlass

    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    torch.manual_seed(23)
    device = torch.device("cuda")
    batch_size, seq_len_q = 2, 4
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
    cache_seqs = torch.tensor([kv_len, kv_len], dtype=torch.int32, device=device)
    # Per-token rank-local bounds: sequence 0 permutes non-causal values in
    # [K - S_q, K] (the table must win over the causal formula); sequence 1's
    # leading token owns no local KV (stats must gate it out per token).
    kv_bounds = torch.tensor(
        [253, 252, 255, 254, 0, 256, 253, 256], dtype=torch.int32, device=device
    )
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
    inputs = [
        q_latent,
        q_rope,
        c_latent,
        c_rope,
        page_table,
        cache_seqs,
        output,
        workspace,
        softmax_stats,
        kv_bounds,
    ]
    scale_kwargs = {
        "softmax_scale": softmax_scale,
        "output_scale": 1.0,
    }
    if input_dtype == torch.float8_e4m3fn:
        inputs.extend(
            [
                torch.tensor([softmax_scale], device=device),
                torch.tensor([1.0], device=device),
            ]
        )
        scale_kwargs = {}
    runner.forward(
        inputs,
        tactic=((128, 128), (128, 256), 4, False),
        **scale_kwargs,
    )

    output_atol = 5e-2 if input_dtype == torch.float8_e4m3fn else 3e-2
    for b in range(batch_size):
        page_ids = page_table[:, b].long()
        key_latent = c_latent[:, :, page_ids].permute(2, 0, 1).reshape(kv_len, latent_dim).float()
        key_rope = c_rope[:, :, page_ids].permute(2, 0, 1).reshape(kv_len, rope_dim).float()
        for t in range(seq_len_q):
            bound = int(kv_bounds[b * seq_len_q + t])
            stats_row = softmax_stats[b * seq_len_q + t]
            if bound == 0:
                # The reduction gates zero-KV tokens out of the CP merge with
                # the canonical empty pair; the output row is unspecified.
                assert torch.isneginf(stats_row[:, 0]).all()
                torch.testing.assert_close(stats_row[:, 1], torch.zeros_like(stats_row[:, 1]))
                continue
            query_latent = q_latent[:, :, t, b].float()
            query_rope = q_rope[:, :, t, b].float()
            scores = (
                query_latent @ key_latent[:bound].transpose(0, 1)
                + query_rope @ key_rope[:bound].transpose(0, 1)
            ) * softmax_scale
            expected_output = torch.softmax(scores, dim=-1) @ key_latent[:bound]
            torch.testing.assert_close(
                output_storage[b, t].float(),
                expected_output,
                rtol=output_atol,
                atol=output_atol,
            )
            actual_log_partition = stats_row[:, 0] + stats_row[:, 1].log()
            expected_log_partition = torch.logsumexp(scores, dim=-1)
            torch.testing.assert_close(
                actual_log_partition, expected_log_partition, rtol=1e-4, atol=1e-4
            )


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_cute_dsl_mla_helix_bounds_tile_boundary(input_dtype: torch.dtype) -> None:
    """Make the masked-span widening load-bearing: with S_q=2 and K=129
    (two 128-wide QK tiles), the causal span masks only the last tile while
    the minimum per-token bound (K - S_q = 127) lands in the first, so a
    port that drops the widening returns unmasked probability mass."""
    import cutlass

    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    torch.manual_seed(29)
    device = torch.device("cuda")
    batch_size, seq_len_q, num_heads = 1, 2, 96
    latent_dim, rope_dim, page_size, kv_len = 512, 64, 64, 129
    blocks_per_sequence = (kv_len + page_size - 1) // page_size
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
    cache_seqs = torch.tensor([kv_len], dtype=torch.int32, device=device)
    kv_bounds = torch.tensor([127, 129], dtype=torch.int32, device=device)
    output_storage = torch.empty(
        batch_size, seq_len_q, num_heads, latent_dim, device=device, dtype=torch.bfloat16
    )
    output = output_storage.permute(2, 3, 1, 0)
    softmax_stats = torch.empty(
        batch_size * seq_len_q, num_heads, 2, device=device, dtype=torch.float32
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
    inputs = [
        q_latent,
        q_rope,
        c_latent,
        c_rope,
        page_table,
        cache_seqs,
        output,
        workspace,
        softmax_stats,
        kv_bounds,
    ]
    scale_kwargs = {"softmax_scale": softmax_scale, "output_scale": 1.0}
    if input_dtype == torch.float8_e4m3fn:
        inputs.extend(
            [
                torch.tensor([softmax_scale], device=device),
                torch.tensor([1.0], device=device),
            ]
        )
        scale_kwargs = {}
    runner.forward(
        inputs,
        tactic=((128, 128), (128, 256), 1, False),
        **scale_kwargs,
    )

    page_ids = page_table[:, 0].long()
    key_latent = (
        c_latent[:, :, page_ids].permute(2, 0, 1).reshape(num_pages * page_size, latent_dim).float()
    )
    key_rope = (
        c_rope[:, :, page_ids].permute(2, 0, 1).reshape(num_pages * page_size, rope_dim).float()
    )
    for t in range(seq_len_q):
        bound = int(kv_bounds[t])
        query_latent = q_latent[:, :, t, 0].float()
        query_rope = q_rope[:, :, t, 0].float()
        scores = (
            query_latent @ key_latent[:bound].transpose(0, 1)
            + query_rope @ key_rope[:bound].transpose(0, 1)
        ) * softmax_scale
        actual_log_partition = softmax_stats[t, :, 0] + softmax_stats[t, :, 1].log()
        expected_log_partition = torch.logsumexp(scores, dim=-1)
        torch.testing.assert_close(
            actual_log_partition, expected_log_partition, rtol=1e-4, atol=1e-4
        )
