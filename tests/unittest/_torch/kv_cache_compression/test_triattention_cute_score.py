# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The SM100 TriAttention CuTe scorer (the only score path) vs PyTorch oracles.

Two layers of coverage over ``_FixedScoreGroup.launch`` -- the exact
production entry point. The kernel-numerics matrix drives a single-layer
group across the supported page geometries (permuted physical pages, ragged
valid lengths, GQA group 4 riding the padded MMA tile) against inline
oracle math. The launch-path matrix drives multi-layer groups across the
named production geometries (Qwen3, GPT-OSS, the originally validated
128-token-page shape) against the shared pure-PyTorch oracle, sweeps
request counts up to the group capacity, and checks the per-request
decode-width metadata the selection reduce kernels consume. The contract
tests pin the loud-failure behavior: unsupported geometry, removed
aggregations, and request counts beyond capacity raise -- there is no
fallback score kernel.
"""

import pytest
import torch
from conftest import encode_block_offsets as _encode_block_offsets
from conftest import torch_tri_score_oracle as _torch_tri_score_oracle

from tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels import (
    _FixedScoreGroup,
)

requires_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="TriAttention score requires SM100",
)


@requires_sm100
@pytest.mark.parametrize(
    "tokens_per_block,page_permutation,valid_lens,num_freqs,num_q_heads",
    [
        # The originally validated geometry: 128-token pages, identity table.
        (128, [0, 1], None, 32, 8),
        # GPT-OSS geometry: 32-token pages; a 64-token compute tile spans two
        # pages, so a shuffled physical-page table catches fragment mix-ups.
        (32, [3, 1, 4, 7, 5, 0, 2, 6], None, 32, 8),
        # Ragged tails land mid-tile: the second page fragment of the last
        # tile is clamped, and scores past the valid length are unspecified.
        (32, [3, 1, 4, 7, 5, 0, 2, 6], [250, 198], 32, 8),
        # Qwen3 geometry: 128-element K rows (64 frequencies) and GQA group
        # 4, which rides the MMA tile N=8 with zeroed padding columns.
        (32, [3, 1, 4, 7, 5, 0, 2, 6], None, 64, 4),
        (32, [3, 1, 4, 7, 5, 0, 2, 6], [250, 198], 64, 4),
    ],
)
def test_cute_score_matches_torch_mean_oracle(
    tokens_per_block: int,
    page_permutation: list,
    valid_lens: "list | None",
    num_freqs: int,
    num_q_heads: int,
) -> None:
    pytest.importorskip("cutlass")

    torch.manual_seed(20260720)
    device = torch.device("cuda")
    seq_len = 256
    num_pages = seq_len // tokens_per_block
    assert sorted(page_permutation) == list(range(num_pages))
    pool = (
        0.125 * torch.randn(num_pages, 2, 1, tokens_per_block, 2 * num_freqs, device=device)
    ).to(torch.bfloat16)
    q_real = 0.125 * torch.randn(1, num_q_heads, num_freqs, device=device)
    q_imag = 0.125 * torch.randn_like(q_real)
    mlr_coef = 0.125 * torch.randn_like(q_real)
    freq_scale_sq = torch.linspace(0.5, 1.5, num_freqs, device=device)
    omega = torch.linspace(0.01, 0.03, num_freqs, device=device)
    offsets = torch.tensor([1.0, 2.0, 4.0], device=device)
    round_starts = torch.tensor([float(seq_len), float(seq_len + 1)], device=device)
    phase = (round_starts[:, None, None] + offsets[None, :, None]) * omega[None, None]
    mean_cos = torch.cos(phase).mean(dim=1).contiguous()
    mean_sin = torch.sin(phase).mean(dim=1).contiguous()

    # Native block-offset staging layout ([pool_slot, request, K/V plane,
    # block] int32): K-plane entries encode physical_page * kv_factor with
    # kv_factor == 2. Both requests read the same (permuted) page sequence.
    k_plane = [2 * page for page in page_permutation]
    v_plane = [2 * page + 1 for page in page_permutation]
    block_offsets = torch.tensor(
        [[[k_plane, v_plane], [k_plane, v_plane]]], dtype=torch.int32, device=device
    )
    group = _FixedScoreGroup(
        [pool],
        [0],
        2,
        2,
        seq_len,
        num_q_heads,
        block_offsets,
        [0],
        q_real,
        q_imag,
        mlr_coef,
        freq_scale_sq,
        omega,
        offsets,
        output_width=seq_len,
    )
    keys = (
        torch.cat([pool[page, 0, 0] for page in page_permutation], dim=0)
        .reshape(seq_len, 2 * num_freqs)
        .float()
    )
    k_real = keys[:, :num_freqs]
    k_imag = keys[:, num_freqs:]
    magnitude = torch.sqrt(k_real.square() + k_imag.square())
    if valid_lens is None:
        valid_lens = [seq_len, seq_len]
    valid_seq_lens = torch.tensor(valid_lens, dtype=torch.int32, device=device)
    valid_widths = torch.tensor(valid_lens, dtype=torch.int32, device=device)
    token_starts_device = torch.zeros(2, dtype=torch.int32, device=device)
    for request_count in (1, 2):
        group.output.fill_(float("nan"))
        actual = group.launch(
            request_count,
            valid_seq_lens,
            valid_widths,
            token_starts_device,
            mean_cos,
            mean_sin,
        )
        assert actual.shape == (request_count, 1, num_q_heads, seq_len)
        for request in range(request_count):
            rotated_real = freq_scale_sq * (k_real * mean_cos[request] + k_imag * mean_sin[request])
            rotated_imag = freq_scale_sq * (k_imag * mean_cos[request] - k_real * mean_sin[request])
            expected = (
                q_real[0, :, None] * rotated_real[None]
                + q_imag[0, :, None] * rotated_imag[None]
                + mlr_coef[0, :, None] * freq_scale_sq[None, None] * magnitude[None]
            ).sum(dim=-1)
            valid = valid_lens[request]
            torch.testing.assert_close(
                actual[request, 0, :, :valid],
                expected[:, :valid],
                rtol=5.0e-3,
                atol=5.0e-3,
            )

    torch.cuda.synchronize()
    # The CuTe runner is the only score path; prove setup actually built it.
    assert group._cute_score_runner is not None


def _build_case(
    *,
    max_requests: int,
    num_layers: int,
    page_count: int,
    tokens_per_block: int,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    prompt_len: int,
    seed: int,
    offsets: tuple = (1.0, 2.0, 4.0),
):
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(seed)
    num_freqs = head_dim // 2
    # The 0.125 scaling keeps the BF16 key/coefficient products small so the
    # kernel-vs-oracle tolerance can stay tight across the frequency sum.
    pools = [
        (
            0.125
            * torch.randn(
                max_requests * page_count,
                2,
                num_kv_heads,
                tokens_per_block,
                head_dim,
                device=device,
            )
        ).to(torch.bfloat16)
        for _ in range(num_layers)
    ]
    page_ids = torch.randperm(max_requests * page_count).view(max_requests, page_count).to(device)
    q_real = 0.125 * torch.randn(num_layers, num_q_heads, num_freqs, device=device)
    q_imag = 0.125 * torch.randn(num_layers, num_q_heads, num_freqs, device=device)
    mlr_coef = 0.125 * torch.randn(num_layers, num_q_heads, num_freqs, device=device)
    freq_scale_sq = torch.rand(num_freqs, device=device) + 0.5
    omega = torch.rand(num_freqs, device=device) * 0.05
    offsets_t = torch.tensor(offsets, dtype=torch.float32, device=device)
    capacity = page_count * tokens_per_block
    group = _FixedScoreGroup(
        pools,
        list(range(num_layers)),
        max_requests,
        page_count,
        capacity,
        num_q_heads,
        _encode_block_offsets(page_ids),
        [0] * num_layers,
        q_real,
        q_imag,
        mlr_coef,
        freq_scale_sq,
        omega,
        offsets_t,
        output_width=capacity - prompt_len,
    )
    round_starts = (torch.arange(max_requests, dtype=torch.int32, device=device) + 9).contiguous()
    token_starts = torch.full((max_requests,), prompt_len, dtype=torch.int32, device=device)
    # Ragged valid lengths whose tails land mid-page and mid-compute-tile.
    seq_lens = [capacity - ((request * 3) % 5) for request in range(max_requests)]
    valid_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    phase = (round_starts.float()[:, None, None] + offsets_t[None, :, None]) * omega[None, None, :]
    mean_cos = torch.cos(phase).mean(dim=1).contiguous()
    mean_sin = torch.sin(phase).mean(dim=1).contiguous()
    # Everything the PyTorch oracle needs to rebuild the reference leg
    # independently (it recomputes its own mean phases from these).
    oracle_inputs = dict(
        page_ids=page_ids,
        q_real=q_real,
        q_imag=q_imag,
        mlr_coef=mlr_coef,
        freq_scale_sq=freq_scale_sq,
        omega=omega,
        offsets=offsets_t,
    )
    return (
        group,
        token_starts,
        valid_seq_lens,
        seq_lens,
        mean_cos,
        mean_sin,
        oracle_inputs,
    )


def _geometry(max_requests, num_layers, page_count, tokens_per_block, head_dim, num_q, num_kv):
    return dict(
        max_requests=max_requests,
        num_layers=num_layers,
        page_count=page_count,
        tokens_per_block=tokens_per_block,
        head_dim=head_dim,
        num_q_heads=num_q,
        num_kv_heads=num_kv,
    )


# One entry per supported production geometry: the Qwen3 shape (64
# frequencies, GQA group 4 riding the padded MMA tile), the GPT-OSS shape
# (32 frequencies, group 8, 32-token pages spanning two page fragments per
# compute tile), and the originally validated 128-token-page shape.
_CASES = [
    pytest.param(_geometry(4, 2, 4, 32, 128, 8, 2), id="qwen3_f64_group4_tpb32"),
    pytest.param(_geometry(2, 3, 4, 32, 64, 8, 1), id="gptoss_f32_group8_tpb32"),
    pytest.param(_geometry(2, 2, 2, 128, 64, 8, 1), id="original_f32_group8_tpb128"),
]


@requires_sm100
@pytest.mark.parametrize("case", _CASES)
def test_cute_kernel_matches_torch_oracle(case):
    pytest.importorskip("cutlass")
    case = dict(case)  # parametrize reuses the dict across reruns
    prompt_len = 5
    max_requests = case["max_requests"]
    num_layers = case["num_layers"]
    (
        group,
        token_starts,
        valid_seq_lens,
        seq_lens,
        mean_cos,
        mean_sin,
        oracle_inputs,
    ) = _build_case(prompt_len=prompt_len, seed=20260719, **case)
    device = group.output.device

    oracle = _torch_tri_score_oracle(
        group._cute_layer_pools,
        oracle_inputs["page_ids"],
        seq_lens,
        [int(start) for start in range(9, 9 + max_requests)],
        oracle_inputs["q_real"],
        oracle_inputs["q_imag"],
        oracle_inputs["mlr_coef"],
        oracle_inputs["freq_scale_sq"],
        oracle_inputs["omega"],
        oracle_inputs["offsets"],
        list(range(num_layers)),
    )

    # The fused runner dispatches every request count up to the group
    # capacity; cover one, an intermediate count, and the capacity.
    for request_count in dict.fromkeys((1, max_requests - 1, max_requests)):
        group.output.fill_(float("nan"))
        valid_widths = torch.full((max_requests,), -1, dtype=torch.int32, device=device)
        scores = group.launch(
            request_count,
            valid_seq_lens,
            valid_widths,
            token_starts,
            mean_cos,
            mean_sin,
        )
        assert scores.shape == (
            request_count,
            num_layers,
            case["num_q_heads"],
            group.output_width,
        )
        # The launch owns the per-request decode widths the selection
        # reduce kernels consume.
        assert valid_widths[:request_count].tolist() == [
            seq_lens[request] - prompt_len for request in range(request_count)
        ]
        for request in range(request_count):
            width = seq_lens[request] - prompt_len
            for layer in range(num_layers):
                torch.testing.assert_close(
                    scores[request, layer, :, :width],
                    oracle[request * num_layers + layer][:, prompt_len : prompt_len + width],
                    rtol=5e-3,
                    atol=5e-3,
                )


def _tiny_unsupported_group(dtype: torch.dtype):
    """A geometry far outside the CuTe contract (constructor accepts it)."""
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(20260722)
    num_layers, max_requests, page_count, tokens_per_block, head_dim = 2, 2, 2, 4, 8
    num_freqs = head_dim // 2
    pools = [
        torch.randn(max_requests * page_count, 2, 1, tokens_per_block, head_dim, device=device).to(
            dtype
        )
        for _ in range(num_layers)
    ]
    page_ids = (
        torch.arange(max_requests * page_count, device=device)
        .view(max_requests, page_count)
        .contiguous()
    )
    capacity = page_count * tokens_per_block
    group = _FixedScoreGroup(
        pools,
        list(range(num_layers)),
        max_requests,
        page_count,
        capacity,
        2,
        _encode_block_offsets(page_ids),
        [0] * num_layers,
        torch.randn(num_layers, 2, num_freqs, device=device),
        torch.randn(num_layers, 2, num_freqs, device=device),
        torch.randn(num_layers, 2, num_freqs, device=device),
        torch.rand(num_freqs, device=device) + 0.5,
        torch.rand(num_freqs, device=device) * 0.05,
        torch.tensor([1.0, 2.0], dtype=torch.float32, device=device),
        output_width=capacity - 1,
    )
    device_args = dict(dtype=torch.int32, device=device)
    return group, (
        torch.full((max_requests,), capacity, **device_args),
        torch.empty(max_requests, **device_args),
        torch.ones(max_requests, **device_args),
        torch.zeros(max_requests, num_freqs, dtype=torch.float32, device=device),
        torch.zeros(max_requests, num_freqs, dtype=torch.float32, device=device),
    )


# The loud-failure contract, one representative per guard family. All three
# guards fire before any kernel work, in this order: removed aggregation,
# request count beyond the group capacity (previously exercised on a
# production-shaped SM100 group; the check is layered before compilation, so
# the tiny group covers the same code path), unsupported geometry at setup.
@pytest.mark.parametrize(
    "dtype,request_count,launch_kwargs,match",
    [
        pytest.param(
            torch.float32, 1, {}, "TriAttention score requires SM100", id="unsupported_geometry"
        ),
        pytest.param(
            torch.bfloat16, 1, {"aggregation": "max"}, "max aggregation", id="max_aggregation"
        ),
        pytest.param(torch.bfloat16, 3, {}, "exceeds fixed score capacity", id="beyond_capacity"),
    ],
)
def test_score_launch_contract_raises(dtype, request_count, launch_kwargs, match):
    group, launch_args = _tiny_unsupported_group(dtype)
    with pytest.raises(ValueError, match=match):
        group.launch(request_count, *launch_args, **launch_kwargs)
