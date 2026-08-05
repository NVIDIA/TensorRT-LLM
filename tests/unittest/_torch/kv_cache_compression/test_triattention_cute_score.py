# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The SM100 TriAttention CuTe scorer (the only score path) vs oracles.

The launch matrix drives the named production geometries against the
pure-PyTorch oracle; the contract test pins the no-fallback loud raise.
"""

import pytest
import torch
from conftest import encode_block_offsets as _encode_block_offsets
from conftest import launch_split_scores as _launch_split_scores
from conftest import make_cute_buffers as _make_cute_buffers
from conftest import torch_tri_score_oracle as _torch_tri_score_oracle
from conftest import write_block_offsets as _write_block_offsets

requires_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="TriAttention score requires SM100",
)


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
    # 0.125 scaling keeps the kernel-vs-oracle tolerance tight.
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
    tri = _make_cute_buffers(
        eviction_mode="per_head",
        layer_pools=pools,
        max_requests=max_requests,
        seq_len=capacity,
        num_q_heads=num_q_heads,
        q_real=q_real,
        q_imag=q_imag,
        mlr_coef=mlr_coef,
        freq_scale_sq=freq_scale_sq,
        omega=omega,
        offsets=offsets_t,
        decode_width=capacity - prompt_len,
    )
    _write_block_offsets(tri, _encode_block_offsets(page_ids))
    logical_source_lengths = (
        torch.arange(max_requests, dtype=torch.int32, device=device) + 9
    ).contiguous()
    prompt_lengths = torch.full((max_requests,), prompt_len, dtype=torch.int32, device=device)
    # Mid-page/mid-tile tails; 58 leaves a fully-invalid trailing fragment.
    tail_cuts = (0, 58, 3, 33)
    seq_lens = [capacity - tail_cuts[request % len(tail_cuts)] for request in range(max_requests)]
    source_lengths = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    phase = (logical_source_lengths.float()[:, None, None] + offsets_t[None, :, None]) * omega[
        None, None, :
    ]
    mean_cos = torch.cos(phase).mean(dim=1).contiguous()
    mean_sin = torch.sin(phase).mean(dim=1).contiguous()
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
        tri,
        pools,
        prompt_lengths,
        source_lengths,
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


# One entry per supported production geometry.
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
        tri,
        pools,
        prompt_lengths,
        source_lengths,
        seq_lens,
        mean_cos,
        mean_sin,
        oracle_inputs,
    ) = _build_case(prompt_len=prompt_len, seed=20260719, **case)
    device = tri._score_scratch.device

    oracle = _torch_tri_score_oracle(
        pools,
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

    # Every request count used by the runtime dispatches through the launcher.
    for request_count in dict.fromkeys((1, max_requests - 1, max_requests)):
        decode_lengths = torch.full((max_requests,), -1, dtype=torch.int32, device=device)
        scores = _launch_split_scores(
            tri,
            request_count,
            source_lengths,
            decode_lengths,
            prompt_lengths,
            mean_cos,
            mean_sin,
        )
        assert scores.shape == (
            request_count,
            num_layers,
            case["num_q_heads"],
            tri._selection_width_capacity,
        )
        # The score leg owns the per-request decode widths the selection
        # reduce kernels consume.
        assert decode_lengths[:request_count].tolist() == [
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


def test_unsupported_geometry_raises_at_buffer_construction():
    pytest.importorskip("cutlass")
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(20260722)
    num_layers, max_requests, page_count, tokens_per_block, head_dim = 2, 2, 2, 4, 8
    num_freqs = head_dim // 2
    # fp32, 4-token pages, 4 freqs: outside the contract on every device.
    pools = [
        torch.randn(max_requests * page_count, 2, 1, tokens_per_block, head_dim, device=device)
        for _ in range(num_layers)
    ]
    calib = torch.randn(num_layers, 2, num_freqs, device=device)
    # No rewrap: the buffer build's own contract error surfaces directly
    # (the fp32 pools trip the BF16 gate first, at TMA descriptor encoding).
    with pytest.raises(TypeError, match="BF16"):
        _make_cute_buffers(
            eviction_mode="per_head",
            layer_pools=pools,
            max_requests=max_requests,
            seq_len=page_count * tokens_per_block,
            num_q_heads=2,
            q_real=calib,
            q_imag=calib.clone(),
            mlr_coef=calib.clone(),
            freq_scale_sq=torch.rand(num_freqs, device=device) + 0.5,
            omega=torch.rand(num_freqs, device=device) * 0.05,
            offsets=torch.tensor([1.0, 2.0], dtype=torch.float32, device=device),
            decode_width=page_count * tokens_per_block - 1,
        )
