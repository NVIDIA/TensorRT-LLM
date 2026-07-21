# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The CuTe score kernel vs an independent PyTorch oracle, through group.launch.

The SM100 CuTe-DSL kernel (``triattention_cute_score.py``) is the ONLY score
implementation. These tests drive it through ``_FixedScoreGroup.launch`` --
the exact production entry point -- across the supported production
geometries, with multi-layer segments, permuted page tables, ragged valid
lengths, and per-request prompt windows, and compare against a pure-PyTorch
oracle that recomputes everything independently. They also pin the
loud-failure contract: unsupported geometry, removed aggregations, and
uncompiled request counts raise instead of routing to another kernel.
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


# One entry per supported production geometry: the Qwen3 shape (64
# frequencies, GQA group 4 riding the padded MMA tile), the GPT-OSS shape
# (32 frequencies, group 8, 32-token pages spanning two page fragments per
# compute tile), and the originally validated 128-token-page shape.
_CASES = [
    pytest.param(
        dict(
            max_requests=4,
            num_layers=2,
            page_count=4,
            tokens_per_block=32,
            head_dim=128,
            num_q_heads=8,
            num_kv_heads=2,
        ),
        id="qwen3_f64_group4_tpb32",
    ),
    pytest.param(
        dict(
            max_requests=2,
            num_layers=3,
            page_count=4,
            tokens_per_block=32,
            head_dim=64,
            num_q_heads=8,
            num_kv_heads=1,
        ),
        id="gptoss_f32_group8_tpb32",
    ),
    pytest.param(
        dict(
            max_requests=2,
            num_layers=2,
            page_count=2,
            tokens_per_block=128,
            head_dim=64,
            num_q_heads=8,
            num_kv_heads=1,
        ),
        id="original_f32_group8_tpb128",
    ),
]

_QWEN3_CASE = dict(_CASES[0].values[0])


class TestTriAttentionScoreLaunch:
    @requires_sm100
    @pytest.mark.parametrize("case", _CASES)
    def test_cute_kernel_matches_torch_oracle(self, case):
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

        # The runner precompiles exactly the request counts production
        # launches: one and the full group capacity.
        for request_count in dict.fromkeys((1, max_requests)):
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
            # reduce kernels consume (the deleted C++ op used to write them).
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

    @requires_sm100
    def test_uncompiled_request_count_raises(self):
        """A request count outside the precompiled variants fails loudly.

        There is no fallback kernel, so ``supports()`` misses must raise
        instead of silently scoring through a slower path.
        """
        pytest.importorskip("cutlass")
        (
            group,
            token_starts,
            valid_seq_lens,
            _,
            mean_cos,
            mean_sin,
            _,
        ) = _build_case(prompt_len=5, seed=20260719, **_QWEN3_CASE)
        valid_widths = torch.empty(
            _QWEN3_CASE["max_requests"], dtype=torch.int32, device=group.output.device
        )
        with pytest.raises(RuntimeError, match="no compiled variant"):
            group.launch(
                _QWEN3_CASE["max_requests"] - 1,
                valid_seq_lens,
                valid_widths,
                token_starts,
                mean_cos,
                mean_sin,
            )

    def _tiny_unsupported_group(self, dtype: torch.dtype):
        """A geometry far outside the CuTe contract (constructor accepts it)."""
        device = torch.device("cuda", torch.cuda.current_device())
        torch.manual_seed(20260722)
        num_layers, max_requests, page_count, tokens_per_block, head_dim = 2, 2, 2, 4, 8
        num_freqs = head_dim // 2
        pools = [
            torch.randn(
                max_requests * page_count, 2, 1, tokens_per_block, head_dim, device=device
            ).to(dtype)
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

    def test_unsupported_geometry_raises(self):
        """Score setup outside the CuTe contract raises; nothing falls back."""
        group, launch_args = self._tiny_unsupported_group(torch.float32)
        with pytest.raises(ValueError, match="TriAttention score requires SM100"):
            group.launch(1, *launch_args)

    def test_max_aggregation_raises(self):
        """Max aggregation was removed with the C++ score stack."""
        group, launch_args = self._tiny_unsupported_group(torch.bfloat16)
        with pytest.raises(ValueError, match="max aggregation"):
            group.launch(1, *launch_args, aggregation="max")
