# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness coverage for the optional SM100 TriAttention CuTe scorer."""

import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="TriAttention CuTe score kernel requires SM100",
)
def test_cute_score_matches_torch_mean_oracle(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("cutlass")
    monkeypatch.setenv("TRTLLM_TRIATTENTION_CUTE_SCORE", "1")

    from tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels import (
        _FixedScoreGroup,
    )

    torch.manual_seed(20260720)
    device = torch.device("cuda")
    seq_len = 256
    num_q_heads = 8
    num_freqs = 32
    pool = (0.125 * torch.randn(2, 2, 1, 128, 64, device=device)).to(torch.bfloat16)
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
    # kv_factor == 2. Both requests read pool pages [0, 1].
    block_offsets = torch.tensor(
        [[[[0, 2], [1, 3]], [[0, 2], [1, 3]]]], dtype=torch.int32, device=device
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
    keys = pool[:, 0, 0].reshape(seq_len, 2 * num_freqs).float()
    k_real = keys[:, :num_freqs]
    k_imag = keys[:, num_freqs:]
    magnitude = torch.sqrt(k_real.square() + k_imag.square())
    valid_seq_lens = torch.tensor([seq_len, seq_len], dtype=torch.int32, device=device)
    valid_widths = torch.tensor([seq_len, seq_len], dtype=torch.int32, device=device)
    round_starts_device = torch.tensor([seq_len, seq_len + 1], dtype=torch.int32, device=device)
    token_starts_device = torch.zeros(2, dtype=torch.int32, device=device)
    for request_count in (1, 2):
        group.output.fill_(float("nan"))
        actual = group.launch(
            request_count,
            valid_seq_lens,
            valid_widths,
            round_starts_device,
            token_starts_device,
            mean_cos,
            mean_sin,
            "mean",
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
            torch.testing.assert_close(
                actual[request, 0],
                expected,
                rtol=5.0e-3,
                atol=5.0e-3,
            )

    torch.cuda.synchronize()
    # Fails loudly if setup silently fell back to the C++ score ops (whose
    # scores would also match the oracle here).
    assert group._cute_score_runner is not None
