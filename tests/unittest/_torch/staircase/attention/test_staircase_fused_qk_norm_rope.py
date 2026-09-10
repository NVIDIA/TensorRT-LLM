# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fused_qk_norm_rope catalog entry."""

import torch

from tensorrt_llm._torch.staircase.catalog.attention.fused_qk_norm_rope import fused_qk_norm_rope

assert torch.cuda.is_available(), "fused_qk_norm_rope requires a CUDA device"

# The kernel rotates the bf16-rounded (x1, x2) pair, so the per-element
# error is absolute in the pair magnitude (up to ~5 after RMS norm with
# weights near 1), not relative to each output element: 0.02 ~= 5 ulp of
# bf16 (2^-8) at magnitude 5. Observed max abs err across all cases is
# 0.016; the default bf16 atol of 1e-5 is unreachable for near-zero
# outputs produced by rotating large pairs.
ATOL = 0.02
RTOL = 1.6e-2  # torch default for bf16


def _inv_freq(rotary_dim: int, base: float, factor: float, low: float, high: float) -> torch.Tensor:
    """YaRN-blended inverse frequencies; plain RoPE when factor == 1."""
    half = rotary_dim // 2
    j = torch.arange(half, dtype=torch.float32, device="cuda")
    pos_freqs = base ** (2.0 * j / rotary_dim)
    inv_extrapolation = 1.0 / pos_freqs
    inv_interpolation = 1.0 / (factor * pos_freqs)
    if high == low:
        high = high + 0.001  # kernel guards the ramp singularity the same way
    ramp = ((j - low) / (high - low)).clamp(0.0, 1.0)
    extrapolation_factor = 1.0 - ramp
    return (
        inv_interpolation * (1.0 - extrapolation_factor) + inv_extrapolation * extrapolation_factor
    )


def _ref(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    base: float,
    is_neox: bool,
    position_ids: torch.Tensor,
    factor: float = 1.0,
    low: float = 0.0,
    high: float = 0.0,
    attention_factor: float = 1.0,
    is_qk_norm: bool = True,
    use_gemma: bool = False,
    use_mrope: bool = False,
    mrope_section1: int = 0,
    mrope_section2: int = 0,
) -> torch.Tensor:
    """fp32 reference: per-head RMS norm on q/k heads, then RoPE; v untouched."""
    num_heads = num_heads_q + num_heads_k + num_heads_v
    x = qkv.float().view(-1, num_heads, head_dim).clone()
    half = rotary_dim // 2
    inv_freq = _inv_freq(rotary_dim, base, factor, low, high)
    if use_mrope:
        angle = position_ids.float()[:, :, None] * inv_freq  # [3, T, half]
        cos, sin = angle.cos(), angle.sin()

        def pick(c: torch.Tensor) -> torch.Tensor:
            out = c[0].clone()
            out[:, 1 : mrope_section1 * 3 : 3] = c[1][:, 1 : mrope_section1 * 3 : 3]
            out[:, 2 : mrope_section2 * 3 : 3] = c[2][:, 2 : mrope_section2 * 3 : 3]
            return out

        cos, sin = pick(cos), pick(sin)
    else:
        angle = position_ids.float()[:, None] * inv_freq  # [T, half]
        cos, sin = angle.cos(), angle.sin()
    cos = cos[:, None, :] * attention_factor
    sin = sin[:, None, :] * attention_factor
    for start, count, weight in (
        (0, num_heads_q, q_weight.float()),
        (num_heads_q, num_heads_k, k_weight.float()),
    ):
        h = x[:, start : start + count, :]
        if is_qk_norm:
            h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
            h = h * ((1.0 + weight) if use_gemma else weight)
        r = h[..., :rotary_dim]
        if is_neox:
            x1, x2 = r[..., :half], r[..., half:]
            rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        else:
            x1, x2 = r[..., ::2], r[..., 1::2]
            rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
        x[:, start : start + count, :] = torch.cat([rotated, h[..., rotary_dim:]], -1)
    return x.view(qkv.shape[0], -1)


def _check(
    num_tokens: int,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    base: float,
    is_neox: bool,
    factor: float = 1.0,
    low: float = 0.0,
    high: float = 0.0,
    attention_factor: float = 1.0,
    is_qk_norm: bool = True,
    use_gemma: bool = False,
    use_mrope: bool = False,
    mrope_section1: int = 0,
    mrope_section2: int = 0,
) -> None:
    width = (num_heads_q + 2 * num_heads_kv) * head_dim
    qkv = torch.randn(num_tokens, width, dtype=torch.bfloat16, device="cuda")
    q_w = torch.randn(head_dim, dtype=torch.bfloat16, device="cuda") * 0.1 + 1.0
    k_w = torch.randn(head_dim, dtype=torch.bfloat16, device="cuda") * 0.1 + 1.0
    shape = (3, num_tokens) if use_mrope else (num_tokens,)
    pos = torch.randint(0, 32768, shape, dtype=torch.int32, device="cuda")
    expected = _ref(
        qkv, num_heads_q, num_heads_kv, num_heads_kv, head_dim, rotary_dim,
        eps, q_w, k_w, base, is_neox, pos, factor, low, high, attention_factor,
        is_qk_norm, use_gemma, use_mrope, mrope_section1, mrope_section2,
    ).to(torch.bfloat16)  # fmt: skip
    fused_qk_norm_rope(
        qkv, num_heads_q, num_heads_kv, num_heads_kv, head_dim, rotary_dim,
        eps, q_w, k_w, base, is_neox, pos, factor, low, high, attention_factor,
        is_qk_norm, use_gemma, use_mrope, mrope_section1, mrope_section2,
    )  # fmt: skip
    torch.testing.assert_close(qkv, expected, rtol=RTOL, atol=ATOL)
    # v heads must pass through bit-exactly
    v = qkv.view(num_tokens, -1, head_dim)[:, num_heads_q + num_heads_kv :, :]
    ev = expected.view(num_tokens, -1, head_dim)[:, num_heads_q + num_heads_kv :, :]
    assert torch.equal(v, ev)


def test_bf16_neox_head_dims() -> None:
    torch.manual_seed(0)
    # decode-like shapes over every supported head_dim
    for head_dim in (64, 128, 256):
        _check(
            2, 8, 2, head_dim=head_dim, rotary_dim=head_dim, eps=1e-6,
            base=10000.0, is_neox=True,
        )  # fmt: skip


def test_bf16_neox_prefill() -> None:
    torch.manual_seed(1)
    # prefill-like: many tokens, Qwen3-32B-like head layout
    _check(
        8192, 32, 8, head_dim=128, rotary_dim=128, eps=1e-6,
        base=1000000.0, is_neox=True,
    )  # fmt: skip


def test_bf16_interleaved() -> None:
    torch.manual_seed(2)
    for num_tokens in (1, 65):
        _check(
            num_tokens, 8, 2, head_dim=128, rotary_dim=128, eps=1e-6,
            base=10000.0, is_neox=False,
        )  # fmt: skip


def test_bf16_gemma_norm() -> None:
    torch.manual_seed(3)
    _check(
        17, 4, 4, head_dim=64, rotary_dim=64, eps=1e-6,
        base=10000.0, is_neox=True, use_gemma=True,
    )  # fmt: skip


def test_bf16_rope_only() -> None:
    torch.manual_seed(4)
    # is_qk_norm=False skips the norm entirely; weights are ignored
    _check(
        17, 8, 2, head_dim=128, rotary_dim=128, eps=1e-6,
        base=10000.0, is_neox=True, is_qk_norm=False,
    )  # fmt: skip


def test_bf16_partial_rotary() -> None:
    torch.manual_seed(5)
    # norm covers the full head_dim; rope covers only the first rotary_dim
    for is_neox in (True, False):
        _check(
            17, 8, 2, head_dim=128, rotary_dim=64, eps=1e-6,
            base=10000.0, is_neox=is_neox,
        )  # fmt: skip


def test_bf16_yarn() -> None:
    torch.manual_seed(6)
    _check(
        33, 8, 2, head_dim=128, rotary_dim=128, eps=1e-6,
        base=1000000.0, is_neox=True,
        factor=4.0, low=4.0, high=20.0, attention_factor=1.2,
    )  # fmt: skip


def test_bf16_mrope_interleaved() -> None:
    torch.manual_seed(7)
    _check(
        19, 8, 2, head_dim=128, rotary_dim=128, eps=1e-6,
        base=10000.0, is_neox=True,
        use_mrope=True, mrope_section1=16, mrope_section2=24,
    )  # fmt: skip


def test_rejects_out_of_contract() -> None:
    torch.manual_seed(8)
    pos = torch.arange(4, dtype=torch.int32, device="cuda")
    w = torch.ones(128, dtype=torch.bfloat16, device="cuda")

    def call(qkv, head_dim=128, q_w=w, position_ids=pos):
        fused_qk_norm_rope(
            qkv, 8, 2, 2, head_dim, head_dim, 1e-6, q_w, w,
            10000.0, True, position_ids,
        )  # fmt: skip

    # non-bf16 qkv
    for dtype in (torch.float16, torch.float32):
        try:
            call(torch.randn(4, 12 * 128, dtype=dtype, device="cuda"))
            raise AssertionError(f"{dtype} qkv unexpectedly accepted")
        except RuntimeError:
            pass
    # unsupported head_dim
    try:
        w96 = torch.ones(96, dtype=torch.bfloat16, device="cuda")
        call(
            torch.randn(4, 12 * 96, dtype=torch.bfloat16, device="cuda"),
            head_dim=96,
            q_w=w96,
        )
        raise AssertionError("head_dim=96 unexpectedly accepted")
    except RuntimeError:
        pass
    # non-int32 position_ids
    try:
        call(
            torch.randn(4, 12 * 128, dtype=torch.bfloat16, device="cuda"),
            position_ids=pos.to(torch.int64),
        )
        raise AssertionError("int64 position_ids unexpectedly accepted")
    except RuntimeError:
        pass
