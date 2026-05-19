# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Numerical-equivalence test for the vLLM-ported fused inverse-RoPE +
FP8 1x128 quantize op.

Compares (mla_rope_inplace -> fp8_batched_quantize_1x128_permute102) against
the new fused op `trtllm::fused_inv_rope_fp8_quant_vllm_port` on each of its
backends:

  - default — CuTe DSL kernel (preferred on SM100; matches Triton at large M
    and is 12-19% faster at small M)
  - `TLLM_DISABLE_CUTE_DSL_FUSED_INV_ROPE=1` — explicit Triton fallback

Run inside a CUDA Blackwell (SM100f) container (the TRT-LLM build sqsh).
"""

import os

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401  (registers the fused op)
from tensorrt_llm._utils import is_sm_100f

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_sm_100f()),
    reason="Requires SM100 family GPU",
)


def _ref_path(
    o_bf16, position_ids, rotary_cos_sin, num_heads_tp, n_groups, nope_dim, rope_dim, is_neox
):
    """Reference: in-place mla_rope_inplace -> fp8_batched_quantize_1x128_permute102."""
    o = o_bf16.clone().contiguous()
    torch.ops.trtllm.mla_rope_inplace(
        o, position_ids.view(-1), rotary_cos_sin, num_heads_tp, nope_dim, rope_dim, True, is_neox
    )
    # Reshape to [N, n_groups, heads_per_group * head_dim] to match the BMM site.
    num_tokens = o.shape[0]
    grouped = o.view(num_tokens, n_groups, -1)
    fp8_ref, scale_ref = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(grouped)
    return fp8_ref, scale_ref


def _fused_path(
    o_bf16, position_ids, rotary_cos_sin, n_groups, heads_per_group, nope_dim, rope_dim, is_neox
):
    o = o_bf16.contiguous()
    fp8_fused, scale_fused = torch.ops.trtllm.fused_inv_rope_fp8_quant_vllm_port(
        o,
        position_ids.view(-1),
        rotary_cos_sin,
        n_groups,
        heads_per_group,
        nope_dim,
        rope_dim,
        128,
        is_neox,
    )
    return fp8_fused, scale_fused


def _run_neox_compare(num_tokens, backend_env):
    """Compare the fused op (under the given backend env) against the reference
    (mla_rope_inplace -> fp8_batched_quantize_1x128_permute102) path.

    `backend_env` is a dict of env vars to set for the duration of the call;
    pass `{}` for the default (Triton) backend.
    """
    torch.manual_seed(0)
    device = "cuda"
    n_groups = 4
    heads_per_group = 8  # smaller for the test
    num_heads = n_groups * heads_per_group
    nope_dim, rope_dim = 448, 64
    head_dim = nope_dim + rope_dim  # 512
    max_pos = 2048

    o_bf16 = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    # Simulate the TRT-LLM rotary_cos_sin layout: (max_pos, 2, rope_dim/2) fp32.
    half = rope_dim // 2
    cos = torch.randn(max_pos, half, dtype=torch.float32, device=device).cos()
    sin = torch.randn(max_pos, half, dtype=torch.float32, device=device).sin()
    rotary_cos_sin = torch.stack([cos, sin], dim=1).contiguous()
    assert rotary_cos_sin.shape == (max_pos, 2, half)

    position_ids = torch.randint(0, max_pos, (num_tokens,), dtype=torch.int32, device=device)

    fp8_ref, scale_ref = _ref_path(
        o_bf16, position_ids, rotary_cos_sin, num_heads, n_groups, nope_dim, rope_dim, is_neox=True
    )

    # Apply backend env vars (e.g. TLLM_FUSED_INV_ROPE_FP8_USE_CUTE_DSL=1)
    # only for the duration of the fused call.
    saved = {k: os.environ.get(k) for k in backend_env}
    try:
        for k, v in backend_env.items():
            os.environ[k] = v
        fp8_fused, scale_fused = _fused_path(
            o_bf16,
            position_ids,
            rotary_cos_sin,
            n_groups,
            heads_per_group,
            nope_dim,
            rope_dim,
            is_neox=True,
        )
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    assert fp8_ref.shape == fp8_fused.shape, (fp8_ref.shape, fp8_fused.shape)
    assert scale_ref.shape == scale_fused.shape, (scale_ref.shape, scale_fused.shape)

    # Compare in dequantized BF16 space (FP8 truncation + FMA reordering can
    # differ by 1 ULP).
    def dequant(fp8, scale):
        # fp8: [G, T, D]; scale: [G, D/128, pad_up(T,4)]
        G, T, D = fp8.shape
        f = fp8.to(torch.float32)
        # Expand scale to [G, T, D]
        s = scale.permute(0, 2, 1).contiguous()  # [G, pad_up_T, D/128]
        s = s[:, :T, :]
        s = s.unsqueeze(-1).expand(G, T, D // 128, 128).reshape(G, T, D)
        return f * s

    deq_ref = dequant(fp8_ref, scale_ref)
    deq_fused = dequant(fp8_fused, scale_fused)
    abs_diff = (deq_ref - deq_fused).abs()
    rel = abs_diff.mean() / (deq_ref.abs().mean() + 1e-9)
    backend_str = ",".join(f"{k}={v}" for k, v in backend_env.items()) or "triton-default"
    print(
        f"[NEOX num_tokens={num_tokens} backend={backend_str}] "
        f"mean abs diff = {abs_diff.mean().item():.4e}  rel = {rel.item():.4e}  "
        f"max = {abs_diff.max().item():.4e}"
    )
    # FP8 e4m3 has ~3 mantissa bits; 1-ULP tolerance scales with the value.
    # Allow a generous 1% relative bound.
    assert rel.item() < 1e-2, f"relative mismatch {rel.item()}"


@pytest.mark.parametrize("num_tokens", [3, 64, 257, 512, 1024, 2048, 8192])
def test_fused_inv_rope_fp8_quant_neox(num_tokens):
    """Default backend (CuTe DSL on SM100)."""
    _run_neox_compare(num_tokens, backend_env={})


@pytest.mark.parametrize("num_tokens", [3, 64, 257, 512, 1024, 2048, 8192])
def test_fused_inv_rope_fp8_quant_neox_triton_fallback(num_tokens):
    """Explicit Triton fallback (`TLLM_DISABLE_CUTE_DSL_FUSED_INV_ROPE=1`)."""
    _run_neox_compare(num_tokens, backend_env={"TLLM_DISABLE_CUTE_DSL_FUSED_INV_ROPE": "1"})


if __name__ == "__main__":
    test_fused_inv_rope_fp8_quant_neox(64)
    test_fused_inv_rope_fp8_quant_neox(257)
    test_fused_inv_rope_fp8_quant_neox_triton_fallback(64)
    test_fused_inv_rope_fp8_quant_neox_triton_fallback(257)
    print("OK")
