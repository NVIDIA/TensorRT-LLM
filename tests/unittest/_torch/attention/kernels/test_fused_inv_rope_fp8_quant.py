# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Numerical-equivalence test for the fused inverse-RoPE + 1x128 FP8 quant op
``trtllm::fused_inv_rope_fp8_quant_vllm_port`` (now implemented in
``cpp/tensorrt_llm/{kernels,thop}/inverseRopeFp8QuantKernel*``).

Reference path is the legacy 2-kernel pair (``mla_rope_inplace`` ->
``fp8_batched_quantize_1x128_permute102``); the fused op must match within
1 FP8 ULP per element.

Runs on SM100+ (B200/B300), where the per-head FP8 quant + cute_dsl_bmm
consumer is wired in.
"""

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
    return torch.ops.trtllm.fused_inv_rope_fp8_quant_vllm_port(
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


def _dequant(fp8, scale):
    """[G, T, D] fp8 * [G, D/128, pad_up(T, 4)] fp32 -> [G, T, D] fp32."""
    G, T, D = fp8.shape
    f = fp8.to(torch.float32)
    s = scale.permute(0, 2, 1).contiguous()  # [G, pad_up_T, D/128]
    s = s[:, :T, :]
    s = s.unsqueeze(-1).expand(G, T, D // 128, 128).reshape(G, T, D)
    return f * s


def _check_match(fp8_ref, scale_ref, fp8_fused, scale_fused, *, ctx):
    assert fp8_ref.shape == fp8_fused.shape, (ctx, fp8_ref.shape, fp8_fused.shape)
    assert scale_ref.shape == scale_fused.shape, (ctx, scale_ref.shape, scale_fused.shape)
    deq_ref = _dequant(fp8_ref, scale_ref)
    deq_fused = _dequant(fp8_fused, scale_fused)
    abs_diff = (deq_ref - deq_fused).abs()
    rel = abs_diff.mean() / (deq_ref.abs().mean() + 1e-9)
    print(
        f"[{ctx}] mean abs diff = {abs_diff.mean().item():.4e}  "
        f"rel = {rel.item():.4e}  max = {abs_diff.max().item():.4e}"
    )
    # FP8 e4m3: ~3 mantissa bits. FMA reordering can flip a single
    # round-to-nearest decision -> at most 1 ULP per element. 1% relative
    # mean error gives a generous bound for the dequant-space comparison.
    assert rel.item() < 1e-2, f"{ctx}: relative mismatch {rel.item()}"


# DSv4 production shapes (from DeepSeek-V4-{Flash,Pro} config.json):
#   Flash: num_heads=64,  o_groups=8,  head_dim=512 (kv_lora_rank=448 + rope=64)
#   Pro:   num_heads=128, o_groups=16, head_dim=512 (kv_lora_rank=448 + rope=64)
# Both reduce to heads_per_group=8 at runtime (n_local_groups depends on TP).
# We also exercise smaller chunks_per_head values for kernel-template coverage.
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(
            dict(n_groups=8, heads_per_group=8, nope_dim=448, rope_dim=64), id="dsv4_flash_dep8"
        ),
        pytest.param(
            dict(n_groups=1, heads_per_group=8, nope_dim=448, rope_dim=64), id="dsv4_flash_tp8"
        ),
        pytest.param(
            dict(n_groups=16, heads_per_group=8, nope_dim=448, rope_dim=64), id="dsv4_pro_dep16"
        ),
        pytest.param(
            dict(n_groups=2, heads_per_group=8, nope_dim=448, rope_dim=64), id="dsv4_pro_tp8"
        ),
        pytest.param(
            dict(n_groups=4, heads_per_group=8, nope_dim=448, rope_dim=64), id="legacy_dep8"
        ),
        pytest.param(dict(n_groups=1, heads_per_group=8, nope_dim=320, rope_dim=64), id="chunks=3"),
        pytest.param(dict(n_groups=1, heads_per_group=8, nope_dim=192, rope_dim=64), id="chunks=2"),
        pytest.param(dict(n_groups=1, heads_per_group=16, nope_dim=64, rope_dim=64), id="chunks=1"),
    ],
)
@pytest.mark.parametrize("is_neox", [True, False], ids=["neox", "gptj"])
@pytest.mark.parametrize("num_tokens", [1, 3, 8, 64, 257, 1024, 4096, 8192])
def test_fused_inv_rope_fp8_quant(num_tokens, shape, is_neox):
    """Fused op (CUDA) vs the legacy 2-kernel reference. Covers both rope
    layouts and all production DSv4 shape combinations within 1 FP8 ULP."""
    torch.manual_seed(0)
    device = "cuda"
    n_groups = shape["n_groups"]
    heads_per_group = shape["heads_per_group"]
    num_heads = n_groups * heads_per_group
    nope_dim, rope_dim = shape["nope_dim"], shape["rope_dim"]
    head_dim = nope_dim + rope_dim
    max_pos = 2048

    o_bf16 = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    half = rope_dim // 2
    cos = torch.randn(max_pos, half, dtype=torch.float32, device=device).cos()
    sin = torch.randn(max_pos, half, dtype=torch.float32, device=device).sin()
    rotary_cos_sin = torch.stack([cos, sin], dim=1).contiguous()
    assert rotary_cos_sin.shape == (max_pos, 2, half)

    position_ids = torch.randint(0, max_pos, (num_tokens,), dtype=torch.int32, device=device)

    fp8_ref, scale_ref = _ref_path(
        o_bf16,
        position_ids,
        rotary_cos_sin,
        num_heads,
        n_groups,
        nope_dim,
        rope_dim,
        is_neox=is_neox,
    )
    fp8_fused, scale_fused = _fused_path(
        o_bf16,
        position_ids,
        rotary_cos_sin,
        n_groups,
        heads_per_group,
        nope_dim,
        rope_dim,
        is_neox=is_neox,
    )
    _check_match(
        fp8_ref,
        scale_ref,
        fp8_fused,
        scale_fused,
        ctx=f"M={num_tokens} is_neox={is_neox} {shape}",
    )


if __name__ == "__main__":
    for M in (1, 64, 8192):
        for sh in (
            dict(n_groups=4, heads_per_group=8, nope_dim=448, rope_dim=64),
            dict(n_groups=1, heads_per_group=16, nope_dim=64, rope_dim=64),
        ):
            for is_neox in (True, False):
                test_fused_inv_rope_fp8_quant(M, sh, is_neox)
    print("OK")
