import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


def _reference_q_norm(q: torch.Tensor, head_dim: int, eps: float) -> torch.Tensor:
    q_view = q.view(-1, head_dim).float()
    inv_rms = torch.rsqrt(q_view.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (q_view * inv_rms).to(q.dtype).view_as(q)


@pytest.mark.parametrize("num_tokens", [1, 7, 129])
@pytest.mark.parametrize("num_heads", [1, 16])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.skipif(
                not torch.cuda.is_bf16_supported(), reason="Requires BF16 support"
            ),
        ),
    ],
)
def test_deepseek_v4_q_norm_matches_reference(num_tokens, num_heads, dtype):
    torch.manual_seed(0)
    device = "cuda"
    head_dim = 512
    eps = 1e-6
    q = torch.randn(num_tokens, num_heads * head_dim, dtype=dtype, device=device).contiguous()

    out = torch.ops.trtllm.deepseek_v4_q_norm(q, num_heads, head_dim, eps)
    ref = _reference_q_norm(q, head_dim, eps)

    atol = 3.2e-2 if dtype == torch.bfloat16 else 5e-3
    rtol = 8e-3 if dtype == torch.bfloat16 else 5e-3
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


def test_deepseek_v4_q_norm_torch_compile_fullgraph():
    torch.manual_seed(0)
    num_heads = 16
    head_dim = 512
    eps = 1e-6
    q = torch.randn(8, num_heads * head_dim, dtype=torch.float16, device="cuda").contiguous()

    def q_norm(q):
        return torch.ops.trtllm.deepseek_v4_q_norm(q, num_heads, head_dim, eps)

    out = torch.compile(q_norm, backend="eager", fullgraph=True)(q)
    ref = _reference_q_norm(q, head_dim, eps)

    torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)


def _reference_q_norm_fused_fp8(
    q: torch.Tensor,
    num_heads: int,
    head_dim: int,
    nope_dim: int,
    eps: float,
    quant_scale_qkv: float,
):
    """Reference: per-row RMSNorm in fp32; split last column-axis into nope/rope;
    nope path multiplied by quant_scale_qkv then cast to fp8_e4m3; rope path cast
    back to input dtype.
    """
    num_tokens = q.shape[0]
    rope_dim = head_dim - nope_dim
    q_view = q.view(num_tokens * num_heads, head_dim).float()
    inv_rms = torch.rsqrt(q_view.pow(2).mean(dim=-1, keepdim=True) + eps)
    normalized = q_view * inv_rms

    nope_fp32 = normalized[:, :nope_dim] * quant_scale_qkv
    rope_fp32 = normalized[:, nope_dim:]

    quant_q_nope = nope_fp32.to(torch.float8_e4m3fn).view(num_tokens, num_heads * nope_dim)
    q_pe = rope_fp32.to(q.dtype).view(num_tokens, num_heads * rope_dim)
    return quant_q_nope, q_pe


@pytest.mark.parametrize("num_tokens", [1, 7, 129])
@pytest.mark.parametrize("num_heads", [1, 16, 128])
@pytest.mark.parametrize("quant_scale_qkv", [None, 0.5, 2.0])
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.skipif(
                not torch.cuda.is_bf16_supported(), reason="Requires BF16 support"
            ),
        ),
        torch.float16,
    ],
)
def test_deepseek_v4_q_norm_fused_fp8_matches_reference(
    num_tokens, num_heads, quant_scale_qkv, dtype
):
    torch.manual_seed(0)
    device = "cuda"
    head_dim = 512
    nope_dim = 448
    eps = 1e-6
    q = torch.randn(num_tokens, num_heads * head_dim, dtype=dtype, device=device).contiguous()

    if quant_scale_qkv is None:
        scale_tensor = None
        scale_value = 1.0
    else:
        scale_tensor = torch.tensor([quant_scale_qkv], dtype=torch.float32, device=device)
        scale_value = quant_scale_qkv

    rope_dim = head_dim - nope_dim
    quant_q_nope = q.new_empty((num_tokens, num_heads * nope_dim), dtype=torch.float8_e4m3fn)
    q_pe = q.new_empty((num_tokens, num_heads * rope_dim))
    torch.ops.trtllm.deepseek_v4_q_norm_fused_fp8(
        q, quant_q_nope, q_pe, num_heads, head_dim, nope_dim, eps, scale_tensor
    )

    ref_quant_q_nope, ref_q_pe = _reference_q_norm_fused_fp8(
        q, num_heads, head_dim, nope_dim, eps, scale_value
    )

    atol_pe = 3.2e-2 if dtype == torch.bfloat16 else 5e-3
    rtol_pe = 8e-3 if dtype == torch.bfloat16 else 5e-3
    torch.testing.assert_close(q_pe, ref_q_pe, atol=atol_pe, rtol=rtol_pe)

    # FP8 nope is bit-identical to torch's reference except at FP8 grid
    # midpoints where the kernel's PTX cvt (round-to-nearest-even) and torch's
    # round-half-away-from-zero disagree by exactly 1 ULP. Allow up to 1 ULP
    # per element, with a hard cap so a real per-row bug still fires.
    quant_q_nope_f32 = quant_q_nope.to(torch.float32)
    ref_quant_q_nope_f32 = ref_quant_q_nope.to(torch.float32)
    diff = quant_q_nope_f32 - ref_quant_q_nope_f32
    n_mismatched = int((diff != 0.0).sum().item())
    if n_mismatched > 0:
        ref_abs = ref_quant_q_nope_f32.abs().clamp(min=2**-6)
        krn_abs = quant_q_nope_f32.abs().clamp(min=2**-6)
        step = torch.maximum(ref_abs, krn_abs).log2().floor().exp2() * (2**-3)
        n_beyond_1ulp = int((diff.abs() > step * 1.001).sum().item())
        assert n_beyond_1ulp == 0, (
            f"FP8 nope: {n_beyond_1ulp}/{n_mismatched} mismatches exceed 1 FP8 ULP"
        )
    assert n_mismatched <= 16, (
        f"FP8 nope: {n_mismatched} mismatched elements exceeds 16-element cap"
    )


def test_deepseek_v4_q_norm_fused_fp8_zero_rows():
    """Edge case: 0 tokens should not launch the kernel and should not crash."""
    num_heads = 4
    head_dim = 512
    nope_dim = 448
    rope_dim = head_dim - nope_dim
    eps = 1e-6
    q = torch.empty(0, num_heads * head_dim, dtype=torch.bfloat16, device="cuda")
    quant_q_nope = q.new_empty((0, num_heads * nope_dim), dtype=torch.float8_e4m3fn)
    q_pe = q.new_empty((0, num_heads * rope_dim))
    torch.ops.trtllm.deepseek_v4_q_norm_fused_fp8(
        q, quant_q_nope, q_pe, num_heads, head_dim, nope_dim, eps, None
    )
    assert quant_q_nope.shape == (0, num_heads * nope_dim)
    assert q_pe.shape == (0, num_heads * rope_dim)


@pytest.mark.parametrize("num_tokens", [1, 7, 129])
def test_deepseek_v4_q_b_layernorm_fused_fp8_returns_3d_q_pe(num_tokens):
    """Lock down the q_pe dim==3 contract expected by thop.attention's
    sparse-MLA context branch (TORCH_CHECK(q_pe->dim() == 3))."""
    import types
    from types import SimpleNamespace

    from tensorrt_llm._torch.modules.attention import MLA

    num_heads = 16
    qk_head_dim = 512
    kv_lora_rank = 448
    rope_dim = qk_head_dim - kv_lora_rank
    stub = SimpleNamespace(
        num_heads_tp=num_heads,
        qk_head_dim=qk_head_dim,
        kv_lora_rank=kv_lora_rank,
        q_b_layernorm=SimpleNamespace(variance_epsilon=1e-6),
    )
    fused = types.MethodType(MLA._deepseek_v4_q_b_layernorm_fused_fp8, stub)
    q_proj = torch.randn(
        num_tokens, num_heads * qk_head_dim, dtype=torch.bfloat16, device="cuda"
    ).contiguous()

    placeholder_q, quant_q_buffer, q_pe, scale = fused(q_proj)

    assert q_pe.shape == (num_tokens, num_heads, rope_dim)
    assert q_pe.stride(2) == 1
    assert q_pe.is_contiguous()
    assert quant_q_buffer.shape == (num_tokens, num_heads * qk_head_dim)
    assert quant_q_buffer.dtype == torch.float8_e4m3fn
    assert placeholder_q.data_ptr() == q_proj.data_ptr()
    assert scale.shape == (1,) and scale.dtype == torch.float32
    assert float(scale.item()) == 1.0


@pytest.mark.parametrize("num_tokens", [1, 7, 129])
@pytest.mark.parametrize("num_heads", [1, 16, 128])
def test_deepseek_v4_q_norm_fused_fp8_interleaved_layout(num_tokens, num_heads):
    """The per-head stride of `quant_q_out` is inferred from the buffer shape:
    `H * nope_dim` -> packed, `H * head_dim` -> interleaved (rope slot left
    untouched). This test verifies both shapes write the same nope values to
    the right offsets and produce identical q_pe.
    """
    torch.manual_seed(0)
    device = "cuda"
    head_dim = 512
    nope_dim = 448
    rope_dim = head_dim - nope_dim
    eps = 1e-6
    q = torch.randn(
        num_tokens, num_heads * head_dim, dtype=torch.bfloat16, device=device
    ).contiguous()
    scale_tensor = torch.tensor([0.5], dtype=torch.float32, device=device)

    # Packed [N, H*nope_dim] layout.
    packed_quant_q_nope = q.new_empty((num_tokens, num_heads * nope_dim), dtype=torch.float8_e4m3fn)
    packed_q_pe = q.new_empty((num_tokens, num_heads * rope_dim))
    torch.ops.trtllm.deepseek_v4_q_norm_fused_fp8(
        q, packed_quant_q_nope, packed_q_pe, num_heads, head_dim, nope_dim, eps, scale_tensor
    )

    # Interleaved [N, H*head_dim] layout; rope slot must be untouched.
    interleaved = q.new_empty((num_tokens, num_heads * head_dim), dtype=torch.float8_e4m3fn)
    q_pe_interleaved = q.new_empty((num_tokens, num_heads * rope_dim))
    torch.ops.trtllm.deepseek_v4_q_norm_fused_fp8(
        q, interleaved, q_pe_interleaved, num_heads, head_dim, nope_dim, eps, scale_tensor
    )

    # The nope slots of `interleaved` should match the packed output element-by-element.
    interleaved_3d = interleaved.view(num_tokens, num_heads, head_dim)
    packed_3d = packed_quant_q_nope.view(num_tokens, num_heads, nope_dim)
    nope_slice = interleaved_3d[:, :, :nope_dim].contiguous()
    torch.testing.assert_close(
        nope_slice.to(torch.float32),
        packed_3d.to(torch.float32),
        atol=0.0,
        rtol=0.0,
        msg="Interleaved nope output should match packed output bit-for-bit",
    )
    # q_pe should be identical in both modes.
    torch.testing.assert_close(q_pe_interleaved, packed_q_pe, atol=0.0, rtol=0.0)
