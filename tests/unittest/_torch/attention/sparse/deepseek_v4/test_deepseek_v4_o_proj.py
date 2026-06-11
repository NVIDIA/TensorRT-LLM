"""
Test for DeepSeek-V4 output projection (_deepseek_v4_o_proj).
"""

from types import SimpleNamespace

import pytest
import torch
from _torch.helpers import per_block_cast_to_fp8_e8m0, per_token_cast_to_fp8_e8m0
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.attention_backend.interface import PositionalEmbeddingParams, RopeParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv3 import weight_dequant
from tensorrt_llm._torch.modules.attention import MLA
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from ..test_sparse_mla_forward import RopeConfig, _calc_diff, apply_rotary_emb, precompute_freqs_cis

FP8_O_PROJ_DIFF_TOL = 2e-3


def _per_token_fp8_quant_dequant(x: torch.Tensor) -> torch.Tensor:
    """Simulate the fused inverse-RoPE FP8 quantization consumed by o_a_proj."""
    original_shape = x.shape
    flattened_x = x.reshape(-1, original_shape[-1])
    fp8_x, scale = per_token_cast_to_fp8_e8m0(flattened_x)
    dequant_x = (fp8_x.view(flattened_x.shape[0], -1, 128).float() * scale.unsqueeze(-1)).view_as(
        flattened_x
    )
    return dequant_x.to(x.dtype).reshape(original_shape)


def calculate_reference_deepseek_v4_o_proj(
    attn_out_latent,
    o_a_proj,
    o_b_proj_weight,
    freqs_cis,
    n_local_groups,
    qk_nope_head_dim,
    qk_rope_head_dim,
    device,
    is_fp8: bool = False,
):
    """
    Reference implementation for DeepSeek-V4 output projection based on ref/model.py.

    Args:
        attn_out_latent: [num_tokens, num_heads, qk_head_dim] attention output
        o_a_proj: [n_local_groups, o_lora_rank, num_heads * qk_head_dim // n_groups]
        o_b_proj_weight: [hidden_size, n_groups * o_lora_rank]
        freqs_cis: [num_toknes, rope_head_dim / 2] rotary embeddings
        n_local_groups: Number of local output projection groups
        qk_nope_head_dim: Dimension of non-positional part
        qk_rope_head_dim: Dimension of positional part
        device: Device to run on
        is_fp8: Whether test fp8 or bf16

    Returns:
        output: [num_tokens, hidden_size] projected output
    """
    num_tokens = attn_out_latent.shape[0]
    qk_nope_head_dim + qk_rope_head_dim

    # Apply RoPE to attn_out_pe
    attn_out_latent = attn_out_latent.unsqueeze(0)
    apply_rotary_emb(attn_out_latent[..., -qk_rope_head_dim:], freqs_cis, inverse=True)

    # Reshape for grouped projection
    attn_out_grouped = attn_out_latent.view(num_tokens, n_local_groups, -1)
    if is_fp8:
        attn_out_grouped = _per_token_fp8_quant_dequant(
            attn_out_grouped.transpose(0, 1).contiguous()
        ).transpose(0, 1)

    # Apply o_a_proj: einsum equivalent to bmm
    o_lora = torch.einsum("tgd,grd->tgr", attn_out_grouped, o_a_proj)

    # Flatten and apply o_b_proj, [num_tokens, n_local_groups * o_lora_rank]
    o_lora_flat = o_lora.flatten(1)
    if is_fp8:
        o_lora_flat = o_lora_flat.to(torch.float8_e4m3fn).to(torch.bfloat16)
    output = torch.nn.functional.linear(o_lora_flat, o_b_proj_weight)  # [num_tokens, hidden_size]

    return output


def _build_dsv4_o_proj_case(num_tokens: int, dtype_str: str, device: torch.device):
    """Build an MLA module, inputs, and reference-path tensors for the DeepSeek-V4
    o_proj tests. Shared by the correctness test and the DSV4_FUSE_OPROJ fused
    fp8-equivalence test so both exercise an identical setup.

    Returns:
        (mla, attn_out_latent, position_ids, refs) where ``refs`` is a namespace
        carrying the dequantized weights / freqs_cis / dims the analytic
        reference path consumes.
    """
    dtype = torch.bfloat16

    # Model configuration matching the reference model
    num_heads = 64
    q_lora_rank = 1024
    kv_lora_rank = 448
    qk_nope_head_dim = 448
    qk_rope_head_dim = 64
    v_head_dim = 512
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    hidden_size = 4096
    max_position_embeddings = 65536
    o_lora_rank = 1024
    num_groups = 8
    n_local_groups = num_groups  # no TP in this test

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Create RoPE config
    rope_config = RopeConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        rope_scaling={
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 4,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 65536,
            "type": "yarn",
        },
        max_position_embeddings=max_position_embeddings,
        rope_theta=10000.0,
        qk_rope_head_dim=qk_rope_head_dim,
        model_type="deepseek_v4",
    )

    # Setup model config with deepseek_v4 sparse attention
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    pretrained_config = SimpleNamespace(
        rms_norm_eps=1e-6,
    )

    # Create sparse attention config for deepseek_v4
    sparse_config = DeepSeekV4SparseAttentionConfig(
        index_n_heads=32,
        index_head_dim=128,
        index_topk=512,
    )

    quant_config = QuantConfig()
    if dtype_str == "fp8":
        quant_config.quant_algo = QuantAlgo.FP8_BLOCK_SCALES
        quant_config.group_size = 128

    model_config = ModelConfig(
        mapping=mapping,
        pretrained_config=pretrained_config,
        sparse_attention_config=sparse_config,
        quant_config=quant_config,
    )

    # Setup positional embedding params
    pos_embd_params = PositionalEmbeddingParams(
        type=PositionEmbeddingType.yarn,
        rope=RopeParams.from_config(rope_config),
        is_neox=False,
    )

    # Create MLA module with deepseek_v4 configuration
    mla = MLA(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=1,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        predicted_tokens_per_seq=1,
        max_position_embeddings=max_position_embeddings,
        bias=False,
        pos_embd_params=pos_embd_params,
        layer_idx=0,
        dtype=dtype,
        config=model_config,
        num_groups=num_groups,
        o_lora_rank=o_lora_rank,
    ).to(device)

    # Initialize weights
    nn_init_std = 0.02
    fp8_a_weight = fp8_a_scale = o_a_proj_bf16 = fp8_b_weight_dequant = dim = None
    with torch.no_grad():
        # Initialize o_a_proj weights
        if dtype_str == "bf16":
            mla.o_a_proj.data = (
                torch.randn(
                    n_local_groups,
                    o_lora_rank,
                    num_heads * qk_head_dim // num_groups,
                    dtype=dtype,
                    device=device,
                )
                * nn_init_std
            )
        elif dtype_str == "fp8":
            dim = num_heads * qk_head_dim // num_groups
            o_a_proj_bf16 = (
                torch.randn(n_local_groups, o_lora_rank, dim, dtype=torch.bfloat16, device=device)
                * nn_init_std
            )

            fp8_a_weight, fp8_a_scale = per_block_cast_to_fp8_e8m0(o_a_proj_bf16.reshape(-1, dim))
            fp8_a_weight = fp8_a_weight.reshape(n_local_groups, o_lora_rank, dim)
            mla.o_a_proj.data = fp8_a_weight
            mla.o_a_proj_scale.data = fp8_a_scale
            # mla.o_a_proj_dequant is None for DSv4 on SM100: PR #14254
            # decouples the FP8-native o_a_proj path from
            # use_cute_dsl_blockscaling_bmm, so DSv4 unconditionally uses the
            # fused inv-RoPE + FP8 quant + cute-dsl BMM chain and never needs
            # the bf16-dequant fallback buffer. The reference path below uses
            # o_a_proj_bf16 directly.

        # Initialize o_b_proj weights
        if dtype_str == "bf16":
            mla.o_b_proj.weight.data = (
                torch.randn(hidden_size, num_groups * o_lora_rank, dtype=dtype, device=device)
                * nn_init_std
            )
        elif dtype_str == "fp8":
            # For FP8, properly quantize using fp8_quantize_1x128_sf_transpose
            o_b_proj_weight_bf16 = (
                torch.randn(
                    hidden_size, num_groups * o_lora_rank, dtype=torch.bfloat16, device=device
                )
                * nn_init_std
            )

            # Quantize the weight
            fp8_b_weight, fp8_b_scale = per_block_cast_to_fp8_e8m0(o_b_proj_weight_bf16)
            fp8_b_weight_dequant = weight_dequant(fp8_b_weight, fp8_b_scale).bfloat16()
            mla.o_b_proj.weight.data = fp8_b_weight
            mla.o_b_proj.weight_scale.data = fp8_b_scale

    # Generate test inputs
    # Note: for deepseek_v4, kv_lora_rank equals qk_head_dim
    attn_out_latent = torch.randn(num_tokens, num_heads, qk_head_dim, dtype=dtype, device=device)
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device=device)

    # Reference weights: the FP8-native path consumes the quantized o_a_proj plus
    # block scales (dequantized here), not the original BF16 weight.
    if dtype_str == "bf16":
        o_a_proj_ref = mla.o_a_proj.data
        o_b_proj_weight_ref = mla.o_b_proj.weight.data
    else:
        o_a_proj_ref = (
            weight_dequant(
                fp8_a_weight.reshape(-1, dim).contiguous(),
                fp8_a_scale.contiguous(),
            )
            .bfloat16()
            .reshape(o_a_proj_bf16.shape)
        )
        o_b_proj_weight_ref = fp8_b_weight_dequant

    freqs_cis = precompute_freqs_cis(
        qk_rope_head_dim,
        num_tokens,
        max_position_embeddings,
        rope_config.rope_theta,
        rope_config.rope_scaling["factor"],
        rope_config.rope_scaling["beta_fast"],
        rope_config.rope_scaling["beta_slow"],
    ).to(device)

    refs = SimpleNamespace(
        o_a_proj=o_a_proj_ref,
        o_b_proj_weight=o_b_proj_weight_ref,
        freqs_cis=freqs_cis,
        n_local_groups=n_local_groups,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
    )
    return mla, attn_out_latent, position_ids, refs


@skip_pre_blackwell
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("num_tokens", [1, 16, 128])
@pytest.mark.parametrize("dtype_str", ["bf16", "fp8"])
def test_deepseek_v4_o_proj(num_tokens: int, dtype_str: str):
    """Test DeepSeek-V4 output projection (_deepseek_v4_o_proj)."""
    print(
        f"\n{'=' * 80}\nTesting: deepseek_v4_o_proj num_tokens={num_tokens} dtype={dtype_str}\n{'=' * 80}"
    )

    if dtype_str == "fp8" and get_sm_version() < 100:
        pytest.skip("FP8 is not supported on pre-Blackwell architectures")

    device = torch.device("cuda")

    mla, attn_out_latent, position_ids, refs = _build_dsv4_o_proj_case(
        num_tokens, dtype_str, device
    )

    # Call the deepseek_v4 output projection (mla_rope_inplace modifies attn_out_latent
    # in-place, so clone before passing to preserve original for reference)
    output = mla._deepseek_v4_o_proj(attn_out_latent.clone(), position_ids)

    reference_output = calculate_reference_deepseek_v4_o_proj(
        attn_out_latent=attn_out_latent,
        o_a_proj=refs.o_a_proj,
        o_b_proj_weight=refs.o_b_proj_weight,
        freqs_cis=refs.freqs_cis[0:num_tokens],
        n_local_groups=refs.n_local_groups,
        qk_nope_head_dim=refs.qk_nope_head_dim,
        qk_rope_head_dim=refs.qk_rope_head_dim,
        device=device,
        is_fp8=dtype_str == "fp8",
    )

    # Validate output shapes
    assert output.shape == reference_output.shape, (
        f"Shape mismatch: output {output.shape} vs reference {reference_output.shape}"
    )
    assert output.dtype == reference_output.dtype, (
        f"Dtype mismatch: output {output.dtype} vs reference {reference_output.dtype}"
    )
    assert torch.isfinite(output).all(), "Output contains non-finite values"
    assert torch.isfinite(reference_output).all(), "Reference output contains non-finite values"

    # Compare results
    abs_error = (output - reference_output).abs()
    max_error = abs_error.max().item()
    mean_error = abs_error.mean().item()

    print(f"  Max error: {max_error:.6f}")
    print(f"  Mean error: {mean_error:.6f}")

    if dtype_str == "fp8":
        diff = _calc_diff(output, reference_output)
        assert diff < FP8_O_PROJ_DIFF_TOL, f"{diff=}"
    else:
        torch.testing.assert_close(output, reference_output, rtol=0.1, atol=0.1)
        print(f"  ✓ Test passed for num_tokens={num_tokens}, dtype={dtype_str}\n")


@skip_pre_blackwell
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("num_tokens", [1, 16, 128, 256])
def test_deepseek_v4_o_proj_fused_fp8_equivalence(num_tokens: int, monkeypatch):
    """The opt-in DSV4_FUSE_OPROJ fused fp8 epilogue must be numerically
    equivalent to the default unfused path.

    Fused (DSV4_FUSE_OPROJ=1): o_a's CuTe-DSL GEMM emits o_lora directly as fp8
    e4m3 + packed-UE8M0 1x128 scale factors, fed straight to DeepGEMM.
    Unfused (default): o_a emits bf16 o_lora, then ``o_b_proj`` runs the separate
    1x128 quant + DeepGEMM. Since the fusion only folds the *same* quant into o_a's
    epilogue, the two production paths must match far tighter than the fp8-vs-bf16
    reference bar.
    """
    if get_sm_version() < 100:
        pytest.skip("DSV4_FUSE_OPROJ fp8 fusion requires Blackwell (SM100+)")

    device = torch.device("cuda")
    mla, attn_out_latent, position_ids, refs = _build_dsv4_o_proj_case(num_tokens, "fp8", device)

    # Guard: confirm the fused branch's static gates hold, so this test actually
    # exercises the fused path instead of silently falling back to the unfused one.
    assert mla.o_a_proj.dtype == torch.float8_e4m3fn
    assert mla.n_local_groups == mla.num_groups
    assert getattr(mla.o_b_proj, "tp_size", 1) == 1
    assert mla.o_b_proj.has_fp8_block_scales
    assert not getattr(mla.o_b_proj, "use_cute_dsl_blockscaling_mm", False)

    # Unfused (default): DSV4_FUSE_OPROJ unset.
    monkeypatch.delenv("DSV4_FUSE_OPROJ", raising=False)
    out_unfused = mla._deepseek_v4_o_proj(attn_out_latent.clone(), position_ids)

    # Fused: opt in.
    monkeypatch.setenv("DSV4_FUSE_OPROJ", "1")
    out_fused = mla._deepseek_v4_o_proj(attn_out_latent.clone(), position_ids)

    assert out_fused.shape == out_unfused.shape, (
        f"Shape mismatch: fused {out_fused.shape} vs unfused {out_unfused.shape}"
    )
    assert out_fused.dtype == out_unfused.dtype, (
        f"Dtype mismatch: fused {out_fused.dtype} vs unfused {out_unfused.dtype}"
    )
    assert torch.isfinite(out_fused).all(), "Fused output contains non-finite values"

    # Analytic reference (same correctness bar as the unfused correctness test).
    reference_output = calculate_reference_deepseek_v4_o_proj(
        attn_out_latent=attn_out_latent,
        o_a_proj=refs.o_a_proj,
        o_b_proj_weight=refs.o_b_proj_weight,
        freqs_cis=refs.freqs_cis[0:num_tokens],
        n_local_groups=refs.n_local_groups,
        qk_nope_head_dim=refs.qk_nope_head_dim,
        qk_rope_head_dim=refs.qk_rope_head_dim,
        device=device,
        is_fp8=True,
    )

    diff_vs_ref = _calc_diff(out_fused, reference_output)
    diff_vs_unfused = _calc_diff(out_fused, out_unfused)
    print(
        f"\n  num_tokens={num_tokens}  diff(fused,ref)={diff_vs_ref:.3e}  "
        f"diff(fused,unfused)={diff_vs_unfused:.3e}"
    )

    assert diff_vs_ref < FP8_O_PROJ_DIFF_TOL, f"fused vs reference {diff_vs_ref=}"
    assert diff_vs_unfused < 1e-3, f"fused vs unfused {diff_vs_unfused=}"
