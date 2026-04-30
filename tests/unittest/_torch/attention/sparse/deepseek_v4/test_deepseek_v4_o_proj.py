"""
Test for DeepSeek-V4 output projection (_deepseek_v4_o_proj).
"""

from types import SimpleNamespace

import pytest
import torch
from _torch.helpers import per_block_cast_to_fp8_e8m0
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

    # Apply o_a_proj: einsum equivalent to bmm
    o_lora = torch.einsum("tgd,grd->tgr", attn_out_grouped, o_a_proj)

    # Flatten and apply o_b_proj, [num_tokens, n_local_groups * o_lora_rank]
    o_lora_flat = o_lora.flatten(1)
    if is_fp8:
        o_lora_flat = o_lora_flat.to(torch.float8_e4m3fn).to(torch.bfloat16)
    output = torch.nn.functional.linear(o_lora_flat, o_b_proj_weight)  # [num_tokens, hidden_size]

    return output


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
            mla.o_a_proj_dequant.data = o_a_proj_bf16

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

    # Call the deepseek_v4 output projection (mla_rope_inplace modifies attn_out_latent
    # in-place, so clone before passing to preserve original for reference)
    output = mla._deepseek_v4_o_proj(attn_out_latent.clone(), position_ids)

    # Calculate reference output
    if dtype_str == "bf16":
        o_a_proj_ref = mla.o_a_proj.data
        o_b_proj_weight_ref = mla.o_b_proj.weight.data
    else:
        # For FP8, convert back to bf16 for reference calculation
        o_a_proj_ref = o_a_proj_bf16
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

    reference_output = calculate_reference_deepseek_v4_o_proj(
        attn_out_latent=attn_out_latent,
        o_a_proj=o_a_proj_ref,
        o_b_proj_weight=o_b_proj_weight_ref,
        freqs_cis=freqs_cis[0:num_tokens],
        n_local_groups=n_local_groups,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
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
        assert diff < 1e-3, f"{diff=}"
    else:
        torch.testing.assert_close(output, reference_output, rtol=0.1, atol=0.1)
        print(f"  ✓ Test passed for num_tokens={num_tokens}, dtype={dtype_str}\n")
