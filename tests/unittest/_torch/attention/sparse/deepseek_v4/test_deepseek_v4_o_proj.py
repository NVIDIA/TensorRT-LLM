"""
Test for DeepSeek-V4 output projection (_deepseek_v4_o_proj).
"""

from types import SimpleNamespace

import pytest
import torch
from _torch.helpers import per_block_cast_to_fp8_e8m0, per_token_cast_to_fp8_e8m0
from utils.util import skip_pre_blackwell

import tensorrt_llm._torch.modules.attention as attention_module
from tensorrt_llm._torch.attention_backend.interface import PositionalEmbeddingParams, RopeParams
from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv3 import weight_dequant
from tensorrt_llm._torch.modules.attention import MLA, _select_dsv4_ob_split_k
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from ..test_sparse_mla_forward import RopeConfig, _calc_diff, apply_rotary_emb, precompute_freqs_cis

FP8_O_PROJ_DIFF_TOL = 2e-3


@pytest.mark.parametrize(
    ("num_tokens", "expected_split"),
    [(1, 2), (16, 2), (32, 2), (64, 2), (96, 2), (128, 2), (160, 1), (16384, 1)],
)
def test_select_dsv4_ob_split_k_auto_policy(num_tokens, expected_split, monkeypatch):
    monkeypatch.delenv("TRTLLM_DSV4_OB_SPLIT_K", raising=False)
    assert _select_dsv4_ob_split_k(num_tokens) == expected_split


def test_select_dsv4_ob_split_k_overrides(monkeypatch):
    monkeypatch.setenv("TRTLLM_DSV4_OB_SPLIT_K", "4")
    assert _select_dsv4_ob_split_k(32) == 4
    assert _select_dsv4_ob_split_k(32, configured_split=2) == 2
    with pytest.raises(ValueError, match="unsupported.*split-K factor"):
        _select_dsv4_ob_split_k(32, configured_split=3)


def test_dsv4_fmha_epilogue_output_uses_fused_oproj():
    attn_fp8 = torch.empty((16, 4, 4096), device="meta", dtype=torch.float8_e4m3fn)
    attn_scale = torch.empty((16, 32, 4), device="meta")
    expected = torch.empty((8, 7168), device="meta", dtype=torch.bfloat16)
    calls = []
    module = SimpleNamespace(
        _should_use_fused_oproj=lambda: True,
        _fused_oa_ob_proj=lambda fp8, scale, num_tokens: calls.append((fp8, scale, num_tokens))
        or expected,
    )

    output = MLA._deepseek_v4_o_proj(module, (attn_fp8, attn_scale))

    assert output is expected
    assert calls == [(attn_fp8, attn_scale, 4)]


@pytest.mark.parametrize(
    ("num_tokens", "num_splits", "expected"),
    [
        (1, 2, ((128, 32), (2, 1), True, None, True)),
        (32, 2, ((128, 32), (2, 1), True, None, True)),
        (64, 2, ((256, 64), (2, 1), True, 8, True)),
        (96, 2, ((256, 128), (2, 1), True, None, True)),
        (128, 2, ((256, 128), (2, 1), True, None, True)),
        (256, 1, ((256, 128), (2, 1), True, None, True)),
        (448, 1, ((256, 208), (2, 1), False, None, True)),
        (512, 1, ((256, 208), (2, 1), False, None, True)),
        (704, 1, ((256, 144), (2, 1), True, None, True)),
        (2240, 1, ((256, 224), (2, 1), True, None, True)),
        (2304, 1, ((256, 224), (2, 1), False, None, False)),
        (3776, 1, ((256, 224), (2, 1), True, None, True)),
        (4096, 1, ((256, 224), (2, 1), False, None, True)),
        (5888, 1, ((256, 224), (2, 1), False, None, True)),
        (9472, 1, ((256, 224), (2, 1), False, None, True)),
        (16384, 1, ((256, 240), (2, 1), True, None, True)),
    ],
)
def test_dsv4_ob_cute_tactic(num_tokens, num_splits, expected):
    runner = cute_dsl_custom_ops.CuteDSLFp8SplitKGemmRunner
    assert runner._get_tactic(num_tokens, 7168, num_splits, 74) == expected


def test_dsv4_ob_split_k_one_uses_cute_dsl_and_caches_weight_scale(monkeypatch):
    from tensorrt_llm import deep_gemm
    from tensorrt_llm.quantization.utils import fp8_utils

    m, n, k = 256, 7168, 16384
    packed_k = k // 512
    weight_scale_storage = torch.empty((packed_k, n), device="meta", dtype=torch.int32)
    transformed_scale = torch.as_strided(weight_scale_storage, (n, packed_k), (1, n))
    module = SimpleNamespace(
        hidden_size=n,
        dtype=torch.bfloat16,
        ob_split_k=1,
        o_b_proj=SimpleNamespace(
            weight=torch.empty((n, k), device="meta", dtype=torch.float8_e4m3fn),
            weight_scale=torch.empty((n // 128, k // 128), device="meta"),
        ),
    )
    activation = torch.empty((m, k), device="meta", dtype=torch.float8_e4m3fn)
    activation_scale = torch.empty_strided((m, packed_k), (1, m), device="meta", dtype=torch.int32)
    calls = []
    transforms = []

    def fail_deep_gemm(*args, **kwargs):
        raise AssertionError("supported SK1 must not dispatch to DeepGEMM")

    def splitk_gemm(a, sfa, b, sfb, partials, num_splits):
        calls.append((a, sfa, b, sfb, partials, num_splits))

    def transform_weight_scale(scale, **kwargs):
        transforms.append((scale, kwargs))
        return transformed_scale

    monkeypatch.setattr(attention_module, "IS_CUTLASS_DSL_AVAILABLE", True)
    monkeypatch.setattr(deep_gemm, "fp8_gemm_nt", fail_deep_gemm)
    monkeypatch.setattr(fp8_utils, "transform_sf_into_required_layout", transform_weight_scale)
    monkeypatch.setattr(torch.ops.trtllm, "dsv4_fp8_splitk_gemm", splitk_gemm)
    output = MLA._fused_ob_gemm(module, activation, activation_scale, m)
    cached_output = MLA._fused_ob_gemm(module, activation, activation_scale, m)

    assert output.shape == (m, n)
    assert cached_output.shape == (m, n)
    assert output.device.type == "meta"
    assert len(transforms) == 1
    assert transforms[0][0] is module.o_b_proj.weight_scale
    assert module.o_b_proj._ob_wsf_int is transformed_scale
    assert len(calls) == 2
    assert calls[0][0] is activation
    assert calls[0][1] is activation_scale
    assert calls[0][2] is module.o_b_proj.weight
    assert calls[0][3] is transformed_scale
    assert calls[0][4].shape == (1, m, n)
    assert calls[0][5] == 1


@pytest.mark.parametrize("m", [1, 32, 64, 128])
def test_dsv4_ob_auto_split_through_m128_uses_cute_dsl(m, monkeypatch):
    from tensorrt_llm import deep_gemm

    n, k = 7168, 16384
    packed_k = k // 512
    weight_scale_storage = torch.empty((packed_k, n), device="meta", dtype=torch.int32)
    module = SimpleNamespace(
        hidden_size=n,
        dtype=torch.bfloat16,
        ob_split_k=None,
        o_b_proj=SimpleNamespace(
            weight=torch.empty((n, k), device="meta", dtype=torch.float8_e4m3fn),
            weight_scale=torch.as_strided(weight_scale_storage, (n, packed_k), (1, n)),
        ),
    )
    activation = torch.empty((m, k), device="meta", dtype=torch.float8_e4m3fn)
    aligned_m = (m + 3) // 4 * 4
    activation_scale = torch.empty_strided(
        (m, packed_k), (1, aligned_m), device="meta", dtype=torch.int32
    )
    calls = []

    def fail_deep_gemm(*args, **kwargs):
        raise AssertionError("automatic M<=128 SK2 must not dispatch to DeepGEMM")

    def splitk_gemm(a, sfa, b, sfb, partials, num_splits):
        calls.append((a, sfa, b, sfb, partials, num_splits))

    monkeypatch.setattr(attention_module, "IS_CUTLASS_DSL_AVAILABLE", True)
    monkeypatch.setattr(deep_gemm, "fp8_gemm_nt", fail_deep_gemm)
    monkeypatch.setattr(torch.ops.trtllm, "dsv4_fp8_splitk_gemm", splitk_gemm)
    output = MLA._fused_ob_gemm(module, activation, activation_scale, m)

    assert output.shape == (2 * m, n)
    assert len(calls) == 1
    assert calls[0][4].shape == (2, m, n)
    assert calls[0][5] == 2


@pytest.mark.parametrize("m", [160])
def test_dsv4_ob_unsplit_mid_m_uses_deep_gemm(m, monkeypatch):
    from tensorrt_llm import deep_gemm
    from tensorrt_llm.quantization.utils import fp8_utils

    n, k = 7168, 16384
    packed_k = k // 512
    module = SimpleNamespace(
        hidden_size=n,
        dtype=torch.bfloat16,
        ob_split_k=None,
        o_b_proj=SimpleNamespace(
            weight=torch.empty((n, k), device="meta", dtype=torch.float8_e4m3fn),
            weight_scale=torch.empty((n // 128, k // 128), device="meta"),
        ),
    )
    activation = torch.empty((m, k), device="meta", dtype=torch.float8_e4m3fn)
    activation_scale = torch.empty_strided((m, packed_k), (1, m), device="meta", dtype=torch.int32)
    calls = []

    def deep_gemm_nt(inputs, weights, output, **kwargs):
        calls.append((inputs, weights, output, kwargs))

    def fail_cute(*args, **kwargs):
        raise AssertionError("default unsplit M below 256 must dispatch to DeepGEMM")

    def fail_transform(*args, **kwargs):
        raise AssertionError("DeepGEMM fallback must not transform the weight scale")

    monkeypatch.setattr(attention_module, "IS_CUTLASS_DSL_AVAILABLE", True)
    monkeypatch.setattr(deep_gemm, "fp8_gemm_nt", deep_gemm_nt)
    monkeypatch.setattr(fp8_utils, "transform_sf_into_required_layout", fail_transform)
    monkeypatch.setattr(torch.ops.trtllm, "dsv4_fp8_splitk_gemm", fail_cute)
    output = MLA._fused_ob_gemm(module, activation, activation_scale, m)

    assert output.shape == (m, n)
    assert output.device.type == "meta"
    assert len(calls) == 1
    assert calls[0][0] == (activation, activation_scale)
    assert calls[0][1] == (module.o_b_proj.weight, module.o_b_proj.weight_scale)
    assert calls[0][2] is output


@skip_pre_blackwell
@pytest.mark.parametrize(
    ("num_tokens", "num_splits"),
    [(1, 2), (2, 2), (32, 2), (64, 2), (128, 2), (256, 1), (512, 1), (64, 4)],
)
def test_dsv4_pro_fp8_splitk_gemm_partials(num_tokens: int, num_splits: int):
    if get_sm_version() // 10 != 10:
        pytest.skip("dsv4_fp8_splitk_gemm requires the SM100 family")

    n, k = 7168, 16384
    aligned_m = (num_tokens + 3) // 4 * 4
    packed_k = k // 512
    a = torch.full((num_tokens, k), 0.03125, device="cuda", dtype=torch.float8_e4m3fn)
    b = torch.full((n, k), 0.03125, device="cuda", dtype=torch.float8_e4m3fn)

    # Pack four unit UE8M0 scales per int32.
    sfa_storage = torch.full((aligned_m * packed_k,), 0x7F7F7F7F, device="cuda", dtype=torch.int32)
    sfb_storage = torch.full((n * packed_k,), 0x7F7F7F7F, device="cuda", dtype=torch.int32)
    sfa = torch.as_strided(sfa_storage, (num_tokens, packed_k), (1, aligned_m))
    sfb = torch.as_strided(sfb_storage, (n, packed_k), (1, n))
    partials = torch.empty((num_splits, num_tokens, n), device="cuda", dtype=torch.bfloat16)

    torch.ops.trtllm.dsv4_fp8_splitk_gemm(a, sfa, b, sfb, partials, num_splits)
    expected_partial = (k // num_splits) * 0.03125 * 0.03125
    torch.testing.assert_close(
        partials.float(), torch.full_like(partials, expected_partial, dtype=torch.float32)
    )


@skip_pre_blackwell
def test_dsv4_pro_fp8_splitk_gemm_packed_scales():
    if get_sm_version() // 10 != 10:
        pytest.skip("dsv4_fp8_splitk_gemm requires the SM100 family")

    m, n, k, num_splits = 64, 128, 2048, 4
    k_blocks = k // 128
    torch.manual_seed(1234)
    a = (torch.randn((m, k), device="cuda") * 0.125).to(torch.float8_e4m3fn)
    b = (torch.randn((n, k), device="cuda") * 0.125).to(torch.float8_e4m3fn)
    exp_a = torch.randint(124, 131, (m, k_blocks), device="cuda")
    exp_b = torch.randint(124, 131, (n // 128, k_blocks), device="cuda").repeat_interleave(
        128, dim=0
    )

    def pack_scales(exponents: torch.Tensor, aligned_rows: int) -> torch.Tensor:
        rows, num_blocks = exponents.shape
        grouped = exponents.reshape(rows, num_blocks // 4, 4).to(torch.int32)
        packed = (
            grouped[..., 0]
            | (grouped[..., 1] << 8)
            | (grouped[..., 2] << 16)
            | (grouped[..., 3] << 24)
        )
        storage = torch.zeros((num_blocks // 4, aligned_rows), device="cuda", dtype=torch.int32)
        storage[:, :rows] = packed.transpose(0, 1)
        return torch.as_strided(storage, packed.shape, (1, aligned_rows))

    sfa = pack_scales(exp_a, m)
    sfb = pack_scales(exp_b, n)
    partials = torch.empty((num_splits, m, n), device="cuda", dtype=torch.bfloat16)
    torch.ops.trtllm.dsv4_fp8_splitk_gemm(a, sfa, b, sfb, partials, num_splits)

    scale_a = torch.exp2(exp_a.float() - 127.0)
    scale_b = torch.exp2(exp_b.float() - 127.0)
    a_dequant = (a.float().reshape(m, k_blocks, 128) * scale_a[..., None]).reshape(m, k)
    b_dequant = (b.float().reshape(n, k_blocks, 128) * scale_b[..., None]).reshape(n, k)
    split_k = k // num_splits
    expected = torch.stack(
        [
            a_dequant[:, split * split_k : (split + 1) * split_k]
            @ b_dequant[:, split * split_k : (split + 1) * split_k].T
            for split in range(num_splits)
        ]
    )
    torch.testing.assert_close(partials.float(), expected, rtol=0.02, atol=0.05)


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
    o_proj tests. Shared by the correctness test and the default fused
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
            # SM100 keeps only the quantized O_a weights.

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

    # Dequantize the weights actually consumed by the FP8 path.
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

    # Preserve the input because inverse RoPE is in-place.
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
@pytest.mark.parametrize("num_tokens", [1, 16, 32, 128, 256])
def test_deepseek_v4_o_proj_fused_fp8_equivalence(num_tokens: int, monkeypatch):
    """The default fused FP8 epilogue must match the disabled baseline.

    Fused (default): o_a's CuTe-DSL GEMM emits o_lora directly as fp8
    e4m3 + packed-UE8M0 1x128 scale factors, fed straight to o_b.
    Disabled: o_a emits bf16 o_lora, then ``o_b_proj`` runs the separate 1x128
    quant + DeepGEMM. Since the fusion only folds the *same* quant into o_a's
    epilogue, the two paths must match far tighter than the fp8-vs-bf16 reference.
    """
    if get_sm_version() < 100:
        pytest.skip("fused DeepSeek-V4 FP8 O-projection requires Blackwell (SM100+)")

    device = torch.device("cuda")
    mla, attn_out_latent, position_ids, refs = _build_dsv4_o_proj_case(num_tokens, "fp8", device)

    # Ensure the fused path is eligible.
    assert mla.o_a_proj.dtype == torch.float8_e4m3fn
    assert mla.n_local_groups == mla.num_groups
    assert getattr(mla.o_b_proj, "tp_size", 1) == 1
    assert mla.o_b_proj.has_fp8_block_scales
    assert not getattr(mla.o_b_proj, "use_cute_dsl_blockscaling_mm", False)

    # Explicit kill switch retains the unfused fallback.
    monkeypatch.setenv("TRTLLM_DSV4_DISABLE_FUSED_OPROJ", "1")
    out_unfused = mla._deepseek_v4_o_proj(attn_out_latent.clone(), position_ids)

    # Unset is the production default and must take the fused path.
    monkeypatch.delenv("TRTLLM_DSV4_DISABLE_FUSED_OPROJ", raising=False)
    mla.ob_split_k = 1
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
        attn_out_latent=attn_out_latent.clone(),
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

    expected_smem_epilogue = num_tokens <= 32
    expected_smem_row_iters = (num_tokens + 15) // 16 if expected_smem_epilogue else 1
    fp8out_keys = [
        key
        for key in cute_dsl_custom_ops.CuteDSLFp8BlackwellBmmRunner.kernel_cache
        if key[0] == "fp8out"
        and key[-2] == expected_smem_row_iters
        and key[-1] == expected_smem_epilogue
    ]
    assert len(fp8out_keys) == 1

    if num_tokens == 16:
        compiled_gemm = cute_dsl_custom_ops.CuteDSLFp8BlackwellBmmRunner.kernel_cache[
            fp8out_keys[0]
        ]
        out_cached = mla._deepseek_v4_o_proj(attn_out_latent.clone(), position_ids)
        assert (
            cute_dsl_custom_ops.CuteDSLFp8BlackwellBmmRunner.kernel_cache[fp8out_keys[0]]
            is compiled_gemm
        )
        torch.testing.assert_close(out_cached, out_unfused, rtol=0, atol=0)
