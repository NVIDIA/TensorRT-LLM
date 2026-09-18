# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test for DeepSeek-V4 output projection (_deepseek_v4_o_proj).
"""

from types import SimpleNamespace

import pytest
import torch
from _torch.helpers import per_block_cast_to_fp8_e8m0, per_token_cast_to_fp8_e8m0
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch import cute_dsl_utils
from tensorrt_llm._torch.attention.backends.interface import PositionalEmbeddingParams, RopeParams
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import module as dsv4
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.module import (
    project_sparse_attn_output,
)
from tensorrt_llm._torch.attention.mla import MLA
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv3 import weight_dequant
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
        use_cute_dsl_blockscaling_mm=dtype_str == "fp8",
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
    assert mla.mha is None
    assert not hasattr(mla, "kv_b_proj")
    assert not hasattr(mla, "v_b_proj")
    assert not hasattr(mla, "o_proj")

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
            # SM107 re-lays weight_scale to UE8M0 K32 at load time; mirror the loader.
            mla.o_b_proj.quant_method.post_load_weights(mla.o_b_proj)

    # Generate test inputs
    # Note: for deepseek_v4, kv_lora_rank equals qk_head_dim
    attn_out_latent = torch.randn(num_tokens, num_heads, qk_head_dim, dtype=dtype, device=device)
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device=device)

    # The non-fused MLA path stores attention output as a flattened 2D buffer.
    # mla_rope_inplace modifies it in place, so preserve the 3D reference input.
    output = project_sparse_attn_output(mla, [attn_out_latent.clone().flatten(1)], position_ids)

    # Calculate reference output
    if dtype_str == "bf16":
        o_a_proj_ref = mla.o_a_proj.data
        o_b_proj_weight_ref = mla.o_b_proj.weight.data
    else:
        # Match the FP8-native o_a_proj path: the runtime BMM consumes
        # quantized o_a_proj plus block scales, not the original BF16 weight.
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
        assert diff < FP8_O_PROJ_DIFF_TOL, f"{diff=}"
    else:
        torch.testing.assert_close(output, reference_output, rtol=0.1, atol=0.1)
        print(f"  ✓ Test passed for num_tokens={num_tokens}, dtype={dtype_str}\n")


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "sm,dsl,rubin,expected",
    [
        (100, True, False, "blackwell"),
        (103, True, True, "blackwell"),
        (107, True, True, "rubin"),
        (107, True, False, None),
        (107, False, True, None),
        (100, False, False, None),
        (90, True, True, None),
        (120, True, True, None),
    ],
)
def test_dsv4_q_b_dispatch(
    monkeypatch: pytest.MonkeyPatch, sm: int, dsl: bool, rubin: bool, expected: str | None
) -> None:
    """Select the correct GEMM independently of the host GPU and installed DSL."""
    torch.manual_seed(42)
    monkeypatch.setattr(dsv4, "get_sm_version", lambda: sm)
    monkeypatch.setattr(dsv4, "IS_CUTLASS_DSL_AVAILABLE", dsl)
    monkeypatch.setattr(dsv4, "IS_CUTLASS_DSL_RUBIN_AVAILABLE", rubin)
    # Exercise the contiguous conversion as well as dispatch.
    q = torch.randn(32, 7, dtype=torch.bfloat16).t()
    weight = torch.randn(32, 16, dtype=torch.bfloat16).t()
    calls = []

    def gemm(name: str, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
        assert a.is_contiguous() and b.is_contiguous()
        calls.append(name)
        out.copy_(torch.nn.functional.linear(a, b))

    for arch in ("blackwell", "rubin"):
        monkeypatch.setattr(
            torch.ops.trtllm,
            f"cute_dsl_bf16_gemm_{arch}",
            lambda a, b, out, name=arch: gemm(name, a, b, out),
            raising=False,
        )
    output = dsv4._q_b_proj_cute_dsl_bf16(q, weight)
    torch.testing.assert_close(output, torch.nn.functional.linear(q, weight))
    assert calls == ([] if expected is None else [expected])


def _output_projection(
    device: str, enabled: bool, rank: int = 16, dim: int = 32
) -> SimpleNamespace:
    return SimpleNamespace(
        num_heads_tp=2,
        n_local_groups=2,
        qk_nope_head_dim=dim - 8,
        qk_rope_head_dim=8,
        o_lora_rank=rank,
        o_a_proj=torch.randn(2, rank, dim, dtype=torch.bfloat16, device=device),
        o_b_proj=torch.nn.Identity(),
        use_cute_dsl_bf16_bmm=enabled,
        inverse_rotary_emb=SimpleNamespace(rotary_cos_sin=None, is_neox=False),
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "sm,enabled,rubin,rank,dim,expected",
    [
        (107, True, True, 16, 32, "rubin"),
        (107, False, True, 16, 32, "bmm"),
        (107, True, False, 16, 32, "bmm"),
        (107, True, True, 15, 32, "bmm"),
        (107, True, True, 16, 30, "bmm"),
        (100, True, True, 16, 32, "bmm"),
        (90, True, True, 16, 32, "bmm"),
    ],
)
def test_dsv4_o_a_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    sm: int,
    enabled: bool,
    rubin: bool,
    rank: int,
    dim: int,
    expected: str,
) -> None:
    """Check each Rubin BMM gate and preserve the transposed output layout."""
    torch.manual_seed(42)
    monkeypatch.setattr(dsv4, "get_sm_version", lambda: sm)
    monkeypatch.setattr(dsv4, "IS_CUTLASS_DSL_RUBIN_AVAILABLE", rubin)
    monkeypatch.setattr(torch.ops.trtllm, "mla_rope_inplace", lambda *args: None)
    calls = []

    def bmm(name: str, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
        calls.append(name)
        assert not out.is_contiguous()
        out.copy_(torch.bmm(a, b))

    monkeypatch.setattr(torch.ops.trtllm, "bmm_out", lambda a, b, out: bmm("bmm", a, b, out))
    monkeypatch.setattr(
        torch.ops.trtllm,
        "cute_dsl_bf16_bmm_rubin",
        lambda a, b, out: bmm("rubin", a, b.transpose(1, 2), out),
        raising=False,
    )
    model = _output_projection("cpu", enabled, rank, dim)
    attn = torch.randn(7, 2 * dim, dtype=torch.bfloat16)
    output = dsv4.project_sparse_attn_output(model, [attn], torch.arange(7))
    reference = (
        torch.bmm(attn.view(7, 2, dim).transpose(0, 1), model.o_a_proj.transpose(1, 2))
        .transpose(0, 1)
        .flatten(1)
    )
    torch.testing.assert_close(output, reference)
    assert calls == [expected]


@pytest.mark.parametrize("num_tokens", [1, 17, 256])
@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_dsv4_rubin_bf16_projections(
    monkeypatch: pytest.MonkeyPatch, num_tokens: int, use_cuda_graph: bool
) -> None:
    """Compare real Rubin projections with linear/BMM in eager and CUDA graphs."""
    torch.manual_seed(42)
    if not torch.cuda.is_available() or get_sm_version() != 107:
        pytest.skip("requires SM107")
    if not cute_dsl_utils.IS_CUTLASS_DSL_RUBIN_AVAILABLE:
        pytest.skip("requires Rubin CuTe DSL")
    # Isolate the projections; RoPE numerics have separate DSV4 output-projection coverage.
    monkeypatch.setattr(torch.ops.trtllm, "mla_rope_inplace", lambda *args: None)
    model = _output_projection("cuda", True)
    q = torch.randn(num_tokens, 32, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(64, 32, dtype=torch.bfloat16, device="cuda")
    attn = torch.randn(num_tokens, 64, dtype=torch.bfloat16, device="cuda")
    positions = torch.arange(num_tokens, device="cuda")

    def run() -> tuple[torch.Tensor, torch.Tensor]:
        return (
            dsv4._q_b_proj_cute_dsl_bf16(q, weight),
            dsv4.project_sparse_attn_output(model, [attn], positions),
        )

    if use_cuda_graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            q_out, o_out = run()
        graph.replay()
    else:
        q_out, o_out = run()
    torch.cuda.synchronize()
    torch.testing.assert_close(q_out, torch.nn.functional.linear(q, weight), atol=0.0625, rtol=0.01)
    model.use_cute_dsl_bf16_bmm = False
    reference_o = dsv4.project_sparse_attn_output(model, [attn], positions)
    torch.testing.assert_close(o_out, reference_o, atol=0.0625, rtol=0.01)
