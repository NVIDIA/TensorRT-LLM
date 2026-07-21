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

import cutlass
import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.attention_backend.cute_dsl.fmha import (
    _COMPILE_CACHE,
    _quantize_blockscaled_one,
    _quantize_fp8_v,
    clear_cute_dsl_fmha_cache,
    cute_dsl_fmha_fwd,
)


def _require_supported_gpu_arch() -> str:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for CuTe DSL FMHA kernels.")

    compute_capability = torch.cuda.get_device_capability()
    gpu_arch = f"sm_{compute_capability[0]}{compute_capability[1]}a"
    if gpu_arch not in ("sm_100a", "sm_103a"):
        pytest.skip("CuTe DSL FMHA smoke tests require a supported Blackwell-class GPU.")

    return gpu_arch


def test_cute_dsl_jit_compile_smoke() -> None:
    """Compile (or fetch from cache) the kernel once and verify the cache grew."""
    _require_supported_gpu_arch()
    device = torch.device("cuda:0")
    batch_size, seq_len, num_heads, head_dim = 1, 128, 1, 128
    sm_scale = head_dim**-0.5

    torch.manual_seed(0)
    q = (
        torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=device)
        * 0.4
    )
    k = torch.randn_like(q) * 0.4
    v = torch.randn_like(q) * 0.4
    out = torch.empty_like(q)

    clear_cute_dsl_fmha_cache()
    cute_dsl_fmha_fwd(q, k, v, out, is_causal=False, sm_scale=sm_scale)
    torch.cuda.synchronize()
    assert len(_COMPILE_CACHE) == 1


def _make_tensor(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    ref_dtype: torch.dtype,
    scale: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a (B, S, H, D) tensor in `dtype` plus its dequantized reference."""
    tensor_f32 = (
        torch.randn(
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            dtype=torch.float32,
            device=device,
        )
        * 0.4
    )
    if dtype == torch.float8_e4m3fn:
        tensor = (tensor_f32 / scale).to(dtype)
        ref = (tensor.float() * scale).to(ref_dtype)
    else:
        tensor = tensor_f32.to(dtype)
        ref = tensor.to(ref_dtype)
    return tensor, ref


def _sdpa_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    """Reference: SDPA on (B, S, H, D) tensors, with GQA broadcast if needed."""
    q_bhsd = q.transpose(1, 2)
    k_bhsd = k.transpose(1, 2)
    v_bhsd = v.transpose(1, 2)
    if q_bhsd.shape[1] != k_bhsd.shape[1]:
        repeat = q_bhsd.shape[1] // k_bhsd.shape[1]
        k_bhsd = k_bhsd.repeat_interleave(repeat, dim=1)
        v_bhsd = v_bhsd.repeat_interleave(repeat, dim=1)
    out = F.scaled_dot_product_attention(
        q_bhsd, k_bhsd, v_bhsd, is_causal=is_causal, scale=sm_scale
    )
    return out.transpose(1, 2)


@pytest.mark.parametrize(
    ("batch_size", "seq_len_q", "seq_len_kv", "is_causal"),
    [
        pytest.param(1, 256, 256, False, id="b1_s256_nocausal"),
        pytest.param(1, 512, 512, True, id="b1_s512_causal"),
        pytest.param(2, 256, 256, False, id="b2_s256_nocausal"),
    ],
)
@pytest.mark.parametrize(
    ("qk_dtype", "pv_dtype", "out_dtype", "atol", "rtol"),
    [
        pytest.param(
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.bfloat16,
            5e-2,
            5e-2,
            id="bf16_fp8_bf16",
        ),
        pytest.param(
            torch.bfloat16,
            torch.bfloat16,
            torch.bfloat16,
            2e-2,
            2e-2,
            id="bf16",
        ),
    ],
)
@pytest.mark.parametrize(
    ("num_heads", "num_heads_kv"),
    [
        pytest.param(1, 1, id="mha_1h"),
        pytest.param(4, 4, id="mha_4h"),
        pytest.param(4, 2, id="gqa_4h_2kv"),
    ],
)
@pytest.mark.parametrize("head_dim", [128])
def test_cute_dsl_fmha_context_forward(
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    is_causal: bool,
    qk_dtype: torch.dtype,
    pv_dtype: torch.dtype,
    out_dtype: torch.dtype,
    num_heads: int,
    num_heads_kv: int,
    head_dim: int,
    atol: float,
    rtol: float,
) -> None:
    _require_supported_gpu_arch()
    device = torch.device("cuda:0")
    sm_scale = head_dim**-0.5
    scale_v = 0.06 if pv_dtype == torch.float8_e4m3fn else 1.0

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    q, q_ref = _make_tensor(
        batch_size, seq_len_q, num_heads, head_dim, qk_dtype, out_dtype, 1.0, device
    )
    k, k_ref = _make_tensor(
        batch_size, seq_len_kv, num_heads_kv, head_dim, qk_dtype, out_dtype, 1.0, device
    )
    v, v_ref = _make_tensor(
        batch_size, seq_len_kv, num_heads_kv, head_dim, pv_dtype, out_dtype, scale_v, device
    )
    out = torch.empty(batch_size, seq_len_q, num_heads, head_dim, dtype=out_dtype, device=device)
    lse = torch.empty(batch_size, seq_len_q, num_heads, dtype=torch.float32, device=device)

    cute_dsl_fmha_fwd(
        q,
        k,
        v,
        out,
        is_causal=is_causal,
        sm_scale=sm_scale,
        scale_v=scale_v,
        lse=lse,
    )
    torch.cuda.synchronize()

    out_ref = _sdpa_ref(q_ref, k_ref, v_ref, is_causal, sm_scale)
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    ("qk_sf_vec", "qk_cutlass_dtype_name", "atol", "rtol"),
    [
        pytest.param(32, "Float8E4M3FN", 8e-2, 8e-2, id="mxfp8"),
        pytest.param(16, "Float4E2M1FN", 2e-1, 2e-1, id="nvfp4"),
    ],
)
@pytest.mark.parametrize(
    ("batch_size", "seq_len_q", "seq_len_kv", "is_causal"),
    [
        pytest.param(1, 256, 256, False, id="b1_s256_nocausal"),
        pytest.param(1, 512, 512, True, id="b1_s512_causal"),
    ],
)
@pytest.mark.parametrize(
    ("num_heads", "num_heads_kv"),
    [
        pytest.param(4, 4, id="mha"),
        pytest.param(4, 2, id="gqa"),
    ],
)
@pytest.mark.parametrize("v_block_size", [0, 1])
def test_cute_dsl_fmha_blockscaled_forward(
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    v_block_size: int,
    is_causal: bool,
    qk_sf_vec: int,
    qk_cutlass_dtype_name: str,
    num_heads: int,
    num_heads_kv: int,
    atol: float,
    rtol: float,
) -> None:
    """End-to-end MXFP8 / NVFP4 block-scaled Q@K path through cute_dsl_fmha_fwd.

    Drives the block-scaled kernel with TRT-LLM-quantized Q/K and FP8 V using either
    one tensor scale or per-head-per-channel scales.
    """
    _require_supported_gpu_arch()

    device = torch.device("cuda:0")
    head_dim = 128  # kernel-imposed for block-scaled MXFP8 / NVFP4
    sm_scale = head_dim**-0.5

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    q_bf16 = (
        torch.randn(
            batch_size,
            seq_len_q,
            num_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.5
    )
    k_bf16 = (
        torch.randn(
            batch_size,
            seq_len_kv,
            num_heads_kv,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.5
    )
    v_bf16 = (
        torch.randn(
            batch_size,
            seq_len_kv,
            num_heads_kv,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.5
    )

    # Exercise both V modes: one tensor scale (0) and an (H, D) scale tensor (1).
    q_q, q_sf, scale_q = _quantize_blockscaled_one(q_bf16, qk_sf_vec)
    k_q, k_sf, scale_k = _quantize_blockscaled_one(k_bf16, qk_sf_vec)
    v_q, scale_v, scale_v_channels = _quantize_fp8_v(v_bf16, per_head_channel=v_block_size == 1)
    qk_cutlass_dtype = getattr(cutlass, qk_cutlass_dtype_name)

    out = torch.empty(
        batch_size,
        seq_len_q,
        num_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    lse = torch.empty(
        batch_size,
        seq_len_q,
        num_heads,
        dtype=torch.float32,
        device=device,
    )

    cute_dsl_fmha_fwd(
        q_q,
        k_q,
        v_q,
        out,
        is_causal=is_causal,
        scale_v=scale_v,
        scale_v_channels=scale_v_channels,
        sm_scale=sm_scale,
        lse=lse,
        scale_q=scale_q,
        scale_k=scale_k,
        qk_sf_vec=qk_sf_vec,
        q_sf=q_sf,
        k_sf=k_sf,
        qk_cutlass_dtype=qk_cutlass_dtype,
    )
    torch.cuda.synchronize()

    out_ref = _sdpa_ref(q_bf16, k_bf16, v_bf16, is_causal, sm_scale)
    assert torch.isfinite(out).all(), "Block-scaled FMHA produced NaN / Inf"
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)
