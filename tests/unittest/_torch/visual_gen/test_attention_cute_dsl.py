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

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.attention import cute_dsl_fmha_fwd
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.attention.fmha import (
    get_cute_dsl_fmha_cubin,
)


def test_cute_dsl_cubin_kernel_can_import_and_load() -> None:
    gpu_arch = _require_supported_gpu_arch()
    try:
        kernel = get_cute_dsl_fmha_cubin(
            torch.bfloat16,
            torch.bfloat16,
            torch.bfloat16,
            128,
            is_causal=False,
            is_persistent=False,
            varlen=True,
            enable_tvm_ffi=True,
            gpu_arch=gpu_arch,
        )
    except ImportError as exc:
        pytest.skip(str(exc))

    assert kernel is not None


def _require_supported_gpu_arch() -> str:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for CuTe DSL FMHA kernels.")

    compute_capability = torch.cuda.get_device_capability()
    gpu_arch = f"sm_{compute_capability[0]}{compute_capability[1]}a"
    if gpu_arch not in ("sm_100a", "sm_103a"):
        pytest.skip("CuTe DSL FMHA smoke tests require a supported Blackwell-class GPU.")

    return gpu_arch


def _make_indptr(lens: list[int], device: torch.device) -> torch.Tensor:
    lens_tensor = torch.tensor(lens, dtype=torch.int32, device=device)
    return torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            lens_tensor.cumsum(0).int(),
        ]
    )


def _make_tensor(
    max_len: int,
    total_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    ref_dtype: torch.dtype,
    scale: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    tensor_f32 = torch.randn(
        max_len + total_len,
        num_heads,
        head_dim,
        dtype=torch.float32,
        device=device,
    )
    tensor_f32 *= 0.1
    if dtype == torch.float8_e4m3fn:
        tensor = (tensor_f32 / scale).to(dtype)
        ref = (tensor.float() * scale).to(ref_dtype)
    else:
        tensor = tensor_f32.to(dtype)
        ref = tensor.to(ref_dtype)
    return tensor[max_len:], ref[max_len:]


def _sdpa_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    is_causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    out = []
    for batch_idx in range(qo_indptr.numel() - 1):
        q_start = int(qo_indptr[batch_idx].item())
        q_end = int(qo_indptr[batch_idx + 1].item())
        kv_start = int(kv_indptr[batch_idx].item())
        kv_end = int(kv_indptr[batch_idx + 1].item())

        q_i = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
        k_i = k[kv_start:kv_end].transpose(0, 1).unsqueeze(0)
        v_i = v[kv_start:kv_end].transpose(0, 1).unsqueeze(0)
        if q_i.shape[1] != k_i.shape[1]:
            repeat_factor = q_i.shape[1] // k_i.shape[1]
            k_i = k_i.repeat_interleave(repeat_factor, dim=1)
            v_i = v_i.repeat_interleave(repeat_factor, dim=1)
        out_i = F.scaled_dot_product_attention(
            q_i,
            k_i,
            v_i,
            is_causal=is_causal,
            scale=sm_scale,
        )
        out.append(out_i.squeeze(0).transpose(0, 1))
    return torch.cat(out, dim=0)


@pytest.mark.parametrize(
    ("q_lens", "kv_lens", "is_causal"),
    [
        pytest.param([256], [256], False, id="single_nocausal"),
        pytest.param([512], [512], True, id="single_causal"),
        pytest.param([64, 128], [128, 512], False, id="varlen_nocausal"),
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
def test_cute_dsl_fmha_context_forward_cubin_smoke(
    q_lens: list[int],
    kv_lens: list[int],
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
    gpu_arch = _require_supported_gpu_arch()
    device = torch.device("cuda:0")
    sm_scale = head_dim**-0.5
    scale_v = 0.06 if pv_dtype == torch.float8_e4m3fn else 1.0

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    qo_indptr = _make_indptr(q_lens, device)
    kv_indptr = _make_indptr(kv_lens, device)
    total_q = int(qo_indptr[-1].item())
    total_kv = int(kv_indptr[-1].item())
    max_qo_len = max(q_lens)
    max_kv_len = max(kv_lens)

    q, q_ref = _make_tensor(
        max_qo_len, total_q, num_heads, head_dim, qk_dtype, out_dtype, 1.0, device
    )
    k, k_ref = _make_tensor(
        max_kv_len, total_kv, num_heads_kv, head_dim, qk_dtype, out_dtype, 1.0, device
    )
    v, v_ref = _make_tensor(
        max_kv_len, total_kv, num_heads_kv, head_dim, pv_dtype, out_dtype, scale_v, device
    )
    out_storage = torch.empty(
        max_qo_len + total_q,
        num_heads,
        head_dim,
        dtype=out_dtype,
        device=device,
    )
    out = out_storage[max_qo_len:]
    kernel_fn = get_cute_dsl_fmha_cubin(
        qk_dtype,
        pv_dtype,
        out_dtype,
        head_dim,
        is_causal,
        is_persistent=False,
        varlen=True,
        enable_tvm_ffi=True,
        gpu_arch=gpu_arch,
    )

    cute_dsl_fmha_fwd(
        q,
        k,
        v,
        out,
        qo_indptr,
        kv_indptr,
        is_causal=is_causal,
        sm_scale=sm_scale,
        scale_v=scale_v,
        max_qo_len=max_qo_len,
        max_kv_len=max_kv_len,
        kernel_fn=kernel_fn,
    )
    torch.cuda.synchronize()

    out_ref = _sdpa_ref(
        q_ref,
        k_ref,
        v_ref,
        qo_indptr,
        kv_indptr,
        is_causal,
        sm_scale,
    )
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)
