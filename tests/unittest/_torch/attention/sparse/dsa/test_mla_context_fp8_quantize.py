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
"""Absorption-mode FP8 Q quantize of the sparse MLA context (``invokeMLAContextFp8Quantize``).

In absorption mode the kernel converts Q only (K and V stay in the paged cache), so its grid is
sized by the Q tokens (``acc_q_len``) and not by ``total_kv_len``. The test calls the launcher of
the installed ``libtensorrt_llm.so`` through a small JIT binding, for multi-request steps on deep
prefixes (up to 657k keys), 96 / 128 heads, both Q layouts (576 = 512 + 64, 512 = 448 + 64) and
FP8 / NVFP4 caches, and checks that

- the FP8 Q equals the reference conversion (BF16 x scale in FP32, saturating RNE to E4M3);
- it does not depend on ``total_kv_len`` (a deep step equals the same Q without a prefix);
- nothing is written past the last Q token;
- the BMM1 scale is the Q dequant scale times the host scale (also times log2(e)) and the
  BMM2 scale is 1.
"""

import hashlib
import struct
import tempfile
from pathlib import Path

import pytest
import torch


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "cpp" / "tensorrt_llm" / "kernels" / "mlaKernels.h").is_file():
            return p
    return here.parents[6]


_CPP = _repo_root() / "cpp"

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 9,
        reason="FP8 E4M3 needs sm89+",
    ),
]

_BINDING = r"""
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "tensorrt_llm/kernels/mlaKernels.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"

namespace tk = tensorrt_llm::kernels;

torch::Tensor mla_ctx_fp8_quantize(torch::Tensor q, torch::Tensor out,
    torch::Tensor quant_scale_qkv, torch::Tensor dequant_scale_q, int64_t total_kv_len,
    bool rope_append, bool nvfp4_cache, double host_bmm1_scale)
{
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kBFloat16 && q.is_contiguous()
            && q.dim() == 3,
        "q must be a contiguous [tokens, heads, dim] BF16 CUDA tensor");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kByte && out.is_contiguous()
            && out.numel() >= q.numel(),
        "out must be a contiguous uint8 CUDA tensor of at least q.numel() bytes");
    auto scales = torch::full({3}, -1.0, q.options().dtype(at::kFloat));
    tk::MlaParams<__nv_bfloat16> p{};
    p.q_buf = reinterpret_cast<__nv_bfloat16*>(q.data_ptr());
    p.quant_q_buf = out.data_ptr();
    p.acc_q_len = static_cast<int32_t>(q.size(0));
    p.head_num = static_cast<int32_t>(q.size(1));
    p.absorption_mode = true;
    p.cache_type = nvfp4_cache ? tk::KvCacheDataType::NVFP4 : tk::KvCacheDataType::FP8;
    p.meta.rope_append = rope_append ? 1 : 0;
    p.quant_scale_qkv = quant_scale_qkv.data_ptr<float>();
    p.dequant_scale_q = dequant_scale_q.data_ptr<float>();
    p.dequant_scale_kv = nullptr;
    p.quant_scale_o = nullptr;
    p.bmm1_scale = scales.data_ptr<float>();
    p.bmm2_scale = scales.data_ptr<float>() + 2;
    p.host_bmm1_scale = static_cast<float>(host_bmm1_scale);
    tk::invokeMLAContextFp8Quantize<__nv_bfloat16>(
        p, static_cast<int>(total_kv_len), c10::cuda::getCurrentCUDAStream().stream());
    return scales;
}
"""

_EXT = None


def _ext():
    """JIT binding to the launcher in the installed libtensorrt_llm.so (built once per process)."""
    global _EXT
    if _EXT is None:
        hdr = _CPP / "tensorrt_llm" / "kernels" / "mlaKernels.h"
        if not hdr.is_file():
            pytest.skip(f"no TRT-LLM C++ headers under {_CPP}")
        from torch.utils.cpp_extension import load_inline

        import tensorrt_llm  # noqa: F401  (loads libtensorrt_llm.so)

        libs = Path(tensorrt_llm.__file__).resolve().parent / "libs"
        _EXT = load_inline(
            name="mla_ctx_fp8_quantize_binding",
            cpp_sources=[_BINDING],
            functions=["mla_ctx_fp8_quantize"],
            extra_include_paths=[str(_CPP / "include"), str(_CPP)],
            extra_cflags=["-std=c++20", "-DENABLE_BF16", "-DENABLE_FP8", "-DENABLE_FP4"],
            extra_ldflags=[f"-L{libs}", "-ltensorrt_llm", f"-Wl,-rpath,{libs}"],
            with_cuda=True,
            build_directory=tempfile.mkdtemp(prefix="mla_ctx_fp8_quantize_"),
        )
    return _EXT


# Steps as (new tokens, cached tokens) per request: the Q rows are the new tokens of all
# requests, the keys their cached + new tokens. The launcher only sees the two totals.
STEPS = {
    "fresh_8k": [(8192, 0)],
    "2k_on_510k": [(2048, 522240)],
    "4x2k_on_126k": [(2048, 129024)] * 4,
    "mixed_4req": [(1, 400000), (4093, 0), (3000, 250000), (77, 7)],
}
HEAD_DIM = {True: 576, False: 512}  # rope_append: 512 + 64; else 448 + 64
QUANT_SCALE = 0.75
DEQUANT_SCALE_Q = 1.3125
LOG2E = 1.4426950408889634
GUARD_ROWS = 3
GUARD_BYTE = 0xA5


def _seed(*key) -> int:
    return int.from_bytes(hashlib.sha256(repr(key).encode()).digest()[:4], "little")


def _make_q(step, heads, rope_append):
    acc_q = sum(n for n, _ in STEPS[step])
    gen = torch.Generator(device="cuda").manual_seed(_seed(step, heads, rope_append))
    shape = (acc_q, heads, HEAD_DIM[rope_append])
    q = torch.randn(shape, device="cuda", generator=gen) * 3
    # Saturation: values past the E4M3 range at fixed places.
    q[0, 0, :4] = torch.tensor([1000.0, -3000.0, 448.0 / QUANT_SCALE, -600.0], device="cuda")
    return q.to(torch.bfloat16).contiguous()


def _reference(q):
    """What quantCopy computes: float(q) * scale, saturating RNE to E4M3."""
    x = q.float() * torch.tensor(QUANT_SCALE, dtype=torch.float32, device=q.device)
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)


def _sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def _f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _launch(q, total_kv_len, rope_append, nvfp4_cache, host_bmm1_scale):
    n = q.numel()
    guard = GUARD_ROWS * q.shape[1] * q.shape[2]
    out = torch.full((n + guard,), GUARD_BYTE, dtype=torch.uint8, device="cuda")
    qs = torch.tensor([QUANT_SCALE], dtype=torch.float32, device="cuda")
    dq = torch.tensor([DEQUANT_SCALE_Q], dtype=torch.float32, device="cuda")
    args = (q, out, qs, dq, total_kv_len, rope_append, nvfp4_cache, host_bmm1_scale)
    scales = _ext().mla_ctx_fp8_quantize(*args)
    torch.cuda.synchronize()
    return out, scales.cpu()


@pytest.mark.parametrize("nvfp4_cache", [True, False], ids=["nvfp4", "fp8"])
@pytest.mark.parametrize("rope_append", [True, False], ids=["d576", "d512"])
@pytest.mark.parametrize("heads", [96, 128])
@pytest.mark.parametrize("step", list(STEPS))
def test_absorption_fp8_quantize_covers_q_only(step, heads, rope_append, nvfp4_cache):
    q = _make_q(step, heads, rope_append)
    acc_q = q.shape[0]
    total_kv = sum(n + c for n, c in STEPS[step])
    host_bmm1_scale = 1.0 / (HEAD_DIM[rope_append] ** 0.5)
    ref = _reference(q).reshape(-1)
    n = q.numel()
    # dequant_q x dequant_kv (1, no KV scale) x host scale; then x log2(e); BMM2: 1 x 1.
    bmm1 = _f32(_f32(DEQUANT_SCALE_Q) * _f32(host_bmm1_scale))
    expect_scales = [bmm1, _f32(bmm1 * _f32(LOG2E)), 1.0]

    shas = set()
    for kv in sorted({acc_q, total_kv}):
        out, scales = _launch(q, kv, rope_append, nvfp4_cache, host_bmm1_scale)
        got = out[:n]
        # Values, so a signed zero does not count; the launches are compared byte for byte below.
        fp8 = torch.float8_e4m3fn
        bad = (got.view(fp8).float() != ref.view(fp8).float()).nonzero()
        assert bad.numel() == 0, (
            f"total_kv_len {kv}: {bad.numel()} FP8 values differ from the reference, "
            f"first at flat index {bad[0].item()}"
        )
        assert bool((out[n:] == GUARD_BYTE).all()), f"total_kv_len {kv}: written past the last Q"
        assert scales.tolist() == expect_scales, f"total_kv_len {kv}: scales {scales.tolist()}"
        shas.add(_sha(got))
    assert len(shas) == 1, "the FP8 Q depends on total_kv_len"
