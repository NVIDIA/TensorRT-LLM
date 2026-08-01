# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Quality test: fused RMSNorm+NVFP4-quantize (modules/fused_ops/
rmsnorm_fp4_quant, backed by flashinfer's CuTe-DSL ``rmsnorm_fp4quant``) vs
the unfused reference chain
(trtllm::flashinfer_rmsnorm -> trtllm::fp4_quantize).

The fused kernel quantizes the fp32 norm result directly, without the
intermediate bf16 round the unfused chain performs when materializing the
normed tensor, so its outputs are near- but not byte-identical (~1.4% of
payload nibbles / ~2% of block scales differ by one code step at serving
shapes). The tests therefore compare dequantized values, not bytes:

- the fused path's quantization error against the fp32 norm must match the
  unfused chain's (both are roundings of the same quantity);
- dequantization reads the scale factors at the swizzled offsets of
  get_sf_out_offset_128x4, so any layout or scale-convention bug shows up
  as a gross error, not a tolerance miss;
- a small byte-mismatch bound documents the near-parity property itself.
"""

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers trtllm torch ops)
import tensorrt_llm._torch.custom_ops.flashinfer_custom_ops  # noqa: F401
from tensorrt_llm._torch.modules.fused_ops.gelu_tanh_mul_fp4_quant import sf_swizzled_offsets
from tensorrt_llm._torch.modules.fused_ops.rmsnorm_fp4_quant import (
    rmsnorm_fp4_quant,
    rmsnorm_fp4_quant_available,
)

pytestmark = pytest.mark.skipif(
    not rmsnorm_fp4_quant_available(),
    reason="flashinfer CuTe-DSL rmsnorm_fp4quant unavailable",
)

_E2M1_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_E2M1_LUT = _E2M1_LUT + [-v for v in _E2M1_LUT]


def _reference(x, w, eps, gs):
    n = torch.ops.trtllm.flashinfer_rmsnorm(x, w, eps)
    q, sf = torch.ops.trtllm.fp4_quantize(n, gs, 16, False)
    return q.view(torch.uint8), sf.view(torch.uint8)


def _norm_fp32(x, w, eps):
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()


def _block_scales(sf, m, h):
    """Block scales [m, h//16] read at the swizzled offsets (as fp32)."""
    valid = sf_swizzled_offsets(m, h // 16, sf.device)
    return sf.view(-1)[valid].view(torch.float8_e4m3fn).float().view(m, h // 16)


def _dequant(q, sf, m, h, gs):
    """Dequantize payload + swizzled scales to fp32 [m, h]."""
    lut = torch.tensor(_E2M1_LUT, device=q.device)
    vals = torch.empty((m, h), device=q.device, dtype=torch.float32)
    vals[:, 0::2] = lut[(q & 0xF).long()]
    vals[:, 1::2] = lut[(q >> 4).long()]
    return vals * _block_scales(sf, m, h).repeat_interleave(16, dim=1) / gs


def _check(x, w, eps, gs):
    fq, fsf = rmsnorm_fp4_quant(x, w, eps, gs)
    rq, rsf = _reference(x.contiguous(), w, eps, gs)
    assert fq.shape == rq.shape and fsf.numel() == rsf.numel()
    assert fq.dtype == torch.uint8 and fsf.dtype == torch.uint8

    m, h = x.shape
    n32 = _norm_fp32(x.contiguous(), w, eps)
    err_fused = (_dequant(fq, fsf, m, h, gs) - n32).abs().mean().item()
    err_ref = (_dequant(rq, rsf, m, h, gs) - n32).abs().mean().item()
    # Both outputs are NVFP4 roundings of the same fp32 norm; the fused
    # path must not be a worse quantizer than the unfused chain (measured
    # equal to ~0.1% at serving shapes; the slack covers rounding-path
    # differences on small inputs).
    assert err_fused <= err_ref * 1.05 + 1e-6, f"{err_fused=} vs {err_ref=}"
    # Near-parity: the two paths differ only where a rounding boundary is
    # crossed (~1.4% of bytes at serving shapes). Blocks whose e4m3 scale
    # underflows to zero are excluded: their payload never contributes to
    # the dequantized value and the implementations fill it differently.
    live = (_block_scales(rsf, m, h) != 0) & (_block_scales(fsf, m, h) != 0)
    live_bytes = live.repeat_interleave(8, dim=1)  # 8 payload bytes / block
    if live_bytes.any():
        mm = (fq != rq)[live_bytes].float().mean().item()
        assert mm < 5e-2, f"payload byte mismatch fraction {mm}"


# 5376 is the Gemma4-31B hidden size; 228 the pinned decode batch; 7700 the
# profiled serving prefill size; 333 exercises padded tail rows.
@pytest.mark.parametrize("n_tokens", [1, 7, 228, 333, 7700])
@pytest.mark.parametrize("hidden", [5376, 512])
def test_rmsnorm_fp4_quant_quality(n_tokens, hidden):
    torch.manual_seed(1234)
    x = (
        torch.randn((n_tokens, hidden), dtype=torch.bfloat16, device="cuda")
        * torch.rand((n_tokens, 1), dtype=torch.bfloat16, device="cuda")
        * 3
    )
    w = torch.rand((hidden,), dtype=torch.bfloat16, device="cuda") + 0.5
    n = torch.ops.trtllm.flashinfer_rmsnorm(x, w, 1e-6)
    gs = (448.0 * 6.0 / n.abs().max().float()).reshape(1)
    _check(x, w, 1e-6, gs)


@pytest.mark.parametrize("gs_value", [1e6, 1.0, 1e-6])
def test_rmsnorm_fp4_quant_extreme_scales(gs_value):
    """Saturating / degenerate global scales must behave like the unfused op."""
    torch.manual_seed(7)
    x = torch.randn((333, 5376), dtype=torch.bfloat16, device="cuda")
    w = torch.rand((5376,), dtype=torch.bfloat16, device="cuda") + 0.5
    gs = torch.tensor([gs_value], dtype=torch.float32, device="cuda")
    _check(x, w, 1e-6, gs)


def test_rmsnorm_fp4_quant_strided_rows():
    """Row-strided input (a view of a wider buffer)."""
    torch.manual_seed(3)
    n, h = 65, 5376
    buf = torch.randn((n, h + 512), dtype=torch.bfloat16, device="cuda")
    x = buf[:, :h]
    w = torch.rand((h,), dtype=torch.bfloat16, device="cuda") + 0.5
    normed = torch.ops.trtllm.flashinfer_rmsnorm(x.contiguous(), w, 1e-6)
    gs = (448.0 * 6.0 / normed.abs().max().float()).reshape(1)
    _check(x, w, 1e-6, gs)


def test_rmsnorm_fp4_quant_zero_row():
    """An all-zero row must dequantize to exactly zero (zero block scales)."""
    torch.manual_seed(5)
    x = torch.randn((9, 5376), dtype=torch.bfloat16, device="cuda")
    x[4] = 0
    w = torch.rand((5376,), dtype=torch.bfloat16, device="cuda") + 0.5
    gs = torch.tensor([100.0], dtype=torch.float32, device="cuda")
    fq, fsf = rmsnorm_fp4_quant(x, w, 1e-6, gs)
    dq = _dequant(fq, fsf, 9, 5376, gs)
    assert dq[4].abs().max().item() == 0.0
    _check(x, w, 1e-6, gs)


def test_rmsnorm_fp4_quant_cuda_graph():
    """The kernel must be CUDA-graph capturable and replay deterministically
    (the serving decode path replays it inside captured graphs)."""
    torch.manual_seed(11)
    x = torch.randn((228, 5376), dtype=torch.bfloat16, device="cuda")
    w = torch.rand((5376,), dtype=torch.bfloat16, device="cuda") + 0.5
    gs = torch.tensor([100.0], dtype=torch.float32, device="cuda")

    eager_q, eager_sf = rmsnorm_fp4_quant(x, w, 1e-6, gs)  # also JIT warmup
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        graph_q, graph_sf = rmsnorm_fp4_quant(x, w, 1e-6, gs)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(graph_q, eager_q)
    valid = sf_swizzled_offsets(228, 5376 // 16, x.device)
    assert torch.equal(graph_sf.view(-1)[valid], eager_sf.view(-1)[valid])


if __name__ == "__main__":
    test_rmsnorm_fp4_quant_quality(7700, 5376)
    test_rmsnorm_fp4_quant_quality(228, 5376)
    test_rmsnorm_fp4_quant_extreme_scales(1e6)
    test_rmsnorm_fp4_quant_strided_rows()
    test_rmsnorm_fp4_quant_zero_row()
    test_rmsnorm_fp4_quant_cuda_graph()
    print("ALL QUALITY CHECKS PASSED")
