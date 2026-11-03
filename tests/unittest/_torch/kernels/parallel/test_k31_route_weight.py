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
"""``k31_route_quant._routing_weight`` (fp32 quotient, fp64 only near a bf16 rounding midpoint) returns exactly the
bf16 bits of the fp64 expression ``sigmoid * scale / (sum of the 16 sigmoids + 1e-20)``: random and router-like
sigmoids at several scales, quotients pushed onto bf16 midpoints, tiny sums and denormal quotients."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)

TOPK = 16


def _kernels():
    import cutlass
    import cutlass.cute as cute

    from tensorrt_llm._torch.moe.kernels.blackwell import k31_route_quant as rq

    @cute.kernel
    def kernel(sig: cute.Tensor, out: cute.Tensor, scale: cutlass.Float64):
        lane, _, _ = cute.arch.thread_idx()
        tok, _, _ = cute.arch.block_idx()
        s_sig = cutlass.Array(cutlass.Float32, 32, space=cutlass.AddressSpace.smem, alignment=16)
        if lane < cutlass.Int32(TOPK):
            s_sig.store(cutlass.Float32(sig[tok, lane]), idx=lane)
        cute.arch.barrier()
        expert = lane % cutlass.Int32(TOPK)
        fast = rq._routing_weight(s_sig, expert, lane, scale)
        ref = rq._routing_weight_fp64(s_sig, expert, lane, scale)
        if lane < cutlass.Int32(TOPK):
            out[tok, 0, lane] = fast
            out[tok, 1, lane] = ref

    @cute.jit
    def launch(sig: cute.Tensor, out: cute.Tensor, scale: cutlass.Float64, n: cutlass.Int32):
        kernel(sig, out, scale).launch(grid=(n, 1, 1), block=(32, 1, 1))

    return launch


_FN = {}


def _run(sig, scale):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    sig = sig.contiguous().float().cuda()
    n = sig.shape[0]
    out = torch.zeros((n, 2, TOPK), dtype=torch.int16, device="cuda")
    args = [from_dlpack(sig), from_dlpack(out)]
    if n not in _FN:
        _FN[n] = cute.compile(_kernels(), *args, cutlass.Float64(scale), cutlass.Int32(n))
    _FN[n](*args, cutlass.Float64(scale), cutlass.Int32(n))
    torch.cuda.synchronize()
    return out


def _check(sig, scale):
    out = _run(sig, scale)
    bad = (out[:, 0] != out[:, 1]).nonzero()
    assert bad.numel() == 0, f"{bad.shape[0]} weights differ, first {bad[:4].tolist()}"


@pytest.mark.parametrize("scale", [1.0, 2.5, 2.827, 1.7320508075688772])
def test_random(scale):
    g = torch.Generator().manual_seed(int(scale * 1000))
    _check(torch.rand(1 << 16, TOPK, generator=g), scale)


def test_router_like():
    g = torch.Generator().manual_seed(5)
    logits = torch.randn(1 << 14, 896, generator=g) * 1.6 - 0.8
    bias = (torch.rand(896, generator=g) - 0.5) * 0.19
    sig = torch.sigmoid(logits)
    idx = torch.topk(sig + bias, TOPK, dim=1).indices
    _check(torch.gather(sig, 1, idx), 2.827)


def test_midpoints():
    """Quotients on and next to bf16 rounding midpoints (the fp64 fallback)."""
    g = torch.Generator().manual_seed(7)
    n = 1 << 14
    sig = torch.rand(n, TOPK, generator=g) * 0.9 + 0.05
    vals = torch.cat([sig, torch.zeros_like(sig)], 1)
    for off in (16, 8, 4, 2, 1):
        vals = vals + vals[:, torch.arange(32) ^ off]
    total = vals[:, 0].double()
    scale = 2.5
    # Lane 0's weight -> a bf16 midpoint (+- a few fp32 ulps), keeping the total fixed by moving lane 1.
    w = (sig[:, 0].double() * scale / total).float()
    bits = w.view(torch.int32)
    target = (bits & ~0xFFFF) | 0x8000
    target = target + torch.randint(-12, 13, (n,), generator=g, dtype=torch.int32)
    w_t = target.view(torch.float32).double()
    s0 = (w_t * total / scale).float()
    delta = s0 - sig[:, 0]
    sig[:, 0] = s0
    sig[:, 1] = (sig[:, 1] - delta).clamp(min=1e-3)
    _check(sig, scale)


def test_tiny_and_denormal():
    g = torch.Generator().manual_seed(9)
    sig = torch.rand(1 << 12, TOPK, generator=g)
    sig[: 1 << 10] *= 1e-13  # tiny sums
    sig[1 << 10 : 1 << 11, 0] = 1e-39  # denormal quotient
    sig[1 << 11 : (1 << 11) + 16] = 0.0
    sig[1 << 11 : (1 << 11) + 16, 0] = 0.5
    _check(sig, 2.5)
