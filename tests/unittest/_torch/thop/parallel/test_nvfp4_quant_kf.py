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
"""``trtllm::fp4_quantize_kf`` must equal ``fp4_quantize(x, gs, 16, False, False)`` byte for byte.

The generated NVFP4 quantization kernel (used by ``fp4_quantize`` for BF16 rows of width 3584 with linear
scales when ``TRTLLM_NVFP4_QUANT_KF=1``) writes E2M1 codes (two per byte, the even element in the low nibble) and
one E4M3 scale per 16 elements. Codes and scales must equal the production kernel's for M from 1 to 32768 on
activation-like rows, massive-activation channels, values on E2M1 code points and rounding midpoints, saturating
scales, all-zero and tiny blocks, BF16 subnormals, E4M3-subnormal scales, infinities, many global scales, other
widths, and under CUDA graph replay. Against a PyTorch reference (IEEE reciprocals) the scales must be equal and the
codes may differ only by one step on the rare elements within an ulp of a rounding midpoint.
"""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

# fp4_quantize latches TRTLLM_NVFP4_QUANT_KF on its first call: keep it off here so that _production runs the
# production kernel (test_routing sets it in subprocesses).
os.environ.pop("TRTLLM_NVFP4_QUANT_KF", None)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100/SM103/SM107 GPU",
)

K_K3 = 3584  # the Kimi K3 routed-expert input width, the shape fp4_quantize routes to the kernel
MS = [
    1,
    2,
    3,
    17,
    18,
    31,
    64,
    127,
    128,
    129,
    255,
    280,
    511,
    1023,
    1024,
    1025,
    2047,
    4096,
    13356,
    32768,
]
MODES = ["activations", "outliers", "ties", "saturate", "zeros", "tiny", "edge"]
RCP6_BITS = 0x3E2AAAAB  # rcp.approx(6.0f), the factor the kernels multiply the block amax by
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# The calibration scale 2688 / amax(bf16) is a ratio of small integers and BF16 values, so ~0.2 % of the products
# x * os land within an ulp of an E2M1 rounding midpoint, where IEEE and rcp.approx reciprocals legitimately round
# apart. The reference comparison moves the global scale off that lattice.
GENERIC_SCALE = 1.000123456


def _load_ops():
    import tensorrt_llm  # noqa: F401  (registers the trtllm ops)


def _global_scale(x):
    """The model's calibration: 448 * 6 / amax(x)."""
    return (448.0 * 6.0 / x.float().abs().amax().clamp_min(1e-12)).reshape(1)


def _inputs(m, k, mode, seed):
    """BF16 [m, k] and its FP32 global scale [1]."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, generator=gen, device="cuda")
    kb = k // 16
    if mode == "activations":
        x = x * torch.exp2(torch.randint(-6, 7, (m, 1), generator=gen, device="cuda").float())
    elif mode == "outliers":
        # massive-activation channels, a few hundred times the rest
        channels = torch.randint(0, k, (max(1, k // 512),), generator=gen, device="cuda")
        x[:, channels] *= 300.0
    elif mode == "ties":
        # power-of-two block scales and global scale 64: x * os lands exactly on E2M1 code points and midpoints
        points = torch.tensor(
            [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0],
            device="cuda",
        )
        idx = torch.randint(0, points.numel(), (m, kb, 16), generator=gen, device="cuda")
        sign = torch.where(torch.rand(m, kb, 16, generator=gen, device="cuda") < 0.5, -1.0, 1.0)
        vals = points[idx] * sign
        vals[:, :, 5] = 6.0  # the block amax, 6 * 2^e, makes the scale an exact power of two
        e = torch.randint(-8, 3, (m, kb, 1), generator=gen, device="cuda").float()
        x = (vals * torch.exp2(e)).view(m, k)
        return x.to(torch.bfloat16), torch.tensor([64.0], device="cuda")
    elif mode == "saturate":
        # block scales far above 448 saturate the scale, then the codes saturate at 6
        x = x * torch.exp2(
            torch.randint(0, 8, (m, kb, 1), generator=gen, device="cuda").float()
        ).repeat_interleave(16, dim=1).view(m, k)
        xb = x.to(torch.bfloat16)
        return xb, _global_scale(xb) * 64.0
    elif mode == "zeros":
        keep = torch.rand(m, kb, 1, generator=gen, device="cuda") < 0.5
        x = (x.view(m, kb, 16) * keep).view(m, k)
        x[::3] = 0.0
        x[1::7] = -0.0
    elif mode == "tiny":
        # blocks whose scale rounds to 0 (output scale rcp(0) = inf) next to normal ones
        tiny = torch.rand(m, kb, 1, generator=gen, device="cuda") < 0.3
        x = torch.where(tiny, x.view(m, kb, 16) * 1e-7, x.view(m, kb, 16)).view(m, k)
        x = torch.where(x == 0, torch.full_like(x, 1e-30), x)  # keep 0 * inf out of the tiny blocks
        xb = x.to(torch.bfloat16)
        return xb, _global_scale(xb)
    elif mode == "edge":
        # BF16 subnormals; block scales in the E4M3 subnormal range; scales that round to 0 in blocks that hold exact
        # zeros (0 * inf); +-inf elements (the global scale comes from the finite values)
        blocks = x.view(m, kb, 16)
        kind = torch.randint(0, 4, (m, kb, 1), generator=gen, device="cuda")
        zeroed = blocks * 1e-9
        zeroed[..., ::2] = 0.0
        blocks = torch.where(kind == 1, blocks * 1e-39, blocks)
        blocks = torch.where(kind == 2, blocks * 2e-5, blocks)
        blocks = torch.where(kind == 3, zeroed, blocks)
        xb = blocks.reshape(m, k).to(torch.bfloat16)
        gs = _global_scale(xb)
        inf = torch.rand(m, k, generator=gen, device="cuda") < 1e-4
        sign = torch.where(torch.rand(m, k, generator=gen, device="cuda") < 0.5, -1.0, 1.0)
        xb = torch.where(inf, (sign * float("inf")).to(torch.bfloat16), xb)
        return xb, gs
    xb = x.to(torch.bfloat16)
    return xb, _global_scale(xb)


def _e2m1_codes(v):
    """Round-to-nearest-even E2M1 codes (saturating at 6), sign in bit 3."""
    a = v.abs()
    code = torch.zeros_like(a, dtype=torch.int32)
    for i in range(1, len(E2M1_VALUES)):
        mid = (E2M1_VALUES[i - 1] + E2M1_VALUES[i]) / 2
        # above the midpoint, or on it when the upper code is even
        code = torch.where((a > mid) | ((a == mid) & (i % 2 == 0)), i, code)
    return (code | (torch.signbit(v).to(torch.int32) << 3)).to(torch.uint8)


def _reference(x, gs):
    """PyTorch NVFP4 reference with IEEE reciprocals: scale bytes exact, codes up to rare last-bit ties."""
    m, k = x.shape
    xb = x.float().view(m, k // 16, 16)
    amax = xb.abs().amax(-1)
    rcp6 = torch.tensor([RCP6_BITS], dtype=torch.int32, device=x.device).view(torch.float32)
    sf = (gs * (amax * rcp6)).clamp(max=448.0).to(torch.float8_e4m3fn)
    os_ = torch.where(amax != 0, 1.0 / (sf.float() * (1.0 / gs)), torch.zeros_like(amax))
    codes = _e2m1_codes(xb * os_.unsqueeze(-1)).view(m, k)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed, sf.view(torch.uint8).reshape(-1)


def _code_steps(q):
    """Unpacked E2M1 codes as signed step indices (both zeros map to 0)."""
    codes = torch.stack([q & 0xF, q >> 4], dim=-1).to(torch.int32)
    mag = codes & 0x7
    return torch.where((codes & 0x8) != 0, -mag, mag)


def _assert_same(actual, expected, what):
    q, s = actual
    q_ref, s_ref = expected
    assert q.shape == q_ref.shape and q.dtype == torch.uint8, f"{what}: codes {q.shape} {q.dtype}"
    assert s.shape == s_ref.shape and s.dtype == torch.uint8, (
        f"{what}: scales {s.shape} vs {s_ref.shape}"
    )
    mismatch = int((s != s_ref).sum())
    assert mismatch == 0, f"{what}: {mismatch} scale bytes differ"
    mismatch = int((q != q_ref).sum())
    assert mismatch == 0, f"{what}: {mismatch} code bytes differ"


def _production(x, gs):
    return torch.ops.trtllm.fp4_quantize(x, gs, 16, False, False)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("m", MS)
def test_bytes_equal_production(m, mode):
    _load_ops()
    x, gs = _inputs(m, K_K3, mode, seed=m * 131 + MODES.index(mode))
    kf = torch.ops.trtllm.fp4_quantize_kf(x, gs)
    _assert_same(kf, _production(x, gs), f"vs production m={m} {mode}")


# The tiny and edge blocks scale by inf (0 * inf in the edge blocks): production defines those bytes.
@pytest.mark.parametrize("mode", ["activations", "outliers", "ties", "saturate", "zeros"])
@pytest.mark.parametrize("m", [1, 17, 128, 1025, 4096])
def test_matches_reference(m, mode):
    _load_ops()
    x, gs = _inputs(m, K_K3, mode, seed=m * 137 + MODES.index(mode))
    if (
        mode != "ties"
    ):  # the ties mode keeps gs = 64: exact products on the code points and midpoints
        gs = gs * GENERIC_SCALE
    q, s = torch.ops.trtllm.fp4_quantize_kf(x, gs)
    q_ref, s_ref = _reference(x, gs)
    assert int((s != s_ref).sum()) == 0, f"m={m} {mode}: scale bytes differ from the reference"
    steps = (_code_steps(q) - _code_steps(q_ref)).abs()
    assert int(steps.max()) <= 1, (
        f"m={m} {mode}: a code differs from the reference by more than one step"
    )
    flips = int((steps != 0).sum())
    assert flips <= (0 if mode == "ties" else 16), (
        f"m={m} {mode}: {flips} codes differ from the reference"
    )


@pytest.mark.parametrize("seed", range(4))
def test_many_global_scales(seed):
    """rcp(gs) and rcp(SF * rcp(gs)) over 256 global scales, log-uniform in [2^-8, 2^16]."""
    _load_ops()
    gen = torch.Generator(device="cuda").manual_seed(1000 + seed)
    x = (torch.randn(64, K_K3, generator=gen, device="cuda") * 4.0).to(torch.bfloat16)
    exponents = torch.rand(256, generator=gen, device="cuda") * 24.0 - 8.0
    for gs in torch.exp2(exponents).float().split(1):
        _assert_same(
            torch.ops.trtllm.fp4_quantize_kf(x, gs), _production(x, gs), f"gs={gs.item()!r}"
        )


@pytest.mark.parametrize("k", [16, 48, 256, 2048, 7168, 7184])
@pytest.mark.parametrize("m", [1, 33, 1025, 4096])
def test_other_widths(m, k):
    _load_ops()
    x, gs = _inputs(m, k, "activations", seed=m + k)
    _assert_same(torch.ops.trtllm.fp4_quantize_kf(x, gs), _production(x, gs), f"m={m} k={k}")


def test_leading_dims():
    _load_ops()
    x, gs = _inputs(6 * 50, K_K3, "activations", seed=3)
    q, s = torch.ops.trtllm.fp4_quantize_kf(x.view(6, 50, K_K3), gs)
    assert q.shape == (6, 50, K_K3 // 2)
    _assert_same((q.view(300, -1), s), _production(x, gs), "[6, 50, K]")


def test_empty_batch():
    _load_ops()
    gs = torch.ones(1, device="cuda")
    q, s = torch.ops.trtllm.fp4_quantize_kf(
        torch.empty(0, K_K3, dtype=torch.bfloat16, device="cuda"), gs
    )
    assert q.shape == (0, K_K3 // 2) and s.numel() == 0


def test_rejects_unsupported_inputs():
    _load_ops()
    gs = torch.ones(1, device="cuda")
    with pytest.raises(RuntimeError):
        torch.ops.trtllm.fp4_quantize_kf(torch.randn(4, 40, device="cuda").to(torch.bfloat16), gs)
    with pytest.raises(RuntimeError):
        torch.ops.trtllm.fp4_quantize_kf(torch.randn(4, K_K3, device="cuda"), gs)
    buf = torch.randn(4 * K_K3 + 8, device="cuda").to(torch.bfloat16)
    misaligned = buf[8:].view(4, K_K3)  # 16-byte offset: contiguous but not 32-byte aligned
    with pytest.raises(RuntimeError):
        torch.ops.trtllm.fp4_quantize_kf(misaligned, gs)
    # the production op still serves it, with or without TRTLLM_NVFP4_QUANT_KF
    gs = _global_scale(misaligned)
    q, s = _production(misaligned, gs)
    _assert_same((q, s), _production(misaligned.clone(), gs), "misaligned")


@pytest.mark.parametrize("m", [18, 4096])
def test_cuda_graph_replay(m):
    _load_ops()
    x, gs = _inputs(m, K_K3, "activations", seed=7)
    torch.ops.trtllm.fp4_quantize_kf(x, gs)  # warm-up outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = torch.ops.trtllm.fp4_quantize_kf(x, gs)
    for seed in (8, 9):
        x_new, gs_new = _inputs(m, K_K3, "outliers", seed=seed)
        x.copy_(x_new)
        gs.copy_(gs_new)
        graph.replay()
        torch.cuda.synchronize()
        _assert_same(out, _production(x, gs), f"graph m={m} seed={seed}")


_ROUTING_PROBE = textwrap.dedent(
    """
    import hashlib
    import torch
    import tensorrt_llm  # noqa: F401

    def kernels(k, swizzled):
        x = torch.randn(256, k, device="cuda").to(torch.bfloat16)
        gs = torch.ones(1, device="cuda")
        torch.ops.trtllm.fp4_quantize(x, gs, 16, False, swizzled)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            torch.ops.trtllm.fp4_quantize(x, gs, 16, False, swizzled)
            torch.cuda.synchronize()
        return " ".join(e.name for e in prof.events())

    def digest(m):
        gen = torch.Generator(device="cuda").manual_seed(m)
        x = torch.randn(m, 3584, generator=gen, device="cuda")
        x[:, torch.randint(0, 3584, (7,), generator=gen, device="cuda")] *= 300.0
        x = x.to(torch.bfloat16)
        gs = (2688.0 / x.float().abs().amax()).reshape(1)
        q, s = torch.ops.trtllm.fp4_quantize(x, gs, 16, False, False)
        return hashlib.sha256(q.cpu().numpy().tobytes() + s.cpu().numpy().tobytes()).hexdigest()[:16]

    print("K3584_LINEAR", "nvfp4QuantKfKernel" in kernels(3584, False))
    print("K7168_LINEAR", "nvfp4QuantKfKernel" in kernels(7168, False))
    print("K3584_SWIZZLED", "nvfp4QuantKfKernel" in kernels(3584, True))
    print("DIGEST", "-".join(digest(m) for m in (1, 300, 1025, 8192)))
    """
)


def _probe(flag):
    env = {k: v for k, v in os.environ.items() if k != "TRTLLM_NVFP4_QUANT_KF"}
    if flag is not None:
        env["TRTLLM_NVFP4_QUANT_KF"] = flag
    result = subprocess.run(
        [sys.executable, "-c", _ROUTING_PROBE], env=env, capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stderr[-2000:]
    fields = [line.split() for line in result.stdout.splitlines()]
    return {
        f[0]: f[1]
        for f in fields
        if len(f) == 2 and f[0] in ("K3584_LINEAR", "K7168_LINEAR", "K3584_SWIZZLED", "DIGEST")
    }


def test_routing():
    """TRTLLM_NVFP4_QUANT_KF=1 routes only BF16 rows of width 3584 with linear scales, and fp4_quantize returns the same
    bytes with the flag unset, off and on (needs the full build)."""
    probes = {flag: _probe(flag) for flag in (None, "0", "1")}
    for flag, lines in probes.items():
        assert {k: v for k, v in lines.items() if k != "DIGEST"} == {
            "K3584_LINEAR": str(flag == "1"),
            "K7168_LINEAR": "False",
            "K3584_SWIZZLED": "False",
        }, (flag, lines)
    assert len({lines["DIGEST"] for lines in probes.values()}) == 1, probes
