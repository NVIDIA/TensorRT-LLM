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
"""``trtllm::fp8_quantize_1x128_r128c4_kf`` must equal ``fp8_quantize_1x128_packed_ue8m0(x, True)`` byte for byte.

The generated MXFP8 quantization kernel (used by ``fp8_quantize_1x128_packed_ue8m0`` when
``TRTLLM_MXFP8_QUANT_KF=1``) writes BF16 [M, K] as FP8 E4M3 with one UE8M0 scale per 1x128 block,
replicated into the four K32 slots of the R128c4 layout, and zeros the scale rows past M. The FP8
bytes and the whole scale buffer must equal the production kernel's, and a PyTorch reference's, on
activation-like rows, block maxima exactly at 448 * 2^j (where the scale rounds up), all-zero blocks,
tiny (subnormal) and huge magnitudes, M on both sides of the 128-row tiles, K / 128 odd and even,
and under CUDA graph replay.

``fp8_quantize_1x128_cutedsl_ue8m0`` (the quantizer of Kimi K3's projections) writes the same bytes, and from
1024 rows on ``TRTLLM_MXFP8_QUANT_KF=1`` routes it to the same kernel (``test_bytes_equal_cutedsl_op``).
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100/SM103/SM107 GPU",
)

MS = [1, 2, 18, 64, 127, 128, 129, 777, 4096, 13356]
# K / 128 odd (1, 3) and even (12, 56, 96).
KS = [128, 384, 1536, 7168, 12288]
MODES = ["activations", "boundary", "zeros", "tiny", "huge", "mixed"]
RCP448_BITS = 0x3B124925  # the FP32 constant the kernels multiply amax by (rcp.approx(448))


def _load_ops():
    import tensorrt_llm  # noqa: F401  (registers the trtllm ops)


def _inputs(m, k, mode, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, generator=gen, device="cuda")
    kb = k // 128
    if mode == "activations":
        x = x * torch.exp2(torch.randint(-6, 7, (m, 1), generator=gen, device="cuda").float())
    elif mode == "boundary":
        j = torch.randint(-8, 8, (m, kb), generator=gen, device="cuda").float()
        sign = torch.where(torch.rand(m, kb, generator=gen, device="cuda") < 0.5, -1.0, 1.0)
        x = x.view(m, kb, 128) * torch.exp2(j - 4).unsqueeze(-1)
        x[:, :, 7] = sign * 448.0 * torch.exp2(j)
        x = x.view(m, k)
    elif mode == "zeros":
        keep = torch.rand(m, kb, 1, generator=gen, device="cuda") < 0.5
        x = (x.view(m, kb, 128) * keep).view(m, k)
        x[::3] = 0.0
    elif mode == "tiny":
        x = x * torch.exp2(torch.randint(-140, -100, (m, k), generator=gen, device="cuda").float())
    elif mode == "huge":
        x = x.sign() * torch.exp2(
            torch.randint(100, 127, (m, k), generator=gen, device="cuda").float()
        )
    elif mode == "mixed":
        block_scale = torch.exp2(
            torch.randint(-40, 41, (m, kb, 1), generator=gen, device="cuda").float()
        )
        x = (x.view(m, kb, 128) * block_scale).reshape(m, k)
    return x.to(torch.bfloat16)


def _reference(x):
    m, k = x.shape
    kb, mb = k // 128, (m + 127) // 128
    xb = x.float().view(m, kb, 128)
    rcp448 = torch.tensor([RCP448_BITS], dtype=torch.int32).view(torch.float32).item()
    bits = (xb.abs().amax(-1).clamp_min(1e-10) * rcp448).contiguous().view(torch.int32)
    e8 = (((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)).clamp(max=254)
    q = (xb * torch.exp2((127 - e8).float()).unsqueeze(-1)).view(m, k).to(torch.float8_e4m3fn)
    r = torch.arange(m, device=x.device).view(-1, 1)
    c = torch.arange(kb * 4, device=x.device).view(1, -1)
    offsets = ((((r // 128) * kb + c // 4) * 32 + r % 32) * 4 + (r % 128) // 32) * 4 + c % 4
    scales = torch.zeros(mb * 128 * kb * 4, dtype=torch.uint8, device=x.device)
    scales[offsets.reshape(-1)] = e8.to(torch.uint8).repeat_interleave(4, dim=1).reshape(-1)
    return q, scales


def _assert_same(actual, expected, what):
    q, s = actual
    q_ref, s_ref = expected
    assert q.shape == q_ref.shape and q.dtype == torch.float8_e4m3fn, what
    mismatch = int((q.view(torch.uint8) != q_ref.view(torch.uint8)).sum())
    assert mismatch == 0, f"{what}: {mismatch} FP8 bytes differ"
    assert s.shape == s_ref.shape and s.dtype == torch.uint8, (
        f"{what}: scale buffer {s.shape} vs {s_ref.shape}"
    )
    mismatch = int((s != s_ref).sum())
    assert mismatch == 0, f"{what}: {mismatch} scale bytes differ"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("k", KS)
@pytest.mark.parametrize("m", MS)
def test_bytes_equal_production(m, k, mode):
    _load_ops()
    x = _inputs(m, k, mode, seed=m * 131 + k)
    kf = torch.ops.trtllm.fp8_quantize_1x128_r128c4_kf(x)
    _assert_same(
        kf,
        torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(x, True),
        f"vs production m={m} k={k} {mode}",
    )
    if m * k <= 4096 * 7168:
        _assert_same(kf, _reference(x), f"vs reference m={m} k={k} {mode}")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("k", KS)
@pytest.mark.parametrize("m", MS)
def test_bytes_equal_cutedsl_op(m, k, mode):
    """``fp8_quantize_1x128_cutedsl_ue8m0`` quantizes the GEMM inputs of Kimi K3's projections.

    The generated kernel must write its bytes, and ``TRTLLM_MXFP8_QUANT_KF=1`` routes the op to that kernel from
    1024 rows on: with the flag unset this compares the kernel against the production kernel, with the flag set
    the op's own output (kernel at 1024+ rows) is held against the PyTorch reference.
    """
    _load_ops()
    x = _inputs(m, k, mode, seed=m * 131 + k + 5)
    op = torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(x)
    _assert_same(
        torch.ops.trtllm.fp8_quantize_1x128_r128c4_kf(x), op, f"kf vs cutedsl op m={m} k={k} {mode}"
    )
    if m * k <= 4096 * 7168:
        _assert_same(op, _reference(x), f"cutedsl op vs reference m={m} k={k} {mode}")


def test_padded_scale_rows_are_zero():
    _load_ops()
    m, k = 129, 1536
    x = _inputs(m, k, "activations", seed=1)
    _, s = torch.ops.trtllm.fp8_quantize_1x128_r128c4_kf(x)
    _, s_ref = _reference(x)
    pad = s_ref == 0
    assert int(pad.sum()) >= (256 - m) * (k // 32)
    assert int((s[pad] != 0).sum()) == 0


def test_empty_batch():
    _load_ops()
    q, s = torch.ops.trtllm.fp8_quantize_1x128_r128c4_kf(
        torch.empty(0, 7168, dtype=torch.bfloat16, device="cuda")
    )
    assert q.shape == (0, 7168) and s.numel() == 0


def test_rejects_unsupported_inputs():
    _load_ops()
    with pytest.raises(RuntimeError):
        torch.ops.trtllm.fp8_quantize_1x128_r128c4_kf(
            torch.randn(4, 192, device="cuda").to(torch.bfloat16)
        )
    with pytest.raises(RuntimeError):
        torch.ops.trtllm.fp8_quantize_1x128_r128c4_kf(torch.randn(4, 1536, device="cuda"))
    buf = torch.randn(4 * 1536 + 8, device="cuda").to(torch.bfloat16)
    misaligned = buf[8:].view(4, 1536)  # 16-byte offset: contiguous but not 32-byte aligned
    with pytest.raises(RuntimeError):
        torch.ops.trtllm.fp8_quantize_1x128_r128c4_kf(misaligned)
    # the production op still serves it
    _assert_same(
        torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(misaligned, True),
        _reference(misaligned),
        "misaligned",
    )


@pytest.mark.parametrize("m", [18, 4096])
def test_cuda_graph_replay(m):
    _load_ops()
    k = 7168
    x = _inputs(m, k, "activations", seed=7)
    torch.ops.trtllm.fp8_quantize_1x128_r128c4_kf(x)  # warm-up outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = torch.ops.trtllm.fp8_quantize_1x128_r128c4_kf(x)
    for seed in (8, 9):
        x.copy_(_inputs(m, k, "boundary", seed=seed))
        graph.replay()
        torch.cuda.synchronize()
        _assert_same(
            out,
            torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(x, True),
            f"graph m={m} seed={seed}",
        )
