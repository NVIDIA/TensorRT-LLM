# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ssm/short_conv_update op (FLA's ``causal_conv1d_update``).

Kimi K3's sequential verify steps each q / k / v section once per verify token: x the step's column of the
projection (a strided bf16 ``[N, 1, D]`` view of the fused rows at TP16; a contiguous one in the TP4 cell), the cache
a dense bf16 ``[N, D, 4]`` copy of the slot windows left-padded with a zero column, the module's bf16 taps viewed
``[D, 4]``, SiLU. Every step's cache is the next step's. The reference is the entry's fp32 shift + dot: the cache holds
raw inputs, so it is bit for bit; y differs only in the fp32 summation order before its rounding to bf16, so it is
gated at one bf16 step (2 ** -7 relative).
"""

import importlib
import zlib

import pytest
import torch

__extra_import_path__ = ["../../.."]
from _validating import validating

short_conv_update = importlib.import_module(
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl."
    "kimi_k3_mxfp4__sm_100__tp16_moetp4ep4.ops.ssm.short_conv_update"
).short_conv_update

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="certified on SM100 and SM103",
)

STEPS = 3  # one verify round's q / k / v column per step
SECTIONS = 4  # the fused [q | k | v | g] rows
SECTION = 1  # the k section: neither the first nor the last column block
RTOL, ATOL = 2.0**-7, 1e-5  # one bf16 step


def _make(cell, seed: int) -> dict:
    spec = cell.spec
    n, d, w = spec["rows"], spec["dim"], spec["width"]
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*s, scale=1.0):
        return torch.randn(*s, generator=gen, device="cuda") * scale

    if spec["strided_x"]:
        fused = rnd(n, STEPS, SECTIONS * d).bfloat16()
        xs = [fused[:, t : t + 1, SECTION * d : (SECTION + 1) * d] for t in range(STEPS)]
    else:
        xs = [rnd(n, 1, d).bfloat16() for _ in range(STEPS)]
    # Two rows around the requests' cache rows, which must stay as they are.
    backing = rnd(n + 2, d, w).bfloat16()
    backing[1 : n + 1, :, 0] = 0  # the zero column the layer pads the W - 1 window with
    return {
        "xs": xs,
        "backing": backing,
        "cache": backing[1 : n + 1],
        "weight": rnd(d, w, scale=0.3).bfloat16(),
    }


@pytest.mark.parametrize(
    "cell", short_conv_update.CELLS, ids=[c.why[:40] for c in short_conv_update.CELLS]
)
def test_steps_against_reference(cell) -> None:
    """STEPS chained steps: the cache argument updated in place and returned, bit for bit with the reference's; y in
    one bf16 step of it, shaped like x; x, the weight and the rows around the cache untouched."""
    t = _make(cell, zlib.crc32(cell.why.encode()) % 2**31)
    cache, ref_cache = t["cache"], t["cache"].clone()
    assert cache.is_contiguous()
    backing0 = t["backing"].clone()
    xs0 = [x.clone() for x in t["xs"]]
    w0 = t["weight"].clone()
    n = cache.shape[0]
    with torch.inference_mode():
        for step, x in enumerate(t["xs"]):
            with validating(short_conv_update):
                y, out_cache = short_conv_update(x, cache, t["weight"], "silu")
            y_ref, ref_out = short_conv_update.reference(x, ref_cache, t["weight"], "silu")
            torch.cuda.synchronize()
            assert out_cache.data_ptr() == cache.data_ptr() and ref_out is ref_cache, step
            assert torch.equal(cache, ref_cache), step
            assert y.shape == x.shape and y.dtype == torch.bfloat16 and y.is_contiguous(), step
            torch.testing.assert_close(y, y_ref, rtol=RTOL, atol=ATOL)
        # After STEPS steps the cache's last STEPS columns are the inputs, oldest first.
        for step, x in enumerate(t["xs"]):
            col = cache.shape[-1] - STEPS + step
            assert torch.equal(cache[:, :, col], x.reshape(n, -1))
    assert torch.equal(t["backing"][0], backing0[0]) and torch.equal(t["backing"][-1], backing0[-1])
    assert all(torch.equal(a, b) for a, b in zip(t["xs"], xs0)) and torch.equal(t["weight"], w0)


def test_strided_cache_is_not_updated_in_place() -> None:
    """Negative control for the contiguous-cache contract: a cache that is a strided view is copied by FLA's input
    guard, so the result is right but the argument is not updated; the layer would lose the step silently. The
    guard refuses it."""
    cell = short_conv_update.CELLS[0]
    t = _make(cell, 5)
    n, d, w = t["cache"].shape
    with torch.inference_mode():
        strided = torch.empty(n, w, d, dtype=torch.bfloat16, device="cuda").transpose(1, 2)
        strided.copy_(t["cache"])
        before = strided.clone()
        ref = t["cache"].clone()
        with pytest.raises(AssertionError, match="contiguous"), validating(short_conv_update):
            short_conv_update(t["xs"][0], strided, t["weight"], "silu")
        y, out_cache = short_conv_update(t["xs"][0], strided, t["weight"], "silu")
        y_ref, _ = short_conv_update.reference(t["xs"][0], ref, t["weight"], "silu")
        torch.cuda.synchronize()
    assert torch.equal(strided, before), "a strided cache was written in place"
    assert out_cache.data_ptr() != strided.data_ptr()
    assert torch.equal(out_cache, ref)
    torch.testing.assert_close(y, y_ref, rtol=RTOL, atol=ATOL)
