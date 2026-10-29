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
"""Kimi K3 fused MoE residual add: ``residual + bf16(routed + shared)`` bit for bit.

The Triton kernel serves every token count up to ``MAX_FUSED_ROWS`` (one compiled kernel: the
element count is a runtime argument), eagerly and in a CUDA graph; larger inputs take the two
BF16 adds, unless the caller lifts the row cap (``max_rows=None``, context forwards under
``TRTLLM_K3_FUSE_MOE_RESIDUAL_CTX``).
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

HIDDEN = 7168


class _LaunchSpy:
    """Counts launches of a Triton kernel (``kernel[grid](...)``) and forwards them."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.launches = 0

    def __getitem__(self, grid):
        self.launches += 1
        return self.kernel[grid]


def _inputs(m: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return tuple(
        (scale * torch.randn((m, HIDDEN), device="cuda", generator=gen)).to(torch.bfloat16)
        for scale in (1.0, 0.3, 4.0)
    )


def _assert_bitwise(out: torch.Tensor, ref: torch.Tensor) -> None:
    assert out.dtype == ref.dtype == torch.bfloat16 and out.shape == ref.shape
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))


def _record_once_logs(mod, monkeypatch) -> list:
    """Records the module's ``info_once`` / ``warning_once`` calls as ``(name, key, message)``."""
    records = []
    for name in ("info_once", "warning_once"):
        monkeypatch.setattr(
            mod.logger,
            name,
            lambda *msg, key, _name=name: records.append((_name, key, " ".join(map(str, msg)))),
        )
    return records


@pytest.mark.parametrize("m", [1, 7, 96, 97, 128, 224, 256, 257])
def test_fused_add_matches_two_bf16_adds(m, monkeypatch):
    from tensorrt_llm._torch.modules import kimi_k3_moe_residual as mod

    spy = _LaunchSpy(mod._moe_residual_add_kernel)
    monkeypatch.setattr(mod, "_moe_residual_add_kernel", spy)
    routed, shared, residual = _inputs(m, seed=m)
    out = mod._fused_moe_residual_add(routed, shared, residual)
    _assert_bitwise(out, residual + (routed + shared))
    assert spy.launches == (1 if m <= mod.MAX_FUSED_ROWS else 0)


def test_fused_add_ineligible_inputs_fall_back(monkeypatch):
    from tensorrt_llm._torch.modules import kimi_k3_moe_residual as mod

    spy = _LaunchSpy(mod._moe_residual_add_kernel)
    monkeypatch.setattr(mod, "_moe_residual_add_kernel", spy)
    routed, shared, residual = _inputs(16, seed=3)
    for args in (
        (routed.float(), shared.float(), residual.float()),
        (routed[:, :4096], shared[:, :4096], residual[:, :4096]),
        (routed.t().contiguous().t(), shared, residual),
    ):
        out = mod._fused_moe_residual_add(*args)
        torch.testing.assert_close(out, args[2] + (args[0] + args[1]), rtol=0, atol=0)
    assert spy.launches == 0


def test_fused_add_in_cuda_graph():
    """Captured once at each of three token counts, replayed on new inputs."""
    from tensorrt_llm._torch.modules.kimi_k3_moe_residual import _fused_moe_residual_add

    for m in (12, 128, 256):
        routed, shared, residual = _inputs(m, seed=100 + m)
        _fused_moe_residual_add(routed, shared, residual)  # compile outside the capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            # A chain: each add feeds the next one's residual, as consecutive layers do.
            out = _fused_moe_residual_add(routed, shared, residual)
            out = _fused_moe_residual_add(routed, shared, out)
        for seed in (1, 2):
            new = _inputs(m, seed=1000 * seed + m)
            for dst, src in zip((routed, shared, residual), new):
                dst.copy_(src)
            graph.replay()
            ref = new[2] + (new[0] + new[1])
            ref = ref + (new[0] + new[1])
            torch.cuda.synchronize()
            _assert_bitwise(out, ref)


@pytest.mark.parametrize("m", [1, 256, 257, 4096, 13356])
def test_uncapped_fused_add_matches_two_bf16_adds(m, monkeypatch):
    """``max_rows=None``: context row counts launch the kernel and stay bitwise."""
    from tensorrt_llm._torch.modules import kimi_k3_moe_residual as mod

    spy = _LaunchSpy(mod._moe_residual_add_kernel)
    monkeypatch.setattr(mod, "_moe_residual_add_kernel", spy)
    logs = _record_once_logs(mod, monkeypatch)
    routed, shared, residual = _inputs(m, seed=m)
    out = mod._fused_moe_residual_add(routed, shared, residual, max_rows=None)
    _assert_bitwise(out, residual + (routed + shared))
    assert spy.launches == 1
    context = m > mod.MAX_FUSED_ROWS
    expected = [("info_once", "kimi_k3_moe_residual_ctx")] if context else []
    assert [(name, key) for name, key, _ in logs] == expected


def test_uncapped_fused_add_ineligible_inputs_fall_back(monkeypatch):
    from tensorrt_llm._torch.modules import kimi_k3_moe_residual as mod

    spy = _LaunchSpy(mod._moe_residual_add_kernel)
    monkeypatch.setattr(mod, "_moe_residual_add_kernel", spy)
    logs = _record_once_logs(mod, monkeypatch)
    routed, shared, residual = _inputs(512, seed=5)
    wide = torch.empty((512, 2 * HIDDEN), dtype=torch.bfloat16, device="cuda")
    wide[:, :HIDDEN] = routed
    out = mod._fused_moe_residual_add(wide[:, :HIDDEN], shared, residual, max_rows=None)
    _assert_bitwise(out, residual + (routed + shared))
    assert spy.launches == 0
    # Not a silent fallback: one warning naming the inputs' state.
    assert [(name, key) for name, key, _ in logs] == [
        ("warning_once", "kimi_k3_moe_residual_ctx_fallback")
    ]
    assert "contiguous [False, True, True]" in logs[0][2]


@pytest.mark.parametrize("over", [False, True])
def test_element_cap_keeps_the_aten_adds(over, monkeypatch):
    """Inputs above the int32-offset element cap take the two BF16 adds."""
    from tensorrt_llm._torch.modules import kimi_k3_moe_residual as mod

    m = 300
    spy = _LaunchSpy(mod._moe_residual_add_kernel)
    monkeypatch.setattr(mod, "_moe_residual_add_kernel", spy)
    monkeypatch.setattr(mod, "_MAX_FUSED_NUMEL", m * HIDDEN - (1 if over else 0))
    routed, shared, residual = _inputs(m, seed=7)
    out = mod._fused_moe_residual_add(routed, shared, residual, max_rows=None)
    _assert_bitwise(out, residual + (routed + shared))
    assert spy.launches == (0 if over else 1)


#: BF16 bit patterns: signed zeros, infinities, the smallest and largest subnormals, the
#: smallest normal, the largest finite values, and 1, 1 + 2^-7 and 2^-8 (sums that tie).
_SPECIAL_BITS = (
    0x0000,
    0x8000,
    0x7F80,
    0xFF80,
    0x0001,
    0x8001,
    0x007F,
    0x807F,
    0x0080,
    0x8080,
    0x7F7F,
    0xFF7F,
    0x3F80,
    0x3F81,
    0x3B80,
    0xBB80,
)


def _special_inputs(m: int, seed: int):
    """Each element: a random normal value, a random bit pattern or one of ``_SPECIAL_BITS``."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    specials = torch.tensor(_SPECIAL_BITS, dtype=torch.int32, device="cuda")
    tensors = []
    for _ in range(3):
        normal = torch.randn((m, HIDDEN), device="cuda", generator=gen).to(torch.bfloat16)
        bits = torch.randint(0, 2**16, (m, HIDDEN), device="cuda", generator=gen)
        picks = torch.randint(0, len(_SPECIAL_BITS), (m, HIDDEN), device="cuda", generator=gen)
        kind = torch.randint(0, 3, (m, HIDDEN), device="cuda", generator=gen)
        bits = torch.where(kind == 1, bits, specials[picks]).to(torch.int32)
        patterns = torch.where(bits >= 2**15, bits - 2**16, bits).to(torch.int16)
        tensors.append(torch.where(kind == 0, normal, patterns.view(torch.bfloat16)))
    return tuple(tensors)


def test_fused_add_special_values_match_two_bf16_adds(monkeypatch):
    """Zeros, subnormals, infinities, overflow and ties: bit for bit; NaN wherever ATen has NaN."""
    from tensorrt_llm._torch.modules import kimi_k3_moe_residual as mod

    spy = _LaunchSpy(mod._moe_residual_add_kernel)
    monkeypatch.setattr(mod, "_moe_residual_add_kernel", spy)
    routed, shared, residual = _special_inputs(64, seed=11)
    out = mod._fused_moe_residual_add(routed, shared, residual)
    ref = residual + (routed + shared)
    assert spy.launches == 1
    nan = torch.isnan(ref)
    assert nan.any() and torch.equal(torch.isnan(out), nan)
    assert torch.equal(out.view(torch.int16)[~nan], ref.view(torch.int16)[~nan])
