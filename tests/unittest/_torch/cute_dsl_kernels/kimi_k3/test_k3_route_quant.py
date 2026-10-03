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
"""trtllm::k3_route_quant (CuTe DSL top-16 routing + MXFP8 latent, M <= 64) at every M in 1..8 and at 16, 32, 64:
top-16 ids, routing weights, MXFP8 codes and scales bit for bit against trtllm::kimi_k3_noaux_tc_mxfp8_quant and
against the unfused chain (trtllm::noaux_tc_op + trtllm::mxfp8_quantize), run-to-run and early-trigger bits, each M's
rows the bits of the same rows of the 64-row call; plus ties, saturation, lane overflow and zero / large / denormal
latent rows at M = 1, 3, 8. The ids are also compared with a stable PyTorch sort of sigmoid + bias (reported)."""

import functools

import pytest
import torch


def _is_sm100() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 10


pytestmark = pytest.mark.skipif(not _is_sm100(), reason="needs an SM100-family GPU")

E, K, H = 896, 16, 3584
SCALE = 2.827
M_CASES = list(range(1, 9)) + [16, 32, 64]


def _ops():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.cute_dsl_kernels.k3_route_quant import op  # noqa: F401

    return torch.ops.trtllm


def _bits(t: torch.Tensor) -> torch.Tensor:
    view = {torch.bfloat16: torch.int16, torch.float32: torch.int32, torch.float8_e4m3fn: torch.uint8}
    return t.contiguous().view(view.get(t.dtype, t.dtype))


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


def _torch_ids(scores, bias):
    """Top-16 of sigmoid + bias, descending, ties to the lower id (stable sort of -key)."""
    key = torch.sigmoid(scores) + bias
    return torch.sort(-key, dim=1, stable=True).indices[:, :K].int()


def _unfused(scores, bias, hidden):
    """The stock chain the fused C++ op replaces: noaux_tc routing, then the MXFP8 quantization of the latent."""
    ops = _ops()
    weights, ids = ops.noaux_tc_op(scores, bias, 1, 1, K, SCALE)
    quantized, scales = ops.mxfp8_quantize(hidden, False, alignment=256)
    m = scores.shape[0]
    return ids.int(), weights.to(torch.bfloat16), quantized.view(torch.float8_e4m3fn), scales.view(m, -1)


@functools.lru_cache(maxsize=None)
def _random(m_max: int = 64):
    gen = torch.Generator(device="cuda").manual_seed(20260928)
    s = torch.randn(m_max, E, generator=gen, device="cuda") * 2.5
    b = torch.randn(E, generator=gen, device="cuda") * 0.1
    h = (torch.randn(m_max, H, generator=gen, device="cuda") * 0.7).bfloat16()
    return s, b, h


def _check(name, s, b, h):
    ops = _ops()
    got = ops.k3_route_quant(s, b, h, SCALE)
    cpp = ops.kimi_k3_noaux_tc_mxfp8_quant(s, b, h, SCALE)
    unf = _unfused(s, b, h)
    vs_cpp = [_same(g, w) for g, w in zip(got, cpp)]
    vs_unfused = [_same(g, w) for g, w in zip(got, unf)]
    torch_ids = torch.equal(got[0], _torch_ids(s, b))
    rerun = all(_same(a, c) for a, c in zip(ops.k3_route_quant(s, b, h, SCALE), got))
    early = all(_same(a, c) for a, c in zip(ops.k3_route_quant(s, b, h, SCALE, True), got))
    print(f"OPCHECK op=k3_route_quant case={name} M={s.shape[0]} vs_cpp(ids,w,q,sf)={vs_cpp} "
          f"vs_unfused(ids,w,q,sf)={vs_unfused} torch_ids={torch_ids} det={rerun} early_same={early}")  # fmt: skip
    return got, all(vs_cpp), all(vs_unfused), torch_ids, rerun, early


@pytest.mark.parametrize("m", M_CASES)
def test_k3_route_quant(m):
    s64, b, h64 = _random()
    s, h = s64[:m].contiguous(), h64[:m].contiguous()
    got, vs_cpp, vs_unfused, torch_ids, rerun, early = _check("random", s, b, h)
    full = _ops().k3_route_quant(s64, b, h64, SCALE)
    rows = all(_same(g, f[:m]) for g, f in zip(got, full))
    print(f"OPCHECK op=k3_route_quant case=random M={m} rows_as_m64={rows}")
    assert vs_cpp and vs_unfused and rerun and early and rows


def _edge_cases():
    gen = torch.Generator(device="cuda").manual_seed(20260929)
    m = 8
    b = torch.randn(E, generator=gen, device="cuda") * 0.1
    h = (torch.randn(m, H, generator=gen, device="cuda") * 0.7).bfloat16()
    s = torch.randn(m, E, generator=gen, device="cuda") * 2.5
    s[:, 100:140] = 3.0
    b_tied = b.clone()
    b_tied[100:140] = 0.25  # 40 equal keys compete for the top 16: ties go to the lower id
    yield "40_tied_keys", s, b_tied, h
    yield "all_equal_logits", torch.full((m, E), 0.3, device="cuda"), torch.zeros(E, device="cuda"), h
    yield "huge_logits", torch.randn(m, E, generator=gen, device="cuda") * 40.0, b, h
    s = torch.randn(m, E, generator=gen, device="cuda")
    s[:, 3::32] += 20.0  # every winner in one selection lane (the exact fallback)
    yield "16_winners_one_lane", s, b, h
    s = torch.randn(m, E, generator=gen, device="cuda")
    s[:, [5, 37, 69, 101, 133]] += 20.0
    yield "5_winners_one_lane", s, b, h
    h2 = h.clone()
    h2[0] = 0
    h2[1, :64] = 0
    h2[2] = h2[2] * 3e4
    h2[3, ::7] = torch.tensor(1e-39).bfloat16()
    h2[4, 5] = torch.tensor(-3e38).bfloat16()
    yield "zero_large_denormal_rows", torch.randn(m, E, generator=gen, device="cuda"), b, h2
    yield "bias_parameter", torch.randn(m, E, generator=gen, device="cuda"), torch.nn.Parameter(b.clone()), h


EDGE_CASES = [
    "40_tied_keys",
    "all_equal_logits",
    "huge_logits",
    "16_winners_one_lane",
    "5_winners_one_lane",
    "zero_large_denormal_rows",
    "bias_parameter",
]


@pytest.mark.parametrize("case", EDGE_CASES)
def test_k3_route_quant_edge_cases(case):
    """The PyTorch sort is reported, not asserted: its sigmoid may tie or split keys the kernels' sigmoid does not."""
    name, s, b, h = next(c for c in _edge_cases() if c[0] == case)
    for m in (1, 3, 8):
        _, vs_cpp, vs_unfused, _, rerun, early = _check(name, s[:m].contiguous(), b, h[:m].contiguous())
        assert vs_cpp and vs_unfused and rerun and early


@pytest.mark.parametrize("m", [0, 65])
def test_token_limit(m):
    ops = _ops()
    s = torch.zeros(m, E, device="cuda")
    h = torch.zeros(m, H, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError):
        ops.k3_route_quant(s, torch.zeros(E, device="cuda"), h, SCALE)
