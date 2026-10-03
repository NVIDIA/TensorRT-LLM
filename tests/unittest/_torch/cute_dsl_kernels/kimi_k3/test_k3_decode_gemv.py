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
"""trtllm::k3_decode_gemv / k3_decode_gemv_tail (CuTe DSL decode GEMV, M <= 8) at the Kimi K3 TP16 per-rank shapes,
at every M in 1..8: error against an fp64 product and against the stock path (cuBLAS F.linear; RMSNorm -> slice ->
concat -> linear for the tail), for the tail also against torch with the kernel's arithmetic (fp32 accumulators, the
RMS on the latent one, one bf16 rounding), run-to-run identical bits and each M's rows bit-identical to the same rows
of the 8-row call. 0 and 9 rows are refused."""

import functools

import pytest
import torch
import torch.nn.functional as F


def _is_sm100() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 10


pytestmark = pytest.mark.skipif(not _is_sm100(), reason="needs an SM100-family GPU")

M_ALL = list(range(1, 9))
TOL = 8e-3  # max |y - ref| / max |ref|
# (N, K) per rank at TP16: the KDA o_proj (one CTA per weight tile) and the KDA input projection (split-K).
SHAPES = {"kda_o_proj": (7168, 768), "kda_qkvg": (3208, 7168)}
HIDDEN, LATENT, WIDTH, PAD, ACT = 7168, 3584, 224, 256, 384
EPS = 1e-6


def _ops():
    import tensorrt_llm  # noqa: F401  (registers the trtllm ops)
    from tensorrt_llm._torch.cute_dsl_kernels.k3_decode_gemv import op  # noqa: F401

    return torch.ops.trtllm


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _rel(y: torch.Tensor, ref: torch.Tensor) -> float:
    return (y.double() - ref.double()).abs().max().item() / ref.double().abs().max().item()


def _report(op, case, m, y, ref, stock=None, **flags):
    extra = "" if stock is None else f" vs_stock={_rel(y, stock):.2e}"
    marks = " ".join(f"{k}={v}" for k, v in flags.items())
    abs_err = (y.double() - ref.double()).abs().max().item()
    print(f"OPCHECK op={op} case={case} M={m} abs={abs_err:.3e} rel={_rel(y, ref):.3e}{extra} {marks}")


@functools.lru_cache(maxsize=None)
def _weight(n: int, k: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(n, k, generator=gen, device="cuda") * 0.03).bfloat16()


@functools.lru_cache(maxsize=None)
def _rows(k: int, seed: int, scale: float = 1.0) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(8, k, generator=gen, device="cuda") * scale).bfloat16()


@pytest.mark.parametrize("m", M_ALL)
@pytest.mark.parametrize("name", list(SHAPES))
def test_k3_decode_gemv(name, m):
    ops = _ops()
    n, k = SHAPES[name]
    w = _weight(n, k, 1)
    x8 = _rows(k, 2)
    x = x8[:m].contiguous()
    y = ops.k3_decode_gemv(x, w, True)
    again = ops.k3_decode_gemv(x, w, True)
    y8 = ops.k3_decode_gemv(x8, w, True)
    ref = x.double() @ w.double().t()
    deterministic = torch.equal(_bits(y), _bits(again))
    m_invariant = torch.equal(_bits(y), _bits(y8[:m]))
    _report("k3_decode_gemv", name, m, y, ref, F.linear(x, w), det=deterministic, rows_as_m8=m_invariant)
    assert y.shape == (m, n)
    assert _rel(y, ref) <= TOL and _rel(y, F.linear(x, w)) <= TOL
    assert deterministic and m_invariant


def _tail_weight() -> torch.Tensor:
    w = _weight(HIDDEN, PAD + ACT, 3).clone()
    w[:, WIDTH:PAD] = 0
    return w


def _tail_ref(latent, act, w, lo):
    lat = latent.double()
    normed = lat * torch.rsqrt(lat.pow(2).mean(dim=1, keepdim=True) + EPS)
    x = torch.cat([normed[:, lo : lo + WIDTH], act.double()], dim=1)
    return x @ torch.cat([w[:, :WIDTH], w[:, PAD:]], dim=1).double().t()


def _tail_stock(latent, act, w, lo):
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    norm = RMSNorm(hidden_size=LATENT, eps=EPS, dtype=torch.bfloat16).cuda()
    norm.weight.data.fill_(1.0)
    x = torch.cat([norm(latent)[:, lo : lo + WIDTH], act], dim=1)
    return F.linear(x, torch.cat([w[:, :WIDTH], w[:, PAD:]], dim=1))


def _tail_fp32(latent, act, w, lo):
    """The tail with the kernel's arithmetic: fp32 accumulators of the latent slice and of the activation, the latent
    one scaled by the RMS of the whole latent row, one bf16 rounding."""
    lat = latent.float()
    scale = torch.rsqrt(lat.pow(2).mean(dim=1, keepdim=True) + EPS)
    acc_lat = lat[:, lo : lo + WIDTH] @ w[:, :WIDTH].float().t()
    return (acc_lat * scale + act.float() @ w[:, PAD:].float().t()).bfloat16()


@pytest.mark.parametrize("m", M_ALL)
@pytest.mark.parametrize("rank", [0, 7, 15])
def test_k3_decode_gemv_tail(rank, m):
    ops = _ops()
    w = _tail_weight()
    lat8, act8 = _rows(LATENT, 4, 0.8), _rows(ACT, 5, 0.5)
    lat, act, lo = lat8[:m].contiguous(), act8[:m].contiguous(), rank * WIDTH

    def call(lat_, act_):
        return ops.k3_decode_gemv_tail(lat_, act_, w, lo, WIDTH, EPS, True)

    y = call(lat, act)
    again = call(lat, act)
    y8 = call(lat8, act8)
    fp32 = _tail_fp32(lat, act, w, lo)
    ref = _tail_ref(lat, act, w, lo)
    stock = _tail_stock(lat, act, w, lo)
    deterministic = torch.equal(_bits(y), _bits(again))
    m_invariant = torch.equal(_bits(y), _bits(y8[:m]))
    _report("k3_decode_gemv_tail", f"rank{rank}", m, y, ref, stock, det=deterministic, rows_as_m8=m_invariant,
            vs_fp32=f"{_rel(y, fp32):.2e}")  # fmt: skip
    assert _rel(y, ref) <= TOL and _rel(y, stock) <= TOL and _rel(y, fp32) <= TOL
    assert deterministic and m_invariant


@pytest.mark.parametrize("m", [0, 9, 16])
def test_token_limit(m):
    ops = _ops()
    from tensorrt_llm._torch.cute_dsl_kernels.k3_decode_gemv import op

    w = _weight(*SHAPES["kda_o_proj"], 1)
    x = torch.zeros(m, w.shape[1], dtype=torch.bfloat16, device="cuda")
    assert not op.supports(x, w)
    with pytest.raises(ValueError):
        ops.k3_decode_gemv(x, w, True)
    lat = torch.zeros(m, LATENT, dtype=torch.bfloat16, device="cuda")
    act = torch.zeros(m, ACT, dtype=torch.bfloat16, device="cuda")
    assert not op.supports_tail(lat, act, _tail_weight(), WIDTH)
