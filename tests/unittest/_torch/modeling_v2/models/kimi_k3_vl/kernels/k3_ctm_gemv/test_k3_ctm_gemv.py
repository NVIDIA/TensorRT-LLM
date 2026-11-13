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
"""trtllm::k3_ctm_gemv_gated, trtllm::k3_ctm_gemv_tail and trtllm::k3_situ_mul at the Kimi K3 TP16 per-rank shapes,
at every M in 1..8: error against an fp64 product of the unfused activation (torch's bf16 roundings) and against the
stock path (cuBLAS F.linear; the Triton SiTU), run-to-run identical bits, each M's rows bit-identical to the same rows
of the 8-row call. The CTM GEMVs that a k3_decode_gemv op also computes return its bits: trtllm::k3_ctm_gemv and the
gated GEMV at split 1, and the tail. The supports() checks refuse 0, 9 and 16 rows."""

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

# k3_ctm_gemv at split 1: (N, K, push). The MLA o_proj, the drafter o_proj (tuned and synthetic).
SPLIT1 = {
    "o_proj_s1": (7168, 768, False),
    "drafter_o_proj": (7168, 384, True),
    "drafter_o_proj_dummy": (7168, 256, True),
}
# k3_ctm_gemv_long: (N, K, sig_col0, split, ring, push), as each call site passes them.
LONG = {
    "mla_qkv_a_gate": (
        2880,
        7168,
        2112,
        6,
        6,
        True,
    ),  # [W_a; W_g], the gate columns as bf16(sigmoid)
    "dense_gate_up": (4224, 7168, -1, 4, 5, False),
    "dense_down": (7168, 2112, -1, 2, 6, False),  # K 2112 ends in a half k-tile
    "drafter_qkv": (512, 7168, -1, 8, 6, True),
    "drafter_gate_up": (1792, 7168, -1, 8, 6, True),
    "drafter_gate_up_dummy": (1536, 7168, -1, 8, 6, True),
    "kda_qkvg": (
        3208,
        7168,
        -1,
        5,
        6,
        True,
    ),  # KDA q/k/v/g/f_a/b: 25 whole tiles and an 8-row last one
}
AG_COLS, G_COL0, K_O = (
    2880,
    2112,
    768,
)  # MLA: [q_a 1536 | kv_a 576 | gate 768] rows of the fused projection
HIDDEN, LATENT, WIDTH, PAD, ACT = 7168, 3584, 224, 256, 384
EPS = 1e-6
SITU = {
    "k3": (4.0, 25.0),
    "plain": (1.0, None),
}  # (beta, linear_beta): the K3 checkpoint's, and the defaults


def _ops():
    import tensorrt_llm  # noqa: F401  (registers the trtllm ops)
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3_ctm_gemv import (
        op,  # noqa: F401
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3_decode_gemv import (
        op as _op,  # noqa: F401
    )

    return torch.ops.trtllm


def _ctm():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3_ctm_gemv import (
        op,
    )

    return op


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _rel(y: torch.Tensor, ref: torch.Tensor) -> float:
    return (y.double() - ref.double()).abs().max().item() / ref.double().abs().max().item()


def _report(op, case, m, y, ref, stock=None, **flags):
    extra = "" if stock is None else f" vs_stock={_rel(y, stock):.2e}"
    marks = " ".join(f"{k}={v}" for k, v in flags.items())
    abs_err = (y.double() - ref.double()).abs().max().item()
    print(
        f"OPCHECK op={op} case={case} M={m} abs={abs_err:.3e} rel={_rel(y, ref):.3e}{extra} {marks}"
    )


@functools.lru_cache(maxsize=None)
def _weight(n: int, k: int, seed: int, scale: float = 0.02) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(n, k, generator=gen, device="cuda") * scale).bfloat16()


@functools.lru_cache(maxsize=None)
def _rows(k: int, seed: int, scale: float = 1.0) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(8, k, generator=gen, device="cuda") * scale).bfloat16()


def _checks(call, args8, m):
    """The M-row call (the first M rows of each 8-row input), a rerun and the 8-row call."""
    args = [a[:m].contiguous() for a in args8]
    y = call(*args)
    again = call(*args)
    y8 = call(*args8)
    return args, y, torch.equal(_bits(y), _bits(again)), torch.equal(_bits(y), _bits(y8[:m]))


@pytest.mark.parametrize("m", M_ALL)
@pytest.mark.parametrize("name", list(SPLIT1))
def test_k3_ctm_gemv_split1_is_k3_decode_gemv(name, m):
    """At split 1 the CTM GEMV returns the bits of k3_decode_gemv on the same rows."""
    ops = _ops()
    n, k, push = SPLIT1[name]
    w = _weight(n, k, 1, 0.03)
    x = _rows(k, 2)[:m].contiguous()
    y = ops.k3_ctm_gemv(x, w, True, 1, push)
    assert torch.equal(_bits(y), _bits(ops.k3_decode_gemv(x, w, True)))


@pytest.mark.parametrize("m", M_ALL)
@pytest.mark.parametrize("gate_sigmoid", [False, True], ids=["sigmoid_given", "sigmoid_in_kernel"])
@pytest.mark.parametrize("split", [2, 1])
def test_k3_ctm_gemv_gated(split, gate_sigmoid, m):
    """The MLA o_proj with its output gate: (a * sigmoid(g)) @ W^T, g the gate columns of the fused projection rows
    (or a * s with s = bf16(sigmoid(g)) already there, the in-model form), against the unfused torch roundings."""
    ops = _ops()
    w = _weight(HIDDEN, K_O, 3, 0.03)
    a8 = _rows(K_O, 4)
    if gate_sigmoid:
        ag8 = _rows(AG_COLS, 5, 2.0)
    else:
        gen = torch.Generator(device="cuda").manual_seed(6)
        ag8 = torch.rand(8, AG_COLS, generator=gen, device="cuda").bfloat16()
    assert _ctm().supports_gated(a8, ag8, G_COL0, w, split)

    def call(a_, ag_):
        return ops.k3_ctm_gemv_gated(a_, ag_, G_COL0, w, True, split, gate_sigmoid)

    (a, ag), y, det, minv = _checks(call, [a8, ag8], m)
    g = ag[:, G_COL0 : G_COL0 + K_O]
    b = a * (g.sigmoid() if gate_sigmoid else g)  # bf16, as the unfused model path
    ref = b.double() @ w.double().t()
    decode = ops.k3_decode_gemv(b, w, True)
    same_decode = torch.equal(_bits(y), _bits(decode))
    _report("k3_ctm_gemv_gated", f"s{split}_{'sig' if gate_sigmoid else 'given'}", m, y, ref, F.linear(b, w),
            det=det, rows_as_m8=minv, eq_unfused_k3_decode_gemv=same_decode)  # fmt: skip
    assert _rel(y, ref) <= TOL and _rel(y, F.linear(b, w)) <= TOL
    assert det and minv
    if split == 1:
        assert same_decode


@pytest.mark.parametrize("m", M_ALL)
@pytest.mark.parametrize("rank", [0, 7, 15])
def test_k3_ctm_gemv_tail(rank, m):
    """The row-parallel MoE tail on the CTM kernel: the bits of k3_decode_gemv_tail."""
    ops = _ops()
    w = _weight(HIDDEN, PAD + ACT, 11, 0.03).clone()
    w[:, WIDTH:PAD] = 0
    lo = rank * WIDTH

    def call(lat_, act_):
        return ops.k3_ctm_gemv_tail(lat_, act_, w, lo, WIDTH, EPS, True)

    (lat, act), y, det, minv = _checks(call, [_rows(LATENT, 12, 0.8), _rows(ACT, 13, 0.5)], m)
    lat64 = lat.double()
    normed = lat64 * torch.rsqrt(lat64.pow(2).mean(dim=1, keepdim=True) + EPS)
    ref = (
        torch.cat([normed[:, lo : lo + WIDTH], act.double()], dim=1)
        @ torch.cat([w[:, :WIDTH], w[:, PAD:]], dim=1).double().t()
    )
    decode = ops.k3_decode_gemv_tail(lat, act, w, lo, WIDTH, EPS, True)
    same_decode = torch.equal(_bits(y), _bits(decode))
    _report(
        "k3_ctm_gemv_tail",
        f"rank{rank}",
        m,
        y,
        ref,
        det=det,
        rows_as_m8=minv,
        eq_k3_decode_gemv_tail=same_decode,
    )
    assert _rel(y, ref) <= TOL
    assert det and minv and same_decode


def _situ_ref(gu, beta, linear_beta):
    """SituAndMul's eager fp32 path (in fp64)."""
    k = gu.shape[1] // 2
    g, u = gu[:, :k].double(), gu[:, k:].double()
    a = beta * torch.tanh(g / beta) * torch.sigmoid(g)
    if linear_beta is not None:
        u = linear_beta * torch.tanh(u / linear_beta)
    return a * u


@pytest.mark.parametrize("m", M_ALL)
@pytest.mark.parametrize("situ", list(SITU))
def test_k3_situ_mul(situ, m):
    """The dense MLP's SiTU-and-mul at TP16 (gate_up [M, 4224] -> [M, 2112]) against the Triton SituAndMul."""
    ops = _ops()
    from tensorrt_llm._torch.modules.situ import SituAndMul

    beta, linear_beta = SITU[situ]
    gu8 = _rows(2 * 2112, 14, 2.0)
    (gu,), y, det, minv = _checks(lambda g_: ops.k3_situ_mul(g_, beta, linear_beta), [gu8], m)
    stock = SituAndMul(beta=beta, linear_beta=linear_beta, use_fused_activation=True)(gu)
    ref = _situ_ref(gu, beta, linear_beta)
    identical = (_bits(y) == _bits(stock)).float().mean().item()
    _report(
        "k3_situ_mul",
        situ,
        m,
        y,
        ref,
        stock,
        det=det,
        rows_as_m8=minv,
        frac_eq_triton=f"{identical:.4f}",
    )
    assert y.shape == (m, 2112)
    assert _rel(y, ref) <= TOL and _rel(y, stock) <= TOL
    assert det and minv


def test_k3_situ_mul_rejects_zero_linear_beta():
    ops = _ops()
    gu = _rows(2 * 2112, 14, 2.0)
    with pytest.raises(ValueError, match="linear_beta"):
        ops.k3_situ_mul(gu, 1.0, 0.0)
    assert ops.k3_situ_mul(gu, 1.0, None).shape == (8, 2112)


@pytest.mark.parametrize("m", [0, 9, 16])
def test_token_limit(m):
    op = _ctm()
    _ops()
    x = torch.zeros(m, 7168, dtype=torch.bfloat16, device="cuda")
    for n, k, _, split, ring, _ in LONG.values():
        assert not op.supports_long(
            torch.zeros(m, k, dtype=torch.bfloat16, device="cuda"), _weight(n, k, 9), split, ring
        )
    with pytest.raises(ValueError):
        torch.ops.trtllm.k3_ctm_gemv_long(x, _weight(2880, 7168, 9), 2112, 6, 6, True, True)
    assert not op.supports(
        torch.zeros(m, 768, dtype=torch.bfloat16, device="cuda"), _weight(7168, 768, 1, 0.03), 1
    )
    assert not op.supports_situ_mul(torch.zeros(m, 4224, dtype=torch.bfloat16, device="cuda"))
    assert not op.supports_swiglu(
        torch.zeros(m, 1792, dtype=torch.bfloat16, device="cuda"), _weight(7168, 896, 7), 2
    )
