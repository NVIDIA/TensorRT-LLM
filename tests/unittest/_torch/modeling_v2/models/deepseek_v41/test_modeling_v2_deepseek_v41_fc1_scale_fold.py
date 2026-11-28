# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every routed-expert loader of the family puts an expert's gate and up halves on ONE FC1 global weight scale, the
larger of the checkpoint's `w1` / `w3` `weight_scale_2` (`ops/moe/nvfp4_fc1_scale_fold`): dep2's, dep's and tep8's
`load_experts`, and dep's `load_megamoe_experts`.

A synthetic checkpoint (the checkpoint's own names) holds experts whose up scale is 1.5x the gate's, the reverse, a
15x ratio, and equal scales, every half with an amax block scale (448). Proved, per loader: the FC1 alpha the runner
gets is built from the larger scale, the stored block scales are finite (scaling 448 UP by 1.5 and casting to e4m3
gives NaN), each half's block scale x the shared scale equals the checkpoint's block scale x its own scale within one
e4m3 rounding, and an expert with equal scales keeps its bytes. The runners' scale swizzles (a CUDA op, and dep2's
CuTe-DSL layout) are replaced by plain concatenation so the stored scales can be read back; the rest is the loaders'
own code. Imports the targets, so a CUDA build is needed."""

import importlib
from types import SimpleNamespace as NS

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="imports the targets (CUDA build)"
)

_FAMILY = "tensorrt_llm._torch.modeling_v2.models.deepseek_v41"
DEV = "cuda"
PREFIX = "layers.3.ffn.experts."
E4M3 = torch.float8_e4m3fn
RTOL = 2.0**-4 * 1.01  # one round-to-nearest of a normal e4m3 value, plus float32 slack
HIDDEN, INTER = 256, 128
# (ws1, ws3) per expert: up larger by 1.5, gate larger by 1.5, a 15x ratio, equal
PAIRS = [(0.01, 0.015), (0.015, 0.01), (0.002, 0.03), (0.01, 0.01)]


def _block_scales(shape, g):
    x = torch.exp2(torch.rand(shape, generator=g) * 10.807 - 2.0).clamp(max=448.0)
    x[..., 0] = 448.0  # the amax block of every row
    return x.to(E4M3)


def _checkpoint(seed=0):
    g = torch.Generator().manual_seed(seed)
    w = {}
    for e, (ws1, ws3) in enumerate(PAIRS):
        q = f"{PREFIX}{e}."
        for name, (n, k), ws2 in (
            ("w1", (INTER, HIDDEN), ws1),
            ("w3", (INTER, HIDDEN), ws3),
            ("w2", (HIDDEN, INTER), 0.01),
        ):
            w[q + f"{name}.weight"] = torch.randint(
                0, 256, (n, k // 2), generator=g, dtype=torch.uint8
            )
            w[q + f"{name}.weight_scale"] = _block_scales((n, k // 16), g)
            w[q + f"{name}.input_scale"] = 0.002 + 0.01 * torch.rand((), generator=g)
            w[q + f"{name}.weight_scale_2"] = torch.tensor(ws2)
    return w


def _i1(ck):
    return max(
        max(float(ck[f"{PREFIX}{e}.w1.input_scale"]), float(ck[f"{PREFIX}{e}.w3.input_scale"]))
        for e in range(len(PAIRS))
    )


def _check_halves(gate, up, ck, e):
    """`gate`, `up` `[I, H/16]` e4m3 as stored for expert `e`."""
    ws1, ws3 = PAIRS[e]
    ws13 = max(ws1, ws3)
    for got, w, own in ((gate, "w1", ws1), (up, "w3", ws3)):
        orig = ck[f"{PREFIX}{e}.{w}.weight_scale"].to(DEV)
        assert torch.isfinite(got.float()).all(), (e, w)
        torch.testing.assert_close(
            got.float() * torch.tensor(ws13, device=DEV),
            orig.float() * torch.tensor(own, device=DEV),
            rtol=RTOL,
            atol=0,
        )
        if ws1 == ws3:
            assert torch.equal(got.view(torch.uint8), orig.view(torch.uint8))


def _runner_moe(inter_t):
    e = len(PAIRS)

    def f32(*shape):
        return torch.zeros(shape, dtype=torch.float32, device=DEV)

    return NS(
        cfg=NS(n_routed_experts=e, moe_intermediate_size=INTER, swiglu_limit=10.0),
        inter_t=inter_t, tp=1, tp_rank=0, offset=0, local=e,
        o1_gate=f32(e), o1=f32(e), o2=f32(e), clamp=f32(e), g1=f32(1), fc2_global=f32(1), fc2_alpha=f32(e),
        fc1_w=torch.zeros(e, 2 * inter_t, HIDDEN // 2, dtype=torch.uint8, device=DEV),
        fc1_s=torch.zeros(e, 2 * inter_t, HIDDEN // 16, dtype=E4M3, device=DEV),
        fc2_w=torch.zeros(e, HIDDEN, inter_t // 2, dtype=torch.uint8, device=DEV),
        fc2_s=torch.zeros(e, HIDDEN, inter_t // 16, dtype=E4M3, device=DEV),
    )  # fmt: skip


@pytest.mark.parametrize("target", ["dep", "tep8_moe_tp4_ep2"])
def test_trtllm_runner_loaders_share_the_larger_fc1_scale(target, monkeypatch):
    """dep / tep8 `load_experts` (`fp4_block_scale_moe_runner`): `o1_gate`, `o1`, `clamp` from the larger scale."""
    wts = importlib.import_module(f"{_FAMILY}.flash_nvfp4__sm_103__{target}.weights")
    loaders = importlib.import_module(
        f"{_FAMILY}.loaders"
    )  # the dep / tep8 expert loader is the family's
    monkeypatch.setattr(
        loaders, "swizzle", lambda x: x
    )  # the runner's 128x4 swizzle: keep rows readable
    inter_t = wts.padded_inter(INTER, 1)
    ck = _checkpoint()
    moe = _runner_moe(inter_t)
    wts.load_experts(moe, ck, PREFIX, DEV)

    ws13 = torch.tensor([max(p) for p in PAIRS], device=DEV)
    i1 = _i1(ck)
    torch.testing.assert_close(moe.o1_gate, i1 * ws13, rtol=1e-6, atol=0)
    torch.testing.assert_close(moe.clamp, 10.0 / (i1 * ws13), rtol=1e-6, atol=0)
    isc2 = torch.stack([ck[f"{PREFIX}{e}.w2.input_scale"] for e in range(len(PAIRS))]).to(DEV)
    # o1 = the FC1 alpha over FC2's activation scale: the layer's largest (one grid for the gated intermediate)
    fc2_in = isc2.max()
    torch.testing.assert_close(moe.o1, i1 * ws13 / fc2_in, rtol=1e-5, atol=0)
    inv = torch.argsort(loaders.fc1_perm(2 * inter_t, DEV))  # undo the runner's row order
    for e in range(len(PAIRS)):
        s = moe.fc1_s[e].view(torch.uint8)[inv].view(E4M3)  # [up | gate], each padded to inter_t
        _check_halves(s[inter_t : inter_t + INTER], s[:INTER], ck, e)


def test_dep2_loader_shares_the_larger_fc1_scale(monkeypatch):
    """dep2 `load_experts` (CuTe-DSL grouped GEMMs): `o1_gate` from the larger scale."""
    wts = importlib.import_module(f"{_FAMILY}.flash_nvfp4__sm_103__dep2.weights")

    def layout(up_w, up_s, gate_w, gate_s, down_w, down_s):  # plain [up | gate], no CuTe-DSL tiling
        return torch.cat([up_w, gate_w]), torch.cat([up_s, gate_s]), down_w, down_s

    monkeypatch.setattr(wts, "cute_dsl_moe_layout", layout)
    inter_t = wts.padded_inter(INTER, 1)
    ck = _checkpoint(1)
    moe = _runner_moe(inter_t)
    wts.load_experts(moe, ck, PREFIX, DEV)

    ws13 = torch.tensor([max(p) for p in PAIRS], device=DEV)
    torch.testing.assert_close(moe.o1_gate, _i1(ck) * ws13, rtol=1e-6, atol=0)
    for e in range(len(PAIRS)):
        s = moe.fc1_s[e]
        _check_halves(s[inter_t : inter_t + INTER], s[:INTER], ck, e)


def test_dep_megamoe_loader_shares_the_larger_fc1_scale():
    """dep `load_megamoe_experts`: `fc1_alpha` from the larger scale, the packed FC1 scales on it."""
    wts = importlib.import_module(f"{_FAMILY}.flash_nvfp4__sm_103__dep.weights")
    e = len(PAIRS)
    ck = _checkpoint(2)
    got = {}
    moe = NS(
        mc=NS(local_slots=e, ep_rank=0, num_experts=e, hidden=HIDDEN, inter=INTER),
        assignments=list(range(e)),
        set_experts=lambda fc1, fc1_sf, fc2, fc2_sf, scalars, device: got.update(
            fc1_sf=fc1_sf, scalars=scalars
        ),
    )
    wts.load_megamoe_experts(moe, ck, PREFIX, DEV)

    ws13 = torch.tensor([max(p) for p in PAIRS], device=DEV)
    torch.testing.assert_close(got["scalars"]["fc1_alpha"], _i1(ck) * ws13, rtol=1e-6, atol=0)
    # invert the kernel's FC1 scale layout: 128x4 atoms (by the destination offset `megamoe_pack_experts`'s
    # reference documents), then gate / up interleaved in groups of 16 rows
    rows, cols = 2 * INTER, HIDDEN // 16
    r = torch.arange(rows, device=DEV)[:, None]
    c = torch.arange(cols, device=DEV)[None, :]
    dest = (
        (((r // 128) * -(-cols // 4) + c // 4) * 32 + r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4
    )
    sf = got["fc1_sf"][:, dest.reshape(-1)].reshape(e, INTER // 16, 2, 16, cols)
    for x in range(e):
        gate = sf[x, :, 0].reshape(INTER, cols).view(E4M3)
        up = sf[x, :, 1].reshape(INTER, cols).view(E4M3)
        _check_halves(gate, up, ck, x)
