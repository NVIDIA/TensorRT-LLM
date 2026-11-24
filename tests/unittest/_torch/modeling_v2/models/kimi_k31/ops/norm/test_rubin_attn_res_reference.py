# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU test of the K3.1 Rubin attn_res entries: their references on small shapes, their guards
and their declarations. No kernel runs here (the GPU tests drive the cells)."""

import inspect

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm._rubin_comm_common import (  # noqa: E501
    lamport_allreduce_reference,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm import (
    _rubin_norm_common as common,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm import (
    rubin_attn_res_lamport_add_rmsnorm as lam,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm import (
    rubin_attn_res_moe_add_rmsnorm as moe,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm import (
    rubin_attn_res_rmsnorm as rms,
)

pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 7),
        reason="K3.1 Rubin (sm_107) entries; the CPU stage runs this file",
    ),
]

ENTRIES = {
    "rubin_attn_res_rmsnorm": rms.rubin_attn_res_rmsnorm,
    "rubin_attn_res_add_rmsnorm": rms.rubin_attn_res_add_rmsnorm,
    "rubin_attn_res_moe_add_rmsnorm": moe.rubin_attn_res_moe_add_rmsnorm,
    "rubin_attn_res_moe_add_rmsnorm_mx": moe.rubin_attn_res_moe_add_rmsnorm_mx,
    "rubin_attn_res_lamport_add_rmsnorm": lam.rubin_attn_res_lamport_add_rmsnorm,
}


def _rows(m, h, n, seed=0):
    g = torch.Generator().manual_seed(seed)

    def rn(*shape, sc=1.0):
        return (torch.randn(*shape, generator=g) * sc).to(torch.bfloat16)

    return dict(
        layer=rn(m, h),
        addend=rn(m, h, sc=0.5),
        shared=rn(m, h, sc=0.5),
        block=rn(n - 1, m, h),
        res_w=rn(h, sc=0.05),
        rms_w=(1.0 + rn(h, sc=0.1).float()).to(torch.bfloat16),
        out_w=(1.0 + rn(h, sc=0.1).float()).to(torch.bfloat16),
    )


def _ulps(a: torch.Tensor, b: torch.Tensor) -> int:
    """Largest distance in bf16 steps between same-signed values (sign-magnitude bits)."""
    ia, ib = a.view(torch.int16).to(torch.int32), b.view(torch.int16).to(torch.int32)
    ia = torch.where(ia < 0, -(ia & 0x7FFF), ia)
    ib = torch.where(ib < 0, -(ib & 0x7FFF), ib)
    return int((ia - ib).abs().max())


def _mix64(layer, block, res_w, rms_w, eps):
    v = torch.cat((block, layer.unsqueeze(0)), dim=0).double()
    r = torch.rsqrt(v.square().mean(-1, keepdim=True) + eps)
    logits = (v * r * (res_w.double() * rms_w.double())).sum(-1)
    return (torch.softmax(logits, 0).unsqueeze(-1) * v).sum(0)


@pytest.mark.parametrize("name", sorted(ENTRIES))
def test_entry_declares_its_cells(name) -> None:
    entry = ENTRIES[name]
    assert entry.CELLS, "no cells"
    whys = [c.why for c in entry.CELLS]
    assert all(whys) and len(set(whys)) == len(whys), "every cell needs its own why"
    specs = [repr(sorted(c.spec.items())) for c in entry.CELLS]
    assert len(set(specs)) == len(specs), "a spec is listed twice"
    # early_trigger is a compile-time constant of the kernel: keyword-only, no default, so every
    # call site states it.
    early = inspect.signature(type(entry).__call__).parameters["early_trigger"]
    assert early.kind is inspect.Parameter.KEYWORD_ONLY
    assert early.default is inspect.Parameter.empty


def test_single_gpu_lamport_cells_have_no_finish_flags() -> None:
    """One-rank cells read a finished call; the others finish one over a tray's ranks, and the MoE
    tail's (all-gather rows) and a block boundary's (``replace``) also over helix8's two trays."""
    for entry in (lam.rubin_attn_res_lamport_add_rmsnorm,):
        for cell in entry.CELLS:
            if cell.spec["world"] == 1:
                assert "two_shot" not in cell.spec and "early_poll" not in cell.spec
            elif cell.spec.get("finish_ag") or cell.spec.get("replace"):
                assert cell.spec["world"] in (lam.WORLD, lam.TAIL_WORLD), cell.why
            else:
                assert cell.spec["world"] == lam.WORLD


@pytest.mark.parametrize("n", [2, 3, 5, 9])
def test_mix_reference_is_the_fp32_mixture_rounded_once(n) -> None:
    t = _rows(3, 64, n, seed=n)
    got = common.mix_reference(t["layer"], t["block"], t["res_w"], t["rms_w"], 1e-5)
    want = _mix64(t["layer"], t["block"], t["res_w"], t["rms_w"], 1e-5).to(torch.bfloat16)
    assert got.dtype == torch.bfloat16 and got.shape == t["layer"].shape
    assert _ulps(got, want) <= 1


def test_rmsnorm_reference_rounds_the_normalized_value_and_the_product() -> None:
    t = _rows(4, 64, 2, seed=1)
    x = t["layer"]
    got = common.rmsnorm_reference(x, t["out_w"], 1e-5)
    xf = x.float()
    normed = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-5)).to(torch.bfloat16)
    want = (normed.float() * t["out_w"].float()).to(torch.bfloat16)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


def test_add_reference_is_a_bf16_add_and_flushes_denormals_when_fast() -> None:
    t = _rows(2, 64, 2, seed=2)
    a, b = t["layer"], t["addend"]
    assert torch.equal(common.bf16_add(a, b, ftz=True).view(torch.int16), (a + b).view(torch.int16))
    tiny = torch.full((1, 8), 2.0**-130, dtype=torch.bfloat16)  # a bf16 (and fp32) denormal
    zero = torch.zeros_like(tiny)
    assert torch.equal(common.bf16_add(tiny, zero, ftz=False), tiny)
    assert torch.equal(common.bf16_add(tiny, zero, ftz=True), zero)


def test_add_rmsnorm_reference_mixes_the_updated_prefix() -> None:
    t = _rows(3, 64, 4, seed=3)
    w = (t["res_w"], t["rms_w"], t["out_w"], 1e-5, 1e-5)
    updated, out = rms.rubin_attn_res_add_rmsnorm.reference(
        t["layer"], t["addend"], t["block"], *w, early_trigger=True
    )
    assert torch.equal(updated, t["layer"] + t["addend"])
    expect = rms.rubin_attn_res_rmsnorm.reference(updated, t["block"], *w, early_trigger=False)
    assert torch.equal(out, expect)


def test_moe_residual_reference_is_two_round_to_nearest_adds() -> None:
    t = _rows(3, 64, 3, seed=4)
    got = moe.moe_residual_reference(t["layer"], t["addend"], t["shared"])
    assert torch.equal(got, t["layer"] + (t["addend"] + t["shared"]))


def test_mxfp8_reference_scales_codes_and_atoms() -> None:
    m, k = 3, 128
    g = torch.Generator().manual_seed(5)
    x = (torch.randn(m, k, generator=g) * 3.0).to(torch.bfloat16)
    x[0, :32] = 0  # an all-zero group takes the amax floor
    xq, xs = common.mxfp8_reference(x)
    assert xq.dtype == torch.float8_e4m3fn and xq.shape == (m, k)
    assert xs.dtype == torch.uint8 and xs.shape == (m, k // 32)
    # amax <= 448 * 2^(e - 127): the codes never saturate; decoding recovers x to e4m3 precision.
    amax = x.float().abs().view(m, k // 32, 32).amax(-1)
    assert bool((amax <= 448.0 * torch.pow(2.0, xs.float() - 127)).all())
    decoded = xq.float().view(m, k // 32, 32) * torch.pow(2.0, xs.float() - 127).unsqueeze(-1)
    err = (decoded - x.float().view(m, k // 32, 32)).abs()
    assert bool((err <= amax.unsqueeze(-1) * 2.0**-3 + 1e-30).all())
    assert int(xs[0, 0]) == ((0x38D2 >> 7) - 8 + int((0x38D2 & 0x7F) > 0x60))
    # The atom layout puts the same exponents at (r % 32) * 16 + (r % 128 // 32) * 4 + g % 4.
    _, atoms = common.mxfp8_reference(x, True, 1)
    assert atoms.shape == ((k // 128) * 512,)
    for r in range(m):
        for grp in range(k // 32):
            off = (grp // 4) * 512 + (r % 32) * 16 + (r % 128 // 32) * 4 + grp % 4
            assert int(atoms[off]) == int(xs[r, grp])


def test_lamport_references_add_the_rank_ordered_sum() -> None:
    t = _rows(3, 64, 5, seed=6)
    g = torch.Generator().manual_seed(7)
    partials = [(torch.randn(3, 64, generator=g) * 0.25).to(torch.bfloat16) for _ in range(4)]
    w = (t["res_w"], t["rms_w"], t["out_w"], 1e-5, 1e-5)
    addend = lamport_allreduce_reference(partials, 4)
    for add_rn in (False, True):
        updated, out = lam.rubin_attn_res_lamport_add_rmsnorm.reference(
            t["layer"], partials, t["block"], *w, 4, add_rn
        )
        assert torch.equal(updated, common.bf16_add(t["layer"], addend, ftz=not add_rn))
        expect = common.mix_rmsnorm_reference(updated, t["block"], *w)
        assert torch.equal(out, expect)


def test_guards_refuse_what_the_kernel_misreads() -> None:
    t = _rows(2, 7168, 3, seed=8)
    w = (t["res_w"], t["rms_w"], t["out_w"], 1e-5, 1e-5)
    # Host tensors: the launcher would hand their addresses to the GPU.
    with pytest.raises(AssertionError, match="device"):
        rms.rubin_attn_res_rmsnorm.is_valid(t["layer"], t["block"], *w, early_trigger=True)
    small = _rows(2, 64, 3, seed=9)
    with pytest.raises(AssertionError, match=r"\[M, 7168\]"):
        rms.rubin_attn_res_rmsnorm.is_valid(small["layer"], small["block"], *w, early_trigger=True)
    with pytest.raises(AssertionError, match="early_poll"):
        common.check_finish(0, 0, 0, True, False)
    with pytest.raises(AssertionError, match="rank"):
        common.check_finish(4, 4, 4, False, False)


class _Launcher:
    """Stands in for a kernel launcher: records what ``__call__`` hands it."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "launched"


# The Lamport entry's trailing launcher flags, passed by name.
FINISH_KEYWORDS = (
    "finish_world",
    "finish_chunk",
    "finish_two_shot",
    "mc_base",
    "rank",
    "add_rn",
    "early_poll",
    "entry_trigger",
    "finish_ag",
)


@pytest.mark.parametrize(
    "module,launcher,entry,n_positional,keywords",
    [
        (rms, "_launch_rmsnorm", "rubin_attn_res_rmsnorm", 7, ()),
        (rms, "_launch_add_rmsnorm", "rubin_attn_res_add_rmsnorm", 8, ()),
        (moe, "_launch_moe_add", "rubin_attn_res_moe_add_rmsnorm", 9, ()),
        (
            moe,
            "_launch_moe_add_mx",
            "rubin_attn_res_moe_add_rmsnorm_mx",
            9,
            ("entry_trigger", "mx_atom", "mx_atom_blocks", "mx_exact"),
        ),
        (lam, "_launch_add", "rubin_attn_res_lamport_add_rmsnorm", 9, FINISH_KEYWORDS),
    ],
)
def test_call_hands_every_argument_to_the_launcher(
    monkeypatch, module, launcher, entry, n_positional, keywords
) -> None:
    """``__call__`` is the launch and nothing else: the kernel's leading arguments in order, the
    flags by name (their position may change when the kernel file moves), early_trigger, and the
    other keyword-only arguments (``updated``, ``replace``) by name."""
    stub = _Launcher()
    monkeypatch.setattr(module, launcher, stub, raising=False)
    op = getattr(module, entry)
    params = inspect.signature(type(op).__call__).parameters
    names = [
        p
        for p, v in params.items()
        if p not in ("self", "early_trigger") and v.kind is not inspect.Parameter.KEYWORD_ONLY
    ]
    extra = {
        p: object()
        for p, v in params.items()
        if p != "early_trigger" and v.kind is inspect.Parameter.KEYWORD_ONLY
    }
    args = [object() for _ in names]
    assert op(*args, early_trigger=True, **extra) == "launched"
    ((got_args, got_kwargs),) = stub.calls
    assert list(got_args) == args[:n_positional]
    assert got_kwargs.pop("early_trigger") is True
    assert set(got_kwargs) == set(keywords) | set(extra)
    for name in keywords:
        assert got_kwargs[name] is args[names.index(name)]
    for name, value in extra.items():
        assert got_kwargs[name] is value
