# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the K3.1 Rubin entries rubin_attn_res_rmsnorm and rubin_attn_res_add_rmsnorm.

Every cell: shapes and dtypes, bit-identical repeated calls, untouched inputs, the updated prefix
bit for bit against the reference's bf16 add, and the output against the entry's fp32
reference (cosine / relative L2) and against the C++ per-token op the kernel replaces (one bf16
step in a small fraction of the elements). Needs the kernel at its family path
(``kernels/norm/rubin_attn_res.py``).
"""

import zlib

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="sm_107 kernel (VR200)",
)

import tensorrt_llm._torch.custom_ops  # noqa: E402,F401 — registers the C++ attn_res ops
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.norm import (  # noqa: E402
    rubin_attn_res as kernel,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm import (  # noqa: E402
    _rubin_norm_common as common,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm.rubin_attn_res_rmsnorm import (  # noqa: E402,E501
    rubin_attn_res_add_rmsnorm,
    rubin_attn_res_rmsnorm,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

H = common.HIDDEN
EPS = common.EPS
# Against the fp32 reference: the kernel rounds where the reference rounds; fast-math exp2 /
# rsqrt and its reduction order remain (the C++ ops measured <= 6e-6 relative L2 there). A
# dropped candidate moves the output by tens of percent.
MIN_COSINE, MAX_RELATIVE_L2 = 0.9999, 1e-3
# Against the C++ op: at most two bf16 steps of an element (plus 2^-8 of the row RMS where the
# mixture cancels), in at most 2 % of the elements (the kernel's own test measures this).
MAX_DIFF_FRACTION = 2e-2


def _inputs(rows: int, candidates: int, seed: int, scale: float = 1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rn(*shape, sc=1.0):
        return (torch.randn(*shape, generator=g, device="cuda") * sc).to(torch.bfloat16)

    return dict(
        layer=rn(rows, H, sc=scale),
        addend=rn(rows, H, sc=0.5 * scale),
        block=rn(candidates - 1, rows, H, sc=scale),
        res_w=rn(H, sc=0.05),
        rms_w=(1.0 + rn(H, sc=0.1).float()).to(torch.bfloat16),
        out_w=(1.0 + rn(H, sc=0.1).float()).to(torch.bfloat16),
    )


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16)


def _assert_reference_close(out: torch.Tensor, ref: torch.Tensor, what: str) -> None:
    a, e = out.float().flatten(), ref.float().flatten()
    cosine = torch.nn.functional.cosine_similarity(a, e, dim=0).item()
    rel = ((a - e).norm() / (e.norm() + 1e-12)).item()
    assert cosine > MIN_COSINE and rel < MAX_RELATIVE_L2, f"{what}: cos {cosine:.7f} rel {rel:.2e}"


def _assert_cpp_close(out: torch.Tensor, cpp: torch.Tensor, what: str) -> None:
    a, e = out.float(), cpp.float()
    diff = (a - e).abs()
    bound = (
        torch.maximum(a.abs(), e.abs()) * 2.0**-6 + 2.0**-8 * e.pow(2).mean(-1, keepdim=True).sqrt()
    )
    assert bool((diff <= bound).all()), (what, float((diff - bound).max()))
    frac = float((diff > 0).float().mean())
    assert frac <= MAX_DIFF_FRACTION, (what, frac)


def _eps(spec) -> tuple[float, float]:
    return spec.get("eps", EPS), spec.get("out_eps", EPS)


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


def test_constants_match_the_kernel() -> None:
    assert (common.HIDDEN, common.MAX_TOKENS) == (kernel.HIDDEN_SIZE, kernel.MAX_TOKENS)


@pytest.mark.parametrize(
    "cell", rubin_attn_res_rmsnorm.CELLS, ids=[c.why[:56] for c in rubin_attn_res_rmsnorm.CELLS]
)
def test_rmsnorm_cells(cell, certifying) -> None:  # noqa: F811 — fixture resolved by name
    certifying(rubin_attn_res_rmsnorm)
    spec = cell.spec
    t = _inputs(spec["rows"], spec["candidates"], _seed(cell))
    eps, out_eps = _eps(spec)
    args = (t["layer"], t["block"], t["res_w"], t["rms_w"], t["out_w"], eps, out_eps)
    before = [x.clone() for x in args[:5]]
    out = rubin_attn_res_rmsnorm(*args, early_trigger=spec["early_trigger"])
    again = rubin_attn_res_rmsnorm(*args, early_trigger=spec["early_trigger"])
    ref = rubin_attn_res_rmsnorm.reference(*args, early_trigger=spec["early_trigger"])
    assert out.shape == t["layer"].shape and out.dtype == torch.bfloat16
    assert torch.equal(_bits(out), _bits(again)), "two identical calls disagree"
    _assert_reference_close(out, ref, "output")
    m, k = spec["rows"], spec["candidates"] - 1
    cpp = torch.ops.trtllm.attn_res_rmsnorm_fwd(
        t["layer"].view(m, 1, H), t["block"].view(k, m, 1, H), *args[2:]
    ).view(m, H)
    _assert_cpp_close(out, cpp, "output vs C++")
    for old, new in zip(before, args[:5]):
        assert torch.equal(_bits(old), _bits(new)), "an input was mutated"


@pytest.mark.parametrize(
    "cell",
    rubin_attn_res_add_rmsnorm.CELLS,
    ids=[c.why[:56] for c in rubin_attn_res_add_rmsnorm.CELLS],
)
def test_add_rmsnorm_cells(cell, certifying) -> None:  # noqa: F811 — fixture resolved by name
    certifying(rubin_attn_res_add_rmsnorm)
    spec = cell.spec
    t = _inputs(spec["rows"], spec["candidates"], _seed(cell))
    eps, out_eps = _eps(spec)
    args = (t["layer"], t["addend"], t["block"], t["res_w"], t["rms_w"], t["out_w"], eps, out_eps)
    before = [x.clone() for x in args[:6]]
    updated, out = rubin_attn_res_add_rmsnorm(*args, early_trigger=spec["early_trigger"])
    updated2, out2 = rubin_attn_res_add_rmsnorm(*args, early_trigger=spec["early_trigger"])
    ref_updated, ref_out = rubin_attn_res_add_rmsnorm.reference(
        *args, early_trigger=spec["early_trigger"]
    )
    for x in (updated, out):
        assert x.shape == t["layer"].shape and x.dtype == torch.bfloat16
    assert torch.equal(_bits(updated), _bits(updated2)) and torch.equal(_bits(out), _bits(out2))
    assert torch.equal(_bits(updated), _bits(ref_updated)), "the add is not the reference's"
    assert torch.equal(_bits(updated), _bits(t["layer"] + t["addend"])), "not a torch bf16 add"
    _assert_reference_close(out, ref_out, "output")
    m, k = spec["rows"], spec["candidates"] - 1
    cpp_updated, cpp_out = torch.ops.trtllm.attn_res_add_rmsnorm_fwd(
        t["layer"].view(m, 1, H), t["addend"].view(m, 1, H), t["block"].view(k, m, 1, H), *args[3:]
    )
    assert torch.equal(_bits(updated), _bits(cpp_updated.view(m, H)))
    _assert_cpp_close(out, cpp_out.view(m, H), "output vs C++")
    for old, new in zip(before, args[:6]):
        assert torch.equal(_bits(old), _bits(new)), "an input was mutated"


def test_band_refuses_a_dropped_candidate() -> None:
    t = _inputs(8, 6, seed=5)
    w = (t["res_w"], t["rms_w"], t["out_w"], EPS, EPS)
    out = rubin_attn_res_rmsnorm(t["layer"], t["block"], *w, early_trigger=True)
    wrong = rubin_attn_res_rmsnorm.reference(t["layer"], t["block"][1:], *w, early_trigger=True)
    with pytest.raises(AssertionError):
        _assert_reference_close(out, wrong, "dropped candidate")


def test_early_trigger_gives_the_same_bits() -> None:
    t = _inputs(96, 9, seed=11)
    args = (t["layer"], t["addend"], t["block"], t["res_w"], t["rms_w"], t["out_w"], EPS, EPS)
    on = rubin_attn_res_add_rmsnorm(*args, early_trigger=True)
    off = rubin_attn_res_add_rmsnorm(*args, early_trigger=False)
    for a, b in zip(on, off):
        assert torch.equal(_bits(a), _bits(b))


def test_guard_refuses_what_the_kernel_misreads() -> None:
    t = _inputs(8, 4, seed=3)
    w = (t["res_w"], t["rms_w"], t["out_w"], EPS, EPS)
    bad_inputs = (
        (t["layer"].float(), t["block"]),  # fp32 rows read as bf16
        (t["layer"].t().contiguous().t(), t["block"]),  # not contiguous
        (t["layer"], t["block"][:, :4]),  # block rows do not match
    )
    for layer, block in bad_inputs:
        with pytest.raises(AssertionError):
            rubin_attn_res_rmsnorm.is_valid(layer, block, *w, early_trigger=True)


def test_cuda_graph_replay_matches_eager() -> None:
    for rows, candidates in ((8, 5), (96, 9)):
        t = _inputs(rows, candidates, seed=7)
        args = (t["layer"], t["addend"], t["block"], t["res_w"], t["rms_w"], t["out_w"], EPS, EPS)
        eager = rubin_attn_res_add_rmsnorm(*args, early_trigger=True)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = rubin_attn_res_add_rmsnorm(*args, early_trigger=True)
        graph.replay()
        for x in captured:
            x.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        for a, b in zip(eager, captured):
            assert torch.equal(_bits(a), _bits(b)), f"rows={rows} N={candidates}"
