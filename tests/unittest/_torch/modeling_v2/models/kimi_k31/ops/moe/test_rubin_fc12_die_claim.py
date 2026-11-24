# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""rubin_fc12's die-local claim variant ``die_claim="uniform"`` (SM107).

The variant changes only how a die-local claim is compiled: the whole scheduler warp runs the
search on values lane 0 broadcasts, lane 0 alone issues the atomics, and the divisions take
FastDivmod divisors built on the host (``kernels/moe/rubin_fc12_claim.py``). It issues the engine
claim's atomics in the same order, so each CTA claims the same tiles and the output equals the
engine kernel's bit for bit wherever FC12 reproduces itself (rows with at most two local experts:
``0 + a + b`` rounds the same in either order).

Off the GPU: the host divisors and their Int32 bound at the TP2 x EP4 (helix8) and EP8 geometries
for both MMA N, the bound at its edge, a model of every rounded-up byte division the claim makes,
the launcher's choice of the variant (and its fallback outside the bound), the cache key, and the
subclass against the engine kernel (``__call__`` and ``kernel`` are the engine's but for the
claim and the ``l2_prefetch`` blocks, which test_rubin_fc12_l2_prefetch.py covers). On SM107:
the exact rows of the variant against the engine claim through the family runner, both
geometries, both MMA N, die schedule on placed weights and forced on plain weights.

A claim that skips or repeats an FC1 tile does not fail an assert: FC2 then waits for an FC1 tile
count it never reaches and the kernel hangs. Run the GPU test under a wall-clock timeout.
"""

import ast
import copy
import importlib.util
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.moe import (
    rubin_fc12_launch as launch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe._rubin_moe_common import (
    token_ordered_sort,
)

_SM107 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
# Certified on SM107 only; other GPUs skip the file, a CPU-only host runs the cpu_only tests.
pytestmark = pytest.mark.skipif(
    torch.cuda.is_available() and not _SM107, reason="certified on SM107 only"
)

HIDDEN, NUM_EXPERTS, TOP_K, TILE = 3584, 896, 16, 128
TWO_MIB = 1 << 21
#: One rank's FC12 window: (local experts, intermediate per rank). helix8 runs MoE TP2 x EP4
#: (224 experts at half of the 3,072 intermediate), the EP8 layout 112 at the full width.
GEOMETRIES = {"tp2ep4": (224, 1536), "ep8": (112, 3072)}
#: The 1-CTA tactics the tuner chooses from, by MMA N (128 is also its fallback).
TACTICS = {
    256: ((TILE, 256, 256), (TILE, 256, 128), (1, 1)),
    128: ((TILE, 128, 256), (TILE, 128, 128), (1, 1)),
}
_PKG = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_ENGINE_KERNEL = (
    _PKG / "_torch/moe/kernels/rubin/rubin_contiguous_grouped_blockscaled_gemm_fused_fc12.py"
)
_CLAIM_KERNEL = Path(launch.__file__).resolve().parent / "rubin_fc12_claim.py"


def _phases(geometry):
    """``((fc1_n, fc1_k), (fc2_n, fc2_k))`` of one rank's window."""
    _, inter = GEOMETRIES[geometry]
    return (2 * inter, HIDDEN), (HIDDEN, inter)


# ---------------------------------------------------------------------------------------------
# No GPU: the divisors, the bound, the launcher's choice
# ---------------------------------------------------------------------------------------------


@pytest.mark.cpu_only
@pytest.mark.parametrize("mma_n", [128, 256])
@pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
def test_divisors_of_the_k31_geometries(geometry, mma_n):
    """The host-side divisor values, as the kernel derives them from the weight shapes; every
    K3.1 phase is far inside the Int32 bound."""
    for n, k in _phases(geometry):
        assert launch.fc12_die_claim_divisors(n, k, mma_n) == (n // mma_n, k // 2 * mma_n)
        assert launch.fc12_die_claim_fits(n, k, mma_n)
    fc1, fc2 = (launch.fc12_die_claim_divisors(n, k, 256) for n, k in _phases(geometry))
    assert (fc1, fc2) == {
        "tp2ep4": ((12, 458752), (14, 196608)),
        "ep8": ((24, 458752), (14, 393216)),
    }[geometry]


@pytest.mark.cpu_only
def test_the_int32_bound_at_its_edge():
    """``tiles_per_m * tile_bytes + 2 MiB + tile_bytes`` must stay below 2^31: with 1 MiB tiles
    (K 8192, MMA N 256) 2044 N tiles end at 2^31 - 1 MiB, 2045 at 2^31; no N tile at all fails."""
    assert launch.fc12_die_claim_divisors(2044 * 256, 8192, 256) == (2044, 1 << 20)
    assert launch.fc12_die_claim_fits(2044 * 256, 8192, 256)
    assert not launch.fc12_die_claim_fits(2045 * 256, 8192, 256)
    assert not launch.fc12_die_claim_fits(128, HIDDEN, 256)


@pytest.mark.cpu_only
@pytest.mark.parametrize("mma_n", [128, 256])
@pytest.mark.parametrize("phase", [0, 1], ids=["fc1", "fc2"])
@pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
def test_rounded_division_stays_below_the_bound(geometry, phase, mma_n):
    """Every byte distance the claim divides by ``tile_bytes`` lies in (0, tiles_per_m *
    tile_bytes + 2 MiB), and it resumes past the current N tile: the bound covers the device's
    Int32 division and the search moves forward."""
    gemm_n, gemm_k = _phases(geometry)[phase]
    tiles_per_m, tile_bytes = launch.fc12_die_claim_divisors(gemm_n, gemm_k, mma_n)
    expert_bytes = gemm_n * gemm_k // 2
    bound = tiles_per_m * tile_bytes + TWO_MIB
    num_local = GEOMETRIES[geometry][0]
    for b_base in (4 << 20, (1 << 40) + (12 << 20), (1 << 44) + (2 << 20)):
        for expert in range(0, num_local, 7):
            first_middle = b_base + expert * expert_bytes + tile_bytes // 2
            for claim_n in range(tiles_per_m):
                middle = first_middle + claim_n * tile_bytes
                next_chunk = ((middle >> 21) + 1) << 21
                numerator = next_chunk - first_middle + tile_bytes - 1
                assert 0 < numerator < bound < 1 << 31
                assert numerator // tile_bytes > claim_n


@pytest.mark.cpu_only
def test_the_launch_takes_the_variant_only_where_it_applies():
    """The variant runs on die-local launches inside its bound; elsewhere the engine claim."""
    (n1, k1), (n2, k2) = _phases("tp2ep4")
    assert launch.fc12_die_claim("uniform", True, n1, k1, n2, k2, 256) == "uniform"
    assert launch.fc12_die_claim("uniform", True, n1, k1, n2, k2, 128) == "uniform"
    assert launch.fc12_die_claim("uniform", False, n1, k1, n2, k2, 256) is None
    assert launch.fc12_die_claim(None, True, n1, k1, n2, k2, 256) is None
    # An FC1 phase outside the bound falls back to the engine claim (same tiles).
    assert launch.fc12_die_claim("uniform", True, 2045 * 256, 8192, n2, k2, 256) is None
    assert launch.fc12_die_claim("uniform", True, n1, k1, 2045 * 256, 8192, 256) is None
    assert launch.DIE_CLAIMS == (None, "uniform")


# ---------------------------------------------------------------------------------------------
# No GPU: the subclass is the engine kernel but for the claim
# ---------------------------------------------------------------------------------------------


def _class(path: Path, name: str) -> ast.ClassDef:
    tree = ast.parse(path.read_text())
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)


def _methods(cls: ast.ClassDef) -> dict:
    return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}


class _DropClaim(ast.NodeTransformer):
    """The subclass's ``__call__`` / ``kernel`` with the claim edits taken out: the divisors and
    their four kernel parameters dropped, the uniform claim calls back on ``_die_local_claim``
    without their two divisor arguments; and the weight-stream L2 prefetch blocks dropped (each a
    ``cutlass.const_expr(self.l2_prefetch > 0)`` block, so not traced at ``l2_prefetch`` 0)."""

    PREFETCH = "cutlass.const_expr(self.l2_prefetch > 0)"

    DIVISORS = {"fc1_die_divisors", "fc2_die_divisors"}
    PARAMS = {
        "fc1_die_tiles_per_m_fdd",
        "fc1_die_tile_bytes_fdd",
        "fc2_die_tiles_per_m_fdd",
        "fc2_die_tile_bytes_fdd",
    }

    def __init__(self):
        self.dropped = []

    def visit_If(self, node):
        if ast.unparse(node.test) == self.PREFETCH:
            self.dropped.append("l2 prefetch block")
            return None
        self.generic_visit(node)
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if len(node.targets) == 1 and ast.unparse(node.targets[0]) in self.DIVISORS:
            self.dropped.append(ast.unparse(node))
            return None
        return node

    def visit_arguments(self, node):
        self.generic_visit(node)
        kept = [a for a in node.args if a.arg not in self.PARAMS]
        self.dropped += [f"param {a.arg}" for a in node.args if a.arg in self.PARAMS]
        node.args = kept
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if ast.unparse(node.func) == "self._die_local_claim_uniform":
            assert [ast.unparse(a) for a in node.args[-2:]] in (
                [f"{p}_die_tiles_per_m_fdd", f"{p}_die_tile_bytes_fdd"] for p in ("fc1", "fc2")
            )
            node.func = ast.parse("self._die_local_claim", mode="eval").body
            node.args = node.args[:-2]
            self.dropped.append("claim call")
        starred = [
            a
            for a in node.args
            if isinstance(a, ast.Starred) and ast.unparse(a.value) in self.DIVISORS
        ]
        if starred:
            node.args = [a for a in node.args if a not in starred]
            self.dropped += [ast.unparse(a) for a in starred]
        return node


@pytest.mark.cpu_only
def test_the_claim_kernel_is_the_engine_kernel_but_the_claim():
    """``__call__`` and ``kernel`` of ``RubinFc12UniformClaimKernel``, with exactly the claim
    edits and the five L2 prefetch blocks taken out, are statement for statement the engine
    kernel's methods; the subclass overrides nothing else and adds only the claim and the prefetch
    helpers. An engine change to either method fails here (the copy must follow it)."""
    engine = _methods(
        _class(_ENGINE_KERNEL, "Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel")
    )
    ours_cls = _class(_CLAIM_KERNEL, "RubinFc12UniformClaimKernel")
    ours = _methods(ours_cls)
    assert [ast.unparse(b) for b in ours_cls.bases] == [
        "Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel"
    ]
    assert set(ours) == {
        "__init__",
        "__call__",
        "kernel",
        "_die_claim_divisors",
        "_die_local_claim_uniform",
        "_die_claim_search",
        "_die_claim_add",
        "_die_claim_max",
        "_l2_prefetch_head",
        "_l2_prefetch_ahead",
        "_l2_prefetch_fc2_head",
    }
    assert not (set(ours) - {"__init__", "__call__", "kernel"}) & set(engine)
    expected = {
        "__call__": [
            "fc1_die_divisors = (None, None)",
            "fc2_die_divisors = (None, None)",
            "fc1_die_divisors = self._die_claim_divisors(fc1_die_geometry)",
            "fc2_die_divisors = self._die_claim_divisors(fc2_die_geometry)",
            "*fc1_die_divisors",
            "*fc2_die_divisors",
        ],
        "kernel": [f"param {p}" for p in sorted(_DropClaim.PARAMS)]
        + ["claim call"] * 4
        + ["l2 prefetch block"] * 5,
    }
    for name in ("__call__", "kernel"):
        drop = _DropClaim()
        reduced = drop.visit(ours[name])
        assert sorted(drop.dropped) == sorted(expected[name]), (name, drop.dropped)
        assert ast.dump(reduced) == ast.dump(engine[name]), name


@pytest.mark.cpu_only
def test_the_kernel_compare_sees_an_engine_change():
    """The compare is not vacuous: the engine ``kernel`` with one statement changed (its first
    warp-uniform read) no longer matches."""
    engine = _methods(
        _class(_ENGINE_KERNEL, "Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel")
    )
    mutated = copy.deepcopy(engine["kernel"])
    first = next(
        n
        for n in ast.walk(mutated)
        if isinstance(n, ast.Assign) and "make_warp_uniform" in ast.unparse(n.value)
    )
    first.value = first.value.args[0]
    ours = _methods(_class(_CLAIM_KERNEL, "RubinFc12UniformClaimKernel"))["kernel"]
    reduced = _DropClaim().visit(ours)
    assert ast.dump(reduced) == ast.dump(engine["kernel"])
    assert ast.dump(reduced) != ast.dump(mutated)


@pytest.mark.cpu_only
def test_the_claim_kernel_needs_die_words():
    """The subclass exists for die-local claiming only; with die words it builds."""
    pytest.importorskip("cutlass")
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.moe import (
        rubin_fc12_claim as claim,
    )

    args = (16, (128, 256, 128), (128, 256, 256), (1, 1), True)
    common = dict(topk=TOP_K, scheduler="l2_atomic")
    kernel = claim.RubinFc12UniformClaimKernel(*args, **common, die_sm_words=(0xCCCCCCCC, 0xCCCC))
    assert kernel.use_die_scheduler
    with pytest.raises(ValueError, match="die_sm_words"):
        claim.RubinFc12UniformClaimKernel(*args, **common)


# ---------------------------------------------------------------------------------------------
# SM107 (two dies): the variant's exact rows against the engine claim
# ---------------------------------------------------------------------------------------------


def _two_dies() -> bool:
    if not _SM107:
        return False
    from tensorrt_llm._torch import die_locality

    return die_locality.locality_domain_count(torch.cuda.current_device()) >= 2


rubin_dies = pytest.mark.skipif(not _two_dies(), reason="needs a two-die SM107 GPU")
#: Rows : tokens per request : expert pool of a request (helix8 b8k7, b16k5, b24k5 and 32).
SHAPES = ("32:8:40", "64:8:40", "96:6:40", "144:6:40")
MODES = ("default", "die_sched", "sched_only")


@pytest.fixture(scope="module")
def windows():
    """Plain and die-placed NVFP4 FC12 operands of one rank's window, per geometry."""
    from tensorrt_llm._torch.die_locality import die_alternating_copy, is_die_alternating

    gen = torch.Generator(device="cuda").manual_seed(100)

    def nibbles(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda", generator=gen)

    def block_scales(*shape):
        values = torch.rand(shape, device="cuda", generator=gen) * 1.5 + 0.5
        return values.to(torch.float8_e4m3fn).view(torch.uint8)

    out = {}
    for geometry, (local, inter) in GEOMETRIES.items():
        plain = dict(
            w1=nibbles(local, 2 * inter, HIDDEN // 2),
            s1=block_scales(local, 2 * inter, HIDDEN // 16),
            a1=torch.full((local,), 2.0e-6, device="cuda"),
            w2=nibbles(local, HIDDEN, inter // 2),
            s2=block_scales(local, HIDDEN, inter // 16),
            a2=torch.full((local,), 2e-5, device="cuda"),
        )
        placed = dict(plain)
        for key in ("w1", "w2"):
            placed[key] = die_alternating_copy(plain[key])
            assert placed[key] is not None and is_die_alternating(placed[key])
        out[geometry] = (plain, placed)
    return out


def _routing(m, per_request, pool, seed, offset, local):
    gen = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(m // per_request):
        request_pool = torch.randperm(NUM_EXPERTS, generator=gen)[:pool]
        for _ in range(per_request):
            rows.append(request_pool[torch.randperm(pool, generator=gen)[:TOP_K]])
    ids = torch.stack(rows).to(torch.int32).cuda()
    scales = (torch.rand((m, TOP_K), generator=gen) + 0.1).cuda()
    tables = token_ordered_sort(ids, offset, local, TILE)
    in_window = ((ids >= offset) & (ids < offset + local)).sum(dim=1)
    return scales, tables, in_window


def _inputs(w, x4, sf, scales, tables, out):
    t2e, t2l, e2p, p2e, _, num_tiles = tables
    return [
        x4.view(torch.float4_e2m1fn_x2),
        w["w1"].view(torch.float4_e2m1fn_x2),
        sf.view(torch.uint8),
        w["s1"],
        w["a1"],
        t2e,
        t2l,
        p2e,
        num_tiles,
        torch.tensor([4.0], device="cuda"),
        w["w2"].view(torch.float4_e2m1fn_x2),
        w["s2"],
        w["a2"],
        out,
        scales,
        e2p,
    ]


@rubin_dies
@pytest.mark.parametrize("mma_n", sorted(TACTICS, reverse=True))
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("geometry", sorted(GEOMETRIES, reverse=True))
def test_uniform_claim_rows_equal_the_engine_claim(monkeypatch, windows, geometry, shape, mma_n):
    """The variant's exact rows equal the engine claim's bit for bit with the die schedule on
    placed weights (die_sched), forced on plain weights (sched_only) and off (default: the token
    floor above the rows, so the variant is not taken); all rows finite and within the
    atomic-order noise; the variant's kernel was built exactly for the die-local runs."""
    pytest.importorskip("cutlass")
    from tensorrt_llm._torch import die_locality
    from tensorrt_llm._torch.utils import ActivationType

    local, _ = GEOMETRIES[geometry]
    offset = local * 2  # the third window: experts 448.. (TP2 x EP4) / 224.. (EP8)
    plain, placed = windows[geometry]
    m, per_request, pool = (int(v) for v in shape.split(":"))
    gen = torch.Generator(device="cuda").manual_seed(m)
    x = torch.randn((m, HIDDEN), device="cuda", generator=gen).to(torch.bfloat16)
    x4, sf = torch.ops.trtllm.fp4_quantize(
        x, ((448 * 6) / x.abs().max().float()).reshape(1), 16, False, False
    )
    sf = sf.view(m, HIDDEN // 16)
    scales, tables, in_window = _routing(
        m, per_request, pool, 1000 + m + per_request, offset, local
    )
    exact = in_window <= 2
    assert int(exact.sum()) > 0
    words = die_locality.pack_sm_die_words(die_locality.sm_die_table(torch.cuda.current_device()))
    real_words = launch.fc12_die_sm_words
    scalars = dict(
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        num_local_experts=local,
        local_expert_offset=offset,
        tile_size=TILE,
        scaling_vector_size=16,
        swiglu_limit=float("inf"),
        ep_size=NUM_EXPERTS // local,
        enable_alltoall=False,
        activation_type=ActivationType.SiTu,
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )
    levers = dict(mma_n=mma_n, die_sched=True, lookahead=1, sm_reserve=16)

    def run(w, die_min_tokens, die_claim):
        runner = launch.RubinFc12Runner(
            **scalars, **levers, die_min_tokens=die_min_tokens, die_claim=die_claim
        )
        out = torch.full((m, HIDDEN), float("nan"), dtype=torch.bfloat16, device="cuda")
        runner(_inputs(w, x4, sf, scales, tables, out), tactic=TACTICS[mma_n])
        torch.cuda.synchronize()
        return out.clone()

    def claim_keys():
        return {
            k
            for k in launch.RubinFc12Runner.kernel_cache
            if tuple(k[3]) == TACTICS[mma_n][0] and ("die_claim", "uniform") in k
        }

    for mode in MODES:
        if mode == "sched_only":
            monkeypatch.setattr(
                launch,
                "fc12_die_sm_words",
                lambda fc1, fc2, cluster, tokens, **kw: words if cluster == 1 else None,
            )
        else:
            monkeypatch.setattr(launch, "fc12_die_sm_words", real_words)
        w = placed if mode == "die_sched" else plain
        # default: the die schedule's token floor above the rows, so no launch claims die-locally
        # and the variant must not be taken.
        floor = 4096 if mode == "default" else 24
        before = claim_keys()
        ref = run(w, floor, None)
        got = run(w, floor, "uniform")
        label = f"{geometry} {shape} MMA N {mma_n} {mode}"
        assert torch.isfinite(ref.float()).all() and torch.isfinite(got.float()).all(), label
        assert torch.equal(got[exact].view(torch.int16), ref[exact].view(torch.int16)), (
            f"{label}: exact rows differ from the engine claim"
        )
        rel = (got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-30)
        assert rel <= 5e-3, f"{label}: relative L2 {rel:.3e}"
        if mode == "default":
            assert claim_keys() == before, f"{label}: the variant was built off the die schedule"
        elif mode == "die_sched":
            assert claim_keys(), f"{label}: no uniform-claim kernel in the cache"
