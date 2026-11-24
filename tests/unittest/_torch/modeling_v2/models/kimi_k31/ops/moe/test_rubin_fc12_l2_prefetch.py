# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""rubin_fc12's weight-stream L2 prefetch ``l2_prefetch`` (SM107).

The uniform claim kernel (``kernels/moe/rubin_fc12_claim.py``) takes a depth P (0..8): its TMA-B
warp prefetches the B / SFB weight stream into L2 P K-blocks ahead of its loads, through the loads'
own TMA atoms and slices, the first FC2 tile's first P K-blocks before the FC1 -> FC2 drain
included. A prefetch writes no shared memory and arrives on no barrier, so the loads, the MMA and
the finalize are the kernel's at P 0: rows the finalize sums from at most two local experts are
bitwise the same, the others within FC12's atomic-order noise (``0 + a + b`` rounds the same in
either order; a third add does not).

Off the GPU: the depths the launcher takes (only with the uniform claim), the depth a launch runs,
the keys (nothing added at P 0, so today's tuned tactics and compiled kernels stay keyed as they
were), the INFO line once per process, the entry and the helix8 target passing the depth through,
where the five prefetch blocks sit in the kernel, and a model of the TMA-B warp's issue order. On
SM107: the exact rows at P 2, 4 and 6 against P 0 through the family runner at helix8's window.
"""

import __future__

import ast
import copy
import importlib.util
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.moe import (
    rubin_fc12_launch as launch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe import rubin_fc12 as mod
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe._rubin_moe_common import (
    token_ordered_sort,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.rubin_fc12 import (
    rubin_fc12,
)
from tensorrt_llm._torch.utils import ActivationType

_SM107 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
# Certified on SM107 only; other GPUs skip the file, a CPU-only host runs the cpu_only tests.
pytestmark = pytest.mark.skipif(
    torch.cuda.is_available() and not _SM107, reason="certified on SM107 only"
)

HIDDEN, NUM_EXPERTS, TOP_K, TILE = 3584, 896, 16, 128
#: helix8's FC12 levers but the prefetch (TP2 x EP4: MMA N 256, die-local uniform claim).
HELIX8 = dict(mma_n=256, die_sched=True, die_min_tokens=24, lookahead=1, sm_reserve=16)
#: K-blocks of one FC1 tile (K = the 3,584 latent) and of one FC2 tile (K = the 1,536 intermediate
#: per rank) at the kernel's 256-wide K block.
FC1_K_BLOCKS, FC2_K_BLOCKS = HIDDEN // 256, 1536 // 256
_LINE = "rubin_fc12: weight stream L2 prefetch {} K-blocks ahead of the loads"
_PKG = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_FAMILY = _PKG / "_torch/_experimental/modeling_v2/models/kimi_k31"
_CLAIM_KERNEL = _FAMILY / "kernels/moe/rubin_fc12_claim.py"
_MODELING = _FAMILY / "kimi_k31_pro_nvfp4__sm_107__helix8/modeling.py"


def _method(tree: ast.AST, cls: str, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"{cls}.{name} not found")


# ---------------------------------------------------------------------------------------------
# No GPU: the launcher
# ---------------------------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_the_depths_the_launcher_takes():
    """0..8 with the uniform claim, 0 without it; anything else, or a depth on the engine
    kernel's claim (which has no prefetch), is refused."""
    for depth in range(9):
        launch.check_levers(**HELIX8, die_claim="uniform", l2_prefetch=depth)
    launch.check_levers(**HELIX8, die_claim=None, l2_prefetch=0)
    launch.check_levers(**HELIX8, die_claim="uniform")
    for bad in (-1, 9, 16, True, 2.0, "4", None):
        with pytest.raises(ValueError, match="l2_prefetch must be an int"):
            launch.check_levers(**HELIX8, die_claim="uniform", l2_prefetch=bad)
    for depth in (1, 4, 8):
        with pytest.raises(ValueError, match="needs die_claim 'uniform'"):
            launch.check_levers(**HELIX8, die_claim=None, l2_prefetch=depth)
    assert launch.L2_PREFETCH == tuple(range(9))


@pytest.mark.cpu_only
def test_the_kernel_takes_the_launchers_depths():
    """The kernel's own bound (read from its source: the module imports the CuTe DSL) is the
    launcher's."""
    tree = ast.parse(_CLAIM_KERNEL.read_text())
    (bound,) = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "L2_PREFETCH_MAX"
    ]
    assert launch.L2_PREFETCH == tuple(range(bound + 1))


@pytest.mark.cpu_only
def test_a_launch_runs_the_depth_only_on_the_uniform_claim(monkeypatch):
    """The depth a launch runs: the runner's on a launch of the uniform claim kernel; 0 where the
    launch falls back to the engine kernel (below the die schedule's token floor, outside the
    Int32 bound), and 0 when off."""
    monkeypatch.setattr(launch, "_l2_prefetch_logged", set())
    (n1, k1), (n2, k2) = (2 * 1536, HIDDEN), (HIDDEN, 1536)
    claim = launch.fc12_die_claim("uniform", True, n1, k1, n2, k2, 256)
    assert launch.fc12_l2_prefetch(4, claim) == 4
    off_schedule = launch.fc12_die_claim("uniform", False, n1, k1, n2, k2, 256)
    assert launch.fc12_l2_prefetch(4, off_schedule) == 0
    assert launch.fc12_l2_prefetch(4, None) == 0
    assert launch.fc12_l2_prefetch(0, "uniform") == 0


@pytest.mark.cpu_only
def test_the_keys_add_nothing_at_zero():
    """What the depth adds to ``unique_id`` and to the kernel cache key: nothing at 0 (today's
    tuned tactics and compiled kernels keep their keys), one tagged entry otherwise; and the
    runner uses exactly these: ``unique_id`` ends with the depth's key, the launch keys and builds
    its kernel on the depth it runs, and passes the kernel a depth only when it is nonzero."""
    assert launch.fc12_l2_prefetch_key(0) == ()
    assert launch.fc12_l2_prefetch_key(4) == (("l2_prefetch", 4),)
    tree = ast.parse(Path(launch.__file__).read_text())
    ret = _method(tree, "RubinFc12Runner", "unique_id").body[-1]
    assert isinstance(ret, ast.Return) and isinstance(ret.value, ast.BinOp)
    assert ast.unparse(ret.value.right) == "fc12_l2_prefetch_key(self.l2_prefetch)"
    forward = _method(tree, "RubinFc12Runner", "forward")
    lines = [ast.unparse(n) for n in ast.walk(forward) if isinstance(n, ast.stmt)]
    assert "l2_prefetch = fc12_l2_prefetch(self.l2_prefetch, die_claim)" in lines
    assert "cache_key += fc12_l2_prefetch_key(l2_prefetch)" in lines
    kwargs = [
        ast.unparse(k.value)
        for n in ast.walk(forward)
        if isinstance(n, ast.Call)
        for k in n.keywords
        if k.arg is None
    ]
    assert kwargs == ["dict(l2_prefetch=l2_prefetch) if l2_prefetch else {}"]


@pytest.mark.cpu_only
def test_unique_id_at_zero_is_the_kernels_without_it():
    """With the CuTe DSL here (the runner is defined only then): the runner's ``unique_id`` at
    P 0 is the 16 values it had before the prefetch existed; P 4 appends its key."""
    cls = getattr(launch, "RubinFc12Runner", None)
    if cls is None:
        pytest.skip("no CuTe DSL with Rubin support here: the runner is not defined")
    ids = []
    for depth in (0, 4):
        runner = object.__new__(cls)  # the constructor checks the GPU's SM version
        runner.__dict__.update(
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            num_local_experts=224,
            local_expert_offset=0,
            tile_size=TILE,
            scaling_vector_size=16,
            swiglu_limit=float("inf"),
            activation_type=0,
            situ_beta=4.0,
            situ_linear_beta=25.0,
            die_claim="uniform",
            l2_prefetch=depth,
            **HELIX8,
        )
        ids.append(runner.unique_id())
    assert len(ids[0]) == 16 and ids[0][10:] == (256, True, 24, 1, 16, "uniform")
    assert ids[1] == ids[0] + (("l2_prefetch", 4),)


@pytest.mark.cpu_only
def test_the_depth_is_logged_once_per_process(monkeypatch):
    """One INFO line per depth and process, on the first launch that runs it; none when off or on
    the engine kernel. It is the witness of a real job (the line, once per rank)."""
    from tensorrt_llm.logger import logger

    lines = []
    monkeypatch.setattr(launch, "_l2_prefetch_logged", set())
    monkeypatch.setattr(logger, "info", lambda msg, *a, **k: lines.append(msg))
    for _ in range(3):
        launch.fc12_l2_prefetch(0, "uniform")
        launch.fc12_l2_prefetch(4, None)
    assert lines == []
    for _ in range(3):
        launch.fc12_l2_prefetch(4, "uniform")
    assert len(lines) == 1 and lines[0].startswith(_LINE.format(4))
    launch.fc12_l2_prefetch(2, "uniform")
    assert len(lines) == 2 and lines[1].startswith(_LINE.format(2))


# ---------------------------------------------------------------------------------------------
# No GPU: the entry and the helix8 target pass the depth through
# ---------------------------------------------------------------------------------------------


def _call_args(rows=8):
    """One call's operands on the CPU (EP8 rank 0's 112 experts, small weights: the entry and its
    guard read no weight values)."""
    local, hidden, inter = 112, 256, 64
    g = torch.Generator().manual_seed(rows)
    ids = torch.stack([torch.randperm(local, generator=g)[:TOP_K] for _ in range(rows)])
    tables = token_ordered_sort(ids, 0, local, TILE)

    def fp4(*shape):
        return torch.zeros(*shape, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)

    return dict(
        input=fp4(rows, hidden // 2),
        fc1_weight=fp4(local, 2 * inter, hidden // 2),
        input_scale=torch.zeros(rows, hidden // 16, dtype=torch.uint8),
        fc1_weight_scale=torch.zeros(local, 2 * inter, hidden // 16, dtype=torch.uint8),
        fc1_alpha=torch.ones(local),
        tile_idx_to_group_idx=tables[0],
        tile_idx_to_mn_limit=tables[1],
        permuted_idx_to_expanded_idx=tables[3],
        num_non_exiting_tiles=tables[5],
        global_sf=torch.ones(1),
        fc2_weight=fp4(local, hidden, inter // 2),
        fc2_weight_scale=torch.zeros(local, hidden, inter // 16, dtype=torch.uint8),
        fc2_alpha=torch.ones(local),
        output=torch.zeros(rows, hidden, dtype=torch.bfloat16),
        token_final_scales=torch.rand(rows, TOP_K, generator=g),
        expanded_idx_to_permuted_idx=tables[2],
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        num_local_experts=local,
        local_expert_offset=0,
        tile_size=TILE,
        swiglu_limit=float("inf"),
        ep_size=8,
        enable_alltoall=False,
        activation_type=int(ActivationType.SiTu),
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )


@pytest.mark.cpu_only
def test_the_entry_passes_the_depth(monkeypatch):
    """``l2_prefetch`` is keyword only and defaults to 0 on the entry and its guard; the entry
    hands it to the launcher unchanged and the guard refuses what the launcher refuses."""
    import inspect

    for method in (type(rubin_fc12).__call__, type(rubin_fc12).is_valid):
        param = inspect.signature(method).parameters["l2_prefetch"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY and param.default == 0
    seen = {}
    monkeypatch.setattr(mod, "launch_rubin_fc12", lambda inputs, **kw: seen.update(kw))
    args = _call_args()
    rubin_fc12(**args, **HELIX8, die_claim="uniform")
    assert seen["l2_prefetch"] == 0
    rubin_fc12(**args, **HELIX8, die_claim="uniform", l2_prefetch=4)
    assert seen["l2_prefetch"] == 4 and seen["die_claim"] == "uniform"
    rubin_fc12.is_valid(**args, **HELIX8, die_claim="uniform", l2_prefetch=4)
    for bad in (dict(die_claim="uniform", l2_prefetch=9), dict(die_claim=None, l2_prefetch=4)):
        with pytest.raises(ValueError, match="l2_prefetch"):
            rubin_fc12.is_valid(**args, **HELIX8, **bad)


def _constants(tree: ast.Module) -> dict:
    """``tree``'s module-level constants that evaluate on their own, in source order (as the
    helix8 MoE front test reads them)."""
    ns: dict = {"MappingProxyType": MappingProxyType, "torch": torch}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            ns[target.id] = eval(compile(ast.Expression(node.value), str(_MODELING), "eval"), ns)
        except Exception:
            continue
    return ns


@pytest.mark.cpu_only
def test_helix8_passes_its_depth_to_fc12():
    """helix8's ``_moe_fc12``, run from its modeling.py over its constants and a recording
    ``rubin_fc12``: every launch takes ``_FC12_L2_PREFETCH`` (2) with the uniform claim, and a
    changed constant reaches the launch (the microbench arms set it per arm)."""
    tree = ast.parse(_MODELING.read_text(), filename=str(_MODELING))
    ns = _constants(tree)
    assert ns["_FC12_L2_PREFETCH"] == 2 and ns["_FC12_DIE_CLAIM"] == "uniform"
    (fn,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_moe_fc12"]
    seen = []
    ns.update(rubin_fc12=lambda *a, **k: seen.append(k), _moe_sort=None)
    module = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(fn)], type_ignores=[]))
    flags = __future__.annotations.compiler_flag
    exec(compile(module, str(_MODELING), "exec", flags=flags, dont_inherit=True), ns)
    u8 = dict(dtype=torch.uint8)
    layer = SimpleNamespace(
        fc1_w=torch.zeros(224, 4, 8, **u8),
        fc2_w=torch.zeros(224, 16, 4, **u8),
        fc1_s=None,
        fc31_alpha=None,
        fc2_in_scale=None,
        fc2_s=None,
        fc2_alpha=None,
        expert_offset=0,
    )
    x4, ids = torch.zeros(96, 8, **u8), torch.zeros(96, TOP_K, dtype=torch.int32)
    for depth in (4, 0, 6):
        ns["_FC12_L2_PREFETCH"] = depth
        ns["_moe_fc12"](layer, x4, None, ids, None, (None,) * 6, None)
        assert seen[-1]["l2_prefetch"] == depth and seen[-1]["die_claim"] == "uniform"
    assert len(seen) == 3


# ---------------------------------------------------------------------------------------------
# No GPU: where the prefetch sits in the kernel, and the TMA-B warp's issue order
# ---------------------------------------------------------------------------------------------

_PREFETCH = "cutlass.const_expr(self.l2_prefetch > 0)"


def _blocks(fn: ast.FunctionDef) -> list:
    """``(block, its siblings, its index)`` of every prefetch block of ``fn``, in source order."""
    found = []
    for parent in ast.walk(fn):
        for field in ("body", "orelse"):
            body = getattr(parent, field, None)
            if not isinstance(body, list):
                continue
            for i, node in enumerate(body):
                if isinstance(node, ast.If) and ast.unparse(node.test) == _PREFETCH:
                    found.append((node, body, i))
    return sorted(found, key=lambda f: f[0].lineno)


@pytest.mark.cpu_only
def test_the_prefetch_blocks_sit_where_the_design_puts_them():
    """The five blocks of the TMA-B warp, each with the loads' own atoms and slices: (1) at an FC1
    tile start, before its first stage is acquired; (2) after K-block k's two loads, before the
    producer state advances; (3) after the FC1 tiles, before the FC1 stages drain, for the first
    FC2 tile when there is one; (4) as (2) in the FC2 K loop; (5) at the end of an FC2 tile, for
    the next one when there is one. No block assigns a name (nothing new is carried by a loop)."""
    tree = ast.parse(_CLAIM_KERNEL.read_text())
    kernel = _method(tree, "RubinFc12UniformClaimKernel", "kernel")
    blocks = _blocks(kernel)
    assert len(blocks) == 5
    calls = [" ".join(ast.unparse(b[0].body[0]).split()) for b in blocks]
    after = [ast.unparse(b[1][b[2] + 1]) if b[2] + 1 < len(b[1]) else None for b in blocks]
    before = [ast.unparse(b[1][b[2] - 1]) for b in blocks]
    assert calls[0] == (
        "self._l2_prefetch_head(tma_atom_fc1_b, tma_atom_fc1_sfb, tFC1BgB_slice, "
        "tFC1BgSFB_slice, k_tile_cnt)"
    )
    assert after[0] == "fc1_b_producer_state.reset_count()"
    assert calls[1] == (
        "self._l2_prefetch_ahead(tma_atom_fc1_b, tma_atom_fc1_sfb, tFC1BgB_slice, "
        "tFC1BgSFB_slice, fc1_b_producer_state.count, k_tile_cnt)"
    )
    assert before[1].startswith("cute.copy(tma_atom_fc1_sfb, tFC1BgSFB_k,")
    assert after[1] == "fc1_b_producer_state.advance()"
    fc2_head = (
        "self._l2_prefetch_fc2_head(tma_atom_fc2_b, tma_atom_fc2_sfb, tFC2BgB, tFC2BgSFB, "
        "tile_info[1], tile_info[2], fc2_k_tile_cnt)"
    )
    assert calls[2] == f"if tile_info[3] == 1: {fc2_head}"
    assert after[2] == "fc1_b_pipeline.producer_tail(fc1_b_producer_state)"
    assert calls[3] == (
        "self._l2_prefetch_ahead(tma_atom_fc2_b, tma_atom_fc2_sfb, fc2_b_gmem, fc2_sfb_gmem, "
        "fc2_b_producer_state.count, fc2_k_tile_cnt)"
    )
    assert before[3].startswith(
        "cute.copy(tma_atom_fc2_sfb, fc2_sfb_gmem[None, fc2_b_producer_state.count]"
    )
    assert after[3] == "fc2_b_producer_state.advance()"
    assert calls[4] == f"if is_valid_fc2_tile: {fc2_head}"
    assert before[4] == "tile_info_consumer_state.advance()" and after[4] is None
    for block, _, _ in blocks:
        assert len(block.body) == 1 and not block.orelse
        assert not [n for n in ast.walk(block) if isinstance(n, (ast.Assign, ast.AugAssign))]
    # The loads read the same slices: FC1 K-block k of tFC1BgB_slice / tFC1BgSFB_slice, FC2 tile
    # (tile_info[1], tile_info[2]) of tFC2BgB / tFC2BgSFB, as _l2_prefetch_fc2_head slices it.
    src = ast.unparse(kernel)
    assert "tFC1BgB_k = tFC1BgB_slice[None, fc1_b_producer_state.count]" in src
    assert "tFC1BgSFB_k = tFC1BgSFB_slice[None, fc1_b_producer_state.count]" in src
    assert "fc2_b_gmem = tFC2BgB[None, fc2_n_tile, None, fc2_expert]" in src
    assert "fc2_sfb_gmem = tFC2BgSFB[None, fc2_sfb_n_tile, None, fc2_expert]" in src
    helper = ast.unparse(_method(tree, "RubinFc12UniformClaimKernel", "_l2_prefetch_fc2_head"))
    assert "tFC2BgB[None, fc2_n_tile, None, fc2_expert]" in helper
    assert "tFC2BgSFB[None, fc2_sfb_n_tile, None, fc2_expert]" in helper


def _warp9(tiles, depth):
    """The TMA-B warp's issue order over one CTA's ``tiles`` (``("fc1" | "fc2", tile id)`` in
    claim order), as the kernel issues it at prefetch depth ``depth``: ``("pf", phase, tile, k)``,
    ``("load", phase, tile, k)`` and ``("drain",)``."""
    out, k_cnt = [], {"fc1": FC1_K_BLOCKS, "fc2": FC2_K_BLOCKS}

    def head(phase, tile):
        out.extend(("pf", phase, tile, k) for k in range(min(depth, k_cnt[phase])))

    fc1 = [t for t in tiles if t[0] == "fc1"]
    fc2 = [t for t in tiles if t[0] == "fc2"]
    for phase, tile in fc1:
        if depth:
            head(phase, tile)  # block 1
        for k in range(k_cnt[phase]):
            out.append(("load", phase, tile, k))
            if depth and k + depth < k_cnt[phase]:
                out.append(("pf", phase, tile, k + depth))  # block 2
    if depth and fc2:
        head(*fc2[0])  # block 3
    out.append(("drain",))
    for i, (phase, tile) in enumerate(fc2):
        for k in range(k_cnt[phase]):
            out.append(("load", phase, tile, k))
            if depth and k + depth < k_cnt[phase]:
                out.append(("pf", phase, tile, k + depth))  # block 4
        if depth and i + 1 < len(fc2):
            head(*fc2[i + 1])  # block 5
    return out


@pytest.mark.cpu_only
@pytest.mark.parametrize("depth", [2, 4, 6])
@pytest.mark.parametrize(
    "tiles",
    [
        [("fc1", 0), ("fc1", 1), ("fc2", 0), ("fc2", 1), ("fc2", 2)],
        [("fc1", 0), ("fc2", 0)],
        [("fc2", 0), ("fc2", 1)],
        [("fc1", 0), ("fc1", 1)],
        [],
    ],
    ids=["fc1x2_fc2x3", "fc1_fc2", "fc2_only", "fc1_only", "none"],
)
def test_the_issue_order_model(tiles, depth):
    """Every prefetched K-block is one the CTA then loads, once, at most ``depth`` K-blocks of its
    tile ahead; none lies past its tile; the first FC2 tile's first K-blocks go out before the
    drain; the loads are those of P 0 in the same order."""
    events = _warp9(tiles, depth)
    loads = [e for e in events if e[0] == "load"]
    assert [e for e in _warp9(tiles, 0) if e[0] == "load"] == loads
    prefetched = [e[1:] for e in events if e[0] == "pf"]
    assert len(prefetched) == len(set(prefetched))
    k_cnt = {"fc1": FC1_K_BLOCKS, "fc2": FC2_K_BLOCKS}
    for i, event in enumerate(events):
        if event[0] != "pf":
            continue
        phase, tile, k = event[1:]
        assert k < k_cnt[phase]
        load_at = events.index(("load", phase, tile, k))
        assert load_at > i, event
        done = sum(1 for e in events[:i] if e[0] == "load" and e[1:3] == (phase, tile))
        assert k - done < depth or done == 0, event
    # Every K-block of every tile is prefetched before its load.
    assert {e[1:] for e in loads} == set(prefetched)
    fc2 = [t for t in tiles if t[0] == "fc2"]
    if fc2:
        drain = events.index(("drain",))
        first = {("pf", *fc2[0], k) for k in range(min(depth, FC2_K_BLOCKS))}
        assert first <= set(events[:drain])


# ---------------------------------------------------------------------------------------------
# SM107 (two dies): the prefetching kernel's exact rows against P 0
# ---------------------------------------------------------------------------------------------


def _two_dies() -> bool:
    if not _SM107:
        return False
    from tensorrt_llm._torch import die_locality

    return die_locality.locality_domain_count(torch.cuda.current_device()) >= 2


rubin_dies = pytest.mark.skipif(not _two_dies(), reason="needs a two-die SM107 GPU")
#: Rows : tokens per request : expert pool of a request (helix8 b8k7, b16k5, b24k5).
SHAPES = ("64:8:40", "96:6:40", "144:6:40")
#: helix8's window: TP2 x EP4, 224 experts at half of the 3,072 intermediate.
LOCAL, INTER = 224, 1536
TACTIC = ((TILE, 256, 256), (TILE, 256, 128), (1, 1))


@pytest.fixture(scope="module")
def window():
    """helix8's FC12 operands of one rank's window, placed in die-alternating memory."""
    from tensorrt_llm._torch.die_locality import die_alternating_copy, is_die_alternating

    gen = torch.Generator(device="cuda").manual_seed(104)

    def nibbles(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda", generator=gen)

    def block_scales(*shape):
        values = torch.rand(shape, device="cuda", generator=gen) * 1.5 + 0.5
        return values.to(torch.float8_e4m3fn).view(torch.uint8)

    w = dict(
        w1=nibbles(LOCAL, 2 * INTER, HIDDEN // 2),
        s1=block_scales(LOCAL, 2 * INTER, HIDDEN // 16),
        a1=torch.full((LOCAL,), 2.0e-6, device="cuda"),
        w2=nibbles(LOCAL, HIDDEN, INTER // 2),
        s2=block_scales(LOCAL, HIDDEN, INTER // 16),
        a2=torch.full((LOCAL,), 2e-5, device="cuda"),
    )
    for key in ("w1", "w2"):
        w[key] = die_alternating_copy(w[key])
        assert w[key] is not None and is_die_alternating(w[key])
    return w


def _routing(m, per_request, pool, seed, offset):
    gen = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(m // per_request):
        request_pool = torch.randperm(NUM_EXPERTS, generator=gen)[:pool]
        for _ in range(per_request):
            rows.append(request_pool[torch.randperm(pool, generator=gen)[:TOP_K]])
    ids = torch.stack(rows).to(torch.int32).cuda()
    scales = (torch.rand((m, TOP_K), generator=gen) + 0.1).cuda()
    tables = token_ordered_sort(ids, offset, LOCAL, TILE)
    in_window = ((ids >= offset) & (ids < offset + LOCAL)).sum(dim=1)
    return scales, tables, in_window


@rubin_dies
@pytest.mark.parametrize("depth", [2, 4, 6])
@pytest.mark.parametrize("shape", SHAPES)
def test_prefetch_rows_equal_p0(window, shape, depth):
    """At P 2, 4 and 6 the uniform claim kernel's exact rows equal its P 0 rows bit for bit, all
    rows are finite and within the atomic-order noise, and the launch built (and keyed) a
    prefetching kernel; P 0 built none."""
    pytest.importorskip("cutlass")
    offset = LOCAL * 2  # the third window: experts 448..671
    m, per_request, pool = (int(v) for v in shape.split(":"))
    gen = torch.Generator(device="cuda").manual_seed(m)
    x = torch.randn((m, HIDDEN), device="cuda", generator=gen).to(torch.bfloat16)
    x4, sf = torch.ops.trtllm.fp4_quantize(
        x, ((448 * 6) / x.abs().max().float()).reshape(1), 16, False, False
    )
    sf = sf.view(m, HIDDEN // 16)
    scales, tables, in_window = _routing(m, per_request, pool, 2000 + m + depth, offset)
    exact = in_window <= 2
    assert int(exact.sum()) > 0
    t2e, t2l, e2p, p2e, _, num_tiles = tables
    scalars = dict(
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        num_local_experts=LOCAL,
        local_expert_offset=offset,
        tile_size=TILE,
        scaling_vector_size=16,
        swiglu_limit=float("inf"),
        ep_size=NUM_EXPERTS // LOCAL,
        enable_alltoall=False,
        activation_type=ActivationType.SiTu,
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )

    def prefetching_keys():
        return {k for k in launch.RubinFc12Runner.kernel_cache if ("l2_prefetch", depth) in k}

    def run(p):
        runner = launch.RubinFc12Runner(**scalars, **HELIX8, die_claim="uniform", l2_prefetch=p)
        out = torch.full((m, HIDDEN), float("nan"), dtype=torch.bfloat16, device="cuda")
        inputs = [
            x4.view(torch.float4_e2m1fn_x2),
            window["w1"].view(torch.float4_e2m1fn_x2),
            sf.view(torch.uint8),
            window["s1"],
            window["a1"],
            t2e,
            t2l,
            p2e,
            num_tiles,
            torch.tensor([4.0], device="cuda"),
            window["w2"].view(torch.float4_e2m1fn_x2),
            window["s2"],
            window["a2"],
            out,
            scales,
            e2p,
        ]
        runner(inputs, tactic=TACTIC)
        torch.cuda.synchronize()
        return out.clone()

    before = prefetching_keys()
    ref = run(0)
    assert prefetching_keys() == before, "P 0 built a prefetching kernel"
    got = run(depth)
    assert prefetching_keys(), f"no kernel keyed ('l2_prefetch', {depth})"
    label = f"{shape} P {depth}"
    assert torch.isfinite(ref.float()).all() and torch.isfinite(got.float()).all(), label
    assert torch.equal(got[exact].view(torch.int16), ref[exact].view(torch.int16)), (
        f"{label}: exact rows differ from P 0"
    )
    rel = (got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-30)
    assert rel <= 5e-3, f"{label}: relative L2 {rel:.3e}"
