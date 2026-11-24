# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The rubin_fc12 entry and its family launcher off the GPU: the launch is the engine runner's
but for the five levers it takes as arguments and the die-local claim's variant, the levers equal
the engine's environment reads, the guard, the die placement builder, the construction check of
the lever variables, and the NVFP4 MoE reference the family's MoE entries share. The claim
variant's own checks are in test_rubin_fc12_die_claim.py.

Its cells run on SM107 in test_rubin_fc12_cells.py, on operands in the layout the kernel reads.
"""

import ast
import inspect
import itertools
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.moe import (
    rubin_fc12_launch as launch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe import (
    _rubin_moe_common as common,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe import rubin_fc12 as mod
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.rubin_fc12 import (
    check_fc12_lever_env,
    fc12_tactics,
    rubin_fc12,
    rubin_fc12_place_per_die,
)
from tensorrt_llm._torch.utils import ActivationType

_SM107 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.skipif(torch.cuda.is_available() and not _SM107, reason="certified on SM107 only"),
]

HIDDEN = 3584  # the latent the routed experts see
#: The TEP8 generation levers (tep8's ``_FC12_*`` launch constants): SM reserve 16, die-local
#: claiming from 24.
TEP8 = dict(mma_n=None, die_sched=True, die_min_tokens=24, lookahead=1, sm_reserve=16)
#: The engine environment that sets the same levers (the serving configuration's values).
TEP8_ENV = {
    "TRTLLM_K3_FC12_SM_RESERVE": "16",
    "TRTLLM_K3_FC12_DIE_LAYOUT": "1",
    "TRTLLM_K3_FC12_DIE_SCHED": "1",
    "TRTLLM_K3_FC12_DIE_SCHED_MIN_TOKENS": "24",
}
_ENGINE_SRC = Path(inspect.getfile(launch)).resolve().parents[7] / (
    "_torch/custom_ops/cute_dsl_custom_ops.py"
)


def _tactics(num_local, inter, tile, mma_n, rows=144, top_k=16, accept=lambda *a: True):
    asked = []

    def can_implement(tiler, inst, cluster, fc1_shape, fc2_shape):
        asked.append(tiler[1])
        assert fc1_shape[1:] == (2 * inter, HIDDEN, num_local)
        assert fc2_shape[1:] == (HIDDEN, inter, num_local)
        return accept(tiler, inst, cluster)

    m = rows * top_k + num_local * tile
    return fc12_tactics(
        m, 2 * inter, HIDDEN, num_local, HIDDEN, inter, tile, mma_n, can_implement
    ), asked


@pytest.mark.parametrize("num_local, inter", [(224, 1536), (112, 3072)], ids=["tp2_ep4", "ep8"])
@pytest.mark.parametrize("tile", [128, 256])
def test_tactics_follow_mma_n(num_local, inter, tile):
    """None: N 128 then 256; a pinned N: that N only, same M tile, instruction and cluster; the
    kernel is asked about the offered N only."""
    base, asked = _tactics(num_local, inter, tile, None)
    assert [t[0][1] for t in base] == [128, 256] and asked == [128, 256]
    for tiler, inst, cluster in base:
        assert tiler == (tile, tiler[1], 256) and inst == (tile, tiler[1], 128)
        assert cluster == (tile // 128, 1)
    for pinned in (256, 128):
        tactics, asked = _tactics(num_local, inter, tile, pinned)
        assert tactics == [t for t in base if t[0][1] == pinned] and asked == [pinned]


def test_a_pinned_n_that_does_not_divide_leaves_no_tactic():
    """FC1 N 3200: 128 divides it, 256 does not."""
    assert [t[0][1] for t in _tactics(8, 1600, 128, None)[0]] == [128]
    assert _tactics(8, 1600, 128, 256)[0] == []


def test_the_kernel_decides_the_rest():
    tactics, _ = _tactics(112, 3072, 128, None, accept=lambda tiler, *_: tiler[1] == 256)
    assert [t[0][1] for t in tactics] == [256]


# The launch against the engine runner


def _method(tree: ast.AST, cls: str, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"{cls}.{name} not found")


class _DropLeverArgs(ast.NodeTransformer):
    """The launcher's three helper calls, with the lever arguments the engine reads from the
    environment instead removed, and the die-local claim's variant (its choice, its cache key
    entry and its kernel class) and its weight-stream L2 prefetch (its depth, its cache key entry
    and its constructor argument) removed: what is left must be the engine's call."""

    PREFETCH_KWARGS = "dict(l2_prefetch=l2_prefetch) if l2_prefetch else {}"

    def visit_Assign(self, node):
        self.generic_visit(node)
        if [ast.unparse(t) for t in node.targets] == ["die_claim"]:
            assert ast.unparse(node.value.func) == "fc12_die_claim"
            return None
        if [ast.unparse(t) for t in node.targets] == ["l2_prefetch"]:
            assert ast.unparse(node.value) == "fc12_l2_prefetch(self.l2_prefetch, die_claim)"
            return None
        return node

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        if ast.unparse(node) == "cache_key += fc12_l2_prefetch_key(l2_prefetch)":
            return None
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        if ast.unparse(node.test) == "die_claim is not None":
            assert [ast.unparse(b) for b in node.body] == [
                "cache_key += (('die_claim', die_claim),)"
            ]
            return None
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Call) and getattr(node.func.func, "id", None) == (
            "fc12_kernel_class"
        ):
            assert [ast.unparse(a) for a in node.func.args] == [
                "self.__class__.kernel_class",
                "die_claim",
            ]
            node.func = node.func.args[0]
            prefetch = [
                k
                for k in node.keywords
                if k.arg is None and ast.unparse(k.value) == self.PREFETCH_KWARGS
            ]
            assert len(prefetch) == 1
            node.keywords = [k for k in node.keywords if k not in prefetch]
        name = getattr(node.func, "id", None)
        if name == "fc12_die_sm_words":
            node.keywords = [
                k for k in node.keywords if k.arg not in ("die_sched", "die_min_tokens")
            ]
        elif name == "fc12_sched_lookahead":
            assert ast.unparse(node.args[0]) == "self.lookahead"
            node.args = node.args[1:]
        elif name == "fc12_max_active_clusters":
            assert ast.unparse(node.args[2]) == "self.sm_reserve"
            node.args = node.args[:2]
        return node


@pytest.mark.parametrize("name", ["forward", "get_tuning_config"])
def test_the_launch_is_the_engine_runners_but_the_lever_reads(name):
    """Statement for statement the engine runner's ``forward`` / ``get_tuning_config``: the only
    difference is that the die schedule, its token floor, the lookahead and the SM reserve are
    the runner's arguments instead of environment reads, and that the die-local claim's variant
    picks the kernel class and joins the cache key. With ``die_claim`` None, the same tactic and
    lever values the launch therefore issues the same memset, compiles the same kernel and passes
    it the same arguments (the GPU test compares the bits)."""
    engine = _method(
        ast.parse(_ENGINE_SRC.read_text()),
        "Sm107BlockScaledContiguousGroupedGemmFusedFc12Runner",
        name,
    )
    ours = _method(ast.parse(Path(inspect.getfile(launch)).read_text()), "RubinFc12Runner", name)
    ours = _DropLeverArgs().visit(ours)
    assert ast.dump(ours) == ast.dump(engine)


def test_the_launch_check_sees_a_changed_launch():
    """The statement compare is not vacuous: one changed argument of the engine runner's launch
    (PDL off) or a lever read left in place fails it."""
    engine_src = _ENGINE_SRC.read_text()
    ours = _DropLeverArgs().visit(
        _method(ast.parse(Path(inspect.getfile(launch)).read_text()), "RubinFc12Runner", "forward")
    )
    for old, new in (
        ("use_pdl=True,", "use_pdl=False,"),
        ("sched_words = 33 if die_sched else 1", "sched_words = 33"),
    ):
        assert engine_src.count(old) >= 1
        mutated = _method(
            ast.parse(engine_src.replace(old, new)),
            "Sm107BlockScaledContiguousGroupedGemmFusedFc12Runner",
            "forward",
        )
        assert ast.dump(ours) != ast.dump(mutated), old


def test_lever_helpers_equal_the_engine_reads(monkeypatch):
    """Each helper with its lever as an argument returns what the engine helper returns with the
    lever in the environment: SM reserve, lookahead, die schedule and its floor."""
    from tensorrt_llm._torch import die_locality
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as engine

    for reserve, cluster, full in itertools.product((0, 1, 16, 40), (1, 2), (1, 74, 148)):
        monkeypatch.setenv("TRTLLM_K3_FC12_SM_RESERVE", str(reserve))
        assert launch.fc12_max_active_clusters(full, cluster, reserve) == (
            engine.fc12_max_active_clusters(full, cluster)
        )
    for lookahead, cluster in itertools.product((1, 2), (1, 2)):
        monkeypatch.setenv("TRTLLM_K3_FC12_SCHED_LOOKAHEAD", str(lookahead))
        assert launch.fc12_sched_lookahead(lookahead, cluster) == engine.fc12_sched_lookahead(
            cluster
        )
    table = (0, 1) * 74
    for alternating, has_table in itertools.product((True, False), (True, False)):
        monkeypatch.setattr(die_locality, "is_die_alternating", lambda t, a=alternating: a)
        monkeypatch.setattr(
            die_locality, "sm_die_table", lambda i, h=has_table: table if h else None
        )
        w = torch.empty(2, 2)
        for sched, floor, cluster, tokens in itertools.product(
            (False, True), (1, 24), (1, 2), (None, 8, 24, 96)
        ):
            monkeypatch.setenv("TRTLLM_K3_FC12_DIE_SCHED", "1" if sched else "0")
            monkeypatch.setenv("TRTLLM_K3_FC12_DIE_SCHED_MIN_TOKENS", str(floor))
            ours = launch.fc12_die_sm_words(
                w, w, cluster, tokens, die_sched=sched, die_min_tokens=floor
            )
            assert ours == engine.fc12_die_sm_words(w, w, cluster, tokens)
            if ours is not None:
                assert ours == die_locality.pack_sm_die_words(table)


def test_levers_outside_the_kernel_are_refused():
    for bad in (
        dict(mma_n=64),
        dict(lookahead=3),
        dict(sm_reserve=-1),
        dict(die_min_tokens=0),
        dict(die_sched=1),
        dict(die_claim="fast"),
        dict(die_claim="uniform", die_sched=False),
    ):
        with pytest.raises(ValueError, match="rubin_fc12"):
            launch.check_levers(**{**TEP8, **bad})
    launch.check_levers(**TEP8, die_claim="uniform")


def _runner_class():
    cls = getattr(launch, "RubinFc12Runner", None)
    if cls is None:
        pytest.skip("no CuTe DSL with Rubin support here: the runner is not defined")
    return cls


def test_runner_tactics_and_cache_identity(monkeypatch):
    """The runner hands its constructor N to fc12_tactics and keys its tuning on every lever; read
    on meta tensors with the kernel's can_implement stubbed."""
    cls = _runner_class()
    asked = []

    def can_implement(**kwargs):
        asked.append(kwargs["mma_tiler"][1])
        return True

    monkeypatch.setattr(cls.kernel_class, "can_implement", staticmethod(can_implement))
    u8 = dict(dtype=torch.uint8, device="meta")
    i32 = dict(dtype=torch.int32, device="meta")
    num_local, inter, rows = 224, 1536, 144
    permuted = rows * 16 + num_local * 128
    inputs = [
        torch.empty((rows, HIDDEN // 2), **u8),
        torch.empty((num_local, 2 * inter, HIDDEN // 2), **u8),
        torch.empty((rows, HIDDEN // 16), **u8),
        torch.empty((num_local, 2 * inter, HIDDEN // 16), **u8),
        torch.empty((num_local,), dtype=torch.float32, device="meta"),
        torch.empty((permuted // 128,), **i32),
        torch.empty((permuted // 128,), **i32),
        torch.empty((permuted,), **i32),
        torch.empty((1,), **i32),
        torch.empty((1,), dtype=torch.float32, device="meta"),
        torch.empty((num_local, HIDDEN, inter // 2), **u8),
    ]
    ids = []
    for mma_n in (None, 256):
        runner = object.__new__(cls)  # the constructor checks the GPU's SM version
        runner.__dict__.update(
            num_experts=896,
            top_k=16,
            num_local_experts=num_local,
            local_expert_offset=0,
            tile_size=128,
            scaling_vector_size=16,
            swiglu_limit=float("inf"),
            activation_type=0,
            situ_beta=None,
            situ_linear_beta=None,
            die_claim=None,
            l2_prefetch=0,
            **{**TEP8, "mma_n": mma_n},
        )
        asked.clear()
        tactics = runner.get_valid_tactics(inputs, None)
        assert [t[0][1] for t in tactics] == ([128, 256] if mma_n is None else [256])
        assert asked == ([128, 256] if mma_n is None else [256])
        ids.append(runner.unique_id())
    assert ids[0] != ids[1] and ids[1][10:] == (256, True, 24, 1, 16, None)
    assert (
        cls.kernel_cache
        is not getattr(
            __import__("tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops", fromlist=["x"]),
            "Sm107BlockScaledContiguousGroupedGemmFusedFc12Runner",
        ).kernel_cache
    )
    with pytest.raises(ValueError, match="mma_n"):
        cls(896, 16, num_local, 0, 128, **{**TEP8, "mma_n": 64})


def test_the_entry_passes_every_lever_to_the_launch(monkeypatch):
    """``__call__`` hands the 16 tensors in the engine op's order, the scalars, the soft caps in
    the engine's canonical form and the five levers to the launcher, unchanged."""
    seen = {}
    monkeypatch.setattr(
        mod, "launch_rubin_fc12", lambda inputs, **kw: seen.update(inputs=inputs, **kw)
    )
    args = _call_args()
    rubin_fc12(**args, **TEP8)
    names = list(inspect.signature(type(rubin_fc12).__call__).parameters)[1:17]
    assert [id(t) for t in seen["inputs"]] == [id(args[n]) for n in names]
    assert {k: seen[k] for k in TEP8} == TEP8
    assert (seen["situ_beta"], seen["situ_linear_beta"]) == (4.0, 25.0)
    rubin_fc12(**{**args, "situ_beta": -1.0}, **TEP8)
    assert seen["situ_beta"] is None
    assert seen["die_claim"] is None
    rubin_fc12(**args, **TEP8, die_claim="uniform")
    assert seen["die_claim"] == "uniform"


# The guard


def _call_args(tile=128, sort_tile=None, rows=8):
    """One call's operands on the CPU: dense and NVFP4-typed, small weights (the guard reads no
    weight shape), and the tables ``token_ordered_sort`` cuts at ``sort_tile`` (default ``tile``)
    for EP8 rank 0's 112 experts."""
    local, top_k, hidden, inter = 112, 16, 256, 64
    g = torch.Generator().manual_seed(rows)
    ids = torch.stack([torch.randperm(local, generator=g)[:top_k] for _ in range(rows)])
    tables = common.token_ordered_sort(ids, 0, local, sort_tile or tile)

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
        token_final_scales=torch.rand(rows, top_k, generator=g),
        expanded_idx_to_permuted_idx=tables[2],
        num_experts=896,
        top_k=top_k,
        num_local_experts=local,
        local_expert_offset=0,
        tile_size=tile,
        swiglu_limit=float("inf"),
        ep_size=8,
        enable_alltoall=False,
        activation_type=int(ActivationType.SiTu),
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )


def test_guard_admits_tables_cut_for_the_tile():
    """The routing tail's tables at 128 and moe_sort's at 256; zeroed counters with a zero
    output; the helix8 levers."""
    rubin_fc12.is_valid(**_call_args(), **TEP8)
    rubin_fc12.is_valid(**_call_args(tile=256), **{**TEP8, "mma_n": 256})
    rubin_fc12.is_valid(**_call_args(), zeroed_counters=torch.zeros(8, dtype=torch.int32), **TEP8)


def test_guard_rejects_tables_cut_for_another_tile():
    """The routing tail's tables read at 256. Their permuted rows are a multiple of 256, so the
    runner's own row check passes them."""
    args = _call_args(tile=256, sort_tile=128)
    assert args["permuted_idx_to_expanded_idx"].numel() % 256 == 0
    with pytest.raises(AssertionError, match="not cut for tile 256"):
        rubin_fc12.is_valid(**args, **TEP8)


def test_guard_rejects_what_the_runner_passes_unchecked():
    """The runner hands the kernel bare pointers: dtypes it does not check, strides, the NVFP4
    block size, the routing shape and the prezeroed buffers; and levers the kernel lacks."""
    base = _call_args()
    hidden = base["output"].shape[1]
    weights = base["token_final_scales"]
    zeroed = torch.zeros(8, dtype=torch.int32)
    for bad, match in (
        (dict(expanded_idx_to_permuted_idx=base["expanded_idx_to_permuted_idx"].long()), "int32"),
        (dict(tile_idx_to_group_idx=base["tile_idx_to_group_idx"].long()), "int32"),
        (dict(token_final_scales=weights.to(torch.bfloat16)), "FP32"),
        (dict(token_final_scales=weights.t().contiguous().t()), "token_final_scales must be"),
        (dict(output=torch.zeros(8, 2 * hidden, dtype=torch.bfloat16)[:, :hidden]), "output must"),
        (dict(scaling_vector_size=32), "NVFP4 blocks"),
        (dict(tile_idx_to_mn_limit=base["tile_idx_to_mn_limit"][:-1]), "differ in length"),
        (dict(top_k=8), "routing of"),
        (dict(output=torch.zeros(9, hidden, dtype=torch.bfloat16)), "routing of"),
        (dict(output=torch.ones(8, hidden, dtype=torch.bfloat16), zeroed_counters=zeroed), "zero"),
        (dict(zeroed_counters=torch.ones(8, dtype=torch.int32)), "are not zero"),
    ):
        with pytest.raises(AssertionError, match=match):
            rubin_fc12.is_valid(**{**base, **bad}, **TEP8)
    with pytest.raises(ValueError, match="lookahead"):
        rubin_fc12.is_valid(**base, **{**TEP8, "lookahead": 4})


def test_levers_have_no_defaults():
    """A call site states every lever: none of the five has a default on the entry. The die-local
    claim's variant defaults to the engine kernel's claim (keyword only)."""
    for method in (type(rubin_fc12).__call__, type(rubin_fc12).is_valid):
        params = inspect.signature(method).parameters
        for lever in TEP8:
            assert params[lever].kind is inspect.Parameter.KEYWORD_ONLY
            assert params[lever].default is inspect.Parameter.empty
        assert params["die_claim"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["die_claim"].default is None


def test_activation_defaults_to_the_engine_ops():
    """SwiGLU, as cute_dsl_nvfp4_fc12_fused_rubin's default; 0 is ActivationType.InvalidType,
    which the kernel refuses to build."""
    for method in (type(rubin_fc12).__call__, type(rubin_fc12).is_valid):
        default = inspect.signature(method).parameters["activation_type"].default
        assert default == int(ActivationType.Swiglu)


# Construction: the lever variables and the die placement


def test_lever_env_unset_or_equal():
    """The TEP8 / helix8 generation lever variables equal the target's levers; another value,
    or any of them on a target without FC12, raises; unset always passes."""
    levers = dict(die_layout=True, die_sched=True, die_min_tokens=24, lookahead=1, sm_reserve=16)
    check_fc12_lever_env(TEP8_ENV, **levers)
    check_fc12_lever_env({}, **levers)
    check_fc12_lever_env({"TRTLLM_K3_FC12_SCHED_LOOKAHEAD": "1"}, **levers)
    for name, value in (
        ("TRTLLM_K3_FC12_SM_RESERVE", "8"),
        ("TRTLLM_K3_FC12_DIE_SCHED", "0"),
        ("TRTLLM_K3_FC12_DIE_SCHED_MIN_TOKENS", "1"),
        ("TRTLLM_K3_FC12_SCHED_LOOKAHEAD", "2"),
        ("TRTLLM_K3_FC12_DIE_LAYOUT", "0"),
    ):
        with pytest.raises(ValueError, match=name):
            check_fc12_lever_env({**TEP8_ENV, name: value}, **levers)
    none = dict.fromkeys(levers)
    check_fc12_lever_env({}, **none)
    with pytest.raises(ValueError, match="runs no FC12"):
        check_fc12_lever_env({"TRTLLM_K3_FC12_SM_RESERVE": "16"}, **none)


def test_place_per_die_rehomes_both_weights_of_every_layer(monkeypatch):
    """Each layer's FC1 and FC2 parameters point at their die-alternating copies (same bytes),
    layer by layer with the originals released in between; the SM -> die table is measured only
    for die-local claiming; a device without die memory keeps every weight where it was."""
    from tensorrt_llm._torch import die_locality

    events = []

    def copy(t):
        events.append(("copy", t.shape))
        return t.clone()

    monkeypatch.setattr(die_locality, "die_alternating_copy", copy)
    monkeypatch.setattr(die_locality, "sm_die_table", lambda i: events.append(("table", i)))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append(("sync",)))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append(("free",)))
    g = torch.Generator().manual_seed(0)
    w = torch.nn.ParameterDict(
        {
            f"l{i}_moe_{n}_w": torch.nn.Parameter(
                torch.randint(0, 255, (2, 8, 4), generator=g).to(torch.uint8), requires_grad=False
            )
            for i in (1, 2)
            for n in ("fc1", "fc2")
        }
    )
    before = {k: (p.data_ptr(), p.detach().clone()) for k, p in w.items()}
    assert rubin_fc12_place_per_die(w, [1, 2], die_sched=True) == 2
    for k, (ptr, data) in before.items():
        assert w[k].data_ptr() != ptr and torch.equal(w[k], data)
    assert [e[0] for e in events] == ["copy", "copy", "sync", "free"] * 2 + ["table"]
    events.clear()
    assert rubin_fc12_place_per_die(w, [1], die_sched=False) == 1
    assert "table" not in [e[0] for e in events]
    monkeypatch.setattr(die_locality, "die_alternating_copy", lambda t: None)
    kept = {k: p.data_ptr() for k, p in w.items()}
    assert rubin_fc12_place_per_die(w, [1, 2], die_sched=True) == 0
    assert {k: p.data_ptr() for k, p in w.items()} == kept


# The NVFP4 numerics and the reference


def test_e2m1_values_low_nibble_first():
    codes = torch.arange(256, dtype=torch.uint8).reshape(1, 256)
    values = common.e2m1_values(codes).reshape(256, 2)
    for byte, (lo, hi) in enumerate(values.tolist()):
        assert lo == common._E2M1[byte & 0xF] and hi == common._E2M1[byte >> 4]


def test_quantize_nvfp4_keeps_grid_values_and_rounds_to_even():
    grid = torch.tensor(common._E2M1[:8] * 2).reshape(1, 16)  # one block, amax 6
    for g in (1.0, 0.5, 8.0):
        x = grid / g
        assert torch.equal(common.quantize_nvfp4(x, g), x)
    assert torch.equal(common.quantize_nvfp4(torch.zeros(2, 16), 1.0), torch.zeros(2, 16))
    ties = torch.tensor([[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0] * 2])
    assert common.quantize_nvfp4(ties, 1.0).tolist()[0][:7] == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0]


def test_dequantize_nvfp4_applies_block_scales():
    codes = torch.full((2, 16), 0x22, dtype=torch.uint8)  # 1.0, 1.0
    scales = torch.tensor([[0.5, 2.0], [1.0, 4.0]]).to(torch.float8_e4m3fn)
    out = common.dequantize_nvfp4(codes, scales)
    assert out[0, :16].eq(0.5).all() and out[0, 16:].eq(2.0).all() and out[1, 16:].eq(4.0).all()


def _random_experts(num, hidden, inter, seed, input_scale=1.0):
    g = torch.Generator().manual_seed(seed)

    def proj(n, k):
        codes = torch.randint(0, 256, (n, k // 2), generator=g, dtype=torch.uint8)
        scales = (torch.rand(n, k // 16, generator=g) * 2 + 0.25).to(torch.float8_e4m3fn)
        return codes, scales, torch.tensor(2.0**-6), torch.tensor(input_scale)

    return [
        {"w1": proj(inter, hidden), "w3": proj(inter, hidden), "w2": proj(hidden, inter)}
        for _ in range(num)
    ]


def _loop_reference(x, ids, weights, experts, offset, beta, linear_beta):
    """Token by token and slot by slot (an independent formulation of the same arithmetic)."""
    g1 = 1.0 / max(float(e[w][3]) for e in experts for w in ("w1", "w3"))
    g2 = 1.0 / max(float(e["w2"][3]) for e in experts)
    out = torch.zeros(x.shape[0], experts[0]["w2"][0].shape[0], dtype=torch.float64)
    for t, k in itertools.product(range(x.shape[0]), range(ids.shape[1])):
        j = int(ids[t, k]) - offset
        if not 0 <= j < len(experts):
            continue
        e = experts[j]
        w = {n: (common.dequantize_nvfp4(e[n][0], e[n][1]) * e[n][2]).double() for n in e}
        xt = common.quantize_nvfp4(x[t : t + 1].float(), g1).double()[0]
        gate, up = w["w1"] @ xt, w["w3"] @ xt
        act = mod.situ(gate, up, beta, linear_beta).float().unsqueeze(0)
        out[t] += float(weights[t, k]) * (w["w2"] @ common.quantize_nvfp4(act, g2).double()[0])
    return out


@pytest.mark.parametrize("input_scale", [1.0, 0.02])
def test_reference_matches_a_token_loop(input_scale):
    hidden, inter, num, top_k, rows = 64, 32, 6, 2, 5
    experts = _random_experts(num, hidden, inter, 1, input_scale)
    g = torch.Generator().manual_seed(2)
    x = (torch.randn(rows, hidden, generator=g) * 0.2).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(num + 2, generator=g)[:top_k] for _ in range(rows)])
    weights = torch.rand(rows, top_k, generator=g)
    for offset in (0, 2):
        ref = rubin_fc12.reference(
            x,
            ids,
            weights,
            experts,
            local_expert_offset=offset,
            situ_beta=8.0,
            situ_linear_beta=20.0,
        )
        loop = _loop_reference(x, ids, weights, experts, offset, 8.0, 20.0)
        assert ref.dtype == torch.float32
        torch.testing.assert_close(ref.double(), loop, rtol=1e-5, atol=1e-6)


def test_reference_windows_sum_to_the_layer():
    """Ids outside a window add nothing, so two ranks' windows sum to one rank holding both (each
    window here quantizes under the same global scales)."""
    hidden, inter = 64, 32
    experts = _random_experts(4, hidden, inter, 3)
    g = torch.Generator().manual_seed(4)
    x = torch.randn(3, hidden, generator=g).to(torch.bfloat16)
    ids = torch.tensor([[0, 3], [1, 2], [2, 0]])
    weights = torch.rand(3, 2, generator=g)
    kw = dict(situ_beta=8.0, situ_linear_beta=20.0)
    whole = mod.nvfp4_situ_moe_reference(x, ids, weights, experts, local_expert_offset=0, **kw)
    left = mod.nvfp4_situ_moe_reference(x, ids, weights, experts[:2], local_expert_offset=0, **kw)
    right = mod.nvfp4_situ_moe_reference(x, ids, weights, experts[2:], local_expert_offset=2, **kw)
    torch.testing.assert_close(left + right, whole, rtol=1e-6, atol=1e-6)
