# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""k3 (q4), k4 (q5) and k5 (q6) beside k7 (q8) on the same target (CPU only, on the sources):
``RuntimeLimits`` and the drafter's q; the gates that take each q (the chain, ``FrontOnce``, N6,
``select_accept``, the o-projection's x buffer, the AttnRes carve-out); routing (the wide fork
takes k3 / k4 / k5 / k7, the ULL target k7 only; the switches alike at every k); the kernels' own
q bounds.
"""

from __future__ import annotations

import ast
import re
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from _helix16_wide_cpu import (
    FAMILY_PKG,
    family_dir,
    function_node,
    literal,
    modeling_path,
    source_objects,
)

# Fixtures and helpers of the front and routing tests (pytest registers imported fixtures here too).
from test_helix16_wide_front_once import _front_once, front, policy  # noqa: F401
from test_helix16_wide_routing import SWITCHES, WIDE, XIN, ull_route  # noqa: F401

pytestmark = pytest.mark.cpu_only

DRAFT_NAMES = (
    "DRAFT_LEN", "DRAFT_TILE_REQUESTS", "_SELECT_ACCEPT_MAX_REQUESTS", "DRAFT_LENS",
    "VERIFY_MAX_ROWS", "MAX_REQUESTS", "max_requests", "verify_tokens", "oproj_capacity",
    "draft_tile_requests", "select_accept_max_requests",
)  # fmt: skip
#: q -> (k, the largest B, its words, its rows, its row-count count)
Q = {
    8: (7, 16, "sixteen", 128, 30),
    6: (5, 21, "twenty-one", 126, 39),
    5: (4, 25, "twenty-five", 125, 45),
    4: (3, 32, "thirty-two", 128, 56),
}
RECORD = {"tail": "tuned", "front": "tmem_merged"}  # the wide fork's record at the defaults


@pytest.fixture(scope="module")
def dl():
    return SimpleNamespace(**source_objects(modeling_path(), DRAFT_NAMES, {}))


@pytest.fixture(scope="module")
def limits():
    """``RuntimeLimits`` in a registered module of its own (a dataclass resolves its string
    annotations through ``sys.modules``)."""
    module = types.ModuleType("_k31_helix16_wide_draft_lens_limits")
    module.dataclass = dataclass
    sys.modules[module.__name__] = module
    try:
        source_objects(modeling_path(), (*DRAFT_NAMES, "RuntimeLimits"), module.__dict__)
        yield module.RuntimeLimits
    finally:
        sys.modules.pop(module.__name__, None)


# --- RuntimeLimits ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", sorted(Q))
def test_the_limits_at_each_q(limits, q):
    """B 1..top at q rows a request: {1..B} and {qb} rows; the largest B's rows and row counts;
    0 or top + 1 requests and a short token budget refused with their messages."""
    k, top, words, rows, count = Q[q]
    for b in range(1, top + 1):
        lim = limits(max_batch_size=b, max_num_tokens=q * b, tokens=q)
        assert lim.tokens == q and lim.max_rows == q * b == max(lim.rows) <= 128
        assert lim.rows == tuple(sorted({*range(1, b + 1), *(q * r for r in range(1, b + 1))}))
    largest = limits(top, rows, q)
    assert max(largest.rows) == rows and len(largest.rows) == count
    requests = f"^The native flat runtime supports one through {words} GEN requests$"
    with pytest.raises(ValueError, match=requests):
        limits(max_batch_size=top + 1, max_num_tokens=16384, tokens=q)
    with pytest.raises(ValueError, match=requests):
        limits(max_batch_size=0, max_num_tokens=q, tokens=q)
    with pytest.raises(
        ValueError,
        match=f"^The engine token budget must cover the configured DL{k} verification batch$",
    ):
        limits(max_batch_size=top, max_num_tokens=q * top - 1, tokens=q)


@pytest.mark.parametrize("tokens", [1, 2, 3, 7, 9])
def test_other_verify_lengths_are_refused(limits, tokens):
    with pytest.raises(ValueError, match="q4 or q5 or q6 or q8"):
        limits(max_batch_size=1, max_num_tokens=16, tokens=tokens)


def test_the_verify_length_is_the_drafters(dl, limits):
    spec = lambda k: SimpleNamespace(spec_config=SimpleNamespace(max_draft_len=k))  # noqa: E731
    assert [dl.verify_tokens(spec(k)) for k in (3, 4, 5, 7)] == [4, 5, 6, 8]
    # Another draft length (or none) keeps q8: validate_flat_config then refuses it.
    for k in (None, 0, 1, 2, 6, 8):
        assert dl.verify_tokens(spec(k)) == 8
    assert dl.verify_tokens(SimpleNamespace(spec_config=None)) == 8
    assert [dl.max_requests(t) for t in (4, 5, 6, 8)] == [32, 25, 21, 16]
    assert dl.MAX_REQUESTS == 32 and dl.DRAFT_LENS == (3, 4, 5, 7) and dl.VERIFY_MAX_ROWS == 128
    assert limits(max_batch_size=16, max_num_tokens=128) == limits(16, 128, 8)  # q8 by default


def test_validate_flat_config_takes_every_draft_len():
    source = ast.unparse(function_node(modeling_path(), "validate_flat_config"))
    assert "if spec.max_draft_len not in DRAFT_LENS:" in source
    assert "spec.max_draft_len != DRAFT_LEN" not in source


# --- the gates -------------------------------------------------------------------------------


def test_the_chains_gate_takes_q4_to_q8_up_to_128_rows(dl):
    chain = function_node(modeling_path(), "FlatChainRuntime.__init__")
    test = next(
        n.test
        for n in ast.walk(chain)
        if isinstance(n, ast.If) and "DRAFT_LENS" in ast.unparse(n.test)
    )
    gate = compile(ast.Expression(test), "gate", "eval")
    for tokens in (4, 5, 6, 8):
        top = 128 // tokens
        rows = tuple(sorted({*range(1, top + 1), *(tokens * b for b in range(1, top + 1))}))
        assert not eval(gate, {"tokens": tokens, "rows": rows, "DRAFT_LENS": dl.DRAFT_LENS})
        assert eval(gate, {"tokens": tokens, "rows": (*rows, 129), "DRAFT_LENS": dl.DRAFT_LENS})
    for tokens in (2, 3, 7, 9):
        assert eval(gate, {"tokens": tokens, "rows": (1,), "DRAFT_LENS": dl.DRAFT_LENS})


@pytest.mark.parametrize("q", [4, 5, 6, 8])
def test_front_once_takes_whole_q_row_requests(front, policy, q):  # noqa: F811
    wider = tuple(range(q * (64 // q + 1), 129, q))
    once, *_ = _front_once(front, policy, [], rows=wider, q=q)
    assert sorted(once.plans) == list(wider)
    for rows in ((wider[0] + 1,), (64,), (132,)):
        with pytest.raises(ValueError, match=f"multiples of {q}"):
            _front_once(front, policy, [], rows=rows, q=q, cap=136)


def test_the_o_projection_x_buffer(dl):
    """q8: the step's rows; q4 / q5 / q6 above 64 rows: whole 64-row x tiles (128)."""
    for q in (4, 5, 6, 8):
        want = [r if q == 8 or r <= 64 else 128 for r in range(1, 129)]
        assert [dl.oproj_capacity(r, q) for r in range(1, 129)] == want, q


def test_n6_requests_and_launch_at_each_q():
    """N6's tiles at ``draft_tile_requests(block)`` requests on Rubin, the ULL target's elsewhere."""
    rubin = {"on": True}
    ns = {"_rubin_rows": lambda device: rubin["on"], "RANKED_TILES": (128, 64, 8, 1, None)}
    names = (
        "DRAFT_TILE_WARPS", "DRAFT_TILE_CAP", "DRAFT_TILES", "_DRAFT_RANKED_TILES_RUBIN",
        "_DRAFT_RANKED_TILES_RUBIN_REST", "_ranked_tiles", "draft_ranked_launch", *DRAFT_NAMES,
    )  # fmt: skip
    source_objects(modeling_path(), names, ns)
    requests, launch = ns["draft_tile_requests"], ns["draft_ranked_launch"]
    assert requests(8) == ns["DRAFT_TILE_REQUESTS"] == range(9, 17)
    assert (requests(6), requests(5), requests(4)) == (range(9, 22), range(9, 26), range(9, 33))
    for block in (2, 3, 7, 9):  # another block keeps 9-16
        assert requests(block) == range(9, 17)
    on = ((64, 64, 8, 1, None), True, 64)
    rest = ((128, 128, 8, 1, None), False, 0)
    one = ((128, 128, 8, 2, None), False, 0)  # one request: two column splits
    for block, top in ((8, 16), (6, 21), (5, 25), (4, 32)):
        for b in range(1, top + 2):
            off = one if b == 1 else rest
            assert launch(block * b, None, block=block) == (on if 9 <= b <= top else off)
    rubin["on"] = False
    assert launch(8 * 12, None, block=8) == ((128, 64, 8, 1, None), False, 0)


def test_select_accept_fuses_every_request_of_a_step(dl):
    accept = dl.select_accept_max_requests
    assert [accept(k) for k in (3, 4, 5, 7)] == [32, 25, 21, 16]
    assert dl._SELECT_ACCEPT_MAX_REQUESTS == 16
    for k in (1, 2, 6, 8):  # another draft length keeps 16
        assert accept(k) == 16
    fused = ast.unparse(function_node(modeling_path(), "DFlashWorker._fused_select_accept"))
    assert (
        "if max(num_contexts, num_gens) > select_accept_max_requests(self.max_draft_len):" in fused
    )


def test_the_attn_res_before_the_kda_layers_keeps_q8s_carveout():
    """The carve-out of the AttnRes before every KDA layer keys on q8's fused head at every q: at
    q8 the head's rows, at another q whether the head would run at q8 here."""
    chain = function_node(modeling_path(), "FlatChainRuntime.__init__")
    text = ast.unparse(chain)
    assert "carveout = head_carveout if before_head and i in self.kda_weights else None" in text
    assign = next(
        n for n in ast.walk(chain)
        if isinstance(n, ast.Assign) and [ast.unparse(t) for t in n.targets] == ["before_head"]
    )  # fmt: skip
    expr = compile(ast.Expression(assign.value), "before_head", "eval")
    for tokens in (8, 6):
        for head_rows in (frozenset(), frozenset({8})):
            for supported in (False, True):
                if head_rows and (tokens != 8 or not supported):
                    continue  # q8 runs the head only where it is supported; q6 never
                seen = []
                flat = SimpleNamespace(head_supported=lambda w, t: seen.append(t) or supported)
                ns = {"self": SimpleNamespace(head_rows=head_rows, kda_weights={1: "w1", 3: "w3"}),
                      "tokens": tokens, "DRAFT_LEN": 7, "FlatKda": flat}  # fmt: skip
                value = eval(expr, ns)  # noqa: S307 (the source's own expression)
                if tokens == 8:
                    assert value == bool(head_rows) and not seen  # the head's rows
                else:
                    assert value == supported and set(seen) == {8}


# --- routing ---------------------------------------------------------------------------------


def _features():
    try:
        import importlib

        return importlib.import_module(f"{FAMILY_PKG}.ops.common.rubin_ull_features")
    except Exception as error:  # noqa: BLE001 (any import failure of the engine's dependencies)
        pytest.skip(f"the recipe module does not import on this host: {error}")


def _recipe(features, **changes):
    fields = dict(
        indexer_k_dtype="fp8", kv_cache_quant_algo="FP8", spec_mode="DFLASH", max_draft_len=7,
        moe_backend="CUTEDSL", role="generation",
        layout=features.Layout(
            world_size=16, tp_size=1, pp_size=1, cp_size=16, helix=True, attention_dp=False,
            moe_tp_size=16, moe_ep_size=1,
        ),
    )  # fmt: skip
    fields.update(changes)
    return fields


def test_decide_keeps_k7_only_and_decide_wide_adds_k3_k4_k5():
    features = _features()
    assert features.WIDE_DRAFT_LENS == (3, 4, 5, features.DRAFT_LEN) == (3, 4, 5, 7)
    for k in (7, 5, 4, 3):
        wide = features.decide_wide(**_recipe(features, max_draft_len=k))
        assert wide == features.decide(**_recipe(features)) and wide.reason is None
    assert features.decide(**_recipe(features, max_draft_len=5)).reason == (
        "max_draft_len 5: ULL verifies blocks of 8 rows (max_draft_len 7) (" + features.RECIPE + ")"
    )
    for k in (None, 0, 1, 2, 6, 8):
        reason = features.decide_wide(**_recipe(features, max_draft_len=k)).reason
        assert reason == (
            f"max_draft_len {k}: the ULL wide-batch target verifies blocks of 4 or 5 or 6 or 8 rows "
            f"(max_draft_len 3 or 4 or 5 or 7) ({features.RECIPE})"
        )
    # The other checks, in decide's order, are the same for both.
    for changes in (
        dict(indexer_k_dtype="fp4"), dict(kv_cache_quant_algo="NVFP4"), dict(spec_mode=None),
        dict(moe_backend="TRTLLM"), dict(role="context"), dict(spec_mode="MTP", max_draft_len=5),
        dict(moe_backend="TRTLLM", max_draft_len=5),
    ):  # fmt: skip
        ours = features.decide_wide(**_recipe(features, **{"max_draft_len": 7, **changes}))
        theirs = features.decide(**_recipe(features, **{"max_draft_len": 7, **changes}))
        assert ours.reason is not None
        if changes.get("max_draft_len", 7) == 7:
            assert ours == theirs


@pytest.mark.parametrize(
    "draft, wide, to",
    [(7, False, XIN), (7, True, WIDE), (5, True, WIDE), (4, True, WIDE), (3, True, WIDE),
     (5, False, None), (4, False, None), (3, False, None), (6, True, None), (8, True, None)],
)  # fmt: skip
def test_each_draft_length_routes_or_is_refused(ull_route, monkeypatch, draft, wide, to):  # noqa: F811
    routing = sys.modules[f"{FAMILY_PKG}.routing"]
    route, targets, selected, ull, config = ull_route
    for name in SWITCHES:
        monkeypatch.delenv(name, raising=False)
    config.spec_config.max_draft_len = draft
    monkeypatch.setenv("TLLM_K31_HELIX16_WIDE", "1" if wide else "0")
    if to is None:
        lens = (
            "4 or 5 or 6 or 8 rows (max_draft_len 3 or 4 or 5 or 7)"
            if wide
            else "8 rows (max_draft_len 7)"
        )
        with pytest.raises(routing.NoTarget, match=re.escape(lens)):
            route()
        assert selected == [] and config.extra_attrs == {}
        return
    assert route() is targets[to[0]]
    assert selected == [ull.ULL16_GENERATION]
    assert config.extra_attrs.get("kimi_k31_helix16_wide") == (RECORD if to is WIDE else None)


@pytest.mark.parametrize(
    "env", [{}, {"TLLM_K31_HELIX16_WIDE_TAIL": "rows", "TLLM_K31_HELIX16_WIDE_FRONT": "once"}]
)
def test_the_switches_take_every_k_alike(ull_route, monkeypatch, env):  # noqa: F811
    """No switch is refused at another q: the record holds no k (the target reads it off the
    drafter) and is the same at k3, k4, k5 and k7."""
    route, targets, _, _, config = ull_route
    records = {}
    for draft in (7, 5, 4, 3):
        config.extra_attrs.clear()
        config.spec_config.max_draft_len = draft
        monkeypatch.setenv("TLLM_K31_HELIX16_WIDE", "1")
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        assert route() is targets[WIDE[0]]
        records[draft] = dict(config.extra_attrs["kimi_k31_helix16_wide"])
    assert records[3] == records[4] == records[5] == records[7]


def test_the_levers_take_no_q():
    """The levers' tables and ``WideMoeOptions`` take no q: they key on rows or requests."""
    for name in ("step_levers", "fc12_row_levers", "WideMoeOptions", "WideTailRows"):
        names = set()
        for n in ast.walk(function_node(modeling_path(), name)):
            names.update(getattr(n, field, None) for field in ("id", "attr", "arg"))
        assert not names & {"tokens", "q", "DRAFT_LEN", "DRAFT_LENS", "max_draft_len"}, name


# --- the kernels' own q bounds ---------------------------------------------------------------


def test_the_kernels_take_q4_q5_q6():
    """The DSA scorer at q4 / q5 / q6 (its bound, its epilogue in one warpgroup and the tensor
    memory), the key shard's 256-column atom, the verify kernels' draft bound; the fused KDA head
    is q8's only and the KDA layers never build MR8's prepared (even-q) verify."""
    score = family_dir() / "kernels/attention/rubin_ull_dsa_index_score.py"
    text = score.read_text()
    assert (
        "assert rows % q == 0 and 1 <= q <= 8 and rows // q <= MAX_REQUESTS and pitch > 0" in text
    )
    assert literal(score, "MAX_REQUESTS") >= 32
    ns = source_objects(
        score, ("ROWS_PER_WARPGROUP", "score_epilogue", "HEADS", "TMEM_COLUMNS"), {}
    )
    assert [ns["score_epilogue"](q) for q in (4, 5, 6)] == [(2, 1), (1, 2), (1, 2)]
    for q in (4, 5, 6):
        accs, split = ns["score_epilogue"](q)
        assert -(-q // split) <= ns["ROWS_PER_WARPGROUP"]
        assert accs * q * ns["HEADS"] <= ns["TMEM_COLUMNS"]
    ready = ast.unparse(function_node(modeling_path(), "FlatDsa.key_shard_ready"))
    assert "next_n * 32 <= 256" in ready
    for name in ("dflash2_greedy_verify.py", "dflash2_block_verify.py"):
        assert literal(family_dir() / "kernels/spec" / name, "MAX_DRAFTS") >= 4
    assert literal(modeling_path(), "HEAD_ROWS") == (8,)
    supported = ast.unparse(function_node(modeling_path(), "FlatKda.head_supported"))
    assert "tokens != 8" in supported
    init = ast.unparse(function_node(modeling_path(), "FlatKda.__init__"))
    assert "core.prepared = frozenset()" in init
