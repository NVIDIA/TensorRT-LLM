# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for attention-DP groups (``tensorrt_llm/_torch/adp_groups.py``, ``TRTLLM_ADPTP_ATTN_TP``): DPxTPy-EPz
with x groups of y tensor-parallel attention ranks (MSO H8 = DP2TP4-EP8, H16 = DP4TP4-EP16, H32 = DP8TP4-EP32).

No torch / CUDA import: ``adp_groups`` is pure Python and is loaded by path; the real ``DefaultADPRouter`` is loaded
from source with its package imports stubbed. Runs with ``python -m unittest`` or pytest.
"""

import importlib.util
import os
import re
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
ADP_GROUPS = ROOT / "tensorrt_llm/_torch/adp_groups.py"
ENV = "TRTLLM_ADPTP_ATTN_TP"
CHECK_ENV = "TRTLLM_ADPTP_CHECK"
GROUP_COUNTS = (1, 2, 4, 8)  # W1: G groups of T=4 (world 4 / 8 / 16 / 32)
T = 4


def load_adp_groups():
    spec = importlib.util.spec_from_file_location("tensorrt_llm._torch.adp_groups", ADP_GROUPS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


G = load_adp_groups()


class _Mapping:
    """The attributes adp_groups reads from ``tensorrt_llm.mapping.Mapping``."""

    def __init__(self, world=8, tp_rank=0, enable_attention_dp=True, tp_size=None, pp_size=1, cp_size=1,
                 enable_lm_head_tp_in_adp=False):
        self.world_size = world
        self.tp_size = world if tp_size is None else tp_size
        self.tp_rank = tp_rank
        self.rank = tp_rank
        self.pp_size = pp_size
        self.cp_size = cp_size
        self.enable_attention_dp = enable_attention_dp
        self.enable_lm_head_tp_in_adp = enable_lm_head_tp_in_adp


def mapping(world=8, tp_rank=0, **kw):
    return _Mapping(world, tp_rank, **kw)


class TestAdpGroups(unittest.TestCase):
    def test_env_names(self):
        self.assertEqual(G.ENV, ENV)
        self.assertEqual(G.CHECK_ENV, CHECK_ENV)

    def test_unset_or_one_is_the_stock_answer(self):
        """Switch unset (or 1): every helper returns stock attention DP's answer (default-off == base)."""
        for env in ({}, {ENV: ""}, {ENV: "1"}, {ENV: " "}):
            for world in (2, 4, 8, 16, 32):
                for r in range(world):
                    m = mapping(world, r)
                    with self.subTest(env=env, world=world, rank=r):
                        self.assertEqual(G.group_attn_tp(m, env), 1)
                        self.assertFalse(G.is_grouped(m, env))
                        self.assertEqual(G.attn_tp_size(m, env), 1)  # stock: 1 if enable_attention_dp
                        self.assertEqual(G.attn_tp_rank(m, env), 0)
                        self.assertEqual(G.dp_size(m, env), world)  # stock: tp_size DP ranks
                        self.assertEqual(G.dp_rank(m, env), r)
                        self.assertTrue(G.is_group_leader(m, env))
                        self.assertEqual(G.attention_shard(r, world, True, env), (0, 1))
                        self.assertEqual(G.attention_shard(r, world, False, env), (r, world))
        states = [types.SimpleNamespace(rank=r) for r in range(4)]
        self.assertIs(G.route_states(states, 1)[2], states[2])
        self.assertEqual(G.sliced_rank_tokens([3, 5, 0, 9], 1), [3, 5, 0, 9])
        self.assertEqual(G.collapse([1, 2, 3], 1), [1, 2, 3])
        self.assertEqual(G.expand({0: "a", 1: "b"}, 1, 2), {0: "a", 1: "b"})
        self.assertEqual(G.diverged_groups([1, 2, 3], 1), [])
        # the os.environ default path
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(G.group_attn_tp(mapping(8)), 1)
            self.assertTrue(G.check_enabled())

    def test_no_attention_dp_is_plain_tp(self):
        m = mapping(4, 2, enable_attention_dp=False)
        env = {ENV: "4"}
        self.assertEqual(G.group_attn_tp(m, env), 1)
        self.assertEqual(G.attn_tp_size(m, env), 4)
        self.assertEqual(G.attn_tp_rank(m, env), 2)
        self.assertEqual(G.dp_size(m, env), 1)
        self.assertEqual(G.dp_rank(m, env), 0)
        self.assertEqual(G.attn_tp_size(None, env), 1)
        self.assertEqual(G.group_attn_tp(None, env), 1)
        self.assertEqual(G.attention_shard(2, 4, False, env), (2, 4))

    def test_legacy_alias(self):
        """The M3.1 names are read only when the new one is unset; the new name wins."""
        m = mapping(8, 5)
        self.assertEqual(G.group_attn_tp(m, {G.LEGACY_ENV: "4"}), 4)
        self.assertEqual(G.group_attn_tp(m, {ENV: "2", G.LEGACY_ENV: "4"}), 2)
        self.assertEqual(G.group_attn_tp(m, {ENV: "", G.LEGACY_ENV: "4"}), 4)
        self.assertEqual(G.group_attn_tp(m, {ENV: "1", G.LEGACY_ENV: "4"}), 1)
        self.assertFalse(G.check_enabled({G.LEGACY_CHECK_ENV: "0"}))
        self.assertTrue(G.check_enabled({CHECK_ENV: "1", G.LEGACY_CHECK_ENV: "0"}))
        with self.assertRaisesRegex(ValueError, G.LEGACY_ENV):
            G.group_attn_tp(m, {G.LEGACY_ENV: "x"})

    def test_dp2tp4_layout(self):
        env = {ENV: "4"}
        for r in range(8):
            m = mapping(8, r)
            with self.subTest(rank=r):
                self.assertEqual(G.group_attn_tp(m, env), 4)
                self.assertEqual(G.attn_tp_size(m, env), 4)
                self.assertEqual(G.attn_tp_rank(m, env), r % 4)
                self.assertEqual(G.dp_size(m, env), 2)
                self.assertEqual(G.dp_rank(m, env), r // 4)
                self.assertEqual(G.is_group_leader(m, env), r in (0, 4))
                self.assertIn(r, G.group_tp_ranks(r // 4, 4))
                self.assertEqual(G.attention_shard(r, 8, True, env), (r % 4, 4))

    def test_one_group_is_attention_tp_over_every_rank(self):
        env = {ENV: "4"}
        for r in range(4):
            m = mapping(4, r)
            self.assertEqual((G.dp_size(m, env), G.dp_rank(m, env), G.attn_tp_rank(m, env)), (1, 0, r))

    def test_bad_switch_or_mapping_raises(self):
        for raw in ("x", "0", "-4", "2.5"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                G.group_attn_tp(mapping(8), {ENV: raw})
        for m, why in (
            (mapping(8, tp_size=8), None),
            (mapping(6), "does not divide"),
            (mapping(8, pp_size=2), "pp_size"),
            (mapping(8, cp_size=2), "cp_size"),
            (mapping(8, enable_lm_head_tp_in_adp=True), "enable_lm_head_tp_in_adp"),
        ):
            got = G.topology_mismatch(m, 4)
            if why is None:
                self.assertIsNone(got)
            else:
                self.assertIn(why, got)
                with self.assertRaises(ValueError):
                    G.group_attn_tp(m, {ENV: "4"})
        self.assertIn("does not divide", G.topology_mismatch(mapping(4), 8))
        self.assertIn("enable_attention_dp is off", G.topology_mismatch(mapping(4, enable_attention_dp=False), 4))
        with self.assertRaises(ValueError):
            G.attention_shard(0, 6, True, {ENV: "4"})

    def test_row_slices_cover_the_rows_once(self):
        for t in (1, 2, 4, 8):
            for rows in range(0, 70):
                seen = []
                widths = set()
                for k in range(t):
                    start, stop, width = G.row_slice(rows, t, k)
                    self.assertLessEqual(stop - start, width)
                    seen += list(range(start, stop))
                    widths.add(width)
                with self.subTest(t=t, rows=rows):
                    self.assertEqual(seen, list(range(rows)))
                    self.assertEqual(len(widths), 1)
                    # The group all-gather of t width-row outputs holds the real rows first.
                    self.assertGreaterEqual(t * widths.pop(), rows)
        with self.assertRaises(ValueError):
            G.row_slice(5, 4, 4)

    def test_sliced_rank_tokens(self):
        self.assertEqual(G.sliced_rank_tokens([9] * 4 + [0] * 4, 4), [3] * 4 + [0] * 4)
        self.assertEqual(G.sliced_rank_tokens([32] * 4 + [17] * 4, 4), [8] * 4 + [5] * 4)
        with self.assertRaises(ValueError):
            G.sliced_rank_tokens([9, 9, 9, 8], 4)
        with self.assertRaises(ValueError):
            G.sliced_rank_tokens([9] * 6, 4)

    def test_collapse_and_expand(self):
        states = [f"r{i}" for i in range(8)]
        self.assertEqual(G.collapse(states, 4), ["r0", "r4"])
        self.assertEqual(G.expand({0: "a", 1: "b"}, 4, 8), {r: "ab"[r // 4] for r in range(8)})
        with self.assertRaises(ValueError):
            G.collapse(states[:6], 4)

    def test_route_states_rekey_to_groups(self):
        @dataclass
        class St:
            rank: int
            num_active_requests: int = 0

        per_rank = [St(r, 10 * (r // 4)) for r in range(8)]
        routed = G.route_states(per_rank, 4)
        self.assertEqual([(s.rank, s.num_active_requests) for s in routed], [(0, 0), (1, 10)])
        self.assertEqual([s.rank for s in per_rank], list(range(8)))  # the gathered states are not mutated
        self.assertEqual(G.route_states(per_rank, 1), per_rank)

    def test_batch_signature_and_divergence(self):
        a = G.batch_signature([(1, 100, 7), (2, 50, 9)])
        self.assertEqual(a, G.batch_signature(iter([(1, 100, 7), (2, 50, 9)])))
        self.assertTrue(-(1 << 63) <= a < (1 << 63))
        self.assertNotEqual(a, G.batch_signature([(2, 50, 9), (1, 100, 7)]))  # order matters
        self.assertNotEqual(a, G.batch_signature([(1, 100, 7), (2, 50, 8)]))  # a different sampled token
        self.assertNotEqual(a, G.batch_signature([(1, 101, 7), (2, 50, 9)]))
        self.assertEqual(G.batch_signature([]), G.batch_signature(iter(())))
        self.assertEqual(G.diverged_groups([a] * 4 + [a, a, a + 1, a], 4), [1])
        self.assertEqual(G.diverged_groups([a] * 8, 4), [])
        self.assertTrue(G.check_enabled({}))
        self.assertFalse(G.check_enabled({CHECK_ENV: "0"}))

    def test_router_contract(self):
        for cls in ("DefaultADPRouter", "KVCacheAwareADPRouter", "ConversationAwareADPRouter"):
            self.assertIsNone(G.router_mismatch(cls))
        self.assertIn("SomeOtherRouter", G.router_mismatch("SomeOtherRouter"))

    def test_module_imports_no_torch(self):
        src = ADP_GROUPS.read_text()
        self.assertNotIn("import torch", src)
        self.assertNotIn("tensorrt_llm", re.sub(r'""".*?"""', "", src, flags=re.S).split("ENV =")[0])


class TestAdpGroupsW1(unittest.TestCase):
    """W1 generality: G = 1 / 2 / 4 / 8 groups of 4 (MSO H32 = G 8), every rank of every G."""

    def test_rank_maps(self):
        env = {ENV: str(T)}
        for g in GROUP_COUNTS:
            world = g * T
            leaders = []
            for r in range(world):
                m = mapping(world, r)
                with self.subTest(groups=g, rank=r):
                    self.assertEqual(G.group_attn_tp(m, env), T)
                    self.assertEqual(G.attn_tp_size(m, env), T)
                    self.assertEqual(G.dp_size(m, env), g)
                    self.assertEqual(G.dp_rank(m, env), r // T)
                    self.assertEqual(G.attn_tp_rank(m, env), r % T)
                    self.assertEqual(G.is_group_leader(m, env), r % T == 0)
                    self.assertIn(r, G.group_tp_ranks(r // T, T))
                    self.assertEqual(G.attention_shard(r, world, True, env), (r % T, T))
                if G.is_group_leader(m, env):
                    leaders.append(r)
            self.assertEqual(leaders, [k * T for k in range(g)])
            # the groups partition the world
            self.assertEqual(sorted(x for k in range(g) for x in G.group_tp_ranks(k, T)), list(range(world)))

    def test_attention_tp2(self):
        env = {ENV: "2"}
        for world in (8, 16):
            for r in range(world):
                m = mapping(world, r)
                self.assertEqual((G.dp_size(m, env), G.dp_rank(m, env), G.attn_tp_rank(m, env)),
                                 (world // 2, r // 2, r % 2))

    def test_route_states_and_expand_round_trip(self):
        class S:
            def __init__(self, rank):
                self.rank = rank

        for g in GROUP_COUNTS:
            world = g * T
            states = [S(r) for r in range(world)]
            routed = G.route_states(states, T)
            with self.subTest(groups=g):
                self.assertEqual([s.rank for s in routed], list(range(g)))
                self.assertEqual([s.rank for s in states], list(range(world)))
                per_group = {k: f"req{k}" for k in range(g)}
                self.assertEqual(G.expand(per_group, T, world), {r: f"req{r // T}" for r in range(world)})
                self.assertEqual(G.collapse(list(range(world)), T), [k * T for k in range(g)])

    def test_ep_dispatch_sizes(self):
        for g in GROUP_COUNTS:
            per_group_rows = [5 * k + 1 for k in range(g)]
            counts = [per_group_rows[r // T] for r in range(g * T)]
            got = G.sliced_rank_tokens(counts, T)
            with self.subTest(groups=g):
                self.assertEqual(got, [-(-per_group_rows[r // T] // T) for r in range(g * T)])

    def test_diverged_groups(self):
        for g in GROUP_COUNTS:
            sig = [7] * (g * T)
            self.assertEqual(G.diverged_groups(sig, T), [])
            sig[g * T - 1] = 8
            self.assertEqual(G.diverged_groups(sig, T), [g - 1])


def _load_router_module():
    """adp_router.py from source with its package imports stubbed (no torch)."""
    sched = ROOT / "tensorrt_llm/_torch/pyexecutor/scheduler"
    logger = types.SimpleNamespace(debug=lambda *a, **k: None, info=lambda *a, **k: None,
                                   warning=lambda *a, **k: None, info_once=lambda *a, **k: None,
                                   warning_once=lambda *a, **k: None)
    trt = types.ModuleType("tensorrt_llm")
    trt.__path__ = []
    tpkg = types.ModuleType("tensorrt_llm._torch")
    tpkg.__path__ = []
    tpkg.adp_groups = G
    stubs = {
        "tensorrt_llm": trt,
        "tensorrt_llm.logger": types.SimpleNamespace(logger=logger),
        "tensorrt_llm._torch": tpkg,
        "tensorrt_llm._torch.adp_groups": G,
        "tensorrt_llm._torch.pyexecutor": types.ModuleType("tensorrt_llm._torch.pyexecutor"),
        "tensorrt_llm._torch.pyexecutor.scheduler": types.ModuleType("tensorrt_llm._torch.pyexecutor.scheduler"),
        "tensorrt_llm._torch.pyexecutor.llm_request": types.SimpleNamespace(LlmRequestState=object),
    }
    for pkg in ("tensorrt_llm._torch.pyexecutor", "tensorrt_llm._torch.pyexecutor.scheduler"):
        stubs[pkg].__path__ = []
    with patch.dict(sys.modules, stubs):
        full = "tensorrt_llm._torch.pyexecutor.scheduler.adp_router"
        spec = importlib.util.spec_from_file_location(full, sched / "adp_router.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
    return mod


class TestDefaultRouterOverGroups(unittest.TestCase):
    def setUp(self):
        try:
            self.R = _load_router_module()
        except (ImportError, TypeError, SyntaxError) as err:  # the router grew a heavier import
            self.skipTest(f"adp_router not loadable without torch: {err}")

    def _items(self, n):
        return [
            types.SimpleNamespace(id=i, child_req_ids=None,
                                  request=types.SimpleNamespace(num_input_tokens=100 + i, py_scheduling_params=None))
            for i in range(n)
        ]

    def test_routes_to_every_group(self):
        """The real DefaultADPRouter over the executor's group states: every request lands on a group index the
        executor reads (a plain collapse keeps ranks 0/4 as keys and raises KeyError: 1)."""
        router = self.R.DefaultADPRouter(dist=None)
        per_rank = [self.R.RankState(rank=r) for r in range(8)]
        assigned, expected = router.route_requests(G.route_states(per_rank, 4), self._items(6), 64)
        self.assertEqual(sorted(assigned), [0, 1])
        self.assertEqual(sorted(len(v) for v in assigned.values()), [3, 3])
        self.assertEqual(expected, 3)
        with self.assertRaises(KeyError):
            router.route_requests(G.collapse(per_rank, 4), self._items(6), 64)

    def test_every_group_count(self):
        router = self.R.DefaultADPRouter(dist=None)
        for g in GROUP_COUNTS:
            per_rank = [self.R.RankState(rank=r) for r in range(g * T)]
            assigned, _ = router.route_requests(G.route_states(per_rank, T), self._items(4 * g), 64)
            with self.subTest(groups=g):
                self.assertEqual(sorted(assigned), list(range(g)))
                self.assertEqual([len(assigned[k]) for k in range(g)], [4] * g)

    def test_stock_states_unchanged(self):
        router = self.R.DefaultADPRouter(dist=None)
        per_rank = [self.R.RankState(rank=r) for r in range(8)]
        stock, _ = router.route_requests(per_rank, self._items(8), 64)
        via, _ = router.route_requests(G.route_states(per_rank, 1), self._items(8), 64)
        self.assertEqual({k: [i.id for i in v] for k, v in stock.items()},
                         {k: [i.id for i in v] for k, v in via.items()})


class TestBuiltInSitesSource(unittest.TestCase):
    """Every site that hard-coded 'attention DP => attention TP 1 / one DP rank per tp rank' asks adp_groups."""

    SITES = {
        "tensorrt_llm/_torch/attention/attention.py": [
            "adp_groups.group_attn_tp(self.mapping)", "enable_attention_dp=sub_mapping_adp"],
        "tensorrt_llm/_torch/model_config.py": ["adp_groups.attn_tp_size(self.mapping)"],
        "tensorrt_llm/_torch/pyexecutor/_util.py": [
            "adp_groups.dp_size(self._mapping)", "adp_groups.attn_tp_size(self._mapping)"],
        "tensorrt_llm/_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py": [
            "adp_groups.attn_tp_size(mapping)", "tp_size = kv_head_tp_size(mapping)"],
        "tensorrt_llm/_torch/pyexecutor/resource_manager.py": ["adp_groups.attn_tp_size(mapping)"],
        "tensorrt_llm/_torch/pyexecutor/scheduler/adp_router.py": ["adp_groups.collapse(list(all_data)"],
        "tensorrt_llm/_torch/pyexecutor/connectors/mooncake_store/worker.py": ["adp_groups.attention_shard("],
        "tensorrt_llm/_torch/pyexecutor/py_executor.py": [
            "adp_groups.router_mismatch", "adp_groups.route_states(all_rank_states", "adp_groups.diverged_groups",
            "responses = []", "self._adp_dp_rank if self._adp_group_tp > 1 else self.dist.tp_rank",
            "self.dist.tp_size // getattr(self, \"_adp_group_tp\", 1)"],
    }

    def test_sites(self):
        for rel, needles in self.SITES.items():
            src = (ROOT / rel).read_text()
            for needle in needles:
                with self.subTest(rel=rel, needle=needle):
                    self.assertIn(needle, src)

    def test_stock_rules_gone(self):
        """No KV-head / attention-module site still hard-codes 'ADP => tp 1' (the rule moved into adp_groups)."""
        stale = {
            "tensorrt_llm/_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py":
                "tp_size = 1 if mapping.enable_attention_dp else mapping.tp_size",
            "tensorrt_llm/_torch/pyexecutor/resource_manager.py":
                "tp_size = 1 if mapping.enable_attention_dp else mapping.tp_size",
            "tensorrt_llm/_torch/pyexecutor/_util.py":
                "tp_size = 1 if self._mapping.enable_attention_dp else self._mapping.tp_size",
            "tensorrt_llm/_torch/model_config.py":
                "attn_tp_size = self.mapping.attn_tp_size if not self.mapping.enable_attention_dp else 1",
        }
        for rel, needle in stale.items():
            with self.subTest(rel=rel):
                self.assertNotIn(needle, (ROOT / rel).read_text())

    def test_no_m31_env_in_product_code(self):
        """Only adp_groups knows the legacy M3.1 names (aliases)."""
        for rel in self.SITES:
            src = (ROOT / rel).read_text()
            with self.subTest(rel=rel):
                self.assertNotIn("TRTLLM_M31_ADPTP", src)


if __name__ == "__main__":
    unittest.main()
