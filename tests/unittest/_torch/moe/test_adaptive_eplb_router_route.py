# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contract of the adaptive route the MegaMoE dispatch router applies (adaptive_eplb_router_reference.py).

The module is loaded by path (it is plain Python); its tables come from the production policy.build_route_tables.
"""

import importlib.util
import pathlib
import random
import unittest

import pytest

from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.policy import build_route_tables

pytestmark = pytest.mark.cpu_only

_SPEC = importlib.util.spec_from_file_location(
    "router_route", pathlib.Path(__file__).resolve().parent / "adaptive_eplb_router_reference.py"
)
rr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rr)

Q32 = 1 << 32
MAX = Q32 - 1


def build_tables(expert_slots, shares, num_experts, num_slots):
    """policy.build_route_tables over the layout ``expert_slots`` give."""
    layout = [-1] * num_slots
    for expert, copies in enumerate(expert_slots):
        for slot in copies:
            layout[slot] = expert
    return build_route_tables(layout, shares, num_experts, num_slots)


def random_layout(rng, num_experts=24, ep=4, local=8, zero_one=0.3):
    """A layout with every expert at home and replicas on free slots; shares with 0/1 copies."""
    per = num_experts // ep
    num_slots = ep * local
    expert_slots = [[(e // per) * local + e % per] for e in range(num_experts)]
    free = [s for s in range(num_slots) if s % local >= per]
    rng.shuffle(free)
    for s in free[: len(free) // 2]:
        expert_slots[rng.randrange(num_experts)].append(s)
    shares = [0.0] * num_slots
    for copies in expert_slots:
        if len(copies) > 1 and rng.random() < zero_one:
            hot = rng.randrange(len(copies))
            for k, s in enumerate(copies):
                shares[s] = 1.0 if k == hot else 0.0
        else:
            for s in copies:
                shares[s] = rng.random()
    return expert_slots, shares, num_experts, num_slots


class RouterRouteTest(unittest.TestCase):
    def test_copy_shares_are_the_q32_shares_and_honor_zero_and_one(self):
        rng = random.Random(7)
        for _ in range(200):
            expert_slots, shares, e, s = random_layout(rng)
            counts, starts, slots, cumulative = build_tables(expert_slots, shares, e, s)
            for x in range(e):
                c, st = counts[x], starts[x]
                cum = cumulative[st : st + c]
                widths = rr.copy_shares(c, cum)
                self.assertEqual(sum(widths), Q32)
                weights = [shares[slots[st + k]] for k in range(c)]
                for k in range(c):
                    # Every point of a copy's range goes to it, the neighbours' to them.
                    lo = sum(widths[:k])
                    if widths[k]:
                        self.assertEqual(rr.copy_of(lo, c, cum), k)
                        self.assertEqual(rr.copy_of(lo + widths[k] - 1, c, cum), k)
                    if weights[k] == 0.0 and sum(weights) > 0:
                        self.assertEqual(widths[k], 0, "a share of 0 takes no point")
                    if weights[k] == 1.0 and sum(weights) == 1.0:
                        self.assertEqual(widths[k], Q32, "a share of 1 takes every point")
                        for p in (0, MAX, rng.randrange(Q32)):
                            self.assertEqual(rr.copy_of(p, c, cum), k)

    def test_realized_split_error_is_bounded(self):
        """(c): rows of each copy against its share over random tokens of one rank's bucket."""
        rng = random.Random(5)
        for tokens in (8, 32, 64, 128, 256):
            errors = []
            for _ in range(300):
                copies = rng.randrange(2, 5)
                weights = [rng.random() for _ in range(copies)]
                if rng.random() < 0.3:
                    weights = [1.0 if k == 0 else 0.0 for k in range(copies)]
                expert_slots = [list(range(copies))]
                shares = weights + [0.0] * 4
                cum = build_tables(expert_slots, shares, 1, copies + 4)[3][:copies]
                picks = sorted(rng.sample(range(tokens), rng.randrange(1, tokens + 1)))
                ep_rank = rng.randrange(32)
                got = [0] * copies
                for t in picks:
                    got[rr.copy_of(rr.point(0, t, ep_rank), copies, cum)] += 1
                total = sum(weights)
                for k in range(copies):
                    errors.append(abs(got[k] - len(picks) * weights[k] / total))
                    if weights[k] == 0.0:
                        self.assertEqual(got[k], 0)
            # A copy is off its share by at most a few rows at every bucket size.
            self.assertLessEqual(max(errors), max(4.0, 0.2 * tokens), tokens)


if __name__ == "__main__":
    unittest.main()
