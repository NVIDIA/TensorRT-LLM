# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source check of the cross-proxy fences in the Kimi K3 attention-residual kernels (PTX ISA, memory consistency
model, proxies).

The consumers read the V (and delta) slots with generic-proxy shared loads and release each slot with an mbarrier
arrive on ``bar_consumed``. The producer then refills the slot with ``cp.async.bulk``, an async-proxy write. Accesses
to one location through two proxies need a cross-proxy fence: every reading thread issues
``fence.proxy.async.shared::cta`` after its last read and before the release. Without it the order rests on how the
compiler schedules the instructions. No test of values can see that, so the kernels' sources are read.

  pytest test_attn_res_proxy_fences.py
"""

import pathlib
import re

import pytest

pytestmark = pytest.mark.cpu_only

_SRC = pathlib.Path(__file__).resolve().parents[5] / "cpp/tensorrt_llm/kernels/kimiK3AttnRes"

# The kernels whose consumers release cp.async.bulk-filled slots, with the number of releases each file has.
RELEASES = {"attnResFwd.cu": 3, "attnResFwdPersistentFused.cu": 1}

RELEASE = re.compile(r"\b(cute::arrive_barrier|mbarrier_arrive)\(\s*plan\.bar_consumed\b")
FENCE = 'asm volatile("fence.proxy.async.shared::cta;" ::: "memory");'
# What may sit between the fence and the release: the warp sync that gathers every lane's reads before lane 0 releases,
# and the lane-0 guard. Any other statement could hold a read after the fence.
BETWEEN = re.compile(r"^(__syncwarp\(\);|if \(lane == 0\)|\{)$")


def _code(line):
    return line.split("//", 1)[0].strip()


def violations(lines):
    """(line number, release) of every ``bar_consumed`` release whose closest preceding statement, past the warp sync
    and the lane-0 guard, is not the cross-proxy fence."""
    out = []
    for i, line in enumerate(lines):
        if not RELEASE.search(_code(line)):
            continue
        for j in range(i - 1, -1, -1):
            code = _code(lines[j])
            if not code or BETWEEN.match(code):
                continue
            if code != FENCE:
                out.append((i + 1, line.strip()))
            break
        else:
            out.append((i + 1, line.strip()))
    return out


@pytest.mark.parametrize("name", sorted(RELEASES))
def test_slot_release_follows_cross_proxy_fence(name):
    lines = (_SRC / name).read_text().splitlines()
    releases = [i for i, line in enumerate(lines) if RELEASE.search(_code(line))]
    assert len(releases) == RELEASES[name], (
        f"{name}: {len(releases)} bar_consumed releases, expected {RELEASES[name]} "
        "(update RELEASES if the kernels changed)"
    )
    assert any("cp.async.bulk.shared::cta.global" in line for line in lines), (
        f"{name}: no cp.async.bulk refill; the fence may no longer be needed"
    )
    assert violations(lines) == [], (
        f"{name}: slot releases without fence.proxy.async.shared::cta right before them: {violations(lines)}"
    )


def test_check_flags_a_release_without_the_fence():
    """The check itself: the release pattern without the fence, as before the fix, is reported."""
    lines = [
        "// Every lane's reads of the chunk's slots before lane 0 releases them.",
        "__syncwarp();",
        "if (lane == 0)",
        "{",
        "    cute::arrive_barrier(plan.bar_consumed[chunk_slot]);",
        "}",
    ]
    assert violations(lines) == [(5, "cute::arrive_barrier(plan.bar_consumed[chunk_slot]);")]
    assert violations([FENCE] + lines) == []
    assert violations([FENCE, "float x = buf[0];"] + lines) == [
        (7, "cute::arrive_barrier(plan.bar_consumed[chunk_slot]);")
    ]
