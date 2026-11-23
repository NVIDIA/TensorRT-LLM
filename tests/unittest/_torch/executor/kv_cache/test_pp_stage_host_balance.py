# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Per-stage host tier sizing math of ``pp_stage_host_balance``, on the host only.

The module imports nothing from the package; it is executed from its file so
the tests run without the compiled bindings.
"""

import importlib.util
import math
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[5]
_PATH = _ROOT / "tensorrt_llm/_torch/pyexecutor/kv_cache/pp_stage_host_balance.py"

GiB = 1 << 30
MAX_SEQ_LEN = 996_672
TOKENS_PER_BLOCK = 32
WINDOW = 2048
# Bytes per token of one FP8 target layer (16 KV heads x 64) and one BF16 draft layer
# (8 KV heads x 128), K and V.
TARGET_LAYER = 2 * 16 * 64
DRAFT_LAYER = 2 * 8 * 128 * 2
POOL_RATIO = [0.15, 0.85]
PARTITION = [15, 16, 16, 15]
DRAFT_LAYERS = 5


@pytest.fixture(scope="module")
def balance():
    spec = importlib.util.spec_from_file_location("pp_stage_host_balance_under_test", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stage_layers(rank):
    """Bytes per token, window and sinks of one stage's layers, in layer order."""
    start = sum(PARTITION[:rank])
    layer_bytes, windows = [], []
    for layer_id in range(start, start + PARTITION[rank]):
        layer_bytes.append(TARGET_LAYER)
        windows.append(MAX_SEQ_LEN if layer_id % 4 == 1 else WINDOW)
    if rank == len(PARTITION) - 1:
        layer_bytes += [DRAFT_LAYER] * DRAFT_LAYERS
        windows += [WINDOW] * DRAFT_LAYERS
    return layer_bytes, windows, [None] * len(layer_bytes)


def _draft_stage_ratio():
    """The last stage's pool_ratio after the draft-inclusive rescale."""
    sliding_target = 11 * TARGET_LAYER
    growth = (sliding_target + DRAFT_LAYERS * DRAFT_LAYER) / sliding_target
    weights = [POOL_RATIO[0] * growth, POOL_RATIO[1]]
    return [weight / sum(weights) for weight in weights]


def _costs(balance):
    costs = []
    for rank in range(4):
        ratio = _draft_stage_ratio() if rank == 3 else POOL_RATIO
        cost, _ = balance.full_attention_token_cost(
            ratio, *_stage_layers(rank), MAX_SEQ_LEN, TOKENS_PER_BLOCK
        )
        costs.append(cost)
    return costs


def _shapes(balance, *, host, other):
    return [
        balance.StageTierShape("node", cost, host, extra)
        for cost, extra in zip(_costs(balance), other)
    ]


def _tokens(shape, host):
    return (host + shape.other_bytes) / shape.full_token_cost


def test_full_attention_cost_of_the_released_stages(balance):
    """Four full-attention layers per stage: 8,192 B per token over the full share."""
    for rank in range(3):
        cost, share = balance.full_attention_token_cost(
            POOL_RATIO, *_stage_layers(rank), MAX_SEQ_LEN, TOKENS_PER_BLOCK
        )
        assert share == POOL_RATIO[1]
        assert cost == pytest.approx(4 * TARGET_LAYER / POOL_RATIO[1])
    ratio = _draft_stage_ratio()
    cost, share = balance.full_attention_token_cost(
        ratio, *_stage_layers(3), MAX_SEQ_LEN, TOKENS_PER_BLOCK
    )
    assert share == pytest.approx(ratio[1])
    assert share == pytest.approx(0.748, abs=5e-4)
    assert cost == pytest.approx(4 * TARGET_LAYER / ratio[1])


def test_full_attention_cost_follows_the_life_cycle_order(balance):
    """A layout that starts with a full-attention layer lists the full share first."""
    layer_bytes = [100, 10, 100, 10]
    windows = [None, WINDOW, 0, WINDOW]
    cost, share = balance.full_attention_token_cost(
        [0.8, 0.2], layer_bytes, windows, [None] * 4, MAX_SEQ_LEN, TOKENS_PER_BLOCK
    )
    assert share == 0.8
    assert cost == pytest.approx(200 / 0.8)


def test_full_attention_cost_declines_unmatched_layouts(balance):
    layers = _stage_layers(0)
    args = (MAX_SEQ_LEN, TOKENS_PER_BLOCK)
    assert balance.full_attention_token_cost(None, *layers, *args) is None
    assert balance.full_attention_token_cost([1.0], *layers, *args) is None
    # All sliding-window: no full-attention life cycle.
    assert (
        balance.full_attention_token_cost([1.0], [10, 10], [WINDOW, WINDOW], [None, None], *args)
        is None
    )
    # Two full-attention life cycles (sinks split them).
    assert (
        balance.full_attention_token_cost([0.5, 0.5], [10, 10], [None, None], [None, 64], *args)
        is None
    )
    assert balance.full_attention_token_cost([1.0], [10], [None, None], [None], *args) is None


def test_full_attention_cost_declines_a_page_size_shared_with_another_life_cycle(balance):
    """Pages of one size share a pool below the device, so the share does not bound them."""
    args = (MAX_SEQ_LEN, TOKENS_PER_BLOCK)
    # One full-attention layer and two sliding-window layers of half its bytes: same page size.
    assert (
        balance.full_attention_token_cost(
            [0.5, 0.5], [100, 50, 50], [None, WINDOW, WINDOW], [None] * 3, *args
        )
        is None
    )
    # One byte apart: separate pools, the share applies.
    cost, share = balance.full_attention_token_cost(
        [0.5, 0.5], [100, 50, 51], [None, WINDOW, WINDOW], [None] * 3, *args
    )
    assert (cost, share) == (pytest.approx(200.0), 0.5)


def test_host_only_split_gives_every_stage_the_same_host_full_tokens(balance):
    """Device and remote left out: 279 GiB per rank becomes about 269.8 / 306.6 GiB."""
    host = 279 * GiB
    shapes = _shapes(balance, host=host, other=[0] * 4)
    plan, reason = balance.plan_stage_host_tiers(shapes)
    assert reason == ""
    sizes = [size / GiB for size in plan.host_bytes]
    assert sizes[:3] == pytest.approx([269.8] * 3, abs=0.05)
    assert sizes[3] == pytest.approx(306.6, abs=0.05)
    assert 4 * host - 4 * balance.HOST_TIER_GRANULARITY <= sum(plan.host_bytes) <= 4 * host
    tokens = [_tokens(shape, size) for shape, size in zip(shapes, plan.host_bytes)]
    granule_tokens = balance.HOST_TIER_GRANULARITY / min(s.full_token_cost for s in shapes)
    assert max(tokens) - min(tokens) <= granule_tokens


def test_all_tiers_split_raises_the_binding_stage_and_keeps_the_total(balance):
    """Device quotas and a 180 GiB remote level per rank, as a single-lender context server."""
    host = 279 * GiB
    device = [119.03 * GiB, 119.32 * GiB, 119.32 * GiB, 121.91 * GiB]
    other = [int(d) + 180 * GiB for d in device]
    shapes = _shapes(balance, host=host, other=other)
    plan, reason = balance.plan_stage_host_tiers(shapes)
    assert reason == ""
    assert all(size % balance.HOST_TIER_GRANULARITY == 0 for size in plan.host_bytes)
    assert 4 * host - 4 * balance.HOST_TIER_GRANULARITY <= sum(plan.host_bytes) <= 4 * host
    before = [_tokens(shape, host) for shape in shapes]
    after = [_tokens(shape, size) for shape, size in zip(shapes, plan.host_bytes)]
    granule_tokens = balance.HOST_TIER_GRANULARITY / min(s.full_token_cost for s in shapes)
    assert max(after) - min(after) <= granule_tokens
    assert plan.full_tokens == pytest.approx(min(after), abs=granule_tokens)
    # The last stage was the binding one; the common count lies between it and the others.
    assert min(before) == before[3]
    assert before[3] < plan.full_tokens < min(before[:3])
    assert plan.host_bytes[3] > host > max(plan.host_bytes[:3])


def test_equal_stages_keep_the_configured_host_tier(balance):
    """The same cost and tiers on every rank give back the configured size exactly."""
    shape = balance.StageTierShape("node", 8192 / 0.85, 279 * GiB, 300 * GiB)
    plan, reason = balance.plan_stage_host_tiers([shape] * 4)
    assert reason == ""
    assert plan.host_bytes == (279 * GiB,) * 4


def test_plan_is_declined_when_a_rank_cannot_take_part(balance):
    shapes = _shapes(balance, host=279 * GiB, other=[0] * 4)
    plan, reason = balance.plan_stage_host_tiers(shapes[:3] + [None])
    assert plan is None and "ranks [3]" in reason
    assert balance.plan_stage_host_tiers([]) == (None, "no ranks")


def test_plan_is_declined_across_hosts(balance):
    shapes = _shapes(balance, host=279 * GiB, other=[0] * 4)
    shapes[2] = balance.StageTierShape(
        "other", *[getattr(shapes[2], f) for f in ("full_token_cost", "host_bytes", "other_bytes")]
    )
    plan, reason = balance.plan_stage_host_tiers(shapes)
    assert plan is None and "2 hosts" in reason


def test_plan_is_declined_when_a_host_tier_would_vanish(balance):
    """A rank whose other tiers already exceed its share would need a negative host tier."""
    cost = 8192 / 0.85
    shapes = [
        balance.StageTierShape("node", cost, 1 * GiB, 0),
        balance.StageTierShape("node", cost, 1 * GiB, 100 * GiB),
    ]
    plan, reason = balance.plan_stage_host_tiers(shapes)
    assert plan is None and reason.startswith("rank 1 would get 0.00 GiB")


def test_plan_depends_only_on_the_gathered_shapes(balance):
    shapes = _shapes(balance, host=279 * GiB, other=[300 * GiB] * 4)
    assert balance.plan_stage_host_tiers(shapes) == balance.plan_stage_host_tiers(list(shapes))


def test_describe_plan_has_one_line_per_rank(balance):
    shapes = _shapes(balance, host=279 * GiB, other=[0] * 4)
    plan, _ = balance.plan_stage_host_tiers(shapes)
    lines = balance.describe_plan(shapes, plan)
    assert len(lines) == 4
    assert lines[3].startswith("rank 3: host tier 279.00 -> ")
    balanced = float(lines[3].split(" -> ")[1].split(" GiB")[0])
    assert balanced == pytest.approx(306.6, abs=0.05)
    assert all(math.isfinite(_tokens(s, h)) for s, h in zip(shapes, plan.host_bytes))
