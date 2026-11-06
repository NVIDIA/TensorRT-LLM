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
"""Per-request split-KV plan of the standalone MLA DSpark drafter's block decode.

``TLLM_DSPARK_MLA_SPLIT_PLAN=1`` launches the target MLA decode's planner once
per drafter step and hands the per-request split counts to the cute-dsl block
decode. These tests need the CuTe DSL (for the decode runner's geometry), not a
GPU: the planner kernel is replaced by its host mirror, and the decode ops by
recorders. Planner kernel against mirror and the decode kernel's output with a
plan are GPU tests (test_cute_dsl_mla_split_plan.py and
test_dspark_fp8_draft_cache.py).
"""

import types
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention_backend.fmha import cute_dsl_mla
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.models import modeling_dspark
from tensorrt_llm._torch.models.modeling_dspark import MLADSparkForCausalLM

pytestmark = pytest.mark.skipif(
    not IS_CUTLASS_DSL_AVAILABLE, reason="the decode runner needs the CuTe DSL"
)

# Stand-in for the decode runner's occupancy query; the plans expected below follow this wave.
MAX_ACTIVE_BLOCKS = 212
WAVE_CLUSTERS = MAX_ACTIVE_BLOCKS // 2


@pytest.fixture
def wave(monkeypatch):
    """The decode runner's occupancy, which needs a device to query."""
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    monkeypatch.setattr(
        CuteDSLNVMlaDecodeBlackwellRunner,
        "_get_max_active_blocks",
        classmethod(lambda cls: MAX_ACTIVE_BLOCKS),
    )


@pytest.fixture
def launches(monkeypatch, wave):
    """Record the plan launches and compute each plan on the host."""
    recorded = []

    def launch(cache_seqs, plan, *, num_requests, rows_per_request, wave_clusters, **kwargs):
        recorded.append(
            dict(
                num_requests=num_requests,
                rows_per_request=rows_per_request,
                wave_clusters=wave_clusters,
                **kwargs,
            )
        )
        plan[:num_requests] = torch.tensor(
            cute_dsl_mla.plan_mla_split_kv(
                cache_seqs[:num_requests].tolist(),
                rows_per_request=rows_per_request,
                wave_clusters=wave_clusters,
                **kwargs,
            ),
            dtype=torch.int32,
        )

    monkeypatch.setattr(cute_dsl_mla, "launch_mla_split_kv_plan", launch)
    return recorded


def _drafter(heads=8, enabled=True):
    """What ``_mla_split_kv_plan`` and ``_cute_dsl_block_decode`` read."""
    stub = SimpleNamespace(
        _num_heads=heads,
        _mla_split_plan=None,
        _mla_split_plan_enabled=enabled,
        _mla_max_split_kv=0,
        kv_lora_rank=512,
        softmax_scale=0.1,
    )
    for name in ("_mla_split_kv_plan", "_cute_dsl_block_decode"):
        setattr(stub, name, types.MethodType(getattr(MLADSparkForCausalLM, name), stub))
    return stub


def _step(ctx_lens, block_size=5):
    """A block fixup of these context lengths: its lengths, tables and bounds only."""
    seq_lens = torch.tensor([n + block_size for n in ctx_lens], dtype=torch.int32)
    return SimpleNamespace(
        seq_lens_i32=seq_lens,
        page_tables_i32=torch.zeros(len(ctx_lens), 4, dtype=torch.int32),
        kv_bounds_i32=seq_lens.repeat_interleave(block_size),
    )


def test_split_plan_is_opt_in(monkeypatch):
    env = modeling_dspark._DSPARK_MLA_SPLIT_PLAN_ENV
    monkeypatch.delenv(env, raising=False)
    assert not modeling_dspark._dspark_mla_split_plan_enabled()
    monkeypatch.setenv(env, "0")
    assert not modeling_dspark._dspark_mla_split_plan_enabled()
    monkeypatch.setenv(env, "1")
    assert modeling_dspark._dspark_mla_split_plan_enabled()


@pytest.mark.parametrize(
    "heads,block_size,rows",
    [
        # TP8: a block of any length is one M tile per request.
        (8, 5, 1),
        (8, 6, 1),
        (8, 7, 1),
        # All heads on a rank (attention DP): the integer fold of block tokens into the M tile.
        (64, 4, 2),
        (64, 5, 5),
        (64, 6, 3),
    ],
)
def test_split_plan_geometry_is_the_decode_launch(wave, heads, block_size, rows):
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    geometry = CuteDSLNVMlaDecodeBlackwellRunner.get_split_plan_geometry(heads, block_size)
    assert geometry == (rows, WAVE_CLUSTERS)
    # The target decode's planner sizes its launch the same way.
    assert geometry == (
        CuteDSLNVMlaDecodeBlackwellRunner.get_folded_seq_len_q(heads, block_size),
        WAVE_CLUSTERS,
    )


@pytest.mark.parametrize(
    "batch,expected",
    [
        # One wave takes every request's 32-way split.
        (3, 32),
        # Past it, fewer splits per request keep the batch in one wave.
        (4, 26),
        (5, 21),
        (8, 13),
        (16, 6),
    ],
)
def test_split_plan_at_the_tp8_drafter_shape(launches, batch, expected):
    drafter = _drafter(heads=8)
    plan = drafter._mla_split_kv_plan(_step([137000] * batch), 5, fp8=False)
    assert plan.dtype == torch.int32 and plan.tolist() == [expected] * batch
    assert launches == [
        dict(
            num_requests=batch,
            rows_per_request=1,
            wave_clusters=WAVE_CLUSTERS,
            piece_cost=modeling_dspark._MLA_SPLIT_PLAN_PIECE_COST_TILES,
        )
    ]


def test_split_plan_follows_each_requests_length(launches):
    """The longest request sets the piece size; the others take as many pieces as they hold."""
    plan = _drafter()._mla_split_kv_plan(_step([3000, 17000, 64000, 137000, 90000]), 5, fp8=False)
    assert plan.tolist() == [1, 4, 15, 32, 21]


def test_split_plan_is_launched_once_per_step_and_pool_dtype(launches):
    drafter = _drafter()
    step = _step([137000] * 5)
    first = drafter._mla_split_kv_plan(step, 5, fp8=False)
    # The other layers of the step reuse it.
    assert drafter._mla_split_kv_plan(step, 5, fp8=False) is first
    assert len(launches) == 1
    # A new step plans again.
    next_step = _step([137000] * 5)
    drafter._mla_split_kv_plan(next_step, 5, fp8=False)
    assert len(launches) == 2
    # An FP8 pool plans at its own piece cost.
    drafter._mla_split_kv_plan(next_step, 5, fp8=True)
    assert [launch["piece_cost"] for launch in launches] == [
        modeling_dspark._MLA_SPLIT_PLAN_PIECE_COST_TILES,
        modeling_dspark._MLA_SPLIT_PLAN_PIECE_COST_TILES,
        modeling_dspark._MLA_SPLIT_PLAN_PIECE_COST_TILES_FP8,
    ]
    assert (
        modeling_dspark._MLA_SPLIT_PLAN_PIECE_COST_TILES_FP8
        > modeling_dspark._MLA_SPLIT_PLAN_PIECE_COST_TILES
    )


def test_split_plan_leaves_a_batch_beyond_the_planner_to_the_uniform_split(launches):
    drafter = _drafter()
    step = _step([1000] * (cute_dsl_mla._SPLIT_PLAN_MAX_REQUESTS + 1))
    assert drafter._mla_split_kv_plan(step, 5, fp8=False) is None
    assert drafter._mla_split_kv_plan(step, 5, fp8=False) is None
    assert launches == []


@pytest.fixture
def decode_ops(monkeypatch):
    """Record what the block decode hands to each cute-dsl decode op."""
    calls = []

    def op(name):
        def record(*args, **kwargs):
            calls.append((name, args, kwargs))

        return record

    ops = SimpleNamespace(
        cute_dsl_mla_decode_fp16_blackwell=op("bf16"),
        cute_dsl_mla_decode_fp8_blackwell=op("fp8"),
    )
    monkeypatch.setattr(torch.ops, "trtllm", ops)
    return calls


@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
@pytest.mark.parametrize("enabled", [False, True], ids=["off", "on"])
def test_block_decode_hands_the_plan_to_the_kernel_after_kv_bounds(
    launches, decode_ops, enabled, fp8
):
    block_size, page_size, heads = 5, 64, 8
    drafter = _drafter(heads=heads, enabled=enabled)
    drafter._assert_cute_dsl_can_implement = lambda *args: None
    drafter._mla_decode_workspace = lambda device, numel: torch.empty(numel, dtype=torch.uint8)
    drafter._mla_fp8_decode_scales = lambda device: (torch.ones(1), torch.ones(1))
    step = _step([137000] * 5, block_size)
    query = torch.zeros(5, block_size, heads, 576, dtype=torch.bfloat16)
    pool = torch.zeros(
        4, 1, 1, page_size, 576, dtype=torch.float8_e4m3fn if fp8 else torch.bfloat16
    )

    out = drafter._cute_dsl_block_decode(query, pool, step, block_size)

    assert out.shape == (5, block_size, heads, 512) and out.dtype == query.dtype
    ((name, args, kwargs),) = decode_ops
    assert name == ("fp8" if fp8 else "bf16")
    # ..., max_batch_size, softmax_stats, kv_bounds, block_split_kvs
    assert args[-4] == 5 and args[-3] is None and args[-2] is step.kv_bounds_i32
    if enabled:
        assert args[-1] is drafter._mla_split_plan[2]
        assert args[-1].tolist() == [21] * 5
    else:
        assert args[-1] is None and drafter._mla_split_plan is None and launches == []
    # The BF16 decode keeps its split cap; the FP8 decode has none.
    assert kwargs == ({} if fp8 else {"max_split_kv": 0})
