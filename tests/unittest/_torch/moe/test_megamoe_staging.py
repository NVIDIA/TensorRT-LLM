# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the MegaMoE CuteDSL backend's staging of the next run."""

import types
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.mega_moe import mega_moe_cute_dsl as mm

pytestmark = pytest.mark.cpu_only

IMPL = mm.TrtllmCutedslMegaMoeNvfp4Impl


def test_the_staging_of_the_next_run_is_handed_out_only_for_a_run_that_stages():
    """``_staging_of_next_run`` (where ``quantize_input`` writes) hands out a bucket's staging only
    for a forward that ``run_moe`` would stage."""
    t = torch.zeros(4, 2)
    idx = torch.full((4, 2), -1, dtype=torch.int64)
    bufs = mm._MegaMoeBuffers(
        activation=t,
        activation_sf=t,
        topk_weights=t,
        combine_output=t,
        shared_workspace=t,
        peer_offsets=(0,),
        topk_idx_local=idx,
    )
    acquired = []
    backend = SimpleNamespace(
        _active_launch_max_tokens=None,
        tactic_autotune=False,
        _weights_created=True,
        ep_size=4,
        _genphase_direct_input_buckets={4},
        dtype=torch.bfloat16,
        hidden_size=4,
        routing_method=SimpleNamespace(experts_per_token=2),
        _local_staging_cache={},
        _select_launch_max_tokens=lambda hint: 4 if hint <= 4 else 8,
        _acquire_buffers=lambda **kw: acquired.append(kw) or bufs,
    )
    backend._may_read_inputs_in_place = types.MethodType(IMPL._may_read_inputs_in_place, backend)
    cpu = torch.device("cpu")
    staging = types.MethodType(IMPL._staging_of_next_run, backend)
    assert staging(2, cpu) is None  # no bucket hint yet
    backend._active_launch_max_tokens = 3
    assert staging(2, cpu) is None  # the staging is not built
    backend._local_staging_cache[mm._StagingKey(4, 2, 4, "cpu", torch.bfloat16)] = {}
    assert staging(2, cpu) is bufs
    assert acquired[-1]["launch_max_T"] == 4 and acquired[-1]["device"] == cpu
    assert staging(2, torch.device("meta")) is None  # a staging set of another device
    assert staging(4, cpu) is None  # an exact-size direct-input bucket
    assert staging(5, cpu) is None  # more rows than the hinted bucket holds
    backend._active_launch_max_tokens = 5
    assert staging(5, cpu) is None  # bucket 8 is selected but not built


def test_run_moe_launches_from_the_staging_only_the_rows_quantize_input_put_there():
    """``run_moe`` takes the staging's rows as already quantized only when ``quantize_input``
    recorded that it quantized exactly those rows into it; any other pairing raises instead of
    copying the rows again or launching stale ones."""
    act = torch.zeros(8, 4, dtype=torch.uint8)
    act_sf = torch.zeros(8, 1, dtype=torch.uint8)
    t = torch.zeros(8, 2)
    bufs = mm._MegaMoeBuffers(
        activation=act,
        activation_sf=act_sf,
        topk_weights=t,
        combine_output=t,
        shared_workspace=t,
        peer_offsets=(0,),
        topk_idx_local=torch.full((8, 2), -1, dtype=torch.int64),
    )
    stage_activation = []
    backend = SimpleNamespace(
        _active_launch_max_tokens=None,
        _quantized_into=None,
        ep_size=1,
        ep_rank=0,
        tactic_autotune=False,
        _genphase_direct_input_buckets=set(),
        _padding_drop_buckets=set(),
        _select_launch_max_tokens=lambda hint: 8,
        _acquire_buffers=lambda **kw: bufs,
        _stage_inputs=lambda **kw: stage_activation.append(kw["stage_activation"]),
        _launch_megamoe_kernel=lambda **kw: "out",
    )
    backend._may_read_inputs_in_place = types.MethodType(IMPL._may_read_inputs_in_place, backend)
    routes = torch.zeros(3, 2, dtype=torch.int64)

    def run(x: torch.Tensor, x_sf: torch.Tensor) -> str:
        return IMPL._run_moe(
            backend,
            x=x,
            x_sf=x_sf,
            topk_idx=routes,
            topk_weights=routes.float(),
            weight_view=None,
            num_tokens=x.shape[0],
            top_k=2,
            hidden=8,
            device=torch.device("cpu"),
            output_dtype=torch.bfloat16,
        )

    staged, fresh = (
        (act[:3], act_sf[:3]),
        (torch.zeros(3, 4, dtype=torch.uint8), act_sf[:3].clone()),
    )
    backend._quantized_into = (act.data_ptr(), 3)
    assert run(*staged) == "out" and stage_activation == [False]
    assert run(*fresh) == "out" and stage_activation == [False, True]
    for record, inputs in (
        ((act.data_ptr(), 3), fresh),  # quantized into this staging, launched from elsewhere
        ((act.data_ptr(), 2), staged),  # another row count
        (None, staged),  # a view of the staging that this run's quantize_input did not write
    ):
        backend._quantized_into = record
        with pytest.raises(RuntimeError, match="quantize_input quantized"):
            run(*inputs)
        assert backend._quantized_into is None  # consumed by the launch it was meant for
    assert stage_activation == [False, True]


@pytest.mark.parametrize("drops_padding_rows", [True, False])
def test_run_moe_stages_with_the_forward_word_only_where_the_kernel_masks_dropped_routes(
    monkeypatch: pytest.MonkeyPatch, drops_padding_rows: bool
):
    """A bucket whose kernel's top-k reduce masks routes with an invalid expert stages the routes
    under the forward's live-row word, so the rows of padding requests are dropped; any other
    bucket stages every row (its reduce would read the dropped rows' stale cells)."""
    t = torch.zeros(8, 2)
    bufs = mm._MegaMoeBuffers(
        activation=torch.zeros(8, 4, dtype=torch.uint8),
        activation_sf=torch.zeros(8, 1, dtype=torch.uint8),
        topk_weights=t,
        combine_output=t,
        shared_workspace=t,
        peer_offsets=(0,),
        topk_idx_local=torch.full((8, 2), -1, dtype=torch.int64),
    )
    staged = []
    backend = SimpleNamespace(
        _active_launch_max_tokens=None,
        _quantized_into=None,
        ep_size=1,
        ep_rank=0,
        tactic_autotune=False,
        _genphase_direct_input_buckets=set(),
        _padding_drop_buckets={8} if drops_padding_rows else {16},
        _select_launch_max_tokens=lambda hint: 8,
        _acquire_buffers=lambda **kw: bufs,
        _stage_inputs=lambda **kw: staged.append(kw["live_row_word"]),
        _launch_megamoe_kernel=lambda **kw: "out",
    )
    backend._may_read_inputs_in_place = types.MethodType(IMPL._may_read_inputs_in_place, backend)
    word = torch.zeros(1, dtype=torch.int32)
    monkeypatch.setattr(mm.live_rows, "current_word", lambda device: word)
    routes = torch.zeros(3, 2, dtype=torch.int64)
    IMPL._run_moe(
        backend,
        x=torch.zeros(3, 4, dtype=torch.uint8),
        x_sf=torch.zeros(3, 1, dtype=torch.uint8),
        topk_idx=routes,
        topk_weights=routes.float(),
        weight_view=None,
        num_tokens=3,
        top_k=2,
        hidden=8,
        device=torch.device("cpu"),
        output_dtype=torch.bfloat16,
    )
    assert len(staged) == 1 and staged[0] is (word if drops_padding_rows else None)
