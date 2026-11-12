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
