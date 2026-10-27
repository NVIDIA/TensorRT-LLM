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
"""In-forward Mamba snapshot at P in the GDN prefill vs the two-iteration path ([.., P), then [P, L))."""

import types

import pytest
import torch

from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_fn
from tensorrt_llm._torch.modules.mamba.gdn_mixer import (
    _inforward_split_conv_states,
    _inforward_split_prefill_core,
    chunk_gated_delta_rule,
)
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import InforwardSnapshotEntry

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

NUM_K_HEADS, NUM_V_HEADS, HEAD_DIM = 4, 8, 128
CONV_DIM, D_CONV = 256, 4
NUM_SLOTS = 8

# (request id, live slot, chunk length, plan entry or None)
_BATCH = [
    (11, 5, 40, None),
    (12, 2, 70, InforwardSnapshotEntry(row=0, point=66, num_before=66, num_after=4)),
    (13, 7, 100, InforwardSnapshotEntry(row=1, point=98, num_before=98, num_after=2)),
]


def _split():
    plan = {rid: entry for rid, _, _, entry in _BATCH if entry is not None}
    slots = {rid: slot for rid, slot, _, _ in _BATCH}
    manager = types.SimpleNamespace(
        conv_state_shape=[CONV_DIM, D_CONV - 1],
        inforward_snapshot_rows=4,
        inforward_snapshot_max_tail=16,
        inforward_snapshot_plan=plan,
        state_index=lambda rid: slots[rid],
    )
    attn = types.SimpleNamespace(
        request_ids=[rid for rid, _, _, _ in _BATCH],
        seq_lens=torch.tensor([length for _, _, length, _ in _BATCH], dtype=torch.int32),
    )
    metadata = Mamba2Metadata(max_batch_size=8, chunk_size=128)
    metadata._prepare_inforward_split(attn, manager, plan, len(_BATCH), len(_BATCH))
    return metadata.inforward_split


def _inputs(total, seed=0):
    torch.manual_seed(seed)
    q = torch.nn.functional.normalize(
        torch.randn(1, total, NUM_K_HEADS, HEAD_DIM, device="cuda"), dim=-1
    ).to(torch.bfloat16)
    k = torch.nn.functional.normalize(
        torch.randn(1, total, NUM_K_HEADS, HEAD_DIM, device="cuda"), dim=-1
    ).to(torch.bfloat16)
    v = (torch.randn(1, total, NUM_V_HEADS, HEAD_DIM, device="cuda") * 0.1).to(torch.bfloat16)
    g = -torch.rand(1, total, NUM_V_HEADS, device="cuda") * 0.05
    beta = torch.rand(1, total, NUM_V_HEADS, device="cuda")
    return q, k, v, g, beta


def _chunk(q, k, v, g, beta, initial, start, end):
    out, final = chunk_gated_delta_rule(
        q=q[:, start:end],
        k=k[:, start:end],
        v=v[:, start:end],
        g=g[:, start:end],
        beta=beta[:, start:end],
        initial_state=initial.unsqueeze(0),
        output_final_state=True,
        cu_seqlens=torch.tensor([0, end - start], dtype=torch.long, device="cuda"),
        head_first=False,
        use_qk_l2norm_in_kernel=False,
    )
    return out, final[0]


def test_split_prefill_core_matches_two_iterations():
    split = _split()
    total = sum(length for _, _, length, _ in _BATCH)
    q, k, v, g, beta = _inputs(total)
    ssm_states = (torch.randn(NUM_SLOTS, NUM_V_HEADS, HEAD_DIM, HEAD_DIM, device="cuda") * 0.01).to(
        torch.bfloat16
    )
    stage = torch.zeros(4, NUM_V_HEADS, HEAD_DIM, HEAD_DIM, dtype=torch.bfloat16, device="cuda")

    expected_out = torch.empty(1, total, NUM_V_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    expected_live = ssm_states.clone()
    expected_stage = stage.clone()
    start = 0
    for _, slot, length, entry in _BATCH:
        before = length if entry is None else entry.num_before
        out, state = _chunk(q, k, v, g, beta, ssm_states[slot], start, start + before)
        expected_out[:, start : start + before] = out
        expected_live[slot] = state.to(ssm_states.dtype)
        if entry is not None:
            # The tail as a context iteration of its own, from the state at P.
            out, end_state = _chunk(
                q, k, v, g, beta, state.to(ssm_states.dtype), start + before, start + length
            )
            expected_out[:, start + before : start + length] = out
            expected_stage[entry.row] = end_state.to(stage.dtype)
        start += length

    out = _inforward_split_prefill_core(q, k, v, g, beta, ssm_states, split, stage, None)
    torch.cuda.synchronize()
    # Batched and per-request chunked prefill may round differently in bf16, and the
    # tails start from the unrounded state at P: compare to bf16 tolerance.
    tol = dict(rtol=1.6e-2, atol=1e-3)
    torch.testing.assert_close(ssm_states, expected_live, **tol)
    torch.testing.assert_close(stage, expected_stage, **tol)
    torch.testing.assert_close(out.to(expected_out.dtype), expected_out, **tol)
    # Slots of other requests are untouched.
    untouched = [i for i in range(NUM_SLOTS) if i not in {slot for _, slot, _, _ in _BATCH}]
    assert torch.equal(ssm_states[untouched], expected_live[untouched])


def test_split_conv_states_match_two_iterations():
    split = _split()
    total = sum(length for _, _, length, _ in _BATCH)
    torch.manual_seed(1)
    mixed_qkv = torch.randn(total, CONV_DIM, device="cuda").to(torch.bfloat16)
    weight = (torch.randn(CONV_DIM, D_CONV, device="cuda") * 0.3).to(torch.bfloat16)
    bias = (torch.randn(CONV_DIM, device="cuda") * 0.1).to(torch.bfloat16)
    conv_states = torch.randn(NUM_SLOTS, CONV_DIM, D_CONV - 1, device="cuda").to(torch.bfloat16)
    stage = torch.zeros(4, CONV_DIM, D_CONV - 1, dtype=torch.bfloat16, device="cuda")
    slots = [slot for _, slot, _, _ in _BATCH]
    lengths = [length for _, _, length, _ in _BATCH]

    # Two-iteration reference, one request at a time.
    expected_out = torch.empty(CONV_DIM, total, dtype=torch.bfloat16, device="cuda")
    expected_live = conv_states.clone()
    expected_stage = stage.clone()
    start = 0
    for _, slot, length, entry in _BATCH:
        cuts = [0, length] if entry is None else [0, entry.num_before, length]
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            x = mixed_qkv[start + lo : start + hi].t().contiguous()
            expected_out[:, start + lo : start + hi] = causal_conv1d_fn(
                x,
                weight,
                bias,
                query_start_loc=torch.tensor([0, hi - lo], dtype=torch.int32, device="cuda"),
                cache_indices=torch.tensor([slot], dtype=torch.int32, device="cuda"),
                has_initial_state=torch.ones(1, dtype=torch.bool, device="cuda"),
                conv_states=expected_live,
            )
            if entry is not None and lo == 0:
                at_point = expected_live[slot].clone()
        if entry is not None:
            expected_stage[entry.row] = expected_live[slot]
            expected_live[slot] = at_point
        start += length

    # In-forward path: whole chunks, then the state swap.
    window = mixed_qkv.index_select(0, split.conv_win)
    cu = torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), dtype=torch.int32)
    out = causal_conv1d_fn(
        mixed_qkv.t().contiguous(),
        weight,
        bias,
        query_start_loc=cu.cuda(),
        cache_indices=torch.tensor(slots, dtype=torch.int32, device="cuda"),
        has_initial_state=torch.ones(len(slots), dtype=torch.bool, device="cuda"),
        conv_states=conv_states,
    )
    _inforward_split_conv_states(conv_states, window, split, stage)
    torch.cuda.synchronize()
    assert torch.equal(conv_states, expected_live)
    assert torch.equal(stage, expected_stage)
    assert torch.equal(out, expected_out)
