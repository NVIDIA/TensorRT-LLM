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

from tensorrt_llm._torch.modules.mamba import gdn_inforward_split
from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_fn
from tensorrt_llm._torch.modules.mamba.gdn_mixer import (
    _gdn_prefill_uses_flashinfer,
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


@pytest.mark.parametrize("linear_gate", [False, True], ids=["log_gate", "linear_gate"])
def test_split_prefill_core_matches_two_iterations(linear_gate):
    if linear_gate and not _gdn_prefill_uses_flashinfer():
        pytest.skip("only the FlashInfer scan takes the linear-space gate")
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

    # The reference scans take the log-space gate; the split path gets what the mixer passes it:
    # alpha = exp(g) when the FlashInfer scan is the resolved one.
    out = _inforward_split_prefill_core(
        q,
        k,
        v,
        g.exp() if linear_gate else g,
        beta,
        ssm_states,
        split,
        stage,
        None,
        g_is_linear=linear_gate,
    )
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
        at_point = None
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


def _random_states(rows, seed):
    torch.manual_seed(seed)
    return torch.randn(rows, NUM_V_HEADS, HEAD_DIM, HEAD_DIM, device="cuda").to(torch.bfloat16)


def test_split_kernels_match_index_ops():
    """Each gdn_inforward_split launch writes exactly the bytes of the index ops it replaces."""
    split = _split()
    total = sum(length for _, _, length, _ in _BATCH)
    q, k, v, g, beta = _inputs(total, seed=2)
    # One final state per chunked-prefill sequence: a split request runs as two.
    final = _random_states(split.gather.numel(), seed=3)
    ssm_states = _random_states(NUM_SLOTS, seed=4)
    assert gdn_inforward_split.split_core_ok(
        q,
        k,
        v,
        g,
        beta,
        final,
        ssm_states,
        ssm_states[:4],
        q.new_empty(1, total, NUM_V_HEADS, HEAD_DIM),
    )

    # Final states into the live slots, and the tails' initial states.
    expected_ssm = ssm_states.clone()
    expected_ssm.index_copy_(0, split.live, final.index_select(0, split.main_rows))
    tail_init = gdn_inforward_split.split_state_rows(final, ssm_states, split)
    assert torch.equal(ssm_states, expected_ssm)
    assert torch.equal(tail_init, final.index_select(0, split.split_rows))

    # Tail tokens.
    gathered = gdn_inforward_split.gather_tail_tokens(q, k, v, g, beta, split)
    for got, x in zip(gathered, (q, k, v, g, beta)):
        assert torch.equal(got, x.index_select(1, split.tail_tok))

    # Tail final states into the staging rows, tail outputs into the output.
    tail_final = _random_states(split.k, seed=5)
    tail_out = torch.randn(1, split.num_tail_tokens, NUM_V_HEADS, HEAD_DIM, device="cuda").to(
        torch.bfloat16
    )
    stage = _random_states(4, seed=6)
    core_out = torch.randn(1, total, NUM_V_HEADS, HEAD_DIM, device="cuda").to(torch.bfloat16)
    expected_stage, expected_out = stage.clone(), core_out.clone()
    expected_stage.index_copy_(0, split.stage_rows, tail_final)
    expected_out[0].index_copy_(0, split.tail_tok, tail_out[0])
    # Rows whose inner dims are not packed are refused, with nothing written.
    wide = torch.zeros(1, total, NUM_V_HEADS, 2 * HEAD_DIM, dtype=core_out.dtype, device="cuda")
    assert not gdn_inforward_split.scatter_tail(
        tail_final, stage, tail_out, wide[..., :HEAD_DIM], split
    )
    strided_stage = stage.transpose(2, 3)
    assert not gdn_inforward_split.scatter_tail(
        tail_final, strided_stage, tail_out, core_out, split
    )
    assert not wide.any()
    assert gdn_inforward_split.scatter_tail(tail_final, stage, tail_out, core_out, split)
    assert torch.equal(stage, expected_stage)
    assert torch.equal(core_out, expected_out)


def test_split_conv_states_kernel_matches_index_ops():
    split = _split()
    total = sum(length for _, _, length, _ in _BATCH)
    torch.manual_seed(7)
    mixed_qkv = torch.randn(total, CONV_DIM, device="cuda").to(torch.bfloat16)
    conv_states = torch.randn(NUM_SLOTS, CONV_DIM, D_CONV - 1, device="cuda").to(torch.bfloat16)
    stage = torch.randn(4, CONV_DIM, D_CONV - 1, device="cuda").to(torch.bfloat16)
    expected_states, expected_stage = conv_states.clone(), stage.clone()
    _inforward_split_conv_states(
        expected_states, mixed_qkv.index_select(0, split.conv_win), split, expected_stage
    )
    assert gdn_inforward_split.split_conv_states_ok(conv_states, mixed_qkv, stage)
    gdn_inforward_split.split_conv_states(conv_states, mixed_qkv, split, stage)
    assert torch.equal(conv_states, expected_states)
    assert torch.equal(stage, expected_stage)


def test_split_prefill_core_fused_copies_bitwise():
    """fused_copies only changes how the states and tokens move: outputs and states are bit-identical."""
    split = _split()
    total = sum(length for _, _, length, _ in _BATCH)
    q, k, v, g, beta = _inputs(total, seed=8)
    results = []
    for fused_copies in (False, True):
        ssm_states = _random_states(NUM_SLOTS, seed=9) * 0.01
        stage = torch.zeros(4, NUM_V_HEADS, HEAD_DIM, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        out = _inforward_split_prefill_core(
            q,
            k,
            v,
            g,
            beta,
            ssm_states,
            split,
            stage,
            None,
            g_is_linear=False,
            fused_copies=fused_copies,
        )
        results.append((out, ssm_states, stage))
    for reference, fused in zip(*results):
        assert torch.equal(reference, fused)


@pytest.mark.parametrize("fused_copies", [False, True])
def test_split_prefill_core_alpha_bitwise(fused_copies):
    """alpha = exp(g) as the gate (g_is_linear; the tails gather it in place of g) gives the bits of g alone."""
    if not _gdn_prefill_uses_flashinfer():
        pytest.skip("only the FlashInfer scan takes the linear-space gate")
    split = _split()
    total = sum(length for _, _, length, _ in _BATCH)
    q, k, v, g, beta = _inputs(total, seed=10)
    results = []
    for linear_gate in (False, True):
        ssm_states = _random_states(NUM_SLOTS, seed=11) * 0.01
        stage = torch.zeros(4, NUM_V_HEADS, HEAD_DIM, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        out = _inforward_split_prefill_core(
            q,
            k,
            v,
            torch.exp(g) if linear_gate else g,
            beta,
            ssm_states,
            split,
            stage,
            None,
            g_is_linear=linear_gate,
            fused_copies=fused_copies,
        )
        results.append((out, ssm_states, stage))
    for reference, with_alpha in zip(*results):
        assert torch.equal(reference, with_alpha)


def test_split_kernels_compile_once_for_every_batch():
    """The state-row split and the tail scatter take the batch's live / split counts at run time: one compiled
    variant each serves counts of 1, 2, 16 and 3 (no in-step compile for a new count class)."""
    from _torch.helpers import triton_num_compiled

    def states(rows, seed):
        return _random_states(rows, seed)

    def run(n_live, k):
        cuda = torch.device("cuda")
        split = types.SimpleNamespace(
            live=torch.arange(n_live, dtype=torch.int32, device=cuda) % NUM_SLOTS,
            main_rows=torch.arange(n_live, dtype=torch.int32, device=cuda),
            split_rows=torch.arange(k, dtype=torch.int32, device=cuda),
            k=k,
            stage_rows=torch.arange(k, dtype=torch.int32, device=cuda),
            tail_tok=torch.arange(2, dtype=torch.int32, device=cuda),
        )
        final = states(max(n_live, k), seed=n_live)
        ssm = states(NUM_SLOTS, seed=k)
        expected_ssm = ssm.clone()
        expected_ssm.index_copy_(0, split.live.long(), final.index_select(0, split.main_rows))
        tail_init = gdn_inforward_split.split_state_rows(final, ssm, split)
        if n_live <= NUM_SLOTS:  # distinct live slots
            assert torch.equal(ssm, expected_ssm)
        assert torch.equal(tail_init, final.index_select(0, split.split_rows))
        stage = states(k, seed=7)
        tail_final = states(k, seed=8)
        tail_out = torch.randn(1, 2, NUM_V_HEADS, HEAD_DIM, device=cuda).to(torch.bfloat16)
        core_out = torch.zeros(1, 4, NUM_V_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=cuda)
        assert gdn_inforward_split.scatter_tail(tail_final, stage, tail_out, core_out, split)
        assert torch.equal(stage, tail_final)

    run(3, 3)
    before = (
        triton_num_compiled(gdn_inforward_split._split_state_rows_kernel),
        triton_num_compiled(gdn_inforward_split._tail_scatter_kernel),
    )
    for n_live, k in ((1, 1), (2, 2), (16, 16), (8, 3)):
        run(n_live, k)
    assert (
        triton_num_compiled(gdn_inforward_split._split_state_rows_kernel),
        triton_num_compiled(gdn_inforward_split._tail_scatter_kernel),
    ) == before


def test_conv_state_and_tail_gather_kernels_compile_once_for_every_batch():
    """The conv-state split and the tail-token gather take their counts as grid sizes and their strides from the
    model's fixed widths: the variant compiled for the first split batch (the startup warmup's) serves every later
    split count, so serving compiles none of the four split kernels."""
    from _torch.helpers import triton_num_compiled

    cuda = torch.device("cuda")
    q, k, v, g, beta = _inputs(64, seed=3)

    def run(num_splits, num_tail):
        split = types.SimpleNamespace(
            k=num_splits,
            tail_slots=torch.arange(num_splits, dtype=torch.long, device=cuda) % NUM_SLOTS,
            stage_rows=torch.arange(num_splits, dtype=torch.long, device=cuda),
            conv_win=torch.arange(num_splits * (D_CONV - 1), dtype=torch.long, device=cuda) % 64,
            tail_tok=torch.arange(num_tail, dtype=torch.long, device=cuda),
        )
        mixed_qkv = torch.randn(64, CONV_DIM, device=cuda).to(torch.bfloat16)
        conv_states = torch.randn(NUM_SLOTS, CONV_DIM, D_CONV - 1, device=cuda).to(torch.bfloat16)
        stage = torch.zeros(NUM_SLOTS, CONV_DIM, D_CONV - 1, dtype=torch.bfloat16, device=cuda)
        gdn_inforward_split.split_conv_states(conv_states, mixed_qkv, split, stage)
        outs = gdn_inforward_split.gather_tail_tokens(q, k, v, g, beta, split)
        for x, o in zip((q, k, v, g, beta), outs):
            assert torch.equal(o, x.index_select(1, split.tail_tok))

    kernels = (
        gdn_inforward_split._split_conv_state_kernel,
        gdn_inforward_split._tail_token_gather_kernel,
    )
    run(2, 4)
    before = tuple(triton_num_compiled(kernel) for kernel in kernels)
    for num_splits, num_tail in ((1, 1), (3, 17), (8, 64), (5, 2)):
        run(num_splits, num_tail)
    assert tuple(triton_num_compiled(kernel) for kernel in kernels) == before
