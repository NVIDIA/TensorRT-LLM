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
"""``fused_select_accept`` with the KDA replay record, and the DSpark worker's gate for it.

The kernel's replay write must equal ``KDAReplayState.record_acceptance`` on main's replay
count, which is the first int32 of each slot's 16-byte record (a strided 1-D view).
``K3DFlashWorker._kda_replay_operands`` hands the kernel those operands only when
``update_mamba_states`` is ``KimiK3HybridCacheManager``'s replay record and nothing else.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.dspark.worker import (
    K3DFlashWorker,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.spec.k3_fused_select_accept import (  # noqa: E501
    k3_fused_select_accept,
)
from tensorrt_llm._torch.modules.kimi_kda.cache_manager import (
    KDAIntermediateState,
    KDAReplayState,
    KimiK3HybridCacheManager,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2 import (
    MambaHybridCacheManagerV2,
)

_RECORD_INT32S = 4  # 16-byte replay-count record per slot


def _pairs(num_rows, n_slots, vocab, device, generator):
    """Interleaved (value, global index) candidates with distinct values per row."""
    vals = torch.rand((num_rows, n_slots), generator=generator, device=device)
    idx = torch.randint(0, vocab, (num_rows, n_slots), generator=generator, device=device)
    buf = torch.stack((vals, idx.to(torch.float32)), dim=-1).reshape(num_rows, 2 * n_slots)
    selected = idx.gather(1, vals.argmax(dim=1, keepdim=True)).squeeze(1).to(torch.int32)
    return (buf.contiguous(), buf.stride(0), 2, n_slots), selected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("num_ctx,num_gens,draft_len", [(0, 5, 3), (2, 7, 5), (1, 1, 1)])
def test_replay_record_matches_record_acceptance(num_ctx, num_gens, draft_len):
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(1234 + num_ctx * 10 + num_gens)
    width = draft_len + 1
    batch = num_ctx + num_gens
    pairs, selected = _pairs(num_ctx + num_gens * width, 4, 1000, device, gen)

    # Drafts agree with the target's selection on a per-request prefix.
    target = selected[num_ctx:].view(num_gens, width)
    agree = torch.randint(0, draft_len + 1, (num_gens,), generator=gen, device=device)
    cols = torch.arange(draft_len, device=device)
    draft = torch.where(cols[None, :] < agree[:, None], target[:, :draft_len], -1).to(torch.int32)
    expected_num_accepted = 1 + agree.to(torch.int32)

    capacity = 3 * batch + 2
    records = torch.randint(0, 9, (capacity, _RECORD_INT32S), dtype=torch.int32, device=device)
    perm = torch.randperm(capacity, generator=gen, device=device)
    state_indices = perm[:batch].to(torch.int32).contiguous()
    dummy = torch.zeros(batch, dtype=torch.bool, device=device)
    dummy[num_ctx] = True  # first generation request is a dummy: its slot keeps its value
    kv_lens = torch.randint(10, 50, (batch,), dtype=torch.int32, device=device)

    reference = KDAReplayState(draft_len)
    ref_records = records.clone()
    reference.prev_num_accepted_tokens = ref_records[:, 0]
    reference.record_acceptance(state_indices[num_ctx:], expected_num_accepted - 1, dummy[num_ctx:])

    fused_records = records.clone()
    prev = fused_records[:, 0]
    assert prev.stride(0) == _RECORD_INT32S
    fused_kv_lens = kv_lens.clone()
    accepted, num_accepted, rewind = k3_fused_select_accept(
        pairs,
        draft,
        num_ctx,
        batch,
        kda_replay=(prev, state_indices, dummy),
        kv_lens=fused_kv_lens,
    )
    torch.cuda.synchronize()

    assert torch.equal(num_accepted[num_ctx:], expected_num_accepted)
    assert torch.equal(
        num_accepted[:num_ctx], torch.ones(num_ctx, dtype=torch.int32, device=device)
    )
    assert torch.equal(accepted[num_ctx:], target)
    assert torch.equal(accepted[:num_ctx, 0], selected[:num_ctx])
    # The replay count matches main's eager record and the rest of each record is untouched.
    assert torch.equal(fused_records, ref_records)
    assert torch.equal(rewind, 1 - expected_num_accepted)
    assert torch.equal(fused_kv_lens[:num_ctx], kv_lens[:num_ctx])
    assert torch.equal(fused_kv_lens[num_ctx:], kv_lens[num_ctx:] + 1)


def _k3_manager(*, state, state_indices=None, context_only=False, mamba_layers=2):
    mgr = object.__new__(KimiK3HybridCacheManager)
    mgr.kda_context_only = context_only
    mgr.local_num_mamba_layers = mamba_layers
    mgr._speculative_state = state
    mgr._dummy_request_mask_buffer = SimpleNamespace(tensor=torch.zeros(8, dtype=torch.bool))
    mgr.get_state_indices = lambda: state_indices
    return mgr


def _replay_state():
    state = KDAReplayState(3)
    state.prev_num_accepted_tokens = torch.zeros((16, _RECORD_INT32S), dtype=torch.int32)[:, 0]
    return state


def _metadata(mgr, state_indices):
    return SimpleNamespace(
        kv_cache_manager=mgr, mamba_metadata=SimpleNamespace(state_indices=state_indices)
    )


def test_gate_hands_over_the_replay_record():
    state = _replay_state()
    indices = torch.arange(8, dtype=torch.int32)
    mgr = _k3_manager(state=state)
    fusable, operands = K3DFlashWorker._kda_replay_operands(_metadata(mgr, indices))
    assert fusable
    prev, state_indices, dummy = operands
    assert prev is state.prev_num_accepted_tokens
    assert state_indices is indices
    assert dummy is mgr._dummy_request_mask


def test_gate_falls_back_to_the_manager_state_indices():
    indices = torch.arange(8, dtype=torch.int32)
    mgr = _k3_manager(state=_replay_state(), state_indices=indices)
    fusable, operands = K3DFlashWorker._kda_replay_operands(_metadata(mgr, None))
    assert fusable and operands[1] is indices


@pytest.mark.parametrize("context_only,mamba_layers", [(True, 2), (False, 0)])
def test_gate_fuses_with_nothing_to_record(context_only, mamba_layers):
    mgr = _k3_manager(state=_replay_state(), context_only=context_only, mamba_layers=mamba_layers)
    assert K3DFlashWorker._kda_replay_operands(_metadata(mgr, None)) == (True, None)


def test_gate_refuses_without_a_replay_record():
    mgr = _k3_manager(state=KDAIntermediateState())
    indices = torch.arange(8, dtype=torch.int32)
    assert K3DFlashWorker._kda_replay_operands(_metadata(mgr, indices)) == (False, None)


def test_gate_refuses_strided_state_indices():
    mgr = _k3_manager(state=_replay_state())
    indices = torch.arange(16, dtype=torch.int32)[::2]
    assert K3DFlashWorker._kda_replay_operands(_metadata(mgr, indices)) == (False, None)


def test_gate_refuses_other_managers():
    other = object.__new__(MambaHybridCacheManagerV2)
    assert K3DFlashWorker._kda_replay_operands(_metadata(other, None)) == (False, None)

    class _Overrides(KimiK3HybridCacheManager):
        def update_mamba_states(self, *args, **kwargs):
            pass

    mgr = object.__new__(_Overrides)
    assert K3DFlashWorker._kda_replay_operands(_metadata(mgr, None)) == (False, None)
