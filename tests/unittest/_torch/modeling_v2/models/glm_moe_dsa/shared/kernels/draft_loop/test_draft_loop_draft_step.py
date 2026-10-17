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
"""draft_loop fused draft-step metadata update versus the native draft-loop tail.

The native reference is the literal per-step tail of
``Eagle3OneModelWorker._forward_linear_draft_loop`` (position advance, step-0
switch to one query token, KV-length update, ``update_for_spec_dec``) running
on a real ``DSAtrtllmAttentionMetadata`` prepared from a real ``DSACacheManager``
at the served GEN configuration. After every step every tensor and scalar
attribute either path changed must be identical.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _draft_loop_fixtures import (  # noqa: E402
    MTP_K,
    add_gen_requests,
    assert_same_state,
    changed_keys,
    make_dsa_cache_manager,
    make_gen_dsa_metadata,
    native_draft_step_tail,
    tensor_state,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.draft_loop import (  # noqa: E402, E501
    draft_step as draft_step_mod,
)


def _step(md, use_draft_loop, i, batch_size, num_accepted, position_ids, gather_ids):
    pos_arg = position_ids if i + 1 < MTP_K else None
    gather = gather_ids if i == 0 else slice(0, batch_size)
    if use_draft_loop:
        result = draft_step_mod.draft_step_update(
            md, i, batch_size, 0, MTP_K, num_accepted, pos_arg, gather_ids if i == 0 else None
        )
        assert result is not None, draft_step_mod.step_ineligibility(md, batch_size, 0)
        return result.position_ids
    return native_draft_step_tail(md, i, batch_size, 0, MTP_K, num_accepted, position_ids, gather)


def _enter_draft(md):
    """The native loop's draft-KV entry: rebind to the draft buffers, refresh."""
    if md.draft_kv_cache_manager is None:
        return None
    from tensorrt_llm._torch.speculative.interface import prepare_attn_metadata_for_draft_replay

    saved = prepare_attn_metadata_for_draft_replay(md, md.draft_kv_cache_manager)
    md.on_update_kv_lens()
    return saved


def _exit_draft(md, saved):
    if saved is not None:
        from tensorrt_llm._torch.speculative.interface import (
            restore_attn_metadata_after_draft_replay,
        )

        restore_attn_metadata_after_draft_replay(md, saved)


def _run_loop(md, use_draft_loop, num_accepted, position_ids, gather_ids):
    batch_size = md.num_seqs
    md.prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda")
    saved = _enter_draft(md)
    states = []
    pos = position_ids
    for i in range(MTP_K):
        pos = _step(md, use_draft_loop, i, batch_size, num_accepted, pos, gather_ids)
        torch.cuda.synchronize()
        states.append((tensor_state(md), None if pos is None else pos.clone()))
    _exit_draft(md, saved)
    md.restore_from_spec_dec()
    md.on_update()
    return states, tensor_state(md)


def _inputs(batch_size, past_kv_lens, accepted):
    num_accepted = torch.tensor(accepted, dtype=torch.int32, device="cuda")
    # Target verify positions: past_kv_len .. past_kv_len + K for each request.
    pos = torch.cat(
        [torch.arange(n, n + MTP_K + 1, dtype=torch.int32) for n in past_kv_lens]
    ).cuda()
    gather_ids = (
        torch.arange(batch_size, dtype=torch.int64, device="cuda") * (MTP_K + 1)
        + num_accepted.to(torch.int64)
        - 1
    )
    return num_accepted, pos, gather_ids


CASES = [
    # (past KV lengths, accepted counts)
    ([391061], [4]),
    ([391061], [1]),
    ([391061], [6]),
    ([2100], [3]),
    ([65536 - 3], [6]),
    ([(1 << 20) - 64], [2]),
    ([491061, 32768, 131072], [6, 1, 3]),
    ([4096, 4097, 4098, 4099, 4100, 4101, 4102, 300000], [1, 2, 3, 4, 5, 6, 4, 3]),
]


@pytest.mark.parametrize("separate_draft", [False, True], ids=["shared_kv", "draft_kv"])
@pytest.mark.parametrize("past_kv_lens, accepted", CASES)
def test_step_update_matches_native(past_kv_lens, accepted, separate_draft):
    batch_size = len(past_kv_lens)
    cache_manager, mapping, sparse = make_dsa_cache_manager(batch_size, max(past_kv_lens))
    draft_manager = None
    try:
        ids = list(range(batch_size))
        add_gen_requests(cache_manager, ids, past_kv_lens)
        if separate_draft:
            draft_manager, _, _ = make_dsa_cache_manager(batch_size, max(past_kv_lens))
            add_gen_requests(draft_manager, ids, past_kv_lens)
        md_native = make_gen_dsa_metadata(
            cache_manager, mapping, sparse, ids, past_kv_lens, draft_cache_manager=draft_manager
        )
        md_draft_loop = make_gen_dsa_metadata(
            cache_manager, mapping, sparse, ids, past_kv_lens, draft_cache_manager=draft_manager
        )
        reason = draft_step_mod.step_ineligibility(md_draft_loop, batch_size, 0)
        assert reason is None, reason
        print(
            f"expand_for_dsl={md_draft_loop.expand_for_dsl} factor={md_draft_loop.dsl_expand_factor} "
            f"num_sms={md_draft_loop.num_sms} split_kv={draft_step_mod._SPLIT_KV_BY_SMS[md_draft_loop.num_sms]} "
            f"width={md_draft_loop.gen_token_stride}"
        )
        base_native, base_draft_loop = tensor_state(md_native), tensor_state(md_draft_loop)
        inputs = _inputs(batch_size, past_kv_lens, accepted)
        steps_native, final_native = _run_loop(md_native, False, *inputs)
        steps_draft_loop, final_draft_loop = _run_loop(md_draft_loop, True, *inputs)
        for i, ((s_n, p_n), (s_w, p_w)) in enumerate(zip(steps_native, steps_draft_loop)):
            keys = changed_keys(base_native, s_n) | changed_keys(base_draft_loop, s_w)
            # The query lengths always change (K + 1 -> 1); the step buffers are
            # compared whether or not they changed in this step.
            assert "_seq_lens_cuda" in keys
            assert_same_state(s_n, s_w, keys | set(_SNAPSHOT_FIELDS), f"step {i}")
            if p_n is None:
                assert p_w is None
            else:
                assert p_w is not None and p_w.dtype == p_n.dtype and torch.equal(p_w, p_n)
        keys = changed_keys(base_native, final_native) | changed_keys(
            base_draft_loop, final_draft_loop
        )
        assert_same_state(final_native, final_draft_loop, keys, "after restore")
    finally:
        cache_manager.shutdown()
        if draft_manager is not None:
            draft_manager.shutdown()


def test_ineligible_short_context_is_untouched():
    """Dense top-k (short context) keeps the native chain; draft_loop changes nothing."""
    cache_manager, mapping, sparse = make_dsa_cache_manager(1, 1000)
    try:
        add_gen_requests(cache_manager, [0], [1000])
        md = make_gen_dsa_metadata(cache_manager, mapping, sparse, [0], [1000])
        assert draft_step_mod.step_ineligibility(md, 1, 0) == "dense top-k fallback active"
        md.prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda")
        before = tensor_state(md)
        num_accepted, pos, gather_ids = _inputs(1, [1000], [3])
        assert (
            draft_step_mod.draft_step_update(md, 0, 1, 0, MTP_K, num_accepted, pos, gather_ids)
            is None
        )
        torch.cuda.synchronize()
        assert not changed_keys(before, tensor_state(md))
        md.restore_from_spec_dec()
    finally:
        cache_manager.shutdown()


def test_schedule_matches_deepgemm():
    """The draft_loop schedule equals DeepGEMM's on random 1-D/2-D context lengths."""
    from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata

    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    split_kv = draft_step_mod.schedule_self_check(num_sms, torch.device("cuda"))
    assert split_kv is not None
    gen = torch.Generator().manual_seed(80)
    for trial in range(300):
        num_requests = int(torch.randint(1, 17, (1,), generator=gen))
        next_n = int(torch.randint(1, 9, (1,), generator=gen))
        hi = [64, 4096, 1 << 20][trial % 3]
        lens = torch.randint(0, hi + 15, (num_requests, 1), generator=gen, dtype=torch.int32)
        lens = lens.expand(-1, next_n).contiguous().cuda()
        native = get_paged_mqa_logits_metadata(lens, 64, num_sms)
        ours = draft_step_mod.paged_mqa_schedule(lens, num_sms, split_kv)
        assert torch.equal(native, ours), (lens.tolist(), next_n)


_SNAPSHOT_FIELDS = (
    "kv_lens_cuda",
    "kv_lens_cuda_2d",
    "scheduler_metadata_buffer",
    "scheduler_metadata_buffer_full_next_n",
    "kv_lens_expanded_cuda",
    "scheduler_metadata_buffer_expanded",
    "req_idx_per_token",
    "slot_mapping_fp8",
    "slot_mapping_scale",
    "gen_kv_indptr",
    "gen_cached_token_indptr",
)


def _capture_loop(md, use_draft_loop, kv_init, num_accepted, pos0, gather_ids):
    """Capture prepare_for_spec_dec + 5 steps + restore, with in-graph snapshots."""
    batch_size = md.num_seqs
    snaps = [{f: torch.empty_like(getattr(md, f)) for f in _SNAPSHOT_FIELDS} for _ in range(MTP_K)]
    seq_snaps = [torch.empty((batch_size,), dtype=torch.int32, device="cuda") for _ in range(MTP_K)]
    pos_snaps = [
        torch.empty((batch_size,), dtype=pos0.dtype, device="cuda") for _ in range(MTP_K - 1)
    ]

    def body():
        md.kv_lens_cuda[:batch_size].copy_(kv_init)
        md.prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda")
        saved = _enter_draft(md)
        pos = pos0
        for i in range(MTP_K):
            pos = _step(md, use_draft_loop, i, batch_size, num_accepted, pos, gather_ids)
            for f in _SNAPSHOT_FIELDS:
                snaps[i][f].copy_(getattr(md, f))
            seq_snaps[i].copy_(md.seq_lens_cuda[:batch_size])
            if pos is not None:
                pos_snaps[i].copy_(pos)
        _exit_draft(md, saved)
        md.restore_from_spec_dec()
        md.on_update()

    # Warm up eagerly (JIT and the schedule self-check), then reset host state.
    host_seq = md._seq_lens.clone()
    num_contexts = md.num_contexts
    body()
    md._seq_lens.copy_(host_seq)
    md.num_contexts = num_contexts
    md.use_spec_decoding = True
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    md._seq_lens.copy_(host_seq)
    md.num_contexts = num_contexts
    return graph, snaps, seq_snaps, pos_snaps


@pytest.mark.parametrize("separate_draft", [False, True], ids=["shared_kv", "draft_kv"])
def test_cuda_graph_replays_match_native(separate_draft):
    past = [391061, 120000]
    batch_size = len(past)
    cache_manager, mapping, sparse = make_dsa_cache_manager(batch_size, max(past) + 64)
    draft_manager = None
    try:
        ids = list(range(batch_size))
        add_gen_requests(cache_manager, ids, past)
        if separate_draft:
            draft_manager, _, _ = make_dsa_cache_manager(batch_size, max(past) + 64)
            add_gen_requests(draft_manager, ids, past)
        md_native = make_gen_dsa_metadata(
            cache_manager, mapping, sparse, ids, past, draft_cache_manager=draft_manager
        )
        md_draft_loop = make_gen_dsa_metadata(
            cache_manager, mapping, sparse, ids, past, draft_cache_manager=draft_manager
        )
        kv_init = md_native.kv_lens_cuda[:batch_size].clone()
        num_accepted, pos0, gather_ids = _inputs(batch_size, past, [3, 5])
        g_native = _capture_loop(md_native, False, kv_init, num_accepted, pos0, gather_ids)
        g_draft_loop = _capture_loop(md_draft_loop, True, kv_init, num_accepted, pos0, gather_ids)
        gen = torch.Generator().manual_seed(5)
        for replay in range(12):
            shift = int(torch.randint(-40, 40, (1,), generator=gen))
            kv_init.copy_(torch.tensor([n + MTP_K + 1 + shift for n in past], dtype=torch.int32))
            num_accepted.copy_(
                torch.randint(1, MTP_K + 2, (batch_size,), generator=gen, dtype=torch.int32)
            )
            gather_ids.copy_(torch.arange(batch_size) * (MTP_K + 1) + num_accepted.cpu() - 1)
            pos0.copy_(torch.cat([torch.arange(n + shift, n + shift + MTP_K + 1) for n in past]))
            g_native[0].replay()
            g_draft_loop[0].replay()
            torch.cuda.synchronize()
            for i in range(MTP_K):
                for f in _SNAPSHOT_FIELDS:
                    assert torch.equal(g_native[1][i][f], g_draft_loop[1][i][f]), (replay, i, f)
                assert torch.equal(g_native[2][i], g_draft_loop[2][i]), (replay, i, "seq_lens")
                if i < MTP_K - 1:
                    assert torch.equal(g_native[3][i], g_draft_loop[3][i]), (replay, i, "pos")
    finally:
        cache_manager.shutdown()
        if draft_manager is not None:
            draft_manager.shutdown()
