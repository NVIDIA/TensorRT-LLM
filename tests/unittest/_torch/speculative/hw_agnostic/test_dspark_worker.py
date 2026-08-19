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
"""GPU unit tests for the DSpark worker and metadata plumbing.

Covers the framework-side logic that does NOT need the full draft model:
``DSparkSpecMetadata`` hidden-state capture (incl. the mHC hc-mean reduction)
and ``DSparkWorker`` slot / rolling-KV-window management. The end-to-end block
draft and acceptance path is covered by the DSpark test in
``integration/defs/accuracy/test_llm_api_pytorch.py``.
"""

import types

import pytest
import torch

from tensorrt_llm._torch.speculative.dspark import DSparkSpecMetadata, DSparkWorker
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="DSpark metadata/worker allocate CUDA buffers"
)

HIDDEN = 128
NCAP = 3
HC_MULT = 4


def _make_metadata(max_num_requests=8, max_num_tokens=64, layers=(58, 59, 60)):
    return DSparkSpecMetadata(
        max_draft_len=5,
        max_total_draft_tokens=5,
        spec_dec_mode=SpeculativeDecodingMode.DSPARK,
        max_num_requests=max_num_requests,
        layers_to_capture=list(layers),
        hidden_size=HIDDEN,
        max_num_tokens=max_num_tokens,
        dtype=torch.bfloat16,
    )


def test_metadata_buffer_and_layer_lookup():
    meta = _make_metadata()
    assert meta.num_capture_layers == NCAP
    assert meta.captured_hidden_states.shape == (64, HIDDEN * NCAP)
    # sorted, O(1) membership
    assert meta.is_layer_capture(58) and meta.is_layer_capture(60)
    assert not meta.is_layer_capture(0) and not meta.is_layer_capture(61)


def test_metadata_capture_plain_hidden():
    """A [num_tokens, hidden] capture is stored at the layer's slice as-is."""
    meta = _make_metadata()
    hs = torch.randn(4, HIDDEN, device="cuda", dtype=torch.bfloat16)
    meta.maybe_capture_hidden_states(59, hs)  # layer 59 -> capture index 1
    got = meta.get_hidden_states(4)
    assert torch.equal(got[:, HIDDEN : 2 * HIDDEN], hs)


def test_metadata_capture_hc_mean_reduction():
    """A flattened mHC residual [N, hc_mult*hidden] is reduced by mean over hc."""
    meta = _make_metadata()
    mhc = torch.randn(4, HC_MULT * HIDDEN, device="cuda", dtype=torch.bfloat16)
    meta.maybe_capture_hidden_states(58, mhc)  # layer 58 -> capture index 0
    expected = mhc.reshape(4, HC_MULT, HIDDEN).mean(dim=1)
    got = meta.get_hidden_states(4)
    assert torch.equal(got[:, 0:HIDDEN], expected)


def test_metadata_no_capture_for_unlisted_layer():
    meta = _make_metadata()
    meta.captured_hidden_states.zero_()
    meta.maybe_capture_hidden_states(10, torch.randn(4, HIDDEN, device="cuda"))
    assert torch.count_nonzero(meta.get_hidden_states(4)) == 0


def test_metadata_prepare_batch_indices():
    meta = _make_metadata()
    meta.request_ids = [7, 3, 5]
    meta.prepare()
    assert meta.batch_indices_cuda[:3].tolist() == [0, 1, 2]


def _make_worker():
    cfg = types.SimpleNamespace(
        max_draft_len=5,
        spec_dec_mode=SpeculativeDecodingMode.DSPARK,
        confidence_threshold=0.5,
    )
    from tensorrt_llm.mapping import Mapping

    return DSparkWorker(cfg, Mapping())


def _fake_draft_model(num_stages=3, window_size=128, head_dim=64):
    return types.SimpleNamespace(
        num_stages=num_stages,
        block_size=5,
        _attn_params={
            "window_size": window_size,
            "head_dim": head_dim,
            "n_heads": 128,
            "softmax_scale": 1.0,
            "eps": 1e-6,
        },
    )


def test_worker_lazy_init_window_buffers():
    worker = _make_worker()
    dm = _fake_draft_model(num_stages=3, window_size=128, head_dim=64)
    meta = _make_metadata(max_num_requests=8)
    worker._lazy_init(dm, meta)
    # max_batch (8) request slots + 1 scratch row for padded / unknown IDs.
    assert worker._kv_windows.shape == (9, 3, 128, 64)
    assert worker._ctx_len.shape == (9,)
    assert worker._valid_len.shape == (9,)
    assert worker._position_initialized.shape == (9,)
    assert worker._scratch_slot == 8
    # Dummy-id floor separates real request ids from CUDA-graph padding ids.
    assert worker._graph_dummy_id_floor == (1 << 64) - 1 - worker.max_draft_len
    # The scratch row is never handed out through the free pool.
    assert list(worker._free_slots) == list(range(8))
    assert worker._batch_to_slot is not None
    assert worker._batch_to_slot.shape == (8,)
    assert worker._batch_to_slot.device.type == "cuda"
    # idempotent
    buf_id = id(worker._kv_windows)
    worker._lazy_init(dm, meta)
    assert id(worker._kv_windows) == buf_id


def test_worker_rejects_mismatched_block_size():
    worker = _make_worker()
    draft_model = _fake_draft_model()
    draft_model.block_size = 4

    with pytest.raises(ValueError, match="block_size must equal worker max_draft_len"):
        worker._lazy_init(draft_model, _make_metadata())


def test_worker_slot_assignment_and_reset():
    worker = _make_worker()
    worker._lazy_init(_fake_draft_model(), _make_metadata(max_num_requests=4))

    s0 = worker._assign_slot(100, reset=False)
    s1 = worker._assign_slot(101, reset=False)
    assert s0 != s1
    # same request id -> same slot (no reset)
    assert worker._assign_slot(100, reset=False) == s0

    # mark a position, then reset -> slot freed + window/pos cleared
    worker._ctx_len[s0] = 42
    worker._valid_len[s0] = 8
    worker._position_initialized[s0] = True
    worker._kv_windows[s0].fill_(1.0)
    s0b = worker._assign_slot(100, reset=True)
    assert int(worker._ctx_len[s0b]) == 0
    assert int(worker._valid_len[s0b]) == 0
    assert not bool(worker._position_initialized[s0b])
    assert float(worker._kv_windows[s0b].abs().sum()) == 0.0


def test_worker_slot_exhaustion_preserves_live_request():
    worker = _make_worker()
    worker._lazy_init(_fake_draft_model(), _make_metadata(max_num_requests=1))

    slot = worker._assign_slot(100, reset=False)
    worker._ctx_len[slot] = 42
    worker._kv_windows[slot].fill_(1.0)

    with pytest.raises(RuntimeError, match="no free rolling-window slots"):
        worker._assign_slot(101, reset=False)

    assert worker._req_to_slot == {100: slot}
    assert int(worker._ctx_len[slot]) == 42
    assert torch.all(worker._kv_windows[slot] == 1.0)


def test_seed_context_windows_preserves_state_across_prefill_chunks():
    class DraftModel:
        num_stages = 1
        block_size = 5
        _attn_params = {
            "window_size": 8,
            "head_dim": 4,
            "n_heads": 128,
            "softmax_scale": 1.0,
            "eps": 1e-6,
        }

        def __init__(self):
            self.written_positions = []

        def write_context_windows(self, hidden, positions, windows):
            self.written_positions.append(positions.clone())
            windows.add_(1)

    worker = _make_worker()
    draft_model = DraftModel()
    metadata = types.SimpleNamespace(
        max_num_requests=1,
        request_ids=[100],
        get_hidden_states=lambda _num_tokens: torch.zeros(
            3, HIDDEN * NCAP, device="cuda", dtype=torch.bfloat16
        ),
    )
    worker._lazy_init(draft_model, metadata)

    first_chunk = types.SimpleNamespace(num_contexts=1, _seq_lens=[3])
    worker._seed_context_windows(
        draft_model, metadata, first_chunk, torch.tensor([[0, 1, 2]], device="cuda"), 3
    )
    slot = worker._req_to_slot[100]
    assert int(worker._ctx_len[slot]) == 3
    assert int(worker._valid_len[slot]) == 3
    assert bool(worker._position_initialized[slot])

    metadata.get_hidden_states = lambda _num_tokens: torch.zeros(
        2, HIDDEN * NCAP, device="cuda", dtype=torch.bfloat16
    )
    second_chunk = types.SimpleNamespace(num_contexts=1, _seq_lens=[2])
    worker._seed_context_windows(
        draft_model, metadata, second_chunk, torch.tensor([[3, 4]], device="cuda"), 2
    )

    assert int(worker._ctx_len[slot]) == 5
    assert int(worker._valid_len[slot]) == 5
    assert [positions.tolist() for positions in draft_model.written_positions] == [
        [1, 2, 3],
        [4, 5],
    ]
    assert torch.all(worker._kv_windows[slot] == 2.0)


def test_prepare_builds_batch_to_slot_on_batched_path():
    """prepare() mirrors the host slot map into _batch_to_slot (default batched path)."""
    worker = _make_worker()
    meta = _make_metadata(max_num_requests=4)
    worker._lazy_init(_fake_draft_model(), meta)  # batched is the default
    meta._dspark_worker = worker

    # Assign slots for two requests (as the prefill path would).
    sa = worker._assign_slot(100, reset=True)
    sb = worker._assign_slot(101, reset=True)

    meta.request_ids = [101, 100]
    meta.prepare()
    # Mirror reflects request-order -> slot.
    assert worker._batch_to_slot[:2].tolist() == [sb, sa]


def test_prepare_frees_stale_slots_on_batched_path():
    """A request that drops out of the batch returns its slot to the free pool."""
    worker = _make_worker()
    meta = _make_metadata(max_num_requests=4)
    worker._lazy_init(_fake_draft_model(), meta)
    meta._dspark_worker = worker

    sa = worker._assign_slot(100, reset=True)
    worker._assign_slot(101, reset=True)
    worker._ctx_len[sa] = 17
    worker._valid_len[sa] = 8
    worker._position_initialized[sa] = True

    # Only request 101 survives; 100's slot must be freed + cleared.
    meta.request_ids = [101]
    meta.prepare()
    assert 100 not in worker._req_to_slot
    assert sa in worker._free_slots
    assert int(worker._ctx_len[sa]) == 0
    assert int(worker._valid_len[sa]) == 0
    assert not bool(worker._position_initialized[sa])


def test_prepare_maps_unknown_request_to_scratch_row_not_slot_zero():
    """Padded / unknown request IDs route to the scratch row, never a live slot.

    Regression for the rolling-window aliasing bug (GitHub #16767): an unknown
    request id (CUDA-graph padding, ADP idle request, or a disagg seed forward
    without a real id) must not overwrite the request that owns slot 0.
    """
    worker = _make_worker()
    meta = _make_metadata(max_num_requests=4)
    worker._lazy_init(_fake_draft_model(), meta)
    meta._dspark_worker = worker

    # A live request takes the first free slot (0) and populates its window.
    s_real = worker._assign_slot(100, reset=True)
    assert s_real == 0
    worker._ctx_len[s_real] = 17
    worker._kv_windows[s_real].fill_(1.0)

    # Batch contains the live request plus an unknown id (e.g. graph padding).
    meta.request_ids = [100, 999]
    meta.prepare()

    # The unknown id maps to the scratch row, not to slot 0.
    assert worker._batch_to_slot[:2].tolist() == [s_real, worker._scratch_slot]
    assert worker._scratch_slot != s_real
    # It is not silently registered as a real request and did not consume a slot.
    assert 999 not in worker._req_to_slot
    assert list(worker._free_slots) == [1, 2, 3]
    # The live request's rolling window and position are untouched.
    assert int(worker._ctx_len[s_real]) == 17
    assert torch.all(worker._kv_windows[s_real] == 1.0)


def test_prepare_assigns_slots_to_disagg_generation_requests():
    """Disagg gen requests (never seeded here) get distinct slots, not one shared row.

    Regression for GitHub #16767: on the disaggregated generation server the
    prompt is prefilled (and the DSpark window seeded) on the *context* server,
    so ``_seed_context_windows`` never runs here and ``_req_to_slot`` stays empty.
    ``prepare()`` must therefore assign each real generation request its own
    rolling-window slot instead of collapsing them all onto the shared scratch
    row (which corrupts drafts and collapses accept length at batch size > 1).
    """
    worker = _make_worker()
    meta = _make_metadata(max_num_requests=4)
    worker._lazy_init(_fake_draft_model(), meta)
    meta._dspark_worker = worker

    # All-generation batch (num_contexts == 0), no prior seeding.
    meta.request_ids = [1000, 1001]
    meta.num_generations = 2
    meta.prepare()

    s0 = worker._req_to_slot[1000]
    s1 = worker._req_to_slot[1001]
    assert s0 != s1
    assert s0 != worker._scratch_slot and s1 != worker._scratch_slot
    assert worker._batch_to_slot[:2].tolist() == [s0, s1]

    # Stable across steps: the same ids keep their slots (no churn / reassignment).
    meta.prepare()
    assert worker._req_to_slot[1000] == s0
    assert worker._req_to_slot[1001] == s1
    assert worker._batch_to_slot[:2].tolist() == [s0, s1]


def test_prepare_keeps_dummy_generation_requests_on_scratch_row():
    """ADP-idle (id 0) and CUDA-graph padding dummies never consume a real slot."""
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID
    from tensorrt_llm._torch.pyexecutor.llm_request import ATTENTION_DP_DUMMY_REQUEST_ID

    worker = _make_worker()
    meta = _make_metadata(max_num_requests=4)
    worker._lazy_init(_fake_draft_model(), meta)
    meta._dspark_worker = worker

    graph_dummy = CUDA_GRAPH_DUMMY_REQUEST_ID - worker.max_draft_len
    meta.request_ids = [1000, ATTENTION_DP_DUMMY_REQUEST_ID, graph_dummy]
    meta.num_generations = 3
    meta.prepare()

    s_real = worker._req_to_slot[1000]
    assert s_real != worker._scratch_slot
    # Dummies are neither registered nor given a real slot; they map to scratch.
    assert ATTENTION_DP_DUMMY_REQUEST_ID not in worker._req_to_slot
    assert graph_dummy not in worker._req_to_slot
    assert worker._batch_to_slot[:3].tolist() == [
        s_real,
        worker._scratch_slot,
        worker._scratch_slot,
    ]
    # Exactly one real slot consumed.
    assert list(worker._free_slots) == [1, 2, 3]


class _RecordingDraftModel:
    num_stages = 1
    block_size = 5
    _attn_params = {
        "window_size": 8,
        "head_dim": 4,
        "n_heads": 128,
        "softmax_scale": 1.0,
        "eps": 1e-6,
    }

    def __init__(self):
        self.forward_calls = []

    def write_context_windows_batched(self, *args):
        pass

    def forward_batched(self, main_hidden, bonus, start_pos, **kwargs):
        self.forward_calls.append(
            {
                "main_hidden": main_hidden.clone(),
                "bonus": bonus.clone(),
                "start_pos": start_pos.clone(),
                "valid_len": kwargs["valid_len"].clone(),
            }
        )
        logits = torch.zeros(main_hidden.shape[0], self.block_size, 8, device=main_hidden.device)
        return None, None, logits


def test_generation_state_cuda_graph_bootstrap_and_replay():
    """Position bootstrap and valid-length advancement remain graph replay safe."""
    worker = _make_worker()
    worker._lazy_init(_fake_draft_model(window_size=8), _make_metadata(max_num_requests=2))
    slots = torch.tensor([0, 1], device="cuda", dtype=torch.long)
    num_accepted = torch.tensor([1, 2], device="cuda", dtype=torch.long)
    input_positions = torch.tensor([4016, 87], device="cuda", dtype=torch.long)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        worker._advance_generation_state(slots, num_accepted, input_positions)

    worker._ctx_len[slots] = 0
    worker._valid_len[slots] = 0
    worker._position_initialized[slots] = False
    graph.replay()
    assert worker._ctx_len[slots].tolist() == [4017, 89]
    assert worker._valid_len[slots].tolist() == [1, 2]
    assert worker._position_initialized[slots].tolist() == [True, True]

    num_accepted.copy_(torch.tensor([2, 3], device="cuda"))
    graph.replay()
    assert worker._ctx_len[slots].tolist() == [4019, 92]
    assert worker._valid_len[slots].tolist() == [3, 5]


def test_disagg_position_bootstrap_uses_actual_positions_and_target_width():
    """A gen-only worker bootstraps absolute positions once and indexes packed
    target rows by their runtime width rather than the configured K+1 width."""
    worker = _make_worker()
    draft_model = _RecordingDraftModel()
    metadata = types.SimpleNamespace(max_num_requests=2)
    worker._lazy_init(draft_model, metadata)

    slots = [worker._assign_slot(1000, reset=False), worker._assign_slot(1001, reset=False)]
    worker._batch_to_slot[:2] = torch.tensor(slots, device="cuda")

    captured = torch.stack(
        [
            torch.full((HIDDEN * NCAP,), 1.0, device="cuda", dtype=torch.bfloat16),
            torch.full((HIDDEN * NCAP,), 2.0, device="cuda", dtype=torch.bfloat16),
        ]
    )
    metadata.get_hidden_states = lambda num_tokens: captured[:num_tokens]
    attn_metadata = types.SimpleNamespace(num_ctx_tokens=0)
    accepted = torch.tensor([[11], [22]], device="cuda", dtype=torch.int32)
    num_accepted = torch.ones(2, device="cuda", dtype=torch.int32)
    position_ids = torch.tensor([[4016, 87]], device="cuda")

    worker._draft_gen_block_batched(
        draft_model,
        metadata,
        attn_metadata,
        accepted,
        num_accepted,
        num_contexts=0,
        batch_size=2,
        total_target_tokens=2,
        position_ids=position_ids,
    )

    first_call = draft_model.forward_calls[-1]
    torch.testing.assert_close(first_call["main_hidden"], captured)
    assert first_call["start_pos"].tolist() == [4017, 88]
    assert worker._ctx_len[slots].tolist() == [4017, 88]
    assert first_call["valid_len"].tolist() == [1, 1]
    assert worker._valid_len[slots].tolist() == [1, 1]
    assert worker._position_initialized[slots].tolist() == [True, True]

    # Existing slots retain their state. The normal K+1 target layout must still
    # select each accepted bonus hidden from its own packed request row.
    target_width = worker.max_draft_len + 1
    captured = torch.stack(
        [
            torch.full((HIDDEN * NCAP,), float(i), device="cuda", dtype=torch.bfloat16)
            for i in range(2 * target_width)
        ]
    )
    metadata.get_hidden_states = lambda num_tokens: captured[:num_tokens]
    accepted = torch.arange(2 * target_width, device="cuda", dtype=torch.int32).reshape(
        2, target_width
    )
    num_accepted = torch.tensor([2, 3], device="cuda", dtype=torch.int32)
    unrelated_positions = torch.zeros((1, 2 * target_width), device="cuda", dtype=torch.long)

    worker._draft_gen_block_batched(
        draft_model,
        metadata,
        attn_metadata,
        accepted,
        num_accepted,
        num_contexts=0,
        batch_size=2,
        total_target_tokens=2 * target_width,
        position_ids=unrelated_positions,
    )

    second_call = draft_model.forward_calls[-1]
    torch.testing.assert_close(second_call["main_hidden"][0], captured[1])
    torch.testing.assert_close(second_call["main_hidden"][1], captured[target_width + 2])
    assert second_call["start_pos"].tolist() == [4019, 91]
    assert second_call["valid_len"].tolist() == [3, 4]
    assert worker._valid_len[slots].tolist() == [3, 4]


def test_forward_mixed_batch_routes_through_base_entries(monkeypatch):
    """Mixed (context + gen) batch: ``forward`` must route acceptance and
    production through the unified ``SpecWorkerBase`` entries, one-hot-fill the
    context requests' draft-prob rows, and assemble
    ``next_draft_tokens = [ctx zeros ; gen argmax]``.

    Spies replace the base sampling entries and the heavy sub-calls (context
    seeding, per-request draft backbone) so this exercises the worker's
    context/gen orchestration — the exact surface the #15775 refactor changed —
    without a real draft model or MPI.
    """
    worker = _make_worker()
    worker.guided_decoder = None
    dm = _fake_draft_model(num_stages=3, window_size=128, head_dim=64)

    K = worker.max_draft_len
    vocab = 16
    num_contexts, num_gens = 2, 3
    batch_size = num_contexts + num_gens

    meta = _make_metadata(max_num_requests=8)
    meta.request_ids = [10, 11, 20, 21, 22]  # 2 context + 3 gen
    meta.prepare()

    attn_metadata = types.SimpleNamespace(
        num_seqs=batch_size,
        num_contexts=num_contexts,
        num_ctx_tokens=0,
        num_tokens=batch_size,
    )

    # Acceptance: return a fixed verified prefix (one accepted token per request).
    accepted = torch.arange(batch_size * (K + 1), dtype=torch.int32, device="cuda").reshape(
        batch_size, K + 1
    )
    num_accepted = torch.ones(batch_size, dtype=torch.int32, device="cuda")
    accept_calls = {}

    def fake_accept(logits, am, sm):
        accept_calls["args"] = (am, sm)
        return accepted, num_accepted

    monkeypatch.setattr(worker, "sample_and_accept_draft_tokens", fake_accept)
    # Context-window seeding is covered by its own test; stub it out here.
    monkeypatch.setattr(worker, "_seed_context_windows", lambda *a, **k: None)

    # The gen-block helper now returns the corrected block logits [num_gens,K,vocab].
    gen_logits = torch.randn(num_gens, K, vocab, device="cuda")
    monkeypatch.setattr(worker, "_draft_gen_block_batched", lambda *a, **k: gen_logits)

    sdt_calls = {}
    # The gen scatter publishes the FULL (post-TP-gather) vocab width, which is
    # wider than the sharded gen_logits width (`vocab`). The worker must pass this
    # published width (draft_probs_last_dim) to write_context_onehot_draft_probs,
    # NOT gen_logits.shape[-1].
    FULL_VOCAB = 97

    def fake_sample_draft_tokens(gl, sm, bs, *, num_contexts):
        sdt_calls["logits"] = gl
        sdt_calls["batch_size"] = bs
        sdt_calls["num_contexts"] = num_contexts
        sm.draft_probs_last_dim = FULL_VOCAB  # simulate the full-vocab scatter
        return gl.argmax(dim=-1).to(torch.int32)

    monkeypatch.setattr(worker, "sample_draft_tokens", fake_sample_draft_tokens)

    onehot_calls = {}
    monkeypatch.setattr(
        worker,
        "write_context_onehot_draft_probs",
        lambda sm, nc, ng, k, gv: onehot_calls.update(nc=nc, ng=ng, k=k, gv=gv),
    )

    input_ids = torch.zeros(batch_size, dtype=torch.long, device="cuda")
    position_ids = torch.zeros(batch_size, dtype=torch.long, device="cuda")
    hidden = torch.zeros(batch_size, HIDDEN, device="cuda", dtype=torch.bfloat16)
    logits = torch.zeros(batch_size, vocab, device="cuda")

    out = worker.forward(input_ids, position_ids, hidden, logits, attn_metadata, meta, dm)

    # Acceptance went through the unified entry with the right metadata objects.
    assert accept_calls["args"] == (attn_metadata, meta)
    # Production fed the [num_gens, K, vocab] block logits to the base sampler,
    # with num_contexts so it slices the gen segment.
    assert sdt_calls["num_contexts"] == num_contexts
    assert sdt_calls["logits"].shape == (num_gens, K, vocab)
    # Context rows one-hot-filled with the *scatter* width (draft_probs_last_dim,
    # FULL_VOCAB), not the sharded gen_logits width (vocab).
    assert onehot_calls == {"nc": num_contexts, "ng": num_gens, "k": K, "gv": FULL_VOCAB}

    # next_draft_tokens = [context zeros ; gen argmax]; gen subset is not polluted
    # by the context rows.
    nd = out["next_draft_tokens"]
    assert nd.shape == (batch_size, K)
    assert torch.all(nd[:num_contexts] == 0)
    assert torch.equal(nd[num_contexts:], gen_logits.argmax(dim=-1).to(torch.int32))
    # Verified tokens are surfaced unchanged.
    assert torch.equal(out["new_tokens"], accepted)
    assert torch.equal(out["new_tokens_lens"], num_accepted)


def test_forward_guided_batch_masks_and_advances_matcher_per_step(monkeypatch):
    """Guided mixed (context + gen) batch: ``forward`` must walk the K draft
    positions left-to-right, advancing the grammar matcher
    (``add_draft_batch``) and masking each position's logits in place
    (``execute_draft_batch``) BEFORE that position is sampled.

    This is the guided sibling of
    ``test_forward_mixed_batch_routes_through_base_entries``; it pins the
    parts static reading can't settle — the step-major contiguous layout the
    bitmask kernel requires, the zero-padded context rows, the seed token fed
    to the matcher, and mask-before-sample ordering — with a fake guided
    decoder (no grammar backend / draft model / MPI).
    """
    worker = _make_worker()
    dm = _fake_draft_model(num_stages=3, window_size=128, head_dim=64)

    K = worker.max_draft_len
    vocab = 16
    num_contexts, num_gens = 2, 3
    batch_size = num_contexts + num_gens
    BANNED = 3  # grammar forbids this draft-vocab column
    FULL_VOCAB = 97  # post-TP-gather scatter width (wider than sharded `vocab`)

    meta = _make_metadata(max_num_requests=8)
    meta.request_ids = [10, 11, 20, 21, 22]  # 2 context + 3 gen
    meta.prepare()

    attn_metadata = types.SimpleNamespace(
        num_seqs=batch_size,
        num_contexts=num_contexts,
        num_ctx_tokens=0,
        num_tokens=batch_size,
    )

    # Acceptance: one accepted token per request (num_accepted == 1), so the
    # matcher seed is accepted_tokens[:, 0].
    accepted = torch.arange(batch_size * (K + 1), dtype=torch.int32, device="cuda").reshape(
        batch_size, K + 1
    )
    num_accepted = torch.ones(batch_size, dtype=torch.int32, device="cuda")
    monkeypatch.setattr(
        worker, "sample_and_accept_draft_tokens", lambda *a, **k: (accepted, num_accepted)
    )
    monkeypatch.setattr(worker, "_seed_context_windows", lambda *a, **k: None)

    gen_logits = torch.randn(num_gens, K, vocab, device="cuda")
    monkeypatch.setattr(worker, "_draft_gen_block_batched", lambda *a, **k: gen_logits)

    class FakeGuidedDecoder:
        def __init__(self):
            self.add_steps = []
            self.add_tokens = []
            self.exec_records = []

        def execute(self, logits, d2t=None):
            # Target-verify masking (``_execute_guided_decoder_if_present`` in
            # ``_forward_impl``); a no-op here — this test asserts only the
            # draft-loop behavior, not target-side masking.
            pass

        def add_draft_batch(self, new_tokens, num_accepted_tokens, draft_step=0):
            self.add_steps.append(draft_step)
            self.add_tokens.append(new_tokens.clone())

        def execute_draft_batch(self, logits, draft_step=0):
            # Record the exact tensor properties the bitmask kernel requires,
            # BEFORE mutating: contiguous [batch, vocab] with zero-padded
            # context rows.
            self.exec_records.append(
                dict(
                    step=draft_step,
                    contiguous=logits.is_contiguous(),
                    shape=tuple(logits.shape),
                    ctx_sum=float(logits[:num_contexts].abs().sum()),
                )
            )
            # Simulate the grammar bitmask applied in place before sampling.
            logits[:, BANNED] = float("-inf")

    guided = FakeGuidedDecoder()
    worker.guided_decoder = guided

    sampled_per_step = []

    def fake_sample_draft_tokens(gl, sm, bs, *, draft_step):
        # The sampler must see the already-masked logits (mask-before-sample).
        assert torch.all(gl[:, BANNED] == float("-inf"))
        sm.draft_probs_last_dim = FULL_VOCAB  # simulate the full-vocab scatter
        tokens = gl.argmax(dim=-1).to(torch.int32)
        sampled_per_step.append(tokens.clone())
        return tokens

    monkeypatch.setattr(worker, "sample_draft_tokens", fake_sample_draft_tokens)

    onehot_calls = {}
    monkeypatch.setattr(
        worker,
        "write_context_onehot_draft_probs",
        lambda sm, nc, ng, k, gv: onehot_calls.update(nc=nc, ng=ng, k=k, gv=gv),
    )

    input_ids = torch.zeros(batch_size, dtype=torch.long, device="cuda")
    position_ids = torch.zeros(batch_size, dtype=torch.long, device="cuda")
    hidden = torch.zeros(batch_size, HIDDEN, device="cuda", dtype=torch.bfloat16)
    logits = torch.zeros(batch_size, vocab, device="cuda")

    out = worker.forward(input_ids, position_ids, hidden, logits, attn_metadata, meta, dm)

    # Matcher advanced once per draft position, in order 0..K-1.
    assert guided.add_steps == list(range(K))
    assert [r["step"] for r in guided.exec_records] == list(range(K))

    # Each step's logits were the contiguous [batch, vocab] slice the bitmask
    # kernel requires, with context rows zero-padded (they carry no draft).
    for rec in guided.exec_records:
        assert rec["contiguous"]
        assert rec["shape"] == (batch_size, vocab)
        assert rec["ctx_sum"] == 0.0

    # Step 0's matcher seed is the last accepted token (accepted[:, 0], since
    # num_accepted == 1); each later step is fed the previous step's full-batch
    # sample.
    assert torch.equal(guided.add_tokens[0], accepted[:, 0])
    for k in range(1, K):
        assert torch.equal(guided.add_tokens[k], sampled_per_step[k - 1])

    # Context rows are one-hot-filled at the scatter width (draft_probs_last_dim),
    # NOT the sharded gen_logits width.
    assert onehot_calls == {"nc": num_contexts, "ng": num_gens, "k": K, "gv": FULL_VOCAB}

    # Output excludes context rows and matches the per-step masked samples.
    nd = out["next_draft_tokens"]
    assert nd.shape == (batch_size, K)
    assert torch.all(nd[:num_contexts] == 0)
    gen_draft = nd[num_contexts:]
    expected_gen = torch.stack([s[num_contexts:] for s in sampled_per_step], dim=1)
    assert torch.equal(gen_draft, expected_gen)
