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
"""draft-loop bookkeeping versus the native MTP one-model worker.

Unit level: fused acceptance (strict + forced-acceptance override, including
the draw counter) against ``_sample_and_accept_draft_tokens_base``; draft-0
inputs against ``prepare_1st_drafter_inputs``; the row gather and the
post-loop finalize against the native indexing.

Worker level: the real ``MTPEagleWorker.forward`` of the target's adopted worker
(``GlmMtpWorker``) on decode steps -- the native draft loop -- on real DSA metadata,
with draft_loop off and on. A stub MTP layer (consuming the real embedding callable)
records what every draft step sees; outputs, per-step inputs, per-step
metadata and the final metadata state must be identical, eagerly and under
CUDA-graph replay.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

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
    tensor_state,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (  # noqa: E402
    dispatch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.draft_loop import (  # noqa: E402, E501
    draft_loop as draft_loop_mod,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.draft_loop import (  # noqa: E402, E501
    draft_step as draft_step_mod,
)

VOCAB = 1024
HIDDEN = 6144


def _make_worker(force: float):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.spec_worker import (  # noqa: E501
        GlmMtpWorker,
    )
    from tensorrt_llm._torch.speculative.eagle3 import MTPEagleWorker
    from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

    worker = GlmMtpWorker.adopt(MTPEagleWorker(MTPDecodingConfig(max_draft_len=MTP_K)))
    worker.force_num_accepted_tokens = force
    return worker


def _spec_metadata(batch_size, draft_tokens, is_cuda_graph=False):
    return SimpleNamespace(
        runtime_draft_len=MTP_K,
        draft_tokens=draft_tokens,
        batch_indices_cuda=torch.arange(max(batch_size, 8), dtype=torch.int32, device="cuda"),
        is_cuda_graph=is_cuda_graph,
        is_all_greedy_sample=True,
        use_rejection_sampling=False,
        enable_penalty=False,
        wants_advanced_draft_sampling=False,
        all_rank_num_tokens=None,
        subseq_all_rank_num_tokens=None,
        spec_resource_manager=None,
        max_num_requests=batch_size,
        gather_ids=None,
        context_prompt_lookahead_tokens=None,
        is_ragged_verify=False,
    )


def _target_logits(targets: torch.Tensor) -> torch.Tensor:
    """Float logits whose argmax per row is ``targets``."""
    logits = torch.randn((targets.numel(), VOCAB), device="cuda")
    logits[torch.arange(targets.numel(), device="cuda"), targets.long()] = 100.0
    return logits


def _random_case(gen, batch_size):
    """Draft tokens and target tokens with a random leading-match length per row."""
    draft = torch.randint(0, VOCAB, (batch_size, MTP_K), generator=gen, dtype=torch.int32)
    target = torch.randint(0, VOCAB, (batch_size, MTP_K + 1), generator=gen, dtype=torch.int32)
    for b in range(batch_size):
        n = int(torch.randint(0, MTP_K + 1, (1,), generator=gen))
        target[b, :n] = draft[b, :n]
        if n < MTP_K and target[b, n] == draft[b, n]:
            target[b, n] = (draft[b, n] + 1) % VOCAB
    return draft.cuda().reshape(-1), target.cuda()


@pytest.mark.parametrize("force", [0.0, 2.61, 0.5, 3.0, 5.0, 7.0])
@pytest.mark.parametrize("batch_size", [1, 3, 8])
def test_accept_matches_native(force, batch_size):
    native, fused = _make_worker(force), _make_worker(force)
    gen = torch.Generator().manual_seed(int(force * 100) + batch_size)
    md_stub = SimpleNamespace(num_seqs=batch_size, num_contexts=0, num_ctx_tokens=0)
    for call in range(40):
        draft, target = _random_case(gen, batch_size)
        logits = _target_logits(target.reshape(-1))
        spec = _spec_metadata(batch_size, draft)
        acc_n, num_n = native._sample_and_accept_draft_tokens_base(
            logits, draft.reshape(batch_size, MTP_K), 0, batch_size, spec
        )
        ops = draft_loop_mod.DraftLoopOps(fused, md_stub, spec)
        assert draft_loop_mod.DraftLoopOps.ineligibility(fused, md_stub, spec) is None
        acc_w, num_w = ops.accept(logits)
        assert acc_w.dtype == acc_n.dtype and torch.equal(acc_w, acc_n), call
        assert num_w.dtype == num_n.dtype and torch.equal(num_w, num_n), call
        if native._force_accept_rng_counter is not None:
            assert torch.equal(native._force_accept_rng_counter, fused._force_accept_rng_counter)
        # Draft-0 inputs and step-0 rows.
        inputs = native.prepare_1st_drafter_inputs(
            input_ids=torch.zeros((batch_size * (MTP_K + 1),), dtype=torch.int32, device="cuda"),
            position_ids=None,
            hidden_states=None,
            accepted_tokens=acc_n,
            attn_metadata=md_stub,
            spec_metadata=spec,
            draft_model=None,
        )
        assert torch.equal(ops.input_ids0, inputs["input_ids"])
        assert ops.input_ids0.data_ptr() != acc_w.data_ptr()
        rows = (spec.batch_indices_cuda[:batch_size] * (MTP_K + 1)).long() + num_n - 1 + 0
        assert torch.equal(ops.gather_ids, rows)


def test_gather_rows_and_finalize_match_native():
    worker = _make_worker(0.0)
    gen = torch.Generator(device="cuda").manual_seed(3)
    for batch_size in (1, 2, 6):
        hidden = torch.randn((batch_size * (MTP_K + 1), HIDDEN), generator=gen, device="cuda")
        hidden = hidden.to(torch.bfloat16)
        rows = torch.randint(0, hidden.shape[0], (batch_size,), generator=gen, device="cuda")
        assert torch.equal(draft_loop_mod.gather_rows(hidden, rows), hidden[rows])

        draft = [
            torch.randint(0, VOCAB, (batch_size,), generator=gen, device="cuda", dtype=torch.int32)
            for _ in range(MTP_K)
        ]
        accepted = torch.randint(
            0, VOCAB, (batch_size, MTP_K + 1), generator=gen, device="cuda", dtype=torch.int32
        )
        num_acc = torch.randint(
            1, MTP_K + 2, (batch_size,), generator=gen, device="cuda", dtype=torch.int32
        )
        spec = _spec_metadata(batch_size, None)
        ops = draft_loop_mod.DraftLoopOps(
            worker, SimpleNamespace(num_seqs=batch_size, num_contexts=0), spec
        )
        ops.accepted_tokens, ops.num_accepted_tokens = accepted, num_acc
        stacked = torch.stack(draft, dim=1)
        expected_new = worker._prepare_next_new_tokens(
            accepted, stacked, spec.batch_indices_cuda, batch_size, num_acc
        )
        got = ops.finalize(draft)
        assert torch.equal(got, stacked)
        assert torch.equal(ops.next_new_tokens, expected_new)


# ----------------------------------------------------------------------------
# Worker level: the adopted worker's draft loop on decode steps
# ----------------------------------------------------------------------------

_STEP_FIELDS = (
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


class _Recorder:
    """Per-step copies of what the MTP layer sees (static buffers under capture)."""

    def __init__(self):
        self.steps = []
        self.static = None
        self.index = 0

    def record(self, md, input_ids, position_ids, hidden, emb):
        items = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "hidden": hidden,
            "emb": emb,
            "seq_lens_cuda": md.seq_lens_cuda,
        }
        items.update({f: getattr(md, f) for f in _STEP_FIELDS})
        if torch.cuda.is_current_stream_capturing():
            if self.static is None:
                self.static = []
            if self.index == len(self.static):
                self.static.append({k: torch.empty_like(v) for k, v in items.items()})
            for k, v in items.items():
                self.static[self.index][k].copy_(v)
            self.index += 1
        else:
            self.steps.append({k: v.clone() for k, v in items.items()})
            self.steps[-1]["num_tokens"] = md.num_tokens


class _StubHead:
    def __call__(self, hidden, lm_head, attn_metadata, return_context_logits):
        return hidden.float() @ lm_head.weight.float().t()


class _StubMTPLayer(torch.nn.Module):
    """Deterministic draft layer: embeds the ids with the given callable."""

    def __init__(self, recorder):
        super().__init__()
        self.recorder = recorder
        self.shared_head = _StubHead()

    def forward(
        self,
        input_ids,
        position_ids,
        hidden_states,
        embed_tokens,
        attn_metadata,
        all_rank_num_tokens=None,
        spec_metadata=None,
        **kwargs,
    ):
        emb = embed_tokens(input_ids)
        self.recorder.record(attn_metadata, input_ids, position_ids, hidden_states, emb)
        mix = emb.float() + 0.5 * hidden_states.float()
        mix = mix + (position_ids.float() % 97).unsqueeze(-1) * 1e-2
        return mix.to(hidden_states.dtype)


def _draft_model(recorder, seed=11):
    from tensorrt_llm._torch.modules.embedding import Embedding

    gen = torch.Generator(device="cuda").manual_seed(seed)
    embed = Embedding(VOCAB, HIDDEN, dtype=torch.bfloat16).cuda()
    embed.weight.data = torch.randn((VOCAB, HIDDEN), generator=gen, device="cuda").to(
        torch.bfloat16
    )
    lm_head = SimpleNamespace(
        weight=torch.randn((VOCAB, HIDDEN), generator=gen, device="cuda").to(torch.bfloat16)
    )
    return SimpleNamespace(
        mtp_layers=[_StubMTPLayer(recorder)], embed_tokens=embed, lm_head=lm_head
    )


def _forward_inputs(batch_size, past, gen):
    draft, target = _random_case(gen, batch_size)
    n = batch_size * (MTP_K + 1)
    return {
        "input_ids": torch.randint(0, VOCAB, (n,), generator=gen, dtype=torch.int32).cuda(),
        "position_ids": torch.cat(
            [torch.arange(p, p + MTP_K + 1, dtype=torch.int32) for p in past]
        ).cuda()[None],
        "hidden_states": torch.randn((n, HIDDEN), generator=gen).to(torch.bfloat16).cuda(),
        "logits": _target_logits(target.reshape(-1)),
        "draft_tokens": draft,
    }


class _ResourceManagerStub:
    """Resource manager exposing only the separate draft KV cache manager."""

    def __init__(self, draft_manager):
        self.draft_manager = draft_manager

    def get_resource_manager(self, kind):
        from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType

        return self.draft_manager if kind == ResourceManagerType.DRAFT_KV_CACHE_MANAGER else None


def _run_forward(worker, md, spec, draft_model, inputs):
    resource_manager = None
    if md.draft_kv_cache_manager is not None:
        # One-model MTP with a separate draft KV cache, as the GEN runner sets up.
        worker.use_separate_draft_kv_cache = True
        resource_manager = _ResourceManagerStub(md.draft_kv_cache_manager)
    with dispatch.step(dispatch.Phase.DECODE):
        return worker.forward(
            input_ids=inputs["input_ids"],
            position_ids=inputs["position_ids"],
            hidden_states=inputs["hidden_states"],
            logits=inputs["logits"],
            attn_metadata=md,
            spec_metadata=spec,
            draft_model=draft_model,
            resource_manager=resource_manager,
        )


_OUTPUT_KEYS = ("new_tokens", "new_tokens_lens", "next_draft_tokens", "next_new_tokens")


@pytest.mark.parametrize("force", [0.0, 2.61])
@pytest.mark.parametrize("past", [[391061], [391061, 120000, 4100]])
@pytest.mark.parametrize("separate_draft", [False, True], ids=["shared_kv", "draft_kv"])
def test_worker_forward_matches_native(monkeypatch, force, past, separate_draft):
    batch_size = len(past)
    cache_manager, mapping, sparse = make_dsa_cache_manager(batch_size, max(past))
    draft_manager = None
    calls = {"accept": 0, "step": 0}
    real_accept, real_step = draft_loop_mod.DraftLoopOps.accept, draft_step_mod.draft_step_update

    def spy_accept(self, logits):
        calls["accept"] += 1
        return real_accept(self, logits)

    def spy_step(*args, **kwargs):
        calls["step"] += 1
        return real_step(*args, **kwargs)

    monkeypatch.setattr(draft_loop_mod.DraftLoopOps, "accept", spy_accept)
    monkeypatch.setattr(draft_step_mod, "draft_step_update", spy_step)
    try:
        ids = list(range(batch_size))
        add_gen_requests(cache_manager, ids, past)
        if separate_draft:
            draft_manager, _, _ = make_dsa_cache_manager(batch_size, max(past))
            add_gen_requests(draft_manager, ids, past)
        gen = torch.Generator().manual_seed(len(past) + int(force * 10))
        for it in range(3):
            inputs = _forward_inputs(batch_size, past, gen)
            results = {}
            for name, features in (
                ("native", frozenset()),
                ("draft_loop", frozenset({"draft_loop"})),
            ):
                monkeypatch.setattr(dispatch, "enabled_features", lambda f=features: f)
                recorder = _Recorder()
                worker = _make_worker(force)
                md = make_gen_dsa_metadata(
                    cache_manager, mapping, sparse, ids, past, draft_cache_manager=draft_manager
                )
                base = tensor_state(md)
                spec = _spec_metadata(batch_size, inputs["draft_tokens"].clone())
                out = _run_forward(worker, md, spec, _draft_model(recorder), inputs)
                torch.cuda.synchronize()
                results[name] = (out, recorder.steps, base, tensor_state(md), worker)
            assert calls["accept"] == it + 1 and calls["step"] == (it + 1) * MTP_K
            (o_n, s_n, b_n, f_n, w_n), (o_w, s_w, b_w, f_w, w_w) = (
                results["native"],
                results["draft_loop"],
            )
            for k in _OUTPUT_KEYS:
                assert o_n[k].dtype == o_w[k].dtype and torch.equal(o_n[k], o_w[k]), (it, k)
            assert len(s_n) == len(s_w) == MTP_K
            for i, (a, b) in enumerate(zip(s_n, s_w)):
                for k in a:
                    if isinstance(a[k], torch.Tensor):
                        assert a[k].dtype == b[k].dtype and torch.equal(a[k], b[k]), (it, i, k)
                    else:
                        assert a[k] == b[k], (it, i, k)
            keys = changed_keys(b_n, f_n) | changed_keys(b_w, f_w)
            assert_same_state(f_n, f_w, keys, f"iteration {it} final metadata")
            if w_n._force_accept_rng_counter is not None:
                assert torch.equal(w_n._force_accept_rng_counter, w_w._force_accept_rng_counter)
    finally:
        cache_manager.shutdown()
        if draft_manager is not None:
            draft_manager.shutdown()


def _capture_and_replay(
    monkeypatch,
    features,
    cache_manager,
    mapping,
    sparse,
    ids,
    past,
    replay_inputs,
    between=None,
    keep_graph=False,
    draft_manager=None,
):
    """Capture one worker forward for ``features``, replay it per input set.

    Returns per-replay clones of the outputs, the per-step records and the
    draw counter. ``between`` runs after capture and before the replays.
    """
    batch_size = len(ids)
    monkeypatch.setattr(dispatch, "enabled_features", lambda f=features: f)
    recorder = _Recorder()
    worker = _make_worker(2.61)
    md = make_gen_dsa_metadata(
        cache_manager, mapping, sparse, ids, past, draft_cache_manager=draft_manager
    )
    kv_init = md.kv_lens_cuda[:batch_size].clone()
    static = {k: v.clone() for k, v in replay_inputs[0].items()}
    spec = _spec_metadata(batch_size, static["draft_tokens"], is_cuda_graph=True)
    draft_model = _draft_model(recorder)
    host_seq = md._seq_lens.clone()
    # The forced-acceptance pool is built eagerly, never under capture.
    worker._ensure_force_accept_rng_state(torch.device("cuda"))
    for _ in range(2):  # eager warmup (forced acceptance is skipped here)
        md.kv_lens_cuda[:batch_size].copy_(kv_init)
        _run_forward(worker, md, spec, draft_model, static)
        md._seq_lens.copy_(host_seq)
        md.num_contexts = 0
        md.on_update()
    md.kv_lens_cuda[:batch_size].copy_(kv_init)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _run_forward(worker, md, spec, draft_model, static)
    md._seq_lens.copy_(host_seq)
    if between is not None:
        between()
    results = []
    for replay, inputs in enumerate(replay_inputs):
        for k, v in inputs.items():
            static[k].copy_(v)
        md.kv_lens_cuda[:batch_size].copy_(kv_init)
        graph.replay()
        try:
            torch.cuda.synchronize()
        except (RuntimeError, torch.AcceleratorError) as err:
            raise AssertionError(f"{sorted(features)} graph replay {replay} faulted") from err
        results.append(
            (
                {k: out[k].clone() for k in _OUTPUT_KEYS},
                [{k: v.clone() for k, v in step.items()} for step in recorder.static],
                worker._force_accept_rng_counter.clone(),
            )
        )
    if keep_graph:
        return results, graph, md, static
    del graph
    torch.cuda.synchronize()
    return results


def _graph_setup():
    past = [391061, 200000]
    cache_manager, mapping, sparse = make_dsa_cache_manager(len(past), max(past) + 64)
    ids = list(range(len(past)))
    add_gen_requests(cache_manager, ids, past)
    gen = torch.Generator().manual_seed(21)
    replay_inputs = [_forward_inputs(len(past), past, gen) for _ in range(10)]
    return cache_manager, mapping, sparse, ids, past, replay_inputs


@pytest.mark.parametrize(
    "scenario",
    [
        "native",
        "native_then_self_check",
        "draft_loop",
        "native_then_new_metadata",
        "native_then_native_eager",
        "native_then_draft_loop_eager",
        "native_then_native_capture",
        "native_then_draft_loop_capture",
    ],
)
def test_graph_isolation_diagnostics(monkeypatch, scenario):
    """A captured graph replays cleanly after other work (fault localization)."""
    cache_manager, mapping, sparse, ids, past, replay_inputs = _graph_setup()
    try:
        between = None
        if scenario.startswith("native_then_") and scenario != "native_then_self_check":
            eager_features = {
                "native_then_native_eager": frozenset(),
                "native_then_draft_loop_eager": frozenset({"draft_loop"}),
                "native_then_native_capture": frozenset(),
                "native_then_draft_loop_capture": frozenset({"draft_loop"}),
            }.get(scenario)
            second_graphs = []

            def between():
                if scenario.endswith("_capture"):
                    # A second captured graph that stays alive during the replays.
                    second_graphs.append(
                        _capture_and_replay(
                            monkeypatch,
                            eager_features,
                            cache_manager,
                            mapping,
                            sparse,
                            ids,
                            past,
                            replay_inputs[:1],
                            keep_graph=True,
                        )
                    )
                    monkeypatch.setattr(dispatch, "enabled_features", lambda: frozenset())
                    return
                md2 = make_gen_dsa_metadata(cache_manager, mapping, sparse, ids, past)
                if eager_features is None:
                    return
                monkeypatch.setattr(dispatch, "enabled_features", lambda: eager_features)
                worker2 = _make_worker(2.61)
                spec2 = _spec_metadata(len(ids), replay_inputs[0]["draft_tokens"].clone())
                for _ in range(2):
                    _run_forward(worker2, md2, spec2, _draft_model(_Recorder()), replay_inputs[0])
                torch.cuda.synchronize()
                monkeypatch.setattr(dispatch, "enabled_features", lambda: frozenset())

        if scenario == "native_then_self_check":

            def between():
                draft_step_mod._SPLIT_KV_BY_SMS.clear()
                assert (
                    draft_step_mod.schedule_self_check(
                        torch.cuda.get_device_properties(0).multi_processor_count,
                        torch.device("cuda"),
                    )
                    is not None
                )

        features = frozenset({"draft_loop"}) if scenario == "draft_loop" else frozenset()
        results = _capture_and_replay(
            monkeypatch,
            features,
            cache_manager,
            mapping,
            sparse,
            ids,
            past,
            replay_inputs[:3],
            between,
        )
        assert len(results) == 3
    finally:
        cache_manager.shutdown()


@pytest.mark.parametrize("separate_draft", [False, True], ids=["shared_kv", "draft_kv"])
def test_worker_forward_cuda_graph(monkeypatch, separate_draft):
    """Captured native and draft_loop forwards replay identically (forced acceptance 2.61).

    Outputs, per-step records and draw counters are compared over 10 replays
    with fresh inputs.
    """
    # The draft_loop schedule self-check runs eagerly before any capture, as in a
    # serving warmup.
    assert draft_step_mod.schedule_self_check(
        torch.cuda.get_device_properties(0).multi_processor_count, torch.device("cuda")
    )
    cache_manager, mapping, sparse, ids, past, replay_inputs = _graph_setup()
    draft_manager = None
    try:
        if separate_draft:
            draft_manager, _, _ = make_dsa_cache_manager(len(past), max(past) + 64)
            add_gen_requests(draft_manager, ids, past)
        native = _capture_and_replay(
            monkeypatch,
            frozenset(),
            cache_manager,
            mapping,
            sparse,
            ids,
            past,
            replay_inputs,
            draft_manager=draft_manager,
        )
        fused = _capture_and_replay(
            monkeypatch,
            frozenset({"draft_loop"}),
            cache_manager,
            mapping,
            sparse,
            ids,
            past,
            replay_inputs,
            draft_manager=draft_manager,
        )
        for replay, ((o_n, r_n, c_n), (o_w, r_w, c_w)) in enumerate(zip(native, fused)):
            for k in _OUTPUT_KEYS:
                assert torch.equal(o_n[k], o_w[k]), (replay, k)
            assert len(r_n) == len(r_w) == MTP_K
            for i in range(MTP_K):
                for k in r_n[i]:
                    assert torch.equal(r_n[i][k], r_w[i][k]), (replay, i, k)
            assert torch.equal(c_n, c_w), replay
    finally:
        cache_manager.shutdown()
        if draft_manager is not None:
            draft_manager.shutdown()
