# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TorchSampler host-path fast paths for the RL rollout configuration (one token and one beam per
request, logprobs and stop words on): the batched update_requests, the single-step batch metadata /
indexer / grouping shortcuts, the steady decode-layout cache (_SteadyDecodeLayout) and the O(window)
stop-word admission must reproduce the generic per-request paths exactly."""
import pytest
import torch

import tensorrt_llm._torch.pyexecutor.sampler.sampler as sampler_mod
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.sampler.finish_reasons import FinishReasonsHandler
from tensorrt_llm._torch.pyexecutor.sampler.sampler import TorchSampler
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import SamplingConfig
from tensorrt_llm.sampling_params import LogprobMode, SamplingParams

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
VOCAB, STOP = 64, 7


def _params(kind, num_logprobs, mode):
    common = dict(logprobs=num_logprobs, logprobs_simple_format=num_logprobs == 0, logprobs_mode=mode,
                  stop_token_ids=[STOP])
    if kind == "greedy":
        return SamplingParams(top_k=1, **common)
    if kind == "temperature":
        return SamplingParams(temperature=1.0, top_p=1.0, top_k=0, **common)
    if kind == "topk":
        return SamplingParams(temperature=0.8, top_k=8, **common)
    raise ValueError(kind)


def _request(i, kinds, num_logprobs, mode, *, seq_slot=None, with_logprobs=True):
    sp = _params(kinds[i % len(kinds)], num_logprobs, mode)
    return LlmRequest(
        request_id=i, max_new_tokens=32, input_tokens=[1, 2, 3, 4 + i % 5],
        sampling_config=SamplingConfig(sp._get_sampling_config()), seq_slot=i if seq_slot is None else seq_slot,
        is_streaming=False, return_log_probs=with_logprobs, num_logprobs=num_logprobs if with_logprobs else None,
        logprobs_simple_format=num_logprobs == 0, logprobs_mode=mode, stop_words_list=[[STOP]],
    )


def _sampler(n_slots, *, single_step_fastpath=None, steady_layout=None):
    s = TorchSampler(TorchSampler.Args(max_seq_len=256, max_draft_len=0, max_beam_width=1, max_num_sequences=n_slots,
                                       max_total_draft_tokens=0, disable_overlap_scheduler=False))
    if single_step_fastpath is not None:
        s._single_step_fastpath = single_step_fastpath
    if steady_layout is not None:
        s._steady_layout_enabled = steady_layout
    return s


def _scheduled(gen, ctx=()):
    s = ScheduledRequests()
    s.context_requests_chunking = []
    s.context_requests_last_chunk = list(ctx)
    s.generation_requests = list(gen)
    return s


def _snapshot(reqs):
    rows = []
    for r in reqs:
        lp = r.py_result.log_probs
        lp = lp() if callable(lp) else lp
        rows.append((r.py_request_id, tuple(r.get_tokens(0)), r.is_finished, r.py_decoding_iter, repr(lp)))
    return rows


CASES = {
    "temperature-raw-top1": (("temperature",), 1, LogprobMode.RAW),
    "greedy-simple": (("greedy",), 0, LogprobMode.RAW),
    "mixed-strategies-processed": (("greedy", "temperature", "topk"), 1, LogprobMode.PROCESSED),
    "temperature-raw-top2-partial": (("temperature",), 2, LogprobMode.RAW),
    "temperature-raw-top2": (("temperature",), 2, LogprobMode.RAW),
    "mixed-strategies-raw-simple": (("temperature", "greedy"), 0, LogprobMode.RAW),
    "mixed-strategies-processed-partial": (("greedy", "temperature", "topk"), 1, LogprobMode.PROCESSED),
}


# ---------------------------------------------------------------------------------------------
# Single-step fast paths: mixed (context + decode) batches whose request set changes every step,
# so the steady-layout cache never applies.
# ---------------------------------------------------------------------------------------------


def _run_single_step(fastpath, *, kinds, num_logprobs, mode, steps, n_gen, n_ctx_per_step, partial_logprobs=False):
    n_slots = n_gen + n_ctx_per_step * steps + 2
    sampler = _sampler(n_slots, single_step_fastpath=fastpath, steady_layout=False)
    gen = [_request(i, kinds, num_logprobs, mode, seq_slot=i, with_logprobs=not (partial_logprobs and i % 3 == 0))
           for i in range(n_gen)]
    ctx_all = [_request(1000 + j, kinds, num_logprobs, mode, seq_slot=n_gen + j) for j in range(n_ctx_per_step * steps)]
    with torch.inference_mode():
        # the initial decode requests enter through the context list: that is what setup registers
        sampler.setup_sampler_step(_scheduled([], gen))
    snapshots = []
    for step in range(steps):
        ctx = ctx_all[step * n_ctx_per_step:(step + 1) * n_ctx_per_step]
        sched = _scheduled(gen, ctx)
        g = torch.Generator(device="cuda").manual_seed(100 + step)
        logits = torch.randn(len(ctx) + len(gen), VOCAB, device="cuda", generator=g) * 3
        logits[:, STOP] = -50.0
        with torch.inference_mode():
            sampler.setup_sampler_step(sched)
            state = sampler.sample_async(sched, model_outputs={"logits": logits},
                                         num_context_logits_prefix_sum=[0] + [k + 1 for k in range(len(ctx))],
                                         resource_manager=None)
            sampler.update_requests(state, resource_manager=None)
        torch.cuda.synchronize()
        snapshots.append(_snapshot(ctx + gen))
        gen = gen + ctx  # the context requests decode with the others from now on
    return snapshots


SINGLE_STEP_CASES = ["temperature-raw-top1", "greedy-simple", "mixed-strategies-processed",
                     "temperature-raw-top2-partial", "mixed-strategies-processed-partial"]


@pytest.mark.parametrize("case", SINGLE_STEP_CASES)
def test_single_step_fast_paths_match_generic_path(case):
    kinds, num_logprobs, mode = CASES[case]
    kw = dict(kinds=kinds, num_logprobs=num_logprobs, mode=mode, steps=4, n_gen=6, n_ctx_per_step=2,
              partial_logprobs=case.endswith("partial"))
    assert _run_single_step(True, **kw) == _run_single_step(False, **kw)


def test_single_step_metadata_matches_generic_metadata():
    sampler = _sampler(16, single_step_fastpath=True, steady_layout=False)
    gen = [_request(i, ("temperature",), 1, LogprobMode.RAW, seq_slot=i) for i in range(5)]
    ctx = [_request(100 + j, ("temperature",), 1, LogprobMode.RAW, seq_slot=8 + j) for j in range(2)]
    sched = _scheduled(gen, ctx)
    logits = torch.randn(7, VOCAB, device="cuda")
    prefix = [0, 1, 2]
    reqs_ref, md_ref, logits_ref = TorchSampler._select_generated_logits(
        sched, logits, num_context_logits_prefix_sum=prefix)
    md_fast = sampler._single_step_metadata(7)
    reqs_got, md_got, logits_got = TorchSampler._select_generated_logits(
        sched, logits, num_context_logits_prefix_sum=prefix, single_step_metadata=md_fast)
    assert reqs_got == reqs_ref and md_got is md_fast
    for name in ("req_num_generated_tokens", "req_num_generated_tokens_output", "req_num_beams", "req_num_steps",
                 "req_offsets"):
        a, b = getattr(md_got, name), getattr(md_ref, name)
        assert torch.equal(a, b) and a.dtype == b.dtype, name
    assert logits_got.data_ptr() == logits_ref.data_ptr() and logits_got.shape == logits_ref.shape
    assert sampler._single_step_metadata(sampler._ones_host.numel()) is None


def test_single_strategy_group_matches_generic_grouping():
    """The grouper's single-strategy shortcut must emit the same group (and raw-logprobs mask) as the
    general sort-based grouping, for uniform and for partially-logprob batches."""
    from tensorrt_llm._torch.pyexecutor.sampler.sampler_strategy import (
        FlashInferGroupedStrategySampler,
        _CachingRequestGrouper,
    )
    reqs = [_request(i, ("temperature",), 1, LogprobMode.RAW, seq_slot=i, with_logprobs=i % 2 == 0) for i in range(6)]
    seq_slots = torch.tensor([r.py_seq_slot for r in reqs], dtype=torch.int32)

    def group(grouper):
        for r in reqs:
            grouper.prepare_for_new_request(r, r.py_seq_slot)
        return grouper.group_requests_by_strategy_key(
            reqs, strategy_to_key=FlashInferGroupedStrategySampler.strategy_grouping_key, pin_memory=False,
            seq_slots=seq_slots, vocab_size=VOCAB)

    fast, fast_raw = group(_CachingRequestGrouper(8))
    # force the general path by making the strategies look different on the first call
    grouper = _CachingRequestGrouper(8)
    for r in reqs:
        grouper.prepare_for_new_request(r, r.py_seq_slot)
    generic, generic_raw = grouper.group_requests_by_strategy_key(
        reqs, strategy_to_key=FlashInferGroupedStrategySampler.strategy_grouping_key, pin_memory=False,
        seq_slots=seq_slots, vocab_size=VOCAB)
    assert list(fast) == list(generic) and len(fast) == 1
    (fv,), (gv,) = fast.values(), generic.values()
    assert torch.equal(fv.indices, gv.indices) and fv.indices.dtype == gv.indices.dtype
    assert fv.strategies == gv.strategies
    assert torch.equal(fv.speculation_needs_probs_indices, gv.speculation_needs_probs_indices)
    assert torch.equal(fv.need_processed_logprobs, gv.need_processed_logprobs)
    assert torch.equal(fast_raw, generic_raw) and fast_raw.tolist() == [True, False, True, False, True, False]


# ---------------------------------------------------------------------------------------------
# Steady decode-layout cache: consecutive single-step decode iterations over an unchanged request
# set reuse the recorded layout; any change of the batch (or of its sequence slots) records anew.
# ---------------------------------------------------------------------------------------------


def _step_logits(step, n, stop_row):
    g = torch.Generator(device="cuda").manual_seed(1000 + step)
    logits = torch.randn(n, VOCAB, device="cuda", generator=g) * 3
    logits[:, STOP] = -50.0
    if stop_row is not None:
        logits[stop_row, STOP] = 50.0
    return logits


def _run_schedule(enabled, monkeypatch, *, kinds, num_logprobs, mode, schedule):
    """schedule: list of steps; each step is (gen_request_ids, stop_row or None, new_request_ids[, reslot]).
    Requests are created lazily and kept alive across steps like the executor does. ``reslot``
    ({request_id: new_slot}) re-assigns sequence slots before the step, as the slot manager does
    for a request that was paused (KV pressure, recompute after a weight update) and comes back
    through a context step."""
    sampler = _sampler(16, steady_layout=enabled)
    pool = {}
    select_calls = []
    orig_select = TorchSampler._select_generated_logits

    def spy_select(*args, **kwargs):
        select_calls.append(1)
        return orig_select(*args, **kwargs)

    snapshots = []
    with monkeypatch.context() as p:
        p.setattr(TorchSampler, "_select_generated_logits", staticmethod(spy_select))
        p.setattr(TorchSampler, "_collect_new_requests_for_setup", lambda self, r: r.all_requests())
        for step, spec in enumerate(schedule):
            ids, stop_row, ctx_ids = spec[:3]
            reslot = spec[3] if len(spec) > 3 else None
            for i in list(ids) + list(ctx_ids):
                if i not in pool:
                    pool[i] = _request(i, kinds, num_logprobs, mode)
            for i, slot in (reslot or {}).items():
                pool[i].seq_slot = slot
                pool[i].py_seq_slot = slot
            ctx = [pool[i] for i in ctx_ids]
            gen = [pool[i] for i in ids]
            sched = _scheduled(gen, ctx)
            # context requests come first in the logits (one logit each, last chunk)
            logits = _step_logits(step, len(ctx) + len(gen), stop_row)
            with torch.inference_mode():
                sampler.setup_sampler_step(sched)
                state = sampler.sample_async(sched, model_outputs={"logits": logits},
                                             num_context_logits_prefix_sum=[0] + [k + 1 for k in range(len(ctx))],
                                             resource_manager=None)
                sampler.update_requests(state, resource_manager=None)
            torch.cuda.synchronize()
            snapshots.append(_snapshot(ctx + gen))
    return snapshots, len(select_calls), sampler


STEADY = [((0, 1, 2, 3, 4), None, ()), ((0, 1, 2, 3, 4), None, ()), ((0, 1, 2, 3, 4), None, ()),
          ((0, 1, 2, 3, 4), None, ())]
# request 2 samples the stop word at step 2 and is dropped from the batch afterwards
STOP_AND_SHRINK = [((0, 1, 2, 3, 4), None, ()), ((0, 1, 2, 3, 4), None, ()), ((0, 1, 2, 3, 4), 2, ()),
                   ((0, 1, 3, 4), None, ()), ((0, 1, 3, 4), None, ())]
# a context request joins at step 2 (mixed batch: no cache), then decodes with the others
GROW_WITH_CONTEXT = [((0, 1, 2), None, ()), ((0, 1, 2), None, ()), ((0, 1, 2), None, (5,)),
                     ((0, 1, 2, 5), None, ()), ((0, 1, 2, 5), None, ()), ((0, 1, 2, 5), None, ())]
# the same request set in a different order is a different layout
REORDER = [((0, 1, 2, 3), None, ()), ((0, 1, 2, 3), None, ()), ((3, 2, 1, 0), None, ()), ((3, 2, 1, 0), None, ())]
# the batch is paused and comes back through a context step (re-prefill) with the SAME request
# tuple but new sequence slots (pause() resets the slot, the slot manager hands out free ones):
# the layout recorded before the pause must not be reused for the decode steps after it
RESLOT_AFTER_CONTEXT = [((0, 1, 2, 3), None, ()), ((0, 1, 2, 3), None, ()),
                        ((), None, (0, 1, 2, 3), {0: 9, 1: 8, 2: 11, 3: 10}),
                        ((0, 1, 2, 3), None, ()), ((0, 1, 2, 3), None, ())]
SCHEDULES = {"STEADY": STEADY, "STOP_AND_SHRINK": STOP_AND_SHRINK, "GROW_WITH_CONTEXT": GROW_WITH_CONTEXT,
             "REORDER": REORDER}
STEADY_CASES = ["greedy-simple", "temperature-raw-top2", "mixed-strategies-processed", "mixed-strategies-raw-simple",
                "temperature-raw-top1"]


@pytest.mark.parametrize("case", STEADY_CASES)
@pytest.mark.parametrize("schedule_name", list(SCHEDULES))
def test_layout_cache_matches_generic_path(monkeypatch, case, schedule_name):
    kinds, num_logprobs, mode = CASES[case]
    schedule = SCHEDULES[schedule_name]
    ref, ref_selects, _ = _run_schedule(False, monkeypatch, kinds=kinds, num_logprobs=num_logprobs, mode=mode,
                                        schedule=schedule)
    out, out_selects, sampler = _run_schedule(True, monkeypatch, kinds=kinds, num_logprobs=num_logprobs,
                                              mode=mode, schedule=schedule)
    assert out == ref  # tokens, finish state, decoding iter and logprobs per step
    # the generic path derives the layout every step; the cache only on layout changes
    assert ref_selects == len(schedule)
    mixed_steps = sum(1 for step in schedule if step[2])
    steady_layout_changes = len({step[0] for step in schedule if not step[2]})
    # every mixed step goes through the generic path; each new pure-decode layout records once
    # (consecutive identical layouts share one record; the schedules above never revisit a layout)
    assert out_selects == mixed_steps + steady_layout_changes, (out_selects, mixed_steps, steady_layout_changes)
    assert sampler._steady_layout is not None and sampler._steady_layout.metadata is not None


@pytest.mark.parametrize("case", ["greedy-simple", "mixed-strategies-processed"])
def test_slot_reassignment_after_context_records_new_layout(monkeypatch, case):
    kinds, num_logprobs, mode = CASES[case]
    ref, _, _ = _run_schedule(False, monkeypatch, kinds=kinds, num_logprobs=num_logprobs, mode=mode,
                              schedule=RESLOT_AFTER_CONTEXT)
    out, out_selects, sampler = _run_schedule(True, monkeypatch, kinds=kinds, num_logprobs=num_logprobs,
                                              mode=mode, schedule=RESLOT_AFTER_CONTEXT)
    assert out == ref  # the tokens land in the requests, not in the slots the old layout remembered
    # step 1 records, step 2 reuses, the context step is generic, step 4 records a new layout
    # (same ids, new slots), step 5 reuses it
    assert out_selects == 3
    layout = sampler._steady_layout
    assert layout.signature == ((0, 9), (1, 8), (2, 11), (3, 10))
    assert layout.seq_slots_host.tolist() == [9, 8, 11, 10]


def test_stop_word_finish_and_lengths_advance(monkeypatch):
    kinds, num_logprobs, mode = CASES["greedy-simple"]
    out, _, sampler = _run_schedule(True, monkeypatch, kinds=kinds, num_logprobs=num_logprobs, mode=mode,
                                    schedule=STOP_AND_SHRINK)
    step2 = {row[0]: row for row in out[2]}
    assert step2[2][2] is True and step2[2][1][-1] == STOP  # request 2 finished on the stop word
    assert all(not step2[i][2] for i in (0, 1, 3, 4))
    layout = sampler._steady_layout
    assert layout.signature == ((0, 0), (1, 1), (3, 3), (4, 4))
    # the sampler sees each request's length before the current step's token is appended:
    # 4 prompt tokens + 4 tokens committed by the first four steps
    assert layout.seq_lens_host.tolist() == [8, 8, 8, 8]
    assert layout.seq_lens_cuda.tolist() == [8, 8, 8, 8]
    assert layout.stop_words_prep is not None and layout.stop_words_prep[0] == 1
    # the recorded layout carries the grouping, logprob and scatter caches of the generic path
    assert layout.grouped is not None and layout.batch_req_indices is not None
    assert layout.batch_dest_indices_1d_cuda is not None and layout.logprobs_index is not None


def test_layout_disabled_by_env_records_nothing(monkeypatch):
    kinds, num_logprobs, mode = CASES["greedy-simple"]
    _, selects, sampler = _run_schedule(False, monkeypatch, kinds=kinds, num_logprobs=num_logprobs, mode=mode,
                                        schedule=STEADY)
    assert selects == len(STEADY) and sampler._steady_layout is None


# ---------------------------------------------------------------------------------------------
# update_requests: the batched fast path against the generic per-request loop.
# ---------------------------------------------------------------------------------------------

N_UPDATE = 5


def _requests_for_update(*, simple_format, num_logprobs, stop_words):
    sp = SamplingParams(top_k=1, logprobs=num_logprobs, logprobs_simple_format=simple_format,
                        stop_token_ids=[STOP] if stop_words else None)
    return [
        LlmRequest(
            request_id=i, max_new_tokens=8, input_tokens=[1, 2, 3],
            sampling_config=SamplingConfig(sp._get_sampling_config()), seq_slot=i, is_streaming=False,
            return_log_probs=True, num_logprobs=num_logprobs, logprobs_simple_format=simple_format,
            stop_words_list=[[STOP]] if stop_words else None,  # one single-token stop word
        )
        for i in range(N_UPDATE)
    ]


def _run_update_requests(fastpath, monkeypatch, *, simple_format, num_logprobs, stop_words):
    sampler = TorchSampler(TorchSampler.Args(max_seq_len=64, max_draft_len=0, max_beam_width=1,
                                             max_num_sequences=N_UPDATE, max_total_draft_tokens=0,
                                             disable_overlap_scheduler=False))
    sampler._batch_fastpath_eligible = fastpath
    reqs = _requests_for_update(simple_format=simple_format, num_logprobs=num_logprobs, stop_words=stop_words)
    torch.manual_seed(0)
    logits = torch.randn(N_UPDATE, VOCAB, device="cuda")
    logits[:, STOP] = -10.0
    logits[2, STOP] = 20.0  # request 2 samples the stop word
    calls = []
    orig = sampler_mod.add_new_tokens_to_requests

    def spy(requests, tokens, beam):
        calls.append(len(requests))
        return orig(requests, tokens, beam)

    with monkeypatch.context() as p:
        p.setattr(TorchSampler, "_collect_new_requests_for_setup", lambda self, r: r.all_requests())
        p.setattr(sampler_mod, "add_new_tokens_to_requests", spy)
        with torch.inference_mode():
            sched = _scheduled(reqs)
            sampler.setup_sampler_step(sched)
            state = sampler.sample_async(sched, model_outputs={"logits": logits},
                                         num_context_logits_prefix_sum=[0], resource_manager=None)
            sampler.update_requests(state, resource_manager=None)
    out = []
    for r in reqs:
        lp = r.py_result.log_probs
        lp = lp() if callable(lp) else lp
        out.append((r.get_tokens(0)[-1], r.is_finished, r.py_decoding_iter, r.py_num_accepted_draft_tokens,
                    r.py_rewind_len, r.py_num_draft_tokens_verified, repr(lp)))
    counters = sampler._finish_reasons_handler.store.num_accepted_draft_tokens_host[:N_UPDATE].tolist()
    return out, counters, calls


@pytest.mark.parametrize("simple_format,num_logprobs", [(True, 0), (False, 0), (False, 2)])
@pytest.mark.parametrize("stop_words", [True, False])
def test_update_requests_fastpath_matches_generic_loop(monkeypatch, simple_format, num_logprobs, stop_words):
    kw = dict(simple_format=simple_format, num_logprobs=num_logprobs, stop_words=stop_words)
    ref, ref_counters, ref_calls = _run_update_requests(False, monkeypatch, **kw)
    fast, fast_counters, fast_calls = _run_update_requests(True, monkeypatch, **kw)
    assert ref_calls == [] and fast_calls == [N_UPDATE]  # the generic path never batches; the fast path batches all
    assert fast == ref  # tokens, finish state, decoding iter, draft counters and logprobs
    if stop_words:
        # both paths zero the stop-word bookkeeping counter of every request that carries stop words
        # (slots without stop words are scratch and never written by either path)
        assert fast_counters == [0] * N_UPDATE and ref_counters == [0] * N_UPDATE
        assert fast[2][1] is True  # request 2 sampled the stop word -> finished
        assert all(not row[1] for i, row in enumerate(fast) if i != 2)
    else:
        assert all(not row[1] for row in fast)
    assert all(row[2] == 1 for row in fast)
    assert all(row[6] != "None" and "[" in row[6] for row in fast)  # one logprob entry appended per request


# ---------------------------------------------------------------------------------------------
# Admission: the stop-word setup copies only the past-token window and caches the padded
# stop-word tensor per list; the values the finish-reason kernels read must be unchanged.
# ---------------------------------------------------------------------------------------------


def _finish_handler(max_len=8, max_num=4):
    return FinishReasonsHandler(max_stop_word_length=max_len, max_num_stop_words=max_num, max_num_sequences=16,
                                max_beam_width=1, max_tokens=1, max_seq_len=100000)


def _admitted_request(slot, prompt, stop_words=None):
    return LlmRequest(request_id=slot, seq_slot=slot, input_tokens=prompt, max_new_tokens=8,
                      stop_words_list=stop_words if stop_words else None,
                      sampling_config=SamplingConfig(), is_streaming=False)


@pytest.mark.parametrize("prompt_len", [3, 7, 40, 5000])
def test_past_tokens_window_matches_full_token_list(prompt_len):
    handler = _finish_handler(max_len=8)
    prompt = [1000 + i for i in range(prompt_len)]
    request = _admitted_request(1, prompt)
    got = handler._get_past_tokens(request).cpu()
    window = 7
    expected = torch.zeros(window, 1, dtype=torch.int32)
    tail = prompt[-window:]
    expected[window - len(tail):, 0] = torch.tensor(tail, dtype=torch.int32)
    assert torch.equal(got, expected)
    # generated tokens count as well
    request.add_new_token(77, 0)
    got = handler._get_past_tokens(request).cpu()
    tail = (prompt + [77])[-window:]
    expected[window - len(tail):, 0] = torch.tensor(tail, dtype=torch.int32)
    assert torch.equal(got, expected)


def test_setup_sampler_step_end_to_end_values_unchanged():
    """Two admitted requests sharing a stop list: the store rows equal a fresh handler's rows
    built through the same path (cache hit vs miss)."""
    stop = [[151645, 198], [151658, 198, 151667]]
    prompts = [[5, 6, 7, 8, 9, 10, 11, 12, 13], [1, 2]]

    def build(handler, slots):
        requests = [_admitted_request(s, p, stop) for s, p in zip(slots, prompts)]
        handler.setup_new_request_handling()
        for r in requests:
            handler.prepare_for_new_request(r)
        host = torch.tensor([slots, handler.new_max_lens, handler.new_end_ids], dtype=torch.int32)
        cuda = host.cuda()
        handler.update_for_new_request(seq_slots_cuda_long=cuda[0].long(), max_lengths_cuda=cuda[1],
                                       end_ids_cuda=cuda[2], seq_slots_host=host[0], all_sampling_requests=requests)
        torch.cuda.synchronize()
        st = handler.store
        # past_tokens_cuda row 0 is never written by the setup (the first sampling step shifts the
        # rows); compare the rows the setup fills.
        return st.stop_words_cuda[..., slots].cpu().clone(), st.past_tokens_cuda[1:, slots].cpu().clone()

    a = _finish_handler()
    words_a, past_a = build(a, [3, 4])
    assert len(a._stop_words_cache) == 1
    b = _finish_handler()
    words_b, past_b = build(b, [3, 4])
    assert torch.equal(words_a, words_b) and torch.equal(past_a, past_b)
    # the past-token rows hold the prompt tails (shifted by one row for the first sampling step)
    window = a._max_stop_word_length - 1
    assert past_a[:, 0, 0].tolist()[-len(prompts[0][-window:]):] == prompts[0][-window:]
    assert past_a[:, 1, 0].tolist()[-2:] == prompts[1]
