# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFlash2 rejection sampling on the helix targets' workers (helix8, helix16), one GPU.

The sampled path is the worker's own: ``_sample_block_path`` (global top-32, the selector's
sampled walk, the proposal store) then ``_verify_sampled_block`` (target probabilities, proposal
load, seeded block verification, greedy rows by the strict rule), on the real kernels
(``_rubin_rs_rig``). Checked here:

* the law: rounds of draft + verification over a Markov target emit the target's joint law of
  the first three tokens (chi-square), at one request (the selector's sequential-scan path), four
  requests with a placeholder-draft row (the store's point mass) and 64 requests with greedy and
  placeholder rows, a T = 0.6 target and the fused probabilities; greedy rows emit the target's
  argmax chain exactly. The generations run as CUDA graph replays (fresh uniforms per replay);
* capture on one set of inputs and replay on another equals the eager step on the second, outputs
  and proposal store alike, with no Triton / CuTe DSL compile after the warmup pass and no
  device-to-host synchronization in the eager step;
* forced acceptance on the sampled path forces the count and keeps the tokens;
* temperature 0 with ``use_rejection_sampling`` on is the flag-off step: the same outputs, the same
  kernels, no proposal store and none of the sampled path's state.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from _rubin_rs_rig import (
    GREEDY_T,
    VOCAB,
    Bufs,
    K,
    Markov,
    build_worker,
    chi_square_gate,
    cuda_kernels,
    generate,
    module,
    sampled_spec,
    stub_drafter,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 3), (10, 7)),
    reason="the DFlash2 sampled path's kernels are certified on sm_103 / sm_107",
)

_CASES = ("helix8", "helix16")


def _compile_log():
    """The worker contract test's Triton / CuTe DSL compile counter."""
    path = Path(__file__).with_name("test_rubin_dflash2_worker_contract.py")
    spec = importlib.util.spec_from_file_location("_k31_rs_contract", path)
    mod = sys.modules.get(spec.name)
    if mod is None:
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    return mod._CompileLog()


def _mode(name: str):
    from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode

    return AdvancedSamplingMode(name)


# --- the law ---------------------------------------------------------------------------------

# (id, requests, replays, advanced sampling mode, target temperature, greedy rows, placeholder rows)
_LAW = (
    ("c1", 1, 40000, "no_topk_no_topp", 1.0, (), ()),
    ("b4_placeholder", 4, 10000, "no_topk_no_topp", 1.0, (), (3,)),
    ("b64_mixed_t06_full", 64, 700, "full", 0.6, (5, 21, 37, 53), tuple(range(3, 64, 8))),
)


@pytest.mark.parametrize("cfg", _LAW, ids=[c[0] for c in _LAW])
@pytest.mark.parametrize("case", _CASES)
def test_sampled_rounds_emit_the_target_law(case, cfg):
    _, n, replays, mode, temperature, greedy_rows, placeholder_rows = cfg
    torch.manual_seed(0)
    drafter = stub_drafter()
    worker = build_worker(case, drafter)
    spec = sampled_spec(n, mode=_mode(mode), temperature=temperature, greedy_rows=greedy_rows)
    markov = Markov()
    bufs = Bufs.make(n, rounds=3)
    placeholder = None
    if placeholder_rows:
        placeholder = torch.zeros(n, dtype=torch.bool, device="cuda")
        placeholder[list(placeholder_rows)] = True

    def run():
        generate(worker, drafter, spec, markov, bufs, placeholder=placeholder)

    run()  # eager warmup: the proposal store, compiles
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    samples = torch.empty(replays, n, 3, dtype=torch.int64, device="cuda")
    taus = torch.zeros(n, dtype=torch.int64, device="cuda")
    for r in range(replays):
        graph.replay()
        samples[r].copy_(bufs.out[:, :3])
        taus.add_(bufs.taus.sum(0))
    torch.cuda.synchronize()

    sampled = torch.ones(n, dtype=torch.bool)
    sampled[list(greedy_rows)] = False
    first = samples[:, sampled.cuda()].reshape(-1, 3)
    chi2, dof = chi_square_gate(first, markov.joint3(temperature))
    print(f"{case} {cfg[0]}: {first.shape[0]} samples, chi2 {chi2:.1f} on {dof} dof")
    chain = torch.tensor(markov.greedy_chain(3), device="cuda")
    for row in greedy_rows:
        assert bool((samples[:, row] == chain).all()), f"greedy row {row} left the argmax chain"
    mean_tau = float(taus[sampled.cuda()].sum()) / (replays * 3 * int(sampled.sum()))
    print(f"{case} {cfg[0]}: mean accepted drafts per round {mean_tau:.3f}")
    assert mean_tau > 0.2


# --- capture / replay on new inputs, no compile after warmup, no host sync ---------------------


def _state(worker, bufs) -> dict:
    """The step's outputs and the store rows of its requests' slots (0 .. n - 1; the store's
    last row is the CUDA-graph padding slot, which no request here writes)."""
    store = worker._dflash2_proposals
    n = bufs.last.shape[0]
    return {
        "out": bufs.out.clone(),
        "num_accepted": bufs.num_accepted.clone(),
        "taus": bufs.taus.clone(),
        "store.candidate_ids": store.candidate_ids[:n].clone(),
        "store.q": store.q[:n].view(torch.int32).clone(),
        "store.tokens": store.tokens[:n].clone(),
    }


@pytest.mark.parametrize("n", [1, 4])
@pytest.mark.parametrize("case", _CASES)
def test_sampled_step_replay_on_new_inputs_equals_eager(case, n):
    drafter = stub_drafter()
    worker = build_worker(case, drafter)
    spec = sampled_spec(n, seed=11)
    markov = Markov()
    bufs = Bufs.make(n, rounds=2)
    placeholder = torch.zeros(n, dtype=torch.bool, device="cuda")
    placeholder[-1] = n > 1

    def run():
        generate(worker, drafter, spec, markov, bufs, rounds=2, placeholder=placeholder)

    def inputs_b():
        spec.request_seeds.copy_(torch.arange(n, device="cuda") * 31 + 99)
        spec.request_offsets.fill_(5 << 20)
        bufs.hidden.copy_(torch.randn_like(bufs.hidden.float()).to(bufs.hidden.dtype))

    run()  # warmup
    torch.cuda.synchronize()
    with _compile_log() as log:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        torch.manual_seed(2)
        inputs_b()
        hidden_b = bufs.hidden.clone()
        graph.replay()
        torch.cuda.synchronize()
        replayed = _state(worker, bufs)
        inputs_b()
        bufs.hidden.copy_(hidden_b)
        for t in (worker._dflash2_proposals.candidate_ids, worker._dflash2_proposals.tokens):
            t.zero_()
        worker._dflash2_proposals.q.zero_()
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            run()
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
        eager = _state(worker, bufs)
    for name in eager:
        assert torch.equal(replayed[name], eager[name]), f"{name} differs"
    assert not log.compiled, f"compiled after the warmup pass: {log.compiled}"


# --- forced acceptance on the sampled path ----------------------------------------------------


@pytest.mark.parametrize("case", _CASES)
def test_forced_acceptance_forces_the_count_and_keeps_the_tokens(case):
    n = 8
    drafter = stub_drafter()
    markov = Markov()
    results = {}
    for force in (0.0, 3.0):
        worker = build_worker(case, drafter, force=force)
        spec = sampled_spec(n, seed=5)
        bufs = Bufs.make(n, rounds=1)
        generate(worker, drafter, spec, markov, bufs, rounds=1)
        results[force] = (worker._dflash2_proposals.tokens.clone(), bufs.num_accepted.clone())
    torch.cuda.synchronize()
    assert torch.equal(results[0.0][0], results[3.0][0]), "the drafts differ under forcing"
    assert bool((results[3.0][1] == 4).all()), results[3.0][1]
    assert bool((results[0.0][1] <= K + 1).all())


# --- temperature 0 with the flag on is the flag-off step ---------------------------------------


@pytest.mark.parametrize("case", _CASES)
def test_greedy_step_is_bitwise_the_flag_off_step(case):
    """All-greedy batches: acceptance and (helix16) the draft agreement through its broadcast
    give the same outputs from the same kernels with the flag on as off, and the sampled path
    leaves nothing behind (no proposal store, no agreement totals or readout buffer)."""
    n = 4
    g = torch.Generator(device="cuda").manual_seed(9)
    logits = 3.0 * torch.randn(n * (K + 1), VOCAB, device="cuda", generator=g)
    target = logits.view(n, K + 1, VOCAB).argmax(-1)
    drafts = torch.randint(0, VOCAB, (n, K), device="cuda", generator=g)
    for row in range(n):
        drafts[row, : 2 * row] = target[row, : 2 * row]
    agreed_in = torch.randint(0, VOCAB, (n, K), dtype=torch.int32, device="cuda", generator=g)
    attn = SimpleNamespace(num_seqs=n, num_contexts=0)
    runs = {}
    for flag in (False, True):
        worker = build_worker(case, stub_drafter())
        spec = sampled_spec(n, flag=flag, all_greedy=True)
        spec.draft_tokens.copy_(drafts.reshape(-1))
        mod = module(case)
        if case == "helix16":
            worker._draft_agreement = mod.ZeroPaddedReduce(
                lambda buf: buf.clone(),
                0,
                K,
                torch.zeros(n, mod._DRAFT_AGREEMENT_WIDTH, dtype=torch.float32, device="cuda"),
            )

        def step():
            accepted, num = worker.sample_and_accept_draft_tokens(logits, attn, spec)
            out = {
                "next_draft_tokens": agreed_in.clone(),
                "next_new_tokens": torch.cat([accepted[:, :1], agreed_in], dim=1),
            }
            worker._agree_draft_tokens(out)
            return accepted, num, out["next_draft_tokens"], out["next_new_tokens"]

        step()  # warm
        outputs, kernels = cuda_kernels(step)
        runs[flag] = (outputs, kernels)
        assert worker._dflash2_proposals is None
        assert getattr(worker, "_rs_agree_count", None) is None
        assert getattr(worker, "_rs_agree_rows", None) is None
        assert "_rs_agree_host" not in worker.__dict__
        if case == "helix16":
            assert not worker._draft_agreement.buffer[:, K:].any()
    (off, k_off), (on, k_on) = runs[False], runs[True]
    for a, b in zip(off, on):
        assert torch.equal(a, b)
    assert k_off == k_on, f"kernels differ:\n off {k_off}\n on  {k_on}"
    assert int(off[1].min()) >= 1 and float(off[1].float().mean()) > 1.0
    assert GREEDY_T < 1e-4


@pytest.mark.parametrize("case", _CASES)
def test_the_law_gate_rejects_a_wrong_proposal(case, monkeypatch):
    """The gate's power: verifying the walk's drafts against a q other than the walk's (uniform
    over the candidates) must move the emitted law off the target's, and the gate must say so."""
    mod = module(case)
    load = mod.k31_dflash2_proposal_load

    def wrong_q(*args):
        candidate_ids, q = load(*args)
        return candidate_ids, torch.full_like(q, 1.0 / q.shape[-1])

    monkeypatch.setattr(mod, "k31_dflash2_proposal_load", wrong_q)
    n, replays = 4, 5000
    drafter = stub_drafter()
    worker = build_worker(case, drafter)
    spec = sampled_spec(n)
    markov = Markov()
    bufs = Bufs.make(n, rounds=3)
    samples = torch.empty(replays, n, 3, dtype=torch.int64, device="cuda")
    generate(worker, drafter, spec, markov, bufs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        generate(worker, drafter, spec, markov, bufs)
    for r in range(replays):
        graph.replay()
        samples[r].copy_(bufs.out[:, :3])
    torch.cuda.synchronize()
    with pytest.raises(AssertionError):
        chi_square_gate(samples.reshape(-1, 3), markov.joint3(1.0))
