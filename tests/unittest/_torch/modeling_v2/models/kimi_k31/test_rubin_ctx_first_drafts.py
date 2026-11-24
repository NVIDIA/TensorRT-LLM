# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A request's first block of drafts made on the disaggregated context server, and its handoff.

The tep8 context server's DFlash2 worker drafts, on a request's last prefill chunk, the block a
generation step would draft right after it (``DFlashWorker._draft_context_rows``); the sampler
sends a row without drafts (``CONTEXT_NO_DRAFT_TOKEN``) as none; the helix16 generation server
counts a full sent block as real drafts. Each NVFP4 GEN receiver drops the slot's stored
proposals (``prepare_disagg_generation_request``). Checked here:

GPU (sm_103 / sm_107, the worker contract's toy drafter):
* the context rows' drafts, block hidden states and pool bytes equal a generation step's that
  accepted one token after a history one shorter (the same history ``[0, L)``, the same bonus at
  ``L``): the block reads the history, sits at ``[L, L + block)`` in the row's own pages, and the
  drafts are its greedy walk;
* the history length and its position offset stay as the prefill left them (the length the
  context server hands the generation server);
* a row whose block would leave its pages, a chunk that is not the last, a row with no history
  slot and a history the unified cache dropped draft nothing, write nothing and send none;
* the drafting makes no device-to-host synchronization;
* the handoff, outside inference mode, drops a slot of the helix16 worker's own proposal store
  (made and walked by its sampled step in inference mode), and the next sampled verification
  takes that row as a point mass.

CPU: the sampler's empty send, the engine's last-chunk record, the generation server's handoff
(full blocks only, never zeros or a short list), the slot's proposals
dropped so a sampled first step verifies the sent block as a point mass (also from a store made in
inference mode, the handoff outside it), and forced acceptance reading the flag on the request's
first step only.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

__extra_import_path__ = ["."]
import test_rubin_dflash2_worker_contract as C  # noqa: E402

_TEP8 = C._Case("kimi_k31_pro_nvfp4__sm_107__tep8")
_H16 = "kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1"
_NVFP4_GEN_RECEIVERS = (
    pytest.param("sm_107__dep16", "Sm107Dep16", id="dep16"),
    pytest.param("sm_107__helix8", "Sm107Helix8", id="helix8"),
    pytest.param("sm_107__helix16_moetp16ep1", "Sm107Helix16Moetp16ep1", id="helix16"),
    pytest.param("sm_107__tep8", "Sm107Tep8", id="tep8"),
    pytest.param("sm_10x__default", "Sm10xDefault", id="default"),
)
_K = 7
_BLOCK = _K + 1
_NO_DRAFT = -1


# --- the context server's first block (GPU) ---


class _Manager:
    """The unified draft cache's host side the context drafter reads: each request's pages."""

    has_standalone_draft_cache = True

    def __init__(self, num_blocks: dict):
        self.num_blocks = num_blocks

    def get_draft_num_blocks(self, request_id: int) -> int:
        return self.num_blocks[request_id]


def _head(container):
    """A CUDA LM head and the shell's logits processor (the head's product, every row)."""
    lm_head = torch.nn.Linear(C._H, C._VOCAB, bias=False, dtype=torch.bfloat16).cuda()
    torch.nn.init.normal_(lm_head.weight, std=0.5)
    container._head = (lm_head, lambda hidden, head, attn, keep: head(hidden).float())


def _gen_drafts(worker, container, hidden, bonus, spec, n):
    """What the generation path of ``_forward_impl`` drafts from a block's outputs."""
    rows = hidden[worker._block_gather_ids(container, n, _BLOCK, _K, hidden.shape[0])]
    logits = container.logits_processor(rows, container.lm_head, None, True)
    logits = logits.reshape(n, _K, logits.shape[-1])
    logits = worker._apply_dflash2_selector(container, logits, rows.reshape(n, _K, -1), bonus, spec)
    return worker.sample_draft_tokens(logits, spec, n, num_contexts=0), rows


def _setup(monkeypatch, seed=5):
    rig = C._rig(_TEP8, _K, C._FP8, monkeypatch)
    _head(rig.container)
    # The toy drafter's selector rank (8) is below the walk kernel's granularity (256). Both
    # paths make the same select_candidate_path call (the walk has its own tests), so a stand-in
    # keeps the top-k logits and records the anchor tokens each path hands it.
    anchors = []

    def walk(block_logits, candidate_ids, unary_logits, block_hidden_states, anchor_token_ids):
        anchors.append(anchor_token_ids.clone())
        return block_logits

    monkeypatch.setattr(rig.container, "select_candidate_path", walk)
    rig.anchors = anchors
    rig.spec.wants_advanced_draft_sampling = False
    inputs = C._inputs(rig, seed, (30, 63, 97))
    # No draft matches the target: every row accepts its target token alone.
    target = inputs["logits"].view(C._BATCH, _BLOCK, C._VOCAB).argmax(-1)
    inputs["draft_tokens"].copy_(((target[:, :_K] + 1) % C._VOCAB).reshape(-1).to(torch.int32))
    C._load(rig, inputs)
    return rig, inputs


def _pools(rig):
    return [p.view(torch.uint8).clone() for p in rig.pools]


def _context_worker(rig, inputs, *, lengths, num_blocks, last_chunk):
    """The worker as a context server's prefill leaves it: rows 0..2 the requests 100..102 on the
    rig's slots with ``lengths`` of history, row 3 a padding request without a slot."""
    w = rig.worker
    slots = inputs["slots"].tolist()
    w.context_server = True
    w._req_to_slot = {100 + row: slots[row] for row in range(3)}
    w._ctx_len_host = [0] * C._BATCH
    for row in range(3):
        w._ctx_len_host[slots[row]] = lengths[row]
    w._ctx_kv_manager = _Manager({100 + row: num_blocks[row] for row in range(3)} | {999: 1})
    rig.spec.request_ids = [100, 101, 102, 999]
    rig.spec.context_last_chunk = list(last_chunk)
    return w


@C._GPU
def test_context_rows_draft_what_a_generation_step_after_one_token_drafts(monkeypatch):
    rig, inputs = _setup(monkeypatch)
    w, c = rig.worker, rig.container
    # The generation step: history [0, L - 1), one accepted token, so the history becomes [0, L)
    # and the block starts at L with that token as its bonus.
    accepted, num = w.sample_and_accept_draft_tokens(rig.logits, rig.attn, rig.spec)
    assert torch.equal(num.cpu(), torch.ones(C._BATCH, dtype=num.dtype))
    prep = w._prepare_paged_latent_gen_inputs(
        accepted, num, rig.attn, rig.spec, c, C._BATCH * _BLOCK, c.mask_token_id, None
    )
    history = _pools(rig)
    ctx_len = w._ctx_len.clone()
    offset = w._ctx_position_offset.clone()
    hidden = c.dflash_forward(**{name: prep[name] for name in C._DFLASH_FORWARD_KW})
    want, want_rows = _gen_drafts(w, c, hidden, prep["first_prev_tokens"], rig.spec, C._BATCH)
    torch.cuda.synchronize()
    after_block = _pools(rig)

    # The context server: the same history [0, L), the same first token.
    for dst, src in zip(rig.pools, history):
        dst.view(torch.uint8).copy_(src)
    slots = inputs["slots"].tolist()
    lengths = [int(ctx_len[slots[row]]) for row in range(3)]
    assert lengths == [length + 1 for length in (30, 63, 97)]
    _context_worker(
        rig, inputs, lengths=lengths, num_blocks=[C._PAGES_PER_ROW] * 3, last_chunk=[1, 1, 1, 1]
    )
    spy = []
    forward = c.dflash_forward
    monkeypatch.setattr(
        c, "dflash_forward", lambda *a, **k: spy.append(forward(*a, **k)) or spy[-1]
    )
    with _no_host_sync():
        got = w._draft_context_rows(accepted, rig.attn, rig.spec, c, C._BATCH)
    torch.cuda.synchronize()

    assert got.dtype == torch.int32 and got.shape == (C._BATCH, _K)
    assert torch.equal(got[:3], want[:3].to(torch.int32)), (got, want)
    assert got[3].tolist() == [_NO_DRAFT] * _K  # the padding row: no slot, no drafts
    # The walk's anchors: each row's first token, as the generation step's bonus.
    assert len(rig.anchors) == 2 and torch.equal(rig.anchors[1], rig.anchors[0][:3])
    # The block itself: the generation step's rows 0..2 (3 rows here, 4 there: the GEMMs may
    # take other tactics, so to rounding).
    torch.testing.assert_close(spy[0], hidden[: 3 * _BLOCK], rtol=2e-2, atol=2e-2)
    # The pools: the block's latents at [L, L + block) of each row's pages, as the generation
    # step's block leaves them (to an FP8 step); every other byte as the prefill left it (the
    # padding row's block went nowhere).
    table = w._ctx_block_tables
    for layer, (now, gen, before) in enumerate(zip(_pools(rig), after_block, history)):
        rest = now.clone()
        for row in range(3):
            pos = torch.arange(lengths[row], lengths[row] + _BLOCK, device="cuda")
            pages, offs = table[row].long()[pos // C._PAGE], pos % C._PAGE
            a = now[pages, 0, 0, offs].view(C._FP8).float()
            b = gen[pages, 0, 0, offs].view(C._FP8).float()
            assert bool(((a - b).abs() <= 0.13 * b.abs() + 2e-2).all()), (layer, row)
            rest[pages, 0, 0, offs] = before[pages, 0, 0, offs]
        assert torch.equal(rest, before), f"layer {layer}: a byte outside the blocks changed"
    # The history the handoff reads: length and position offset as the prefill left them.
    assert torch.equal(w._ctx_len, ctx_len) and torch.equal(w._ctx_position_offset, offset)


def _no_host_sync():
    import contextlib

    @contextlib.contextmanager
    def check():
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            yield
        finally:
            torch.cuda.set_sync_debug_mode("default")

    return check()


@C._GPU
def test_rows_that_cannot_draft_send_none_and_write_nothing(monkeypatch):
    rig, inputs = _setup(monkeypatch, seed=9)
    w, c = rig.worker, rig.container
    accepted, _ = w.sample_and_accept_draft_tokens(rig.logits, rig.attn, rig.spec)
    length = 2 * C._PAGE - 3  # the block crosses into a third page
    before = _pools(rig)
    cases = {
        # (lengths, pages held, last chunk) per row 0..2
        "pages": ([length] * 3, [2, C._PAGES_PER_ROW, 2], [1, 1, 1, 1]),
        "chunk": ([length] * 3, [C._PAGES_PER_ROW] * 3, [0, 1, 0, 1]),
        "dropped": ([0, length, 0], [C._PAGES_PER_ROW] * 3, [1, 1, 1, 1]),
    }
    for name, (lengths, blocks, last) in cases.items():
        for dst, src in zip(rig.pools, before):
            dst.view(torch.uint8).copy_(src)
        _context_worker(rig, inputs, lengths=lengths, num_blocks=blocks, last_chunk=last)
        got = w._draft_context_rows(accepted, rig.attn, rig.spec, c, C._BATCH)
        torch.cuda.synchronize()
        assert int(got[1].min()) >= 0, name  # row 1 drafts in every case
        for row in (0, 2):
            assert got[row].tolist() == [_NO_DRAFT] * _K, (name, row)
        # Only row 1's block positions changed.
        table = w._ctx_block_tables
        pos = torch.arange(length, length + _BLOCK, device="cuda")
        pages = table[1].long()[pos // C._PAGE]
        for now, old in zip(_pools(rig), before):
            now = now.clone()
            now[pages, 0, 0, pos % C._PAGE] = old[pages, 0, 0, pos % C._PAGE]
            assert torch.equal(now, old), name


def test_only_a_context_server_drafts_its_context_rows():
    """The shell marks the worker from the role routing recorded; without the context role
    (aggregated serving) the context rows keep the zero placeholders the next step verifies as
    such, and a context-KV-only drafter on a context server sends no drafts."""
    import tensorrt_llm

    root = Path(tensorrt_llm.__file__).parent / "_torch/_experimental/modeling_v2/models/kimi_k31"
    source = (root / f"{_TEP8.target}/modeling.py").read_text()
    assert 'self.spec_worker.context_server = core.role == "context"' in source
    tree = ast.parse(source)
    impl = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_forward_impl"
    )
    text = ast.unparse(impl)
    assert "num_contexts > 0 and self.context_server and drafts and" in text
    assert "self.max_draft_len" in text
    assert "CONTEXT_NO_DRAFT_TOKEN" in text and "torch.zeros((num_contexts, K)" in text


# --- the handoff (CPU) ---


def test_the_sampler_sends_no_drafts_for_a_row_marked_without():
    from tensorrt_llm._torch.speculative.interface import CONTEXT_NO_DRAFT_TOKEN
    from tensorrt_llm._torch.speculative.spec_sampler_base import SpecSampler

    assert CONTEXT_NO_DRAFT_TOKEN == _NO_DRAFT
    handle = SpecSampler._request_common_handling
    store = [[5, 6, 7, 8, 9, 10, 11], [_NO_DRAFT] * _K, [0] * _K]
    for slot, want in ((0, store[0][:5]), (1, []), (2, [0] * 5)):
        req = SimpleNamespace(py_seq_slot=slot, py_decoding_iter=0)
        handle(None, req, store, 5)
        assert req.py_draft_tokens == want and req.py_decoding_iter == 1


def _engine_file(rel: str) -> ast.Module:
    import tensorrt_llm

    root = Path(tensorrt_llm.__file__).parent
    return ast.parse((root / rel).read_text())


def test_the_input_preparer_records_each_context_chunk_as_last_or_not():
    """``spec_metadata.context_last_chunk`` is the context requests' ``is_last_context_chunk``, in
    the order their rows are packed (``scheduled_requests.context_requests``)."""
    tree = _engine_file("_torch/pyexecutor/engine/runners/decoder/inputs.py")
    found = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Attribute) and t.attr == "context_last_chunk" for t in node.targets
        )
    ]
    assert len(found) == 1
    comp = found[0].value
    assert isinstance(comp, ast.ListComp)
    assert ast.unparse(comp.elt) == "request.is_last_context_chunk"
    assert ast.unparse(comp.generators[0].iter) == "scheduled_requests.context_requests"
    packed = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and ast.unparse(node.iter) == "scheduled_requests.context_requests"
    ]
    assert packed, "the context rows are packed in scheduled_requests.context_requests order"


def _h16():
    return importlib.import_module(f"{C._PACKAGE}.{_H16}.modeling")


class _Store:
    def __init__(self, rows=6):
        from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_sampling import (  # noqa: E501
            DFlash2ProposalStore,
        )

        self.store = DFlash2ProposalStore(rows, _K, 32, "cpu")


def _gen_model(store=None):
    worker = SimpleNamespace(max_draft_len=_K, _dflash2_proposals=store)
    return SimpleNamespace(spec_worker=worker, mapping=SimpleNamespace(rank=0))


def _request(drafts, slot=2):
    return SimpleNamespace(
        py_seq_slot=slot,
        py_has_real_drafts=False,
        context_phase_params=SimpleNamespace(draft_tokens=drafts),
    )


@pytest.mark.parametrize(
    "drafts,real",
    [
        ([3, 1, 4, 1, 5, 9, 2], True),
        ([0, 0, 0, 0, 0, 0, 7], True),
        (None, False),  # nothing sent: the executor's zero placeholders
        ([], False),
        ([0] * _K, False),  # a context server that sends zeros for no drafts
        ([3, 1, 4], False),  # a short block
        ([3, 1, 4, 1, 5, 9, 2, 6], False),
        ([3, 1, -1, 1, 5, 9, 2], False),
    ],
)
def test_the_generation_server_counts_full_sent_blocks_as_real(drafts, real):
    pytest.importorskip("cutlass")
    hook = _h16().ModelingV2KimiK31ProNvfp4Sm107Helix16Moetp16ep1.prepare_disagg_generation_request
    req = _request(drafts)
    hook(_gen_model(), req)
    assert req.py_has_real_drafts is real


@pytest.mark.parametrize("layout, model_suffix", _NVFP4_GEN_RECEIVERS)
def test_the_slot_proposals_are_dropped_so_a_sampled_first_step_takes_the_point_mass(
    layout, model_suffix
):
    """A previous request's walk in the slot, equal to the sent block, would be read as the block's
    proposal; the handoff drops it, so the verification reads the point mass on the sent block,
    the exact law of a deterministic (greedy) proposal."""
    pytest.importorskip("cutlass")
    mod = importlib.import_module(f"{C._PACKAGE}.kimi_k31_pro_nvfp4__{layout}.modeling")
    model_type = getattr(mod, f"ModelingV2KimiK31ProNvfp4{model_suffix}")
    hook = model_type.prepare_disagg_generation_request
    sent = torch.tensor([[3, 1, 4, 1, 5, 9, 2]])
    slots = torch.tensor([2, 3])
    ids = torch.arange(32).expand(2, _K, 32)
    q = torch.full((2, _K, 32), 1 / 32)
    with torch.inference_mode():
        store = _Store().store
        store.store(slots, ids, q, sent.expand(2, _K))
    assert store.tokens.is_inference() and not torch.is_inference_mode_enabled()
    _, stale_q = store.load(slots[:1], sent)
    assert torch.equal(stale_q, q[:1])  # An exact token match alone reads the old proposal.

    model = _gen_model(store)
    if layout == "sm_10x__default":
        model = SimpleNamespace(llm=model)
    hook(model, _request(sent[0].tolist(), slot=2))
    got_ids, got_q = store.load(slots, sent.expand(2, _K))
    assert torch.equal(got_ids[0], sent[0, :, None].expand(_K, 32))
    assert torch.equal(got_q[0], (torch.arange(32) == 0).float().expand(_K, 32))
    assert torch.equal(got_ids[1], ids[1]) and torch.equal(got_q[1], q[1])
    # Subsequent GEN drafts can write and use a sampled proposal for this same slot.
    with torch.inference_mode():
        store.store(slots[:1], ids[:1], q[:1], sent)
    assert torch.equal(store.load(slots[:1], sent)[1], q[:1])


def test_the_slot_is_dropped_from_a_store_made_in_inference_mode():
    """A served store is made by a forward, under inference mode (its tensors are inference
    tensors); the executor calls the handoff outside it. The drop still writes the row."""
    pytest.importorskip("cutlass")
    hook = _h16().ModelingV2KimiK31ProNvfp4Sm107Helix16Moetp16ep1.prepare_disagg_generation_request
    sent = torch.tensor([[3, 1, 4, 1, 5, 9, 2]])
    with torch.inference_mode():
        store = _Store().store
        store.store(
            torch.tensor([2]), torch.zeros(1, _K, 32, dtype=torch.long), torch.ones(1, _K, 32), sent
        )
        other = store.tokens[3].clone()
    assert store.tokens.is_inference() and not torch.is_inference_mode_enabled()
    hook(_gen_model(store), _request(sent[0].tolist(), slot=2))
    assert store.tokens[2].tolist() == [-1] * _K and torch.equal(store.tokens[3], other)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 3), (10, 7)),
    reason="the DFlash2 sampled path's kernels are certified on sm_103 / sm_107",
)
def test_the_handoff_drops_a_row_of_the_workers_own_store():
    """The helix16 worker's sampled step makes its store (``_proposal_store``) and walks every
    slot in inference mode, as a served forward does; the handoff, outside inference mode, drops
    one slot. The next sampled verification takes that row as a point mass, exactly as a row
    invalidated in the forward, and the other rows as their walks."""
    pytest.importorskip("cutlass")
    from _rubin_rs_rig import VOCAB, Bufs, Markov, build_worker, sampled_spec, stub_drafter

    hook = _h16().ModelingV2KimiK31ProNvfp4Sm107Helix16Moetp16ep1.prepare_disagg_generation_request
    n, slot = 8, 5

    def step(drop):
        drafter = stub_drafter()
        worker = build_worker("helix16", drafter)
        spec = sampled_spec(n, seed=77)
        markov = Markov()
        bufs = Bufs.make(n, rounds=1)
        bufs.last.copy_(torch.arange(n, device="cuda") * 5 % VOCAB)
        with torch.inference_mode():
            drafts = worker._sample_block_path(
                drafter, markov.draft_base[bufs.last], bufs.hidden, bufs.last, spec, 0, n
            )
        store = worker._dflash2_proposals
        assert store is not None and store.tokens.is_inference()
        walked = store.tokens[slot].clone()
        assert torch.equal(walked, drafts[slot].long()), "the slot holds its walk"
        if drop == "handoff":
            assert not torch.is_inference_mode_enabled()
            sent = drafts[slot].tolist()
            hook(
                SimpleNamespace(spec_worker=worker, mapping=SimpleNamespace(rank=0)),
                _request(sent, slot=slot),
            )
            assert store.tokens[slot].tolist() == [-1] * _K
        with torch.inference_mode():
            if drop == "forward":
                store.invalidate(torch.tensor([slot], device="cuda"))
            spec.draft_tokens.copy_(drafts.reshape(-1))
            prefix = torch.cat((bufs.last[:, None], drafts.long()), dim=1)
            logits = markov.log_trans[prefix].view(n * (_K + 1), VOCAB).contiguous()
            attn = SimpleNamespace(num_seqs=n, num_contexts=0)
            accepted, num = worker._verify_sampled_block(logits, attn, spec)
        torch.cuda.synchronize()
        return accepted.clone(), num.clone()

    handoff, forward, kept = step("handoff"), step("forward"), step(None)
    assert torch.equal(handoff[0], forward[0]) and torch.equal(handoff[1], forward[1])
    rows = [r for r in range(n) if r != slot]
    assert torch.equal(handoff[0][rows], kept[0][rows]) and torch.equal(
        handoff[1][rows], kept[1][rows]
    )


def test_forced_acceptance_reads_the_flag_on_the_first_step_only():
    """The executor's real-row marking takes the handoff's flag for the request's first step, then
    every later step of a drafting worker is real."""
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs import DecoderInputPreparer

    spec_config = SimpleNamespace(
        spec_dec_mode=SimpleNamespace(drafts_context_requests=lambda: False)
    )
    first = SimpleNamespace(py_has_real_drafts=True)
    placeholder = SimpleNamespace(py_has_real_drafts=False)
    rows = []
    host = SimpleNamespace(
        lists=SimpleNamespace(
            real_draft_rows=rows, extend_requests=[], generation_requests=[first, placeholder]
        )
    )
    batch = SimpleNamespace(context_requests=[])
    DecoderInputPreparer._mark_real_draft_rows(host, batch, spec_config, _K)
    assert rows == [True, False]
    assert first.py_has_real_drafts and placeholder.py_has_real_drafts
