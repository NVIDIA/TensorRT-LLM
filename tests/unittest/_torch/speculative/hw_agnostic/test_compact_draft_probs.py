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
"""``TRTLLM_RS_COMPACT_DRAFT_PROBS``: the compact draft-q store against the dense one, on host.

The module under test and the built-in bodies it plugs into are loaded from their source
files (``compact_draft_probs.py`` by path; ``SpecWorkerBase._accept_draft_tokens``,
``_rejection_buffers_valid``, ``_sample_and_accept_draft_tokens_rejection`` and
``DFlashWorker._sample_dflash2_block`` compiled from ``interface.py`` / ``dflash.py``), so the
tests run without the package's compiled dependencies. FlashInfer's chain speculative sampling
is replaced by a host transcription of its acceptance rule driven by a seeded generator: the
claim under test is that the verify receives the same draft rows bit for bit, hence the same
accept / reject decisions.
"""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[5]
_SPEC = _ROOT / "tensorrt_llm" / "_torch" / "speculative"


def _load_compact():
    name = "_compact_draft_probs_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SPEC / "compact_draft_probs.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cdp = _load_compact()

K, TOP_K, VOCAB, SLOTS = 7, 16, 997, 12  # slot rows = SLOTS + 1 (dummy row SLOTS)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32)


def _compile_methods(path: Path, class_name: str, names, namespace: dict) -> dict:
    """The named methods of ``class_name`` in ``path``, compiled into ``namespace``."""
    tree = ast.parse(path.read_text())
    found = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in names:
                    item.decorator_list = []
                    found[item.name] = item
    missing = set(names) - set(found)
    assert not missing, f"{class_name} in {path.name} lacks {sorted(missing)}"
    module = ast.Module(body=list(found.values()), type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


# ---- the dense store, as the built-in writers run it --------------------------------------------


def _dense_sampled_write(dense, slots, candidate_ids, proposal, vocab):
    """``DFlashWorker._sample_dflash2_block``'s dense branch (and the draft head kernel's)."""
    steps = candidate_ids.shape[1]
    dense[slots, :steps, :vocab] = dense.new_zeros(())
    arange = torch.arange(steps)
    dense[slots[:, None, None], arange[:, None], candidate_ids] = proposal


def _dense_ctx_onehot(dense, slots, draft_len, vocab):
    """``SpecWorkerBase.write_context_onehot_draft_probs``'s dense branch."""
    dense[:, :draft_len, :vocab].index_fill_(0, slots, 0.0)
    dense[:, :draft_len, 0].index_fill_(0, slots, 1.0)


def _dense_padding_onehot(dense, slots, draft_len, vocab):
    """``SpecMetadata.write_padding_onehot_draft_probs``'s dense branch."""
    dense[slots, :draft_len, :vocab] = 0.0
    dense[slots, :draft_len, 0] = 1.0


def _random_rows(gen, requests, steps=K, top_k=TOP_K, vocab=VOCAB):
    """Distinct candidate ids per row (``torch.topk`` indices) and their sampled distributions."""
    scores = torch.rand((requests, steps, vocab), generator=gen)
    ids = scores.topk(top_k, dim=-1).indices
    logits = torch.randn((requests, steps, top_k), generator=gen) * 3.0
    proposal = torch.softmax(logits, dim=-1)
    # Exact zeros and a negative zero, as a filtered sampler can produce.
    proposal[..., -1] = 0.0
    proposal[..., -2] = -0.0
    return ids, proposal


def _run_writes(seed, steps=60):
    """A seeded sequence of the three writers on a dense and a compact store.

    Returns both stores and the set of slots a live request could read (written at least once).
    """
    gen = torch.Generator().manual_seed(seed)
    dense = torch.full((SLOTS + 1, K, VOCAB), float("nan"))  # uninitialized stand-in
    compact = cdp.CompactDraftProbs(SLOTS + 1, K, VOCAB, device="cpu")
    written = set()
    for _ in range(steps):
        op = int(torch.randint(0, 3, (1,), generator=gen))
        count = int(torch.randint(1, SLOTS + 1, (1,), generator=gen))
        slots = torch.randperm(SLOTS, generator=gen)[:count]
        if op == 0:
            ids, proposal = _random_rows(gen, count)
            _dense_sampled_write(dense, slots, ids, proposal, VOCAB)
            compact.write_rows(slots, ids, proposal, VOCAB)
        elif op == 1:
            _dense_ctx_onehot(dense, slots, K, VOCAB)
            compact.write_onehot_index_fill(slots, K, VOCAB)
        else:
            _dense_padding_onehot(dense, slots, K, VOCAB)
            compact.write_onehot(slots, K, VOCAB)
        written.update(int(s) for s in slots)
    return dense, compact, sorted(written)


# ---- 1. the records rebuild the dense rows bit for bit ------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_records_rebuild_the_dense_rows_bitwise(seed):
    dense, compact, written = _run_writes(seed)
    slots = torch.tensor(written)
    rebuilt = compact.densify(slots, K, VOCAB)
    assert torch.equal(_bits(rebuilt), _bits(dense[slots, :K, :VOCAB]))
    # A batch order other than slot order, with a shorter draft length.
    order = slots.flip(0)
    rebuilt = compact.densify(order, K - 2, VOCAB)
    assert torch.equal(_bits(rebuilt), _bits(dense[order, : K - 2, :VOCAB]))


def test_a_partial_draft_write_keeps_the_other_positions():
    gen = torch.Generator().manual_seed(7)
    dense = torch.zeros((SLOTS + 1, K, VOCAB))
    compact = cdp.CompactDraftProbs(SLOTS + 1, K, VOCAB, device="cpu")
    slots = torch.tensor([1, 4, 9])
    ids, proposal = _random_rows(gen, 3)
    _dense_sampled_write(dense, slots, ids, proposal, VOCAB)
    compact.write_rows(slots, ids, proposal, VOCAB)
    ids, proposal = _random_rows(gen, 3, steps=3)
    _dense_sampled_write(dense, slots, ids, proposal, VOCAB)
    compact.write_rows(slots, ids, proposal, VOCAB)
    assert torch.equal(_bits(compact.densify(slots, K, VOCAB)), _bits(dense[slots]))


def test_the_store_is_small_and_refuses_what_it_cannot_hold():
    big = cdp.CompactDraftProbs(97, 7, 202_048, device="cpu")
    assert big.shape == (97, 7, 202_048)
    assert big.nbytes() < 1 << 20  # vs 97 * 7 * 202048 * 4 B = 0.511 GiB dense
    store = cdp.CompactDraftProbs(SLOTS + 1, K, VOCAB, device="cpu", lanes=8)
    ids, proposal = _random_rows(torch.Generator().manual_seed(0), 2)
    with pytest.raises(ValueError, match="lanes"):
        store.write_rows(torch.tensor([0, 1]), ids, proposal, VOCAB)
    with pytest.raises(ValueError, match="wide rows"):
        store.write_onehot(torch.tensor([0]), K, VOCAB + 1)
    with pytest.raises(ValueError, match="wide rows"):
        store.densify(torch.tensor([0]), K, VOCAB - 1)


def test_the_knob_and_its_refusals(monkeypatch):
    monkeypatch.delenv(cdp.ENV, raising=False)
    assert not cdp.compact_draft_probs_requested()
    for value, on in (("1", True), ("on", True), ("0", False), ("off", False), ("", False)):
        monkeypatch.setenv(cdp.ENV, value)
        assert cdp.compact_draft_probs_requested() is on
    dflash = SimpleNamespace(is_dflash=lambda: True)
    other = SimpleNamespace(is_dflash=lambda: False)
    assert cdp.compact_store_refusal(dflash, 100, 100) is None
    assert cdp.compact_store_refusal(dflash, 100, 0) is None
    assert "differs" in cdp.compact_store_refusal(dflash, 100, 32)
    assert "DFlash" in cdp.compact_store_refusal(other, 100, 100)
    assert "DFlash" in cdp.compact_store_refusal(None, 100, 100)


# ---- 2. the deferred rows and the private-logits scratch ----------------------------------------


def test_deferred_rows_into_scratch_and_the_scratch_rules():
    dense, compact, written = _run_writes(11)
    slots = torch.tensor(written[:5])
    deferred = cdp.DeferredDraftProbs(compact, slots, K, VOCAB)
    assert tuple(deferred.shape) == (5, K, VOCAB) and deferred.dim() == 3
    expected = dense[slots, :K, :VOCAB]

    fresh = deferred.materialize()
    assert torch.equal(_bits(fresh), _bits(expected))

    logits = torch.randn((5 * (K + 1), VOCAB))
    probs = torch.softmax(logits, dim=-1)
    assert cdp.private_verify_scratch(logits, probs) is None  # not marked
    cdp.mark_private_verify_logits(logits)
    scratch = cdp.private_verify_scratch(logits, probs)
    assert scratch is logits
    rows = deferred.materialize(scratch)
    assert rows.untyped_storage().data_ptr() == logits.untyped_storage().data_ptr()
    assert torch.equal(_bits(rows), _bits(expected))

    # Storage shared with a live tensor, a non-FP32 or a non-contiguous buffer is refused.
    aliased = cdp.mark_private_verify_logits(torch.randn((5 * (K + 1), VOCAB)))
    assert cdp.private_verify_scratch(aliased, aliased[: 5 * K]) is None
    half = cdp.mark_private_verify_logits(torch.randn((5 * (K + 1), VOCAB)).half())
    assert cdp.private_verify_scratch(half) is None
    strided = torch.randn((VOCAB, 5 * (K + 1))).t()
    cdp.mark_private_verify_logits(strided)
    assert cdp.private_verify_scratch(strided) is None
    # Too small a scratch falls back to a new buffer.
    small = cdp.mark_private_verify_logits(torch.randn((3, VOCAB)))
    rows = deferred.materialize(small)
    assert rows.untyped_storage().data_ptr() != small.untyped_storage().data_ptr()
    assert torch.equal(_bits(rows), _bits(expected))


# ---- 3. the built-in verify on both stores: same decisions on fixed seeds ------------------------


def _chain_reference(draft_probs, draft_token_ids, target_probs, seed, offset):
    """Host transcription of chain speculative sampling's acceptance rule.

    Position ``i`` accepts draft token ``x`` when ``u * q[x] < p[x]`` (``u`` uniform); the first
    rejection emits a draw from ``relu(p - q)`` renormalized; a fully accepted block emits a
    draw from the bonus row. Draws come from a generator seeded per request by
    (seed, offset). Returns ``(accepted [B, K + 1] int32, counts [B] int32)``.
    """
    batch, steps = draft_token_ids.shape
    accepted = torch.full((batch, steps + 1), -1, dtype=torch.int32)
    counts = torch.zeros(batch, dtype=torch.int32)
    for b in range(batch):
        gen = torch.Generator().manual_seed(int(seed[b]) * 1_000_003 + int(offset[b]))
        emitted = 0
        for i in range(steps):
            x = int(draft_token_ids[b, i])
            u = float(torch.rand((), generator=gen))
            q = float(draft_probs[b, i, x])
            p = float(target_probs[b, i, x])
            if u * q < p:
                accepted[b, i] = x
                emitted += 1
                continue
            residual = (target_probs[b, i] - draft_probs[b, i]).clamp_min(0.0)
            if float(residual.sum()) <= 0.0:
                residual = target_probs[b, i]
            accepted[b, i] = int(torch.multinomial(residual, 1, generator=gen))
            break
        else:
            accepted[b, steps] = int(torch.multinomial(target_probs[b, steps], 1, generator=gen))
        counts[b] = emitted + 1
    return accepted, counts


class _VerifyWorker:
    """The built-in rejection verify bodies on a host stand-in worker."""

    _d2t = None
    force_num_accepted_tokens = 0.0
    probs_alias_logits = False

    def _apply_occurrence_penalties(self, logits, *args):
        return logits

    def _commit_occurrence_counts(self, accepted, batch_size, spec_metadata):
        return accepted

    def _can_use_rejection_sampling(self, spec_metadata):
        return True

    def _sample_and_accept_draft_tokens_base(self, *args):
        raise AssertionError("the strict path ran instead of the rejection verify")

    def _sample_tokens_for_batch(self, logits, spec_metadata, num_contexts, batch_size):
        return logits.argmax(dim=-1).to(torch.int32)

    @staticmethod
    def _zero_padding_rows(logits, *args, **kwargs):
        return logits

    def _compute_probs_from_logits(self, mode, logits, temperatures, top_ks, top_ps, min_ps):
        probs = torch.softmax(logits / temperatures[:, None], dim=-1)
        if self.probs_alias_logits:  # a backend writing the probabilities over the logits
            return logits.copy_(probs)
        return probs

    def _rng_state_per_request(self, spec_metadata, start=0, end=None, repeat=1, *, slot):
        seeds = spec_metadata.request_seeds[start:end]
        return seeds, seeds * 0 + slot

    def _apply_force_accepted_tokens(self, num_accepted, num_contexts, draft_len, spec_metadata):
        return num_accepted


_VERIFY_NAMES = (
    "_accept_draft_tokens",
    "_rejection_buffers_valid",
    "_sample_and_accept_draft_tokens_rejection",
)


def _verify_bodies():
    namespace = {
        "torch": torch,
        "DeferredDraftProbs": cdp.DeferredDraftProbs,
        "private_verify_scratch": cdp.private_verify_scratch,
        "rejection_sampling_one_model": lambda **kw: _chain_reference(
            kw["draft_probs"],
            kw["draft_token_ids"],
            kw["target_probs"],
            kw["seed"],
            kw["offset"],
        ),
    }
    return _compile_methods(_SPEC / "interface.py", "SpecWorkerBase", _VERIFY_NAMES, namespace)


def _metadata(dense, compact, slot_ids, requests):
    return SimpleNamespace(
        draft_probs=dense,
        compact_draft_probs=compact,
        draft_probs_last_dim=VOCAB,
        draft_probs_vocab_size=VOCAB,
        batch_slot_ids=slot_ids,
        full_draft_probs=None,
        is_ragged_verify=False,
        total_verify_tokens=None,
        verify_lens=None,
        qo_indptr=None,
        advanced_sampling_mode=None,
        temperatures=torch.full((requests * (K + 1),), 0.9),
        top_ks=torch.zeros(requests * (K + 1), dtype=torch.int32),
        top_ps=torch.ones(requests * (K + 1)),
        min_ps=torch.zeros(requests * (K + 1)),
        request_seeds=torch.arange(requests, dtype=torch.int64) * 7919 + 13,
        rng_accept_slot=K,
    )


def _make_worker(probs_alias_logits=False):
    bodies = _verify_bodies()
    worker_cls = type("Worker", (_VerifyWorker,), dict(bodies))
    worker = worker_cls()
    worker.probs_alias_logits = probs_alias_logits
    return worker


def _draft_tokens_from(rows, gen):
    """Draft tokens drawn from the stored rows (what the drafter did), [B, K] int32."""
    flat = rows.reshape(-1, rows.shape[-1]).clamp_min(0.0)
    return torch.multinomial(flat, 1, generator=gen).reshape(rows.shape[:2]).to(torch.int32)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("private", [False, True])
def test_accept_reject_decisions_match_the_dense_store(seed, private):
    dense, compact, written = _run_writes(100 + seed)
    gen = torch.Generator().manual_seed(seed)
    order = torch.tensor(written)[torch.randperm(len(written), generator=gen)][:8]
    requests = order.shape[0]
    slot_ids = order.clone()
    draft_tokens = _draft_tokens_from(dense[order, :K, :VOCAB], gen)
    # Target logits close to the draft rows, so both acceptance and rejection occur.
    base = torch.log(dense[order, :K, :VOCAB].clamp_min(1e-6))
    bonus = torch.randn((requests, 1, VOCAB), generator=gen)
    target_logits = torch.cat([base, bonus], dim=1) + 0.7 * torch.randn(
        (requests, K + 1, VOCAB), generator=gen
    )
    target_logits = target_logits.reshape(requests * (K + 1), VOCAB).contiguous()

    worker = _make_worker()
    dense_logits = target_logits.clone()
    accepted_d, counts_d = worker._accept_draft_tokens(
        dense_logits, draft_tokens, 0, requests, _metadata(dense, None, slot_ids, requests)
    )
    compact_logits = target_logits.clone()
    if private:
        cdp.mark_private_verify_logits(compact_logits)
    accepted_c, counts_c = worker._accept_draft_tokens(
        compact_logits, draft_tokens, 0, requests, _metadata(None, compact, slot_ids, requests)
    )
    assert torch.equal(accepted_d, accepted_c)
    assert torch.equal(counts_d, counts_c)
    assert int(counts_d.min()) >= 1 and int(counts_d.max()) <= K + 1
    # The run is informative: some requests reject early, some accept several tokens.
    assert int(counts_d.min()) < int(counts_d.max())
    # Private logits carry the rebuilt rows after the verify; others are untouched.
    assert torch.equal(dense_logits, target_logits)
    if private:
        assert not torch.equal(compact_logits, target_logits)
        head = compact_logits.reshape(-1)[: requests * K * VOCAB].view(requests, K, VOCAB)
        assert torch.equal(_bits(head), _bits(dense[order, :K, :VOCAB]))
    else:
        assert torch.equal(compact_logits, target_logits)


def test_probabilities_written_over_the_logits_keep_the_logits_out_of_reach():
    dense, compact, written = _run_writes(5)
    order = torch.tensor(written[:6])
    requests = order.shape[0]
    gen = torch.Generator().manual_seed(5)
    draft_tokens = _draft_tokens_from(dense[order, :K, :VOCAB], gen)
    logits = torch.randn((requests * (K + 1), VOCAB), generator=gen)
    worker = _make_worker(probs_alias_logits=True)
    accepted_d, counts_d = worker._accept_draft_tokens(
        logits.clone(), draft_tokens, 0, requests, _metadata(dense, None, order, requests)
    )
    private = cdp.mark_private_verify_logits(logits.clone())
    accepted_c, counts_c = worker._accept_draft_tokens(
        private, draft_tokens, 0, requests, _metadata(None, compact, order, requests)
    )
    assert torch.equal(accepted_d, accepted_c) and torch.equal(counts_d, counts_c)


def test_a_vocab_mismatch_fails_closed_to_the_strict_path():
    _, compact, written = _run_writes(3)
    order = torch.tensor(written[:2])
    worker = _make_worker()
    meta = _metadata(None, compact, order, 2)
    meta.draft_probs_last_dim = VOCAB - 1
    draft_tokens = torch.zeros((2, K), dtype=torch.int32)
    logits = torch.randn((2 * (K + 1), VOCAB))
    with pytest.raises(AssertionError, match="strict path"):
        worker._accept_draft_tokens(logits, draft_tokens, 0, 2, meta)


# ---- 4. the DFlash 2 sampled writer: compact branch == dense branch ----------------------------


class _DFlashStandIn:
    def __init__(self, ids, unary):
        self._ids, self._unary = ids, unary

    def _dflash2_global_top_k(self, gen_logits, spec_metadata, top_k, full_vocab):
        return self._ids, self._unary, None


def test_the_dflash2_sampled_writer_stores_the_same_rows():
    (body,) = _compile_methods(
        _SPEC / "dflash.py", "DFlashWorker", ("_sample_dflash2_block",), {"torch": torch}
    ).values()
    gen = torch.Generator().manual_seed(21)
    requests, num_contexts = 5, 2
    ids, proposal = _random_rows(gen, requests)
    tokens = torch.randint(0, VOCAB, (requests, K), generator=gen)
    draft_model = SimpleNamespace(
        candidate_selector=SimpleNamespace(
            predecessor_codebook=torch.zeros((VOCAB, 1)), top_k=TOP_K
        ),
        sample_candidate_path=lambda *args: (tokens, proposal),
    )
    worker = _DFlashStandIn(ids, proposal.log())
    slot_ids = torch.tensor([10, 3, 0, 7, 11, 5, 2])
    stores = []
    for compact in (False, True):
        meta = SimpleNamespace(
            draft_probs=None if compact else torch.zeros((SLOTS + 1, K, VOCAB)),
            compact_draft_probs=(
                cdp.CompactDraftProbs(SLOTS + 1, K, VOCAB, device="cpu") if compact else None
            ),
            draft_probs_last_dim=0,
            batch_slot_ids=slot_ids,
            request_temperatures=torch.ones(8),
            request_top_ks=torch.zeros(8, dtype=torch.int32),
            request_top_ps=torch.ones(8),
            request_min_ps=torch.zeros(8),
            advanced_sampling_mode=None,
        )
        out_tokens, vocab = body(
            worker,
            draft_model,
            torch.empty((requests, K, VOCAB)),
            torch.empty((requests, K, 4)),
            torch.zeros(requests, dtype=torch.int64),
            meta,
            num_contexts,
            num_contexts + requests,
        )
        assert vocab == VOCAB and torch.equal(out_tokens, tokens.to(torch.int32))
        assert meta.draft_probs_last_dim == VOCAB
        stores.append(meta)
    gen_slots = slot_ids[num_contexts : num_contexts + requests]
    dense_rows = stores[0].draft_probs[gen_slots]
    rebuilt = stores[1].compact_draft_probs.densify(gen_slots, K, VOCAB)
    assert torch.equal(_bits(rebuilt), _bits(dense_rows))


# ---- 5. a mixed batch: the context rows are sampled before the rows take the logits' storage ----


def test_a_mixed_batch_samples_the_context_rows_before_the_rows_reuse_the_logits():
    dense, compact, written = _run_writes(41)
    gen = torch.Generator().manual_seed(41)
    contexts, gens = 3, 5
    perm = torch.tensor(written)[torch.randperm(len(written), generator=gen)]
    order = perm[:gens]
    slot_ids = torch.cat([perm[gens : gens + contexts], order])
    draft_tokens = _draft_tokens_from(dense[order, :K, :VOCAB], gen)
    base = torch.log(dense[order, :K, :VOCAB].clamp_min(1e-6))
    gen_logits = torch.cat([base, torch.randn((gens, 1, VOCAB), generator=gen)], dim=1)
    gen_logits = gen_logits + 0.7 * torch.randn((gens, K + 1, VOCAB), generator=gen)
    ctx_logits = torch.randn((contexts, VOCAB), generator=gen)
    logits = torch.cat([ctx_logits, gen_logits.reshape(gens * (K + 1), VOCAB)]).contiguous()
    rows = contexts + gens * (K + 1)

    def meta(dense_store, compact_store):
        m = _metadata(dense_store, compact_store, slot_ids, gens)
        m.temperatures = torch.full((rows,), 0.9)
        m.top_ks = torch.zeros(rows, dtype=torch.int32)
        m.top_ps = torch.ones(rows)
        m.min_ps = torch.zeros(rows)
        m.request_seeds = torch.arange(contexts + gens, dtype=torch.int64) * 7919 + 13
        return m

    worker = _make_worker()
    accepted_d, counts_d = worker._accept_draft_tokens(
        logits.clone(), draft_tokens, contexts, contexts + gens, meta(dense, None)
    )
    private = cdp.mark_private_verify_logits(logits.clone())
    accepted_c, counts_c = worker._accept_draft_tokens(
        private, draft_tokens, contexts, contexts + gens, meta(None, compact)
    )
    # A context row defines only its first token (the rest is torch.empty in the built-in).
    assert torch.equal(accepted_d[:contexts, 0], accepted_c[:contexts, 0])
    assert torch.equal(accepted_d[contexts:], accepted_c[contexts:])
    assert torch.equal(counts_d, counts_c)
    assert int(counts_d[contexts:].min()) < int(counts_d[contexts:].max())
    # The context rows' tokens come from the logits before the rebuilt rows overwrite them.
    assert torch.equal(accepted_c[:contexts, 0], ctx_logits.argmax(dim=-1).to(torch.int32))
    head = private.reshape(-1)[: gens * K * VOCAB].view(gens, K, VOCAB)
    assert torch.equal(_bits(head), _bits(dense[order, :K, :VOCAB]))


# ---- 6. the built-in one-hot writers and the allocation, compiled from interface.py --------------


def _compile_functions(path: Path, names, namespace: dict) -> dict:
    """The named module-level functions of ``path``, compiled into ``namespace``."""
    tree = ast.parse(path.read_text())
    found = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    }
    missing = set(names) - set(found)
    assert not missing, f"{path.name} lacks {sorted(missing)}"
    module = ast.Module(body=list(found.values()), type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


class _HostTorch:
    """``torch`` with the ``device='cuda'`` allocations of the compiled bodies on the host."""

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def empty(*args, device=None, **kwargs):
        return torch.empty(*args, **kwargs)

    @staticmethod
    def zeros(*args, device=None, **kwargs):
        return torch.zeros(*args, **kwargs)

    @staticmethod
    def tensor(*args, device=None, **kwargs):
        return torch.tensor(*args, **kwargs)


class _HostCompactDraftProbs(cdp.CompactDraftProbs):
    def __init__(self, *args, device=None, **kwargs):
        super().__init__(*args, device="cpu", **kwargs)


class _Logger:
    def __init__(self):
        self.lines = []

    def info(self, message):
        self.lines.append(("info", message))

    def warning(self, message):
        self.lines.append(("warning", message))


def _interface_namespace():
    namespace = {
        "torch": _HostTorch(),
        "CompactDraftProbs": _HostCompactDraftProbs,
        "compact_draft_probs_requested": cdp.compact_draft_probs_requested,
        "compact_store_refusal": cdp.compact_store_refusal,
        "logger": _Logger(),
        "_COMPACT_DRAFT_PROBS_LOGGED": False,
        # TRTLLM_DISABLE_ONEHOT_INDEX_FILL unset: the index_fill writers.
        "_ONEHOT_INDEXED_PUT": False,
        # Host-only test: the padding writer's slot upload is not pinned.
        "prefer_pinned": lambda: False,
    }
    _compile_functions(_SPEC / "interface.py", ("_wants_compact_draft_probs",), namespace)
    return namespace


def _alloc_meta(**over):
    fields = dict(
        use_rejection_sampling=True,
        draft_probs=None,
        compact_draft_probs=None,
        vocab_size=VOCAB,
        draft_vocab_size=VOCAB,
        max_num_requests=SLOTS,
        max_draft_len=K,
        num_seq_slots=0,
        batch_slot_ids=None,
        full_draft_probs=None,
        draft_probs_vocab_size=0,
        draft_probs_last_dim=0,
        dummy_slot_row=0,
        spec_dec_mode=SimpleNamespace(is_dflash=lambda: True),
    )
    fields.update(over)
    return SimpleNamespace(**fields)


def test_the_allocation_is_dense_unless_the_knob_asks_for_records(monkeypatch):
    namespace = _interface_namespace()
    (prepare,) = _compile_methods(
        _SPEC / "interface.py", "SpecMetadata", ("prepare_rejection_sampling_buffers",), namespace
    ).values()

    monkeypatch.delenv(cdp.ENV, raising=False)
    meta = _alloc_meta()
    prepare(meta)
    assert meta.compact_draft_probs is None
    assert tuple(meta.draft_probs.shape) == (SLOTS + 1, K, VOCAB)
    assert meta.draft_probs.dtype == torch.float32
    assert meta.dummy_slot_row == SLOTS and meta.draft_probs_vocab_size == VOCAB
    assert namespace["logger"].lines == []  # knob unset: nothing logged

    monkeypatch.setenv(cdp.ENV, "1")
    meta = _alloc_meta()
    prepare(meta)
    assert meta.draft_probs is None and meta.compact_draft_probs is not None
    assert meta.compact_draft_probs.shape == (SLOTS + 1, K, VOCAB)
    assert meta.dummy_slot_row == SLOTS and meta.draft_probs_vocab_size == VOCAB
    prepare(meta)  # idempotent: the records are kept and no dense store appears
    assert meta.draft_probs is None
    assert [kind for kind, _ in namespace["logger"].lines] == ["info"]

    for refused in (
        _alloc_meta(spec_dec_mode=SimpleNamespace(is_dflash=lambda: False)),
        _alloc_meta(draft_vocab_size=VOCAB // 2),
    ):
        prepare(refused)
        assert refused.compact_draft_probs is None and refused.draft_probs is not None
    rejection_off = _alloc_meta(use_rejection_sampling=False)
    prepare(rejection_off)
    assert rejection_off.draft_probs is None and rejection_off.compact_draft_probs is None


@pytest.mark.parametrize("seed", [0, 1])
def test_the_built_in_one_hot_writers_store_the_dense_rows(seed):
    namespace = _interface_namespace()
    (padding,) = _compile_methods(
        _SPEC / "interface.py", "SpecMetadata", ("write_padding_onehot_draft_probs",), namespace
    ).values()
    (context,) = _compile_methods(
        _SPEC / "interface.py", "SpecWorkerBase", ("write_context_onehot_draft_probs",), namespace
    ).values()
    gen = torch.Generator().manual_seed(seed)
    slot_ids = torch.randperm(SLOTS, generator=gen)
    contexts = 4
    metas = []
    for compact in (False, True):
        meta = _alloc_meta(
            draft_probs=None if compact else torch.full((SLOTS + 1, K, VOCAB), float("nan")),
            compact_draft_probs=(
                cdp.CompactDraftProbs(SLOTS + 1, K, VOCAB, device="cpu") if compact else None
            ),
            draft_probs_vocab_size=VOCAB,
            batch_slot_ids=slot_ids,
        )
        # A pure-context step publishes the store's width; a mixed one reuses the gen width.
        context(None, meta, contexts, 0, K)
        assert meta.draft_probs_last_dim == VOCAB
        context(None, meta, contexts, 3, K - 1, VOCAB)
        padding(meta, [int(s) for s in slot_ids[contexts : contexts + 3]], K)
        metas.append(meta)
    read = slot_ids[: contexts + 3]
    rebuilt = metas[1].compact_draft_probs.densify(read, K, VOCAB)
    assert torch.equal(_bits(rebuilt), _bits(metas[0].draft_probs[read]))
