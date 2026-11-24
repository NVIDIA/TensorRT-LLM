# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sampled step's agreement (``DFlashWorker._agree_proposals``): every TP16 rank verifies
rank 0's drafts against rank 0's proposals.

One process plays the 16 ranks. Each rank has its own worker and proposal store and its own
``ZeroPaddedReduce`` over one shared all-reduce: rank 0 (the only writing rank) is reduced first
and every rank gets rank 0's row plus its own (all-zero) row, which is what the MNNVL all-reduce
returns for a broadcast. Checked:

* the packed row at two requests: ``[drafts 7 | ids 7 x 32 | q 7 x 32]``, 455 FP32 columns of
  the 2048, ids by value and ``q`` bit for bit; rank 0 clears the proposal columns after the
  reduce, so the next greedy step stages the row it staged before (7 columns);
* the negative control: one rank's walk is made to diverge (its draft seeds, or its ``q``).
  Agreeing on the tokens only, that rank commits other tokens than rank 0; with the proposals
  packed in, every rank's store equals rank 0's bit for bit (ids, ``q``, tokens), every rank
  commits rank 0's tokens and counts, and only the diverged rank's ``K31_RS_AGREE`` total counts
  the rows that differed;
* the all-gather fallback (no broadcast bound) keeps the store and drafts of a one-rank group;
* the K31_RS_AGREE readout (pinned copy, logged once complete) and the K31_RS_PATH line;
* the K31_RS_AGREE_HASH lines (step=first at the first readout with an agreed row, step=late
  ``_RS_HASH_LATE_STEPS`` later): the store digest, equal on the 16 ranks after the packed
  agreement and different on the diverged rank after a tokens-only one.
"""

import importlib
import re
from types import SimpleNamespace

import pytest
import torch

__extra_import_path__ = [".."]
from _rubin_rs_rig import (  # noqa: E402
    TOP_K,
    VOCAB,
    Bufs,
    K,
    Markov,
    build_worker,
    generate,
    sampled_spec,
    stub_drafter,
)

_PKG = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31."
    "kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1"
)
WORLD = 16
DIVERGENT = 5
_WIDTH = K + 2 * K * TOP_K  # 455

_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 3), (10, 7)),
    reason="the DFlash2 sampled path's kernels are certified on sm_103 / sm_107",
)


def _mod():
    return importlib.import_module(_PKG + ".modeling")


class _Broadcast:
    """The group's all-reduce for a rank-0 broadcast, ranks called in order 0, 1, ...: rank 0's
    call records its row and every call returns the group's sum, rank 0's row plus the other
    ranks' (all-zero) rows."""

    def __init__(self):
        self.staged = []
        self.total = None

    def for_rank(self, rank):
        def all_reduce(buf):
            if rank == 0:
                self.total = buf.clone()
                self.staged.append(buf.clone())
                return self.total.clone()
            assert not buf.any(), "a rank other than 0 wrote into the broadcast row"
            return self.total + buf

        return all_reduce


def _agreement(mod, broadcast, rank, rows, device):
    return mod.ZeroPaddedReduce(
        broadcast.for_rank(rank),
        0 if rank == 0 else None,
        K,
        torch.zeros(rows, mod._DRAFT_AGREEMENT_WIDTH, dtype=torch.float32, device=device),
    )


# --- the packed row --------------------------------------------------------------------------


@_gpu
def test_packed_row_at_two_requests_and_the_greedy_row_after_it():
    """On CUDA: the greedy row's unpack is a Triton kernel."""
    mod = _mod()
    dev = "cuda"
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_sampling import (  # noqa: E501
        DFlash2ProposalStore,
    )

    g = torch.Generator().manual_seed(4)
    worker = object.__new__(mod.DFlashWorker)
    store = DFlash2ProposalStore(6, K, TOP_K, dev)
    store.candidate_ids.copy_(
        torch.randint(0, 163840, store.candidate_ids.shape, generator=g).to(dev)
    )
    store.candidate_ids[3, 0, 0] = 163839
    store.q.copy_(torch.softmax(4 * torch.randn(store.q.shape, generator=g), dim=-1).to(dev))
    store.q[3, 1, :4] = torch.tensor(
        [0.0, 1e-30, 1e-40, 5e-45], device=dev
    )  # zero, tiny, subnormals
    worker._dflash2_proposals = store
    worker._rs_agree_count = torch.zeros(2, dtype=torch.int64, device=dev)
    broadcast = _Broadcast()
    worker._draft_agreement = _agreement(mod, broadcast, 0, 4, dev)
    slots = torch.tensor([3, 1], device=dev)
    drafts = torch.randint(0, 163840, (2, K), dtype=torch.int32, generator=g).to(dev)
    store.tokens[slots] = drafts.long()
    ids, q = store.candidate_ids[slots].clone(), store.q[slots].clone()
    worker._rs_agree_rows = (slots, 0, None)
    out = {
        "next_draft_tokens": drafts.clone(),
        "next_new_tokens": torch.zeros(2, K + 1, device=dev).int(),
    }
    worker._agree_draft_tokens(out)

    row = broadcast.staged[0]
    assert row.shape == (2, mod._DRAFT_AGREEMENT_WIDTH) and _WIDTH == 455
    assert torch.equal(row[:, :K].to(torch.int32), drafts)
    assert torch.equal(row[:, K : K + K * TOP_K].to(torch.int64), ids.view(2, -1))
    assert torch.equal(
        row[:, K + K * TOP_K : _WIDTH].view(torch.int32), q.view(2, -1).view(torch.int32)
    )
    assert not row[:, _WIDTH:].any(), "the packed row uses 455 columns"
    assert not worker._draft_agreement.buffer[:, K:].any(), "rank 0 left proposal columns set"
    assert worker._rs_agree_rows is None
    assert worker._rs_agree_count.tolist() == [0, 2]
    assert torch.equal(store.q[slots].view(torch.int32), q.view(torch.int32))
    assert torch.equal(out["next_draft_tokens"], drafts)

    # The next (greedy) step stages the row the agreement staged before the sampled path existed.
    greedy = torch.randint(0, 163840, (2, K), dtype=torch.int32, generator=g).to(dev)
    worker._agree_draft_tokens(
        {"next_draft_tokens": greedy, "next_new_tokens": torch.zeros(2, K + 1, device=dev).int()}
    )
    expected = torch.zeros(2, mod._DRAFT_AGREEMENT_WIDTH, device=dev)
    expected[:, :K] = greedy
    assert torch.equal(broadcast.staged[1], expected)
    assert worker._rs_agree_count.tolist() == [0, 2]


def test_one_rank_fallback_keeps_the_store():
    mod = _mod()
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_sampling import (  # noqa: E501
        DFlash2ProposalStore,
    )

    worker = object.__new__(mod.DFlashWorker)
    worker.mapping = SimpleNamespace(
        has_cp_helix=lambda: False, tp_size=1, tp_group_shares_requests=True
    )
    store = DFlash2ProposalStore(4, K, TOP_K, "cpu")
    store.candidate_ids.copy_(torch.randint(0, VOCAB, store.candidate_ids.shape))
    store.q.copy_(torch.softmax(torch.randn(store.q.shape), dim=-1))
    drafts = torch.randint(0, VOCAB, (3, K), dtype=torch.int32)
    slots = torch.tensor([2, 0])
    store.tokens[slots] = drafts[1:].long()
    before = (store.candidate_ids.clone(), store.q.clone(), store.tokens.clone())
    worker._dflash2_proposals = store
    worker._rs_agree_count = torch.zeros(2, dtype=torch.int64)
    worker._draft_agreement = None
    worker._rs_agree_rows = (slots, 1, None)
    out = {"next_draft_tokens": drafts.clone(), "next_new_tokens": torch.zeros(3, K + 1).int()}
    worker._agree_draft_tokens(out)
    assert torch.equal(out["next_draft_tokens"], drafts)
    assert torch.equal(out["next_new_tokens"][:, 1:], drafts)
    for a, b in zip(before, (store.candidate_ids, store.q, store.tokens)):
        assert torch.equal(a, b)
    assert worker._rs_agree_count.tolist() == [0, 2]


# --- the negative control --------------------------------------------------------------------


def _store_bits(worker, slots):
    store = worker._dflash2_proposals
    return (
        store.candidate_ids[slots].clone(),
        store.q[slots].view(torch.int32).clone(),
        store.tokens[slots].clone(),
    )


def _run_ranks(diverge: str, pack: bool, n: int = 16, workers=None):
    """Every rank drafts (the DIVERGENT rank's walk perturbed), agrees, then verifies rank 0's
    drafts. Returns per rank: (pre-agreement store bits, post-agreement store bits, accepted,
    num_accepted, agreement totals); ``workers`` (a list) gets the 16 workers."""
    mod = _mod()
    drafter = stub_drafter()
    markov = Markov()
    bufs = Bufs.make(n, rounds=1)
    bufs.last.copy_(torch.arange(n, device="cuda") % VOCAB)
    broadcast = _Broadcast()
    attn = SimpleNamespace(num_seqs=n, num_contexts=0)
    ranks = []
    for rank in range(WORLD):
        worker = build_worker("helix16", drafter)
        worker._draft_agreement = _agreement(mod, broadcast, rank, n, "cuda")
        spec = sampled_spec(n, seed=1234)
        ranks.append((worker, spec))
    slots = ranks[0][1].batch_slot_ids
    pre, drafts = [], []
    for rank, (worker, spec) in enumerate(ranks):
        if rank == DIVERGENT and diverge == "seed":
            spec.request_seeds.add_(1)
        gen_logits = markov.draft_base[bufs.last]
        d = worker._sample_block_path(drafter, gen_logits, bufs.hidden, bufs.last, spec, 0, n)
        if rank == DIVERGENT and diverge == "seed":
            spec.request_seeds.sub_(1)
        if rank == DIVERGENT and diverge == "q":
            store = worker._dflash2_proposals
            flat = store.q[slots].sqrt()
            store.q[slots] = flat / flat.sum(-1, keepdim=True)
        pre.append(_store_bits(worker, slots))
        drafts.append(d)
    results = []
    agreed0 = None
    for rank, (worker, spec) in enumerate(ranks):
        if not pack:
            worker._rs_agree_rows = None  # the tokens-only agreement
        out = {
            "next_draft_tokens": drafts[rank].clone(),
            "next_new_tokens": torch.cat([bufs.last[:, None].int(), drafts[rank]], dim=1),
        }
        worker._agree_draft_tokens(out)
        agreed = out["next_draft_tokens"]
        if rank == 0:
            agreed0 = agreed.clone()
        assert torch.equal(agreed, agreed0), "the tokens did not agree"
        assert not worker._draft_agreement.buffer[:, K:].any()
        spec.draft_tokens.copy_(agreed.reshape(-1))
        prefix = torch.cat((bufs.last[:, None], agreed.long()), dim=1)
        logits = markov.log_trans[prefix].view(n * (K + 1), VOCAB).contiguous()
        accepted, num = worker._verify_sampled_block(logits, attn, spec)
        count = worker._rs_agree_count
        results.append(
            (
                pre[rank],
                _store_bits(worker, slots),
                accepted.clone(),
                num.clone(),
                None if count is None else count.clone(),
            )
        )
    torch.cuda.synchronize()
    if workers is not None:
        workers.extend(worker for worker, _ in ranks)
    return results


@_gpu
@pytest.mark.parametrize("diverge", ["seed", "q"])
def test_tokens_only_agreement_lets_a_diverged_rank_commit_other_tokens(diverge):
    results = _run_ranks(diverge, pack=False)
    _, _, acc0, num0, _ = results[0]
    for rank, (_, _, acc, num, _) in enumerate(results):
        same = torch.equal(acc, acc0) and torch.equal(num, num0)
        if rank == DIVERGENT:
            assert not same, "the diverged rank committed rank 0's tokens without the pack-back"
        else:
            assert same, f"rank {rank} differs from rank 0"


@_gpu
@pytest.mark.parametrize("diverge", ["seed", "q"])
def test_packed_agreement_gives_every_rank_rank_zeros_proposals(diverge):
    results = _run_ranks(diverge, pack=True)
    pre0, post0, acc0, num0, _ = results[0]
    differed = None
    for rank, (pre, post, acc, num, count) in enumerate(results):
        for a, b, name in zip(post, post0, ("candidate ids", "q bits", "tokens")):
            assert torch.equal(a, b), f"rank {rank}: store {name} differ from rank 0's"
        assert torch.equal(acc, acc0) and torch.equal(num, num0), (
            f"rank {rank} committed other tokens"
        )
        rows = (
            (pre[0] != pre0[0]).flatten(1).any(-1)
            | (pre[1] != pre0[1]).flatten(1).any(-1)
            | (pre[2] != pre0[2]).any(-1)
        )
        assert count.tolist() == [int(rows.sum()), num.numel()], (rank, count.tolist())
        if rank == DIVERGENT:
            differed = int(rows.sum())
        else:
            assert int(rows.sum()) == 0
    assert differed and differed > 0, "the negative control did not diverge"
    print(f"{diverge}: {differed} of {num0.numel()} rows differed on rank {DIVERGENT}")


# --- the readout lines -----------------------------------------------------------------------


@_gpu
def test_agreement_readout_and_path_lines(monkeypatch):
    mod = _mod()
    lines = []
    monkeypatch.setattr(mod.logger, "info", lambda msg, *a, **k: lines.append(str(msg)))
    worker = build_worker("helix16", stub_drafter())
    worker._report_rs_agreement()  # the sampled path never ran: nothing
    assert worker._rs_agree_pending is None and "_rs_agree_host" not in worker.__dict__
    n = 2
    spec = sampled_spec(n)
    bufs = Bufs.make(n, rounds=1)
    generate(worker, stub_drafter(), spec, Markov(), bufs, rounds=1)
    spec.is_cuda_graph = True
    generate(worker, stub_drafter(), spec, Markov(), bufs, rounds=1)
    path = [s for s in lines if s.startswith("K31_RS_PATH")]
    assert len(path) == 2 and "phase=eager" in path[0] and "phase=warmup" in path[1], path
    assert "block_verify=1" in path[0] and "gens=2" in path[0]
    worker._rs_agree_count.copy_(torch.tensor([3, 40]))
    worker._report_rs_agreement()
    torch.cuda.synchronize()
    worker._report_rs_agreement()
    agree = [s for s in lines if s.startswith("K31_RS_AGREE ")]
    assert agree == ["K31_RS_AGREE rank=0 mismatched=3 rows=40 steps=1"], agree


# --- the hash lines --------------------------------------------------------------------------

_HASH_LINE = re.compile(
    r"K31_RS_AGREE_HASH rank=(\d+) step=(first|late) h=([0-9a-f]{16}) steps=(\d+) rows=(\d+)$"
)


def _hash_lines(lines):
    return [_HASH_LINE.match(s).groups() for s in lines if s.startswith("K31_RS_AGREE_HASH")]


def _expected_digest(mod, worker):
    store = worker._dflash2_proposals
    n = store.num_rows - 1
    h = int(mod._rs_digest(store.candidate_ids[:n], store.q[:n], store.tokens[:n]))
    return f"{h & 0xFFFFFFFFFFFFFFFF:016x}"


@_gpu
def test_agree_hash_first_and_late_lines(monkeypatch):
    mod = _mod()
    lines = []
    monkeypatch.setattr(mod.logger, "info", lambda msg, *a, **k: lines.append(str(msg)))
    worker = build_worker("helix16", stub_drafter())
    n = 2
    generate(worker, stub_drafter(), sampled_spec(n), Markov(), Bufs.make(n, rounds=1), rounds=1)
    torch.cuda.synchronize()
    lines.clear()
    count = worker._rs_agree_count
    count.zero_()  # nothing agreed yet: readouts every step, no hash line
    worker._report_rs_agreement()  # step 1
    torch.cuda.synchronize()
    count.copy_(torch.tensor([0, 5]))
    worker._report_rs_agreement()  # step 2: step 1's readout (rows=0); step 2's taken
    assert not _hash_lines(lines)
    torch.cuda.synchronize()
    worker._report_rs_agreement()  # step 3: step 2's readout (the every-step one), rows=5
    expected = _expected_digest(mod, worker)
    assert _hash_lines(lines) == [("0", "first", expected, "2", "5")], lines
    assert [s for s in lines if s.startswith("K31_RS_AGREE ")] == [
        "K31_RS_AGREE rank=0 mismatched=0 rows=0 steps=1"
    ]
    # Readouts back to every _RS_AGREE_PERIOD steps; step=late at the first one 2000 steps on.
    worker._dflash2_proposals.q[0, 0, 0] += 1.0
    late = _expected_digest(mod, worker)
    assert late != expected, "the digest missed a changed q word"
    lines.clear()
    for _ in range(mod._RS_HASH_LATE_STEPS + 2 * mod._RS_AGREE_PERIOD):
        worker._report_rs_agreement()
        torch.cuda.synchronize()
    steps = 2 + mod._RS_HASH_LATE_STEPS  # the first periodic readout from there
    while steps % mod._RS_AGREE_PERIOD != 1:
        steps += 1
    assert _hash_lines(lines) == [("0", "late", late, str(steps), "5")], _hash_lines(lines)
    agree = [s for s in lines if s.startswith("K31_RS_AGREE ")]
    assert (
        len(agree) == (mod._RS_HASH_LATE_STEPS + 2 * mod._RS_AGREE_PERIOD) // mod._RS_AGREE_PERIOD
    )


@_gpu
@pytest.mark.parametrize("pack", [True, False])
def test_agree_hash_lines_equal_across_ranks_only_with_the_pack(pack, monkeypatch):
    mod = _mod()
    workers = []
    _run_ranks("q", pack=pack, workers=workers)
    hashes = []
    for worker in workers:
        lines = []
        monkeypatch.setattr(mod.logger, "info", lambda msg, *a, **k: lines.append(str(msg)))
        if not pack:
            worker._rs_agree_count = torch.tensor([0, 16], device="cuda")
        worker._report_rs_agreement()
        torch.cuda.synchronize()
        worker._report_rs_agreement()
        (line,) = _hash_lines(lines)
        assert line[1] == "first" and line[4] == "16"
        hashes.append(line[2])
    for rank, h in enumerate(hashes):
        if pack or rank != DIVERGENT:
            assert h == hashes[0], f"rank {rank}: {h} != {hashes[0]}"
        else:
            assert h != hashes[0], "the diverged rank's store hashed equal to rank 0's"
