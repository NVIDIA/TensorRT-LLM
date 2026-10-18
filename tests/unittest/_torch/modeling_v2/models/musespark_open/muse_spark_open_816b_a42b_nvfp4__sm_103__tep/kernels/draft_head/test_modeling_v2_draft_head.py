# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Muse Spark Open tep8 draft head (fast path ``draft_head``, ``kernels/draft_head/head.py``).

The claim is that the sharded draft head gives the built-in draft tokens bit for
bit. The three entries it chains are gated on their cells in their own tests
(``test_modeling_v2_draft_{shard_topk,topk_merge,path_walk}.py``); here, eight
vocab shards of one set of draft logits are pushed through
``selector_draft_tokens`` (the key exchange emulated) and through the
re-classed DFlash worker, and compared with the built-in worker on the
gathered logits; then the logits-processor wrapper, CUDA-graph replay, the
install-time tie-order probe and the install itself.
"""

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.draft_head import (  # noqa: E501
    head as draft_head,
)

__extra_import_path__ = ["."]
from _draft_head_cases import (  # noqa: E402
    TOP_K,
    TP,
    VOCAB,
    DraftStub,
    builtin_draft_tokens,
    draft_shard_topk,
    draft_step,
    seed_of,
    selector,
)

assert torch.cuda.is_available(), "the vocab-sharded draft head requires a CUDA device"


# --------------------------------------------------------------------------- end to end


def _fake_allgather(shards, keys_by_rank, rank):
    def allgather(tensor, mapping, dim=-1, sizes=None):
        if dim == 0:
            assert torch.equal(tensor, keys_by_rank[rank]), "this rank's keys are not the shard's"
            return torch.cat(keys_by_rank, dim=0)
        assert torch.equal(tensor.reshape(-1, tensor.shape[-1]), shards[rank])
        return torch.cat(shards, dim=-1).view(*tensor.shape[:-1], -1)

    return allgather


class _FakeAllReduce:
    """The fabric all-reduce of the key halves, emulated: the sum of every rank's buffer."""

    def __init__(self, shards, rank):
        self.packed = [
            draft_shard_topk(s, rank=r, num_ranks=len(shards)) for r, s in enumerate(shards)
        ]
        self.rank = rank

    def __call__(self, tensor):
        assert torch.equal(tensor, self.packed[self.rank]), "this rank's halves are not the shard's"
        return torch.stack(self.packed).sum(dim=0)


@pytest.mark.parametrize("exchange", ["allgather", "allreduce"])
@pytest.mark.parametrize(
    "requests,positions,kind",
    [
        (1, 7, "gemm"),
        (1, 7, "grid"),
        (2, 7, "gemm"),
        (1, 3, "gemm"),
        (1, 1, "logits"),
        (1, 7, "zeros"),
    ],
)
def test_selector_draft_tokens_equal_builtin(requests, positions, kind, exchange, monkeypatch):
    shards, block_hidden, anchor = draft_step(
        requests, positions, kind, seed_of(f"{requests}{positions}{kind}")
    )
    expected = builtin_draft_tokens(shards, block_hidden, anchor)
    keys_by_rank = [draft_shard_topk(s) for s in shards]
    for rank in range(TP):
        mapping = types.SimpleNamespace(tp_size=TP, tp_rank=rank)
        monkeypatch.setattr(draft_head, "allgather", _fake_allgather(shards, keys_by_rank, rank))
        allreduce = _FakeAllReduce(shards, rank) if exchange == "allreduce" else None
        local = shards[rank].view(requests, positions, -1)
        got = draft_head.selector_draft_tokens(
            local, block_hidden, anchor, selector(), mapping, allreduce
        )
        assert torch.equal(got, expected), f"rank {rank}: {got.tolist()} != {expected.tolist()}"


def _workers():
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=TP, tp_size=TP, rank=0)
    config = types.SimpleNamespace(max_draft_len=7, attention_backend="VANILLA")
    builtin = DFlashWorker(config, mapping)
    draft_head_on = DFlashWorker(config, mapping)
    draft_head_on.__class__ = draft_head.ShardedDraftHeadDFlashWorker
    return builtin, draft_head_on


def _spec_metadata(advanced=False):
    return types.SimpleNamespace(
        wants_advanced_draft_sampling=advanced,
        is_all_greedy_sample=not advanced,
        draft_vocab_size=VOCAB,
        vocab_size=VOCAB,
    )


@pytest.mark.parametrize("requests,positions", [(1, 7), (2, 5)])
def test_reclassed_worker_drafts_the_builtin_tokens(requests, positions, monkeypatch):
    shards, block_hidden, anchor = draft_step(requests, positions, "gemm", 31 + requests)
    builtin, draft_head_on = _workers()
    stub, md = DraftStub(selector()), _spec_metadata()
    full = torch.cat(shards, dim=-1).float().view(requests, positions, VOCAB)
    expected = builtin.sample_draft_tokens(
        builtin._apply_dflash2_selector(stub, full, block_hidden, anchor, md), md, requests
    )
    keys_by_rank = [draft_shard_topk(s) for s in shards]
    for rank in (0, 5, 7):
        monkeypatch.setattr(draft_head, "allgather", _fake_allgather(shards, keys_by_rank, rank))
        block = draft_head_on._apply_dflash2_selector(
            stub, shards[rank].view(requests, positions, -1), block_hidden, anchor, md
        )
        assert block.shape == (requests, positions, VOCAB), (
            "the stand-in must carry the full vocab width"
        )
        tokens = draft_head_on.sample_draft_tokens(block, md, requests, num_contexts=0)
        assert tokens.dtype == torch.int32 and torch.equal(tokens, expected)
    # With the fabric key exchange the install attaches (the worker's mapping is rank 0).
    draft_head_on.__dict__["_draft_head_allreduce"] = _FakeAllReduce(shards, 0)
    block = draft_head_on._apply_dflash2_selector(
        stub, shards[0].view(requests, positions, -1), block_hidden, anchor, md
    )
    assert torch.equal(
        draft_head_on.sample_draft_tokens(block, md, requests, num_contexts=0), expected
    )


def test_reclassed_worker_falls_back_to_builtin_for_sampled_drafts(monkeypatch):
    shards, block_hidden, anchor = draft_step(1, 7, "gemm", 41)
    builtin, draft_head_on = _workers()
    stub, md = DraftStub(selector()), _spec_metadata(advanced=True)
    full = torch.cat(shards, dim=-1).float().view(1, 7, VOCAB)
    expected = builtin._apply_dflash2_selector(stub, full.clone(), block_hidden, anchor, md)
    monkeypatch.setattr(draft_head, "allgather", _fake_allgather(shards, None, 2))
    got = draft_head_on._apply_dflash2_selector(
        stub, shards[2].view(1, 7, -1), block_hidden, anchor, md
    )
    assert getattr(got, draft_head.TOKENS_ATTR, None) is None
    assert torch.equal(got, expected)


def test_reclassed_worker_passes_full_vocab_logits_through():
    shards, block_hidden, anchor = draft_step(1, 7, "logits", 43)
    builtin, draft_head_on = _workers()
    stub, md = DraftStub(selector()), _spec_metadata()
    full = torch.cat(shards, dim=-1).float().view(1, 7, VOCAB)
    a = builtin._apply_dflash2_selector(stub, full.clone(), block_hidden, anchor, md)
    b = draft_head_on._apply_dflash2_selector(stub, full.clone(), block_hidden, anchor, md)
    assert torch.equal(a, b)


def test_slot_ids_are_cached_outside_capture_only():
    builtin, draft_head_on = _workers()
    a = draft_head_on._draft_slot_ids(None, 1, 8, 7)
    assert torch.equal(a, builtin._draft_slot_ids(None, 1, 8, 7))
    assert draft_head_on._draft_slot_ids(None, 1, 8, 7) is a
    assert torch.equal(
        draft_head_on._draft_slot_ids(None, 3, 8, 7), builtin._draft_slot_ids(None, 3, 8, 7)
    )


def test_shard_logits_processor_is_the_gathered_heads_local_gemm(monkeypatch):
    """The wrapper returns what the column-parallel head computes before its all-gather."""
    from tensorrt_llm._torch import distributed as dist_pkg
    from tensorrt_llm._torch.modules.embedding import LMHead
    from tensorrt_llm._torch.modules.linear import TensorParallelMode
    from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor
    from tensorrt_llm.mapping import Mapping

    vocab, hidden = 4096, 1024
    head = LMHead(
        vocab,
        hidden,
        dtype=torch.bfloat16,
        mapping=Mapping(world_size=TP, tp_size=TP, rank=3),
        tensor_parallel_mode=TensorParallelMode.COLUMN,
        gather_output=True,
        reduce_output=False,
    ).cuda()
    with torch.no_grad():
        head.weight.copy_(torch.randn_like(head.weight, dtype=torch.float32).bfloat16())
    assert draft_head.gathers_column_shards(head)
    h = torch.randn(7, hidden, device="cuda").bfloat16()
    captured = {}

    def fake_allgather(output, mapping, dim=-1, sizes=None):
        captured["local"] = output.clone()
        return output

    monkeypatch.setattr(dist_pkg, "allgather", fake_allgather)
    processor = draft_head.ShardDraftLogitsProcessor(LogitsProcessor())
    shard = processor(h, head, None, True)
    builtin = LogitsProcessor()(h, head, None, True)  # the gathered forward, all-gather faked
    assert shard.dtype == torch.bfloat16 and shard.shape == (7, vocab // TP)
    assert torch.equal(shard, captured["local"])
    assert torch.equal(shard.float(), builtin)


# --------------------------------------------------------------------------- CUDA graph


@pytest.mark.parametrize("exchange", ["allgather", "allreduce"])
def test_cuda_graph_replay_follows_new_logits(exchange, monkeypatch):
    """``selector_draft_tokens`` captured once and replayed on new data."""
    sel = selector()
    requests, positions = 1, 7
    shards, block_hidden, anchor = draft_step(requests, positions, "gemm", 51)
    full = torch.cat(shards, dim=-1)
    shard = VOCAB // TP
    local = full[:, 2 * shard : 3 * shard]

    # Every rank's keys, recomputed from the captured buffer on each replay.
    def graph_allgather(tensor, mapping, dim=-1, sizes=None):
        assert dim == 0
        return torch.cat(
            [draft_shard_topk(full[:, r * shard : (r + 1) * shard]) for r in range(TP)], dim=0
        )

    def graph_allreduce(tensor):
        return torch.stack(
            [
                draft_shard_topk(full[:, r * shard : (r + 1) * shard], rank=r, num_ranks=TP)
                for r in range(TP)
            ]
        ).sum(dim=0)

    monkeypatch.setattr(draft_head, "allgather", graph_allgather)
    mapping = types.SimpleNamespace(tp_size=TP, tp_rank=2)
    allreduce = graph_allreduce if exchange == "allreduce" else None

    def step():
        return draft_head.selector_draft_tokens(
            local.view(requests, positions, shard), block_hidden, anchor, sel, mapping, allreduce
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step()
    for replay in range(4):
        new_shards, new_hidden, new_anchor = draft_step(
            requests, positions, ("gemm", "grid")[replay % 2], 60 + replay
        )
        full.copy_(torch.cat(new_shards, dim=-1))
        block_hidden.copy_(new_hidden)
        anchor.copy_(new_anchor)
        graph.replay()
        expected = builtin_draft_tokens(new_shards, new_hidden, new_anchor)
        assert torch.equal(out, expected), f"replay {replay}"


# --------------------------------------------------------------------------- tie-order probe


def _stable_topk(x, k, dim=-1):
    """A top-k that breaks ties by ascending index: torch's selection, another order."""
    values, ids = torch.sort(x, dim=dim, descending=True, stable=True)
    return values.narrow(dim, 0, k), ids.narrow(dim, 0, k)


def test_tie_order_probe_logits_hold_the_tie_families():
    probe = draft_head.tie_order_probe_logits(VOCAB, VOCAB // TP)
    assert probe.dtype == torch.bfloat16 and probe.shape == (draft_head.PROBE_ROWS, VOCAB)
    again = draft_head.tie_order_probe_logits(VOCAB, VOCAB // TP)
    assert torch.equal(probe.view(torch.int16), again.view(torch.int16)), "not deterministic"
    assert (probe[0] == 0).all() and not torch.signbit(probe[0]).any()
    assert torch.signbit(probe[1]).any() and not torch.signbit(probe[1]).all()
    assert probe[5].isinf().sum() == 20 and probe[6].isnan().sum() == 3
    edges = [c for r in range(TP) for c in (r * (VOCAB // TP), (r + 1) * (VOCAB // TP) - 1)]
    assert (probe[7, edges] == 7.0).all() and (probe[7] == 7.5).sum() == 3


def test_tie_order_probe_agrees_with_this_torch():
    assert draft_head.tie_order_reason(selector(), TP) is None


def test_tie_order_probe_catches_another_tie_order():
    """The probe's rows separate torch's order from a stable one, and the probe says so."""
    full = draft_head.tie_order_probe_logits(VOCAB, VOCAB // TP).cuda().float().view(1, -1, VOCAB)
    torch_values, torch_ids = torch.topk(full, TOP_K, dim=-1)
    stable_values, stable_ids = _stable_topk(full, TOP_K)
    differ = (torch_ids != stable_ids).any(dim=-1).view(-1)
    assert differ.sum() >= 4, "the probe must hold rows whose tie order is torch's alone"
    reason = draft_head.tie_order_reason(selector(), TP, topk=_stable_topk)
    assert reason is not None and "orders top-k ties differently" in reason


# --------------------------------------------------------------------------- install


def _head(gather_output=True, vocab=VOCAB):
    from tensorrt_llm._torch.modules.embedding import LMHead
    from tensorrt_llm._torch.modules.linear import TensorParallelMode
    from tensorrt_llm.mapping import Mapping

    return LMHead(
        vocab,
        64,
        dtype=torch.bfloat16,
        mapping=Mapping(world_size=TP, tp_size=TP, rank=0),
        tensor_parallel_mode=TensorParallelMode.COLUMN,
        gather_output=gather_output,
        reduce_output=False,
    )


def test_install_reclasses_the_worker_and_wraps_the_processor_once():
    from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker

    worker, _ = _workers()
    assert type(worker) is DFlashWorker
    inner = LogitsProcessor()
    drafter = types.SimpleNamespace(
        candidate_selector=selector(), logits_processor=inner, lm_head=_head()
    )
    llm = types.SimpleNamespace(spec_worker=worker, draft_model=drafter)
    built, keys = [], object()
    assert not draft_head.installed(llm)
    assert draft_head.install(llm, make_allreduce=lambda m: built.append(m) or keys)
    assert built == [worker.mapping] and worker.__dict__["_draft_head_allreduce"] is keys
    assert type(worker) is draft_head.ShardedDraftHeadDFlashWorker and draft_head.installed(llm)
    assert isinstance(drafter.logits_processor, draft_head.ShardDraftLogitsProcessor)
    assert drafter.logits_processor.inner is inner
    wrapper = drafter.logits_processor
    assert draft_head.install(llm, make_allreduce=lambda m: built.append(m) or keys)
    assert drafter.logits_processor is wrapper and len(built) == 1  # idempotent


def test_install_refuses_a_group_without_the_mnnvl_key_exchange():
    from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor

    worker, _ = _workers()
    drafter = types.SimpleNamespace(
        candidate_selector=selector(), logits_processor=LogitsProcessor(), lm_head=_head()
    )
    llm = types.SimpleNamespace(spec_worker=worker, draft_model=drafter)
    with pytest.raises(ValueError, match="MNNVL FP32 all-reduce of the key exchange"):
        draft_head.install(llm, make_allreduce=lambda m: None)


def test_install_refuses_a_head_it_cannot_shard():
    from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor

    uneven = _head()
    uneven.gather_output_sizes = [VOCAB // TP + 1] * (TP - 1) + [VOCAB // TP - (TP - 1)]
    assert not draft_head.gathers_column_shards(uneven)
    worker, _ = _workers()
    for head in (_head(gather_output=False), _head(vocab=TP * 4096), uneven):
        drafter = types.SimpleNamespace(
            candidate_selector=selector(), logits_processor=LogitsProcessor(), lm_head=head
        )
        llm = types.SimpleNamespace(spec_worker=worker, draft_model=drafter)
        with pytest.raises(ValueError, match="draft_head cannot run"):
            draft_head.install(llm, make_allreduce=lambda m: None)
        assert not draft_head.installed(llm)


def test_install_refuses_when_the_tie_order_probe_fails(monkeypatch):
    from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker

    monkeypatch.setattr(draft_head, "tie_order_reason", lambda sel, ranks: "probe rows [3] differ")
    worker, _ = _workers()
    inner = LogitsProcessor()
    drafter = types.SimpleNamespace(
        candidate_selector=selector(), logits_processor=inner, lm_head=_head()
    )
    llm = types.SimpleNamespace(spec_worker=worker, draft_model=drafter)
    built = []
    with pytest.raises(ValueError, match="probe rows"):
        draft_head.install(llm, make_allreduce=lambda m: built.append(m))
    assert type(worker) is DFlashWorker and drafter.logits_processor is inner and not built
