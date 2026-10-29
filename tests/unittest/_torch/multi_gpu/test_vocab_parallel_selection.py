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
"""Vocab-parallel token selection must match selection on gathered logits.

Every rank holds a vocab shard of the logits; the vocab-parallel argmax and the
DSpark Markov chain must produce the same token ids as ``torch.argmax`` over the
gathered logits (ties to the lowest global index, NaN first), on every rank,
eagerly and under CUDA-graph replay, for both pair-exchange backends.
"""

import os
import pickle
import sys
import traceback
from types import SimpleNamespace

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

pytestmark = pytest.mark.threadleak(enabled=False)

HIDDEN = 256
SHARD = 3000  # not a multiple of the 2048-wide kernel tile


def _run(fn, *args):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        fn(rank, *args)
    except Exception:
        traceback.print_exc()
        raise
    return True


def _make_head(rank, tp, comm_backend):
    os.environ["TLLM_SPEC_VOCAB_PARALLEL_COMM"] = comm_backend
    os.environ["TLLM_TEST_MNNVL"] = "1"
    from tensorrt_llm._torch.modules.embedding import LMHead
    from tensorrt_llm._torch.modules.linear import TensorParallelMode
    from tensorrt_llm._torch.speculative.vocab_parallel import VocabParallelHead

    mapping = Mapping(world_size=tp, tp_size=tp, rank=rank)
    vocab = SHARD * tp
    lm_head = LMHead(
        vocab,
        HIDDEN,
        dtype=torch.bfloat16,
        mapping=mapping,
        tensor_parallel_mode=TensorParallelMode.COLUMN,
        gather_output=True,
    ).cuda()
    gen = torch.Generator(device="cuda").manual_seed(7)
    full = (torch.randn(vocab, HIDDEN, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    lm_head.weight.data.copy_(full[rank * SHARD : (rank + 1) * SHARD])
    assert VocabParallelHead.supported(lm_head, vocab)
    head = VocabParallelHead(lm_head, vocab)
    assert head.exchange.backend == comm_backend
    return head, vocab


def _special_rows(full):
    """Overwrite rows with ties across shards, all -inf, NaN and +inf."""
    vocab = full.shape[-1]
    full[0] = 1.0  # all equal -> index 0
    full[1] = -float("inf")  # all -inf -> index 0
    full[2, vocab - 1] = float("nan")
    full[2, SHARD + 5] = float("nan")  # first NaN wins
    full[3, 2 * SHARD - 1] = float("inf")
    full[3, vocab - 2] = float("inf")
    full[4] = 0.0
    full[4, vocab - 1] = 2.0
    full[4, 1] = 2.0  # tie across shards -> lower rank
    return full


@torch.inference_mode()
def _argmax_check(rank, tp, comm_backend):
    head, vocab = _make_head(rank, tp, comm_backend)
    for rows in (1, 7, 64, 300):
        gen = torch.Generator(device="cuda").manual_seed(rows)
        full = torch.randint(-4, 4, (rows, vocab), device="cuda", generator=gen).to(torch.bfloat16)
        if rows >= 5:
            full = _special_rows(full)
        local = full[:, rank * SHARD : (rank + 1) * SHARD].contiguous()
        expected = torch.argmax(full.float(), dim=-1)
        assert torch.equal(head.argmax(local), expected)
        assert torch.equal(head.argmax(local, out_dtype=torch.int32), expected.to(torch.int32))
        # Gathered logits round-trip exactly.
        assert torch.equal(
            head.gather_full_logits(local).view(torch.int32), full.float().view(torch.int32)
        )
    # Real LM head + CUDA graph replay.
    hidden = torch.randn(48, HIDDEN, device="cuda").to(torch.bfloat16)
    expected = torch.argmax(head.lm_head(hidden).float(), dim=-1)
    out = head.argmax(head.local_logits(hidden))
    assert torch.equal(out, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_g = head.argmax(head.local_logits(hidden))
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out_g, expected)


def _markov_reference(full, anchors, w1, w2):
    from tensorrt_llm._torch.models.modeling_speculative import dspark_markov_chain_logits

    def plain_argmax(x):
        return torch.argmax(x, dim=-1).type(torch.int32).long()

    biased = dspark_markov_chain_logits(full.float(), anchors, w1, w2, argmax_fn=plain_argmax)
    return torch.argmax(biased, dim=-1).to(torch.int32)


@torch.inference_mode()
def _markov_check(rank, tp, comm_backend):
    head, vocab = _make_head(rank, tp, comm_backend)
    gen = torch.Generator(device="cuda").manual_seed(11)
    w1 = torch.randn(vocab, 256, device="cuda", generator=gen).to(torch.bfloat16)
    w2 = (torch.randn(vocab, 256, device="cuda", generator=gen) * 0.1).to(torch.bfloat16)
    for num_reqs, block in ((1, 1), (1, 7), (4, 7), (32, 6), (128, 6)):
        full = (torch.randn(num_reqs, block, vocab, device="cuda", generator=gen) * 3).to(
            torch.bfloat16
        )
        anchors = torch.randint(0, vocab, (num_reqs,), device="cuda", generator=gen)
        anchors[0] = -1 if num_reqs > 1 else anchors[0]  # out-of-vocab anchor -> zero bias
        local = full[..., rank * SHARD : (rank + 1) * SHARD].contiguous()
        expected = _markov_reference(full, anchors, w1, w2)
        assert torch.equal(head.markov_tokens(local, anchors, w1, w2), expected)
        # Without a Markov head: the per-position argmax.
        plain = head.markov_tokens(local, anchors, None, None)
        assert torch.equal(plain, torch.argmax(full.float(), dim=-1).to(torch.int32))
    # Ties: quantized logits and a zero bias make many cross-shard ties.
    full = torch.randint(0, 3, (16, 6, vocab), device="cuda", generator=gen).to(torch.bfloat16)
    anchors = torch.randint(0, vocab, (16,), device="cuda", generator=gen)
    zero_w2 = torch.zeros_like(w2)
    local = full[..., rank * SHARD : (rank + 1) * SHARD].contiguous()
    assert torch.equal(
        head.markov_tokens(local, anchors, w1, zero_w2),
        _markov_reference(full, anchors, w1, zero_w2),
    )
    # CUDA graph replay.
    full = (torch.randn(8, 6, vocab, device="cuda", generator=gen) * 3).to(torch.bfloat16)
    local = full[..., rank * SHARD : (rank + 1) * SHARD].contiguous()
    anchors = torch.randint(0, vocab, (8,), device="cuda", generator=gen)
    expected = _markov_reference(full, anchors, w1, w2)
    head.markov_tokens(local, anchors, w1, w2)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = head.markov_tokens(local, anchors, w1, w2)
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, expected)


@torch.inference_mode()
def _sample_check(rank, tp, comm_backend):
    from tensorrt_llm._torch.speculative.vocab_parallel import VocabShardedLogits

    head, vocab = _make_head(rank, tp, comm_backend)
    rows = 64
    gen = torch.Generator(device="cuda").manual_seed(3)
    full = (torch.randn(rows, vocab, device="cuda", generator=gen) * 2).to(torch.bfloat16)
    local = full[:, rank * SHARD : (rank + 1) * SHARD].contiguous()
    temps = torch.full((rows,), 1.0, device="cuda")
    temps[::4] = 1e-5  # greedy rows inside a sampled batch
    seeds = torch.full((rows,), 42, dtype=torch.int64, device="cuda")
    offsets = torch.arange(rows, dtype=torch.int64, device="cuda") * 8
    out = head.sample(local, temps, seeds, offsets, 1e-4)
    # Identical on every rank, deterministic, greedy rows are the argmax.
    gathered = MPI.COMM_WORLD.allgather(out.cpu())
    assert all(torch.equal(g, gathered[0]) for g in gathered)
    assert torch.equal(head.sample(local, temps, seeds, offsets, 1e-4), out)
    argmax = torch.argmax(full.float(), dim=-1)
    assert torch.equal(out[::4], argmax[::4])
    assert bool(((out >= 0) & (out < vocab)).all())
    # A different offset gives a different draw for the sampled rows.
    out2 = head.sample(local, temps, seeds, offsets + 1, 1e-4)
    assert not torch.equal(out2[1::4], out[1::4])
    # Distribution: one row, many independent draws, compare to softmax.
    logits = torch.zeros(1, vocab, device="cuda")
    logits[0, :8] = torch.tensor([2.0, 1.0, 0.5, 0.0, -1.0, 1.5, 0.25, 3.0])
    logits[0, 8:] = -float("inf")
    logits[0, SHARD : SHARD + 2] = torch.tensor([2.5, 0.75])
    local = (
        logits[:, rank * SHARD : (rank + 1) * SHARD].to(torch.bfloat16).repeat(4096, 1).contiguous()
    )
    n = local.shape[0]
    draws = head.sample(
        local,
        torch.full((n,), 0.7, device="cuda"),
        torch.full((n,), 5, dtype=torch.int64, device="cuda"),
        torch.zeros((n,), dtype=torch.int64, device="cuda"),
        1e-4,
    )
    counts = torch.bincount(draws, minlength=vocab).float()
    probs = torch.softmax(logits[0].to(torch.bfloat16).float() / 0.7, dim=-1)
    support = probs > 0
    assert counts[~support].sum() == 0
    expected = probs[support] * n
    chi2 = (((counts[support] - expected) ** 2) / expected).sum().item()
    assert chi2 < 40.0, chi2  # 9 degrees of freedom; p ~ 1e-5
    # Through the spec-worker wrapper.
    sharded = VocabShardedLogits(head, local[:rows])
    md = SimpleNamespace(
        is_all_greedy_sample=True, temperatures=temps, seeds=seeds, offsets=offsets
    )
    assert torch.equal(sharded.select_tokens(md), head.argmax(local[:rows]))
    md.is_all_greedy_sample = False
    assert torch.equal(
        sharded.select_tokens(md), head.sample(local[:rows], temps, seeds, offsets, 1e-4)
    )


@torch.inference_mode()
def _worker_accept_check(rank, tp, comm_backend):
    from tensorrt_llm._torch.speculative.interface import SpecWorkerBase
    from tensorrt_llm._torch.speculative.vocab_parallel import VocabShardedLogits

    class _Stub(SpecWorkerBase):
        @property
        def max_draft_len(self):
            return 8

        def _forward_impl(self, *args, **kwargs):
            raise NotImplementedError

    head, vocab = _make_head(rank, tp, comm_backend)
    num_gens, draft_len = 6, 6
    gen = torch.Generator(device="cuda").manual_seed(5)
    full = torch.randint(0, 5, (num_gens * (draft_len + 1), vocab), device="cuda", generator=gen)
    full = full.to(torch.bfloat16)
    target = torch.argmax(full.float(), dim=-1).view(num_gens, draft_len + 1)
    draft = target[:, :draft_len].clone().to(torch.int32)
    draft[1::2, 2] = (draft[1::2, 2] + 1) % vocab
    md = SimpleNamespace(is_all_greedy_sample=True, is_cuda_graph=False)
    local = full[:, rank * SHARD : (rank + 1) * SHARD].contiguous()
    for fused in ("0", "1"):
        os.environ["TLLM_SPEC_TAIL_FUSION"] = fused
        worker = _Stub()
        ref = worker._sample_and_accept_draft_tokens_base(full.float(), draft, 0, num_gens, md)
        out = worker._sample_and_accept_draft_tokens_base(
            VocabShardedLogits(head, local), draft, 0, num_gens, md
        )
        assert torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1])
    os.environ.pop("TLLM_SPEC_TAIL_FUSION", None)


@torch.inference_mode()
def _embedding_check(rank, tp, comm_backend):
    os.environ["TLLM_TEST_MNNVL"] = "1"
    from tensorrt_llm._torch.modules.embedding import Embedding
    from tensorrt_llm._torch.modules.linear import TensorParallelMode
    from tensorrt_llm._torch.speculative.spec_tail_kernels import vocab_shard_embedding

    mapping = Mapping(world_size=tp, tp_size=tp, rank=rank)
    vocab, hidden = 1000 * tp, 7168
    embedding = Embedding(
        vocab,
        hidden,
        dtype=torch.bfloat16,
        mapping=mapping,
        tensor_parallel_mode=TensorParallelMode.COLUMN,
        gather_output=True,
    ).cuda()
    gen = torch.Generator(device="cuda").manual_seed(9)
    full = torch.randn(vocab, hidden, device="cuda", generator=gen).to(torch.bfloat16)
    embedding.weight.data.copy_(full[rank * 1000 : (rank + 1) * 1000])
    for n in (1, 5, 129):
        ids = torch.randint(0, vocab, (n,), device="cuda", generator=gen)
        ids[0] = vocab - 3
        expected = embedding(ids)
        out = vocab_shard_embedding(embedding, ids)
        assert out is not None and torch.equal(out, expected)
        assert torch.equal(out, full[ids])


def _submit(executor, fn, comm_backend):
    tp = executor.num_workers
    results = executor.map(_run, *zip(*[(fn, tp, comm_backend)] * tp))
    for r in results:
        assert r is True


_needs_gpus = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
_WORLD_SIZES = [2, 4] if torch.cuda.device_count() >= 4 else [2]


@_needs_gpus
@pytest.mark.parametrize("comm_backend", ["nccl", "mnnvl"])
@pytest.mark.parametrize("mpi_pool_executor", _WORLD_SIZES, indirect=True)
def test_vocab_parallel_argmax(comm_backend, mpi_pool_executor):
    _submit(mpi_pool_executor, _argmax_check, comm_backend)


@_needs_gpus
@pytest.mark.parametrize("comm_backend", ["nccl", "mnnvl"])
@pytest.mark.parametrize("mpi_pool_executor", _WORLD_SIZES, indirect=True)
def test_vocab_parallel_markov_tokens(comm_backend, mpi_pool_executor):
    _submit(mpi_pool_executor, _markov_check, comm_backend)


@_needs_gpus
@pytest.mark.parametrize("comm_backend", ["nccl", "mnnvl"])
@pytest.mark.parametrize("mpi_pool_executor", _WORLD_SIZES, indirect=True)
def test_vocab_parallel_sample(comm_backend, mpi_pool_executor):
    _submit(mpi_pool_executor, _sample_check, comm_backend)


@_needs_gpus
@pytest.mark.parametrize("comm_backend", ["mnnvl"])
@pytest.mark.parametrize("mpi_pool_executor", _WORLD_SIZES, indirect=True)
def test_vocab_parallel_strict_accept(comm_backend, mpi_pool_executor):
    _submit(mpi_pool_executor, _worker_accept_check, comm_backend)


@_needs_gpus
@pytest.mark.parametrize("mpi_pool_executor", _WORLD_SIZES, indirect=True)
def test_vocab_shard_embedding(mpi_pool_executor):
    _submit(mpi_pool_executor, _embedding_check, "mnnvl")
