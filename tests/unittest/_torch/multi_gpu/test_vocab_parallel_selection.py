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
"""Vocab-parallel token selection must match selection on the gathered logits.

Every rank holds a vocab shard of the logits. The vocab-parallel argmax and the
DSpark Markov chain must give the ids ``torch.argmax`` gives on the gathered
logits (ties to the lowest global index, NaN first), and the Gumbel-max sampler
the draw a single shard holding the whole vocab gives, on every rank, eagerly
and under CUDA-graph replay, over the MNNVL and the default all-reduce.
"""

import os
import pickle
import platform
import sys
import traceback
from types import SimpleNamespace

import cloudpickle
import pytest
import torch
from _torch.helpers import dspark_markov_chain_logits_reference
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory
from tensorrt_llm._torch.nccl_window_graph import nccl_window_graph_capture
from tensorrt_llm._torch.nccl_window_tensor_scope import discard_nccl_window_tensor_outputs
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

pytestmark = pytest.mark.threadleak(enabled=False)

# The MNNVL all-reduce needs fabric-backed NVLink memory (GB200/GB300).
_MNNVL = pytest.param(
    True,
    id="mnnvl",
    marks=pytest.mark.skipif(
        platform.machine().lower() != "aarch64" or not MnnvlMemory.supports_mnnvl(),
        reason="needs fabric-backed MNNVL",
    ),
)
_EXCHANGES = [pytest.param(False, id="auto"), _MNNVL]

HIDDEN = 256
SHARD = 3000  # not a multiple of the 2048-wide kernel tile


def _run(fn, *args):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    # _make_head sets TLLM_TEST_MNNVL in this pooled MPI worker; later tasks on
    # the same worker must see the value it had before.
    saved_mnnvl = os.environ.get("TLLM_TEST_MNNVL")
    try:
        # The exchanges may return NCCL-window tensors, whose leases must end in a
        # tensor scope and whose graph captures need a window graph-pool owner (the
        # model engine and its CUDA-graph runner provide both). The scope is only
        # active with a CUDA tensor among its inputs.
        with discard_nccl_window_tensor_outputs(torch.empty(1, device="cuda")):
            fn(rank, *args)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if saved_mnnvl is None:
            os.environ.pop("TLLM_TEST_MNNVL", None)
        else:
            os.environ["TLLM_TEST_MNNVL"] = saved_mnnvl
    return True


def _make_head(rank, tp, mnnvl):
    # Selects MNNVL for the pair all-reduce on a single node.
    os.environ["TLLM_TEST_MNNVL"] = "1" if mnnvl else "0"
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
    assert VocabParallelHead.supports(lm_head)
    head = VocabParallelHead(lm_head)
    assert (head._exchange.all_reduce.mnnvl_allreduce is not None) == mnnvl
    return head, vocab


def _capture(fn, inputs):
    """A CUDA graph of ``fn()`` and its output, captured as the CUDA-graph runner does.

    An eager warm-up whose NCCL-window buffers are released first, then the capture
    under a window graph-pool owner, which reuses those buffers.
    """
    with discard_nccl_window_tensor_outputs(inputs):
        fn()
    graph = torch.cuda.CUDAGraph()
    with nccl_window_graph_capture(graph, torch.cuda.graph_pool_handle()):
        out = fn()
    return graph, out


def _shard(full, rank):
    return full[..., rank * SHARD : (rank + 1) * SHARD].contiguous()


def _special_rows(full):
    """Ties across shards, all -inf, NaN and +inf rows."""
    vocab = full.shape[-1]
    full[0] = 1.0
    full[1] = -float("inf")
    full[2, vocab - 1] = float("nan")
    full[2, SHARD + 5] = float("nan")
    full[3, 2 * SHARD - 1] = float("inf")
    full[3, vocab - 2] = float("inf")
    full[4] = 0.0
    full[4, vocab - 1] = 2.0
    full[4, 1] = 2.0
    return full


@torch.inference_mode()
def _argmax_check(rank, tp, mnnvl):
    head, vocab = _make_head(rank, tp, mnnvl)
    for rows in (1, 7, 64, 300):
        gen = torch.Generator(device="cuda").manual_seed(rows)
        full = torch.randint(-4, 4, (rows, vocab), device="cuda", generator=gen).to(torch.bfloat16)
        if rows >= 5:
            full = _special_rows(full)
        local = _shard(full, rank)
        expected = torch.argmax(full.float(), dim=-1)
        assert torch.equal(head.argmax(local), expected)
        assert torch.equal(head.argmax(local, out_dtype=torch.int32), expected.to(torch.int32))
        # The gather fallback reproduces the gathered logits bit for bit.
        assert torch.equal(
            head.gather_logits(local).view(torch.int32), full.float().view(torch.int32)
        )
    # Through the real LM head, and under CUDA-graph replay.
    hidden = torch.randn(48, HIDDEN, device="cuda").to(torch.bfloat16)
    expected = torch.argmax(head.lm_head(hidden).float(), dim=-1)
    assert torch.equal(head.argmax(head.local_logits(hidden)), expected)
    graph, out = _capture(lambda: head.argmax(head.local_logits(hidden)), hidden)
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, expected)


@torch.inference_mode()
def _sample_check(rank, tp, mnnvl):
    from tensorrt_llm._torch.speculative.vocab_parallel import (
        VocabShardedLogits,
        _local_sample_kernel,
        _PairExchange,
        _select_tokens,
    )

    head, vocab = _make_head(rank, tp, mnnvl)
    for rows in (1, 64, 257):
        gen = torch.Generator(device="cuda").manual_seed(rows)
        full = (torch.randn(rows, vocab, device="cuda", generator=gen) * 2).to(torch.bfloat16)
        temps = torch.full((rows,), 0.9, device="cuda")
        temps[::4] = 1e-5  # greedy rows inside a sampled batch
        seeds = torch.randint(0, 1 << 40, (rows,), device="cuda", generator=gen)
        offsets = torch.arange(rows, dtype=torch.int64, device="cuda") * 32
        out = head.sample(_shard(full, rank), temps, seeds, offsets, 1e-4)
        # The draw of one shard holding the whole vocab: the noise is keyed by
        # the global vocab id, so the split over ranks cannot change it.
        single = _select_tokens(
            _local_sample_kernel,
            full,
            (temps, seeds, offsets, 1e-4),
            0,
            _PairExchange(None, 1, 0),
            torch.int64,
        )
        assert torch.equal(out, single)
        assert torch.equal(out[::4], torch.argmax(full.float(), dim=-1)[::4])
        gathered = MPI.COMM_WORLD.allgather(out.cpu())
        assert all(torch.equal(g, gathered[0]) for g in gathered)
    # Through the spec-worker wrapper.
    sharded = VocabShardedLogits(head, _shard(full, rank))
    md = SimpleNamespace(
        is_all_greedy_sample=True, temperatures=temps, seeds=seeds, offsets=offsets
    )
    assert torch.equal(sharded.select_tokens(md), torch.argmax(full.float(), dim=-1))
    md.is_all_greedy_sample = False
    assert torch.equal(sharded.select_tokens(md), single)


def _markov_reference(full, anchors, w1, w2):
    biased = dspark_markov_chain_logits_reference(full.float(), anchors, w1, w2)
    return torch.argmax(biased, dim=-1).to(torch.int32)


@torch.inference_mode()
def _markov_check(rank, tp, mnnvl):
    head, vocab = _make_head(rank, tp, mnnvl)
    gen = torch.Generator(device="cuda").manual_seed(11)
    w1 = torch.randn(vocab, 256, device="cuda", generator=gen).to(torch.bfloat16)
    w2 = (torch.randn(vocab, 256, device="cuda", generator=gen) * 0.1).to(torch.bfloat16)
    for num_reqs, block in ((1, 1), (1, 7), (4, 7), (32, 6), (128, 6)):
        full = (torch.randn(num_reqs, block, vocab, device="cuda", generator=gen) * 3).to(
            torch.bfloat16
        )
        anchors = torch.randint(0, vocab, (num_reqs,), device="cuda", generator=gen)
        if num_reqs > 1:
            anchors[0] = -1  # out-of-vocab anchor: zero bias
        expected = _markov_reference(full, anchors, w1, w2)
        assert torch.equal(head.markov_tokens(_shard(full, rank), anchors, w1, w2), expected)
    # The biased shard logits of a vocab-sharded draft head.
    from tensorrt_llm._torch.distributed import AllReduceStrategy
    from tensorrt_llm._torch.speculative.vocab_parallel import markov_chain_logits, tp_pair_exchange

    def global_argmax(step_logits):
        gathered = MPI.COMM_WORLD.allgather(step_logits.cpu())
        return torch.cat(gathered, dim=-1).argmax(dim=-1).cuda()

    exchange = tp_pair_exchange(head.lm_head.mapping, AllReduceStrategy.NCCL)
    full = torch.randn(6, 7, vocab, device="cuda", generator=gen) * 3
    anchors = torch.randint(0, vocab, (6,), device="cuda", generator=gen)
    local = _shard(full, rank)
    biased = markov_chain_logits(local, anchors, w1, w2, exchange)
    # The eager chain on the shard: this rank's markov_w2 rows, global argmax.
    expected = dspark_markov_chain_logits_reference(
        local, anchors, w1, w2[rank * SHARD : (rank + 1) * SHARD], argmax_fn=global_argmax
    )
    assert torch.equal(biased.view(torch.int32), expected.view(torch.int32))
    # Quantized logits and a zero bias make many cross-shard ties.
    full = torch.randint(0, 3, (16, 6, vocab), device="cuda", generator=gen).to(torch.bfloat16)
    anchors = torch.randint(0, vocab, (16,), device="cuda", generator=gen)
    zero_w2 = torch.zeros_like(w2)
    assert torch.equal(
        head.markov_tokens(_shard(full, rank), anchors, w1, zero_w2),
        _markov_reference(full, anchors, w1, zero_w2),
    )
    # CUDA-graph replay.
    full = (torch.randn(8, 6, vocab, device="cuda", generator=gen) * 3).to(torch.bfloat16)
    local = _shard(full, rank)
    anchors = torch.randint(0, vocab, (8,), device="cuda", generator=gen)
    expected = _markov_reference(full, anchors, w1, w2)
    graph, out = _capture(lambda: head.markov_tokens(local, anchors, w1, w2), (local, anchors))
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, expected)


@torch.inference_mode()
def _worker_accept_check(rank, tp, mnnvl):
    """Strict acceptance on shard logits equals acceptance on gathered logits."""
    from tensorrt_llm._torch.speculative.interface import SpecWorkerBase
    from tensorrt_llm._torch.speculative.vocab_parallel import VocabShardedLogits

    class _Worker(SpecWorkerBase):
        @property
        def max_draft_len(self):
            return 8

        def _forward_impl(self, *args, **kwargs):
            raise NotImplementedError

    head, vocab = _make_head(rank, tp, mnnvl)
    num_ctx, num_gens, draft_len = 2, 6, 6
    rows = num_ctx + num_gens * (draft_len + 1)
    gen = torch.Generator(device="cuda").manual_seed(5)
    full = torch.randint(0, 5, (rows, vocab), device="cuda", generator=gen).to(torch.bfloat16)
    target = torch.argmax(full.float(), dim=-1)[num_ctx:].view(num_gens, draft_len + 1)
    draft = target[:, :draft_len].clone().to(torch.int32)
    draft[1::2, 2] = (draft[1::2, 2] + 1) % vocab
    md = SimpleNamespace(
        is_all_greedy_sample=True,
        is_cuda_graph=False,
        is_ragged_verify=False,
        sharded_target_logits=False,
    )
    worker = _Worker()
    batch = num_ctx + num_gens
    ref = worker._sample_and_accept_draft_tokens_base(full.float(), draft, num_ctx, batch, md)
    out = worker._sample_and_accept_draft_tokens_base(
        VocabShardedLogits(head, _shard(full, rank)), draft, num_ctx, batch, md
    )
    assert torch.equal(out[0][:num_ctx, 0], ref[0][:num_ctx, 0])
    assert torch.equal(out[0][num_ctx:], ref[0][num_ctx:])
    assert torch.equal(out[1], ref[1])


_CHECKS = {
    "argmax": _argmax_check,
    "sample": _sample_check,
    "markov": _markov_check,
    "worker_accept": _worker_accept_check,
}


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("mnnvl", _EXCHANGES)
@pytest.mark.parametrize("check", list(_CHECKS))
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_vocab_parallel_selection_tp2(check, mnnvl, mpi_pool_executor):
    tp = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(_run, *zip(*[(_CHECKS[check], tp, mnnvl)] * tp))
    assert all(results)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs")
@pytest.mark.parametrize("mnnvl", _EXCHANGES)
@pytest.mark.parametrize("check", list(_CHECKS))
@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
def test_vocab_parallel_selection_tp4(check, mnnvl, mpi_pool_executor):
    tp = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(_run, *zip(*[(_CHECKS[check], tp, mnnvl)] * tp))
    assert all(results)
