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
"""TRTLLM_MTP_TAIL_TRIM: the trimmed MTP draft-step glue is bit-identical.

Single GPU: the fused local-argmax pack kernel against the original
``_get_local_max_and_combined`` and a mocked MTP-Eagle draft loop with the trim
on vs off. Two GPUs (``mpi_pool_executor``): the dedicated MNNVL mailbox
exchange against the NCCL allgather, eager and under CUDA-graph replay, once
right after a model-sized MNNVL allreduce on the shared workspace.
"""

import gc
import os
import pickle
import sys
import traceback
from contextlib import contextmanager
from types import SimpleNamespace

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from utils.util import skip_num_gpus_less_than

import tensorrt_llm
from tensorrt_llm._torch.distributed import AllReduce
from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce, allgather
from tensorrt_llm._torch.speculative import eagle3 as eagle3_mod
from tensorrt_llm._torch.speculative.eagle3 import Eagle3OneModelWorker
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase
from tensorrt_llm._torch.speculative.mtp_tail_trim import (
    DraftArgmaxMailbox,
    gather_draft_argmax_pairs,
    local_argmax_pack,
)
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# needed since we reuse the mpi executor pool, first test running will leak a thread
pytestmark = pytest.mark.threadleak(enabled=False)


def _ref_combined(logits, tp_rank):
    """The original path: SpecWorkerBase._get_local_max_and_combined."""
    fake = SimpleNamespace(mapping=SimpleNamespace(tp_rank=tp_rank))
    return SpecWorkerBase._get_local_max_and_combined(fake, logits)


def _draft_tokens(gathered):
    """The original consumer: SpecWorkerBase._get_draft_tokens_from_gathered."""
    return SpecWorkerBase._get_draft_tokens_from_gathered(SimpleNamespace(), gathered)


def _make_logits(rows, vocab, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, vocab, generator=g, device="cuda").to(dtype)
    if rows > 1:
        x[1, :] = x[1, 0]  # all equal -> index 0
    if rows > 2:
        x[2, 5] = x[2, vocab - 3] = x[2].max() + 1  # tie -> lower index
    if rows > 3:
        x[3, :] = float("-inf")  # all -inf -> index 0
    if rows > 4:
        x[4, 7] = float("nan")  # NaN wins
        x[4, 9] = float("nan")
    if rows > 5:
        x[5, vocab - 1] = float("inf")
    return x


def _bits(t):
    return t.contiguous().view(torch.int32)


def _assert_same(actual, expected):
    """Equal values, NaN == NaN (the fp32 sum in the allreduce may
    canonicalize a NaN payload; argmax treats every NaN alike)."""
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "rows,vocab", [(1, 19360), (6, 19360), (7, 77), (9, 38720), (3, 8192), (2, 151936)]
)
@pytest.mark.parametrize("tp_rank", [0, 5])
def test_local_argmax_pack_matches_torch(dtype, rows, vocab, tp_rank):
    logits = _make_logits(rows, vocab, dtype, seed=rows * 1000 + vocab)
    ref = _ref_combined(logits, tp_rank)
    out = torch.empty(rows, 2, dtype=torch.float32, device="cuda")
    local_argmax_pack(logits, tp_rank * vocab, out, slot=0)
    assert torch.equal(_bits(out), _bits(ref)), (out, ref)

    # Mailbox layout: the pair at slot 2 * rank, exact zeros elsewhere.
    box = torch.full((rows, 128), 7.0, dtype=torch.float32, device="cuda")
    local_argmax_pack(logits, tp_rank * vocab, box, slot=2 * tp_rank)
    expect = torch.zeros_like(box)
    expect[:, 2 * tp_rank : 2 * tp_rank + 2] = ref
    assert torch.equal(_bits(box), _bits(expect))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_local_argmax_pack_strided_rows_and_cuda_graph():
    base = _make_logits(8, 19360 + 64, torch.bfloat16, seed=3)
    logits = base[:, :19360]  # row stride != vocab
    ref = _ref_combined(logits, 2)
    out = torch.empty(8, 2, dtype=torch.float32, device="cuda")
    local_argmax_pack(logits, 2 * 19360, out)
    assert torch.equal(_bits(out), _bits(ref))

    static_in = logits.clone()
    static_out = torch.empty(8, 2, dtype=torch.float32, device="cuda")
    local_argmax_pack(static_in, 2 * 19360, static_out)  # warm up / compile
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        local_argmax_pack(static_in, 2 * 19360, static_out)
    for seed in (11, 12):
        new = _make_logits(8, 19360, torch.bfloat16, seed=seed)
        static_in.copy_(new)
        graph.replay()
        assert torch.equal(_bits(static_out), _bits(_ref_combined(new, 2)))


# --------------------------------------------------------------------------
# Mocked MTP-Eagle draft loop: trim on vs off must feed every draft step the
# same inputs and produce the same draft tokens.


class _FakeMetadata:
    def __init__(self, seq_lens, kv_lens, num_contexts, device):
        self._seq_lens = seq_lens.clone()
        self._seq_lens_cuda = seq_lens.clone().to(device)
        self.kv_lens_cuda = kv_lens.clone().to(device)
        self.num_contexts = num_contexts
        self.num_ctx_tokens = int(seq_lens[:num_contexts].sum())
        self.kv_cache_manager = None
        self.use_spec_decoding = True
        self.kv_snapshots = []

    @property
    def seq_lens_cuda(self):
        return self._seq_lens_cuda

    def on_update(self):
        pass

    def update_for_spec_dec(self):
        self.kv_snapshots.append(self.kv_lens_cuda.clone())


class _FakeMTPLayer:
    def __init__(self, hidden, vocab, device):
        g = torch.Generator(device="cpu").manual_seed(0)
        self.w_h = (torch.randn(hidden, hidden, generator=g) * 0.2).to(device)
        self.embed = torch.randn(vocab, hidden, generator=g).to(device)
        self.head = torch.randn(hidden, vocab, generator=g).to(device)
        self.calls = []

    def __call__(
        self,
        embed_tokens,
        all_rank_num_tokens,
        input_ids,
        position_ids,
        hidden_states,
        attn_metadata,
        spec_metadata,
    ):
        self.calls.append((input_ids.clone(), position_ids.clone(), hidden_states.clone()))
        out = torch.tanh(
            hidden_states @ self.w_h
            + self.embed[input_ids.long()]
            + position_ids.float().unsqueeze(-1) * 0.01
        )
        return out

    def shared_head(self, hidden_states, lm_head, attn_metadata, return_context_logits):
        return hidden_states @ self.head


def _run_mocked_draft_loop(monkeypatch, trim, draft_len=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden, vocab = 32, 97
    num_contexts, batch_size = 1, 3
    ctx_len = 4
    seq_lens = torch.tensor(
        [ctx_len] + [draft_len + 1] * (batch_size - num_contexts), dtype=torch.int32
    )
    num_tokens = int(seq_lens.sum())
    g = torch.Generator(device="cpu").manual_seed(1)
    input_ids = torch.randint(0, vocab, (num_tokens,), generator=g, dtype=torch.int32).to(device)
    position_ids = torch.arange(num_tokens, dtype=torch.int32).to(device) + 40
    hidden_states = torch.randn(num_tokens, hidden, generator=g).to(device)
    num_accepted = torch.tensor([1, 3, 1], dtype=torch.int32).to(device)
    kv_lens = torch.tensor([ctx_len, 50, 60], dtype=torch.int32)

    md = _FakeMetadata(seq_lens, kv_lens, num_contexts, device)
    layer = _FakeMTPLayer(hidden, vocab, device)
    draft_model = SimpleNamespace(mtp_layers=[layer], lm_head=None, embed_tokens=None)
    spec_metadata = SimpleNamespace(
        runtime_draft_len=draft_len,
        batch_indices_cuda=torch.arange(8, dtype=torch.int32).to(device),
        wants_advanced_draft_sampling=False,
        all_rank_num_tokens=None,
        subseq_all_rank_num_tokens=None,
    )

    @contextmanager
    def draft_context(attn_metadata, manager):
        yield attn_metadata

    def run_draft_forward(dm, inputs, sm, i):
        return dm.mtp_layers[0](embed_tokens=None, all_rank_num_tokens=None, **inputs), None

    def sample_draft_tokens(logits, sm, bs, draft_step=None, mapping_lm_head_tp=None):
        return torch.argmax(logits, dim=-1).type(torch.int32)

    worker = SimpleNamespace(
        is_mtp_eagle=True,
        model_config=None,
        guided_decoder=None,
        sa_enhancer=None,
        draft_kv_cache_context=draft_context,
        _run_draft_forward=run_draft_forward,
        sample_draft_tokens=sample_draft_tokens,
    )
    monkeypatch.setattr(eagle3_mod, "mtp_tail_trim_enabled", lambda: trim)
    inputs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "hidden_states": hidden_states,
        "attn_metadata": md,
        "spec_metadata": spec_metadata,
    }
    tokens = Eagle3OneModelWorker._forward_linear_draft_loop(
        worker,
        inputs,
        md,
        spec_metadata,
        draft_model,
        None,
        num_contexts,
        batch_size,
        num_accepted,
        None,
    )
    return tokens, layer.calls, md.kv_snapshots


@pytest.mark.parametrize("draft_len", [1, 2, 5])
def test_draft_loop_trim_is_bit_identical(monkeypatch, draft_len):
    ref_tokens, ref_calls, ref_kv = _run_mocked_draft_loop(monkeypatch, False, draft_len)
    out_tokens, out_calls, out_kv = _run_mocked_draft_loop(monkeypatch, True, draft_len)
    assert torch.equal(ref_tokens, out_tokens)
    assert len(ref_calls) == len(out_calls) == draft_len
    for (ri, rp, rh), (oi, op, oh) in zip(ref_calls, out_calls):
        assert torch.equal(ri, oi)
        assert torch.equal(rp, op)
        assert rh.dtype == oh.dtype and torch.equal(rh, oh)
    assert len(ref_kv) == len(out_kv)
    for a, b in zip(ref_kv, out_kv):
        assert torch.equal(a, b)


# --------------------------------------------------------------------------
# Multi-rank: the dedicated MNNVL mailbox exchange vs the NCCL allgather.

_MNNVL_TEST_ENV = ("TLLM_TEST_MNNVL", "TRTLLM_FORCE_MNNVL_AR")


def _check_mailbox_matches_nccl(mapping: Mapping, rank: int, world: int) -> None:
    model_ar = AllReduce(mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16)
    assert model_ar.mnnvl_allreduce is not None
    big = torch.randn(6, 6144, device="cuda").to(torch.bfloat16)  # model-sized payload
    model_ar(big)  # the model workspace exists before the mailbox eligibility check
    box = DraftArgmaxMailbox.get(mapping)
    assert box is not None, "dedicated mailbox did not initialise"

    def ref_path(logits):
        return allgather(_ref_combined(logits, rank), mapping, dim=-1)

    def check(out, ref):
        _assert_same(out, ref)
        torch.testing.assert_close(_draft_tokens(out), _draft_tokens(ref), rtol=0, atol=0)

    # rows > max_rows takes the NCCL fallback with the pack kernel.
    for rows in (1, 2, 6, 64, box.max_rows, box.max_rows + 1):
        for vocab in (19360, 77):
            logits = _make_logits(rows, vocab, torch.bfloat16, seed=100 * rows + vocab + 7 * rank)
            if rows > 6:  # same max on every rank -> lowest rank must win
                logits[6, :] = 0
                logits[6, 3] = 5
            check(gather_draft_argmax_pairs(logits, mapping), ref_path(logits))

    # A model-sized allreduce right before a one-row exchange: the mailbox's
    # own workspace keeps the model's dirty Lamport buffer out of its launch.
    logits = _make_logits(1, 19360, torch.bfloat16, seed=41 + rank)
    model_ar(big)
    check(gather_draft_argmax_pairs(logits, mapping), ref_path(logits))

    # CUDA graph: model allreduce + pack + mailbox, replayed on fresh logits.
    static_logits = _make_logits(1, 19360, torch.bfloat16, seed=rank)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        model_ar(big)
        gather_draft_argmax_pairs(static_logits, mapping)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            model_ar(big)
            graph_out = gather_draft_argmax_pairs(static_logits, mapping)
    for it in range(3):
        new = _make_logits(1, 19360, torch.bfloat16, seed=1000 * it + rank)
        static_logits.copy_(new)
        graph.replay()
        torch.cuda.synchronize()
        check(graph_out, ref_path(new))


def _mailbox_worker(world: int) -> bool:
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    previous_env = {name: (name in os.environ, os.environ.get(name)) for name in _MNNVL_TEST_ENV}
    os.environ["TLLM_TEST_MNNVL"] = "1"
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
    mapping = Mapping(world_size=world, tp_size=world, rank=rank)
    try:
        MPI.COMM_WORLD.barrier()
        MNNVLAllReduce.allreduce_mnnvl_workspaces.pop(mapping, None)
        DraftArgmaxMailbox._instances.pop(mapping, None)
        gc.collect()
        MPI.COMM_WORLD.barrier()
        _check_mailbox_matches_nccl(mapping, rank, world)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        DraftArgmaxMailbox._instances.pop(mapping, None)
        MNNVLAllReduce.allreduce_mnnvl_workspaces.pop(mapping, None)
        gc.collect()
        for name, (was_present, value) in previous_env.items():
            if was_present:
                os.environ[name] = value
            else:
                os.environ.pop(name, None)
    return True


@skip_num_gpus_less_than(2)
@pytest.mark.gpu2
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_mailbox_matches_nccl_allgather(mpi_pool_executor):
    world = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(_mailbox_worker, [world] * world)
    for r in results:
        assert r is True
