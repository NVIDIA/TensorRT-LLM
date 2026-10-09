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
"""``speculative/draft_argmax.py``: the TP exchange of the MTP draft argmax pairs.

CPU: ``MNNVLAllReduce.max_one_shot_tokens`` against the one-shot threshold at TP 2..64.

Single GPU: the Triton (index, max) pack kernel against the ``torch.max`` producer bitwise
(bf16 / fp16 / fp32, ties, all -inf rows, NaN, strided rows, CUDA-graph replay).

Two GPUs (``mpi_pool_executor``, ``gpu2``): ``DraftArgmaxExchange`` against the allgather
producer under the NCCL, AUTO and (on MNNVL hardware) MNNVL strategies in one worker
invocation, eager and under CUDA-graph replay, the fallback above ``max_rows``, and the tagged
MNNVL workspace living next to the model's default one.
"""

import gc
import os
import pickle
import platform
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from utils.util import skip_single_gpu

import tensorrt_llm
from tensorrt_llm._mnnvl_utils import MnnvlMemory
from tensorrt_llm._torch.distributed import AllReduce
from tensorrt_llm._torch.distributed.ops import _MNNVL_ONE_SHOT_THRESHOLD_BYTES, MNNVLAllReduce
from tensorrt_llm._torch.models.modeling_utils import MetaInitMode
from tensorrt_llm._torch.speculative.draft_argmax import (
    ROW_WIDTH,
    WORKSPACE_TAG,
    DraftArgmaxExchange,
    gather_argmax_pairs_allgather,
    local_argmax_pack,
    local_argmax_pairs,
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

_TP_RANK = 5  # rank offset / slot used by the single-GPU pack tests
_SLOT = 2 * _TP_RANK


def _draft_tokens(gathered):
    """Eager form of SpecWorkerBase._get_draft_tokens_from_gathered."""
    best = torch.argmax(gathered[..., 1::2], dim=-1, keepdim=True)
    return torch.gather(gathered[..., 0::2], -1, best).squeeze(-1).to(torch.int32)


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


def _assert_same_values(actual, expected):
    """Equal values, NaN == NaN: the fp32 SUM may canonicalize a NaN payload
    and turns a -0.0 max into +0.0; the consuming argmax compares values."""
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


def _assert_packed_row(row, ref):
    """``row[:, _SLOT:_SLOT + 2]`` is ``ref`` bitwise, the other slots are exact +0.0."""
    assert torch.equal(_bits(row[:, _SLOT : _SLOT + 2]), _bits(ref))
    rest = torch.ones(ROW_WIDTH, dtype=torch.bool, device=row.device)
    rest[_SLOT : _SLOT + 2] = False
    assert (_bits(row[:, rest]) == 0).all()


# --------------------------------------------------------------------------
# CPU-only: the one-shot sizing the default max_rows comes from.


@pytest.mark.parametrize("tp_size", [2, 4, 8, 16, 32, 64])
def test_max_one_shot_tokens_matches_the_one_shot_threshold(tp_size):
    assert 2 * tp_size <= ROW_WIDTH
    rows = MNNVLAllReduce.max_one_shot_tokens(ROW_WIDTH, tp_size, torch.float32)
    payload = rows * ROW_WIDTH * 4 * tp_size
    assert rows >= 1
    assert payload <= _MNNVL_ONE_SHOT_THRESHOLD_BYTES < payload + ROW_WIDTH * 4 * tp_size
    # The one-shot sizing rule: the whole payload fits one Lamport buffer.
    assert (
        MNNVLAllReduce.get_required_workspace_size(rows, ROW_WIDTH, tp_size, torch.float32)
        == payload
    )


# --------------------------------------------------------------------------
# Single GPU: the pack kernel vs torch.max.


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "rows,vocab", [(1, 19360), (6, 19360), (7, 77), (9, 38720), (3, 8192), (2, 151936)]
)
def test_local_argmax_pack_matches_torch(dtype, rows, vocab):
    logits = _make_logits(rows, vocab, dtype, seed=rows * 1000 + vocab)
    row = torch.full((rows, ROW_WIDTH), 7.0, dtype=torch.float32, device="cuda")
    local_argmax_pack(logits, _TP_RANK * vocab, row, slot=_SLOT)
    _assert_packed_row(row, local_argmax_pairs(logits, _TP_RANK))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_local_argmax_pack_strided_rows_and_cuda_graph():
    vocab = 19360
    base = _make_logits(8, vocab + 64, torch.bfloat16, seed=3)
    logits = base[:, :vocab]  # row stride != vocab
    out = torch.empty(8, ROW_WIDTH, dtype=torch.float32, device="cuda")
    local_argmax_pack(logits, _TP_RANK * vocab, out, slot=_SLOT)
    _assert_packed_row(out, local_argmax_pairs(logits, _TP_RANK))

    static_in = logits.clone()
    static_out = torch.empty_like(out)
    local_argmax_pack(static_in, _TP_RANK * vocab, static_out, slot=_SLOT)  # warm up / compile
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        local_argmax_pack(static_in, _TP_RANK * vocab, static_out, slot=_SLOT)
    for seed in (11, 12):
        new = _make_logits(8, vocab, torch.bfloat16, seed=seed)
        static_in.copy_(new)
        graph.replay()
        _assert_packed_row(static_out, local_argmax_pairs(new, _TP_RANK))


# --------------------------------------------------------------------------
# Two GPUs: DraftArgmaxExchange vs the allgather producer.

_MNNVL_TEST_ENV = "TLLM_TEST_MNNVL"
_TEST_MAX_ROWS = 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_exchange_built_under_meta_init_allocates_staging_on_first_use():
    """The worker constructs the exchange inside the model constructor, i.e. under
    MetaInitMode, where torch.empty ignores the requested device; the staging
    buffer must therefore not be allocated at construction (a meta tensor would
    reach the pack kernel with data_ptr 0)."""
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    with MetaInitMode():
        module = DraftArgmaxExchange(mapping, AllReduceStrategy.AUTO, max_rows=4)
    assert module._staging is None
    staging = module._staging_rows(2)
    assert staging.device.type == "cuda" and not staging.is_meta and staging.data_ptr() != 0
    assert staging.shape == (2, ROW_WIDTH) and module._staging.shape == (4, ROW_WIDTH)
    logits = _make_logits(2, 77, torch.bfloat16, seed=3)
    out = module(logits)  # tp 1: the all-reduce is the identity, the pairs come back as packed
    assert torch.equal(_bits(out), _bits(gather_argmax_pairs_allgather(logits, mapping)))


def _mnnvl_available() -> bool:
    """The predicate the MNNVL all-reduce tests gate on."""
    return (
        platform.machine().lower() == "aarch64"
        and torch.cuda.device_count() >= 2
        and MnnvlMemory.supports_mnnvl()
    )


def _check_gathered(out, ref):
    assert out.shape == ref.shape and out.dtype == torch.float32
    _assert_same_values(out, ref)
    assert torch.equal(_draft_tokens(out), _draft_tokens(ref))


def _check_exchange_matches_allgather(mapping: Mapping, strategy: AllReduceStrategy) -> None:
    rank, world = mapping.tp_rank, mapping.tp_size
    # Built the way the engine builds it: inside the model constructor, under
    # MetaInitMode (every torch.empty lands on the meta device there).
    with MetaInitMode():
        module = DraftArgmaxExchange(mapping, strategy, max_rows=_TEST_MAX_ROWS)
    assert module.max_rows == _TEST_MAX_ROWS
    assert module._staging is None  # allocated by the first forward, outside the mode
    workspaces = MNNVLAllReduce.allreduce_mnnvl_workspaces
    if strategy == AllReduceStrategy.MNNVL:
        # The MNNVL arm must really run the MNNVL kernels, on the tagged workspace only.
        mnnvl = module.allreduce.mnnvl_allreduce
        assert mnnvl is not None and mnnvl.workspace_key == (mapping, WORKSPACE_TAG)
        assert (mapping, WORKSPACE_TAG) in workspaces and mapping not in workspaces
    elif strategy == AllReduceStrategy.NCCL:
        assert module.allreduce.mnnvl_allreduce is None

    for rows in (1, 3, _TEST_MAX_ROWS):
        for vocab in (19360, 77):
            logits = _make_logits(rows, vocab, torch.bfloat16, seed=100 * rows + vocab + 7 * rank)
            if rows > 6:  # same max on every rank -> the lowest rank must win
                logits[6, :] = 0
                logits[6, 3] = 5
            _check_gathered(module(logits), gather_argmax_pairs_allgather(logits, mapping))

    # Above max_rows forward runs the allgather producer itself: identical bits.
    big = _make_logits(_TEST_MAX_ROWS + 1, 77, torch.bfloat16, seed=rank)
    out = module(big)
    assert out.shape == (_TEST_MAX_ROWS + 1, 2 * world)
    assert torch.equal(_bits(out), _bits(gather_argmax_pairs_allgather(big, mapping)))

    # CUDA graph: pack + all-reduce captured once, replayed on fresh logits.
    static_logits = _make_logits(3, 19360, torch.bfloat16, seed=rank)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        module(static_logits)  # warm up / compile
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            graph_out = module(static_logits)
    for it in range(3):
        new = _make_logits(3, 19360, torch.bfloat16, seed=1000 * it + rank)
        static_logits.copy_(new)
        graph.replay()
        torch.cuda.synchronize()
        _check_gathered(graph_out, gather_argmax_pairs_allgather(new, mapping))


def _check_tagged_workspace_is_separate(mapping: Mapping) -> None:
    rank, world = mapping.tp_rank, mapping.tp_size
    workspaces = MNNVLAllReduce.allreduce_mnnvl_workspaces
    model_ar = AllReduce(mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16)
    assert model_ar.mnnvl_allreduce is not None
    assert model_ar.mnnvl_allreduce.workspace_key is mapping  # untagged key: the mapping itself
    module = DraftArgmaxExchange(mapping, AllReduceStrategy.MNNVL, max_rows=8)
    assert module.allreduce.mnnvl_allreduce is not None

    default_ws = workspaces[mapping]
    tagged_ws = workspaces[(mapping, WORKSPACE_TAG)]
    assert default_ws is not tagged_ws
    assert default_ws["uc_buffer"].data_ptr() != tagged_ws["uc_buffer"].data_ptr()
    assert default_ws["buffer_flags"].data_ptr() != tagged_ws["buffer_flags"].data_ptr()
    # The exchange sized its own Lamport buffers; the model default is far larger.
    assert tagged_ws["buffer_size_bytes"] == MNNVLAllReduce.get_required_workspace_size(
        8, ROW_WIDTH, world, torch.float32
    )
    assert default_ws["buffer_size_bytes"] > tagged_ws["buffer_size_bytes"]
    # A second AllReduce with the same tag shares the tagged workspace, not the model's.
    other = AllReduce(
        mapping=mapping,
        strategy=AllReduceStrategy.MNNVL,
        dtype=torch.float32,
        workspace_tag=WORKSPACE_TAG,
    )
    assert other.mnnvl_allreduce is not None
    assert other.mnnvl_allreduce.workspace_key == (mapping, WORKSPACE_TAG)
    assert workspaces[(mapping, WORKSPACE_TAG)] is tagged_ws
    assert workspaces[mapping] is default_ws

    # A model-sized all-reduce right before a one-row exchange, and the model
    # all-reduce again right after: the two workspaces do not see each other's
    # dirty Lamport buffers.
    big = torch.randn(6, 6144, device="cuda").to(torch.bfloat16)
    model_ar(big)
    logits = _make_logits(1, 19360, torch.bfloat16, seed=41 + rank)
    _check_gathered(module(logits), gather_argmax_pairs_allgather(logits, mapping))
    ones = torch.full((2, 128), rank + 1, dtype=torch.bfloat16, device="cuda")
    expect = torch.full_like(ones, world * (world + 1) // 2)
    torch.testing.assert_close(model_ar(ones), expect)
    _check_gathered(module(logits), gather_argmax_pairs_allgather(logits, mapping))


def _drop_workspaces(mapping: Mapping) -> None:
    for key in (mapping, (mapping, WORKSPACE_TAG)):
        MNNVLAllReduce.allreduce_mnnvl_workspaces.pop(key, None)
    gc.collect()


def _run_on_rank(world: int, check, *args, test_mnnvl: bool = False) -> None:
    """Run ``check(mapping, *args)`` on this rank with the MNNVL registry clean
    before and after; ``test_mnnvl`` sets the single-node MNNVL bypass the
    MNNVL all-reduce tests use, restored on exit."""
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    mapping = Mapping(world_size=world, tp_size=world, rank=rank)
    previous_env = os.environ.get(_MNNVL_TEST_ENV)
    if test_mnnvl:
        os.environ[_MNNVL_TEST_ENV] = "1"
    try:
        MPI.COMM_WORLD.barrier()
        _drop_workspaces(mapping)
        MPI.COMM_WORLD.barrier()
        check(mapping, *args)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        _drop_workspaces(mapping)
        if previous_env is None:
            os.environ.pop(_MNNVL_TEST_ENV, None)
        else:
            os.environ[_MNNVL_TEST_ENV] = previous_env


def _exchange_worker(world: int) -> bool:
    for strategy in (AllReduceStrategy.NCCL, AllReduceStrategy.AUTO):
        _run_on_rank(world, _check_exchange_matches_allgather, strategy)
    if _mnnvl_available():
        _run_on_rank(
            world, _check_exchange_matches_allgather, AllReduceStrategy.MNNVL, test_mnnvl=True
        )
    return True


def _tagged_workspace_worker(world: int) -> bool:
    _run_on_rank(world, _check_tagged_workspace_is_separate, test_mnnvl=True)
    return True


@skip_single_gpu
@pytest.mark.gpu2
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_draft_argmax_exchange_matches_allgather(mpi_pool_executor):
    world = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(_exchange_worker, [world] * world)
    for r in results:
        assert r is True


@skip_single_gpu
@pytest.mark.gpu2
@pytest.mark.skipif(not _mnnvl_available(), reason="MNNVL not available")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_tagged_mnnvl_workspace_is_separate_from_the_default(mpi_pool_executor):
    world = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(_tagged_workspace_worker, [world] * world)
    for r in results:
        assert r is True
