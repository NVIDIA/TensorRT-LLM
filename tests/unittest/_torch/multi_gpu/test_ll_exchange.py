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
"""The one-launch TP exchanges (distributed/ll_exchange.py) against the same reduction over every rank's
inputs: key max / gather and greedy argmax of TPKeyExchange, its argmax and temperature sample of
vocabulary-sharded logits (also through the spec worker), and TPGroupExchange's subgroup all-gather,
eager and replayed from a CUDA graph (the epochs keep advancing across launches)."""

import math
import pickle
import sys
import traceback
from types import SimpleNamespace

import cloudpickle
import pytest
import torch
from _torch.helpers import gumbel_max_reference
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._torch.distributed.ll_exchange import (
    TPGroupExchange,
    TPKeyExchange,
    unpack_argmax_tokens,
)
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import GREEDY_TEMPERATURE_THRESHOLD
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
        reason="needs SM90+ (PDL)",
    ),
]


def _mapping(tp):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    MPI.COMM_WORLD.barrier()
    return Mapping(world_size=tp, tp_size=tp, rank=rank), rank


def _keys(m, seed, r):
    g = torch.Generator(device="cuda").manual_seed(seed * 31 + r)
    return torch.randint(-(2**62), 2**62, (m,), generator=g, device="cuda", dtype=torch.int64)


def _key_exchange(tp):
    mapping, rank = _mapping(tp)
    ex = TPKeyExchange.create(mapping, max_rows=64)
    assert ex is not None, "no peer memory for the exchange"
    try:
        # Consecutive launches of one channel with changing row counts (alternating slots).
        for step, m in enumerate((1, 64, 7, 512, 33)):
            all_keys = torch.stack([_keys(m, step, r) for r in range(tp)])
            gather = torch.empty_like(all_keys)
            max_out, _ = ex.exchange(all_keys[rank].clone(), gather_out=gather)
            torch.cuda.synchronize()
            assert torch.equal(max_out, all_keys.max(0).values), ("max", m)
            assert torch.equal(gather, all_keys), ("gather", m)

        # Greedy argmax over a vocabulary sharded by rank: ties go to the lowest global index.
        m, vocab = 16, 1000
        g = torch.Generator(device="cuda").manual_seed(5)
        logits = torch.randn((m, tp * vocab), generator=g, device="cuda")
        logits[0, :] = 0.0
        logits[0, 3::vocab] = 1.0  # one maximum per shard, all equal: token 3
        logits[1, vocab - 1] = logits[1, 2 * vocab - 1] = 50.0  # tied across the first two shards
        shard = logits[:, rank * vocab : (rank + 1) * vocab]
        vals, lidx = shard.max(-1)
        lidx = lidx.to(torch.int32)
        for _ in range(3):
            tokens = ex.argmax_tokens(vals, lidx, rank * vocab)
            torch.cuda.synchronize()
            ref = logits.argmax(-1).to(torch.int32)
            ref[0] = 3
            ref[1] = vocab - 1
            assert torch.equal(tokens, ref), (tokens.tolist(), ref.tolist())

        # CUDA-graph replays with fresh inputs copied into the captured buffers.
        m = 24
        keys = _keys(m, 100, rank)
        max_buf = torch.empty_like(keys)
        ex.exchange(keys, max_out=max_buf)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ex.exchange(keys, max_out=max_buf)
        for step in range(5):
            all_keys = torch.stack([_keys(m, 200 + step, r) for r in range(tp)])
            keys.copy_(all_keys[rank])
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(max_buf, all_keys.max(0).values), ("graph", step)
    finally:
        MPI.COMM_WORLD.barrier()
        ex.close()


SHARD = 1000  # vocabulary columns per rank


def _argmax_logits(tp):
    mapping, rank = _mapping(tp)
    ex = TPKeyExchange.create(mapping, max_rows=64)
    assert ex is not None, "no peer memory for the exchange"
    try:
        for dtype in (torch.bfloat16, torch.float32):
            for m in (1, 6, 64, 300):
                g = torch.Generator(device="cuda").manual_seed(m)
                # Few distinct values: ties inside a shard, across shards and across the programs of a row.
                full = torch.randint(-3, 3, (m, tp * SHARD), generator=g, device="cuda").to(dtype)
                if m >= 6:
                    full[0] = 0.0
                    full[0, 3::SHARD] = 1.0  # one maximum per shard, all equal: token 3
                    full[1, SHARD - 1] = full[1, 2 * SHARD - 1] = 50.0  # tied across two shards
                    full[2] = -float("inf")
                    full[3] = -1.0
                    full[3, SHARD + 2] = 0.0
                    full[3, 7] = -0.0  # equal to +0.0: token 7
                    full[4, -1] = 50.0  # last token of the last shard
                # This rank's columns of the full rows (row stride tp * SHARD).
                local = full[:, rank * SHARD : (rank + 1) * SHARD]
                expected = torch.argmax(full.float(), dim=-1).to(torch.int32)
                if m >= 6:
                    assert expected[:5].tolist() == [3, SHARD - 1, 0, 7, tp * SHARD - 1]
                for split in (None, 1, 3, 32):
                    keys = torch.empty(m, dtype=torch.int64, device="cuda")
                    tokens = ex.argmax_logits(local, rank * SHARD, max_out=keys, split=split)
                    torch.cuda.synchronize()
                    assert torch.equal(tokens, expected), (dtype, m, split)
                    assert torch.equal(unpack_argmax_tokens(keys), expected), (dtype, m, split)

        # CUDA-graph replays with fresh logits copied into the captured buffer.
        m = 24
        local = torch.zeros((m, SHARD), dtype=torch.bfloat16, device="cuda")
        ex.argmax_logits(local, rank * SHARD)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            tokens = ex.argmax_logits(local, rank * SHARD)
        for step in range(3):
            g = torch.Generator(device="cuda").manual_seed(100 + step)
            full = torch.randint(-3, 3, (m, tp * SHARD), generator=g, device="cuda")
            local.copy_(full[:, rank * SHARD : (rank + 1) * SHARD])
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(tokens, torch.argmax(full, dim=-1).to(torch.int32)), step
    finally:
        MPI.COMM_WORLD.barrier()
        ex.close()


def _single_shard_sample(full, temperatures, seed):
    """The draw of one shard holding the whole vocabulary (the noise of a token does not depend on
    the sharding)."""
    return gumbel_max_reference(full, temperatures, seed, GREEDY_TEMPERATURE_THRESHOLD)


def _chi_square(counts, probs, z=5.0):
    """Pearson statistic of ``counts`` against ``probs`` (bins under 5 expected draws merged) and its
    critical value at ``z`` standard deviations (Wilson-Hilferty; z = 5 rejects a correct sampler
    with probability 3e-7)."""
    expected = probs.double() * counts.sum()
    big = expected >= 5
    obs, exp = counts[big].double(), expected[big]
    rest_obs, rest_exp = counts[~big].sum().double(), expected[~big].sum()
    if rest_exp >= 5:
        obs, exp = torch.cat([obs, rest_obs.view(1)]), torch.cat([exp, rest_exp.view(1)])
    else:
        obs[-1] += rest_obs
        exp[-1] += rest_exp
    stat = float(((obs - exp) ** 2 / exp).sum())
    dof = obs.numel() - 1
    crit = dof * (1 - 2 / (9 * dof) + z * math.sqrt(2 / (9 * dof))) ** 3
    return stat, crit


def _gumbel_sample(tp, base_seed=0):
    from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

    mapping, rank = _mapping(tp)
    ex = TPKeyExchange.create(mapping, max_rows=64)
    assert ex is not None, "no peer memory for the exchange"
    vocab, rows, groups = tp * SHARD, 512, 4
    group_temps = torch.tensor([0.6, 1.0, 1.5, 0.5 * GREEDY_TEMPERATURE_THRESHOLD], device="cuda")
    temps = group_temps.repeat_interleave(rows // groups)  # the last group stays greedy
    g = torch.Generator(device="cuda").manual_seed(base_seed + 1)
    dists = (torch.randn((groups, vocab), generator=g, device="cuda") * 3).to(torch.bfloat16)
    full = dists.repeat_interleave(rows // groups, dim=0)
    local = full[:, rank * SHARD : (rank + 1) * SHARD]
    greedy = torch.argmax(full.float(), dim=-1).to(torch.int32)
    try:
        # Each row draws its own noise: the rows of a group are independent samples of its logits.
        counts = torch.zeros((groups - 1, vocab), dtype=torch.int64, device="cuda")
        seed = torch.zeros(1, dtype=torch.int64, device="cuda")
        for step in range(32):
            seed.fill_(base_seed * 1000 + step + 1)
            tokens = ex.argmax_logits(
                local,
                rank * SHARD,
                temperature=temps,
                seed=seed,
                greedy_threshold=GREEDY_TEMPERATURE_THRESHOLD,
                split=3 if step % 2 else None,
            )
            assert torch.equal(tokens, _single_shard_sample(full, temps, seed)), step
            for group in range(groups - 1):
                counts[group] += torch.bincount(
                    tokens.view(groups, -1)[group].long(), minlength=vocab
                )
        gathered = MPI.COMM_WORLD.allgather(tokens.cpu())
        assert all(torch.equal(t, gathered[0]) for t in gathered)
        assert torch.equal(tokens.view(groups, -1)[-1], greedy.view(groups, -1)[-1])
        for group in range(groups - 1):
            probs = torch.softmax(dists[group].float() / group_temps[group], dim=-1)
            stat, crit = _chi_square(counts[group].cpu(), probs.cpu())
            assert stat < crit, (float(group_temps[group]), stat, crit)

        # Through the spec worker: every temperature-only batch advances its device seed.
        class _Worker(SpecWorkerBase):
            @property
            def max_draft_len(self):
                return 1

            def _forward_impl(self, *args, **kwargs):
                raise NotImplementedError

        worker = _Worker(key_exchange=ex)
        worker.mapping = mapping
        worker.cache_derived_state()
        md = SimpleNamespace(
            is_all_greedy_sample=False, sharded_target_logits=True, temperatures=temps
        )
        for step in (1, 2):
            tokens = worker._sample_tokens_for_batch(local, md, 0, rows)
            assert int(worker._gumbel_seed) == step
            assert torch.equal(tokens, _single_shard_sample(full, temps, worker._gumbel_seed))
        md.is_all_greedy_sample = True
        assert torch.equal(worker._sample_tokens_for_batch(local, md, 0, rows), greedy)
        torch.cuda.synchronize()
    finally:
        MPI.COMM_WORLD.barrier()
        ex.close()


def _group_allgather(tp):
    mapping, rank = _mapping(tp)
    group_size = 2
    ex = TPGroupExchange.create(mapping, group_size, max_bytes=64 * 4 * 256 * 2, max_merge_rows=64)
    assert ex is not None, "no peer memory for the exchange"
    try:
        members = range(rank // group_size * group_size, (rank // group_size + 1) * group_size)
        for step, rows in enumerate((1, 9, 64, 3)):

            def x_of(r):
                g = torch.Generator(device="cuda").manual_seed(step * 17 + r)
                return torch.randn((rows, 4, 256), generator=g, device="cuda").to(torch.bfloat16)

            out = ex.allgather(x_of(rank))
            torch.cuda.synchronize()
            ref = torch.cat([x_of(r).reshape(rows, -1) for r in members], dim=-1)
            assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), ("allgather", rows)
    finally:
        MPI.COMM_WORLD.barrier()
        ex.close()


def _worker(fn, tp):
    try:
        with torch.inference_mode():
            fn(tp)
    except Exception:
        traceback.print_exc()
        raise
    return True


CASES = {
    "key_exchange": _key_exchange,
    "argmax_logits": _argmax_logits,
    "gumbel_sample": _gumbel_sample,
    "group_allgather": _group_allgather,
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_ll_exchange(case, mpi_pool_executor):
    tp = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < tp:
        pytest.skip(f"needs {tp} GPUs")
    results = mpi_pool_executor.map(_worker, [CASES[case]] * tp, [tp] * tp)
    assert all(results)
