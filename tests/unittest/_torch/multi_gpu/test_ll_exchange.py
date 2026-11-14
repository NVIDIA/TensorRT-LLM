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
vocabulary-sharded logits (also through the spec worker), and TPGroupExchange's subgroup all-gather and the
RoPE kernel's in-kernel q all-gather, eager and replayed from a CUDA graph (the epochs keep advancing across
launches); TPGroupExchange's all-to-all merge against an eager merge of every member's partials."""

import math
import pickle
import sys
import traceback
from types import SimpleNamespace

import cloudpickle
import pytest
import torch
from _torch.helpers import (
    fast_gumbel_max_reference,
    gumbel_max_reference,
    late_copy,
    triton_num_compiled,
)
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
                # The row-split launch shape draws the same tokens.
                ex.argmax_row_split = True
                tokens = ex.argmax_logits(local, rank * SHARD)
                ex.argmax_row_split = False
                torch.cuda.synchronize()
                assert torch.equal(tokens, expected), (dtype, m, "row split")

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
            ex.argmax_row_split = step % 4 == 0
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


def _fast_gumbel_sample(tp):
    """fast_gumbel: the four-draws kernel's tokens are fast_gumbel_max_reference's over the whole vocabulary on
    every rank (whatever the sharding), greedy below the threshold, and distributed as softmax(logits / T)."""
    mapping, rank = _mapping(tp)
    ex = TPKeyExchange.create(mapping, max_rows=64, fast_gumbel=True)
    assert ex is not None, "no peer memory for the exchange"
    vocab, rows, groups = tp * SHARD, 512, 4
    group_temps = torch.tensor([0.6, 1.0, 1.5, 0.5 * GREEDY_TEMPERATURE_THRESHOLD], device="cuda")
    g = torch.Generator(device="cuda").manual_seed(3)
    dists = (torch.randn((groups, vocab), generator=g, device="cuda") * 3).to(torch.bfloat16)
    try:
        counts = torch.zeros((groups - 1, vocab), dtype=torch.int64, device="cuda")
        seed = torch.zeros(1, dtype=torch.int64, device="cuda")
        for step, m in enumerate([rows] * 32 + [1, 7, 33]):
            temps = group_temps.repeat_interleave(rows // groups)[:m]
            full = dists.repeat_interleave(rows // groups, dim=0)[:m]
            local = full[:, rank * SHARD : (rank + 1) * SHARD]
            seed.fill_(step + 1)
            tokens = ex.argmax_logits(
                local,
                rank * SHARD,
                temperature=temps,
                seed=seed,
                greedy_threshold=GREEDY_TEMPERATURE_THRESHOLD,
            )
            expected = fast_gumbel_max_reference(full, temps, seed, GREEDY_TEMPERATURE_THRESHOLD)
            assert torch.equal(tokens, expected), step
            gathered = MPI.COMM_WORLD.allgather(tokens.cpu())
            assert all(torch.equal(t, gathered[0]) for t in gathered)
            if m == rows:
                for group in range(groups - 1):
                    counts[group] += torch.bincount(
                        tokens.view(groups, -1)[group].long(), minlength=vocab
                    )
                greedy = torch.argmax(full.float(), dim=-1).to(torch.int32)
                assert torch.equal(tokens.view(groups, -1)[-1], greedy.view(groups, -1)[-1])
        for group in range(groups - 1):
            probs = torch.softmax(dists[group].float() / group_temps[group], dim=-1)
            stat, crit = _chi_square(counts[group].cpu(), probs.cpu())
            assert stat < crit, (float(group_temps[group]), stat, crit)
        torch.cuda.synchronize()
    finally:
        MPI.COMM_WORLD.barrier()
        ex.close()


def _sample_accept(tp):
    """fused_accept.sample_and_strict_accept: the seed bump, the four-draws Gumbel sample of argmax_logits and the
    strict acceptance (natural, forced base, forced base + pool draw) in one launch give the three launches' accepted
    tokens, counts, seed and forced-acceptance counter, eager and replayed from a CUDA graph."""
    from tensorrt_llm._torch.speculative.fused_accept import (
        fused_strict_accept,
        sample_and_strict_accept,
    )

    mapping, rank = _mapping(tp)
    ex = TPKeyExchange.create(mapping, max_rows=64, fast_gumbel=True)
    assert ex is not None, "no peer memory for the exchange"
    k, vocab = 6, tp * SHARD
    pool = torch.rand(1024, generator=torch.Generator(device="cuda").manual_seed(5), device="cuda")

    def inputs(num_contexts, num_gens, step):
        g = torch.Generator(device="cuda").manual_seed(100 * step + num_gens)
        m = num_contexts + num_gens * (k + 1)
        full = (torch.randn((m, vocab), generator=g, device="cuda") * 3).to(torch.bfloat16)
        temps = torch.rand(m, generator=g, device="cuda") + 0.3
        temps[::3] = 0.5 * GREEDY_TEMPERATURE_THRESHOLD  # greedy rows
        greedy = torch.argmax(full.float(), dim=-1).to(torch.int32)[num_contexts:].view(-1, k + 1)
        draft = greedy[:, :k].clone()
        draft[1::2, 2] += 1  # leading matches stop early on odd requests
        return full[:, rank * SHARD : (rank + 1) * SHARD].contiguous(), temps, draft

    def stock(logits, temps, draft, nc, b, seed, force):
        seed.add_(1)
        tokens = ex.argmax_logits(
            logits,
            rank * SHARD,
            temperature=temps,
            seed=seed,
            greedy_threshold=GREEDY_TEMPERATURE_THRESHOLD,
        )
        return fused_strict_accept(tokens, draft, nc, b, force)

    def fused(logits, temps, draft, nc, b, seed, force):
        return sample_and_strict_accept(
            ex, logits, rank * SHARD, temps, seed, GREEDY_TEMPERATURE_THRESHOLD, draft, nc, b, force
        )

    def check(a, b, nc, what):
        (acc_a, n_a), (acc_b, n_b) = a, b
        assert torch.equal(n_a, n_b), what
        assert torch.equal(acc_a[:nc, 0], acc_b[:nc, 0]) and torch.equal(acc_a[nc:], acc_b[nc:]), (
            what
        )

    try:
        for force_kind in ("natural", "base", "draw"):
            counters = [torch.zeros(1, dtype=torch.int64, device="cuda") for _ in range(2)]

            def force_of(i):
                if force_kind == "natural":
                    return None
                if force_kind == "base":
                    return (3, k + 1, 0.0, None, None, 6007, 1009, 1024)
                return (3, k + 1, 0.6, pool, counters[i], 6007, 1009, 1024)

            seeds = [torch.full((1,), 7, dtype=torch.int64, device="cuda") for _ in range(2)]
            for step, (nc, ng) in enumerate(((0, 5), (2, 3), (0, 8), (1, 0))):
                logits, temps, draft = inputs(nc, ng, step)
                b = nc + ng
                ref = stock(logits, temps, draft, nc, b, seeds[0], force_of(0))
                got = fused(logits, temps, draft, nc, b, seeds[1], force_of(1))
                assert got is not None
                torch.cuda.synchronize()
                check(ref, got, nc, (force_kind, step))
                assert torch.equal(seeds[0], seeds[1]) and torch.equal(counters[0], counters[1])
            # CUDA graph: one captured fused call replayed on new logits; the stock launches give the reference.
            logits, temps, draft = inputs(0, 5, 9)
            fused(logits, temps, draft, 0, 5, seeds[1], force_of(1))  # compile outside the capture
            stock(logits, temps, draft, 0, 5, seeds[0], force_of(0))
            torch.cuda.synchronize()
            MPI.COMM_WORLD.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                got = fused(logits, temps, draft, 0, 5, seeds[1], force_of(1))
            for replay in range(3):
                new_logits, new_temps, new_draft = inputs(0, 5, 20 + replay)
                for static, new in zip((logits, temps, draft), (new_logits, new_temps, new_draft)):
                    static.copy_(new)
                graph.replay()
                ref = stock(logits, temps, draft, 0, 5, seeds[0], force_of(0))
                torch.cuda.synchronize()
                check(ref, got, 0, (force_kind, "replay", replay))
                assert torch.equal(seeds[0], seeds[1]) and torch.equal(counters[0], counters[1])
            del graph

        # The split (programs per row) follows the row count: launches at other splits reuse the compiled kernel
        # and draw the same tokens.
        from tensorrt_llm._torch.distributed import ll_exchange
        from tensorrt_llm._torch.speculative import fused_accept

        real_split = ll_exchange._fast_gumbel_split
        seeds = [torch.full((1,), 11, dtype=torch.int64, device="cuda") for _ in range(2)]
        compiled = triton_num_compiled(fused_accept._sample_accept_kernel)
        try:
            for step, split in enumerate((4, 6, 8, 2)):
                ll_exchange._fast_gumbel_split = lambda m, v_shard, num_sms, s=split: s
                logits, temps, draft = inputs(0, 5, 40 + step)
                ref = stock(logits, temps, draft, 0, 5, seeds[0], None)
                got = fused(logits, temps, draft, 0, 5, seeds[1], None)
                torch.cuda.synchronize()
                check(ref, got, 0, ("split", split))
        finally:
            ll_exchange._fast_gumbel_split = real_split
        assert triton_num_compiled(fused_accept._sample_accept_kernel) == compiled
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
        # An epoch is a plain counter whose parity picks the slot half: exchanges across its int32 wrap (-1 -> 0 -> 1)
        # deliver like any other. Every rank's previous launch has finished before any rank moves its epochs.
        MPI.COMM_WORLD.barrier()
        ex._state.fill_(-1)

        def wrap_x(r):
            g = torch.Generator(device="cuda").manual_seed(1000 + r)
            return torch.randn((64, 4, 256), generator=g, device="cuda").to(torch.bfloat16)

        x = wrap_x(rank)
        ref = torch.cat([wrap_x(r).reshape(64, -1) for r in members], dim=-1)
        blocks = 64 * 4 * 256 * 2 // 4 // 256
        for launch in range(2):
            out = ex.allgather(x)
            torch.cuda.synchronize()
            assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), (
                "allgather across the wrap",
                launch,
            )
        assert (ex._state[0, :blocks] == 1).all(), ex._state[0, :blocks]
        # Refused before any launch: an integer payload (its all-ones words would arrive rewritten), a pushed
        # merge of two partials per row (its producer pushes one), and an in-kernel gather on an exchange created
        # without its head dim.
        for dtype in (torch.int32, torch.float64):
            with pytest.raises(ValueError, match="floating-point"):
                ex.allgather(torch.zeros(4, 256, dtype=dtype, device="cuda"))
        o = torch.zeros(group_size, 2, 4, 4, 256, dtype=torch.bfloat16, device="cuda")
        lse = torch.zeros(group_size, 2, 4, 4, device="cuda")
        out = torch.empty(4, 4, 256, dtype=torch.bfloat16, device="cuda")
        with pytest.raises(ValueError, match="one partial"):
            ex.alltoall_merge(o, lse, out, pushed=True)
        with pytest.raises(ValueError, match="head dim"):
            ex.rope_gather_args(2, 4, 256)
    finally:
        MPI.COMM_WORLD.barrier()
        ex.close()


def _rope_gather(tp):
    """The RoPE / FP8-append kernel's in-kernel q all-gather (fused_qkv_rope_gate_fp8_append_gather) is bit-identical
    to the plain kernel followed by TPGroupExchange.allgather of its FP8 query: every output, the KV pool and the
    gathered q, over repeated calls of changing token counts and CUDA-graph replays (epochs keep advancing)."""
    from tensorrt_llm._torch.attention.backends.fmha.gqa_decode import KvAppendPlan
    from tensorrt_llm._torch.attention.kernels.fused_qk_norm_rope_gate import (
        fused_qkv_rope_gate_fp8_append,
        fused_qkv_rope_gate_fp8_append_gather,
    )

    mapping, rank = _mapping(tp)
    hq, hkv, d, rot, page, max_pages = 4, 1, 256, 64, 64, 8
    mrope, channel = (11, 11, 10), 2
    ex = TPGroupExchange.create(
        mapping, tp, max_bytes=1 << 20, max_merge_rows=64, num_channels=3, gather_head_dim=d
    )
    assert ex is not None, "no peer memory for the exchange"
    g = torch.Generator(device="cuda").manual_seed(1)
    q_weight = (torch.randn(d, generator=g, device="cuda") * 0.1).to(torch.bfloat16)
    k_weight = (torch.randn(d, generator=g, device="cuda") * 0.1).to(torch.bfloat16)
    cos_sin = torch.randn(4096, 2, rot // 2, generator=g, device="cuda")
    scale = torch.tensor([0.05], device="cuda")
    inv_scale = 1.0 / scale

    def call_inputs(q_len, b, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        seq = torch.randint(q_len, page * max_pages, (b,), generator=g, device="cuda").to(
            torch.int32
        )
        tables = torch.randperm(b * 2 * max_pages + 4, generator=g, device="cuda")[
            : b * 2 * max_pages
        ]
        tables = tables.to(torch.int32).view(b, 2, max_pages).contiguous()
        pos = (seq[:, None] - q_len + torch.arange(q_len, device="cuda")).reshape(-1)
        positions = pos.expand(3, -1).contiguous()
        # q / gate / K / V differ per member, as the heads of a TP rank do
        gq = torch.Generator(device="cuda").manual_seed(seed * 101 + rank)
        qkv = torch.randn(b * q_len, (2 * hq + 2 * hkv) * d, generator=gq, device="cuda")
        return qkv.to(torch.bfloat16), positions, seq, tables

    def plan_of(q_len, b, pool, seq, tables):
        return KvAppendPlan(0, b * q_len, b, q_len, pool, tables, seq, inv_scale, scale, 1.0 / 16)

    def run(gather, q_len, b, qkv, positions, seq, tables, pool):
        plan = plan_of(q_len, b, pool, seq, tables)
        args = (qkv, q_weight, k_weight, cos_sin, positions, 1e-6, hq, hkv, d, rot, mrope, plan)
        if gather:
            ga = (*ex.rope_gather_args(channel, b * q_len * hq, d), True)
            return fused_qkv_rope_gate_fp8_append_gather(*args, ga)
        outs = fused_qkv_rope_gate_fp8_append(*args)
        return (*outs, ex.allgather(outs[2], channel=0))

    def pools(b):
        return [
            torch.zeros(
                b * 2 * max_pages + 4, hkv, page, d, dtype=torch.float8_e4m3fn, device="cuda"
            )
            for _ in range(2)
        ]

    def check(got, ref, pool_got, pool_ref, what):
        torch.cuda.synchronize()
        # (qkv, gate, q8, scales, gathered q); the kernels write scales [0:5] only
        got, ref = list(got), list(ref)
        got[3], ref[3] = got[3][:5], ref[3][:5]
        for i, (x, y) in enumerate(zip(got, ref)):
            assert x.shape == y.shape and torch.equal(x.view(torch.uint8), y.view(torch.uint8)), (
                what,
                i,
            )
        assert torch.equal(pool_got.view(torch.uint8), pool_ref.view(torch.uint8)), (
            what,
            "kv pool",
        )

    try:
        for step, (q_len, b) in enumerate(((7, 3), (1, 5), (7, 8), (7, 3), (1, 1))):
            inputs = call_inputs(q_len, b, step)
            pool_g, pool_p = pools(b)
            got = run(True, q_len, b, *inputs, pool_g)
            ref = run(False, q_len, b, *inputs, pool_p)
            check(got, ref, pool_g, pool_p, ("eager", step))

        # CUDA graph: one captured gather call replayed on new inputs.
        q_len, b = 7, 4
        qkv, positions, seq, tables = call_inputs(q_len, b, 100)
        pool_g, pool_p = pools(b)
        run(True, q_len, b, qkv, positions, seq, tables, pool_g)  # compile outside the capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            got = run(True, q_len, b, qkv, positions, seq, tables, pool_g)
        for replay in range(4):
            new = call_inputs(q_len, b, 200 + replay)
            for static, x in zip((qkv, positions, seq, tables), new):
                static.copy_(x)
            pool_g.zero_()
            pool_p.zero_()
            graph.replay()
            ref = run(False, q_len, b, qkv, positions, seq, tables, pool_p)
            check(got, ref, pool_g, pool_p, ("replay", replay))
            # an eager gather call between replays shares the epochs
            eager = run(True, q_len, b, qkv, positions, seq, tables, pools(b)[0])
            torch.cuda.synchronize()
            assert torch.equal(eager[4].view(torch.uint8), ref[4].view(torch.uint8)), (
                "eager",
                replay,
            )
        del graph

        # PDL: the positions come from the immediate predecessor, which releases this launch at entry and stores
        # them ~2 ms later; the kernel reads them (and the cos / sin rows they select) only after its wait.
        q_len, b = 7, 3
        qkv, positions, seq, tables = call_inputs(q_len, b, 300)
        pool_g, pool_p = pools(b)
        ref = run(False, q_len, b, qkv, positions, seq, tables, pool_p)
        late = torch.zeros_like(positions)
        late_copy(positions, late, 1000)  # compile outside the timed launch
        torch.cuda.synchronize()
        late.zero_()
        torch.cuda.synchronize()
        MPI.COMM_WORLD.barrier()
        late_copy(positions, late, 2_000_000)
        got = run(True, q_len, b, qkv, late, seq, tables, pool_g)
        check(got, ref, pool_g, pool_p, ("pdl predecessor writes the positions",))
    finally:
        MPI.COMM_WORLD.barrier()
        ex.close()


def _receive_words(ex):
    """This rank's receive buffers of ``ex`` as int32 words (a device copy)."""
    from cuda.bindings import runtime as cudart

    nbytes = ex.channel_bytes * ex.num_channels
    words = torch.empty(nbytes // 4, dtype=torch.int32, device="cuda")
    torch.cuda.synchronize()
    (err,) = cudart.cudaMemcpy(
        words.data_ptr(),
        ex._buffers.local_ptr,
        nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
    )
    assert err == cudart.cudaError_t.cudaSuccess
    return words


def _merge_reference(o_all, lse_all, grank, valid=None, gate=None, extra_o=None, extra_lse=None):
    """alltoall_merge's result at member ``grank`` from every member's ``o`` [G, P, T, Hh, D] / ``lse`` [G, P, T, Hh]:
    the LSE-weighted mean (base 2, FP32) of the valid partials it receives and the extra partial, rounded to BF16,
    then gated."""
    o = torch.stack([x[grank] for x in o_all]).float()  # [source, P, T, Hh, D]
    lse = torch.stack([x[grank] for x in lse_all])  # [source, P, T, Hh]
    ok = lse > float("-inf")
    if valid is not None:
        ok = ok & (valid[:, :, :, None] != 0)
    m = torch.where(ok, lse, float("-inf")).amax(dim=(0, 1))
    if extra_lse is not None:
        m = torch.maximum(m, extra_lse)
    w = torch.where(ok, torch.exp2(lse - m), 0.0)
    num = (w[..., None] * o).sum(dim=(0, 1))
    den = w.sum(dim=(0, 1))
    if extra_o is not None:
        wx = torch.where(extra_lse > float("-inf"), torch.exp2(extra_lse - m), 0.0)
        num = num + wx[..., None] * extra_o.float()
        den = den + wx
    res = torch.where(den[..., None] > 0, num / torch.where(den > 0, den, 1.0)[..., None], 0.0)
    res = res.to(torch.bfloat16)
    if gate is not None:
        res = (res.float() * torch.sigmoid(gate.float())).to(torch.bfloat16)
    return res


def _group_merge(tp):
    """TPGroupExchange's Lamport words against references over every member's inputs: its all-gather (one and
    several blocks per program) against the concatenation, its all-to-all merge (BF16 and E4M3 out, gate, extra
    partial, valid mask, one and two partials per row) against an eager merge, and the RoPE kernel's in-kernel q
    all-gather against the plain kernel and every member's FP8 query; eager and replayed from a CUDA graph. After
    every call each receive word is empty again."""
    from tensorrt_llm._torch.attention.backends.fmha.gqa_decode import KvAppendPlan
    from tensorrt_llm._torch.attention.kernels.fused_qk_norm_rope_gate import (
        fused_qkv_rope_gate_fp8_append,
        fused_qkv_rope_gate_fp8_append_gather,
    )
    from tensorrt_llm._torch.distributed.ll_exchange import LAMPORT_EMPTY

    mapping, rank = _mapping(tp)
    hq, hkv, d, rot, page, max_pages, mrope = 4, 1, 256, 64, 64, 8, (11, 11, 10)
    ex = TPGroupExchange.create(
        mapping, tp, max_bytes=1 << 20, max_merge_rows=1024, num_channels=3, gather_head_dim=d
    )
    assert ex is not None, "no peer memory for the exchange"
    g = torch.Generator(device="cuda").manual_seed(1)
    q_weight = (torch.randn(d, generator=g, device="cuda") * 0.1).to(torch.bfloat16)
    k_weight = (torch.randn(d, generator=g, device="cuda") * 0.1).to(torch.bfloat16)
    cos_sin = torch.randn(4096, 2, rot // 2, generator=g, device="cuda")
    scale = torch.tensor([0.05], device="cuda")
    out_scale = torch.tensor([0.37], device="cuda")

    def inputs(step, member):
        """Every input of one step's calls from ``member``; each member's own data differs from its peers'."""
        gr = torch.Generator(device="cuda").manual_seed(step * 101 + member)
        rows = (1, 9, 200, 64, 3)[step % 5]  # 200 rows: several 256-word blocks per program
        x = torch.randn((rows, 4, 256), generator=gr, device="cuda").to(torch.bfloat16)
        t, p = (5, 16, 3)[step % 3], 1 + step % 2
        o = torch.randn((tp, p, t, hq, d), generator=gr, device="cuda").to(torch.bfloat16)
        lse = torch.randn((tp, p, t, hq), generator=gr, device="cuda") * 4
        lse[0, 0, 0, 0] = float("-inf")
        valid = (torch.rand((tp, p, t), generator=gr, device="cuda") > 0.2).to(torch.int32)
        gate = torch.randn((t, hq, d), generator=gr, device="cuda").to(torch.bfloat16)
        extra_o = torch.randn((t, hq, d), generator=gr, device="cuda").to(torch.bfloat16)
        extra_lse = torch.randn((t, hq), generator=gr, device="cuda") * 4
        q_len, b = (7, 3) if step % 2 else (1, 5)
        # tables / positions: the same on every member
        gs = torch.Generator(device="cuda").manual_seed(step)
        tables = torch.randperm(b * 2 * max_pages + 4, generator=gs, device="cuda")[
            : b * 2 * max_pages
        ]
        tables = tables.to(torch.int32).view(b, 2, max_pages).contiguous()
        seq = torch.randint(q_len, page * max_pages, (b,), generator=gs, device="cuda").to(
            torch.int32
        )
        pos = (seq[:, None] - q_len + torch.arange(q_len, device="cuda")).reshape(-1)
        qkv = torch.randn(b * q_len, (2 * hq + 2 * hkv) * d, generator=gr, device="cuda")
        rope = (q_len, b, qkv.to(torch.bfloat16), pos.expand(3, -1).contiguous(), seq, tables)
        return x, (o, lse, valid, gate, extra_o, extra_lse), rope

    def rope_call(rope, pool, gather):
        q_len, b, qkv, positions, seq, tables = rope
        plan = KvAppendPlan(0, b * q_len, b, q_len, pool, tables, seq, 1.0 / scale, scale, 1.0 / 16)
        args = (qkv, q_weight, k_weight, cos_sin, positions, 1e-6, hq, hkv, d, rot, mrope, plan)
        if gather:
            ga = (*ex.rope_gather_args(2, b * q_len * hq, d), True)
            return list(fused_qkv_rope_gate_fp8_append_gather(*args, ga))
        return list(fused_qkv_rope_gate_fp8_append(*args))

    def new_pool(rope):
        b = rope[1]
        return torch.zeros(
            b * 2 * max_pages + 4, hkv, page, d, dtype=torch.float8_e4m3fn, device="cuda"
        )

    def run(x, merge, rope):
        o, lse, valid, gate, extra_o, extra_lse = merge
        t = o.shape[2]
        gathered = ex.allgather(x, channel=0)
        out = torch.empty((t, hq, d), dtype=torch.bfloat16, device="cuda")
        ex.alltoall_merge(o, lse, out, gate=gate, valid=valid, channel=1)
        out8 = torch.empty((t, hq, d), dtype=torch.float8_e4m3fn, device="cuda")
        ex.alltoall_merge(
            o,
            lse,
            None,
            extra_o=extra_o,
            extra_lse=extra_lse,
            out_fp8=out8,
            out_scale=out_scale,
            channel=1,
        )
        pool = new_pool(rope)
        return [gathered, out, out8, pool, *rope_call(rope, pool, gather=True)]

    def check(got, step, what):
        """``got`` (run's outputs from this member's inputs of ``step``) against the references."""
        members = [inputs(step, m) for m in range(tp)]
        x, merge, rope = members[rank]
        o, lse, valid, gate, extra_o, extra_lse = merge
        gathered, out, out8, pool, *rope_got = got
        torch.cuda.synchronize()
        ref = torch.cat([mx.reshape(mx.shape[0], -1) for mx, _, _ in members], dim=-1)
        assert torch.equal(gathered.view(torch.int16), ref.view(torch.int16)), (what, "allgather")
        o_all = [mm[0] for _, mm, _ in members]
        lse_all = [mm[1] for _, mm, _ in members]
        ref = _merge_reference(o_all, lse_all, rank, valid=valid, gate=gate)
        torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2, msg=str(what))
        ref8 = _merge_reference(o_all, lse_all, rank, extra_o=extra_o, extra_lse=extra_lse)
        ref8 = (ref8.float() * (1.0 / out_scale)).to(torch.float8_e4m3fn).float()
        # within one E4M3 step of the BF16 reference's quant
        step8 = ref8.abs() * 0.125 + 2**-9
        assert bool(((out8.float() - ref8).abs() <= step8).all()), (what, "fp8")
        pool_ref = new_pool(rope)
        rope_ref = rope_call(rope, pool_ref, gather=False)
        q8_all = [rope_call(mr, new_pool(mr), gather=False)[2] for _, _, mr in members]
        rope_ref.append(torch.cat([q.reshape(q.shape[0], -1) for q in q8_all], dim=-1))
        # the kernels write scales [0:5] only
        rope_got[3], rope_ref[3] = rope_got[3][:5], rope_ref[3][:5]
        for i, (a, b) in enumerate(zip(rope_got, rope_ref)):
            assert a.shape == b.shape and torch.equal(a.view(torch.uint8), b.view(torch.uint8)), (
                what,
                "rope",
                i,
            )
        assert torch.equal(pool.view(torch.uint8), pool_ref.view(torch.uint8)), (what, "kv pool")
        # Every member has read its words of this step before any member starts the next one.
        MPI.COMM_WORLD.barrier()
        assert bool((_receive_words(ex) == LAMPORT_EMPTY).all()), (what, "Lamport words left")
        MPI.COMM_WORLD.barrier()

    try:
        assert bool((_receive_words(ex) == LAMPORT_EMPTY).all()), "Lamport buffers start empty"
        for step in range(6):
            check(run(*inputs(step, rank)), step, ("eager", step))
        # CUDA graph: one captured step replayed on new inputs of the same shapes (step 1 + 30 k).
        x, merge, rope = inputs(1, rank)
        run(x, merge, rope)  # compile outside the capture
        check_steps = [1 + 30 * (replay + 1) for replay in range(3)]
        torch.cuda.synchronize()
        MPI.COMM_WORLD.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            got = run(x, merge, rope)
        for replay, step in enumerate(check_steps):
            nx, nmerge, nrope = inputs(step, rank)
            for static, new in zip((x, *merge, *rope[2:]), (nx, *nmerge, *nrope[2:])):
                static.copy_(new)
            got[3].zero_()  # the KV pool the captured RoPE kernel appends to
            graph.replay()
            check(got, step, ("replay", replay))
        del graph
    finally:
        MPI.COMM_WORLD.barrier()
        ex.close()


def _peer_fill_failure(tp):
    """A sentinel fill that fails on one rank declines the peer buffers on every rank (no rank is left waiting in
    the barrier the fill is agreed in)."""
    from cuda.bindings import runtime as cudart

    from tensorrt_llm._torch.distributed.ll_exchange import _PeerBuffers

    mapping, rank = _mapping(tp)
    real = cudart.cudaMemset

    def memset(ptr, value, nbytes):
        # Only the sentinel fill: IpcMemory's own zero fill (before its handle exchange) stays real.
        if rank == 1 and value == 0xFF:
            return (cudart.cudaError_t.cudaErrorInvalidValue,)
        return real(ptr, value, nbytes)

    cudart.cudaMemset = memset
    try:
        buffers = _PeerBuffers.create(mapping, 1 << 20, fill=0xFF)
    finally:
        cudart.cudaMemset = real
    MPI.COMM_WORLD.barrier()
    assert buffers is None, "a rank kept peer buffers whose fill failed on another rank"


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
    "fast_gumbel_sample": _fast_gumbel_sample,
    "sample_accept": _sample_accept,
    "group_allgather": _group_allgather,
    "rope_gather": _rope_gather,
    "group_merge": _group_merge,
    "peer_fill_failure": _peer_fill_failure,
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_ll_exchange(case, mpi_pool_executor):
    tp = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < tp:
        pytest.skip(f"needs {tp} GPUs")
    results = mpi_pool_executor.map(_worker, [CASES[case]] * tp, [tp] * tp)
    assert all(results)


def test_group_exchange_is_cached_per_gather_head_dim(monkeypatch):
    """get_tp_group_exchange returns the exchange created for the same group, channel count and in-kernel gather
    head dim (a PDL-off engine's exchange has no RoPE-gather channel for a later PDL-on one)."""
    from tensorrt_llm._torch.distributed import ll_exchange

    created = []

    def create(cls, mapping, group_size, max_bytes, max_merge_rows, num_channels, gather_head_dim):
        created.append(gather_head_dim)
        return SimpleNamespace(
            transport="ipc", gather_head_dim=gather_head_dim, num_channels=num_channels
        )

    monkeypatch.setattr(ll_exchange.TPGroupExchange, "create", classmethod(create))
    monkeypatch.setattr(ll_exchange, "_GROUP_EXCHANGES", {})
    mapping = SimpleNamespace(tp_group=[0, 1], tp_rank=0, tp_size=2)
    a = ll_exchange.get_tp_group_exchange(mapping, 2, 1 << 20, 64, 3, gather_head_dim=256)
    b = ll_exchange.get_tp_group_exchange(mapping, 2, 1 << 20, 64, 3, gather_head_dim=128)
    assert ll_exchange.get_tp_group_exchange(mapping, 2, 1 << 20, 64, 3, gather_head_dim=256) is a
    assert b.gather_head_dim == 128 and created == [256, 128]
    c = ll_exchange.get_tp_group_exchange(mapping, 2, 1 << 20, 64, 2, gather_head_dim=256)
    assert c is not a and c.num_channels == 2 and a.num_channels == 3
