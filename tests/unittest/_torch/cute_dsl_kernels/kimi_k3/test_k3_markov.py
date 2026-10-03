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
"""``trtllm::k3_markov`` (the Kimi K3 DSpark vanilla-Markov draft chain over a vocab-sharded draft head) on W GPUs of
one NVLink domain (MPI, one rank per GPU), against two references:

- exact: the chain in the arithmetic of the C++ ``trtllm::dspark_markov_chain`` it replaces (the TeKit port's op when
  the build has it, else the same arithmetic in torch, ``Group.emulated_chain``; with both, the 'emulation' column
  compares them) + the TP-gathered greedy sampler (``SpecWorkerBase.greedy_sample_draft_with_tp_gather``: each rank's
  first maximum, an all-gather of (index, value), the first maximum over the ranks) + the next_new_tokens assembly
  (``SpecWorkerBase._prepare_next_new_tokens``). The corrected logits (fp32 bits), the tokens and next_new must be
  bit-identical to it on every rank, and identical across the ranks;
- main's torch chain (``dspark_markov_step_bias``: an F.linear of the bf16 weights, rounded to bf16 once, whose
  summation order differs), fed k3_markov's tokens as the anchors: the corrected logits within one bf16 ulp of the bias
  and every token a global maximum within that tolerance (the 'torch chain' column; the tolerance adds the fp32
  accumulation bound, 2^-16 of the sum of the products' magnitudes, for sums near zero).

Checks (``report``); every case is also rerun (bit-identical) and run with the folded KV-length rewind
(``kv_lens[first + b] = max(kv_lens[first + b] - rewind[b], 0)``: positive, negative and clamped rewinds, every other
output unchanged):

- every split, B = 1 .. 8 requests x K in (1, 3, 7) block positions, fp32 and bf16 base logits [B, K, S] (bf16 is
  compared with the reference fed ``base.float()``), on two vocab shards: S = 10240 (the TP16 slice; every rank
  pushes each exchange entry into ``--copies`` = 16 / W slots, the exchange volume of 16 ranks) and S = 163840 / W.
  ``pick_grid`` must take every split of the TP16 slice; a shard it rejects is reported (not run), and ``supports``
  and the op must reject it too;
- crafted, at several (B, K): exact ties within an 8-row block, across blocks, warps, CTAs and ranks (identical
  markov_w2 rows and base values: the lowest vocabulary index wins), anchors outside the vocabulary (zero bias), NaN
  logits (never win: compared with a NaN-ignoring sampler), rewind and acceptance edges (a partial rewind from row 0,
  an empty one, num_accepted 0, a wider accepted, num_accepted and the accepted rows as the model's [num_contexts:]
  slices);
- mixed steps: G generation requests after num_contexts context requests, num_accepted / accepted_rows the model's
  int32 slices at element num_contexts and the rewind from rewind_first = num_contexts, also against the same call on
  contiguous copies (bit for bit);
- 60 eager calls with B, K and the dtype varying, interleaved with MNNVL all-reduces (and C++ chain calls when the build
  has the op: the Lamport rotations of k3_markov's workspaces and of the all-reduce's, which the C++ chain shares);
- CUDA graphs of [all-reduce, k3_markov, k3_markov with the rewind], B in (1, 8) x K in (1, 7), fp32 and bf16,
  replayed 20 times with rewritten inputs.

Timing (``time``): per split at S = 10240 with the copies, CUDA graphs of back-to-back calls, each call on its own
markov_w2 shard copy (> 200 MB between two reads of one: HBM-cold), ABBA rounds behind an MPI barrier; the max over the
ranks of the median us per call of the production path (the fp32 cast of the bf16 head logits, the C++ chain or else
main's torch chain with the TP-gathered argmax per position, the gathered sampler, the next_new assembly; it exchanges
over the W ranks only) and of k3_markov on the bf16 logits.

    srun -n W --mpi=pmix python3 test_k3_markov.py [report | time] [--copies C] [--skip-perf] [--rounds R]

W = 2 or 4 GPUs of one NVLink domain, every GPU of the node visible to every rank (rank r runs on GPU r % count).
Without a mode the checks run, then the timing (unless ``--skip-perf``); the exit code is nonzero when a check fails.
The checks also run under pytest on W >= 2 ranks (``srun -n W --mpi=pmix python3 -m pytest -p no:cacheprovider
test_k3_markov.py``); with fewer ranks the module is skipped.
"""

import argparse
import contextlib
import os
import statistics
import sys
import traceback
import zlib
from typing import Optional

import pytest
import torch

VOCAB = 163840  # the draft vocabulary: rows of markov_w1 and markov_w2
MARKOV_RANK = 256
TP16_SHARD = VOCAB // 16
HIDDEN = 7168  # the all-reduce's row width
BATCHES = tuple(range(1, 9))
BLOCKS = (1, 3, 7)
DTYPES = (torch.float32, torch.bfloat16)
DTYPE_NAMES = {torch.float32: "fp32", torch.bfloat16: "bf16"}
CRAFTED = {
    "ties": "ties",
    "anchors": "anchors outside the vocabulary",
    "nan": "NaN logits",
    "edges": "rewind / acceptance edges",
}
CRAFTED_SPLITS = ((1, 1), (1, 7), (3, 3), (8, 1), (8, 7))
# Mixed steps: G generation requests after num_contexts context requests (DSpark K = 7, bf16 logits, the TP16 slice).
MIXED_CONTEXTS = (0, 1, 3)
MIXED_GENS = (1, 2, 4, 8)
GRAPH_SPLITS = ((1, 1), (1, 7), (8, 1), (8, 7))
# Local rows tied on every rank (with S / 2 + 3 and S - 5): 3 and 5 share an 8-row block, 11 is the warp's next block,
# then the next warps and CTAs (a CTA owns 128 rows of the TP16 slice, 320 of 163840 / 4).
TIE_ROWS = (3, 5, 11, 19, 67, 131, 323)
HBM_COLD_BYTES = 200 << 20
FIELDS = ("corrected", "tokens", "next_new", "rerun", "ranks", "rewind", "torch", "emulation")
HEADER = (
    "| case | B | K | S | dtype | corrected | tokens | next_new | rerun | ranks identical | rewind | torch chain "
    "| emulation | result |\n"
    "| :-- | --: | --: | --: | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |"
)


def _sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


def _world_size() -> int:
    """MPI ranks of this launch (1 without mpi4py)."""
    try:
        from mpi4py import MPI
    except ImportError:
        return 1
    return MPI.COMM_WORLD.Get_size()


pytestmark = [
    pytest.mark.skipif(not _sm100(), reason="needs SM100 (MNNVL multicast, TMA bulk copies, PDL)"),
    pytest.mark.skipif(
        _world_size() < 2, reason="needs >= 2 MPI ranks: srun -n W --mpi=pmix python3 -m pytest"
    ),
]


def case_seed(*key) -> int:
    """A seed from a case's parameters, the same in every process (unlike hash())."""
    return zlib.crc32(repr(key).encode())


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32)


def identical(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Same shape, dtype and bits (NaN payloads included)."""
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(bits(a), bits(b))


def same_outputs(a, b) -> bool:
    return all(identical(x, y) for x, y in zip(a, b))


def differing_words(a: torch.Tensor, b: torch.Tensor) -> int:
    """32-bit words of ``a`` that differ from ``b`` (all of them when the shapes or dtypes differ)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return max(a.numel(), 1)
    return int((bits(a) != bits(b)).sum())


def acceptance(
    batch: int, block: int, gen, extra_rows: int = 1, extra_width: int = 0, min_accepted: int = 1
):
    """(accepted [B + extra_rows, K + 1 + extra_width], num_accepted [B] in [min_accepted, K + 1], distinct accepted
    rows [B]), int32 and the same on every rank (``gen`` is a shared generator)."""
    rows = batch + extra_rows
    width = block + 1 + extra_width
    accepted = torch.randint(
        0, VOCAB, (rows, width), generator=gen, device="cuda", dtype=torch.int32
    )
    num_accepted = torch.randint(
        min_accepted, block + 2, (batch,), generator=gen, device="cuda", dtype=torch.int32
    )
    accepted_rows = torch.randperm(rows, generator=gen, device="cuda")[:batch].to(torch.int32)
    return accepted, num_accepted, accepted_rows


def next_new_reference(acc, tokens: torch.Tensor) -> torch.Tensor:
    """SpecWorkerBase._prepare_next_new_tokens: [accepted[rows[b], num_accepted[b] - 1], tokens[b, :]]."""
    accepted, num_accepted, rows = acc
    first = accepted[rows.long(), num_accepted.long() - 1].unsqueeze(1)
    return torch.cat([first, tokens], dim=1)


def rewind_spec(batch: int, seed: int, count: Optional[int] = None, first: int = 2, after: int = 1):
    """(kv_lens, rewind, rewind_first, the kv_lens it must leave) for the folded KV-length rewind of ``count`` (B by
    default) rows from ``first``: a rewind past the length (clamped to 0), a negative one (the length grows), one to
    exactly 0, random others; ``after`` untouched rows follow."""
    count = batch if count is None else count
    gen = torch.Generator().manual_seed(seed)
    kv_lens = torch.randint(0, 50, (first + count + after,), generator=gen, dtype=torch.int32)
    amounts = torch.randint(-6, 9, (count,), generator=gen, dtype=torch.int32)
    amounts[0] = kv_lens[first] + 5
    if count > 1:
        amounts[1] = -1 - amounts[1].abs()
    if count > 2:
        amounts[2] = kv_lens[first + 2]
    want = kv_lens.clone()
    want[first : first + count] = (kv_lens[first : first + count] - amounts).clamp_min(0)
    return kv_lens.cuda(), amounts.cuda(), first, want.cuda()


class Group:
    """This rank of the W-rank TP group: the mapping, the MNNVL all-reduce (whose workspace the C++ chain shares), the
    Markov weights (the same on every rank), the C++ chain's scratch, and the reference path."""

    def __init__(self, copies: Optional[int] = None):
        from mpi4py import MPI

        self.comm = MPI.COMM_WORLD
        self.rank, self.world = self.comm.Get_rank(), self.comm.Get_size()
        gpus = torch.cuda.device_count()
        torch.cuda.set_device(self.rank % gpus)
        import tensorrt_llm  # noqa: F401  (registers the trtllm ops)
        from tensorrt_llm._torch.cute_dsl_kernels.k3_markov import op
        from tensorrt_llm._torch.distributed import AllReduceFusionOp, AllReduceParams, allgather
        from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce
        from tensorrt_llm.mapping import Mapping

        self.op, self.kernel, self.allgather = op, op._kernel_module(), allgather
        # The C++ chain k3_markov replaces: the TeKit port registers it, main does not.
        try:
            self.cpp = torch.ops.trtllm.dspark_markov_chain
        except (AttributeError, RuntimeError):
            self.cpp = None
        self.copies = copies or max(1, 16 // self.world)
        self.mapping = Mapping(
            world_size=self.world, rank=self.rank, gpus_per_node=gpus, tp_size=self.world
        )
        # The model's all-reduces own this workspace; the C++ chain exchanges through its Lamport buffers too.
        self.all_reduce_module = MNNVLAllReduce(self.mapping, torch.bfloat16)
        self.mnnvl_workspaces = MNNVLAllReduce.allreduce_mnnvl_workspaces
        weights = torch.Generator(device="cuda").manual_seed(7)
        self.w1 = (
            torch.randn(VOCAB, MARKOV_RANK, generator=weights, device="cuda") * 0.25
        ).bfloat16()
        self.w2 = (
            torch.randn(VOCAB, MARKOV_RANK, generator=weights, device="cuda") * 0.25
        ).bfloat16()
        local = torch.Generator(device="cuda").manual_seed(100 + self.rank)
        self.ar_input = torch.randn(8, HIDDEN, generator=local, device="cuda").bfloat16()
        self.ar_params = AllReduceParams(
            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
            residual=torch.zeros(8, HIDDEN, dtype=torch.bfloat16, device="cuda"),
            norm_weight=torch.ones(HIDDEN, dtype=torch.bfloat16, device="cuda"),
            eps=1e-5,
        )
        # The C++ chain's grid-sync words (zero; every launch leaves them zero) and per-CTA partial maxima, for the
        # largest split: dsparkMarkovSyncWords = K + K B + 1, dsparkMarkovPartials = K B grid pairs, grid <= the SMs.
        self.sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        block, batch = max(BLOCKS), max(BATCHES)
        self.sync_words = torch.zeros(block + block * batch + 1, dtype=torch.int32, device="cuda")
        self.partials = torch.zeros(
            2 * block * batch * self.sms, dtype=torch.float32, device="cuda"
        )

    def say(self, *args) -> None:
        if self.rank == 0:
            print(*args, flush=True)

    def say_row(self, row: dict) -> None:
        cells = [row[c] for c in ("case", "B", "K", "S", "dtype", *FIELDS)]
        self.say(
            "| " + " | ".join(str(c) for c in cells) + f" | {'PASS' if row['ok'] else 'FAIL'} |"
        )

    def all_ranks(self, flag: bool) -> bool:
        return all(self.comm.allgather(bool(flag)))

    def shards(self):
        """(S, push copies): the TP16 slice with the copies, then this group's full-vocabulary shard."""
        shards = [(TP16_SHARD, self.copies)]
        if VOCAB // self.world != TP16_SHARD:
            shards.append((VOCAB // self.world, 1))
        return shards

    def shard(self, kind: str):
        return (TP16_SHARD, self.copies) if kind == "tp16" else (VOCAB // self.world, 1)

    def shard_weights(self, shard: int) -> torch.Tensor:
        return self.w2[self.rank * shard : (self.rank + 1) * shard]

    def generators(self, seed: int):
        """(shared, local) CUDA generators: shared draws are the same on every rank (anchors, acceptance), local ones
        are this rank's (its logits)."""
        shared = torch.Generator(device="cuda").manual_seed(seed)
        local = torch.Generator(device="cuda").manual_seed(seed + 7919 * (self.rank + 1))
        return shared, local

    def all_reduce(self):
        return self.all_reduce_module(self.ar_input, all_reduce_params=self.ar_params)

    def emulated_chain(self, base, first, w2s, shard: int) -> torch.Tensor:
        """The C++ chain's corrected logits (fp32 [B, K, S]) in torch, position by position with the TP-gathered greedy
        token (NaN never wins) as the next anchor: lane l's FMA chain from +0 over columns 8 l .. 8 l + 7 (the products
        of bf16 values are exact in fp32, so fp32 adds), the 32 lane sums added in pairs at lane distances 16, 8, 4, 2,
        1, the sum rounded to bf16 (nearest even) and added to the fp32 base; an anchor outside the vocabulary adds a
        zero bias."""
        batch, block = base.shape[:2]
        w2f = w2s.float().view(1, shard, 32, 8)
        base32 = base.float()
        prev = first.long()
        out = []
        for k in range(block):
            valid = (prev >= 0) & (prev < VOCAB)
            w1r = self.w1[prev.clamp(0, VOCAB - 1)].float().view(batch, 1, 32, 8)
            products = w2f * w1r
            lanes = torch.zeros(batch, shard, 32, device="cuda")
            for j in range(8):
                lanes = lanes + products[..., j]
            for half in (16, 8, 4, 2, 1):
                lanes = lanes[..., :half] + lanes[..., half : 2 * half]
            bias = lanes[..., 0].bfloat16().float()
            bias = torch.where(valid[:, None], bias, torch.zeros_like(bias))
            corrected = base32[:, k] + bias
            out.append(corrected)
            prev = self.sampler(corrected.unsqueeze(1), shard, ignore_nan=True)[:, 0].long()
        return torch.stack(out, dim=1)

    def chain(self, base, first, w2s, shard: int) -> torch.Tensor:
        """The exact reference's corrected logits: the C++ chain when the build has it, else its torch emulation."""
        if self.cpp is not None:
            return self.cpp_chain(base, first, w2s, shard)
        return self.emulated_chain(base, first, w2s, shard)

    def torch_chain_failures(
        self, base, first, w2s, shard: int, out, ignore_nan: bool = False
    ) -> int:
        """Main's torch chain at every position, fed k3_markov's tokens as the anchors: ``dspark_markov_step_bias``
        (an F.linear of the bf16 weights, one bf16 rounding) added to the fp32 base. The dot products' summation
        orders differ, so k3_markov's corrected logits must be within one bf16 ulp of the bias plus the fp32
        accumulation bound (2^-16 of the sum of the products' magnitudes, which covers a sum near zero) of it, and its
        token a global maximum of it within that tolerance. Returns this rank's failing (b, k) positions."""
        from tensorrt_llm._torch.models.modeling_speculative import (
            dspark_markov_step_bias,
            markov_prev_embeddings,
        )

        corrected, tokens = out[0], out[1]
        base32, lo = base.float(), self.rank * shard
        bad = 0
        for k in range(base.shape[1]):
            prev = first.long() if k == 0 else tokens[:, k - 1].long()
            bias = dspark_markov_step_bias(prev, self.w1, w2s).float()
            ref = base32[:, k] + bias
            magnitude = torch.nn.functional.linear(
                markov_prev_embeddings(prev, self.w1).float().abs(), w2s.float().abs()
            )
            tol = bias.abs() * 2.0**-7 + magnitude * 2.0**-16 + ref.abs().nan_to_num(0.0) * 2.0**-22
            got = corrected[:, k]
            close = ((got - ref).abs() <= tol) | (got.isnan() & ref.isnan())
            masked = ref.masked_fill(ref.isnan(), float("-inf"))
            token = tokens[:, k].long()
            mine = (token >= lo) & (token < lo + shard)
            at = masked.gather(1, (token - lo).clamp(0, shard - 1).unsqueeze(1)).squeeze(1)
            at = torch.where(mine, at, torch.full_like(at, float("-inf")))
            stats = torch.stack(
                [masked.max(dim=-1).values, at, tol.nan_to_num(0.0).max(dim=-1).values]
            )
            every = torch.stack(self.comm.allgather(stats.cpu()))  # [ranks, 3, B]
            best, value, slack = (
                every[:, 0].max(0).values,
                every[:, 1].max(0).values,
                every[:, 2].max(0).values,
            )
            token_ok = value >= best - 2 * slack
            # A NaN wins the plain sampler's row; k3_markov's NaN-free choice is checked by the exact reference.
            if not ignore_nan:
                token_ok |= torch.stack(self.comm.allgather(ref.isnan().any(dim=-1).cpu())).any(0)
            rows_ok = close.all(dim=-1).cpu()
            bad += int((~(rows_ok & token_ok)).sum())
            for b in (
                (~rows_ok).nonzero().flatten().tolist()[:2]
            ):  # the worst element of a failing row, for the log
                excess = ((got[b] - ref[b]).abs() - tol[b]).nan_to_num(float("-inf"))
                v = int(excess.argmax())
                print(f"torch chain: rank {self.rank} position {k} request {b} column {v}: k3 {got[b, v].item():.9g}, "
                      f"main {ref[b, v].item():.9g}, bias {bias[b, v].item():.9g}, sum |products| "
                      f"{magnitude[b, v].item():.6g}, tolerance {tol[b, v].item():.3g}", flush=True)  # fmt: skip
        return bad

    def main_reference(self, base, first, w2s, shard: int, acc):
        """Main's path: its torch chain (``dspark_markov_step_bias``) with the TP-gathered greedy token per position,
        then next_new: (corrected fp32 [B, K, S], tokens int32 [B, K], next_new int32 [B, K + 1])."""
        from tensorrt_llm._torch.models.modeling_speculative import dspark_markov_step_bias

        base32, prev = base.float(), first.long()
        corrected, tokens = [], []
        for k in range(base.shape[1]):
            step = base32[:, k] + dspark_markov_step_bias(prev, self.w1, w2s).float()
            token = self.sampler(step.unsqueeze(1), shard)[:, 0]
            corrected.append(step)
            tokens.append(token)
            prev = token.long()
        tokens = torch.stack(tokens, dim=1)
        return torch.stack(corrected, dim=1), tokens, next_new_reference(acc, tokens)

    def cpp_chain(self, base, first, w2s, shard: int) -> torch.Tensor:
        """trtllm::dspark_markov_chain on fp32 logits (bf16 ones cast with .float(), as the model's head did)."""
        ws = self.mnnvl_workspaces[self.mapping]
        return self.cpp(
            base.float(), first, self.w1, w2s, self.rank * shard, self.sync_words, self.partials,
            ws["uc_buffer"].view(torch.bfloat16).view(3, -1), ws["buffer_flags"],
        )  # fmt: skip

    def sampler(
        self, corrected: torch.Tensor, shard: int, ignore_nan: bool = False
    ) -> torch.Tensor:
        """SpecWorkerBase.greedy_sample_draft_with_tp_gather on [B, K, S] corrected logits: each rank's first maximum
        (NaN as -inf with ``ignore_nan``), the all-gather of (index, value), the first maximum over the ranks."""
        flat = corrected.reshape(-1, shard)
        if ignore_nan:
            flat = flat.masked_fill(flat.isnan(), float("-inf"))
        values, argmax = torch.max(flat, dim=-1, keepdim=True)
        index = (argmax.to(torch.int32) + self.rank * shard).float()
        combined = torch.stack([index, values.float()], dim=-1).flatten(-2)
        gathered = self.allgather(combined, self.mapping, dim=-1)
        best = torch.argmax(gathered[..., 1::2], dim=-1, keepdim=True)
        tokens = torch.gather(gathered[..., 0::2], -1, best).squeeze(-1).to(torch.int32)
        return tokens.view(corrected.shape[:-1])

    def reference(self, base, first, w2s, shard: int, acc, ignore_nan: bool = False):
        """The production path: (corrected fp32 [B, K, S], tokens int32 [B, K], next_new int32 [B, K + 1])."""
        corrected = self.chain(base, first, w2s, shard)
        tokens = self.sampler(corrected, shard, ignore_nan)
        return corrected, tokens, next_new_reference(acc, tokens)

    def k3(self, base, first, w2s, shard: int, acc, copies: int, rewind=None):
        """k3_markov as the model calls it (op.markov_chain); ``rewind``: (kv_lens, rewind, rewind_first, ...)."""
        kv_lens, amounts, rewind_first = (None, None, 0) if rewind is None else rewind[:3]
        return self.op.markov_chain(
            self.mapping, base, first, self.w1, w2s, self.rank * shard, *acc, push_copies=copies,
            kv_lens=kv_lens, rewind=amounts, rewind_first=rewind_first,
        )  # fmt: skip

    def same_on_ranks(self, *tensors) -> bool:
        every = self.comm.allgather([t.cpu() for t in tensors])
        return all(identical(a, b) for other in every[1:] for a, b in zip(every[0], other))

    def summarize(self, failures: dict, **cells) -> dict:
        """A table row from this rank's failure counts, summed over the ranks (a field without a count: '-')."""
        every = self.comm.allgather(failures)
        row = dict(cells, ok=True)
        for field in FIELDS:
            if field not in failures:
                row[field] = "-"
                continue
            count = sum(r[field] for r in every)
            row[field] = "yes" if count == 0 else f"no ({count})"
            row["ok"] = row["ok"] and count == 0
        return row


def shard_support(g: Group, shard: int, copies: int):
    """(rejected splits, consistent, description) of a shard: ``pick_grid`` for every split, ``supports`` agreeing
    with it, and k3_markov raising ValueError on a rejected split."""
    grids = {(b, k): g.op.pick_grid(shard, k, b) for b in BATCHES for k in BLOCKS}
    w2s = g.shard_weights(shard)
    agree = all(
        g.op.supports(torch.empty(b, k, shard, dtype=dtype, device="cuda"), g.w1, w2s) == (grid > 0)
        for (b, k), grid in grids.items()
        for dtype in DTYPES
    )
    rejected = {split for split, grid in grids.items() if grid == 0}
    used = sorted({grid for grid in grids.values() if grid > 0})
    line = f"S = {shard} ({g.world * copies} slots per exchange): "
    if used:
        rows = "/".join(str(shard // grid) for grid in used)
        line += f"pick_grid {'/'.join(map(str, used))} CTAs of {rows} rows; "
    raises = True
    if rejected:
        b, k = min(rejected)
        shared, _ = g.generators(case_seed("rejected", shard))
        base = torch.zeros(b, k, shard, device="cuda")
        first = torch.zeros(b, dtype=torch.long, device="cuda")
        try:
            g.k3(base, first, w2s, shard, acceptance(b, k, shared), copies)
            raises = False
        except ValueError:
            pass
        budget, multiple = g.kernel.SMEM_ROW_BUDGET, g.kernel.WARPS * g.kernel.BLOCK_ROWS
        max_grid = min(g.op.WORKSPACE_MAX_GRID, g.sms) // 2 * 2
        line += (
            f"REJECTED for {len(rejected)} of {len(grids)} splits (rows per CTA a multiple of {multiple}, "
            f"at most SMEM_ROW_BUDGET = {budget}; an even grid of at most min(WORKSPACE_MAX_GRID = "
            f"{g.op.WORKSPACE_MAX_GRID}, {g.sms} SMs): S <= {max_grid * (budget // multiple * multiple)}); "
            f"k3_markov {'raises ValueError' if raises else 'DOES NOT RAISE'}; "
        )
    line += f"supports() {'agrees' if agree else 'DISAGREES'}"
    return rejected, g.all_ranks(agree and raises), line


def check(
    g: Group,
    case: str,
    shard: int,
    copies: int,
    base,
    first,
    w2s,
    acc,
    rewind,
    ignore_nan: bool = False,
    expect: Optional[torch.Tensor] = None,
    empty_rewind: bool = False,
) -> dict:
    """One case: the reference, k3_markov, its rerun and its call with the KV-length rewind (and with an empty one)."""
    batch, block = base.shape[:2]
    ref = g.reference(base, first, w2s, shard, acc, ignore_nan)
    out = g.k3(base, first, w2s, shard, acc, copies)
    again = g.k3(base, first, w2s, shard, acc, copies)
    kv_lens, amounts, _, kv_want = rewind
    rewind_ok = True
    if empty_rewind:  # kv_lens given, nothing to rewind: kv_lens untouched
        spare = kv_lens.clone()
        empty = g.k3(base, first, w2s, shard, acc, copies, rewind=(spare, amounts[:0], 0))
        rewind_ok = identical(spare, kv_lens) and same_outputs(empty, out)
    rewound = g.k3(base, first, w2s, shard, acc, copies, rewind=rewind)
    rewind_ok = rewind_ok and identical(kv_lens, kv_want) and same_outputs(rewound, out)
    tokens_ok = identical(out[1], ref[1]) and (expect is None or identical(out[1], expect))
    failures = dict(
        corrected=differing_words(out[0], ref[0]),
        tokens=int(not tokens_ok),
        next_new=int(not identical(out[2], ref[2])),
        rerun=int(not same_outputs(again, out)),
        ranks=int(not g.same_on_ranks(out[1], out[2])),
        rewind=int(not rewind_ok),
        torch=g.torch_chain_failures(base, first, w2s, shard, out, ignore_nan),
    )
    if g.cpp is not None:
        failures["emulation"] = differing_words(g.emulated_chain(base, first, w2s, shard), ref[0])
    return g.summarize(
        failures, case=case, B=batch, K=block, S=shard, dtype=DTYPE_NAMES[base.dtype]
    )


def random_case(g: Group, shard: int, copies: int, batch: int, block: int, dtype) -> dict:
    seed = case_seed("random", shard, batch, block, dtype)
    shared, local = g.generators(seed)
    base = (torch.randn(batch, block, shard, generator=local, device="cuda") * 3.0).to(dtype)
    first = torch.randint(0, VOCAB, (batch,), generator=shared, device="cuda")
    acc, rewind = acceptance(batch, block, shared), rewind_spec(batch, seed)
    return check(g, "random", shard, copies, base, first, g.shard_weights(shard), acc, rewind)


def crafted_case(
    g: Group, kind: str, shard: int, copies: int, batch: int, block: int, dtype
) -> dict:
    seed = case_seed(kind, shard, batch, block, dtype)
    shared, local = g.generators(seed)
    base = torch.randn(batch, block, shard, generator=local, device="cuda") * 3.0
    first = torch.randint(0, VOCAB, (batch,), generator=shared, device="cuda")
    w2s = g.shard_weights(shard)
    acc = acceptance(batch, block, shared)
    rewind = rewind_spec(batch, seed)
    extra = {}
    if kind == "ties":
        rows = [r for r in TIE_ROWS if r < shard] + [shard // 2 + 3, shard - 5]
        w2s = w2s.clone()
        w2s[rows] = g.w2[11]  # one markov_w2 row, so one bias, at every tie row of every rank
        expect = torch.empty(batch, block, dtype=torch.int32)
        tied = []
        for b in range(batch):
            for k in range(block):
                # The tie set of (b, k): every rank from the lead one, minus the lead's first `drop` rows, so the winner
                # (its first kept row) moves over the levels of the reduction and over the ranks.
                lead, drop = (b + k) % g.world, (b + 2 * k) % (len(rows) - 1)
                expect[b, k] = lead * shard + rows[drop]
                if g.rank >= lead:
                    tied += [(b, k, r) for r in (rows[drop:] if g.rank == lead else rows)]
        if tied:
            base[tuple(torch.tensor(tied, device="cuda").t())] = 40.0
        extra["expect"] = expect.cuda()
    elif kind == "anchors":
        outside = torch.tensor([-1, VOCAB, VOCAB + 5, -100], device="cuda")
        first[0::2] = outside[: (batch + 1) // 2]
    elif kind == "nan":
        nan = float("nan")
        b_idx = torch.arange(batch, device="cuda").repeat_interleave(block)
        k_idx = torch.arange(block, device="cuda").repeat(batch)
        row = 9 + 64 * ((b_idx + 3 * k_idx) % 5)
        base[:, :, 7] = nan  # one row of every rank at every position
        # A dominant value on one rank per (b, k), between NaN rows of its 8-row block on every rank.
        base[b_idx, k_idx, row - 1] = nan
        base[b_idx, k_idx, row + 1] = nan
        mine = (b_idx + k_idx + 1) % g.world == g.rank
        base[b_idx[mine], k_idx[mine], row[mine]] = 50.0
        if g.rank == 1:
            base[0, block // 2] = nan  # a whole position of request 0 on one rank
        extra["ignore_nan"] = True
    else:  # rewind and acceptance edges
        accepted, num_accepted, rows = acceptance(
            batch, block, shared, extra_rows=3, extra_width=2, min_accepted=0
        )
        num_accepted[0] = 0  # accepted column -1: the last one, as torch indexing
        # num_accepted and the accepted rows as the model passes them: [num_contexts:] slices (a 4-byte offset).
        pad = torch.zeros(1, dtype=torch.int32, device="cuda")
        acc = (accepted, torch.cat([pad, num_accepted])[1:], torch.cat([pad, rows])[1:])
        rewind = rewind_spec(batch, seed, count=max(1, batch // 2), first=0, after=3)
        extra["empty_rewind"] = True
    return check(g, CRAFTED[kind], shard, copies, base.to(dtype), first, w2s, acc, rewind, **extra)


def mixed_step_case(g: Group, contexts: int, batch: int, block: int = 7) -> dict:
    """A mixed step as DSparkWorker passes it: ``batch`` generation requests after ``contexts`` context requests,
    num_accepted / accepted_rows the int32 slices [contexts:] of the step's per-request tensors and the KV-length
    rewind on the whole kv_lens from rewind_first = ``contexts``. Against the reference, and bit for bit against the
    same call on contiguous copies (kv_lens[contexts:] from rewind_first 0) in the 'rerun' column."""
    shard, copies = TP16_SHARD, g.copies
    seed = case_seed("mixed", contexts, batch, block)
    shared, local = g.generators(seed)
    base = (torch.randn(batch, block, shard, generator=local, device="cuda") * 3.0).bfloat16()
    first = torch.randint(0, VOCAB, (batch,), generator=shared, device="cuda")
    w2s = g.shard_weights(shard)
    accepted, num_accepted, rows = acceptance(contexts + batch, block, shared)
    sliced = (accepted, num_accepted[contexts:], rows[contexts:])
    copied = (accepted, num_accepted[contexts:].clone(), rows[contexts:].clone())
    kv_lens, amounts, _, kv_want = rewind_spec(batch, seed, first=contexts)
    kv_copy = kv_lens[contexts:].clone()
    ref = g.reference(base, first, w2s, shard, copied)
    out = g.k3(base, first, w2s, shard, sliced, copies, rewind=(kv_lens, amounts, contexts))
    out_copies = g.k3(base, first, w2s, shard, copied, copies, rewind=(kv_copy, amounts, 0))
    failures = dict(
        corrected=differing_words(out[0], ref[0]),
        tokens=int(not identical(out[1], ref[1])),
        next_new=int(not identical(out[2], ref[2])),
        rerun=int(not same_outputs(out, out_copies)),
        ranks=int(not g.same_on_ranks(out[1], out[2])),
        rewind=int(not (identical(kv_lens, kv_want) and identical(kv_copy, kv_want[contexts:]))),
        torch=g.torch_chain_failures(base, first, w2s, shard, out),
    )
    return g.summarize(failures, case=f"mixed step, {contexts} context requests first", B=batch, K=block,
                       S=shard, dtype="bf16")  # fmt: skip


def eager_interleave(g: Group, calls: int = 60) -> dict:
    """``calls`` eager k3_markov calls with B, K and the dtype varying (every fifth on S = 163840 / W when the kernel
    splits it), every other one after an MNNVL all-reduce and every third followed by an extra C++ chain call, each
    against the reference."""
    batches = (1, 8, 2, 5, 3, 7, 4, 6)
    full = VOCAB // g.world
    failures = dict(corrected=0, tokens=0, next_new=0)
    shards, outputs = set(), []
    for i in range(calls):
        batch, block = batches[i % len(batches)], BLOCKS[i % len(BLOCKS)]
        dtype = DTYPES[i % len(DTYPES)]
        on_full = i % 5 == 4 and g.op.pick_grid(full, block, batch) > 0
        shard, copies = (full, 1) if on_full else (TP16_SHARD, g.copies)
        shards.add(shard)
        shared, local = g.generators(case_seed("eager", i))
        base = (torch.randn(batch, block, shard, generator=local, device="cuda") * 3.0).to(dtype)
        first = torch.randint(0, VOCAB, (batch,), generator=shared, device="cuda")
        acc = acceptance(batch, block, shared)
        w2s = g.shard_weights(shard)
        if i % 2 == 0:
            g.all_reduce()
        out = g.k3(base, first, w2s, shard, acc, copies)
        if i % 3 == 0 and g.cpp is not None:
            g.cpp_chain(base, first, w2s, shard)
        ref = g.reference(base, first, w2s, shard, acc)
        failures["corrected"] += int(differing_words(out[0], ref[0]) > 0)
        failures["tokens"] += int(not identical(out[1], ref[1]))
        failures["next_new"] += int(not identical(out[2], ref[2]))
        outputs += [out[1], out[2]]
    failures["ranks"] = int(not g.same_on_ranks(*outputs))
    case = (
        f"{calls} eager calls, all-reduces{' and C++ chains' if g.cpp is not None else ''} between"
    )
    shards = ", ".join(map(str, sorted(shards)))
    return g.summarize(failures, case=case, B="1-8", K="1, 3, 7", S=shards, dtype="fp32, bf16")


def capture(fn, calls: int, stream):
    """A CUDA graph of fn(0) .. fn(calls - 1), after two eager calls (compilation, NCCL and workspace setup)."""
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn(0)
        fn(1)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for i in range(calls):
            fn(i)
    torch.cuda.synchronize()
    return graph


def graph_replays(g: Group, batch: int, block: int, dtype, replays: int = 20) -> dict:
    """A CUDA graph of [MNNVL all-reduce, k3_markov, k3_markov with the KV-length rewind] at S = 10240 with the
    copies, replayed with every input rewritten in place, each replay against the reference."""
    shard, copies = TP16_SHARD, g.copies
    w2s = g.shard_weights(shard)
    shared, _ = g.generators(case_seed("graph", batch, block, dtype))
    base = torch.zeros(batch, block, shard, dtype=dtype, device="cuda")
    first = torch.zeros(batch, dtype=torch.long, device="cuda")
    acc = acceptance(batch, block, shared)
    kv_lens, amounts, rewind_first, _ = rewind_spec(batch, 0)
    outs = {}

    def rewrite(rep: int) -> torch.Tensor:
        """New inputs, in place; returns the kv_lens the replay must leave."""
        seed = case_seed("graph", batch, block, dtype, rep)
        shared, local = g.generators(seed)
        base.copy_(torch.randn(batch, block, shard, generator=local, device="cuda") * 3.0)
        first.copy_(torch.randint(0, VOCAB, (batch,), generator=shared, device="cuda"))
        for dst, src in zip(acc, acceptance(batch, block, shared)):
            dst.copy_(src)
        new_kv_lens, new_amounts, _, want = rewind_spec(batch, seed)
        kv_lens.copy_(new_kv_lens)
        amounts.copy_(new_amounts)
        return want

    def body(_):
        # Held with the graph, so that no allocation inside the graph reuses the all-reduce output's address.
        outs["all_reduce"] = g.all_reduce()
        outs["plain"] = g.k3(base, first, w2s, shard, acc, copies)
        outs["rewind"] = g.k3(
            base, first, w2s, shard, acc, copies, rewind=(kv_lens, amounts, rewind_first)
        )

    rewrite(-1)
    graph = capture(body, 1, torch.cuda.Stream())
    failures = dict(corrected=0, tokens=0, next_new=0, rerun=0, rewind=0)
    outputs = []
    for rep in range(replays):
        want = rewrite(rep)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        ref = g.reference(base, first, w2s, shard, acc)
        plain, rewound = outs["plain"], outs["rewind"]
        failures["corrected"] += int(differing_words(plain[0], ref[0]) > 0)
        failures["tokens"] += int(not identical(plain[1], ref[1]))
        failures["next_new"] += int(not identical(plain[2], ref[2]))
        failures["rerun"] += int(not same_outputs(rewound, plain))
        failures["rewind"] += int(not identical(kv_lens, want))
        outputs += [plain[1].clone(), plain[2].clone()]
    outs.clear()
    del graph
    failures["ranks"] = int(not g.same_on_ranks(*outputs))
    case = f"graph [all-reduce, k3_markov, k3_markov + rewind] x {replays} replays"
    return g.summarize(failures, case=case, B=batch, K=block, S=shard, dtype=DTYPE_NAMES[dtype])


def report(g: Group) -> int:
    """The checks as a markdown table (rank 0); 0 when every check passes on every rank."""
    hosts = g.comm.allgather(f"{os.uname().nodename}:{torch.cuda.current_device()}")
    g.say(
        f"{torch.cuda.get_device_name()} x {g.world} ({', '.join(hosts)}); k3_markov: {g.op.__file__}"
    )
    ok = True
    runs = []
    for shard, copies in g.shards():
        rejected, consistent, line = shard_support(g, shard, copies)
        g.say(line)
        # The TP16 slice must take every split; a larger shard may be beyond the kernel (the model then falls back).
        ok = ok and consistent and not (shard == TP16_SHARD and rejected)
        runs.append((shard, copies, rejected))
    g.say(
        "A failing cell shows its count over the ranks: differing words of the corrected logits, "
        "else failing ranks or calls."
    )
    g.say(
        "Exact reference: "
        + ("trtllm::dspark_markov_chain (C++); 'emulation': its torch emulation against it"
           if g.cpp is not None else "the C++ chain's arithmetic in torch (this build has no C++ chain)")
        + "; 'torch chain': main's torch chain fed k3_markov's tokens, positions beyond one bf16 ulp of the bias plus"
        " the fp32 accumulation bound"
    )  # fmt: skip
    g.say(HEADER)
    rows = []
    for shard, copies, rejected in runs:
        for batch in BATCHES:
            for block in BLOCKS:
                if (batch, block) in rejected:
                    continue
                for dtype in DTYPES:
                    rows.append(random_case(g, shard, copies, batch, block, dtype))
                    g.say_row(rows[-1])
        for kind in CRAFTED:
            for index, (batch, block) in enumerate(CRAFTED_SPLITS):
                if (batch, block) in rejected:
                    continue
                dtype = DTYPES[index % len(DTYPES)]
                rows.append(crafted_case(g, kind, shard, copies, batch, block, dtype))
                g.say_row(rows[-1])
    for contexts in MIXED_CONTEXTS:
        for batch in MIXED_GENS:
            rows.append(mixed_step_case(g, contexts, batch))
            g.say_row(rows[-1])
    rows.append(eager_interleave(g))
    g.say_row(rows[-1])
    for batch, block in GRAPH_SPLITS:
        for dtype in DTYPES:
            rows.append(graph_replays(g, batch, block, dtype))
            g.say_row(rows[-1])
    not_run = [f"S = {shard}: {len(rejected)} splits" for shard, _, rejected in runs if rejected]
    if not_run:
        g.say("Not run (rejected by pick_grid, see above): " + ", ".join(not_run))
    ok = g.all_ranks(ok and all(row["ok"] for row in rows))
    g.say("ALL PASS" if ok else "FAIL")
    return 0 if ok else 1


def replay_us(g: Group, graph, stream) -> float:
    """One replay, started behind an MPI barrier, in us."""
    torch.cuda.synchronize()
    g.comm.Barrier()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record()
        graph.replay()
        end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1e3


def timing(g: Group, rounds: int = 10) -> None:
    """Per split at S = 10240 with the copies: graphs of back-to-back calls of the production path and of k3_markov,
    each call on its own markov_w2 shard copy; the max over the ranks of the median us per call."""
    shard, copies = TP16_SHARD, g.copies
    shard_bytes = shard * MARKOV_RANK * 2
    # One shard copy per call: > 200 MB of other copies between two reads of one.
    calls = -(-HBM_COLD_BYTES // shard_bytes) + 1
    gen = torch.Generator(device="cuda").manual_seed(31 + g.rank)
    w2_copies = [
        (torch.randn(shard, MARKOV_RANK, generator=gen, device="cuda") * 0.25).bfloat16()
        for _ in range(calls)
    ]
    stream = torch.cuda.Stream()
    g.say(
        f"{torch.cuda.get_device_name()} x {g.world}; S = {shard}, {g.world * copies} slots per k3_markov "
        f"exchange; graphs of {calls} calls, each on its own markov_w2 shard copy ({calls * shard_bytes >> 20} "
        f"MB: HBM-cold); {rounds} ABBA rounds; max over the ranks of the median us per call"
    )
    chain = "C++ chain" if g.cpp is not None else "main's torch chain"
    g.say(f"| B | K | grid | production (cast, {chain}, sampler, next_new) | k3_markov | speedup |")
    g.say("| --: | --: | --: | --: | --: | --: |")
    for batch in BATCHES:
        for block in BLOCKS:
            shared, local = g.generators(case_seed("time", batch, block))
            inputs = []
            for _ in range(4):
                base = torch.randn(batch, block, shard, generator=local, device="cuda") * 3.0
                first = torch.randint(0, VOCAB, (batch,), generator=shared, device="cuda")
                inputs.append((base.bfloat16(), first, acceptance(batch, block, shared)))

            def production(i):
                base, first, acc = inputs[i % len(inputs)]
                if g.cpp is not None:
                    return g.reference(base, first, w2_copies[i], shard, acc)
                return g.main_reference(base, first, w2_copies[i], shard, acc)

            def fused(i):
                base, first, acc = inputs[i % len(inputs)]
                return g.k3(base, first, w2_copies[i], shard, acc, copies)

            graphs = [capture(production, calls, stream), capture(fused, calls, stream)]
            for graph in graphs:
                replay_us(g, graph, stream)
            times = [[], []]
            for rnd in range(rounds):
                for arm in (0, 1) if rnd % 2 == 0 else (1, 0):
                    times[arm].append(replay_us(g, graphs[arm], stream) / calls)
            medians = g.comm.allgather([statistics.median(t) for t in times])
            prod_us, k3_us = (max(m[arm] for m in medians) for arm in (0, 1))
            grid = g.op.pick_grid(shard, block, batch)
            g.say(
                f"| {batch} | {block} | {grid} | {prod_us:.2f} | {k3_us:.2f} | {prod_us / k3_us:.2f}x |"
            )
            del graphs


# ----------------------------------------------------------------------------------------------------------------
# pytest (srun -n W --mpi=pmix python3 -m pytest -p no:cacheprovider test_k3_markov.py): every rank runs the same
# tests in the same order, and every collective of a test happens before its assert.
# ----------------------------------------------------------------------------------------------------------------

_state = {}


def group() -> Group:
    """This process' rank of the group, built by the first test (its workspaces are allocated collectively)."""
    if "group" not in _state:
        _state["group"] = Group()
    return _state["group"]


@contextlib.contextmanager
def collective():
    """Inference mode; an exception on one rank aborts the job (its peers would wait for it in a collective)."""
    with torch.inference_mode():
        try:
            yield
        except Exception:
            traceback.print_exc()
            from mpi4py import MPI

            MPI.COMM_WORLD.Abort(1)
            raise


def test_shard_support():
    with collective():
        g = group()
        results = [(shard, *shard_support(g, shard, copies)) for shard, copies in g.shards()]
    for shard, rejected, consistent, line in results:
        assert consistent, line
        assert not (shard == TP16_SHARD and rejected), line


@pytest.mark.parametrize("dtype", DTYPES, ids=[DTYPE_NAMES[d] for d in DTYPES])
@pytest.mark.parametrize("block", BLOCKS)
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("shard_kind", ["tp16", "full"])
def test_split(shard_kind, batch, block, dtype):
    with collective():
        g = group()
        shard, copies = g.shard(shard_kind)
        if g.op.pick_grid(shard, block, batch) == 0:
            pytest.skip(f"pick_grid rejects S = {shard} (see test_shard_support)")
        row = random_case(g, shard, copies, batch, block, dtype)
    assert row["ok"], row


@pytest.mark.parametrize(
    "index", range(len(CRAFTED_SPLITS)), ids=[f"{b}x{k}" for b, k in CRAFTED_SPLITS]
)
@pytest.mark.parametrize("kind", list(CRAFTED))
@pytest.mark.parametrize("shard_kind", ["tp16", "full"])
def test_crafted(shard_kind, kind, index):
    batch, block = CRAFTED_SPLITS[index]
    with collective():
        g = group()
        shard, copies = g.shard(shard_kind)
        if g.op.pick_grid(shard, block, batch) == 0:
            pytest.skip(f"pick_grid rejects S = {shard} (see test_shard_support)")
        row = crafted_case(g, kind, shard, copies, batch, block, DTYPES[index % len(DTYPES)])
    assert row["ok"], row


@pytest.mark.parametrize("batch", MIXED_GENS)
@pytest.mark.parametrize("contexts", MIXED_CONTEXTS)
def test_mixed_step(contexts, batch):
    with collective():
        row = mixed_step_case(group(), contexts, batch)
    assert row["ok"], row


def test_eager_interleave():
    with collective():
        row = eager_interleave(group())
    assert row["ok"], row


@pytest.mark.parametrize("dtype", DTYPES, ids=[DTYPE_NAMES[d] for d in DTYPES])
@pytest.mark.parametrize("batch,block", GRAPH_SPLITS, ids=[f"{b}x{k}" for b, k in GRAPH_SPLITS])
def test_graph_replay(batch, block, dtype):
    with collective():
        row = graph_replays(group(), batch, block, dtype)
    assert row["ok"], row


# ----------------------------------------------------------------------------------------------------------------
# Script (srun -n W --mpi=pmix python3 test_k3_markov.py [report | time] ...)
# ----------------------------------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="trtllm::k3_markov op check (see the module docstring)"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("report", "time"),
        help="the checks only, or the timing only (default: the checks, then the timing)",
    )
    parser.add_argument(
        "--copies",
        type=int,
        default=None,
        help="push copies per rank at S = 10240 (default 16 / W: the exchange volume of 16 ranks)",
    )
    parser.add_argument("--skip-perf", action="store_true", help="no timing after the checks")
    parser.add_argument("--rounds", type=int, default=10, help="ABBA timing rounds")
    args = parser.parse_args()
    if args.copies is not None and args.copies < 1:
        parser.error("--copies must be >= 1")
    if not _sm100() or _world_size() < 2:
        print(
            "needs SM100 GPUs and >= 2 MPI ranks: srun -n W --mpi=pmix python3 test_k3_markov.py",
            flush=True,
        )
        return 2
    with torch.inference_mode():
        g = Group(args.copies)
        status = 0 if args.mode == "time" else report(g)
        if args.mode == "time" or (args.mode is None and not args.skip_perf):
            timing(g, args.rounds)
    return status


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # a rank stopping here would leave its peers waiting in a collective
        traceback.print_exc()
        from mpi4py import MPI

        MPI.COMM_WORLD.Abort(1)
