# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A one-GPU rig for the DFlash2 sampled path of the helix targets' workers (helix8, helix16).

The worker is the target's own ``DFlashWorker`` (the engine's constructor, one-rank mapping) with
a stub drafter holding only the candidate selector, at the shapes the selector's kernels take: 32
candidates a slot and a rank of 256. The drafter's logits are full-vocabulary (the global top-32
is this rank's), so ``_sample_block_path`` and ``_verify_sampled_block`` run the target's real
kernels (global top-32, ``rubin_linear``, the selector draft, the target probabilities, the
proposal load, seeded block verification, greedy verification) on a 64-token vocabulary.

:func:`generate` emits tokens by rounds of (sampled draft, block verification) over a Markov
target (``Markov``): each verified position's target logits are the log-transition row of the
position's prefix token, so the target's joint law of the first tokens is known exactly.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch

PACKAGE = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
TARGETS = {
    "helix8": "kimi_k31_pro_nvfp4__sm_107__helix8",
    "helix16": "kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1",
}
VOCAB, TOP_K, RANK, HIDDEN, K = 64, 32, 256, 256, 7
GREEDY_T = 1e-5  # the engine's greedy sentinel temperature
_INT32_MAX = 2**31 - 1


def module(case: str):
    return importlib.import_module(f"{PACKAGE}.{TARGETS[case]}.modeling")


def stub_drafter(seed: int = 3) -> SimpleNamespace:
    """The drafter fields the sampled path reads: the selector's codebooks and projection."""
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape, scale):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16).cuda()

    selector = SimpleNamespace(
        top_k=TOP_K,
        predecessor_codebook=rnd(VOCAB, RANK, scale=0.15),
        successor_codebook=rnd(VOCAB, RANK, scale=0.15),
        hidden_projection_weight=rnd(RANK, HIDDEN, scale=1.0 / math.sqrt(HIDDEN)),
    )
    return SimpleNamespace(has_candidate_selector=True, candidate_selector=selector)


def build_worker(case: str, drafter, *, force: float = 0.0, k: int = K):
    """The target's worker as the engine builds it (one-rank mapping), on the stub drafter."""
    from tensorrt_llm.mapping import Mapping

    spec_config = SimpleNamespace(
        max_draft_len=k,
        attention_backend="VANILLA",
        context_kv_dtype="fp8",
        context_kv_only=False,
        use_rejection_sampling=True,
        tokens_per_gen_step=k + 1,
    )
    worker = module(case).DFlashWorker(spec_config, Mapping(world_size=1, tp_size=1, rank=0))
    worker._draft_model_ref = lambda: drafter
    worker.force_num_accepted_tokens = float(force)
    return worker


def sampled_spec(
    n: int,
    *,
    mode=None,
    temperature: float = 1.0,
    greedy_rows=(),
    seed: int = 1234,
    k: int = K,
    flag: bool = True,
    all_greedy: bool = False,
) -> SimpleNamespace:
    """A generation-only step's speculative metadata (the fields the sampled path reads)."""
    from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode

    dev = torch.device("cuda")
    rows = n * (k + 1)
    request_t = torch.full((n,), GREEDY_T if all_greedy else temperature, device=dev)
    for row in greedy_rows:
        request_t[row] = GREEDY_T
    return SimpleNamespace(
        runtime_draft_len=k,
        draft_tokens=torch.zeros(n * k, dtype=torch.int32, device=dev),
        is_ragged_verify=False,
        enable_penalty=False,
        use_rejection_sampling=flag,
        is_all_greedy_sample=all_greedy,
        is_cuda_graph=False,
        batch_slot_ids=torch.arange(n, dtype=torch.int64, device=dev),
        dummy_slot_row=0,
        max_num_requests=n,
        num_seq_slots=n,
        draft_probs=None,
        advanced_sampling_mode=mode or AdvancedSamplingMode.NO_TOPK_NO_TOPP,
        temperatures=request_t.repeat_interleave(k + 1),
        top_ks=torch.full((rows,), _INT32_MAX, dtype=torch.int32, device=dev),
        top_ps=torch.ones(rows, device=dev),
        min_ps=torch.zeros(rows, device=dev),
        seeds=torch.arange(rows, dtype=torch.int64, device=dev) + seed,
        offsets=torch.zeros(rows, dtype=torch.int64, device=dev),
        request_temperatures=request_t,
        request_seeds=torch.arange(n, dtype=torch.int64, device=dev) * 7919 + seed,
        request_offsets=torch.zeros(n, dtype=torch.int64, device=dev),
        rng_accept_slot=2 * k + 1,
        rng_draft_slot=lambda step: k + 1 + step,
        real_draft_rows=None,
    )


class Markov:
    """A first-order Markov target over the vocabulary: each token's next-token row has mass on
    ``support`` tokens only (top-k style zeros), starting from token 0."""

    def __init__(self, seed: int = 21, support: int = 6) -> None:
        g = torch.Generator().manual_seed(seed)
        logits = 1.5 * torch.randn(VOCAB, VOCAB, generator=g)
        keep = logits.topk(support, dim=-1).indices
        mask = torch.zeros(VOCAB, VOCAB, dtype=torch.bool).scatter_(1, keep, True)
        trans = torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=-1).double()
        self.trans = trans
        # Target logits: log p, a finite floor off the support (exp underflows to 0 in FP32).
        self.log_trans = torch.where(mask, trans.log(), torch.full_like(trans, -1e4)).float().cuda()
        # The drafter's block logits by anchor: slot 0 near the target's row, later slots noise.
        base = 2.0 * torch.randn(VOCAB, K, VOCAB, generator=g)
        base[:, 0] = torch.where(mask, trans.log(), torch.full_like(trans, -8.0)).float() + 0.7 * (
            torch.randn(VOCAB, VOCAB, generator=g)
        )
        self.draft_base = base.contiguous().cuda()

    def tempered(self, temperature: float) -> torch.Tensor:
        p = torch.where(self.trans > 0, self.trans ** (1.0 / temperature), 0.0)
        return p / p.sum(-1, keepdim=True)

    def joint3(self, temperature: float) -> torch.Tensor:
        """The exact law of the first three tokens, ``[V, V, V]`` float64."""
        t = self.tempered(temperature)
        return t[0][:, None, None] * t[:, :, None] * t[None, :, :]

    def greedy_chain(self, length: int) -> list[int]:
        out, last = [], 0
        for _ in range(length):
            last = int(self.trans[last].argmax())
            out.append(last)
        return out


@dataclass
class Bufs:
    """Fixed tensors of a generation (a captured graph reads and writes the same storage)."""

    last: torch.Tensor  # [n] int64, the anchor of the next draft
    lens: torch.Tensor  # [n] int64, tokens emitted so far
    out: torch.Tensor  # [n, width] int64, emitted tokens (-1 unwritten)
    logits: torch.Tensor  # [n * (K + 1), V] FP32 target logits of the step
    hidden: torch.Tensor  # [n, K, H] BF16 drafter hidden states of the block slots
    num_accepted: torch.Tensor  # [n] int32, the last round's counts
    taus: torch.Tensor  # [rounds, n] int64, the rounds' accepted draft counts

    @classmethod
    def make(cls, n: int, rounds: int, seed: int = 5) -> "Bufs":
        g = torch.Generator().manual_seed(seed)
        dev = torch.device("cuda")
        return cls(
            last=torch.zeros(n, dtype=torch.int64, device=dev),
            lens=torch.zeros(n, dtype=torch.int64, device=dev),
            out=torch.full((n, rounds * (K + 1) + 1), -1, dtype=torch.int64, device=dev),
            logits=torch.zeros(n * (K + 1), VOCAB, dtype=torch.float32, device=dev),
            hidden=(torch.randn(n, K, HIDDEN, generator=g) * 0.5).to(torch.bfloat16).to(dev),
            num_accepted=torch.zeros(n, dtype=torch.int32, device=dev),
            taus=torch.zeros(rounds, n, dtype=torch.int64, device=dev),
        )


def generate(
    worker,
    drafter,
    spec,
    markov: Markov,
    bufs: Bufs,
    *,
    rounds: int = 3,
    placeholder: Optional[torch.Tensor] = None,
    offset_stride: int = 1 << 16,
) -> None:
    """``rounds`` rounds of (sampled draft, block verification) from token 0 into ``bufs.out``.

    ``placeholder`` ``[n]`` bool: those rows verify deterministic drafts (a function of the
    anchor) instead of the walk's, as a request verifying placeholder drafts does; the store
    then holds other tokens and the verification takes the point mass. Every round advances the
    requests' offsets, so each round (and each graph replay) draws fresh uniforms.
    """
    n = bufs.last.shape[0]
    attn = SimpleNamespace(num_seqs=n, num_contexts=0)
    cols = torch.arange(K + 1, device=bufs.out.device)
    bufs.last.zero_()
    bufs.lens.zero_()
    bufs.out.fill_(-1)
    for r in range(rounds):
        gen_logits = markov.draft_base[bufs.last]
        drafts = worker._sample_block_path(drafter, gen_logits, bufs.hidden, bufs.last, spec, 0, n)
        if placeholder is not None:
            fixed = ((bufs.last * 7 + 3) % VOCAB).to(torch.int32)
            drafts = torch.where(placeholder[:, None], fixed[:, None].expand(n, K), drafts)
        spec.draft_tokens.copy_(drafts.reshape(-1))
        prefix = torch.cat((bufs.last[:, None], drafts.long()), dim=1)
        bufs.logits.copy_(markov.log_trans[prefix].view(n * (K + 1), VOCAB))
        accepted, num = worker._verify_sampled_block(bufs.logits, attn, spec)
        bufs.num_accepted.copy_(num)
        tau = num.long() - 1
        bufs.taus[r].copy_(tau)
        valid = cols[None, :] <= tau[:, None]
        pos = (bufs.lens[:, None] + cols[None, :]).clamp_max(bufs.out.shape[1] - 1)
        bufs.out.scatter_(1, pos, torch.where(valid, accepted.long(), bufs.out.gather(1, pos)))
        bufs.lens.add_(num.long())
        bufs.last.copy_(accepted.long().gather(1, tau[:, None]).squeeze(1))
        spec.request_offsets.add_(offset_stride)


def chi_square_gate(first: torch.Tensor, exact: torch.Tensor) -> tuple[float, int]:
    """Chi-square of the first three tokens' counts against ``exact``; cells expecting fewer
    than 5 counts are pooled into one. Asserts no count off the support and
    ``chi2 < dof + 6 sqrt(2 dof)``; returns ``(chi2, dof)``."""
    first = first.cpu()
    s = first.shape[0]
    assert bool((first >= 0).all()), "a row emitted fewer than three tokens"
    index = first[:, 0] * VOCAB * VOCAB + first[:, 1] * VOCAB + first[:, 2]
    counts = torch.bincount(index, minlength=VOCAB**3).double()
    expected = exact.flatten() * s
    support = expected > 0
    assert float(counts[~support].sum()) == 0.0, "emitted a token the target gives no mass"
    big = expected >= 5
    chi2 = float(((counts[big] - expected[big]) ** 2 / expected[big]).sum())
    small = support & ~big
    cells = int(big.sum())
    if bool(small.any()):
        e, c = float(expected[small].sum()), float(counts[small].sum())
        chi2 += (c - e) ** 2 / e
        cells += 1
    dof = cells - 1
    assert chi2 < dof + 6 * math.sqrt(2 * dof), (chi2, dof)
    return chi2, dof


def cuda_kernels(fn):
    """``(fn(), [CUDA kernel names in launch order])``."""
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        out = fn()
        torch.cuda.synchronize()
    events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    events.sort(key=lambda e: e.time_range.start)
    return out, [e.name for e in events]
