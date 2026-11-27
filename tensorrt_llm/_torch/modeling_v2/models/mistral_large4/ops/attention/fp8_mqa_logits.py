# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ragged FP8 MQA logits: the DSA indexer's prefill scores, in one DeepGEMM call.

Wraps the pybind `tensorrt_llm.deep_gemm.fp8_mqa_logits`, not a
`torch.ops.trtllm.*` op: the ragged prefill scorer has no registered-op form,
and the engine's DSA indexer (`Indexer._call_mqa_logits` in
`tensorrt_llm/_torch/attention/backends/sparse/dsa/indexer.py`) calls the
pybind directly. A call given a split-KV schedule (`mqa_logits_schedule`) goes
through `fp8_fp4_mqa_logits`, of which the legacy pybind is the schedule-less
form: the same kernel, its keys spread over every SM.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.deep_gemm import fp8_fp4_mqa_logits as _deep_gemm_fp8_fp4_mqa_logits
from tensorrt_llm.deep_gemm import fp8_mqa_logits as _deep_gemm_fp8_mqa_logits
from tensorrt_llm.deep_gemm import get_mqa_logits_metadata as _deep_gemm_mqa_logits_metadata
from tensorrt_llm.deep_gemm import get_num_sms as _deep_gemm_num_sms


def mqa_logits_schedule(
    cu_seq_len_k_start: torch.Tensor,
    cu_seq_len_k_end: torch.Tensor,
    num_kv_tokens: int,
    num_heads: int,
) -> torch.Tensor:
    """The split-KV schedule of one `fp8_mqa_logits` call over these windows and keys.

    DeepGEMM's `get_mqa_logits_metadata`: one single-CTA kernel on the current stream
    (no host sync) that prices each 4-row query block (`128 / num_heads` rows) by the
    256-key splits of its window span and cuts the summed splits into one contiguous run
    per SM, so a block's keys may be spread over several SMs. It returns an int32 tensor
    the call takes as `schedule_meta`. The schedule is a function of the windows, the
    key count, the head count and the device's SM count only: a caller that scores the
    same rows and windows against several layers' keys builds it once."""
    return _deep_gemm_mqa_logits_metadata(
        cu_seq_len_k_start, cu_seq_len_k_end, num_kv_tokens, num_heads
    )


#: The DSA indexer of Mistral Large 4: 32 heads of 128, fp8 keys with one fp32
#: scale per key.
_INDEXER = dict(heads=32, head_dim=128)

#: The Mistral Large 4 FP8 target's fullest context step: max_batch_size 16
#: requests packed into its 9216-token budget (no KV reuse, no chunked
#: prefill, so no cached prefix). Lengths are chosen so request boundaries fall
#: inside the kernel's 4-row query blocks, and the first request is a single
#: token.
_PACKED_16 = [
    (0, n) for n in (1, 7, 33, 100, 255, 257, 513, 1023, 1025, 2047, 2049, 931, 299, 301, 201, 174)
]

#: The Mistral Large 4 NVFP4 target's fullest context step: 16 requests packed into
#: its 32768-token budget, the same way. Request starts (and so window starts)
#: at keys 0, 1, 8, 41, 141, 396, 653, 1166, 2189, 3214, 5261, 7310, 11405,
#: 15502, 23693 and 31886; windows up to 8193 keys long.
_PACKED_16_32K = [
    (0, n)
    for n in (1, 7, 33, 100, 255, 257, 513, 1023, 1025, 2047, 2049, 4095, 4097, 8191, 8193, 882)
]

#: The NVFP4 tep4, tp4 and tep8 deployments at their 1M envelope (max_seq_len 1048576,
#: chunked prefill in 32768-token steps): the last chunk of a 1048576-token
#: prompt, behind its 1015808-token cached prefix. Each chunk is scored against
#: the request's whole KV, cached + new, so this step's keys are 1048576. The
#: dep4 deployment's cells use it too: the 32nd and last indexer chunk of a
#: 1048576-token prompt at its 1M envelope (chunked prefill in 32768-token
#: chunks), 32768 new tokens over 1015808 cached, so the chunk gathers
#: n_k = 1048576 keys and its row t sees keys [0, 1015809 + t).
_LAST_CHUNK_1M = [(1015808, 32768)]

#: Under the engine's query split (tep4 and tp4, a step of 8192 or more context
#: query tokens) rank 0 scores the first 8192 rows of that chunk against the
#: keys up to its share's end, cached + 8192 = 1024000.
_RANK0_SHARE_1M = [(1015808, 8192)]

#: Rank 1's share of the same chunk: rows 8192..16383 against the keys up to
#: its share's end, cached + 16384 = 1032192.
_RANK1_SHARE_1M = [(1015808, 16384)]

#: 16 requests' 16-token chunk tails behind 61440-token prefixes, packed into
#: one indexer chunk as the engine packs them while sum(q) x sum(kv) stays
#: within its 2**28 packed-logits budget: 256 x 983296 = 251,723,776.
_TAILS_16_1M = [(61440, 16) for _ in range(16)]

#: Two 1048576-token prompts' 1-token final chunks, packed into one indexer
#: chunk: 2 x 2097152 = 4,194,304 packed logits. The second request's keys and
#: window start at 1048576 = 2**20.
_FINAL_TOKENS_2X1M = [(1048575, 1), (1048575, 1)]

#: Three requests' final chunks packed up to the 2**28 budget: two near-1M
#: prompts' 50-token tails and a 277613-token prompt's 13-token tail,
#: 113 x 2373713 = 268,229,569 packed logits. 2373713 keys sit just under the
#: 2,373,856-token per-rank pool of the tp4 deployment's delivered boot, which bounds
#: a packed chunk's keys while block reuse is off.
_FINAL_CHUNKS_POOL = [(1048000, 50), (1048000, 50), (277600, 13)]

#: A packed indexer chunk at the dep4 deployment's 1M envelope: the engine groups
#: whole requests into one chunk while sum(new) x sum(cached + new) <= 2**28,
#: so 16 requests' 16-token last chunks over 65520 cached keys each are one
#: 1048576-key block of 256 rows, each request's windows inside its own 65536
#: keys.
_PACKED_16_TAILS_1M = [(65520, 16) for _ in range(16)]

#: Under the tep8 deployment's query split (eight ranks, a step of 8192 or more
#: context query tokens) rank r scores rows [4096 r, 4096 (r + 1)) of a
#: 32768-token chunk against the keys up to its share's end, cached +
#: 4096 (r + 1): the request's rows up to there are the step. Rank 3's share of
#: the last chunk of a 1048576-token prompt sees 1032192 keys, rank 0's 1019904.
_TEP8_RANK3_SHARE_1M = [(1015808, 16384)]
_TEP8_RANK0_SHARE_1M = [(1015808, 4096)]


class Fp8MqaLogits(OpWrapper):
    """Score every query row against every key of one ragged prefill chunk.

    Returns `[T, N]` fp32 logits (a row-padded view, not contiguous) with

        logits[i, j] = k_scale[j] * sum_h weights[i, h] * relu(q[i, h, :] . k[j, :])

    for `cu_seq_len_k_start[i] <= j < cu_seq_len_k_end[i]`, and -inf
    elsewhere when `clean_logits`. `kv` is the pair `(k [N, D] fp8, k_scale [N]
    fp32)`. The fp8 codes of `q` enter the dot product raw: there is no q-side
    scale, so the caller folds it, with the indexer's `D**-0.5 * H**-0.5`, into
    `weights`.
    """

    ARCHS = frozenset({Arch.SM_103})

    # `seqs` lists each request of one context step as (cached keys, new
    # tokens); token `t` of a request sees its own keys up to `cached + t`.
    # `rows`, when present, is the query tile `[lo, hi)` the call scores,
    # passed as views into the step's buffers the way a tiling caller slices
    # them; without it the call scores the whole step. The "Mistral Large 4 1M"
    # cells are the tep4 and tp4 deployments' calls at their 1M envelope: each
    # scores every indexer prefill chunk in query tiles of floor(2**28 / n_k)
    # rows, cut from the chunk's first row. The "Mistral Large 4 tep8 1M" cells are
    # the tep8 deployment's, which tiles the same way from the first row of its
    # rank's share of the chunk under the engine's query split; `share`, when
    # present, is that share `[lo, hi)` of the step's rows (the whole step below
    # the split threshold), the rows a tile is cut from. Rank 7's last tile at
    # 1048576, 524288, 262144, 131072 and 65536 keys and the 16 packed chunk
    # tails are the "Mistral Large 4 1M" cells of the same calls. `budget`, when
    # present (the dep4 deployment's 1M cells), is the logits budget whose
    # floor(budget / keys)-row tiles the call is one of (the whole step when it
    # has no `rows`).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one request, no cache: causal windows over the chunk's own keys",
            spec=dict(**_INDEXER, seqs=[(0, 2048)]),
        ),
        Cell(
            why="a cached prefix: every window starts at 0 and runs past the chunk",
            spec=dict(**_INDEXER, seqs=[(4096, 1024)]),
        ),
        Cell(
            why="two requests in one call: the second's windows start mid-row",
            spec=dict(**_INDEXER, seqs=[(0, 300), (1700, 500)]),
        ),
        Cell(
            why="a query count that is not a multiple of the 4-row query block",
            spec=dict(**_INDEXER, seqs=[(100, 37)]),
        ),
        Cell(
            why="Mistral Large 4 32768 keys in one call, T = N = 32768: the largest [T, N] the NVFP4 "
            "target's envelope (max_num_tokens = max_seq_len = indexer_max_chunk_size = 32768) "
            "admits, for a caller that does not tile; its [32768, 33024]-element fp32 rows span "
            "4,328,521,728 bytes, rows 16257 on past 2**31 bytes and 32514 on past 2**32",
            spec=dict(**_INDEXER, seqs=[(0, 32768)]),
        ),
        Cell(
            why="Mistral Large 4 32768 keys, the last of the floor(2**24 / 32768) = 512-row query "
            "tiles the NVFP4 target cuts: views 32256 rows into q and weights, windows of "
            "32257..32768 keys reaching the last one",
            spec=dict(**_INDEXER, seqs=[(0, 32768)], rows=(32256, 32768)),
        ),
        Cell(
            why="Mistral Large 4 32768 tokens in one call, 16 packed requests (max_batch_size): a "
            "1-token request, windows up to 8193 keys, request boundaries inside 4-row query "
            "blocks",
            spec=dict(**_INDEXER, seqs=_PACKED_16_32K),
        ),
        Cell(
            why="Mistral Large 4 1M, first 256-row tile at 1048576 keys: rows 0..255 of the last "
            "32768-token chunk of a 1048576-token prompt, behind its 1015808-token cached "
            "prefix, windows of 1015809..1016064 keys; floor(2**28 / 1048576) = 256 rows, a "
            "256 x 1048832 fp32 tile of 1,074,003,968 bytes",
            spec=dict(**_INDEXER, seqs=_LAST_CHUNK_1M, rows=(0, 256)),
        ),
        Cell(
            why="Mistral Large 4 1M, last 256-row tile at 1048576 keys: rows 32512..32767 of the "
            "last chunk, windows of 1048321..1048576 keys reaching the last key at "
            "max_seq_len",
            spec=dict(**_INDEXER, seqs=_LAST_CHUNK_1M, rows=(32512, 32768)),
        ),
        Cell(
            why="Mistral Large 4 tep8 1M, an unsplit 8191-row tail chunk: 8191 new tokens behind "
            "1040385 cached, below the 8192-row split threshold, so one rank scores every row "
            "against 1048576 keys; 31 256-row tiles and this 255-row tail 7936..8190, windows "
            "of 1048322..1048576 keys",
            spec=dict(**_INDEXER, seqs=[(1040385, 8191)], rows=(7936, 8191), share=(0, 8191)),
        ),
    )

    note = """
    FP8 MQA prefill indexer logits with optional split-KV scheduling. Covers causal windows,
    cached prefixes, packed requests, non-multiple-of-4 rows, 32K and 1M query tiles.
    """

    def __call__(
        self,
        q: torch.Tensor,
        kv: Tuple[torch.Tensor, torch.Tensor],
        weights: torch.Tensor,
        cu_seq_len_k_start: torch.Tensor,
        cu_seq_len_k_end: torch.Tensor,
        clean_logits: bool = True,
        max_seqlen_k: int = 0,
        schedule_meta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if schedule_meta is None:
            return _deep_gemm_fp8_mqa_logits(
                q,
                kv,
                weights,
                cu_seq_len_k_start,
                cu_seq_len_k_end,
                clean_logits,
                max_seqlen_k,
            )
        # The legacy pybind is exactly this call without the schedule (DeepGEMM
        # `csrc/apis/attention.hpp`): q unscaled, fp32 logits.
        return _deep_gemm_fp8_fp4_mqa_logits(
            (q, None),
            kv,
            weights,
            cu_seq_len_k_start,
            cu_seq_len_k_end,
            clean_logits,
            max_seqlen_k,
            torch.float32,
            schedule_meta,
        )

    def reference(
        self,
        q: torch.Tensor,
        kv: Tuple[torch.Tensor, torch.Tensor],
        weights: torch.Tensor,
        cu_seq_len_k_start: torch.Tensor,
        cu_seq_len_k_end: torch.Tensor,
        clean_logits: bool = True,
        max_seqlen_k: int = 0,
        schedule_meta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The scores in fp64, rounded once to fp32.

        Outside each row's window the entry is -inf when `clean_logits`, else
        NaN, which the test's gate reads as "unspecified". `schedule_meta` is
        not read: it moves which SM computes a logit, not its value.
        """
        k, k_scale = kv
        qd, kd, wd = q.float().double(), k.float().double(), weights.double()
        n = k.shape[0]
        # One head at a time keeps the transient at [T, N] rather than [T, H, N].
        full = torch.zeros(q.shape[0], n, dtype=torch.float64, device=q.device)
        for h in range(q.shape[1]):
            full += (qd[:, h] @ kd.t()).relu() * wd[:, h : h + 1]
        full *= k_scale.double().unsqueeze(0)
        ks = cu_seq_len_k_start.long().unsqueeze(1)
        ke = cu_seq_len_k_end.long().unsqueeze(1)
        fill = float("-inf") if clean_logits else float("nan")
        if max_seqlen_k == 0:
            cols = torch.arange(n, device=q.device).unsqueeze(0)
            out = full.masked_fill((cols < ks) | (cols >= ke), fill)
        else:
            cols = torch.arange(max_seqlen_k, device=q.device).unsqueeze(0)
            src = (ks + cols).clamp(max=n - 1)
            out = torch.gather(full, 1, src).masked_fill(cols >= ke - ks, fill)
        return out.float()

    def is_valid(
        self,
        q: torch.Tensor,
        kv: Tuple[torch.Tensor, torch.Tensor],
        weights: torch.Tensor,
        cu_seq_len_k_start: torch.Tensor,
        cu_seq_len_k_end: torch.Tensor,
        clean_logits: bool = True,
        max_seqlen_k: int = 0,
        schedule_meta: Optional[torch.Tensor] = None,
    ) -> None:
        if schedule_meta is not None:
            # The kernel trusts the schedule: one built for other windows, another key
            # count or another SM count scores the wrong spans and answers wrongly. Its
            # live words (per-SM starts and split counts, then each query block's span)
            # must be the ones this call's own schedule holds; the pad between them is
            # never written.
            fresh = mqa_logits_schedule(
                cu_seq_len_k_start, cu_seq_len_k_end, kv[0].shape[0], q.shape[1]
            )
            sms = _deep_gemm_num_sms()
            span_off = (3 * sms + 1) // 2 * 2
            blocks = -(-q.shape[0] // (128 // q.shape[1]))
            live = [slice(0, 3 * sms), slice(span_off, span_off + 2 * blocks)]
            assert schedule_meta.dtype == torch.int32 and schedule_meta.numel() >= fresh.numel(), (
                f"schedule_meta {schedule_meta.dtype} [{schedule_meta.numel()}] is not this "
                f"call's int32 [{fresh.numel()}] schedule"
            )
            assert all(torch.equal(schedule_meta[s], fresh[s]) for s in live), (
                "schedule_meta was not built for this call's windows, keys and SM count"
            )
        # fp16 weights are accepted and select a second kernel, which answers
        # at half precision: measured relative RMS error 4.8e-4 against the
        # reference, where the fp32 path measures 6.8e-8.
        assert weights.dtype == torch.float32, (
            f"weights must be fp32; got {weights.dtype}, which selects a kernel that answers "
            f"at half precision"
        )
        if cu_seq_len_k_start.numel():
            # A negative start is not read as 0: with every start at -1, every
            # logit of the call came back -inf.
            assert int(cu_seq_len_k_start.min()) >= 0, "a window starts before key 0"
            # An inverted window is not read as empty: its row came back with
            # finite logits where an empty window gives -inf.
            assert bool((cu_seq_len_k_start <= cu_seq_len_k_end).all()), (
                "a window ends before it starts"
            )


fp8_mqa_logits = Fp8MqaLogits()
