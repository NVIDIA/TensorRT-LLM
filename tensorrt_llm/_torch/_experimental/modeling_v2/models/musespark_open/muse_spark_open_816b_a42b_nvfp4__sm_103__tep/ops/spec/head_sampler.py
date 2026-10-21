# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The target head's token sampler split over two kernels, with the vocabulary gather folded in.

The built-in head of this target rounds the LM-head GEMM to BF16, all-gathers
the eight vocabulary shards (NCCL ring + ``cat``), widens the row to FP32,
applies the soft cap in four elementwise kernels (``x * m``, ``/ cap``,
``tanh``, ``cap * t``) and samples each row with the shipped fused sampler
(``torch.ops.trtllm.fused_sample_from_logits``): one CTA of 1024 threads per
row, which for a row without top-k / top-p / min-p folds thread ``t``'s
elements ``4 (t + 1024 j) + {0..3}`` into one online-softmax state, reduces the
1024 states, scans their masses, draws one Philox uniform and walks the
crossing thread's elements again.

``SplitSample`` (fast path ``split_sampler``) takes the gathered **BF16** row instead:
kernel A runs the same 1024 per-thread folds with 128 of them per CTA and 8
CTAs per row, applying the soft cap in registers with the torch chain's
rounding statements; kernel B, one 1024-thread CTA per row, runs the shipped
block reduce, scan, Philox draw and walk on the states kernel A wrote. Every
virtual thread folds the same elements in the same order with the same
operations, so the states, the row reductions and the tokens are the built-in
ones bit for bit. A row with a filter runs the shipped kernel body inside
kernel B, reading the BF16 row through the cap.

Above ``MAX_ROWS`` (128) rows the shipped sampler runs a 512-thread block per row
instead (``fusedSamplingTokensNarrowKernel``); a call of more rows runs 512
virtual threads per row the same way (4 CTAs of kernel A per row, a 512-thread
kernel B), up to ``WIDE_MAX_ROWS`` rows (fast path ``wide_split_sampler`` lets the
head take them). ``virtual_threads(rows)`` is the shipped block of a call.

``GatherSplitSample`` (fast path ``lm_head_inbox``) also replaces the gather: each rank's
LM-head GEMM writes its ``[rows, shard]`` output straight into its slot of a
multicast inbox (``HeadGatherWorkspace``) whose words hold a sentinel between
calls (``0xFFFF`` in both BF16 halves: a NaN no GEMM epilogue produces); kernel
A polls this rank's copy, copies every vector into the gathered BF16 row
``[rows, vocab]``, resets the word, and folds. The gathered row keeps the
GEMM's bits, negative zeros included.

Both ops are CUDA C++ built from ``_head_sampler/`` (copies of the shipped
``cpp/tensorrt_llm/kernels/fusedSampling`` headers plus ``head_sampler.cu``)
with ``torch.utils.cpp_extension`` and without ``--use_fast_math`` -- the
shipped library's arithmetic -- as one extension per device architecture
and source version (``library_name``: ``LIBRARY_NAME``, the arch target and
``source_digest``) under torch's extension build root; ``load_library``
builds it there on first use (torch serializes concurrent builds) and loads
it. The per-arch name keeps an sm_103a and an sm_107a build apart when both
share one build root. The digest of the sources and flags keeps builds of
different sources apart: torch rebuilds an extension of an unchanged name only
when ninja finds a source newer than its object, so two source versions
mounted at the same path under one build root (a cache seeded from another
run, or two code versions served one after another from one cache) would
otherwise load whichever build is newest.
"""

from __future__ import annotations

import functools
import hashlib
import threading
from pathlib import Path
from typing import Optional, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.mapping import Mapping

#: The CUDA sources of the library, in ``_head_sampler/``.
SOURCE_DIR = Path(__file__).resolve().parent / "_head_sampler"
SOURCES = (
    "head_sampler.cu",
    "fusedSamplingKernels.h",
    "fusedSamplingKernelsInternal.h",
    "fusedSamplingKernelsCommon.cuh",
)
#: nvcc flags. No ``--use_fast_math``: the shipped sampler library is built without it,
#: and with it the soft cap's ``tanhf`` and the sampler's arithmetic change bits.
NVCC_FLAGS = (
    "-O3",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-lineinfo",
)
#: Prefix of the torch extension the library is built and loaded as (``library_name``).
LIBRARY_NAME = "musespark_open_head_sampler"

#: Threads of the shipped single-CTA block at up to 128 rows; one virtual thread each.
VIRTUAL_THREADS = 1024
#: Threads of the shipped single-CTA block above 128 rows (its 512-thread narrow block).
NARROW_VIRTUAL_THREADS = 512
#: 32-bit words of one virtual thread's online-softmax state (max, sum, argmax).
STATE_WORDS = 3
#: Rows of the shipped sampler's 1024-thread single-CTA block (its
#: ``kWideBlockMaxRows``), 16 requests of a K=7 verify block; the head's default row cap.
MAX_ROWS = 128
#: Most rows ``SplitSample`` takes (128 requests of a K=7 verify block); above
#: ``MAX_ROWS`` it reproduces the shipped 512-thread block.
WIDE_MAX_ROWS = 1024
#: Rows of one rank's slot of the head-gather inbox (``HeadGatherWorkspace``):
#: one request's last accepted token and 7 drafts.
INBOX_ROWS = 8
#: Threads per CTA of kernel A (8 CTAs per row).
FOLD_THREADS = 128
#: A poll waiting longer than this traps: a peer that never published. The trap ends the
#: CUDA context (where the built-in NCCL all-gather would keep waiting), so any skew between
#: two ranks' LM-head GEMMs above it -- a debugger or profiler pausing one rank, a host stall --
#: is fatal.
POLL_TIMEOUT_S = 600
#: Vectors (4 elements) of one virtual thread kernel B's walk stages in shared memory.
WALK_VECTORS = 64

_LOCK = threading.Lock()
_LOADED: Optional[str] = None


def arch_target() -> str:
    """The arch-specific target of the current device: ``103a`` or ``107a``."""
    capability = torch.cuda.get_device_capability()
    if capability == (10, 3):
        return "103a"
    elif capability == (10, 7):
        return "107a"
    raise RuntimeError(
        f"the head sampler is built for sm_103a / sm_107a, not sm_{capability[0]}{capability[1]}"
    )


@functools.lru_cache(maxsize=None)
def source_digest() -> str:
    """12 hex digits of SHA-256 over the library's sources (``SOURCES``) and ``NVCC_FLAGS``."""
    digest = hashlib.sha256()
    for name in SOURCES:
        digest.update(name.encode() + b"\0")
        digest.update((SOURCE_DIR / name).read_bytes())
    digest.update("\0".join(NVCC_FLAGS).encode())
    return digest.hexdigest()[:12]


def library_name(target: str) -> str:
    """The torch extension name of the library built for ``target`` (``arch_target``) from these sources."""
    return f"{LIBRARY_NAME}_sm{target}_{source_digest()}"


def build(verbose: bool = False) -> str:
    """Compile (if needed) and load the library under torch's extension build root; its path."""
    from torch.utils.cpp_extension import load

    target = arch_target()
    name = library_name(target)
    path = load(
        name=name,
        sources=[str(SOURCE_DIR / SOURCES[0])],
        extra_cuda_cflags=list(NVCC_FLAGS) + [f"-gencode=arch=compute_{target},code=sm_{target}"],
        extra_include_paths=[str(SOURCE_DIR)],
        is_python_module=False,
        verbose=verbose,
    )
    return str(path) if path else name


def load_library() -> str:
    """Build and load the library once per process; returns its path."""
    global _LOADED
    with _LOCK:
        if _LOADED is None:
            _LOADED = build()
        return _LOADED


def load_library_agreed(mapping: Mapping) -> str:
    """``load_library`` on every rank of ``mapping``'s TP group, then an agreement (collective).

    Succeeds on every rank or raises on every rank, so a rank whose load fails
    does not leave its peers waiting in the next collective.
    """
    from tensorrt_llm._torch.distributed.ops import (
        _get_mnnvl_workspace_comm,
        _mnnvl_workspace_all_succeeded,
    )

    comm = _get_mnnvl_workspace_comm(mapping)
    error: Optional[BaseException] = None
    path = ""
    try:
        path = load_library()
    except Exception as exc:  # noqa: BLE001 - every failure must reach the agreement
        error = exc
    if not _mnnvl_workspace_all_succeeded(comm, error is None):
        raise RuntimeError(
            "[musespark_open] split_sampler: the head sampler library failed to load on at least one rank"
            f"{f' (this rank: {error!r})' if error is not None else ''}"
        ) from error
    return path


@functools.lru_cache(maxsize=None)
def cap_constants(multiplier: float, soft_cap: float) -> Tuple[float, float, float]:
    """The FP32 values the torch chain uses: ``fp32(m)``, ``fp32(1) / fp32(cap)``, ``fp32(cap)``.

    ``logits * m`` and ``cap * t`` convert the Python scalar to FP32; ``logits / cap``
    multiplies by the FP32 reciprocal ATen computes on the host in FP32.
    """
    m = torch.tensor(multiplier, dtype=torch.float32)
    cap = torch.tensor(soft_cap, dtype=torch.float32)
    inv = torch.tensor(1.0, dtype=torch.float32) / cap
    return float(m), float(inv), float(cap)


def reference_capped(logits: torch.Tensor, multiplier: float, soft_cap: float) -> torch.Tensor:
    """The built-in head's soft cap (``MuseSparkOpenLogitsProcessor``), statement for statement."""
    x = logits.float()
    x = x.float() * multiplier
    return soft_cap * torch.tanh(x / soft_cap)


def virtual_threads(rows: int) -> int:
    """The shipped sampler's threads per row for a call of ``rows`` rows."""
    return VIRTUAL_THREADS if rows <= MAX_ROWS else NARROW_VIRTUAL_THREADS


def state_words(rows: int) -> int:
    """32-bit words of kernel A's per-virtual-thread states for a call of ``rows`` rows."""
    return rows * virtual_threads(rows) * STATE_WORDS


def new_states(rows: int, device: torch.device) -> torch.Tensor:
    """Scratch for kernel A's per-virtual-thread states, for every call of at most ``rows`` rows."""
    words = max(state_words(r) for r in {min(rows, MAX_ROWS), rows})
    return torch.empty(words, dtype=torch.int32, device=device)


def _neutral_or_filtered_rows_ok(vocab: int) -> bool:
    # The walk stages WALK_VECTORS vectors per 1024-thread virtual thread, twice that per
    # 512-thread one: the same vocabulary bound.
    return vocab % 4 == 0 and vocab // 4 <= WALK_VECTORS * VIRTUAL_THREADS


class SplitSample(OpWrapper):
    """One token per row of BF16 head logits, sampled as the built-in head samples their soft-capped FP32 row.

    ``logits`` ``[rows, vocab]`` BF16 are the LM head's gathered output before the
    soft cap; ``multiplier`` and ``soft_cap`` are the processor's constants. The
    other arguments are ``fused_sample_from_logits``'s. ``states`` is scratch of
    ``new_states(rows)``. Returns int32 ``[rows]``: bitwise the tokens
    ``fused_sample_from_logits(soft_cap * tanh(logits.float() * multiplier / soft_cap), …)``
    returns.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="the target's verify block: 8 neutral rows (T 1), per-row RNG, N(0, 40) logits",
            spec=dict(rows=8, kind="n40", temperature=1.0, rng="row"),
        ),
        Cell(
            why="shared RNG entry: the row index is the Philox subsequence",
            spec=dict(rows=8, kind="n10", temperature=1.0, rng="shared"),
        ),
        Cell(
            why="wide logits: a few tokens carry the mass",
            spec=dict(rows=8, kind="n120", temperature=1.0, rng="row"),
        ),
        Cell(
            why="spiky row: 64 outliers",
            spec=dict(rows=8, kind="spiky", temperature=1.0, rng="row"),
        ),
        Cell(
            why="one dominant token: the walk crosses on its first elements",
            spec=dict(rows=8, kind="peaked", temperature=1.0, rng="row"),
        ),
        Cell(
            why="flat row: every thread carries mass, the crossing thread anywhere",
            spec=dict(rows=8, kind="flat", temperature=1.0, rng="row"),
        ),
        Cell(
            why="saturated soft cap: ties after the cap, argmax tie-breaking",
            spec=dict(rows=8, kind="sat", temperature=1.0, rng="row"),
        ),
        Cell(
            why="temperature other than 1 on a neutral row",
            spec=dict(rows=8, kind="n40", temperature=0.7, rng="row"),
        ),
        Cell(
            why="greedy sentinel temperature on a neutral row (1 / 1e-6 scaling)",
            spec=dict(rows=8, kind="n40", temperature=0.0, rng="row"),
        ),
        Cell(
            why="one row (a single-row call counts as per-row RNG)",
            spec=dict(rows=1, kind="n40", temperature=1.0, rng="row"),
        ),
        Cell(
            why="filtered rows fall back to the shipped body: top-k, top-p, min-p, mixed with neutral rows",
            spec=dict(rows=8, kind="n40", temperature=1.0, rng="row", filters="mixed"),
        ),
        Cell(
            why="16 rows: the verify blocks of 2 requests",
            spec=dict(rows=16, kind="n40", temperature=1.0, rng="row"),
        ),
        Cell(
            why="64 rows: the verify blocks of 8 requests",
            spec=dict(rows=64, kind="n40", temperature=1.0, rng="row"),
        ),
        Cell(
            why="128 rows: 16 requests, the shipped sampler's widest 1024-thread call",
            spec=dict(rows=128, kind="n40", temperature=1.0, rng="row"),
        ),
        Cell(
            why="128 rows, shared RNG entry: the row index is the Philox subsequence",
            spec=dict(rows=128, kind="spiky", temperature=0.7, rng="shared"),
        ),
        Cell(
            why="128 rows with filtered rows on the shipped body, mixed with neutral rows",
            spec=dict(rows=128, kind="n40", temperature=1.0, rng="row", filters="mixed"),
        ),
        Cell(
            why="129 rows: the shipped sampler's first 512-thread call",
            spec=dict(rows=129, kind="n40", temperature=1.0, rng="row"),
        ),
        Cell(
            why="256 rows (32 requests), spiky rows, shared RNG entry",
            spec=dict(rows=256, kind="spiky", temperature=0.7, rng="shared"),
        ),
        Cell(
            why="256 rows with filtered rows on the shipped 512-thread body",
            spec=dict(rows=256, kind="n40", temperature=1.0, rng="row", filters="mixed"),
        ),
        Cell(
            why="512 rows (64 requests), flat rows: the crossing thread anywhere",
            spec=dict(rows=512, kind="flat", temperature=1.0, rng="row"),
        ),
        Cell(
            why="1024 rows (128 requests, a 1P1D generation step), saturated soft cap",
            spec=dict(rows=1024, kind="sat", temperature=1.0, rng="row"),
        ),
    )

    note = """
    Metadata consumed: none. Two launches (kernel A grid (rows, V / 128) x 128
    threads; kernel B grid rows x V threads, V = ``virtual_threads(rows)``: 1024
    up to 128 rows, 512 above), both with programmatic dependent launch when
    ``pdl``; each waits on its predecessor before its first global read.

    Call site: the target DFlash worker's ``_sample_tokens_for_batch`` under
    the ``split_sampler`` fast path (``target_head.py``), on non-greedy batches with the fused
    sampling mode, on decode steps of at most 128 rows (16 requests at K=7), or
    of at most ``WIDE_MAX_ROWS`` rows under ``wide_split_sampler``.

    Numerics: bitwise the built-in soft cap + ``fused_sample_from_logits``,
    by construction (same per-thread fold order and operations, the shipped
    reduce / scan / RNG code) and by test (per-thread states, row values and
    tokens). When the target falls where two spans of the float scan overlap
    (an exclusive prefix and the previous prefix plus its mass round apart),
    two threads claim it; the shipped kernel lets each walk and keeps the
    crossing written last, and kernel B walks every claimant and keeps the
    later crossing position (then the higher thread) -- the same token
    whenever at most one claimant crosses, or the two cross at different
    positions in one warp; only a same-position tie, where the shipped
    kernel's own result depends on which store lands, is decided by the
    rule. The single-walker path runs when exactly one thread claims.

    Hard limits
      rows <= WIDE_MAX_ROWS (``states`` holds ``state_words(rows)`` words, as
      ``new_states(rows)`` gives), vocab % 4 == 0, vocab / 4 <= 64 * 1024 (the
      walk stages one virtual thread's vectors in shared memory).

    Not certified: torch.compile.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        min_ps: torch.Tensor,
        seed: Optional[torch.Tensor],
        offset: Optional[torch.Tensor],
        multiplier: float,
        soft_cap: float,
        states: torch.Tensor,
        pdl: bool = False,
    ) -> torch.Tensor:
        m, inv, cap = cap_constants(multiplier, soft_cap)
        torch.ops.musespark_open_head_sampler.fold_plain(
            logits, temperatures, states, m, inv, cap, FOLD_THREADS, pdl
        )
        return torch.ops.musespark_open_head_sampler.draw(
            logits, temperatures, top_ks, top_ps, min_ps, seed, offset, states, m, inv, cap, pdl
        )

    def reference(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        min_ps: torch.Tensor,
        seed: Optional[torch.Tensor],
        offset: Optional[torch.Tensor],
        multiplier: float,
        soft_cap: float,
        states: torch.Tensor,
        pdl: bool = False,
    ) -> torch.Tensor:
        """The built-in head: the processor's soft cap, then the shipped fused sampler.

        ``states`` and ``pdl`` are ignored.
        """
        capped = reference_capped(logits, multiplier, soft_cap)
        return torch.ops.trtllm.fused_sample_from_logits(
            capped, temperatures, top_ks, top_ps, min_ps, seed, offset
        )

    def is_valid(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        min_ps: torch.Tensor,
        seed: Optional[torch.Tensor],
        offset: Optional[torch.Tensor],
        multiplier: float,
        soft_cap: float,
        states: torch.Tensor,
        pdl: bool = False,
    ) -> None:
        if logits.dtype != torch.bfloat16 or logits.dim() != 2:
            raise ValueError("logits must be BF16 [rows, vocab]")
        rows, vocab = logits.shape
        if not 1 <= rows <= WIDE_MAX_ROWS:
            raise ValueError(f"1..{WIDE_MAX_ROWS} rows")
        if not _neutral_or_filtered_rows_ok(vocab):
            raise ValueError("vocab must be a multiple of 4 and at most 4 * 64 * 1024")
        if states.numel() < state_words(rows):
            raise ValueError("states is smaller than new_states(rows)")


split_sample = SplitSample()


class GatherSplitSample(OpWrapper):
    """``SplitSample`` whose logits are gathered from a multicast inbox by kernel A itself.

    ``inbox`` is this rank's copy of the inbox, ``[ranks][slot_rows][shard]`` BF16
    (any dtype view of those bytes), every word of rows < ``rows`` of every slot
    published by its rank's GEMM this call (or about to be); ``gathered`` is the
    ``[rows, ranks * shard]`` BF16 row this writes. Every consumed word is reset
    to the sentinel. ``status`` (int32 ``[1]``) gets bit 1 if a poll timed out (the
    kernel then traps). With ``sample=False`` only the gather runs and the
    return value is ``gathered``.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="the target's verify block: 8 rows, 8 shards of 25256, inbox published before the call",
            spec=dict(rows=8, ranks=8, shard=25256, kind="n40", publish="before"),
        ),
        Cell(
            why="the inbox published while kernel A already polls (concurrent producer)",
            spec=dict(rows=8, ranks=8, shard=25256, kind="n40", publish="during"),
        ),
        Cell(
            why="negative zeros and saturated caps survive the gather bit for bit",
            spec=dict(rows=8, ranks=8, shard=25256, kind="sat", publish="before", zeros=True),
        ),
        Cell(
            why="gather only (the materialize path for greedy / non-fused steps)",
            spec=dict(rows=8, ranks=8, shard=25256, kind="n40", publish="before", sample=False),
        ),
        Cell(
            why="fewer rows than slots: the other rows' words stay untouched",
            spec=dict(rows=3, ranks=8, shard=25256, kind="n10", publish="before"),
        ),
    )

    note = """
    Metadata consumed: none. Kernel A polls with ``ld.relaxed.sys`` (8 bytes:
    published when none of the four BF16 halves is the sentinel ``0xFFFF``),
    copies, resets with ``st.relaxed.sys``. One call per inbox use; every rank
    issues the same calls in the same order (the inbox protocol is in
    ``HeadGatherWorkspace``).

    Numerics: the gathered row is the producers' bits (no canonicalization);
    tokens as ``SplitSample``.

    Hard limits: as ``SplitSample``; ``rows <= slot_rows``; ``shard % 4 == 0``.
    """

    def __call__(
        self,
        inbox: torch.Tensor,
        gathered: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        min_ps: torch.Tensor,
        seed: Optional[torch.Tensor],
        offset: Optional[torch.Tensor],
        multiplier: float,
        soft_cap: float,
        states: torch.Tensor,
        status: torch.Tensor,
        shard: int,
        slot_rows: int,
        sample: bool = True,
        pdl: bool = False,
    ) -> torch.Tensor:
        m, inv, cap = cap_constants(multiplier, soft_cap)
        torch.ops.musespark_open_head_sampler.fold_gather(
            inbox,
            gathered,
            temperatures,
            states,
            status,
            m,
            inv,
            cap,
            shard // 4,
            slot_rows,
            POLL_TIMEOUT_S,
            sample,
            FOLD_THREADS,
            pdl,
        )
        if not sample:
            return gathered
        return torch.ops.musespark_open_head_sampler.draw(
            gathered, temperatures, top_ks, top_ps, min_ps, seed, offset, states, m, inv, cap, pdl
        )

    def reference(
        self,
        inbox: torch.Tensor,
        gathered: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        min_ps: torch.Tensor,
        seed: Optional[torch.Tensor],
        offset: Optional[torch.Tensor],
        multiplier: float,
        soft_cap: float,
        states: torch.Tensor,
        status: torch.Tensor,
        shard: int,
        slot_rows: int,
        sample: bool = True,
        pdl: bool = False,
    ) -> torch.Tensor:
        """The built-in: concatenate the shards in rank order, cap, sample (no inbox side effects).

        ``inbox`` must hold the published words (read, not reset); ``gathered``,
        ``states``, ``status`` and ``pdl`` are ignored.
        """
        rows = gathered.shape[0]
        ranks = gathered.shape[1] // shard
        slots = inbox.view(torch.bfloat16).reshape(-1)[: ranks * slot_rows * shard]
        logits = slots.view(ranks, slot_rows, shard)[:, :rows].permute(1, 0, 2).reshape(rows, -1)
        if not sample:
            return logits.contiguous()
        capped = reference_capped(logits, multiplier, soft_cap)
        return torch.ops.trtllm.fused_sample_from_logits(
            capped, temperatures, top_ks, top_ps, min_ps, seed, offset
        )

    def is_valid(
        self,
        inbox: torch.Tensor,
        gathered: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        min_ps: torch.Tensor,
        seed: Optional[torch.Tensor],
        offset: Optional[torch.Tensor],
        multiplier: float,
        soft_cap: float,
        states: torch.Tensor,
        status: torch.Tensor,
        shard: int,
        slot_rows: int,
        sample: bool = True,
        pdl: bool = False,
    ) -> None:
        if gathered.dtype != torch.bfloat16 or gathered.dim() != 2:
            raise ValueError("gathered must be BF16 [rows, vocab]")
        rows, vocab = gathered.shape
        if shard % 4 or vocab % shard:
            raise ValueError("the vocabulary is a whole number of shards of whole vectors")
        if not 1 <= rows <= min(slot_rows, MAX_ROWS):
            raise ValueError("rows must fit the inbox slots and the scratch")
        if not _neutral_or_filtered_rows_ok(vocab):
            raise ValueError("vocab must be a multiple of 4 and at most 4 * 64 * 1024")
        if inbox.numel() * inbox.element_size() < (vocab // shard) * slot_rows * shard * 2:
            raise ValueError("inbox is smaller than ranks x slot_rows x shard BF16")


gather_split_sample = GatherSplitSample()


class HeadGatherWorkspace:
    """The multicast inbox of the head gather for one tensor-parallel group, and its scratch.

    Layout: ``[ranks][INBOX_ROWS][shard]`` BF16; rank ``r``'s GEMM writes rows
    ``0..rows-1`` of slot ``r`` through the multicast address (``slot``), which
    the fabric delivers to every rank's copy. Every word holds the sentinel
    between calls.

    Protocol (one buffer, the consumer resets): (1) rank ``r``'s slot words are
    written only by rank ``r``'s GEMM, once per call; this rank's copy is
    otherwise written only by this rank's kernel A, which resets a word after
    reading it, and read by nothing else. (2) At the start of every call every
    word of every copy is the sentinel: true after construction (fill, device
    synchronize, then the group agreement, a collective), and restored by every
    call's kernel A, which reads and resets every word of rows < ``rows``. (3) A
    peer's next-call store into this copy follows, through the collectives of
    the next step's forward (the step-start embedding all-reduce and every
    layer's hops), this rank's issue of its next-step contributions, which its
    stream orders after this call's kernel A completed; kernel A's reset stores
    are performed at this GPU's L2 before it completes. (4) Every rank runs the
    same sequence of calls: whether a step uses the inbox depends only on
    rank-uniform facts (row count, no context rows, host flags).

    Construction is collective over the group (the library load and its
    agreement, then the fabric allocation, sentinel fill and a second agreement)
    and either succeeds on every rank or raises on every rank; ``library`` is the
    loaded library's path. The load is agreed before the allocation because the
    allocation is itself a collective: a rank whose load failed must not skip it
    while its peers wait in it.
    """

    def __init__(self, mapping: Mapping, shard: int):
        from tensorrt_llm._torch.distributed.ops import (
            _get_mnnvl_workspace_comm,
            _make_mnnvl_mcast_buffer,
            _mnnvl_device_index,
            _mnnvl_workspace_all_succeeded,
        )

        self.ranks = mapping.tp_size
        self.rank = mapping.tp_rank
        self.shard = int(shard)
        self.vocab = self.ranks * self.shard
        self.device = torch.device("cuda", _mnnvl_device_index(mapping))
        self._comm = _get_mnnvl_workspace_comm(mapping)
        self.slot_bytes = INBOX_ROWS * self.shard * 2
        self.library = ""
        error: Optional[BaseException] = None
        try:
            if self.shard % 4:
                raise RuntimeError(f"shard {self.shard} is not a whole number of 4-element vectors")
            if torch.cuda.get_device_capability(self.device) not in ((10, 3), (10, 7)):
                raise RuntimeError("the head gather runs on SM103 or SM107 only")
            self.library = load_library()
        except Exception as exc:  # noqa: BLE001 - every failure must reach the agreement
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError(
                "[musespark_open] lm_head_inbox: the head sampler library failed to load on at "
                "least one rank"
                f"{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error
        try:
            total = self.ranks * self.slot_bytes
            self._handle = _make_mnnvl_mcast_buffer(
                self._comm, total, mapping, mapping.is_multi_node()
            )
            self.inbox = self._handle.get_uc_buffer(self.rank, (total // 4,), torch.int32, 0)
            self._mc = self._handle.get_mc_buffer((total // 2,), torch.bfloat16, 0)
            with torch.cuda.device(self.device):
                self.inbox.fill_(-1)
                self.gathered = torch.empty(
                    (INBOX_ROWS, self.vocab), dtype=torch.bfloat16, device=self.device
                )
                self.states = new_states(INBOX_ROWS, self.device)
                self.status = torch.zeros(1, dtype=torch.int32, device=self.device)
                torch.cuda.synchronize(self.device)
        except Exception as exc:  # noqa: BLE001 - every failure must reach the agreement
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError(
                "[musespark_open] lm_head_inbox: head gather workspace failed on at least one rank"
                f"{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error

    def slot(self, rows: int) -> torch.Tensor:
        """This rank's ``[rows, shard]`` slot at the multicast address: the GEMM's ``out``."""
        start = self.rank * INBOX_ROWS * self.shard
        return self._mc[start : start + rows * self.shard].view(rows, self.shard)

    def errors(self) -> int:
        """The poll status word (synchronizes)."""
        return int(self.status.item())


_SHARED: dict = {}


def shared_gather_workspace(mapping: Mapping, shard: int) -> HeadGatherWorkspace:
    """The one head-gather workspace of this group and shard width, built on first use (collective)."""
    key = (mapping, int(shard))
    workspace = _SHARED.get(key)
    if workspace is None:
        workspace = HeadGatherWorkspace(mapping, shard)
        _SHARED[key] = workspace
    return workspace
