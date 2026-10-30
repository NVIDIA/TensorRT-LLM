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

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper
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


# ---------------------------------------------------------------------------------------
# The vocabulary-parallel head (fast path ``vocab_parallel_head``): no logits all-gather.

#: Fewest rows the vocabulary-parallel head takes (two K=7 verify blocks); 8-row steps keep
#: the ``lm_head_inbox`` gather.
VP_MIN_ROWS = 16
#: Claimants of a row whose crossings the ranks exchange (``kVpClaimants``).
VP_CLAIMANTS = 4
#: Greedy words per (rank, row): one per ``vp_fold`` CTA of the 1024-thread block.
VP_GREEDY_SLOTS = VIRTUAL_THREADS // FOLD_THREADS
#: ``vp_draw``'s per-row record for ``vp_decide`` (int32 words).
VP_INFO_WORDS = 2 + VP_CLAIMANTS
#: Most ranks (``kVpMaxRanks``).
VP_MAX_RANKS = 8


class VocabParallelLayout:
    """The exchange buffer of the vocabulary-parallel head for ``ranks`` shards of ``shard``
    elements and calls of at most ``max_rows`` rows.

    One copy per rank, every 32-bit word the sentinel ``0xFFFFFFFF`` between calls:
    ``[ranks][state_stride]`` 16-byte states (``vp_fold``'s per-virtual-thread partials), then
    ``[ranks][greedy_stride]`` 8-byte greedy words at ``greedy_offset``, then
    ``[ranks][cand_stride]`` 4-byte candidates at ``cand_offset``; ``nbytes`` in all.
    """

    def __init__(self, ranks: int, shard: int, max_rows: int):
        if not 1 <= ranks <= VP_MAX_RANKS:
            raise ValueError(f"1..{VP_MAX_RANKS} ranks, got {ranks}")
        if shard % 4:
            raise ValueError(f"shard {shard} is not a whole number of 4-element vectors")
        if not VP_MIN_ROWS <= max_rows <= WIDE_MAX_ROWS:
            raise ValueError(f"max_rows must be in {VP_MIN_ROWS}..{WIDE_MAX_ROWS}, got {max_rows}")
        self.ranks = int(ranks)
        self.shard = int(shard)
        self.shard_vecs = self.shard // 4
        self.vocab = self.ranks * self.shard
        self.max_rows = int(max_rows)
        widest = max(
            -(-self.shard_vecs // virtual_threads(r)) for r in {min(max_rows, MAX_ROWS), max_rows}
        )
        if widest > 64:
            raise ValueError(f"shard {shard} is too wide for the walk stage")
        self.state_stride = max(r * virtual_threads(r) for r in {min(max_rows, MAX_ROWS), max_rows})
        self.greedy_stride = self.max_rows * VP_GREEDY_SLOTS
        self.cand_stride = self.max_rows * VP_CLAIMANTS
        self.greedy_offset = self.ranks * self.state_stride * 16
        self.cand_offset = self.greedy_offset + self.ranks * self.greedy_stride * 8
        self.nbytes = -(-(self.cand_offset + self.ranks * self.cand_stride * 4) // 16) * 16
        self.staging_bytes = self.max_rows * self.shard * 2

    def strides(self) -> dict:
        return dict(
            state_stride=self.state_stride,
            greedy_stride=self.greedy_stride,
            cand_stride=self.cand_stride,
        )


class _VocabParallelRank:
    """One rank's view: its exchange copy, the publish targets, every rank's staging rows."""

    def __init__(
        self,
        layout: VocabParallelLayout,
        rank: int,
        copy: int,
        publish: list,
        staging: int,
        staging_of: list,
        info: torch.Tensor,
        status: torch.Tensor,
    ):
        self.layout = layout
        self.rank = int(rank)
        self.copy = int(copy)
        self.publish = [int(x) for x in publish]
        self.staging = int(staging)
        self.staging_of = [int(x) for x in staging_of]
        self.info = info
        self.status = status

    def fold(self, local, temperatures, top_ks, top_ps, min_ps, multiplier, soft_cap, greedy, pdl):
        m, inv, cap = cap_constants(multiplier, soft_cap)
        lay = self.layout
        torch.ops.musespark_open_head_sampler.vp_fold(
            local,
            temperatures,
            top_ks,
            top_ps,
            min_ps,
            self.staging,
            self.publish,
            self.rank,
            lay.ranks,
            lay.state_stride,
            lay.greedy_offset,
            lay.greedy_stride,
            lay.cand_stride,
            greedy,
            m,
            inv,
            cap,
            pdl,
        )

    def draw(
        self,
        pending,
        local,
        temperatures,
        top_ks,
        top_ps,
        min_ps,
        seed,
        offset,
        multiplier,
        soft_cap,
        pdl,
    ):
        m, inv, cap = cap_constants(multiplier, soft_cap)
        lay = self.layout
        return torch.ops.musespark_open_head_sampler.vp_draw(
            pending,
            local,
            temperatures,
            top_ks,
            top_ps,
            min_ps,
            seed,
            offset,
            self.info,
            self.status,
            self.copy,
            self.staging_of,
            self.publish,
            self.rank,
            lay.ranks,
            lay.state_stride,
            lay.greedy_stride,
            lay.cand_offset,
            lay.cand_stride,
            POLL_TIMEOUT_S,
            m,
            inv,
            cap,
            pdl,
        )

    def decide(self, tokens, pdl):
        lay = self.layout
        torch.ops.musespark_open_head_sampler.vp_decide(
            tokens,
            self.info,
            self.status,
            self.copy,
            self.rank,
            lay.ranks,
            lay.shard_vecs,
            lay.state_stride,
            lay.greedy_stride,
            lay.cand_offset,
            lay.cand_stride,
            POLL_TIMEOUT_S,
            pdl,
        )

    def argmax(self, rows, pdl):
        lay = self.layout
        return torch.ops.musespark_open_head_sampler.vp_argmax(
            self.status,
            rows,
            self.copy,
            self.rank,
            lay.ranks,
            lay.shard_vecs,
            lay.state_stride,
            lay.greedy_offset,
            lay.greedy_stride,
            lay.cand_stride,
            POLL_TIMEOUT_S,
            pdl,
        )


def vocab_parallel_sample(
    view: _VocabParallelRank,
    pending: torch.Tensor,
    local: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    seed: Optional[torch.Tensor],
    offset: Optional[torch.Tensor],
    multiplier: float,
    soft_cap: float,
    pdl: bool = False,
) -> torch.Tensor:
    """One rank's sampled tokens (int32 ``[rows]``) of the step whose shard ``local`` is.

    ``vp_fold``, ``vp_draw`` and ``vp_decide`` in order; every rank of the group makes the same
    call with its own shard. ``pending`` is the ``[rows, vocab]`` BF16 row: written for rows with
    a top-k / top-p / min-p filter only (gathered from the ranks' staging rows), left as it is
    otherwise.
    """
    rows = local.shape[0]
    view.fold(local, temperatures, top_ks, top_ps, min_ps, multiplier, soft_cap, False, pdl)
    tokens = view.draw(
        pending,
        local,
        temperatures,
        top_ks,
        top_ps,
        min_ps,
        seed,
        offset,
        multiplier,
        soft_cap,
        pdl,
    )
    view.decide(tokens[:rows], pdl)
    return tokens


def vocab_parallel_argmax(
    view: _VocabParallelRank,
    local: torch.Tensor,
    multiplier: float,
    soft_cap: float,
    pdl: bool = False,
) -> torch.Tensor:
    """One rank's greedy tokens (int64 ``[rows]``): ``torch.argmax`` of the soft-capped FP32 row."""
    view.fold(
        local, view.status, view.status, view.status, view.status, multiplier, soft_cap, True, pdl
    )
    return view.argmax(local.shape[0], pdl)


def vocab_parallel_reference(
    locals_: list,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    seed: Optional[torch.Tensor],
    offset: Optional[torch.Tensor],
    multiplier: float,
    soft_cap: float,
    greedy: bool,
) -> torch.Tensor:
    """The built-in head on the rank-order concatenation of the shards: ``torch.argmax`` of the
    soft-capped FP32 row (greedy) or ``fused_sample_from_logits`` on it."""
    capped = reference_capped(torch.cat(list(locals_), dim=1), multiplier, soft_cap)
    if greedy:
        return torch.argmax(capped, dim=-1)
    return torch.ops.trtllm.fused_sample_from_logits(
        capped, temperatures, top_ks, top_ps, min_ps, seed, offset
    )


class VocabParallelEmulation:
    """``ranks`` virtual ranks on one GPU: every value is stored into each rank's copy (instead
    of one multicast store), every rank's staging rows are plain tensors. The kernels are the
    served ones; a call runs every rank's ``vp_fold``, then every ``vp_draw``, then every
    ``vp_decide`` (so no poll waits)."""

    def __init__(self, ranks: int, shard: int, max_rows: int, device):
        load_library()
        self.layout = VocabParallelLayout(ranks, shard, max_rows)
        lay = self.layout
        self.device = torch.device(device)
        self.copies = [
            torch.full((lay.nbytes // 4,), -1, dtype=torch.int32, device=self.device)
            for _ in range(ranks)
        ]
        self.staging = [
            torch.empty((max_rows, shard), dtype=torch.bfloat16, device=self.device)
            for _ in range(ranks)
        ]
        publish = [c.data_ptr() for c in self.copies]
        staging_of = [s.data_ptr() for s in self.staging]
        self.status = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.views = [
            _VocabParallelRank(
                lay,
                r,
                self.copies[r].data_ptr(),
                publish,
                staging_of[r],
                staging_of,
                torch.empty(max_rows * VP_INFO_WORDS, dtype=torch.int32, device=self.device),
                self.status,
            )
            for r in range(ranks)
        ]

    def sample(
        self,
        locals_,
        temperatures,
        top_ks,
        top_ps,
        min_ps,
        seed,
        offset,
        multiplier,
        soft_cap,
        pdl=False,
    ):
        """Every rank's tokens (list of int32 ``[rows]``) and pending rows."""
        rows = locals_[0].shape[0]
        pendings = [
            torch.zeros((rows, self.layout.vocab), dtype=torch.bfloat16, device=self.device)
            for _ in self.views
        ]
        for view, local in zip(self.views, locals_):
            view.fold(local, temperatures, top_ks, top_ps, min_ps, multiplier, soft_cap, False, pdl)
        tokens = [
            view.draw(
                pending,
                local,
                temperatures,
                top_ks,
                top_ps,
                min_ps,
                seed,
                offset,
                multiplier,
                soft_cap,
                pdl,
            )
            for view, pending, local in zip(self.views, pendings, locals_)
        ]
        for view, t in zip(self.views, tokens):
            view.decide(t, pdl)
        return tokens, pendings

    def argmax(self, locals_, multiplier, soft_cap, pdl=False):
        for view, local in zip(self.views, locals_):
            view.fold(
                local,
                self.status,
                self.status,
                self.status,
                self.status,
                multiplier,
                soft_cap,
                True,
                pdl,
            )
        return [view.argmax(locals_[0].shape[0], pdl) for view in self.views]

    def all_sentinel(self) -> bool:
        return all(bool((c == -1).all()) for c in self.copies)


class VocabParallelWorkspace:
    """The vocabulary-parallel head's fabric buffers for one tensor-parallel group.

    ``exchange``: the multicast exchange buffer (``VocabParallelLayout``), each rank's copy filled
    with the sentinel; values are stored through the multicast address, read and reset in the
    own copy. ``staging``: ``[max_rows, shard]`` BF16 per rank (unicast; peers read it), written
    by ``vp_fold`` for rows with a filter only.

    Protocol: (1) every exchange slot has one writer per call (``vp_fold`` per (rank, row,
    virtual thread) or (rank, row, CTA); ``vp_draw`` per (rank, row, claimant)); every rank
    reads each slot of its own copy once per call and resets it. (2) At the start of every call
    every word of every copy is the sentinel: after construction (fill, synchronize, agreement)
    and restored by the call's readers. (3) Reuse across calls: a peer stores into this copy (and
    into its staging rows) again only in a later call, which it reaches after a collective of
    the group issued between the two calls (the decode step's all-reduces and hops, the
    drafter's hops). This rank publishes into that collective only after this call's last
    kernel has completed: every kernel between them is a plain launch or waits for its
    predecessor (``griddepcontrol.wait``) before it completes, and the collective publishes
    after its own wait -- so an all-PDL chain (the Lamport hops) orders it as an NCCL call
    does. Two calls with no collective between them are not ordered: a peer's next fold can
    overwrite a word this rank has not read. (4) Every rank makes the same calls: the path
    depends only on rank-uniform facts.

    Construction is collective over the group and succeeds or raises on every rank.
    """

    def __init__(self, mapping: Mapping, shard: int, max_rows: int):
        from tensorrt_llm._torch.distributed.ops import (
            _get_mnnvl_workspace_comm,
            _make_mnnvl_mcast_buffer,
            _mnnvl_device_index,
            _mnnvl_workspace_all_succeeded,
        )

        self.ranks = mapping.tp_size
        self.rank = mapping.tp_rank
        self.device = torch.device("cuda", _mnnvl_device_index(mapping))
        self._comm = _get_mnnvl_workspace_comm(mapping)
        self.library = ""
        error: Optional[BaseException] = None
        try:
            self.layout = VocabParallelLayout(self.ranks, shard, max_rows)
            if torch.cuda.get_device_capability(self.device) not in ((10, 3), (10, 7)):
                raise RuntimeError("the vocabulary-parallel head runs on SM103 or SM107 only")
            self.library = load_library()
        except Exception as exc:  # noqa: BLE001 - every failure must reach the agreement
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError(
                "[musespark_open] vocab_parallel_head: the head sampler library failed to load on at "
                "least one rank"
                f"{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error
        lay = self.layout
        try:
            multi_node = mapping.is_multi_node()
            self._exchange = _make_mnnvl_mcast_buffer(self._comm, lay.nbytes, mapping, multi_node)
            self._staging = _make_mnnvl_mcast_buffer(
                self._comm, lay.staging_bytes, mapping, multi_node
            )
            self.copy = self._exchange.get_uc_buffer(self.rank, (lay.nbytes // 4,), torch.int32, 0)
            self._mc = self._exchange.get_mc_buffer((lay.nbytes // 4,), torch.int32, 0)
            self.staging_of = [
                self._staging.get_uc_buffer(peer, (lay.staging_bytes // 2,), torch.bfloat16, 0)
                for peer in range(self.ranks)
            ]
            with torch.cuda.device(self.device):
                self.copy.fill_(-1)
                self.info = torch.empty(
                    lay.max_rows * VP_INFO_WORDS, dtype=torch.int32, device=self.device
                )
                self.status = torch.zeros(1, dtype=torch.int32, device=self.device)
                torch.cuda.synchronize(self.device)
        except Exception as exc:  # noqa: BLE001 - every failure must reach the agreement
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError(
                "[musespark_open] vocab_parallel_head: workspace failed on at least one rank"
                f"{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error
        self.view = _VocabParallelRank(
            lay,
            self.rank,
            self.copy.data_ptr(),
            [self._mc.data_ptr()],
            self.staging_of[self.rank].data_ptr(),
            [t.data_ptr() for t in self.staging_of],
            self.info,
            self.status,
        )

    @property
    def max_rows(self) -> int:
        return self.layout.max_rows

    @property
    def vocab(self) -> int:
        return self.layout.vocab

    def nbytes(self) -> int:
        """Device memory this rank holds for the head (exchange copy + staging rows)."""
        return self.layout.nbytes + self.layout.staging_bytes

    def errors(self) -> int:
        """The poll status word (synchronizes)."""
        return int(self.status.item())
