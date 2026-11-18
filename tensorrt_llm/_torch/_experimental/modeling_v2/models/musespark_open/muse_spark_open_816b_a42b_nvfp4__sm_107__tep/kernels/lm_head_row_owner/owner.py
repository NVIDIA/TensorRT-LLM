# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``lm_head_row_owner``: the TEP4 verify head with each request's logits rows owned by one rank.

On a decode step of ``B`` requests (``8 B`` verify rows at K = 7) the built-in head runs the
LM-head GEMM on every rank (``[8 B, shard]``), all-gathers the four vocabulary shards
(NCCL ring + ``cat``), and every rank then soft-caps, computes the target probabilities of,
rebuilds the draft rows of and chain-verifies all ``B`` requests: the same work four times.

With this lever, on steps of ``MIN_REQUESTS..MAX_REQUESTS`` requests, the requests are split
into four contiguous owner slices of ``q = ceil(B / 4)`` requests (rank ``o`` owns
``[o q, min(B, (o + 1) q))``) and:

1. **Producer.** Every rank's LM-head GEMM (``torch.mm`` into a strided ``out``; the kernel the
   built-in ``F.linear`` runs, ``CERTIFIED_PRODUCERS``) stores its ``[8 B, shard]`` output
   straight into the owners' pages: a stitched virtual range per ``q`` maps request ``j``'s 8
   rows (``REQUEST_BYTES``, one VMM allocation) onto the owner's physical allocation, so one
   GEMM with row stride ``LDC`` writes every owner's rows. No logits gather.
2. **Owner.** Each owner polls its rows (Lamport: every page word holds the sentinel
   ``0xFFFF`` in each BF16 half between calls, a NaN no GEMM epilogue produces), soft-caps
   them in FP32 with the built-in arithmetic and resets the words to the sentinel
   (``row_owner.cu`` ``ownerCapKernel``), then runs the built-in verify body on its slice:
   CUDA-graph padding rows zeroed, ``_compute_probs_from_logits`` (fused) on its ``8 q`` rows
   (``8 q >= 136`` rows: the shipped sampler's 512-thread block, the block of the built-in's
   ``8 B >= 520`` rows), the compact draft rows of its requests, and the chain speculative
   sampling kernel with the Philox subsequence of the request's index in the batch
   (``chain_row_base.cu``: the stock kernel numbers subsequences by block index).
3. **Exchange.** Each owner's ``[q, K + 2]`` int32 tokens and counts are all-gathered over the
   TP group (NCCL); every rank then finishes the built-in body (forced acceptance, the
   accepted-count override) on the whole batch.

Every per-row kernel sees the bytes the built-in path gives it on the owner rank, so the
tokens, counts and every intermediate row are the owner rank's built-in ones bit for bit. The
draft-probability rows are rank-local and not bitwise equal across TP ranks, and the built-in
emits rank 0's decision, so a request owned by another rank can (rarely) be decided
differently from the built-in: a numeric lever. Any other consumer of the
step's logits (a non-rejection verify, a greedy batch, a refused shape) first gathers the
whole row from the pages (``pollGatherKernel``, then a TP barrier, then each owner resets its
rows) and gets the built-in's gathered row; a row nothing consumed is drained (owner poll +
reset) by the head's forward hook.

Protocol safety (why no step can hang or read stale words)
---------------------------------------------------------
* *Agreement.* Whether a step takes the pages, and how it is consumed, depends only on
  rank-uniform host facts: the row count, ``attn_metadata.num_seqs`` / ``num_contexts``, the
  installed levers, the census (agreed at install), and the spec metadata's host fields
  (rejection sampling, greedy, penalties, ragged verify, sampling mode), which every TP rank
  holds alike. So every rank issues the same GEMM, the same consume kernel and the same
  collectives.
* *Progress.* An owner's poll waits only for the peers' LM-head GEMMs of the same step. A
  peer's GEMM is issued on its main stream after everything of that step before the head
  (the model forward and its collectives, which the owner has already completed: its poll is
  issued after its own GEMM on its own main stream), and depends on no work the owner issues
  later; the producer never needs the owner's SMs (remote stores). So every poll terminates
  once the peers reach their GEMM. A peer that never does (a host stall, a rank that took
  another path) is a protocol failure: the poll gives up after ``POLL_TIMEOUT_MS``, records
  the failure in ``status`` and traps (the CUDA context ends loudly, where the built-in NCCL
  gather would keep waiting).
* *Words are final when seen.* The certified producer (one cuBLAS kernel per row count, no
  split-K reduce: ``CERTIFIED_PRODUCERS`` and the install census) stores each element of the
  output once, with its final value; the sentinel is unreachable (the RN conversion of any
  NaN is ``0x7FFF``), so a word without a sentinel half is the published value.
* *Reset before reuse (graph-replay epoch safety).* There is no epoch counter: the invariant
  is that between two head calls every page word is the sentinel. A consume resets exactly
  the words the step's GEMMs wrote (the owner's live rows), each after reading it; the
  gather resets them after a TP barrier that follows every rank's reads. Every consume then
  ends with a TP collective the owner issues after its reset kernel on the same stream (the
  verify: the result exchange; the gather and the drain: a TP barrier after the reset). The
  next writer of a word is a peer's GEMM of a later step, issued on the peer's stream after
  that collective has completed on the peer, which it can only do once the owner has
  joined it, that is after the owner's reset; so the reset is performed in the owner's
  memory before the next write arrives, with no assumption on what the caller runs between
  two head calls.
  Rows of a smaller batch are a subset of the rows a larger batch writes, and a step never
  writes rows it does not consume, so the invariant holds for any sequence of batch sizes and
  of captured graphs replayed in any order; nothing in the protocol depends on which graph
  ran before.
* *Failure inside a step.* A row the step did not consume is drained by the head's forward
  hook (``always_call``), on every rank alike; a step that raises between the GEMM and the
  consume on only some ranks ends the server anyway (the next collective cannot complete).
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import math
import os
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from tensorrt_llm._torch.speculative.compact_draft_probs import (
    DeferredDraftProbs,
    mark_private_verify_logits,
    private_verify_scratch,
)
from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase
from tensorrt_llm.logger import logger

from ... import dispatch as _tp_agreement

#: Fewest requests of a step the lever takes: the owner slice must hold at least 17 requests,
#: ``8 * 17 = 136`` rows, so that the owner's fused probabilities run the shipped sampler's
#: 512-thread block, the block of the built-in's call of more than 128 rows.
MIN_REQUESTS = 65
#: Most requests of a step the lever takes (the census's row counts; the generation
#: server's ``max_batch_size``).
MAX_REQUESTS = 96
#: Verify rows of one request (the last accepted token and K = 7 drafts).
ROWS_PER_REQUEST = 8
#: TP width the owners split over.
WORLD = 4
#: Elements (BF16) of one page row: 512 KiB, a whole number of rows per VMM allocation.
LDC = 262144
ROW_BYTES = LDC * 2
#: One request's rows: one VMM allocation (``cuMemMap`` maps whole allocations only).
REQUEST_BYTES = ROWS_PER_REQUEST * ROW_BYTES
#: Allocations (requests) per owner.
SLOTS = math.ceil(MAX_REQUESTS / WORLD)
#: Smallest owner slice the lever stitches.
MIN_SLICE = math.ceil(MIN_REQUESTS / WORLD)
#: A poll waiting longer than this traps (a peer that never ran its GEMM); the split
#: sampler's inbox uses the same bound. Tests lower it.
POLL_TIMEOUT_MS = 600_000
#: (torch, cuBLASLt, SM count) for which the LM-head producer census (``torch.mm`` of
#: ``[8 B, 8192] x [50512, 8192]^T`` into a row stride of ``LDC``, B 65..96, all four GPUs)
#: found one kernel per row count, the one the contiguous call runs, no split-K reduce, and
#: bitwise equal outputs.
CERTIFIED_PRODUCERS = frozenset({("2.14.0a0+b2c75dd062.nv26.09", 130800, 212)})

LIBRARY_NAME = "musespark_open_sm107_row_owner"
SOURCE_DIR = Path(__file__).resolve().parent
PROTOCOL_SOURCE = SOURCE_DIR / "row_owner.cu"
CHAIN_SOURCE = SOURCE_DIR / "chain_row_base.cu"
#: No ``--use_fast_math``: the soft cap's ``tanhf`` and products keep the torch chain's bits.
NVCC_FLAGS = ("-O3", "-lineinfo")

_LOCK = threading.Lock()
_PROTOCOL_LOADED: Optional[str] = None
_CHAIN_MODULE = None


def _digest(path: Path, *extra: str) -> str:
    digest = hashlib.sha256(path.read_bytes())
    for value in extra:
        digest.update(b"\0" + value.encode())
    return digest.hexdigest()[:12]


def arch_target() -> str:
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (10, 7):
        raise RuntimeError(f"lm_head_row_owner is built for sm_107a, not sm_{major}{minor}")
    return "107a"


def load_protocol_library() -> str:
    """Build (if needed) and load the owner protocol kernels once per process; their path."""
    global _PROTOCOL_LOADED
    with _LOCK:
        if _PROTOCOL_LOADED is None:
            from torch.utils.cpp_extension import load

            target = arch_target()
            name = f"{LIBRARY_NAME}_sm{target}_{_digest(PROTOCOL_SOURCE, *NVCC_FLAGS)}"
            path = load(
                name=name,
                sources=[str(PROTOCOL_SOURCE)],
                extra_cuda_cflags=list(NVCC_FLAGS)
                + [f"-gencode=arch=compute_{target},code=sm_{target}"],
                is_python_module=False,
                verbose=False,
            )
            _PROTOCOL_LOADED = str(path) if path else name
        return _PROTOCOL_LOADED


def chain_module():
    """The row-base chain kernel, built by flashinfer's JIT with flashinfer's own flags.

    The stock ``chain_speculative_sampling`` is flashinfer's JIT module ``sampling``; this
    copy of its kernel is compiled by the same JIT (``gen_jit_spec``: the same nvcc, CCCL,
    arch mapping and ``-use_fast_math`` flags), so its arithmetic is the stock kernel's.
    """
    global _CHAIN_MODULE
    with _LOCK:
        if _CHAIN_MODULE is None:
            from flashinfer.jit import env as jit_env
            from flashinfer.jit.core import gen_jit_spec

            spec = gen_jit_spec(
                f"musespark_open_chain_row_base_{_digest(CHAIN_SOURCE)}",
                [CHAIN_SOURCE],
                extra_include_paths=[jit_env.FLASHINFER_CSRC_DIR],
            )
            _CHAIN_MODULE = spec.build_and_load()
        return _CHAIN_MODULE


def _ops():
    load_protocol_library()
    return torch.ops.musespark_open_sm107_row_owner


def owner_slice(batch: int, rank: int, world: int = WORLD) -> Tuple[int, int, int]:
    """``(q, first, count)``: the slice size of a batch, and rank's first request and live count."""
    q = math.ceil(batch / world)
    first = rank * q
    return q, first, max(0, min(batch, first + q) - first)


def producer_refusal(device: torch.device) -> Optional[str]:
    """Why the certified LM-head producer may not publish on ``device`` here, or None (rank-local)."""
    from ...ops.comm import projection_publish as _publish

    torch.cuda.init()
    key = (
        str(torch.__version__),
        _publish.cublaslt_version(),
        torch.cuda.get_device_properties(device).multi_processor_count,
    )
    if key not in CERTIFIED_PRODUCERS:
        return (
            f"the LM-head producer census does not cover (torch, cuBLASLt, SMs) = {key} "
            f"(certified: {sorted(CERTIFIED_PRODUCERS)})"
        )
    overrides = [name for name in _publish.CUBLAS_WORKSPACE_ENV if os.environ.get(name)]
    if overrides:
        return f"{', '.join(overrides)} resize the cuBLAS workspaces the producer census ran with"
    return None


def _check(result):
    """A ``cuda.bindings.driver`` call's value, raising on an error code."""
    if isinstance(result, tuple):
        err, *rest = result
    else:
        err, rest = result, []
    if int(err) != 0:
        raise RuntimeError(f"CUDA driver error {err}")
    if not rest:
        return None
    return rest[0] if len(rest) == 1 else rest


class OwnerPages:
    """This rank's ``SLOTS`` request allocations, the peers' imported, and one stitched range per slice size.

    Collective over the TP group (the handle exchange and its agreement). The stitched range
    of slice size ``q`` is ``WORLD * q * REQUEST_BYTES`` of virtual address space: owner
    ``o``'s ``j``-th allocation at ``(o q + j) * REQUEST_BYTES``, so request ``r`` of a batch
    with that slice size is at ``r * REQUEST_BYTES`` and row ``i`` at ``i * ROW_BYTES``. Every
    page word starts as the sentinel.
    """

    def __init__(self, mapping, device: torch.device) -> None:
        from cuda.bindings import driver as cu

        self._cu = cu
        self.rank = int(mapping.tp_rank)
        self.world = int(mapping.tp_size)
        self.device = device
        location = cu.CUmemLocation()
        location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        location.id = int(device.index)
        kinds = (
            cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
            cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
        )
        errors = []
        self.handles = None
        for kind in kinds:
            fabric = kind == cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
            prop = cu.CUmemAllocationProp()
            prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
            prop.location = location
            prop.requestedHandleTypes = kind
            mine, payload, error = [], None, None
            try:
                granularity = _check(
                    cu.cuMemGetAllocationGranularity(
                        prop,
                        cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
                    )
                )
                if REQUEST_BYTES % int(granularity):
                    raise RuntimeError(
                        f"{REQUEST_BYTES} B is not a multiple of the granularity {granularity}"
                    )
                self.granularity = int(granularity)
                mine = [_check(cu.cuMemCreate(REQUEST_BYTES, prop, 0)) for _ in range(SLOTS)]
                exported = [_check(cu.cuMemExportToShareableHandle(h, kind, 0)) for h in mine]
                payload = [bytes(e.data) if fabric else int(e) for e in exported]
            except Exception as exc:  # noqa: BLE001 - every rank's outcome must reach the agreement
                error = f"{type(exc).__name__}: {exc}"
            everyone = _tp_agreement.Distributed.get(mapping).tp_allgather(
                (error, payload, os.getpid())
            )
            failed = [why for why, _, _ in everyone if why is not None]
            if failed:
                for handle in mine:
                    cu.cuMemRelease(handle)
                errors.append(f"{'fabric' if fabric else 'posix fd'}: {sorted(set(failed))}")
                continue
            self.handles = self._import(everyone, mine, kind, fabric)
            self.kind = "fabric" if fabric else "posix fd"
            break
        if self.handles is None:
            raise RuntimeError(f"no shareable VMM handle type works on every rank: {errors}")
        self.access = cu.CUmemAccessDesc()
        self.access.location = location
        self.access.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        self.ranges: Dict[int, int] = {}
        for q in range(MIN_SLICE, SLOTS + 1):
            self.ranges[q] = self._stitch(q)
        self._views: Dict[Tuple[str, int, int], torch.Tensor] = {}
        # Every word of this rank's pages to the sentinel (0xFF bytes).
        own = self.ranges[SLOTS] + self.rank * SLOTS * REQUEST_BYTES
        _check(cu.cuMemsetD8(own, 0xFF, SLOTS * REQUEST_BYTES))
        torch.cuda.synchronize(device)

    def _import(self, everyone, mine, kind, fabric):
        cu = self._cu
        handles = []
        syscall = ctypes.CDLL(None, use_errno=True).syscall
        for rank, (_, payload, pid) in enumerate(everyone):
            if rank == self.rank:
                handles.append(mine)
                continue
            imported = []
            for data in payload:
                if fabric:
                    imported.append(_check(cu.cuMemImportFromShareableHandle(data, kind)))
                    continue
                # pidfd_open / pidfd_getfd: the peer's descriptor in this process.
                pidfd = syscall(434, pid, 0)
                fd = syscall(438, pidfd, data, 0) if pidfd >= 0 else -1
                if pidfd < 0 or fd < 0:
                    raise RuntimeError(f"pidfd import failed (errno {ctypes.get_errno()})")
                imported.append(_check(cu.cuMemImportFromShareableHandle(fd, kind)))
            handles.append(imported)
        return handles

    def _stitch(self, q: int) -> int:
        cu = self._cu
        slab = q * REQUEST_BYTES
        va = int(_check(cu.cuMemAddressReserve(self.world * slab, self.granularity, 0, 0)))
        for owner in range(self.world):
            for j in range(q):
                _check(
                    cu.cuMemMap(
                        va + owner * slab + j * REQUEST_BYTES,
                        REQUEST_BYTES,
                        0,
                        self.handles[owner][j],
                        0,
                    )
                )
            _check(cu.cuMemSetAccess(va + owner * slab, slab, [self.access], 1))
        return va

    def _view(self, kind: str, q: int, rows: int, address: int, columns: int) -> torch.Tensor:
        from tensorrt_llm._dlpack_utils import pack_strided_memory

        key = (kind, q, rows)
        view = self._views.get(key)
        if view is None:
            # DLPack carries no BF16 code here: int16 views, reinterpreted by the callers.
            view = pack_strided_memory(
                address, columns * 2, ROW_BYTES, rows, torch.int16, int(self.device.index)
            )
            self._views[key] = view
        return view

    def producer(self, q: int, rows: int, shard: int) -> torch.Tensor:
        """``[rows, shard]`` BF16, row stride ``LDC``: this rank's columns of the stitched range."""
        address = self.ranges[q] + self.rank * shard * 2
        return self._view("producer", q, rows, address, shard).view(torch.bfloat16)

    def owned(self, q: int, vocab: int) -> torch.Tensor:
        """``[8 q, vocab]`` int16, row stride ``LDC``: this rank's own rows in the range of ``q``."""
        address = self.ranges[q] + self.rank * q * REQUEST_BYTES
        return self._view("owned", q, q * ROWS_PER_REQUEST, address, vocab)

    def stitched(self, q: int, rows: int, vocab: int) -> torch.Tensor:
        """``[rows, vocab]`` int16, row stride ``LDC``: every owner's rows of the range of ``q``."""
        return self._view("stitched", q, rows, self.ranges[q], vocab)


class _OwnerSlots:
    """The spec-metadata fields ``SpecWorkerBase._zero_padding_rows`` reads, for an owner slice."""

    is_ragged_verify = False

    def __init__(self, batch_slot_ids: torch.Tensor, dummy_slot_row: int) -> None:
        self.batch_slot_ids = batch_slot_ids
        self.dummy_slot_row = dummy_slot_row


class RowOwnerHead:
    """The owner pages and the step logic of ``lm_head_row_owner`` (attached to the split sampler's head)."""

    def __init__(self, mapping, lm_head, multiplier: float, soft_cap: float) -> None:
        """Collective over the TP group: refusal agreement, library loads, pages, census.

        Raises:
            ValueError: on every rank alike, when some rank cannot run the lever.
        """
        from tensorrt_llm._torch.modules.logit_soft_cap import soft_cap_constants

        self.mapping = mapping
        self.rank = int(mapping.tp_rank)
        self.world = int(mapping.tp_size)
        self.device = lm_head.weight.device
        self.shard = int(lm_head.weight.shape[0])
        self.hidden = int(lm_head.weight.shape[1])
        self.vocab = self.shard * self.world
        self.cap = soft_cap_constants(multiplier, soft_cap)
        reason = self.refusal(mapping, lm_head)
        if reason is None:
            reason = producer_refusal(self.device)
        if reason is None:
            try:
                load_protocol_library()
                chain_module()
            except Exception as exc:  # noqa: BLE001 - every failure must reach the agreement
                reason = f"the owner kernels failed to build ({type(exc).__name__}: {exc})"
        reason = _tp_agreement.agreed_refusal(mapping, reason, facts=(self.shard, self.hidden))
        if reason is not None:
            raise ValueError(f"[musespark_open] lm_head_row_owner cannot run here: {reason}")
        error = None
        try:
            self.pages = OwnerPages(mapping, self.device)
        except RuntimeError as exc:
            error = str(exc)
        reason = _tp_agreement.agreed_refusal(mapping, error)
        if reason is not None:
            raise ValueError(f"[musespark_open] lm_head_row_owner cannot run here: {reason}")
        self.status = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.barrier_word = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.batches = self._census(lm_head)
        self._declined_logged: set = set()
        self.verified_steps = 0
        logger.info(
            f"[musespark_open] lm_head_row_owner: decode steps of {MIN_REQUESTS}..{MAX_REQUESTS} "
            f"requests write each request's LM-head rows into its owner's pages "
            f"({SLOTS} x {REQUEST_BYTES >> 20} MiB per rank, {self.pages.kind} handles); "
            f"owners verify their slice; census certified {len(self.batches)} batch sizes"
        )

    @staticmethod
    def refusal(mapping, lm_head) -> Optional[str]:
        """Why this rank cannot run the lever (rank-local), or None."""
        from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode

        if int(mapping.tp_size) != WORLD:
            return f"the owner split is built for TP{WORLD}, not TP{mapping.tp_size}"
        if not isinstance(lm_head, Linear) or lm_head.tp_mode != TensorParallelMode.COLUMN:
            return "the LM head is not a column-parallel Linear"
        if lm_head.bias is not None or lm_head.weight.dtype != torch.bfloat16:
            return "the LM head is not an unbiased BF16 projection"
        if getattr(lm_head, "lora", None) is not None:
            return "the LM head has LoRA"
        if getattr(lm_head, "use_custom_cublas_mm", False) or getattr(
            lm_head, "use_cute_dsl_bf16_gemm", False
        ):
            return "the LM head's GEMM is not the built-in F.linear (cuBLAS) call"
        shard = int(lm_head.weight.shape[0])
        if shard * 2 > ROW_BYTES // WORLD or (shard * WORLD) % 8:
            return f"vocabulary shards of {shard} do not fit the {LDC}-element page rows"
        try:
            arch_target()
        except RuntimeError as exc:
            return str(exc)
        return None

    def _census(self, lm_head) -> frozenset:
        """The batch sizes whose strided GEMM equals the built-in's bit for bit (agreed); pages reset after.

        Every rank writes its columns into every owner's pages (the real ranges), reads its
        own columns back and compares them with ``F.linear``'s contiguous output.
        """
        import torch.nn.functional as F

        # Install runs with grad mode on; the served weight may require grad (``out=`` refuses it).
        weight = lm_head.weight.detach()
        generator = torch.Generator(device=self.device).manual_seed(20261115 + self.rank)
        rows_max = MAX_REQUESTS * ROWS_PER_REQUEST
        x_all = torch.randn(
            (rows_max, self.hidden), generator=generator, device=self.device, dtype=torch.float32
        ).to(torch.bfloat16)
        ok = []
        for batch in range(MIN_REQUESTS, MAX_REQUESTS + 1):
            rows = batch * ROWS_PER_REQUEST
            q = math.ceil(batch / WORLD)
            x = x_all[:rows]
            reference = F.linear(x, weight)
            produced = self.pages.producer(q, rows, self.shard)
            torch.mm(x, weight.t(), out=produced)
            torch.cuda.synchronize(self.device)
            if torch.equal(reference.view(torch.int16), produced.view(torch.int16)):
                ok.append(batch)
        del x_all
        everyone = _tp_agreement.Distributed.get(self.mapping).tp_allgather(ok, small_payload=True)
        agreed = frozenset(set(everyone[0]).intersection(*map(set, everyone[1:])))
        # Every rank finished writing (the allgather above follows each rank's synchronize):
        # each owner resets its own pages.
        own = self.pages.owned(SLOTS, self.vocab)
        own.fill_(-1)
        torch.cuda.synchronize(self.device)
        _tp_agreement.Distributed.get(self.mapping).tp_allgather(None, small_payload=True)
        missing = sorted(set(range(MIN_REQUESTS, MAX_REQUESTS + 1)) - agreed)
        if missing:
            logger.warning(
                f"[musespark_open] lm_head_row_owner: the strided LM-head GEMM differs from the "
                f"built-in's at batch sizes {missing}; those steps keep the built-in head"
            )
        return agreed

    # ------------------------------------------------------------------ producer
    def takes(self, hidden_states: torch.Tensor, attn_metadata) -> bool:
        """Whether this decode call writes into the pages (rank-uniform host facts)."""
        rows = int(hidden_states.shape[0])
        if rows % ROWS_PER_REQUEST:
            return False
        batch = rows // ROWS_PER_REQUEST
        return (
            batch in self.batches
            and getattr(attn_metadata, "num_seqs", None) == batch
            and hidden_states.shape[1] == self.hidden
        )

    def produce(self, lm_head, hidden_states: torch.Tensor) -> torch.Tensor:
        """The LM-head GEMM into the owners' pages; returns the step's pending ``[rows, vocab]`` row.

        The row is a placeholder (allocated, not written) until a consumer gathers it.
        """
        rows = int(hidden_states.shape[0])
        q = math.ceil(rows // ROWS_PER_REQUEST / WORLD)
        torch.mm(
            hidden_states, lm_head.weight.detach().t(), out=self.pages.producer(q, rows, self.shard)
        )
        return torch.empty((rows, self.vocab), dtype=torch.bfloat16, device=hidden_states.device)

    # ------------------------------------------------------------------ consumers
    def _timeout_ms(self) -> int:
        return int(POLL_TIMEOUT_MS)

    def gather_into(self, logits: torch.Tensor) -> None:
        """The whole gathered BF16 row into ``logits`` (the step's placeholder); resets the pages.

        Every rank reads every owner's rows (polled), a TP barrier, then each owner resets its
        live rows, then a TP barrier (no peer writes the pages again before the reset).
        ``logits`` then holds the built-in's gathered row bit for bit.
        """
        from tensorrt_llm._torch.distributed import allgather

        rows = int(logits.shape[0])
        batch = rows // ROWS_PER_REQUEST
        q, _, live = owner_slice(batch, self.rank, self.world)
        ops = _ops()
        ops.poll_gather(
            self.pages.stitched(q, rows, self.vocab),
            logits.view(torch.int16),
            self.status,
            self._timeout_ms(),
        )
        # Every rank's reads are issued before its barrier contribution (stream order).
        allgather(self.barrier_word, self.mapping, dim=0)
        ops.owner_cap(
            self.pages.owned(q, self.vocab),
            live * ROWS_PER_REQUEST,
            None,
            self.vocab,
            *self.cap,
            self.status,
            self._timeout_ms(),
            False,
        )
        allgather(self.barrier_word, self.mapping, dim=0)

    def drain(self, rows: int) -> None:
        """Poll and reset this owner's live rows of a step nothing consumed, then a TP barrier."""
        from tensorrt_llm._torch.distributed import allgather

        batch = rows // ROWS_PER_REQUEST
        q, _, live = owner_slice(batch, self.rank, self.world)
        _ops().owner_cap(
            self.pages.owned(q, self.vocab),
            live * ROWS_PER_REQUEST,
            None,
            self.vocab,
            *self.cap,
            self.status,
            self._timeout_ms(),
            True,
        )
        allgather(self.barrier_word, self.mapping, dim=0)

    def verify_refusal(
        self, worker, logits, draft_tokens, draft_probs, num_contexts, batch_size, spec_metadata
    ) -> Optional[str]:
        """Why the owner verify cannot stand in for the built-in rejection body on this step, or None.

        Host facts every rank holds alike.
        """
        rows = int(logits.shape[0])
        k = int(draft_tokens.shape[1]) if draft_tokens.dim() == 2 else -1
        if num_contexts != 0:
            return "a step with context rows"
        if spec_metadata.is_ragged_verify:
            return "a ragged verify"
        if k != ROWS_PER_REQUEST - 1 or draft_tokens.shape[0] != batch_size:
            return f"draft tokens of shape {tuple(draft_tokens.shape)}"
        if rows != batch_size * ROWS_PER_REQUEST:
            return f"{rows} logits rows for {batch_size} requests"
        if not spec_metadata.advanced_sampling_mode.is_fused:
            return "a sampling mode other than the fused one"
        if spec_metadata.min_ps is None or spec_metadata.top_ks is None:
            return "no per-row sampling parameters"
        if getattr(spec_metadata, "batch_slot_ids", None) is None:
            return "no batch slot ids"
        if getattr(worker, "_d2t", None) is not None:
            return "a draft-to-target vocabulary map"
        if isinstance(draft_probs, DeferredDraftProbs):
            if draft_probs.vocab != self.vocab or draft_probs.draft_len != k:
                return "compact draft rows of another vocabulary or length"
        elif not (
            isinstance(draft_probs, torch.Tensor)
            and draft_probs.dtype == torch.float32
            and draft_probs.dim() == 3
            and tuple(draft_probs.shape) == (batch_size, k, self.vocab)
            and draft_probs.is_contiguous()
        ):
            return "dense draft rows of another shape"
        return None

    def _rows(self, values: torch.Tensor, first: int, count: int, total: int) -> torch.Tensor:
        """``values[first : first + count]``, the rows past ``total`` repeating row ``total - 1``."""
        if first + count <= total:
            return values[first : first + count]
        index = torch.arange(first, first + count, device=values.device).clamp_(max=total - 1)
        return values.index_select(0, index)

    def verify(
        self, worker, head, logits, draft_tokens, draft_probs, batch_size, spec_metadata
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The built-in ``_sample_and_accept_draft_tokens_rejection`` of a decode step, owner-sliced.

        Consumes the pending row. Returns the built-in's ``(accepted_tokens, num_accepted_tokens)``.
        """
        from tensorrt_llm._torch.distributed import allgather

        k = int(draft_tokens.shape[1])
        width = k + 1
        q, first, live = owner_slice(batch_size, self.rank, self.world)
        rows = q * width
        ops = _ops()
        capped = torch.empty((rows, self.vocab), dtype=torch.float32, device=logits.device)
        ops.owner_cap(
            self.pages.owned(q, self.vocab),
            live * width,
            capped,
            self.vocab,
            *self.cap,
            self.status,
            self._timeout_ms(),
            True,
        )
        head.release_row_owner()
        # The built-in body from here, on the owner's requests (padded to q with the last one).
        slots = self._rows(spec_metadata.batch_slot_ids, first, q, batch_size)
        SpecWorkerBase._zero_padding_rows(
            capped,
            _OwnerSlots(slots, getattr(spec_metadata, "dummy_slot_row", 0)),
            0,
            q,
            width,
            packed_verify=True,
            inplace=True,
        )
        total = batch_size * width
        start = first * width
        probs = worker._compute_probs_from_logits(
            spec_metadata.advanced_sampling_mode,
            capped,
            self._rows(spec_metadata.temperatures, start, rows, total),
            self._rows(spec_metadata.top_ks, start, rows, total),
            self._rows(spec_metadata.top_ps, start, rows, total),
            self._rows(spec_metadata.min_ps, start, rows, total),
        )
        target_probs = probs.view(q, width, self.vocab)[:live]
        if isinstance(draft_probs, DeferredDraftProbs):
            mine = DeferredDraftProbs(
                draft_probs.store,
                draft_probs.slot_ids[first : first + live],
                draft_probs.draft_len,
                draft_probs.vocab,
            )
            mark_private_verify_logits(capped)
            owner_draft = mine.materialize(private_verify_scratch(capped, probs, target_probs))
        else:
            owner_draft = draft_probs[first : first + live]
        full_draft_tokens = draft_tokens.to(torch.int32).contiguous()
        seed, offset = worker._rng_state_per_request(
            spec_metadata, 0, batch_size, slot=spec_metadata.rng_accept_slot
        )
        tokens = torch.empty((live, width), dtype=torch.int32, device=logits.device)
        accepted_num = torch.zeros(live, dtype=torch.int32, device=logits.device)
        emitted = torch.zeros(live, dtype=torch.int32, device=logits.device)
        chain_module().chain_speculative_sampling_row_base(
            owner_draft,
            full_draft_tokens[first : first + live],
            target_probs,
            tokens,
            accepted_num,
            emitted,
            True,
            seed.contiguous(),
            offset.contiguous(),
            first,
        )
        mine = torch.zeros((q, width + 1), dtype=torch.int32, device=logits.device)
        mine[:live, :width] = tokens
        mine[:live, width] = emitted + 1
        everyone = allgather(mine, self.mapping, dim=0)
        gen_accepted = everyone[:batch_size, :width]
        gen_num_accepted = everyone[:batch_size, width]
        # SpecWorkerBase._sample_and_accept_draft_tokens_rejection from its kernel on (no
        # context rows, uniform verify).
        accepted_tokens = torch.empty((batch_size, width), dtype=torch.int, device=logits.device)
        num_accepted_tokens = torch.ones(batch_size, dtype=torch.int, device=logits.device)
        if worker.force_num_accepted_tokens != 0.0:
            gen_accepted[:, 1 : k + 1].copy_(full_draft_tokens)
        accepted_tokens[0:] = gen_accepted
        num_accepted_tokens[0:] = gen_num_accepted
        num_accepted_tokens = worker._apply_force_accepted_tokens(
            num_accepted_tokens, 0, k, spec_metadata=spec_metadata
        )
        self.verified_steps += 1
        return accepted_tokens, num_accepted_tokens

    def log_declined(self, reason: str) -> None:
        if reason not in self._declined_logged:
            self._declined_logged.add(reason)
            logger.info(
                f"[musespark_open] lm_head_row_owner: {reason}; such steps gather the whole row "
                "from the pages and run the built-in verify"
            )


class RowOwnerVerifyDFlashWorker(DFlashWorker):
    """The DFlash worker whose rejection verify of a row-owned head row runs owner-sliced.

    Never constructed: ``install_worker`` re-classes the worker as a class derived from this
    one and its current class, before the step folds compose on top (``verify_fold``'s
    ``rejection_fold`` then folds the bookkeeping of the accepted tokens and counts this
    returns, as it does the built-in's).
    """

    def _sample_and_accept_draft_tokens_rejection(
        self, logits, draft_tokens, draft_probs, num_contexts, batch_size, spec_metadata
    ):
        head = self.__dict__.get("_split_sampler_head")
        owner = getattr(head, "row_owner", None)
        if owner is None or not head.pending_row_owner or not head.owns(logits):
            return super()._sample_and_accept_draft_tokens_rejection(
                logits, draft_tokens, draft_probs, num_contexts, batch_size, spec_metadata
            )
        reason = owner.verify_refusal(
            self, logits, draft_tokens, draft_probs, num_contexts, batch_size, spec_metadata
        )
        if reason is not None:
            owner.log_declined(reason)
            return super()._sample_and_accept_draft_tokens_rejection(
                head.materialize(logits),
                draft_tokens,
                draft_probs,
                num_contexts,
                batch_size,
                spec_metadata,
            )
        return owner.verify(
            self, head, logits, draft_tokens, draft_probs, batch_size, spec_metadata
        )


_COMPOSED: Dict[type, type] = {}


def install_worker(spec_worker) -> bool:
    """Re-class the DFlash worker for the owner verify; idempotent. Call before ``verify_fold``.

    Raises:
        ValueError: if the worker is not a DFlash worker or already overrides the rejection body.
    """
    if isinstance(spec_worker, RowOwnerVerifyDFlashWorker):
        return False
    if not isinstance(spec_worker, DFlashWorker):
        raise ValueError(
            "[musespark_open] lm_head_row_owner cannot run here: the spec worker is "
            f"{type(spec_worker).__name__}, not a DFlashWorker"
        )
    cls = type(spec_worker)
    name = "_sample_and_accept_draft_tokens_rejection"
    if getattr(cls, name) is not getattr(DFlashWorker, name) or name in vars(spec_worker):
        raise ValueError(
            f"[musespark_open] lm_head_row_owner cannot run here: the spec worker class {cls.__name__} "
            f"already overrides {name}"
        )
    composed = _COMPOSED.get(cls)
    if composed is None:
        composed = type(
            f"RowOwner{cls.__name__}",
            (RowOwnerVerifyDFlashWorker, cls),
            {
                "__module__": __name__,
                "__doc__": f"``{cls.__qualname__}`` with ``RowOwnerVerifyDFlashWorker``'s verify.",
            },
        )
        _COMPOSED[cls] = composed
    spec_worker.__class__ = composed
    return True


def install(llm, mapping) -> RowOwnerHead:
    """Attach the owner head to the installed split sampler's head (collective); returns it.

    Call after ``split_sampler`` and ``wide_split_sampler`` installed (the head takes steps of
    up to ``MAX_REQUESTS * 8`` rows) and after ``install_worker``. Idempotent.

    Raises:
        ValueError: on every rank alike, when the lever cannot run here.
    """
    from ..split_sampler import target_head as _target_head

    reason = None
    head = None
    if not _target_head.installed(llm):
        reason = "the split_sampler head is not installed"
    else:
        head = llm.logits_processor._split_head
        if head.row_owner is not None:
            return head.row_owner
        if head.max_rows < MAX_REQUESTS * ROWS_PER_REQUEST:
            reason = "wide_split_sampler is not on (the head takes at most 128 rows)"
        elif head.vp is not None or head.s2 is not None or head.pv_gather is not None:
            reason = "another head path (vocab_parallel_head / distributed_head_sampler) takes these steps"
        elif head.rs is not None:
            reason = "the fused rejection sampler (rejection_sampler) verifies these steps"
        elif not isinstance(getattr(llm, "spec_worker", None), RowOwnerVerifyDFlashWorker):
            reason = "the DFlash worker does not run the owner verify (install_worker)"
    reason = _tp_agreement.agreed_refusal(mapping, reason)
    if reason is not None:
        raise ValueError(f"[musespark_open] lm_head_row_owner cannot run here: {reason}")
    owner = RowOwnerHead(mapping, llm.lm_head, head.multiplier, head.soft_cap)
    head.row_owner = owner
    return owner


@functools.lru_cache(maxsize=None)
def describe() -> str:
    return (
        f"LM-head rows owned by one rank of {WORLD} on decode steps of "
        f"{MIN_REQUESTS}..{MAX_REQUESTS} requests"
    )
