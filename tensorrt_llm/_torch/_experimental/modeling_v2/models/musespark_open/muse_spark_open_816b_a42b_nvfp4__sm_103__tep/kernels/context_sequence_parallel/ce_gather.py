# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A sequence-parallel layer's attention input all-gathered by the copy engines, QKV in row blocks.

What it replaces. On a sequence-parallel context step every layer all-gathers its
attention input (``[shard, 8192]`` BF16 per rank) with an NCCL ring kernel and then
runs the QKV projection over all rows. The ring kernel needs SMs, so it runs next to
nothing and the projection waits for its last byte.

What runs instead, on steps of at least ``MIN_ROWS`` rows. Rank ``r`` gathers on a
second stream ("copy stream"):

* after the attention input is produced, the copy engines copy it into rows
  ``[r * shard, (r + 1) * shard)`` of each peer's receive buffer, peers ``r+1, r+2, r+3``
  in turn, and after each copy ``cuStreamWriteValue32`` sets the flag word that peer
  keeps for source ``r`` (default flags: a fence before the write);
* the QKV projection runs in row blocks on the main stream: rank ``r``'s own rows from
  its own attention input first, then each peer's rows from the receive buffer after a
  ``cuStreamWaitValue32`` on that peer's flag, peers in the order their copies arrive
  (``r-1, r-2, r-3``: every sender copies to its next rank first). After the last
  wait the flags are reset (fenced writes) and the main stream joins the copy stream.

The receive buffers and flag words are one ``McastGPUBuffer`` per rank (the allocation
of the projection-publish transport), used through its unicast peer mappings only.

Ownership and progress (cuFlow):

* One writer per region and layer: rank ``p``'s copy writes ``recv_r[p * shard...]``;
  one reader: rank ``r``'s block GEMM for ``p``.
* Publication: the copy completes, then the fenced flag write; acquire: the stream-side
  wait, then the GEMM. The reader resets its flags after its last wait.
* Visibility: the waits carry no ``CU_STREAM_WAIT_VALUE_FLUSH`` (the device reports
  ``CAN_FLUSH_REMOTE_WRITES`` = 0). The copied rows are visible to the reader's GEMM
  because the sender orders the copy before its fenced flag write on one stream and
  the NVLink peer writes land in the reader's memory before the flag word does. This
  rests on that ordering of the peer-write path, not on a documented acquire of the
  wait; the unit and in-engine checks under graph replay are bit exact.
* Consumers: on a step this takes, the view ``gather`` returns never holds this
  rank's own rows (``project`` reads them from the attention input itself), so
  ``project`` must stay its only reader; ``settle`` completes the view in place for
  any other reader.
* Reuse (one buffer for every layer): rank ``p`` writes ``recv_r`` again for layer
  ``L + 1`` only after its own layer ``L`` passed the o_proj and experts
  reduce-scatters, which need rank ``r``'s contributions, issued after ``r``'s layer
  ``L`` block GEMMs and flag resets; between steps the forward's final all-gather does
  the same. This needs one collective of all ranks between two gathers; every layer
  has two.
* Progress: a wait targets a copy its peer enqueued right after its own producer of the
  same layer, which depends only on collectives this rank already passed. Copy
  engines take no SM and the waits hold none, so nothing assumes co-residency. A peer
  that never copies leaves the stream waiting without an error: ``start_watch``'s
  thread sees a forward that began and did not end for ``stall_s`` seconds
  (``first_stall_s`` before this rank ended its first one), reads this rank's flag
  words on another stream, logs the sources that never arrived and aborts.

Numerics: bitwise. The gathered rows are the same bytes, and each row block's GEMM is
``torch.matmul`` with the projection's own weight, which computes every row as the
whole-step GEMM does on the cuBLAS this was certified with (the in-engine checksum
arm compares the hidden states).
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable, List, Optional, Tuple

import torch
from cuda.bindings import driver as _cu
from cuda.bindings import runtime as _cudart

from tensorrt_llm._torch.distributed import allgather
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - the image ships Triton
    triton = None

#: Seconds a sequence-parallel forward may stay begun and not ended before the watch
#: aborts, once this rank has ended one (a served context step takes well under a second;
#: a later step may still load or compile a kernel).
STALL_S = 120.0
#: The same bound while this rank has not yet ended any sequence-parallel forward: the
#: first one (the warmup) compiles and loads the step's kernels, which on a cold cache
#: takes longer than ``STALL_S``.
FIRST_STALL_S = 900.0
#: The fewest rows of a step whose attention input is gathered here: below it the
#: flags' fixed per-flag cost eats the overlap (measured: next to no gain per layer at
#: 2048 rows, a clear and growing gain from 4096 rows up).
MIN_ROWS = 4096
#: int32 words between two flag words (one 128-byte line each).
FLAG_STRIDE = 32
#: Seconds between two reads of the forward counters by the watch thread.
WATCH_INTERVAL_S = 1.0

_ALIGN = 2 << 20


def _check(result):
    err = result[0] if isinstance(result, tuple) else result
    if err != _cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"[musespark_open] ce_gather: CUDA driver call failed: {err}")


def _write_value(address: int, value: int) -> None:
    """``cuStreamWriteValue32`` on the current stream with the default (fenced) flags."""
    _check(
        _cu.cuStreamWriteValue32(
            _cu.CUstream(torch.cuda.current_stream().cuda_stream),
            _cu.CUdeviceptr(address),
            value,
            _cu.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
        )
    )


def _wait_value(address: int, value: int) -> None:
    """``cuStreamWaitValue32`` (equal) on the current stream."""
    _check(
        _cu.cuStreamWaitValue32(
            _cu.CUstream(torch.cuda.current_stream().cuda_stream),
            _cu.CUdeviceptr(address),
            value,
            _cu.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
        )
    )


class _HostWords:
    """Pinned host int32 words a kernel updates at system scope and a host thread reads unsynchronized.

    Triton reads ``data_ptr()`` and ``dtype`` of a pointer argument; the words' device
    address is their host address (unified addressing).
    """

    dtype = torch.int32

    def __init__(self, count: int) -> None:
        # Without pinned memory there is no mapped device pointer: the error below makes
        # ``CopyEngineGather.build`` refuse, and the NCCL all-gather runs instead.
        self.host = torch.zeros(count, dtype=torch.int32, pin_memory=prefer_pinned())
        status, pointer = _cudart.cudaHostGetDevicePointer(self.host.data_ptr(), 0)
        if status != _cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaHostGetDevicePointer of the forward counters: {status}")
        self._pointer = int(pointer)

    def data_ptr(self) -> int:
        return self._pointer


if triton is not None:

    @triton.jit
    def _count_kernel(words_ptr, INDEX: tl.constexpr):
        """Add one to ``words[INDEX]`` (pinned host memory) at system scope."""
        tl.atomic_add(words_ptr + INDEX, 1, sem="release", scope="sys")


#: The forward counters' slots.
_BEGUN, _ENDED = 0, 1


def projection_refusal(linear) -> Optional[str]:
    """Why ``linear`` (a layer's ``qkv_proj``) cannot run in row blocks here, or None."""
    from tensorrt_llm._torch.modules.linear import UnquantizedLinearMethod

    weight = getattr(linear, "weight", None)
    if not isinstance(getattr(linear, "quant_method", None), UnquantizedLinearMethod):
        return "the QKV projection is quantized"
    if weight is None or weight.dtype != torch.bfloat16 or weight.dim() != 2:
        return "the QKV projection's weight is not a 2-D BF16 matrix"
    if getattr(linear, "bias", None) is not None:
        return "the QKV projection has a bias"
    if getattr(linear, "use_cute_dsl_bf16_gemm", False) or getattr(
        linear, "use_custom_cublas_mm", False
    ):
        return "the QKV projection runs a custom GEMM"
    if getattr(linear, "lora", None) is not None:
        return "the QKV projection carries LoRA"
    return None


class CopyEngineGather:
    """The copy-engine gather of every sequence-parallel layer of one model, and its row-block QKV.

    Built (``build``, collective over the TP group) on the first eager prefill step, before
    any CUDA graph is captured; until then, and on steps it does not take, ``gather``
    is the NCCL all-gather the built-in layer runs.
    """

    def __init__(self, mapping, max_rows: int, hidden: int, comm=None) -> None:
        self.mapping = mapping
        self.rank = mapping.tp_rank
        self.world = mapping.tp_size
        self.hidden = int(hidden)
        # A step of ``rows`` rows gathers ``world * ceil(rows / world)`` rows.
        self.max_rows = -(-int(max_rows) // self.world) * self.world
        self._comm = comm
        self.built = False
        self.refused: Optional[str] = None
        self.allocated = 0
        self._pending: Optional[Tuple[torch.Tensor, torch.Tensor, int, int]] = None
        self._watch: Optional[threading.Thread] = None
        self._watch_stop: Optional[threading.Event] = None
        #: Set (instead of aborting) when the watch was started with ``abort=False`` (tests).
        self.stalled: Optional[dict] = None
        self._flag_snap: Optional[torch.Tensor] = None
        self.gathers = 0

    # ------------------------------------------------------------------ build

    def build(self) -> None:
        """Allocate the receive buffers and flags on every rank (collective); refuse alike on failure."""
        from tensorrt_llm._torch.distributed.ops import (
            _get_mnnvl_workspace_comm,
            _make_mnnvl_mcast_buffer,
            _mnnvl_workspace_all_succeeded,
        )

        if self.built:
            return
        self.built = True
        comm = self._comm if self._comm is not None else _get_mnnvl_workspace_comm(self.mapping)
        refusal = None
        if torch.cuda.is_current_stream_capturing():
            refusal = "the first prefill step is captured"
        elif triton is None:
            refusal = "Triton is not available"
        if not _mnnvl_workspace_all_succeeded(comm, refusal is None):
            self.refused = refusal or "another rank refused"
            return
        error: Optional[BaseException] = None
        try:
            recv_bytes = -(-self.max_rows * self.hidden * 2 // _ALIGN) * _ALIGN
            flag_words = self.world * FLAG_STRIDE
            total = recv_bytes + flag_words * 4
            torch.cuda.synchronize()
            free_before = torch.cuda.mem_get_info()[0]
            self._handle = _make_mnnvl_mcast_buffer(
                comm, total, self.mapping, self.mapping.is_multi_node()
            )
            self.recv: List[torch.Tensor] = [
                self._handle.get_uc_buffer(p, (self.max_rows, self.hidden), torch.bfloat16, 0)
                for p in range(self.world)
            ]
            self.flags = self._handle.get_uc_buffer(
                self.rank, (flag_words,), torch.int32, recv_bytes // 4
            )
            self._flag_base = [
                self._handle.get_uc_buffer(
                    p, (flag_words,), torch.int32, recv_bytes // 4
                ).data_ptr()
                for p in range(self.world)
            ]
            self.flags.zero_()
            self.copy_stream = torch.cuda.Stream()
            self.watch_stream = torch.cuda.Stream()
            self.counters = _HostWords(2)
            # Launch both counter kernels and the flag read once now, while no stream waits:
            # a kernel's first launch loads its module, and that load does not return while
            # any stream of the device is blocked on a flag a peer has not yet written.
            self._count(_BEGUN)
            self._count(_ENDED)
            self.flag_words()
            torch.cuda.synchronize()
            self.counters.host.zero_()
            self.allocated = max(0, free_before - torch.cuda.mem_get_info()[0])
        except Exception as exc:  # noqa: BLE001 - every failure must reach the agreement
            error = exc
        if not _mnnvl_workspace_all_succeeded(comm, error is None):
            self.refused = f"the buffers could not be allocated: {error!r}"
            return
        logger.info(
            f"[musespark_open] ce_gather: sequence-parallel steps of {MIN_ROWS}..{self.max_rows} rows "
            f"gather the attention input on the copy engines ({total / 2**20:.1f} MiB requested, "
            f"{self.allocated / 2**20:.1f} MiB of device memory on this rank)"
        )

    @property
    def armed(self) -> bool:
        return self.built and self.refused is None

    def takes(self, rows: int) -> bool:
        """Whether a step of ``rows`` rows gathers here (host facts, alike on every rank)."""
        return self.armed and MIN_ROWS <= rows <= self.max_rows

    # --------------------------------------------------------------- protocol

    def _flag_address(self, owner: int, source: int) -> int:
        return self._flag_base[owner] + 4 * source * FLAG_STRIDE

    def arrival_order(self) -> List[int]:
        """The sources in the order their copies reach this rank."""
        return [(self.rank - k) % self.world for k in range(1, self.world)]

    def gather(self, attn_input: torch.Tensor, rows: int) -> torch.Tensor:
        """The step's attention input, ``[rows, hidden]``: the hook the built-in layer calls.

        On a step this takes, the returned view's peer rows are complete only after
        the row-block projection's waits, so it is handed to that projection only
        (``project``); every other step gets the NCCL all-gather.
        """
        if not self.takes(rows):
            return allgather(attn_input, self.mapping, dim=0)[:rows]
        if self._pending is not None:
            raise RuntimeError(
                "[musespark_open] ce_gather: a gathered attention input was not projected"
            )
        shard = int(attn_input.shape[0])
        if shard * self.world < rows or not attn_input.is_contiguous():
            raise ValueError(
                f"[musespark_open] ce_gather: a [{shard}, ...] shard cannot hold {rows} rows"
            )
        rank = self.rank
        mine = slice(rank * shard, (rank + 1) * shard)
        main = torch.cuda.current_stream()
        self.copy_stream.wait_stream(main)
        with torch.cuda.stream(self.copy_stream):
            for k in range(1, self.world):
                peer = (rank + k) % self.world
                self.recv[peer][mine].copy_(attn_input)
                _write_value(self._flag_address(peer, rank), 1)
        view = self.recv[rank][:rows]
        self._pending = (view, attn_input, shard, rows)
        self.gathers += 1
        return view

    def is_pending(self, tensor: torch.Tensor) -> bool:
        """Whether ``tensor`` is the view ``gather`` returned and nothing has projected yet."""
        pending = self._pending
        return (
            pending is not None
            and tensor.data_ptr() == pending[0].data_ptr()
            and tensor.shape == pending[0].shape
        )

    def project(self, weight: torch.Tensor) -> torch.Tensor:
        """``view @ weight.T`` for the pending view, in row blocks behind the flags."""
        view, own, shard, rows = self._pending
        self._pending = None
        out = torch.empty((rows, weight.shape[0]), dtype=view.dtype, device=view.device)
        w_t = weight.t()
        rank = self.rank

        def block(source: int) -> Optional[slice]:
            lo, hi = source * shard, min((source + 1) * shard, rows)
            return slice(lo, hi) if hi > lo else None

        own_rows = block(rank)
        if own_rows is not None:
            torch.matmul(own[: own_rows.stop - own_rows.start], w_t, out=out[own_rows])
        order = self.arrival_order()
        for index, source in enumerate(order):
            _wait_value(self._flag_address(rank, source), 1)
            if index == len(order) - 1:
                for done in order:
                    _write_value(self._flag_address(rank, done), 0)
            rows_of = block(source)
            if rows_of is not None:
                torch.matmul(self.recv[rank][rows_of], w_t, out=out[rows_of])
        torch.cuda.current_stream().wait_stream(self.copy_stream)
        return out

    def settle(self) -> None:
        """Complete the pending view in place (every flag waited and reset, own rows copied) and join."""
        view, own, shard, rows = self._pending
        self._pending = None
        order = self.arrival_order()
        for source in order:
            _wait_value(self._flag_address(self.rank, source), 1)
        for source in order:
            _write_value(self._flag_address(self.rank, source), 0)
        lo, hi = self.rank * shard, min((self.rank + 1) * shard, rows)
        if hi > lo:
            view[lo:hi].copy_(own[: hi - lo])
        torch.cuda.current_stream().wait_stream(self.copy_stream)

    # ------------------------------------------------------------- the watch

    def _count(self, index: int) -> None:
        _count_kernel[(1,)](self.counters, INDEX=index, num_warps=1)

    def begin(self) -> None:
        """Count a sequence-parallel forward as begun (a one-thread kernel; captured with the step)."""
        if self.armed:
            self._count(_BEGUN)

    def end(self) -> None:
        """Count a sequence-parallel forward as ended."""
        if self.armed:
            self._count(_ENDED)

    def flag_words(self, bound_s: float = 5.0) -> Optional[List[int]]:
        """This rank's flag word per source, or None if not read within ``bound_s``.

        One contiguous device-to-host copy of the flag block on the watch stream: a
        copy-engine transfer into a pinned buffer kept from the first call, with no kernel
        and no allocation, so it completes while the compute streams wait on a flag.
        """
        # Inference mode on every thread: the buffer is first made by ``build`` inside the
        # engine's inference mode and later refilled from the watch's helper thread.
        with torch.inference_mode():
            snap = self._flag_snap
            if snap is None:
                snap = torch.empty(
                    self.flags.numel(), dtype=torch.int32, pin_memory=prefer_pinned()
                )
                self._flag_snap = snap
            with torch.cuda.stream(self.watch_stream):
                snap.copy_(self.flags, non_blocking=True)
                done = torch.cuda.Event()
                done.record()
        start = time.monotonic()
        while not done.query():
            if time.monotonic() - start > bound_s:
                return None
            time.sleep(0.001)
        return [int(v) for v in snap[::FLAG_STRIDE]]

    def _flag_words_bounded(self, bound_s: float = 5.0) -> Optional[List[int]]:
        """``flag_words`` from a helper thread, abandoned after ``bound_s`` even if a CUDA call blocks."""
        result: List[Optional[List[int]]] = []
        reader = threading.Thread(
            target=lambda: result.append(self.flag_words(bound_s)),
            name="musespark_open.ce_gather_flags",
            daemon=True,
        )
        reader.start()
        reader.join(bound_s + 1.0)
        return result[0] if result else None

    def start_watch(
        self,
        stall_s: Optional[float] = None,
        interval_s: float = WATCH_INTERVAL_S,
        first_stall_s: Optional[float] = None,
        abort: bool = True,
        on_stall: Optional[Callable[[dict], None]] = None,
    ) -> bool:
        """Watch the forward counters from a host thread; returns whether a thread runs.

        A forward that began and has not ended for ``stall_s`` seconds (default
        ``STALL_S``) is waiting on a flag that
        no peer set: the thread logs this rank's flag words (the sources still at 0 never
        arrived) and aborts the process, or with ``abort=False`` records the report in
        ``stalled`` and calls ``on_stall`` (tests). Until this rank has ended one forward
        the bound is ``first_stall_s`` (default ``FIRST_STALL_S``), at least ``stall_s``.
        """
        if not self.armed or self._watch is not None:
            return self._watch is not None
        if stall_s is None:
            stall_s = STALL_S
        if first_stall_s is None:
            first_stall_s = FIRST_STALL_S
        first_stall_s = max(first_stall_s, stall_s)
        stop = threading.Event()
        device = torch.cuda.current_device()

        def watch():
            torch.cuda.set_device(device)
            seen_end, since = None, None
            while not stop.wait(interval_s):
                begun = int(self.counters.host[_BEGUN])
                ended = int(self.counters.host[_ENDED])
                now = time.monotonic()
                if begun <= ended or ended != seen_end:
                    seen_end, since = ended, now
                    continue
                if now - since < (stall_s if ended > 0 else first_stall_s):
                    continue
                logger.error(
                    f"[musespark_open] ce_gather: rank {self.rank} has waited {now - since:.1f} s in a "
                    f"sequence-parallel forward (begun {begun}, ended {ended}); reading its flag words"
                )
                words = self._flag_words_bounded()
                missing = (
                    None if words is None else [s for s in self.arrival_order() if words[s] == 0]
                )
                report = dict(
                    rank=self.rank,
                    begun=begun,
                    ended=ended,
                    stalled_s=round(now - since, 3),
                    flags=words,
                    missing=missing,
                )
                logger.error(
                    f"[musespark_open] ce_gather: rank {self.rank} has waited {now - since:.1f} s in a "
                    f"sequence-parallel forward (begun {begun}, ended {ended}); flag words per source "
                    f"{words}, sources that never arrived {missing}"
                )
                if abort:
                    os.abort()
                self.stalled = report
                if on_stall is not None:
                    on_stall(report)
                return

        self._watch_stop = stop
        self._watch = threading.Thread(
            target=watch, name="musespark_open.ce_gather_watch", daemon=True
        )
        self._watch.start()
        return True

    def stop_watch(self) -> None:
        """Stop the watch thread, if one runs (tests)."""
        if self._watch is not None:
            self._watch_stop.set()
            self._watch.join()
            self._watch = None


class CopyEngineGatherQKV:
    """Mixin for a layer's ``self_attn.qkv_proj``: the pending gathered view runs in row blocks."""

    _sp_ce_gather: Optional[CopyEngineGather] = None

    def forward(self, input, *args, **kwargs):
        gather = self._sp_ce_gather
        if gather is not None and isinstance(input, torch.Tensor) and gather.is_pending(input):
            if not args and not kwargs.get("lora_params"):
                return gather.project(self.weight)
            gather.settle()
        return super().forward(input, *args, **kwargs)


_CLASSES = {}


def _with_mixin(base: type) -> type:
    if issubclass(base, CopyEngineGatherQKV):
        return base
    if base not in _CLASSES:
        _CLASSES[base] = type(
            "CopyEngineGatherQKVOver" + base.__name__,
            (CopyEngineGatherQKV, base),
            {"__module__": __name__},
        )
    return _CLASSES[base]


def install(model) -> Optional[CopyEngineGather]:
    """Hook every layer's attention-input gather and return the gather, or None.

    Changes nothing and returns None when a layer's projection cannot run in row blocks
    (logged); the buffers are built later (``CopyEngineGather.build``).
    """
    layers = list(model.layers)
    reasons = {layer.layer_idx: projection_refusal(layer.self_attn.qkv_proj) for layer in layers}
    refused = {idx: why for idx, why in reasons.items() if why is not None}
    if refused:
        logger.warning(
            f"[musespark_open] ce_gather: not armed: {sorted(set(refused.values()))} "
            f"(layers {sorted(refused)}); the attention input is all-gathered by NCCL"
        )
        return None
    config = model.model_config
    gather = CopyEngineGather(
        model._sp_mapping, config.max_num_tokens, config.pretrained_config.hidden_size
    )
    for layer in layers:
        qkv = layer.self_attn.qkv_proj
        qkv.__class__ = _with_mixin(type(qkv))
        qkv._sp_ce_gather = gather
        layer._sp_attn_input_gather = gather.gather
    return gather
