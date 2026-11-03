# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM (Apache-2.0):
# https://github.com/vllm-project/vllm/blob/642c18d64c4cb6db3ded8fbe1786718cd4005fcb/vllm/models/kimi_k3/nvidia/specialized_decode/runtime/prefetch.py
"""Kimi-K3 decode L2 weight prefetch at fixed points of each layer (``TRTLLM_K3_L2PF_V2``).

This ports the prefetch controller of vLLM's Kimi-K3 specialized decode
(``specialized_decode/runtime/prefetch.py``, ``PrefetchController``) to the TRT-LLM K3 decoder
layers. Small-batch decode is bound by HBM bandwidth, and a kernel that starts on weights that are
not in L2 stalls on DRAM, while other kernels (the KDA recurrence, the MLA decode, the
all-reduces) leave HBM mostly idle. At fixed points of each MoE layer, a small kernel
(:class:`~tensorrt_llm._torch.kernels.blackwell.k3_l2_prefetch_table.L2PrefetchTable`)
on a side stream pulls the weights the next kernels read into L2.

Points (``fork(point, layer_idx, num_tokens)``), in the order a MoE layer reaches them:

* ``attention_moe``: inside the attention block, for KDA once its input projections are issued
  and for MLA once its decode core (and the Helix exchange) is issued. Prefetches this layer's MoE
  front (router, latent_down) and shared-expert (gate_up, down) weights.
* ``oproj_down``: after the attention output (o_proj, its all-reduce, the post-attention residual
  mix and RMSNorm), right before the MoE block. Prefetches the MoE front again.
* ``next_attention``: once the routed experts (FC2) are done. Prefetches the next layer's
  attention weights in the order its kernels read them.

How many bytes a point prefetches depends on the step's token count, through a table in vLLM's
``batch_table`` syntax: ``"lo-hi:MiB[.carveout],..."``, for example ``"1-8:64,9-64:50"``. The cap
applies in consumption order: the first MiB of the point's weights are prefetched, and a step
whose token count is not in the table prefetches nothing at that point. The model leaves out
weights that an older prefetch switch already covers at that token count
(``TRTLLM_K3_MOE_L2_PREFETCH=front``, ``TRTLLM_K3_MOE_L2_PREFETCH_NEXT_MLA``).

The default tables are sized for the Rubin (SM107) L2 in the helix8 decode.
Per rank, the MoE front is 39.5 MB (BF16 router, FP8 latent_down) and the shared expert 17 MB. A
KDA layer's attention is 60 MB, of which 48 MB is read before the recurrence. An MLA layer's
attention is 87 MB (q side for all 96 heads), of which 51 MB is read before the decode core:

* Bulk-prefetched lines stay in L2 until they are first read, so a point's bytes occupy L2 until
  their consumer runs. Each point therefore prefetches at most 64 MiB, which keeps the bytes
  outstanding at once well within the L2.
* From 16 tokens per step (two requests with mtp7) up, ``next_attention`` stops at the weights the
  next layer reads before its KDA recurrence or MLA decode core (50 MiB: KDA QKVG, [f_a | b], f_b;
  MLA [q_a | kv_a], q_b, W_UK). Nothing it prefetched then waits in L2 across a core whose own
  reused data (KDA recurrent state, MLA split-KV partials) grows with the batch. At 8 tokens it
  also covers the o_proj of a KDA layer and the W_UV of an MLA layer (64 MiB).
* The MoE front and shared expert are read right after the attention block, so ``attention_moe``
  and ``oproj_down`` cover them whole (56 and 40 MiB).
* Nothing is prefetched above 64 tokens per step (8 requests). There the step is mostly
  HBM-bound, and the prefetch would take bandwidth from the routed experts. vLLM stops above
  8 tokens per step on GB300 (TP8), and Rubin has more HBM bandwidth.

Streams and dependencies: the front and shared-expert points are on the critical path. They run on
a stream of the greatest priority and launch with the max-shared carveout, so the SMs their
one-warp CTAs land on keep the shared-memory split a GEMM launched next needs (see
:mod:`~tensorrt_llm._torch.kernels.blackwell.k3_l2_prefetch_table`). ``next_attention``
runs on a stream of normal priority, with the driver's default carveout unless its batch table
gives one. Each fork records an event on the current (main) stream and the
prefetch stream waits for it, so a prefetch starts once the work it follows is complete and never
runs ahead of it. No main-stream kernel waits for a prefetch until
:meth:`L2PrefetchController.join`. The model calls ``join`` where the last MoE layer joins its
shared-expert stream (and again at the end of the step, where it does nothing if that join already
ran), so a captured CUDA graph always rejoins the prefetch streams. Prefetch tables are built on
first use outside graph capture (the CUDA graph warmup runs) and are then replayed as captured.
"""

from __future__ import annotations

import os
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import torch

from ...logger import logger
from .multi_stream_utils import do_multi_stream

if TYPE_CHECKING:
    from ..kernels.blackwell.k3_l2_prefetch_table import L2PrefetchTable

#: Switch: ``0`` (default) off, ``1`` every point, or a comma list of point names.
ENV_VAR = "TRTLLM_K3_L2PF_V2"
#: Prefetch points, in the order a MoE layer reaches them.
POINTS = ("attention_moe", "oproj_down", "next_attention")
#: Points on the critical path: greatest stream priority, max-shared carveout.
URGENT_POINTS = frozenset({"attention_moe", "oproj_down"})
#: Preferred shared-memory carveout (percent) of the urgent launches.
URGENT_CARVEOUT = 100
#: Per point, ``"lo-hi:MiB[.carveout]"`` by tokens per step (``<ENV_VAR>_<POINT>`` overrides).
DEFAULT_TABLES = {
    "attention_moe": "1-64:56",
    "oproj_down": "1-64:40",
    "next_attention": "1-8:64,9-64:50",
}
#: One-warp CTAs of each point's launch (``<ENV_VAR>_CTAS="<attention_moe>,<oproj_down>,<next>"``).
DEFAULT_CTAS = {"attention_moe": 32, "oproj_down": 8, "next_attention": 32}

#: ``(point, layer_idx, num_tokens) -> tensors`` a point prefetches for a layer, in read order.
TensorsFn = Callable[[str, int, int], Sequence[torch.Tensor]]
#: Tokens per step -> ``(MiB cap, carveout or None)``.
BatchTable = Dict[int, Tuple[int, Optional[int]]]


def parse_points(value: str) -> FrozenSet[str]:
    """The points ``TRTLLM_K3_L2PF_V2`` selects: none for ``""`` / ``"0"``, all for ``"1"``,
    else the comma-separated point names."""
    value = value.strip()
    if value in ("", "0"):
        return frozenset()
    if value == "1":
        return frozenset(POINTS)
    points = frozenset(v.strip() for v in value.split(",") if v.strip())
    if not points <= set(POINTS):
        raise ValueError(f"{ENV_VAR} must be 0, 1 or a comma list of {POINTS}, got {value!r}")
    return points


def batch_table(spec: str) -> BatchTable:
    """Tokens per step -> ``(MiB cap, carveout)`` from ``"lo-hi:cap[.carveout],..."`` (vLLM's
    ``batch_table`` syntax; later entries override earlier ones; a cap of 0 prefetches nothing)."""
    table: BatchTable = {}
    for item in (s.strip() for s in spec.split(",")):
        if not item:
            continue
        span, sep, value = item.partition(":")
        lo, _, hi = span.partition("-")
        cap, _, carveout = value.partition(".")
        try:
            first, last, mib = int(lo), int(hi or lo), int(cap)
            percent = int(carveout) if carveout else None
        except ValueError as err:
            raise ValueError(f"bad L2 prefetch table entry {item!r} in {spec!r}") from err
        if (
            not sep
            or first < 1
            or last < first
            or mib < 0
            or (percent is not None and not 0 <= percent <= 100)
        ):
            raise ValueError(f"bad L2 prefetch table entry {item!r} in {spec!r}")
        table.update({m: (mib, percent) for m in range(first, last + 1)})
    return table


def parse_ctas(spec: str) -> Dict[str, int]:
    """CTAs per point from ``"<attention_moe>,<oproj_down>,<next_attention>"`` (empty: defaults)."""
    if not spec.strip():
        return dict(DEFAULT_CTAS)
    values = [v.strip() for v in spec.split(",")]
    if len(values) != len(POINTS) or not all(v.isdigit() and int(v) > 0 for v in values):
        raise ValueError(f"{ENV_VAR}_CTAS must be {len(POINTS)} positive integers, got {spec!r}")
    return dict(zip(POINTS, (int(v) for v in values)))


class L2PrefetchConfig(NamedTuple):
    """What ``TRTLLM_K3_L2PF_V2`` prefetches: the enabled points, each point's table by tokens per
    step, and each point's CTA count."""

    points: FrozenSet[str]
    tables: Dict[str, BatchTable]
    ctas: Dict[str, int]

    @classmethod
    def from_env(
        cls,
        environ: Optional[Mapping[str, str]] = None,
        points: Optional[FrozenSet[str]] = None,
    ) -> "L2PrefetchConfig":
        """Read ``TRTLLM_K3_L2PF_V2_<POINT>`` and ``TRTLLM_K3_L2PF_V2_CTAS`` from ``environ``
        (default: the process environment); the enabled points are ``points``, or without them
        those ``TRTLLM_K3_L2PF_V2`` in ``environ`` selects."""
        env = os.environ if environ is None else environ
        tables = {
            point: batch_table(env.get(f"{ENV_VAR}_{point.upper()}", DEFAULT_TABLES[point]))
            for point in POINTS
        }
        if points is None:
            points = parse_points(env.get(ENV_VAR, "0"))
        return cls(points, tables, parse_ctas(env.get(f"{ENV_VAR}_CTAS", "")))


def tensor_ranges(tensors: Sequence[torch.Tensor]) -> List[Tuple[int, int]]:
    """``(address, bytes)`` of each dense CUDA tensor in order (16-byte aligned inwards), without
    empty, non-dense or repeated ones."""
    from ..kernels.blackwell.k3_l2_prefetch import tensor_range

    ranges = []
    seen = set()
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda or not tensor.numel():
            continue
        try:
            address, nbytes = tensor_range(tensor)
        except ValueError:
            continue
        if nbytes > 0 and address not in seen:
            seen.add(address)
            ranges.append((address, nbytes))
    return ranges


class L2PrefetchController:
    """Prefetch launches per ``(point, layer, tokens)`` on two side streams, joined once per step.

    Args:
        config: points, tables and CTA counts (:meth:`L2PrefetchConfig.from_env`).
        tensors_for: ``(point, layer_idx, num_tokens) -> tensors`` that the point prefetches for
            that layer, in the order their kernels read them. It is called outside graph capture
            only, the first time a ``(point, layer, tokens)`` is forked.
    """

    def __init__(self, config: L2PrefetchConfig, tensors_for: TensorsFn):
        self.config = config
        self._tensors_for = tensors_for
        # Whether the current step prefetches (set by begin_step).
        self.active = False
        # (point, layer, tokens) -> table (None: nothing to prefetch there).
        self._tables: Dict[Tuple[str, int, int], Optional[L2PrefetchTable]] = {}
        # Tables shared by equal (ranges, ctas, carveout), and the tensors their ranges point into.
        self._shared: Dict[tuple, L2PrefetchTable] = {}
        self._owners: Dict[tuple, Tuple[torch.Tensor, ...]] = {}
        self._events: Dict[Tuple[str, int], torch.cuda.Event] = {}
        # urgent -> stream, and the streams with launches the main stream has not joined yet.
        self._streams: Dict[bool, torch.cuda.Stream] = {}
        self._pending: Dict[bool, torch.cuda.Stream] = {}

    @property
    def pending(self) -> bool:
        """Whether a prefetch stream holds launches the current stream has not joined."""
        return bool(self._pending)

    def begin_step(self, num_tokens: int, generation_only: bool) -> None:
        """Arm the forks of a step of ``num_tokens`` tokens: only generation-only steps that run
        with multi-stream (CUDA graph warmup and capture) and that some enabled point serves."""
        self.active = (
            generation_only
            and do_multi_stream()
            and not torch.compiler.is_compiling()
            and any(
                self.config.tables[p].get(num_tokens, (0, None))[0] > 0 for p in self.config.points
            )
        )

    def fork(self, point: str, layer_idx: int, num_tokens: int) -> bool:
        """Prefetch ``point``'s weights of layer ``layer_idx`` on its side stream, after the work
        issued so far on the current stream. Returns whether a prefetch was launched."""
        if not self.active or point not in self.config.points:
            return False
        table = self._table(point, layer_idx, num_tokens)
        if table is None:
            return False
        urgent = point in URGENT_POINTS
        stream = self._stream(urgent)
        event = self._events.get((point, layer_idx))
        if event is None:
            event = self._events[(point, layer_idx)] = torch.cuda.Event()
        event.record()
        stream.wait_event(event)
        with torch.cuda.stream(stream):
            table.launch()
        self._pending[urgent] = stream
        return True

    def join(self) -> None:
        """Make the current stream wait for every prefetch launched since the last join."""
        if not self._pending:
            return
        current = torch.cuda.current_stream()
        for stream in self._pending.values():
            current.wait_stream(stream)
        self._pending.clear()

    def table_ranges(
        self, point: str, layer_idx: int, num_tokens: int
    ) -> Tuple[Tuple[int, int], ...]:
        """The ranges a fork of ``(point, layer_idx, num_tokens)`` prefetches (empty: none),
        building its table if needed."""
        table = self._table(point, layer_idx, num_tokens)
        return () if table is None else table.ranges

    def _stream(self, urgent: bool) -> torch.cuda.Stream:
        stream = self._streams.get(urgent)
        if stream is None:
            priority = torch.cuda.Stream.priority_range()[1] if urgent else 0
            stream = torch.cuda.Stream(device=torch.cuda.current_device(), priority=priority)
            self._streams[urgent] = stream
        return stream

    def _table(self, point: str, layer_idx: int, num_tokens: int) -> Optional[L2PrefetchTable]:
        key = (point, layer_idx, num_tokens)
        if key in self._tables:
            return self._tables[key]
        if torch.cuda.is_current_stream_capturing():
            # Tables are copied to the device and compiled in the warmup runs before capture.
            logger.warning_once(
                f"{ENV_VAR}: {point} of layer {layer_idx} at {num_tokens} tokens first reached "
                "under CUDA graph capture; not prefetched",
                key=f"k3_l2pf_v2_capture_{point}_{num_tokens}",
            )
            return None
        mib, carveout = self.config.tables[point].get(num_tokens, (0, None))
        table = None
        if mib > 0:
            from ..kernels.blackwell.k3_l2_prefetch_table import L2PrefetchTable, capped_ranges

            tensors = tuple(self._tensors_for(point, layer_idx, num_tokens))
            ranges = tuple(capped_ranges(tensor_ranges(tensors), mib << 20))
            if ranges:
                if carveout is None and point in URGENT_POINTS:
                    carveout = URGENT_CARVEOUT
                ctas = self.config.ctas[point]
                shared_key = (ranges, ctas, carveout)
                table = self._shared.get(shared_key)
                if table is None:
                    device = next(
                        t.device for t in tensors if isinstance(t, torch.Tensor) and t.is_cuda
                    )
                    table = L2PrefetchTable(ranges, device, ctas=ctas, carveout=carveout)
                    self._shared[shared_key] = table
                    self._owners[shared_key] = tensors
        self._tables[key] = table
        return table


__all__ = [
    "DEFAULT_CTAS",
    "DEFAULT_TABLES",
    "ENV_VAR",
    "POINTS",
    "URGENT_POINTS",
    "L2PrefetchConfig",
    "L2PrefetchController",
    "batch_table",
    "parse_ctas",
    "parse_points",
    "tensor_ranges",
]
