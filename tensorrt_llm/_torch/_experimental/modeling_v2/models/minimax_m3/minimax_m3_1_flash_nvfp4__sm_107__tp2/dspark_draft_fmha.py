# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DRAFT_FMHA_BATCH2_SPLIT: the DSpark drafter's batch-2 window decode as two per-request calls on two streams.

The drafter's block decode attends each layer's sliding window through FlashInfer's TRTLLM-Gen decode
(``trtllm_batch_decode_with_kv_cache``, q_len_per_req = draft length, HND 64-token pages). At batch 2 the library picks
a kernel whose grid is 9 x 2 x 2 = 36 CTAs, few for the device, while one request alone with the window-capped
``max_seq_len`` (window + 64 rows, rounded up to a page) runs a better-occupied kernel: the batch-2 kernel choice, not
the work, is the cost. This path therefore splits a batch-2 call into one call per request, each with the
window-capped ``max_seq_len``, request 0 on the current stream and request 1 on an aux stream forked and joined with
events (inside the CUDA graph when captured). Request 1 gets its own workspace and multi-CTA counter buffer (zeroed
once; the kernel re-zeroes them), so the two calls share no scratch.

Not bitwise against the batch-2 call (another kernel and KV split per request): drafter-only precision. The attended
keys are the window either way (the cap only sizes the tile and the KV split, as DRAFT_WINDOW_CAPPED_DECODE).

Installed on the drafter's own op set (``_dflash_trtllm_gen_ops``, a frozen dataclass: a copy with this callable as its
``batch_decode_with_kv_cache``), so no in-tree file changes. Calls it does not split run the library call unchanged,
with one logged line per reason:

- ``[M3 dispatch] DRAFT_FMHA_BATCH2_SPLIT not taken: rank R: <why>`` for the structural cases (a batch other than 2, a
  layer without a sliding window): expected on every step mix (batch 1 at least);
- ``[M3 dispatch] DRAFT_FMHA_BATCH2_SPLIT declined: rank R: <why>`` for a batch-2 window call it cannot take (an
  argument it does not forward or not at its library default, shapes, no caller counters, a capture before the aux
  buffers exist, a caller workspace that grew after a capture used the aux copies): never expected in a supported
  configuration.

The aux buffers are allocated on the first eligible call outside capture (the CUDA-graph warmup's eager forward runs
before each capture). Once a capture has recorded them they are never freed or replaced: a later call whose caller
buffers outgrow them declines; before any capture, a reallocation keeps the old buffers alive (an eager kernel on the
aux stream may still read them).

The engagement line ``[M3 dispatch] DRAFT_FMHA_BATCH2_SPLIT taken: rank R: ... under CUDA-graph capture`` is logged once
per process, at the first split recorded into a CUDA graph (the first eager split logs an ``eager`` line). It is not
evidence that a graph replayed.

Opt-in: ``TRTLLM_M31_ENABLE_DRAFT_FMHA_BATCH2_SPLIT=1``, read at import.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch

from tensorrt_llm.logger import logger

from .kernels import m3_report
from .m31_switches import m31_enabled

DFMHA_B2 = m31_enabled("DRAFT_FMHA_BATCH2_SPLIT")
#: The decode batch this path splits (the two-request step).
BATCH = 2  # capacity: fixed: the two-request decode step
#: Rows the window cap adds past the window (the in-tree window-capped decode formula,
#: modeling_dflash._decode_max_seq_len).
CAP_PAD_ROWS = 64  # capacity: fixed: the in-tree window-cap margin (block_size <= 64)

_PASS_THROUGH = (
    "bmm1_scale",
    "bmm2_scale",
    "window_left",
    "kv_layout",
    "backend",
    "bmm1_scale_log2",
)
#: Every argument the split forwards or sets per request; any other argument must hold its library default.
_FORWARDED = frozenset(
    _PASS_THROUGH
    + (
        "query",
        "kv_cache",
        "workspace_buffer",
        "block_tables",
        "seq_lens",
        "max_seq_len",
        "out",
        "enable_pdl",
        "q_len_per_req",
        "uses_shared_paged_kv_idx",
        "multi_ctas_kv_counter_buffer",
    )
)
#: The library arguments the split leaves at their defaults (the drafter passes these values or omits them).
_DEFAULTS = {
    "sinks": None,
    "max_q_len": None,
    "cum_seq_lens_q": None,
    "kv_cache_sf": None,
    "mask": None,
    "lse": None,
    "out_dtype": None,
    "o_sf_scale": None,
    "o_sf_vec_size": None,
    "skip_softmax_threshold_scale_factor": None,
    "bf16q_fp8kv_transform_mode": None,
    "return_lse": False,
    "enable_block_sparse_attention": False,
    "o_scale": 1.0,
}
#: Kinds of a call the split does not take: structural (expected) or declined (never expected).
NOT_TAKEN, DECLINED = "not taken", "declined"


def window_cap(window_left: int, page_size: int) -> int:
    """The window-capped ``max_seq_len``: the window plus 64 rows, rounded up to a page."""
    return -(-(int(window_left) + 1 + CAP_PAD_ROWS) // int(page_size)) * int(page_size)


def _page_size(kv_cache) -> Optional[int]:
    k = kv_cache[0] if isinstance(kv_cache, (tuple, list)) else None
    if not isinstance(k, torch.Tensor) or k.dim() != 4:
        return None
    return int(k.shape[-2])  # HND pages [pages, kv heads, page size, head dim]


def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


class SplitDecode:
    """``batch_decode_with_kv_cache`` with the batch-2 window decode split per request on two streams; any other call
    goes to ``inner`` (the library call) unchanged."""

    def __init__(self, inner: Callable[..., Any], rank=None) -> None:
        self.inner = inner
        self.rank = rank
        self._stream: Optional[torch.cuda.Stream] = None
        self._fork: Optional[torch.cuda.Event] = None
        self._join: Optional[torch.cuda.Event] = None
        self._ws: Optional[torch.Tensor] = None
        self._counters: Optional[torch.Tensor] = None
        # Aux buffers replaced before any capture: kept alive (an eager kernel on the aux stream may still read them).
        self._retired: List[Tuple[torch.Tensor, torch.Tensor]] = []
        # Whether a CUDA graph recorded the aux buffers: from then on they are never replaced.
        self._captured = False
        self._logged: Set[Tuple[str, str]] = set()
        self.taken = 0  # host calls taken (captures + eager calls; replays are not counted)

    # ----------------------------------------------------------------------------------------------- eligibility

    def classify(self, args: tuple, kw: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        """(kind, why) of a call that keeps the library path, None when it is split (``NOT_TAKEN``: structural;
        ``DECLINED``: a batch-2 window call it cannot take). Shapes, dtypes and host arguments only: fixed per captured
        graph and the same on every rank. Allocates the aux buffers on the first eligible call outside capture."""
        if args:
            return DECLINED, "positional arguments"
        q = kw.get("query")
        bt = kw.get("block_tables")
        seq = kw.get("seq_lens")
        out = kw.get("out")
        qlen = kw.get("q_len_per_req")
        if not isinstance(q, torch.Tensor) or not isinstance(bt, torch.Tensor):
            return DECLINED, "no query / block tables"
        if bt.dim() != 2 or int(bt.shape[0]) != BATCH:
            return NOT_TAKEN, f"batch {int(bt.shape[0]) if bt.dim() >= 1 else -1}"
        window_left = kw.get("window_left", -1)
        if not isinstance(window_left, int) or window_left < 0:
            return NOT_TAKEN, f"window_left {window_left!r} (not a sliding-window layer)"
        if not isinstance(qlen, int) or not 1 <= qlen <= CAP_PAD_ROWS:
            return DECLINED, f"q_len_per_req {qlen!r}"
        if q.dim() != 3 or int(q.shape[0]) != BATCH * qlen or not q.is_contiguous():
            return DECLINED, f"query {tuple(q.shape)} contiguous {q.is_contiguous()}"
        if not isinstance(out, torch.Tensor) or out.shape != q.shape or not out.is_contiguous():
            return DECLINED, "no contiguous out of the query's shape"
        if not isinstance(seq, torch.Tensor) or seq.dim() != 1 or int(seq.shape[0]) != BATCH:
            return DECLINED, "seq_lens is not one length per request"
        if kw.get("backend", "auto") != "trtllm-gen" or kw.get("kv_layout", "HND") != "HND":
            return DECLINED, f"backend {kw.get('backend')!r} layout {kw.get('kv_layout')!r}"
        for name, value in kw.items():
            if name in _FORWARDED:
                continue
            if name not in _DEFAULTS:
                return DECLINED, f"argument {name}"
            default = _DEFAULTS[name]
            if value is not default and not (
                isinstance(value, (bool, float))
                and type(value) is type(default)
                and value == default
            ):
                return DECLINED, f"{name} given"
        if not kw.get("uses_shared_paged_kv_idx", True):
            return DECLINED, "per-K/V page tables"
        page = _page_size(kw.get("kv_cache"))
        if page is None:
            return DECLINED, "kv_cache is not a (K, V) pair of HND page pools"
        ws = kw.get("workspace_buffer")
        counters = kw.get("multi_ctas_kv_counter_buffer")
        if not isinstance(ws, torch.Tensor) or not isinstance(counters, torch.Tensor):
            return DECLINED, "no workspace / multi-CTA counter buffer of the caller's"
        if not self._ready(ws, counters):
            if torch.cuda.is_current_stream_capturing():
                return DECLINED, "aux workspace not allocated before capture"
            if self._captured:
                return DECLINED, (
                    "the caller's workspace / counters outgrew the aux copies after a CUDA graph captured them"
                )
            self._allocate(ws, counters)
        return None

    def why_not(self, args: tuple, kw: Dict[str, Any]) -> Optional[str]:
        """Why a call keeps the library path (None: it is split)."""
        verdict = self.classify(args, kw)
        return None if verdict is None else verdict[1]

    def _ready(self, ws: torch.Tensor, counters: torch.Tensor) -> bool:
        return (
            self._ws is not None
            and self._ws.device == ws.device
            and _nbytes(self._ws) >= _nbytes(ws)
            and self._counters is not None
            and _nbytes(self._counters) >= _nbytes(counters)
        )

    def _allocate(self, ws: torch.Tensor, counters: torch.Tensor) -> None:
        """The aux stream, its events and request 1's workspace (the caller's size) and zeroed counter buffer. Only
        before any capture; buffers it replaces stay alive in ``_retired``."""
        dev = ws.device
        if self._stream is None or self._stream.device != dev:
            self._stream = torch.cuda.Stream(device=dev)
            self._fork = torch.cuda.Event()
            self._join = torch.cuda.Event()
        if self._ws is not None:
            self._retired.append((self._ws, self._counters))
        self._ws = torch.empty_like(ws)
        self._counters = torch.zeros_like(counters)

    # ------------------------------------------------------------------------------------------------------ call

    def _log(self, key: Tuple[str, str], message: str) -> None:
        if key not in self._logged:
            self._logged.add(key)
            logger.info(message)

    def __call__(self, *args, **kw):
        verdict = self.classify(args, kw)
        if verdict is not None:
            kind, why = verdict
            self._log(
                (kind, why),
                f"[M3 dispatch] DRAFT_FMHA_BATCH2_SPLIT {kind}: rank {self.rank}: {why}; one library decode call",
            )
            return self.inner(*args, **kw)
        q, out, bt, seq = kw["query"], kw["out"], kw["block_tables"], kw["seq_lens"]
        qlen = int(kw["q_len_per_req"])
        cap = min(int(kw["max_seq_len"]), window_cap(kw["window_left"], _page_size(kw["kv_cache"])))
        common = {k: kw[k] for k in _PASS_THROUGH if k in kw}
        common.update(
            kv_cache=kw["kv_cache"],
            max_seq_len=cap,
            enable_pdl=kw.get("enable_pdl"),
            q_len_per_req=qlen,
            uses_shared_paged_kv_idx=True,
        )

        def one(r: int, ws: torch.Tensor, counters: torch.Tensor) -> None:
            rows = slice(r * qlen, (r + 1) * qlen)
            self.inner(
                query=q[rows],
                workspace_buffer=ws,
                block_tables=bt[r : r + 1],
                seq_lens=seq[r : r + 1],
                out=out[rows],
                multi_ctas_kv_counter_buffer=counters,
                **common,
            )

        captured = torch.cuda.is_current_stream_capturing()
        if captured:
            self._captured = True
        main = torch.cuda.current_stream(q.device)
        side = self._stream
        self._fork.record(main)
        side.wait_event(self._fork)
        with torch.cuda.stream(side):
            one(1, self._ws, self._counters)
        one(0, kw["workspace_buffer"], kw["multi_ctas_kv_counter_buffer"])
        self._join.record(side)
        main.wait_event(self._join)
        self.taken += 1
        what = (
            f"batch-2 drafter window decode as {BATCH} per-request calls (q_len {qlen}, heads {int(q.shape[1])}, "
            f"window_left {int(kw['window_left'])}, max_seq_len {cap} each, request 1 on the aux stream, own "
            "workspace + counters)"
        )
        if captured:
            self._log(
                ("on", "captured"),
                f"[M3 dispatch] DRAFT_FMHA_BATCH2_SPLIT taken: rank {self.rank}: {what} under CUDA-graph capture",
            )
        else:
            self._log(
                ("on", "eager"),
                f"[M3 sm107 tp2] DRAFT_FMHA_BATCH2_SPLIT eager split on rank {self.rank}: {what}",
            )
        m3_report.work(
            "DRAFT_FMHA_BATCH2_SPLIT", f"{BATCH} x 1 request, q_len {qlen}, max_seq_len {cap}"
        )
        return out


def install(draft_model, rank=None) -> Tuple[bool, str]:
    """Route the drafter's batch-2 window decode through ``SplitDecode``. Returns (installed, why / what): config only
    (the switches and the drafter's backend), the same on every rank, no collective."""
    if not DFMHA_B2:
        return False, "off: default off: opt in with TRTLLM_M31_ENABLE_DRAFT_FMHA_BATCH2_SPLIT=1"
    if getattr(draft_model, "dflash_attention_backend", None) != "TRTLLM":
        return (
            False,
            f"off: drafter attention backend {getattr(draft_model, 'dflash_attention_backend', None)!r} "
            "(TRTLLM needed)",
        )
    ops = getattr(draft_model, "_dflash_trtllm_gen_ops", None)
    if ops is None or not dataclasses.is_dataclass(ops):
        return False, "off: the drafter has no TRTLLM-Gen op set"
    current = getattr(ops, "batch_decode_with_kv_cache", None)
    if isinstance(current, SplitDecode):
        return True, "on (already installed)"
    if current is None:
        return False, "off: the op set has no batch decode"
    draft_model._dflash_trtllm_gen_ops = dataclasses.replace(
        ops, batch_decode_with_kv_cache=SplitDecode(current, rank)
    )
    return (
        True,
        f"on: batch-{BATCH} window decode as per-request calls (max_seq_len = window + {CAP_PAD_ROWS} rows, page "
        "rounded) on two streams",
    )


def installed(draft_model) -> Optional[SplitDecode]:
    ops = getattr(draft_model, "_dflash_trtllm_gen_ops", None)
    fn = getattr(ops, "batch_decode_with_kv_cache", None)
    return fn if isinstance(fn, SplitDecode) else None


__all__ = [
    "BATCH",
    "CAP_PAD_ROWS",
    "DECLINED",
    "DFMHA_B2",
    "NOT_TAKEN",
    "SplitDecode",
    "install",
    "installed",
    "window_cap",
]
