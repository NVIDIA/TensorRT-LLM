# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Standalone benchmark for replay_selective_state_update (Triton kernel).

Suitable for nsight-compute (ncu) and nsight-systems (nsys) capture.

Fixed model config: NVIDIA-Nemotron-3-Super-120B-A12B at TP=8
  nheads=16, head_dim=64, d_state=128, ngroups=1

mtp_len is the per-request sequence length processed by replay: in MTP it
equals num_draft_tokens + 1 target token, so --mtp-lengths 6 models 5 drafts
+ 1 target.

Baseline kernel (--baseline [triton|flashinfer]):
  Calls selective_state_update with T=mtp_len tokens and disable_state_update=True,
  matching the MTP scoring pass in mamba2_mixer.py exactly.

Timing methodology
==================

All in-bench timing comes from CUPTI's Activity API (1 ns kernel
timestamps from the GPU profiling fabric).  cudaEvent.elapsed_time() was
removed — its ~0.5 us resolution overshoots CUPTI by ~50% on short kernels
in graphs, and we have no other use for it here.  See the CUPTI block
lower in this file for the timer source.

Three modes:

  --cupti --cuda-graph (default)
      Capture one CUDA graph per cell (warmup + timed iters inlined),
      replay once, read kernel start/end from CUPTI.  ~20× faster than
      nsys-wrapped capture and matches it to within ~1% / noise floor.

  --cupti --no-cuda-graph
      Eager loop with CUPTI.  Per-kernel timestamps are still accurate, but
      the per-iter SPAN (max(end) - min(start)) now includes the Python
      launch latency BETWEEN consecutive kernels in run_fn (~100 µs on
      Hopper/Blackwell).  Graph capture and PDL hide that latency; eager
      mode honestly reports it.  For per-kernel timing in eager mode, look
      at per_kernel.start_us/end_us in --json-detailed output rather than
      the span percentiles.  Useful when graph capture is undesirable.

  --no-cupti  (with or without --cuda-graph)
      No in-bench timing — just runs the kernels for an external profiler
      (nsys / ncu) to time.  In-process CUPTI conflicts with nsys's own
      subscriber, so disable ours when wrapping in nsys.  Bench output
      reports zeros for median/p95/p99; trust the external trace.

JSON output schema (--json-output PATH)
=======================================

Designed to be parsed by collect.py / report.py without touching sqlite or
NVTX traces.  Future agents: prefer reading this JSON over re-running nsys.

  {
    "metadata": {timestamp, cmd, tp_size, warmup, iters, variant, cupti},
    "results": {
      "<key>": {median, p95, p99, n, [iters_us], [per_kernel]}
    }
  }

Key format mirrors collect.py's kernel_data.json convention:
  incremental/{batch}/{mtp}/{sd}/k{prev_k}/{sweep_parts}/tp{tp}
  triton/{batch}/{mtp}/{sd}/tp{tp}
  flashinfer/{batch}/{mtp}/{sd}/tp{tp}

  - <sd> is normalized: bf16 / fp16 / fp32 / int8 / int16 / fp8.
  - <sweep_parts> is e.g. "M16_W1_S3_SR0_RECT0_WC1" — flags concatenated by
    underscore in canonical (M, W, S, pW, pS, H, R, CT, SR, RECT, WC) order.
  - All numeric values in microseconds (us).

Per-record fields:
  - median, p95, p99: span statistics (us).  Span = max(kernel_end_ns) -
    min(kernel_start_ns) across the iter's kernels — same convention as
    nsys-derived collect.py used to use.
  - n: number of timed iters that contributed.
  - iters_us: list of length n, raw per-iter spans (only with --json-detailed).
  - per_kernel: {<kernel_name>: {start_us: [...], end_us: [...]}} where
    timestamps are RELATIVE to that iter's first kernel start, in us.  Lets
    you see PDL overlap directly without an external profiler.  Only with
    --json-detailed.

Example usage:
  # Basic sweep (default = --cupti, just summary stats)
  python benchmark_replay_selective_state_update.py \\
      --batch-sizes 1,2,4 --mtp-lengths 1,4,8 --warmup 5 --iters 20

  # JSON output, summary stats only (compact)
  python benchmark_replay_selective_state_update.py \\
      --batch-sizes 16 --mtp-lengths 6 --json-output /tmp/out.json

  # JSON output, full per-iter / per-kernel data (for PDL analysis etc.)
  python benchmark_replay_selective_state_update.py \\
      --batch-sizes 16 --mtp-lengths 6 \\
      --json-output /tmp/out.json --json-detailed

  # nsys capture (--no-cupti so our subscriber doesn't conflict)
  nsys profile --capture-range=cudaProfilerApi \\
      python benchmark_replay_selective_state_update.py --profile --no-cupti

  # ncu capture (--no-cupti --no-cuda-graph: each kernel replayable solo)
  ncu --target-processes all \\
      python benchmark_replay_selective_state_update.py --profile \\
          --no-cupti --no-cuda-graph \\
          --batch-sizes 1 --mtp-lengths 4 --warmup 5 --iters 5
"""

import argparse
import importlib
import itertools
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from einops import repeat


def _import_mamba_kernels_fast():
    """Load kernel modules directly (~40s faster than a full tensorrt_llm init).
    Use --full-import as the fallback if module dependencies change.

    Strategy: stub the parent packages (tensorrt_llm, tensorrt_llm._torch,
    tensorrt_llm._torch.modules) in sys.modules with __path__ set, but do
    NOT execute their __init__.py.  Then load the leaf kernel modules.
    When a kernel body imports e.g. tensorrt_llm._utils.get_sm_version,
    Python's machinery resolves it against our stub's __path__ and loads
    only _utils.py — skipping the heavy tensorrt_llm package init.
    """
    import types

    repo_root = Path(__file__).resolve().parents[5]
    trtllm_dir = repo_root / "tensorrt_llm"
    mamba_pkg = "tensorrt_llm._torch.modules.mamba"
    mamba_dir = trtllm_dir / "_torch" / "modules" / "mamba"

    def _stub_pkg(fqn: str, pkg_dir: Path):
        """Register a stub package in sys.modules without running its
        __init__.py.  Sets __path__ so Python can resolve submodule imports
        against the real directory on disk."""
        if fqn in sys.modules:
            return
        stub = types.ModuleType(fqn)
        stub.__path__ = [str(pkg_dir)]
        sys.modules[fqn] = stub

    # Stub the parent chain so `from tensorrt_llm._utils import ...` (and
    # similar) work without triggering tensorrt_llm/__init__.py.
    _stub_pkg("tensorrt_llm", trtllm_dir)
    _stub_pkg("tensorrt_llm._torch", trtllm_dir / "_torch")
    _stub_pkg("tensorrt_llm._torch.modules", trtllm_dir / "_torch" / "modules")

    def _load(mod_name: str, file_name: str):
        fqn = f"{mamba_pkg}.{mod_name}" if mod_name else mamba_pkg
        if fqn in sys.modules:
            return sys.modules[fqn]
        spec = importlib.util.spec_from_file_location(
            fqn,
            mamba_dir / file_name,
            submodule_search_locations=[str(mamba_dir)] if file_name == "__init__.py" else [],
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[fqn] = mod
        spec.loader.exec_module(mod)
        return mod

    # 1. Package __init__ (defines PAD_SLOT_ID = -1)
    _load("", "__init__.py")
    # 2. softplus helper (used by both kernel modules)
    _load("softplus", "softplus.py")
    # 3. The actual kernels
    replay_mod = _load("replay_selective_state_update", "replay_selective_state_update.py")
    checkpoint_mod = _load("checkpointing_state_update", "checkpointing_state_update.py")
    base_mod = _load("selective_state_update", "selective_state_update.py")
    conv1d_mod = _load("causal_conv1d_triton", "causal_conv1d_triton.py")

    return (
        replay_mod.replay_selective_state_update,
        checkpoint_mod.checkpointing_state_update,
        base_mod.selective_state_update,
        conv1d_mod.causal_conv1d_update,
    )


def _import_mamba_kernels_full():
    """Import via the standard tensorrt_llm package (slow but safe)."""
    from tensorrt_llm._torch.modules.mamba.causal_conv1d_triton import causal_conv1d_update
    from tensorrt_llm._torch.modules.mamba.checkpointing_state_update import (
        checkpointing_state_update,
    )
    from tensorrt_llm._torch.modules.mamba.replay_selective_state_update import (
        replay_selective_state_update,
    )
    from tensorrt_llm._torch.modules.mamba.selective_state_update import selective_state_update

    return (
        replay_selective_state_update,
        checkpointing_state_update,
        selective_state_update,
        causal_conv1d_update,
    )


# Use fast import by default; --full-import parsed later but we need the
# functions at module level.  Check sys.argv early.
if "--full-import" in sys.argv:
    (
        replay_selective_state_update,
        checkpointing_state_update,
        selective_state_update,
        causal_conv1d_update,
    ) = _import_mamba_kernels_full()
else:
    try:
        (
            replay_selective_state_update,
            checkpointing_state_update,
            selective_state_update,
            causal_conv1d_update,
        ) = _import_mamba_kernels_fast()
    except Exception as e:  # noqa: BLE001 - exit loudly; don't hide a fast-import regression
        print(
            f"ERROR: fast import failed ({type(e).__name__}: {e})\n"
            "Re-run with --full-import for the slow but stable path, "
            "then file a bug or fix _import_mamba_kernels_fast.",
            file=sys.stderr,
        )
        sys.exit(1)


_VARIANT_FNS = {
    "replay": lambda: replay_selective_state_update,
    "checkpointing": lambda: checkpointing_state_update,
}

# Model config defaults (Nemotron-3-Super-120B full model).
# --tp-size divides nheads and ngroups to get the per-GPU slice.
#   TP=1: nheads=128, ngroups=8
#   TP=4: nheads=32,  ngroups=2
#   TP=8: nheads=16,  ngroups=1  (default)
NHEADS = 128
HEAD_DIM = 64
D_STATE = 128
NGROUPS = 8
TP_SIZE = 8  # default; overridden by --tp-size

# L2 flush buffer: ~128 MB — larger than L2 on A100/H100/B200
_L2_FLUSH_SIZE = 32 * 1024 * 1024  # float32 elements → 128 MB
_l2_flush: torch.Tensor | None = None


def _init_l2_flush() -> None:
    global _l2_flush
    _l2_flush = torch.empty(_L2_FLUSH_SIZE, dtype=torch.float32, device="cuda")


def _flush_l2() -> None:
    """Evict L2 by writing to a large buffer then synchronising."""
    assert _l2_flush is not None
    _l2_flush.fill_(0.0)
    torch.cuda.synchronize()


def _resolve_prev_ks(args, mtp_len: int) -> list[int]:
    """Resolve prev_k values for one mtp_len cell.

    Two input modes (mutually exclusive in spirit; absolute wins if both given):
      --prev-tokens-int "0,10,11,16"  → use literal integers, clamped to
        [0, max_window] (where max_window is the cache T-axis capacity).
      --prev-tokens-fracs "0,0.5,1.0" → fractions of mtp_len, clamped to
        [0, mtp_len] (current behavior).

    For replay-style checkpointing the cache holds up to max_window old
    tokens, so absolute integers are the right knob.  Fractions are kept
    for back-compat with prior placeholder runs.
    """
    upper = getattr(args, "max_window", 0) or mtp_len
    if getattr(args, "prev_tokens_int", None):
        return sorted(set(max(0, min(upper, int(v))) for v in args.prev_tokens_int))
    return sorted(
        set(min(mtp_len, max(0, round(f * mtp_len))) for f in args.prev_tokens_fracs)
    )


# Tensor construction helpers


def _build_tensors(
    batch: int,
    mtp_len: int,
    state_dtype: torch.dtype,
    act_dtype: torch.dtype,
    nheads: int,
    head_dim: int,
    d_state: int,
    ngroups: int,
    max_window: int | None = None,
):
    """
    Build all tensors for one benchmark configuration.

    nheads/ngroups are already TP-split (i.e. full_nheads // tp_size).

    Returns:
      state0                   : (batch, nheads, head_dim, d_state) – initial SSM state
      x, dt, B, C              : (batch, mtp_len, ...) – token inputs for both kernels
      A, dt_bias, D            : SSM parameters (float32, tie_hdim strides)
      prev_tokens              : (batch,)
      out_incr                 : pre-allocated output for replay kernel (batch, mtp_len, nheads, head_dim)
      out_base                 : pre-allocated output for baseline kernel   (batch, mtp_len, nheads, head_dim)
      intermediate_states_buffer: for baseline kernel (batch, mtp_len, nheads, head_dim, d_state)
    """
    device = "cuda"

    torch.manual_seed(42)

    # --- SSM parameters (float32, tie_hdim strides) ---
    A_base = -torch.rand(nheads, device=device) - 0.5
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)  # stride(-1)=0, stride(-2)=0

    dt_bias_base = torch.randn(nheads, device=device, dtype=torch.float32)
    dt_bias = repeat(dt_bias_base, "h -> h p", p=head_dim)  # stride(-1)=0

    D_base = torch.randn(nheads, device=device, dtype=torch.float32)
    D = repeat(D_base, "h -> h p", p=head_dim)

    # --- SSM state ---
    # Quantized dtypes need their own initializer (torch.randn doesn't accept
    # int) and a parallel fp32 scales tensor (per-(head, dim) channel decode
    # scale, broadcast over dstate).  Quant state is filled with realistic-
    # range values via fp32 → quant; scales are derived consistently so the
    # initial state isn't garbage on dequant.
    _QUANT_BENCH = {
        torch.int8: 127.0,
        torch.int16: 32767.0,
        torch.float8_e4m3fn: 448.0,
    }
    if state_dtype in _QUANT_BENCH:
        quant_max = _QUANT_BENCH[state_dtype]
        state_fp32 = torch.randn(
            batch, nheads, head_dim, d_state, device=device, dtype=torch.float32
        )
        amax = state_fp32.abs().amax(dim=-1)  # (batch, nheads, head_dim)
        encode_scale = quant_max / amax.clamp(min=1e-30)
        state_scales0 = (1.0 / encode_scale).to(torch.float32)  # decode scale
        scaled = state_fp32 * encode_scale.unsqueeze(-1)
        if state_dtype == torch.float8_e4m3fn:
            state0 = scaled.clamp(-quant_max, quant_max).to(state_dtype)
        else:
            state0 = scaled.round().clamp(-quant_max, quant_max).to(state_dtype)
    else:
        state0 = torch.randn(
            batch, nheads, head_dim, d_state, device=device, dtype=state_dtype
        )
        state_scales0 = None

    # --- Cache tensors for replay kernel ---
    # max_window is the cache T-axis capacity; defaults to mtp_len (the
    # placeholder/degenerate case where every step is a checkpoint step).
    # For real replay-style checkpointing, max_window > mtp_len.
    cache_T = max_window if max_window is not None else mtp_len
    # old_x: single-buffered (cache, max_window, nheads, dim)
    old_x = torch.randn(batch, cache_T, nheads, head_dim, device=device, dtype=act_dtype)
    # old_B: double-buffered (cache, 2, max_window, ngroups, dstate)
    old_B = torch.randn(batch, 2, cache_T, ngroups, d_state, device=device, dtype=act_dtype)
    # old_dt: double-buffered (cache, 2, nheads, max_window) fp32 — T contiguous
    old_dt = torch.randn(batch, 2, nheads, cache_T, device=device, dtype=torch.float32)
    # old_dA_cumsum: double-buffered (cache, 2, nheads, max_window) fp32 — T contiguous
    old_dA_cumsum = torch.randn(batch, 2, nheads, cache_T, device=device, dtype=torch.float32)
    # cache_buf_idx: which buffer to read (0 or 1)
    cache_buf_idx = torch.zeros(batch, device=device, dtype=torch.int32)

    # --- Token inputs (used by both replay and baseline kernels) ---
    x = torch.randn(batch, mtp_len, nheads, head_dim, device=device, dtype=act_dtype)
    # dt must match D's dtype (fp32) for flashinfer — force it for all paths.
    dt_base = torch.randn(batch, mtp_len, nheads, device=device, dtype=torch.float32)
    dt = repeat(dt_base, "b t h -> b t h p", p=head_dim)  # tie_hdim
    B = torch.randn(batch, mtp_len, ngroups, d_state, device=device, dtype=act_dtype)
    C = torch.randn(batch, mtp_len, ngroups, d_state, device=device, dtype=act_dtype)

    # prev_tokens placeholder — overwritten per-run
    prev_tokens = torch.zeros(batch, device=device, dtype=torch.int32)

    out_incr = torch.zeros(batch, mtp_len, nheads, head_dim, device=device, dtype=act_dtype)
    out_base = torch.zeros(batch, mtp_len, nheads, head_dim, device=device, dtype=act_dtype)

    # intermediate_states_buffer is only consumed by the fp/baseline path;
    # for quantized state dtypes we'll skip baselines entirely, so the buffer
    # dtype falls back to fp32 to keep selective_state_update happy.
    int_buffer_dtype = state_dtype if state_dtype not in _QUANT_BENCH else torch.float32
    intermediate_states_buffer = torch.zeros(
        batch, mtp_len, nheads, head_dim, d_state, device=device, dtype=int_buffer_dtype
    )

    # --- Conv1d tensors (for --with-conv1d mode) ---
    d_inner = nheads * head_dim
    conv_dim = d_inner + 2 * ngroups * d_state
    d_conv = 4  # conv kernel width for Nemotron/Mamba2

    # xbc_input: (batch, conv_dim, mtp_len) — "hot" input from in_proj.
    # Match production layout: in_proj output is (batch*mtp_len, conv_dim)
    # contiguous, then .view(batch, mtp_len, conv_dim).transpose(1, 2)
    # gives strides (mtp_len*conv_dim, 1, conv_dim) — NOT the standard
    # (conv_dim*mtp_len, mtp_len, 1) of a freshly allocated 3D tensor.
    # Conv1d preserves input strides in its output, so downstream split
    # + view inherits the correct layout without needing .contiguous().
    xbc_input_flat = torch.randn(batch * mtp_len, conv_dim, device=device, dtype=act_dtype)
    xbc_input = xbc_input_flat.view(batch, mtp_len, conv_dim).transpose(1, 2)
    # conv_state: (batch, conv_dim, d_conv) — "cold" cache
    conv_state = torch.randn(batch, conv_dim, d_conv, device=device, dtype=act_dtype)
    # conv_weight: (conv_dim, d_conv) — parameter
    conv_weight = torch.randn(conv_dim, d_conv, device=device, dtype=act_dtype)
    # conv_bias: (conv_dim,) — parameter
    conv_bias = torch.randn(conv_dim, device=device, dtype=act_dtype)

    return (
        state0,
        state_scales0,
        old_x,
        old_B,
        old_dt,
        old_dA_cumsum,
        cache_buf_idx,
        x,
        dt,
        B,
        C,
        A,
        dt_bias,
        D,
        prev_tokens,
        out_incr,
        out_base,
        intermediate_states_buffer,
        xbc_input,
        conv_state,
        conv_weight,
        conv_bias,
        d_inner,
        conv_dim,
    )


# =============================================================================
# CUPTI in-process kernel timing
#
# Self-contained module-in-a-file.  Reads kernel start/end timestamps directly
# from the GPU profiling fabric via NVIDIA's cupti-python bindings (1 ns
# resolution), avoiding two pitfalls of the cuda-events path:
#
#   1. cudaEvent.elapsed_time() resolution (~0.5 us) is too coarse for the
#      short kernels we care about, especially with PDL + cuda graphs at
#      small batch — events recorded inside a graph have proven noisy.
#   2. nsys is the only known accurate alternative, but the
#      profile-export-sqlite-parse pipeline is heavy and out-of-process.
#
# This is functionally equivalent to wrapping each cell in nsys, except it
# runs in the same Python process with no serialization.  When this proves
# out, lift `CuptiKernelTimer` and `_time_kernel_cuda_graph_cupti` into a
# proper TRT-LLM utility module — there is no benchmark-specific code below.
# =============================================================================


# Substring match: kernels run_fn launches that we want to time.  Mirrors
# the parser in scripts/.../collect.py so cupti and nsys-based outputs agree.
_CUPTI_KEEP_KERNEL_SUBSTRINGS = (
    "_replay_precompute",
    "_checkpointing_precompute",
    "_rectangle_precompute",
    "_dynamic_precompute",
    "_replay_state_update",
    "_checkpointing_main",
    "_rectangle_main",
    "_dynamic_main",
    "selective_scan_update",
    "selective_state_update",
    "causal_conv1d_update",
)


class CuptiKernelTimer:
    """Process-singleton wrapper around CUPTI's CONCURRENT_KERNEL activity.

    CUPTI's callbacks are global (one subscriber per process), so the timer
    is constructed lazily once via `CuptiKernelTimer.get()`.  cupti-python
    parses the activity buffer for us — `buffer_completed` receives a Python
    list of typed activity objects, not a raw byte buffer — so no FFI is
    needed.

    Usage:
        timer = CuptiKernelTimer.get()
        timer.start()                     # arms; drops any stale records
        <run cuda graph and synchronize>
        records = timer.stop()            # flush; list of tuples per kernel
                                           # (name, start_ns, end_ns, corr,
                                           #  graph_id, graph_node_id, stream)

    The callback fires from a CUPTI worker thread, so a lock guards the
    record buffer.  Records are kept tiny (tuple of ints + str) to minimize
    Python overhead in the hot path of the callback.
    """

    _instance = None
    _import_error = None

    @classmethod
    def get(cls) -> "CuptiKernelTimer":
        if cls._instance is not None:
            return cls._instance
        if cls._import_error is not None:
            raise cls._import_error
        try:
            from cupti import cupti as _c
        except ImportError as e:  # pragma: no cover — env-dependent
            cls._import_error = e
            raise
        cls._instance = cls._init(_c)
        return cls._instance

    @classmethod
    def _init(cls, _c) -> "CuptiKernelTimer":
        import threading

        self = object.__new__(cls)
        self._c = _c
        self._records: list[tuple] = []
        self._lock = threading.Lock()

        # CUPTI callback contract (from cupti-python-samples/cupti_common.py):
        #   buffer_requested() -> (buffer_size, max_num_records)
        #   buffer_completed(activities: list)
        # Setting max_num_records=0 (unbounded) avoids spurious buffer
        # requests.  8 MiB matches the sample defaults.
        def _buf_req():
            return (8 * 1024 * 1024, 0)

        kernel_kinds = (_c.ActivityKind.CONCURRENT_KERNEL, _c.ActivityKind.KERNEL)

        def _buf_done(activities):
            recs = []
            for a in activities:
                if a.kind not in kernel_kinds:
                    continue
                # start/end == 0 means CUPTI couldn't time this kernel.
                if a.start == 0 or a.end == 0:
                    continue
                recs.append((
                    a.name,
                    int(a.start),
                    int(a.end),
                    int(a.correlation_id),
                    int(a.graph_id),
                    int(a.graph_node_id),
                    int(a.stream_id),
                ))
            if recs:
                with self._lock:
                    self._records.extend(recs)

        # Hold strong refs so the C side never sees GC'd Python callables.
        self._buf_req = _buf_req
        self._buf_done = _buf_done

        _c.activity_register_callbacks(_buf_req, _buf_done)
        _c.activity_enable(_c.ActivityKind.CONCURRENT_KERNEL)
        return self

    def start(self) -> None:
        """Arm capture: flush any stale records, then clear the buffer."""
        self._c.activity_flush_all(1)
        with self._lock:
            self._records.clear()

    def stop(self) -> list[tuple]:
        """Flush and return all kernel records since the last start()."""
        self._c.activity_flush_all(1)
        with self._lock:
            return list(self._records)


# =============================================================================
# Timing helpers
# =============================================================================


def _stats_from_spans(spans_us: list[float]) -> dict:
    """Compute median / p95 / p99 / n from a per-iter span list."""
    s = sorted(spans_us)
    return {
        "median": statistics.median(s),
        "p95": s[int(0.95 * len(s))],
        "p99": s[int(0.99 * len(s))],
        "n": len(s),
    }


def _stats_from_cupti_records(records, warmup, iters, tag):
    """Bin a flat CUPTI kernel record stream into per-iter spans + per-kernel
    relative timestamps.  Used by both graph and eager CUPTI paths.

    `records` are tuples (name, start_ns, end_ns, ...) — see CuptiKernelTimer.
    The first warmup*K records are dropped; the rest are chunked into K-tuples.
    """
    records = [
        r for r in records
        if any(s in r[0] for s in _CUPTI_KEEP_KERNEL_SUBSTRINGS)
    ]
    records.sort(key=lambda r: r[1])  # by start_ns

    total = len(records)
    expected_iters = warmup + iters
    if total == 0 or total % expected_iters != 0:
        names_seen = sorted({r[0] for r in records})
        raise RuntimeError(
            f"CUPTI capture mismatch for {tag!r}: got {total} records, "
            f"expected a multiple of {expected_iters} (warmup+iters={expected_iters}). "
            f"Kernel names captured: {names_seen}"
        )
    K = total // expected_iters
    timed = records[warmup * K:]

    spans_us: list[float] = []
    per_kernel: dict[str, dict[str, list[float]]] = {}
    for i in range(iters):
        chunk = timed[i * K:(i + 1) * K]
        iter_start_ns = min(r[1] for r in chunk)
        iter_end_ns = max(r[2] for r in chunk)
        spans_us.append((iter_end_ns - iter_start_ns) / 1000.0)
        for r in chunk:
            name = r[0]
            slot = per_kernel.setdefault(name, {"start_us": [], "end_us": []})
            slot["start_us"].append((r[1] - iter_start_ns) / 1000.0)
            slot["end_us"].append((r[2] - iter_start_ns) / 1000.0)

    out = _stats_from_spans(spans_us)
    out["iters_us"] = spans_us
    out["per_kernel"] = per_kernel
    return out


def _time_kernel_cuda_graph(
    args,
    run_fn,
    reset_fn,
    tag: str,
) -> dict:
    """CUDA-graph CUPTI timer.

    Captures one CUDA graph (warmup + timed iters inlined), replays once,
    reads kernel start/end timestamps from CUPTI (1 ns resolution).
    """
    timer = CuptiKernelTimer.get()
    warmup = args.warmup
    iters = args.iters

    # Eager warmup before graph capture (triggers Triton autotune if active).
    reset_fn()
    run_fn()
    torch.cuda.synchronize()

    reset_fn()
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(warmup + iters):
            reset_fn()
            if args.l2_flush:
                _l2_flush.fill_(0.0)
            run_fn()

    torch.cuda.synchronize()

    timer.start()
    torch.cuda.nvtx.range_push(tag)
    g.replay()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    records = timer.stop()

    return _stats_from_cupti_records(records, warmup, iters, tag)


def _time_kernel_eager(
    args,
    run_fn,
    reset_fn,
    tag: str,
) -> dict:
    """Non-graph CUPTI timer (for ncu wrapping, debugging, etc.).

    Each iter runs serially with sync between, but kernel start/end still
    come from CUPTI — same accuracy as the graph path, just slower per-iter
    (extra Python + sync overhead).
    """
    timer = CuptiKernelTimer.get()
    warmup = args.warmup
    iters = args.iters

    timer.start()
    torch.cuda.nvtx.range_push(tag)
    for _ in range(warmup + iters):
        reset_fn()
        if args.l2_flush:
            _flush_l2()  # includes synchronize
        run_fn()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    records = timer.stop()

    return _stats_from_cupti_records(records, warmup, iters, tag)


def _run_kernel_untimed(args, run_fn, reset_fn, tag: str) -> dict:
    """No in-bench timing: just run the kernels for an external profiler
    (nsys / ncu) to time externally.  Returns a stats dict full of zeros so
    downstream code (table, JSON) doesn't break.
    """
    warmup = args.warmup
    iters = args.iters

    if args.cuda_graph:
        # Eager warmup before capture (Triton autotune)
        reset_fn(); run_fn(); torch.cuda.synchronize()
        reset_fn(); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(warmup + iters):
                reset_fn()
                if args.l2_flush:
                    _l2_flush.fill_(0.0)
                run_fn()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(tag)
        g.replay()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    else:
        torch.cuda.nvtx.range_push(tag)
        for _ in range(warmup + iters):
            reset_fn()
            if args.l2_flush:
                _flush_l2()
            run_fn()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

    spans_us = [0.0] * iters
    out = _stats_from_spans(spans_us)
    out["iters_us"] = spans_us
    out["per_kernel"] = {}
    return out


def _time_kernel(args, run_fn, reset_fn, tag: str) -> dict:
    """Dispatch to graph-CUPTI / eager-CUPTI / no-timer path.

    --cupti: in-process CUPTI Activity API timing (default).  Use --no-cupti
    when running under nsys (in-process CUPTI conflicts with nsys's own
    subscriber); the bench then runs the kernels for nsys to time externally.
    """
    if not getattr(args, "cupti", True):
        return _run_kernel_untimed(args, run_fn, reset_fn, tag)
    if args.cuda_graph:
        return _time_kernel_cuda_graph(args, run_fn, reset_fn, tag)
    return _time_kernel_eager(args, run_fn, reset_fn, tag)


# Per-config benchmark (consolidated baseline + replay)


def _compile_warmup_phase(args, batch_sizes, mtp_lengths, state_dtypes, act_dtypes,
                          baseline_fn, max_workers: int) -> None:
    """Run each config once in parallel to compile + cache Triton kernels.

    Triton's ``compile()`` releases the GIL, so a ThreadPoolExecutor
    fans out shape compilations across CPU cores in one shared CUDA
    context.  Compiled binaries land in Triton's on-disk cache (default
    ``~/.triton/cache``) and the subsequent sequential measurement
    phase loads them with no compile cost.

    Parallel measurement would race for GPU time and skew numbers, so
    only the warmup is parallelized; timing stays serial.
    """
    from concurrent.futures import ThreadPoolExecutor

    sr_modes_list = getattr(args, "sr_modes_list", ["RN"])

    rect_list = getattr(args, "rectangle_for_nowrite_list", [False])
    write_modes_list = getattr(args, "write_modes_list", [args.write_checkpoint])

    configs = []
    for batch in batch_sizes:
        for mtp_len in mtp_lengths:
            prev_ks = _resolve_prev_ks(args, mtp_len)
            for state_dtype in state_dtypes:
                for act_dtype in act_dtypes:
                    for sr_mode in sr_modes_list:
                        for write_ckpt in write_modes_list:
                            # Rectangle is only meaningful for nowrite cells.
                            effective_rect_list = (
                                [False] if write_ckpt else rect_list
                            )
                            for rect in effective_rect_list:
                                configs.append((
                                    batch, mtp_len, prev_ks, state_dtype, act_dtype,
                                    sr_mode, rect, write_ckpt,
                                ))

    print(f"[compile-warmup] {len(configs)} configs across {max_workers} threads")
    t0 = time.perf_counter()

    def _warm(cfg):
        batch, mtp_len, prev_ks, state_dtype, act_dtype, sr_mode, rect, write_ckpt = cfg
        _bench_config(
            args, batch, mtp_len, prev_ks, state_dtype, act_dtype, baseline_fn,
            sr_mode=sr_mode, rectangle_for_nowrite=rect,
            write_checkpoint=write_ckpt, warmup_only=True,
        )

    errors = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_warm, cfg) for cfg in configs]
        for cfg, fut in zip(configs, futures):
            try:
                fut.result()
            except Exception as e:
                errors.append((cfg, e))

    if errors:
        for cfg, e in errors:
            print(f"[compile-warmup] FAILED config {cfg}: {type(e).__name__}: {e}",
                  file=sys.stderr)
        raise errors[0][1]

    print(f"[compile-warmup] done in {time.perf_counter() - t0:.1f}s")


def _bench_config(
    args,
    batch: int,
    mtp_len: int,
    prev_ks: list[int],
    state_dtype: torch.dtype,
    act_dtype: torch.dtype,
    baseline_fn,
    sr_mode: str = "RN",
    rectangle_for_nowrite: bool = False,
    write_checkpoint: bool = True,
    warmup_only: bool = False,
) -> None:
    """
    Benchmark one (batch, mtp_len, dtype) configuration.

    Runs the baseline kernel (if baseline_fn is not None) followed by the
    replay kernel for each prev_k value.  Tensors are built once and
    shared across all runs in this config.

    When ``warmup_only`` is True, calls each kernel exactly once instead of
    timing it.  Used by the parallel-warmup phase to populate Triton's
    persistent compile cache across all configs concurrently.  No timing
    output is produced.
    """
    state_dtype_name = str(state_dtype).split(".")[-1]
    act_dtype_name = str(act_dtype).split(".")[-1]

    (
        state0,
        state_scales0,
        old_x0,
        old_B0,
        old_dt0,
        old_dA_cumsum0,
        cache_buf_idx0,
        x,
        dt,
        B,
        C,
        A,
        dt_bias,
        D,
        prev_tokens,
        out_incr,
        out_base,
        intermediate_states_buffer,
        xbc_input0,
        conv_state0,
        conv_weight,
        conv_bias,
        d_inner,
        conv_dim,
    ) = _build_tensors(
        batch,
        mtp_len,
        state_dtype,
        act_dtype,
        args.tp_nheads,
        args.head_dim,
        args.d_state,
        args.tp_ngroups,
        max_window=getattr(args, "max_window", None) or None,
    )

    nheads = args.tp_nheads
    ngroups = args.tp_ngroups
    head_dim = args.head_dim
    d_state = args.d_state
    with_conv1d = getattr(args, "with_conv1d", False)
    use_philox = (sr_mode == "SR")
    variant_fn = _VARIANT_FNS[args.variant]()

    # SR rounding: allow fp16 and the quantized dtypes (int8/int16/fp8).
    # bf16/fp32 SR is not supported (no PTX path for bf16; fp32 doesn't need
    # rounding).  When sweeping --sr-modes RN,SR over a mixed dtype set,
    # silently skip the SR cell for unsupported dtypes — the RN cell still
    # prints, and other dtypes still get their SR row.
    rand_seed = None
    _SR_SUPPORTED = (
        torch.float16, torch.int8, torch.int16, torch.float8_e4m3fn,
    )
    if use_philox:
        if state_dtype not in _SR_SUPPORTED:
            return
        rand_seed = torch.randint(0, 2**62, (1,), device="cuda", dtype=torch.int64)

    is_quantized = state_dtype in (torch.int8, torch.int16, torch.float8_e4m3fn)

    state_work = state0.clone()
    state_scales_work = state_scales0.clone() if state_scales0 is not None else None
    old_x_work = old_x0.clone()
    old_B_work = old_B0.clone()
    old_dt_work = old_dt0.clone()
    old_dA_cumsum_work = old_dA_cumsum0.clone()
    cache_buf_idx_work = cache_buf_idx0.clone()
    xbc_input_work = xbc_input0.clone()
    conv_state_work = conv_state0.clone()

    def _reset():
        state_work.copy_(state0)
        if state_scales_work is not None:
            state_scales_work.copy_(state_scales0)
        old_x_work.copy_(old_x0)
        old_B_work.copy_(old_B0)
        old_dt_work.copy_(old_dt0)
        old_dA_cumsum_work.copy_(old_dA_cumsum0)
        cache_buf_idx_work.copy_(cache_buf_idx0)
        if with_conv1d:
            conv_state_work.copy_(conv_state0)

    def _reset_conv1d_realistic():
        """Realistic reset: cold cache, L2 flush, then hot in_proj output."""
        # 1. Reset cold state (cache tensors, SSM state)
        state_work.copy_(state0)
        if state_scales_work is not None:
            state_scales_work.copy_(state_scales0)
        old_x_work.copy_(old_x0)
        old_B_work.copy_(old_B0)
        old_dt_work.copy_(old_dt0)
        old_dA_cumsum_work.copy_(old_dA_cumsum0)
        cache_buf_idx_work.copy_(cache_buf_idx0)
        conv_state_work.copy_(conv_state0)
        # 2. L2 flush (evicts cold state from cache)
        if _l2_flush is not None:
            _l2_flush.fill_(0.0)
        # 3. Write hot tensors (simulates in_proj output landing in L2)
        xbc_input_work.copy_(xbc_input0)

    # Silently skip the baseline row for any (baseline, state_dtype, SR)
    # combo it can't run.  Better than erroring on a partial sweep — our
    # kernel rows still print.  Compatibility:
    #   * Quantized states (int8 / int16 / fp8): no baseline supports them.
    #   * Triton baseline (selective_state_update): no rand_seed kwarg.
    #   * flashinfer baseline: rand_seed only on fp16 state.
    def _baseline_supports() -> bool:
        if baseline_fn is None:
            return False
        if is_quantized:
            return False
        if use_philox:
            if args.baseline == "triton":
                return False
            if args.baseline == "flashinfer" and state_dtype != torch.float16:
                return False
        return True

    if baseline_fn is not None and not _baseline_supports():
        if not warmup_only:
            sr_tag = " + SR" if use_philox else ""
            print(
                f"# Skipping {args.baseline} baseline for "
                f"state_dtype={state_dtype_name}{sr_tag} (unsupported)."
            )
        baseline_fn = None

    show_kernel_col = baseline_fn is not None

    def _conv1d_split(xbc_in, conv_st, launch_dependent_kernels=False):
        """Run conv1d update and split output into (x, B, C) views.

        The input tensor's strides are preserved through conv1d and the
        transpose+view chain.  With the production-matching layout
        (contiguous (batch*T, conv_dim) viewed as (batch, conv_dim, T)),
        the output after transpose+view has stride(-1)==1 and
        stride(1)==dim, satisfying both our kernel and flashinfer.
        """
        xbc_result = causal_conv1d_update(
            xbc_in,
            conv_st,
            conv_weight,
            conv_bias,
            activation="silu",
            launch_dependent_kernels=launch_dependent_kernels,
        )
        xbc_flat = xbc_result.transpose(1, 2).view(batch * mtp_len, conv_dim)
        x_flat, B_flat, C_flat = torch.split(
            xbc_flat, [d_inner, ngroups * d_state, ngroups * d_state], dim=-1
        )
        x_conv = x_flat.view(batch, mtp_len, nheads, head_dim)
        B_conv = B_flat.view(batch, mtp_len, ngroups, d_state)
        C_conv = C_flat.view(batch, mtp_len, ngroups, d_state)
        return x_conv, B_conv, C_conv

    # --- Baseline ---
    if baseline_fn is not None:
        tag = f"base_b{batch}_mtp{mtp_len}_s{state_dtype_name}_a{act_dtype_name}"

        philox_kwargs = {}
        if rand_seed is not None and args.baseline == "flashinfer":
            philox_kwargs = {"rand_seed": rand_seed, "philox_rounds": args.philox_rounds}

        if with_conv1d:

            def _run_baseline():
                x_conv, B_conv, C_conv = _conv1d_split(xbc_input_work, conv_state_work)
                baseline_fn(
                    state_work,
                    x=x_conv,
                    dt=dt,
                    A=A,
                    B=B_conv,
                    C=C_conv,
                    D=D,
                    dt_bias=dt_bias,
                    dt_softplus=True,
                    out=out_base,
                    disable_state_update=True,
                    intermediate_states_buffer=intermediate_states_buffer,
                    cache_steps=mtp_len,
                    **philox_kwargs,
                )
        else:

            def _run_baseline():
                baseline_fn(
                    state_work,
                    x=x,
                    dt=dt,
                    A=A,
                    B=B,
                    C=C,
                    D=D,
                    dt_bias=dt_bias,
                    dt_softplus=True,
                    out=out_base,
                    disable_state_update=True,
                    intermediate_states_buffer=intermediate_states_buffer,
                    cache_steps=mtp_len,
                    **philox_kwargs,
                )

        reset_fn = _reset_conv1d_realistic if with_conv1d else _reset
        if warmup_only:
            reset_fn()
            _run_baseline()
            torch.cuda.synchronize()
        else:
            stats = _time_kernel(args, _run_baseline, reset_fn, tag)

            _print_row(
                show_kernel_col,
                args.baseline,
                batch,
                mtp_len,
                "N/A",
                state_dtype_name,
                act_dtype_name,
                stats,
                json_results=getattr(args, "_json_results", None),
                tp_size=args.tp_size,
                json_detailed=getattr(args, "json_detailed", False),
            )

    # --- Sweep parameter parsing (invariant across prev_k) ---
    def _parse_sweep(val):
        if val is None:
            return [None]
        return [int(v) for v in val.split(",")]

    block_size_m_values = _parse_sweep(args.block_size_m)
    num_warps_values = _parse_sweep(args.num_warps)
    num_stages_values = _parse_sweep(args.num_stages)
    precompute_num_warps_values = _parse_sweep(args.precompute_num_warps)
    precompute_num_stages_values = _parse_sweep(args.precompute_num_stages)
    heads_per_block_values = _parse_sweep(args.heads_per_block)
    maxnreg_values = _parse_sweep(args.maxnreg)
    num_ctas_values = _parse_sweep(args.num_ctas)

    # --- Replay kernel, one row per prev_k ---
    # Cache T-axis capacity (for prev_k validity check on the nowrite path).
    max_window = getattr(args, "max_window", 0) or mtp_len
    for prev_k in prev_ks:
        # On the nowrite path, new tokens append at [prev_k, prev_k+T) of the
        # active buffer, so prev_k+T must fit within max_window.  Skip
        # silently for combinations that don't satisfy this — lets a single
        # nsys run sweep both write modes against a shared prev_k list.
        if not write_checkpoint and prev_k + mtp_len > max_window:
            continue
        prev_tokens.fill_(prev_k)
        tag = f"incr_b{batch}_mtp{mtp_len}_k{prev_k}_s{state_dtype_name}_a{act_dtype_name}"

        for (
            block_size_m,
            num_warps,
            num_stages,
            precompute_num_warps,
            precompute_num_stages,
            heads_per_block,
            maxnreg,
            num_ctas,
        ) in itertools.product(
            block_size_m_values,
            num_warps_values,
            num_stages_values,
            precompute_num_warps_values,
            precompute_num_stages_values,
            heads_per_block_values,
            maxnreg_values,
            num_ctas_values,
        ):

            def _run_incr(
                prev_k=prev_k,
                block_size_m=block_size_m,
                num_warps=num_warps,
                num_stages=num_stages,
                precompute_num_warps=precompute_num_warps,
                precompute_num_stages=precompute_num_stages,
                heads_per_block=heads_per_block,
                maxnreg=maxnreg,
                num_ctas=num_ctas,
            ):
                if with_conv1d:
                    x_call, B_call, C_call = _conv1d_split(
                        xbc_input_work, conv_state_work, launch_dependent_kernels=args.external_pdl
                    )
                    extra_kwargs = {"launch_with_pdl": args.external_pdl}
                else:
                    x_call, B_call, C_call = x, B, C
                    extra_kwargs = {}
                # write_checkpoint is only meaningful for the checkpointing
                # variant; replay variant ignores the kwarg.  state_scales
                # is also checkpointing-only (replay kernel doesn't quantize).
                if args.variant == "checkpointing":
                    extra_kwargs["write_checkpoint"] = write_checkpoint
                    extra_kwargs["rectangle_for_nowrite"] = rectangle_for_nowrite
                    if state_scales_work is not None:
                        extra_kwargs["state_scales"] = state_scales_work
                    if getattr(args, "use_tma_state", False):
                        extra_kwargs["_use_tma_state"] = True
                    if getattr(args, "use_tma_state_load_replay", False):
                        extra_kwargs["_use_tma_state_load_replay"] = True
                    if getattr(args, "use_tma_state_store_replay", False):
                        extra_kwargs["_use_tma_state_store_replay"] = True
                variant_fn(
                    state_work,
                    old_x_work,
                    old_B_work,
                    old_dt_work,
                    old_dA_cumsum_work,
                    cache_buf_idx_work,
                    prev_tokens,
                    x=x_call,
                    dt=dt,
                    A=A,
                    B=B_call,
                    C=C_call,
                    out=out_incr,
                    D=D,
                    dt_bias=dt_bias,
                    dt_softplus=True,
                    state_batch_indices=None,
                    rand_seed=rand_seed,
                    philox_rounds=args.philox_rounds,
                    use_internal_pdl=args.internal_pdl,
                    _block_size_m=block_size_m,
                    _num_warps=num_warps,
                    _num_stages=num_stages,
                    _precompute_num_warps=precompute_num_warps,
                    _precompute_num_stages=precompute_num_stages,
                    _heads_per_block=heads_per_block,
                    _maxnreg=maxnreg,
                    _num_ctas=num_ctas,
                    **extra_kwargs,
                )

            parts = []
            if block_size_m is not None:
                parts.append(f"M={block_size_m}")
            if num_warps is not None:
                parts.append(f"W={num_warps}")
            if num_stages is not None:
                parts.append(f"S={num_stages}")
            if precompute_num_warps is not None:
                parts.append(f"pW={precompute_num_warps}")
            if precompute_num_stages is not None:
                parts.append(f"pS={precompute_num_stages}")
            if heads_per_block is not None:
                parts.append(f"H={heads_per_block}")
            if maxnreg is not None:
                parts.append(f"R={maxnreg}")
            if num_ctas is not None:
                parts.append(f"CT={num_ctas}")
            parts.append(f"SR={1 if use_philox else 0}")
            parts.append(f"RECT={1 if rectangle_for_nowrite else 0}")
            parts.append(f"WC={1 if write_checkpoint else 0}")
            sweep_suffix = (" " + ",".join(parts)) if parts else ""
            sweep_tag = tag + sweep_suffix.replace(" ", "_").replace(",", "_")

            reset_fn = _reset_conv1d_realistic if with_conv1d else _reset
            if warmup_only:
                reset_fn()
                _run_incr()
                torch.cuda.synchronize()
            else:
                stats = _time_kernel(args, _run_incr, reset_fn, sweep_tag)

                _print_row(
                    show_kernel_col,
                    args.variant,
                    batch,
                    mtp_len,
                    prev_k,
                    state_dtype_name,
                    act_dtype_name,
                    stats,
                    sweep_suffix,
                    json_results=getattr(args, "_json_results", None),
                    tp_size=args.tp_size,
                    json_detailed=getattr(args, "json_detailed", False),
                )


# Map full torch dtype name → short tag used in JSON keys (matches collect.py).
_DTYPE_SHORT = {
    "float32": "fp32", "bfloat16": "bf16", "float16": "fp16",
    "int8": "int8", "int16": "int16", "float8_e4m3fn": "fp8",
}


def _build_json_key(
    kernel_name, batch, mtp_len, prev_k, state_dtype_name, sweep_suffix, tp_size
):
    """Build a key matching collect.py's kernel_data.json convention:

      incremental/{batch}/{mtp}/{sd}/k{k}/{sweep_parts}/tp{tp}
      triton/{batch}/{mtp}/{sd}/tp{tp}
      flashinfer/{batch}/{mtp}/{sd}/tp{tp}

    `kernel_name` is what _print_row receives: variant name for the timed
    kernel (replay/checkpointing) or baseline name for the baseline row.
    Variant rows collapse to "incremental" — the variant choice is captured
    by the sweep flags collect.py would otherwise apply via --variant.
    """
    if kernel_name in ("replay", "checkpointing"):
        kind = "incremental"
    else:
        kind = kernel_name  # "triton" / "flashinfer"

    sd = _DTYPE_SHORT.get(state_dtype_name, state_dtype_name)
    parts = [kind, str(batch), str(mtp_len), sd]
    if prev_k != "N/A":
        parts.append(f"k{prev_k}")
    if sweep_suffix:
        # sweep_suffix format: " M=4,W=1,S=1,SR=0,RECT=0,WC=1"
        # collect.py format:    "M4_W1_S1_SR0_RECT0_WC0"
        # Strip leading/trailing whitespace, drop '=', commas → underscores.
        parts.append(
            sweep_suffix.strip().replace("=", "").replace(",", "_")
        )
    parts.append(f"tp{tp_size}")
    return "/".join(parts)


def _print_row(
    show_kernel_col,
    kernel_name,
    batch,
    mtp_len,
    prev_k,
    state_dtype_name,
    act_dtype_name,
    stats,
    sweep_suffix="",
    json_results=None,
    tp_size=None,
    json_detailed=False,
):
    """Print one summary row and optionally accumulate stats for JSON output.

    `stats` is a dict from _time_kernel: {median, p95, p99, n, iters_us,
    [per_kernel]}.  The summary table only shows the headline percentiles.
    JSON output captures median/p95/p99/n by default; with json_detailed=True
    it also captures the per-iter and per-kernel data.
    """
    kernel_col = f"{kernel_name:>11} | " if show_kernel_col else ""
    print(
        f"| {kernel_col}{batch:>5} | {mtp_len:>7} | {str(prev_k):>6} | "
        f"{state_dtype_name:>11} | {act_dtype_name:>9} | "
        f"{stats['median']:>9.2f} | {stats['p95']:>7.2f} | {stats['p99']:>7.2f} |"
        f"{sweep_suffix}"
    )
    if json_results is not None:
        key = _build_json_key(
            kernel_name, batch, mtp_len, prev_k, state_dtype_name,
            sweep_suffix, tp_size,
        )
        if json_detailed:
            json_results[key] = stats
        else:
            json_results[key] = {
                k: stats[k] for k in ("median", "p95", "p99", "n")
                if k in stats
            }


# Main benchmark loop


def _run_benchmark(args) -> None:
    # JSON accumulator — populated by _print_row when --json-output is set.
    # Stash on args so we don't need to thread a dict through every helper.
    args._json_results = {} if getattr(args, "json_output", None) else None

    assert args.nheads % args.tp_size == 0, (
        f"nheads ({args.nheads}) must be divisible by tp_size ({args.tp_size})"
    )
    assert args.ngroups % args.tp_size == 0, (
        f"ngroups ({args.ngroups}) must be divisible by tp_size ({args.tp_size})"
    )
    args.tp_nheads = args.nheads // args.tp_size
    args.tp_ngroups = args.ngroups // args.tp_size

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    mtp_lengths = [int(x) for x in args.mtp_lengths.split(",")]

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "fp16": torch.float16,
        "int8": torch.int8,
        "int16": torch.int16,
        "fp8": torch.float8_e4m3fn,
    }
    state_dtypes = [dtype_map[s] for s in args.state_dtypes.split(",")]
    act_dtypes = [dtype_map[s] for s in args.act_dtypes.split(",")]

    # Resolve baseline function
    if args.baseline == "flashinfer":
        from flashinfer.mamba import selective_state_update as baseline_fn
    elif args.baseline == "triton":
        baseline_fn = selective_state_update
    else:
        baseline_fn = None

    # --with-conv1d uses its own realistic L2 flush (cold cache flush then
    # hot in_proj write).  Override the generic l2_flush to avoid double-flushing.
    if args.with_conv1d:
        args.l2_flush = False
        _init_l2_flush()  # still needed for the realistic reset's flush step
    elif args.l2_flush:
        _init_l2_flush()

    if args.compile_threads > 0:
        _compile_warmup_phase(
            args, batch_sizes, mtp_lengths, state_dtypes, act_dtypes,
            baseline_fn, max_workers=args.compile_threads,
        )

    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()

    # Print header
    if baseline_fn is not None:
        print(
            f"| {'kernel':>11} | {'batch':>5} | {'mtp_len':>7} | {'prev_k':>6} | "
            f"{'state_dtype':>11} | {'act_dtype':>9} | "
            f"{'median_us':>9} | {'p95_us':>7} | {'p99_us':>7} |"
        )
        print(
            f"|{'-' * 13}|{'-' * 7}|{'-' * 9}|{'-' * 8}|"
            f"{'-' * 13}|{'-' * 11}|{'-' * 11}|{'-' * 9}|{'-' * 9}|"
        )
    else:
        print(
            f"| {'batch':>5} | {'mtp_len':>7} | {'prev_k':>6} | "
            f"{'state_dtype':>11} | {'act_dtype':>9} | "
            f"{'median_us':>9} | {'p95_us':>7} | {'p99_us':>7} |"
        )
        print(
            f"|{'-' * 7}|{'-' * 9}|{'-' * 8}|{'-' * 13}|{'-' * 11}|{'-' * 11}|{'-' * 9}|{'-' * 9}|"
        )

    sr_modes_list = getattr(args, "sr_modes_list", ["RN"])
    rect_list = getattr(args, "rectangle_for_nowrite_list", [False])
    write_modes_list = getattr(args, "write_modes_list", [args.write_checkpoint])

    for batch in batch_sizes:
        for mtp_len in mtp_lengths:
            # Resolve prev_k fractions → clamped integers in [0, mtp_len]
            prev_ks = _resolve_prev_ks(args, mtp_len)
            for state_dtype in state_dtypes:
                for act_dtype in act_dtypes:
                    for sr_mode in sr_modes_list:
                        for write_ckpt in write_modes_list:
                            # Rectangle only meaningful for nowrite cells.
                            effective_rect_list = (
                                [False] if write_ckpt else rect_list
                            )
                            for rect in effective_rect_list:
                                _bench_config(
                                    args, batch, mtp_len, prev_ks, state_dtype, act_dtype,
                                    baseline_fn, sr_mode=sr_mode,
                                    rectangle_for_nowrite=rect,
                                    write_checkpoint=write_ckpt,
                                )

    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()

    if args.json_output and args._json_results is not None:
        import json
        payload = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "cmd": " ".join(sys.argv),
                "tp_size": args.tp_size,
                "warmup": args.warmup,
                "iters": args.iters,
                "variant": args.variant,
                "cupti": getattr(args, "cupti", False),
            },
            "results": args._json_results,
        }
        tmp = args.json_output + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, args.json_output)
        print(f"\nJSON results written to: {args.json_output} "
              f"({len(args._json_results)} entries)")


# CLI


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark replay_selective_state_update Triton kernel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--nheads",
        type=int,
        default=NHEADS,
        help="Full-model nheads (divided by --tp-size for per-GPU slice)",
    )
    parser.add_argument(
        "--ngroups",
        type=int,
        default=NGROUPS,
        help="Full-model ngroups (divided by --tp-size for per-GPU slice)",
    )
    parser.add_argument(
        "--head-dim", type=int, default=HEAD_DIM, help="Head dimension (not TP-split)"
    )
    parser.add_argument(
        "--d-state", type=int, default=D_STATE, help="SSM state dimension (not TP-split)"
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=TP_SIZE,
        help="Tensor parallel size; divides nheads and ngroups",
    )
    parser.add_argument(
        "--batch-sizes", default="1,2,4,8", help="Comma-separated decode batch sizes"
    )
    parser.add_argument(
        "--mtp-lengths",
        default="1,2,4,8",
        help="Comma-separated per-request sequence lengths (num_draft_tokens + 1 target)",
    )
    parser.add_argument(
        "--state-dtypes",
        default="fp32",
        help="Comma-separated state dtypes: fp16,bf16,fp32,int8,int16,fp8.  "
        "Quantized dtypes (int8/int16/fp8) require the checkpointing variant "
        "and skip baselines (selective_state_update doesn't accept them).",
    )
    parser.add_argument(
        "--act-dtypes",
        default="bf16",
        help="Comma-separated activation dtypes for x/B/C/dt: fp32,bf16",
    )
    parser.add_argument("--warmup", type=int, default=20, help="Number of warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Number of timed iterations")
    parser.add_argument(
        "--compile-threads",
        type=int,
        default=64,
        help="Number of THREADS used in the compile-warmup phase (one call "
        "per (batch, mtp_len, prev_k, dtype, sweep) cell, parallelized over "
        "N threads).  Triton compile releases the GIL, so threads compile "
        "in parallel and populate the persistent cache for free hits during "
        "the sequential timed phase.  0 disables the phase.  Default 64.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Wrap timed region in cudaProfilerStart/Stop (for ncu --target-processes all)",
    )
    parser.add_argument(
        "--l2-flush",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2 eviction between iterations",
    )
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture all warmup + timed iterations in a "
        "single CUDA graph with per-iteration events "
        "inside the graph, eliminating all host overhead.",
    )
    parser.add_argument(
        "--cupti",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Time kernels via CUPTI Activity API (1 ns from the GPU "
        "profiling fabric); per-iter span = max(kernel_end) - "
        "min(kernel_start). Default ON.  --no-cupti disables in-bench "
        "timing entirely (kernels still run, but median/p95/p99 are zero) "
        "— use when wrapping the bench in nsys/ncu, where the external "
        "profiler provides timings and our CUPTI subscriber would conflict.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="If set, write per-cell results to this JSON file in the "
        "shape consumed by collect.py / report.py.  See the 'JSON output "
        "schema' section at the top of this file.",
    )
    parser.add_argument(
        "--json-detailed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When --json-output is set, also include iters_us (raw per-iter "
        "spans) and per_kernel (per-iter relative start/end timestamps for "
        "each kernel) — useful for PDL overlap analysis but adds ~4 KB/cell. "
        "Default off keeps records to ~40 bytes (median/p95/p99/n only).",
    )
    parser.add_argument(
        "--prev-tokens-fracs",
        default="0,0.5,1.0",
        type=lambda s: [float(x) for x in s.split(",")],
        help="Fractions of mtp_len to use as prev_num_accepted_tokens "
        "for the replay kernel sweep. Values are rounded "
        "and clamped to [0, mtp_len].",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        nargs="?",
        const="triton",
        choices=[None, "triton", "flashinfer"],
        help="Baseline to benchmark alongside the replay kernel. "
        "'triton': native Triton selective_state_update. "
        "'flashinfer': flashinfer selective_state_update (same signature). "
        "Pass --baseline alone for 'triton'. Default: no baseline.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save results (file or directory). "
        "If a directory, writes benchmark_replay_<timestamp>.txt inside it.",
    )
    parser.add_argument(
        "--block-size-m",
        type=str,
        default=None,
        help="Override BLOCK_SIZE_M: single value or comma-separated sweep (e.g. '4,8,16,32').",
    )
    parser.add_argument(
        "--num-warps",
        type=str,
        default=None,
        help="Override num_warps: single value or comma-separated sweep (e.g. '1,2,4').",
    )
    parser.add_argument(
        "--internal-pdl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Internal PDL between precompute and main kernels (default: on).",
    )
    parser.add_argument(
        "--num-stages",
        type=str,
        default=None,
        help="Override num_stages for the main kernel (comma-separated sweep).",
    )
    parser.add_argument(
        "--precompute-num-warps",
        type=str,
        default=None,
        help="Override num_warps for precompute kernel (comma-separated sweep).",
    )
    parser.add_argument(
        "--precompute-num-stages",
        type=str,
        default=None,
        help="Override num_stages for precompute kernel (comma-separated sweep).",
    )
    parser.add_argument(
        "--max-window",
        type=int,
        default=16,
        help="Cache T-axis capacity (max replay buffer length).  Default 16 "
        "matches Nemotron-3-Super-120B production.  Pass 0 to fall back to "
        "mtp_len (degenerate every-step-checkpoint case, mostly unused).",
    )
    parser.add_argument(
        "--prev-tokens-int",
        type=lambda s: [int(x) for x in s.split(",")] if s else None,
        default=None,
        help="Absolute prev_num_accepted_tokens values to test, comma-separated "
        "(e.g. '0,10,11,16').  Clamped to [0, max_window].  When set, "
        "overrides --prev-tokens-fracs.",
    )
    parser.add_argument(
        "--write-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether the checkpointing kernel should write the post-replay "
        "state to HBM.  True = checkpoint step (default).  False = "
        "non-checkpoint step (skip state HBM write + Philox).  No effect on "
        "the replay variant.  Ignored if --write-modes is set.",
    )
    parser.add_argument(
        "--write-modes",
        type=str,
        default=None,
        help="Comma-separated 0/1 values to sweep both write modes in a "
        "single nsys process — for apples-to-apples comparison of write "
        "vs nowrite (replay) vs nowrite (rectangle) within one timeline. "
        "Skips silently for (write=False, prev_k+T>max_window) combos. "
        "When set, overrides --write-checkpoint.",
    )
    parser.add_argument(
        "--with-conv1d",
        action="store_true",
        help="Include conv1d kernel before replay SSM. "
        "Uses realistic L2 flush: cold caches flushed, hot in_proj output "
        "kept warm. Measures conv1d → precompute → main span.",
    )
    parser.add_argument(
        "--external-pdl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="External PDL: conv1d launches dependents, precompute waits. "
        "Only relevant with --with-conv1d. --no-external-pdl disables.",
    )
    parser.add_argument(
        "--heads-per-block",
        type=str,
        default=None,
        help="Override HEADS_PER_BLOCK for precompute kernel (comma-separated sweep).",
    )
    parser.add_argument(
        "--maxnreg",
        type=str,
        default=None,
        help="Override maxnreg for the main kernel (comma-separated sweep).",
    )
    parser.add_argument(
        "--num-ctas",
        type=str,
        default=None,
        help="Override num_ctas for the main kernel (comma-separated sweep).",
    )
    parser.add_argument(
        "--sr-modes",
        type=str,
        default="RN",
        help="Comma-separated rounding modes to sweep: any combination of "
        "{RN, SR}.  SR (stochastic rounding) is silently skipped for state "
        "dtypes that don't support it (bf16, fp32).  Default 'RN' matches "
        "legacy --philox-rounding=False behavior.",
    )
    parser.add_argument(
        "--rectangle-for-nowrite",
        type=str,
        default="0",
        help="Comma-separated 0/1 values: 0 = replay-style nowrite kernel, "
        "1 = dedicated rectangle nowrite kernel.  Sweep both with '0,1' to "
        "compare in one invocation.  Silently no-op for write cells (the "
        "write path always uses replay-style).  Only applies to the "
        "checkpointing variant.",
    )
    parser.add_argument(
        "--use-tma-state",
        action="store_true",
        help="Use TMA (host-built tensor descriptor) for the state load in "
        "_rectangle_main_kernel.  Only applies to the rectangle nowrite path "
        "of the checkpointing variant; ignored otherwise.",
    )
    parser.add_argument(
        "--use-tma-state-load-replay",
        action="store_true",
        help="Use TMA for state LOAD in _checkpointing_main_kernel (replay "
        "main, both WC=0 and WC=1 paths).  Independent from rect TMA.",
    )
    parser.add_argument(
        "--use-tma-state-store-replay",
        action="store_true",
        help="Use TMA for state STORE in _checkpointing_main_kernel (replay "
        "main, WC=1 path only — no-op for WC=0).  Independent from rect TMA "
        "and from --use-tma-state-load-replay.",
    )
    parser.add_argument(
        "--philox-rounding",
        action="store_true",
        help="DEPRECATED — equivalent to --sr-modes SR.  Retained for "
        "backward compatibility; use --sr-modes for new scripts.  fp16 SR "
        "and fp8 SR require sm_100a (Blackwell B200+).",
    )
    parser.add_argument(
        "--philox-rounds",
        type=int,
        default=5,
        help="Number of Philox PRNG rounds.  Default 5 matches the "
        "Nemotron-3-Super-120B production config (mamba_ssm_philox_rounds=5 "
        "in examples/configs and tests/integration/perf configs).  The "
        "wrapper's generic fallback default is 10; callers without explicit "
        "config see 10.  Only consulted when --philox-rounding is enabled.",
    )
    parser.add_argument(
        "--variant",
        choices=["replay", "checkpointing"],
        default="replay",
        help="Which kernel to time as the 'replay' row.  'replay' = today's "
        "kernel (selective_state_update.py:replay).  'checkpointing' = "
        "checkpointing_state_update.py.  Both share the same wrapper signature.",
    )
    parser.add_argument(
        "--full-import",
        action="store_true",
        help="Use standard tensorrt_llm import path instead of fast direct "
        "module loading. Slower (~40s startup) but guaranteed correct "
        "if the fast path breaks due to package changes.",
    )
    args = parser.parse_args()

    # Backward-compat: --philox-rounding implies --sr-modes SR if --sr-modes
    # was left at the default.  If both are set explicitly, error.
    sr_modes_default = (args.sr_modes == "RN")
    if args.philox_rounding:
        if not sr_modes_default and args.sr_modes != "SR":
            parser.error(
                "--philox-rounding (deprecated) is incompatible with explicit "
                f"--sr-modes={args.sr_modes!r}.  Use --sr-modes SR (or "
                "RN,SR) instead and drop --philox-rounding."
            )
        args.sr_modes = "SR"

    sr_modes = [m.strip() for m in args.sr_modes.split(",") if m.strip()]
    for m in sr_modes:
        if m not in ("RN", "SR"):
            parser.error(f"--sr-modes value must be RN or SR, got {m!r}")
    args.sr_modes_list = sr_modes

    rect_modes = [v.strip() for v in args.rectangle_for_nowrite.split(",") if v.strip()]
    rect_list = []
    for v in rect_modes:
        if v not in ("0", "1"):
            parser.error(f"--rectangle-for-nowrite value must be 0 or 1, got {v!r}")
        rect_list.append(v == "1")
    args.rectangle_for_nowrite_list = rect_list

    if args.write_modes is not None:
        wm = [v.strip() for v in args.write_modes.split(",") if v.strip()]
        write_list = []
        for v in wm:
            if v not in ("0", "1"):
                parser.error(f"--write-modes value must be 0 or 1, got {v!r}")
            write_list.append(v == "1")
        args.write_modes_list = write_list
    else:
        args.write_modes_list = [args.write_checkpoint]
    return args


class _Tee:
    """Write to both stdout and a file simultaneously."""

    def __init__(self, path: str):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._file = open(path, "w")  # noqa: SIM115
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()


if __name__ == "__main__":
    _args = _parse_args()

    _out_path = None
    if _args.output != "-":
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _fname = f"benchmark_replay_{_ts}.txt"
        if _args.output is None:
            _out_path = os.path.expanduser(f"~/nemo_logs/{_fname}")
        elif os.path.isdir(_args.output) or _args.output.endswith("/"):
            _out_path = os.path.join(_args.output, _fname)
        else:
            _out_path = _args.output

    if _out_path:
        _tee = _Tee(_out_path)
        sys.stdout = _tee
        print(f"# benchmark_replay_selective_state_update  {datetime.now().isoformat()}")
        print(f"# cmd: {' '.join(sys.argv)}")

    try:
        _run_benchmark(_args)
    finally:
        if _out_path:
            sys.stdout = _tee._stdout
            _tee.close()
            print(f"\nResults saved to: {_out_path}")
