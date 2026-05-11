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

import numpy as np
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
    # slot_perm placeholder — overwritten per-run by mix pre_iter_fn when
    # sort_slots is enabled.  Identity by default so cells that don't sort
    # (or pure-batch cells) get a meaningful identity perm if the kernel
    # ends up reading it (USE_PERM=False makes this path unused).
    slot_perm_buf = torch.arange(batch, device=device, dtype=torch.int32)

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
        slot_perm_buf,
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
    "_persistent_main",
    "selective_scan_update",
    "selective_state_update",
    "causal_conv1d_update",
)


def _kernels_per_iter_incremental(
    mode: str,
    with_conv1d: bool,
    *,
    persistent_skip_empty: bool = True,
) -> int:
    """Expected number of CUPTI-tracked kernels per iter for the incremental
    kernel chain, given the dispatch mode and the conv1d flag.

    Used to validate CUPTI record counts (no auto-inference — silent
    mis-timing is the failure mode we're guarding against).

    `persistent_skip_empty=True` (today's behavior): the
    `mode='persistent_main'` launch helper host-early-outs when its half
    is empty (n_writes=0 or n_writes=batch in pure scenarios), so only
    one of the two persistent_main_kernel launches actually fires per
    iter.  With `persistent_skip_empty=False` (future no-eo mode), both
    halves always launch and K bumps by 1.

    `persistent_dynamic` always launches 1 main; not affected by the flag.
    """
    if mode == "monolithic":
        k = 2  # precomp + main
    elif mode == "dynamic":
        k = 2  # dynamic_precomp + dynamic_main
    elif mode == "maindl":
        k = 3  # 1 dynamic_precomp + 2 mains (write + nowrite)
    elif mode in ("doublelaunch", "dlgrouped"):
        k = 4  # 2 precomp + 2 main
    elif mode == "dl_write_only":
        k = 2  # 1 precomp + 1 main (write only)
    elif mode == "persistent_dynamic":
        k = 2  # 1 dynamic_precomp + 1 persistent_main
    elif mode == "persistent_main":
        k = 2 if persistent_skip_empty else 3  # see docstring
    else:
        raise ValueError(f"_kernels_per_iter_incremental: unknown mode {mode!r}")
    if with_conv1d:
        k += 1
    return k


def _kernels_per_iter_baseline(with_conv1d: bool) -> int:
    """Expected kernels per iter for triton / flashinfer baselines.

    Both baselines run a single state-update kernel; `--with-conv1d`
    prepends one conv1d kernel.
    """
    return 2 if with_conv1d else 1


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


def _stats_from_cupti_records(records, warmup, iters, tag, expected_K):
    """Bin a flat CUPTI kernel record stream into per-iter spans + per-kernel
    relative timestamps.  Used by both graph and eager CUPTI paths.

    `records` are tuples (name, start_ns, end_ns, ...) — see CuptiKernelTimer.
    `expected_K` is the kernels-per-iter count the caller declares; we
    validate the CUPTI total matches `expected_K * (warmup + iters)` exactly.
    On mismatch we dump per-name record counts so missing or extra kernels
    are obvious (most common cause: a new dispatch mode whose kernels lack
    a matching entry in `_CUPTI_KEEP_KERNEL_SUBSTRINGS`, silently filtering
    them out).
    """
    records = [
        r for r in records
        if any(s in r[0] for s in _CUPTI_KEEP_KERNEL_SUBSTRINGS)
    ]
    records.sort(key=lambda r: r[1])  # by start_ns

    total = len(records)
    expected_iters = warmup + iters
    expected_total = expected_K * expected_iters
    if total != expected_total:
        from collections import Counter
        name_counts = dict(Counter(r[0] for r in records))
        raise RuntimeError(
            f"CUPTI capture mismatch for {tag!r}: expected {expected_K} "
            f"kernels/iter × {expected_iters} iters (warmup+iters) = "
            f"{expected_total} records, got {total}.  Kernel record counts: "
            f"{name_counts}.  If a kernel name is missing, add a substring "
            f"to _CUPTI_KEEP_KERNEL_SUBSTRINGS; if present but the count is "
            f"wrong, adjust _kernels_per_iter_* for this mode."
        )
    K = expected_K
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


_PRE_GRAPH_WARMUP_ITERS = 3  # standard practice; see commit msg / design doc


def _time_kernel_cuda_graph(
    args,
    run_fn,
    reset_fn,
    tag: str,
    *,
    expected_K: int,
    pre_iter_fn=None,
    iters_override: int | None = None,
) -> dict:
    """CUDA-graph CUPTI timer (graph-per-iter design).

    Captures one CUDA graph holding a single iter's worth of work (reset
    + l2_flush + run_fn) and replays it `warmup + iters` times.  Per-iter
    setup (`pre_iter_fn`, e.g. mix-mode PNAT/n_writes copies) runs OUTSIDE
    the graph, on the same CUDA stream so the order
    `pre_iter_fn → l2_flush → run_fn` is preserved on every replay.

    Why graph-per-iter (vs the older "one giant graph holding all iters"
    design): instantiating a CUDA graph is expensive — proportional to
    graph size — so a single small graph instantiated once is much
    cheaper than one big graph instantiated for each cell of a sweep.
    Replays are cheap regardless.

    Why pre_iter_fn outside the graph: it depends on the iter index `i`
    (different sample per iter), but graph capture would bake in the
    capture-time `i`.  Putting the per-iter copy outside the graph also
    has a useful side effect: PNAT (the per-iter input) is loaded into
    L2 by the copy, then evicted by the in-graph L2 flush, so the kernel
    reads PNAT cold — closer to production behavior than the old design.

    Pre-graph eager warmup (3 iters): forces PyTorch's caching allocator
    + Triton's autotune cache to settle before capture so the graph
    doesn't bake in init-only allocations.

    ``iters_override`` (if not None) overrides ``args.iters`` for this
    call.  Used to give mix scenarios a higher iter count than pure
    (more iters = more independent mix draws averaged in).
    """
    timer = CuptiKernelTimer.get()
    warmup = args.warmup
    iters = iters_override if iters_override is not None else args.iters

    # Pre-graph eager warmup: full per-iter chain × N.  Settles allocator
    # + autotune; pre_iter_fn included so any side effects it has are
    # exercised before capture.
    for _ in range(_PRE_GRAPH_WARMUP_ITERS):
        reset_fn()
        if pre_iter_fn is not None:
            pre_iter_fn(0)
        run_fn()
    torch.cuda.synchronize()

    # Reset just before capture so warmup state changes don't bleed in.
    reset_fn()
    torch.cuda.synchronize()

    # Capture ONE iter.  pre_iter_fn deliberately not in here.
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        reset_fn()
        if args.l2_flush:
            _l2_flush.fill_(0.0)
        run_fn()
    torch.cuda.synchronize()

    # Time: replay the per-iter graph `warmup + iters` times, with
    # pre_iter_fn called between replays on the same stream.  CUPTI
    # records every kernel launch; _stats_from_cupti_records validates
    # against expected_K and slices warmup off the front.
    timer.start()
    torch.cuda.nvtx.range_push(tag)
    for i in range(warmup + iters):
        if pre_iter_fn is not None:
            pre_iter_fn(i)
        g.replay()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    records = timer.stop()

    return _stats_from_cupti_records(records, warmup, iters, tag, expected_K)


def _time_kernel_eager(
    args,
    run_fn,
    reset_fn,
    tag: str,
    *,
    expected_K: int,
    pre_iter_fn=None,
    iters_override: int | None = None,
) -> dict:
    """Non-graph CUPTI timer (for ncu wrapping, debugging, etc.).

    Each iter runs serially with sync between, but kernel start/end still
    come from CUPTI — same accuracy as the graph path, just slower per-iter
    (extra Python + sync overhead).
    """
    timer = CuptiKernelTimer.get()
    warmup = args.warmup
    iters = iters_override if iters_override is not None else args.iters

    timer.start()
    torch.cuda.nvtx.range_push(tag)
    # Unified warmup+iters loop; CUPTI filters by warmup count internally.
    for i in range(warmup + iters):
        reset_fn()
        if args.l2_flush:
            _flush_l2()  # includes synchronize
        if pre_iter_fn is not None:
            pre_iter_fn(i)
        run_fn()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    records = timer.stop()

    return _stats_from_cupti_records(records, warmup, iters, tag, expected_K)


def _run_kernel_untimed(args, run_fn, reset_fn, tag: str) -> dict:
    """No in-bench timing: just run the kernels for an external profiler
    (nsys / ncu) to time externally.  Returns a stats dict full of zeros so
    downstream code (table, JSON) doesn't break.

    Note: pre_iter_fn / iters_override aren't plumbed here yet — mix-mode
    benchmarking relies on CUPTI.  Add when a use-case lands.
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


def _time_kernel(
    args, run_fn, reset_fn, tag: str,
    *,
    expected_K: int,
    pre_iter_fn=None,
    iters_override: int | None = None,
) -> dict:
    """Dispatch to graph-CUPTI / eager-CUPTI / no-timer path.

    --cupti: in-process CUPTI Activity API timing (default).  Use --no-cupti
    when running under nsys (in-process CUPTI conflicts with nsys's own
    subscriber); the bench then runs the kernels for nsys to time externally.

    `expected_K` is the kernels-per-iter count the caller declares
    (computed via _kernels_per_iter_*).  CUPTI paths validate against it
    explicitly; the no-timer fallback ignores it (no records to validate).
    """
    if not getattr(args, "cupti", True):
        if pre_iter_fn is not None:
            raise RuntimeError(
                "_time_kernel: pre_iter_fn requires CUPTI (mix-mode); "
                "got --no-cupti.  Re-run with CUPTI on or plumb pre_iter_fn "
                "through _run_kernel_untimed."
            )
        return _run_kernel_untimed(args, run_fn, reset_fn, tag)
    if args.cuda_graph:
        return _time_kernel_cuda_graph(
            args, run_fn, reset_fn, tag,
            expected_K=expected_K,
            pre_iter_fn=pre_iter_fn,
            iters_override=iters_override,
        )
    return _time_kernel_eager(
        args, run_fn, reset_fn, tag,
        expected_K=expected_K,
        pre_iter_fn=pre_iter_fn,
        iters_override=iters_override,
    )


# Per-config benchmark (consolidated baseline + replay)


def _warm_one_config(args, cfg, baseline_fn) -> None:
    """Module-level worker for the compile-warmup process pool.

    Module-level so ProcessPoolExecutor can pickle it (nested functions
    aren't picklable).  Each worker process holds its own GIL → no
    serialization between concurrent compiles.

    ``cfg`` is a tuple of (outer_cfg, inner_overrides):
      * outer_cfg = (batch, mtp_len, prev_ks, state_dtype, act_dtype,
                     sr_mode, rect, write_ckpt, mode,
                     sort_slots, reverse_nowrite, hardcode_sort)
      * inner_overrides = dict of args attribute name -> single-value string
                         to clamp the per-cell inner-knob sweep to ONE
                         combination.  Triggers exactly one Triton compile
                         per worker invocation, so N workers achieve
                         N-way concurrency regardless of outer config
                         count.  (Prior design fanned out only at outer
                         granularity, capping concurrency at ~10 even
                         with --compile-threads 50.)

    ``baseline_fn`` is optional — when ``None``, only the checkpointing
    kernel is warmed (the baseline-selection kernel can be warmed once in
    the parent if needed).  This lets us avoid pickling C-extension
    function references across processes.
    """
    outer_cfg, inner_overrides = cfg
    (batch, mtp_len, prev_ks, state_dtype, act_dtype, sr_mode,
     rect, write_ckpt, mode, sort_slots, reverse_nowrite, hardcode_sort) = outer_cfg
    # Clone args and override inner-knob sweep lists to single values.
    # _bench_config then iterates a 1×1×...×1 cartesian inside.
    import argparse as _ap
    args_copy = _ap.Namespace(**vars(args))
    for k, v in inner_overrides.items():
        setattr(args_copy, k, v)
    _bench_config(
        args_copy, batch, mtp_len, prev_ks, state_dtype, act_dtype, baseline_fn,
        sr_mode=sr_mode, rectangle_for_nowrite=rect,
        write_checkpoint=write_ckpt, mode=mode,
        sort_slots=sort_slots, reverse_nowrite=reverse_nowrite,
        hardcode_sort=hardcode_sort,
        warmup_only=True,
    )


def _compile_warmup_phase(args, batch_sizes, mtp_lengths, state_dtypes, act_dtypes,
                          baseline_fn, max_workers: int) -> None:
    """Parallel compile-warmup using a ProcessPoolExecutor with `spawn`
    start method.

    Each worker process holds its own GIL and its own CUDA context, so
    Triton compiles (Python AST/codegen + LLVM/ptxas) run truly in
    parallel.  Previous ThreadPoolExecutor design hit GIL contention
    in the Python codegen phase, capping throughput at ~1-2 cores even
    with 28 threads (observed: 4 R threads vs 28 in pool).

    Compiled binaries land in Triton's on-disk cache (TRITON_CACHE_DIR
    or default ~/.triton/cache).  Workers share the cache via filesystem
    — first to write any given (kernel_source × constexpr_set) hash
    wins; concurrent writes to the SAME hash are wasteful but not
    corrupting.

    spawn start method avoids inheriting parent CUDA state (which is
    unsafe after fork on Linux with active CUDA contexts).  Per-worker
    import + CUDA init costs ~10s, amortized over each worker's many
    compiles.  baseline_fn is intentionally NOT passed to workers to
    avoid pickling complications; the parent compiles the baseline
    kernel itself before launching the pool when applicable.
    """
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    sr_modes_list = getattr(args, "sr_modes_list", ["RN"])

    rect_list = getattr(args, "rectangle_for_nowrite_list", [False])
    write_modes_list = getattr(args, "write_modes_list", [args.write_checkpoint])
    modes_list = getattr(args, "modes_list", ["monolithic"])
    sort_list = getattr(args, "sort_slots_list", [False])
    rev_list = getattr(args, "reverse_nowrite_list", [False])
    hsort_list = getattr(args, "hardcode_sort_list", [False])

    # Compile-warmup task enumeration: outer × inner cartesian.
    # CRITICAL: only enumerate axes that change the kernel's COMPILE signature.
    # Drop runtime axes (batch, prev_k) that produce identical kernel hashes —
    # otherwise we'd pay ~50-100ms of bench setup per redundant cache-hit task.
    #
    # Batches collapsed to first only: batch is a runtime int passed to the
    # kernel, not a constexpr; all batches share the same compiled kernel.
    # prev_k is already a list passed into _bench_config (not enumerated here).
    configs = []
    _compile_batches = batch_sizes[:1]  # collapse runtime axis
    for batch in _compile_batches:
        for mtp_len in mtp_lengths:
            prev_ks = _resolve_prev_ks(args, mtp_len)
            for state_dtype in state_dtypes:
                for act_dtype in act_dtypes:
                    for sr_mode in sr_modes_list:
                        for mode in modes_list:
                            effective_write_modes = (
                                write_modes_list if mode == "monolithic" else [True]
                            )
                            for write_ckpt in effective_write_modes:
                                if mode == "monolithic":
                                    effective_rect_list = (
                                        [False] if write_ckpt else rect_list
                                    )
                                else:
                                    effective_rect_list = rect_list
                                for rect in effective_rect_list:
                                    # Note: "persistent_main" is included in
                                    # the dl-family for sort/hsort sweep
                                    # eligibility — it consumes the same
                                    # slot_perm and benefits from the same
                                    # write-first clustering.  It additionally
                                    # requires _n_writes (count of write
                                    # slots) which the bench computes from
                                    # the pure-scenario PNAT (mix scenarios
                                    # not yet supported for persistent_main).
                                    is_dl_family = mode in (
                                        "doublelaunch", "dlgrouped", "maindl",
                                        "dl_write_only", "persistent_main",
                                        "persistent_dynamic",
                                    )
                                    # Match the timed-run skip: sort=1 only
                                    # makes sense when there's a mix scenario.
                                    can_sort = is_dl_family and (args.mix_csv is not None)
                                    effective_sort_list = (
                                        sort_list if can_sort else [False]
                                    )
                                    effective_hsort_list = (
                                        hsort_list if can_sort else [False]
                                    )
                                    for sort_slots in effective_sort_list:
                                        effective_rev_list = (
                                            rev_list if sort_slots else [False]
                                        )
                                        for reverse_nowrite in effective_rev_list:
                                            for hardcode_sort in effective_hsort_list:
                                                if sort_slots and hardcode_sort:
                                                    continue
                                                configs.append((
                                                    batch, mtp_len, prev_ks,
                                                    state_dtype, act_dtype,
                                                    sr_mode, rect, write_ckpt, mode,
                                                    sort_slots, reverse_nowrite,
                                                    hardcode_sort,
                                                ))

    # Enumerate inner-knob cartesian — same axes _bench_config iterates
    # internally.  Each (outer × inner) tuple becomes one task; workers
    # then trigger exactly one Triton compile per task, giving true
    # N-way concurrency with --compile-threads N.
    def _ps(val):
        if val is None or (isinstance(val, str) and not val):
            return [None]
        if isinstance(val, str):
            return [v.strip() for v in val.split(",") if v.strip()]
        return [val]

    # CPS (cta_per_sm) collapsed to first value: NUM_PERSISTENT is runtime now,
    # so different CPS values share the same compiled kernel.  Collapsing here
    # avoids enumerating 4-8x redundant tasks that would each pay ~50-100ms
    # bench setup overhead for a cache hit on the same kernel hash.
    _cps_for_compile = _ps(args.cta_per_sm)[:1]
    knob_axes = [
        ("block_size_m",                 _ps(args.block_size_m)),
        ("num_warps",                    _ps(args.num_warps)),
        ("num_stages",                   _ps(args.num_stages)),
        ("precompute_num_warps",         _ps(args.precompute_num_warps)),
        ("precompute_num_stages",        _ps(args.precompute_num_stages)),
        ("heads_per_block",              _ps(args.heads_per_block)),
        ("maxnreg",                      _ps(args.maxnreg)),
        ("num_ctas",                     _ps(args.num_ctas)),
        ("cta_per_sm",                   _cps_for_compile),  # collapsed (runtime)
        ("num_loop_stages",              _ps(args.num_loop_stages)),
        ("flatten",                      _ps(args.flatten)),
        ("warp_specialize",              _ps(args.warp_specialize)),
        ("use_tma_rect_load",            _ps(args.use_tma_rect_load)),
        ("use_tma_replay_write_load",    _ps(args.use_tma_replay_write_load)),
        ("use_tma_replay_nowrite_load",  _ps(args.use_tma_replay_nowrite_load)),
        ("use_tma_replay_write_store",   _ps(args.use_tma_replay_write_store)),
    ]
    import itertools as _it
    inner_combos = list(_it.product(*(values for _, values in knob_axes)))

    # Cross product outer × inner.  Override only knobs that have an
    # explicit value (skip None — those leave args.<knob> at its CLI default,
    # which _bench_config handles via its own _parse_sweep).
    tasks = []
    for outer in configs:
        for inner_tuple in inner_combos:
            inner_overrides = {
                name: str(val)
                for (name, _), val in zip(knob_axes, inner_tuple)
                if val is not None
            }
            tasks.append((outer, inner_overrides))

    # Shuffle to reduce cross-worker race on the same kernel hash.  Two
    # workers picking adjacent tasks (same mode, neighboring knob value)
    # could both miss + compile the same kernel hash; shuffling spreads
    # workloads across different kernel hash families.
    import random as _r
    _r.shuffle(tasks)

    print(f"[compile-warmup] {len(tasks)} compile tasks "
          f"({len(configs)} outer × {len(inner_combos)} inner combos) "
          f"across {max_workers} processes (ProcessPoolExecutor, spawn start)")
    t0 = time.perf_counter()

    ctx = multiprocessing.get_context("spawn")
    errors = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        # baseline_fn=None: workers compile only the checkpointing kernel.
        # Baseline kernels (if any) get compiled lazily in the parent during
        # the timing phase — usually just one extra compile, negligible.
        futures = {
            ex.submit(_warm_one_config, args, task, None): task
            for task in tasks
        }
        for fut in futures:
            try:
                fut.result()
            except Exception as e:
                errors.append((futures[fut], e))

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
    mode: str = "monolithic",
    mix_samples_cpu=None,
    mix_label: str = "",
    sort_slots: bool = False,
    reverse_nowrite: bool = False,
    perm_samples_cpu=None,
    hardcode_sort: bool = False,
    mix_samples_sorted_cpu=None,
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
        slot_perm_buf,
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
            stats = _time_kernel(
                args, _run_baseline, reset_fn, tag,
                expected_K=_kernels_per_iter_baseline(with_conv1d),
            )

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
    # Persistent-only sweep dims; ignored when the cell's mode != persistent_main.
    cta_per_sm_values = _parse_sweep(args.cta_per_sm)
    num_loop_stages_values = _parse_sweep(args.num_loop_stages)
    flatten_values = _parse_sweep(args.flatten)
    warp_specialize_values = _parse_sweep(args.warp_specialize)
    # TMA toggles — independent 0/1 sweep per path.  The skip-dupe at the
    # top of the inner loop body collapses cells where a flag's path is
    # unreachable, so e.g. monolithic + WC=True only runs the value=0
    # cells for nowrite-load and rect-load.
    use_tma_rect_load_values = _parse_sweep(args.use_tma_rect_load)
    use_tma_replay_write_load_values = _parse_sweep(args.use_tma_replay_write_load)
    use_tma_replay_nowrite_load_values = _parse_sweep(args.use_tma_replay_nowrite_load)
    use_tma_replay_write_store_values = _parse_sweep(args.use_tma_replay_write_store)

    # --- Replay kernel ---
    # Cache T-axis capacity (for prev_k validity check on the nowrite path).
    max_window = getattr(args, "max_window", 0) or mtp_len

    # Build the list of scenarios to time.  A scenario is one cell in the
    # output: pure-mode scenarios fill prev_tokens with one constant before
    # the timing loop; mix-mode scenarios feed a pre-baked per-iter samples
    # tensor, with the per-iter copy captured inside the CUDA graph.  Pure
    # and mix can coexist in one call so a single nsys trace covers both.
    scenarios = []
    for prev_k in prev_ks:
        # On the nowrite path, new tokens append at [prev_k, prev_k+T) of
        # the active buffer, so prev_k+T must fit within max_window.
        # mode != monolithic dispatches per-slot from PNAT, so any
        # prev_k <= max_window is valid for those modes.
        if mode == "monolithic" and not write_checkpoint and prev_k + mtp_len > max_window:
            continue
        scenarios.append({
            "label": f"k{prev_k}",
            "print_label": prev_k,
            "fill": prev_k,
            "pre_iter": None,
            "iters": None,  # use args.iters
        })
    # Mix scenario: skip on monolithic (mono on mixed PNAT corrupts the
    # wrong-mode slots).  Persistent_main + mix is now supported: bench
    # pre-bakes both a per-iter PNAT samples tensor and a per-iter
    # n_writes samples tensor; pre_iter_fn copies row i of each into the
    # kernel-input tensors (PNAT and n_writes_dev) on the same stream as
    # the captured CUDA graph, so they're cold w.r.t. the in-graph L2
    # flush.
    if mix_samples_cpu is not None and mode != "monolithic":
        device = state_work.device
        # Hardcode-sort: per-iter prev_tokens are CPU-sorted write-first.
        # Kernel runs USE_PERM=False but the EO gate sees clustered modes.
        # Output is scrambled (we don't permute x/B/C/dt to match) but
        # timing is meaningful — isolates clustering benefit from the
        # per-program perm-load overhead in --sort-slots.
        src = mix_samples_sorted_cpu if (hardcode_sort and mix_samples_sorted_cpu is not None) else mix_samples_cpu
        samples_gpu = torch.from_numpy(src).to(device=device, dtype=torch.int32)

        # For persistent_main + mix: pre-compute the per-iter n_writes
        # (count of slots needing the write path = PNAT+T > max_window)
        # and the (1,) scratch the kernel reads from.  Both halves of
        # persistent_main always launch in mix scenarios (host can't
        # cheaply read n_writes per iter without a sync), so the kernel's
        # slot-range derivation must be correct from device n_writes.
        # persistent_dynamic doesn't need n_writes (the kernel ignores
        # n_writes_dev when IS_DYNAMIC=True via Triton DCE), but we still
        # allocate a sentinel scratch so the wrapper API is uniform.
        n_writes_samples_gpu = None
        n_writes_dev_mix = None
        if mode in ("persistent_main", "persistent_dynamic"):
            # n_writes per iter = number of slots that overflow the window.
            n_writes_per_iter = ((src + mtp_len) > max_window).sum(axis=1).astype(np.int32)
            n_writes_samples_gpu = torch.from_numpy(n_writes_per_iter).to(
                device=device, dtype=torch.int32
            )
            n_writes_dev_mix = torch.zeros(1, dtype=torch.int32, device=device)

        # Build _mix_pre_iter — the closure that runs OUTSIDE the captured
        # graph between replays.  Updates: prev_tokens (always),
        # slot_perm_buf (when sort_slots), n_writes_dev_mix (persistent).
        if sort_slots and perm_samples_cpu is not None:
            perm_samples_gpu = torch.from_numpy(perm_samples_cpu).to(
                device=device, dtype=torch.int32
            )
            if n_writes_samples_gpu is not None:
                def _mix_pre_iter(i, _s=samples_gpu, _ps=perm_samples_gpu,
                                  _ns=n_writes_samples_gpu, _pt=prev_tokens,
                                  _pm=slot_perm_buf, _nw=n_writes_dev_mix):
                    _pt.copy_(_s[i])
                    _pm.copy_(_ps[i])
                    _nw.copy_(_ns[i:i+1])
            else:
                def _mix_pre_iter(i, _s=samples_gpu, _ps=perm_samples_gpu,
                                  _pt=prev_tokens, _pm=slot_perm_buf):
                    _pt.copy_(_s[i])
                    _pm.copy_(_ps[i])
        else:
            if n_writes_samples_gpu is not None:
                def _mix_pre_iter(i, _s=samples_gpu, _ns=n_writes_samples_gpu,
                                  _pt=prev_tokens, _nw=n_writes_dev_mix):
                    _pt.copy_(_s[i])
                    _nw.copy_(_ns[i:i+1])
            else:
                def _mix_pre_iter(i, _s=samples_gpu, _pt=prev_tokens):
                    _pt.copy_(_s[i])

        # Mix iters override: if --mix-iters set, use it; else use args.iters.
        mix_iters = getattr(args, "mix_iters", None)
        scenarios.append({
            "label": f"mix{mix_label}",
            "print_label": "mix",
            "fill": None,
            "pre_iter": _mix_pre_iter,
            "iters": mix_iters,  # None => use args.iters
            # Pass through to _run_incr so the wrapper receives _n_writes_dev
            # (mix scenarios) instead of _n_writes (pure scenarios).
            "n_writes_dev": n_writes_dev_mix,
        })

    # Pure scenarios don't pre-allocate n_writes_dev; mix scenarios do.
    # Default empty-halves skip: True for pure (host knows n_writes,
    # production-equivalent host-skip), False for mix (host can't read
    # device n_writes per iter without sync, must always launch both).
    for scn in scenarios:
        scenario_n_writes_dev = scn.get("n_writes_dev")  # None for pure
        scenario_skip_empty = scenario_n_writes_dev is None
        if scn["fill"] is not None:
            prev_tokens.fill_(scn["fill"])
        prev_k_for_print = scn["print_label"]
        scenario_pre_iter = scn["pre_iter"]
        scenario_iters = scn.get("iters")  # None => use args.iters
        tag = f"incr_b{batch}_mtp{mtp_len}_{scn['label']}_s{state_dtype_name}_a{act_dtype_name}"

        for (
            block_size_m,
            num_warps,
            num_stages,
            precompute_num_warps,
            precompute_num_stages,
            heads_per_block,
            maxnreg,
            num_ctas,
            cta_per_sm,
            num_loop_stages,
            flatten,
            warp_specialize,
            use_tma_rect_load,
            use_tma_replay_write_load,
            use_tma_replay_nowrite_load,
            use_tma_replay_write_store,
        ) in itertools.product(
            block_size_m_values,
            num_warps_values,
            num_stages_values,
            precompute_num_warps_values,
            precompute_num_stages_values,
            heads_per_block_values,
            maxnreg_values,
            num_ctas_values,
            cta_per_sm_values,
            num_loop_stages_values,
            flatten_values,
            warp_specialize_values,
            use_tma_rect_load_values,
            use_tma_replay_write_load_values,
            use_tma_replay_nowrite_load_values,
            use_tma_replay_write_store_values,
        ):
            # Skip-dupe for TMA flag sweeps: a flag whose code path isn't
            # reachable in this cell produces identical timing for value=0
            # and value=1.  We canonicalize by skipping value=1 cells when
            # the flag's path is unreachable.  Path reachability rules:
            #   * write path (replay write-load + write-store): mono+WC=True,
            #     OR any non-monolithic mode.
            #   * rect path (rect-load): rectangle_for_nowrite=True AND a
            #     nowrite path exists in this mode (mono+WC=False, OR any
            #     non-monolithic mode).
            #   * replay-nowrite path (nowrite-load): nowrite path exists
            #     AND rect isn't taking it: mono+WC=False+rect=False, OR
            #     any non-monolithic mode with rect=False.
            _is_mono = (mode == "monolithic")
            _write_path = (_is_mono and write_checkpoint) or (not _is_mono)
            _rect_path = rectangle_for_nowrite and (
                (not _is_mono) or (_is_mono and not write_checkpoint)
            )
            _replay_nowrite_path = (
                (_is_mono and not write_checkpoint and not rectangle_for_nowrite)
                or ((not _is_mono) and not rectangle_for_nowrite)
            )
            def _set(v):  # flag set to a non-zero sweep value
                return v is not None and v != 0
            if (_set(use_tma_rect_load) and not _rect_path
                    or _set(use_tma_replay_write_load) and not _write_path
                    or _set(use_tma_replay_nowrite_load) and not _replay_nowrite_path
                    or _set(use_tma_replay_write_store) and not _write_path):
                continue

            # Pre-allocate n_writes_dev tensor OUTSIDE the captured graph for
            # persistent modes in pure scenarios.  Mix scenarios already have
            # `scenario_n_writes_dev` pre-allocated.  The wrapper's fallback
            # `torch.tensor([...], device=...)` allocation would invalidate
            # the CUDA-graph capture stream — must allocate here, before the
            # `_run_incr` lambda (which is what gets captured) is defined.
            # For persistent_dynamic the kernel ignores the value (IS_DYNAMIC
            # DCE's the load); we still need a valid pointer.  For
            # persistent_main pure, the value is constant per cell so we set
            # it once here.
            _n_writes_dev_pure: torch.Tensor | None = None
            _host_n_writes_pure: int | None = None
            if mode in ("persistent_main", "persistent_dynamic") and scenario_n_writes_dev is None:
                _n_writes_dev_pure = torch.zeros(1, dtype=torch.int32, device=state_work.device)
                if mode == "persistent_main":
                    scn_fill = scn["fill"]
                    is_write_scenario_local = (scn_fill + mtp_len) > max_window
                    _host_n_writes_pure = batch if is_write_scenario_local else 0
                    _n_writes_dev_pure.fill_(_host_n_writes_pure)

            def _run_incr(
                block_size_m=block_size_m,
                num_warps=num_warps,
                num_stages=num_stages,
                precompute_num_warps=precompute_num_warps,
                precompute_num_stages=precompute_num_stages,
                heads_per_block=heads_per_block,
                maxnreg=maxnreg,
                num_ctas=num_ctas,
                cta_per_sm=cta_per_sm,
                num_loop_stages=num_loop_stages,
                flatten=flatten,
                warp_specialize=warp_specialize,
                use_tma_rect_load=use_tma_rect_load,
                use_tma_replay_write_load=use_tma_replay_write_load,
                use_tma_replay_nowrite_load=use_tma_replay_nowrite_load,
                use_tma_replay_write_store=use_tma_replay_write_store,
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
                    extra_kwargs["mode"] = mode
                    if sort_slots:
                        extra_kwargs["slot_perm"] = slot_perm_buf
                    # reverse_nowrite is meaningful in two ways:
                    #   - with slot_perm: walk the perm tail-first
                    #   - without slot_perm (hardcode-sort): walk pid_b
                    #     itself tail-first via the REVERSE_PERM constexpr
                    if sort_slots or (hardcode_sort and reverse_nowrite):
                        extra_kwargs["reverse_nowrite"] = reverse_nowrite
                    if state_scales_work is not None:
                        extra_kwargs["state_scales"] = state_scales_work
                    if use_tma_rect_load:  # 1 → True, 0/None → False
                        extra_kwargs["_use_tma_rect_load"] = True
                    if use_tma_replay_write_load:
                        extra_kwargs["_use_tma_replay_write_load"] = True
                    if use_tma_replay_nowrite_load:
                        extra_kwargs["_use_tma_replay_nowrite_load"] = True
                    if use_tma_replay_write_store:
                        extra_kwargs["_use_tma_replay_write_store"] = True
                    # persistent_main needs n_writes (count of write-mode
                    # slots in the pre-sorted batch) as a host-side int.
                    # Pure scenarios: every slot has the same PNAT, so
                    # n_writes is either 0 (all nowrite) or batch (all
                    # write) depending on whether PNAT+T overflows the
                    # window.  Mix scenarios are skipped earlier.
                    if mode in ("persistent_main", "persistent_dynamic"):
                        # Per-cell sweep values for persistent-only knobs.
                        # Apply to both persistent variants.  _parse_sweep
                        # returns [None] when the user didn't pass the flag,
                        # in which case we leave the wrapper's defaults.
                        if cta_per_sm is not None:
                            extra_kwargs["_cta_per_sm"] = cta_per_sm
                        if num_loop_stages is not None:
                            extra_kwargs["_num_loop_stages"] = num_loop_stages
                        if flatten is not None:
                            extra_kwargs["_flatten"] = bool(flatten)
                        if warp_specialize is not None:
                            extra_kwargs["_warp_specialize"] = bool(warp_specialize)
                    if mode in ("persistent_main", "persistent_dynamic"):
                        # persistent_main + mix REQUIRES sort: the kernel
                        # partitions slots [0, n_writes) = write half,
                        # [n_writes, batch) = nowrite half.  This only holds
                        # if PNAT is monotone (writes first), which sort
                        # provides via either:
                        #   sort_slots=1 → USE_PERM reads slot_perm to remap
                        #   hardcode_sort=1 → PNAT itself is CPU-pre-sorted
                        # persistent_dynamic doesn't need sort (per-slot
                        # runtime dispatch); persistent_main pure scenarios
                        # are trivially sorted (homogeneous PNAT).
                        if (mode == "persistent_main"
                                and scenario_n_writes_dev is not None
                                and not (sort_slots or hardcode_sort)):
                            raise AssertionError(
                                "persistent_main + mix requires sort_slots=1 "
                                "or hardcode_sort=1 — kernel partitions slots "
                                "by index, which is only valid when PNAT is "
                                "monotone (writes first).  Without sort, the "
                                "partition silently mismatches actual slot "
                                "modes.  Re-run with --sort-slots 1 or "
                                "--hardcode-sort 1."
                            )
                        # n_writes plumbing: pure scenarios pass an int
                        # (host knows the value, can host-skip empty halves);
                        # mix scenarios pass a (1,) device tensor that the
                        # bench's pre_iter_fn updates per replay.
                        # _persistent_skip_empty_halves=False on mix so both
                        # halves always launch (kernel uses device n_writes
                        # to derive its slot range).
                        if scenario_n_writes_dev is not None:
                            # Mix path: caller-allocated tensor, updated per
                            # iter by scenario_pre_iter outside capture.
                            extra_kwargs["_n_writes_dev"] = scenario_n_writes_dev
                            extra_kwargs["_persistent_skip_empty_halves"] = False
                        elif mode == "persistent_main":
                            # Pure: caller pre-allocated `_n_writes_dev_pure`
                            # outside this lambda (so the alloc doesn't land
                            # inside the captured graph).  Pass both the
                            # tensor and the host int so the wrapper can use
                            # host-skip when `_persistent_skip_empty_halves`.
                            extra_kwargs["_n_writes"] = _host_n_writes_pure
                            extra_kwargs["_n_writes_dev"] = _n_writes_dev_pure
                            extra_kwargs["_persistent_skip_empty_halves"] = scenario_skip_empty
                        elif mode == "persistent_dynamic":
                            # persistent_dynamic pure: kernel ignores n_writes
                            # via IS_DYNAMIC DCE, but the wrapper needs a
                            # valid (1,) tensor pointer.  Pass the pre-allocated
                            # zero tensor to avoid any in-capture alloc.
                            extra_kwargs["_n_writes_dev"] = _n_writes_dev_pure
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
            # Persistent-only knobs (only meaningful when MODE=persistent_main;
            # printed unconditionally so output rows are uniformly comparable
            # across modes when the user passed these sweeps).
            if cta_per_sm is not None:
                parts.append(f"CPS={cta_per_sm}")
            if num_loop_stages is not None:
                parts.append(f"LS={num_loop_stages}")
            if flatten is not None:
                parts.append(f"FL={flatten}")
            if warp_specialize is not None:
                parts.append(f"WS={warp_specialize}")
            # TMA sweep tags.  Four wrapper-level flags map to three
            # kernel-level constexprs (rect-load and replay-nowrite-load
            # share `USE_TMA_LOAD_NOWRITE`, picked by the wrapper based on
            # RECTANGLE).  TMARL specifically gates the rectangle path's
            # state load; TMANL specifically gates the replay-style
            # nowrite path's state load.  Distinct because their measured
            # perf profiles differ (see CHECKPOINTING_DESIGN.md item #17:
            # rect TMA is "not a win" while replay-nowrite TMA is the
            # biggest measured win at int8 b>=64).
            if use_tma_rect_load is not None:
                parts.append(f"TMARL={use_tma_rect_load}")    # rect path load
            if use_tma_replay_write_load is not None:
                parts.append(f"TMAWL={use_tma_replay_write_load}")    # replay-write load
            if use_tma_replay_nowrite_load is not None:
                parts.append(f"TMANL={use_tma_replay_nowrite_load}")  # replay-NOWRITE load (NOT rect)
            if use_tma_replay_write_store is not None:
                parts.append(f"TMAWS={use_tma_replay_write_store}")   # replay-write store
            parts.append(f"SR={1 if use_philox else 0}")
            parts.append(f"RECT={1 if rectangle_for_nowrite else 0}")
            parts.append(f"WC={1 if write_checkpoint else 0}")
            parts.append(f"MODE={mode}")
            parts.append(f"SORT={1 if sort_slots else 0}")
            parts.append(f"REVN={1 if reverse_nowrite else 0}")
            parts.append(f"HSORT={1 if hardcode_sort else 0}")
            sweep_suffix = (" " + ",".join(parts)) if parts else ""
            sweep_tag = tag + sweep_suffix.replace(" ", "_").replace(",", "_")

            reset_fn = _reset_conv1d_realistic if with_conv1d else _reset
            if warmup_only:
                reset_fn()
                if scenario_pre_iter is not None:
                    scenario_pre_iter(0)
                _run_incr()
                torch.cuda.synchronize()
            else:
                stats = _time_kernel(
                    args, _run_incr, reset_fn, sweep_tag,
                    expected_K=_kernels_per_iter_incremental(
                        mode, with_conv1d=with_conv1d,
                        persistent_skip_empty=scenario_skip_empty,
                    ),
                    pre_iter_fn=scenario_pre_iter,
                    iters_override=scenario_iters,
                )

                _print_row(
                    show_kernel_col,
                    args.variant,
                    batch,
                    mtp_len,
                    prev_k_for_print,
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
    modes_list = getattr(args, "modes_list", ["monolithic"])
    sort_list = getattr(args, "sort_slots_list", [False])
    rev_list = getattr(args, "reverse_nowrite_list", [False])
    hsort_list = getattr(args, "hardcode_sort_list", [False])

    # Pre-load AL distribution for mix mode (if --mix-csv set).
    mix_al = None
    mix_label = ""
    if args.mix_csv is not None:
        from pathlib import Path as _Path
        from checkpoint_mix_sim import load_al_distribution as _load_al
        mix_label = _Path(args.mix_csv).stem
        # T (= mtp_len) varies per cell; load once with the LARGEST mtp so
        # we have enough columns; the loader normalizes the dist anyway.
        mix_al = _load_al(_Path(args.mix_csv), T=max(mtp_lengths), column=args.mix_csv_column)

    for batch in batch_sizes:
        for mtp_len in mtp_lengths:
            # Resolve prev_k fractions → clamped integers in [0, mtp_len]
            prev_ks = _resolve_prev_ks(args, mtp_len)

            # Pre-generate mix samples once per (batch, mtp_len) cell so all
            # tuning configs see the same per-iter prev_tokens vectors —
            # tuning differences become signal, mix-noise is shared.
            # Size the sample buffer for the LARGER of args.iters and
            # args.mix_iters since mix scenarios use mix_iters.
            mix_samples_cpu = None
            perm_samples_cpu = None  # per-iter slot perm sorted write-first
            mix_samples_sorted_cpu = None  # per-iter prev_tokens, write-first
            if mix_al is not None:
                from checkpoint_mix_sim import sample_steady_state_pnat as _sample_pnat
                _max_window = getattr(args, "max_window", 0) or mtp_len
                _max_iters = max(args.iters, getattr(args, "mix_iters", None) or args.iters)
                mix_samples_cpu = _sample_pnat(
                    mix_al, T=mtp_len, window=_max_window, batch=batch,
                    K=args.warmup + _max_iters, seed=args.mix_seed,
                )
                if any(sort_list) or any(hsort_list):
                    # write-first stable argsort: kind='stable' preserves
                    # original-slot order within each mode group.
                    write_mask = (
                        mix_samples_cpu + mtp_len > _max_window
                    ).astype(np.int8)  # 1 = write, 0 = nowrite
                    perm_idx = np.argsort(
                        -write_mask, kind="stable", axis=-1
                    ).astype(np.int32)
                    if any(sort_list):
                        perm_samples_cpu = perm_idx
                    if any(hsort_list):
                        # Apply the perm to the prev_tokens samples themselves.
                        # Result row i = mix_samples_cpu[i] reordered such
                        # that write-mode entries come first.
                        mix_samples_sorted_cpu = np.take_along_axis(
                            mix_samples_cpu, perm_idx, axis=-1
                        ).astype(mix_samples_cpu.dtype)

            for state_dtype in state_dtypes:
                for act_dtype in act_dtypes:
                    for sr_mode in sr_modes_list:
                        for mode in modes_list:
                            # Non-monolithic modes ignore write_checkpoint
                            # (per-slot from PNAT) — collapse the sweep so we
                            # don't duplicate identical cells.
                            effective_write_modes = (
                                write_modes_list if mode == "monolithic" else [True]
                            )
                            for write_ckpt in effective_write_modes:
                                # Rectangle is meaningful for: nowrite cells in
                                # monolithic; always for dynamic / doublelaunch
                                # (constexpr knob).
                                if mode == "monolithic":
                                    effective_rect_list = (
                                        [False] if write_ckpt else rect_list
                                    )
                                else:
                                    effective_rect_list = rect_list
                                for rect in effective_rect_list:
                                    # Sort/reverse only meaningful for the
                                    # dl-family early-out kernels AND only
                                    # against the mix scenario (the actual
                                    # sort experiment).  Pure k= scenarios
                                    # under sort=1 would just run a
                                    # USE_PERM=True kernel against an
                                    # identity perm — same data point as
                                    # sort=0 + extra compile.  Skip sort=1
                                    # when no mix is configured; mono /
                                    # dynamic also skip sort=1; reverse=1
                                    # with sort=0 is a no-op (skip).
                                    # Note: "persistent_main" is included in
                                    # the dl-family for sort/hsort sweep
                                    # eligibility — it consumes the same
                                    # slot_perm and benefits from the same
                                    # write-first clustering.  It additionally
                                    # requires _n_writes (count of write
                                    # slots) which the bench computes from
                                    # the pure-scenario PNAT (mix scenarios
                                    # not yet supported for persistent_main).
                                    is_dl_family = mode in (
                                        "doublelaunch", "dlgrouped", "maindl",
                                        "dl_write_only", "persistent_main",
                                        "persistent_dynamic",
                                    )
                                    can_sort = (
                                        is_dl_family and mix_samples_cpu is not None
                                    )
                                    effective_sort_list = (
                                        sort_list if can_sort else [False]
                                    )
                                    effective_hsort_list = (
                                        hsort_list if can_sort else [False]
                                    )
                                    for sort_slots in effective_sort_list:
                                        for hardcode_sort in effective_hsort_list:
                                            # sort_slots and hardcode_sort
                                            # are alternative experiments
                                            # for the same idea — skip the
                                            # combined cell to avoid double
                                            # interpretation.
                                            if sort_slots and hardcode_sort:
                                                continue
                                            # rev=1 is meaningful with EITHER
                                            # sort_slots=1 (perm-based) or
                                            # hardcode_sort=1 (raw pid_b
                                            # subtraction in unsorted-perm
                                            # path).  rev=1 with both 0 is
                                            # a no-op.
                                            effective_rev_list = (
                                                rev_list if (sort_slots or hardcode_sort) else [False]
                                            )
                                            for reverse_nowrite in effective_rev_list:
                                                _bench_config(
                                                    args, batch, mtp_len,
                                                    prev_ks, state_dtype,
                                                    act_dtype, baseline_fn,
                                                    sr_mode=sr_mode,
                                                    rectangle_for_nowrite=rect,
                                                    write_checkpoint=write_ckpt,
                                                    mode=mode,
                                                    mix_samples_cpu=mix_samples_cpu,
                                                    mix_label=mix_label,
                                                    sort_slots=sort_slots,
                                                    reverse_nowrite=reverse_nowrite,
                                                    perm_samples_cpu=perm_samples_cpu,
                                                    hardcode_sort=hardcode_sort,
                                                    mix_samples_sorted_cpu=mix_samples_sorted_cpu,
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
        "--cta-per-sm",
        type=str,
        default=None,
        help="CTAs per SM in the 1D persistent grid for mode=persistent_main "
        "(comma-separated sweep).  num_persistent = cta_per_sm × NUM_SMS.  "
        "Default = 1 (one CTA per SM).  Replaces the old --num-persistent.  "
        "Ignored for non-persistent_main modes.",
    )
    parser.add_argument(
        "--num-loop-stages",
        type=str,
        default=None,
        help="num_stages on the inner tl.range(...) persistent loop for "
        "mode=persistent_main (comma-separated sweep).  Default = 2.  Note: "
        "this is loop-level, NOT the kernel-arg num_stages (which only "
        "pipelines dot-feeding loads).  Watch Triton issue #8259 — "
        "num_stages>1 + flatten=True can corrupt stores in non-dot kernels.  "
        "Ignored for non-persistent_main modes.",
    )
    parser.add_argument(
        "--flatten",
        type=str,
        default=None,
        help="`flatten` arg on tl.range(...) for mode=persistent_main "
        "(comma-separated 0/1 sweep).  Default = 1.  Ignored for "
        "non-persistent_main modes.",
    )
    parser.add_argument(
        "--warp-specialize",
        type=str,
        default=None,
        help="`warp_specialize` arg on tl.range(...) for mode=persistent_main "
        "(comma-separated 0/1 sweep).  Default = 0.  Triton 3.6 only "
        "supports it on simple matmul loops; our scan loop probably won't "
        "pattern-match — exposed as a knob for sweep experiments.  Requires "
        "num_warps >= 4 if 1.  Ignored for non-persistent_main modes.",
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
        "--use-tma-rect-load",
        type=str,
        default=None,
        help="Comma-separated 0/1 sweep.  Use TMA (host-built tensor "
        "descriptor) for state load in the rectangle nowrite path.  "
        "Cells where the rect path isn't reachable (e.g. mode=monolithic "
        "+ WC=True) skip the value=1 case as a dupe.",
    )
    parser.add_argument(
        "--use-tma-replay-write-load",
        type=str,
        default=None,
        help="Comma-separated 0/1 sweep.  TMA state LOAD in replay main "
        "when WC=True.  Independent from nowrite-load and rect TMA — see "
        "CHECKPOINTING_DESIGN.md item #17 for measured perf.",
    )
    parser.add_argument(
        "--use-tma-replay-nowrite-load",
        type=str,
        default=None,
        help="Comma-separated 0/1 sweep.  TMA state LOAD in replay main "
        "when WC=False.  Design doc reports the largest win on this path "
        "(int8 b>=64: -8 to -12%%).",
    )
    parser.add_argument(
        "--use-tma-replay-write-store",
        type=str,
        default=None,
        help="Comma-separated 0/1 sweep.  TMA state STORE in replay main "
        "(WC=True path only — no-op for WC=False).  Independent from all "
        "load TMA flags.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="monolithic",
        help="Comma-separated dispatch modes to sweep, any of "
        "{monolithic,dynamic,doublelaunch}.  monolithic = today's behavior "
        "(one kernel pair, write_checkpoint applied to whole batch); "
        "dynamic = single kernel pair that dispatches per-slot at runtime "
        "based on PNAT (rectangle_for_nowrite picks RECTANGLE constexpr); "
        "doublelaunch = two kernel pairs launched in sequence with "
        "EARLY_OUT=True, each handling slots whose mode matches it.  "
        "Only applies to the checkpointing variant; non-monolithic modes "
        "ignore --write-modes (per-slot from PNAT).",
    )
    parser.add_argument(
        "--mix-csv",
        type=str,
        default=None,
        help="Path to AL histogram CSV (cols: AL, count).  When set, an "
        "additional 'mix' cell is emitted per (batch, mtp, dtype, sr, "
        "mode, RECT, M, W, ...) combo where prev_tokens varies per iter, "
        "drawn from the steady-state PNAT distribution induced by the "
        "AL histogram.  Mix cells run only on dynamic and doublelaunch "
        "modes (mono on a mixed batch corrupts wrong-mode slots).  "
        "Each iteration of the captured CUDA graph has a different "
        "pre-baked prev_tokens vector; warmup iters use distinct samples "
        "from the timed iters so nsys-included warmup leaks don't bias.",
    )
    parser.add_argument(
        "--mix-csv-column",
        type=int,
        default=1,
        help="Column index (0-based) in the AL histogram CSV for the "
        "count/probability column.  Default 1 (second column).",
    )
    parser.add_argument(
        "--mix-seed",
        type=int,
        default=42,
        help="RNG seed for the steady-state PNAT sampler.  Same seed "
        "across runs => same per-slot samples for reproducible "
        "comparisons.",
    )
    parser.add_argument(
        "--sort-slots",
        type=str,
        default="0",
        help="Comma-separated 0/1.  When 1, mix scenarios pre-sort slots "
        "write-first (write slots at the head of slot_perm, nowrite at the "
        "tail) and the dl-family kernels read pid_b through that perm — "
        "clusters early-outs at one end of the grid.  Only meaningful for "
        "doublelaunch/dlgrouped/maindl with mix scenarios; mono/dynamic "
        "and pure-batch cells skip sort=1.",
    )
    parser.add_argument(
        "--reverse-nowrite",
        type=str,
        default="0",
        help="Comma-separated 0/1.  When 1 (and --sort-slots 1), the "
        "nowrite-side kernels in dlgrouped/doublelaunch/maindl walk the "
        "perm in reverse so both halves of the dl chain front-load real "
        "work.  reverse=1 with sort=0 is skipped (no perm to reverse).",
    )
    parser.add_argument(
        "--hardcode-sort",
        type=str,
        default="0",
        help="Comma-separated 0/1.  When 1, the per-iter prev_tokens "
        "samples are pre-sorted write-first OFFLINE (CPU-side) before "
        "the timed region — kernel runs unchanged (USE_PERM=False) but "
        "the EO gate sees sorted PNAT so early-outs cluster naturally. "
        "Zero per-program load cost vs --sort-slots; output is "
        "scrambled (we don't permute x/B/C/dt) but timing is meaningful. "
        "Used to isolate whether clustering helps independent of the "
        "perm-load overhead in the sort-slots path.",
    )
    parser.add_argument(
        "--mix-iters",
        type=int,
        default=None,
        help="Iteration count override for mix scenarios (each iter is a "
        "different per-slot prev_tokens draw).  Default (None) uses "
        "--iters.  Mix scenarios benefit from more iters since each "
        "iter samples a different mix; pure scenarios don't.",
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

    sort_modes = [v.strip() for v in (args.sort_slots or "0").split(",") if v.strip()]
    sort_list = []
    for v in sort_modes:
        if v not in ("0", "1"):
            parser.error(f"--sort-slots value must be 0 or 1, got {v!r}")
        sort_list.append(v == "1")
    args.sort_slots_list = sort_list

    rev_modes = [v.strip() for v in (args.reverse_nowrite or "0").split(",") if v.strip()]
    rev_list = []
    for v in rev_modes:
        if v not in ("0", "1"):
            parser.error(f"--reverse-nowrite value must be 0 or 1, got {v!r}")
        rev_list.append(v == "1")
    args.reverse_nowrite_list = rev_list

    hsort_modes = [v.strip() for v in (args.hardcode_sort or "0").split(",") if v.strip()]
    hsort_list = []
    for v in hsort_modes:
        if v not in ("0", "1"):
            parser.error(f"--hardcode-sort value must be 0 or 1, got {v!r}")
        hsort_list.append(v == "1")
    args.hardcode_sort_list = hsort_list

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

    modes_raw = [v.strip() for v in args.modes.split(",") if v.strip()]
    valid_modes = {
        "monolithic", "dynamic", "doublelaunch", "dlgrouped", "maindl",
        "dl_write_only", "persistent_main", "persistent_dynamic",
    }
    for m in modes_raw:
        if m not in valid_modes:
            parser.error(
                f"--modes value must be one of {sorted(valid_modes)}, got {m!r}"
            )
    args.modes_list = modes_raw or ["monolithic"]
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
