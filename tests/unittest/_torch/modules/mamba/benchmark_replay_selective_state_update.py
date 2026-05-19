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

Baseline kernel (--baseline flashinfer_pr3324):
  Calls FlashInfer PR 3324's checkpointing_ssu kernel against the same replay
  cache tensors as the replay kernel.

Timing methodology
==================

All in-bench timing comes from CUPTI's Activity API (1 ns kernel
timestamps from the GPU profiling fabric).  cudaEvent.elapsed_time() was
removed — its ~0.5 us resolution overshoots CUPTI by ~50% on short kernels
in graphs, and we have no other use for it here.  See the CUPTI block
lower in this file for the timer source.

Three modes:

  --cupti --cuda-graph (default)
      Capture a small CUDA graph for the cell, replay it for warmup + timed
      iterations, and read kernel start/end from CUPTI.  Raw CUPTI buffers are
      parsed out-of-process on the timed path, with a cached ordinal plan used
      to keep only the kernels we care about.

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
    "metadata": {timestamp, cmd, tp_size, warmup, iters, cupti},
    "results": {
      "<key>": {median, p95, p99, n, iters_us, [n_writes_per_iter],
                [kmix_bucket_score], [per_kernel]}
    }
  }

Key format mirrors collect.py's kernel_data.json convention:
  incremental/{batch}/{mtp}/{sd}/k{prev_k}/{sweep_parts}/tp{tp}
  flashinfer_pr3324/{batch}/{mtp}/{sd}/k{prev_k}/{sweep_parts}/tp{tp}

  - <sd> is normalized: bf16 / fp16 / fp32 / int8 / int16 / fp8.
  - <sweep_parts> is e.g. "M16_W1_S3_SR0_RECT0" — flags concatenated by
    underscore in canonical order. pS/R/CT appear only when explicitly swept.
  - All numeric values in microseconds (us).

Per-record fields:
  - median, p95, p99: span statistics (us).  Span = max(kernel_end_ns) -
    min(kernel_start_ns) across the iter's kernels — same convention as
    nsys-derived collect.py used to use.
  - n: number of timed iters that contributed.
  - iters_us: list of length n, raw per-iter spans.
  - n_writes_per_iter: for mix rows, list of length n with the number of
    write-path slots in each timed iteration.
  - kmix_bucket_score: for mix rows, binomial-PDF-weighted expected time
    from per-n_writes bucket medians.
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
import atexit
import csv
import ctypes
import importlib
import itertools
import json
import math
import multiprocessing as mp
import os
import queue
import statistics
import sys
import threading
import time
from datetime import datetime
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import torch
from einops import repeat

REPLAY_WORK_POSITION_IN_DECODE_BATCH = 0
REPLAY_WORK_CACHE_SLOT = 1
REPLAY_WORK_PNAT = 2
REPLAY_WORK_CACHE_BUF_IDX = 3
REPLAY_WORK_ITEM_WIDTH = 4

DEFAULT_PMIX_T = 6
DEFAULT_PMIX_LABEL = "pmix_T6"
DEFAULT_PMIX_REPLAY_COUNTS = {
    1: 539622,
    2: 329144,
    3: 336224,
    4: 495832,
    5: 444398,
    6: 1007008,
}


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

    utils_stub = types.ModuleType("tensorrt_llm._utils")

    def _fast_get_sm_version():
        prop = torch.cuda.get_device_properties(0)
        return prop.major * 10 + prop.minor

    utils_stub.get_sm_version = _fast_get_sm_version
    sys.modules["tensorrt_llm._utils"] = utils_stub

    def _load(mod_name: str, file_name: str):
        fqn = f"{mamba_pkg}.{mod_name}" if mod_name else mamba_pkg
        if fqn in sys.modules:
            return sys.modules[fqn]
        kwargs = {}
        if file_name == "__init__.py":
            kwargs["submodule_search_locations"] = [str(mamba_dir)]
        spec = importlib.util.spec_from_file_location(fqn, mamba_dir / file_name, **kwargs)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[fqn] = mod
        spec.loader.exec_module(mod)
        return mod

    # 1. Package __init__ (defines PAD_SLOT_ID = -1)
    _load("", "__init__.py")
    # 2. softplus helper (used by both kernel modules)
    _load("softplus", "softplus.py")
    # 3. replay_selective_state_update only needs replay-work-item constants
    # from mamba2_metadata at import time.  Stub those constants instead of
    # importing the full metadata module and its scheduler/attention deps.
    metadata_stub = types.ModuleType(f"{mamba_pkg}.mamba2_metadata")
    metadata_stub.REPLAY_WORK_POSITION_IN_DECODE_BATCH = (
        REPLAY_WORK_POSITION_IN_DECODE_BATCH
    )
    metadata_stub.REPLAY_WORK_CACHE_SLOT = REPLAY_WORK_CACHE_SLOT
    metadata_stub.REPLAY_WORK_PNAT = REPLAY_WORK_PNAT
    metadata_stub.REPLAY_WORK_CACHE_BUF_IDX = REPLAY_WORK_CACHE_BUF_IDX
    metadata_stub.REPLAY_WORK_ITEM_WIDTH = REPLAY_WORK_ITEM_WIDTH
    sys.modules[f"{mamba_pkg}.mamba2_metadata"] = metadata_stub
    # 4. The actual kernels
    replay_mod = _load("replay_selective_state_update", "replay_selective_state_update.py")
    conv1d_mod = _load("causal_conv1d_triton", "causal_conv1d_triton.py")

    return (
        replay_mod.replay_selective_state_update,
        replay_mod._resolve_tuning,
        conv1d_mod.causal_conv1d_update,
    )


def _import_mamba_kernels_full():
    """Import via the standard tensorrt_llm package (slow but safe)."""
    replay_mod = importlib.import_module(
        "tensorrt_llm._torch.modules.mamba.replay_selective_state_update"
    )
    from tensorrt_llm._torch.modules.mamba.causal_conv1d_triton import causal_conv1d_update

    return (
        replay_mod.replay_selective_state_update,
        replay_mod._resolve_tuning,
        causal_conv1d_update,
    )


# Use fast import by default; --full-import parsed later but we need the
# functions at module level.  Check sys.argv early.
if "--full-import" in sys.argv:
    (
        replay_selective_state_update,
        resolve_replay_tuning,
        causal_conv1d_update,
    ) = _import_mamba_kernels_full()
else:
    try:
        (
            replay_selective_state_update,
            resolve_replay_tuning,
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


def _al_counts_to_distribution(
    al_to_count: dict[int, float],
    T: int,
    label: str,
) -> np.ndarray:
    """Return a length-(T + 1) probability vector indexed by accepted length."""
    if not al_to_count:
        raise SystemExit(f"no numeric AL rows found in {label}")
    if max(al_to_count) > T:
        raise SystemExit(
            f"AL histogram {label} contains accepted length {max(al_to_count)} "
            f"> T={T}"
        )
    if min(al_to_count) < 0:
        raise SystemExit(f"AL histogram {label} contains negative accepted lengths")
    if al_to_count.get(0, 0.0) > 0.0:
        print(
            f"[WARN] {label} contains AL=0 mass; spec decoding normally "
            "accepts at least the target token.",
            file=sys.stderr,
        )

    dist = np.zeros(T + 1, dtype=np.float64)
    for accepted_length, count in al_to_count.items():
        dist[accepted_length] = count
    total = dist.sum()
    if total == 0.0:
        raise SystemExit(f"AL distribution sums to zero in {label}")
    return dist / total


def _load_al_distribution(path: Path, T: int, column: int = 1) -> np.ndarray:
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise SystemExit(f"empty CSV: {path}")

    al_to_count: dict[int, float] = {}
    for row in rows:
        if not row or not row[0].strip():
            continue
        try:
            accepted_length = int(float(row[0]))
            count = float(row[column])
        except (IndexError, ValueError):
            continue
        al_to_count[accepted_length] = (
            al_to_count.get(accepted_length, 0.0) + count
        )

    return _al_counts_to_distribution(al_to_count, T, str(path))


def _load_builtin_pmix_distribution(T: int) -> np.ndarray:
    if T != DEFAULT_PMIX_T:
        raise SystemExit(
            f"--pmix uses the built-in T{DEFAULT_PMIX_T} histogram; "
            f"use --mtp-lengths {DEFAULT_PMIX_T} or pass --mix-csv for another T."
        )
    return _al_counts_to_distribution(
        DEFAULT_PMIX_REPLAY_COUNTS,
        T,
        f"built-in {DEFAULT_PMIX_LABEL}",
    )


def _markov_stationary(al_dist: np.ndarray, T: int, window: int) -> np.ndarray:
    """Stationary PNAT distribution for the replay-window Markov chain."""
    n_states = window + 1
    transition = np.zeros((n_states, n_states), dtype=np.float64)
    for pnat in range(n_states):
        is_write = pnat + T > window
        for accepted_length in range(1, T + 1):
            prob = al_dist[accepted_length]
            if prob == 0.0:
                continue
            next_pnat = accepted_length if is_write else pnat + accepted_length
            assert 0 <= next_pnat <= window, (
                f"unreachable transition: pnat={pnat} al={accepted_length} "
                f"write={is_write} window={window}"
            )
            transition[pnat, next_pnat] += prob

    eigvals, eigvecs = np.linalg.eig(transition.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    if abs(eigvals[idx] - 1.0) > 1e-6:
        raise SystemExit(
            "stationary distribution eigensolve failed: closest eigenvalue "
            f"to 1 is {eigvals[idx]}"
        )

    pi = np.real(eigvecs[:, idx])
    pi = np.maximum(pi, 0.0)
    if pi.sum() == 0.0:
        pi = np.ones(n_states, dtype=np.float64) / n_states
    for _ in range(2000):
        new_pi = pi @ transition
        if np.allclose(new_pi, pi, atol=1e-12, rtol=0):
            pi = new_pi
            break
        pi = new_pi
    return pi / pi.sum()


def _sample_steady_state_pnat(
    al_dist: np.ndarray,
    T: int,
    window: int,
    batch: int,
    K: int,
    seed: int = 42,
) -> np.ndarray:
    pi = _markov_stationary(al_dist, T, window)
    rng = np.random.default_rng(seed)
    return rng.choice(
        np.arange(window + 1, dtype=np.int64),
        size=(K, batch),
        p=pi,
    ).astype(np.int32)


def _build_replay_work_items_cpu(
    pnat_samples: np.ndarray,
    T: int,
    window: int,
    cache_buf_idx_samples: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build write-first replay work items for one or more PNAT sample rows."""
    samples = np.asarray(pnat_samples, dtype=np.int32)
    squeeze = samples.ndim == 1
    if squeeze:
        samples = samples[None, :]
    if samples.ndim != 2:
        raise ValueError(f"pnat_samples must be 1D or 2D, got shape {samples.shape}")

    n_samples, batch = samples.shape
    write_mask = samples + T > window
    order = np.argsort(-write_mask.astype(np.int8), kind="stable", axis=1)
    positions = np.broadcast_to(
        np.arange(batch, dtype=np.int32), (n_samples, batch)
    )

    if cache_buf_idx_samples is None:
        cache_buf_idx_samples = np.zeros_like(samples, dtype=np.int32)
    else:
        cache_buf_idx_samples = np.asarray(cache_buf_idx_samples, dtype=np.int32)
        if cache_buf_idx_samples.ndim == 1:
            cache_buf_idx_samples = np.broadcast_to(
                cache_buf_idx_samples[None, :], samples.shape
            )
        if cache_buf_idx_samples.shape != samples.shape:
            raise ValueError(
                "cache_buf_idx_samples shape must match pnat_samples, got "
                f"{cache_buf_idx_samples.shape} and {samples.shape}"
            )

    work_items = np.empty(
        (n_samples, batch, REPLAY_WORK_ITEM_WIDTH), dtype=np.int32
    )
    work_items[:, :, REPLAY_WORK_POSITION_IN_DECODE_BATCH] = np.take_along_axis(
        positions, order, axis=1
    )
    work_items[:, :, REPLAY_WORK_CACHE_SLOT] = work_items[
        :, :, REPLAY_WORK_POSITION_IN_DECODE_BATCH
    ]
    work_items[:, :, REPLAY_WORK_PNAT] = np.take_along_axis(samples, order, axis=1)
    work_items[:, :, REPLAY_WORK_CACHE_BUF_IDX] = np.take_along_axis(
        cache_buf_idx_samples, order, axis=1
    )
    n_writes = write_mask.sum(axis=1).astype(np.int32)

    if squeeze:
        return n_writes[:1], work_items[0]
    return n_writes, work_items


# Tensor construction helpers

# Module-level cache for tensor buffers shared across cells.  Keyed by all
# the "fixed" dimensions (state_dtype, act_dtype, max_window, mtp_len,
# nheads, head_dim, d_state, ngroups).  Within a key, the batch dim grows
# in place: if a new cell requests a batch <= cached max_batch, we return
# views (slices) of the existing tensors; if batch > cached max_batch, we
# realloc at the new batch (which becomes the new max).  Tensors never shrink.
#
# Rationale: torch.randn/zeros for these tensor shapes at b=512 takes
# ~10-30ms per call.  At ~895 cells/min with 5 different batch sizes,
# we were re-allocating every cell.  Caching saves the bulk of that per-cell
# overhead, raising GPU util in the timing phase.
#
# Reset state lives in caller (state_work = state0.copy_), so cached state0
# is purely a reference whose contents stay fixed once allocated.  This is
# fine: it's only read by the reset path.
_TENSOR_CACHE: dict = {}


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
      replay_work_items        : packed per-slot replay metadata (batch, 4)
      out_incr                 : pre-allocated output for replay kernel (batch, mtp_len, nheads, head_dim)
      out_base                 : pre-allocated output for baseline kernel   (batch, mtp_len, nheads, head_dim)
    """
    device = "cuda"

    # Cache lookup — grow batch in place if needed; else return views.
    cache_key = (state_dtype, act_dtype, max_window, mtp_len,
                 nheads, head_dim, d_state, ngroups)
    cached = _TENSOR_CACHE.get(cache_key)
    if cached is not None and cached["max_batch"] >= batch:
        # Hit — return slices for current batch.
        b = batch
        return (
            cached["state0"][:b],
            cached["state_scales0"][:b] if cached["state_scales0"] is not None else None,
            cached["old_x"][:b],
            cached["old_B"][:b],
            cached["old_dt"][:b],
            cached["old_dA_cumsum"][:b],
            cached["cache_buf_idx"][:b],
            cached["x"][:b],
            cached["dt"][:b],
            cached["B"][:b],
            cached["C"][:b],
            cached["A"],
            cached["dt_bias"],
            cached["D"],
            cached["prev_tokens"][:b],
            cached["replay_work_items"][:b],
            cached["out_incr"][:b],
            cached["out_base"][:b],
            cached["xbc_input"][:b],
            cached["conv_state"][:b],
            cached["conv_weight"],
            cached["conv_bias"],
            cached["d_inner"],
            cached["conv_dim"],
        )

    # Miss or grow.  Allocate at new max_batch (existing data, if any, is
    # released — caller code re-fills via reset paths anyway).  Rebind
    # `batch` locally to alloc_batch so the existing allocation code below
    # uses the larger size; keep request_batch for the final slice.
    request_batch = batch
    alloc_batch = batch if cached is None else max(batch, cached["max_batch"])
    batch = alloc_batch

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
    replay_work_items = torch.empty(
        batch, REPLAY_WORK_ITEM_WIDTH, device=device, dtype=torch.int32
    )
    position_in_decode_batch = torch.arange(batch, device=device, dtype=torch.int32)
    replay_work_items[:, REPLAY_WORK_POSITION_IN_DECODE_BATCH] = (
        position_in_decode_batch
    )
    replay_work_items[:, REPLAY_WORK_CACHE_SLOT] = position_in_decode_batch
    replay_work_items[:, REPLAY_WORK_PNAT] = 0
    replay_work_items[:, REPLAY_WORK_CACHE_BUF_IDX] = 0

    out_incr = torch.zeros(batch, mtp_len, nheads, head_dim, device=device, dtype=act_dtype)
    out_base = torch.zeros(batch, mtp_len, nheads, head_dim, device=device, dtype=act_dtype)

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

    # Store full-batch buffers in cache and return slices at request_batch.
    _TENSOR_CACHE[cache_key] = {
        "max_batch": alloc_batch,
        "state0": state0,
        "state_scales0": state_scales0,
        "old_x": old_x,
        "old_B": old_B,
        "old_dt": old_dt,
        "old_dA_cumsum": old_dA_cumsum,
        "cache_buf_idx": cache_buf_idx,
        "x": x,
        "dt": dt,
        "B": B,
        "C": C,
        "A": A,
        "dt_bias": dt_bias,
        "D": D,
        "prev_tokens": prev_tokens,
        "replay_work_items": replay_work_items,
        "out_incr": out_incr,
        "out_base": out_base,
        "xbc_input": xbc_input,
        "conv_state": conv_state,
        "conv_weight": conv_weight,
        "conv_bias": conv_bias,
        "d_inner": d_inner,
        "conv_dim": conv_dim,
    }
    rb = request_batch
    return (
        state0[:rb],
        state_scales0[:rb] if state_scales0 is not None else None,
        old_x[:rb],
        old_B[:rb],
        old_dt[:rb],
        old_dA_cumsum[:rb],
        cache_buf_idx[:rb],
        x[:rb],
        dt[:rb],
        B[:rb],
        C[:rb],
        A,
        dt_bias,
        D,
        prev_tokens[:rb],
        replay_work_items[:rb],
        out_incr[:rb],
        out_base[:rb],
        xbc_input[:rb],
        conv_state[:rb],
        conv_weight,
        conv_bias,
        d_inner,
        conv_dim,
    )


# =============================================================================
# CUPTI in-process kernel timing
#
# Self-contained module-in-a-file.  Reads kernel start/end timestamps directly
# from the GPU profiling fabric via CUPTI's Activity API (1 ns
# resolution), avoiding two pitfalls of the cuda-events path:
#
#   1. cudaEvent.elapsed_time() resolution (~0.5 us) is too coarse for the
#      short kernels we care about, especially with PDL + cuda graphs at
#      small batch — events recorded inside a graph have proven noisy.
#   2. nsys is the only known accurate alternative, but the
#      profile-export-sqlite-parse pipeline is heavy and out-of-process.
#
# This is functionally equivalent to wrapping each cell in nsys, except it
# runs in the same benchmark process and sends raw activity buffers to a
# parser process instead of materializing Python objects in the CUPTI callback.
# =============================================================================


# Substring match: kernels run_fn launches that we want to time.  Mirrors
# the parser in scripts/.../collect.py so cupti and nsys-based outputs agree.
_CUPTI_KEEP_KERNEL_SUBSTRINGS = (
    "_dynamic_precompute",
    "_persistent_main",
    "selective_scan_update",
    "selective_state_update",
    "causal_conv1d_update",
)


def _kernels_per_iter_incremental(
    mode: str,
    with_conv1d: bool,
) -> int:
    """Expected number of CUPTI-tracked kernels per iter for the incremental
    kernel chain, given the dispatch mode and the conv1d flag.

    Used to validate CUPTI record counts (no auto-inference — silent
    mis-timing is the failure mode we're guarding against).

    `persistent_main` always launches both write and nowrite halves.  The
    half ranges are derived from device `n_writes` inside the kernel so CUDA
    graphs can replay with changing write counts.
    `persistent_dynamic` always launches 1 main; not affected by the flag.
    """
    if mode == "persistent_dynamic":
        k = 2  # 1 dynamic_precomp + 1 persistent_main
    elif mode == "persistent_main":
        k = 3  # 1 dynamic_precomp + write-main + nowrite-main
    else:
        raise ValueError(
            f"mode must be resolved before CUPTI kernel counting, got {mode!r}"
        )
    if with_conv1d:
        k += 1
    return k


def _dtype_key_for_replay_tuning(dtype: torch.dtype) -> str:
    return {
        torch.float32: "fp32",
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
        torch.int8: "int8",
        torch.int16: "int16",
        torch.float8_e4m3fn: "fp8",
    }.get(dtype, str(dtype))


def _resolve_effective_replay_mode(
    args,
    batch: int,
    state_dtype: torch.dtype,
    use_philox: bool,
    mode: str | None,
) -> str:
    if mode is not None:
        return mode
    table_entry = resolve_replay_tuning(
        batch,
        args.tp_nheads,
        _dtype_key_for_replay_tuning(state_dtype),
        "SR" if use_philox else "RN",
    )
    if table_entry is None:
        return "persistent_dynamic"
    table_mode, _ = table_entry
    return table_mode


def _kernels_per_iter_baseline(with_conv1d: bool) -> int:
    """Expected kernels per iter for the FlashInfer PR3324 baseline.

    The baseline runs a single state-update kernel; `--with-conv1d` prepends
    one conv1d kernel.
    """
    return 2 if with_conv1d else 1


def _is_flashinfer_pr3324_baseline(args) -> bool:
    return getattr(args, "baseline", None) == "flashinfer_pr3324"


def _prepare_flashinfer_jit_workspace() -> None:
    # The container home cache can be read-only under sandboxed runs. Set a
    # writable default before importing flashinfer.jit, which resolves this at
    # import time.
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp/flashinfer")


_LIBCUPTI_CANDIDATES = (
    os.environ.get("CUPTI_LIBRARY_PATH"),
    "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcupti.so.13",
    "libcupti.so.13",
    "libcupti.so",
)
_CUPTI_SUCCESS = 0
_CUPTI_ERROR_MAX_LIMIT_REACHED = 12
_CUPTI_ERROR_INVALID_KIND = 21
_CUPTI_ACTIVITY_KIND_KERNEL = 3
_CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL = 10
_CUPTI_ACTIVITY_ATTR_ZEROED_OUT_ACTIVITY_BUFFER = 5
_CUPTI_HOST_BUFFER_BYTES = 1024 * 1024
_CUPTI_HOST_BUFFER_COUNT = 16

# Multiprocessing start method for compile-warmup + CUPTI parser children.
# Set in __main__ from --mp-start-method.  "spawn" (default) is robust; each
# child re-imports torch/triton/etc (~15s).  "forkserver" preloads once and
# forks cheaply (~1s/child) — see __main__ block for the preload setup.
_MP_START_METHOD = "spawn"
_DEFAULT_CUDA_GRAPH_GROUP_ITERS_PURE = 1
_DEFAULT_CUDA_GRAPH_GROUP_ITERS_MIX = 4


def _load_libcupti() -> ctypes.CDLL:
    errors = []
    for candidate in _LIBCUPTI_CANDIDATES:
        if not candidate:
            continue
        try:
            return ctypes.CDLL(candidate)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    raise ImportError("Unable to load libcupti: " + "; ".join(errors))


class _CuptiActivityKernel11Prefix(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("kind", ctypes.c_int),
        ("cache_config", ctypes.c_uint8),
        ("shared_memory_config", ctypes.c_uint8),
        ("registers_per_thread", ctypes.c_uint16),
        ("partitioned_global_cache_requested", ctypes.c_int),
        ("partitioned_global_cache_executed", ctypes.c_int),
        ("start", ctypes.c_uint64),
        ("end", ctypes.c_uint64),
        ("completed", ctypes.c_uint64),
        ("device_id", ctypes.c_uint32),
        ("context_id", ctypes.c_uint32),
        ("stream_id", ctypes.c_uint32),
        ("grid_x", ctypes.c_int32),
        ("grid_y", ctypes.c_int32),
        ("grid_z", ctypes.c_int32),
        ("block_x", ctypes.c_int32),
        ("block_y", ctypes.c_int32),
        ("block_z", ctypes.c_int32),
        ("static_shared_memory", ctypes.c_int32),
        ("dynamic_shared_memory", ctypes.c_int32),
        ("local_memory_per_thread", ctypes.c_uint32),
        ("local_memory_total", ctypes.c_uint32),
        ("correlation_id", ctypes.c_uint32),
        ("grid_id", ctypes.c_int64),
        ("name", ctypes.c_void_p),
        ("reserved0", ctypes.c_void_p),
        ("queued", ctypes.c_uint64),
        ("submitted", ctypes.c_uint64),
        ("launch_type", ctypes.c_uint8),
        ("is_shared_memory_carveout_requested", ctypes.c_uint8),
        ("shared_memory_carveout_requested", ctypes.c_uint8),
        ("padding", ctypes.c_uint8),
        ("shared_memory_executed", ctypes.c_uint32),
        ("graph_node_id", ctypes.c_uint64),
    ]


def _configure_cupti_get_next_record(libcupti) -> None:
    libcupti.cuptiActivityGetNextRecord.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    libcupti.cuptiActivityGetNextRecord.restype = ctypes.c_int


def _parse_cupti_buffer_ptr(libcupti, buffer_ptr: int, valid_size: int, *, include_names: bool):
    records = []
    zero_ts_count = 0
    zero_ts_names: dict[str, int] = {}
    record_ptr = ctypes.c_void_p(None)
    while True:
        result = libcupti.cuptiActivityGetNextRecord(
            ctypes.c_void_p(buffer_ptr),
            valid_size,
            ctypes.byref(record_ptr),
        )
        if result == _CUPTI_SUCCESS:
            kind = ctypes.cast(record_ptr, ctypes.POINTER(ctypes.c_int)).contents.value
            if kind not in (_CUPTI_ACTIVITY_KIND_KERNEL, _CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL):
                continue
            kernel = ctypes.cast(record_ptr, ctypes.POINTER(_CuptiActivityKernel11Prefix)).contents
            name = None
            if include_names:
                if kernel.name:
                    name = ctypes.string_at(kernel.name).decode("utf-8", errors="replace")
                else:
                    name = "?"
            if kernel.start == 0 or kernel.end == 0:
                zero_ts_count += 1
                if name is not None:
                    zero_ts_names[name] = zero_ts_names.get(name, 0) + 1
                continue
            if include_names:
                records.append((
                    name,
                    int(kernel.start),
                    int(kernel.end),
                    int(kernel.correlation_id),
                    0,
                    int(kernel.graph_node_id),
                    int(kernel.stream_id),
                ))
            else:
                records.append((
                    int(kernel.start),
                    int(kernel.end),
                    int(kernel.correlation_id),
                    int(kernel.graph_node_id),
                    int(kernel.stream_id),
                ))
        elif result == _CUPTI_ERROR_MAX_LIMIT_REACHED:
            break
        elif result == _CUPTI_ERROR_INVALID_KIND:
            break
        else:
            raise RuntimeError(f"cuptiActivityGetNextRecord failed with CUptiResult={result}")
    return records, zero_ts_count, zero_ts_names


def _apply_cupti_filter_plan(numeric_records, filter_plan):
    if not filter_plan:
        return [
            (None, start, end, corr, 0, graph_node_id, stream_id)
            for start, end, corr, graph_node_id, stream_id in sorted(numeric_records)
        ]

    filtered = []
    replay_idx = 0
    record_idx = 0
    for start, end, corr, graph_node_id, stream_id in sorted(numeric_records):
        if replay_idx >= len(filter_plan):
            break
        records_per_replay, ordinal_names = filter_plan[replay_idx]
        if record_idx < len(ordinal_names):
            name = ordinal_names[record_idx]
            if name is not None:
                filtered.append((name, start, end, corr, 0, graph_node_id, stream_id))
        record_idx += 1
        if record_idx >= records_per_replay:
            replay_idx += 1
            record_idx = 0
    return filtered


def _cupti_parser_worker(input_queue, output_queue, ready_event) -> None:
    libcupti = _load_libcupti()
    _configure_cupti_get_next_record(libcupti)
    shared_blocks: dict[str, shared_memory.SharedMemory] = {}
    records_by_generation: dict[int, list[tuple[int, int, int, int, int]]] = {}
    zero_ts_by_generation: dict[int, int] = {}
    ready_event.set()
    while True:
        item = input_queue.get()
        if item is None:
            break
        kind = item[0]
        if kind == "buffer":
            _, generation, buffer_id, name, valid_size = item
            shm = shared_blocks.get(name)
            if shm is None:
                shm = shared_memory.SharedMemory(name=name)
                shared_blocks[name] = shm
            shared_char = ctypes.c_char.from_buffer(shm.buf)
            try:
                parser_ptr = ctypes.addressof(shared_char)
                records, zero_ts_count, _ = _parse_cupti_buffer_ptr(
                    libcupti,
                    parser_ptr,
                    valid_size,
                    include_names=False,
                )
                records_by_generation.setdefault(generation, []).extend(records)
                zero_ts_by_generation[generation] = zero_ts_by_generation.get(generation, 0) + zero_ts_count
                ctypes.memset(parser_ptr, 0, len(shm.buf))
            except Exception as exc:  # pragma: no cover - diagnostic worker path
                output_queue.put({"kind": "error", "generation": generation, "error": repr(exc)})
            finally:
                del shared_char
            output_queue.put({"kind": "buffer_done", "generation": generation, "buffer_id": buffer_id})
        elif kind == "finish":
            if len(item) == 4:
                _, generation, filter_plan, stats_request = item
            else:
                _, generation, filter_plan = item
                stats_request = None
            try:
                raw_records = records_by_generation.pop(generation, [])
                zero_ts_count = zero_ts_by_generation.pop(generation, 0)
                filtered_records = _apply_cupti_filter_plan(raw_records, filter_plan)
                stats = None
                parser_stats_ms = 0.0
                stats_ready = stats_request is not None
                if stats_request is not None:
                    stats_start_s = time.perf_counter()
                    stats = _stats_from_cupti_records(
                        filtered_records,
                        int(stats_request["warmup"]),
                        int(stats_request["iters"]),
                        str(stats_request["tag"]),
                        int(stats_request["expected_K"]),
                        zero_ts_count=zero_ts_count,
                        zero_ts_names={},
                        include_details=bool(stats_request.get("include_details", True)),
                    )
                    parser_stats_ms = 1000.0 * (time.perf_counter() - stats_start_s)
                    filtered_records = []
                output_queue.put({
                    "kind": "finish_done",
                    "generation": generation,
                    "records": filtered_records,
                    "zero_ts_count": zero_ts_count,
                    "zero_ts_names": {},
                    "raw_record_count": len(raw_records),
                    "stats": stats,
                    "stats_ready": stats_ready,
                    "parser_stats_ms": parser_stats_ms,
                })
            except Exception as exc:  # pragma: no cover - diagnostic worker path
                output_queue.put({"kind": "error", "generation": generation, "error": repr(exc)})
        else:
            output_queue.put({"kind": "error", "generation": -1, "error": f"unknown parser message {kind!r}"})
    for shm in shared_blocks.values():
        shm.close()


class CuptiKernelTimer:
    """Raw CUPTI Activity timer with out-of-process parsing for timed runs.

    CUPTI's callback gives us raw activity buffers.  The callback only hands
    shared-memory buffer metadata to a parser process, so the main process
    avoids the cupti-python per-record object creation cost during the timed
    path.  A single local calibration replay may parse names in-process to
    build an ordinal filter plan for a just-captured CUDA graph.
    """

    _instance = None
    _import_error = None

    _request_callback_type = ctypes.CFUNCTYPE(
        None,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    )
    _complete_callback_type = ctypes.CFUNCTYPE(
        None,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
    )

    @classmethod
    def get(cls) -> "CuptiKernelTimer":
        if cls._instance is not None:
            return cls._instance
        if cls._import_error is not None:
            raise cls._import_error
        try:
            cls._instance = cls()
            return cls._instance
        except ImportError as exc:  # pragma: no cover - env-dependent
            cls._import_error = exc
            raise

    def __init__(self) -> None:
        self._libcupti = _load_libcupti()
        self._configure_functions()
        self._lock = threading.Lock()
        self._shared_buffers: dict[int, shared_memory.SharedMemory] = {}
        self._buffer_id_by_ptr: dict[int, int] = {}
        self._free_buffer_ids: list[int] = []
        self._local_completed: list[tuple[int, int]] = []
        self._mode = "drop"
        self._generation = 0
        self._finish_results: dict[int, dict] = {}
        self._parser_errors: list[str] = []
        self._filter_plan = ()
        self._last_start_timing: dict[str, float] = {}
        self._last_stop_timing: dict[str, float] = {}
        self._current_flush_period_ms = 0
        self._mp_ctx = mp.get_context(_MP_START_METHOD)
        # Retry parser-process spawn: concurrent bench instances on the same
        # node race on POSIX named semaphores in /dev/shm — child can die in
        # pickle.load with FileNotFoundError in SemLock._rebuild before
        # signalling ready_event.  Detect early-dead child via is_alive() so
        # we don't waste the full timeout, and retry up to 3x with jitter.
        last_err = None
        for _spawn_attempt in range(3):
            self._parse_input_queue = self._mp_ctx.Queue()
            self._parse_output_queue = self._mp_ctx.Queue()
            ready_event = self._mp_ctx.Event()
            self._parse_process = self._mp_ctx.Process(
                target=_cupti_parser_worker,
                args=(self._parse_input_queue, self._parse_output_queue, ready_event),
            )
            self._parse_process.start()
            deadline = time.time() + 30.0
            spawn_ok = False
            while time.time() < deadline:
                if ready_event.wait(timeout=0.5):
                    spawn_ok = True
                    break
                if not self._parse_process.is_alive():
                    break
            if spawn_ok:
                last_err = None
                break
            last_err = (f"attempt {_spawn_attempt + 1}: "
                        f"alive={self._parse_process.is_alive()}, "
                        f"exitcode={self._parse_process.exitcode}")
            try:
                if self._parse_process.is_alive():
                    self._parse_process.terminate()
                self._parse_process.join(timeout=2.0)
            except Exception:
                pass
            time.sleep(0.5 + 0.5 * _spawn_attempt)
        if last_err is not None:
            raise RuntimeError(
                f"CUPTI parser process did not initialize after 3 attempts: {last_err}"
            )

        self._set_zeroed_host_buffer_attr()
        for _ in range(_CUPTI_HOST_BUFFER_COUNT):
            self._free_buffer_ids.append(self._allocate_shared_buffer())

        self._request_callback = self._request_callback_type(self._request_buffer)
        self._complete_callback = self._complete_callback_type(self._complete_buffer)
        self._check(self._libcupti.cuptiActivityRegisterCallbacks(
            self._request_callback,
            self._complete_callback,
        ))
        self._check(self._libcupti.cuptiActivityEnable(_CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL))
        atexit.register(self.close)

    def _configure_functions(self) -> None:
        self._libcupti.cuptiActivityRegisterCallbacks.argtypes = [
            self._request_callback_type,
            self._complete_callback_type,
        ]
        self._libcupti.cuptiActivityRegisterCallbacks.restype = ctypes.c_int
        self._libcupti.cuptiActivityEnable.argtypes = [ctypes.c_int]
        self._libcupti.cuptiActivityEnable.restype = ctypes.c_int
        self._libcupti.cuptiActivityFlushAll.argtypes = [ctypes.c_uint32]
        self._libcupti.cuptiActivityFlushAll.restype = ctypes.c_int
        self._libcupti.cuptiActivityFlushPeriod.argtypes = [ctypes.c_uint32]
        self._libcupti.cuptiActivityFlushPeriod.restype = ctypes.c_int
        self._libcupti.cuptiActivitySetAttribute.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
        ]
        self._libcupti.cuptiActivitySetAttribute.restype = ctypes.c_int
        _configure_cupti_get_next_record(self._libcupti)

    def _set_zeroed_host_buffer_attr(self) -> None:
        value_obj = ctypes.c_uint8(1)
        size_obj = ctypes.c_size_t(ctypes.sizeof(value_obj))
        result = self._libcupti.cuptiActivitySetAttribute(
            _CUPTI_ACTIVITY_ATTR_ZEROED_OUT_ACTIVITY_BUFFER,
            ctypes.byref(size_obj),
            ctypes.byref(value_obj),
        )
        if result != _CUPTI_SUCCESS:
            print(
                "[WARN] CUPTI zeroed host-buffer attribute failed; "
                f"continuing with default CUPTI buffer handling (CUptiResult={result}).",
                file=sys.stderr,
            )

    def _check(self, result: int) -> None:
        if result != _CUPTI_SUCCESS:
            raise RuntimeError(f"CUPTI call failed with CUptiResult={result}")

    def _allocate_shared_buffer(self) -> int:
        buffer_id = len(self._shared_buffers)
        shm = shared_memory.SharedMemory(create=True, size=_CUPTI_HOST_BUFFER_BYTES)
        shared_char = ctypes.c_char.from_buffer(shm.buf)
        try:
            ptr = ctypes.addressof(shared_char)
        finally:
            del shared_char
        if ptr % 8 != 0:
            shm.close()
            shm.unlink()
            raise RuntimeError("CUPTI shared-memory activity buffer was not 8-byte aligned")
        self._shared_buffers[buffer_id] = shm
        self._buffer_id_by_ptr[ptr] = buffer_id
        return buffer_id

    def _buffer_ptr(self, buffer_id: int) -> int:
        shm = self._shared_buffers[buffer_id]
        shared_char = ctypes.c_char.from_buffer(shm.buf)
        try:
            return ctypes.addressof(shared_char)
        finally:
            del shared_char

    def _request_buffer(self, buffer, size, max_num_records) -> None:
        with self._lock:
            if self._free_buffer_ids:
                buffer_id = self._free_buffer_ids.pop()
            else:
                buffer_id = self._allocate_shared_buffer()
            ptr = self._buffer_ptr(buffer_id)
        buffer[0] = ptr
        size[0] = _CUPTI_HOST_BUFFER_BYTES
        max_num_records[0] = 0

    def _complete_buffer(self, context, stream_id, buffer, size, valid_size) -> None:
        del context, stream_id, size
        buffer_ptr = int(buffer)
        valid_size_int = int(valid_size)
        with self._lock:
            mode = self._mode
            generation = self._generation
            buffer_id = self._buffer_id_by_ptr[buffer_ptr]
            if valid_size_int == 0 or mode == "drop":
                self._free_buffer_ids.append(buffer_id)
                return
            if mode == "local":
                self._local_completed.append((buffer_id, valid_size_int))
                return
            shm = self._shared_buffers[buffer_id]
        self._parse_input_queue.put(("buffer", generation, buffer_id, shm.name, valid_size_int))

    def _handle_parser_result(self, result: dict) -> None:
        kind = result.get("kind")
        if kind == "buffer_done":
            with self._lock:
                self._free_buffer_ids.append(int(result["buffer_id"]))
        elif kind == "finish_done":
            self._finish_results[int(result["generation"])] = result
        elif kind == "error":
            self._parser_errors.append(str(result.get("error")))

    def _drain_parser_results(self) -> None:
        while True:
            try:
                result = self._parse_output_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_parser_result(result)

    def is_generation_ready(self, generation: int) -> bool:
        self._drain_parser_results()
        return generation in self._finish_results or bool(self._parser_errors)

    def _flush(self, flag: int) -> None:
        self._check(self._libcupti.cuptiActivityFlushAll(flag))

    def _set_flush_period_ms(self, period_ms: int) -> None:
        if period_ms == self._current_flush_period_ms:
            return
        self._check(self._libcupti.cuptiActivityFlushPeriod(period_ms))
        self._current_flush_period_ms = period_ms

    def _begin(
        self,
        mode: str,
        filter_plan=(),
        flush_period_ms: int = 0,
        collect_timing: bool = False,
    ) -> int:
        start_timing: dict[str, float] = {}
        with self._lock:
            self._mode = "drop"
        phase_start_s = time.perf_counter() if collect_timing else 0.0
        self._flush(1)
        if collect_timing:
            start_timing["forced_flush_ms"] = 1000.0 * (time.perf_counter() - phase_start_s)
        phase_start_s = time.perf_counter() if collect_timing else 0.0
        self._drain_parser_results()
        if collect_timing:
            start_timing["drain_ms"] = 1000.0 * (time.perf_counter() - phase_start_s)
        with self._lock:
            self._generation += 1
            generation = self._generation
            self._mode = mode
            self._local_completed = []
            self._filter_plan = filter_plan
        if flush_period_ms > 0:
            phase_start_s = time.perf_counter() if collect_timing else 0.0
            self._set_flush_period_ms(flush_period_ms)
            if collect_timing:
                start_timing["period_enable_ms"] = 1000.0 * (
                    time.perf_counter() - phase_start_s
                )
        self._last_start_timing = start_timing
        return generation

    def capture_names(self, replay_fn) -> tuple[list[tuple], int, dict]:
        """Run a small calibration replay and parse kernel names locally."""
        self._begin("local")
        replay_fn()
        torch.cuda.synchronize()
        self._flush(0)
        records: list[tuple] = []
        zero_ts_count = 0
        zero_ts_names: dict[str, int] = {}
        with self._lock:
            completed = list(self._local_completed)
            self._local_completed = []
            self._mode = "drop"
        for buffer_id, valid_size in completed:
            ptr = self._buffer_ptr(buffer_id)
            recs, zeros, zero_names = _parse_cupti_buffer_ptr(
                self._libcupti,
                ptr,
                valid_size,
                include_names=True,
            )
            records.extend(recs)
            zero_ts_count += zeros
            for name, count in zero_names.items():
                zero_ts_names[name] = zero_ts_names.get(name, 0) + count
            ctypes.memset(ptr, 0, _CUPTI_HOST_BUFFER_BYTES)
            with self._lock:
                self._free_buffer_ids.append(buffer_id)
        records.sort(key=lambda r: r[1])
        return records, zero_ts_count, zero_ts_names

    def start(
        self,
        filter_plan=(),
        flush_period_ms: int = 0,
        collect_timing: bool = False,
    ) -> None:
        self._begin("parser", filter_plan, flush_period_ms, collect_timing)

    def stop_async(
        self,
        collect_timing: bool = False,
        stats_request: dict | None = None,
    ) -> tuple[int, dict[str, float]]:
        stop_timing: dict[str, float] = {}
        generation = self._generation
        phase_start_s = time.perf_counter() if collect_timing else 0.0
        self._set_flush_period_ms(0)
        if collect_timing:
            stop_timing["period_disable_ms"] = 1000.0 * (
                time.perf_counter() - phase_start_s
            )
        phase_start_s = time.perf_counter() if collect_timing else 0.0
        self._flush(0)
        if collect_timing:
            stop_timing["flush_ms"] = 1000.0 * (time.perf_counter() - phase_start_s)
        with self._lock:
            self._mode = "drop"
            filter_plan = self._filter_plan
        self._parse_input_queue.put(("finish", generation, filter_plan, stats_request))
        self._last_stop_timing = stop_timing
        return generation, stop_timing

    def wait_for_generation_result(
        self,
        generation: int,
        stop_timing: dict[str, float] | None = None,
        collect_timing: bool = False,
    ) -> dict:
        if stop_timing is None:
            stop_timing = {}
        phase_start_s = time.perf_counter() if collect_timing else 0.0
        deadline = time.perf_counter() + 10.0
        while time.perf_counter() < deadline:
            result = self._finish_results.pop(generation, None)
            if result is not None:
                if collect_timing:
                    stop_timing["parser_wait_ms"] = 1000.0 * (
                        time.perf_counter() - phase_start_s
                    )
                    stop_timing["total_ms"] = (
                        stop_timing.get("period_disable_ms", 0.0)
                        + stop_timing.get("flush_ms", 0.0)
                        + stop_timing["parser_wait_ms"]
                    )
                self._last_stop_timing = stop_timing
                return result
            timeout_s = max(0.0, min(0.01, deadline - time.perf_counter()))
            try:
                parser_result = self._parse_output_queue.get(timeout=timeout_s)
            except queue.Empty:
                continue
            self._handle_parser_result(parser_result)
            if self._parser_errors:
                raise RuntimeError("CUPTI parser process failed: " + "; ".join(self._parser_errors))
        raise TimeoutError("Timed out waiting for CUPTI parser process")

    def wait_for_generation(
        self,
        generation: int,
        stop_timing: dict[str, float] | None = None,
        collect_timing: bool = False,
    ) -> tuple[list[tuple], int, dict, int]:
        result = self.wait_for_generation_result(generation, stop_timing, collect_timing)
        return (
            list(result["records"]),
            int(result["zero_ts_count"]),
            dict(result["zero_ts_names"]),
            int(result["raw_record_count"]),
        )

    def stop(self, collect_timing: bool = False) -> tuple[list[tuple], int, dict, int]:
        generation, stop_timing = self.stop_async(collect_timing)
        return self.wait_for_generation(generation, stop_timing, collect_timing)

    def last_start_timing(self) -> dict[str, float]:
        return dict(self._last_start_timing)

    def last_stop_timing(self) -> dict[str, float]:
        return dict(self._last_stop_timing)

    def close(self) -> None:
        parse_process = getattr(self, "_parse_process", None)
        if parse_process is not None and parse_process.is_alive():
            self._parse_input_queue.put(None)
            parse_process.join(timeout=5.0)
            if parse_process.is_alive():
                parse_process.terminate()
                parse_process.join(timeout=1.0)
        for shm in getattr(self, "_shared_buffers", {}).values():
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass


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


def _binomial_pmf(n: int, p: float) -> tuple[float, ...]:
    return tuple(
        math.comb(n, k) * (p**k) * ((1.0 - p)**(n - k))
        for k in range(n + 1)
    )


def _kmix_bucket_score(
    iters_us,
    n_writes_per_iter,
    batch: int,
    write_frac: float | None,
) -> float | None:
    """Compute the search-driver kmix score from per-iteration timings."""
    if (
        write_frac is None
        or not iters_us
        or not n_writes_per_iter
        or len(iters_us) != len(n_writes_per_iter)
    ):
        return None

    buckets: dict[int, list[float]] = {}
    for span, n_writes in zip(iters_us, n_writes_per_iter):
        k = int(n_writes)
        if 0 <= k <= batch:
            buckets.setdefault(k, []).append(float(span))
    if not buckets:
        return None

    pmf = _binomial_pmf(batch, write_frac)
    numerator = 0.0
    denominator = 0.0
    for k, spans in buckets.items():
        numerator += pmf[k] * statistics.median(spans)
        denominator += pmf[k]
    return numerator / denominator if denominator > 0.0 else None


def _stats_from_cupti_records(records, warmup, iters, tag, expected_K,
                              zero_ts_count: int = 0,
                              zero_ts_names: dict | None = None,
                              include_details: bool = True):
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
        if r[0] is not None and any(s in r[0] for s in _CUPTI_KEEP_KERNEL_SUBSTRINGS)
    ]
    records.sort(key=lambda r: r[1])  # by start_ns

    total = len(records)
    expected_iters = warmup + iters
    expected_total = expected_K * expected_iters
    if total != expected_total:
        from collections import Counter
        name_counts = dict(Counter(r[0] for r in records))
        # Non-fatal: skip this cell instead of killing the whole sweep.
        # Mismatch may be a CUPTI dropped-records issue (rare configs),
        # not necessarily a K-table bug.  Log so the user can investigate
        # the specific cell post-hoc; return None so the caller can skip
        # writing a JSON row.
        zero_msg = ""
        if zero_ts_count:
            zero_msg = (
                f" + {zero_ts_count} records with start/end=0 "
                f"(dropped by callback, breakdown {zero_ts_names}). "
                f"Total observed kernel records (timed + zero-ts) = "
                f"{total + zero_ts_count} / {expected_total}."
            )
        print(
            f"[WARN] CUPTI capture mismatch for {tag!r}: expected "
            f"{expected_K} kernels/iter × {expected_iters} iters "
            f"(warmup+iters) = {expected_total} records, got {total}. "
            f"Kernel record counts: {name_counts}.{zero_msg} SKIPPING cell.",
            file=sys.stderr,
            flush=True,
        )
        # Per-record dump: (name, start_ns_rel, end_ns_rel, corr_id, graph_id, stream_id).
        # Times relative to first record so absolute ns isn't drowning output.
        # Limit dump to first 30 records to avoid flooding logs at high K.
        if records:
            t0_ns = records[0][1]
            for i, r in enumerate(records[:30]):
                # r = (name, start_ns, end_ns, corr_id, graph_id, graph_node_id, stream_id)
                rel_start = (r[1] - t0_ns) / 1000.0  # us
                rel_end = (r[2] - t0_ns) / 1000.0
                print(
                    f"  rec[{i:3d}] name={r[0]!r} start={rel_start:.2f}us "
                    f"end={rel_end:.2f}us corr={r[3]} graph={r[4]} stream={r[6]}",
                    file=sys.stderr,
                    flush=True,
                )
            if len(records) > 30:
                print(f"  ... ({len(records) - 30} more records elided)",
                      file=sys.stderr, flush=True)
        return None
    K = expected_K
    timed = records[warmup * K:]

    spans_us: list[float] = []
    per_kernel: dict[str, dict[str, list[float]]] = {}
    for i in range(iters):
        chunk = timed[i * K:(i + 1) * K]
        iter_start_ns = min(r[1] for r in chunk)
        iter_end_ns = max(r[2] for r in chunk)
        spans_us.append((iter_end_ns - iter_start_ns) / 1000.0)
        if include_details:
            for r in chunk:
                name = r[0]
                slot = per_kernel.setdefault(name, {"start_us": [], "end_us": []})
                slot["start_us"].append((r[1] - iter_start_ns) / 1000.0)
                slot["end_us"].append((r[2] - iter_start_ns) / 1000.0)

    out = _stats_from_spans(spans_us)
    out["iters_us"] = spans_us
    if include_details:
        out["per_kernel"] = per_kernel
    return out


_PRE_GRAPH_WARMUP_ITERS = 1
_CUPTI_FILTER_PLAN_CACHE: dict[tuple, tuple[int, tuple[str | None, ...]]] = {}


class _HostTiming:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.values: dict[str, float | int | bool] = {}
        self._total_start_s = time.perf_counter() if enabled else 0.0
        self._phase_start_s = 0.0

    def start(self) -> None:
        if self.enabled:
            self._phase_start_s = time.perf_counter()

    def stop(self, key: str) -> None:
        if self.enabled:
            self.values[key] = 1000.0 * (time.perf_counter() - self._phase_start_s)

    def add(self, key: str, value: float | int | bool) -> None:
        if self.enabled:
            self.values[key] = value

    def stop_total(self) -> None:
        if self.enabled:
            self.values["total_ms"] = 1000.0 * (time.perf_counter() - self._total_start_s)

    def attach(self, stats: dict | None) -> None:
        if self.enabled and stats is not None:
            stats["host_timing"] = self.values


class _PendingCuptiStats:

    def __init__(
        self,
        timer: CuptiKernelTimer,
        generation: int,
        stop_timing: dict[str, float],
        host_timing: _HostTiming,
        *,
        warmup: int,
        iters: int,
        tag: str,
        expected_K: int,
        expected_raw_record_count: int,
    ) -> None:
        self._timer = timer
        self._generation = generation
        self._stop_timing = stop_timing
        self._host_timing = host_timing
        self._warmup = warmup
        self._iters = iters
        self._tag = tag
        self._expected_K = expected_K
        self._expected_raw_record_count = expected_raw_record_count

    def is_ready(self) -> bool:
        return self._timer.is_generation_ready(self._generation)

    def resolve(self) -> dict | None:
        result = self._timer.wait_for_generation_result(
            self._generation,
            self._stop_timing,
            collect_timing=self._host_timing.enabled,
        )
        for key, value in self._timer.last_stop_timing().items():
            self._host_timing.add(f"cupti_stop_{key}", value)
        raw_record_count = int(result["raw_record_count"])
        if raw_record_count != self._expected_raw_record_count:
            print(
                f"[WARN] CUPTI raw-record mismatch for {self._tag!r}: expected "
                f"{self._expected_raw_record_count}, got {raw_record_count}. SKIPPING cell.",
                file=sys.stderr,
            )
            return None

        if result.get("stats_ready"):
            stats = result.get("stats")
            self._host_timing.add("stats_ms", 0.0)
            self._host_timing.add("parser_stats_ms", float(result.get("parser_stats_ms", 0.0)))
        else:
            self._host_timing.start()
            stats = _stats_from_cupti_records(
                list(result["records"]),
                self._warmup,
                self._iters,
                self._tag,
                self._expected_K,
                zero_ts_count=int(result["zero_ts_count"]),
                zero_ts_names=dict(result["zero_ts_names"]),
            )
            self._host_timing.stop("stats_ms")
        self._host_timing.attach(stats)
        return stats


def _target_name_or_none(name: str | None) -> str | None:
    if name is None:
        return None
    if any(s in name for s in _CUPTI_KEEP_KERNEL_SUBSTRINGS):
        return name
    return None


def _capture_group_graph(
    args,
    run_fn,
    reset_fn,
    group_iters: int,
    graph_pre_iter_fn=None,
) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for j in range(group_iters):
            if graph_pre_iter_fn is not None:
                graph_pre_iter_fn(j)
            reset_fn()
            if args.l2_flush:
                _l2_flush.fill_(0.0)
            run_fn()
    return graph


def _graph_group_iters(args, total_iters: int, pre_iter_fn, pre_iter_group_factory) -> int:
    """Pick the graph-group size unconditionally; the caller is expected to
    round total_iters up to a multiple of this so all iters fit in clean
    replays.  Sample arrays are pre-padded at allocation (see _sample_pnat
    call site) so the per-replay window can index past the user-requested
    iter count by up to group_iters-1 extra samples.
    """
    if pre_iter_fn is not None and pre_iter_group_factory is None:
        # Per-iter callback without a group-factory: can't batch.
        return 1
    requested = getattr(args, "cuda_graph_group_iters", None)
    if requested is None:
        return (
            _DEFAULT_CUDA_GRAPH_GROUP_ITERS_MIX
            if pre_iter_group_factory is not None
            else _DEFAULT_CUDA_GRAPH_GROUP_ITERS_PURE
        )
    return max(1, int(requested))


def _get_cupti_filter_plan(timer: CuptiKernelTimer, graph, cache_key: tuple | None,
                           group_iters: int) -> tuple[int, tuple[str | None, ...]]:
    full_cache_key = None if cache_key is None else (cache_key, group_iters)
    if full_cache_key is not None:
        cached = _CUPTI_FILTER_PLAN_CACHE.get(full_cache_key)
        if cached is not None:
            return cached

    records, zero_ts_count, zero_ts_names = timer.capture_names(graph.replay)
    if zero_ts_count:
        print(
            f"[WARN] CUPTI calibration saw {zero_ts_count} zero-timestamp records "
            f"(breakdown {zero_ts_names}); continuing with nonzero records.",
            file=sys.stderr,
        )
    ordinal_names = tuple(_target_name_or_none(r[0]) for r in records)
    target_count = sum(name is not None for name in ordinal_names)
    if target_count == 0:
        raise RuntimeError("CUPTI calibration did not find any target kernel records")
    plan = (len(records), ordinal_names)
    if full_cache_key is not None:
        _CUPTI_FILTER_PLAN_CACHE[full_cache_key] = plan
    return plan


def _time_kernel_cuda_graph(
    args,
    run_fn,
    reset_fn,
    tag: str,
    *,
    expected_K: int,
    pre_iter_fn=None,
    pre_iter_group_factory=None,
    iters_override: int | None = None,
    cupti_plan_key: tuple | None = None,
) -> dict:
    """CUDA-graph CUPTI timer (graph-per-iter design).

    Captures one CUDA graph holding a small group of logical iterations
    (per-iter setup + reset + l2_flush + run_fn) and replays it enough
    times to cover `warmup + iters`.

    Why graph-per-iter (vs the older "one giant graph holding all iters"
    design): instantiating a CUDA graph is expensive — proportional to
    graph size — so a single small graph instantiated once is much
    cheaper than one big graph instantiated for each cell of a sweep.
    Replays are cheap regardless.

    Mix cells use a per-replay device window: an outside-graph copy loads
    the next group of PNAT/n_writes samples, then graph-captured per-iter
    copies update kernel inputs before each reset + L2 flush + run.

    Pre-graph eager warmup: forces PyTorch's caching allocator
    + Triton's autotune cache to settle before capture so the graph
    doesn't bake in init-only allocations.

    ``iters_override`` (if not None) overrides ``args.iters`` for this
    call.  Used to give mix scenarios a higher iter count than pure
    (more iters = more independent mix draws averaged in).
    """
    host_timing = _HostTiming(bool(getattr(args, "host_timing", False)))
    timer = CuptiKernelTimer.get()
    warmup = args.warmup
    iters = iters_override if iters_override is not None else args.iters

    # Pre-graph eager warmup: full per-iter chain once.  This settles
    # Triton/PyTorch setup and wrapper-side intermediate allocations;
    # skipping it risks lazy work leaking into graph capture.
    warmup_iters = _PRE_GRAPH_WARMUP_ITERS
    host_timing.add("pre_graph_warmup_iters", warmup_iters)
    host_timing.start()
    for _ in range(warmup_iters):
        reset_fn()
        if pre_iter_fn is not None:
            pre_iter_fn(0)
        run_fn()
    if warmup_iters > 0:
        torch.cuda.synchronize()
    host_timing.stop("pre_graph_warmup_ms")

    total_iters = warmup + iters
    group_iters = _graph_group_iters(args, total_iters, pre_iter_fn, pre_iter_group_factory)
    # Args are rounded at argparse-time so warmup+iters/mix_iters are already
    # multiples of the relevant group_iters.  Assert here to catch any caller
    # bypassing argparse.
    assert total_iters % group_iters == 0, (
        f"total_iters={total_iters} not a multiple of group_iters={group_iters}; "
        f"args.warmup/iters/mix_iters should be rounded post-argparse."
    )
    pre_replay_fn = None
    graph_pre_iter_fn = None
    if pre_iter_group_factory is not None and group_iters > 1:
        pre_replay_fn, graph_pre_iter_fn = pre_iter_group_factory(group_iters)

    # Reset just before capture so warmup state changes don't bleed in.
    host_timing.start()
    reset_fn()
    torch.cuda.synchronize()
    host_timing.stop("pre_capture_reset_ms")

    # Capture a small group of identical logical iterations.  Mix/pre_iter
    # cells can group when they provide a graph-side pre-iter updater backed
    # by a per-replay device window.
    host_timing.start()
    g = _capture_group_graph(args, run_fn, reset_fn, group_iters, graph_pre_iter_fn)
    host_timing.stop("graph_capture_ms")

    if pre_replay_fn is not None:
        host_timing.start()
        pre_replay_fn(0)
        torch.cuda.synchronize()
        host_timing.stop("graph_preload_ms")

    plan_cache_key = None if cupti_plan_key is None else (cupti_plan_key, group_iters)
    host_timing.add("cupti_plan_cached", (
        plan_cache_key is not None and plan_cache_key in _CUPTI_FILTER_PLAN_CACHE
    ))
    host_timing.start()
    records_per_replay, ordinal_names = _get_cupti_filter_plan(
        timer,
        g,
        cupti_plan_key,
        group_iters,
    )
    host_timing.stop("cupti_plan_ms")
    target_count = sum(name is not None for name in ordinal_names)
    expected_targets_per_replay = expected_K * group_iters
    if target_count != expected_targets_per_replay:
        print(
            f"[WARN] CUPTI calibration mismatch for {tag!r}: expected "
            f"{expected_targets_per_replay} target records in a {group_iters}-iter graph replay, "
            f"got {target_count} target records out of {records_per_replay} total records.",
            file=sys.stderr,
        )

    # Time: replay the grouped graph enough times to cover warmup+iters.
    # Mix cells preload one device window per replay on the same stream.
    # CUPTI records every kernel launch; _stats_from_cupti_records
    # validates against expected_K and slices warmup off the front.
    graph_replays = total_iters // group_iters
    filter_plan = ((records_per_replay, ordinal_names),) * graph_replays
    cupti_flush_period_ms = max(0, int(getattr(args, "cupti_flush_period_ms", 0)))
    host_timing.start()
    timer.start(
        filter_plan,
        flush_period_ms=cupti_flush_period_ms,
        collect_timing=host_timing.enabled,
    )
    host_timing.stop("cupti_start_ms")
    for key, value in timer.last_start_timing().items():
        host_timing.add(f"cupti_start_{key}", value)
    torch.cuda.nvtx.range_push(tag)
    host_timing.start()
    for i in range(graph_replays):
        if pre_replay_fn is not None:
            pre_replay_fn(i)
        elif pre_iter_fn is not None:
            pre_iter_fn(i)
        g.replay()
    host_timing.stop("graph_enqueue_ms")
    host_timing.start()
    torch.cuda.synchronize()
    host_timing.stop("graph_sync_ms")
    torch.cuda.nvtx.range_pop()
    expected_raw_record_count = records_per_replay * graph_replays
    host_timing.start()
    if int(getattr(args, "cupti_defer_depth", 1)) > 1:
        generation, stop_timing = timer.stop_async(
            collect_timing=host_timing.enabled,
            stats_request={
                "warmup": warmup,
                "iters": iters,
                "tag": tag,
                "expected_K": expected_K,
                "include_details": bool(getattr(args, "json_detailed", False)),
            },
        )
        host_timing.stop("cupti_stop_ms")
        for key, value in timer.last_stop_timing().items():
            host_timing.add(f"cupti_stop_{key}", value)
        host_timing.stop_total()
        host_timing.add("graph_group_iters", group_iters)
        host_timing.add("graph_replays", graph_replays)
        host_timing.add("cupti_records_per_replay", records_per_replay)
        host_timing.add("cupti_target_records_per_replay", target_count)
        host_timing.add("cupti_raw_records_expected", expected_raw_record_count)
        host_timing.add("cupti_flush_period_ms", cupti_flush_period_ms)
        return _PendingCuptiStats(
            timer,
            generation,
            stop_timing,
            host_timing,
            warmup=warmup,
            iters=iters,
            tag=tag,
            expected_K=expected_K,
            expected_raw_record_count=expected_raw_record_count,
        )

    records, zero_ts_count, zero_ts_names, raw_record_count = timer.stop(
        collect_timing=host_timing.enabled,
    )
    host_timing.stop("cupti_stop_ms")
    for key, value in timer.last_stop_timing().items():
        host_timing.add(f"cupti_stop_{key}", value)
    if raw_record_count != expected_raw_record_count:
        print(
            f"[WARN] CUPTI raw-record mismatch for {tag!r}: expected "
            f"{records_per_replay} total records/replay × {graph_replays} replays "
            f"= {expected_raw_record_count}, got {raw_record_count}. SKIPPING cell.",
            file=sys.stderr,
        )
        return None

    host_timing.start()
    stats = _stats_from_cupti_records(
        records,
        warmup,
        iters,
        tag,
        expected_K,
        zero_ts_count=zero_ts_count,
        zero_ts_names=zero_ts_names,
        include_details=bool(getattr(args, "json_detailed", False)),
    )
    host_timing.stop("stats_ms")
    host_timing.stop_total()
    host_timing.add("graph_group_iters", group_iters)
    host_timing.add("graph_replays", graph_replays)
    host_timing.add("cupti_records_per_replay", records_per_replay)
    host_timing.add("cupti_target_records_per_replay", target_count)
    host_timing.add("cupti_raw_records", raw_record_count)
    host_timing.add("cupti_raw_records_expected", expected_raw_record_count)
    host_timing.add("cupti_flush_period_ms", cupti_flush_period_ms)
    host_timing.attach(stats)
    return stats


def _time_kernel_eager(
    args,
    run_fn,
    reset_fn,
    tag: str,
    *,
    expected_K: int,
    pre_iter_fn=None,
    iters_override: int | None = None,
    cupti_plan_key: tuple | None = None,
) -> dict:
    """Non-graph CUPTI timer (for ncu wrapping, debugging, etc.).

    Each iter runs serially with sync between, but kernel start/end still
    come from CUPTI — same accuracy as the graph path, just slower per-iter
    (extra Python + sync overhead).
    """
    host_timing = _HostTiming(bool(getattr(args, "host_timing", False)))
    timer = CuptiKernelTimer.get()
    warmup = args.warmup
    iters = iters_override if iters_override is not None else args.iters

    del cupti_plan_key

    def _run_eager_loop():
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

    host_timing.start()
    records, zero_ts_count, zero_ts_names = timer.capture_names(_run_eager_loop)
    host_timing.stop("timed_loop_and_cupti_parse_ms")

    host_timing.start()
    stats = _stats_from_cupti_records(
        records,
        warmup,
        iters,
        tag,
        expected_K,
        zero_ts_count=zero_ts_count,
        zero_ts_names=zero_ts_names,
        include_details=bool(getattr(args, "json_detailed", False)),
    )
    host_timing.stop("stats_ms")
    host_timing.stop_total()
    host_timing.attach(stats)
    return stats


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
    pre_iter_group_factory=None,
    iters_override: int | None = None,
    cupti_plan_key: tuple | None = None,
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
            pre_iter_group_factory=pre_iter_group_factory,
            iters_override=iters_override,
            cupti_plan_key=cupti_plan_key,
        )
    return _time_kernel_eager(
        args, run_fn, reset_fn, tag,
        expected_K=expected_K,
        pre_iter_fn=pre_iter_fn,
        iters_override=iters_override,
        cupti_plan_key=cupti_plan_key,
    )


# Per-config benchmark (consolidated baseline + replay)


def _warm_one_config(args, cfg, baseline_fn) -> None:
    """Module-level worker for the compile-warmup process pool.

    Module-level so ProcessPoolExecutor can pickle it (nested functions
    aren't picklable).  Each worker process holds its own GIL → no
    serialization between concurrent compiles.

    ``cfg`` is a tuple of (outer_cfg, inner_overrides_or_list):
      * outer_cfg = (batch, mtp_len, prev_ks, state_dtype, act_dtype,
                     sr_mode, rect, mode, hardcode_sort)
      * inner_overrides_or_list = dict of args attribute name -> value-string,
        OR a list of such dicts.  In the list form (CPS-grouped task) the
        worker compiles each entry sequentially within the same process so
        Triton's in-process kernel cache catches value-spec hits across
        related entries (e.g. CPS={1,2} and {4,8} each form a `div_by_16`
        spec bucket; the second compile in a bucket short-circuits).

    ``baseline_fn`` is optional — when ``None``, only the replay kernel is
    warmed (the baseline-selection kernel can be warmed once in
    the parent if needed).  This lets us avoid pickling C-extension
    function references across processes.
    """
    outer_cfg, inner_overrides_or_list = cfg
    overrides_list = (inner_overrides_or_list
                      if isinstance(inner_overrides_or_list, list)
                      else [inner_overrides_or_list])
    (batch, mtp_len, prev_ks, state_dtype, act_dtype, sr_mode,
     rect, mode, hardcode_sort) = outer_cfg
    import argparse as _ap
    for inner_overrides in overrides_list:
        # Fresh clone per entry: prevents knob-value leakage between
        # consecutive cells in a CPS-grouped task (entries may set
        # different non-CPS knobs in degenerate edge cases).
        args_copy = _ap.Namespace(**vars(args))
        for k, v in inner_overrides.items():
            setattr(args_copy, k, v)
        _bench_config(
            args_copy, batch, mtp_len, prev_ks, state_dtype, act_dtype, baseline_fn,
            sr_mode=sr_mode, rectangle_for_nowrite=rect, mode=mode,
            hardcode_sort=hardcode_sort,
            warmup_only=True,
        )


def _compile_warmup_phase(args, batch_sizes, mtp_lengths, state_dtypes, act_dtypes,
                          baseline_fn, max_workers: int) -> None:
    _cw_t0 = time.perf_counter()
    def _cw(label: str) -> None:
        dt = time.perf_counter() - _cw_t0
        print(f"[compile-warmup] t={dt:7.2f}s  {label}", file=sys.stderr, flush=True)
    _cw("entered _compile_warmup_phase")
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
    modes_list = getattr(args, "modes_list", ["persistent_dynamic"])
    hsort_list = getattr(args, "hardcode_sort_list", [False])

    # Compile-warmup task enumeration: outer × inner cartesian.
    # CRITICAL: only enumerate axes that change the kernel's COMPILE signature.
    # Drop runtime axes (batch, prev_k) that produce identical kernel hashes —
    # otherwise we'd pay ~50-100ms of bench setup per redundant cache-hit task.
    #
    # Batches collapse to first only when the caller has forced an explicit
    # mode/knob set. With mode=None, _DEFAULT_TUNING depends on effective
    # batch and can choose different constexpr knobs or even a different mode.
    # prev_k is already a list passed into _bench_config (not enumerated here).
    configs = []
    _default_tuning_requested = any(m is None for m in modes_list)
    _compile_batches = batch_sizes if _default_tuning_requested else batch_sizes[:1]
    for batch in _compile_batches:
        for mtp_len in mtp_lengths:
            prev_ks = _resolve_prev_ks(args, mtp_len)
            for state_dtype in state_dtypes:
                for act_dtype in act_dtypes:
                    for sr_mode in sr_modes_list:
                        for mode in modes_list:
                            for rect in rect_list:
                                can_sort = (
                                    args.mix_csv is not None
                                    or getattr(args, "pmix", False)
                                )
                                effective_hsort_list = (
                                    hsort_list if can_sort else [False]
                                )
                                for hardcode_sort in effective_hsort_list:
                                    configs.append((
                                        batch, mtp_len, prev_ks,
                                        state_dtype, act_dtype, sr_mode,
                                        rect, mode, hardcode_sort,
                                    ))

    # Enumerate inner-knob signatures.  Two paths:
    #   (1) --cell-list mode (preferred when set): pull exactly the cells
    #       that will be timed from args._cell_list_set.  No synthetic
    #       cartesian — we only pre-compile what will run.
    #   (2) Sweep-args mode: cartesian over split-aware axes that read
    #       BOTH unsplit (args.X) and per-half (args.X_write/_nowrite)
    #       knob settings.  Older code read only args.X and silently
    #       enumerated 1 inner combo when callers set only the per-half
    #       versions (all cell-list usage, plus any --block-size-m-write/
    #       _nowrite CLI invocation), causing massive in-process JIT
    #       compile tax for persistent_main especially.
    #
    # In BOTH paths we GROUP tasks by non-CPS signature so each worker
    # process compiles all CPS values for its group sequentially.
    # NUM_PERSISTENT = CPS * num_sms is a runtime int but Triton auto-
    # specializes on `div_by_16`, partitioning {CPS=1,2} (132,264) from
    # {CPS=4,8} (528,1056) into two distinct compiled variants.  By
    # keeping all CPS variants for one (M,W,S,LS,TMA,...) signature in
    # the same worker, the second compile in each spec bucket hits the
    # in-process Triton cache (no disk-cache round trip).

    def _ps(val):
        if val is None or (isinstance(val, str) and not val):
            return [None]
        if isinstance(val, str):
            return [v.strip() for v in val.split(",") if v.strip()]
        return [val]

    def _split_or_pair(shared_attr, w_attr, nw_attr):
        """Return list of (write_val, nowrite_val) strings.

        Reads shared (args.X), write-side (args.X_write), and nowrite-side
        (args.X_nowrite) values.  If both per-half attrs are None, emits
        tied pairs (v,v) over the shared values.  If either per-half is
        set, cartesian-iterates per-half values, falling back to shared
        for whichever side is None.
        """
        w = _ps(getattr(args, w_attr, None))
        nw = _ps(getattr(args, nw_attr, None))
        s = _ps(getattr(args, shared_attr, None))
        if w == [None] and nw == [None]:
            return [(v, v) for v in s]
        if w == [None]:
            w = s
        if nw == [None]:
            nw = s
        return [(a, b) for a in w for b in nw]

    _cw(f"built {len(configs)} outer configs")
    cell_set = getattr(args, "_cell_list_set", set())
    cell_keys = getattr(args, "_cell_list_keys", ())
    # CPS keys are runtime ints (kernel value-specializes on `div_by_16`);
    # cells differing only on CPS values can SHARE a worker so the second
    # CPS value in a div_by_16 bucket hits the in-process Triton cache.
    _cps_keys = ("cta_per_sm_write", "cta_per_sm_nowrite", "cta_per_sm")

    if cell_set:
        # ============== CELL-LIST PATH ==============
        # Build tasks DIRECTLY from cells.  Each cell carries its OWN
        # outer-axis values (RECT, MODE, SR, HSORT) so we
        # pair each cell with its specific outer config — NOT the union-
        # cartesian of all cells' outer values.  Previously the OUTER ×
        # CELL cartesian doubled task count when a cell-list spanned both
        # RECT=0 and RECT=1 (or any other outer-axis split); half the
        # tasks then failed the cell-list filter inside the worker and
        # wasted dispatch overhead.  This path is O(|unique cell groups|).
        from collections import defaultdict as _dd
        cell_groups: dict = _dd(list)
        for tup in cell_set:
            d = dict(zip(cell_keys, tup))
            cell_outer = (
                "SR" if d.get("SR", 0) else "RN",       # sr_mode
                bool(d.get("RECT", 0)),                  # rect
                d.get("MODE", "persistent_dynamic"),     # mode
                bool(d.get("HSORT", 0)),                 # hardcode_sort
            )
            inner = {}
            for k, v in d.items():
                if k in _CELL_LIST_KEY_TO_ARG:
                    inner[_CELL_LIST_KEY_TO_ARG[k]] = str(v)
            non_cps_sig = tuple(sorted((k, v) for k, v in inner.items() if k not in _cps_keys))
            cell_groups[(cell_outer, non_cps_sig)].append(inner)

        # CLI-runtime axes (batch/mtp/dtype) are NOT in cell-list — they
        # come from CLI args and cartesian here (typically just 1 combo).
        cli_outers = []
        for _b in _compile_batches:
            for _m in mtp_lengths:
                _pk = _resolve_prev_ks(args, _m)
                for _sd in state_dtypes:
                    for _ad in act_dtypes:
                        cli_outers.append((_b, _m, _pk, _sd, _ad))

        tasks = []
        for cli_outer in cli_outers:
            for (cell_outer, _sig), inner_list in cell_groups.items():
                outer_cfg = (*cli_outer, *cell_outer)
                tasks.append((outer_cfg, inner_list))
        n_groups = len(cell_groups)
        n_total_cells = sum(len(g) for g in cell_groups.values())
        n_outer_used = len(cli_outers)
    else:
        # ============== SWEEP-ARGS PATH ==============
        # Build inner_dicts via cartesian over knob axes, then cross with
        # the `configs` outer cartesian.  Existing behavior.
        m_pairs   = _split_or_pair("block_size_m",     "block_size_m_write",     "block_size_m_nowrite")
        w_pairs   = _split_or_pair("num_warps",        "num_warps_write",        "num_warps_nowrite")
        ns_pairs  = _split_or_pair("num_stages",       "num_stages_write",       "num_stages_nowrite")
        cps_pairs = _split_or_pair("cta_per_sm",       "cta_per_sm_write",       "cta_per_sm_nowrite")
        ls_pairs  = _split_or_pair("num_loop_stages",  "num_loop_stages_write",  "num_loop_stages_nowrite")
        pw_vals   = _ps(args.precompute_num_warps)
        ps_vals   = _ps(args.precompute_num_stages)
        h_vals    = _ps(args.heads_per_block)
        mr_vals   = _ps(args.maxnreg)
        ct_vals   = _ps(args.num_ctas)
        fl_vals   = _ps(args.flatten)
        wsp_vals  = _ps(args.warp_specialize)
        trl_vals  = _ps(args.use_tma_rect_load)
        twl_vals  = _ps(args.use_tma_replay_write_load)
        tnl_vals  = _ps(args.use_tma_replay_nowrite_load)
        tws_vals  = _ps(args.use_tma_replay_write_store)
        import itertools as _it
        inner_dicts = []
        for ((mw, mnw), (ww, wnw), (sw, snw), (cw, cnw), (lw, lnw),
             pw, ps_, h, mr, ct, fl, wsp,
             trl, twl, tnl, tws) in _it.product(
                m_pairs, w_pairs, ns_pairs, cps_pairs, ls_pairs,
                pw_vals, ps_vals, h_vals, mr_vals, ct_vals,
                fl_vals, wsp_vals,
                trl_vals, twl_vals, tnl_vals, tws_vals):
            d = {}
            for k, v in (
                ("block_size_m_write", mw),
                ("block_size_m_nowrite", mnw),
                ("num_warps_write", ww),
                ("num_warps_nowrite", wnw),
                ("num_stages_write", sw),
                ("num_stages_nowrite", snw),
                ("cta_per_sm_write", cw),
                ("cta_per_sm_nowrite", cnw),
                ("num_loop_stages_write", lw),
                ("num_loop_stages_nowrite", lnw),
                ("precompute_num_warps", pw),
                ("precompute_num_stages", ps_),
                ("heads_per_block", h),
                ("maxnreg", mr),
                ("num_ctas", ct),
                ("flatten", fl),
                ("warp_specialize", wsp),
                ("use_tma_rect_load", trl),
                ("use_tma_replay_write_load", twl),
                ("use_tma_replay_nowrite_load", tnl),
                ("use_tma_replay_write_store", tws),
            ):
                if v is not None:
                    d[k] = str(v)
            inner_dicts.append(d)

        groups: dict = {}
        for d in inner_dicts:
            sig = tuple(sorted((k, v) for k, v in d.items() if k not in _cps_keys))
            groups.setdefault(sig, []).append(d)
        tasks = []
        for outer in configs:
            for sig, group in groups.items():
                tasks.append((outer, group))
        n_groups = len(groups)
        n_total_cells = sum(len(g) for g in groups.values())
        n_outer_used = len(configs)

    # Shuffle ACROSS tasks (preserve within-group CPS sequence for in-process
    # cache adjacency — within-group order is intentional, not shuffled).
    import random as _r
    _r.shuffle(tasks)

    _cw(f"built {len(tasks)} tasks covering {n_total_cells} cells in {n_groups} groups")
    print(f"[compile-warmup] {len(tasks)} compile tasks "
          f"({n_outer_used} outer × {n_groups} cell-groups "
          f"covering {n_total_cells} cells, CPS-grouped"
          + (", per-cell outer" if cell_set else "")
          + f") across {max_workers} processes (ProcessPoolExecutor, {_MP_START_METHOD} start)")
    t0 = time.perf_counter()

    ctx = multiprocessing.get_context(_MP_START_METHOD)
    errors = []
    _cw("about to create ProcessPoolExecutor")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        _cw("ProcessPoolExecutor created, about to submit tasks")
        # baseline_fn=None: workers compile only the replay kernel.
        # Baseline kernels (if any) get compiled lazily in the parent during
        # the timing phase — usually just one extra compile, negligible.
        futures = {
            ex.submit(_warm_one_config, args, task, None): task
            for task in tasks
        }
        _cw(f"submitted {len(futures)} tasks, waiting for results")
        _n_done = 0
        for fut in futures:
            try:
                fut.result()
            except Exception as e:
                errors.append((futures[fut], e))
            _n_done += 1
            # Progress beacons at 10/25/50/75/100% to gauge effective parallelism.
            if _n_done in (max(1, len(futures)//10),
                           max(1, len(futures)//4),
                           max(1, len(futures)//2),
                           max(1, (3*len(futures))//4),
                           len(futures)):
                _cw(f"{_n_done}/{len(futures)} tasks complete")

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
    mode: str = "persistent_dynamic",
    mix_samples_cpu=None,
    mix_label: str = "",
    hardcode_sort: bool = False,
    mix_samples_sorted_cpu=None,
    mix_write_frac: float | None = None,
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
        replay_work_items_buf,
        out_incr,
        out_base,
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
    mode = _resolve_effective_replay_mode(
        args, batch, state_dtype, use_philox, mode
    )

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

    is_pr3324_baseline = _is_flashinfer_pr3324_baseline(args)

    # Silently skip the baseline row for any (baseline, state_dtype, SR)
    # combo it can't run.  Better than erroring on a partial sweep — our
    # kernel rows still print.  Compatibility:
    #   * flashinfer_pr3324: enable the dtypes that match this benchmark's
    #     semantics. Skip int16 because PR3324 treats it as raw int16 state,
    #     without our per-channel state_scale path; skip bf16 because we do not
    #     use bf16 state.
    def _baseline_supports() -> bool:
        if baseline_fn is None:
            return False
        assert is_pr3324_baseline, args.baseline
        if state_dtype not in (
            torch.float32, torch.float16, torch.int8, torch.float8_e4m3fn,
        ):
            return False
        return not (use_philox and state_dtype == torch.float32)

    if baseline_fn is not None and not _baseline_supports():
        if not warmup_only:
            sr_tag = " + SR" if use_philox else ""
            print(
                f"# Skipping {args.baseline} baseline for "
                f"state_dtype={state_dtype_name}{sr_tag} (unsupported)."
            )
        baseline_fn = None

    show_kernel_col = args.baseline is not None

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
    # Per-main split-knob sweeps.  Default = same as the shared sweep (so each
    # combo is tied).  When set independently, the inner loop sweeps the
    # cross-product (write × nowrite); --skip-diagonal drops the tied subset.
    def _split_or_share(split_csv, shared_values):
        return _parse_sweep(split_csv) if split_csv else shared_values
    block_size_m_write_values = _split_or_share(args.block_size_m_write, block_size_m_values)
    block_size_m_nowrite_values = _split_or_share(args.block_size_m_nowrite, block_size_m_values)
    num_warps_write_values = _split_or_share(args.num_warps_write, num_warps_values)
    num_warps_nowrite_values = _split_or_share(args.num_warps_nowrite, num_warps_values)
    num_stages_write_values = _split_or_share(args.num_stages_write, num_stages_values)
    num_stages_nowrite_values = _split_or_share(args.num_stages_nowrite, num_stages_values)
    cta_per_sm_write_values = _split_or_share(args.cta_per_sm_write, cta_per_sm_values)
    cta_per_sm_nowrite_values = _split_or_share(args.cta_per_sm_nowrite, cta_per_sm_values)
    num_loop_stages_write_values = _split_or_share(args.num_loop_stages_write, num_loop_stages_values)
    num_loop_stages_nowrite_values = _split_or_share(args.num_loop_stages_nowrite, num_loop_stages_values)
    # Whether any *_write / *_nowrite knob was independently set — used by
    # --skip-diagonal to know if the cross-product is non-trivial.  Without
    # any split, the per-main values == shared values and skip-diagonal is
    # a no-op (which is correct).
    _any_split = any(getattr(args, name) for name in (
        "block_size_m_write", "block_size_m_nowrite",
        "num_warps_write", "num_warps_nowrite",
        "num_stages_write", "num_stages_nowrite",
        "cta_per_sm_write", "cta_per_sm_nowrite",
        "num_loop_stages_write", "num_loop_stages_nowrite",
    ))
    # TMA toggles — independent 0/1 sweep per path.  The skip-dupe at the
    # top of the inner loop body collapses cells where a flag's path is
    # unreachable for the current rectangle_for_nowrite setting.
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
    if not (getattr(args, "mix_only", False) and mix_samples_cpu is not None):
        for prev_k in prev_ks:
            # Persistent modes dispatch per-slot from PNAT, so any prev_k
            # <= max_window is valid.
            scenarios.append({
                "label": f"k{prev_k}",
                "print_label": prev_k,
                "fill": prev_k,
                "pre_iter": None,
                "iters": None,  # use args.iters
            })
    # Mix scenario: bench pre-bakes per-iter PNAT samples, n_writes, and
    # replay_work_items. Grouped graph capture copies window rows into the
    # persistent kernel-input tensors before each in-graph L2 flush, so timed
    # kernels read the metadata cold.
    if mix_samples_cpu is not None:
        device = state_work.device
        # Hardcode-sort: per-iter prev_tokens are CPU-sorted write-first.
        # Output is scrambled (we don't permute x/B/C/dt to match) but
        # timing is meaningful as a clustering experiment.
        src = mix_samples_sorted_cpu if (hardcode_sort and mix_samples_sorted_cpu is not None) else mix_samples_cpu
        samples_gpu = torch.from_numpy(src).to(device=device, dtype=torch.int32)

        n_writes_per_iter_all, replay_work_items_samples_cpu = (
            _build_replay_work_items_cpu(src, mtp_len, max_window)
        )
        n_writes_samples_gpu = torch.from_numpy(n_writes_per_iter_all).to(
            device=device, dtype=torch.int32
        )
        replay_work_items_samples_gpu = torch.from_numpy(
            replay_work_items_samples_cpu
        ).to(device=device, dtype=torch.int32)
        n_writes_mix = torch.zeros(1, dtype=torch.int32, device=device)

        def _mix_pre_iter(
            i,
            _s=samples_gpu,
            _ns=n_writes_samples_gpu,
            _wi=replay_work_items_samples_gpu,
            _pt=prev_tokens,
            _nw=n_writes_mix,
            _rwi=replay_work_items_buf,
        ):
            _pt.copy_(_s[i])
            _nw.copy_(_ns[i:i + 1])
            _rwi.copy_(_wi[i])

        def _mix_pre_iter_group_factory(
            group_iters,
            _s=samples_gpu,
            _ns=n_writes_samples_gpu,
            _wi=replay_work_items_samples_gpu,
            _pt=prev_tokens,
            _nw=n_writes_mix,
            _rwi=replay_work_items_buf,
        ):
            sample_window = torch.empty(
                (group_iters, _s.shape[1]), device=_s.device, dtype=_s.dtype,
            )
            n_writes_window = torch.empty(
                (group_iters,), device=_ns.device, dtype=_ns.dtype
            )
            work_items_window = torch.empty(
                (group_iters, _wi.shape[1], _wi.shape[2]),
                device=_wi.device,
                dtype=_wi.dtype,
            )

            def _pre_replay(replay_idx):
                start = replay_idx * group_iters
                end = start + group_iters
                sample_window.copy_(_s[start:end])
                n_writes_window.copy_(_ns[start:end])
                work_items_window.copy_(_wi[start:end])

            def _graph_pre_iter(j):
                _pt.copy_(sample_window[j])
                _nw.copy_(n_writes_window[j:j + 1])
                _rwi.copy_(work_items_window[j])

            return _pre_replay, _graph_pre_iter

        # Mix iters override: if --mix-iters set, use it; else use args.iters.
        mix_iters = getattr(args, "mix_iters", None)
        scenarios.append({
            "label": f"mix{mix_label}",
            "print_label": "mix",
            "fill": None,
            "pre_iter": _mix_pre_iter,
            "pre_iter_group_factory": _mix_pre_iter_group_factory,
            "iters": mix_iters,  # None => use args.iters
            "n_writes": n_writes_mix,
            # Full per-iter n_writes array (size = warmup + iters).  Used by
            # the JSON-detailed output to pair each iter's span with its
            # mix composition for post-hoc bucketing analysis.
            "n_writes_per_iter": n_writes_per_iter_all,
        })

    # Pure scenarios use one constant n_writes/work-items row. Mix scenarios
    # update both tensors per iter. persistent_main always launches both
    # halves because n_writes is device-resident for graph replay.
    for scn in scenarios:
        scenario_n_writes = scn.get("n_writes")
        if scn["fill"] is not None:
            prev_tokens.fill_(scn["fill"])
            n_writes_cpu, work_items_cpu = _build_replay_work_items_cpu(
                np.full((batch,), scn["fill"], dtype=np.int32),
                mtp_len,
                max_window,
            )
            scenario_n_writes = torch.from_numpy(n_writes_cpu).to(
                device=state_work.device, dtype=torch.int32
            )
            replay_work_items_buf.copy_(
                torch.from_numpy(work_items_cpu).to(
                    device=state_work.device, dtype=torch.int32
                )
            )
        prev_k_for_print = scn["print_label"]
        scenario_pre_iter = scn["pre_iter"]
        scenario_pre_iter_group_factory = scn.get("pre_iter_group_factory")
        scenario_iters = scn.get("iters")  # None => use args.iters
        tag = f"incr_b{batch}_mtp{mtp_len}_{scn['label']}_s{state_dtype_name}_a{act_dtype_name}"

        scenario_per_iter_nw = None
        if getattr(args, "json_detailed", False) or scn["fill"] is None:
            eff_iters = scenario_iters if scenario_iters is not None else args.iters
            if scn["fill"] is not None:
                if getattr(args, "json_detailed", False):
                    is_write = scn["fill"] + mtp_len > max_window
                    scenario_per_iter_nw = [batch if is_write else 0] * eff_iters
            else:
                nw_full = scn.get("n_writes_per_iter")
                if nw_full is not None:
                    scenario_per_iter_nw = (
                        nw_full[args.warmup:args.warmup + eff_iters].tolist()
                    )

        if baseline_fn is not None and is_pr3324_baseline:
            baseline_suffix_parts = [f"SR={int(use_philox)}"]
            hsort_is_swept = len(getattr(args, "hardcode_sort_list", [False])) > 1
            if scn["fill"] is None and (hardcode_sort or hsort_is_swept):
                baseline_suffix_parts.append(f"HSORT={1 if hardcode_sort else 0}")
            baseline_sweep_suffix = ",".join(baseline_suffix_parts)
            baseline_key = _build_json_key(
                args.baseline,
                batch,
                mtp_len,
                prev_k_for_print,
                state_dtype_name,
                baseline_sweep_suffix,
                args.tp_size,
            )
            run_baseline = True
            if not warmup_only:
                baseline_seen_keys = getattr(args, "_baseline_seen_keys", None)
                if baseline_seen_keys is None:
                    baseline_seen_keys = set()
                    args._baseline_seen_keys = baseline_seen_keys
                if (
                    baseline_key in getattr(args, "_done_keys", set())
                    or baseline_key in baseline_seen_keys
                ):
                    run_baseline = False
                else:
                    baseline_seen_keys.add(baseline_key)

            base_tag = (
                f"base_pr3324_b{batch}_mtp{mtp_len}_{scn['label']}_"
                f"s{state_dtype_name}_a{act_dtype_name}"
            )

            def _run_pr3324_baseline():
                if with_conv1d:
                    x_call, B_call, C_call = _conv1d_split(
                        xbc_input_work,
                        conv_state_work,
                        launch_dependent_kernels=args.external_pdl,
                    )
                else:
                    x_call, B_call, C_call = x, B, C
                baseline_fn(
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
                    out=out_base,
                    D=D,
                    dt_bias=dt_bias,
                    dt_softplus=True,
                    state_batch_indices=None,
                    state_scale=state_scales_work,
                    rand_seed=rand_seed,
                    philox_rounds=args.philox_rounds,
                    enable_pdl=with_conv1d and args.external_pdl,
                )

            reset_fn = _reset_conv1d_realistic if with_conv1d else _reset
            if warmup_only:
                reset_fn()
                if scenario_pre_iter is not None:
                    scenario_pre_iter(0)
                _run_pr3324_baseline()
                torch.cuda.synchronize()
            elif run_baseline:
                stats = _time_kernel(
                    args,
                    _run_pr3324_baseline,
                    reset_fn,
                    base_tag,
                    expected_K=_kernels_per_iter_baseline(with_conv1d),
                    pre_iter_fn=scenario_pre_iter,
                    pre_iter_group_factory=scenario_pre_iter_group_factory,
                    iters_override=scenario_iters,
                    cupti_plan_key=(
                        "baseline",
                        args.baseline,
                        batch,
                        mtp_len,
                        prev_k_for_print,
                        state_dtype_name,
                        act_dtype_name,
                        with_conv1d,
                        bool(args.l2_flush),
                        bool(args.external_pdl),
                        bool(use_philox),
                        _kernels_per_iter_baseline(with_conv1d),
                    ),
                )
                _submit_result_job(
                    args,
                    stats,
                    show_kernel_col=show_kernel_col,
                    kernel_name=args.baseline,
                    batch=batch,
                    mtp_len=mtp_len,
                    prev_k=prev_k_for_print,
                    state_dtype_name=state_dtype_name,
                    act_dtype_name=act_dtype_name,
                    sweep_suffix=baseline_sweep_suffix,
                    per_iter_nw=scenario_per_iter_nw,
                    kmix_bucket_write_frac=mix_write_frac
                    if scn["fill"] is None else None,
                    skipped_tag=base_tag,
                )

        # Iteration over per-cell knob combos.
        # When NO per-main split is requested (_any_split=False), each row in
        # the cross-product gives the same value to both write_main and
        # nowrite_main (current behavior — backward-compat).  When ANY split
        # IS requested, we iterate the write and nowrite axes independently
        # (cross-product blowup is the user's responsibility — they typically
        # pair this with --skip-diagonal to drop the tied subset).
        if _any_split:
            _iter_axes = (
                block_size_m_write_values, block_size_m_nowrite_values,
                num_warps_write_values, num_warps_nowrite_values,
                num_stages_write_values, num_stages_nowrite_values,
                precompute_num_warps_values,
                precompute_num_stages_values,
                heads_per_block_values,
                maxnreg_values, num_ctas_values,
                cta_per_sm_write_values, cta_per_sm_nowrite_values,
                num_loop_stages_write_values, num_loop_stages_nowrite_values,
                flatten_values, warp_specialize_values,
                use_tma_rect_load_values,
                use_tma_replay_write_load_values,
                use_tma_replay_nowrite_load_values,
                use_tma_replay_write_store_values,
            )
        else:
            # Tied: one value per shared knob.  Wrap in single-element list for
            # uniform iteration; the body sets w/nw both to the shared value.
            _iter_axes = (
                block_size_m_values, [None],
                num_warps_values, [None],
                num_stages_values, [None],
                precompute_num_warps_values,
                precompute_num_stages_values,
                heads_per_block_values,
                maxnreg_values, num_ctas_values,
                cta_per_sm_values, [None],
                num_loop_stages_values, [None],
                flatten_values, warp_specialize_values,
                use_tma_rect_load_values,
                use_tma_replay_write_load_values,
                use_tma_replay_nowrite_load_values,
                use_tma_replay_write_store_values,
            )
        # Iteration source: when --cell-list is active AND this is the main
        # timing path (not a compile-warmup worker), iterate the cell set
        # DIRECTLY (one yield per cell).  The earlier design iterated the
        # full inner cartesian and filtered each iteration via membership in
        # args._cell_list_set — that's O(cartesian) which blows up to
        # billions of iterations when the cell-list spans wide split-knob
        # values (CPS, LS, M, W, S each contributing a Wx*Wnw factor on top
        # of TMA flags), producing 50+ min of CPU spin per bench call before
        # any actual timing.  Direct iteration is O(|cell_list|).
        #
        # IMPORTANT exception for workers (warmup_only=True): _warm_one_config
        # clamps args.*_write/_nowrite via inner_overrides to single values,
        # making the cartesian 1×1×...×1 = 1 iter, which is exactly the one
        # cell that worker was given.  If we used cell-list-direct iteration
        # here, every worker would iterate ALL 2884 cells instead of just
        # its assigned one — turning compile-warmup into 28-way duplication.
        # (Observed: 256 tasks in 233s under that bug vs ~18s correct.)
        if getattr(args, "_cell_list_set", None) and not warmup_only:
            def _gen_from_cell_list():
                keys = args._cell_list_keys
                for tup in args._cell_list_set:
                    d = dict(zip(keys, tup))
                    yield (
                        d.get("Mw"), d.get("Mnw"),
                        d.get("Ww"), d.get("Wnw"),
                        d.get("Sw"), d.get("Snw"),
                        d.get("pW"), d.get("pS"),
                        d.get("H"),
                        d.get("R"), d.get("CT"),
                        d.get("CPSw"), d.get("CPSnw"),
                        d.get("LSw"), d.get("LSnw"),
                        d.get("FL"), d.get("WS"),
                        d.get("TMARL"), d.get("TMAWL"),
                        d.get("TMANL"), d.get("TMAWS"),
                    )
            _iter_source = _gen_from_cell_list()
        else:
            _iter_source = itertools.product(*_iter_axes)

        for (
            block_size_m_w,
            block_size_m_nw,
            num_warps_w,
            num_warps_nw,
            num_stages_w,
            num_stages_nw,
            precompute_num_warps,
            precompute_num_stages,
            heads_per_block,
            maxnreg,
            num_ctas,
            cta_per_sm_w,
            cta_per_sm_nw,
            num_loop_stages_w,
            num_loop_stages_nw,
            flatten,
            warp_specialize,
            use_tma_rect_load,
            use_tma_replay_write_load,
            use_tma_replay_nowrite_load,
            use_tma_replay_write_store,
        ) in _iter_source:
            # When tied, _nw values were placeholder None; fill from _w (the
            # shared value).  When split, _w and _nw came from independent lists.
            if not _any_split:
                block_size_m_nw = block_size_m_w
                num_warps_nw = num_warps_w
                num_stages_nw = num_stages_w
                cta_per_sm_nw = cta_per_sm_w
                num_loop_stages_nw = num_loop_stages_w
            # Skip-diagonal: when split is on, drop the tied subset (same as a
            # prior shared-knob sweep would cover).
            if _any_split and args.skip_diagonal and (
                block_size_m_w == block_size_m_nw and
                num_warps_w == num_warps_nw and
                num_stages_w == num_stages_nw and
                cta_per_sm_w == cta_per_sm_nw and
                num_loop_stages_w == num_loop_stages_nw
            ):
                continue
            # Backward-compat aliases used by the existing body below.  When
            # tied, these are simply the shared value.  When split, the
            # _write copy is used for sweep_tag and grouping (a stable choice
            # so the tag is unique per (write, nowrite) combo).
            block_size_m = block_size_m_w
            num_warps = num_warps_w
            num_stages = num_stages_w
            cta_per_sm = cta_per_sm_w
            num_loop_stages = num_loop_stages_w
            # Skip-dupe for TMA flag sweeps: a flag whose code path isn't
            # reachable in this cell produces identical timing for value=0
            # and value=1.  We canonicalize by skipping value=1 cells when
            # the flag's path is unreachable.  Path reachability rules:
            #   * write path (replay write-load + write-store): always true.
            #   * rect path (rect-load): rectangle_for_nowrite=True.
            #   * replay-nowrite path (nowrite-load): rect isn't taking it.
            _write_path = True  # both halves exist for persistent modes
            _rect_path = rectangle_for_nowrite
            _replay_nowrite_path = not rectangle_for_nowrite
            def _set(v):  # flag set to a non-zero sweep value
                return v is not None and v != 0
            if (_set(use_tma_rect_load) and not _rect_path
                    or _set(use_tma_replay_write_load) and not _write_path
                    or _set(use_tma_replay_nowrite_load) and not _replay_nowrite_path
                    or _set(use_tma_replay_write_store) and not _write_path):
                continue

            assert scenario_n_writes is not None

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
                extra_kwargs["rectangle_for_nowrite"] = rectangle_for_nowrite
                extra_kwargs["mode"] = mode
                extra_kwargs["n_writes"] = scenario_n_writes
                extra_kwargs["replay_work_items"] = replay_work_items_buf
                if state_scales_work is not None:
                    extra_kwargs["state_scales"] = state_scales_work
                if use_tma_rect_load is not None:
                    extra_kwargs["_use_tma_rect_load"] = bool(use_tma_rect_load)
                if use_tma_replay_write_load is not None:
                    extra_kwargs["_use_tma_replay_write_load"] = bool(use_tma_replay_write_load)
                if use_tma_replay_nowrite_load is not None:
                    extra_kwargs["_use_tma_replay_nowrite_load"] = bool(use_tma_replay_nowrite_load)
                if use_tma_replay_write_store is not None:
                    extra_kwargs["_use_tma_replay_write_store"] = bool(use_tma_replay_write_store)
                if cta_per_sm is not None:
                    extra_kwargs["_cta_per_sm"] = cta_per_sm
                if num_loop_stages is not None:
                    extra_kwargs["_num_loop_stages"] = num_loop_stages
                if flatten is not None:
                    extra_kwargs["_flatten"] = bool(flatten)
                if warp_specialize is not None:
                    extra_kwargs["_warp_specialize"] = bool(warp_specialize)

                replay_selective_state_update(
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
                    # Per-main overrides (None = tied to shared above; explicit
                    # only when the inner loop is iterating split axes).
                    _block_size_m_write=block_size_m_w if _any_split else None,
                    _block_size_m_nowrite=block_size_m_nw if _any_split else None,
                    _num_warps_write=num_warps_w if _any_split else None,
                    _num_warps_nowrite=num_warps_nw if _any_split else None,
                    _num_stages_write=num_stages_w if _any_split else None,
                    _num_stages_nowrite=num_stages_nw if _any_split else None,
                    _cta_per_sm_write=cta_per_sm_w if _any_split else None,
                    _cta_per_sm_nowrite=cta_per_sm_nw if _any_split else None,
                    _num_loop_stages_write=num_loop_stages_w if _any_split else None,
                    _num_loop_stages_nowrite=num_loop_stages_nw if _any_split else None,
                    **extra_kwargs,
                )

            parts = []
            # Tuned wrapper knobs emit "auto" when unset, meaning the wrapper
            # resolves them from _DEFAULT_TUNING per cell.  pS/R/CT are not
            # tuning-table knobs today, so leave them out unless explicitly
            # swept.
            def _val(v):
                return "auto" if v is None else v

            # When tied (not _any_split), emit the shared single-value tag
            # (M=8 etc).  When split, emit explicit Mw / Mnw tags so cells
            # with the same shared value but different per-main values get
            # unique JSON keys.
            def _emit_split(name_w, name_nw, val_w, val_nw):
                if val_w is None and val_nw is None:
                    parts.append(f"{name_w[:-1]}=auto")  # tied form, both auto
                    return
                if not _any_split or val_w == val_nw:
                    parts.append(f"{name_w[:-1]}={_val(val_w)}")
                else:
                    parts.append(f"{name_w}={_val(val_w)}")
                    parts.append(f"{name_nw}={_val(val_nw)}")
            _emit_split("Mw", "Mnw", block_size_m_w, block_size_m_nw)
            _emit_split("Ww", "Wnw", num_warps_w, num_warps_nw)
            _emit_split("Sw", "Snw", num_stages_w, num_stages_nw)
            parts.append(f"pW={_val(precompute_num_warps)}")
            parts.append(f"H={_val(heads_per_block)}")
            optional_tag_parts = (
                ("pS", precompute_num_stages),
                ("R", maxnreg),
                ("CT", num_ctas),
            )
            for _name, _value in optional_tag_parts:
                if _value is not None:
                    parts.append(f"{_name}={_value}")
            # Persistent-only knobs (only meaningful when MODE=persistent_main;
            # printed unconditionally so output rows are uniformly comparable
            # across modes when the user passed these sweeps).
            _emit_split("CPSw", "CPSnw", cta_per_sm_w, cta_per_sm_nw)
            _emit_split("LSw", "LSnw", num_loop_stages_w, num_loop_stages_nw)
            parts.append(f"FL={_val(flatten)}")
            parts.append(f"WS={_val(warp_specialize)}")
            # TMA sweep tags.  Four wrapper-level flags map to three
            # kernel-level constexprs (rect-load and replay-nowrite-load
            # share `USE_TMA_LOAD_NOWRITE`, picked by the wrapper based on
            # RECTANGLE).  TMARL specifically gates the rectangle path's
            # state load; TMANL specifically gates the replay-style
            # nowrite path's state load.
            parts.append(f"TMARL={_val(use_tma_rect_load)}")
            parts.append(f"TMAWL={_val(use_tma_replay_write_load)}")
            parts.append(f"TMANL={_val(use_tma_replay_nowrite_load)}")
            parts.append(f"TMAWS={_val(use_tma_replay_write_store)}")
            parts.append(f"SR={1 if use_philox else 0}")
            parts.append(f"RECT={'auto' if rectangle_for_nowrite is None else (1 if rectangle_for_nowrite else 0)}")
            parts.append(f"MODE={_val(mode)}")
            parts.append(f"HSORT={1 if hardcode_sort else 0}")
            sweep_suffix = (" " + ",".join(parts)) if parts else ""
            sweep_tag = tag + sweep_suffix.replace(" ", "_").replace(",", "_")

            reset_fn = _reset_conv1d_realistic if with_conv1d else _reset
            # --cell-list filter: only time cells whose (canonical-knob-values)
            # tuple is in the loaded set.  Robust to bench gaining new knobs
            # (old cell-list files keep working: any keys they don't list
            # become wildcards that retain CLI defaults).
            if args._cell_list_keys:
                _tup = _current_cell_tuple(args, locals())
                if _tup is None or _tup not in args._cell_list_set:
                    continue
            # Resume from JSONL: skip cells already recorded.  Built the same
            # way _print_row builds JSON keys; must stay in sync.
            done_keys = getattr(args, "_done_keys", None)
            if done_keys:
                # One key per scenario (k=6, k=11, mix) — skip the whole cell
                # only if ALL of its scenarios are already done.  We don't
                # know which scenarios will be emitted here without
                # re-evaluating the inner scenario loop; conservatively skip
                # only when the prev_k_for_print's specific key is done.
                _resume_key = _build_json_key(
                    "replay", batch, mtp_len, prev_k_for_print,
                    state_dtype_name, sweep_suffix, args.tp_size,
                )
                if _resume_key in done_keys:
                    continue
            if warmup_only:
                reset_fn()
                if scenario_pre_iter is not None:
                    scenario_pre_iter(0)
                _run_incr()
                torch.cuda.synchronize()
            else:
                # Inline retry: CUPTI sometimes loses records under PDL +
                # high cell count; retrying the SAME cell often catches it
                # because the failure is transient at the kernel-launch level.
                # Per --cupti-retry budget.  On final failure, append tag to
                # the skipped list for an external rerun in a fresh process.
                defer_results = (
                    args.cuda_graph
                    and getattr(args, "cupti", True)
                    and int(getattr(args, "cupti_defer_depth", 1)) > 1
                )
                retry_budget = 0 if defer_results else max(0, getattr(args, "cupti_retry", 1))
                stats = None
                expected_K = _kernels_per_iter_incremental(
                    mode, with_conv1d=with_conv1d,
                )
                plan_key = (
                    "incremental",
                    "replay",
                    mode,
                    batch,
                    mtp_len,
                    state_dtype_name,
                    act_dtype_name,
                    with_conv1d,
                    bool(args.l2_flush),
                    bool(args.external_pdl),
                    bool(args.internal_pdl),
                    bool(use_philox),
                    bool(rectangle_for_nowrite),
                    bool(hardcode_sort),
                    scenario_pre_iter is not None,
                    expected_K,
                )
                for attempt in range(retry_budget + 1):
                    stats = _time_kernel(
                        args, _run_incr, reset_fn, sweep_tag,
                        expected_K=expected_K,
                        pre_iter_fn=scenario_pre_iter,
                        pre_iter_group_factory=scenario_pre_iter_group_factory,
                        iters_override=scenario_iters,
                        cupti_plan_key=plan_key,
                    )
                    if stats is not None:
                        break
                    if attempt < retry_budget:
                        print(
                            f"[retry] CUPTI mismatch on {sweep_tag!r}; "
                            f"retrying ({attempt + 1}/{retry_budget})",
                            file=sys.stderr,
                            flush=True,
                        )
                if stats is None:
                    args._skipped_cells.append(sweep_tag)

                per_iter_nw = scenario_per_iter_nw if stats is not None else None

                if stats is not None:
                    _submit_result_job(
                        args,
                        stats,
                        show_kernel_col=show_kernel_col,
                        kernel_name="replay",
                        batch=batch,
                        mtp_len=mtp_len,
                        prev_k=prev_k_for_print,
                        state_dtype_name=state_dtype_name,
                        act_dtype_name=act_dtype_name,
                        sweep_suffix=sweep_suffix,
                        per_iter_nw=per_iter_nw,
                        kmix_bucket_write_frac=mix_write_frac
                        if scn["fill"] is None else None,
                        skipped_tag=sweep_tag,
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
      flashinfer_pr3324/{batch}/{mtp}/{sd}/k{k}/{sweep_parts}/tp{tp}

    Replay rows collapse to "incremental"; baseline rows keep their baseline
    name.
    """
    if kernel_name == "replay":
        kind = "incremental"
    else:
        kind = kernel_name

    sd = _DTYPE_SHORT.get(state_dtype_name, state_dtype_name)
    parts = [kind, str(batch), str(mtp_len), sd]
    if prev_k != "N/A":
        parts.append(f"k{prev_k}")
    if sweep_suffix:
        # sweep_suffix format: " M=4,W=1,S=1,SR=0,RECT=0"
        # collect.py format:    "M4_W1_S1_SR0_RECT0"
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
    tp_size=None,
    json_detailed=False,
    jsonl_path=None,
    jsonl_host=None,
    jsonl_gpu=None,
):
    """Print one summary row and append the result to the JSONL sidecar.

    `stats` is a dict from _time_kernel: {median, p95, p99, n, iters_us,
    [n_writes_per_iter], [kmix_bucket_score], [per_kernel]}. The summary
    table shows kmix_bucket_score when present, otherwise median. JSONL
    captures the compact per-iter spans +
    n_writes_per_iter by default; with json_detailed=True it also captures
    per-kernel data.

    When `jsonl_path` is provided, appends one JSON line per row to the
    JSONL sidecar (crash-safe incremental persistence; lets a killed sweep
    resume from the last completed cell on rerun, even across hosts).  Open
    per-write because `args` is pickled to ProcessPoolExecutor workers and
    file handles aren't picklable.  JSONL is the canonical artifact — the
    bench no longer writes a final `.json` summary; use `jsonl_to_json.py`
    if a one-shot `.json` snapshot is needed.
    """
    kernel_col = f"{kernel_name:>11} | " if show_kernel_col else ""
    headline_us = stats.get("kmix_bucket_score", stats["median"])
    print(
        f"| {kernel_col}{batch:>5} | {mtp_len:>7} | {str(prev_k):>6} | "
        f"{state_dtype_name:>11} | {act_dtype_name:>9} | "
        f"{headline_us:>9.2f} | {stats['p95']:>7.2f} | {stats['p99']:>7.2f} |"
        f"{sweep_suffix}"
    )
    if jsonl_path is not None:
        key = _build_json_key(
            kernel_name, batch, mtp_len, prev_k, state_dtype_name,
            sweep_suffix, tp_size,
        )
        if json_detailed:
            row_stats = stats
        else:
            row_stats = {
                k: stats[k]
                for k in (
                    "median", "p95", "p99", "n", "iters_us",
                    "n_writes_per_iter", "kmix_bucket_score",
                )
                if k in stats
            }
        if "host_timing" in stats:
            row_stats["host_timing"] = stats["host_timing"]
        # Append to JSONL sidecar if a path is set (incremental persistence).
        # Open per-write because args is pickled to ProcessPoolExecutor
        # workers, and file handles aren't picklable.  A clean SIGTERM or
        # Python exception will leave the file consistent up to the last
        # newline; catastrophic kills can leave a partial last line, which
        # the resume reader tolerates via json.JSONDecodeError pass.
        if jsonl_path is not None:
            # Wall-clock timestamp (float seconds since UNIX epoch) at write
            # time.  Lets post-hoc analysis diff consecutive rows to derive
            # per-cell wall budget and identify startup-bound vs steady-state
            # segments (cells/sec, downtime between bench invocations) without
            # needing to instrument the bench's outer loops separately.
            import time as _time
            rec = {"key": key, "stats": row_stats, "t": _time.time()}
            if jsonl_host is not None:
                rec["host"] = jsonl_host
            if jsonl_gpu is not None:
                rec["gpu"] = jsonl_gpu
            with open(jsonl_path, "a") as f:
                f.write(json.dumps(rec) + "\n")


def _finish_result_job(args, job: dict) -> None:
    result = job["result"]
    if isinstance(result, _PendingCuptiStats):
        stats = result.resolve()
    else:
        stats = result

    if stats is None:
        skipped_tag = job.get("skipped_tag")
        if skipped_tag is not None:
            args._skipped_cells.append(skipped_tag)
        return

    per_iter_nw = job.get("per_iter_nw")
    if per_iter_nw is not None:
        stats["n_writes_per_iter"] = per_iter_nw
        kmix_bucket_score = _kmix_bucket_score(
            stats.get("iters_us"),
            per_iter_nw,
            job["batch"],
            job.get("kmix_bucket_write_frac"),
        )
        if kmix_bucket_score is not None:
            stats["kmix_bucket_score"] = kmix_bucket_score

    _print_row(
        job["show_kernel_col"],
        job["kernel_name"],
        job["batch"],
        job["mtp_len"],
        job["prev_k"],
        job["state_dtype_name"],
        job["act_dtype_name"],
        stats,
        job.get("sweep_suffix", ""),
        tp_size=args.tp_size,
        json_detailed=getattr(args, "json_detailed", False),
        jsonl_path=getattr(args, "_jsonl_path", None),
        jsonl_host=getattr(args, "_jsonl_host", None),
        jsonl_gpu=getattr(args, "_jsonl_gpu", None),
    )


def _drain_pending_results(args, *, force: bool = False) -> None:
    pending_results = getattr(args, "_pending_results", None)
    if not pending_results:
        return

    max_pending = max(1, int(getattr(args, "cupti_defer_depth", 1)))
    while pending_results:
        first_result = pending_results[0]["result"]
        should_block = force or len(pending_results) >= max_pending
        if (
            not should_block
            and isinstance(first_result, _PendingCuptiStats)
            and not first_result.is_ready()
        ):
            break
        job = pending_results.pop(0)
        _finish_result_job(args, job)


def _submit_result_job(
    args,
    result,
    *,
    show_kernel_col,
    kernel_name,
    batch,
    mtp_len,
    prev_k,
    state_dtype_name,
    act_dtype_name,
    sweep_suffix="",
    per_iter_nw=None,
    kmix_bucket_write_frac=None,
    skipped_tag=None,
) -> None:
    job = {
        "result": result,
        "show_kernel_col": show_kernel_col,
        "kernel_name": kernel_name,
        "batch": batch,
        "mtp_len": mtp_len,
        "prev_k": prev_k,
        "state_dtype_name": state_dtype_name,
        "act_dtype_name": act_dtype_name,
        "sweep_suffix": sweep_suffix,
        "per_iter_nw": per_iter_nw,
        "kmix_bucket_write_frac": kmix_bucket_write_frac,
        "skipped_tag": skipped_tag,
    }
    if isinstance(result, _PendingCuptiStats):
        args._pending_results.append(job)
        _drain_pending_results(args)
    else:
        _finish_result_job(args, job)


# Cell-list mode — canonical knob-key mapping to argparse args + local
# loop variable.  See _load_cell_list_into_args / inner-loop filter.
#
# Each entry: cell-key → (args attribute name, comma-separated string flag)
# For split (write/nowrite) knobs, we use Xw / Xnw keys.  Tied forms (M, W,
# S, CPS, LS) accepted on load and expanded to their w/nw variants.
_CELL_LIST_KEY_TO_ARG = {
    "Mw":    "block_size_m_write",
    "Mnw":   "block_size_m_nowrite",
    "Ww":    "num_warps_write",
    "Wnw":   "num_warps_nowrite",
    "Sw":    "num_stages_write",
    "Snw":   "num_stages_nowrite",
    "CPSw":  "cta_per_sm_write",
    "CPSnw": "cta_per_sm_nowrite",
    "LSw":   "num_loop_stages_write",
    "LSnw":  "num_loop_stages_nowrite",
    "pW":    "precompute_num_warps",
    "pS":    "precompute_num_stages",
    "H":     "heads_per_block",
    "R":     "maxnreg",
    "CT":    "num_ctas",
    "FL":    "flatten",
    "WS":    "warp_specialize",
    "TMARL": "use_tma_rect_load",
    "TMAWL": "use_tma_replay_write_load",
    "TMANL": "use_tma_replay_nowrite_load",
    "TMAWS": "use_tma_replay_write_store",
    "RECT":  "rectangle_for_nowrite",
    "HSORT": "hardcode_sort",
    # MODE and SR get special handling (string values):
    # MODE → args.modes (single mode name)
    # SR → args.sr_modes ("RN" if 0, "SR" if 1)
}

# Split-knob tied form: "M" expands to both "Mw" and "Mnw".
_CELL_LIST_TIED_EXPANSIONS = {
    "M":   ("Mw", "Mnw"),
    "W":   ("Ww", "Wnw"),
    "S":   ("Sw", "Snw"),
    "CPS": ("CPSw", "CPSnw"),
    "LS":  ("LSw", "LSnw"),
}

def _normalize_cell(cell: dict) -> dict:
    """Expand tied-form keys (M, W, S, CPS, LS) to their w/nw variants.
    Returns a new dict with only canonical split-or-plain keys.
    """
    out = dict(cell)
    for tied, (w_key, nw_key) in _CELL_LIST_TIED_EXPANSIONS.items():
        if tied in out:
            v = out.pop(tied)
            out.setdefault(w_key, v)
            out.setdefault(nw_key, v)
    return out


def _load_cell_list_into_args(args) -> None:
    """Read --cell-list JSON, normalize, override args.* knob ranges, and
    populate args._cell_list_keys + args._cell_list_set for the inner-loop
    filter.  Errors out if cells aren't uniform (different key sets).
    """
    with open(args.cell_list) as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        sys.exit(f"--cell-list: expected JSON list, got {type(raw).__name__}")
    cells = [_normalize_cell(c) for c in raw]
    if not cells:
        print("[cell-list] empty list — nothing to time", file=sys.stderr)
        return
    allowed_keys = set(_CELL_LIST_KEY_TO_ARG) | {"MODE", "SR"}
    unknown_keys = sorted({
        key for cell in cells for key in cell
        if key not in allowed_keys
    })
    if unknown_keys:
        sys.exit(f"--cell-list: unknown keys {unknown_keys}")
    # All cells must share the same key set (uniform schema)
    keys0 = frozenset(cells[0].keys())
    for i, c in enumerate(cells[1:], start=1):
        if frozenset(c.keys()) != keys0:
            sys.exit(
                f"--cell-list: cells must have uniform key sets; cell[0] "
                f"has {sorted(keys0)} but cell[{i}] has {sorted(c.keys())}"
            )

    # Auto-cover: collect per-knob value set across all cells
    cover: dict = {}
    for c in cells:
        for k, v in c.items():
            cover.setdefault(k, set()).add(v)
    # Apply overrides
    for key, vals in cover.items():
        if key in _CELL_LIST_KEY_TO_ARG:
            arg_name = _CELL_LIST_KEY_TO_ARG[key]
            vals_str = ",".join(str(v) for v in sorted(vals))
            setattr(args, arg_name, vals_str)
        elif key == "MODE":
            args.modes = ",".join(sorted({str(v) for v in vals}))
        elif key == "SR":
            args.sr_modes = ",".join(sorted({"SR" if v else "RN" for v in vals}))
        else:
            print(f"[cell-list] WARNING: unknown key {key!r} in cells; "
                  f"will not override any args.* attribute (the value will "
                  f"still be matched in the filter if a matching local var "
                  f"is in scope)", file=sys.stderr)

    # Canonical key order (sorted) for tuple matching in the inner loop
    args._cell_list_keys = tuple(sorted(keys0))
    args._cell_list_set = {
        tuple(c[k] for k in args._cell_list_keys) for c in cells
    }
    print(f"[cell-list] loaded {len(cells)} cells with keys "
          f"{list(args._cell_list_keys)}; overrode args.* to auto-cover",
          file=sys.stderr)


# Maps cell-list key → name of the local variable in _bench_config's inner
# loop.  Used to extract the "current cell" tuple for the filter check.
# Keep in sync with the loop-variable names; the filter is lenient about
# missing names (it picks them up from the inner scope at runtime).
_CELL_LIST_KEY_TO_LOCAL = {
    "Mw":    "block_size_m_w",
    "Mnw":   "block_size_m_nw",
    "Ww":    "num_warps_w",
    "Wnw":   "num_warps_nw",
    "Sw":    "num_stages_w",
    "Snw":   "num_stages_nw",
    "CPSw":  "cta_per_sm_w",
    "CPSnw": "cta_per_sm_nw",
    "LSw":   "num_loop_stages_w",
    "LSnw":  "num_loop_stages_nw",
    "pW":    "precompute_num_warps",
    "pS":    "precompute_num_stages",
    "H":     "heads_per_block",
    "R":     "maxnreg",
    "CT":    "num_ctas",
    "FL":    "flatten",
    "WS":    "warp_specialize",
    "TMARL": "use_tma_rect_load",
    "TMAWL": "use_tma_replay_write_load",
    "TMANL": "use_tma_replay_nowrite_load",
    "TMAWS": "use_tma_replay_write_store",
    "RECT":  "rectangle_for_nowrite",
    "MODE":  "mode",
    "HSORT": "hardcode_sort",
    "SR":    "use_philox",
}


def _current_cell_tuple(args, locals_dict: dict) -> tuple | None:
    """Build the (key1=val1, key2=val2, ...) tuple for the current inner-loop
    iteration, matching args._cell_list_keys' order.  Used by the inner-loop
    filter to check membership in args._cell_list_set.  Returns None if any
    expected local is missing (the bench evolved a knob name — caller skips).
    """
    if not args._cell_list_keys:
        return None
    vals = []
    for k in args._cell_list_keys:
        local_name = _CELL_LIST_KEY_TO_LOCAL.get(k, k)
        if local_name not in locals_dict:
            return None
        v = locals_dict[local_name]
        # Coerce bools to ints to match cell-list JSON (1/0)
        if isinstance(v, bool):
            v = int(v)
        vals.append(v)
    return tuple(vals)


# Main benchmark loop


def _run_benchmark(args) -> None:
    # Phase-timing markers — emit timestamped checkpoints so a captured-stdout
    # run can later attribute wall time to setup vs compile-warmup vs prewarm
    # vs timing.  Single-line format makes log-grepping trivial.
    _phase_t0 = time.perf_counter()
    def _phase(label: str) -> None:
        dt = time.perf_counter() - _phase_t0
        print(f"[phase] t={dt:7.2f}s  {label}", file=sys.stderr, flush=True)
    _phase("enter _run_benchmark")

    # Pending-results FIFO for srxl's deferred CUPTI parsing pipeline.  Each
    # entry holds a _PendingCuptiStats handle; _drain_pending_results pulls
    # ready entries and routes them to _print_row (which appends to JSONL).
    args._pending_results = []

    # JSONL incremental sidecar.  Path = `<json_output>.jsonl`.  Each completed
    # cell appends one line `{"key": <json_key>, "stats": {...}, "host": <h>}`
    # to this file as it finishes timing.  On startup we read this sidecar (if
    # present) and populate _done_keys so a killed bench can resume without
    # redoing already-timed cells.  Crash-safe by construction: append-only
    # writes survive SIGTERM/SIGKILL/reboot mid-sweep.
    #
    # Resume is host-blind: _done_keys includes records from any host, so a
    # bench restarted on a different node fills in the missing cells without
    # redoing cells already covered elsewhere.  Cross-host *timings* aren't
    # directly comparable, but each JSONL record carries its `host` stamp so
    # the analyzer can group/compare per host.  This bench no longer writes a
    # final `.json` summary — the JSONL is the canonical artifact; use the
    # `jsonl_to_json.py` helper if a one-shot `.json` snapshot is needed.
    #
    # Note: we store only paths/strings on `args` because args is pickled to
    # ProcessPoolExecutor workers during compile-warmup, and file handles
    # (TextIOWrapper) aren't picklable.  _print_row open-appends per cell.
    args._jsonl_path = None
    args._done_keys: set[str] = set()
    args._baseline_seen_keys: set[str] = set()
    args._jsonl_host = None  # hostname stamp for the current run
    args._jsonl_gpu = None   # GPU device id stamp (current process visibility)
    if getattr(args, "json_output", None):
        import socket
        args._jsonl_host = socket.gethostname()
        # Capture GPU id once at startup.  Used by the oracle-cache layer in
        # search_driver to attribute timings to a specific (host, gpu) pair
        # for cross-process pruning.  os.environ['CUDA_VISIBLE_DEVICES']
        # is the right source pre-torch-init (it's what the harness sets);
        # post-init we could use torch.cuda.current_device() but we keep it
        # to env to avoid forcing a CUDA init at this point in startup.
        args._jsonl_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        args._jsonl_path = args.json_output + ".jsonl"
        # Read existing JSONL if present: load every record's key into the
        # skip set regardless of host (gap-fill on a new node).
        if os.path.exists(args._jsonl_path):
            n_loaded = 0
            host_counts: dict[str, int] = {}
            with open(args._jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        # Tolerate partial last line from a crash mid-write.
                        continue
                    k = rec.get("key")
                    if k is None:
                        continue
                    args._done_keys.add(k)
                    n_loaded += 1
                    rec_host = rec.get("host")
                    if rec_host:
                        host_counts[rec_host] = host_counts.get(rec_host, 0) + 1
            if n_loaded:
                host_summary = ", ".join(
                    f"{h}={n}" for h, n in sorted(host_counts.items())
                ) if host_counts else "(no host stamps)"
                print(
                    f"[resume] {args._jsonl_path}: loaded {n_loaded} prior "
                    f"cell results across hosts [{host_summary}]; sweep will "
                    f"skip them.  New cells stamp host={args._jsonl_host}.",
                    file=sys.stderr,
                )

        # Sidecar metadata: cmd, host, tp_size, cupti, etc. Written
        # once at startup; helps later analysis identify how this JSONL was
        # produced even though there's no top-level .json wrapper anymore.
        meta_path = args.json_output + ".meta.json"
        meta_payload = {
            "timestamp": datetime.now().isoformat(),
            "host": args._jsonl_host,
            "cmd": " ".join(sys.argv),
            "tp_size": getattr(args, "tp_size", None),
            "warmup": getattr(args, "warmup", None),
            "iters": getattr(args, "iters", None),
            "cupti": getattr(args, "cupti", False),
        }
        # Append to a list so successive runs (gap-fill, retry) keep history.
        existing_meta = []
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    existing_meta = json.load(f)
                if not isinstance(existing_meta, list):
                    existing_meta = [existing_meta]
            except (OSError, json.JSONDecodeError):
                existing_meta = []
        existing_meta.append(meta_payload)
        # Bench is sometimes invoked with --json-output pointing into a dir
        # the caller hasn't created (subprocess driver, search loop, etc.).
        # Ensure the dir exists before writing the meta sidecar OR the JSONL.
        os.makedirs(os.path.dirname(os.path.abspath(meta_path)), exist_ok=True)
        tmp = meta_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(existing_meta, f, indent=2)
        os.replace(tmp, meta_path)

    # Skipped cells accumulator — populated by _bench_config when CUPTI capture
    # mismatch causes a cell to be skipped.  Written to args.skipped_output
    # (or derived from json_output) at end of run.
    args._skipped_cells = []

    # Cell-list filter (replaces the old --retry-cells tag-string filter).
    # When set, the sweep iterates ONLY the cells described in the list.
    #
    # Each entry in the JSON file is a dict of canonical knob keys → values,
    # using the same names that appear in the sweep_tag (Mw/Mnw, Ww/Wnw,
    # Sw/Snw, pW, pS, H, R, CT, CPSw/CPSnw, LSw/LSnw, FL, WS, TMARL,
    # TMAWL, TMANL, TMAWS, SR, RECT, MODE, HSORT).  Each
    # cell may also use the tied forms M / W / S / CPS / LS (single value
    # applied to both write and nowrite halves).
    #
    # On load we:
    #   - Override the bench's CLI knob args (`args.block_size_m_write`,
    #     etc.) with the union of values present across all cells per knob,
    #     so the cartesian iteration auto-covers the list.
    #   - Build `args._cell_list_keys` (the canonical key order used by
    #     every cell — must be uniform across the list) and
    #     `args._cell_list_set` (frozen tuples for O(1) membership check
    #     inside the inner loop).
    #
    # In the inner loop, we build the current iteration's tuple and skip
    # cells not in the set.  Dict-matching is robust to bench gaining new
    # knobs (old cell-list files keep working — newly-added knobs simply
    # aren't matched on, so they retain CLI defaults).
    # Cell-list state may already have been populated by main() (so that
    # the args.*_list derivations downstream see the override).  Default to
    # empty if not.
    _phase(f"done loading _done_keys ({len(args._done_keys)} entries)")

    if not hasattr(args, "_cell_list_keys"):
        args._cell_list_keys: tuple = ()
        args._cell_list_set: set = set()
        if getattr(args, "cell_list", None):
            _load_cell_list_into_args(args)
    _phase(f"done loading cell-list ({len(args._cell_list_set)} cells)")

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
    if args.mix_csv is not None and len(mtp_lengths) != 1:
        sys.exit("--mix-csv requires exactly one --mtp-lengths value")

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
    if args.json_output and len(act_dtypes) != 1:
        sys.exit("--json-output requires exactly one --act-dtypes value")

    # Resolve baseline function.
    if args.baseline == "flashinfer_pr3324":
        _prepare_flashinfer_jit_workspace()
        from flashinfer_checkpointing_ssu_pr3324 import checkpointing_ssu as baseline_fn
    else:
        baseline_fn = None

    # --with-conv1d uses its own realistic L2 flush (cold cache flush then
    # hot in_proj write).  Override the generic l2_flush to avoid double-flushing.
    if args.with_conv1d:
        args.l2_flush = False
        _init_l2_flush()  # still needed for the realistic reset's flush step
    elif args.l2_flush:
        _init_l2_flush()

    _phase("about to enter compile-warmup")
    if args.compile_threads > 0:
        _compile_warmup_phase(
            args, batch_sizes, mtp_lengths, state_dtypes, act_dtypes,
            baseline_fn, max_workers=args.compile_threads,
        )
    _phase("returned from compile-warmup")

    # Pre-warm the per-(state_dtype, act_dtype, mtp_len, ...) tensor cache at
    # the largest requested batch size.  Without this, the timing loop would
    # progressively grow the cache as it encounters larger batches (e.g.,
    # iterate 1 -> 16 -> 64 -> 128 -> 512 = 5 separate growth allocations,
    # each freeing the previous buffers).  Pre-warming at max-batch up front
    # makes every subsequent timing cell a view-slice (zero alloc cost).
    _max_batch = max(batch_sizes)
    for state_dtype in state_dtypes:
        for act_dtype in act_dtypes:
            for mtp_len in mtp_lengths:
                _build_tensors(
                    _max_batch, mtp_len, state_dtype, act_dtype,
                    args.tp_nheads, args.head_dim, args.d_state, args.tp_ngroups,
                    max_window=getattr(args, "max_window", None) or None,
                )
    _phase("done tensor prewarm — entering timing")

    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()

    # Print header
    mix_enabled = args.mix_csv is not None or getattr(args, "pmix", False)
    headline_name = "score_us" if mix_enabled else "median_us"
    if mix_enabled:
        print(
            "kmix bucket score: mix rows report bucket-weighted score_us; "
            "non-mix rows report median_us."
        )
    if baseline_fn is not None:
        print(
            f"| {'kernel':>11} | {'batch':>5} | {'mtp_len':>7} | {'prev_k':>6} | "
            f"{'state_dtype':>11} | {'act_dtype':>9} | "
            f"{headline_name:>9} | {'p95_us':>7} | {'p99_us':>7} |"
        )
        print(
            f"|{'-' * 13}|{'-' * 7}|{'-' * 9}|{'-' * 8}|"
            f"{'-' * 13}|{'-' * 11}|{'-' * 11}|{'-' * 9}|{'-' * 9}|"
        )
    else:
        print(
            f"| {'batch':>5} | {'mtp_len':>7} | {'prev_k':>6} | "
            f"{'state_dtype':>11} | {'act_dtype':>9} | "
            f"{headline_name:>9} | {'p95_us':>7} | {'p99_us':>7} |"
        )
        print(
            f"|{'-' * 7}|{'-' * 9}|{'-' * 8}|{'-' * 13}|{'-' * 11}|{'-' * 11}|{'-' * 9}|{'-' * 9}|"
        )

    sr_modes_list = getattr(args, "sr_modes_list", ["RN"])
    rect_list = getattr(args, "rectangle_for_nowrite_list", [False])
    modes_list = getattr(args, "modes_list", ["persistent_dynamic"])
    hsort_list = getattr(args, "hardcode_sort_list", [False])

    # Pre-load AL distribution for mix mode.
    mix_al = None
    mix_label = ""
    if getattr(args, "pmix", False):
        mix_label = DEFAULT_PMIX_LABEL
        mix_al = _load_builtin_pmix_distribution(T=max(mtp_lengths))
    elif args.mix_csv is not None:
        mix_csv = Path(args.mix_csv)
        mix_label = mix_csv.stem
        # T (= mtp_len) varies per cell; load once with the LARGEST mtp so
        # we have enough columns; the loader normalizes the dist anyway.
        mix_al = _load_al_distribution(
            mix_csv, T=max(mtp_lengths), column=args.mix_csv_column
        )

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
            mix_samples_sorted_cpu = None  # per-iter prev_tokens, write-first
            mix_write_frac = None
            if mix_al is not None:
                _max_window = getattr(args, "max_window", 0) or mtp_len
                mix_pi = _markov_stationary(mix_al, mtp_len, _max_window)
                mix_write_frac = float(
                    mix_pi[_max_window - mtp_len + 1:].sum()
                )
                _max_iters = max(args.iters, getattr(args, "mix_iters", None) or args.iters)
                mix_samples_cpu = _sample_steady_state_pnat(
                    mix_al, T=mtp_len, window=_max_window, batch=batch,
                    K=args.warmup + _max_iters, seed=args.mix_seed,
                )
                if any(hsort_list):
                    # write-first stable argsort: kind='stable' preserves
                    # original-slot order within each mode group.
                    write_mask = (
                        mix_samples_cpu + mtp_len > _max_window
                    ).astype(np.int8)  # 1 = write, 0 = nowrite
                    perm_idx = np.argsort(
                        -write_mask, kind="stable", axis=-1
                    ).astype(np.int32)
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
                            for rect in rect_list:
                                can_sort = mix_samples_cpu is not None
                                cell_list_active = bool(
                                    getattr(args, "_cell_list_keys", ())
                                )
                                effective_hsort_list = (
                                    hsort_list if (can_sort or cell_list_active)
                                    else [False]
                                )
                                for hardcode_sort in effective_hsort_list:
                                    _bench_config(
                                        args, batch, mtp_len, prev_ks,
                                        state_dtype, act_dtype, baseline_fn,
                                        sr_mode=sr_mode,
                                        rectangle_for_nowrite=rect,
                                        mode=mode,
                                        mix_samples_cpu=mix_samples_cpu,
                                        mix_label=mix_label,
                                        hardcode_sort=hardcode_sort,
                                        mix_samples_sorted_cpu=mix_samples_sorted_cpu,
                                        mix_write_frac=mix_write_frac,
                                    )

    _drain_pending_results(args, force=True)

    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()

    # JSONL is the canonical artifact (written incrementally per cell with
    # host stamps).  No clean-exit `.json` write — use `jsonl_to_json.py` to
    # materialize a snapshot when an analyzer wants one.
    if args.json_output and args._jsonl_path is not None:
        print(f"\nJSONL results: {args._jsonl_path} "
              f"(meta sidecar: {args.json_output}.meta.json)")

    # Write the skipped-cells sidecar.  Caller can convert this list to a
    # --cell-list JSON (one dict per skipped cell) to drive a retry pass in
    # a fresh process.
    skipped_path = getattr(args, "skipped_output", None)
    if skipped_path is None and args.json_output:
        # Derive default: foo.json -> foo.skipped.json
        skipped_path = args.json_output.rsplit(".", 1)[0] + ".skipped.json"
    if skipped_path is not None and args._skipped_cells:
        payload = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "cmd": " ".join(sys.argv),
                "skipped_count": len(args._skipped_cells),
            },
            "skipped": args._skipped_cells,
        }
        tmp = skipped_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, skipped_path)
        print(f"Skipped {len(args._skipped_cells)} cells (CUPTI mismatch); "
              f"tags written to: {skipped_path}", file=sys.stderr)
    elif args._skipped_cells:
        # No output path but there are skipped cells — emit a stderr summary.
        print(f"Skipped {len(args._skipped_cells)} cells (CUPTI mismatch); "
              f"first 5: {args._skipped_cells[:5]}", file=sys.stderr)


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
        default=str(DEFAULT_PMIX_T),
        help="Comma-separated per-request sequence lengths (num_draft_tokens + 1 target)",
    )
    parser.add_argument(
        "--state-dtypes",
        default="fp32",
        help="Comma-separated state dtypes: fp16,bf16,fp32,int8,int16,fp8.  "
        "FlashInfer PR3324 baseline supports fp32/fp16/int8/fp8 only.",
    )
    parser.add_argument(
        "--act-dtypes",
        default="bf16",
        help="Comma-separated activation dtypes for x/B/C/dt: fp32,bf16",
    )
    parser.add_argument("--warmup", type=int, default=4,
                        help="Number of warmup iterations.  Default aligns with "
                        "the graph group-iters (default 4 for mix scenarios) so "
                        "warmup + iters / mix-iters lands on a clean multiple "
                        "without per-args rounding overhead.  Earlier default of "
                        "20 was overkill for steady-state warming.")
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
        "--mp-start-method",
        choices=("spawn", "forkserver"),
        default="spawn",
        help="multiprocessing start method for compile-warmup workers AND "
        "the CUPTI parser child process.  'spawn' (default) is robust but "
        "each child re-imports the bench module (~15s torch+triton import "
        "cost).  'forkserver' starts a server once, preloads the bench "
        "module ONCE, then forks children cheaply (~1s each).  When 4 "
        "benches run concurrently with --compile-threads 26 each, spawn "
        "still incurs 4*26=104 imports per round; forkserver cuts this to "
        "4 (one per server).",
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
        "--cuda-graph-group-iters",
        type=int,
        default=None,
        help="Capture this many logical benchmark iterations per graph "
        "replay when warmup + iters is divisible by this value. Default "
        f"auto-selects {_DEFAULT_CUDA_GRAPH_GROUP_ITERS_PURE} for pure "
        f"cells and {_DEFAULT_CUDA_GRAPH_GROUP_ITERS_MIX} for mix cells. "
        "Mix cells use a per-replay device window so they can group "
        "iterations too.",
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
        "--cupti-flush-period-ms",
        type=int,
        default=0,
        help="If >0, ask CUPTI to periodically flush activity buffers during "
        "the timed CUDA-graph region. This can overlap raw-buffer parsing with "
        "long timed cells; 0 leaves flushing explicit at the end of each cell.",
    )
    parser.add_argument(
        "--cupti-defer-depth",
        type=int,
        default=4,
        help="Maximum number of CUDA-graph CUPTI timing results that may be "
        "left for the parser process while the main process starts later cells. "
        "1 preserves synchronous per-cell parsing and inline retry behavior.",
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
        help="When --json-output is set, also include per_kernel "
        "(per-iter relative start/end timestamps for each kernel). Compact "
        "JSON always includes iters_us, and mix rows include "
        "n_writes_per_iter. Default off keeps records compact.",
    )
    parser.add_argument(
        "--host-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Attach benchmark host-side phase timings to JSON/JSONL results. "
        "Useful for diagnosing benchmark overhead, but it adds roughly 1 KB "
        "per compact JSONL row and several perf_counter calls per cell.",
    )
    parser.add_argument(
        "--cupti-retry",
        type=int,
        default=1,
        help="On CUPTI capture mismatch (kernel record count != expected), "
        "retry the cell this many times in-process before giving up.  CUPTI "
        "gets racy after thousands of cells in one process (PDL + small "
        "kernels occasionally lose records); a single retry usually catches "
        "transient cases.  Set 0 to disable and skip on first mismatch.",
    )
    parser.add_argument(
        "--skipped-output",
        default=None,
        help="Path to write the list of cells that failed CUPTI capture even "
        "after --cupti-retry retries (JSON list of sweep_tag strings).  "
        "Default: derived from --json-output by replacing .json with "
        ".skipped.json.",
    )
    parser.add_argument(
        "--cell-list",
        default=None,
        help="Path to a JSON list of cell dicts (one per cell to time).  "
        "Each dict has canonical knob keys → values: Mw, Mnw, Ww, Wnw, Sw, "
        "Snw, pW, pS, H, R, CT, CPSw, CPSnw, LSw, LSnw, FL, WS, TMARL, "
        "TMAWL, TMANL, TMAWS, SR, RECT, MODE, HSORT (tied forms M / W / S / "
        "CPS / LS are also accepted and auto-expanded). "
        "When set, bench's CLI knob ranges are auto-overridden to the "
        "per-knob union across all cells, and the inner-loop filter skips "
        "any iteration whose knob-value tuple isn't in the list.  All cells "
        "must share the same key set (uniform schema).",
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
        const="flashinfer_pr3324",
        choices=["flashinfer_pr3324"],
        help="Baseline to benchmark alongside the replay kernel. "
        "'flashinfer_pr3324': FlashInfer PR 3324 checkpointing_ssu, "
        "benchmarked per prev_k for fp32/fp16/int8/fp8 state. "
        "Pass --baseline alone for flashinfer_pr3324. Default: no baseline.",
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
        "--block-size-m-write", type=str, default=None,
        help="Sweep BLOCK_SIZE_M for the WRITE main only (overrides --block-size-m "
        "for the write half).  Tied to --block-size-m if unset.",
    )
    parser.add_argument(
        "--block-size-m-nowrite", type=str, default=None,
        help="Sweep BLOCK_SIZE_M for the NOWRITE main only.  Tied to --block-size-m if unset.",
    )
    parser.add_argument(
        "--num-warps-write", type=str, default=None,
        help="Sweep num_warps for the WRITE main only.  Tied to --num-warps if unset.",
    )
    parser.add_argument(
        "--num-warps-nowrite", type=str, default=None,
        help="Sweep num_warps for the NOWRITE main only.  Tied to --num-warps if unset.",
    )
    parser.add_argument(
        "--num-stages-write", type=str, default=None,
        help="Sweep num_stages for the WRITE main only.  Tied to --num-stages if unset.",
    )
    parser.add_argument(
        "--num-stages-nowrite", type=str, default=None,
        help="Sweep num_stages for the NOWRITE main only.  Tied to --num-stages if unset.",
    )
    parser.add_argument(
        "--cta-per-sm-write", type=str, default=None,
        help="Sweep cta_per_sm for the WRITE persistent_main only.  Tied to --cta-per-sm if unset.",
    )
    parser.add_argument(
        "--cta-per-sm-nowrite", type=str, default=None,
        help="Sweep cta_per_sm for the NOWRITE persistent_main only.  Tied to --cta-per-sm if unset.",
    )
    parser.add_argument(
        "--num-loop-stages-write", type=str, default=None,
        help="Sweep num_loop_stages for the WRITE persistent_main only.  Tied to --num-loop-stages if unset.",
    )
    parser.add_argument(
        "--num-loop-stages-nowrite", type=str, default=None,
        help="Sweep num_loop_stages for the NOWRITE persistent_main only.  Tied to --num-loop-stages if unset.",
    )
    parser.add_argument(
        "--skip-diagonal", action=argparse.BooleanOptionalAction, default=False,
        help="When sweeping any per-main *_write / *_nowrite knobs, skip cells "
        "where ALL splittable knobs satisfy write_value == nowrite_value (i.e. "
        "the 'diagonal' that's already covered by a prior shared-knob sweep). "
        "Useful for incremental sweeps that extend earlier results without redoing "
        "the tied-knob cells.",
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
        default=None,
        help="Comma-separated 0/1 values: 0 = replay-style nowrite kernel, "
        "1 = dedicated rectangle nowrite kernel.  Sweep both with '0,1' to "
        "compare in one invocation.  Silently no-op for write cells (the "
        "write path always uses replay-style).  When unset (default), the "
        "wrapper resolves from the _DEFAULT_TUNING lookup per (batch, dtype, "
        "sr) cell.",
    )
    parser.add_argument(
        "--use-tma-rect-load",
        type=str,
        default=None,
        help="Comma-separated 0/1 sweep.  Use TMA (host-built tensor "
        "descriptor) for state load in the rectangle nowrite path.  "
        "Cells where the rect path isn't reachable (rectangle_for_nowrite=False) "
        "skip the value=1 case as a dupe.",
    )
    parser.add_argument(
        "--use-tma-replay-write-load",
        type=str,
        default=None,
        help="Comma-separated 0/1 sweep.  TMA state LOAD in replay main "
        "for the checkpoint/write half.  Independent from nowrite-load and "
        "rect TMA — see CHECKPOINTING_DESIGN.md item #17 for measured perf.",
    )
    parser.add_argument(
        "--use-tma-replay-nowrite-load",
        type=str,
        default=None,
        help="Comma-separated 0/1 sweep.  TMA state LOAD in replay main "
        "for the replay/nowrite half.  Design doc reports the largest win "
        "on this path (int8 b>=64: -8 to -12%%).",
    )
    parser.add_argument(
        "--use-tma-replay-write-store",
        type=str,
        default=None,
        help="Comma-separated 0/1 sweep.  TMA state STORE in replay main "
        "for the checkpoint/write half.  Independent from all load TMA flags.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=None,
        help="Comma-separated dispatch modes to sweep, any of "
        "{persistent_dynamic, persistent_main}.  "
        "persistent_dynamic = single persistent-CTA kernel that dispatches "
        "per-slot at runtime based on PNAT.  "
        "persistent_main = persistent-CTA kernel with two halves (write + "
        "nowrite), using caller-provided n_writes and write-first "
        "replay_work_items. "
        "When unset (default), the wrapper resolves from the _DEFAULT_TUNING "
        "lookup per (batch, dtype, sr) cell.",
    )
    parser.add_argument(
        "--mix-csv",
        type=str,
        default=None,
        help="Path to AL histogram CSV (cols: AL, count).  When set, an "
        "additional 'mix' cell is emitted per (batch, mtp, dtype, sr, "
        "mode, RECT, M, W, ...) combo where prev_tokens varies per iter, "
        "drawn from the steady-state PNAT distribution induced by the "
        "AL histogram.  Both persistent modes support mix scenarios.  "
        "Each iteration of the captured CUDA graph has a different "
        "pre-baked prev_tokens vector; warmup iters use distinct samples "
        "from the timed iters so nsys-included warmup leaks don't bias. "
        "Mutually exclusive with --pmix.",
    )
    parser.add_argument(
        "--pmix",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=f"Use the built-in production-like T{DEFAULT_PMIX_T} accepted-length "
        f"histogram from the replay-count column. Requires --mtp-lengths {DEFAULT_PMIX_T} "
        "and is mutually exclusive with --mix-csv.",
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
        "--hardcode-sort",
        type=str,
        default="0",
        help="Comma-separated 0/1.  When 1, the per-iter prev_tokens "
        "samples are pre-sorted write-first OFFLINE (CPU-side) before "
        "the timed region — kernel runs unchanged (USE_PERM=False) but "
        "the EO gate sees sorted PNAT so early-outs cluster naturally. "
        "Output is scrambled (we don't permute x/B/C/dt) but timing is "
        "meaningful.",
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
        "--mix-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When --pmix or --mix-csv is set, emit only mix scenarios and skip the "
        "pure prev_k sibling scenarios. Default: false.",
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
        "--full-import",
        action="store_true",
        help="Use standard tensorrt_llm import path instead of fast direct "
        "module loading. Slower (~40s startup) but guaranteed correct "
        "if the fast path breaks due to package changes.",
    )
    args = parser.parse_args()
    if args.pmix and args.mix_csv is not None:
        parser.error(
            "--pmix and --mix-csv are mutually exclusive; use --pmix for the "
            f"built-in T{DEFAULT_PMIX_T} distribution or --mix-csv for a custom histogram."
        )
    if args.pmix:
        mtp_lengths_for_pmix = [
            int(x) for x in args.mtp_lengths.split(",") if x.strip()
        ]
        if any(t != DEFAULT_PMIX_T for t in mtp_lengths_for_pmix):
            parser.error(
                f"--pmix uses the built-in T{DEFAULT_PMIX_T} histogram; "
                f"use --mtp-lengths {DEFAULT_PMIX_T} or pass --mix-csv for another T."
            )
    if args.mix_only and args.mix_csv is None and not args.pmix:
        parser.error("--mix-only requires --pmix or --mix-csv")

    # Round iter counts up so warmup + iters (and warmup + mix_iters) are clean
    # multiples of the graph group-iters used downstream.  Default mix group is
    # 4, default pure group is 2.  An explicit --cuda-graph-group-iters can
    # request a larger group.  We round to the max of the two so all scenarios
    # in a single run (pure + mix) share a clean total_iters.  The cost is at
    # most (group-1) extra iters per scenario — negligible — and the win is
    # that graph_group_iters never falls back to 1 (which caused ~5x slowdown
    # in observed benchmark walls).
    _group_for_rounding = max(
        _DEFAULT_CUDA_GRAPH_GROUP_ITERS_MIX,
        _DEFAULT_CUDA_GRAPH_GROUP_ITERS_PURE,
        getattr(args, "cuda_graph_group_iters", None) or 0,
    )
    def _round_iters_to_group(name, val):
        total = args.warmup + val
        if total % _group_for_rounding == 0:
            return val
        new_total = ((total + _group_for_rounding - 1) // _group_for_rounding) * _group_for_rounding
        new_val = new_total - args.warmup
        print(f"[bench] rounding --{name} {val} → {new_val} so warmup+{name} "
              f"({new_total}) is a multiple of graph group_iters={_group_for_rounding}",
              file=sys.stderr)
        return new_val
    args.iters = _round_iters_to_group("iters", args.iters)
    if getattr(args, "mix_iters", None):
        args.mix_iters = _round_iters_to_group("mix-iters", args.mix_iters)

    # Cell-list (if any) must be applied BEFORE the post-argparse string→list
    # derivations below — those build args.*_list from args.* strings, so a
    # cell-list override of e.g. args.modes='persistent_main' needs to land
    # before args.modes_list is computed.  The function populates args._cell_list_keys
    # and args._cell_list_set, plus overrides args.* knob strings to the
    # per-knob union of values across the listed cells.
    if getattr(args, "cell_list", None):
        _load_cell_list_into_args(args)

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

    # rectangle_for_nowrite=None means "let the wrapper resolve from
    # _DEFAULT_TUNING".  Empty/unset argparse default produces [None] in the
    # sweep list; the kernel call passes None and the wrapper picks per-cell.
    if args.rectangle_for_nowrite is None:
        rect_list = [None]
    else:
        rect_modes = [v.strip() for v in args.rectangle_for_nowrite.split(",") if v.strip()]
        rect_list = []
        for v in rect_modes:
            if v not in ("0", "1"):
                parser.error(f"--rectangle-for-nowrite value must be 0 or 1, got {v!r}")
            rect_list.append(v == "1")
        if not rect_list:
            rect_list = [None]
    args.rectangle_for_nowrite_list = rect_list

    hsort_modes = [v.strip() for v in (args.hardcode_sort or "0").split(",") if v.strip()]
    hsort_list = []
    for v in hsort_modes:
        if v not in ("0", "1"):
            parser.error(f"--hardcode-sort value must be 0 or 1, got {v!r}")
        hsort_list.append(v == "1")
    args.hardcode_sort_list = hsort_list

    # mode=None means "let the wrapper resolve from _DEFAULT_TUNING".  Same
    # convention as --rectangle-for-nowrite.
    if args.modes is None:
        args.modes_list = [None]
    else:
        modes_raw = [v.strip() for v in args.modes.split(",") if v.strip()]
        valid_modes = {
            "persistent_main", "persistent_dynamic",
        }
        for m in modes_raw:
            if m not in valid_modes:
                parser.error(
                    f"--modes value must be one of {sorted(valid_modes)}, got {m!r}"
                )
        args.modes_list = modes_raw if modes_raw else [None]
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

    # Configure multiprocessing start method early — must be before any
    # mp.get_context() that uses the chosen method.  For forkserver, also
    # add this file's dir to sys.path so the forkserver can import this
    # module by basename for preload (otherwise it tries to import
    # __main__, which is a different beast across processes).
    if _args.mp_start_method == "forkserver":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        mp.set_start_method("forkserver", force=True)
        try:
            mp.set_forkserver_preload([
                "benchmark_replay_selective_state_update",
            ])
        except Exception as _e:
            print(f"[warn] set_forkserver_preload failed: {_e!r}; "
                  f"forks will still work but pay full import cost",
                  file=sys.stderr)
    _MP_START_METHOD = _args.mp_start_method

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
