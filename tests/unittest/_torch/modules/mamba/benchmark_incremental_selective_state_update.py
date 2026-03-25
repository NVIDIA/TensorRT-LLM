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
"""
TODO: Better name for mtp-len, which is actually draft_len + 1
Standalone benchmark for incremental_selective_state_update (Triton kernel).

Suitable for nsight-compute (ncu) and nsight-systems (nsys) capture.

Fixed model config: NVIDIA-Nemotron-3-Super-120B-A12B at TP=8
  nheads=16, head_dim=64, d_state=128, ngroups=1

Baseline kernel (--baseline [triton|flashinfer]):
  Calls selective_state_update with T=mtp_len tokens and disable_state_update=True,
  matching the MTP scoring pass in mamba2_mixer.py exactly.

Example usage:
  # Basic sweep
  python benchmark_incremental_selective_state_update.py \\
      --batch-sizes 1,2,4 --mtp-lengths 1,4,8 --warmup 5 --iters 20

  # With CUDA graph (default) and Triton baseline:
  python benchmark_incremental_selective_state_update.py --baseline \\
      --batch-sizes 1,2,4 --mtp-lengths 5,10,20

  # nsys capture (NVTX ranges visible in timeline)
  nsys profile --capture-range=cudaProfilerApi \\
      python benchmark_incremental_selective_state_update.py --profile

  # ncu capture
  ncu --target-processes all \\
      python benchmark_incremental_selective_state_update.py --profile \\
          --batch-sizes 1 --mtp-lengths 4 --warmup 5 --iters 5
"""

import argparse
import importlib
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

import torch
from einops import repeat


def _import_mamba_kernels():
    """Import Mamba Triton kernels without triggering the heavy tensorrt_llm init.

    The kernel files only depend on torch, triton, einops, and a PAD_SLOT_ID
    constant (-1).  Going through ``tensorrt_llm.__init__`` pulls in every
    model, custom C++ op, modelopt, etc. — adding ~40 s of startup.  We
    short-circuit that by pre-loading the tiny ``__init__`` (which defines
    PAD_SLOT_ID) and the ``softplus`` helper, then importing the two kernel
    modules directly.
    """
    mamba_pkg = "tensorrt_llm._torch.modules.mamba"
    repo_root = Path(__file__).resolve().parents[5]
    mamba_dir = repo_root / "tensorrt_llm" / "_torch" / "modules" / "mamba"

    def _load(mod_name: str, file_name: str):
        fqn = f"{mamba_pkg}.{mod_name}" if mod_name else mamba_pkg
        if fqn in sys.modules:
            return sys.modules[fqn]
        spec = importlib.util.spec_from_file_location(
            fqn, mamba_dir / file_name,
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
    incr_mod = _load("incremental_selective_state_update",
                     "incremental_selective_state_update.py")
    base_mod = _load("selective_state_update", "selective_state_update.py")

    return incr_mod.incremental_selective_state_update, base_mod.selective_state_update


incremental_selective_state_update, selective_state_update = _import_mamba_kernels()

# ---------------------------------------------------------------------------
# Fixed model config (Nemotron-3-Super-120B @ TP=8)
# ---------------------------------------------------------------------------
NHEADS = 16
HEAD_DIM = 64
D_STATE = 128
NGROUPS = 1

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


# ---------------------------------------------------------------------------
# Tensor construction helpers
# ---------------------------------------------------------------------------

def _build_tensors(batch: int, mtp_len: int, state_dtype: torch.dtype,
                   act_dtype: torch.dtype):
    """
    Build all tensors for one benchmark configuration.

    Returns:
      state0                   : (batch, nheads, head_dim, d_state) – initial SSM state
      intermediate_update_inputs: (batch, mtp_len, nheads*head_dim + nheads + ngroups*d_state)
                                   packed [old_x | old_dt_base | old_B] for incremental kernel
      x, dt, B, C              : (batch, mtp_len, ...) – token inputs for both kernels
      A, dt_bias, D            : SSM parameters (float32, tie_hdim strides)
      prev_tokens              : (batch,)
      out_incr                 : pre-allocated output for incremental kernel (batch, mtp_len, nheads, head_dim)
      out_base                 : pre-allocated output for baseline kernel   (batch, mtp_len, nheads, head_dim)
      intermediate_states_buffer: for baseline kernel (batch, mtp_len, nheads, head_dim, d_state)
    """
    device = "cuda"
    nheads, head_dim, d_state, ngroups = NHEADS, HEAD_DIM, D_STATE, NGROUPS

    torch.manual_seed(42)

    # --- SSM parameters (float32, tie_hdim strides) ---
    A_base = -torch.rand(nheads, device=device) - 0.5
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)    # stride(-1)=0, stride(-2)=0

    dt_bias_base = torch.randn(nheads, device=device, dtype=torch.float32)
    dt_bias = repeat(dt_bias_base, "h -> h p", p=head_dim)     # stride(-1)=0

    D_base = torch.randn(nheads, device=device, dtype=torch.float32)
    D = repeat(D_base, "h -> h p", p=head_dim)

    # --- SSM state ---
    state0 = torch.randn(batch, nheads, head_dim, d_state,
                         device=device, dtype=state_dtype)

    # --- Old inputs for packing into intermediate_update_inputs ---
    old_x = torch.randn(batch, mtp_len, nheads, head_dim,
                        device=device, dtype=act_dtype)
    old_dt_base = torch.randn(batch, mtp_len, nheads,
                               device=device, dtype=act_dtype)
    old_B = torch.randn(batch, mtp_len, ngroups, d_state,
                        device=device, dtype=act_dtype)

    # intermediate_update_inputs: packed layout for incremental kernel
    old_x_flat = old_x.reshape(batch, mtp_len, nheads * head_dim)
    old_B_flat = old_B.reshape(batch, mtp_len, ngroups * d_state)
    intermediate_update_inputs = torch.cat(
        [old_x_flat, old_dt_base, old_B_flat], dim=-1
    ).contiguous()

    # --- Token inputs (used by both incremental and baseline kernels) ---
    x = torch.randn(batch, mtp_len, nheads, head_dim, device=device, dtype=act_dtype)
    # TODO: For now, dt has to match D (fp32) for flashifner, so always do that
    dt_base = torch.randn(batch, mtp_len, nheads, device=device, dtype=torch.float32)
    dt = repeat(dt_base, "b t h -> b t h p", p=head_dim)    # tie_hdim
    B = torch.randn(batch, mtp_len, ngroups, d_state, device=device, dtype=act_dtype)
    C = torch.randn(batch, mtp_len, ngroups, d_state, device=device, dtype=act_dtype)

    # prev_tokens placeholder — overwritten per-run
    prev_tokens = torch.zeros(batch, device=device, dtype=torch.int32)

    out_incr = torch.zeros(batch, mtp_len, nheads, head_dim,
                           device=device, dtype=act_dtype)
    out_base = torch.zeros(batch, mtp_len, nheads, head_dim,
                           device=device, dtype=act_dtype)

    intermediate_states_buffer = torch.zeros(
        batch, mtp_len, nheads, head_dim, d_state,
        device=device, dtype=state_dtype)

    return (state0, intermediate_update_inputs,
            x, dt, B, C,
            A, dt_bias, D, prev_tokens,
            out_incr, out_base, intermediate_states_buffer)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _compute_stats(latencies_us: list[float]) -> tuple[float, float, float]:
    """Return (median_us, p95_us, p99_us) from a list of latencies."""
    median_us = statistics.median(latencies_us)
    s = sorted(latencies_us)
    p95_us = s[int(0.95 * len(s))]
    p99_us = s[int(0.99 * len(s))]
    return median_us, p95_us, p99_us


def _time_kernel_cuda_graph(
    args, run_fn, reset_fn, tag: str,
) -> tuple[float, float, float]:
    """
    All-in-one CUDA graph timing.

    Captures a single graph containing warmup iterations followed by timed
    iterations with per-iteration event pairs recorded inside the graph.
    One replay, one sync, then all timings are read.
    """
    warmup = args.warmup
    iters = args.iters

    start_events = [torch.cuda.Event(enable_timing=True, external=True)
                     for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True, external=True)
                   for _ in range(iters)]

    # Eager warmup before graph capture (triggers Triton autotune if active)
    reset_fn()
    run_fn()
    torch.cuda.synchronize()

    reset_fn()
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        # Warmup iterations (unrolled into the graph)
        for _ in range(warmup):
            reset_fn()
            if args.l2_flush:
                _l2_flush.fill_(0.0)
            run_fn()

        # Timed iterations with events inside the graph
        for i in range(iters):
            reset_fn()
            if args.l2_flush:
                _l2_flush.fill_(0.0)
            start_events[i].record()
            run_fn()
            end_events[i].record()

    torch.cuda.synchronize()

    # Single replay
    torch.cuda.nvtx.range_push(tag)
    g.replay()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    latencies_us = [
        start_events[i].elapsed_time(end_events[i]) * 1000.0
        for i in range(iters)
    ]
    return _compute_stats(latencies_us)


def _time_kernel_eager(
    args, run_fn, reset_fn, tag: str,
) -> tuple[float, float, float]:
    """Non-CUDA-graph timing path (for debugging, ncu, etc.)."""
    # Warmup
    for _ in range(args.warmup):
        reset_fn()
        run_fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies_us: list[float] = []
    torch.cuda.nvtx.range_push(tag)
    for _ in range(args.iters):
        reset_fn()
        if args.l2_flush:
            _flush_l2()  # includes synchronize
        start_event.record()
        run_fn()
        end_event.record()
        torch.cuda.synchronize()
        latencies_us.append(start_event.elapsed_time(end_event) * 1000.0)
    torch.cuda.nvtx.range_pop()

    return _compute_stats(latencies_us)


def _time_kernel(args, run_fn, reset_fn, tag: str) -> tuple[float, float, float]:
    """Dispatch to CUDA-graph or eager timing path."""
    if args.cuda_graph:
        return _time_kernel_cuda_graph(args, run_fn, reset_fn, tag)
    return _time_kernel_eager(args, run_fn, reset_fn, tag)


# ---------------------------------------------------------------------------
# Per-config benchmark (consolidated baseline + incremental)
# ---------------------------------------------------------------------------

def _bench_config(args, batch: int, mtp_len: int, prev_ks: list[int],
                  state_dtype: torch.dtype, act_dtype: torch.dtype,
                  baseline_fn) -> None:
    """
    Benchmark one (batch, mtp_len, dtype) configuration.

    Runs the baseline kernel (if baseline_fn is not None) followed by the
    incremental kernel for each prev_k value.  Tensors are built once and
    shared across all runs in this config.
    """
    state_dtype_name = str(state_dtype).split(".")[-1]
    act_dtype_name = str(act_dtype).split(".")[-1]

    (state0, intermediate_update_inputs,
     x, dt, B, C,
     A, dt_bias, D, prev_tokens,
     out_incr, out_base, intermediate_states_buffer,
    ) = _build_tensors(batch, mtp_len, state_dtype, act_dtype)

    state_work = state0.clone()
    interm_work = intermediate_update_inputs.clone()

    def _reset():
        state_work.copy_(state0)
        interm_work.copy_(intermediate_update_inputs)

    show_kernel_col = baseline_fn is not None

    # --- Baseline ---
    if baseline_fn is not None:
        tag = f"base_b{batch}_mtp{mtp_len}_s{state_dtype_name}_a{act_dtype_name}"

        def _run_baseline():
            baseline_fn(
                state_work,
                x=x, dt=dt, A=A, B=B, C=C,
                D=D, dt_bias=dt_bias, dt_softplus=True,
                out=out_base,
                disable_state_update=True,
                intermediate_states_buffer=intermediate_states_buffer,
                cache_steps=mtp_len,
            )

        median_us, p95_us, p99_us = _time_kernel(
            args, _run_baseline, _reset, tag)

        _print_row(show_kernel_col, args.baseline, batch, mtp_len, "N/A",
                   state_dtype_name, act_dtype_name, median_us, p95_us, p99_us)

    # --- Incremental kernel, one row per prev_k ---
    for prev_k in prev_ks:
        prev_tokens.fill_(prev_k)
        tag = (f"incr_b{batch}_mtp{mtp_len}_k{prev_k}"
               f"_s{state_dtype_name}_a{act_dtype_name}")

        bsm_values = ([int(x) for x in args.block_size_m.split(",")]
                       if args.block_size_m else [None])
        nw_values = ([int(x) for x in args.num_warps.split(",")]
                      if args.num_warps else [None])
        for bsm in bsm_values:
            for nw in nw_values:
                def _run_incr(prev_k=prev_k, bsm=bsm, nw=nw):
                    incremental_selective_state_update(
                        state_work, interm_work, prev_tokens,
                        x=x, dt=dt, A=A, B=B, C=C,
                        out=out_incr,
                        D=D, dt_bias=dt_bias, dt_softplus=True,
                        state_batch_indices=None,
                        _block_size_m=bsm,
                        _num_warps=nw,
                        _fast_forward_replay=args.fast_forward_replay,
                        _cb_output=args.cb_output,
                    )

                sweep_tag = (tag + (f"_m{bsm}" if bsm else "")
                             + (f"_w{nw}" if nw else ""))
                median_us, p95_us, p99_us = _time_kernel(
                    args, _run_incr, _reset, sweep_tag)

                sweep_suffix = ""
                if bsm is not None or nw is not None:
                    parts = []
                    if bsm is not None:
                        parts.append(f"M={bsm}")
                    if nw is not None:
                        parts.append(f"W={nw}")
                    sweep_suffix = " " + ",".join(parts)

                _print_row(show_kernel_col, "incremental",
                           batch, mtp_len, prev_k,
                           state_dtype_name, act_dtype_name,
                           median_us, p95_us, p99_us,
                           sweep_suffix)


def _print_row(show_kernel_col, kernel_name, batch, mtp_len, prev_k,
               state_dtype_name, act_dtype_name, median_us, p95_us, p99_us,
               sweep_suffix=""):
    kernel_col = f"{kernel_name:>11} | " if show_kernel_col else ""
    print(f"| {kernel_col}{batch:>5} | {mtp_len:>7} | {str(prev_k):>6} | "
          f"{state_dtype_name:>11} | {act_dtype_name:>9} | "
          f"{median_us:>9.2f} | {p95_us:>7.2f} | {p99_us:>7.2f} |"
          f"{sweep_suffix}")


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def _run_benchmark(args) -> None:
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    mtp_lengths = [int(x) for x in args.mtp_lengths.split(",")]

    dtype_map = {"bf16": torch.bfloat16, "fp32": torch.float32}
    state_dtypes = [dtype_map[s] for s in args.state_dtypes.split(",")]
    act_dtypes = [dtype_map[s] for s in args.act_dtypes.split(",")]

    # Resolve baseline function
    if args.baseline == "flashinfer":
        from flashinfer.mamba import selective_state_update as baseline_fn
    elif args.baseline == "triton":
        baseline_fn = selective_state_update
    else:
        baseline_fn = None

    if args.l2_flush:
        _init_l2_flush()

    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()

    # Print header
    if baseline_fn is not None:
        print(f"| {'kernel':>11} | {'batch':>5} | {'mtp_len':>7} | {'prev_k':>6} | "
              f"{'state_dtype':>11} | {'act_dtype':>9} | "
              f"{'median_us':>9} | {'p95_us':>7} | {'p99_us':>7} |")
        print(f"|{'-'*13}|{'-'*7}|{'-'*9}|{'-'*8}|"
              f"{'-'*13}|{'-'*11}|{'-'*11}|{'-'*9}|{'-'*9}|")
    else:
        print(f"| {'batch':>5} | {'mtp_len':>7} | {'prev_k':>6} | "
              f"{'state_dtype':>11} | {'act_dtype':>9} | "
              f"{'median_us':>9} | {'p95_us':>7} | {'p99_us':>7} |")
        print(f"|{'-'*7}|{'-'*9}|{'-'*8}|"
              f"{'-'*13}|{'-'*11}|{'-'*11}|{'-'*9}|{'-'*9}|")

    for batch in batch_sizes:
        for mtp_len in mtp_lengths:
            # Resolve prev_k fractions → clamped integers in [0, mtp_len]
            prev_ks = sorted(set(
                min(mtp_len, max(0, round(f * mtp_len)))
                for f in args.prev_tokens_fracs
            ))
            for state_dtype in state_dtypes:
                for act_dtype in act_dtypes:
                    _bench_config(args, batch, mtp_len, prev_ks,
                                  state_dtype, act_dtype, baseline_fn)

    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark incremental_selective_state_update Triton kernel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch-sizes", default="1,2,4,8",
                        help="Comma-separated decode batch sizes")
    parser.add_argument("--mtp-lengths", default="1,2,4,8",
                        help="Comma-separated MTP speculation depths")
    parser.add_argument("--state-dtypes", default="fp32",
                        help="Comma-separated state dtypes: bf16,fp32")
    parser.add_argument("--act-dtypes", default="bf16",
                        help="Comma-separated activation dtypes for x/B/C/dt: "
                             "fp32,bf16")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Number of warmup iterations")
    parser.add_argument("--iters", type=int, default=100,
                        help="Number of timed iterations")
    parser.add_argument("--profile", action="store_true",
                        help="Wrap timed region in cudaProfilerStart/Stop "
                             "(for ncu --target-processes all)")
    parser.add_argument("--l2-flush", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="L2 eviction between iterations")
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Capture all warmup + timed iterations in a "
                             "single CUDA graph with per-iteration events "
                             "inside the graph, eliminating all host overhead.")
    parser.add_argument("--prev-tokens-fracs", default="0,0.5,1.0",
                        type=lambda s: [float(x) for x in s.split(",")],
                        help="Fractions of mtp_len to use as prev_num_accepted_tokens "
                             "for the incremental kernel sweep. Values are rounded "
                             "and clamped to [0, mtp_len].")
    parser.add_argument(
        "--baseline", default=None, nargs="?", const="triton",
        choices=[None, "triton", "flashinfer"],
        help="Baseline to benchmark alongside the incremental kernel. "
             "'triton': native Triton selective_state_update. "
             "'flashinfer': flashinfer selective_state_update (same signature). "
             "Pass --baseline alone for 'triton'. Default: no baseline.")
    parser.add_argument(
        "--output", default=None,
        help="Path to save results (file or directory). "
             "If a directory, writes benchmark_incremental_<timestamp>.txt inside it.")
    parser.add_argument(
        "--block-size-m", type=str, default=None,
        help="Override BLOCK_SIZE_M: single value or comma-separated sweep (e.g. '4,8,16,32').")
    parser.add_argument(
        "--num-warps", type=str, default=None,
        help="Override num_warps: single value or comma-separated sweep (e.g. '1,2,4').")
    parser.add_argument(
        "--fast-forward-replay", action=argparse.BooleanOptionalAction,
        default=False,
        help="Use fast-forward (parallel) replay vs sequential.")
    parser.add_argument(
        "--cb-output", action=argparse.BooleanOptionalAction,
        default=False,
        help="Use CB formulation for output phase vs sequential state update.")
    return parser.parse_args()


class _Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
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
        _fname = f"benchmark_incremental_{_ts}.txt"
        if _args.output is None:
            _out_path = os.path.expanduser(f"~/nemo_logs/{_fname}")
        elif os.path.isdir(_args.output) or _args.output.endswith("/"):
            _out_path = os.path.join(_args.output, _fname)
        else:
            _out_path = _args.output

    if _out_path:
        _tee = _Tee(_out_path)
        sys.stdout = _tee
        print(f"# benchmark_incremental_selective_state_update  {datetime.now().isoformat()}")
        print(f"# cmd: {' '.join(sys.argv)}")

    try:
        _run_benchmark(_args)
    finally:
        if _out_path:
            sys.stdout = _tee._stdout
            _tee.close()
            print(f"\nResults saved to: {_out_path}")
