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

Example usage:
  # Basic sweep
  python benchmark_replay_selective_state_update.py \\
      --batch-sizes 1,2,4 --mtp-lengths 1,4,8 --warmup 5 --iters 20

  # With CUDA graph (default) and Triton baseline:
  python benchmark_replay_selective_state_update.py --baseline \\
      --batch-sizes 1,2,4 --mtp-lengths 5,10,20

  # nsys capture (NVTX ranges visible in timeline)
  nsys profile --capture-range=cudaProfilerApi \\
      python benchmark_replay_selective_state_update.py --profile

  # ncu capture
  ncu --target-processes all \\
      python benchmark_replay_selective_state_update.py --profile \\
          --batch-sizes 1 --mtp-lengths 4 --warmup 5 --iters 5
"""

import argparse
import importlib
import itertools
import os
import statistics
import sys
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
    base_mod = _load("selective_state_update", "selective_state_update.py")
    conv1d_mod = _load("causal_conv1d_triton", "causal_conv1d_triton.py")

    return (
        replay_mod.replay_selective_state_update,
        base_mod.selective_state_update,
        conv1d_mod.causal_conv1d_update,
    )


def _import_mamba_kernels_full():
    """Import via the standard tensorrt_llm package (slow but safe)."""
    from tensorrt_llm._torch.modules.mamba.causal_conv1d_triton import causal_conv1d_update
    from tensorrt_llm._torch.modules.mamba.replay_selective_state_update import (
        replay_selective_state_update,
    )
    from tensorrt_llm._torch.modules.mamba.selective_state_update import selective_state_update

    return replay_selective_state_update, selective_state_update, causal_conv1d_update


# Use fast import by default; --full-import parsed later but we need the
# functions at module level.  Check sys.argv early.
if "--full-import" in sys.argv:
    replay_selective_state_update, selective_state_update, causal_conv1d_update = (
        _import_mamba_kernels_full()
    )
else:
    try:
        replay_selective_state_update, selective_state_update, causal_conv1d_update = (
            _import_mamba_kernels_fast()
        )
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
    state0 = torch.randn(batch, nheads, head_dim, d_state, device=device, dtype=state_dtype)

    # --- Cache tensors for replay kernel ---
    # old_x: single-buffered (cache, T, nheads, dim)
    old_x = torch.randn(batch, mtp_len, nheads, head_dim, device=device, dtype=act_dtype)
    # old_B: double-buffered (cache, 2, T, ngroups, dstate)
    old_B = torch.randn(batch, 2, mtp_len, ngroups, d_state, device=device, dtype=act_dtype)
    # old_dt: double-buffered (cache, 2, nheads, T) fp32 — T contiguous
    old_dt = torch.randn(batch, 2, nheads, mtp_len, device=device, dtype=torch.float32)
    # old_dA_cumsum: double-buffered (cache, 2, nheads, T) fp32 — T contiguous
    old_dA_cumsum = torch.randn(batch, 2, nheads, mtp_len, device=device, dtype=torch.float32)
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

    intermediate_states_buffer = torch.zeros(
        batch, mtp_len, nheads, head_dim, d_state, device=device, dtype=state_dtype
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


# Timing helpers


def _compute_stats(latencies_us: list[float]) -> tuple[float, float, float]:
    """Return (median_us, p95_us, p99_us) from a list of latencies."""
    median_us = statistics.median(latencies_us)
    s = sorted(latencies_us)
    p95_us = s[int(0.95 * len(s))]
    p99_us = s[int(0.99 * len(s))]
    return median_us, p95_us, p99_us


def _time_kernel_cuda_graph(
    args,
    run_fn,
    reset_fn,
    tag: str,
) -> tuple[float, float, float]:
    """
    All-in-one CUDA graph timing.

    Captures a single graph containing warmup iterations followed by timed
    iterations with per-iteration event pairs recorded inside the graph.
    One replay, one sync, then all timings are read.
    """
    warmup = args.warmup
    iters = args.iters

    start_events = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(iters)]

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

    latencies_us = [start_events[i].elapsed_time(end_events[i]) * 1000.0 for i in range(iters)]
    return _compute_stats(latencies_us)


def _time_kernel_eager(
    args,
    run_fn,
    reset_fn,
    tag: str,
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


# Per-config benchmark (consolidated baseline + replay)


def _bench_config(
    args,
    batch: int,
    mtp_len: int,
    prev_ks: list[int],
    state_dtype: torch.dtype,
    act_dtype: torch.dtype,
    baseline_fn,
) -> None:
    """
    Benchmark one (batch, mtp_len, dtype) configuration.

    Runs the baseline kernel (if baseline_fn is not None) followed by the
    replay kernel for each prev_k value.  Tensors are built once and
    shared across all runs in this config.
    """
    state_dtype_name = str(state_dtype).split(".")[-1]
    act_dtype_name = str(act_dtype).split(".")[-1]

    (
        state0,
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
    )

    nheads = args.tp_nheads
    ngroups = args.tp_ngroups
    head_dim = args.head_dim
    d_state = args.d_state
    with_conv1d = getattr(args, "with_conv1d", False)
    use_philox = getattr(args, "philox_rounding", False)

    # Philox rounding: allocate rand_seed tensor
    rand_seed = None
    if use_philox:
        if state_dtype != torch.float16:
            raise ValueError(f"--philox-rounding requires --state-dtypes fp16, got {state_dtype}")
        if args.baseline == "triton":
            raise ValueError(
                "--philox-rounding not supported with --baseline triton "
                "(only flashinfer and replay support it)"
            )
        rand_seed = torch.randint(0, 2**62, (1,), device="cuda", dtype=torch.int64)

    state_work = state0.clone()
    old_x_work = old_x0.clone()
    old_B_work = old_B0.clone()
    old_dt_work = old_dt0.clone()
    old_dA_cumsum_work = old_dA_cumsum0.clone()
    cache_buf_idx_work = cache_buf_idx0.clone()
    xbc_input_work = xbc_input0.clone()
    conv_state_work = conv_state0.clone()

    def _reset():
        state_work.copy_(state0)
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
            philox_kwargs = {"rand_seed": rand_seed, "philox_rounds": 10}

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
        median_us, p95_us, p99_us = _time_kernel(args, _run_baseline, reset_fn, tag)

        _print_row(
            show_kernel_col,
            args.baseline,
            batch,
            mtp_len,
            "N/A",
            state_dtype_name,
            act_dtype_name,
            median_us,
            p95_us,
            p99_us,
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

    # --- Replay kernel, one row per prev_k ---
    for prev_k in prev_ks:
        prev_tokens.fill_(prev_k)
        tag = f"incr_b{batch}_mtp{mtp_len}_k{prev_k}_s{state_dtype_name}_a{act_dtype_name}"

        for (
            block_size_m,
            num_warps,
            num_stages,
            precompute_num_warps,
            precompute_num_stages,
            heads_per_block,
        ) in itertools.product(
            block_size_m_values,
            num_warps_values,
            num_stages_values,
            precompute_num_warps_values,
            precompute_num_stages_values,
            heads_per_block_values,
        ):

            def _run_incr(
                prev_k=prev_k,
                block_size_m=block_size_m,
                num_warps=num_warps,
                num_stages=num_stages,
                precompute_num_warps=precompute_num_warps,
                precompute_num_stages=precompute_num_stages,
                heads_per_block=heads_per_block,
            ):
                if with_conv1d:
                    x_call, B_call, C_call = _conv1d_split(
                        xbc_input_work, conv_state_work, launch_dependent_kernels=args.external_pdl
                    )
                    extra_kwargs = {"launch_with_pdl": args.external_pdl}
                else:
                    x_call, B_call, C_call = x, B, C
                    extra_kwargs = {}
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
                    use_internal_pdl=args.internal_pdl,
                    _block_size_m=block_size_m,
                    _num_warps=num_warps,
                    _num_stages=num_stages,
                    _precompute_num_warps=precompute_num_warps,
                    _precompute_num_stages=precompute_num_stages,
                    _heads_per_block=heads_per_block,
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
            sweep_suffix = (" " + ",".join(parts)) if parts else ""
            sweep_tag = tag + sweep_suffix.replace(" ", "_").replace(",", "_")

            reset_fn = _reset_conv1d_realistic if with_conv1d else _reset
            median_us, p95_us, p99_us = _time_kernel(args, _run_incr, reset_fn, sweep_tag)

            _print_row(
                show_kernel_col,
                "replay",
                batch,
                mtp_len,
                prev_k,
                state_dtype_name,
                act_dtype_name,
                median_us,
                p95_us,
                p99_us,
                sweep_suffix,
            )


def _print_row(
    show_kernel_col,
    kernel_name,
    batch,
    mtp_len,
    prev_k,
    state_dtype_name,
    act_dtype_name,
    median_us,
    p95_us,
    p99_us,
    sweep_suffix="",
):
    kernel_col = f"{kernel_name:>11} | " if show_kernel_col else ""
    print(
        f"| {kernel_col}{batch:>5} | {mtp_len:>7} | {str(prev_k):>6} | "
        f"{state_dtype_name:>11} | {act_dtype_name:>9} | "
        f"{median_us:>9.2f} | {p95_us:>7.2f} | {p99_us:>7.2f} |"
        f"{sweep_suffix}"
    )


# Main benchmark loop


def _run_benchmark(args) -> None:
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

    dtype_map = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}
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

    for batch in batch_sizes:
        for mtp_len in mtp_lengths:
            # Resolve prev_k fractions → clamped integers in [0, mtp_len]
            prev_ks = sorted(
                set(min(mtp_len, max(0, round(f * mtp_len))) for f in args.prev_tokens_fracs)
            )
            for state_dtype in state_dtypes:
                for act_dtype in act_dtypes:
                    _bench_config(
                        args, batch, mtp_len, prev_ks, state_dtype, act_dtype, baseline_fn
                    )

    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()


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
        "--state-dtypes", default="fp32", help="Comma-separated state dtypes: fp16,bf16,fp32"
    )
    parser.add_argument(
        "--act-dtypes",
        default="bf16",
        help="Comma-separated activation dtypes for x/B/C/dt: fp32,bf16",
    )
    parser.add_argument("--warmup", type=int, default=20, help="Number of warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Number of timed iterations")
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
        "--philox-rounding",
        action="store_true",
        help="Enable Philox stochastic rounding for fp16 state "
        "(rand_seed generated per iteration, philox_rounds=10).",
    )
    parser.add_argument(
        "--full-import",
        action="store_true",
        help="Use standard tensorrt_llm import path instead of fast direct "
        "module loading. Slower (~40s startup) but guaranteed correct "
        "if the fast path breaks due to package changes.",
    )
    return parser.parse_args()


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
