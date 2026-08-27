# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Direct split-K correctness/timing check for the Rubin (SM107) BF16 dense GEMM.

This exercises exactly the path used by the ``trtllm::cute_dsl_bf16_gemm_rubin``
custom op for split-K tactics:

  1. ``RubinBf16PersistentDenseGemmKernel(split_k_slices=S)`` expands the
     scheduler L dimension and computes one K slice per CTA.
  2. Every CTA uses TMA reduce-add to accumulate directly into a pre-zeroed
     BF16/FP32 output; there is no FP32 workspace or reduction kernel.

Reference is ``torch.matmul(A, B^T)``.  ``--split_k_slices 1`` runs the original
single-pass kernel (no workspace / reduction) for a sanity baseline.

Example::

    python run_dense_bf16_split_k_gemm_persistent.py \
        --mnkl 8192,256,7168,1 --split_k_slices 4 \
        --warmup_iterations 3 --iterations 20 --print_duration
"""

import argparse
import importlib.util
import sys
import types
from pathlib import Path
from typing import Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack, make_ptr


def _load_rubin_bf16_kernel():
    """Import the Rubin BF16 kernel without triggering heavy package init.

    The ``tensorrt_llm`` package ``__init__`` needs transformers etc.

    The kernel modules use relative imports within ``cute_dsl_kernels``; the
    rubin module's only absolute import is
    ``tensorrt_llm._torch.cute_dsl_kernels.blackwell.dense_gemm_persistent``.
    We register stub namespace packages with correct ``__path__`` so that
    absolute import resolves to the source files directly.
    """
    try:
        from tensorrt_llm._torch.cute_dsl_kernels.rubin.dense_bf16_gemm_persistent import (
            PersistentDenseGemmKernel as K,
        )

        return K
    except (ModuleNotFoundError, ImportError):
        pass

    repo = Path(__file__).parents[3]
    chain = {
        "tensorrt_llm": repo / "tensorrt_llm",
        "tensorrt_llm._torch": repo / "tensorrt_llm/_torch",
        "tensorrt_llm._torch.cute_dsl_kernels": repo / "tensorrt_llm/_torch/cute_dsl_kernels",
        "tensorrt_llm._torch.cute_dsl_kernels.blackwell": repo
        / "tensorrt_llm/_torch/cute_dsl_kernels/blackwell",
        "tensorrt_llm._torch.cute_dsl_kernels.rubin": repo
        / "tensorrt_llm/_torch/cute_dsl_kernels/rubin",
    }
    for name, path in chain.items():
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [str(path)]
            sys.modules[name] = mod

    def _load(modname, file):
        spec = importlib.util.spec_from_file_location(modname, file)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
        return mod

    bw = "tensorrt_llm._torch.cute_dsl_kernels.blackwell"
    rb = "tensorrt_llm._torch.cute_dsl_kernels.rubin"
    bw_dir = chain[bw]
    rb_dir = chain[rb]
    _load(f"{bw}.utils", bw_dir / "utils.py")
    _load(f"{bw}.custom_pipeline", bw_dir / "custom_pipeline.py")
    _load(f"{bw}.dense_gemm_persistent", bw_dir / "dense_gemm_persistent.py")
    rubin_mod = _load(f"{rb}.dense_bf16_gemm_persistent", rb_dir / "dense_bf16_gemm_persistent.py")
    return rubin_mod.PersistentDenseGemmKernel


RubinBf16PersistentDenseGemmKernel = _load_rubin_bf16_kernel()

_CUTLASS_DTYPE_MAP = {
    "bf16": cutlass.BFloat16,
    "fp32": cutlass.Float32,
}
_TORCH_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def run(
    mnkl: Tuple[int, int, int, int],
    c_dtype_str: str,
    use_2cta_instrs: bool,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    split_k_slices: int,
    tolerance: float = 1e-02,
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,
    noncontiguous_output: bool = False,
    use_cuda_graph: bool = False,
):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    m, n, k, batch = mnkl
    assert batch == 1, "This split-K script only supports linear GEMM (batch=1)."
    torch_c_dtype = _TORCH_DTYPE_MAP[c_dtype_str]

    print("Running Rubin (SM107) BF16 split-K dense GEMM:")
    print(f"  mnkl: {mnkl}, c_dtype: {c_dtype_str}")
    print(f"  mma_tiler_mn: {mma_tiler_mn}, cluster_shape_mn: {cluster_shape_mn}")
    print(f"  use_2cta_instrs: {use_2cta_instrs}, split_k_slices: {split_k_slices}")

    torch.manual_seed(1111)
    device = torch.device("cuda")

    # input: [M, K], weight: [N, K] (both K-major / contiguous), output: [M, N]
    a = torch.randn(m, k, dtype=torch.bfloat16, device=device)
    b = torch.randn(n, k, dtype=torch.bfloat16, device=device)
    if noncontiguous_output:
        # Allocate an oversized buffer and slice so the output is non-contiguous.
        c_full = torch.empty(m, n + 8, dtype=torch_c_dtype, device=device)
        c = c_full[:, :n]
    else:
        c = torch.empty(m, n, dtype=torch_c_dtype, device=device)

    a_batched = a.unsqueeze(0)  # [1, M, K]
    b_batched = b.unsqueeze(0)  # [1, N, K]

    gemm = RubinBf16PersistentDenseGemmKernel(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=use_2cta_instrs,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        split_k_slices=split_k_slices,
    )

    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )
    stream = cuda_stream()

    a_ptr = make_ptr(
        cutlass.BFloat16, a_batched.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    b_ptr = make_ptr(
        cutlass.BFloat16, b_batched.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )

    c_tmp = c.unsqueeze(-1)  # [M, N, 1]
    c_cute = from_dlpack(c_tmp, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    compiled_gemm = cute.compile(
        gemm.wrapper,
        m,
        n,
        k,
        batch,
        a_ptr,
        b_ptr,
        c_cute,
        max_active_clusters,
        stream,
        options="--opt-level 2",
    )

    def run_kernel():
        if split_k_slices > 1:
            c.zero_()
        compiled_gemm(m, n, k, batch, a_ptr, b_ptr, c_cute, cuda_stream())

    timed_kernel = run_kernel
    graph = None
    if use_cuda_graph:
        run_kernel()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_kernel()
        timed_kernel = graph.replay

    timed_kernel()
    torch.cuda.synchronize()

    if not skip_ref_check:
        ref = torch.matmul(a.float(), b.float().t())  # [M, N] fp32
        got = c.float()
        max_err = (got - ref).abs().max().item()
        mean_err = (got - ref).abs().mean().item()
        rel_tol = tolerance
        atol = max(tolerance, 1e-2 * ref.abs().max().item())
        ok = torch.allclose(got, ref, atol=atol, rtol=rel_tol)
        print(f"  ref check: max_err={max_err:.6f}, mean_err={mean_err:.6f}, pass={ok}")
        if not ok:
            raise AssertionError(f"split-K reference check failed: max_err={max_err}")

    exec_time_us = None
    if iterations > 0:
        for _ in range(warmup_iterations):
            timed_kernel()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            timed_kernel()
        end.record()
        torch.cuda.synchronize()
        exec_time_us = start.elapsed_time(end) / iterations * 1e3
    return exec_time_us


def cuda_stream():
    import cuda.bindings.driver as cuda

    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


if __name__ == "__main__":

    def parse_ints(s: str) -> Tuple[int, ...]:
        return tuple(int(x.strip()) for x in s.split(","))

    parser = argparse.ArgumentParser(
        description="Rubin SM107 BF16 split-K dense GEMM correctness/timing."
    )
    parser.add_argument("--mnkl", type=parse_ints, default=(512, 256, 256, 1))
    parser.add_argument("--mma_tiler_mn", type=parse_ints, default=(128, 128))
    parser.add_argument("--cluster_shape_mn", type=parse_ints, default=(1, 1))
    parser.add_argument("--c_dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--use_2cta_instrs", action="store_true")
    parser.add_argument("--split_k_slices", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=1e-02)
    parser.add_argument("--warmup_iterations", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--skip_ref_check", action="store_true")
    parser.add_argument("--noncontiguous_output", action="store_true")
    parser.add_argument("--cuda_graph", action="store_true")
    parser.add_argument("--print_duration", action="store_true")
    args = parser.parse_args()

    if len(args.mnkl) != 4:
        parser.error("--mnkl must contain exactly 4 values")

    exec_time = run(
        args.mnkl,
        args.c_dtype,
        args.use_2cta_instrs,
        args.mma_tiler_mn,
        args.cluster_shape_mn,
        args.split_k_slices,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.noncontiguous_output,
        args.cuda_graph,
    )
    if args.print_duration and exec_time is not None:
        print(f"Execution time: {exec_time:.6f} us")
    print("PASS")
