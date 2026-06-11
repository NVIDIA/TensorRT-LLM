/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/common/config.h"

#include <NvInferRuntime.h>

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::cutlass_kernels
{

// Forward declaration; the typedef below references it by name.
struct MoeLoraDevicePathModule;

// Function-pointer dispatch for the libtorch-dependent GEMM stage of the MoE
// LoRA device path. The implementation lives in th_common (moeOp.cpp) because
// the cudaGraph(SplitK)GroupedGemm wrappers allocate workspace via at::Tensor,
// which cannot be linked from libmoe_gemm_src.a (that archive is also linked
// into the TensorRT plugin, which must not depend on libtorch).
//
// It repacks mod into a MoeLoraGemmGroupArrays, runs the problem builder, and
// dispatches the in/out GEMMs, accumulating into output_base (which the caller
// must initialize). data_type is the scalar dtype (fp16/bf16/fp32).
using MoeLoraDeviceRunFn = void (*)(MoeLoraDevicePathModule const& mod, int64_t num_permuted_tokens,
    int64_t in_hidden_size, int64_t max_lora_rank, int64_t dtype_bytes, int64_t splitk_slices, void const* input_base,
    void* output_base, nvinfer1::DataType data_type, cudaStream_t stream);

// Per-module device-resident scratch for the MoE LoRA capture-safe path.
// Pointers refer to device memory unless noted.
//
// The struct is typed with void* rather than the concrete
// cutlass::gemm::GemmCoord* / int64_t* types so this header can be included
// from moe_kernels.h without dragging in cutlass headers. The concrete types
// are recovered at the call site (matching the contract documented in
// moe_lora_problem_builder.h):
//
//   problem_sizes_*  -> cutlass::gemm::GemmCoord* (device, [P_max])
//   a_ptrs_*/b/d     -> void**                    (device, [P_max])
//   lda/ldb/ldd_*    -> int64_t*                  (device, [P_max])
//   splitk_offsets   -> int64_t*                  (device, [P_max + 1])
//   lowrank_ws_dev   -> void*                     (device, [P_max, max_lora_rank, dtype_bytes])
//   host_max_*       -> cutlass::gemm::GemmCoord* (pinned host, [1])
//
// The split-K in-GEMM's partial-sum scratch is allocated internally by the
// cuda_graph_split_k_grouped_gemm wrapper (sized from the host max-problem
// hint); only the per-problem splitk_offsets are produced here.
//
// out_hidden_size is the trailing dimension of the module's output buffer; it
// is inter_size for fc1/gated and hidden_size for fc2. The output base address
// itself is passed directly to runMoeLoraDeviceModule at the call site.
struct MoeLoraDevicePathModule
{
    // Per-source-token (rank, A_ptr, B_ptr) device mirrors, staged via a
    // pinned-host to device async H2D in FusedMoeRunner::buildMoeLoraParams.
    // These feed launchMoeLoraPointerExpand as ranks_src / ptrs_src.
    int32_t const* ranks_src_dev = nullptr;
    int64_t const* ptrs_src_dev = nullptr;

    // Inner (A) and outer (B) dimensions for this module, fed to the
    // pointer-expand kernel as dim_a / dim_b so it can compute the per-expert
    // offset weight_index * dim * lora_rank. For fc1/gated this is
    // (hidden_size, inter_size); for fc2 it is (inter_size, hidden_size).
    int64_t dim_a = 0;
    int64_t dim_b = 0;

    // Per-permuted-row (rank, A_ptr + offset, B_ptr + offset).
    int32_t* permuted_ranks_dev = nullptr;
    int64_t* permuted_ptrs_dev = nullptr;

    // cuda_graph_(split_k_)grouped_gemm-ready bundle.
    void* problem_sizes_in_dev = nullptr;
    void* problem_sizes_out_dev = nullptr;
    void** a_ptrs_in_dev = nullptr;
    void** b_ptrs_in_dev = nullptr;
    void** d_ptrs_in_dev = nullptr;
    void** b_ptrs_out_dev = nullptr;
    void** d_ptrs_out_dev = nullptr;
    int64_t* lda_in_dev = nullptr;
    int64_t* ldb_in_dev = nullptr;
    int64_t* ldd_in_dev = nullptr;
    int64_t* ldb_out_dev = nullptr;
    int64_t* ldd_out_dev = nullptr;
    int64_t* splitk_offsets_dev = nullptr;

    // Low-rank intermediate workspace shared between the in- and out-GEMM. The
    // split-K partial-sum scratch is owned by the GEMM wrapper, not here.
    void* lowrank_workspace_dev = nullptr;

    // Host (pinned) per-call max problem size hints, required by the
    // cuda_graph_*_grouped_gemm wrappers for kernel selection. The
    // values are upper bounds (max_M, max_N, max_K) safe to fix at
    // warmup time.
    void* host_max_problem_in_pinned = nullptr;
    void* host_max_problem_out_pinned = nullptr;

    // Trailing dimension of the module's output buffer (inter_size for
    // fc1/gated, hidden_size for fc2). The output base address is supplied
    // directly to runMoeLoraDeviceModule at the call site.
    int64_t out_hidden_size = 0;
};

// Top-level device-path bundle attached to LoraParams when the device LoRA
// path is active. enabled == false means the FusedMoeRunner runs the legacy
// host path.
struct MoeLoraDevicePath
{
    bool enabled = false;

    // Scalars common to all three modules. Fixed for the lifetime of the
    // FusedMoeRunner once the scratch is allocated.
    int64_t in_hidden_size = 0;
    int64_t max_lora_rank = 0;
    int64_t dtype_bytes = 0;
    int64_t splitk_slices = 0;

    bool has_gated = false;

    // libtorch-bound GEMM dispatch entry point, populated by moeOp.cpp when the
    // device path is enabled. nullptr means the device path is unavailable from
    // this consumer (for example, the TensorRT plugin).
    MoeLoraDeviceRunFn run = nullptr;

    MoeLoraDevicePathModule fc1;
    MoeLoraDevicePathModule fc2;
    MoeLoraDevicePathModule gated;
};

} // namespace kernels::cutlass_kernels

TRTLLM_NAMESPACE_END
