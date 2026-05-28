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

// Function-pointer dispatch for the libtorch-dependent GEMM stage of the
// MoE LoRA device path. The implementation lives in `th_common`
// (`moeOp.cpp`) because the underlying `cudaGraph(SplitK)GroupedGemm`
// wrappers allocate workspace via `at::Tensor`, which is *not* linkable
// from `libmoe_gemm_src.a` (that archive ends up inside the TensorRT
// plugin `.so`, which deliberately does not depend on libtorch).
//
// Contract:
//   `mod`                   per-module device-resident scratch produced by
//                           `launchMoeLoraPointerExpand` (6b.B) -- the
//                           function pointer's body repacks it into a
//                           `MoeLoraGemmGroupArrays`, calls the problem
//                           builder, and then dispatches the in/out GEMMs.
//   `num_permuted_tokens`   length of the permuted-row tables in `mod`.
//   `in_hidden_size`        K of the in-GEMM (== input row stride).
//   `max_lora_rank`         workspace ldd (and worst-case rank cap).
//   `dtype_bytes`           sizeof(scalar) for the LoRA tensors.
//   `splitk_slices`         split-K factor for the in-GEMM.
//   `input_base`            base of the input matrix (M rows of
//                           `in_hidden_size`).
//   `output_base`           base of the module's output buffer; the impl
//                           ACCUMULATES into it (callers must zero/init).
//   `data_type`             scalar dtype expected by the GEMM wrappers
//                           (fp16/bf16/fp32 in the MVP).
//   `stream`                CUDA stream to launch onto.
using MoeLoraDeviceRunFn = void (*)(MoeLoraDevicePathModule const& mod, int64_t num_permuted_tokens,
    int64_t in_hidden_size, int64_t max_lora_rank, int64_t dtype_bytes, int64_t splitk_slices, void const* input_base,
    void* output_base, nvinfer1::DataType data_type, cudaStream_t stream);


// Per-module device-resident scratch for the MoE LoRA capture-safe path
// introduced in Phase 6b.C. Pointers refer to device memory unless noted.
//
// The struct is intentionally typed with `void*` rather than the concrete
// `cutlass::gemm::GemmCoord*` / `int64_t*` types so this header can be
// included from `moe_kernels.h` without dragging in cutlass headers. The
// concrete types are recovered at the call site (matching the contract
// already documented in `moe_lora_problem_builder.h`):
//
//   problem_sizes_*  -> cutlass::gemm::GemmCoord* (device, [P_max])
//   a_ptrs_*/b/d     -> void**                    (device, [P_max])
//   lda/ldb/ldd_*    -> int64_t*                  (device, [P_max])
//   splitk_offsets   -> int64_t*                  (device, [P_max + 1])
//   lowrank_ws_dev   -> void*                     (device, [P_max, max_lora_rank, dtype_bytes])
//   splitk_ws_dev    -> void*                     (device, [P_max, max_lora_rank * splitk_slices], fp32)
//   host_max_*       -> cutlass::gemm::GemmCoord* (pinned host, [1])
//
// `output_base_dev` points at the device buffer that consumes this
// module's LoRA delta (e.g. `lora_fc1_result_`, `lora_fc2_result_`,
// `lora_gated_result_`). `out_hidden_size` is the trailing dimension of
// that buffer; it differs per module (fc1/gated -> inter_size,
// fc2 -> hidden_size).
struct MoeLoraDevicePathModule
{
    // 6b.A device mirrors: per-source-token (rank, A_ptr, B_ptr) staged via
    // pinned-host -> device async H2D in `FusedMoeRunner::buildMoeLoraParams`.
    // These feed `launchMoeLoraPointerExpand` as `ranks_src` / `ptrs_src`.
    int32_t const* ranks_src_dev = nullptr;
    int64_t const* ptrs_src_dev = nullptr;

    // Inner ("A") and outer ("B") dimensions for this module, fed to the
    // pointer-expand kernel as `dim_a` / `dim_b` so it can compute the
    // per-expert offset `weight_index * dim * lora_rank` (zeroed when the
    // corresponding shared-outer flag is set). For fc1/gated this is
    // (hidden_size, inter_size); for fc2 it's (inter_size, hidden_size).
    int64_t dim_a = 0;
    int64_t dim_b = 0;

    // 6b.B outputs: per-permuted-row (rank, A_ptr+offset, B_ptr+offset).
    int32_t* permuted_ranks_dev = nullptr;
    int64_t* permuted_ptrs_dev = nullptr;

    // 6b.C.1 outputs: cuda_graph_(split_k_)grouped_gemm-ready bundle.
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

    // Workspaces that participate in the in/out GEMM data flow.
    void* lowrank_workspace_dev = nullptr;
    void* splitk_workspace_dev = nullptr;

    // Host (pinned) per-call max problem size hints, required by the
    // cuda_graph_*_grouped_gemm wrappers for kernel selection. The
    // values are upper bounds (max_M, max_N, max_K) safe to fix at
    // warmup time.
    void* host_max_problem_in_pinned = nullptr;
    void* host_max_problem_out_pinned = nullptr;

    // Module-local data-flow knobs.
    void* output_base_dev = nullptr;
    int64_t out_hidden_size = 0;
};

// Top-level device-path bundle attached to LoraParams when the device
// LoRA path is active. nullptr / `enabled == false` means the
// FusedMoeRunner is running the legacy host path (unchanged behavior).
struct MoeLoraDevicePath
{
    bool enabled = false;

    // Shared scalars across all three modules. Static for the lifetime
    // of the FusedMoeRunner once the scratch is allocated.
    int64_t in_hidden_size = 0;
    int64_t max_lora_rank = 0;
    int64_t dtype_bytes = 0;
    int64_t splitk_slices = 0;

    bool has_gated = false;

    // libtorch-bound GEMM dispatch entry point; populated by `moeOp.cpp`
    // when the device path is enabled. nullptr means "device path
    // unavailable from this consumer" (e.g. the bare plugin).
    MoeLoraDeviceRunFn run = nullptr;

    MoeLoraDevicePathModule fc1;
    MoeLoraDevicePathModule fc2;
    MoeLoraDevicePathModule gated;
};

} // namespace kernels::cutlass_kernels
TRTLLM_NAMESPACE_END
