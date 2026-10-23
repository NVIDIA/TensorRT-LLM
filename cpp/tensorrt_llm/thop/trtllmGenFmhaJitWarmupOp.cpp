/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaKernels.h"

#include <torch/library.h>

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// Waits for the background TRTLLM-Gen FMHA JIT warmup sweeps and re-runs each
// finished sweep once on the calling thread, so every kernel the warmup
// derived is compiled before CUDA graph capture and before the engine reports
// ready. Returns the number of sweeps verified here. No-op unless
// TRTLLM_GEN_FMHA_ASYNC_WARMUP=1.
int64_t trtllmGenFmhaJitWarmupDrainAndVerify()
{
    return tensorrt_llm::kernels::TllmGenFmhaKernel::drainAndVerifyAllJITWarmups();
}

// Number of TRTLLM-Gen FMHA kernel-cache misses (NVRTC compiles) the export
// library reported in this process so far. A value that does not change across
// a request proves that the request compiled nothing.
int64_t trtllmGenFmhaJitNumCacheMisses()
{
    return tensorrt_llm::kernels::TllmGenFmhaKernel::numJITCacheMisses();
}

// Number of compile requests whose cache result the export library did not
// report. Non-zero means the miss count above is not a complete measure.
int64_t trtllmGenFmhaJitNumUnknownCacheResults()
{
    return tensorrt_llm::kernels::TllmGenFmhaKernel::numJITUnknownCacheResults();
}

// Whether this process runs the TRTLLM-Gen FMHA JIT warmup on the background
// thread (TRTLLM_GEN_FMHA_ASYNC_WARMUP=1, read once per process).
bool trtllmGenFmhaAsyncJitWarmupEnabled()
{
    return tensorrt_llm::kernels::asyncJITWarmupEnabled();
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("trtllm_gen_fmha_jit_warmup_drain_and_verify() -> int",
        &tensorrt_llm::torch_ext::trtllmGenFmhaJitWarmupDrainAndVerify);
    m.def("trtllm_gen_fmha_jit_num_cache_misses() -> int", &tensorrt_llm::torch_ext::trtllmGenFmhaJitNumCacheMisses);
    m.def("trtllm_gen_fmha_jit_num_unknown_cache_results() -> int",
        &tensorrt_llm::torch_ext::trtllmGenFmhaJitNumUnknownCacheResults);
    m.def("trtllm_gen_fmha_async_jit_warmup_enabled() -> bool",
        &tensorrt_llm::torch_ext::trtllmGenFmhaAsyncJitWarmupEnabled);
}
