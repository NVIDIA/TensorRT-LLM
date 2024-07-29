/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
 *
 * This file is NOT thread safe.
 */
#pragma once
#include <stddef.h>

#ifdef _WIN32

#if COMPILING_DLL
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __declspec(dllimport)
#endif

#else             // _WIN32
#define DLLEXPORT // Nothing.
#endif

#if __cplusplus
extern "C"
{
#endif

    typedef enum
    {
        // sm >= 80
        TLLM_XQA_JIT_HMMA = 0,
        // sm == 90
        TLLM_XQA_JIT_QGMMA = 1
    } tllmXqaJitKernelType;

    typedef struct
    {
        // Compute capability, e.g. 89.
        int sm;

        unsigned int head_size;
        unsigned int num_q_heads;
        unsigned int num_kv_heads;
        unsigned int beam_width;
        unsigned int tokens_per_block;
        bool multi_query_tokens;
        bool paged_kv_cache;

        // Actual type: tensorrt_llm::kernels::Data_type
        int data_type;
        int kv_cache_data_type;

        tllmXqaJitKernelType kernel_type;
    } tllmXqaJitContext;

    // tllmXqaJitProgram is an opaque handle for a program.
    typedef struct _tllmXqaJitProgram* tllmXqaJitProgram;

    typedef enum
    {
        TLLM_XQA_JIT_SUCCESS = 0,
        TLLM_XQA_JIT_INVALID_INPUT = 1,
        TLLM_XQA_JIT_INTERNAL_ERROR = 2,
    } tllmXqaJitStatus;

    // context must outlive prog.
    DLLEXPORT tllmXqaJitStatus tllmXqaJitCreateAndCompileProgram(
        tllmXqaJitProgram* prog, tllmXqaJitContext const* context);
    DLLEXPORT tllmXqaJitStatus tllmXqaJitGetCUBINSize(tllmXqaJitProgram prog, size_t* cubinSizeRet);
    DLLEXPORT tllmXqaJitStatus tllmXqaJitGetCUBIN(tllmXqaJitProgram prog, char* cubin);
    DLLEXPORT tllmXqaJitStatus tllmXqaJitDestroyProgram(tllmXqaJitProgram* prog);

    // Returns the size of the error string associated with the last non-success tllmXqaJit function call (including the
    // trailing \0). Returns 0 if there is no such non-success function call.
    DLLEXPORT size_t tllmXqaJitGetLastErrorStringSize();
    // Returns the error string.
    // Output can be nullptr if the returned value of tllmGetLastErrorStringSize() is 0.
    DLLEXPORT void tllmXqaJitGetLastErrorString(char* output);

#if __cplusplus
} // extern "C"
#endif
