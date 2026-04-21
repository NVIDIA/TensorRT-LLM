/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "../DevKernel.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
// Routing-specific launch macros.
// These macros build on top of LAUNCH_ESC from DevKernel.h.
//
// Unlike the generic LAUNCH_PDL (which instantiates 2 kernels for UsePdl=true/false),
// LAUNCH_PDL_ROUTING instantiates only 1 kernel and passes UsePdl as a runtime field
// in KernelParams.  This halves routing kernel instantiations.
////////////////////////////////////////////////////////////////////////////////////////////////////

#define LAUNCH_PDL_ROUTING(data, coopLaunch, types, kernel, numBlocks, numThreads, smemSize, stream)                   \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaLaunchConfig_t config{};                                                                                   \
        config.gridDim = numBlocks;                                                                                    \
        config.blockDim = numThreads;                                                                                  \
        config.dynamicSmemBytes = smemSize;                                                                            \
        config.stream = (cudaStream_t) stream;                                                                         \
                                                                                                                       \
        cudaLaunchAttribute attributes[2] = {};                                                                        \
        attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;                                         \
        attributes[0].val.programmaticStreamSerializationAllowed = int(data.mUsePdl);                                  \
        attributes[1].id = cudaLaunchAttributeCooperative;                                                             \
        attributes[1].val.cooperative = int(coopLaunch);                                                               \
        config.attrs = attributes;                                                                                     \
        config.numAttrs = 2;                                                                                           \
        auto params = KernelParams<types>::setKernelParams(data);                                                      \
        auto kernelTyped = kernel<KernelParams<types>>;                                                                \
        if (smemSize > 48 * 1024)                                                                                      \
            TLLM_CUDA_CHECK(cudaFuncSetAttribute(kernelTyped, cudaFuncAttributeMaxDynamicSharedMemorySize, smemSize)); \
        TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, kernelTyped, params));                                             \
    } while (0)

// Llama4 dispatch: uses data.mDtypeOutput.
#define LAUNCH_ROUTING_LLAMA4(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream)                       \
    if (data.mDtypeOutput == tg::Dtype::Fp32)                                                                          \
    {                                                                                                                  \
        LAUNCH_PDL_ROUTING(data, coopLaunch,                                                                           \
            LAUNCH_ESC(float, float, 128 /* Always 128 for llama4*/, 1 /* Always 1 for llama4*/), kernel, numBlocks,   \
            numThreads, smemSize, stream);                                                                             \
    }                                                                                                                  \
    else if (data.mDtypeOutput == tg::Dtype::Bfloat16)                                                                 \
    {                                                                                                                  \
        LAUNCH_PDL_ROUTING(data, coopLaunch,                                                                           \
            LAUNCH_ESC(__nv_bfloat16, __nv_bfloat16, 128 /* Always 128 for llama4*/, 1 /* Always 1 for llama4*/),      \
            kernel, numBlocks, numThreads, smemSize, stream);                                                          \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        TLLM_LOG_ERROR("Unsupported dtypeExpW");                                                                       \
    }

// DeepSeek dispatch: uses data.mDtypeOutput.
#define LAUNCH_ROUTING_WITH_NUM_EXPERTS_FORCE_FLOAT_INPUT(data, coopLaunch, kernel, numBlocks, numThreads, smemSize,   \
    stream, extraFlag, forceFloatInput, numExperts, numTopExperts)                                                     \
    if (data.mDtypeOutput == tg::Dtype::Fp32 && extraFlag)                                                             \
    {                                                                                                                  \
        LAUNCH_PDL_ROUTING(data, coopLaunch, LAUNCH_ESC(float, float, numExperts, numTopExperts, true), kernel,        \
            numBlocks, numThreads, smemSize, stream);                                                                  \
    }                                                                                                                  \
    else if (data.mDtypeOutput == tg::Dtype::Fp32)                                                                     \
    {                                                                                                                  \
        LAUNCH_PDL_ROUTING(data, coopLaunch, LAUNCH_ESC(float, float, numExperts, numTopExperts, false), kernel,       \
            numBlocks, numThreads, smemSize, stream);                                                                  \
    }                                                                                                                  \
    else if (data.mDtypeOutput == tg::Dtype::Bfloat16 && extraFlag && forceFloatInput)                                 \
    {                                                                                                                  \
        LAUNCH_PDL_ROUTING(data, coopLaunch, LAUNCH_ESC(float, __nv_bfloat16, numExperts, numTopExperts, true),        \
            kernel, numBlocks, numThreads, smemSize, stream);                                                          \
    }                                                                                                                  \
    else if (data.mDtypeOutput == tg::Dtype::Bfloat16 && extraFlag)                                                    \
    {                                                                                                                  \
        LAUNCH_PDL_ROUTING(data, coopLaunch,                                                                           \
            LAUNCH_ESC(__nv_bfloat16, __nv_bfloat16, numExperts, numTopExperts, true), kernel, numBlocks, numThreads,  \
            smemSize, stream);                                                                                         \
    }                                                                                                                  \
    else if (data.mDtypeOutput == tg::Dtype::Bfloat16 && forceFloatInput)                                              \
    {                                                                                                                  \
        LAUNCH_PDL_ROUTING(data, coopLaunch, LAUNCH_ESC(float, __nv_bfloat16, numExperts, numTopExperts, false),       \
            kernel, numBlocks, numThreads, smemSize, stream);                                                          \
    }                                                                                                                  \
    else if (data.mDtypeOutput == tg::Dtype::Bfloat16)                                                                 \
    {                                                                                                                  \
        LAUNCH_PDL_ROUTING(data, coopLaunch,                                                                           \
            LAUNCH_ESC(__nv_bfloat16, __nv_bfloat16, numExperts, numTopExperts, false), kernel, numBlocks, numThreads, \
            smemSize, stream);                                                                                         \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        TLLM_LOG_ERROR("Unsupported dtypeExpW");                                                                       \
    }

////////////////////////////////////////////////////////////////////////////////////////////////////

// routingCustom dispatch: uses data.mDtypeOutput (OutputT) and data.mDtypeInput (InputT).
// These are routingCustom::Data fields, NOT used by DeepSeek/Llama4 macros.
// Wraps (PreProc, PostProc) into TopKExpertSelect for the standard preprocess→topK→postprocess flow.
#define LAUNCH_ROUTING_WITH_POLICIES(                                                                                  \
    data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, PreProc, PostProc, numExperts, numTopExperts)   \
    if (data.mDtypeOutput == tg::Dtype::Fp32)                                                                          \
    {                                                                                                                  \
        LAUNCH_PDL_ROUTING(data, coopLaunch,                                                                           \
            LAUNCH_ESC(float, float, numExperts, numTopExperts, TopKExpertSelect<PreProc, PostProc>), kernel,          \
            numBlocks, numThreads, smemSize, stream);                                                                  \
    }                                                                                                                  \
    else if (data.mDtypeOutput == tg::Dtype::Bfloat16 && data.mDtypeInput == tg::Dtype::Fp32)                          \
    {                                                                                                                  \
        LAUNCH_PDL_ROUTING(data, coopLaunch,                                                                           \
            LAUNCH_ESC(float, __nv_bfloat16, numExperts, numTopExperts, TopKExpertSelect<PreProc, PostProc>), kernel,  \
            numBlocks, numThreads, smemSize, stream);                                                                  \
    }                                                                                                                  \
    else if (data.mDtypeOutput == tg::Dtype::Bfloat16)                                                                 \
    {                                                                                                                  \
        LAUNCH_PDL_ROUTING(data, coopLaunch,                                                                           \
            LAUNCH_ESC(__nv_bfloat16, __nv_bfloat16, numExperts, numTopExperts, TopKExpertSelect<PreProc, PostProc>),  \
            kernel, numBlocks, numThreads, smemSize, stream);                                                          \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        TLLM_LOG_ERROR("Unsupported dtypeOutput");                                                                     \
    }

// routingCustom dispatch for custom ExpertSelectPolicy types that don't use PreProc/PostProc.
// Use this when the policy does NOT follow the standard preprocess→topK→postprocess pattern.
// ExpertSelect must satisfy the ExpertSelectPolicy concept (see RoutingCustomPolicy.cuh).
#define LAUNCH_ROUTING_WITH_EXPERT_SELECT(                                                                             \
    data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, ExpertSelect, numExperts, numTopExperts)        \
    if (data.mDtypeOutput == tg::Dtype::Fp32)                                                                          \
    {                                                                                                                  \
        LAUNCH_PDL_ROUTING(data, coopLaunch, LAUNCH_ESC(float, float, numExperts, numTopExperts, ExpertSelect),        \
            kernel, numBlocks, numThreads, smemSize, stream);                                                          \
    }                                                                                                                  \
    else if (data.mDtypeOutput == tg::Dtype::Bfloat16 && data.mDtypeInput == tg::Dtype::Fp32)                          \
    {                                                                                                                  \
        LAUNCH_PDL_ROUTING(data, coopLaunch,                                                                           \
            LAUNCH_ESC(float, __nv_bfloat16, numExperts, numTopExperts, ExpertSelect), kernel, numBlocks, numThreads,  \
            smemSize, stream);                                                                                         \
    }                                                                                                                  \
    else if (data.mDtypeOutput == tg::Dtype::Bfloat16)                                                                 \
    {                                                                                                                  \
        LAUNCH_PDL_ROUTING(data, coopLaunch,                                                                           \
            LAUNCH_ESC(__nv_bfloat16, __nv_bfloat16, numExperts, numTopExperts, ExpertSelect), kernel, numBlocks,      \
            numThreads, smemSize, stream);                                                                             \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        TLLM_LOG_ERROR("Unsupported dtypeOutput");                                                                     \
    }

////////////////////////////////////////////////////////////////////////////////////////////////////
