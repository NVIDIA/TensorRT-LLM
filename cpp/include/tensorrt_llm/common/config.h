/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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
#ifndef TRTLLM_CONFIG_H
#define TRTLLM_CONFIG_H

/**
 * \def TRTLLM_ABI_NAMESPACE
 * This macro is used to open an implicitly inline namespace block for the ABI version.
 * This macro can be overridden to change the ABI version.
 * The default ABI version is _v1.
 */
#ifndef TRTLLM_ABI_NAMESPACE
#define TRTLLM_ABI_NAMESPACE _v1
#endif

#ifndef TRTLLM_ABI_NAMESPACE_BEGIN
#define TRTLLM_ABI_NAMESPACE_BEGIN                                                                                     \
    inline namespace TRTLLM_ABI_NAMESPACE                                                                              \
    {
#endif

#ifndef TRTLLM_ABI_NAMESPACE_END
#define TRTLLM_ABI_NAMESPACE_END }
#endif

/**
 * \def TRTLLM_NAMESPACE_BEGIN
 * This macro is used to open a `tensorrt_llm::` namespace block, along with any
 * enclosing namespaces requested by TRTLLM_WRAPPED_NAMESPACE, etc.
 * This macro is defined by TensorRT-LLM and may not be overridden.
 */
#define TRTLLM_NAMESPACE_BEGIN                                                                                         \
    namespace tensorrt_llm                                                                                             \
    {                                                                                                                  \
    TRTLLM_ABI_NAMESPACE_BEGIN

/**
 * \def TRTLLM_NAMESPACE_END
 * This macro is used to close a `tensorrt_llm::` namespace block, along with any
 * enclosing namespaces requested by TRTLLM_WRAPPED_NAMESPACE, etc.
 * This macro is defined by TensorRT-LLM and may not be overridden.
 */
#define TRTLLM_NAMESPACE_END                                                                                           \
    TRTLLM_ABI_NAMESPACE_END                                                                                           \
    }  /* end namespace tensorrt_llm */

#endif // TRTLLM_CONFIG_H
