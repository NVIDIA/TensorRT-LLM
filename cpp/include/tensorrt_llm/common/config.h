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

#if (defined(TRTLLM_NAMESPACE_PREFIX) || defined(TRTLLM_NAMESPACE_POSTFIX)) && !defined(TRTLLM_NAMESPACE_QUALIFIER)
#error TRTLLM requires a definition of TRTLLM_NAMESPACE_QUALIFIER when TRTLLM_NAMESPACE_PREFIX/POSTFIX are defined.
#endif

/**
 * \def TRTLLM_WRAPPED_NAMESPACE
 * If defined, this value will be used as the name of a namespace that wraps the
 * `tensorrt_llm::` namespace.
 * If THRUST_TRTLLM_WRAPPED_NAMESPACE is set, this will inherit that macro's value.
 * This macro should not be used with any other TRTLLM namespace macros.
 */
#ifdef TRTLLM_WRAPPED_NAMESPACE
#define TRTLLM_NAMESPACE_PREFIX                                                                                               \
    namespace TRTLLM_WRAPPED_NAMESPACE                                                                                 \
    {
#define TRTLLM_NAMESPACE_POSTFIX }

#define TRTLLM_NAMESPACE_QUALIFIER ::TRTLLM_WRAPPED_NAMESPACE::tensorrt_llm
#endif

/**
 * \def TRTLLM_NAMESPACE_PREFIX
 * This macro is inserted prior to all `namespace cub { ... }` blocks. It is
 * derived from TRTLLM_WRAPPED_NAMESPACE, if set, and will be empty otherwise.
 * It may be defined by users, in which case TRTLLM_NAMESPACE_PREFIX,
 * TRTLLM_NAMESPACE_POSTFIX, and TRTLLM_NAMESPACE_QUALIFIER must all be set consistently.
 */
#ifndef TRTLLM_NAMESPACE_PREFIX
#define TRTLLM_NAMESPACE_PREFIX
#endif

/**
 * \def TRTLLM_NAMESPACE_POSTFIX
 * This macro is inserted following the closing braces of all
 * `namespace tensorrt_llm { ... }` block. It is defined appropriately when
 * TRTLLM_WRAPPED_NAMESPACE is set, and will be empty otherwise. It may be
 * defined by users, in which case TRTLLM_NAMESPACE_PREFIX, TRTLLM_NAMESPACE_POSTFIX, and
 * TRTLLM_NAMESPACE_QUALIFIER must all be set consistently.
 */
#ifndef TRTLLM_NAMESPACE_POSTFIX
#define TRTLLM_NAMESPACE_POSTFIX
#endif

/**
 * \def TRTLLM_NAMESPACE_QUALIFIER
 * This macro is used to qualify members of tensorrt_llm:: when accessing them from
 * outside of their namespace. By default, this is just `::cub`, and will be
 * set appropriately when TRTLLM_WRAPPED_NAMESPACE is defined. This macro may be
 * defined by users, in which case TRTLLM_NAMESPACE_PREFIX, TRTLLM_NAMESPACE_POSTFIX, and
 * TRTLLM_NAMESPACE_QUALIFIER must all be set consistently.
 */
#ifndef TRTLLM_NAMESPACE_QUALIFIER
#define TRTLLM_NAMESPACE_QUALIFIER ::tensorrt_llm
#endif

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
 * This macro is defined by TRTLLM and may not be overridden.
 */
#define TRTLLM_NAMESPACE_BEGIN                                                                                         \
    TRTLLM_NAMESPACE_PREFIX                                                                                            \
    namespace tensorrt_llm                                                                                                      \
    {                                                                                                                  \
    TRTLLM_ABI_NAMESPACE_BEGIN

/**
 * \def TRTLLM_NAMESPACE_END
 * This macro is used to close a `tensorrt_llm::` namespace block, along with any
 * enclosing namespaces requested by TRTLLM_WRAPPED_NAMESPACE, etc.
 * This macro is defined by TRTLLM and may not be overridden.
 */
#define TRTLLM_NAMESPACE_END                                                                                           \
    TRTLLM_ABI_NAMESPACE_END                                                                                           \
    } /* end namespace tensorrt_llm */                                                                                          \
    TRTLLM_NAMESPACE_POSTFIX

#endif // TRTLLM_CONFIG_H
