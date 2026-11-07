/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/kernels/minimaxM3Sm107Tp4/sampledAccept.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3Sm107Tp4
{
namespace sampledAccept
{
long long sampledAcceptWsWords(int world, int oneRoundK)
{
    return minimaxM3Sm107::sampledAccept::sampledAcceptWsWords(world, oneRoundK);
}

cudaError_t invokeSampledAccept(Args const& a, bool usePdl, cudaStream_t stream)
{
    return minimaxM3Sm107::sampledAccept::invokeSampledAccept<kMaxVs>(a, usePdl, stream);
}
} // namespace sampledAccept
} // namespace minimaxM3Sm107Tp4
} // namespace kernels

TRTLLM_NAMESPACE_END
