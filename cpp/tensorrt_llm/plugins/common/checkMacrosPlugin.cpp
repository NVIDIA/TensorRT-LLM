/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "checkMacrosPlugin.h"

#include "tensorrt_llm/common/logger.h"

namespace tensorrt_llm::plugins
{

void caughtError(std::exception const& e)
{
    TLLM_LOG_EXCEPTION(e);
}

void logError(char const* msg, char const* file, char const* fn, int line)
{
    TLLM_LOG_ERROR("Parameter check failed at: %s::%s::%d, condition: %s", file, fn, line, msg);
}

} // namespace tensorrt_llm::plugins
