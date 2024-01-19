/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include <cstdlib>

namespace tensorrt_llm::common
{

// XQA kernels (optimized kernels for generation phase).
bool forceXQAKernels()
{
    const char* force_xqa_env_var = getenv("TRTLLM_FORCE_XQA");
    static bool forceXQA = false;
    if (force_xqa_env_var != nullptr)
    {
        if (force_xqa_env_var[0] == '1' && force_xqa_env_var[1] == '\0')
        {
            forceXQA = true;
        }
    }
    return forceXQA;
}

// Tune the number of blocks per sequence for accuracy/performance purpose.
bool getEnvMmhaMultiblockDebug()
{
    static bool init = false;
    static bool forceMmhaMaxSeqLenTile = false;
    if (!init)
    {
        init = true;
        const char* enable_mmha_debug_var = std::getenv("TRTLLM_ENABLE_MMHA_MULTI_BLOCK_DEBUG");
        if (enable_mmha_debug_var)
        {
            if (enable_mmha_debug_var[0] == '1' && enable_mmha_debug_var[1] == '\0')
            {
                forceMmhaMaxSeqLenTile = true;
            }
        }
    }
    return forceMmhaMaxSeqLenTile;
}

int getEnvMmhaBlocksPerSequence()
{
    static bool init = false;
    static int mmhaBlocksPerSequence = 0;
    if (!init)
    {
        init = true;
        const char* mmhaBlocksPerSequenceEnv = std::getenv("TRTLLM_MMHA_BLOCKS_PER_SEQUENCE");
        if (mmhaBlocksPerSequenceEnv)
        {
            mmhaBlocksPerSequence = std::atoi(mmhaBlocksPerSequenceEnv);
            if (mmhaBlocksPerSequence <= 0)
            {
                TLLM_LOG_WARNING("Invalid value for TRTLLM_MMHA_BLOCKS_PER_SEQUENCE. Will use default values instead!");
            }
        }
    }
    return mmhaBlocksPerSequence;
}

} // namespace tensorrt_llm::common
