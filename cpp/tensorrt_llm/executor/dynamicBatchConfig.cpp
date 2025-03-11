/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/executor/executor.h"

namespace tensorrt_llm::executor
{

DynamicBatchConfig::DynamicBatchConfig(bool enableBatchSizeTuning, bool enableMaxNumTokensTuning,
    SizeType32 dynamicBatchMovingAverageWindow, std::vector<std::pair<SizeType32, SizeType32>> batchSizeTable)
    : mEnableBatchSizeTuning(enableBatchSizeTuning)
    , mEnableMaxNumTokensTuning(enableMaxNumTokensTuning)
    , mDynamicBatchMovingAverageWindow(dynamicBatchMovingAverageWindow)
    , mBatchSizeTable(batchSizeTable)
{
}

bool DynamicBatchConfig::getEnableBatchSizeTuning() const
{
    return mEnableBatchSizeTuning;
}

bool DynamicBatchConfig::getEnableMaxNumTokensTuning() const
{
    return mEnableMaxNumTokensTuning;
}

SizeType32 DynamicBatchConfig::getDynamicBatchMovingAverageWindow() const
{
    return mDynamicBatchMovingAverageWindow;
}

std::vector<std::pair<SizeType32, SizeType32>> DynamicBatchConfig::getBatchSizeTable() const
{
    return mBatchSizeTable;
}

std::vector<std::pair<SizeType32, SizeType32>> const DynamicBatchConfig::kDefaultBatchSizeTable{
    {144, 128},
    {336, 256},
    {672, 512},
    {832, 768},
    {1280, 1024},
    {1664, 1536},
};

} // namespace tensorrt_llm::executor
