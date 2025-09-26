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

#include <optional>

namespace tensorrt_llm::executor
{

GuidedDecodingParams::GuidedDecodingParams(GuideType guideType, std::optional<std::string> guide)
    : mGuideType{guideType}
    , mGuide{std::move(guide)}
{
    TLLM_CHECK_WITH_INFO(mGuideType == GuideType::kJSON || mGuide.has_value(),
        "The guide string must be provided unless using GuideType::kJSON.");
}

bool GuidedDecodingParams::operator==(GuidedDecodingParams const& other) const
{
    return mGuideType == other.mGuideType && mGuide == other.mGuide;
}

GuidedDecodingParams::GuideType GuidedDecodingParams::getGuideType() const
{
    return mGuideType;
}

std::optional<std::string> GuidedDecodingParams::getGuide() const
{
    return mGuide;
}

} // namespace tensorrt_llm::executor
