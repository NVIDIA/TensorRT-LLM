/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/nativeMemory.h"

#include <torch/extension.h>

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("get_native_memory_usage(int device) -> (str[], int[], int[], int[])");
    m.def("get_native_memory_labels() -> str[]");
}

// These ops take no Tensor argument, so nothing selects a CPU or CUDA kernel; CatchAll serves every dispatch key.
TORCH_LIBRARY_IMPL(trtllm, CatchAll, m)
{
    m.impl("get_native_memory_usage",
        [](int64_t device)
        {
            std::vector<std::string> labels;
            std::vector<int64_t> counts;
            std::vector<int64_t> bytes;
            std::vector<int64_t> unmeasured;
            for (auto const& [label, usage] : tensorrt_llm::common::getNativeMemoryUsage(static_cast<int>(device)))
            {
                labels.push_back(label);
                counts.push_back(usage.count);
                bytes.push_back(usage.bytes);
                unmeasured.push_back(usage.unmeasured);
            }
            return std::make_tuple(labels, counts, bytes, unmeasured);
        });
    m.impl("get_native_memory_labels",
        []()
        {
            auto const& all = tensorrt_llm::common::nativeMemoryLabels::kAll;
            return std::vector<std::string>(all.begin(), all.end());
        });
}
