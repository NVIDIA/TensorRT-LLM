// Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <gtest/gtest.h>

#include "model_instance_state.h"
#include "model_state.h"

#include "tensorrt_llm/common/logger.h"

using namespace triton::backend::inflight_batcher_llm;

namespace triton::backend::inflight_batcher_llm::tests
{

class ModelInstanceStateTest : public ::testing::Test
{
protected:
    void SetUp() override {}

    void TearDown() override
    {
        ;
    }
};

TEST_F(ModelInstanceStateTest, ExecutorConfig)
{

    std::string jsonStr =
        R"(
{
    "parameters": {
        "normalize_log_probs": {
            "string_value": "false"
        },
        "multi_block_mode": {
            "string_value": "false"
        },
        "enable_context_fmha_fp32_acc": {
            "string_value": "true"
        },
        "cuda_graph_mode": {
            "string_value": "true"
        },
        "cuda_graph_cache_size": {
            "string_value": "10"
        },
        "gpt_model_type": {
            "string_value": "inflight_fused_batching"
        }
    }
}
)";

    triton::common::TritonJson::Value modelConfig;
    EXPECT_EQ(modelConfig.Parse(jsonStr), TRITONJSON_STATUSSUCCESS);

    ModelState modelState(nullptr, "test", 1, std::move(modelConfig));

    auto modelInstanceState = ModelInstanceState(&modelState);
    auto executorConfig = modelInstanceState.getExecutorConfigFromParams();
    auto extendedPerfCfg = executorConfig.getExtendedRuntimePerfKnobConfig();

    EXPECT_EQ(executorConfig.getNormalizeLogProbs(), false);
    EXPECT_EQ(extendedPerfCfg.getMultiBlockMode(), false);
    EXPECT_EQ(extendedPerfCfg.getEnableContextFMHAFP32Acc(), true);
    EXPECT_EQ(extendedPerfCfg.getCudaGraphMode(), true);
    EXPECT_EQ(extendedPerfCfg.getCudaGraphCacheSize(), 10);
}
} // namespace triton::backend::inflight_batcher_llm::tests
