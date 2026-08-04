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

#include "model_state.h"

#include "tensorrt_llm/common/logger.h"

#include <fstream>

using namespace triton::backend::inflight_batcher_llm;

class ModelStateTest : public ::testing::Test
{
protected:
    void SetUp() override {}

    void TearDown() override
    {
        ;
    }
};

struct ModelStateTestUtils
{
    static void CompareModelStates(ModelState& ms1, ModelState& ms2)
    {
        EXPECT_EQ(ms1.GetModelName(), ms2.GetModelName());
        EXPECT_EQ(ms1.GetModelVersion(), ms2.GetModelVersion());
        EXPECT_EQ(ms1.GetExecutorWorkerPath(), ms2.GetExecutorWorkerPath());
        EXPECT_EQ(ms1.getDeviceIds(), ms2.getDeviceIds());
        EXPECT_EQ(ms1.IsDecoupled(), ms2.IsDecoupled());

        // Compare a few parameters
        EXPECT_EQ(ms1.GetParameter<std::string>("gpt_model_type"), ms2.GetParameter<std::string>("gpt_model_type"));
        EXPECT_EQ(ms1.GetParameter<int32_t>("max_beam_width"), ms2.GetParameter<int32_t>("max_beam_width"));
        EXPECT_EQ(ms1.GetParameter<bool>("normalize_log_probs"), ms2.GetParameter<bool>("normalize_log_probs"));
    }

    static void TestModelState(std::string name, uint64_t version, std::string jsonFileName)
    {
        // Force the creation of the logger so it gets destroyed last
        tensorrt_llm::common::Logger::getLogger();

        std::ifstream ifs(jsonFileName);
        EXPECT_EQ(ifs.is_open(), true);
        std::stringstream ss;
        ss << ifs.rdbuf();
        ifs.close();

        triton::common::TritonJson::Value modelConfig;
        EXPECT_EQ(modelConfig.Parse(ss.str()), TRITONJSON_STATUSSUCCESS);

        // create origin ModelState
        ModelState original(nullptr, name, version, std::move(modelConfig));

        // copy through deserialize(original.serialize())
        auto packed = original.serialize();
        auto copy = ModelState::deserialize(packed);

        ModelStateTestUtils::CompareModelStates(original, copy);
    }
};

TEST_F(ModelStateTest, 1)
{
    ModelStateTestUtils::TestModelState("first", 3, "first.json");
    ;
}

TEST_F(ModelStateTest, 2)
{
    ModelStateTestUtils::TestModelState("second", 7, "second.json");
    ;
}

TEST_F(ModelStateTest, 3)
{
    ModelStateTestUtils::TestModelState("third", 3, "third.json");
    ;
}
