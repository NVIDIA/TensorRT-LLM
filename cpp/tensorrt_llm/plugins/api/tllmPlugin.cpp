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
#include "tllmPlugin.h"

#include "tensorrt_llm/common/stringUtils.h"
#include "tensorrt_llm/runtime/tllmLogger.h"

#include "tensorrt_llm/plugins/bertAttentionPlugin/bertAttentionPlugin.h"
#include "tensorrt_llm/plugins/gemmPlugin/gemmPlugin.h"
#include "tensorrt_llm/plugins/gptAttentionPlugin/gptAttentionPlugin.h"
#include "tensorrt_llm/plugins/identityPlugin/identityPlugin.h"
#include "tensorrt_llm/plugins/layernormPlugin/layernormPlugin.h"
#include "tensorrt_llm/plugins/layernormQuantizationPlugin/layernormQuantizationPlugin.h"
#include "tensorrt_llm/plugins/lookupPlugin/lookupPlugin.h"
#include "tensorrt_llm/plugins/loraPlugin/loraPlugin.h"
#include "tensorrt_llm/plugins/mixtureOfExperts/mixtureOfExpertsPlugin.h"
#if ENABLE_MULTI_DEVICE
#include "tensorrt_llm/plugins/ncclPlugin/allgatherPlugin.h"
#include "tensorrt_llm/plugins/ncclPlugin/allreducePlugin.h"
#include "tensorrt_llm/plugins/ncclPlugin/recvPlugin.h"
#include "tensorrt_llm/plugins/ncclPlugin/reduceScatterPlugin.h"
#include "tensorrt_llm/plugins/ncclPlugin/sendPlugin.h"
#endif // ENABLE_MULTI_DEVICE
#include "tensorrt_llm/plugins/quantizePerTokenPlugin/quantizePerTokenPlugin.h"
#include "tensorrt_llm/plugins/quantizeTensorPlugin/quantizeTensorPlugin.h"
#include "tensorrt_llm/plugins/rmsnormPlugin/rmsnormPlugin.h"
#include "tensorrt_llm/plugins/rmsnormQuantizationPlugin/rmsnormQuantizationPlugin.h"
#include "tensorrt_llm/plugins/selectiveScanPlugin/selectiveScanPlugin.h"
#include "tensorrt_llm/plugins/smoothQuantGemmPlugin/smoothQuantGemmPlugin.h"
#include "tensorrt_llm/plugins/weightOnlyGroupwiseQuantMatmulPlugin/weightOnlyGroupwiseQuantMatmulPlugin.h"
#include "tensorrt_llm/plugins/weightOnlyQuantMatmulPlugin/weightOnlyQuantMatmulPlugin.h"

#include <array>
#include <cstdlib>

#include <NvInferRuntime.h>

namespace tc = tensorrt_llm::common;

namespace
{

nvinfer1::IPluginCreator* creatorPtr(nvinfer1::IPluginCreator& creator)
{
    return &creator;
}

auto tllmLogger = tensorrt_llm::runtime::TllmLogger();

nvinfer1::ILogger* gLogger{&tllmLogger};

class GlobalLoggerFinder : public nvinfer1::ILoggerFinder
{
public:
    nvinfer1::ILogger* findLogger() override
    {
        return gLogger;
    }
};

GlobalLoggerFinder gGlobalLoggerFinder{};

#if !defined(_MSC_VER)
__attribute__((constructor))
#endif
void initOnLoad()
{
    auto constexpr kLoadPlugins = "TRT_LLM_LOAD_PLUGINS";
    auto const loadPlugins = std::getenv(kLoadPlugins);
    if (loadPlugins && loadPlugins[0] == '1')
    {
        initTrtLlmPlugins(gLogger);
    }
}

bool pluginsInitialized = false;

} // namespace

// New Plugin APIs

extern "C"
{
    bool initTrtLlmPlugins(void* logger, const char* libNamespace)
    {
        if (pluginsInitialized)
            return true;

        if (logger)
        {
            gLogger = static_cast<nvinfer1::ILogger*>(logger);
        }
        setLoggerFinder(&gGlobalLoggerFinder);

        auto registry = getPluginRegistry();
        std::int32_t nbCreators;
        auto creators = getPluginCreators(nbCreators);

        for (std::int32_t i = 0; i < nbCreators; ++i)
        {
            auto const creator = creators[i];
            creator->setPluginNamespace(libNamespace);
            registry->registerCreator(*creator, libNamespace);
            if (gLogger)
            {
                auto const msg = tc::fmtstr("Registered plugin creator %s version %s in namespace %s",
                    creator->getPluginName(), creator->getPluginVersion(), libNamespace);
                gLogger->log(nvinfer1::ILogger::Severity::kVERBOSE, msg.c_str());
            }
        }

        pluginsInitialized = true;
        return true;
    }

    [[maybe_unused]] void setLoggerFinder([[maybe_unused]] nvinfer1::ILoggerFinder* finder)
    {
        tensorrt_llm::plugins::api::LoggerFinder::getInstance().setLoggerFinder(finder);
    }

    [[maybe_unused]] nvinfer1::IPluginCreator* const* getPluginCreators(std::int32_t& nbCreators)
    {
        static tensorrt_llm::plugins::IdentityPluginCreator identityPluginCreator;
        static tensorrt_llm::plugins::BertAttentionPluginCreator bertAttentionPluginCreator;
        static tensorrt_llm::plugins::GPTAttentionPluginCreator gptAttentionPluginCreator;
        static tensorrt_llm::plugins::GemmPluginCreator gemmPluginCreator;
        static tensorrt_llm::plugins::MixtureOfExpertsPluginCreator moePluginCreator;
#if ENABLE_MULTI_DEVICE
        static tensorrt_llm::plugins::SendPluginCreator sendPluginCreator;
        static tensorrt_llm::plugins::RecvPluginCreator recvPluginCreator;
        static tensorrt_llm::plugins::AllreducePluginCreator allreducePluginCreator;
        static tensorrt_llm::plugins::AllgatherPluginCreator allgatherPluginCreator;
        static tensorrt_llm::plugins::ReduceScatterPluginCreator reduceScatterPluginCreator;
#endif // ENABLE_MULTI_DEVICE
        static tensorrt_llm::plugins::LayernormPluginCreator layernormPluginCreator;
        static tensorrt_llm::plugins::RmsnormPluginCreator rmsnormPluginCreator;
        static tensorrt_llm::plugins::SmoothQuantGemmPluginCreator smoothQuantGemmPluginCreator;
        static tensorrt_llm::plugins::LayernormQuantizationPluginCreator layernormQuantizationPluginCreator;
        static tensorrt_llm::plugins::QuantizePerTokenPluginCreator quantizePerTokenPluginCreator;
        static tensorrt_llm::plugins::QuantizeTensorPluginCreator quantizeTensorPluginCreator;
        static tensorrt_llm::plugins::RmsnormQuantizationPluginCreator rmsnormQuantizationPluginCreator;
        static tensorrt_llm::plugins::WeightOnlyGroupwiseQuantMatmulPluginCreator
            weightOnlyGroupwiseQuantMatmulPluginCreator;
        static tensorrt_llm::plugins::WeightOnlyQuantMatmulPluginCreator weightOnlyQuantMatmulPluginCreator;
        static tensorrt_llm::plugins::LookupPluginCreator lookupPluginCreator;
        static tensorrt_llm::plugins::LoraPluginCreator loraPluginCreator;
        static tensorrt_llm::plugins::SelectiveScanPluginCreator selectiveScanPluginCreator;

        static std::array pluginCreators
            = { creatorPtr(identityPluginCreator),
                  creatorPtr(bertAttentionPluginCreator),
                  creatorPtr(gptAttentionPluginCreator),
                  creatorPtr(gemmPluginCreator),
                  creatorPtr(moePluginCreator),
#if ENABLE_MULTI_DEVICE
                  creatorPtr(sendPluginCreator),
                  creatorPtr(recvPluginCreator),
                  creatorPtr(allreducePluginCreator),
                  creatorPtr(allgatherPluginCreator),
                  creatorPtr(reduceScatterPluginCreator),
#endif // ENABLE_MULTI_DEVICE
                  creatorPtr(layernormPluginCreator),
                  creatorPtr(rmsnormPluginCreator),
                  creatorPtr(smoothQuantGemmPluginCreator),
                  creatorPtr(layernormQuantizationPluginCreator),
                  creatorPtr(quantizePerTokenPluginCreator),
                  creatorPtr(quantizeTensorPluginCreator),
                  creatorPtr(rmsnormQuantizationPluginCreator),
                  creatorPtr(weightOnlyGroupwiseQuantMatmulPluginCreator),
                  creatorPtr(weightOnlyQuantMatmulPluginCreator),
                  creatorPtr(lookupPluginCreator),
                  creatorPtr(loraPluginCreator),
                  creatorPtr(selectiveScanPluginCreator),
              };
        nbCreators = pluginCreators.size();
        return pluginCreators.data();
    }

} // extern "C"

namespace tensorrt_llm::plugins::api
{
LoggerFinder& tensorrt_llm::plugins::api::LoggerFinder::getInstance() noexcept
{
    static LoggerFinder instance;
    return instance;
}

void LoggerFinder::setLoggerFinder(nvinfer1::ILoggerFinder* finder)
{
    std::lock_guard<std::mutex> lk(mMutex);
    if (mLoggerFinder == nullptr && finder != nullptr)
    {
        mLoggerFinder = finder;
    }
}

nvinfer1::ILogger* LoggerFinder::findLogger()
{
    std::lock_guard<std::mutex> lk(mMutex);
    if (mLoggerFinder != nullptr)
    {
        return mLoggerFinder->findLogger();
    }
    return nullptr;
}
} // namespace tensorrt_llm::plugins::api
