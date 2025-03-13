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
#include "tensorrt_llm/plugins/api/tllmPlugin.h"

#include "tensorrt_llm/common/stringUtils.h"
#include "tensorrt_llm/runtime/tllmLogger.h"

#include "tensorrt_llm/plugins/bertAttentionPlugin/bertAttentionPlugin.h"
#include "tensorrt_llm/plugins/doraPlugin/doraPlugin.h"
#include "tensorrt_llm/plugins/fp8RowwiseGemmPlugin/fp8RowwiseGemmPlugin.h"
#include "tensorrt_llm/plugins/fusedLayernormPlugin/fusedLayernormPlugin.h"
#include "tensorrt_llm/plugins/gemmPlugin/gemmPlugin.h"
#include "tensorrt_llm/plugins/gemmSwigluPlugin/gemmSwigluPlugin.h"
#include "tensorrt_llm/plugins/gptAttentionPlugin/gptAttentionPlugin.h"
#include "tensorrt_llm/plugins/identityPlugin/identityPlugin.h"
#include "tensorrt_llm/plugins/layernormQuantizationPlugin/layernormQuantizationPlugin.h"
#include "tensorrt_llm/plugins/lookupPlugin/lookupPlugin.h"
#include "tensorrt_llm/plugins/loraPlugin/loraPlugin.h"
#include "tensorrt_llm/plugins/lruPlugin/lruPlugin.h"
#include "tensorrt_llm/plugins/mambaConv1dPlugin/mambaConv1dPlugin.h"
#include "tensorrt_llm/plugins/mixtureOfExperts/mixtureOfExpertsPlugin.h"
#include "tensorrt_llm/plugins/quantizeToFP4Plugin/quantizeToFP4Plugin.h"
#if ENABLE_MULTI_DEVICE
#include "tensorrt_llm/plugins/cpSplitPlugin/cpSplitPlugin.h"
#include "tensorrt_llm/plugins/gemmAllReducePlugin/gemmAllReducePlugin.h"
#include "tensorrt_llm/plugins/ncclPlugin/allgatherPlugin.h"
#include "tensorrt_llm/plugins/ncclPlugin/allreducePlugin.h"
#include "tensorrt_llm/plugins/ncclPlugin/recvPlugin.h"
#include "tensorrt_llm/plugins/ncclPlugin/reduceScatterPlugin.h"
#include "tensorrt_llm/plugins/ncclPlugin/sendPlugin.h"
#endif // ENABLE_MULTI_DEVICE
#include "tensorrt_llm/plugins/cudaStreamPlugin/cudaStreamPlugin.h"
#include "tensorrt_llm/plugins/cumsumLastDimPlugin/cumsumLastDimPlugin.h"
#include "tensorrt_llm/plugins/eaglePlugin/eagleDecodeDraftTokensPlugin.h"
#include "tensorrt_llm/plugins/eaglePlugin/eaglePrepareDrafterInputsPlugin.h"
#include "tensorrt_llm/plugins/eaglePlugin/eagleSampleAndAcceptDraftTokensPlugin.h"
#include "tensorrt_llm/plugins/fp4GemmPlugin/fp4GemmPlugin.h"
#include "tensorrt_llm/plugins/lowLatencyGemmPlugin/lowLatencyGemmPlugin.h"
#include "tensorrt_llm/plugins/lowLatencyGemmSwigluPlugin/lowLatencyGemmSwigluPlugin.h"
#include "tensorrt_llm/plugins/qserveGemmPlugin/qserveGemmPlugin.h"
#include "tensorrt_llm/plugins/quantizePerTokenPlugin/quantizePerTokenPlugin.h"
#include "tensorrt_llm/plugins/quantizeTensorPlugin/quantizeTensorPlugin.h"
#include "tensorrt_llm/plugins/rmsnormQuantizationPlugin/rmsnormQuantizationPlugin.h"
#include "tensorrt_llm/plugins/selectiveScanPlugin/selectiveScanPlugin.h"
#include "tensorrt_llm/plugins/smoothQuantGemmPlugin/smoothQuantGemmPlugin.h"
#include "tensorrt_llm/plugins/topkLastDimPlugin/topkLastDimPlugin.h"
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

nvinfer1::IPluginCreatorInterface* creatorInterfacePtr(nvinfer1::IPluginCreatorInterface& creator)
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
[[maybe_unused]] __attribute__((constructor))
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

namespace tensorrt_llm::plugins::api
{

LoggerManager& tensorrt_llm::plugins::api::LoggerManager::getInstance() noexcept
{
    static LoggerManager instance;
    return instance;
}

void LoggerManager::setLoggerFinder(nvinfer1::ILoggerFinder* finder)
{
    std::lock_guard<std::mutex> lk(mMutex);
    if (mLoggerFinder == nullptr && finder != nullptr)
    {
        mLoggerFinder = finder;
    }
}

[[maybe_unused]] nvinfer1::ILogger* LoggerManager::logger()
{
    std::lock_guard<std::mutex> lk(mMutex);
    if (mLoggerFinder != nullptr)
    {
        return mLoggerFinder->findLogger();
    }
    return nullptr;
}

nvinfer1::ILogger* LoggerManager::defaultLogger() noexcept
{
    return gLogger;
}
} // namespace tensorrt_llm::plugins::api

// New Plugin APIs

extern "C"
{
    bool initTrtLlmPlugins(void* logger, char const* libNamespace)
    {
        if (pluginsInitialized)
        {
            return true;
        }

        if (logger)
        {
            gLogger = static_cast<nvinfer1::ILogger*>(logger);
        }
        setLoggerFinder(&gGlobalLoggerFinder);

        auto registry = getPluginRegistry();

        {
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
        }

        {
            std::int32_t nbCreators;
            auto creators = getCreators(nbCreators);

            for (std::int32_t i = 0; i < nbCreators; ++i)
            {
                auto const creator = creators[i];
                registry->registerCreator(*creator, libNamespace);
            }
        }

        pluginsInitialized = true;
        return true;
    }

    [[maybe_unused]] void setLoggerFinder([[maybe_unused]] nvinfer1::ILoggerFinder* finder)
    {
        tensorrt_llm::plugins::api::LoggerManager::getInstance().setLoggerFinder(finder);
    }

    [[maybe_unused]] nvinfer1::IPluginCreator* const* getPluginCreators(std::int32_t& nbCreators)
    {
        static tensorrt_llm::plugins::IdentityPluginCreator identityPluginCreator;
        static tensorrt_llm::plugins::BertAttentionPluginCreator bertAttentionPluginCreator;
        static tensorrt_llm::plugins::FusedLayernormPluginCreator fusedLayernormPluginCreator;
        static tensorrt_llm::plugins::GPTAttentionPluginCreator gptAttentionPluginCreator;
        static tensorrt_llm::plugins::GemmPluginCreator gemmPluginCreator;
        static tensorrt_llm::plugins::GemmSwigluPluginCreator gemmSwigluPluginCreator;
        static tensorrt_llm::plugins::Fp8RowwiseGemmPluginCreator fp8RowwiseGemmPluginCreator;
        static tensorrt_llm::plugins::MixtureOfExpertsPluginCreator moePluginCreator;
#if ENABLE_MULTI_DEVICE
        static tensorrt_llm::plugins::SendPluginCreator sendPluginCreator;
        static tensorrt_llm::plugins::RecvPluginCreator recvPluginCreator;
        static tensorrt_llm::plugins::AllreducePluginCreator allreducePluginCreator;
        static tensorrt_llm::plugins::AllgatherPluginCreator allgatherPluginCreator;
        static tensorrt_llm::plugins::ReduceScatterPluginCreator reduceScatterPluginCreator;
        static tensorrt_llm::plugins::GemmAllReducePluginCreator gemmAllReducePluginCreator;
#endif // ENABLE_MULTI_DEVICE
        static tensorrt_llm::plugins::SmoothQuantGemmPluginCreator smoothQuantGemmPluginCreator;
        static tensorrt_llm::plugins::QServeGemmPluginCreator qserveGemmPluginCreator;
        static tensorrt_llm::plugins::LayernormQuantizationPluginCreator layernormQuantizationPluginCreator;
        static tensorrt_llm::plugins::QuantizeToFP4PluginCreator quantizeToFP4PluginCreator;
        static tensorrt_llm::plugins::QuantizePerTokenPluginCreator quantizePerTokenPluginCreator;
        static tensorrt_llm::plugins::QuantizeTensorPluginCreator quantizeTensorPluginCreator;
        static tensorrt_llm::plugins::RmsnormQuantizationPluginCreator rmsnormQuantizationPluginCreator;
        static tensorrt_llm::plugins::WeightOnlyGroupwiseQuantMatmulPluginCreator
            weightOnlyGroupwiseQuantMatmulPluginCreator;
        static tensorrt_llm::plugins::WeightOnlyQuantMatmulPluginCreator weightOnlyQuantMatmulPluginCreator;
        static tensorrt_llm::plugins::LookupPluginCreator lookupPluginCreator;
        static tensorrt_llm::plugins::LoraPluginCreator loraPluginCreator;
        static tensorrt_llm::plugins::SelectiveScanPluginCreator selectiveScanPluginCreator;
        static tensorrt_llm::plugins::Fp4GemmPluginCreator fp4GemmPluginCreator;
        static tensorrt_llm::plugins::MambaConv1dPluginCreator mambaConv1DPluginCreator;
        static tensorrt_llm::plugins::lruPluginCreator lruPluginCreator;
        static tensorrt_llm::plugins::CumsumLastDimPluginCreator cumsumLastDimPluginCreator;
        static tensorrt_llm::plugins::TopkLastDimPluginCreator topkLastDimPluginCreator;
        static tensorrt_llm::plugins::LowLatencyGemmPluginCreator lowLatencyGemmPluginCreator;
        static tensorrt_llm::plugins::LowLatencyGemmSwigluPluginCreator lowLatencyGemmSwigluPluginCreator;
        static tensorrt_llm::plugins::EagleDecodeDraftTokensPluginCreator eagleDecodeDraftTokensPluginCreator;
        static tensorrt_llm::plugins::EagleSampleAndAcceptDraftTokensPluginCreator
            eagleSampleAndAcceptDraftTokensPluginCreator;
        static tensorrt_llm::plugins::CudaStreamPluginCreator cudaStreamPluginCreator;

        static std::array pluginCreators
            = { creatorPtr(identityPluginCreator),
                  creatorPtr(bertAttentionPluginCreator),
                  creatorPtr(gptAttentionPluginCreator),
                  creatorPtr(gemmPluginCreator),
                  creatorPtr(gemmSwigluPluginCreator),
                  creatorPtr(fp8RowwiseGemmPluginCreator),
                  creatorPtr(moePluginCreator),
#if ENABLE_MULTI_DEVICE
                  creatorPtr(sendPluginCreator),
                  creatorPtr(recvPluginCreator),
                  creatorPtr(allreducePluginCreator),
                  creatorPtr(allgatherPluginCreator),
                  creatorPtr(reduceScatterPluginCreator),
                  creatorPtr(gemmAllReducePluginCreator),
#endif // ENABLE_MULTI_DEVICE
                  creatorPtr(fusedLayernormPluginCreator),
                  creatorPtr(smoothQuantGemmPluginCreator),
                  creatorPtr(qserveGemmPluginCreator),
                  creatorPtr(layernormQuantizationPluginCreator),
                  creatorPtr(quantizeToFP4PluginCreator),
                  creatorPtr(quantizePerTokenPluginCreator),
                  creatorPtr(quantizeTensorPluginCreator),
                  creatorPtr(rmsnormQuantizationPluginCreator),
                  creatorPtr(weightOnlyGroupwiseQuantMatmulPluginCreator),
                  creatorPtr(weightOnlyQuantMatmulPluginCreator),
                  creatorPtr(lookupPluginCreator),
                  creatorPtr(loraPluginCreator),
                  creatorPtr(selectiveScanPluginCreator),
                  creatorPtr(fp4GemmPluginCreator),
                  creatorPtr(mambaConv1DPluginCreator),
                  creatorPtr(lruPluginCreator),
                  creatorPtr(cumsumLastDimPluginCreator),
                  creatorPtr(topkLastDimPluginCreator),
                  creatorPtr(lowLatencyGemmPluginCreator),
                  creatorPtr(eagleDecodeDraftTokensPluginCreator),
                  creatorPtr(eagleSampleAndAcceptDraftTokensPluginCreator),
                  creatorPtr(lowLatencyGemmSwigluPluginCreator),
                  creatorPtr(cudaStreamPluginCreator),
              };
        nbCreators = pluginCreators.size();
        return pluginCreators.data();
    }

    [[maybe_unused]] nvinfer1::IPluginCreatorInterface* const* getCreators(std::int32_t& nbCreators)
    {
        static tensorrt_llm::plugins::EaglePrepareDrafterInputsPluginCreator eaglePrepareDrafterInputsPluginCreator;
#if ENABLE_MULTI_DEVICE
        static tensorrt_llm::plugins::CpSplitPluginCreator cpSplitPluginCreator;
#endif // ENABLE_MULTI_DEVICE

        static tensorrt_llm::plugins::DoraPluginCreator doraPluginCreator;

        static std::array creators
            = { creatorInterfacePtr(eaglePrepareDrafterInputsPluginCreator),
#if ENABLE_MULTI_DEVICE
                  creatorInterfacePtr(cpSplitPluginCreator),
#endif // ENABLE_MULTI_DEVICE
                  creatorInterfacePtr(doraPluginCreator) };

        nbCreators = creators.size();
        return creators.data();
    }
} // extern "C"
