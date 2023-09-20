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
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "tensorrt_llm/plugins/bertAttentionPlugin/bertAttentionPlugin.h"
#include "tensorrt_llm/plugins/common/checkMacrosPlugin.h"
#include "tensorrt_llm/plugins/common/plugin.h"
#include "tensorrt_llm/plugins/gemmPlugin/gemmPlugin.h"
#include "tensorrt_llm/plugins/gptAttentionPlugin/gptAttentionPlugin.h"
#include "tensorrt_llm/plugins/identityPlugin/identityPlugin.h"
#include "tensorrt_llm/plugins/layernormPlugin/layernormPlugin.h"
#include "tensorrt_llm/plugins/layernormQuantizationPlugin/layernormQuantizationPlugin.h"
#include "tensorrt_llm/plugins/lookupPlugin/lookupPlugin.h"
#if ENABLE_MULTI_DEVICE
#include "tensorrt_llm/plugins/ncclPlugin/allgatherPlugin.h"
#include "tensorrt_llm/plugins/ncclPlugin/allreducePlugin.h"
#include "tensorrt_llm/plugins/ncclPlugin/recvPlugin.h"
#include "tensorrt_llm/plugins/ncclPlugin/sendPlugin.h"
#endif // ENABLE_MULTI_DEVICE
#include "tensorrt_llm/plugins/quantizePerTokenPlugin/quantizePerTokenPlugin.h"
#include "tensorrt_llm/plugins/quantizeTensorPlugin/quantizeTensorPlugin.h"
#include "tensorrt_llm/plugins/rmsnormPlugin/rmsnormPlugin.h"
#include "tensorrt_llm/plugins/rmsnormQuantizationPlugin/rmsnormQuantizationPlugin.h"
#include "tensorrt_llm/plugins/smoothQuantGemmPlugin/smoothQuantGemmPlugin.h"
#include "tensorrt_llm/plugins/weightOnlyGroupwiseQuantMatmulPlugin/weightOnlyGroupwiseQuantMatmulPlugin.h"
#include "tensorrt_llm/plugins/weightOnlyQuantMatmulPlugin/weightOnlyQuantMatmulPlugin.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <mutex>
#include <stack>
#include <unordered_set>
using namespace nvinfer1;
using namespace nvinfer1::plugin;

namespace nvinfer1
{

namespace
{

// This singleton ensures that each plugin is only registered once for a given
// namespace and type, and attempts of duplicate registration are ignored.
class PluginCreatorRegistry
{
public:
    static PluginCreatorRegistry& getInstance()
    {
        static PluginCreatorRegistry instance;
        return instance;
    }

    template <typename CreatorType>
    void addPluginCreator(void* logger, const char* libNamespace)
    {
        // Make accesses to the plugin creator registry thread safe
        std::lock_guard<std::mutex> lock(mRegistryLock);

        std::string errorMsg;
        std::string verboseMsg;

        std::unique_ptr<CreatorType> pluginCreator{new CreatorType{}};
        pluginCreator->setPluginNamespace(libNamespace);

        nvinfer1::ILogger* trtLogger = static_cast<nvinfer1::ILogger*>(logger);
        std::string pluginType = std::string{pluginCreator->getPluginNamespace()}
            + "::" + std::string{pluginCreator->getPluginName()} + " version "
            + std::string{pluginCreator->getPluginVersion()};

        if (mRegistryList.find(pluginType) == mRegistryList.end())
        {
            bool status = getPluginRegistry()->registerCreator(*pluginCreator, libNamespace);
            if (status)
            {
                mRegistry.push(std::move(pluginCreator));
                mRegistryList.insert(pluginType);
                verboseMsg = "Registered plugin creator - " + pluginType;
            }
            else
            {
                errorMsg = "Could not register plugin creator -  " + pluginType;
            }
        }
        else
        {
            verboseMsg = "Plugin creator already registered - " + pluginType;
        }

        if (trtLogger)
        {
            if (!errorMsg.empty())
            {
                trtLogger->log(ILogger::Severity::kERROR, errorMsg.c_str());
            }

            if (!verboseMsg.empty())
            {
                trtLogger->log(ILogger::Severity::kVERBOSE, verboseMsg.c_str());
            }
        }
    }

    ~PluginCreatorRegistry()
    {
        std::lock_guard<std::mutex> lock(mRegistryLock);

        // Release pluginCreators in LIFO order of registration.
        while (!mRegistry.empty())
        {
            mRegistry.pop();
        }
        mRegistryList.clear();
    }

private:
    PluginCreatorRegistry() {}

    std::mutex mRegistryLock;
    std::stack<std::unique_ptr<IPluginCreator>> mRegistry;
    std::unordered_set<std::string> mRegistryList;

public:
    PluginCreatorRegistry(PluginCreatorRegistry const&) = delete;
    void operator=(PluginCreatorRegistry const&) = delete;
};

template <typename CreatorType>
void initializePlugin(void* logger, const char* libNamespace)
{
    PluginCreatorRegistry::getInstance().addPluginCreator<CreatorType>(logger, libNamespace);
}

} // namespace
} // namespace nvinfer1

// New Plugin APIs

extern "C"
{
    bool initLibNvInferPlugins(void* logger, const char* libNamespace)
    {
        nvinfer1::initializePlugin<nvinfer1::plugin::IdentityPluginCreator>(logger, libNamespace);
        nvinfer1::initializePlugin<nvinfer1::plugin::BertAttentionPluginCreator>(logger, libNamespace);
        nvinfer1::initializePlugin<nvinfer1::plugin::GPTAttentionPluginCreator>(logger, libNamespace);
        nvinfer1::initializePlugin<nvinfer1::plugin::GemmPluginCreator>(logger, libNamespace);
#if ENABLE_MULTI_DEVICE
        nvinfer1::initializePlugin<nvinfer1::plugin::SendPluginCreator>(logger, libNamespace);
        nvinfer1::initializePlugin<nvinfer1::plugin::RecvPluginCreator>(logger, libNamespace);
        nvinfer1::initializePlugin<nvinfer1::plugin::AllreducePluginCreator>(logger, libNamespace);
        nvinfer1::initializePlugin<nvinfer1::plugin::AllgatherPluginCreator>(logger, libNamespace);
#endif // ENABLE_MULTI_DEVICE
        nvinfer1::initializePlugin<nvinfer1::plugin::LayernormPluginCreator>(logger, libNamespace);
        nvinfer1::initializePlugin<nvinfer1::plugin::RmsnormPluginCreator>(logger, libNamespace);
        nvinfer1::initializePlugin<nvinfer1::plugin::SmoothQuantGemmPluginCreator>(logger, libNamespace);
        nvinfer1::initializePlugin<nvinfer1::plugin::LayernormQuantizationPluginCreator>(logger, libNamespace);
        nvinfer1::initializePlugin<nvinfer1::plugin::QuantizePerTokenPluginCreator>(logger, libNamespace);
        nvinfer1::initializePlugin<nvinfer1::plugin::QuantizeTensorPluginCreator>(logger, libNamespace);
        nvinfer1::initializePlugin<nvinfer1::plugin::RmsnormQuantizationPluginCreator>(logger, libNamespace);
        nvinfer1::initializePlugin<nvinfer1::plugin::WeightOnlyGroupwiseQuantMatmulPluginCreator>(logger, libNamespace);
        nvinfer1::initializePlugin<nvinfer1::plugin::WeightOnlyQuantMatmulPluginCreator>(logger, libNamespace);
        nvinfer1::initializePlugin<nvinfer1::plugin::LookupPluginCreator>(logger, libNamespace);
        return true;
    }
} // extern "C"
