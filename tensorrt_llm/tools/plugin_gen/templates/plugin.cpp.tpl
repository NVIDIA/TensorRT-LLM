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

#include "plugin.h"
#include <iostream>

using namespace nvinfer1;
using nvinfer1::plugin::[[ plugin_name ]]Creator;
using nvinfer1::plugin::[[ plugin_name ]];

PluginFieldCollection [[ plugin_name ]]Creator::mFC{};
std::vector<PluginField> [[ plugin_name ]]Creator::mPluginAttributes;
static bool triton_kernels_loaded = false;

// constructor
[[ plugin_name ]]::[[ plugin_name ]]( [[ construct_arg_list ]] )
{
  {% for arg in params -%}
  this->[[arg.name]] = [[arg.name]];
  {% endfor %}
}


// Parameterized constructor
[[ plugin_name ]]::[[ plugin_name ]](const void* data, size_t length)
{
    const char *d = static_cast<const char*>(data), *a = d;

    {% for arg in params -%}
    read(d, [[arg.name]]);
    {% endfor %}
    TLLM_CHECK(d == a + getSerializationSize());
}


nvinfer1::IPluginV2DynamicExt* [[ plugin_name ]]::clone() const noexcept
{
  auto* plugin = new [[plugin_name]]([[', '.join(param_names)]]);
  plugin->setPluginNamespace(mNamespace.c_str());
  return plugin;
}


nvinfer1::DimsExprs [[ plugin_name ]]::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputDims, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
  [[ getOutputDimensions_body ]]
}

bool [[ plugin_name ]]::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
  PLUGIN_ASSERT(nbInputs + nbOutputs == [[io_count]]);
  PLUGIN_ASSERT(0 <= pos && pos < nbInputs + nbOutputs);
  PLUGIN_ASSERT(nbInputs == [[input_count]]);
  PLUGIN_ASSERT(nbOutputs == [[output_count]]);

  {% for arg in inputs %}
  if (pos == [[loop.index0]]) {
    {%- if arg.is_tensor -%}
    return inOut[pos].type == DataType::[[ arg.dtype.dtype.to('trt') ]] && inOut[pos].format == TensorFormat::kLINEAR;
    {%- else -%}
    return inOut[pos].type == DataType::[[ arg.dtype.dtype.to('trt') ]];
    {%- endif -%}
  }
  {% endfor %}

  {% for arg in outputs %}
  if (pos == nbInputs + [[loop.index0]])
    return inOut[pos].type == DataType::[[ arg.dtype.dtype.to('trt') ]] && inOut[pos].format == TensorFormat::kLINEAR;
  {% endfor %}
}


void [[ plugin_name ]]::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
  [[ configurePlugin_body ]]
}


size_t [[ plugin_name ]]::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
  [[ getWorkspaceSize_body ]]
}

int [[ plugin_name ]]::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
  // input arguments
  {% for arg in inputs %}
  const auto* [[arg.name]]_buf = reinterpret_cast<const [[arg.dtype.dtype.to("c")]] *>(inputs[ [[arg.offset]] ]);
  {% if arg.dtype.is_tensor -%}
  auto [[arg.name]] = reinterpret_cast<CUdeviceptr>([[arg.name]]_buf);
  {% else %}
  const auto [[arg.name]] = * [[arg.name]]_buf;
  {% endif -%}
  {% endfor %}

  // outputs
  {% for arg in outputs %}
  auto* [[arg.name]]_buf = reinterpret_cast<const [[arg.dtype.dtype.to("c")]] *>(outputs[ [[arg.offset]] ]);
  auto [[arg.name]] = reinterpret_cast<CUdeviceptr>([[arg.name]]_buf);
  {% endfor %}

  // dim size arguments
  {%- for arg in dim_size_args -%}
  {# code field is dedicated for DimSizeArg #}
  [[arg.dtype.dtype.to("c")]] [[arg.name]] = [[arg.code]];
  {%- endfor %}

  // TODO: Check result code
  [[kernel_name]]([[enqueue_body_arg_list]]);

  return 0;
}


nvinfer1::DataType [[ plugin_name ]]::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
  {% for arg in outputs %}
  if (index == [[loop.index0]]) {
    return DataType::[[arg.dtype.dtype.to('trt')]];
  }
  {% endfor %}
}


const char* [[ plugin_name ]]::getPluginType() const noexcept
{
  return "[[ plugin_name ]]";
}

const char* [[ plugin_name ]]::getPluginVersion() const noexcept
{
  return "[[ kernel_version ]]";
}

int [[ plugin_name ]]::getNbOutputs() const noexcept
{
  return [[outputs|length]];
}

int [[ plugin_name ]]::initialize() noexcept
{
  if (triton_kernels_loaded) {
      return 0;
  }
  load_[[kernel_name]]();
  triton_kernels_loaded = true;
  return 0;
}

void [[ plugin_name ]]::terminate() noexcept {
  if (!triton_kernels_loaded) {
      return;
  }
  unload_[[kernel_name]]();
  triton_kernels_loaded = false;
}

size_t [[ plugin_name ]]::getSerializationSize() const noexcept
{
  size_t ret = 0;

  {% for arg in params -%}
  ret += sizeof([[arg.dtype.dtype.to('c')]]);
  {% endfor %}

  return ret;

}

void [[ plugin_name ]]::serialize(void* buffer) const noexcept
{

    char *d = static_cast<char*>(buffer), *a = d;

    {% for arg in params -%}
    write(d, [[arg.name]]);
    {% endfor %}
    TLLM_CHECK(d == a + getSerializationSize());

}

void [[ plugin_name ]]::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void [[ plugin_name ]]::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* [[ plugin_name ]]::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}



[[ plugin_name ]]Creator::[[ plugin_name ]]Creator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();

    {% for arg in params %}
    mPluginAttributes.emplace_back(PluginField("[[arg.name]]", nullptr, PluginFieldType::[[arg.dtype.dtype.to('trt_plugin')]]));
    {% endfor %}

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}


const char* [[ plugin_name ]]Creator::getPluginName() const noexcept
{
    return "[[ plugin_name ]]";
}

const char* [[ plugin_name ]]Creator::getPluginVersion() const noexcept
{
    return "[[ plugin_version ]]";
}

const PluginFieldCollection* [[ plugin_name ]]Creator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* [[ plugin_name ]]Creator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
  const PluginField* fields = fc->fields;

  // declare parameters
  {% for arg in params %}
    [[arg.dtype.dtype.to('c')]] [[arg.name]];
  {% endfor %}

    for (int i = 0; i < fc->nbFields; ++i) {
        const char* attrName = fields[i].name;
  {% for arg in params %}
        if (!strcmp(attrName, "[[arg.name]]"))
        {
            PLUGIN_ASSERT(fields[i].type == PluginFieldType::[[arg.dtype.dtype.to('trt_plugin')]]);
            [[arg.name]] = static_cast<[[arg.dtype.dtype.to('c')]]>(*(static_cast<const [[arg.dtype.dtype.to('c')]]*>(fields[i].data)));
        }
  {% endfor %}
    }

    try
    {
        auto* obj = new [[plugin_name]]([[ ', '.join(param_names) ]]);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;

}


IPluginV2* [[ plugin_name ]]Creator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call [[ plugin_name ]]::destroy()
    try
    {
        auto* obj = new [[ plugin_name ]](serialData, serialLength);
        obj->setPluginNamespace("tensorrt_llm");
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void [[plugin_name]]Creator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* [[plugin_name]]Creator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
