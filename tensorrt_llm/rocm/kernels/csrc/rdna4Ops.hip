/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
#include "ops.h"
#include "wave32.h"

#include <ATen/Dispatch.h>
#include <ATen/core/grad_mode.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace tensorrt_llm::rdna4
{
namespace
{
constexpr int64_t kMaxBlocks = 65535;

void checkHip(hipError_t error)
{
    TORCH_CHECK(error == hipSuccess, "RDNA4 HIP error: ", hipGetErrorString(error));
}

void checkTensor(at::Tensor const& input)
{
    TORCH_CHECK(input.is_cuda(), "Expected a PyTorch HIP tensor");
    TORCH_CHECK(input.is_contiguous(), "RDNA4 kernels require contiguous tensors");
    TORCH_CHECK(input.dim() > 0 && input.size(-1) > 0, "Expected a nonempty final dimension");
    TORCH_CHECK(
        input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
        "Only FP32, FP16 and BF16 storage is supported");
    TORCH_CHECK(!at::GradMode::is_enabled() || !input.requires_grad(), "RDNA4 kernels are inference-only");
}

void checkLike(at::Tensor const& input, at::Tensor const& other)
{
    checkTensor(other);
    TORCH_CHECK(input.device() == other.device() && input.scalar_type() == other.scalar_type(),
        "Input tensors must share device and dtype");
}

void checkWeight(at::Tensor const& input, at::Tensor const& weight, double epsilon)
{
    checkLike(input, weight);
    TORCH_CHECK(weight.dim() == 1 && weight.numel() == input.size(-1), "Invalid normalization weight");
    TORCH_CHECK(
        static_cast<float>(epsilon) > 0 && std::isfinite(static_cast<float>(epsilon)), "Invalid normalization epsilon");
}

hipStream_t currentStream(at::Tensor const& input)
{
    // Obtain PyTorch's current stream, not HIP's default stream. PyTorch exposes
    // HIP tensors under DeviceType::CUDA and supplies the corresponding guard.
    c10::impl::VirtualGuardImpl const implementation(input.device().type());
    c10::hip::HIPStreamMasqueradingAsCUDA const stream(implementation.getStream(input.device()));
    hipDeviceProp_t properties{};
    checkHip(hipGetDeviceProperties(&properties, input.get_device()));
    std::string const target(properties.gcnArchName);
    std::string const architecture = target.substr(0, target.find(':'));
    TORCH_CHECK(architecture == "gfx1200" || architecture == "gfx1201", "Native kernels require RDNA4");
    TORCH_CHECK(properties.warpSize == kWaveSize, "Native kernels require wave32");
    return stream.stream();
}

unsigned int rowBlocks(int64_t rows)
{
    return static_cast<unsigned int>(std::min(rows, kMaxBlocks));
}

unsigned int linearBlocks(int64_t elements)
{
    return rowBlocks(1 + (elements - 1) / kBlockSize);
}

template <typename Scalar, bool AddResidual, bool Centered>
__global__ void normKernel(Scalar const* input, Scalar const* residual, Scalar const* weight, Scalar const* bias,
    Scalar* output, Scalar* updated, int64_t rows, int64_t width, float epsilon)
{
    __shared__ float scratch[kWavesPerBlock];
    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x)
    {
        int64_t const offset = row * width;
        float sum = 0.0F;
        for (int64_t column = threadIdx.x; column < width; column += blockDim.x)
        {
            Scalar stored = input[offset + column];
            if constexpr (AddResidual)
            {
                stored
                    = static_cast<Scalar>(static_cast<float>(stored) + static_cast<float>(residual[offset + column]));
                updated[offset + column] = stored;
            }
            float const value = static_cast<float>(stored);
            sum += Centered ? value : value * value;
        }
        float const total = blockSum(sum, scratch);
        float const mean = Centered ? total / static_cast<float>(width) : 0.0F;
        float variance = total / static_cast<float>(width);
        if constexpr (Centered)
        {
            sum = 0.0F;
            for (int64_t column = threadIdx.x; column < width; column += blockDim.x)
            {
                Scalar const stored = AddResidual ? updated[offset + column] : input[offset + column];
                float const centered = static_cast<float>(stored) - mean;
                sum += centered * centered;
            }
            variance = blockSum(sum, scratch) / static_cast<float>(width);
        }
        float const inverse = rsqrtf(variance + epsilon);
        for (int64_t column = threadIdx.x; column < width; column += blockDim.x)
        {
            Scalar const stored = AddResidual ? updated[offset + column] : input[offset + column];
            float normalized = (static_cast<float>(stored) - mean) * inverse;
            // HF RMSNorm rounds the normalized value before applying the weight.
            if constexpr (!Centered)
            {
                normalized = static_cast<float>(static_cast<Scalar>(normalized));
            }
            float result = normalized * static_cast<float>(weight[column]);
            if constexpr (Centered)
            {
                if (bias != nullptr)
                {
                    result += static_cast<float>(bias[column]);
                }
            }
            output[offset + column] = static_cast<Scalar>(result);
        }
    }
}

template <typename Scalar>
__global__ void activationKernel(Scalar const* gate, Scalar const* up, Scalar* output, int64_t elements, int activation)
{
    int64_t const stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < elements; index += stride)
    {
        float const value = static_cast<float>(gate[index]);
        float activated;
        if (activation == 0)
        {
            activated = value / (1.0F + expf(-value));
        }
        else
        {
            constexpr float kSqrtTwoOverPi = 0.7978845608028654F;
            constexpr float kGeluCubic = 0.044715F;
            activated = 0.5F * value * (1.0F + tanhf(kSqrtTwoOverPi * (value + kGeluCubic * value * value * value)));
        }
        output[index] = static_cast<Scalar>(activated * static_cast<float>(up[index]));
    }
}

template <typename Scalar>
__global__ void rotaryKernel(Scalar const* input, float const* cos, float const* sin, Scalar* output, int64_t elements,
    int64_t heads, int64_t dim, int64_t pairs, bool interleaved)
{
    int64_t const stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < elements; index += stride)
    {
        int64_t const column = index % dim;
        if (column >= 2 * pairs)
        {
            output[index] = input[index];
            continue;
        }
        int64_t const token = index / (heads * dim);
        bool const second = interleaved ? (column % 2 != 0) : (column >= pairs);
        int64_t const pair = interleaved ? column / 2 : column % pairs;
        int64_t const partner
            = interleaved ? (second ? column - 1 : column + 1) : (second ? column - pairs : column + pairs);
        float const value = static_cast<float>(input[index]);
        float const other = static_cast<float>(input[index - column + partner]);
        float const c = cos[token * pairs + pair];
        float const s = sin[token * pairs + pair];
        output[index] = static_cast<Scalar>(value * c + (second ? other : -other) * s);
    }
}

template <typename Scalar>
__global__ void attentionKernel(Scalar const* query, Scalar const* key, Scalar const* value, float const* bias,
    Scalar* output, int64_t rows, int64_t queryHeads, int64_t kvHeads, int64_t queries, int64_t keys, int64_t dim,
    bool causal, int64_t queryStart, float scale)
{
    __shared__ float scratch[kWavesPerBlock];
    int64_t const column = threadIdx.x;
    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x)
    {
        int64_t const queryIndex = row % queries;
        int64_t const head = (row / queries) % queryHeads;
        int64_t const batch = row / (queries * queryHeads);
        int64_t const kvHead = head / (queryHeads / kvHeads);
        int64_t const queryOffset = row * dim;
        int64_t const kvOffset = (batch * kvHeads + kvHead) * keys * dim;
        float accumulator = 0.0F;
        float maximum = -INFINITY;
        float denominator = 0.0F;
        for (int64_t keyIndex = 0; keyIndex < keys; ++keyIndex)
        {
            // These branches are uniform across the complete work-group.
            if (causal && keyIndex > queryStart + queryIndex)
            {
                continue;
            }
            float product = 0.0F;
            if (column < dim)
            {
                product = static_cast<float>(query[queryOffset + column])
                    * static_cast<float>(key[kvOffset + keyIndex * dim + column]);
            }
            float score = blockSum(product, scratch) * scale;
            if (bias != nullptr)
            {
                score += bias[row * keys + keyIndex];
            }
            if (score == -INFINITY)
            {
                continue;
            }
            float const nextMaximum = fmaxf(maximum, score);
            float const previousScale = expf(maximum - nextMaximum);
            float const probability = expf(score - nextMaximum);
            denominator = denominator * previousScale + probability;
            if (column < dim)
            {
                accumulator = accumulator * previousScale
                    + probability * static_cast<float>(value[kvOffset + keyIndex * dim + column]);
            }
            maximum = nextMaximum;
        }
        if (column < dim)
        {
            output[queryOffset + column] = static_cast<Scalar>(denominator == 0.0F ? 0.0F : accumulator / denominator);
        }
    }
}
} // namespace

at::Tensor rmsNorm(at::Tensor const& input, at::Tensor const& weight, double epsilon)
{
    checkTensor(input);
    checkWeight(input, weight, epsilon);
    c10::DeviceGuard const guard(input.device());
    at::Tensor output = at::empty_like(input);
    int64_t const width = input.size(-1);
    int64_t const rows = input.numel() / width;
    if (rows == 0)
    {
        return output;
    }
    hipStream_t const stream = currentStream(input);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "rdna4_rms_norm",
        [&]
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(normKernel<scalar_t, false, false>), dim3(rowBlocks(rows)),
                dim3(kBlockSize), 0, stream, input.data_ptr<scalar_t>(), nullptr, weight.data_ptr<scalar_t>(), nullptr,
                output.data_ptr<scalar_t>(), nullptr, rows, width, static_cast<float>(epsilon));
        });
    checkHip(hipGetLastError());
    return output;
}

std::tuple<at::Tensor, at::Tensor> addRmsNorm(
    at::Tensor const& input, at::Tensor const& residual, at::Tensor const& weight, double epsilon)
{
    checkTensor(input);
    checkLike(input, residual);
    checkWeight(input, weight, epsilon);
    TORCH_CHECK(input.sizes() == residual.sizes(), "Residual shape mismatch");
    c10::DeviceGuard const guard(input.device());
    at::Tensor output = at::empty_like(input);
    at::Tensor updated = at::empty_like(input);
    int64_t const width = input.size(-1);
    int64_t const rows = input.numel() / width;
    if (rows == 0)
    {
        return {output, updated};
    }
    hipStream_t const stream = currentStream(input);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "rdna4_add_rms_norm",
        [&]
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(normKernel<scalar_t, true, false>), dim3(rowBlocks(rows)),
                dim3(kBlockSize), 0, stream, input.data_ptr<scalar_t>(), residual.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(), nullptr, output.data_ptr<scalar_t>(), updated.data_ptr<scalar_t>(), rows,
                width, static_cast<float>(epsilon));
        });
    checkHip(hipGetLastError());
    return {output, updated};
}

at::Tensor layerNorm(
    at::Tensor const& input, at::Tensor const& weight, std::optional<at::Tensor> const& bias, double epsilon)
{
    checkTensor(input);
    checkWeight(input, weight, epsilon);
    if (bias.has_value())
    {
        checkLike(input, *bias);
        TORCH_CHECK(bias->sizes() == weight.sizes(), "Bias shape mismatch");
    }
    c10::DeviceGuard const guard(input.device());
    at::Tensor output = at::empty_like(input);
    int64_t const width = input.size(-1);
    int64_t const rows = input.numel() / width;
    if (rows == 0)
    {
        return output;
    }
    hipStream_t const stream = currentStream(input);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "rdna4_layer_norm",
        [&]
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(normKernel<scalar_t, false, true>), dim3(rowBlocks(rows)),
                dim3(kBlockSize), 0, stream, input.data_ptr<scalar_t>(), nullptr, weight.data_ptr<scalar_t>(),
                bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr, output.data_ptr<scalar_t>(), nullptr, rows,
                width, static_cast<float>(epsilon));
        });
    checkHip(hipGetLastError());
    return output;
}

at::Tensor gatedActivation(at::Tensor const& gate, at::Tensor const& up, int64_t activation)
{
    checkTensor(gate);
    checkLike(gate, up);
    TORCH_CHECK(gate.sizes() == up.sizes(), "Gate/up shape mismatch");
    TORCH_CHECK(activation == 0 || activation == 1, "Invalid activation");
    c10::DeviceGuard const guard(gate.device());
    at::Tensor output = at::empty_like(gate);
    if (gate.numel() == 0)
    {
        return output;
    }
    hipStream_t const stream = currentStream(gate);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, gate.scalar_type(), "rdna4_gated_activation",
        [&]
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(activationKernel<scalar_t>), dim3(linearBlocks(gate.numel())),
                dim3(kBlockSize), 0, stream, gate.data_ptr<scalar_t>(), up.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(), gate.numel(), static_cast<int>(activation));
        });
    checkHip(hipGetLastError());
    return output;
}

at::Tensor rotary(at::Tensor const& input, at::Tensor const& cos, at::Tensor const& sin, bool interleaved)
{
    checkTensor(input);
    TORCH_CHECK(input.dim() == 3 && cos.dim() == 2 && sin.sizes() == cos.sizes(), "Invalid rotary shape");
    TORCH_CHECK(cos.is_contiguous() && sin.is_contiguous() && cos.scalar_type() == at::kFloat
            && sin.scalar_type() == at::kFloat && cos.device() == input.device() && sin.device() == input.device(),
        "Angles must be contiguous FP32 tensors on the input device");
    TORCH_CHECK(cos.size(0) == input.size(0) && cos.size(1) > 0 && cos.size(1) <= input.size(2) / 2,
        "Invalid rotary dimension");
    checkTensor(cos);
    checkTensor(sin);
    c10::DeviceGuard const guard(input.device());
    at::Tensor output = at::empty_like(input);
    if (input.numel() == 0)
    {
        return output;
    }
    hipStream_t const stream = currentStream(input);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "rdna4_rotary",
        [&]
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(rotaryKernel<scalar_t>), dim3(linearBlocks(input.numel())),
                dim3(kBlockSize), 0, stream, input.data_ptr<scalar_t>(), cos.data_ptr<float>(), sin.data_ptr<float>(),
                output.data_ptr<scalar_t>(), input.numel(), input.size(1), input.size(2), cos.size(1), interleaved);
        });
    checkHip(hipGetLastError());
    return output;
}

at::Tensor attention(at::Tensor const& query, at::Tensor const& key, at::Tensor const& value,
    std::optional<at::Tensor> const& bias, bool causal, int64_t queryStart, double scale)
{
    checkTensor(query);
    checkLike(query, key);
    checkLike(query, value);
    TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.sizes() == key.sizes(), "Invalid attention shape");
    TORCH_CHECK(key.size(0) == query.size(0) && key.size(3) == query.size(3) && key.size(1) > 0 && query.size(1) > 0
            && query.size(1) % key.size(1) == 0 && key.size(2) > 0,
        "Invalid attention batch/head geometry");
    TORCH_CHECK(query.size(3) <= kBlockSize && queryStart >= 0
            && queryStart <= std::numeric_limits<int64_t>::max() - query.size(2)
            && std::isfinite(static_cast<float>(scale)),
        "Invalid head dimension, query position or scale");
    if (bias.has_value())
    {
        checkTensor(*bias);
        TORCH_CHECK(bias->is_contiguous() && bias->device() == query.device() && bias->scalar_type() == at::kFloat,
            "Attention bias must be contiguous FP32 on the query device");
        TORCH_CHECK(bias->dim() == 4 && bias->size(0) == query.size(0) && bias->size(1) == query.size(1)
                && bias->size(2) == query.size(2) && bias->size(3) == key.size(2),
            "Invalid attention bias shape");
    }
    c10::DeviceGuard const guard(query.device());
    at::Tensor output = at::empty_like(query);
    int64_t const rows = query.numel() / query.size(3);
    if (rows == 0)
    {
        return output;
    }
    hipStream_t const stream = currentStream(query);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, query.scalar_type(), "rdna4_attention",
        [&]
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(attentionKernel<scalar_t>), dim3(rowBlocks(rows)), dim3(kBlockSize), 0,
                stream, query.data_ptr<scalar_t>(), key.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(),
                bias.has_value() ? bias->data_ptr<float>() : nullptr, output.data_ptr<scalar_t>(), rows, query.size(1),
                key.size(1), query.size(2), key.size(2), query.size(3), causal, queryStart, static_cast<float>(scale));
        });
    checkHip(hipGetLastError());
    return output;
}
} // namespace tensorrt_llm::rdna4
