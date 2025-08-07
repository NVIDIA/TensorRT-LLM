/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/thop/fp8Op.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace th = torch;

namespace torch_ext
{

static int getExp(float v)
{
    int vIntRepr;
    memcpy(&vIntRepr, &v, sizeof(vIntRepr));
    int expBits = (vIntRepr >> 23) & 0xff;
    return expBits - 127;
}

static int getMantissaBits(float v)
{
    int vIntRepr;
    memcpy(&vIntRepr, &v, sizeof(vIntRepr));
    int mantissaBits = vIntRepr & 0x7fffff;
    return mantissaBits;
}

static bool getSign(float v)
{
    int vIntRepr;
    memcpy(&vIntRepr, &v, sizeof(vIntRepr));
    return vIntRepr >> 31;
}

static float makeExpFloat(int expValue)
{
    expValue += 127;
    expValue <<= 23;
    float vFloat;
    memcpy(&vFloat, &expValue, sizeof(vFloat));
    return vFloat;
}

/*
 * E2M1 to float
 * 0111 -> 6
 * 0110 -> 4
 * 0101 -> 3
 * 0100 -> 2
 * 0011 -> 1.5
 * 0010 -> 1
 * 0001 -> 0.5
 * 0000 -> 0
 */
static float const kE2M1ToFloatArray[] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
static float const kE2M1Array[] = {0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5};
static int const kE2M1Count = 8;

uint8_t floatToE2M1(float value)
{
    float absValue = fabs(value);
    TORCH_CHECK_LT(absValue, 8.0f);
    uint8_t result = getSign(value) ? 8 : 0;
    int fp4AbsValue = kE2M1Count - 1;
    for (; fp4AbsValue > 0; --fp4AbsValue)
    {
        if (kE2M1Array[fp4AbsValue] < absValue)
            break;
        // Tie to even.
        if (kE2M1Array[fp4AbsValue] == absValue && !(fp4AbsValue & 1))
            break;
    }
    result |= fp4AbsValue;
    return result;
}

float e2M1ToFloat(uint8_t value)
{
    bool signBit = value & 8;
    auto absValue = value & 7;
    float result = kE2M1ToFloatArray[absValue];
    if (signBit)
        result = -result;
    return result;
}

torch::autograd::variable_list FloatToE2M1AndUFP8SFScale(
    th::Tensor floatTensor, int64_t sfVecSize, int64_t sfType, torch::optional<bool> isSfSwizzledLayout)
{
    CHECK_CPU_INPUT(floatTensor, th::kFloat32);
    auto inputShape = floatTensor.sizes();
    TORCH_CHECK(inputShape.size() == 2, "Input should be 2D tensor.");
    TORCH_CHECK(inputShape[1] % sfVecSize == 0);
    th::Tensor valueE2M1 = th::zeros({inputShape[0], inputShape[1] / 2}, th::dtype(FLOAT4_E2M1X2).requires_grad(false));
    th::Tensor scaleFP8SF
        = th::zeros({tensorrt_llm::computeSwizzledLayoutSFSize(inputShape[0], inputShape[1] / sfVecSize)},
            th::dtype(SF_DTYPE).requires_grad(false));
    th::Tensor repFloat = th::zeros(inputShape, th::dtype(th::kFloat32).requires_grad(false));

    int hiddenDim = inputShape[1];
    int packedFp4HiddenDim = hiddenDim / 2;
    int groupsPerHiddenDim = hiddenDim / sfVecSize;

    // Note: if isSfSwizzledLayout is provided, use its value; otherwise default to true.
    tensorrt_llm::QuantizationSFLayout layout = isSfSwizzledLayout.value_or(true)
        ? tensorrt_llm::QuantizationSFLayout::SWIZZLED
        : tensorrt_llm::QuantizationSFLayout::LINEAR;

    for (size_t vIdx = 0; vIdx < static_cast<size_t>(inputShape[0]); ++vIdx)
    {
        for (int group = 0; group < groupsPerHiddenDim; ++group)
        {
            float const* inputPtr = floatTensor.data_ptr<float>() + vIdx * hiddenDim + group * sfVecSize;
            float* repPtr = repFloat.data_ptr<float>() + vIdx * hiddenDim + group * sfVecSize;
            uint8_t* packedFp4Ptr = valueE2M1.data_ptr<uint8_t>() + vIdx * packedFp4HiddenDim + group * sfVecSize / 2;
            int8_t* scaleFP8SFPtr = static_cast<int8_t*>(scaleFP8SF.data_ptr());

            float maxAbsValue = 0.0f;
            for (int i = 0; i < sfVecSize; ++i)
            {
                maxAbsValue = std::max(maxAbsValue, fabs(inputPtr[i]));
            }
            int scaleExp = getExp(maxAbsValue);
            scaleExp -= 2;
            if (sfType == 0)
            {
                scaleExp = std::max(scaleExp, -126);
                int e8M0Scale = scaleExp + 127;
                TORCH_CHECK_GT(e8M0Scale, 0);
                TORCH_CHECK_LT(e8M0Scale, 255);
                int8_t e8M0ScaleOut = e8M0Scale & 0xff;
                scaleFP8SFPtr[computeSFIndex(vIdx, group, inputShape[0], groupsPerHiddenDim, layout)] = e8M0ScaleOut;
            }
            else
            {
                scaleExp = std::max(scaleExp, -6);
                int e4M3Scale = scaleExp + 7;
                TORCH_CHECK_GT(e4M3Scale, 0);
                TORCH_CHECK_LT(e4M3Scale, 15);
                int8_t e4M3ScaleOut = (e4M3Scale & 0xff) << 3;
                scaleFP8SFPtr[computeSFIndex(vIdx, group, inputShape[0], groupsPerHiddenDim, layout)] = e4M3ScaleOut;
            }
            float scaleFloat = makeExpFloat(scaleExp);
            float invScaleFloat = 1.0 / scaleFloat;
            // printf("vIdx=%ld, group=%d, maxAbsValue=%f, scaleExp=%d, e8M0Scale=%d, scaleFloat=%f,
            // invScaleFloat=%f\n",
            //        vIdx, group, maxAbsValue, scaleExp, e8M0Scale, scaleFloat, invScaleFloat);
            for (int i = 0; i < sfVecSize; ++i)
            {
                float value = inputPtr[i];
                float scaledValue = invScaleFloat * value;
                uint8_t fp4Value = floatToE2M1(scaledValue);
                float e2M1FloatValue = e2M1ToFloat(fp4Value);
                float repResult = e2M1FloatValue * scaleFloat;
                repPtr[i] = repResult;
                uint8_t packedValue = packedFp4Ptr[i / 2];
                if (i % 2 == 0)
                {
                    packedValue &= 0xf0;
                    packedValue |= fp4Value;
                }
                else
                {
                    packedValue &= 0x0f;
                    packedValue |= (fp4Value << 4);
                }
                packedFp4Ptr[i / 2] = packedValue;
                // printf("  i=%d, value=%f, scaledValue=%f, fp4Value=%x, e2M1FloatValue=%f, repResult=%f\n",
                //        i, value, scaledValue, (int)fp4Value, e2M1FloatValue, repResult);
            }
        }
    }

    return {valueE2M1, scaleFP8SF, repFloat};
}

// Preprocess the weights.
torch::autograd::variable_list HalfToE2M1AndUFP8SFScale(
    th::Tensor halfTensor, th::Tensor globalScale, int64_t sfVecSize, int64_t sfType)
{
    CHECK_INPUT(halfTensor, th::kFloat16);
    CHECK_INPUT(globalScale, th::kFloat32);
    auto inputShape = halfTensor.sizes();
    TORCH_CHECK(inputShape.size() == 2 || inputShape.size() == 3, "Input should be 2D or 3D tensor.");
    bool has_experts = inputShape.size() == 3;
    auto num_experts = has_experts ? inputShape[0] : 1;
    auto rows = has_experts ? inputShape[1] : inputShape[0];
    auto cols = has_experts ? inputShape[2] : inputShape[1];

    auto const expert_sf_size = tensorrt_llm::computeSwizzledLayoutSFSize(rows, cols / sfVecSize);

    TORCH_CHECK(cols % sfVecSize == 0);
    std::array<int64_t, 3> shape{num_experts, rows, cols / 2};
    th::IntArrayRef shape_ref(shape.data() + !has_experts, shape.size() - !has_experts);
    th::Tensor valueE2M1 = th::zeros(shape_ref, th::dtype(FLOAT4_E2M1X2).device(torch::kCUDA).requires_grad(false));
    th::Tensor scaleFP8SF
        = th::zeros({num_experts * expert_sf_size}, th::dtype(SF_DTYPE).device(torch::kCUDA).requires_grad(false));

    int const mMultiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();
    for (size_t eIdx = 0; eIdx < static_cast<size_t>(num_experts); eIdx++)
    {
        size_t const expert_elem_offset = rows * cols * eIdx;
        size_t const expert_sf_offset = expert_sf_size * eIdx;
        constexpr int FP4_PER_INT64 = 16;
        constexpr int FP8_PER_INT32 = 4;
        tensorrt_llm::kernels::invokeFP4Quantization(1, rows, cols,
            reinterpret_cast<half*>(halfTensor.data_ptr()) + expert_elem_offset, globalScale.data_ptr<float>() + eIdx,
            reinterpret_cast<int64_t*>(valueE2M1.data_ptr()) + expert_elem_offset / FP4_PER_INT64,
            reinterpret_cast<int32_t*>(scaleFP8SF.data_ptr()) + expert_sf_offset / FP8_PER_INT32, sfType == 0,
            tensorrt_llm::QuantizationSFLayout::SWIZZLED, mMultiProcessorCount, 0);
    }

    return {valueE2M1, scaleFP8SF};
}

// Interleave (and possibly pad) the weights block scaling factor.
// blockScale: [num_experts, rows, cols] or [rows, cols]
// Return: num_experts * pad_up(rows, 128) * pad_up(cols, 4)
th::Tensor BlockScaleInterleave(th::Tensor const& blockScale)
{
    bool is_cuda = blockScale.device().is_cuda();
    if (is_cuda)
    {
        CHECK_INPUT(blockScale, SF_DTYPE);
    }
    else
    {
        CHECK_CPU_INPUT(blockScale, SF_DTYPE);
    }
    auto blockScaleShape = blockScale.sizes();
    TORCH_CHECK(blockScaleShape.size() == 2 || blockScaleShape.size() == 3, "Block Scale should be 2D or 3D tensor.");
    auto num_experts = blockScaleShape.size() == 3 ? blockScaleShape[0] : 1;
    auto rows = blockScaleShape.size() == 3 ? blockScaleShape[1] : blockScaleShape[0];
    auto cols = blockScaleShape.size() == 3 ? blockScaleShape[2] : blockScaleShape[1];

    auto expert_out_size = tensorrt_llm::computeSwizzledLayoutSFSize(rows, cols);
    auto rows_padded = PadUpFn(rows, 128);
    auto cols_padded = PadUpFn(cols, 4);
    TORCH_CHECK(
        expert_out_size == rows_padded * cols_padded, "expert_out_size should be equal to rows_padded * cols_padded.");
    th::Tensor interleavedBlockScale = th::empty(
        {expert_out_size * num_experts}, th::dtype(SF_DTYPE).device(blockScale.device()).requires_grad(false));

    if (is_cuda)
    {
        const thread_local int smCount = tensorrt_llm::common::getMultiProcessorCount();
        auto stream = at::cuda::getCurrentCUDAStream(blockScale.get_device());
        tensorrt_llm::kernels::invokeBlockScaleInterleave(num_experts, rows, rows_padded, cols, cols_padded,
            blockScale.data_ptr<uint8_t>(), static_cast<uint8_t*>(interleavedBlockScale.data_ptr()), smCount, stream);
    }
    else
    {
        for (int eIdx = 0; eIdx < static_cast<int>(num_experts); eIdx++)
        {
            uint8_t* interleavedBlockScalePtr
                = static_cast<uint8_t*>(interleavedBlockScale.data_ptr()) + eIdx * expert_out_size;
            for (int rIdx = 0; rIdx < static_cast<int>(rows_padded); ++rIdx)
            {
                auto globalRowIdx = eIdx * rows + rIdx;
                uint8_t* blockScalePtr = blockScale.data_ptr<uint8_t>() + globalRowIdx * cols;
                for (int cIdx = 0; cIdx < static_cast<int>(cols_padded); ++cIdx)
                {
                    uint8_t sf_ori = 0;
                    if (rIdx < static_cast<int>(rows) && cIdx < static_cast<int>(cols))
                    {
                        sf_ori = blockScalePtr[cIdx];
                    }
                    int sf_index = computeSFIndex(rIdx, cIdx, rows, cols, tensorrt_llm::QuantizationSFLayout::SWIZZLED);
                    interleavedBlockScalePtr[sf_index] = sf_ori;
                }
            }
        }
    }

    return interleavedBlockScale;
}

// Reverse interleave the weights block scaling factor.
// blockScale: [num_experts, rows, cols] or [rows, cols]
// Note: rows and cols are the dimensions of the original unswizzled SFMatrix, so reshape input before passing into
// this function! Return: The same shape as blockScale
th::Tensor BlockScaleInterleaveReverse(th::Tensor const& blockScale)
{
    bool is_cuda = blockScale.device().is_cuda();
    if (is_cuda)
    {
        CHECK_INPUT(blockScale, SF_DTYPE);
    }
    else
    {
        CHECK_CPU_INPUT(blockScale, SF_DTYPE);
    }
    auto blockScaleShape = blockScale.sizes();
    TORCH_CHECK(blockScaleShape.size() == 2 || blockScaleShape.size() == 3, "Block Scale should be 2D or 3D tensor.");
    auto num_experts = blockScaleShape.size() == 3 ? blockScaleShape[0] : 1;
    auto rows = blockScaleShape.size() == 3 ? blockScaleShape[1] : blockScaleShape[0];
    auto cols = blockScaleShape.size() == 3 ? blockScaleShape[2] : blockScaleShape[1];
    TORCH_CHECK(rows % 128 == 0, "rows of Interleaved block scales should be multiple of 128.");
    TORCH_CHECK(cols % 4 == 0, "cols of Interleaved block scales should be multiple of 4.");
    auto expert_out_size = rows * cols;
    th::Tensor reversedBlockScale
        = th::empty(blockScaleShape, th::dtype(SF_DTYPE).device(blockScale.device()).requires_grad(false));

    if (is_cuda)
    {
        const thread_local int smCount = tensorrt_llm::common::getMultiProcessorCount();
        auto stream = at::cuda::getCurrentCUDAStream(blockScale.get_device());
        tensorrt_llm::kernels::invokeBlockScaleInterleaveReverse(num_experts, rows, cols,
            blockScale.data_ptr<uint8_t>(), static_cast<uint8_t*>(reversedBlockScale.data_ptr()), smCount, stream);
    }
    else
    {
        // index in the swizzled SFMatrix -> (eIdx, rIdx, cIdx) in the unswizzled SFMatrix
        std::map<int, std::array<int, 3>> identity;
        for (int eIdx = 0; eIdx < num_experts; eIdx++)
        {
            for (int rIdx = 0; rIdx < rows; ++rIdx)
            {
                for (int cIdx = 0; cIdx < cols; ++cIdx)
                {
                    int sf_index = computeSFIndex(rIdx, cIdx, rows, cols, tensorrt_llm::QuantizationSFLayout::SWIZZLED);
                    identity[eIdx * expert_out_size + sf_index] = std::array<int, 3>{eIdx, rIdx, cIdx};
                }
            }
        }
        uint8_t* blockScalePtr = static_cast<uint8_t*>(blockScale.data_ptr());
        for (int i = 0; i < expert_out_size * num_experts; i++)
        {
            auto loc_2d = identity[i];
            if (loc_2d[1] < rows && loc_2d[2] < cols)
            {
                uint8_t* reversedBlockScalePtr
                    = reversedBlockScale.data_ptr<uint8_t>() + (loc_2d[0] * rows + loc_2d[1]) * cols + loc_2d[2];
                *reversedBlockScalePtr = blockScalePtr[i];
            }
        }
    }

    return reversedBlockScale;
}

th::Tensor E2M1AndUFP8SFScaleToFloat(th::Tensor valueE2M1, th::Tensor scaleFP8SF, int64_t sfVecSize, int64_t sfType)
{
    CHECK_CPU_INPUT(valueE2M1, FLOAT4_E2M1X2);
    CHECK_CPU_INPUT(scaleFP8SF, SF_DTYPE);
    auto packedShape = valueE2M1.sizes();
    auto scaleShape = scaleFP8SF.sizes();
    TORCH_CHECK(packedShape.size() == 2, "valueE2M1 should be 2D tensor.");
    TORCH_CHECK(scaleShape.size() == 1, "scaleFP8SF should be 1D tensor.");
    th::Tensor floatTensor
        = th::zeros({packedShape[0], packedShape[1] * 2}, th::dtype(th::kFloat32).requires_grad(false));

    int hiddenDim = packedShape[1] * 2;
    int packedFp4HiddenDim = hiddenDim / 2;
    int groupsPerHiddenDim = hiddenDim / sfVecSize;

    for (size_t vIdx = 0; vIdx < static_cast<size_t>(packedShape[0]); ++vIdx)
    {
        for (int group = 0; group < groupsPerHiddenDim; ++group)
        {
            float* floatPtr = floatTensor.data_ptr<float>() + vIdx * hiddenDim + group * sfVecSize;
            uint8_t* packedFp4Ptr = valueE2M1.data_ptr<uint8_t>() + vIdx * packedFp4HiddenDim + group * sfVecSize / 2;
            uint8_t* scaleFP8SFPtr = scaleFP8SF.data_ptr<uint8_t>();
            uint8_t fp8Scale = scaleFP8SFPtr[computeSFIndex(
                vIdx, group, packedShape[0], groupsPerHiddenDim, tensorrt_llm::QuantizationSFLayout::SWIZZLED)];
            int scale = fp8Scale;
            if (sfType == 0)
            {
                scale -= 127;
            }
            else
            {
                scale >>= 3;
                scale -= 7;
            }
            float scaleFloat = makeExpFloat(scale);
            for (int i = 0; i < sfVecSize; ++i)
            {
                uint8_t packedFp4 = packedFp4Ptr[i / 2];
                if (i % 2 == 1)
                {
                    packedFp4 >>= 4;
                }
                packedFp4 &= 0xf;
                float value = e2M1ToFloat(packedFp4) * scaleFloat;
                floatPtr[i] = value;
            }
        }
    }
    return floatTensor;
}

// Used by the (fp16 -> int4) quant layer + int4 gemm network.
th::Tensor E2M1AndUFP8SFScaleToFloatV2(th::Tensor valueE2M1, th::Tensor scaleFP8SF,
    std::optional<th::Tensor> globalScale, int64_t sfVecSize, int64_t sfType, bool isSfSwizzledLayout = true)
{
    CHECK_CPU_INPUT(valueE2M1, FLOAT4_E2M1X2);
    CHECK_CPU_INPUT(scaleFP8SF, SF_DTYPE);
    auto packedShape = valueE2M1.sizes();
    auto scaleShape = scaleFP8SF.sizes();
    TORCH_CHECK(packedShape.size() == 2, "valueE2M1 should be 2D tensor.");
    TORCH_CHECK(scaleShape.size() == 1, "scaleFP8SF should be 1D tensor.");
    th::Tensor floatTensor
        = th::zeros({packedShape[0], packedShape[1] * 2}, th::dtype(th::kFloat32).requires_grad(false));

    float globalScaleVal{1.0f};
    if (sfType == 1)
    {
        TORCH_CHECK(globalScale.has_value(), "globalScale is required when sfType is 1.");
        CHECK_CPU_INPUT(globalScale.value(), th::kFloat32);
        globalScaleVal = globalScale->data_ptr<float>()[0];
    }

    int hiddenDim = packedShape[1] * 2;
    int packedFp4HiddenDim = hiddenDim / 2;
    int groupsPerHiddenDim = hiddenDim / sfVecSize;

    tensorrt_llm::QuantizationSFLayout layout = isSfSwizzledLayout ? tensorrt_llm::QuantizationSFLayout::SWIZZLED
                                                                   : tensorrt_llm::QuantizationSFLayout::LINEAR;

    for (size_t vIdx = 0; vIdx < static_cast<size_t>(packedShape[0]); ++vIdx)
    {
        for (int group = 0; group < groupsPerHiddenDim; ++group)
        {
            float* floatPtr = floatTensor.data_ptr<float>() + vIdx * hiddenDim + group * sfVecSize;
            uint8_t* packedFp4Ptr = valueE2M1.data_ptr<uint8_t>() + vIdx * packedFp4HiddenDim + group * sfVecSize / 2;
            uint8_t* scaleFP8SFPtr = scaleFP8SF.data_ptr<uint8_t>();
            uint8_t fp8Scale = scaleFP8SFPtr[computeSFIndex(vIdx, group, packedShape[0], groupsPerHiddenDim, layout)];
            float scaleFloat;
            if (sfType == 0)
            {
                uint32_t tmp = uint32_t(fp8Scale) << 23;
                memcpy(&scaleFloat, &tmp, sizeof(scaleFloat));
            }
            else
            {
                scaleFloat = float(reinterpret_cast<__nv_fp8_e4m3&>(fp8Scale));
            }
            scaleFloat *= globalScaleVal;
            for (int i = 0; i < sfVecSize; ++i)
            {
                uint8_t packedFp4 = packedFp4Ptr[i / 2];
                if (i % 2 == 1)
                {
                    packedFp4 >>= 4;
                }
                packedFp4 &= 0xf;
                float value = e2M1ToFloat(packedFp4) * scaleFloat;
                floatPtr[i] = value;
            }
        }
    }
    return floatTensor;
}

} // namespace torch_ext

static auto float_to_e2m1_and_ufp8sf_scale
    = torch::RegisterOperators("tensorrt_llm::float_to_e2m1_and_ufp8sf_scale", &torch_ext::FloatToE2M1AndUFP8SFScale);

static auto half_to_e2m1_and_ufp8sf_scale
    = torch::RegisterOperators("tensorrt_llm::half_to_e2m1_and_ufp8sf_scale", &torch_ext::HalfToE2M1AndUFP8SFScale);

static auto e2m1_and_ufp8sf_scale_to_float
    = torch::RegisterOperators("tensorrt_llm::e2m1_and_ufp8sf_scale_to_float", &torch_ext::E2M1AndUFP8SFScaleToFloat);

static auto e2m1_and_ufp8sf_scale_to_float_v2 = torch::RegisterOperators(
    "tensorrt_llm::e2m1_and_ufp8sf_scale_to_float_v2", &torch_ext::E2M1AndUFP8SFScaleToFloatV2);

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("block_scale_interleave(Tensor input) -> Tensor");
    m.def("block_scale_interleave_reverse(Tensor input) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("block_scale_interleave", &torch_ext::BlockScaleInterleave);
    m.impl("block_scale_interleave_reverse", &torch_ext::BlockScaleInterleaveReverse);
}

TORCH_LIBRARY_IMPL(trtllm, CPU, m)
{
    m.impl("block_scale_interleave", &torch_ext::BlockScaleInterleave);
    m.impl("block_scale_interleave_reverse", &torch_ext::BlockScaleInterleaveReverse);
}
