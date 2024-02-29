/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include <vector>

namespace tensorrt_llm::runtime
{

class LoraModule
{
public:
    using TensorPtr = ITensor::SharedPtr;

    enum class ModuleType : SizeType
    {
        kINVALID = -1,
        kATTN_QKV = 0,
        kATTN_Q = 1,
        kATTN_K = 2,
        kATTN_V = 3,
        kATTN_DENSE = 4,
        kMLP_H_TO_4H = 5,
        kMLP_4H_TO_H = 6,
        kMLP_GATE = 7,
    };

    explicit constexpr LoraModule(ModuleType const& t, SizeType inDim, SizeType outDim, bool inDimFirst,
        bool outDimFirst, SizeType inTpSplitDim, SizeType outTpSplitDim) noexcept
        : mType(t)
        , mInDim(inDim)
        , mOutDim(outDim)
        , mInDimFirst(inDimFirst)
        , mOutDimFirst(outDimFirst)
        , mInTpSplitDim(inTpSplitDim)
        , mOutTpSplitDim(outTpSplitDim)
    {
    }

    explicit constexpr LoraModule() noexcept
        : LoraModule(ModuleType::kATTN_QKV, 0, 0, false, true, -1, -1)
    {
    }

    explicit constexpr LoraModule(LoraModule const& o) = default;
    constexpr LoraModule& operator=(LoraModule const& o) = default;

    [[nodiscard]] SizeType constexpr flattenedInOutSize(SizeType adapterSize) const noexcept
    {
        return adapterSize * (mInDim + mOutDim);
    }

    [[nodiscard]] SizeType constexpr value() const noexcept
    {
        return static_cast<SizeType>(mType);
    }

    [[nodiscard]] std::string_view constexpr name() const noexcept
    {
        return toModuleName(mType);
    }

    [[nodiscard]] SizeType constexpr inDim() const noexcept
    {
        return mInDim;
    }

    [[nodiscard]] SizeType constexpr outDim() const noexcept
    {
        return mOutDim;
    }

    [[nodiscard]] bool constexpr inDimFirst() const noexcept
    {
        return mInDimFirst;
    }

    [[nodiscard]] bool constexpr outDimFirst() const noexcept
    {
        return mOutDimFirst;
    }

    [[nodiscard]] SizeType constexpr inTpSplitDim() const noexcept
    {
        return mInTpSplitDim;
    }

    [[nodiscard]] SizeType constexpr outTpSplitDim() const noexcept
    {
        return mOutTpSplitDim;
    }

    static std::vector<LoraModule> createLoraModules(std::vector<std::string> const& loraModuleNames,
        SizeType hiddenSize, SizeType mlpHiddenSize, SizeType numAttentionHeads, SizeType numKvAttentionHeads,
        SizeType attentionHeadSize, SizeType tpSize);

    static ModuleType constexpr toModuleType(std::string_view const& name)
    {
        if (name == "attn_qkv")
            return ModuleType::kATTN_QKV;
        else if (name == "attn_q")
            return ModuleType::kATTN_Q;
        else if (name == "attn_k")
            return ModuleType::kATTN_K;
        else if (name == "attn_v")
            return ModuleType::kATTN_V;
        else if (name == "attn_dense")
            return ModuleType::kATTN_DENSE;
        else if (name == "mlp_h_to_4h")
            return ModuleType::kMLP_H_TO_4H;
        else if (name == "mlp_4h_to_h")
            return ModuleType::kMLP_4H_TO_H;
        else if (name == "mlp_gate")
            return ModuleType::kMLP_GATE;
        else
            return ModuleType::kINVALID;
    }

    static std::string_view constexpr toModuleName(ModuleType t) noexcept
    {
        switch (t)
        {
        case ModuleType::kATTN_QKV: return "attn_qkv";
        case ModuleType::kATTN_Q: return "attn_q";
        case ModuleType::kATTN_K: return "attn_k";
        case ModuleType::kATTN_V: return "attn_v";
        case ModuleType::kATTN_DENSE: return "attn_dense";
        case ModuleType::kMLP_H_TO_4H: return "mlp_h_to_4h";
        case ModuleType::kMLP_4H_TO_H: return "mlp_4h_to_h";
        case ModuleType::kMLP_GATE: return "mlp_gate";
        case ModuleType::kINVALID: return "INVALID";
        }
        return "INVALID";
    }

    static std::string_view constexpr toModuleName(SizeType id)
    {
        auto t = LoraModule::ModuleType(id);
        return toModuleName(t);
    }

private:
    ModuleType mType;
    SizeType mInDim;
    SizeType mOutDim;
    bool mInDimFirst;
    bool mOutDimFirst;
    SizeType mInTpSplitDim;
    SizeType mOutTpSplitDim;
};

inline std::ostream& operator<<(std::ostream& output, LoraModule const& module)
{
    return output << "LoraModule(id=" << module.value() << ", "
                  << "name=" << module.name() << ", "
                  << "inDim=" << module.inDim() << ", "
                  << "outDim=" << module.outDim() << ", "
                  << "inTpSplitDim=" << module.inTpSplitDim() << ", "
                  << "outTpSplitDim=" << module.outTpSplitDim() << ")";
}
} // namespace tensorrt_llm::runtime
