/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/runtime/ncclCommunicator.h"
#include "tensorrt_llm/thop/thUtils.h"
#include <memory>

namespace th = torch;

namespace torch_ext
{

class NcclCommunicatorOp : public th::jit::CustomClassHolder
{
public:
    NcclCommunicatorOp(int64_t tpSize, int64_t ppSize, int64_t rank);

    void send(th::Tensor tensor, int64_t toRank) const;
    void recv(th::Tensor& tensor, int64_t fromRank) const;

private:
    int32_t mRank;
    std::shared_ptr<tensorrt_llm::runtime::NcclCommunicator> mPipelineComm;
};

} // namespace torch_ext
