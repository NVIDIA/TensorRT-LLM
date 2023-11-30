/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "generationInput.h"

#include "tensorrt_llm/runtime/generationInput.h"
#include "tensorrt_llm/runtime/torchView.h"

namespace tr = tensorrt_llm::runtime;

using namespace tensorrt_llm::pybind::runtime;

std::shared_ptr<tr::PromptTuningParams> PromptTuningParams::toTrtLlm() const
{
    auto ptt = std::make_shared<tr::PromptTuningParams>();
    if (embeddingTable)
        ptt->embeddingTable = tr::TorchView::of(embeddingTable.value());
    if (tasks)
        ptt->tasks = tr::TorchView::of(tasks.value());
    if (vocabSize)
        ptt->vocabSize = tr::TorchView::of(vocabSize.value());
    ptt->promptTuningEnabled = promptTuningEnabled;
    return ptt;
}

std::shared_ptr<tr::GenerationInput> GenerationInput::toTrtLlm() const
{
    auto input = std::make_shared<tr::GenerationInput>(
        endId, padId, tr::TorchView::of(ids.value()), tr::TorchView::of(lengths.value()), packed);
    if (embeddingBias)
        input->embeddingBias = tr::TorchView::of(embeddingBias.value());
    if (badWordsList)
        input->badWordsList = tr::TorchView::of(badWordsList.value());
    if (stopWordsList)
        input->stopWordsList = tr::TorchView::of(stopWordsList.value());
    input->maxNewTokens = maxNewTokens;
    input->promptTuningParams = *promptTuningParams.toTrtLlm();
    return input;

    return input;
}
