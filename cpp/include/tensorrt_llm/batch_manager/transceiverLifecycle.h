/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace tensorrt_llm::batch_manager
{

enum class LogicalDisposition : std::uint8_t
{
    kACCEPTED,
    kALREADY_TERMINAL,
    kNOT_FOUND,
    kREJECTED,
};

enum class PhysicalDisposition : std::uint8_t
{
    kNOT_EXPOSED,
    kACTIVE,
    kQUIESCING,
    kQUIESCED_SUCCESS,
    kQUIESCED_FAILURE,
    kIN_DOUBT,
};

[[nodiscard]] constexpr bool isReusable(PhysicalDisposition const disposition)
{
    return disposition == PhysicalDisposition::kNOT_EXPOSED || disposition == PhysicalDisposition::kQUIESCED_SUCCESS
        || disposition == PhysicalDisposition::kQUIESCED_FAILURE;
}

struct TransceiverCapabilities
{
    std::uint32_t protocolVersion{0};
    bool qualifiedLegacyMode{false};
    bool attemptIdentity{false};
    bool endpointIncarnation{false};
    bool allocationGenerationLeases{false};
    bool cancelBeforeCreateTombstones{false};
    bool publicationGate{false};
    bool inFlightCancellation{false};
    bool exactWriterTracking{false};
    bool submissionFence{false};
    bool perOperationQuiescence{false};
    bool endpointWideQuiescence{false};
    bool directTransfer{false};
    bool bounceTransfer{false};
    bool multiWriter{false};
    bool generationFirst{false};
    bool pipelineParallel{false};
    bool tensorParallel{false};
    bool attentionDataParallel{false};
    bool terminalResultReplay{false};
};

struct CancelResult
{
    LogicalDisposition logical{LogicalDisposition::kREJECTED};
    PhysicalDisposition physical{PhysicalDisposition::kIN_DOUBT};
    bool retryable{false};
    std::string reason;

    [[nodiscard]] bool safeToReuse() const
    {
        return isReusable(physical);
    }
};

struct ShutdownResult
{
    PhysicalDisposition physical{PhysicalDisposition::kIN_DOUBT};
    std::optional<std::uint64_t> inDoubtContextCount{std::nullopt};
    bool fatal{false};
    std::string reason;

    [[nodiscard]] bool safeToReleaseManagers() const
    {
        return isReusable(physical) && inDoubtContextCount == 0 && !fatal;
    }
};

} // namespace tensorrt_llm::batch_manager
