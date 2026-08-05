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

#include "tensorrt_llm/batch_manager/transceiverLifecycle.h"

#include <utility>

namespace tensorrt_llm::batch_manager::lifecycle_detail
{

enum class LegacyCancelObservation
{
    kCANCELLED_WITHOUT_REUSE_PROOF,
    kCANCELLATION_REQUESTED,
    kACTIVE_UNSUPPORTED,
    kOWNER_NOT_FOUND,
};

inline LegacyCancelObservation classifyLegacyCancelObservation(
    bool const cancellationRequested, bool const cancelledBeforeLocalWork, bool const requestWasFound)
{
    if (cancellationRequested)
    {
        return cancelledBeforeLocalWork ? LegacyCancelObservation::kCANCELLED_WITHOUT_REUSE_PROOF
                                        : LegacyCancelObservation::kCANCELLATION_REQUESTED;
    }
    return requestWasFound ? LegacyCancelObservation::kACTIVE_UNSUPPORTED : LegacyCancelObservation::kOWNER_NOT_FOUND;
}

inline CancelResult makeLegacyCancelResult(LegacyCancelObservation const observation, std::string reason)
{
    switch (observation)
    {
    case LegacyCancelObservation::kCANCELLED_WITHOUT_REUSE_PROOF:
        return {LogicalDisposition::kACCEPTED, PhysicalDisposition::kIN_DOUBT, false, std::move(reason)};
    case LegacyCancelObservation::kCANCELLATION_REQUESTED:
        return {LogicalDisposition::kACCEPTED, PhysicalDisposition::kQUIESCING, true, std::move(reason)};
    case LegacyCancelObservation::kACTIVE_UNSUPPORTED:
        return {LogicalDisposition::kREJECTED, PhysicalDisposition::kACTIVE, true, std::move(reason)};
    case LegacyCancelObservation::kOWNER_NOT_FOUND:
        return {LogicalDisposition::kNOT_FOUND, PhysicalDisposition::kIN_DOUBT, true, std::move(reason)};
    }
    return {LogicalDisposition::kREJECTED, PhysicalDisposition::kIN_DOUBT, false,
        "Unknown legacy cancellation observation"};
}

inline bool legacyCancellationAccepted(CancelResult const& result)
{
    return result.logical == LogicalDisposition::kACCEPTED;
}

inline CancelResult failClosedForPoisonedStorage(CancelResult result)
{
    result.physical = PhysicalDisposition::kIN_DOUBT;
    result.retryable = false;
    result.reason += "; a transfer buffer is poisoned";
    return result;
}

inline ShutdownResult makeLegacyShutdownResult(bool const poisoned)
{
    return {PhysicalDisposition::kIN_DOUBT, std::nullopt, poisoned,
        poisoned ? "C++ cache transceiver has poisoned storage; endpoint replacement is required"
                 : "C++ cache transceiver does not yet provide synchronized endpoint-wide accounting or a submission "
                   "fence"};
}

} // namespace tensorrt_llm::batch_manager::lifecycle_detail
