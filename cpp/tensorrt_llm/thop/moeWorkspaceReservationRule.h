/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <cstddef>
#include <cstdint>

// What the MoE capture-workspace reservation does with one request, and nothing
// else. Extracted for the same reason as the NCCL capture rule: reaching it
// otherwise needs a built libtensorrt_llm and a GPU, so the decision the fix
// turns on was the least testable part of it.
//
// The problem it solves: getWorkspaceInfo reallocates unconditionally while a
// capture is underway, so the block comes from the graph's PrivatePool while the
// runner caches it for the life of the process. torch.cuda.graph.__enter__ calls
// empty_cache() before EVERY capture, so that pool can be erased at the start of
// a later capture, leaving the cached block with a dangling BlockPool*.
//
// The reservation avoids it by allocating eagerly and never allocating under
// capture. Every way it can decline leaves the old behaviour in place -- it must
// never be a new way to fail.

namespace tensorrt_llm::moe_reservation
{

enum class ReservationAction
{
    ServeReservation,   //!< return the reserved block; freeze it
    DeclineMultiStream, //!< MoE has run on >1 eager stream: sharing one block is unsound
    DeclineTooSmall,    //!< no eager call has sized it for this shape
    GrowReservation,    //!< eager path: allocate or enlarge (parking a frozen block)
    UseStreamWorkspace, //!< nothing to do here; the ordinary per-stream path serves
};

struct ReservationState
{
    bool enabled{false};
    bool capturing{false};
    size_t eagerStreamCount{0}; //!< distinct streams MoE has run on OUTSIDE a capture
    int64_t reservedBytes{0};
    int64_t neededBytes{0};
    bool frozen{false}; //!< a captured graph already reads the reservation
};

inline ReservationAction reservationAction(ReservationState const& s)
{
    if (!s.enabled)
    {
        return ReservationAction::UseStreamWorkspace;
    }
    // Counting EAGER streams, not map entries: the fallback path inserts an
    // entry keyed by the capture stream, so counting the map meant one decline
    // disabled the reservation for the rest of the process under a reason that
    // was not the real one.
    bool const singleStream = s.eagerStreamCount <= 1;
    if (s.capturing && !singleStream)
    {
        return ReservationAction::DeclineMultiStream;
    }
    if (s.capturing && s.reservedBytes >= s.neededBytes)
    {
        return ReservationAction::ServeReservation;
    }
    if (s.capturing)
    {
        return ReservationAction::DeclineTooSmall;
    }
    if (s.reservedBytes < s.neededBytes)
    {
        return ReservationAction::GrowReservation;
    }
    return ReservationAction::UseStreamWorkspace;
}

//! Does growing here have to park the current block rather than free it?
//!
//! Once a captured graph has read the reservation, freeing it would pull memory
//! out from under a live graph. Parking is bounded by the number of distinct
//! growth steps, not by the number of graphs.
inline bool growthMustPark(ReservationState const& s)
{
    return reservationAction(s) == ReservationAction::GrowReservation && s.frozen;
}

//! Apply an action to the state, so a test can drive the lifecycle as a
//! sequence rather than as isolated decisions.
//!
//! The subtle part of this rule is not any single branch, it is the order:
//! grow, serve-and-freeze, grow-again-and-park. Reading the branches one at a
//! time cannot see whether freezing and parking agree with each other.
//!
//! parkedCount is incremented when growth has to park rather than free.
//!
//! LIMITATION, stated because the equivalent gap on the NCCL side was worth
//! closing and this one is not closable the same way: unlike
//! nccl_window_rule::applyHandOut, this is a MODEL of what getWorkspaceInfo
//! does, not the code it runs. It cannot be the code, because the real mutation
//! allocates a torch::Tensor and parks the previous one, and this header is
//! deliberately free of torch. So the two can drift.
//!
//! What has to stay true for the model to be faithful, i.e. what to re-check
//! after touching the switch in getWorkspaceInfo:
//!
//!   ServeReservation    sets frozen, changes nothing else
//!   GrowReservation     parks iff growthMustPark(), then makes the reservation
//!                       exactly neededBytes
//!   the three others    change no state at all
inline void applyAction(ReservationState& s, size_t& parkedCount)
{
    switch (reservationAction(s))
    {
    case ReservationAction::ServeReservation: s.frozen = true; break;
    case ReservationAction::GrowReservation:
        if (s.frozen)
        {
            ++parkedCount;
        }
        s.reservedBytes = s.neededBytes;
        break;
    case ReservationAction::DeclineMultiStream:
    case ReservationAction::DeclineTooSmall:
    case ReservationAction::UseStreamWorkspace: break;
    }
}

//! What clearWorkspaces() does: the graphs are gone, so the reservation is
//! released and may grow again.
inline void applyClear(ReservationState& s)
{
    s.reservedBytes = 0;
    s.frozen = false;
    s.eagerStreamCount = 0;
}

} // namespace tensorrt_llm::moe_reservation
