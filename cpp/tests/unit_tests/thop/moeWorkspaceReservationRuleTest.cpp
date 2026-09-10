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

// The decision that keeps MoE from allocating under CUDA graph capture, driven
// on plain types: no GPU, no allocator, no graph. getWorkspaceInfo() calls
// exactly these functions, so what is asserted here is what runs.

#include "tensorrt_llm/thop/moeWorkspaceReservationRule.h"

#include <gtest/gtest.h>

using namespace tensorrt_llm::moe_reservation;

namespace
{

ReservationState st(bool enabled, bool capturing, size_t streams, int64_t reserved, int64_t needed, bool frozen)
{
    return ReservationState{enabled, capturing, streams, reserved, needed, frozen};
}

} // namespace

TEST(MoeWorkspaceReservationRule, DisabledIsInert)
{
    EXPECT_EQ(reservationAction(st(false, true, 1, 0, 100, false)), ReservationAction::UseStreamWorkspace);
    EXPECT_EQ(reservationAction(st(false, false, 1, 0, 100, false)), ReservationAction::UseStreamWorkspace);
}

TEST(MoeWorkspaceReservationRule, EagerSizesItAndTheCaptureReadsIt)
{
    EXPECT_EQ(reservationAction(st(true, false, 1, 0, 100, false)), ReservationAction::GrowReservation)
        << "eager, nothing reserved yet";
    EXPECT_EQ(reservationAction(st(true, true, 1, 100, 100, false)), ReservationAction::ServeReservation)
        << "capture, reservation exactly big enough";
    EXPECT_EQ(reservationAction(st(true, true, 1, 200, 100, false)), ReservationAction::ServeReservation)
        << "capture, reservation larger than needed";
}

// Every decline must leave the pre-existing path in place. The reservation is an
// improvement, not a new way to fail.
TEST(MoeWorkspaceReservationRule, DeclinesFallBackRatherThanFail)
{
    EXPECT_EQ(reservationAction(st(true, true, 1, 50, 100, false)), ReservationAction::DeclineTooSmall);
    EXPECT_EQ(reservationAction(st(true, true, 2, 200, 100, false)), ReservationAction::DeclineMultiStream);
    EXPECT_EQ(reservationAction(st(true, true, 2, 50, 100, false)), ReservationAction::DeclineMultiStream)
        << "multi-stream must win over too-small: sharing one block across streams is unsound at any size";
}

// Nothing has run eagerly, so nothing sized it. Must not be read as
// multi-stream, and must not serve out of an empty reservation.
TEST(MoeWorkspaceReservationRule, NoEagerStreamSeenYet)
{
    EXPECT_EQ(reservationAction(st(true, true, 0, 0, 100, false)), ReservationAction::DeclineTooSmall);
}

TEST(MoeWorkspaceReservationRule, EagerLeavesABigEnoughReservationAlone)
{
    EXPECT_EQ(reservationAction(st(true, false, 1, 200, 100, false)), ReservationAction::UseStreamWorkspace);
}

// A known inefficiency, pinned so it is a property rather than a surprise: on a
// multi-stream model the eager path still grows a reservation that capture will
// always decline.
TEST(MoeWorkspaceReservationRule, EagerGrowsEvenWhenCaptureWillDecline)
{
    EXPECT_EQ(reservationAction(st(true, false, 2, 0, 100, false)), ReservationAction::GrowReservation);
}

TEST(MoeWorkspaceReservationRule, GrowthParksOnlyOnceAGraphReadsIt)
{
    EXPECT_TRUE(growthMustPark(st(true, false, 1, 50, 100, true)));
    EXPECT_FALSE(growthMustPark(st(true, false, 1, 50, 100, false))) << "nothing has baked a pointer in yet";
    EXPECT_FALSE(growthMustPark(st(true, true, 1, 200, 100, true))) << "not growing";
}

// The branches are simple; the ORDER is where this rule can go wrong. Reading
// them one at a time cannot see whether freezing and parking agree.
TEST(MoeWorkspaceReservationRule, Lifecycle)
{
    size_t parked = 0;
    ReservationState s{true, false, 0, 0, 0, false};

    // Warmup at the largest shape, on the model stream.
    s.capturing = false;
    s.eagerStreamCount = 1;
    s.neededBytes = 1000;
    ASSERT_EQ(reservationAction(s), ReservationAction::GrowReservation);
    applyAction(s, parked);

    // The first capture is the largest graph: the engine captures in descending
    // batch size, so the largest shape is already sized by warmup.
    s.capturing = true;
    ASSERT_EQ(reservationAction(s), ReservationAction::ServeReservation);
    applyAction(s, parked);
    EXPECT_TRUE(s.frozen) << "serving must freeze: a graph now holds this address";

    s.neededBytes = 400;
    EXPECT_EQ(reservationAction(s), ReservationAction::ServeReservation) << "a smaller graph is served too";

    // A later eager call at a BIGGER shape. The frozen block is read by a live
    // graph, so growth must park it rather than free it.
    s.capturing = false;
    s.neededBytes = 2000;
    ASSERT_EQ(reservationAction(s), ReservationAction::GrowReservation);
    EXPECT_TRUE(growthMustPark(s));
    applyAction(s, parked);
    EXPECT_EQ(parked, 1u) << "exactly one block parked, none freed";

    s.capturing = true;
    s.neededBytes = 1500;
    EXPECT_EQ(reservationAction(s), ReservationAction::ServeReservation) << "served from the grown block";

    applyClear(s);
    EXPECT_FALSE(s.frozen) << "clearWorkspaces() is the point at which no graph replays against these";
    s.capturing = true;
    s.neededBytes = 100;
    EXPECT_EQ(reservationAction(s), ReservationAction::DeclineTooSmall)
        << "after clear a capture finds nothing reserved and must fall back";
}

// A second eager stream appearing mid-run must take the reservation out of the
// path from then on.
TEST(MoeWorkspaceReservationRule, SecondEagerStreamMidRunStopsItServing)
{
    size_t parked = 0;
    ReservationState s{true, false, 1, 0, 500, false};
    applyAction(s, parked);
    s.capturing = true;
    ASSERT_EQ(reservationAction(s), ReservationAction::ServeReservation);

    s.eagerStreamCount = 2;
    EXPECT_EQ(reservationAction(s), ReservationAction::DeclineMultiStream);
}
