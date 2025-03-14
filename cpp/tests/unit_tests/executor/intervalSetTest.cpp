/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "tensorrt_llm/executor/intervalSet.h"

#include <gtest/gtest.h>

using tensorrt_llm::executor::IntervalSet;
using tensorrt_llm::executor::IdType;

class IntervalSetTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    void SetUp() override {}

    void TearDown() override {}

    IntervalSet<IdType> mIntervalSet;
};

namespace
{

TEST_F(IntervalSetTest, testPublicAPI)
{
    EXPECT_FALSE(mIntervalSet.contains(0));
    EXPECT_EQ(mIntervalSet.getNumElements(), 0);
    mIntervalSet.insert(0);
    mIntervalSet.insert(1);
    mIntervalSet.insert(4);
    mIntervalSet.insert(6);
    EXPECT_TRUE(mIntervalSet.contains(0));
    EXPECT_TRUE(mIntervalSet.contains(4));
    EXPECT_FALSE(mIntervalSet.contains(2125));
    EXPECT_EQ(mIntervalSet.getNumElements(), 4);
    mIntervalSet.insert(6);
    mIntervalSet.insert(4);
    mIntervalSet.insert(1);
    mIntervalSet.insert(0);
    EXPECT_EQ(mIntervalSet.getNumElements(), 4);
    EXPECT_TRUE(mIntervalSet.contains(0));
    EXPECT_TRUE(mIntervalSet.contains(4));
    EXPECT_FALSE(mIntervalSet.contains(9));
    EXPECT_FALSE(mIntervalSet.contains(3));
    EXPECT_FALSE(mIntervalSet.contains(11));
    mIntervalSet.clear();
    EXPECT_EQ(mIntervalSet.getNumElements(), 0);
}

TEST_F(IntervalSetTest, testClear)
{
    for (int i = 0; i < 100; i++)
    {
        if (i % 2 == 0)
        {
            EXPECT_FALSE(mIntervalSet.contains(i));
            mIntervalSet.insert(i);
            EXPECT_TRUE(mIntervalSet.contains(i));
            mIntervalSet.clear();
            EXPECT_EQ(mIntervalSet.getNumElements(), 0);
        }
        EXPECT_FALSE(mIntervalSet.contains(i));
    }
    for (int i = 0; i < 100; i++)
    {
        EXPECT_FALSE(mIntervalSet.contains(i));
    }
    mIntervalSet.clear();
    EXPECT_EQ(mIntervalSet.getNumElements(), 0);
    mIntervalSet.clear();
    EXPECT_EQ(mIntervalSet.getNumElements(), 0);

    for (int i = 19; i >= 10; i--)
    {
        if (i % 2 == 0)
        {
            EXPECT_FALSE(mIntervalSet.contains(i));
            mIntervalSet.insert(i);
            EXPECT_TRUE(mIntervalSet.contains(i));
        }
        else
        {
            EXPECT_FALSE(mIntervalSet.contains(i));
        }
    }
    EXPECT_EQ(mIntervalSet.getNumElements(), 5);
    mIntervalSet.clear();
    EXPECT_EQ(mIntervalSet.getNumElements(), 0);
    mIntervalSet.clear();
    EXPECT_EQ(mIntervalSet.getNumElements(), 0);
}

TEST_F(IntervalSetTest, testRandomInsert)
{
    mIntervalSet.insert(4);
    mIntervalSet.insert(8);
    EXPECT_EQ(mIntervalSet.getNumElements(), 2);
    std::vector<int> idToAdd{9, 7, 5, 1, 6, 0, 2};
    for (auto id : idToAdd)
    {
        mIntervalSet.insert(id);
    }
    for (int i = 0; i < 10; i++)
    {
        if (i != 3)
        {
            EXPECT_TRUE(mIntervalSet.contains(i));
        }
        else
        {
            EXPECT_FALSE(mIntervalSet.contains(i));
        }
    }
    EXPECT_EQ(mIntervalSet.getNumElements(), 9);
    mIntervalSet.insert(3);
    for (int i = 0; i < 10; i++)
    {
        EXPECT_TRUE(mIntervalSet.contains(i));
    }
    EXPECT_EQ(mIntervalSet.getNumElements(), 10);
    mIntervalSet.clear();
    EXPECT_EQ(mIntervalSet.getNumElements(), 0);
}

TEST_F(IntervalSetTest, testTerminatedReqIdIntervals)
{
    mIntervalSet.insert(4);
    mIntervalSet.insert(8);
    EXPECT_EQ(mIntervalSet.getNumElements(), 2);
    // terminatedReqIdIntervals is [[4, 4], [8, 8]]
    EXPECT_EQ(mIntervalSet.getIntervals().size(), 2);
    mIntervalSet.insert(3);
    mIntervalSet.insert(5);
    // terminatedReqIdIntervals is [[3, 5], [8, 8]]
    EXPECT_EQ(mIntervalSet.getNumElements(), 4);
    EXPECT_EQ(mIntervalSet.getIntervals().size(), 2);
    mIntervalSet.insert(9);
    mIntervalSet.insert(7);
    // terminatedReqIdIntervals is [[3, 5], [7, 9]]
    EXPECT_EQ(mIntervalSet.getNumElements(), 6);
    EXPECT_EQ(mIntervalSet.getIntervals().size(), 2);
    mIntervalSet.insert(6);
    // terminatedReqIdIntervals is [[3, 9]]
    EXPECT_EQ(mIntervalSet.getNumElements(), 7);
    EXPECT_EQ(mIntervalSet.getIntervals().size(), 1);
    mIntervalSet.insert(1);
    // terminatedReqIdIntervals is [[1, 1], [3, 9]]
    EXPECT_EQ(mIntervalSet.getNumElements(), 8);
    EXPECT_EQ(mIntervalSet.getIntervals().size(), 2);
    mIntervalSet.insert(0);
    // terminatedReqIdIntervals is [[0, 1], [3, 9]]
    EXPECT_EQ(mIntervalSet.getNumElements(), 9);
    EXPECT_EQ(mIntervalSet.getIntervals().size(), 2);
    mIntervalSet.insert(2);
    // terminatedReqIdIntervals is [[0, 9]]
    EXPECT_EQ(mIntervalSet.getNumElements(), 10);
    EXPECT_EQ(mIntervalSet.getIntervals().size(), 1);
    for (int i = 0; i < 10; i++)
    {
        mIntervalSet.insert(i);
        // terminatedReqIdIntervals is always [[0, 9]]
        EXPECT_EQ(mIntervalSet.getNumElements(), 10);
        EXPECT_EQ(mIntervalSet.getIntervals().size(), 1);
    }
    mIntervalSet.clear();
    EXPECT_EQ(mIntervalSet.getNumElements(), 0);
    EXPECT_EQ(mIntervalSet.getIntervals().size(), 0);

    // Insert continuous decreasing numbers. Interval size is always one.
    for (int i = 19; i >= 10; i--)
    {
        mIntervalSet.insert(i);
        EXPECT_TRUE(mIntervalSet.contains(i));
        EXPECT_EQ(mIntervalSet.getIntervals().size(), 1);
    }
    mIntervalSet.clear();
    EXPECT_EQ(mIntervalSet.getNumElements(), 0);
    EXPECT_EQ(mIntervalSet.getIntervals().size(), 0);

    // Insert 50 disjoint even numbers
    for (int i = 0; i < 100; i++)
    {
        if (i % 2 == 0)
        {
            mIntervalSet.insert(i);
            EXPECT_EQ(mIntervalSet.getNumElements(), (i / 2) + 1);
            EXPECT_EQ(mIntervalSet.getIntervals().size(), (i / 2) + 1);
        }
    }

    // Insert 50 disjoint odd numbers. Interval size should go down as the intervals are merged.
    for (int i = 0; i < 100; i++)
    {
        if (i % 2 != 0)
        {
            mIntervalSet.insert(i);
            EXPECT_EQ(mIntervalSet.getNumElements(), 50 + (i + 1) / 2);
            if (i != 99)
            {
                EXPECT_EQ(mIntervalSet.getIntervals().size(), 50 - (i + 1) / 2);
            }
            else
            {
                EXPECT_EQ(mIntervalSet.getIntervals().size(), 1);
            }
        }
    }
}

} // namespace
