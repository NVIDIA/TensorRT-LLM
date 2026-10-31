/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kvCacheManagerV2TestUtils.h"
#include "kv_cache_manager_v2/blockRadixTree.h"
#include "kv_cache_manager_v2/streamingEventSink.h"

#include <gtest/gtest.h>

#include <memory>
#include <variant>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;

TEST(KvCacheManagerV2StreamingEventTest, EmptyCacheResetsAreObservableWithoutInference)
{
    auto const config = test::makeConfig();
    LifeCycleRegistry const lifeCycles(config);
    auto const sink = std::make_shared<StreamingEventSink>(1);
    BlockRadixTree tree(lifeCycles, config.tokensPerBlock, sink);

    // Repeated empty resets must be visible even when capture has a one-entry cap.
    tree.clear();
    tree.clear();
    auto const events = sink->drainIterationEvents();
    ASSERT_EQ(events.size(), 2);
    for (auto const& event : events)
    {
        EXPECT_TRUE(std::holds_alternative<StreamingAllBlocksClearedData>(event));
    }
    EXPECT_TRUE(sink->drainIterationEvents().empty());
    EXPECT_EQ(sink->getStats().droppedEvents, 0);
}

} // namespace
