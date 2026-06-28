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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/NixlTransferEngine.h"

#include <gtest/gtest.h>

namespace b = tensorrt_llm::executor::kv_cache::bounce;

// Compile/link smoke test: forces NixlTransferEngine.cpp to compile against nixl.h and link
// against NIXL::nixl (verifying the createXferReq/registerMem/getXferStatus API usage is
// correct). It does NOT invoke NIXL at runtime — constructing with a null agent is safe (the
// ctor only stores the pointer).
//
// The real NIXL runtime path (registerRegion/MR lifecycle, postWrite/poll/release, the
// SUCCESS==data-landed poll semantics, and teardown drain) IS exercised over actual RDMA,
// byte-exact + concurrent + multi-agent, by bounceNixlE2ETest.cpp (a single-process two-agent
// loopback harness; it GTEST_SKIPs when no GPU / NIXL backend is available). NixlTransferEngine
// wraps a live nixlAgent, so it cannot be unit-tested in isolation without a real agent — that
// e2e harness is the coverage, not this file.
TEST(NixlTransferEngine, ConstructsAndLinks)
{
    b::NixlTransferEngine engine(/*agent=*/nullptr, /*deviceId=*/0);
    SUCCEED();
}
