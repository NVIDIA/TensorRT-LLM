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

// A "forward stand-in" for the scatter-ordering tests: a kernel that occupies one block per SM and
// spins until the host releases it, the way a persistent cross-rank kernel holds the GPU while it
// waits for its peers.

#include <cuda_runtime_api.h>

namespace bounce_test
{

/// Host-mapped release flag plus the gate launch on `stream`. The kernel keeps running until
/// release() is called. Not copyable; the destructor releases and drains the gate.
class GateKernel
{
public:
    GateKernel();
    ~GateKernel();
    GateKernel(GateKernel const&) = delete;
    GateKernel& operator=(GateKernel const&) = delete;

    /// Launch the spinning gate on `stream` (grid = SM count). Returns the launch status.
    cudaError_t launch(cudaStream_t stream);
    /// Let the gate kernel finish.
    void release();
    /// Whether the gate kernel is still running (event not yet reached).
    [[nodiscard]] bool running() const;

private:
    int volatile* mFlag{nullptr};
    int* mFlagDev{nullptr};
    cudaEvent_t mDone{nullptr};
    cudaStream_t mStream{nullptr};
};

} // namespace bounce_test
