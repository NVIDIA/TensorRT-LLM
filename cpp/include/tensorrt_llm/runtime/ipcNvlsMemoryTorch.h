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

#include "tensorrt_llm/runtime/ipcNvlsMemory.h"

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

namespace tensorrt_llm::runtime
{

//! Create an NVLS rendezvous backed by one multi-backend Torch ProcessGroup.
//! CPU collectives used for handle exchange require a Gloo backend on this
//! same ProcessGroup; CUDA collectives may use its NCCL backend.
IpcNvlsRendezvousPtr makeTorchDistIpcNvlsRendezvous(c10::intrusive_ptr<c10d::ProcessGroup> processGroup);

} // namespace tensorrt_llm::runtime
