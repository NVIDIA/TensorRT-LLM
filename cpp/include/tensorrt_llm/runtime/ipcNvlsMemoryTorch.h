/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
