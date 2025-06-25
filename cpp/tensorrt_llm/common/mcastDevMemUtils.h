/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

// Avoid circular dependency
namespace tensorrt_llm::runtime
{
class McastDeviceMemory;
}

namespace tensorrt_llm::common
{
using McastDeviceMemory = tensorrt_llm::runtime::McastDeviceMemory;
// Register a buffer with the McastDeviceMemory class. This function does not check if the ptr belongs to the buffer!
void registerMcastDevMemBuffer(void* ptr, McastDeviceMemory* buf);
void unregisterMcastDevMemBuffer(McastDeviceMemory* buf);
// Find the buffer object from the given pointer, if it has been registered. This function does not take any size
// information. Thus a derived pointer cannot used as the key.
McastDeviceMemory* findMcastDevMemBuffer(void* ptr);

} // namespace tensorrt_llm::common
