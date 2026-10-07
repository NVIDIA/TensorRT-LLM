/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "userbuffersTensor.h"

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// Custom op interface for create_userbuffers_tensor.
// Python side does not need the UBBuffer object.
torch::Tensor create_userbuffers_tensor_op(at::IntArrayRef shape, torch::ScalarType dtype)
{
    return create_userbuffers_tensor(shape, dtype).first;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("create_userbuffers_tensor", &tensorrt_llm::torch_ext::create_userbuffers_tensor_op);
}
