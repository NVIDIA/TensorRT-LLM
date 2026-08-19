# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generated protobuf bindings used by the SMG adapter."""

try:
    from smg_grpc_proto.generated import trtllm_service_pb2, trtllm_service_pb2_grpc
except ModuleNotFoundError as e:
    if e.name != "smg_grpc_proto":
        raise
    raise ModuleNotFoundError(
        "The SMG gRPC adapter requires the optional 'smg-grpc-proto' package, "
        "which is not part of the default TensorRT-LLM installation. Install it "
        'with: pip install "tensorrt_llm[grpc-smg]"',
        name=e.name,
    ) from e

__all__ = ["trtllm_service_pb2", "trtllm_service_pb2_grpc"]
