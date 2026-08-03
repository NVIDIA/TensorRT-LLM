# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

r"""SMG integration for high-performance communication with external routers.

This module provides a gRPC server interface that accepts pre-tokenized requests
and returns raw token IDs, enabling efficient binary communication with Rust-based
routers like sgl-router.

Key Features:
- Pre-tokenized input (no Python tokenization overhead)
- Raw token ID output (no Python detokenization overhead)
- Streaming support with delta tokens
- Full sampling parameter support
- Guided decoding (JSON schema, regex, grammar)
- LoRA and prompt tuning support
- Disaggregated inference support

Proto definitions are provided by the smg-grpc-proto package (pip install smg-grpc-proto).

Usage:
    python -m tensorrt_llm.commands.serve /path/to/model \
        --grpc \
        --host 0.0.0.0 \
        --port 50051
"""

# Try to import generated protobuf modules from smg-grpc-proto package
try:
    from smg_grpc_proto.generated import trtllm_service_pb2, trtllm_service_pb2_grpc

    PROTOS_AVAILABLE = True
except ImportError:
    PROTOS_AVAILABLE = False
    trtllm_service_pb2 = None
    trtllm_service_pb2_grpc = None

# Import protobuf-dependent modules only when the generated package is present.
if PROTOS_AVAILABLE:
    try:
        from .request_manager import (
            GrpcRequestManager,
            create_disaggregated_params_from_proto,
            create_lora_request_from_proto,
            create_sampling_params_from_proto,
        )

        REQUEST_MANAGER_AVAILABLE = True
    except ImportError:
        REQUEST_MANAGER_AVAILABLE = False
        GrpcRequestManager = None
        create_sampling_params_from_proto = None
        create_lora_request_from_proto = None
        create_disaggregated_params_from_proto = None

    try:
        from .servicer import TrtllmServiceServicer

        SERVICER_AVAILABLE = True
    except ImportError:
        SERVICER_AVAILABLE = False
        TrtllmServiceServicer = None
else:
    REQUEST_MANAGER_AVAILABLE = False
    GrpcRequestManager = None
    create_sampling_params_from_proto = None
    create_lora_request_from_proto = None
    create_disaggregated_params_from_proto = None
    SERVICER_AVAILABLE = False
    TrtllmServiceServicer = None

__all__ = [
    "PROTOS_AVAILABLE",
    "REQUEST_MANAGER_AVAILABLE",
    "SERVICER_AVAILABLE",
    "GrpcRequestManager",
    "TrtllmServiceServicer",
    "create_disaggregated_params_from_proto",
    "create_lora_request_from_proto",
    "create_sampling_params_from_proto",
    "trtllm_service_pb2",
    "trtllm_service_pb2_grpc",
]
