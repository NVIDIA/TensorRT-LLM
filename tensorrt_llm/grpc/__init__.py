# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

r"""TensorRT-LLM gRPC module for high-performance communication with external routers.

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

Usage:
    python -m tensorrt_llm.commands.serve /path/to/model \
        --grpc \
        --host 0.0.0.0 \
        --port 50051
"""

from pathlib import Path

# Module directory for proto files
GRPC_MODULE_DIR = Path(__file__).parent

# Proto file path
PROTO_FILE = GRPC_MODULE_DIR / "trtllm_service.proto"

# Try to import generated protobuf modules
try:
    from . import trtllm_service_pb2, trtllm_service_pb2_grpc

    PROTOS_AVAILABLE = True
except ImportError:
    PROTOS_AVAILABLE = False
    trtllm_service_pb2 = None
    trtllm_service_pb2_grpc = None


def compile_protos():
    """Compile protobuf files to generate Python modules.

    Run this function if the generated *_pb2.py files are missing.
    Alternatively, run: python tensorrt_llm/grpc/compile_protos.py
    """
    from .compile_protos import main as compile_main

    compile_main()


def ensure_protos_available():
    """Ensure protobuf modules are available, compiling if necessary."""
    global PROTOS_AVAILABLE, trtllm_service_pb2, trtllm_service_pb2_grpc

    if not PROTOS_AVAILABLE:
        raise ImportError(
            "gRPC protobuf modules are not available. "
            "Please generate them by running: "
            "python tensorrt_llm/grpc/compile_protos.py"
        )


# Try to import request manager
try:
    from .grpc_request_manager import (
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

# Try to import servicer
try:
    from .grpc_servicer import TrtllmServiceServicer

    SERVICER_AVAILABLE = True
except ImportError:
    SERVICER_AVAILABLE = False
    TrtllmServiceServicer = None

__all__ = [
    "GRPC_MODULE_DIR",
    "PROTO_FILE",
    "PROTOS_AVAILABLE",
    "REQUEST_MANAGER_AVAILABLE",
    "SERVICER_AVAILABLE",
    "compile_protos",
    "ensure_protos_available",
    "trtllm_service_pb2",
    "trtllm_service_pb2_grpc",
    "GrpcRequestManager",
    "TrtllmServiceServicer",
    "create_sampling_params_from_proto",
    "create_lora_request_from_proto",
    "create_disaggregated_params_from_proto",
]
