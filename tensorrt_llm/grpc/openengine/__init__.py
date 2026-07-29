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

r"""openengine gRPC protocol support for TensorRT-LLM (experimental).

The openengine contract (https://github.com/ai-dynamo/openengine) is a
vendor-neutral gRPC protocol for inference engines. Its ``.proto`` sources are
vendored under ``proto/openengine/v1/`` and the Python stubs are generated and
committed alongside this file; see ``PIN.md`` for the pinned commit and
``generate_stubs.sh`` for regeneration.

Unlike the SMG protocol, these stubs require no external pip package: only the
``grpcio`` and ``protobuf`` runtimes (already needed for gRPC serving).

The servicer and proto<->TensorRT-LLM converters are added in follow-up changes.
This first change vendors the protos and the generated stubs only.
"""

from tensorrt_llm.logger import logger

# Importing the stubs needs the grpc/protobuf runtimes. Catch ImportError so
# --grpc degrades to OPENENGINE_PROTOS_AVAILABLE=False when they are absent,
# and log it so a partially regenerated stub set names the missing module.
# Other failures (e.g. a protobuf descriptor/version error) propagate on
# purpose rather than be masked here as "unavailable".
try:
    from . import (
        error_pb2,
        generation_pb2,
        kv_pb2,
        lifecycle_pb2,
        lora_pb2,
        model_pb2,
        openengine_pb2,
        openengine_pb2_grpc,
        server_pb2,
    )

    OPENENGINE_PROTOS_AVAILABLE = True
except ImportError as exc:
    logger.warning(f"openengine gRPC protos unavailable: {exc!r}")
    OPENENGINE_PROTOS_AVAILABLE = False
    error_pb2 = None
    generation_pb2 = None
    kv_pb2 = None
    lifecycle_pb2 = None
    lora_pb2 = None
    model_pb2 = None
    openengine_pb2 = None
    openengine_pb2_grpc = None
    server_pb2 = None

__all__ = [
    "OPENENGINE_PROTOS_AVAILABLE",
    "error_pb2",
    "generation_pb2",
    "kv_pb2",
    "lifecycle_pb2",
    "lora_pb2",
    "model_pb2",
    "openengine_pb2",
    "openengine_pb2_grpc",
    "server_pb2",
]
