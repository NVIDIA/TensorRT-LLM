# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protobuf performance guidance shared by the gRPC launchers."""

from google.protobuf.internal import api_implementation

from tensorrt_llm.logger import logger


def _warn_if_python_protobuf() -> None:
    """Warn when the active protobuf implementation is pure Python."""
    implementation = api_implementation.Type()
    if implementation == "python":
        logger.warning(
            "TensorRT-LLM gRPC is using Python protobuf, which can reduce throughput. Set "
            "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=upb before starting Python "
            "and install a supported protobuf wheel with the upb extension. "
            "Changing the environment after importing protobuf does not change "
            "the active implementation."
        )
