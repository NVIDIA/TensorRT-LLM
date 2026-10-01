# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime requirements shared by the gRPC adapters."""

from google.protobuf.internal import api_implementation


def _require_native_protobuf() -> None:
    """Reject a protobuf runtime without the native upb implementation."""
    implementation = api_implementation.Type()
    if implementation != "upb":
        raise RuntimeError(
            "TensorRT-LLM gRPC requires native protobuf (upb), but the active "
            f"implementation is {implementation!r}. Set "
            "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=upb before starting Python "
            "and install a supported protobuf wheel with the upb extension. "
            "Changing the environment after importing protobuf does not change "
            "the active implementation."
        )
