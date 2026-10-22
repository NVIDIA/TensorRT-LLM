# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable internal facade for TensorRT-LLM's private OpenEngine bindings."""

import json
import re
import warnings
from importlib.resources import files

_MANIFEST = json.loads(
    files(__package__).joinpath("proto", "manifest.json").read_text(encoding="utf-8")
)
_PROTOBUF_GENCODE = str(_MANIFEST["generator"]["protobuf_gencode"])
_PROTOBUF_RUNTIME_MAJOR = int(str(_MANIFEST["runtime_floors"]["protobuf"]).split(".", 1)[0])

# Older Protobuf 6 runtimes near the requirements.txt floor warn for this
# supported Protobuf 5 gencode pairing. Newer 6.x runtimes omit the warning,
# making this version-specific filter a no-op there. Suppress only the known
# compatibility warning for our schema; requirements.txt excludes Protobuf 7.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=(
            rf"^Protobuf gencode version {re.escape(_PROTOBUF_GENCODE)} is exactly one major "
            rf"version older than the runtime version {_PROTOBUF_RUNTIME_MAJOR}\.[0-9.]+ at "
            r"openengine/v1/[A-Za-z0-9_]+\.proto\."
        ),
        category=UserWarning,
    )
    from ._generated import (
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


__all__ = [
    "MINIMUM_CLIENT_REVISION",
    "SCHEMA_RELEASE",
    "SCHEMA_REVISION",
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

SCHEMA_REVISION = int(_MANIFEST["schema_revision"])
MINIMUM_CLIENT_REVISION = int(_MANIFEST["minimum_client_revision"])
SCHEMA_RELEASE = str(_MANIFEST["bsr"]["release"])
