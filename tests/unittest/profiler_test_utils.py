# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Decode the shared metadata envelope for Python and native golden tests."""

import ast
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from tensorrt_llm._profiler import MetadataValue


def decode_profiler_metadata(value: str) -> tuple[str, dict[str, "MetadataValue"]]:
    """Decode a UTF-8 byte-prefixed region and Python literal dictionary.

    Args:
        value: The complete TRTLLM1 envelope.

    Returns:
        The region name and decoded metadata fields.

    Raises:
        ValueError: If the prefix, length, region, or literal dictionary is invalid.
    """
    payload = value.encode("utf-8")
    prefix = b"TRTLLM1"
    if not payload.startswith(prefix):
        raise ValueError("Invalid profiler metadata prefix")
    try:
        separator = payload.index(b":", len(prefix))
        length = payload[len(prefix) : separator]
        if not length.isdigit():
            raise ValueError("Invalid region byte length")
        size = int(length)
        start = separator + 1
        end = start + size
        if size == 0 or end > len(payload):
            raise ValueError("Truncated or empty region")
        region = payload[start:end].decode("utf-8")
        metadata = ast.literal_eval(payload[end:].decode("utf-8"))
    except (SyntaxError, UnicodeError) as error:
        raise ValueError("Invalid profiler metadata envelope") from error
    if not isinstance(metadata, dict) or not all(isinstance(key, str) for key in metadata):
        raise ValueError("Profiler metadata must be a dictionary with string keys")
    return region, cast("dict[str, MetadataValue]", metadata)
