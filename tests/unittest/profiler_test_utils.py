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

"""Shared profiler selection, execution-trace capture, and metadata decoding."""

import ast
import json
import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch

if TYPE_CHECKING:
    from tensorrt_llm._profiler import MetadataValue, ProfilerRegion


def select_regions(monkeypatch: pytest.MonkeyPatch, *regions: "ProfilerRegion") -> None:
    """Select cached custom regions without process startup or repeated patch details."""
    from tensorrt_llm import _profiler

    monkeypatch.setattr(_profiler, "_CUSTOM_REGIONS", frozenset(regions))


@contextmanager
def execution_trace_observer(path: Path) -> Iterator[torch.profiler.ExecutionTraceObserver]:
    """Register a real observer and always release its process-wide callback."""
    observer = torch.profiler.ExecutionTraceObserver()
    try:
        observer.register_callback(str(path))
        yield observer
    finally:
        observer.unregister_callback()


def profiler_nodes(path: Path, *names: str) -> list[dict[str, Any]]:
    """Load observer nodes, optionally restricted to the supplied scope names."""
    nodes = json.loads(path.read_text())["nodes"]
    return [node for node in nodes if not names or node["name"] in names]


def decode_node_metadata(node: dict[str, Any]) -> tuple[str, dict[str, "MetadataValue"]]:
    """Decode the single metadata input of an annotated observer scope."""
    return decode_profiler_metadata(node["inputs"]["values"][0])


def _validate_decoded_value(value: object) -> None:
    if type(value) in (type(None), str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, dict):
        if not all(type(key) is str for key in value):
            raise ValueError("Profiler metadata keys must be strings")
        values = value.values()
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        raise ValueError("Unsupported profiler metadata value")
    for item in values:
        _validate_decoded_value(item)


def decode_profiler_metadata(value: str) -> tuple[str, dict[str, "MetadataValue"]]:
    """Decode JSON framing or the legacy UTF-8-prefixed Python literal format.

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
        json_format = payload.startswith(prefix + b":")
        length_start = len(prefix) + int(json_format)
        separator = payload.index(b":", length_start)
        length = payload[length_start:separator]
        if not length.isdigit():
            raise ValueError("Invalid region byte length")
        size = int(length)
        start = separator + 1
        end = start + size
        if size == 0 or end > len(payload):
            raise ValueError("Truncated or empty region")
        region = payload[start:end].decode("utf-8")
        encoded = payload[end:].decode("utf-8")
        metadata = json.loads(encoded) if json_format else ast.literal_eval(encoded)
    except (SyntaxError, UnicodeError) as error:
        raise ValueError("Invalid profiler metadata envelope") from error
    if not isinstance(metadata, dict):
        raise ValueError("Profiler metadata must be a dictionary with string keys")
    _validate_decoded_value(metadata)
    return region, cast("dict[str, MetadataValue]", metadata)
