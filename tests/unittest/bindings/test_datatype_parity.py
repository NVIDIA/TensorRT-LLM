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
"""Serialization/format-compatibility guard for the internal data-type enum.

``tensorrt_llm.bindings.DataType`` is backed by ``tensorrt_llm::DataType``
(``common/tllmDataType.h``), currently an alias of ``nvinfer1::DataType`` and
slated to become a standalone enum when the TensorRT-engine execution path is
removed. Its enumerator *integer values* must continue to match the legacy
``nvinfer1::DataType`` values so that previously-serialized executor configs
and KV-cache metadata remain byte-compatible, and so the public Python member
set is unchanged.
"""

import pytest

bindings = pytest.importorskip("tensorrt_llm.bindings")

# Legacy nvinfer1::DataType integer values. These MUST remain stable.
LEGACY_DATATYPE_VALUES = {
    "FLOAT": 0,
    "HALF": 1,
    "INT8": 2,
    "INT32": 3,
    "BOOL": 4,
    "UINT8": 5,
    "FP8": 6,
    "BF16": 7,
    "INT64": 8,
    "NVFP4": 10,
}


def test_bindings_datatype_values_match_legacy():
    """Each exposed DataType enumerator keeps its legacy numeric value."""
    dt = bindings.DataType
    for name, value in LEGACY_DATATYPE_VALUES.items():
        assert hasattr(dt, name), f"tensorrt_llm.bindings.DataType is missing '{name}'"
        assert getattr(dt, name).value == value, (
            f"DataType.{name} numeric value changed to {getattr(dt, name).value}; "
            f"expected {value} (breaks serialization compatibility)"
        )


def test_bindings_datatype_member_set_unchanged():
    """The public DataType member set still contains all legacy members."""
    members = {m for m in dir(bindings.DataType) if not m.startswith("_")}
    missing = set(LEGACY_DATATYPE_VALUES) - members
    assert not missing, f"tensorrt_llm.bindings.DataType lost members: {sorted(missing)}"
