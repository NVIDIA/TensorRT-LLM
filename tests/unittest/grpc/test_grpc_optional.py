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
"""Guards for the optional SMG gRPC extra (``pip install tensorrt_llm[grpc-smg]``).

These run correctly with or without the extra installed: the "missing" case is
simulated so it is meaningful in every environment, and the "present" case is
guarded with ``importorskip``.
"""

import sys

import pytest


def test_smg_bindings_missing_gives_actionable_error(monkeypatch):
    """A missing 'smg-grpc-proto' yields an actionable install hint.

    Importing the SMG bindings must fail with a ``pip install
    tensorrt_llm[grpc-smg]`` hint rather than a bare ImportError.
    """
    # Force the optional package to look absent regardless of whether it is
    # actually installed in this environment (setting a sys.modules entry to
    # None makes ``import smg_grpc_proto`` raise ImportError).
    monkeypatch.setitem(sys.modules, "smg_grpc_proto", None)
    monkeypatch.delitem(sys.modules, "smg_grpc_proto.generated", raising=False)
    # Evict any cached bindings module, otherwise ``import`` is a cache hit that
    # returns without re-executing the guard.
    monkeypatch.delitem(sys.modules, "tensorrt_llm.grpc.smg.bindings", raising=False)

    with pytest.raises(ImportError, match=r"tensorrt_llm\[grpc-smg\]"):
        import tensorrt_llm.grpc.smg.bindings  # noqa: F401


def test_smg_bindings_present_smoke():
    """When the extra is installed, the bindings import cleanly.

    They must expose the pb2 modules the SMG adapter depends on.
    """
    pytest.importorskip(
        "smg_grpc_proto",
        reason="SMG gRPC adapter extra not installed",
    )
    from tensorrt_llm.grpc.smg import bindings

    assert bindings.trtllm_service_pb2 is not None
    assert bindings.trtllm_service_pb2_grpc is not None
