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
"""Exercise the real GMS read-only pre-materialize SourceIdentity gate.

These tests drive ``ModelLoader._check_gms_source_identity`` directly -- the
single decision point the GMS RO path consults before ``materialize_module``.
Unlike MX, GMS has no disk fallback, so a verified mismatch must *raise* under
``STRICT`` rather than fall back. The ``ModelLoader`` is constructed via
``__new__`` to bypass its heavy ``__init__``; the GMS backend is a stub
exposing only ``get_source_identity`` so no GMS pool, GPU, or model is touched.
"""

import pytest
from _source_identity_fakes import make_identity as _identity

from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader
from tensorrt_llm._torch.weight_sharing import SourceIdentityMismatchError
from tensorrt_llm.llmapi.llm_args import LoadFormat


class _FakeCheckpointLoader:
    """Minimal checkpoint-loader stub exposing ``checkpoint_format``."""

    def __init__(self, checkpoint_format):
        self.checkpoint_format = checkpoint_format


class _FakeGMSBackend:
    """Minimal GMS backend stub exposing only the gate's fetch seam."""

    def __init__(self, source_identity):
        self._source_identity = source_identity

    def get_source_identity(self):
        return self._source_identity


def _new_loader(local_identity):
    """Construct a loader bypassing heavy base __init__, wire the local id."""
    loader = ModelLoader.__new__(ModelLoader)
    loader._source_identity = local_identity
    return loader


def test_source_identity_is_only_needed_for_mx_or_gms():
    # Default HF/AUTO loading should be a strict no-op: no SourceIdentity
    # construction, no tensor-layout traversal, no behavior change.
    assert not ModelLoader._needs_source_identity(_FakeCheckpointLoader("HF"), LoadFormat.AUTO)
    assert ModelLoader._needs_source_identity(_FakeCheckpointLoader("MX"), LoadFormat.AUTO)
    assert ModelLoader._needs_source_identity(_FakeCheckpointLoader("HF"), LoadFormat.GMS)


def test_gate_passes_on_matching_identity():
    local = _identity(attn_backend="TRTLLM")
    writer = _identity(attn_backend="TRTLLM")
    loader = _new_loader(local)
    # Compatible writer: the gate is a no-op (no raise).
    loader._check_gms_source_identity(_FakeGMSBackend(writer))


def test_gate_raises_on_mismatch():
    local = _identity(attn_backend="TRTLLM")
    writer = _identity(attn_backend="FLASHINFER")
    loader = _new_loader(local)
    # No disk fallback for GMS: an incompatible writer must raise.
    with pytest.raises(SourceIdentityMismatchError):
        loader._check_gms_source_identity(_FakeGMSBackend(writer))


def test_gate_inert_when_writer_identity_unavailable():
    # Publisher metadata not wired yet (get_source_identity returns None);
    # the gate is inert and materialization proceeds without enforcement.
    local = _identity()
    loader = _new_loader(local)
    loader._check_gms_source_identity(_FakeGMSBackend(None))
