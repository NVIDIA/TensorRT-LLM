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

These tests drive `check_gms_source_identity` directly -- the single decision
point the GMS RO path consults before `materialize_module`. Unlike MX, GMS has
no disk fallback, so a verified mismatch must *raise* under `STRICT` rather
than fall back. The GMS backend is a stub exposing only `get_source_identity`
so no GMS pool, GPU, or model is touched.
"""

import pytest
from _source_identity_fakes import make_identity as _identity

from tensorrt_llm._torch.models.checkpoints.gms.model_loading import check_gms_source_identity
from tensorrt_llm._torch.weight_sharing import SourceIdentityMismatchError

pytestmark = pytest.mark.cpu_only


class _FakeGMSBackend:
    """Minimal GMS backend stub exposing only the gate's fetch seam."""

    def __init__(self, source_identity):
        self._source_identity = source_identity

    def get_source_identity(self):
        return self._source_identity


def test_gate_passes_on_matching_identity():
    local = _identity(attn_backend="TRTLLM")
    writer = _identity(attn_backend="TRTLLM")
    # Compatible writer: the gate is a no-op (no raise).
    check_gms_source_identity(local, _FakeGMSBackend(writer))


def test_gate_raises_on_mismatch():
    local = _identity(attn_backend="TRTLLM")
    writer = _identity(attn_backend="FLASHINFER")
    # No disk fallback for GMS: an incompatible writer must raise.
    with pytest.raises(SourceIdentityMismatchError):
        check_gms_source_identity(local, _FakeGMSBackend(writer))


def test_gate_raises_on_checkpoint_artifact_mismatch():
    local = _identity(artifact_key="fine-tune-a")
    writer = _identity(artifact_key="fine-tune-b")
    with pytest.raises(SourceIdentityMismatchError):
        check_gms_source_identity(local, _FakeGMSBackend(writer))


def test_gate_raises_when_writer_identity_unavailable():
    # Publisher metadata not wired yet (get_source_identity returns None);
    # GMS has no disk fallback, so unverified sharing must raise.
    local = _identity()
    with pytest.raises(SourceIdentityMismatchError):
        check_gms_source_identity(local, _FakeGMSBackend(None))
