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
"""Exercise the real MX checkpoint loader pre-transfer SourceIdentity gate.

Upstream `modelexpress` is never imported. The tests run the compatibility
decision against real `SourceIdentity` objects and use a small discovery
client stub for the pinned ModelExpress 0.4.1 API shape.
"""

from types import SimpleNamespace

from _source_identity_fakes import FakeMapping
from _source_identity_fakes import make_identity as _identity

from tensorrt_llm._torch.models.checkpoints.mx.checkpoint_loader import (
    MXCheckpointLoader,
    _build_mx_source_metadata,
)


def _new_loader(local_identity):
    """Construct a loader while bypassing the heavy base initializer."""
    loader = MXCheckpointLoader.__new__(MXCheckpointLoader)
    loader._local_source_identity = local_identity
    return loader


def test_gate_proceeds_on_matching_identity():
    local = _identity(attn_backend="TRTLLM")
    source = _identity(attn_backend="TRTLLM")
    loader = _new_loader(local)
    assert loader._source_metadata_identity_compatible(_build_mx_source_metadata(source)) is True


def test_gate_falls_back_on_mismatch():
    local = _identity(attn_backend="TRTLLM")
    source = _identity(attn_backend="FLASHINFER")
    loader = _new_loader(local)
    assert loader._source_metadata_identity_compatible(_build_mx_source_metadata(source)) is False


def test_gate_falls_back_on_checkpoint_artifact_mismatch():
    local = _identity(artifact_key="fine-tune-a")
    source = _identity(artifact_key="fine-tune-b")
    loader = _new_loader(local)
    assert loader._source_metadata_identity_compatible(_build_mx_source_metadata(source)) is False


def test_gate_falls_back_when_no_local_identity():
    # MX must not consume shared weights unless the receiver identity exists.
    loader = _new_loader(None)
    assert (
        loader._source_metadata_identity_compatible(_build_mx_source_metadata(_identity())) is False
    )


def test_gate_falls_back_when_source_identity_unavailable():
    loader = _new_loader(_identity())
    assert loader._source_metadata_identity_compatible(None) is False


def test_fetch_source_metadata_supports_modelexpress_0_4_1_client_shape_and_close_failure():
    local = _identity()
    loader = MXCheckpointLoader.__new__(MXCheckpointLoader)
    loader._local_source_identity = local
    loader._mx_server_url = "http://mx:8001"
    loader._model_name = None

    class _Client:
        def __init__(self, *, server_url):
            self.server_url = server_url

        def get_metadata(self, mx_source_id, worker_id):
            raise AssertionError("ID-based metadata lookup should not be used for identity queries")

        def list_sources(self, *, identity):
            return SimpleNamespace(
                instances=[SimpleNamespace(mx_source_id="source", worker_id="worker")]
            )

        def close(self):
            raise RuntimeError("close failed")

    metadata = loader._fetch_source_metadata(
        "ckpt",
        _Client,
        lambda **_kw: SimpleNamespace(extra_parameters={}),
    )

    assert metadata == _build_mx_source_metadata(local)


def test_load_weights_pops_source_identity_kwarg():
    # source_identity must be consumed by load_weights and never leak into the
    # HfCheckpointLoader disk-fallback signature. With no server URL configured
    # the loader takes the disk-fallback path; we only assert the kwarg is
    # stored and not forwarded.
    loader = MXCheckpointLoader.__new__(MXCheckpointLoader)
    loader._mx_server_url = None
    loader._p2p_succeeded = False
    captured = {}

    def fake_fallback(checkpoint_dir, mapping, *, reason=None, **kwargs):
        captured["kwargs"] = kwargs
        return {}

    loader._fallback_to_disk = fake_fallback
    identity = _identity()
    loader.load_weights("ckpt", mapping=FakeMapping(), model=None, source_identity=identity)
    assert loader._local_source_identity is identity
    assert "source_identity" not in captured["kwargs"]
    assert "model" not in captured["kwargs"]
