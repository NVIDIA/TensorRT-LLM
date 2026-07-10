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

These tests drive `MXCheckpointLoader._source_identity_compatible` directly —
the single decision point the MX `load_weights` path consults before starting
a P2P transfer. Upstream `modelexpress` is never imported: the discovery
client / identity builder are passed as stubs, and the publisher-identity fetch
seam is patched, so the gate logic runs against real `SourceIdentity` objects
without any model, GPU, or RDMA.
"""

from types import SimpleNamespace

from _source_identity_fakes import FakeMapping
from _source_identity_fakes import make_identity as _identity

from tensorrt_llm._torch.models.checkpoints.mx.checkpoint_loader import (
    MXCheckpointLoader,
    _build_mx_source_metadata,
)


def _new_loader(local_identity, source_identity, fetched=True):
    """Construct a loader bypassing heavy base __init__, wire the seams."""
    loader = MXCheckpointLoader.__new__(MXCheckpointLoader)
    loader._local_source_identity = local_identity
    # Patch the single fetch seam to return the publisher identity (or None).
    loader._fetch_source_identity = lambda *a, **k: source_identity if fetched else None
    return loader


# MxClient / build_identity are only forwarded to the (patched) fetch seam.
_STUB_CLIENT = object()
_STUB_BUILD = object()


def test_gate_proceeds_on_matching_identity():
    local = _identity(attn_backend="TRTLLM")
    source = _identity(attn_backend="TRTLLM")
    loader = _new_loader(local, source)
    assert loader._source_identity_compatible("ckpt", _STUB_CLIENT, _STUB_BUILD) is True


def test_gate_falls_back_on_mismatch():
    local = _identity(attn_backend="TRTLLM")
    source = _identity(attn_backend="FLASHINFER")
    loader = _new_loader(local, source)
    assert loader._source_identity_compatible("ckpt", _STUB_CLIENT, _STUB_BUILD) is False


def test_gate_falls_back_when_no_local_identity():
    # MX must not consume shared weights unless the receiver identity exists.
    loader = _new_loader(None, _identity())
    assert loader._source_identity_compatible("ckpt", _STUB_CLIENT, _STUB_BUILD) is False


def test_gate_falls_back_when_source_identity_unavailable():
    # Publisher identity not yet fetchable (upstream metadata channel pending);
    # reject P2P and fall back to disk rather than sharing unverified weights.
    local = _identity()
    loader = _new_loader(local, None, fetched=False)
    assert loader._source_identity_compatible("ckpt", _STUB_CLIENT, _STUB_BUILD) is False


def test_fetch_source_identity_returns_none_when_metadata_unavailable():
    loader = MXCheckpointLoader.__new__(MXCheckpointLoader)
    loader._mx_server_url = "http://mx:8001"
    loader._model_name = None

    class _Client:
        def __init__(self, *, server_url):
            self.server_url = server_url

        def list_sources(self, *, identity):
            return SimpleNamespace(instances=[])

    assert loader._fetch_source_identity("ckpt", _Client, lambda **_kw: object()) is None


def test_fetch_source_identity_from_source_metadata():
    source = _identity()
    loader = MXCheckpointLoader.__new__(MXCheckpointLoader)
    loader._mx_server_url = "http://mx:8001"
    loader._model_name = None

    class _Client:
        def __init__(self, *, server_url):
            self.server_url = server_url

        def list_sources(self, *, identity):
            instance = SimpleNamespace(metadata=_build_mx_source_metadata(source))
            return SimpleNamespace(instances=[instance])

    assert loader._fetch_source_identity("ckpt", _Client, lambda **_kw: object()) == source


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
