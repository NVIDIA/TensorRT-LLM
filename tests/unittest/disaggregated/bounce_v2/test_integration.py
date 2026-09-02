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
"""Integration smoke tests for the bounce_v2 wiring into the native transport.

Covers the transfer.py-level integration WITHOUT a full TransferWorker (which
needs a live KVCacheManager + model weights — too heavy for a unit test; noted
gap: the Sender path from _process_task_queue through should_use/submit is
exercised only via the engine-level tests in test_reactor_engine.py):

  - the ``TRTLLM_BOUNCE_V2_ENABLE`` env gate of ``create_bounce_v2_engine``
    (NoBounceEngine null object when off, a live engine + parsable handshake
    blob when on — construction errors must RAISE, not silently fall back);
  - the ``RankInfo.bounce_v2_handshake`` optional field: msgpack roundtrip,
    and backward compatibility with rank-info blobs from peers that predate
    the field (deserializes to None, exactly what add_peer treats as
    "bounce not advertised").

Same skip guards as test_mechanism_bindings.py (CUDA + compiled wheel).
"""

from __future__ import annotations

import uuid

import msgpack
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("bounce_v2 integration tests require a CUDA device", allow_module_level=True)

tab = pytest.importorskip(
    "tensorrt_llm.tensorrt_llm_transfer_agent_binding",
    reason="bounce_v2 integration tests require the compiled tensorrt_llm wheel",
)

from tensorrt_llm._torch.disaggregation.bounce_v2.codec import BOUNCE_VERSION  # noqa: E402
from tensorrt_llm._torch.disaggregation.bounce_v2.config import BounceV2Config  # noqa: E402
from tensorrt_llm._torch.disaggregation.bounce_v2.engine import (  # noqa: E402
    BOUNCE_V2_ENV,
    BounceEngine,
    NoBounceEngine,
    create_bounce_v2_engine,
)
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo  # noqa: E402

DEVICE = 0
GIB = 1 << 30


@pytest.fixture
def agent():
    name = f"bv2int_{uuid.uuid4().hex[:8]}"
    a = tab.NixlTransferAgent(tab.BaseAgentConfig(name))
    yield name, a
    torch.cuda.synchronize()
    a.shutdown()


def _rank_info(**overrides) -> RankInfo:
    fields = dict(
        instance_name="ctx",
        instance_rank=0,
        tp_size=1,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        layer_num_per_pp=[2],
        sender_endpoints=["tcp://127.0.0.1:1234"],
        self_endpoint="tcp://127.0.0.1:1235",
        transfer_engine_info=b"engine-desc",
    )
    fields.update(overrides)
    return RankInfo(**fields)


# --------------------------------------------------------------------------------------------
# 1. env gate (create_bounce_v2_engine)
# --------------------------------------------------------------------------------------------
class TestEnvGate:
    @pytest.mark.parametrize("value", [None, "0", "", "off", "no"])
    def test_disabled_returns_null_object(self, monkeypatch, value):
        if value is None:
            monkeypatch.delenv(BOUNCE_V2_ENV, raising=False)
        else:
            monkeypatch.setenv(BOUNCE_V2_ENV, value)
        # agent=None must be fine: the disabled path may not touch it (call
        # sites construct the engine before the binding-level agent is used).
        eng = create_bounce_v2_engine(None, DEVICE, "ctx0")
        assert isinstance(eng, NoBounceEngine)
        # Null-object contract: unconditional call sites never need checks.
        assert eng.local_handshake_blob() == b""
        assert eng.add_peer("p", b"blob") is False
        assert eng.has_peer("p") is False
        assert eng.should_use(None, "p") is False
        eng.forget_peer("p")
        eng.shutdown()  # idempotent no-op
        with pytest.raises(RuntimeError, match="disabled"):
            eng.submit()

    @pytest.mark.parametrize("value", ["1", "true", "True"])
    def test_enabled_constructs_live_engine(self, monkeypatch, agent, value):
        free, _total = torch.cuda.mem_get_info(DEVICE)
        if free < 4 * GIB:
            pytest.skip("factory uses the default 2 GiB arena; not enough free VRAM")
        monkeypatch.setenv(BOUNCE_V2_ENV, value)
        name, raw_agent = agent
        eng = create_bounce_v2_engine(raw_agent, DEVICE, name)
        try:
            assert isinstance(eng, BounceEngine)
            blob = eng.local_handshake_blob()
            parsed = BounceEngine._decode_handshake(blob)
            assert parsed is not None
            version, kind, chunk_cap, arena_cap, endpoint = parsed
            assert version == BOUNCE_VERSION
            assert chunk_cap == BounceV2Config(enabled=True).max_chunk_size_bytes
            assert arena_cap > 0
            assert endpoint.startswith("tcp://")
            # transfer.py wires the blob into rank info as `blob or None`:
            # a live engine's blob must therefore be non-empty (truthy).
            assert blob
        finally:
            eng.shutdown()

    def test_enabled_with_bad_agent_raises(self, monkeypatch):
        """The user explicitly opted in: a construction failure must RAISE.

        A silent NIXL fallback would be a silent ~1000x perf cliff.
        """
        monkeypatch.setenv(BOUNCE_V2_ENV, "1")
        with pytest.raises(RuntimeError, match="register_region"):
            create_bounce_v2_engine(None, DEVICE, "ctx0")
        with pytest.raises(RuntimeError, match="register_region"):
            create_bounce_v2_engine(object(), DEVICE, "ctx0")


# --------------------------------------------------------------------------------------------
# 2. RankInfo handshake field (the rank-info exchange transfer.py rides on)
# --------------------------------------------------------------------------------------------
class TestRankInfoHandshake:
    def test_roundtrip_with_handshake_blob(self):
        blob = b"\x48\x32\x56\x42" + bytes(range(32))  # arbitrary binary payload
        ri = _rank_info(bounce_v2_handshake=blob)
        back = RankInfo.from_bytes(ri.to_bytes())
        assert back.bounce_v2_handshake == blob
        assert back.instance_name == ri.instance_name
        assert back.transfer_engine_info == ri.transfer_engine_info

    def test_roundtrip_disabled_is_none(self):
        # transfer.py sets `local_handshake_blob() or None`, so the disabled
        # path serializes None (not b"").
        ri = _rank_info(bounce_v2_handshake=None)
        raw = ri.to_bytes()
        # Contract: a None handshake OMITS the key entirely — old peers decode
        # with cls(**unpacked) and would crash on an unknown key, so a new
        # rank with bounce off must emit an old-style blob (forward compat).
        assert "bounce_v2_handshake" not in msgpack.unpackb(raw, strict_map_key=False)
        back = RankInfo.from_bytes(raw)
        assert back.bounce_v2_handshake is None

    def test_backward_compat_old_peer_blob_without_field(self):
        """A rank-info blob predating the field must still deserialize.

        The handshake defaults to None — which the receiver's add_peer path
        treats as 'bounce not advertised' (NIXL fallback).
        """
        ri = _rank_info()
        old = msgpack.unpackb(ri.to_bytes(), strict_map_key=False)
        # to_bytes now already omits the None key; pop defensively so this
        # test keeps modeling an OLD peer's blob regardless.
        old.pop("bounce_v2_handshake", None)
        back = RankInfo.from_bytes(msgpack.packb(old))
        assert back.bounce_v2_handshake is None
        assert back.instance_name == ri.instance_name
        # And the sender-side consumer contract on such a value:
        assert NoBounceEngine().add_peer("peer", back.bounce_v2_handshake) is False

    def test_getattr_guard_matches_transfer_py_usage(self):
        """transfer.py reads the field via a getattr default guard.

        Verify both the present and the (simulated legacy-object) absent
        attribute shapes behave with getattr(ri, 'bounce_v2_handshake', None).
        """
        ri = _rank_info(bounce_v2_handshake=b"x")
        assert getattr(ri, "bounce_v2_handshake", None) == b"x"

        class LegacyRankInfo:  # a peer object without the attribute at all
            pass

        assert getattr(LegacyRankInfo(), "bounce_v2_handshake", None) is None
