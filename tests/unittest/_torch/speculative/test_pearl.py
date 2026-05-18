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
"""Unit tests for PEARL speculative decoding wiring.

CPU-only tests covering:
- PEARL message types encode/decode in the 96-byte envelope
- ``PEARLDecodingConfig`` validation and ``spec_dec_mode`` dispatch
- ``SpeculativeDecodingMode.PEARL_ONE_MODEL`` predicates

End-to-end GPU tests live alongside the existing draft-target tests and
are run via ``examples/llm-api/rdma/spec_dec_pearl_target_main.py``
against ``examples/llm-api/rdma/pearl_draft_server.py``.
"""

import pytest

from tensorrt_llm._torch.speculative.draft_api_protocol import DraftApiProtocol
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode

P = DraftApiProtocol
S = DraftApiProtocol.Status
MT = DraftApiProtocol.MessageType


# -------------------- Protocol tests --------------------


def test_pearl_message_types_exist():
    assert int(MT.kPearlVerifyContinue) == 2
    assert int(MT.kPearlPreVerifyToken) == 3
    assert int(MT.kPearlDraftBatch) == 4
    assert int(MT.kPearlRollback) == 5
    assert int(MT.kPearlProbe) == 6


def test_pearl_verify_continue_roundtrip():
    msg = P.Message(
        message_type=MT.kPearlVerifyContinue,
        round_seq_num=11,
        position=23,
        num_tokens=4,  # tokens[0]=gamma_next; tokens[1:4]=accepted state
        tokens=[5, 100, 200, 300] + [0] * 16,
    )
    status, payload = P.encode(msg)
    assert status == S.kOk
    assert len(payload) == P.kMessageBytes

    status2, decoded = P.decode(payload)
    assert status2 == S.kOk
    assert decoded.message_type == int(MT.kPearlVerifyContinue)
    assert decoded.tokens[0] == 5
    assert decoded.tokens[1:4] == [100, 200, 300]
    assert decoded.num_tokens == 4


def test_pearl_pre_verify_token_roundtrip():
    msg = P.Message(
        message_type=MT.kPearlPreVerifyToken,
        round_seq_num=17,
        position=42,
        num_tokens=1,
        tokens=[999] + [0] * 19,
    )
    status, payload = P.encode(msg)
    assert status == S.kOk
    status2, decoded = P.decode(payload)
    assert status2 == S.kOk
    assert decoded.message_type == int(MT.kPearlPreVerifyToken)
    assert decoded.tokens[0] == 999


def test_pearl_pre_verify_token_rejects_wrong_num_tokens():
    # PEARL pre-verify must carry exactly 1 token.
    msg = P.Message(
        message_type=MT.kPearlPreVerifyToken,
        round_seq_num=1,
        position=1,
        num_tokens=2,
        tokens=[1, 2] + [0] * 18,
    )
    assert P.validate(msg) == S.kInvalidNumTokens


def test_pearl_draft_batch_roundtrip():
    msg = P.Message(
        message_type=MT.kPearlDraftBatch,
        round_seq_num=5,
        position=0,
        num_tokens=6,
        tokens=list(range(100, 106)) + [0] * 14,
    )
    status, payload = P.encode(msg)
    assert status == S.kOk
    status2, decoded = P.decode(payload)
    assert status2 == S.kOk
    assert decoded.tokens[:6] == list(range(100, 106))


def test_pearl_rollback_roundtrip():
    msg = P.Message(
        message_type=MT.kPearlRollback,
        round_seq_num=3,
        position=2,  # rollback_count = 2
        num_tokens=1,
        tokens=[42] + [0] * 19,  # tokens[0] = corrected token
    )
    status, payload = P.encode(msg)
    assert status == S.kOk
    status2, decoded = P.decode(payload)
    assert status2 == S.kOk
    assert decoded.message_type == int(MT.kPearlRollback)
    assert decoded.position == 2
    assert decoded.tokens[0] == 42


def test_pearl_probe_allows_zero_tokens():
    msg = P.Message(
        message_type=MT.kPearlProbe,
        round_seq_num=0,
        position=0,
        num_tokens=0,
        tokens=[0] * 20,
    )
    status, payload = P.encode(msg)
    assert status == S.kOk
    status2, decoded = P.decode(payload)
    assert status2 == S.kOk
    assert decoded.num_tokens == 0


def test_existing_protocol_still_works():
    """PEARL extensions must not break the original kDraftToTarget /
    kTargetToDraft validation rules."""
    msg = P.Message(
        message_type=MT.kTargetToDraft,
        round_seq_num=42,
        position=100,
        num_tokens=1,
        tokens=[5421] + [0] * 19,
    )
    status, payload = P.encode(msg)
    assert status == S.kOk
    status2, decoded = P.decode(payload)
    assert status2 == S.kOk

    # T2D must still reject num_tokens != 1
    bad = P.Message(
        message_type=MT.kTargetToDraft,
        round_seq_num=0,
        position=0,
        num_tokens=2,
        tokens=[1, 2] + [0] * 18,
    )
    assert P.validate(bad) == S.kInvalidNumTokens


# -------------------- Mode enum predicates --------------------


def test_pearl_mode_enum():
    mode = SpeculativeDecodingMode.PEARL_ONE_MODEL
    assert mode.is_pearl_one_model()
    assert mode.is_external_drafter()  # PEARL is an external drafter
    assert not mode.is_draft_target_one_model()
    assert not mode.is_pard()


def test_draft_target_mode_unchanged():
    mode = SpeculativeDecodingMode.DRAFT_TARGET_ONE_MODEL
    assert mode.is_draft_target_one_model()
    assert mode.is_external_drafter()
    assert not mode.is_pearl_one_model()


# -------------------- Config validation --------------------


def test_pearl_config_dispatch_to_mode():
    from tensorrt_llm.llmapi import PEARLDecodingConfig

    cfg = PEARLDecodingConfig(
        max_draft_len=4,
        speculative_model="dummy/path",
        draft_offload_enabled=True,
        draft_offload_v2=True,
        draft_offload_server_host="127.0.0.1",
        draft_offload_server_port=47000,
    )
    assert cfg.spec_dec_mode == SpeculativeDecodingMode.PEARL_ONE_MODEL
    assert cfg.decoding_type == "PEARL"


def test_pearl_config_auto_enables_offload():
    """PEARL is meaningless without RDMA offload — the validator should
    force-enable draft_offload_enabled and draft_offload_v2."""
    from tensorrt_llm.llmapi import PEARLDecodingConfig

    cfg = PEARLDecodingConfig(
        max_draft_len=4,
        speculative_model="dummy/path",
        draft_offload_server_host="127.0.0.1",
        draft_offload_server_port=47000,
    )
    assert cfg.draft_offload_enabled is True
    assert cfg.draft_offload_v2 is True


def test_pearl_config_inherits_draft_target_fields():
    from tensorrt_llm.llmapi import DraftTargetDecodingConfig, PEARLDecodingConfig

    cfg = PEARLDecodingConfig(
        max_draft_len=4,
        speculative_model="dummy/path",
        draft_offload_enabled=True,
        draft_offload_v2=True,
        draft_offload_server_host="127.0.0.1",
        draft_offload_server_port=47000,
    )
    # PEARL is a subclass of DraftTarget — isinstance must still match.
    assert isinstance(cfg, DraftTargetDecodingConfig)
    # PEARL-specific fields default to expected values.
    assert cfg.pearl_enable_pre_verify is True
    assert cfg.pearl_enable_post_verify is True
    assert cfg.pearl_adaptive_gamma is True
    assert cfg.pearl_gamma_profile_batch_sizes == [1, 2, 4, 8, 16, 32]


# -------------------- PEARL worker basics (no GPU) --------------------


def test_pearl_worker_select_gamma():
    """``select_gamma_for_batch`` falls back to max_draft_len when no
    table is set, and uses the smallest profiled bs >= current bs once
    profiled."""
    # Direct import; the class needs torch but we only call pure-python
    # methods so the import works.
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not available")

    from tensorrt_llm._torch.speculative.pearl import PEARLOneModelWorker

    # Bypass __init__ — we only test pure-python helpers.
    w = PEARLOneModelWorker.__new__(PEARLOneModelWorker)
    w._adaptive_gamma_enabled = True
    w._gamma_table = {}
    w.spec_config = type("C", (), {"max_draft_len": 8})()
    # No table -> max_draft_len.
    assert w.select_gamma_for_batch(1) == 8

    # With table: smallest bs >= current_bs wins.
    w._gamma_table = {1: 5, 2: 4, 4: 3, 8: 2}
    assert w.select_gamma_for_batch(1) == 5
    assert w.select_gamma_for_batch(2) == 4
    assert w.select_gamma_for_batch(3) == 3  # 4 is smallest >= 3
    assert w.select_gamma_for_batch(8) == 2
    assert w.select_gamma_for_batch(16) == 2  # > all profiled -> largest
