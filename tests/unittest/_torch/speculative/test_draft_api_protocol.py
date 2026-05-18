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
"""Byte-level / semantic tests for ``draft_api_protocol``.

These tests must keep wire compatibility with the izzy reference.  The
``GOLDEN_*`` hex blobs were generated from
``/home/scratch.zhaoyangw_gpu/repo/tekit/doca_rdma/standalone/protocol.py``
— do not regenerate them when the local encoder changes; instead fix the
encoder so it matches the golden bytes again.
"""

import pytest

from tensorrt_llm._torch.speculative.draft_api_protocol import DraftApiProtocol

P = DraftApiProtocol
S = DraftApiProtocol.Status
MT = DraftApiProtocol.MessageType


# Single-string literals — Ruff line-length is suppressed because the golden
# values must be byte-for-byte equal to izzy's encoder output (one 192-char
# hex per 96-byte packet); breaking them across lines invites copy errors.
GOLDEN_T2D_42_100_5421 = "2a0000006400000001010100000000002d15000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"  # noqa: E501
GOLDEN_D2T_7_10_5TOK = "070000000a000000010005000000000064000000c80000002c01000090010000f4010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"  # noqa: E501
GOLDEN_D2T_MAX = "fffffffffeffffff0100140000000000000000000100000002000000030000000400000005000000060000000700000008000000090000000a0000000b0000000c0000000d0000000e0000000f00000010000000110000001200000013000000"  # noqa: E501


def test_message_size_constant():
    assert P.kMessageBytes == 96
    assert P.kMaxTokens == 20


def test_encode_decode_roundtrip_t2d():
    msg = P.Message(
        message_type=MT.kTargetToDraft,
        round_seq_num=42,
        position=100,
        num_tokens=1,
        tokens=[5421] + [0] * 19,
    )
    status, payload = P.encode(msg)
    assert status == S.kOk
    assert len(payload) == P.kMessageBytes

    status2, decoded = P.decode(payload)
    assert status2 == S.kOk
    assert decoded.tokens[0] == 5421
    assert decoded.round_seq_num == 42
    assert decoded.position == 100
    assert decoded.num_tokens == 1
    assert decoded.message_type == int(MT.kTargetToDraft)


def test_encode_decode_roundtrip_d2t():
    msg = P.Message(
        message_type=MT.kDraftToTarget,
        round_seq_num=7,
        position=10,
        num_tokens=5,
        tokens=[100, 200, 300, 400, 500] + [0] * 15,
    )
    status, payload = P.encode(msg)
    assert status == S.kOk

    status2, decoded = P.decode(payload)
    assert status2 == S.kOk
    assert decoded.num_tokens == 5
    assert decoded.tokens[:5] == [100, 200, 300, 400, 500]


@pytest.mark.parametrize(
    "msg_kwargs,expected_hex",
    [
        (
            dict(
                message_type=MT.kTargetToDraft,
                round_seq_num=42,
                position=100,
                num_tokens=1,
                tokens=[5421] + [0] * 19,
            ),
            GOLDEN_T2D_42_100_5421,
        ),
        (
            dict(
                message_type=MT.kDraftToTarget,
                round_seq_num=7,
                position=10,
                num_tokens=5,
                tokens=[100, 200, 300, 400, 500] + [0] * 15,
            ),
            GOLDEN_D2T_7_10_5TOK,
        ),
        (
            dict(
                message_type=MT.kDraftToTarget,
                round_seq_num=0xFFFFFFFF,
                position=0xFFFFFFFE,
                num_tokens=20,
                tokens=list(range(20)),
            ),
            GOLDEN_D2T_MAX,
        ),
    ],
)
def test_byte_compat_with_izzy(msg_kwargs, expected_hex):
    """Encoder must produce byte-identical output to izzy reference."""
    msg = P.Message(**msg_kwargs)
    status, payload = P.encode(msg)
    assert status == S.kOk
    assert payload.hex() == expected_hex


def test_validation_target_to_draft_must_be_1_token():
    msg = P.Message(
        message_type=MT.kTargetToDraft,
        round_seq_num=1,
        position=0,
        num_tokens=2,
        tokens=[1, 2] + [0] * 18,
    )
    assert P.validate(msg) == S.kInvalidNumTokens


def test_validation_draft_to_target_zero_tokens_invalid():
    msg = P.Message(
        message_type=MT.kDraftToTarget,
        round_seq_num=1,
        position=0,
        num_tokens=0,
        tokens=[0] * 20,
    )
    assert P.validate(msg) == S.kInvalidNumTokens


def test_validation_position_overflow():
    options = P.ValidationOptions(max_position=99)
    msg = P.Message(
        message_type=MT.kTargetToDraft,
        round_seq_num=1,
        position=100,
        num_tokens=1,
        tokens=[42] + [0] * 19,
    )
    assert P.validate(msg, options) == S.kInvalidPosition


def test_validation_unsupported_version():
    msg = P.Message(
        version=99,
        message_type=MT.kTargetToDraft,
        round_seq_num=1,
        position=0,
        num_tokens=1,
        tokens=[1] + [0] * 19,
    )
    assert P.validate(msg) == S.kUnsupportedVersion


def test_decode_size_mismatch():
    status, decoded = P.decode(b"\x00" * 95)
    assert status == S.kInvalidSize
    assert decoded is None


def test_decode_reserved_must_be_zero():
    msg = P.Message(
        message_type=MT.kTargetToDraft,
        round_seq_num=1,
        position=0,
        num_tokens=1,
        tokens=[42] + [0] * 19,
    )
    status, payload = P.encode(msg)
    assert status == S.kOk
    # Stamp a non-zero byte into the reserved range (offset 11..15).
    corrupted = bytearray(payload)
    corrupted[P.kReservedOffset] = 0xAA
    status2, decoded = P.decode(bytes(corrupted))
    assert status2 == S.kReservedFieldNonZero
    assert decoded is None


def test_decode_reserved_check_can_be_disabled():
    msg = P.Message(
        message_type=MT.kTargetToDraft,
        round_seq_num=1,
        position=0,
        num_tokens=1,
        tokens=[42] + [0] * 19,
    )
    _, payload = P.encode(msg)
    corrupted = bytearray(payload)
    corrupted[P.kReservedOffset] = 0xAA
    options = P.ValidationOptions(enforce_reserved_zeros=False)
    status, decoded = P.decode(bytes(corrupted), options)
    assert status == S.kOk
    assert decoded.num_tokens == 1


def test_to_string_known_values():
    assert P.to_string(S.kOk) == "kOk"
    assert P.to_string(S.kInvalidSize) == "kInvalidSize"
    # Out-of-range ints fall back through ``_status`` to ``kInvalidBuffer``;
    # this matches izzy's behaviour and is what callers see on encode error.
    assert P.to_string(99) == "kInvalidBuffer"
