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

import pytest
import torch

from tensorrt_llm.llmapi.llm import _WhisperSuppressTokensLogitsProcessor

pytestmark = pytest.mark.cpu_only

NEG_INF = float("-inf")


def _make(suppress=(3, 5), begin=(7,)):
    return _WhisperSuppressTokensLogitsProcessor(
        suppress_token_ids=list(suppress), begin_suppress_token_ids=list(begin)
    )


def _logits(vocab=10):
    return torch.zeros(1, vocab)


def test_suppress_masks_exactly_the_listed_tokens():
    proc = _make()
    logits = _logits()
    proc(req_id=1, logits=logits, token_ids=[[0, 1]], stream_ptr=None, client_id=None)

    masked = {i for i, v in enumerate(logits[0].tolist()) if v == NEG_INF}
    # 7 is begin-suppressed: the first callback sees exactly the prompt.
    assert masked == {3, 5, 7}


def test_begin_suppress_applies_only_while_the_window_is_open():
    proc = _make()
    # First call captures prompt_len=2 and applies begin-suppression.
    proc(req_id=1, logits=_logits(), token_ids=[[0, 1]], stream_ptr=None, client_id=None)
    # Second call is past the prompt: begin-suppression must not re-arm.
    logits = _logits()
    proc(req_id=1, logits=logits, token_ids=[[0, 1, 2]], stream_ptr=None, client_id=None)

    masked = {i for i, v in enumerate(logits[0].tolist()) if v == NEG_INF}
    assert masked == {3, 5}


def test_token_ids_are_immutable():
    """The id sequences must not be mutable in place.

    The device index cache is keyed on device alone, which is only sound
    because the ids it was built from cannot change afterwards.
    """
    proc = _make()
    assert isinstance(proc.suppress_token_ids, tuple)
    assert isinstance(proc.begin_suppress_token_ids, tuple)
    with pytest.raises(TypeError):
        proc.suppress_token_ids[0] = 99


def test_index_tensor_is_built_once_per_device():
    """Repeated calls must reuse one index tensor per device.

    This is the point of the cache: every rebuild is a blocking host-to-device
    copy on the decode path.
    """
    proc = _make()
    for step in range(3):
        proc(
            req_id=1,
            logits=_logits(),
            token_ids=[[0, 1] + [2] * step],
            stream_ptr=None,
            client_id=None,
        )

    device = torch.zeros(1).device
    assert list(proc._suppress_idx) == [device]
    first = proc._suppress_idx[device]
    proc(req_id=2, logits=_logits(), token_ids=[[0, 1]], stream_ptr=None, client_id=None)
    assert proc._suppress_idx[device] is first


def test_empty_lists_mask_nothing():
    proc = _WhisperSuppressTokensLogitsProcessor(suppress_token_ids=[], begin_suppress_token_ids=[])
    logits = _logits()
    proc(req_id=1, logits=logits, token_ids=[[0, 1]], stream_ptr=None, client_id=None)
    assert not torch.isinf(logits).any()
    assert proc._suppress_idx == {}


def test_masks_each_beam_independently():
    proc = _make(suppress=(3,), begin=())
    logits = torch.zeros(2, 10)
    proc(req_id=1, logits=logits, token_ids=[[0, 1], [0, 4]], stream_ptr=None, client_id=None)
    assert logits[0][3] == NEG_INF
    assert logits[1][3] == NEG_INF
    assert torch.isinf(logits).sum() == 2


def test_out_of_range_token_id_raises_every_call():
    """Caching must not turn a hard error into a one-shot one.

    An id outside the vocabulary has to fail on the second call exactly as it
    does on the first.
    """
    proc = _make(suppress=(99,), begin=())
    for _ in range(2):
        with pytest.raises(IndexError):
            proc(req_id=1, logits=_logits(), token_ids=[[0, 1]], stream_ptr=None, client_id=None)
