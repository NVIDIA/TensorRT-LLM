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
"""DeepSeek-V4.1 decoder SWA bounded replay: the runtime half.

The model runs decoder replay; the V4.1 cache manager resolves where it may run.
Its window KV of the layers after the last KV-source layer holds each prompt's last
``window`` positions only, so with block reuse on it needs encoder replay, which
re-forwards a hit's tail rows. Under decoder replay the manager also refuses, at
admission, the context logits and prompt logprobs those layers never compute. It admits
every prompt length the engine admits, and so it needs no second check when a request
paused for recompute comes back with its generated tokens in its prompt.
"""

import os
from unittest.mock import MagicMock

import pytest
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import (
    cache_manager as v4_cache_manager,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.cache_manager import (
    _v41_decoder_replay_refusal,
    _v41_encoder_replay_refusal,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.v41_layout import (
    DEEPSEEK_V41_DECODER_REPLAY_ENV,
    DEEPSEEK_V41_ENCODER_REPLAY_ENV,
    get_deepseek_v41_decoder_replay_setting,
    get_deepseek_v41_layout,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestType
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType
from tensorrt_llm.bindings import SamplingConfig
from tensorrt_llm.mapping import Mapping

from .test_deepseek_v41_encoder_replay import make_manager, prompt
from .test_deepseek_v41_layout import WINDOW, make_v41_config

# candidate_topk_blocks x candidate_block_size of the checkpoint (2048 x 8): past this
# many visible positions layer 20's candidate blocks bind.
CANDIDATE_POSITIONS = 16384


def test_switch_and_refusal():
    """The switch parses like the encoder one; with block reuse on, decoder replay is
    refused exactly when encoder replay is off, naming why encoder replay is off."""
    saved = os.environ.pop(DEEPSEEK_V41_DECODER_REPLAY_ENV, None)
    try:
        assert get_deepseek_v41_decoder_replay_setting() is None
        for value, expected in (("1", True), ("on", True), ("0", False), ("No", False)):
            os.environ[DEEPSEEK_V41_DECODER_REPLAY_ENV] = value
            assert get_deepseek_v41_decoder_replay_setting() is expected
        os.environ[DEEPSEEK_V41_DECODER_REPLAY_ENV] = "maybe"
        with pytest.raises(ValueError, match=DEEPSEEK_V41_DECODER_REPLAY_ENV):
            get_deepseek_v41_decoder_replay_setting()
    finally:
        os.environ.pop(DEEPSEEK_V41_DECODER_REPLAY_ENV, None)
        if saved is not None:
            os.environ[DEEPSEEK_V41_DECODER_REPLAY_ENV] = saved

    layout = get_deepseek_v41_layout(make_v41_config())
    assert _v41_decoder_replay_refusal(layout, True, None) is None
    assert _v41_decoder_replay_refusal(layout, False, "a KV cache connector") is None
    refusal = _v41_decoder_replay_refusal(layout, True, "a KV cache connector")
    assert refusal is not None and "block reuse" in refusal and "a KV cache connector" in refusal
    assert f"last {WINDOW} positions" in refusal
    # A KV connector turns encoder replay off (a manager with a stand-in connector is not
    # built); the manager passes its short reason on to the decoder refusal.
    _, reason = _v41_encoder_replay_refusal(
        layout, Mapping(), 8192, {"kv_connector_manager": object()}
    )
    assert reason.split(": ", 1)[0] == "a KV cache connector"


# (encoder switch, block reuse, manager kwargs, encoder replay resolved, why encoder
# replay is off as the decoder line and error name it -- None where decoder replay may run)
RESOLUTION_CASES = {
    "tp": (None, True, {}, True, None),
    "encoder_off_reuse_off": ("0", False, {}, False, None),
    "encoder_off": ("0", True, {}, False, f"{DEEPSEEK_V41_ENCODER_REPLAY_ENV}=0"),
    "disagg": (None, True, {"is_disagg": True}, False, "disaggregated serving"),
    "disagg_reuse_off": (None, False, {"is_disagg": True}, False, None),
    "small_token_budget": (
        None,
        True,
        {"max_num_tokens": WINDOW},
        False,
        f"max_num_tokens <= {WINDOW} (got {WINDOW})",
    ),
    # Attention DP: each rank's manager replays its own requests' hits, and the model
    # bounds its own requests; both replays run there as under attention TP.
    "attention_dp": (
        None,
        True,
        {"mapping": Mapping(world_size=1, rank=0, tp_size=1, enable_attention_dp=True)},
        True,
        None,
    ),
}


@skip_pre_blackwell
@pytest.mark.parametrize("case", list(RESOLUTION_CASES))
def test_resolves_with_encoder_replay(case, capfd, monkeypatch):
    """Ruling R21 for decoder replay: with block reuse on it needs encoder replay. Unset,
    it resolves on (attention TP or DP) unless reuse is on and encoder replay is off, then
    off with one line naming why; an explicit 1 there raises before any pool exists. An
    explicit 0 is always off."""
    encoder_switch, reuse, kwargs, encoder_on, why = RESOLUTION_CASES[case]
    monkeypatch.setattr(v4_cache_manager, "_V41_DECODER_REPLAY_ANNOUNCED", set())
    refused = why is not None

    def build(decoder_switch):
        if decoder_switch is None:
            monkeypatch.delenv(DEEPSEEK_V41_DECODER_REPLAY_ENV, raising=False)
        else:
            monkeypatch.setenv(DEEPSEEK_V41_DECODER_REPLAY_ENV, decoder_switch)
        manager, _ = make_manager(encoder_on, switch=encoder_switch, block_reuse=reuse, **kwargs)
        manager.shutdown()
        return manager

    capfd.readouterr()
    manager = build(None)
    assert (manager.v41_encoder_replay_off_reason is None) is encoder_on
    assert manager.v41_decoder_replay is not refused, case
    lines = [
        line
        for line in capfd.readouterr().out.splitlines()
        if DEEPSEEK_V41_DECODER_REPLAY_ENV in line
    ]
    if refused:
        assert len(lines) == 1, lines
        assert "resolves off" in lines[0] and why in lines[0] and "block reuse" in lines[0]
        # Once per process: the engine builds a KV-size estimation manager first.
        assert build(None).v41_decoder_replay is False
        assert DEEPSEEK_V41_DECODER_REPLAY_ENV not in capfd.readouterr().out
    else:
        assert lines == [], lines

    if refused:
        with pytest.raises(ValueError, match=DEEPSEEK_V41_DECODER_REPLAY_ENV) as error:
            build("1")
        assert why in str(error.value)
    else:
        assert build("1").v41_decoder_replay is True
    assert build("0").v41_decoder_replay is False
    assert DEEPSEEK_V41_DECODER_REPLAY_ENV not in capfd.readouterr().out


def admission_request(
    request_id: int, *, context_logits: bool, generation_only: bool = False, length: int = 300
):
    request = LlmRequest(
        request_id=request_id,
        max_new_tokens=8,
        input_tokens=prompt(request_id, length),
        sampling_config=SamplingConfig(),
        is_streaming=False,
        return_context_logits=context_logits,
    )
    if generation_only:
        request.py_llm_request_type = LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY
    return request


def admit(resource_manager: ResourceManager, request: LlmRequest) -> None:
    """PyExecutor's own admission check, with everything but the resource managers stubbed."""
    executor = MagicMock()
    executor.kv_cache_transceiver = None
    executor.max_beam_width = 1
    executor.resource_manager = resource_manager
    PyExecutor._validate_request(executor, request)


# (encoder switch, decoder switch, block reuse, decoder replay resolved)
ADMISSION_CASES = {
    "replay_on": (None, None, True, True),
    "switch_off": (None, "0", True, False),
    "resolved_off": ("0", None, True, False),
}


@skip_pre_blackwell
@pytest.mark.parametrize("case", list(ADMISSION_CASES))
def test_admission_refuses_context_logits_under_replay(case, monkeypatch):
    """Under decoder replay a request for context logits or prompt logprobs (which the
    executor serves through context logits) fails at admission with a message naming the
    switch, instead of raising inside the forward, which fails the whole step. Other
    requests, a generation-only request (its context ran on another engine) and every
    request without decoder replay are admitted."""
    encoder_switch, decoder_switch, reuse, decoder_on = ADMISSION_CASES[case]
    if decoder_switch is None:
        monkeypatch.delenv(DEEPSEEK_V41_DECODER_REPLAY_ENV, raising=False)
    else:
        monkeypatch.setenv(DEEPSEEK_V41_DECODER_REPLAY_ENV, decoder_switch)
    manager, _ = make_manager(encoder_switch is None, switch=encoder_switch, block_reuse=reuse)
    try:
        assert manager.v41_decoder_replay is decoder_on
        resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: manager})
        admit(resource_manager, admission_request(1, context_logits=False))
        admit(resource_manager, admission_request(2, context_logits=True, generation_only=True))
        refused = admission_request(3, context_logits=True)
        if decoder_on:
            with pytest.raises(
                ValueError, match="return_context_logits / prompt_logprobs"
            ) as error:
                admit(resource_manager, refused)
            assert f"{DEEPSEEK_V41_DECODER_REPLAY_ENV}=0" in str(error.value)
            assert f"last {WINDOW} positions" in str(error.value)
        else:
            admit(resource_manager, refused)
    finally:
        manager.shutdown()


@skip_pre_blackwell
@pytest.mark.parametrize("decoder_switch", [None, "0"], ids=["replay_on", "switch_off"])
def test_admission_admits_long_and_recomputed_prompts(decoder_switch, monkeypatch):
    """Decoder replay serves every prompt length the engine admits, past the candidate
    blocks' 2048 x 8 positions too, so admission refuses none by length. A request paused
    for recompute comes back with its generated tokens in its prompt
    (``PyExecutor._pause_recompute_request``); admitting it again gives the same answer."""
    if decoder_switch is None:
        monkeypatch.delenv(DEEPSEEK_V41_DECODER_REPLAY_ENV, raising=False)
    else:
        monkeypatch.setenv(DEEPSEEK_V41_DECODER_REPLAY_ENV, decoder_switch)
    max_seq_len = 2 * CANDIDATE_POSITIONS
    manager, _ = make_manager(True, switch=None, max_seq_len=max_seq_len, max_batch_size=1)
    try:
        assert manager.v41_decoder_replay is (decoder_switch is None)
        resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: manager})
        for n, length in enumerate((CANDIDATE_POSITIONS, CANDIDATE_POSITIONS + 1, max_seq_len - 8)):
            admit(resource_manager, admission_request(n + 1, context_logits=False, length=length))

        prompt_len, generated = CANDIDATE_POSITIONS - 384, 1000
        request = LlmRequest(
            request_id=9,
            max_new_tokens=4000,
            input_tokens=prompt(9, prompt_len),
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )
        admit(resource_manager, request)
        for t in range(generated):
            request.add_new_token(t % 1000, 0)
        PyExecutor._pause_recompute_request(MagicMock(), request)
        assert request.prompt_len == prompt_len + generated > CANDIDATE_POSITIONS
        admit(resource_manager, request)
    finally:
        manager.shutdown()


def test_resource_manager_validate_request_skips_managers_without_the_hook():
    """Resource managers that predate the hook, or unset slots, are skipped."""
    seen = []

    class Hooked:
        def validate_request(self, request):
            seen.append(request)

    resource_manager = ResourceManager(
        {
            ResourceManagerType.SEQ_SLOT_MANAGER: object(),
            ResourceManagerType.SPEC_RESOURCE_MANAGER: None,
            ResourceManagerType.KV_CACHE_MANAGER: Hooked(),
        }
    )
    resource_manager.validate_request("request")
    assert seen == ["request"]
