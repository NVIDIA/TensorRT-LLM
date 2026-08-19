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
"""The Kimi K3 target's spec-dec mode gate.

``KimiLinearForCausalLM.__init__`` whitelists the speculative-decoding modes the
target will serve. It is a plain assert on the *target* side, so nothing that
exercises the drafter, the builder or the worker can reach it -- which is how a
K3 engine configured with ``decoding_type: DSpark`` got through every unit test
and then failed at model construction with

    AssertionError: Kimi K3 supports speculative decoding only with SA or DFlash

The gate is the second statement in ``__init__``, and everything past it builds
the real K3 model, which needs weights and 16 GPUs. So rather than stub the
framework out from under it -- the base initializer's arguments construct
``KimiLinearModel`` before the base is even called, so stubbing the base does
not help -- these tests ask a narrower question: did construction fail *at the
gate*, or did it get past it? Anything that fails later has passed the gate,
which is the whole of what is under test here.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.models.modeling_kimi_linear import KimiLinearForCausalLM
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode

# Admitted: SA drafts in-forward with no draft weights; DFlash and DSpark are
# the external-drafter flow, and the target side is identical for both (the
# hidden-state capture in KimiLinearModel.forward is unconditional).
ADMITTED = [
    SpeculativeDecodingMode.SA,
    SpeculativeDecodingMode.DFLASH,
    SpeculativeDecodingMode.DSPARK,
]
# Refused: these need draft heads that no K3 checkpoint ships.
REFUSED = [
    SpeculativeDecodingMode.MTP,
    SpeculativeDecodingMode.EAGLE3_ONE_MODEL,
]

_SPEC_GATE = "speculative decoding"
_PP_GATE = "pipeline parallelism"


def _model_config(mode, *, pp_size=1):
    """The minimum ``__init__`` reads before it reaches the gate."""
    return SimpleNamespace(
        pretrained_config=SimpleNamespace(model_type="kimi_linear", linear_attn_config={}),
        mapping=SimpleNamespace(pp_size=pp_size),
        spec_config=None if mode is None else SimpleNamespace(spec_dec_mode=mode),
    )


def _rejected_by(model_config) -> str | None:
    """Which gate rejected this config, or None if construction got past them.

    Only the two guard asserts count as a rejection; construction is expected
    to fail afterwards on the real model, and that failure means the config was
    admitted. An AssertionError from anywhere else is a genuine problem and is
    re-raised rather than silently read as a rejection.
    """
    try:
        KimiLinearForCausalLM(model_config)
    except AssertionError as exc:
        message = str(exc)
        for gate in (_SPEC_GATE, _PP_GATE):
            if gate in message:
                return gate
        raise
    except Exception:
        return None
    return None


@pytest.mark.parametrize("mode", ADMITTED, ids=lambda m: m.name)
def test_admitted_modes_pass_the_gate(mode):
    assert _rejected_by(_model_config(mode)) is None


def test_no_spec_config_passes_the_gate():
    assert _rejected_by(_model_config(None)) is None


@pytest.mark.parametrize("mode", REFUSED, ids=lambda m: m.name)
def test_refused_modes_are_rejected_at_the_spec_gate(mode):
    assert _rejected_by(_model_config(mode)) == _SPEC_GATE


def test_the_refusal_message_names_the_admitted_modes():
    # The message is the only guidance a user gets, so it has to name the modes
    # that would work -- that is what turns "not supported" into an action.
    with pytest.raises(AssertionError, match="SA, DFlash or DSpark"):
        KimiLinearForCausalLM(_model_config(SpeculativeDecodingMode.MTP))


def test_pipeline_parallelism_is_still_rejected():
    # The pp guard sits ahead of the spec gate; pin the order so a future edit
    # to the mode list cannot let a pp>1 config through.
    config = _model_config(SpeculativeDecodingMode.DSPARK, pp_size=2)
    assert _rejected_by(config) == _PP_GATE
