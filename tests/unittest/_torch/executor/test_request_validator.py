# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor import request_utils


class DummyRequest:
    def __init__(self, beam_width: Optional[int] = 1):
        self.py_multimodal_data = None
        if beam_width is None:
            self.sampling_config = None
        else:
            self.sampling_config = SimpleNamespace(beam_width=beam_width)

    def check_token_id_range(self, _):
        return True


def test_validate_request_rejects_beam_width_mismatch():
    model_engine = SimpleNamespace(model=object())
    sampler = Mock()
    validator = request_utils.RequestValidator(
        model_engine=model_engine,
        sampler=sampler,
        max_beam_width=1,
    )

    with pytest.raises(ValueError, match="max_beam_width"):
        validator.validate_request(DummyRequest(beam_width=2))

    sampler.validate_request.assert_not_called()


def test_validate_requests_collects_failures():
    model_engine = SimpleNamespace(model=object())
    sampler = Mock()

    def _validate_request_side_effect(req):
        if req.sampling_config is None:
            raise RuntimeError("sampler validation failure")

    sampler.validate_request.side_effect = _validate_request_side_effect
    validator = request_utils.RequestValidator(
        model_engine=model_engine,
        sampler=sampler,
        max_beam_width=1,
    )

    valid_request = DummyRequest(beam_width=1)
    invalid_request = DummyRequest(beam_width=None)

    validated, failures = validator.validate_requests([valid_request, invalid_request])

    assert validated == [valid_request]
    assert len(failures) == 1
    assert failures[0].request is invalid_request
    assert failures[0].error_msg == "sampler validation failure"
