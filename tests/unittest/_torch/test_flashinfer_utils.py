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

from unittest import mock

import pytest

from tensorrt_llm._torch import flashinfer_utils


@pytest.fixture(autouse=True)
def reset_pdl_log_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        flashinfer_utils.is_pdl_enabled,
        "_printed",
        False,
        raising=False,
    )


@pytest.mark.parametrize(
    ("env_value", "sm_version", "expected"),
    [
        (None, -1, False),
        (None, 89, False),
        (None, 90, True),
        (None, 100, True),
        ("1", 90, True),
    ],
)
def test_is_pdl_enabled_hardware_gate(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str | None,
    sm_version: int,
    expected: bool,
) -> None:
    if env_value is None:
        monkeypatch.delenv("TRTLLM_ENABLE_PDL", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_ENABLE_PDL", env_value)
    monkeypatch.setattr(flashinfer_utils, "get_sm_version", lambda: sm_version)

    assert flashinfer_utils.is_pdl_enabled() is expected


@pytest.mark.parametrize("sm_version", [-1, 89])
def test_is_pdl_enabled_rejects_unsupported_explicit_enable(
    monkeypatch: pytest.MonkeyPatch,
    sm_version: int,
) -> None:
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "1")
    monkeypatch.setattr(flashinfer_utils, "get_sm_version", lambda: sm_version)

    with mock.patch.object(flashinfer_utils.logger, "info") as log_info:
        with pytest.raises(ValueError, match="TRTLLM_ENABLE_PDL=1 requires SM90 or newer") as error:
            flashinfer_utils.is_pdl_enabled()

    log_info.assert_not_called()
    message = str(error.value)
    if sm_version < 0:
        assert "no CUDA GPU" in message
    else:
        assert f"SM{sm_version}" in message
    assert "Unset TRTLLM_ENABLE_PDL" in message
    assert "set it to 0" in message


@pytest.mark.parametrize("env_value", ["0", "true"])
def test_is_pdl_enabled_explicit_disable_skips_hardware_probe(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str,
) -> None:
    get_sm_version = mock.Mock(side_effect=AssertionError("unexpected hardware probe"))
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", env_value)
    monkeypatch.setattr(flashinfer_utils, "get_sm_version", get_sm_version)

    with mock.patch.object(flashinfer_utils.logger, "info") as log_info:
        assert flashinfer_utils.is_pdl_enabled() is False

    get_sm_version.assert_not_called()
    log_info.assert_not_called()


@pytest.mark.parametrize(
    ("sm_version", "expected", "expected_log"),
    [
        (-1, False, "PDL disabled: no CUDA GPU is available"),
        (89, False, "PDL disabled on SM89: requires SM90 or newer"),
        (90, True, "PDL enabled"),
    ],
)
def test_is_pdl_enabled_logs_once(
    monkeypatch: pytest.MonkeyPatch,
    sm_version: int,
    expected: bool,
    expected_log: str,
) -> None:
    monkeypatch.delenv("TRTLLM_ENABLE_PDL", raising=False)
    monkeypatch.setattr(flashinfer_utils, "get_sm_version", lambda: sm_version)

    with mock.patch.object(flashinfer_utils.logger, "info") as log_info:
        assert flashinfer_utils.is_pdl_enabled() is expected
        assert flashinfer_utils.is_pdl_enabled() is expected

    log_info.assert_called_once_with(expected_log)
