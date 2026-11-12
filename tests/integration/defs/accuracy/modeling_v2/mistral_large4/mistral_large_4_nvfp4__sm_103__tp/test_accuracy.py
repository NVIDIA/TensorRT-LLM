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
"""Whole-model gate for the mistral_large4 modeling_v2 target mistral_large_4_nvfp4__sm_103__tp.

The tests run the implementations in ``../_common.py``, which also holds the layout and its
deployments.
"""

import pytest

from .._common import (
    PROFILES,
    PROFILES_WITHOUT_DSPARK,
    _TpTarget,
    pytest_generate_tests,  # noqa: F401 (the hook parametrizes the deployments)
    skip_not_sm103,
)


class TestModelingV2MistralLarge4Nvfp4Sm103Tp(_TpTarget):
    """mistral-large-4-nvfp4 / sm_103 / tp, the language model."""

    pytestmark = skip_not_sm103

    @pytest.mark.skip_less_device(4)
    def test_gsm8k(self, deployment):
        self._test_gsm8k(deployment)

    @pytest.mark.skip_less_device(4)
    def test_gsm8k_24shot(self, deployment):
        self._test_gsm8k_24shot(deployment)

    @pytest.mark.skip_less_device(4)
    def test_passkey_retrieval(self, deployment):
        self._test_passkey_retrieval(deployment)

    @pytest.mark.skip_less_device(4)
    @pytest.mark.parametrize("profile", PROFILES)
    def test_accuracy(self, deployment, profile, tmp_path, monkeypatch, record_property):
        self._test_accuracy(deployment, profile, tmp_path, monkeypatch, record_property)

    @pytest.mark.skip_less_device(4)
    @pytest.mark.parametrize("profile", PROFILES_WITHOUT_DSPARK)
    def test_accuracy_without_dspark(
        self, deployment, profile, tmp_path, monkeypatch, record_property
    ):
        self._test_accuracy_without_dspark(
            deployment, profile, tmp_path, monkeypatch, record_property
        )

    @pytest.mark.skip_less_device(4)
    def test_llguidance(self, tmp_path, record_property):
        self._test_llguidance(tmp_path, record_property)
