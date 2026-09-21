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
"""Regression tests for ADLogger severity-to-level mapping.

Covers https://github.com/NVIDIA/TensorRT-LLM/issues/19491: ``set_level()`` and the
``AUTO_DEPLOY_LOG_LEVEL`` environment-variable init path must resolve the AutoDeploy-specific
severity names (``verbose``, ``internal_error``) the same way ``log()`` already does via
``_SEVERITY_TO_LEVEL``, instead of falling back to INFO.
"""

import logging

import pytest

from tensorrt_llm._torch.auto_deploy.utils.logger import ADLogger, Singleton

pytestmark = pytest.mark.cpu_only

_SEVERITY_CASES = [
    ("internal_error", logging.CRITICAL),
    ("error", logging.ERROR),
    ("warning", logging.WARNING),
    ("info", logging.INFO),
    ("verbose", logging.DEBUG),
    ("debug", logging.DEBUG),
]


@pytest.fixture
def fresh_ad_logger():
    """Yield a newly constructed ADLogger, restoring prior singleton/logger state after."""
    underlying_logger = logging.getLogger("auto_deploy")
    original_instance = Singleton._instances.get(ADLogger)
    original_level = underlying_logger.level

    Singleton._instances.pop(ADLogger, None)
    try:
        yield ADLogger()
    finally:
        underlying_logger.setLevel(original_level)
        if original_instance is not None:
            Singleton._instances[ADLogger] = original_instance
        else:
            Singleton._instances.pop(ADLogger, None)


@pytest.mark.parametrize("severity,expected_level", _SEVERITY_CASES)
def test_set_level_maps_public_severity_names(fresh_ad_logger, severity, expected_level):
    fresh_ad_logger.set_level(severity)
    assert fresh_ad_logger._logger.level == expected_level


@pytest.mark.parametrize("severity,expected_level", _SEVERITY_CASES)
def test_set_level_is_case_insensitive(fresh_ad_logger, severity, expected_level):
    fresh_ad_logger.set_level(severity.upper())
    assert fresh_ad_logger._logger.level == expected_level


def test_set_level_falls_back_to_info_for_unknown_severity(fresh_ad_logger):
    fresh_ad_logger.set_level("not_a_real_severity")
    assert fresh_ad_logger._logger.level == logging.INFO


@pytest.mark.parametrize("severity,expected_level", _SEVERITY_CASES)
def test_env_variable_init_maps_public_severity_names(monkeypatch, severity, expected_level):
    monkeypatch.setenv("AUTO_DEPLOY_LOG_LEVEL", severity)
    underlying_logger = logging.getLogger("auto_deploy")
    original_instance = Singleton._instances.get(ADLogger)
    original_level = underlying_logger.level

    Singleton._instances.pop(ADLogger, None)
    try:
        logger = ADLogger()
        assert logger._logger.level == expected_level
    finally:
        underlying_logger.setLevel(original_level)
        if original_instance is not None:
            Singleton._instances[ADLogger] = original_instance
        else:
            Singleton._instances.pop(ADLogger, None)
