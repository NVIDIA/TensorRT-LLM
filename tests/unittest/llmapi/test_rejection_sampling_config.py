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
"""Tests for rejection sampling configuration validation."""

import pytest
from pydantic import ValidationError

from tensorrt_llm.llmapi.llm_args import EagleDecodingConfig

# EagleDecodingConfig.validate_eagle_config requires max_draft_len > 0; use a
# minimal value for config-only tests (no model load).
_MIN_DRAFT = 4
# validate_speculative_model requires a non-None draft model path (not loaded in these tests).
_SPEC_MODEL = "dummy-path/eagle-draft"


def test_rejection_sampling_requires_allow_advanced_sampling():
    """use_rejection_sampling=True is rejected unless allow_advanced_sampling=True."""
    with pytest.raises(ValidationError) as exc_info:
        EagleDecodingConfig(
            max_draft_len=_MIN_DRAFT,
            speculative_model=_SPEC_MODEL,
            use_rejection_sampling=True,
            allow_advanced_sampling=False,
        )
    err = str(exc_info.value)
    assert "use_rejection_sampling" in err
    assert "allow_advanced_sampling" in err


def test_rejection_sampling_with_advanced_sampling():
    """Rejection sampling is valid when both flags are enabled."""
    config = EagleDecodingConfig(
        max_draft_len=_MIN_DRAFT,
        speculative_model=_SPEC_MODEL,
        use_rejection_sampling=True,
        allow_advanced_sampling=True,
    )
    assert config.use_rejection_sampling is True
    assert config.allow_advanced_sampling is True


def test_rejection_sampling_disabled_by_default():
    """use_rejection_sampling defaults to False for a minimal valid Eagle config."""
    config = EagleDecodingConfig(
        max_draft_len=_MIN_DRAFT,
        speculative_model=_SPEC_MODEL,
    )
    assert config.use_rejection_sampling is False


def test_advanced_sampling_without_rejection_sampling():
    """allow_advanced_sampling can be True while use_rejection_sampling stays False."""
    config = EagleDecodingConfig(
        max_draft_len=_MIN_DRAFT,
        speculative_model=_SPEC_MODEL,
        allow_advanced_sampling=True,
        use_rejection_sampling=False,
    )
    assert config.allow_advanced_sampling is True
    assert config.use_rejection_sampling is False
