# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from ....conftest import llm_models_root


@pytest.fixture
def model_root():
    return Path(llm_models_root()) / "nemotron_shortcut"


@pytest.fixture
def checkpoint(model_root, request):
    return model_root / request.param
