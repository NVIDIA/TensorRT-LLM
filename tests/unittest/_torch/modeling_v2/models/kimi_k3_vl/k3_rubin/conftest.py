# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin import features


@pytest.fixture
def k3_features(monkeypatch):
    """``k3_features(<field>=<value>, ...)`` sets K3Features fields for one test.

    Only fields read when called take effect this way; a field read at import is a module
    constant of its reader, which the test patches instead.
    """

    def set_fields(**fields):
        monkeypatch.setattr(features, "_ACTIVE", dataclasses.replace(features.active(), **fields))
        return features.active()

    return set_fields
