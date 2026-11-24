# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

H3_V2 = "TLLM_VISUALGEN_MODELING2_H3"


@pytest.fixture(autouse=True)
def h3_modeling_v2(monkeypatch):
    monkeypatch.setenv(H3_V2, "1")
