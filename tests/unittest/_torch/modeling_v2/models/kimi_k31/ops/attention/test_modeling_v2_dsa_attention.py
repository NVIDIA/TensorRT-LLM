# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which Kimi K3.1 DSA backend declaration ``attention_backend_for`` returns: the family's for TRTLLM with the
``dsa`` algorithm (the perf targets' with ``perf``), None for any other backend or sparse algorithm."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_attention import (
    K31DSABackend,
    K31PerfDSABackend,
    attention_backend_for,
)

pytestmark = pytest.mark.cpu_only


def test_attention_backend_for():
    dsa = SimpleNamespace(algorithm="dsa")
    assert attention_backend_for("TRTLLM", dsa) is K31DSABackend
    assert attention_backend_for("trtllm", dsa, perf=True) is K31PerfDSABackend
    assert attention_backend_for("FLASHINFER", dsa) is None
    assert attention_backend_for("TRTLLM", SimpleNamespace(algorithm="rocket")) is None
    assert attention_backend_for("TRTLLM", None) is None
