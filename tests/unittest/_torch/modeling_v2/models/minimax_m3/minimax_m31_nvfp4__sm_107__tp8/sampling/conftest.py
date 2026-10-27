# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sm_107 targets run the sampler with FlashInfer's default top-p renorm mode: routing arms
flashinfer_ops.arm_sm107_top_p_renorm (via interface.arm_sm107_spec_paths) before the model is built."""

import pytest

from tensorrt_llm._torch.pyexecutor.sampler.ops import flashinfer as flashinfer_ops


@pytest.fixture(autouse=True)
def _sm107_top_p_renorm_mode(monkeypatch):
    monkeypatch.setattr(flashinfer_ops, "TOP_P_RENORM_DETERMINISTIC", False)
