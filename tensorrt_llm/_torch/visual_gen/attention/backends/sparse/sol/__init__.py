# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VisualGen SOL sparse attention: shared parameters, the TRT-LLM-owned predictor,
and the TRTLLM and CuTeDSL backends."""

from .backend import SOLCuTeDSLAttention, SOLTrtllmAttention
from .params import SolParams
from .predictor import SolPredictorOutputs, predict, support_reason

__all__ = [
    "SOLCuTeDSLAttention",
    "SOLTrtllmAttention",
    "SolParams",
    "SolPredictorOutputs",
    "predict",
    "support_reason",
]
