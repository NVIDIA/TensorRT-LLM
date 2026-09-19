# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VisualGen SOL sparse attention: shared parameters, the TRT-LLM-owned predictor,
and the TRTLLM and CuTeDSL backends."""

from .backend import SOLCuTeDSLAttention, SOLTrtllmAttention
from .params import SolParams
from .predictor import (
    SolPredictorGeometry,
    SolPredictorOutputs,
    SolPredictorPlan,
    SolPredictorPlanKey,
    SOLSparsePredictor,
)

__all__ = [
    "SOLCuTeDSLAttention",
    "SOLSparsePredictor",
    "SOLTrtllmAttention",
    "SolParams",
    "SolPredictorGeometry",
    "SolPredictorOutputs",
    "SolPredictorPlan",
    "SolPredictorPlanKey",
]
