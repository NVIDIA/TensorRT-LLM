# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight tables and loaders: MiniMax-M3.1 (MXFP8 + packed NVFP4 experts) / sm_107 / tp8.

The tables, transforms and loaders are the family's (``m31_weights``); the splits follow the engine's mapping.
"""

from ..m31_weights import MiniMaxM31DraftWeights, MiniMaxM31Weights, fused_projection

__all__ = ["DRAFT_WEIGHTS", "MODEL_WEIGHTS", "fused_projection"]

MODEL_WEIGHTS = MiniMaxM31Weights("tp8")
DRAFT_WEIGHTS = MiniMaxM31DraftWeights("tp8")
