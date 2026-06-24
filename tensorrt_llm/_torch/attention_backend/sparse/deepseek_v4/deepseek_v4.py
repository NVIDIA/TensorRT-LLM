# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class DeepseekV4AttentionType(Enum):
    # Attention types backed by per-layer sliding-window cache state.
    SWA = 0
    COMPRESSOR_KV = 1
    COMPRESSOR_SCORE = 2
    INDEXER_COMPRESSOR_KV = 3
    INDEXER_COMPRESSOR_SCORE = 4

    # Attention types backed by ratio-shared compressed cache state.
    COMPRESS = 5
    INDEXER_COMPRESS = 6

    # Backward-compatible names used by the standalone compressor primitive.
    COMPRESSOR_STATE = COMPRESSOR_KV
    INDEXER_COMPRESSOR_STATE = INDEXER_COMPRESSOR_KV
