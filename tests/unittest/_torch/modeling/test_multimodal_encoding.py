# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for tensorrt_llm._torch.models.multimodal_encoding."""

from __future__ import annotations

import pytest

from tensorrt_llm._torch.models.multimodal_encoding import MultimodalItem


class TestMultimodalItem:
    def test_basic_fields(self):
        item = MultimodalItem(
            src_param_idx=2,
            item_idx_in_param=1,
            modality="image",
            token_count=5,
            payload={"data": "x"},
        )
        assert item.src_param_idx == 2
        assert item.item_idx_in_param == 1
        assert item.modality == "image"
        assert item.token_count == 5
        assert item.payload == {"data": "x"}

    def test_ghost_sentinel(self):
        item = MultimodalItem(
            src_param_idx=0,
            item_idx_in_param=-1,
            modality="audio",
            token_count=4,
            payload={"data": "y"},
        )
        assert item.item_idx_in_param == -1

    def test_is_frozen(self):
        item = MultimodalItem(
            src_param_idx=0,
            item_idx_in_param=0,
            modality="image",
            token_count=1,
            payload={},
        )
        with pytest.raises(Exception):  # FrozenInstanceError subclass of AttributeError
            item.src_param_idx = 99
