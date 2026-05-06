# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.llm_request import get_multimodal_embedding_lengths


def test_multimodal_embedding_lengths_legacy_without_metadata():
    request = SimpleNamespace(multimodal_lengths=[3, 5])

    assert get_multimodal_embedding_lengths(request) == [3, 5]


def test_multimodal_embedding_lengths_uses_top_level_metadata():
    request = SimpleNamespace(
        multimodal_lengths=[6, 5],
        py_multimodal_data={
            "multimodal_embedding_lengths": [5, 3],
        },
    )

    assert get_multimodal_embedding_lengths(request) == [5, 3]


def test_multimodal_embedding_lengths_uses_layout_metadata_fallback():
    request = SimpleNamespace(
        multimodal_lengths=[6, 5],
        py_multimodal_data={
            "layout_metadata": {
                "multimodal_embedding_lengths": [6, 4],
            },
        },
    )

    assert get_multimodal_embedding_lengths(request) == [6, 4]


def test_multimodal_embedding_lengths_rejects_metadata_length_mismatch():
    request = SimpleNamespace(
        multimodal_lengths=[4],
        py_multimodal_data={
            "multimodal_embedding_lengths": [3, 1],
        },
    )

    with pytest.raises(ValueError, match="length must match"):
        get_multimodal_embedding_lengths(request)


def test_multimodal_embedding_lengths_rejects_lengths_larger_than_prompt_span():
    request = SimpleNamespace(
        multimodal_lengths=[4],
        py_multimodal_data={
            "multimodal_embedding_lengths": [5],
        },
    )

    with pytest.raises(ValueError, match="exceeds"):
        get_multimodal_embedding_lengths(request)
