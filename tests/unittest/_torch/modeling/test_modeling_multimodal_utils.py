# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from typing import Optional, TypedDict

from tensorrt_llm._torch.models.modeling_multimodal_utils import _get_cached_merged_typed_dict


def test_merged_typed_dict_reuses_cached_schema():
    """Two `merged_typed_dict` schemas with the same shape collapse to one cached
    class. This is what makes huggingface_hub's per-schema validator cache hit
    on subsequent ProcessorMixin._merge_kwargs calls."""
    cache: dict = {}
    first = TypedDict("merged_typed_dict", {"do_rescale": Optional[bool]}, total=False)
    second = TypedDict("merged_typed_dict", {"do_rescale": Optional[bool]}, total=False)

    cached_first = _get_cached_merged_typed_dict(first, cache)
    cached_second = _get_cached_merged_typed_dict(second, cache)

    assert cached_first is cached_second  # identity stable across equivalent shapes


def test_non_merged_typed_dict_schemas_pass_through():
    """Schemas whose `__name__` is not the ephemeral `merged_typed_dict` are
    returned unchanged — only HF's per-call class is the one we want to collapse."""
    other = TypedDict("other_typed_dict", {"do_rescale": Optional[bool]}, total=False)
    assert _get_cached_merged_typed_dict(other, {}) is other
