# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared stubs for the multimodal encode/extractor unit tests.

These helpers are imported by the per-model extractor suites
(`test_nano_encode_extractor`, `test_qwen3vl_encode_extractor`) and the
backend-agnostic assembly suite (`test_multimodal_encoding`) so the
`MultimodalParams` stub and the by-param item extractor stay defined in
exactly one place.
"""

from __future__ import annotations

from tensorrt_llm.inputs.multimodal import MultimodalParams


def _identity_extractor(items_by_param):
    """Yield pre-built MultimodalItems keyed by source param index.

    Mirrors the production extractor contract: `MixedModalityAssembly`
    invokes `extract(param_idx, param)` once for every param index in the
    batch (`enumerate(multimodal_params)`), so a param that produces no
    items must yield nothing rather than raising. Returning `[]` for an
    absent key models that empty-yield case (a real extractor such as
    `_nano_extract_items` yields nothing for an item-less param), so use
    `.get(idx, [])` instead of `items_by_param[idx]`.
    """

    def extract(param_idx, _param):
        yield from items_by_param.get(param_idx, [])

    return extract


def _make_param(multimodal_data: dict) -> MultimodalParams:
    """Build a stub MultimodalParams for extractor unit tests."""
    return MultimodalParams(
        multimodal_input=None,
        multimodal_data=multimodal_data,
        multimodal_runtime=None,
    )
