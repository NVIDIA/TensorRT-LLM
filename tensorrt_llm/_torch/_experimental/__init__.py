# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Subpackages here are not covered by the API stability tests.

Anything under this package may change shape or be removed without a
deprecation cycle. Import it from outside `tensorrt_llm` at your own risk;
in-tree callers should reach it through a switch that stays off by default,
the way `modeling_v2` is reached through `TRTLLM_MODELING_V2`.

Nothing is re-exported here on purpose, so this module itself pulls in no
subpackage. That is not the same as saying a subtree here is unreachable at
startup -- `modeling_v2` is imported eagerly by `_torch/models/modeling_auto.py`
-- only that reaching one has to be written down at the import site rather than
happening as a side effect of this package.
"""
