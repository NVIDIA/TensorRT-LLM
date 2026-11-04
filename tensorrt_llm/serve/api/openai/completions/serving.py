# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``POST /v1/completions``."""

from tensorrt_llm.serve._base.serving import GenerateBaseServing


class CompletionServing(GenerateBaseServing):
    """``POST /v1/completions``.

    Takes no dependency beyond those of ``GenerateBaseServing``; the
    composition root builds it with ``_generate_serving_kwargs()``.
    """
