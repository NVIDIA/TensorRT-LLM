# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping

BLOCK_REUSE_STABLE_TOKEN_COUNT_TRACE_KEY = "trtllm-internal-block-reuse-stable-token-count"


def get_block_reuse_stable_token_count(
    trace_headers: Mapping[str, str] | None,
) -> int | None:
    if trace_headers is None:
        return None
    value = trace_headers.get(BLOCK_REUSE_STABLE_TOKEN_COUNT_TRACE_KEY)
    if value is None:
        return None
    try:
        token_count = int(value)
    except ValueError:
        return None
    return token_count if token_count > 0 else None


def set_block_reuse_stable_token_count(
    trace_headers: Mapping[str, str] | None,
    token_count: int | None,
) -> dict[str, str] | None:
    if token_count is None or token_count <= 0:
        return dict(trace_headers) if trace_headers is not None else None
    updated = dict(trace_headers or {})
    updated[BLOCK_REUSE_STABLE_TOKEN_COUNT_TRACE_KEY] = str(token_count)
    return updated
