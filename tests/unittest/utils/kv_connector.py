# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit request, capability, and MPI doubles for KV connector unit tests."""

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock, Mock, patch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState

CONNECTOR_MODULE = "tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector"


class ConnectorAdapterMock(MagicMock):
    supports_attention_dp = True
    capacity_only = False
    reports_failed_loads = False
    requires_block_aligned_reuse = False
    drops_reuse_match_backoff = False
    requires_completion_reservation = False


def connector_request(
    request_id: int,
    *,
    dummy: bool = False,
    context_only: bool = False,
    generation_only: bool = False,
    attention_dp_dummy: bool = False,
) -> MagicMock:
    req = MagicMock(
        spec=LlmRequest,
        is_dummy_request=dummy,
        is_attention_dp_dummy=attention_dp_dummy,
        py_connector_load_failed=False,
        py_decoding_iter=0,
        py_batch_idx=None,
        py_orig_prompt_len=8,
        cached_tokens=0,
        is_context_only_request=context_only,
        is_generation_only_request=generation_only,
    )
    req.request_id = request_id
    req.py_request_id = request_id
    req.is_child = False
    req.is_finished = False
    req.state = (
        LlmRequestState.GENERATION_IN_PROGRESS
        if generation_only and not context_only
        else LlmRequestState.CONTEXT_INIT
    )
    req.context_current_position = 0
    req.context_remaining_length = 8
    req.context_chunk_size = 8
    req.total_input_len_cp = 8
    req.get_tokens.return_value = list(range(8))
    req.py_draft_tokens = []
    req.kv_cache_retention_config = None
    req.cache_salt = None
    return req


@contextmanager
def connector_mpi(*, rank: int = 0, world_size: int = 1) -> Iterator[Mock]:
    """Keep rank, world size, and identity broadcast consistent for a test scope."""
    with (
        patch(f"{CONNECTOR_MODULE}.mpi_rank", return_value=rank),
        patch(f"{CONNECTOR_MODULE}.mpi_world_size", return_value=world_size),
        patch(
            f"{CONNECTOR_MODULE}.mpi_broadcast", side_effect=lambda result, root: result
        ) as broadcast,
    ):
        yield broadcast
