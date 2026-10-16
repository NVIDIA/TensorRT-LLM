# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.speculative import (
    set_spec_metadata_all_rank_num_tokens,
    update_spec_metadata,
)

pytestmark = pytest.mark.cpu_only


def test_update_spec_metadata_handles_parallel_draft_and_dynamic_tree() -> None:
    spec_mode = SimpleNamespace(
        attention_need_spec_dec_mode=Mock(return_value=True),
        is_parallel_draft=Mock(return_value=True),
    )
    spec_metadata = SimpleNamespace(
        spec_dec_mode=spec_mode,
        is_spec_dec_tree=True,
        is_spec_dec_dynamic_tree=True,
    )
    scheduled_requests = SimpleNamespace(
        batch_size=2,
        num_context_requests=1,
        context_requests=[object()],
        generation_requests=[object()],
    )
    attn_metadata = SimpleNamespace(update_spec_dec_param=Mock())
    spec_tree_manager = SimpleNamespace(
        slot_storage=SimpleNamespace(fill_all_slot_ids=Mock()),
    )

    config = SimpleNamespace(
        spec_config=SimpleNamespace(
            get_runtime_tokens_per_gen_step=lambda draft_len: draft_len + 1
        ),
        attention_backend=object(),
        original_max_draft_len=2,
        original_max_total_draft_tokens=6,
        spec_dec_max_total_draft_tokens=5,
    )

    update_spec_metadata(
        spec_metadata,
        config,
        scheduled_requests,
        attn_metadata,
        spec_tree_manager,
        runtime_draft_len=3,
    )

    assert spec_metadata.runtime_draft_len == 3
    assert spec_metadata.runtime_tokens_per_gen_step == 4
    spec_tree_manager.slot_storage.fill_all_slot_ids.assert_called_once_with(
        scheduled_requests.context_requests,
        scheduled_requests.generation_requests,
    )
    attn_metadata.update_spec_dec_param.assert_called_once_with(
        batch_size=2,
        is_spec_decoding_enabled=True,
        is_spec_dec_tree=True,
        is_spec_dec_dynamic_tree=True,
        max_draft_len=6,
        max_total_draft_tokens=6,
        spec_metadata=spec_metadata,
        spec_tree_manager=spec_tree_manager,
        num_contexts=1,
    )


def test_set_spec_metadata_all_rank_counts_for_one_model() -> None:
    mode = SimpleNamespace(
        is_mtp_eagle_one_model=Mock(return_value=True),
        is_eagle3_one_model=Mock(return_value=False),
    )
    spec_metadata = SimpleNamespace(spec_dec_mode=mode)

    set_spec_metadata_all_rank_num_tokens(
        spec_metadata,
        [8, 9],
        [2, 3],
        [1, 2],
    )

    assert spec_metadata.all_rank_num_tokens == [8, 9]
    assert spec_metadata.all_rank_num_seqs == [2, 3]
    assert spec_metadata.all_rank_num_gens == [1, 2]
    assert spec_metadata.subseq_all_rank_num_tokens == [2, 3]
