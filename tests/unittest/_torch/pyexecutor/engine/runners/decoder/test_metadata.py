# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import DecoderRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.metadata import (
    set_spec_metadata_all_rank_num_tokens,
    update_spec_metadata,
)
from tensorrt_llm._torch.speculative.dspark import DSparkSpecMetadata

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


@pytest.mark.parametrize("dspark", [False, True])
@pytest.mark.parametrize("gather_context_logits", [False, True])
def test_forward_selects_current_speculative_rows(
    dspark: bool, gather_context_logits: bool
) -> None:
    logits = torch.arange(24).reshape(6, 4)
    metadata = object.__new__(DSparkSpecMetadata) if dspark else SimpleNamespace()
    runner = object.__new__(DecoderRunner)
    runner._config = SimpleNamespace(without_logits=False)
    runner._inputs = SimpleNamespace(preprocess_inputs=lambda inputs, **kwargs: inputs)
    runner.sample_in_graph_callable = None

    def forward(
        *,
        return_context_logits: bool,
        spec_metadata: SimpleNamespace | DSparkSpecMetadata | None = None,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        assert return_context_logits is (spec_metadata is not None or gather_context_logits)
        if dspark and spec_metadata is not None:
            assert spec_metadata._requires_all_context_logits is gather_context_logits
        return {"logits": logits, "hidden_states": logits + 100}

    runner._model_forward = forward
    for rows in ([1, 2, 4, 5], None, [0, 3]):
        inputs = {}
        if rows is not None:
            metadata.gather_ids = torch.tensor(rows)
            inputs["spec_metadata"] = metadata
        outputs = runner._forward_step(
            inputs,
            enable_spec_decode=rows is not None,
            runtime_draft_len=1 if rows is not None else 0,
            is_dummy=False,
            gather_context_logits=gather_context_logits,
        )
        torch.testing.assert_close(outputs["logits"], logits if rows is None else logits[rows])
        torch.testing.assert_close(outputs["hidden_states"], logits + 100)
        if dspark:
            assert metadata._requires_all_context_logits is None
