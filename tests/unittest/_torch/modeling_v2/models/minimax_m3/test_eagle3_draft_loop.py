# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the MiniMax worker's Python dispatch without importing GPU kernels."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, sentinel

import pytest

pytestmark = pytest.mark.cpu_only

_TORCH_ROOT = Path(__file__).resolve().parents[6] / "tensorrt_llm" / "_torch"


def _load_worker(tp: int) -> type:
    # Execute the actual methods, including the base dispatcher and its signature.
    # Keeping the class definitions preserves zero-argument super() in the override.
    namespace = {}
    sources = (
        (_TORCH_ROOT / "speculative/eagle3.py", "Eagle3OneModelWorker", {"_forward_draft_loop"}),
        (
            _TORCH_ROOT
            / f"_experimental/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp{tp}/modeling.py",
            "M3Eagle3Worker",
            {"_forward_impl", "_forward_draft_loop"},
        ),
    )
    for path, name, methods in sources:
        cls = next(
            node
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.ClassDef) and node.name == name
        )
        cls.body = [
            node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods
        ]
        assert len(cls.body) == len(methods)
        if name == "Eagle3OneModelWorker":
            cls.bases = []
        module = ast.Module(body=[cls], type_ignores=[])
        exec(compile(module, str(path), "exec"), namespace)
    return namespace["M3Eagle3Worker"]


@pytest.mark.parametrize("tp", [4, 8])
@pytest.mark.parametrize("path", ["fallback", "fallback_buffer", "fused", "external"])
def test_eagle3_draft_loop_contract(tp: int, path: str) -> None:
    worker = _load_worker(tp)()
    worker._is_mamba_hybrid_cache = False
    worker._auxiliary_state_handlers = []
    worker.sa_enhancer = None
    worker._uses_external_shared_target_kv = path == "external"
    worker._execute_guided_decoder_if_present = Mock()
    worker.sample_and_accept_draft_tokens = Mock(
        return_value=(sentinel.accepted, sentinel.accepted_counts)
    )
    worker.draft_only_eligible = Mock(return_value=False)
    metadata = SimpleNamespace(
        num_seqs=2,
        num_contexts=1,
        _seq_lens_cuda=sentinel.pre_draft_seq_lens,
        all_rank_num_tokens=sentinel.rank_tokens,
    )
    spec = SimpleNamespace(
        runtime_draft_len=3,
        spec_resource_manager=SimpleNamespace(),
        batch_indices_cuda=sentinel.batch_indices,
    )

    def prepare_metadata(md: SimpleNamespace) -> None:
        # The caller must retain the sequence lengths from before preparation.
        md._seq_lens_cuda = sentinel.draft_seq_lens

    worker._prepare_attn_metadata_for_spec_dec = Mock(side_effect=prepare_metadata)
    worker._restore_attn_metadata_from_spec_dec = Mock()
    worker.prepare_1st_drafter_inputs = Mock(return_value=sentinel.inputs)
    worker.get_draft_kv_cache_manager = Mock(return_value=sentinel.cache)
    worker._m3_fused_loop_ok = Mock(return_value=path == "fused")
    prebuilt = sentinel.next_new if path == "fallback_buffer" else None
    worker._forward_linear_draft_loop = Mock(return_value=(sentinel.drafts, prebuilt))

    def fused_loop(*args: object) -> object:
        worker._m3_next_new_tokens = sentinel.next_new
        return sentinel.drafts

    worker._m3_fused_draft_loop = Mock(side_effect=fused_loop)
    worker._forward_external_shared_target_kv_draft_loop = Mock(return_value=sentinel.drafts)
    worker._prepare_next_new_tokens = Mock(return_value=sentinel.next_new)

    outputs = worker._forward_impl(
        sentinel.input_ids,
        Mock(),
        sentinel.hidden_states,
        sentinel.logits,
        metadata,
        spec,
        sentinel.draft_model,
        sentinel.resources,
    )

    assert outputs["next_draft_tokens"] is sentinel.drafts
    assert outputs["next_new_tokens"] is sentinel.next_new
    assert outputs["new_tokens"] is sentinel.accepted
    assert metadata.use_spec_decoding
    if path.startswith("fallback"):
        worker._forward_linear_draft_loop.assert_called_once_with(
            sentinel.inputs,
            metadata,
            spec,
            sentinel.draft_model,
            sentinel.cache,
            1,
            2,
            sentinel.accepted_counts,
            sentinel.rank_tokens,
            sentinel.accepted,
            sentinel.pre_draft_seq_lens,
        )
    else:
        worker._forward_linear_draft_loop.assert_not_called()
    if path in ("fallback_buffer", "fused"):
        worker._prepare_next_new_tokens.assert_not_called()
    else:
        worker._prepare_next_new_tokens.assert_called_once_with(
            sentinel.accepted, sentinel.drafts, sentinel.batch_indices, 2, sentinel.accepted_counts
        )
