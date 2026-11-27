# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from test_common.llm_data import with_mocked_hf_download_for_single_gpu
from test_common.moe_tactics import sm100f_moe_tactic_or_cutlass
from utils.llm_data import llm_models_root
from utils.spawn_process import SpawnProcessContext, spawn_process
from utils.util import skip_num_gpus_less_than, skip_pre_blackwell

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._device_copy import DeviceCopyBuffer
from tensorrt_llm._torch.attention.backends.sparse.dsa import (
    DSACacheManagerV2, DSAtrtllmAttentionMetadata)
from tensorrt_llm._torch.attention.backends.trtllm import \
    TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.peft.lora.config import LoraConfig
from tensorrt_llm._torch.pyexecutor._util import (
    _derive_draft_max_attention_window,
    _expand_attention_window_pattern_to_global_layers)
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import KeyType
from tensorrt_llm._torch.pyexecutor.py_executor_creator import \
    _extend_full_attention_windows_for_spec_decode
from tensorrt_llm._torch.speculative.eagle3 import (Eagle3OneModelSpecMetadata,
                                                    MTPEagleWorker)
from tensorrt_llm._torch.speculative.eagle3_dynamic_tree import \
    Eagle3OneModelDynamicTreeWorker
from tensorrt_llm._torch.speculative.interface import \
    INVALID_PROMPT_LOOKAHEAD_TOKEN
from tensorrt_llm._torch.speculative.mtp_dynamic_tree import \
    MTPEagleDynamicTreeWorker
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.llmapi import (CudaGraphConfig, Eagle3DecodingConfig,
                                 KvCacheConfig, MoeConfig, MTPDecodingConfig)


def test_mtp_eagle_refreshes_dsa_metadata_before_draft_forward() -> None:
    """Refresh DSA mappings after switching to the draft cache."""
    events = []
    state = {}
    draft_manager = object.__new__(DSACacheManagerV2)

    class _Metadata(DSAtrtllmAttentionMetadata):

        def __init__(self):
            self._seq_lens_cuda = torch.tensor([1], dtype=torch.int32)
            self._num_ctx_tokens = 0

        def set_in_mtp_draft_loop(self, active):
            state["in_mtp_draft_loop"] = active

        def set_mtp_num_accepted(self, value):
            state["mtp_num_accepted"] = value

        def set_skip_topk(self, value):
            state["skip_topk"] = value

        def on_update_kv_lens(self):
            events.append("refresh")

    metadata = _Metadata()

    @contextmanager
    def draft_context(attn_metadata,
                      manager,
                      enable_draft_locality_domain_forkjoin=False):
        assert attn_metadata is metadata
        assert manager is draft_manager
        events.append("switch")
        yield metadata

    class _StopAfterFirstDraftForward(Exception):
        pass

    def run_draft_forward(*args, **kwargs):
        events.append("forward")
        raise _StopAfterFirstDraftForward

    worker = SimpleNamespace(
        is_mtp_eagle=True,
        _draft_takes_kept_rows=False,
        draft_kv_cache_context=draft_context,
        _run_draft_forward=run_draft_forward,
        _step0_draft_layer_picks_rows=lambda draft_model, use_epilogue: False,
    )

    from tensorrt_llm._torch.speculative.eagle3 import Eagle3OneModelWorker

    with pytest.raises(_StopAfterFirstDraftForward):
        Eagle3OneModelWorker._forward_linear_draft_loop(
            worker,
            {"position_ids": torch.tensor([0], dtype=torch.int32)},
            metadata,
            SimpleNamespace(runtime_draft_len=1),
            draft_model=object(),
            draft_kv_cache_manager=draft_manager,
            num_contexts=0,
            batch_size=1,
            num_accepted_tokens=torch.tensor([1], dtype=torch.int32),
            original_all_rank_num_tokens=None,
        )

    assert events == ["switch", "refresh", "forward"]
    # The draft forward raised, so only the ExitStack callback can have cleared
    # this; a leaked flag would let the next target forward reuse the selections.
    assert state == {
        "in_mtp_draft_loop": False,
        "mtp_num_accepted": None,
        "skip_topk": False,
    }


def test_dynamic_tree_metadata_forces_target_mask_prepare_each_step() -> None:
    metadata = TrtllmAttentionMetadata(
        seq_lens=None,
        seq_lens_kv=None,
        num_contexts=0,
        max_num_requests=1,
        max_num_tokens=1,
        max_seq_len=1,
    )
    common_kwargs = dict(
        batch_size=0,
        is_spec_decoding_enabled=False,
        is_spec_dec_tree=True,
        max_draft_len=1,
        max_total_draft_tokens=1,
    )

    metadata.update_spec_dec_param(is_spec_dec_dynamic_tree=True,
                                   **common_kwargs)
    assert metadata.force_prepare_spec_dec_tree_mask

    metadata.update_spec_dec_param(is_spec_dec_dynamic_tree=False,
                                   **common_kwargs)
    assert not metadata.force_prepare_spec_dec_tree_mask


def test_mtp_dynamic_tree_relocation_uses_full_attention_window(
        monkeypatch: pytest.MonkeyPatch) -> None:
    worker = object.__new__(MTPEagleDynamicTreeWorker)
    worker._kv_head_dim_bytes = 256
    worker._accepted_draft_indices_tensor = torch.tensor([[0, 1], [2, -1]],
                                                         dtype=torch.int32)
    worker._num_accepted_tokens_buf = torch.tensor([2, 1], dtype=torch.int32)

    attention_pool_pointers = object()
    attention_block_offsets = object()
    cache_manager = SimpleNamespace(
        num_kv_heads_per_layer=[0, 8, 0, 8],
        kv_cache_pool_mapping=[[0, 0], [2, 0], [1, 0], [2, 1]],
        kv_cache_pool_pointers=[object(),
                                object(), attention_pool_pointers],
        max_attention_window_vec=[None],
        max_seq_len=8192,
        max_total_draft_tokens=31,
        max_blocks_per_seq=256,
        tokens_per_block=32,
    )
    attention_metadata = SimpleNamespace(
        kv_cache_manager=cache_manager,
        kv_lens_cuda=torch.tensor([128, 256], dtype=torch.int32),
        kv_cache_block_offsets=[object(),
                                object(), attention_block_offsets],
    )
    update_op = MagicMock()
    monkeypatch.setattr(
        torch.ops.tensorrt_llm,
        "update_kv_cache_draft_token_location_2d",
        update_op,
    )

    worker._relocate_kv_eagerly(attention_metadata, batch_size=2)

    update_op.assert_called_once()
    args = update_op.call_args.args
    assert args[4] == 2
    assert args[5] == 8
    assert args[8] == cache_manager.max_seq_len
    assert args[9] is attention_pool_pointers
    assert args[10] is attention_block_offsets


def test_eagle3_dynamic_tree_relocation_uses_local_cache_shape(
        monkeypatch: pytest.MonkeyPatch) -> None:
    worker = object.__new__(Eagle3OneModelDynamicTreeWorker)
    worker._kv_head_dim_bytes = 256
    worker._accepted_draft_indices_tensor = torch.tensor([[0, 1], [2, -1]],
                                                         dtype=torch.int32)
    worker._num_accepted_tokens_buf = torch.tensor([2, 1], dtype=torch.int32)

    local_pool_pointers = object()
    local_block_offsets = object()
    cache_manager = SimpleNamespace(
        num_layers=8,
        num_local_layers=2,
        num_kv_heads_per_layer=[8, 8],
        max_attention_window_vec=[None, None],
        max_seq_len=8192,
        max_total_draft_tokens=31,
        max_blocks_per_seq=256,
        tokens_per_block=32,
        kv_cache_pool_mapping=[[1, 0], [1, 1]],
        kv_cache_pool_pointers=[object(), local_pool_pointers],
    )
    attention_metadata = SimpleNamespace(
        kv_cache_manager=cache_manager,
        kv_lens_cuda=torch.tensor([128, 256], dtype=torch.int32),
        kv_cache_block_offsets=[object(), local_block_offsets],
    )
    update_op = MagicMock()
    monkeypatch.setattr(
        torch.ops.tensorrt_llm,
        "update_kv_cache_draft_token_location_2d",
        update_op,
        raising=False,
    )

    worker._relocate_kv_eagerly(attention_metadata, batch_size=2)

    update_op.assert_called_once()
    (
        accepted_draft_indices,
        num_accepted_tokens,
        past_key_value_lengths,
        use_paged_kv_cache,
        layer_count,
        num_kv_heads,
        head_size_in_bytes,
        rewind_draft_token_count,
        max_kv_cache_len,
        pool_pointers,
        block_offsets,
        max_blocks_per_seq,
        tokens_per_block,
        stream,
    ) = update_op.call_args.args
    assert torch.equal(accepted_draft_indices,
                       worker._accepted_draft_indices_tensor)
    assert torch.equal(num_accepted_tokens, worker._num_accepted_tokens_buf)
    assert torch.equal(past_key_value_lengths, attention_metadata.kv_lens_cuda)
    assert use_paged_kv_cache is True
    assert layer_count == cache_manager.num_local_layers
    assert num_kv_heads == 8
    assert head_size_in_bytes == worker._kv_head_dim_bytes
    assert rewind_draft_token_count == cache_manager.max_total_draft_tokens
    assert max_kv_cache_len == cache_manager.max_seq_len
    assert pool_pointers is local_pool_pointers
    assert block_offsets is local_block_offsets
    assert max_blocks_per_seq == cache_manager.max_blocks_per_seq
    assert tokens_per_block == cache_manager.tokens_per_block
    assert stream is None


def test_eagle3_draft_kv_cache_uses_full_window_when_draft_has_no_swa() -> None:
    kv_cache_config = KvCacheConfig(max_attention_window=[128, 131072])
    draft_pretrained_config = SimpleNamespace(num_hidden_layers=3)

    max_attention_window = _derive_draft_max_attention_window(
        kv_cache_config=kv_cache_config,
        draft_pretrained_config=draft_pretrained_config,
        max_seq_len=131072,
        num_draft_layers=3,
    )

    assert max_attention_window is None


def test_eagle3_draft_kv_cache_expands_swa_in_global_layer_order() -> None:
    kv_cache_config = KvCacheConfig(max_attention_window=[128, 131072])
    draft_pretrained_config = SimpleNamespace(
        sliding_window=512,
        layer_types=["sliding_attention", "full_attention"],
    )

    max_attention_window = _derive_draft_max_attention_window(
        kv_cache_config=kv_cache_config,
        draft_pretrained_config=draft_pretrained_config,
        max_seq_len=4096,
        num_draft_layers=3,
    )

    global_windows = _expand_attention_window_pattern_to_global_layers(
        max_attention_window=max_attention_window,
        layer_mask=[False, False, False, False, False, True, True, True],
    )

    # Slots 0-4 are inactive fillers; physical draft layers 5-7 preserve the
    # derived sliding/full/sliding pattern.
    assert global_windows == [512, 512, 512, 512, 512, 512, 4096, 512]


def test_eagle3_draft_kv_cache_rejects_multiple_sliding_window_sizes() -> None:
    # The derivation assumes a single scalar sliding_window per model. A future
    # draft config exposing multiple distinct sliding sizes must fail loudly
    # rather than silently collapse every sliding layer to one size.
    kv_cache_config = KvCacheConfig(max_attention_window=[128, 131072])
    draft_pretrained_config = SimpleNamespace(
        sliding_window=[512, 1024],
        layer_types=["sliding_attention", "full_attention"],
    )

    with pytest.raises(NotImplementedError):
        _derive_draft_max_attention_window(
            kv_cache_config=kv_cache_config,
            draft_pretrained_config=draft_pretrained_config,
            max_seq_len=4096,
            num_draft_layers=3,
        )


def test_eagle3_target_kv_cache_extends_full_window_for_spec_decode() -> None:
    kv_cache_config = KvCacheConfig(max_attention_window=[128, 131072])
    spec_config = Eagle3DecodingConfig(max_draft_len=3,
                                       speculative_model="draft")

    _extend_full_attention_windows_for_spec_decode(
        kv_cache_config=kv_cache_config,
        spec_config=spec_config,
        net_max_seq_len=131072,
        model_engine_max_seq_len=131080,
    )

    assert kv_cache_config.max_attention_window == [128, 131080]


@skip_num_gpus_less_than(1)
def test_eagle3_one_model_capture_uses_real_token_count() -> None:
    # maybe_capture_hidden_states routes through inplace_slice_copy, a CUDA-only
    # custom op, so this test runs on GPU. It verifies that the capture is bounded
    # by the real scheduled token count (num_capture_tokens) and not self.num_tokens.
    #
    # prepare() decrements self.num_tokens to the attention-DP subseq shape while
    # leaving num_capture_tokens at the real (unpadded) count. Simulate that
    # post-prepare state: if the capture wrongly used self.num_tokens it would drop
    # the draft-verification rows (regressing the acceptance rate); if it wrongly
    # used the padded input it would overrun the buffer.
    spec_metadata = Eagle3OneModelSpecMetadata(
        max_num_requests=1,
        max_draft_len=3,
        max_total_draft_tokens=3,
        num_layers=36,
        hidden_size=2,
        max_num_tokens=4,
        dtype=torch.float32,
        layers_to_capture={23},
    )
    assert spec_metadata.hidden_states is not None
    spec_metadata.hidden_states.fill_(-1)
    # Real scheduled count is 4; num_tokens is decremented below it by prepare().
    spec_metadata.num_capture_tokens = 4
    spec_metadata.num_tokens = 1

    # 6 rows simulate CUDA-graph padding beyond the 4 scheduled tokens.
    hidden_states = torch.arange(12, dtype=torch.float32,
                                 device="cuda").reshape(6, 2)
    residual = torch.ones_like(hidden_states)
    spec_metadata.maybe_capture_hidden_states(23, hidden_states, residual)
    torch.cuda.synchronize()

    # All 4 real tokens captured (not truncated to num_tokens=1), padding dropped.
    assert torch.equal(spec_metadata.hidden_states[:4, :2],
                       hidden_states[:4] + residual[:4])
    assert spec_metadata.hidden_states.shape == (4, 2)


@skip_num_gpus_less_than(1)
def test_mtp_eagle_context_input_uses_prompt_lookahead() -> None:
    invalid = INVALID_PROMPT_LOOKAHEAD_TOKEN
    spec_config = MTPDecodingConfig(max_draft_len=3)
    spec_metadata = Eagle3OneModelSpecMetadata(
        max_num_requests=2,
        max_draft_len=3,
        max_total_draft_tokens=3,
        spec_dec_mode=spec_config.spec_dec_mode,
        num_layers=1,
        hidden_size=4,
        max_num_tokens=5,
    )
    spec_metadata.gather_ids = torch.tensor([2, 4],
                                            dtype=torch.int32,
                                            device="cuda")
    spec_metadata.runtime_draft_len = 3
    graph_metadata = spec_metadata.create_cuda_graph_metadata(1)
    assert graph_metadata.context_prompt_lookahead_tokens.shape == (1, )
    assert (graph_metadata.context_prompt_lookahead_tokens.data_ptr()
            != spec_metadata.context_prompt_lookahead_tokens.data_ptr())
    assert torch.all(graph_metadata.context_prompt_lookahead_tokens == invalid)

    # Seed both rows with stale values, including token ID zero, then clear the
    # final-chunk row with the sentinel below.
    spec_metadata.populate_context_prompt_lookahead([0, 22])
    torch.testing.assert_close(
        spec_metadata.context_prompt_lookahead_tokens[:2],
        torch.tensor([0, 22], dtype=torch.int32, device="cuda"),
    )
    spec_metadata.populate_context_prompt_lookahead([0, invalid])

    worker = MTPEagleWorker(spec_config)
    draft_inputs = worker.prepare_1st_drafter_inputs(
        input_ids=torch.tensor([10, 11, 12, 20, 21],
                               dtype=torch.int32,
                               device="cuda"),
        position_ids=torch.arange(5, dtype=torch.int32,
                                  device="cuda").unsqueeze(0),
        hidden_states=torch.zeros((5, 4), device="cuda"),
        accepted_tokens=torch.tensor(
            [[99, 0, 0, 0], [88, 0, 0, 0]],
            dtype=torch.int32,
            device="cuda",
        ),
        attn_metadata=SimpleNamespace(num_contexts=2, num_ctx_tokens=5),
        spec_metadata=spec_metadata,
        draft_model=SimpleNamespace(),
    )

    torch.testing.assert_close(
        draft_inputs["input_ids"],
        torch.tensor([11, 12, 0, 21, 88], dtype=torch.int32, device="cuda"),
    )
    torch.testing.assert_close(
        spec_metadata.context_prompt_lookahead_tokens[:2],
        torch.tensor([0, invalid], dtype=torch.int32, device="cuda"),
    )


@skip_num_gpus_less_than(1)
@pytest.mark.parametrize("fast_paths", [False, True])
def test_mtp_eagle_generation_only_draft_input_ids(fast_paths) -> None:
    """A generation-only step's draft input ids are the accepted tokens: read
    in place on the speculative fast paths, else a concatenated copy."""
    spec_config = MTPDecodingConfig(max_draft_len=3)
    spec_metadata = Eagle3OneModelSpecMetadata(
        max_num_requests=2,
        max_draft_len=3,
        max_total_draft_tokens=3,
        spec_dec_mode=spec_config.spec_dec_mode,
        num_layers=1,
        hidden_size=4,
        max_num_tokens=8,
    )
    spec_metadata.gather_ids = torch.tensor([3, 7],
                                            dtype=torch.int32,
                                            device="cuda")
    spec_metadata.runtime_draft_len = 3
    accepted = torch.arange(8, dtype=torch.int32, device="cuda").view(2, 4)
    worker = MTPEagleWorker(spec_config, spec_decode_fast_paths=fast_paths)
    draft_inputs = worker.prepare_1st_drafter_inputs(
        input_ids=torch.zeros(8, dtype=torch.int32, device="cuda"),
        position_ids=torch.arange(8, dtype=torch.int32,
                                  device="cuda").unsqueeze(0),
        hidden_states=torch.zeros((8, 4), device="cuda"),
        accepted_tokens=accepted,
        attn_metadata=SimpleNamespace(num_contexts=0, num_ctx_tokens=0),
        spec_metadata=spec_metadata,
        draft_model=SimpleNamespace(),
    )
    input_ids = draft_inputs["input_ids"]
    assert input_ids.dtype == torch.int32
    assert torch.equal(input_ids, accepted.flatten())
    assert (input_ids.data_ptr() == accepted.data_ptr()) == fast_paths


@skip_num_gpus_less_than(1)
def test_mtp_eagle_dynamic_tree_context_input_uses_prompt_lookahead() -> None:
    invalid = INVALID_PROMPT_LOOKAHEAD_TOKEN
    worker = object.__new__(MTPEagleDynamicTreeWorker)
    lookahead_tokens = torch.tensor(
        [13, invalid],
        dtype=torch.int32,
        device="cuda",
    )
    draft_inputs = worker._prepare_step0_drafter_inputs(
        input_ids=torch.tensor([10, 11, 12, 20, 21],
                               dtype=torch.int32,
                               device="cuda"),
        position_ids=torch.arange(5, dtype=torch.int32, device="cuda"),
        last_tokens_idx=torch.tensor([2, 4], dtype=torch.int32, device="cuda"),
        hidden_states=torch.zeros((5, 4), device="cuda"),
        accepted_tokens=torch.tensor(
            [[99, 0, 0, 0], [88, 0, 0, 0]],
            dtype=torch.int32,
            device="cuda",
        ),
        attn_metadata=SimpleNamespace(num_contexts=2,
                                      num_seqs=2,
                                      num_ctx_tokens=5),
        spec_metadata=SimpleNamespace(
            context_prompt_lookahead_tokens=lookahead_tokens),
    )

    torch.testing.assert_close(
        draft_inputs["input_ids"],
        torch.tensor([11, 12, 13, 21, 88], dtype=torch.int32, device="cuda"),
    )


@skip_num_gpus_less_than(1)
def test_eagle3_prepare_skipped_uploads_keep_slot_ids_current() -> None:
    slots = {5: 2, 6: 0, 7: 3}
    md = object.__new__(Eagle3OneModelSpecMetadata)
    md.use_rejection_sampling = False
    md.use_dynamic_tree = False
    md.num_tokens = 0
    md.num_generations = 0
    md.batch_indices_cuda = torch.full((8, ),
                                       -1,
                                       dtype=torch.int,
                                       device="cuda")
    md.slot_ids = torch.full((8, ), -1, dtype=torch.long, device="cuda")
    md._batch_indices_upload = DeviceCopyBuffer(md.batch_indices_cuda)
    md._slot_ids_upload = DeviceCopyBuffer(md.slot_ids)
    md.spec_resource_manager = SimpleNamespace(slot_manager=SimpleNamespace(
        get_slot=lambda rid: slots[rid]))
    md.spec_dec_mode = SimpleNamespace(is_mtp_eagle_one_model=lambda: True)

    # Repeated and prefix batches may skip the upload; a reassigned slot may not.
    for request_ids, reassign in [([5, 6], None), ([5, 6], None), ([5], None),
                                  ([5, 6, 7], None), ([5, 6, 7], (6, 4))]:
        if reassign is not None:
            slots[reassign[0]] = reassign[1]
        md.request_ids = request_ids
        md.prepare()
        torch.cuda.synchronize()
        n = len(request_ids)
        assert md.batch_indices_cuda[:n].tolist() == list(range(n))
        assert md.slot_ids[:n].tolist() == [slots[r] for r in request_ids]


def test_external_shared_target_kv_step_prepares_next_new_tokens() -> None:
    accepted = torch.tensor([[11, 12, 13]], dtype=torch.int32)
    num_accepted = torch.tensor([2], dtype=torch.int32)
    drafts = torch.tensor([[21, 22]], dtype=torch.int32)
    worker = SimpleNamespace(
        _execute_guided_decoder_if_present=lambda logits: None,
        sample_and_accept_draft_tokens=lambda *args: (accepted, num_accepted),
        _is_mamba_hybrid_cache=False,
        sa_enhancer=None,
        _uses_external_shared_target_kv=True,
        _forward_external_shared_target_kv_draft_loop=lambda **kwargs: drafts,
        _prepare_next_new_tokens=lambda *args: "next new tokens",
        _auxiliary_state_handlers=[],
    )
    spec_metadata = SimpleNamespace(runtime_draft_len=2,
                                    spec_resource_manager=None,
                                    batch_indices_cuda=torch.tensor([0]),
                                    sharded_target_logits=False)
    attn_metadata = SimpleNamespace(num_seqs=1, num_contexts=0)

    out = MTPEagleWorker._forward_impl(worker,
                                       torch.tensor([1, 2, 3]),
                                       torch.arange(3),
                                       torch.zeros(3, 8),
                                       torch.zeros(3, 16),
                                       attn_metadata,
                                       spec_metadata,
                                       draft_model=None)

    assert out["next_draft_tokens"] is drafts
    assert out["next_new_tokens"] == "next new tokens"


@pytest.fixture(scope="function")
def enforce_single_worker(monkeypatch):
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    yield


def test_kv_lens_runtime_with_eagle3_one_model():
    """
    Validates that kv_lens_runtime correctly excludes num_extra_kv_tokens when
    preparing attention metadata during EAGLE3 one-model speculative decoding.

    Background:
    - EAGLE3 reserves num_extra_kv_tokens = max_draft_len - 1 in KV cache for draft token management
    - kv_lens_runtime becomes host_past_key_value_lengths, which eventually becomes mMaxSeqLenKv in FMHA kernel
    - Bug: mMaxSeqLenKv was incorrectly set to actual_kv_length + num_extra_kv_tokens
    - Fix: mMaxSeqLenKv should be set to actual_kv_length only (without extra tokens)

    This test validates the fix by directly testing the prepare() logic.
    """

    # Test parameters
    num_seqs = 3
    num_extra_kv_tokens = 7  # e.g., max_draft_len = 8, so extra = 7
    prompt_lens = [50, 100, 75]  # These represent actual KV lengths
    seq_lens_q = [1, 1, 1]  # 1 token each in generation
    num_cached_tokens_per_seq = [
        prompt_lens[i] - seq_lens_q[i] for i in range(num_seqs)
    ]

    mock_kv_cache_manager = _make_mock_kv_cache_manager(num_seqs)

    attn_metadata = TrtllmAttentionMetadata(
        max_num_requests=num_seqs,
        max_num_tokens=sum(seq_lens_q),
        kv_cache_manager=mock_kv_cache_manager,
    )

    # Set required attributes
    attn_metadata.request_ids = list(range(1, num_seqs + 1))
    attn_metadata.prompt_lens = prompt_lens
    attn_metadata._seq_lens = torch.tensor(seq_lens_q, dtype=torch.int32)
    # seq_lens_kv is the number of new KV tokens being added in this step (for generation, same as seq_lens_q)
    attn_metadata._seq_lens_kv = torch.tensor(seq_lens_q, dtype=torch.int32)

    # Set KV cache params with num_extra_kv_tokens (EAGLE3 one-model case)
    attn_metadata.kv_cache_params = KVCacheParams(
        use_cache=True,
        num_cached_tokens_per_seq=num_cached_tokens_per_seq,
        num_extra_kv_tokens=num_extra_kv_tokens)

    attn_metadata.prepare()
    actual_kv_lengths = torch.tensor(prompt_lens, dtype=torch.int32)

    # kv_lens_runtime should equal actual KV lengths (without extra tokens)
    kv_lens_runtime = attn_metadata.kv_lens_runtime[:num_seqs]
    assert torch.equal(kv_lens_runtime, actual_kv_lengths), \
        f"kv_lens_runtime should be {actual_kv_lengths.tolist()}, but got {kv_lens_runtime.tolist()}"

    # Internal kv_lens should include extra tokens
    kv_lens_internal = attn_metadata.kv_lens[:num_seqs]
    expected_kv_lens_with_extra = actual_kv_lengths + num_extra_kv_tokens
    assert torch.equal(kv_lens_internal, expected_kv_lens_with_extra), \
        f"kv_lens should be {expected_kv_lens_with_extra.tolist()}, but got {kv_lens_internal.tolist()}"


def _make_mock_kv_cache_manager(num_seqs: int) -> MagicMock:
    mock_kv_cache_manager = MagicMock()
    mock_kv_cache_manager.tokens_per_block = 32
    mock_kv_cache_manager.num_pools = 1
    mock_kv_cache_manager.num_attention_op_pools = 1
    mock_kv_cache_manager.max_blocks_per_seq = 16
    mock_kv_cache_manager.max_batch_size = num_seqs
    mock_kv_cache_manager.max_seq_len = 512
    mock_kv_cache_manager.copy_batch_block_offsets = MagicMock()
    # MagicMock fabricates every attribute; metadata detects an FP4 MLA cache
    # layout from this hook, so a plain (non-FP4) manager must not expose it.
    mock_kv_cache_manager.fp4_mla_layout = None
    mock_kv_cache_manager.mixed_precision_high_layers = {}
    return mock_kv_cache_manager


@pytest.mark.parametrize("spec_signal", [
    None, "num_extra_kv_tokens", "is_spec_decoding_enabled",
    "has_speculative_draft_tokens", "draft_kv_cache_manager"
])
def test_block_offsets_staging_width_spec_gate(spec_signal):
    """prepare() caps the staged block-table width by the batch's max KV
    length only on the non-speculative path.

    Any speculative-decoding signal must disable the cap (max_blocks=None):
    spec kernels address block columns past the host kv_lens snapshot
    (device-side kv_lens advances in draft/tree sub-steps, draft-token blocks
    are allocated ahead), so a host-derived cap leaves columns they
    dereference unstaged. Regression test for the EAGLE3 warmup illegal
    memory access.
    """
    num_seqs = 3
    prompt_lens = [50, 100, 75]
    seq_lens_q = [1, 1, 1]
    num_cached_tokens_per_seq = [
        prompt_lens[i] - seq_lens_q[i] for i in range(num_seqs)
    ]

    mock_kv_cache_manager = _make_mock_kv_cache_manager(num_seqs)
    metadata_kwargs = dict(
        max_num_requests=num_seqs,
        max_num_tokens=sum(seq_lens_q),
        kv_cache_manager=mock_kv_cache_manager,
    )
    mock_draft_manager = None
    if spec_signal == "draft_kv_cache_manager":
        mock_draft_manager = _make_mock_kv_cache_manager(num_seqs)
        metadata_kwargs["draft_kv_cache_manager"] = mock_draft_manager

    attn_metadata = TrtllmAttentionMetadata(**metadata_kwargs)
    if spec_signal == "is_spec_decoding_enabled":
        attn_metadata.is_spec_decoding_enabled = True
    elif spec_signal == "has_speculative_draft_tokens":
        attn_metadata.runtime_features.has_speculative_draft_tokens = True

    attn_metadata.request_ids = list(range(1, num_seqs + 1))
    attn_metadata.prompt_lens = prompt_lens
    attn_metadata._seq_lens = torch.tensor(seq_lens_q, dtype=torch.int32)
    attn_metadata._seq_lens_kv = torch.tensor(seq_lens_q, dtype=torch.int32)
    attn_metadata.kv_cache_params = KVCacheParams(
        use_cache=True,
        num_cached_tokens_per_seq=num_cached_tokens_per_seq,
        num_extra_kv_tokens=(7 if spec_signal == "num_extra_kv_tokens" else 0))

    attn_metadata.prepare()

    if spec_signal is None:
        # Non-speculative: capped at ceil(max kv len / tokens_per_block).
        expected_max_blocks = -(-max(prompt_lens) //
                                mock_kv_cache_manager.tokens_per_block)
    else:
        expected_max_blocks = None
    call_kwargs = mock_kv_cache_manager.copy_batch_block_offsets.call_args.kwargs
    assert call_kwargs["max_blocks"] == expected_max_blocks
    if mock_draft_manager is not None:
        draft_kwargs = mock_draft_manager.copy_batch_block_offsets.call_args.kwargs
        assert draft_kwargs["max_blocks"] is None


def _check_spec_decoding_stats(llm: LLM) -> None:
    stats = llm.get_stats(timeout=2)
    assert stats, "Should have iteration stats"
    # Find iterations with speculation (generation phase)
    iterations_with_spec = []
    for stat in stats:
        if 'specDecodingStats' in stat:
            spec_stats = stat['specDecodingStats']
            if spec_stats.get('numDraftTokens', 0) > 0:
                iterations_with_spec.append(spec_stats)

    # Should have at least some iterations with spec decoding
    assert len(iterations_with_spec) > 0, \
        f"Should have iterations with specDecodingStats (found {len(iterations_with_spec)})"

    # Validate specDecodingStats structure and values
    for spec_stats in iterations_with_spec:
        # Check all fields are present
        assert 'numDraftTokens' in spec_stats
        assert 'numAcceptedTokens' in spec_stats
        assert 'numRequestsWithDraftTokens' in spec_stats
        assert 'acceptanceLength' in spec_stats
        assert 'iterLatencyMS' in spec_stats
        assert 'draftOverhead' in spec_stats

        # Validate value constraints
        assert spec_stats['numDraftTokens'] > 0
        assert 0 <= spec_stats['numAcceptedTokens'] <= spec_stats[
            'numDraftTokens']
        assert spec_stats['numRequestsWithDraftTokens'] > 0
        assert spec_stats['acceptanceLength'] >= 1.0
        assert spec_stats['iterLatencyMS'] >= 0.0
        assert 0.0 <= spec_stats['draftOverhead'] <= 1.0

    # Calculate overall acceptance rate
    total_draft = sum(s['numDraftTokens'] for s in iterations_with_spec)
    total_accepted = sum(s['numAcceptedTokens'] for s in iterations_with_spec)
    acceptance_rate = (total_accepted / total_draft *
                       100) if total_draft > 0 else 0

    # Should have reasonable acceptance rate for Eagle3
    assert acceptance_rate > 5.0, f"Acceptance rate too low: {acceptance_rate:.1f}%"


@pytest.mark.parametrize(
    "use_cuda_graph,disable_overlap_scheduler,enable_block_reuse,enable_chunked_prefill,use_hf_speculative_model",
    [
        pytest.param(False, False, True, False, False,
                     id="eager_overlap_reuse"),
        pytest.param(True, False, True, False, False, id="graph_overlap_reuse"),
        pytest.param(True, True, False, True, True, id="graph_chunked_hf"),
    ])
@pytest.mark.high_cuda_memory
@with_mocked_hf_download_for_single_gpu
def test_llama_eagle3(use_cuda_graph: bool, disable_overlap_scheduler: bool,
                      enable_block_reuse: bool, enable_chunked_prefill: bool,
                      use_hf_speculative_model: bool) -> None:
    """Check acceptance, metrics and short/long greedy parity for representative engines."""
    if torch.cuda.get_device_properties(0).total_memory / 1e9 < 35:
        pytest.skip("Not enough memory to load target + draft model")

    models_path = llm_models_root()
    target_model_dir = f"{models_path}/llama-3.1-model/Llama-3.1-8B-Instruct"
    eagle_model = ("yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
                   if use_hf_speculative_model else
                   f"{models_path}/EAGLE3-LLaMA3.1-Instruct-8B")
    max_draft_len = 4
    check_stats = use_cuda_graph and not disable_overlap_scheduler
    common = dict(
        model=target_model_dir,
        disable_overlap_scheduler=disable_overlap_scheduler,
        cuda_graph_config=CudaGraphConfig(
            batch_sizes=[1]) if use_cuda_graph else None,
        max_batch_size=1,
        kv_cache_config=KvCacheConfig(enable_block_reuse=enable_block_reuse,
                                      max_tokens=8192),
        # Exceed the draft checkpoint's limit to exercise its warmup bound.
        max_seq_len=8192,
        enable_chunked_prefill=enable_chunked_prefill,
    )
    if enable_chunked_prefill:
        common["max_num_tokens"] = 64
    spec_config = Eagle3DecodingConfig(max_draft_len=max_draft_len,
                                       speculative_model=eagle_model)
    long_prompt = ", ".join(str(i) for i in range(1000))
    prompts = [
        "The capital of France is", "The president of the United States is",
        long_prompt
    ]
    parity_params = SamplingParams(max_tokens=10, temperature=0)
    with LLM(**common,
             speculative_config=spec_config,
             enable_iter_perf_stats=check_stats) as llm_spec:
        acceptance_prompt = long_prompt if enable_chunked_prefill else "The future of AI is"
        token_ids = llm_spec.tokenizer.encode(acceptance_prompt)
        if enable_chunked_prefill:
            assert len(token_ids) > common["max_num_tokens"]
        num_tokens = num_drafted = num_accepted = 0
        for output in llm_spec.generate_async(token_ids,
                                              SamplingParams(max_tokens=128,
                                                             temperature=0),
                                              streaming=True):
            new_tokens = output.outputs[0].token_ids
            num_drafted += max_draft_len
            num_accepted += len(new_tokens) - num_tokens - 1
            num_tokens = len(new_tokens)
        assert num_drafted > 0
        assert num_accepted / num_drafted > 0.1
        if check_stats:
            _check_spec_decoding_stats(llm_spec)
        results_spec = llm_spec.generate(prompts, parity_params)
        spec_tokens = [result.outputs[0].token_ids for result in results_spec]

    with LLM(**common) as llm_ref:
        results_ref = llm_ref.generate(prompts, parity_params)
        ref_tokens = [result.outputs[0].token_ids for result in results_ref]
    assert len(spec_tokens) == len(ref_tokens) == len(prompts)
    for actual, expected in zip(spec_tokens, ref_tokens, strict=True):
        assert actual
        assert actual == expected


def test_deepseek_mla_eagle3():
    use_cuda_graph = True
    attn_backend = "TRTLLM"
    disable_overlap_scheduler = False
    enable_block_reuse = False
    enable_chunked_prefill = False

    # Eagle3 one model works with overlap scheduler and block reuse.
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 150:
        pytest.skip("Not enough memory to load target + draft model")

    models_path = llm_models_root()
    eagle_config = {
        "architectures": ["Eagle3DeepseekV3ForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "first_k_dense_replace": 1,
        "hidden_act": "silu",
        "hidden_size": 2560,
        "intermediate_size": 8192,
        "kv_lora_rank": 512,
        "max_position_embeddings": 4096,
        "model_type": "kimi_k2",
        "num_attention_heads": 32,
        "num_hidden_layers": 1,
        "num_key_value_heads": 32,
        "num_nextn_predict_layers": 0,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "rms_norm_eps": 1e-05,
        "rope_scaling": {
            "beta_fast": 1.0,
            "beta_slow": 1.0,
            "factor": 64.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "type": "yarn"
        },
        "rope_theta": 50000.0,
        "routed_scaling_factor": 2.827,
        "scoring_func": "sigmoid",
        "seq_aux": True,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "torch_dtype": "bfloat16",
        "torchscript": False,
        "transformers_version": "4.51.3",
        "use_bfloat16": False,
        "use_cache": True,
        "v_head_dim": 128,
        "vocab_size": 129280,
        "draft_vocab_size": 129280,
        "eagle_config": {
            "use_aux_hidden_state": True,
            "use_input_layernorm_in_first_layer": True,
            "use_last_layernorm": True,
            "use_mtp_layernorm": False
        }
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        eagle_model_dir = Path(temp_dir)
        config_path = eagle_model_dir / "config.json"
        with config_path.open("w") as f:
            json.dump(eagle_config, f, indent=2)
        target_model_dir = f"{models_path}/DeepSeek-V3-Lite/nvfp4_moe_only"

        # bs > 1 gives non-deterministic when doing IFB. There are slight chances
        # that ref and spec does not match 100%
        max_batch_size = 16
        max_draft_len = 3
        kv_cache_config = KvCacheConfig(enable_block_reuse=enable_block_reuse,
                                        max_tokens=8192)
        cuda_graph_config = CudaGraphConfig(
            batch_sizes=[1]) if use_cuda_graph else None

        llm_common_config = dict(
            model=target_model_dir,
            attn_backend=attn_backend,
            disable_overlap_scheduler=disable_overlap_scheduler,
            cuda_graph_config=cuda_graph_config,
            max_batch_size=max_batch_size,
            max_num_tokens=4096,
            max_seq_len=4096,
            kv_cache_config=kv_cache_config,
            enable_chunked_prefill=enable_chunked_prefill,
            load_format="dummy",
        )

        spec_config = Eagle3DecodingConfig(max_draft_len=max_draft_len,
                                           speculative_model=eagle_model_dir,
                                           load_format="dummy")

        llm_spec = LLM(**llm_common_config, speculative_config=spec_config)

        tok_ids = llm_spec.tokenizer.encode("The future of AI is")

        sampling_params = SamplingParams(max_tokens=32, temperature=0)
        for output in llm_spec.generate_async(tok_ids,
                                              sampling_params,
                                              streaming=True):
            pass


def test_multi_eagle3():
    use_cuda_graph = True
    attn_backend = "TRTLLM"
    disable_overlap_scheduler = False
    enable_block_reuse = False
    enable_chunked_prefill = False

    # Eagle3 one model works with overlap scheduler and block reuse.
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 150:
        pytest.skip("Not enough memory to load target + draft model")

    models_path = llm_models_root()
    eagle_config = {
        'architectures': ['LlamaForCausalLMEagle3'],
        'attention_bias': False,
        'attention_dropout': 0.0,
        'bos_token_id': 128000,
        'eos_token_id': [128001, 128008, 128009],
        'eagle_config': {
            'use_aux_hidden_state': False,
            'use_input_layernorm_in_first_layer': True,
            'use_last_layernorm': True,
            'use_mtp_layernorm': False
        },
        'head_dim': 128,
        'hidden_act': 'silu',
        'hidden_size': 4096,
        'initializer_range': 0.02,
        'intermediate_size': 16384,
        'max_position_embeddings': 131072,
        'mlp_bias': False,
        'model_type': 'llama',
        'num_attention_heads': 32,
        'num_eagle_features': 1,
        'num_hidden_layers': 2,
        'num_key_value_heads': 8,
        'pretraining_tp': 1,
        'rms_norm_eps': 1e-05,
        'rope_scaling': {
            'factor': 8.0,
            'high_freq_factor': 4.0,
            'low_freq_factor': 1.0,
            'original_max_position_embeddings': 8192,
            'rope_type': 'llama3'
        },
        'rope_theta': 500000.0,
        'tie_word_embeddings': False,
        'torch_dtype': 'bfloat16',
        'transformers_version': '4.52.4',
        'use_cache': True,
        'vocab_size': 128256,
        'draft_vocab_size': 128256,
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        eagle_model_dir = Path(temp_dir)
        config_path = eagle_model_dir / "config.json"
        with config_path.open("w") as f:
            json.dump(eagle_config, f, indent=2)
        target_model_dir = f"{models_path}/llama-3.1-model/Llama-3.1-8B-Instruct"

        # bs > 1 gives non-deterministic when doing IFB. There are slight chances
        # that ref and spec does not match 100%
        max_batch_size = 16
        max_draft_len = 3
        kv_cache_config = KvCacheConfig(enable_block_reuse=enable_block_reuse,
                                        free_gpu_memory_fraction=0.5,
                                        use_kv_cache_manager_v2=True)
        cuda_graph_config = CudaGraphConfig(
            batch_sizes=[1]) if use_cuda_graph else None

        llm_common_config = dict(
            model=target_model_dir,
            attn_backend=attn_backend,
            disable_overlap_scheduler=disable_overlap_scheduler,
            cuda_graph_config=cuda_graph_config,
            max_batch_size=max_batch_size,
            kv_cache_config=kv_cache_config,
            enable_chunked_prefill=enable_chunked_prefill,
            load_format="dummy",
        )

        spec_config = Eagle3DecodingConfig(max_draft_len=max_draft_len,
                                           speculative_model=eagle_model_dir,
                                           load_format="dummy")

        llm_spec = LLM(**llm_common_config, speculative_config=spec_config)

        tok_ids = llm_spec.tokenizer.encode("The future of AI is")

        sampling_params = SamplingParams(max_tokens=32, temperature=0)
        for output in llm_spec.generate_async(tok_ids,
                                              sampling_params,
                                              streaming=True):
            pass


def _write_tiny_eagle3_lora(directory: Path) -> None:
    """Write a nonzero rank-8 Q/V adapter matching the four-layer test target."""
    from safetensors.torch import save_file

    directory.mkdir()
    rank = 8
    generator = torch.Generator().manual_seed(17)
    weights = {}
    for layer in range(4):
        for module, out_features in (("q_proj", 512), ("v_proj", 128)):
            stem = f"base_model.model.model.layers.{layer}.self_attn.{module}"
            weights[f"{stem}.lora_A.weight"] = torch.randn(rank,
                                                           512,
                                                           generator=generator)
            weights[f"{stem}.lora_B.weight"] = torch.randn(out_features,
                                                           rank,
                                                           generator=generator)
    save_file(weights, str(directory / "adapter_model.safetensors"))
    (directory / "adapter_config.json").write_text(
        json.dumps({
            "peft_type": "LORA",
            "r": rank,
            "lora_alpha": rank,
            "target_modules": ["q_proj", "v_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "inference_mode": True,
        }))


def _run_tiny_eagle3(context: SpawnProcessContext, tmp_path: Path,
                     use_dynamic_tree: bool) -> None:
    """Run one mode in a fresh interpreter, with no nested executor process."""
    from transformers import LlamaConfig

    # Keep real attention, hidden-state capture, and draft verification without
    # depending on pretrained weights or a tokenizer.
    vocab_size = 256
    target_model_dir = tmp_path / "target"
    eagle_model = tmp_path / "draft"
    config = LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=128,
        vocab_size=vocab_size,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        torch_dtype="bfloat16",
    )
    config.save_pretrained(target_model_dir)
    config.architectures = ["LlamaForCausalLMEagle3"]
    config.num_hidden_layers = 1
    config.draft_vocab_size = vocab_size
    config.save_pretrained(eagle_model)

    max_batch_size = 2 if use_dynamic_tree else 4
    max_draft_len = 6
    dynamic_tree_max_top_k = 10
    max_total_draft_tokens = 60
    kv_cache_config = KvCacheConfig(enable_block_reuse=False, max_tokens=1024)
    # The chain's sampled requests deliberately fall back to eager execution;
    # its greedy LoRA requests replay the padded graph in the same engine.
    cuda_graph_config = (CudaGraphConfig(
        batch_sizes=[1, 2]) if use_dynamic_tree else CudaGraphConfig(
            batch_sizes=[4], enable_padding=True, sampling_variants=["greedy"]))

    llm_common_config = dict(
        model=target_model_dir,
        attn_backend="TRTLLM",
        disable_overlap_scheduler=use_dynamic_tree,
        cuda_graph_config=cuda_graph_config,
        max_batch_size=max_batch_size,
        kv_cache_config=kv_cache_config,
        max_seq_len=128,
        max_num_tokens=128,
        load_format="dummy",
        skip_tokenizer_init=True,
    )

    spec_config_kwargs = dict(
        max_draft_len=max_draft_len,
        speculative_model=eagle_model,
        use_rejection_sampling=True,
        load_format="dummy",
        eagle3_layers_to_capture={0, 1, 2},
    )
    if use_dynamic_tree:
        spec_config_kwargs.update(
            use_dynamic_tree=True,
            dynamic_tree_max_topK=dynamic_tree_max_top_k,
            max_total_draft_tokens=max_total_draft_tokens,
        )

    prompts = [[1, 3, 5, 7], [1, 4, 6, 8]]
    sampling_params = SamplingParams(max_tokens=20,
                                     temperature=1.0,
                                     top_p=1.0,
                                     end_id=-1)
    if not use_dynamic_tree:
        llm_common_config["lora_config"] = LoraConfig(max_lora_rank=8,
                                                      max_loras=1,
                                                      max_cpu_loras=1)
        adapter_dir = tmp_path / "adapter"
        _write_tiny_eagle3_lora(adapter_dir)

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
        with LLM(**llm_common_config,
                 speculative_config=Eagle3DecodingConfig(
                     **spec_config_kwargs)) as llm_spec:
            runner = llm_spec._executor.engine.model_engine.cuda_graph_runner
            replay = runner.replay
            replayed: list[KeyType] = []

            def record_replay(key: KeyType, current_inputs: dict,
                              **kwargs) -> torch.Tensor | None:
                replayed.append(key)
                return replay(key, current_inputs, **kwargs)

            patch.setattr(runner, "replay", record_replay)
            results = llm_spec.generate(prompts, sampling_params)
            assert len(results) == len(prompts)
            for result in results:
                assert len(result.outputs) == 1
                token_ids = result.outputs[0].token_ids
                assert token_ids is not None
                assert len(token_ids) == sampling_params.max_tokens
                assert all(0 <= token < vocab_size for token in token_ids)
            if use_dynamic_tree:
                assert replayed
                assert all(not key.is_all_greedy_sample for key in replayed)
            else:
                assert not replayed, "Sampled chain requests must exercise eager decoding"
                lora_prompts = prompts + [[1, 9, 11, 13]]
                greedy = SamplingParams(max_tokens=8, temperature=0, end_id=-1)
                baseline = llm_spec.generate(lora_prompts, greedy)
                replayed.clear()
                adapted = llm_spec.generate(
                    lora_prompts,
                    greedy,
                    lora_request=[LoRARequest("tiny", 1, str(adapter_dir))] * 3)
                assert any(key.batch_size == 4 and key.use_lora_graph
                           for key in replayed)
                restored = llm_spec.generate(lora_prompts, greedy)
                for outputs in (baseline, adapted, restored):
                    assert len(outputs) == 3
                    for result in outputs:
                        assert len(result.outputs) == 1
                        assert len(
                            result.outputs[0].token_ids) == greedy.max_tokens
                        assert all(0 <= token < vocab_size
                                   for token in result.outputs[0].token_ids)
                base_tokens = [
                    result.outputs[0].token_ids for result in baseline
                ]
                adapted_tokens = [
                    result.outputs[0].token_ids for result in adapted
                ]
                restored_tokens = [
                    result.outputs[0].token_ids for result in restored
                ]
                assert adapted_tokens != base_tokens, "The nonzero adapter must affect generation"
                assert restored_tokens == base_tokens, "LoRA must not leak into plain requests"
    context.send("done")


@pytest.mark.parametrize("use_dynamic_tree", [False, True],
                         ids=["chain_eager_lora_graph", "tree_graph"])
@skip_num_gpus_less_than(1)
@pytest.mark.private_mpi_session
def test_llama_eagle3_rejection_sampling_modes(use_dynamic_tree: bool,
                                               tmp_path: Path) -> None:
    # XQA's process-global cubin registry omits q_seq_len / is_spec_dec_tree.
    # Drain pytest's cached MPI workers and keep each mode in a fresh process.
    with spawn_process(_run_tiny_eagle3, tmp_path, use_dynamic_tree) as process:
        process.receive("done", timeout=540)
        assert process.wait(timeout=30) == 0


@pytest.mark.parametrize("disable_overlap_scheduler", [False, True])
@pytest.mark.parametrize("use_cuda_graph", [False, True])
@pytest.mark.high_cuda_memory
@skip_pre_blackwell
def test_nemotron_super_mtp_dynamic_tree_dl6_k10_dt31(
        use_cuda_graph: bool, disable_overlap_scheduler: bool):
    if torch.cuda.device_count() < 8:
        pytest.skip("Nemotron Super dynamic-tree MTP test requires 8 GPUs")

    models_path = llm_models_root()
    model_path = f"{models_path}/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"

    max_batch_size = 1
    max_draft_len = 6
    kv_cache_config = KvCacheConfig(enable_block_reuse=False,
                                    mamba_ssm_cache_dtype="float16",
                                    free_gpu_memory_fraction=0.8)
    cuda_graph_config = CudaGraphConfig(
        batch_sizes=[1]) if use_cuda_graph else None

    moe_tactic = sm100f_moe_tactic_or_cutlass("trtllm_gen")
    llm_common_config = dict(
        model=model_path,
        tensor_parallel_size=8,
        moe_expert_parallel_size=8,
        pipeline_parallel_size=1,
        moe_config=MoeConfig(tactic=moe_tactic),
        disable_overlap_scheduler=disable_overlap_scheduler,
        cuda_graph_config=cuda_graph_config,
        max_batch_size=max_batch_size,
        kv_cache_config=kv_cache_config,
        max_seq_len=8192,
    )
    # The target's tactic does not reach the MTP layers, so they pin it too.
    spec_config = MTPDecodingConfig(max_draft_len=max_draft_len,
                                    use_dynamic_tree=True,
                                    dynamic_tree_max_topK=10,
                                    max_total_draft_tokens=31,
                                    moe_tactic=moe_tactic)

    llm_spec = LLM(**llm_common_config, speculative_config=spec_config)
    prompt = llm_spec.tokenizer.apply_chat_template(
        [{
            "role": "user",
            "content": "The future of AI is"
        }],
        tokenize=False,
        add_generation_prompt=True,
    )
    tok_ids = llm_spec.tokenizer.encode(prompt)

    sampling_params = SamplingParams(max_tokens=128, temperature=0)
    num_tokens = 0
    num_drafted = 0
    num_accepted = 0
    for output in llm_spec.generate_async(tok_ids,
                                          sampling_params,
                                          streaming=True):
        new_tokens = output.outputs[0].token_ids
        num_drafted += max_draft_len
        num_accepted += len(new_tokens) - num_tokens - 1
        num_tokens = len(new_tokens)

    accept_rate = num_accepted / num_drafted
    assert accept_rate > 0.20

    sampling_params = SamplingParams(max_tokens=10, temperature=0)
    results_spec = llm_spec.generate([prompt], sampling_params)
    generated_text_spec = [result.outputs[0].text for result in results_spec]
    llm_spec.shutdown()

    llm_ref = LLM(**llm_common_config)
    results_ref = llm_ref.generate([prompt], sampling_params)
    generated_text_ref = [result.outputs[0].text for result in results_ref]
    llm_ref.shutdown()

    for text_spec, text_ref in zip(generated_text_spec, generated_text_ref):
        assert text_spec == text_ref


@pytest.mark.parametrize("disable_overlap_scheduler", [False])
@pytest.mark.parametrize("use_cuda_graph", [True])
# dl3_k4_dt12_bs1 covers one request against a mask buffer sized for one request on the Blackwell SwapsMmaAb kernel.
@pytest.mark.parametrize(
    "max_draft_len,dynamic_tree_max_topK,max_total_draft_tokens,max_batch_size,kv_cache_max_tokens",
    [
        pytest.param(6, 10, 30, 4, 2048, id="dl6_k10_dt30"),
        pytest.param(
            3, 4, 12, 1, 4096, id="dl3_k4_dt12_bs1", marks=skip_pre_blackwell),
    ])
@pytest.mark.high_cuda_memory
@with_mocked_hf_download_for_single_gpu
def test_llama_eagle3_dynamic_tree(
        use_cuda_graph: bool, disable_overlap_scheduler: bool,
        max_draft_len: int, dynamic_tree_max_topK: int,
        max_total_draft_tokens: int, max_batch_size: int,
        kv_cache_max_tokens: int):
    """Test EAGLE3 dynamic tree speculative decoding with one-model architecture."""
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 35:
        pytest.skip("Not enough memory to load target + draft model")

    models_path = llm_models_root()
    target_model_dir = f"{models_path}/llama-3.1-model/Llama-3.1-8B-Instruct"
    eagle_model = f"{models_path}/EAGLE3-LLaMA3.1-Instruct-8B"

    kv_cache_config = KvCacheConfig(enable_block_reuse=False,
                                    max_tokens=kv_cache_max_tokens,
                                    free_gpu_memory_fraction=0.5)
    cuda_graph_config = CudaGraphConfig(
        batch_sizes=[i for i in range(1, max_batch_size +
                                      1)]) if use_cuda_graph else None

    llm_common_config = dict(
        model=target_model_dir,
        attn_backend="TRTLLM",
        disable_overlap_scheduler=disable_overlap_scheduler,
        cuda_graph_config=cuda_graph_config,
        max_batch_size=max_batch_size,
        kv_cache_config=kv_cache_config,
        max_seq_len=2048,
    )

    spec_config = Eagle3DecodingConfig(
        max_draft_len=max_draft_len,
        speculative_model=eagle_model,
        use_dynamic_tree=True,
        dynamic_tree_max_topK=dynamic_tree_max_topK,
        max_total_draft_tokens=max_total_draft_tokens,
    )

    # Create the LLM instance
    llm_spec = LLM(**llm_common_config, speculative_config=spec_config)

    # Acceptance rate tests
    prompts = [
        "The capital of France is",
        "The president of the United States is",
    ]
    tok_ids = [llm_spec.tokenizer.encode("The future of AI is")]

    sampling_params = SamplingParams(max_tokens=128, temperature=0)

    for i in range(len(tok_ids)):
        num_tokens = 0
        num_drafted = 0
        num_accepted = 0

        for output in llm_spec.generate_async(tok_ids[i],
                                              sampling_params,
                                              streaming=True):
            new_tokens = output.outputs[0].token_ids
            num_drafted += max_draft_len
            num_accepted += len(new_tokens) - num_tokens - 1
            num_tokens = len(new_tokens)

        accept_rate = num_accepted / num_drafted
        assert accept_rate > 0.10

    # Check that batched dynamic-tree decoding produces meaningful text.
    sampling_params = SamplingParams(max_tokens=10, temperature=0)

    results_spec = llm_spec.generate(prompts, sampling_params)
    generated_text_spec = [result.outputs[0].text for result in results_spec]
    llm_spec.shutdown()

    def assert_meaningful_text(text: str) -> None:
        stripped = text.strip()
        assert stripped
        assert "\ufffd" not in stripped
        assert any(ch.isalpha() for ch in stripped)
        words = stripped.lower().split()
        assert not any(
            len(set(words[i:i + 6])) == 1 for i in range(len(words) - 5))

    assert len(generated_text_spec) == len(prompts)
    for text_spec in generated_text_spec:
        assert_meaningful_text(text_spec)


if __name__ == "__main__":
    unittest.main()
