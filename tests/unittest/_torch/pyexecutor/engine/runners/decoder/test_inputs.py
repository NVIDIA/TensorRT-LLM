# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import contextlib
import unittest
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest
import torch
from _torch.pyexecutor.engine.runners.decoder.utils import decoder_config, input_preparer
from _torch.pyexecutor.engine_test_utils import (
    DummyModel,
    _create_request,
    _create_request_with_tokens,
    _make_request_stub,
)
from _torch.pyexecutor.engine_test_utils import kv_cache_manager as managed_kv_cache
from utils.util import skip_pre_blackwell_unittest, skip_ray

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionMetadata,
    AttentionRuntimeFeatures,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.attention.backends.vanilla import VanillaAttention
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs import (
    DecoderInputBuffers,
    DecoderInputPreparer,
    HostInputs,
    _get_context_prompt_lookahead_token,
    _zero_fill_row_spans,
)
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.metadata import DecoderMetadata
from tensorrt_llm._torch.pyexecutor.engine.runners.interface import PreparedInputs
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.speculative.interface import INVALID_PROMPT_LOOKAHEAD_TOKEN
from tensorrt_llm._torch.speculative.spec_sampler_base import SampleStateTensorsSpec
from tensorrt_llm.bindings import SamplingConfig
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalItemMetadata,
)
from tensorrt_llm.llmapi import (
    MTPDecodingConfig,
    NGramDecodingConfig,
    SADecodingConfig,
    SamplingParams,
)
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import CpType, Mapping

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

    from tensorrt_llm._torch.kernels.spec_step_copies.op import StepInputGather
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.config import DecoderRunnerConfig
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig


def _input_preparer(*, spec_config: DecodingBaseConfig | None = None) -> DecoderInputPreparer:
    return input_preparer(decoder_config(is_spec_decode=False, spec_config=spec_config), model=None)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("enabled", "fused"), [(None, False), (False, True), (True, False), (True, True)]
)
def test_input_preparer_configured_confidence_flags(enabled: bool | None, fused: bool) -> None:
    spec_config = (
        None
        if enabled is None
        else SimpleNamespace(
            enable_confidence_scheduling=enabled, enable_fused_confidence_scheduler=fused
        )
    )
    config = SimpleNamespace(is_spec_decode=False, spec_config=spec_config)
    inputs = DecoderInputPreparer(
        buffers=None,
        config=config,
        mapping=Mapping(),
        dist=None,
        model=None,
        get_runtime_tokens_per_gen_step=None,
        iter_states={},
        lora=None,
        use_beam_search=False,
        steady_gen_cache_supported=True,
    )

    assert inputs._config is config
    assert inputs._dspark_confidence_enabled is bool(enabled)
    assert inputs._dspark_trims_submitted_tokens is bool(enabled)
    assert inputs._dspark_device_windows is bool(enabled and fused)
    assert inputs.previous_verify_lens == ([] if enabled else None)
    assert inputs.has_previous_device_draft is False


@pytest.mark.cpu_only
def test_input_preparer_without_config_keeps_confidence_disabled() -> None:
    inputs = DecoderInputPreparer(
        buffers=None,
        config=None,
        mapping=Mapping(),
        dist=None,
        model=None,
        get_runtime_tokens_per_gen_step=None,
        iter_states={},
        lora=None,
        use_beam_search=False,
        steady_gen_cache_supported=True,
    )

    assert inputs._config is None
    assert inputs._dspark_confidence_enabled is False
    assert inputs._dspark_trims_submitted_tokens is False
    assert inputs._dspark_device_windows is False
    assert inputs.previous_verify_lens is None
    assert inputs.has_previous_device_draft is False


@contextlib.contextmanager
def input_preparer_and_kv_cache(
    config: DecoderRunnerConfig | None = None,
    *,
    spec_config: DecodingBaseConfig | None = None,
    model: DummyModel | None = None,
    max_tokens: int = 258,
    max_batch_size: int = 13,
) -> Iterator[tuple[DecoderInputPreparer, KVCacheManagerV2]]:
    config = config or decoder_config(spec_config=spec_config)
    model = (
        model
        if model is not None
        else DummyModel(config.dtype, rope_scaling={"type": "mrope"} if config.use_mrope else None)
    )
    inputs = input_preparer(
        config,
        model=model,
        buffers=DecoderInputBuffers.for_config(config, config.max_beam_width > 1),
    )
    with managed_kv_cache(
        model=model, max_tokens=max_tokens, max_batch_size=max_batch_size
    ) as manager:
        yield inputs, manager


@pytest.mark.cpu_only
class TestCachedKvTokenLogging(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = _input_preparer()
        self.real = SimpleNamespace(is_cuda_graph_dummy=False)
        self.padding = SimpleNamespace(is_cuda_graph_dummy=True)

    def check_counts(
        self,
        counts: list[int],
        groups: tuple[tuple[list[SimpleNamespace], int], ...],
        expected: list[int],
        padding: int,
    ) -> None:
        self.inputs.iter_states["cached_kv_tokens"] = sum(counts)
        self.inputs.record_cached_kv_tokens_per_req(counts, groups)
        self.assertEqual(
            self.inputs.iter_states,
            {
                "cached_kv_tokens": sum(counts),
                "cached_kv_tokens_per_req": expected,
                "cached_kv_tokens_cuda_graph_padding": padding,
            },
        )
        counts[:] = [999]
        self.assertEqual(self.inputs.iter_states["cached_kv_tokens_per_req"], expected)

    def test_mtp_padding_before_generation(self) -> None:
        # Input packing is context, extend (including padding), gen.
        self.check_counts(
            [10, 100, 7, 200],
            (([self.real], 1), ([self.real, self.padding], 1), ([self.real], 1)),
            [10, 100, 200],
            7,
        )

    def test_beams_preserved_after_context_and_padding(self) -> None:
        self.check_counts(
            [10, 100, 101, 7, 7, 200, 201],
            (([self.real], 1), ([self.real, self.padding, self.real], 2)),
            [10, 100, 101, 200, 201],
            14,
        )

    def test_attention_dp_dummy_is_not_graph_padding(self) -> None:
        adp_dummy = SimpleNamespace(
            is_dummy=True, is_attention_dp_dummy=True, is_cuda_graph_dummy=False
        )
        self.check_counts([10, 1, 7], (([self.real, adp_dummy, self.padding], 1),), [10, 1], 7)

    def test_padding_only(self) -> None:
        self.check_counts([7, 7], (([self.padding, self.padding], 1),), [], 14)

    def test_group_row_count_must_match_counts(self) -> None:
        for counts, groups in (
            ([100, 200], (([self.real], 1),)),
            ([100, 101, 200], (([self.real, self.real], 2),)),
        ):
            with self.subTest(counts=counts, groups=groups):
                with self.assertRaises(AssertionError):
                    self.inputs.record_cached_kv_tokens_per_req(counts, groups)

    def test_steady_batch_clears_previous_padding(self) -> None:
        self.inputs.iter_states["cached_kv_tokens_cuda_graph_padding"] = 7
        self.check_counts([100, 200], (), [100, 200], 0)

    def test_empty_batch_clears_previous_values(self) -> None:
        self.inputs.iter_states["cached_kv_tokens_per_req"] = [100]
        self.inputs.iter_states["cached_kv_tokens_cuda_graph_padding"] = 7
        self.check_counts([], (([], 1),), [], 0)


@pytest.mark.cpu_only
def test_context_prompt_lookahead_stops_at_prompt_boundary() -> None:
    """An internal chunk hands back the next prompt token; the last one does not.

    The boundary is read off the request's live chunking state rather than a
    Python-side prompt length, which a preemption leaves stale.
    """
    request = _create_request_with_tokens([10, 11, 12, 13, 14], 1)

    request.context_chunk_size = 2
    assert _get_context_prompt_lookahead_token(request, 2) == 12

    # A reused prefix advances the live cursor while py_prompt_len remains the
    # original prompt length. Internal chunk [2, 4) still looks up token 4.
    request.context_current_position = 2
    request.context_chunk_size = 2
    assert _get_context_prompt_lookahead_token(request, 4) == 14

    # Final chunk [2, 5): nothing follows in the prompt, so the drafter falls
    # back to the token the target just sampled.
    request.context_chunk_size = 3
    assert _get_context_prompt_lookahead_token(request, 5) == INVALID_PROMPT_LOOKAHEAD_TOKEN


@pytest.mark.cpu_only
def test_zero_fill_row_spans_merges_unsorted_and_adjacent_spans() -> None:
    buffer = torch.arange(1, 13, dtype=torch.int32)
    _zero_fill_row_spans(buffer, [(9, 11), (2, 4), (4, 5), (7, 8)])
    assert buffer.tolist() == [1, 2, 0, 0, 0, 6, 7, 0, 9, 0, 0, 12]


@pytest.mark.cpu_only
def test_zero_fill_row_spans_without_spans_is_noop() -> None:
    buffer = torch.arange(4, dtype=torch.int32)
    _zero_fill_row_spans(buffer, [])
    assert buffer.tolist() == [0, 1, 2, 3]


class DecoderInputPreparerTestCase(unittest.TestCase):
    def test_prepare_inputs_returns_steady_gen_fast_inputs(self) -> None:
        inputs = _input_preparer()
        inputs.guided_decoder = Mock()
        inputs._can_use_steady_gen_fast_prepare = Mock(return_value=True)
        fast_inputs = PreparedInputs({"prepared": True})
        inputs._apply_steady_gen_fast_prepare = Mock(return_value=fast_inputs)
        batch = ScheduledRequests()
        new_tokens = torch.zeros(1)

        result = inputs.prepare_inputs(
            batch,
            Mock(),
            Mock(),
            new_tensors_device=SimpleNamespace(new_tokens=new_tokens),
            enable_spec_decode=False,
            runtime_draft_len=3,
            is_dummy=True,
        )

        self.assertIs(result.kwargs, fast_inputs.kwargs)
        self.assertEqual(result.runtime_draft_len, 3)
        inputs.guided_decoder.add_batch.assert_called_once_with(
            batch, new_tokens=new_tokens, runtime_draft_len=3
        )
        inputs._can_use_steady_gen_fast_prepare.assert_called_once_with(
            batch, new_tokens, None, None, True
        )

    def test_prepare_inputs_reports_tree_draft_length(self) -> None:
        inputs = _input_preparer(
            spec_config=SimpleNamespace(is_linear_tree=False, tokens_per_gen_step=8)
        )
        inputs._can_use_steady_gen_fast_prepare = Mock(return_value=True)
        inputs._apply_steady_gen_fast_prepare = Mock(return_value=PreparedInputs({}))

        for enable_spec_decode, expected in ((True, 7), (False, 2)):
            with self.subTest(enable_spec_decode=enable_spec_decode):
                result = inputs.prepare_inputs(
                    ScheduledRequests(),
                    Mock(),
                    Mock(),
                    enable_spec_decode=enable_spec_decode,
                    runtime_draft_len=2,
                    is_dummy=True,
                )
                self.assertEqual(result.runtime_draft_len, expected)

    def test_steady_gen_fast_prepare_skips_dummy_passes(self) -> None:
        inputs = _input_preparer()
        request = _make_request_stub(7)
        inputs._steady_gen_cache = {
            "num_requests": 1,
            "request_ids": [request.py_request_id],
        }
        batch = ScheduledRequests()
        batch.generation_requests = [request]
        new_tokens = torch.zeros(1)

        self.assertTrue(
            inputs._can_use_steady_gen_fast_prepare(batch, new_tokens, None, None, False)
        )
        self.assertFalse(
            inputs._can_use_steady_gen_fast_prepare(batch, new_tokens, None, None, True)
        )

    def test_padding_request_input_id_rows_are_zeroed(self) -> None:
        (preparer, kv_cache_manager) = self.enterContext(input_preparer_and_kv_cache())
        resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})
        attn_metadata = AttentionMetadata(
            max_num_requests=4, max_num_tokens=32, kv_cache_manager=kv_cache_manager
        )
        attn_metadata.is_cuda_graph = False

        generation = _create_request_with_tokens([50, 51, 52, 53, 54], 1)
        generation.py_seq_slot = 0
        generation.py_batch_idx = 0
        padding = _create_request_with_tokens([60, 61, 62], 2)
        padding.is_cuda_graph_dummy = True
        padding.py_seq_slot = 1

        graph_batch = ScheduledRequests()
        graph_batch.generation_requests = [generation, padding]
        new_tokens = torch.zeros((1, 4, 1), dtype=torch.int32, device="cuda")
        new_tokens[0, 0, 0] = 777
        overlap_state = SimpleNamespace(new_tokens=new_tokens)
        inputs = preparer
        inputs._can_use_steady_gen_fast_prepare = Mock(return_value=False)
        # The padding row is written by neither input-id copy, so poison the
        # buffer with an out-of-vocabulary id to catch a stale read.
        preparer._buffers.input_ids_cuda.fill_(1 << 30)

        prepared = inputs.prepare_inputs(
            scheduled_requests=graph_batch,
            kv_cache_manager=kv_cache_manager,
            attn_metadata=attn_metadata,
            new_tensors_device=overlap_state,
            resource_manager=resource_manager,
            enable_spec_decode=False,
            runtime_draft_len=0,
            is_dummy=False,
        )

        self.assertEqual(prepared.kwargs["input_ids"][:2].cpu().tolist(), [777, 0])

    def test_promoted_context_precedes_speculative_overlap_generation(self) -> None:
        spec_config = SADecodingConfig(
            max_draft_len=1,
            draft_len_schedule={1: 1},
        )
        (preparer, kv_cache_manager) = self.enterContext(
            input_preparer_and_kv_cache(spec_config=spec_config)
        )
        resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})
        attn_metadata = AttentionMetadata(
            max_num_requests=4, max_num_tokens=32, kv_cache_manager=kv_cache_manager
        )
        attn_metadata.is_cuda_graph = False
        # A bare Mock auto-vivifies every attribute, so optional metadata
        # fields read by prepare_inputs must be pinned off explicitly.
        spec_metadata = Mock(
            _force_non_greedy_for_capture=False,
            context_prompt_lookahead_tokens=None,
        )

        context = _create_request_with_tokens([11, 22, 33, 44], 1)
        context.context_current_position = 3
        context.context_chunk_size = 1
        context.py_seq_slot = 0
        # A promoted context row must ignore any stale overlap slot.
        context.py_batch_idx = 3

        generation = _create_request_with_tokens([50, 51, 52, 53, 54], 2)
        generation.py_seq_slot = 1
        generation.py_batch_idx = 1
        generation.py_needs_onehot_draft_probs = True

        graph_batch = ScheduledRequests()
        graph_batch.generation_requests = [context, generation]
        new_tokens = torch.zeros((1, 4, 1), dtype=torch.int32, device="cuda")
        new_tokens[0, 0, 0] = 999
        new_tokens[0, 1, 0] = 777
        overlap_state = SampleStateTensorsSpec(
            new_tokens=new_tokens,
            new_tokens_lens=torch.ones(4, dtype=torch.int32, device="cuda"),
            next_draft_tokens=torch.zeros((4, 1), dtype=torch.int32, device="cuda"),
        )

        prepared = preparer.prepare_inputs(
            scheduled_requests=graph_batch,
            kv_cache_manager=kv_cache_manager,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            new_tensors_device=overlap_state,
            resource_manager=resource_manager,
            promoted_context_request_ids=frozenset({context.py_request_id}),
            enable_spec_decode=True,
            runtime_draft_len=0,
            is_dummy=False,
        )

        inputs = prepared.kwargs
        self.assertEqual(inputs["input_ids"][:2].cpu().tolist(), [44, 777])
        self.assertEqual(inputs["position_ids"][0, :2].cpu().tolist(), [3, 4])
        self.assertEqual(
            attn_metadata.request_ids, [context.py_request_id, generation.py_request_id]
        )
        self.assertEqual(attn_metadata.kv_cache_params.num_cached_tokens_per_seq, [3, 5])
        self.assertEqual(preparer._buffers.previous_batch_indices_cuda[:1].cpu().tolist(), [1])
        self.assertEqual(preparer._buffers.previous_pos_id_offsets_cuda[:2].cpu().tolist(), [0, 1])
        self.assertEqual(attn_metadata.num_contexts, 0)
        self.assertEqual(
            spec_metadata.request_ids, [context.py_request_id, generation.py_request_id]
        )
        self.assertFalse(generation.py_needs_onehot_draft_probs)
        spec_metadata.write_padding_onehot_draft_probs.assert_called_once_with(
            [generation.py_seq_slot], 0
        )

    def test_spec_decode_graph_step_gather_kernel_matches_torch_path(self) -> None:
        """A CUDA graph decode step of a speculative engine whose overlap
        gathers run as one StepInputGather launch writes what the torch path
        writes."""
        from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
        from tensorrt_llm._torch.kernels.spec_step_copies import op as spec_step_copies

        if not spec_step_copies.is_supported():
            self.skipTest("the step-copy kernels run on SM 100")
        max_draft_len = 3
        tokens_per_step = max_draft_len + 1
        (preparer, kv_cache_manager) = self.enterContext(
            input_preparer_and_kv_cache(spec_config=SADecodingConfig(max_draft_len=max_draft_len))
        )
        inputs = preparer
        buffers = preparer._buffers
        step_gather = inputs._step_input_gather
        self.assertIsNotNone(step_gather)
        resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        # Three requests that ran in the previous step, in slots out of order.
        slots = [5, 2, 7]
        requests = kv_cache_manager.add_dummy_requests(
            [1, 2, 3], token_nums=[10, 17, 33], is_gen=True, max_num_draft_tokens=max_draft_len
        )
        for request, slot in zip(requests, slots):
            request.is_dummy_request = False
            request.py_seq_slot = slot
        batch = ScheduledRequests()
        batch.generation_requests = requests
        attn_metadata = DecoderMetadata(
            config=preparer._config,
            mapping=Mapping(),
            model=preparer.model,
            buffers=preparer._buffers,
            attention_runtime_features=AttentionRuntimeFeatures(),
        ).setup_attention(kv_cache_manager)
        self.assertIs(type(attn_metadata), TrtllmAttentionMetadata)
        graph_metadata = attn_metadata.create_cuda_graph_metadata(
            len(requests), False, max_draft_len
        )
        spec_metadata = Mock(
            _force_non_greedy_for_capture=False,
            context_prompt_lookahead_tokens=None,
            supports_steady_prepare=False,
        )

        # The sampler's slot stores as the previous step left them.
        num_slots = 8
        generator = torch.Generator(device="cuda").manual_seed(0)
        previous = SampleStateTensorsSpec(
            new_tokens=torch.randint(
                0,
                1000,
                (tokens_per_step, num_slots, 1),
                generator=generator,
                dtype=torch.int32,
                device="cuda",
            ),
            new_tokens_lens=torch.randint(
                1,
                tokens_per_step + 1,
                (num_slots,),
                generator=generator,
                dtype=torch.int32,
                device="cuda",
            ),
            next_draft_tokens=torch.randint(
                0,
                1000,
                (num_slots, max_draft_len),
                generator=generator,
                dtype=torch.int32,
                device="cuda",
            ),
        )

        def step_inputs(step_input_gather: StepInputGather | None) -> list[torch.Tensor]:
            """The buffers one step writes, from a filler they all start at."""
            inputs._step_input_gather = step_input_gather
            for request, slot in zip(requests, slots):
                request.py_batch_idx = slot
            written = (
                buffers.input_ids_cuda,
                buffers.position_ids_cuda,
                buffers.draft_tokens_cuda,
                buffers.previous_pos_id_offsets_cuda,
                buffers.previous_kv_lens_offsets_cuda,
            )
            for buffer in written:
                buffer.fill_(-3)
            inputs.prepare_inputs(
                scheduled_requests=batch,
                kv_cache_manager=kv_cache_manager,
                attn_metadata=graph_metadata,
                spec_metadata=spec_metadata,
                new_tensors_device=previous,
                resource_manager=resource_manager,
                enable_spec_decode=True,
                runtime_draft_len=max_draft_len,
                is_dummy=False,
            )
            torch.cuda.synchronize()
            return [buffer.clone() for buffer in written]

        torch_path = step_inputs(None)
        launched = []
        launch = step_gather.gather

        def recorded_launch(*args: object, **kwargs: object) -> bool:
            launched.append(launch(*args, **kwargs))
            return launched[-1]

        with patch.object(step_gather, "gather", side_effect=recorded_launch):
            kernel_path = step_inputs(step_gather)
        self.assertEqual(launched, [True])
        for expected, actual in zip(torch_path, kernel_path):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # The previous rows' input ids came from the stores, by slot.
        num_tokens = len(requests) * tokens_per_step
        expected_input_ids = previous.new_tokens[:, slots, 0].t().reshape(-1)
        self.assertEqual(kernel_path[0][:num_tokens].tolist(), expected_input_ids.tolist())

    @skip_ray
    def test_prepare_inputs_with_helix_parallelism(self) -> None:
        """Test prepare_inputs with helix parallelism."""

        # Create model engine with helix parallelism.

        # Provide mapping for model engine.
        cp_size = 2
        cp_rank = 0
        cp_config = {"cp_type": CpType.HELIX, "tokens_per_block": 4}
        mapping = Mapping(
            world_size=cp_size,
            tp_size=1,
            pp_size=1,
            cp_size=cp_size,
            cp_config=cp_config,
            rank=cp_rank,
        )
        config = decoder_config(max_num_tokens=512)
        preparer = input_preparer(
            config,
            model=DummyModel(torch.half),
            mapping=mapping,
            buffers=DecoderInputBuffers.for_config(config, False),
        )

        # Create scheduled requests with two generation requests.
        scheduled_requests = ScheduledRequests()
        scheduled_requests.context_requests_last_chunk = []
        prompt_lens = [20, 15]
        gen_requests = []
        for idx in range(len(prompt_lens)):
            req = _create_request(num_tokens=prompt_lens[idx], req_id=idx + 1)
            req.py_prompt_len = prompt_lens[idx]
            req.py_batch_idx = None
            req.is_dummy_request = False
            req.py_seq_slot = idx
            req.sampling_config.beam_width = 1
            req.py_multimodal_data = {}
            req.total_input_len_cp = prompt_lens[idx] * 2
            req.seqlen_this_rank_cp = prompt_lens[idx]
            req.py_decoding_iter = 1
            gen_requests.append(req)
        scheduled_requests.generation_requests = gen_requests

        # Create KV cache manager for attention metadata.
        kv_cache_config = KvCacheConfig(max_tokens=512, enable_block_reuse=False)
        kv_cache_manager = self.enterContext(
            managed_kv_cache(
                model=preparer.model,
                max_tokens=512,
                max_batch_size=4,
                mapping=mapping,
                config=kv_cache_config,
            )
        )
        kv_cache_manager.add_dummy_requests([1, 2], prompt_lens, is_gen=True)
        attn_metadata = TrtllmAttentionMetadata(
            max_num_requests=4,
            max_num_tokens=512,
            kv_cache_manager=kv_cache_manager,
            mapping=mapping,
        )
        attn_metadata.is_cuda_graph = False

        # Initialize model engine buffers.

        result = preparer.prepare_inputs(
            scheduled_requests=scheduled_requests,
            kv_cache_manager=kv_cache_manager,
            attn_metadata=attn_metadata,
            enable_spec_decode=False,
            runtime_draft_len=0,
            is_dummy=False,
        ).kwargs

        # Verify expected keys are present.
        self.assertIsNotNone(result)
        self.assertIn("input_ids", result)
        self.assertIn("position_ids", result)
        self.assertIn("attn_metadata", result)

        # Also, verify that position_ids are properly calculated.
        position_ids = result["position_ids"]
        self.assertIsInstance(position_ids, torch.Tensor)
        expected_positions = [40, 30]
        actual_positions = position_ids.squeeze(0).cpu().tolist()[:2]
        self.assertEqual(
            actual_positions,
            expected_positions,
            f"Position IDs should reflect CP allgather results. Expected: {expected_positions}, "
            f"Got: {actual_positions}",
        )

        # Verify attention metadata is properly configured.
        self.assertEqual(attn_metadata.request_ids, [1, 2])
        self.assertEqual(attn_metadata.prompt_lens, [20, 15])
        self.assertEqual(attn_metadata.num_contexts, 0)

        # Verify KV cache parameters
        self.assertIsNotNone(attn_metadata.kv_cache_params)
        self.assertTrue(attn_metadata.kv_cache_params.use_cache)

        # Verify sequence lengths are correct.
        expected_seq_lens = [1, 1]
        if hasattr(attn_metadata, "seq_lens") and attn_metadata.seq_lens is not None:
            actual_seq_lens = attn_metadata.seq_lens.cpu().tolist()
            self.assertEqual(actual_seq_lens, expected_seq_lens)

    def test_mrope_multi_beam_generation_positions(self) -> None:
        """Every beam row gets its request's `past + delta` on all 3 axes."""
        beam_width = 2
        config = decoder_config(
            max_batch_size=4, max_num_tokens=64, max_beam_width=beam_width, use_mrope=True
        )
        preparer = input_preparer(
            config,
            model=DummyModel(torch.half, rope_scaling={"type": "mrope"}),
            buffers=DecoderInputBuffers.for_config(config, config.max_beam_width > 1),
        )
        # (num_tokens, delta): one delta arrives on the host and one is
        # already a CUDA int32 scalar.
        cases = (
            (4, torch.tensor([[10]], dtype=torch.int32)),
            (7, torch.tensor([[-3]], dtype=torch.int32, device="cuda")),
        )
        requests = []
        for seq_slot, (num_tokens, delta) in enumerate(cases):
            request = LlmRequest(
                request_id=seq_slot + 1,
                max_new_tokens=1,
                input_tokens=list(range(1, num_tokens + 1)),
                sampling_config=SamplingConfig(
                    SamplingParams(n=beam_width, use_beam_search=True)._get_sampling_config()
                ),
                is_streaming=False,
            )
            request.py_seq_slot = seq_slot
            request.py_multimodal_data = {"mrope_config": {"mrope_position_deltas": delta}}
            requests.append(request)
        host = HostInputs(
            use_mrope=True,
            mrope_dummy_seq_slot=config.max_num_tokens,
            extra_inputs=preparer._new_extra_inputs(),
            context_prompt_lookahead=None,
        )
        host.lists.generation_requests.extend(requests)

        batched_mrope_generation_updates: list[tuple[int, int, torch.Tensor]] = []
        preparer._pack_generation_requests(
            host,
            frozenset(),
            None,
            batched_mrope_generation_updates=batched_mrope_generation_updates,
        )
        host.total_num_tokens = len(host.lists.position_ids)
        preparer._copy_position_ids_to_device(
            host, batched_mrope_generation_updates=batched_mrope_generation_updates
        )

        # Both deltas take the batched scalar path, one row per beam.
        self.assertEqual([row for row, _, _ in batched_mrope_generation_updates], [0, 1, 2, 3])
        self.assertEqual(host.lists.mrope_position_ids, [])
        # Without a previous tensor the past length is num_tokens - 1.
        expected_row = [
            num_tokens - 1 + int(delta) for num_tokens, delta in cases for _ in range(beam_width)
        ]
        self.assertEqual(expected_row, [13, 13, 3, 3])
        torch.testing.assert_close(
            preparer._buffers.mrope_position_ids_cuda[:, :, :4],
            torch.tensor([[expected_row]] * 3, dtype=torch.int32, device="cuda"),
            atol=0,
            rtol=0,
        )

    def test_mrope_mixed_batched_and_fallback_positions_match_fallback(self) -> None:
        """Mixed scalar batching matches the old per-request writes exactly."""
        config = decoder_config(max_batch_size=2, max_num_tokens=64, use_mrope=True)
        preparer = input_preparer(
            config,
            model=DummyModel(torch.half, rope_scaling={"type": "mrope"}),
            buffers=DecoderInputBuffers.for_config(config, config.max_beam_width > 1),
        )
        sentinel = 123

        def run(
            delta_dtypes: tuple[torch.dtype, torch.dtype],
        ) -> tuple[HostInputs, list[tuple[int, int, torch.Tensor]], torch.Tensor]:
            requests = []
            for seq_slot, (num_tokens, delta, dtype) in enumerate(
                zip((4, 7), (10, -3), delta_dtypes)
            ):
                request = LlmRequest(
                    request_id=seq_slot + 1,
                    max_new_tokens=1,
                    input_tokens=list(range(1, num_tokens + 1)),
                    sampling_config=SamplingConfig(SamplingParams()._get_sampling_config()),
                    is_streaming=False,
                )
                request.py_seq_slot = seq_slot
                request.py_multimodal_data = {
                    "mrope_config": {
                        "mrope_position_deltas": torch.tensor([[delta]], dtype=dtype, device="cuda")
                    }
                }
                requests.append(request)
            host = HostInputs(
                use_mrope=True,
                mrope_dummy_seq_slot=config.max_num_tokens,
                extra_inputs=preparer._new_extra_inputs(),
                context_prompt_lookahead=None,
            )
            host.lists.generation_requests.extend(requests)
            batched_mrope_generation_updates: list[tuple[int, int, torch.Tensor]] = []
            preparer._pack_generation_requests(
                host,
                frozenset(),
                None,
                batched_mrope_generation_updates=batched_mrope_generation_updates,
            )
            host.total_num_tokens = len(host.lists.position_ids)
            preparer._buffers.mrope_position_ids_cuda.fill_(sentinel)
            preparer._copy_position_ids_to_device(
                host, batched_mrope_generation_updates=batched_mrope_generation_updates
            )
            positions = preparer._buffers.mrope_position_ids_cuda.clone()
            return host, batched_mrope_generation_updates, positions

        mixed_host, mixed_updates, mixed = run((torch.int32, torch.int64))
        _, fallback_updates, fallback = run((torch.int64, torch.int64))

        self.assertEqual([row for row, _, _ in mixed_updates], [0])
        self.assertEqual(
            [(start, end) for start, end, _ in mixed_host.lists.mrope_position_ids], [(1, 2)]
        )
        self.assertEqual(fallback_updates, [])
        torch.testing.assert_close(mixed, fallback, atol=0, rtol=0)
        self.assertTrue(torch.all(mixed[:, :, 2:] == sentinel).item())

    def _setup_mrope_inputs(
        self,
        max_num_tokens: int = 32,
        spec_config: DecodingBaseConfig | None = None,
        **config_changes: Any,
    ) -> tuple[DecoderInputPreparer, KVCacheManagerV2, AttentionMetadata]:
        config = decoder_config(
            **{
                "max_num_tokens": max_num_tokens,
                "spec_config": spec_config,
                "use_mrope": True,
                **config_changes,
            }
        )
        inputs, manager = self.enterContext(
            input_preparer_and_kv_cache(
                config,
                max_tokens=max_num_tokens,
                max_batch_size=4,
            )
        )
        metadata = AttentionMetadata(
            max_num_requests=4, max_num_tokens=max_num_tokens, kv_cache_manager=manager
        )
        metadata.is_cuda_graph = False
        return inputs, manager, metadata

    @staticmethod
    def _make_mrope_gen_request(
        num_tokens: int, req_id: int, seq_slot: int, delta: int | None
    ) -> LlmRequest:
        """Generation request carrying an MRoPE delta when `delta` is not None,
        and no multimodal data at all otherwise (a text-only prompt)."""
        request = _create_request(num_tokens, req_id)
        request.py_prompt_len = num_tokens
        request.py_batch_idx = None
        request.py_seq_slot = seq_slot
        request.sampling_config.beam_width = 1
        if delta is None:
            request.py_multimodal_data = {}
        else:
            request.py_multimodal_data = {
                "mrope_config": {
                    "mrope_position_deltas": torch.tensor([[delta]], dtype=torch.int32)
                },
            }
        return request

    def _make_mrope_spec_batch(
        self, previous_batch: bool
    ) -> tuple[ScheduledRequests, list[LlmRequest]]:
        """Two multimodal verify groups around a text-only one, plus a CUDA graph
        padding dummy, each carrying one draft token.

        Returns the scheduled batch and the multimodal requests."""
        # (num_tokens, req_id, seq_slot, delta); the middle request is text-only.
        requests = [
            self._make_mrope_gen_request(4, 1, 0, 10),
            self._make_mrope_gen_request(5, 2, 1, None),
            self._make_mrope_gen_request(6, 3, 2, 20),
        ]
        for request in requests:
            request.py_draft_tokens = [7]
            if previous_batch:
                # The overlap scheduler's previous step ran these requests at
                # their current slots, which also seeded the delta cache.
                request.py_batch_idx = request.py_seq_slot
                if request.py_multimodal_data:
                    request.py_mrope_delta_cache_slot = request.py_seq_slot
        # The padding dummy carries the zero delta that `add_dummy_requests`
        # attaches on MRoPE models.
        dummy = self._make_mrope_gen_request(2, 4, 3, 0)
        dummy.is_cuda_graph_dummy = True
        dummy.py_draft_tokens = [0]

        scheduled_requests = ScheduledRequests()
        scheduled_requests.context_requests_last_chunk = []
        scheduled_requests.generation_requests = requests + [dummy]
        return scheduled_requests, [requests[0], requests[2]]

    def test_prepare_inputs_spec_decode_adds_mrope_delta(self) -> None:
        """One-model speculative verify groups get the request's MRoPE delta.

        Every token a generation step feeds is text after the prompt's vision
        spans, so its three coordinates are `seq_index + delta`, as for an
        ordinary generation request. Read slots stay dense, one per verify
        group, and a generation-only request seeds the delta cache.
        """
        preparer, kv_cache_manager, attn_metadata = self._setup_mrope_inputs(
            spec_config=SADecodingConfig(max_draft_len=1)
        )
        scheduled_requests, multimodal_requests = self._make_mrope_spec_batch(previous_batch=False)
        spec_metadata = Mock(
            _force_non_greedy_for_capture=False,
            context_prompt_lookahead_tokens=None,
        )

        result = preparer.prepare_inputs(
            scheduled_requests=scheduled_requests,
            kv_cache_manager=kv_cache_manager,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            enable_spec_decode=True,
            runtime_draft_len=1,
            is_dummy=False,
        ).kwargs

        # Each verify group is [last token, draft token] at past_seen + 0/1;
        # the text-only request and the zero-delta dummy keep the scalar
        # positions.
        position_ids = result["position_ids"]
        self.assertEqual(tuple(position_ids.shape), (3, 1, 8))
        expected = torch.tensor(
            [[[13, 14, 4, 5, 25, 26, 1, 2]]] * 3, dtype=torch.int32, device="cuda"
        )
        torch.testing.assert_close(position_ids, expected, atol=0, rtol=0)
        # Slot 32 is the reserved zero slot (max_num_tokens * pp_size).
        self.assertEqual(result["mrope_delta_read_seq_slots"].cpu().tolist(), [0, 32, 2, 32])
        self.assertEqual(result["mrope_delta_write_seq_slots"].cpu().tolist(), [0, 2])
        self.assertEqual(
            [request.py_mrope_delta_cache_slot for request in multimodal_requests], [0, 2]
        )

    def test_prepare_inputs_spec_decode_overlap_keeps_mrope_delta(self) -> None:
        """Under the overlap scheduler the accepted-token correction applied in
        `preprocess_inputs` lands on top of the delta, on all three axes."""
        preparer, kv_cache_manager, attn_metadata = self._setup_mrope_inputs(
            spec_config=SADecodingConfig(max_draft_len=1)
        )
        scheduled_requests, _ = self._make_mrope_spec_batch(previous_batch=True)
        spec_metadata = Mock(
            _force_non_greedy_for_capture=False,
            context_prompt_lookahead_tokens=None,
        )
        # The previous step accepted 2, 1 and 2 tokens in slots 0, 1 and 2.
        overlap_state = SampleStateTensorsSpec(
            new_tokens=torch.zeros((2, 4, 1), dtype=torch.int32, device="cuda"),
            new_tokens_lens=torch.tensor([2, 1, 2, 1], dtype=torch.int32, device="cuda"),
            next_draft_tokens=torch.zeros((4, 1), dtype=torch.int32, device="cuda"),
        )

        inputs = preparer.prepare_inputs(
            scheduled_requests=scheduled_requests,
            kv_cache_manager=kv_cache_manager,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            new_tensors_device=overlap_state,
            enable_spec_decode=True,
            runtime_draft_len=1,
            is_dummy=False,
        ).kwargs

        # The host packs positions from the stale base (past_seen + delta).
        stale = torch.tensor([[[13, 14, 4, 5, 25, 26, 1, 2]]] * 3, dtype=torch.int32, device="cuda")
        torch.testing.assert_close(inputs["position_ids"], stale, atol=0, rtol=0)
        self.assertEqual(inputs["mrope_delta_read_seq_slots"].cpu().tolist(), [0, 32, 2, 32])
        # The delta cache already holds both multimodal requests' deltas.
        self.assertNotIn("mrope_delta_write_seq_slots", inputs)

        inputs = preparer.preprocess_inputs(inputs, enable_spec_decode=True, runtime_draft_len=1)
        corrected = torch.tensor(
            [[[15, 16, 5, 6, 27, 28, 1, 2]]] * 3, dtype=torch.int32, device="cuda"
        )
        torch.testing.assert_close(inputs["position_ids"], corrected, atol=0, rtol=0)

    @staticmethod
    def _make_mrope_ctx_request(
        num_tokens: int, req_id: int, seq_slot: int, delta: int
    ) -> LlmRequest:
        """Single-chunk context request carrying MRoPE metadata.

        Its position on axis `a` of token `i` is `100 + 1000 * a + i`, so a
        spliced span says which axis and token it came from."""
        request = _create_request(num_tokens, req_id)
        request.py_prompt_len = num_tokens
        request.py_batch_idx = None
        request.py_seq_slot = seq_slot
        request.context_current_position = 0
        request.context_chunk_size = num_tokens
        request.sampling_config.beam_width = 1
        positions = (
            100 + 1000 * torch.arange(3).view(3, 1, 1) + torch.arange(num_tokens).view(1, 1, -1)
        ).to(torch.int32)
        request.py_multimodal_data = {
            "mrope_config": {
                "mrope_position_ids": positions,
                "mrope_position_deltas": torch.tensor([[delta]], dtype=torch.int32),
            },
        }
        return request

    @staticmethod
    def _prepare_mrope_spec_inputs(
        preparer: DecoderInputPreparer,
        kv_cache_manager: KVCacheManagerV2,
        attn_metadata: AttentionMetadata,
        scheduled_requests: ScheduledRequests,
    ) -> dict[str, Any]:
        spec_metadata = Mock(
            _force_non_greedy_for_capture=False,
            context_prompt_lookahead_tokens=None,
        )
        result = preparer.prepare_inputs(
            scheduled_requests=scheduled_requests,
            kv_cache_manager=kv_cache_manager,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            enable_spec_decode=True,
            runtime_draft_len=1,
            is_dummy=False,
        ).kwargs
        return result

    def _assert_mrope_positions(
        self, result: dict[str, Any], rows: list[int] | list[list[int]]
    ) -> None:
        """`rows` is one list for all three axes, or one list per axis."""
        if not isinstance(rows[0], list):
            rows = [rows] * 3
        position_ids = result["position_ids"]
        self.assertEqual(tuple(position_ids.shape), (3, 1, len(rows[0])))
        expected = torch.tensor([rows[0:1], rows[1:2], rows[2:3]], dtype=torch.int32, device="cuda")
        torch.testing.assert_close(position_ids, expected, atol=0, rtol=0)

    def test_prepare_inputs_spec_decode_mrope_plain_rows_follow_verify_groups(self) -> None:
        """Verify groups are packed ahead of plain generation requests, and the
        MRoPE delta read slots follow that packing rather than scheduler order.

        The attention kernel indexes the deltas by generation batch index: row
        i of the packed generation batch must read the delta of the request
        packed i-th, so a verify group's slot comes before every plain
        request's, whatever order the scheduler listed them in.
        """
        preparer, kv_cache_manager, attn_metadata = self._setup_mrope_inputs(
            spec_config=MTPDecodingConfig(max_draft_len=1)
        )
        # `verify_*` carry a draft token, `plain_*` do not; `*_mm` have an
        # MRoPE delta, `*_text` none.
        plain_mm = self._make_mrope_gen_request(6, 1, 0, 20)
        verify_mm = self._make_mrope_gen_request(4, 2, 1, 10)
        plain_text = self._make_mrope_gen_request(5, 3, 2, None)
        verify_text = self._make_mrope_gen_request(3, 4, 3, None)
        verify_mm.py_draft_tokens = [7]
        verify_text.py_draft_tokens = [7]

        scheduled_requests = ScheduledRequests()
        scheduled_requests.context_requests_last_chunk = []
        scheduled_requests.generation_requests = [plain_mm, verify_mm, plain_text, verify_text]
        result = self._prepare_mrope_spec_inputs(
            preparer, kv_cache_manager, attn_metadata, scheduled_requests
        )

        # Packed order: verify_mm, verify_text, plain_mm, plain_text. The
        # verify groups hold two rows each, the plain requests one.
        self._assert_mrope_positions(result, [13, 14, 2, 3, 25, 4])
        # Slot 32 is the reserved zero slot (max_num_tokens * pp_size) read by
        # the text-only requests.
        self.assertEqual(result["mrope_delta_read_seq_slots"].cpu().tolist(), [1, 32, 0, 32])
        self.assertEqual(result["mrope_delta_write_seq_slots"].cpu().tolist(), [1, 0])

    def test_prepare_inputs_spec_decode_mrope_context_takes_no_read_slot(self) -> None:
        """A context request packs ahead of the verify groups with its own
        MRoPE positions and takes no generation read slot, so the verify
        groups' slots start the list."""
        preparer, kv_cache_manager, attn_metadata = self._setup_mrope_inputs(
            spec_config=MTPDecodingConfig(max_draft_len=1)
        )
        # A context request with multimodal content takes the placeholder-token
        # split, which reads the vocabulary size.
        preparer.model.config.vocab_size = 1000
        context = self._make_mrope_ctx_request(5, 1, 0, 30)
        verify_mm = self._make_mrope_gen_request(4, 2, 1, 10)
        verify_mm.py_draft_tokens = [7]
        plain_mm = self._make_mrope_gen_request(6, 3, 2, 20)

        scheduled_requests = ScheduledRequests()
        scheduled_requests.context_requests_last_chunk = [context]
        scheduled_requests.generation_requests = [plain_mm, verify_mm]
        result = self._prepare_mrope_spec_inputs(
            preparer, kv_cache_manager, attn_metadata, scheduled_requests
        )

        # The context span keeps the processor's per-axis positions; the
        # generation rows after it are text at position + delta on all axes.
        self._assert_mrope_positions(
            result,
            [
                [100, 101, 102, 103, 104, 13, 14, 25],
                [1100, 1101, 1102, 1103, 1104, 13, 14, 25],
                [2100, 2101, 2102, 2103, 2104, 13, 14, 25],
            ],
        )
        self.assertEqual(result["mrope_delta_read_seq_slots"].cpu().tolist(), [1, 2])
        # The context request seeds the delta cache first, then the verify
        # group and the plain request seed theirs.
        self.assertEqual(result["mrope_delta_write_seq_slots"].cpu().tolist(), [0, 1, 2])
        self.assertEqual(context.py_mrope_delta_cache_slot, 0)

    def _setup_extend_as_context_inputs(
        self,
    ) -> tuple[DecoderInputPreparer, KVCacheManagerV2, AttentionMetadata]:
        inputs, manager, metadata = self._setup_mrope_inputs(
            spec_config=NGramDecodingConfig(max_draft_len=1),
            attention_backend=VanillaAttention,
            disable_overlap_scheduler=True,
        )
        spec_config = inputs._config.spec_config
        self.assertTrue(spec_config.spec_dec_mode.extend_ctx(VanillaAttention))
        self.assertTrue(spec_config.is_linear_tree)
        return inputs, manager, metadata

    def test_prepare_inputs_mrope_verify_as_context_takes_no_read_slot(self) -> None:
        """When the kernel runs verify groups as chunked context rows, their
        rotary comes from `position_ids` alone and only the plain generation
        requests take read slots."""
        preparer, kv_cache_manager, attn_metadata = self._setup_extend_as_context_inputs()
        verify_mm = self._make_mrope_gen_request(4, 1, 0, 10)
        verify_text = self._make_mrope_gen_request(3, 2, 1, None)
        plain_mm = self._make_mrope_gen_request(6, 3, 2, 20)
        plain_text = self._make_mrope_gen_request(5, 4, 3, None)
        verify_mm.py_draft_tokens = [7]
        verify_text.py_draft_tokens = [7]

        scheduled_requests = ScheduledRequests()
        scheduled_requests.context_requests_last_chunk = []
        scheduled_requests.generation_requests = [verify_mm, verify_text, plain_mm, plain_text]
        result = self._prepare_mrope_spec_inputs(
            preparer, kv_cache_manager, attn_metadata, scheduled_requests
        )

        self._assert_mrope_positions(result, [13, 14, 2, 3, 25, 4])
        self.assertEqual(result["mrope_delta_read_seq_slots"].cpu().tolist(), [2, 32])
        # The verify group still seeds the delta cache for the plain steps
        # that follow it.
        self.assertEqual(result["mrope_delta_write_seq_slots"].cpu().tolist(), [0, 2])

    def test_prepare_inputs_mrope_verify_as_context_alone_emits_no_read_slots(self) -> None:
        """A batch whose only delta carriers are verify groups run as context
        rows has nothing for the generation-indexed read."""
        preparer, kv_cache_manager, attn_metadata = self._setup_extend_as_context_inputs()
        verify_mm = self._make_mrope_gen_request(4, 1, 0, 10)
        verify_mm.py_draft_tokens = [7]
        plain_text = self._make_mrope_gen_request(5, 2, 1, None)

        scheduled_requests = ScheduledRequests()
        scheduled_requests.context_requests_last_chunk = []
        scheduled_requests.generation_requests = [verify_mm, plain_text]
        result = self._prepare_mrope_spec_inputs(
            preparer, kv_cache_manager, attn_metadata, scheduled_requests
        )

        self._assert_mrope_positions(result, [13, 14, 4])
        self.assertNotIn("mrope_delta_read_seq_slots", result)
        self.assertEqual(result["mrope_delta_write_seq_slots"].cpu().tolist(), [0])

    @skip_pre_blackwell_unittest
    def test_ced_warmup_passes_dummy_context_requests(self) -> None:
        for warmup in (False, True):
            with self.subTest(warmup=warmup):
                preparer, manager, metadata = self._setup_mrope_inputs(
                    1024, use_mrope=False, disable_overlap_scheduler=True
                )
                manager.has_private_swa_suffix = Mock(return_value=True)
                preparer.model.model_config.pretrained_config.rope_scaling = None
                preparer.model.model = SimpleNamespace(
                    ced_kv_precompute=True, decoder_replay_split=20
                )
                scheduled = ScheduledRequests()
                scheduled.reset_context_requests(
                    manager.add_dummy_requests([0], token_nums=[257], is_gen=False)
                )
                try:
                    prepared = preparer.prepare_inputs(
                        scheduled,
                        manager,
                        metadata,
                        enable_spec_decode=False,
                        runtime_draft_len=0,
                        is_dummy=warmup,
                    )
                    inputs = prepared.kwargs
                    self.assertEqual(inputs["context_requests"], scheduled.context_requests)
                    self.assertTrue(inputs["context_requests"][0].is_dummy)
                    self.assertEqual(metadata.seq_lens.tolist(), [257])
                    self.assertEqual(inputs["input_ids"].numel(), 257)
                finally:
                    for req in scheduled.context_requests:
                        manager.free_resources(req)

    @skip_pre_blackwell_unittest
    def test_ced_inputs_preserve_physical_rows_and_token_sources(self) -> None:
        from tensorrt_llm._torch.pyexecutor.ced_replay import EncoderReplay

        for overlap, mixed in ((False, True), (True, True), (True, False)):
            with self.subTest(overlap=overlap, mixed=mixed):
                preparer, manager, metadata = self._setup_mrope_inputs(
                    5001, use_mrope=False, disable_overlap_scheduler=not overlap
                )
                manager.has_private_swa_suffix = Mock(return_value=True)
                preparer.model.model_config.pretrained_config.rope_scaling = None
                preparer.model.model = SimpleNamespace(
                    ced_kv_precompute=True, decoder_replay_split=20
                )
                scheduled = ScheduledRequests()
                if mixed:
                    context = _create_request_with_tokens(list(range(5000)), 7)
                    context.context_current_position = 4000
                    context.context_chunk_size = 1000
                    context.py_seq_slot = 0
                    context.py_ced_replay = EncoderReplay(7, 123, 4000, 3872)
                    scheduled.context_requests_last_chunk = [context]
                for index, (length, previous_index) in enumerate(((20, 2), (30, 0))):
                    req = _create_request(length, 8 + index)
                    req.py_seq_slot = index + 1
                    req.py_batch_idx = previous_index
                    scheduled.generation_requests.append(req)
                new_tokens = torch.tensor([[[222], [999], [111]]], dtype=torch.int32, device="cuda")
                prepared = preparer.prepare_inputs(
                    scheduled,
                    manager,
                    metadata,
                    new_tensors_device=SimpleNamespace(new_tokens=new_tokens) if overlap else None,
                    enable_spec_decode=False,
                    runtime_draft_len=0,
                    is_dummy=False,
                )
                inputs = prepared.kwargs
                expected_context = list(range(3872, 5000)) if mixed else []
                self.assertEqual(
                    inputs["input_ids"].cpu().tolist(),
                    expected_context + ([111, 222] if overlap else [0, 0]),
                )
                self.assertEqual(
                    inputs["position_ids"].view(-1).cpu().tolist(),
                    expected_context + ([20, 30] if overlap else [19, 29]),
                )
                if mixed:
                    self.assertEqual(inputs["context_requests"], [context])
                    self.assertEqual(metadata.seq_lens.tolist(), [1128, 1, 1])
                    self.assertEqual(context.context_current_position, 4000)
                    self.assertEqual(context.context_chunk_size, 1000)
                    self.assertEqual(context.cached_tokens, 4000)
                    if not overlap:
                        self.assertEqual(
                            list(metadata.kv_cache_params.num_cached_tokens_per_seq),
                            [3872, 19, 29],
                        )
                else:
                    self.assertNotIn("context_requests", inputs)
                # GPU sampling is one step ahead of CPU token append.
                self.assertEqual(
                    [r.max_beam_num_tokens for r in scheduled.generation_requests], [20, 30]
                )

    def test_item_scheduled_chunk_embedding_is_not_retained(self) -> None:
        """The per-forward item-scheduler payload must not replace the
        request's own multimodal data, or each chunked prefill would keep its
        last chunk's joined embedding rows until prefill completes."""
        (preparer, kv_cache_manager) = self.enterContext(input_preparer_and_kv_cache())
        preparer.model.config.vocab_size = 100
        attn_metadata = AttentionMetadata(
            max_num_requests=4, max_num_tokens=32, kv_cache_manager=kv_cache_manager
        )
        attn_metadata.is_cuda_graph = False

        context = _create_request_with_tokens([11, 22, 33, 44], 1)
        context.context_chunk_size = 2
        context.py_seq_slot = 0
        context.py_mm_encoder_state = object()
        multimodal_data = {
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                [("image", 0)], output_embedding_lengths=[2]
            )
        }
        context.py_multimodal_data = multimodal_data
        preparer._mm_item_scheduler = Mock()
        preparer._mm_item_scheduler.build_multimodal_data_for_llm.return_value = {
            "multimodal_embedding": torch.ones((2, 1), dtype=torch.float16),
            "multimodal_embedding_is_chunk": True,
        }
        scheduled_requests = ScheduledRequests()
        scheduled_requests.context_requests_chunking = [context]

        preparer.prepare_inputs(
            scheduled_requests=scheduled_requests,
            kv_cache_manager=kv_cache_manager,
            attn_metadata=attn_metadata,
            enable_spec_decode=False,
            runtime_draft_len=0,
            is_dummy=False,
        )

        preparer._mm_item_scheduler.build_multimodal_data_for_llm.assert_called_once()
        self.assertIs(context.py_multimodal_data, multimodal_data)
        self.assertNotIn("multimodal_embedding", multimodal_data)
