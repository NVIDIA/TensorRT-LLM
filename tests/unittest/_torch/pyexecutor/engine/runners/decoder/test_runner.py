# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import contextlib
import unittest
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest
import torch
from _torch.pyexecutor.engine.runners.decoder.utils import decoder_config
from _torch.pyexecutor.engine_test_utils import _make_request_stub

import tensorrt_llm._torch.pyexecutor.engine.model_call as model_call_module
import tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner as decoder_runner_module
from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import KeyType
from tensorrt_llm._torch.pyexecutor.engine.model_call import ModelCaller
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import DecoderRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs import DecoderInputBuffers
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner import PrefillCompiledModel
from tensorrt_llm._torch.pyexecutor.engine.runners.interface import PreparedInputs, ScheduledInputs
from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.warmup_timer import _WarmupTimer
from tensorrt_llm._torch.utils import (
    get_model_extra_attrs,
    is_torch_compiling,
    set_per_request_prefill_cuda_graph_flag,
    torch_compiling,
)
from tensorrt_llm.mapping import Mapping

if TYPE_CHECKING:
    from typing import Any

    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager

_TWO_MODEL_SPEC_MODE = SimpleNamespace(use_one_engine=lambda: False)


def _make_decoder_runner(
    graph_key: KeyType | None,
    runner_enabled: bool = True,
    private_decoder: bool = False,
    **config_changes: Any,
) -> tuple[DecoderRunner, Mock, Mock, Mock, dict[str, object]]:
    config = decoder_config(
        **{
            "disable_overlap_scheduler": True,
            "is_spec_decode": False,
            **config_changes,
        }
    )
    if config.compile_only_piecewise_graphs:
        model = DecoderModelForCausalLM.__new__(DecoderModelForCausalLM)
        torch.nn.Module.__init__(model)
        model.model = PrefillCompiledModel(torch.nn.Identity(), torch.nn.Identity())
        model.extra_attrs = {}
        model.model_config = SimpleNamespace(pretrained_config=SimpleNamespace(rope_scaling=None))
    else:
        model = SimpleNamespace(
            extra_attrs={},
            model_config=SimpleNamespace(pretrained_config=SimpleNamespace(rope_scaling=None)),
        )
    outputs = {"logits": object()}
    backend = Mock()
    backend.enabled = runner_enabled
    backend.pad_batch.side_effect = lambda batch, *_args: nullcontext(batch)
    graph_metadata = Mock()
    backend.maybe_get_cuda_graph.return_value = (
        (graph_metadata, None, graph_key) if graph_key is not None else (None, None, None)
    )
    backend.get_graph_pool.return_value = None
    backend.needs_capture.return_value = False
    backend.is_warmup_only = False
    backend.replay.return_value = outputs
    buffers = SimpleNamespace(
        input_ids_cuda=torch.zeros(2, dtype=torch.int32),
        draft_tokens_cuda=None,
        previous_pos_id_offsets_cuda=None,
    )
    with (
        patch.object(DecoderInputBuffers, "for_config", return_value=buffers),
        patch.object(decoder_runner_module, "CUDAGraphRunner", return_value=backend),
        patch.object(decoder_runner_module, "CutlassWorkspaceReclaimer"),
    ):
        decoder = DecoderRunner(
            model,
            config,
            model_caller=Mock(),
            mapping=Mapping(),
            dist=None,
            moe_load_balancer=None,
            sparse_attention_config=None,
            attention_runtime_features=AttentionRuntimeFeatures(),
            get_runtime_tokens_per_gen_step=Mock(return_value=1),
            warmup_timer=_WarmupTimer(rank=0),
            iter_states={},
            metrics={},
        )
    metadata = Mock()
    decoder._metadata.attn_metadata = metadata
    decoder._metadata.setup_attention = Mock(return_value=metadata)
    spec_mode = Mock()
    spec_mode.attention_need_spec_dec_mode.return_value = False
    spec_mode.is_parallel_draft.return_value = False
    spec_metadata = Mock(
        spec_dec_mode=spec_mode, is_spec_dec_tree=False, is_spec_dec_dynamic_tree=False
    )
    decoder._metadata.spec_metadata = spec_metadata
    decoder._metadata.setup_speculative = Mock(return_value=spec_metadata)
    decoder._inputs.prepare_inputs = Mock(
        return_value=PreparedInputs(
            {
                "prepared": True,
                "input_ids": buffers.input_ids_cuda,
            }
        )
    )
    decoder._inputs.record_promoted_iter_states = Mock()
    decoder._inputs.record_input_copy = Mock()
    decoder._forward_step = Mock(return_value=outputs)
    decoder._execute_logit_post_processors = Mock()
    resources = Mock()
    peft_cache_manager = Mock(data_type=torch.bfloat16)

    def get_resource_manager(resource_type: ResourceManagerType) -> object:
        if resource_type == ResourceManagerType.PEFT_CACHE_MANAGER:
            return peft_cache_manager
        if resource_type == ResourceManagerType.KV_CACHE_MANAGER:
            return SimpleNamespace(has_private_swa_suffix=Mock(return_value=private_decoder))
        return object()

    resources.get_resource_manager.side_effect = get_resource_manager
    return decoder, backend, resources, metadata, outputs


@pytest.mark.cpu_only
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "ordinary_capture,eligible",
    [(False, True), (False, False), (True, True), (True, False)],
)
def test_decoder_forward_routes_compile_bypass(
    enabled: bool, ordinary_capture: bool, eligible: bool
) -> None:
    """Exercise both the regular forward and ordinary-graph capture callbacks."""
    key = KeyType(batch_size=1, draft_len=0) if ordinary_capture else None
    engine, graph_runner, resources, _, outputs = _make_decoder_runner(
        key, compile_only_piecewise_graphs=enabled
    )
    eager = torch.nn.Identity()
    compiled = torch.nn.Identity()
    eager.forward = Mock(return_value=outputs)
    compiled.forward = Mock(return_value=outputs)
    wrapper = PrefillCompiledModel(eager, compiled)
    engine._model.model = wrapper
    engine._forward_step.side_effect = lambda *args, **kwargs: wrapper()
    graph_runner.needs_capture.return_value = ordinary_capture
    graph_runner.is_warmup_only = ordinary_capture
    graph_runner.capture.side_effect = lambda key, forward, inputs, **kwargs: forward(inputs)
    batch = ScheduledRequests()
    batch.generation_requests = [_make_request_stub(1)]

    with (
        patch(
            "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs.torch.cuda.Event",
            return_value=Mock(),
        ),
        patch(
            "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner.get_per_request_prefill_cuda_graph_flag",
            return_value=eligible,
        ),
    ):
        result = engine.forward(ScheduledInputs(batch=batch), resource_manager=resources)

    assert result["logits"] is outputs["logits"]
    expected_compiled = eligible and not (enabled and ordinary_capture)
    selected = compiled.forward if expected_compiled else eager.forward
    other = eager.forward if expected_compiled else compiled.forward
    selected.assert_called_once_with()
    other.assert_not_called()
    assert not wrapper._bypass_active


@pytest.mark.cpu_only
class DecoderGraphForwardTestCase(unittest.TestCase):
    def test_ced_final_context_cannot_promote_to_decode_graph(self) -> None:
        key = KeyType(batch_size=1, draft_len=0)
        engine, runner, resource_manager, metadata, outputs = _make_decoder_runner(
            key, private_decoder=True
        )
        engine._model.model = SimpleNamespace(ced_kv_precompute=True, decoder_replay_split=20)
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [_make_request_stub(1)]
        with (
            patch("tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs.torch.cuda.Event"),
            patch(
                "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.graphs._make_single_token_context_graph_batch"
            ) as promote,
        ):
            actual_outputs = engine.forward(
                ScheduledInputs(batch=batch), resource_manager=resource_manager
            )
            self.assertEqual(actual_outputs, {**outputs, "runtime_draft_len": 0})
            self.assertIs(actual_outputs["logits"], outputs["logits"])
        promote.assert_not_called()
        runner.replay.assert_not_called()
        engine._forward_step.assert_called_once()
        self.assertIs(engine._inputs.prepare_inputs.call_args.args[2], metadata)

    def test_zero_runtime_draft_speculation_commits_graph_candidate(self) -> None:
        key = KeyType(batch_size=2, draft_len=0)
        engine, runner, resource_manager, semantic_attn_metadata, outputs = _make_decoder_runner(
            key,
            spec_config=SimpleNamespace(
                draft_len_schedule=None,
                spec_dec_mode=_TWO_MODEL_SPEC_MODE,
                is_linear_tree=True,
                get_runtime_tokens_per_gen_step=Mock(return_value=1),
            ),
        )
        graph_attn_metadata = runner.maybe_get_cuda_graph.return_value[0]
        runner.maybe_get_cuda_graph.return_value = (
            graph_attn_metadata,
            engine._metadata.spec_metadata,
            key,
        )
        context = _make_request_stub(1)
        generation = _make_request_stub(2)
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [context]
        batch.generation_requests = [generation]

        with patch(
            "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs.torch.cuda.Event",
            return_value=Mock(),
        ):
            actual_outputs = engine.forward(
                ScheduledInputs(batch=batch, enable_spec_decode=True),
                resource_manager=resource_manager,
            )

        self.assertEqual(actual_outputs, {**outputs, "runtime_draft_len": 0})
        self.assertIsNot(actual_outputs, outputs)
        graph_batch = runner.maybe_get_cuda_graph.call_args.args[0]
        self.assertEqual(graph_batch.generation_requests, [context, generation])
        self.assertTrue(runner.maybe_get_cuda_graph.call_args.kwargs["enable_spec_decode"])
        engine._metadata.spec_metadata.update_is_all_greedy_sample.assert_called_once_with(
            graph_batch.all_requests()
        )
        prepare_args = engine._inputs.prepare_inputs.call_args.args
        self.assertIs(prepare_args[0], graph_batch)
        self.assertIs(prepare_args[3], engine._metadata.spec_metadata)
        self.assertEqual(prepare_args[-1], frozenset({context.py_request_id}))
        semantic_attn_metadata.update_spec_dec_param.assert_called_once()
        self.assertEqual(
            semantic_attn_metadata.update_spec_dec_param.call_args.kwargs["num_contexts"], 1
        )
        prepared_inputs = engine._inputs.prepare_inputs.return_value.kwargs
        runner.replay.assert_called_once_with(
            key, prepared_inputs, position_id_offsets=None, static_inputs_staged=0
        )

    def test_zero_runtime_draft_speculation_graph_miss_is_semantic_eager(self) -> None:
        engine, runner, resource_manager, semantic_attn_metadata, outputs = _make_decoder_runner(
            None,
            spec_config=SimpleNamespace(
                draft_len_schedule=None,
                spec_dec_mode=_TWO_MODEL_SPEC_MODE,
                is_linear_tree=True,
                get_runtime_tokens_per_gen_step=Mock(return_value=1),
            ),
        )
        context = _make_request_stub(1)
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [context]

        with patch(
            "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs.torch.cuda.Event",
            return_value=Mock(),
        ):
            actual_outputs = engine.forward(
                ScheduledInputs(batch=batch, enable_spec_decode=True),
                resource_manager=resource_manager,
            )

        self.assertEqual(actual_outputs, {**outputs, "runtime_draft_len": 0})
        self.assertIsNot(actual_outputs, outputs)
        graph_batch = runner.maybe_get_cuda_graph.call_args.args[0]
        self.assertEqual(graph_batch.generation_requests, [context])
        prepare_args = engine._inputs.prepare_inputs.call_args.args
        self.assertIs(prepare_args[0], batch)
        self.assertIs(prepare_args[2], semantic_attn_metadata)
        self.assertIs(prepare_args[3], engine._metadata.spec_metadata)
        self.assertEqual(prepare_args[-1], frozenset())
        engine._forward_step.assert_called_once()
        runner.replay.assert_not_called()

    def test_zero_runtime_non_linear_tree_speculation_uses_semantic_eager_batch(self) -> None:
        engine, runner, resource_manager, semantic_attn_metadata, outputs = _make_decoder_runner(
            None,
            spec_config=SimpleNamespace(
                draft_len_schedule=None,
                spec_dec_mode=_TWO_MODEL_SPEC_MODE,
                is_linear_tree=False,
                tokens_per_gen_step=8,
                get_runtime_tokens_per_gen_step=Mock(return_value=1),
            ),
        )
        context = _make_request_stub(1)
        generation = _make_request_stub(2)
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [context]
        batch.generation_requests = [generation]

        with patch(
            "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs.torch.cuda.Event",
            return_value=Mock(),
        ):
            actual_outputs = engine.forward(
                ScheduledInputs(batch=batch, enable_spec_decode=True),
                resource_manager=resource_manager,
            )

        self.assertEqual(actual_outputs, {**outputs, "runtime_draft_len": 0})
        self.assertIsNot(actual_outputs, outputs)
        lookup = runner.maybe_get_cuda_graph.call_args
        self.assertIs(lookup.args[0], batch)
        self.assertEqual(lookup.kwargs["promoted_context_request_ids"], frozenset())
        prepare_args = engine._inputs.prepare_inputs.call_args.args
        self.assertIs(prepare_args[0], batch)
        self.assertIs(prepare_args[2], semantic_attn_metadata)
        self.assertIs(prepare_args[3], engine._metadata.spec_metadata)
        self.assertEqual(prepare_args[-1], frozenset())
        engine._forward_step.assert_called_once()
        runner.replay.assert_not_called()

    def test_multimodal_graph_miss_preserves_semantic_payload(self) -> None:
        engine, runner, resource_manager, _, _ = _make_decoder_runner(None)
        engine._model.config = SimpleNamespace(vocab_size=100)
        engine._model.mm_token_ids = torch.tensor([99], dtype=torch.int32)
        context = _make_request_stub(1, prompt_len=3)
        context.get_tokens = Mock(return_value=[99, 11, 22])
        multimodal_data = {
            "multimodal_embedding": object(),
            "mrope_config": {"mrope_position_deltas": object()},
        }
        context.py_multimodal_data = multimodal_data
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [context]

        with patch(
            "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs.torch.cuda.Event",
            return_value=Mock(),
        ):
            engine.forward(ScheduledInputs(batch=batch), resource_manager=resource_manager)

        graph_batch = runner.maybe_get_cuda_graph.call_args.args[0]
        self.assertIsNot(graph_batch, batch)
        self.assertEqual(graph_batch.generation_requests, [context])
        self.assertIs(engine._inputs.prepare_inputs.call_args.args[0], batch)
        self.assertIs(context.py_multimodal_data, multimodal_data)
        self.assertIn("multimodal_embedding", multimodal_data)

    def test_breakable_graph_falls_back_for_context_logits(self) -> None:
        for return_context_logits in (True, False):
            with self.subTest(return_context_logits=return_context_logits):
                engine, _, resource_manager, _, outputs = _make_decoder_runner(None)
                breakable_runner = Mock()
                breakable_runner.is_capturing = False
                breakable_runner.is_warming_up = False
                breakable_runner.has_graph.return_value = True
                breakable_runner.execute.return_value = outputs
                engine._graphs.breakable_cuda_graph_runner = breakable_runner
                prepared = engine._inputs.prepare_inputs.return_value

                def prepare_prefill_graph_batch(*args: object, **kwargs: object) -> PreparedInputs:
                    set_per_request_prefill_cuda_graph_flag(True)
                    return prepared

                engine._inputs.prepare_inputs.side_effect = prepare_prefill_graph_batch
                self.addCleanup(set_per_request_prefill_cuda_graph_flag, False)

                batch = ScheduledRequests()
                batch.context_requests_last_chunk = [_make_request_stub(1)]
                batch.context_requests_last_chunk[
                    0
                ].py_return_context_logits = return_context_logits

                with patch(
                    "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs.torch.cuda.Event",
                    return_value=Mock(),
                ):
                    actual_outputs = engine.forward(
                        ScheduledInputs(batch=batch, gather_context_logits=return_context_logits),
                        resource_manager=resource_manager,
                    )

                self.assertEqual(actual_outputs, {**outputs, "runtime_draft_len": 0})
                self.assertIsNot(actual_outputs, outputs)
                if return_context_logits:
                    breakable_runner.execute.assert_not_called()
                    engine._forward_step.assert_called_once()
                else:
                    breakable_runner.execute.assert_called_once()
                    engine._forward_step.assert_not_called()

    def test_generation_only_forward_does_not_call_new_selector(self) -> None:
        key = KeyType(batch_size=1, draft_len=0)
        engine, runner, resource_manager, _, _ = _make_decoder_runner(key)
        generation = _make_request_stub(2)
        batch = ScheduledRequests()
        batch.generation_requests = [generation]

        with patch(
            "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs.torch.cuda.Event",
            return_value=Mock(),
        ):
            engine.forward(ScheduledInputs(batch=batch), resource_manager=resource_manager)

        lookup = runner.maybe_get_cuda_graph.call_args
        self.assertIs(lookup.args[0], batch)
        self.assertEqual(lookup.kwargs["promoted_context_request_ids"], frozenset())
        self.assertFalse(lookup.kwargs["use_lora_graph"])
        self.assertIs(engine._inputs.prepare_inputs.call_args.args[0], batch)
        self.assertEqual(engine._inputs.prepare_inputs.call_args.args[-1], frozenset())

    def test_generation_lora_request_selects_lora_graph(self) -> None:
        key = KeyType(batch_size=1, draft_len=0, use_lora_graph=True)
        engine, runner, resource_manager, _, _ = _make_decoder_runner(
            key, cuda_graph_specialize_lora=True
        )
        engine._lora = engine._graphs._lora = SimpleNamespace(cuda_graph_manager=object())
        generation = _make_request_stub(2)
        generation.lora_task_id = 42
        batch = ScheduledRequests()
        batch.generation_requests = [generation]

        with patch(
            "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs.torch.cuda.Event",
            return_value=Mock(),
        ):
            engine.forward(ScheduledInputs(batch=batch), resource_manager=resource_manager)

        self.assertTrue(runner.maybe_get_cuda_graph.call_args.kwargs["use_lora_graph"])
        self.assertEqual(
            runner.maybe_get_cuda_graph.call_args.kwargs["peft_cache_data_type"], torch.bfloat16
        )
        self.assertTrue(engine._inputs.prepare_inputs.call_args.kwargs["use_lora_graph"])

    def test_global_incompatibilities_bypass_candidate_selection(self) -> None:
        cases = (
            "graphs_disabled",
            "speculative_nonzero_runtime_draft",
            "beam",
            "extra_inputs",
            "ple_recurrent_state",
            "nested_ple_recurrent_state",
            "context_parallel",
            "context_only_cache",
        )
        for case in cases:
            with self.subTest(case=case):
                engine, runner, resource_manager, _, _ = _make_decoder_runner(
                    None,
                    max_beam_width=2 if case == "beam" else 1,
                    spec_config=SimpleNamespace(
                        draft_len_schedule=None,
                        spec_dec_mode=_TWO_MODEL_SPEC_MODE,
                        is_linear_tree=True,
                        get_runtime_tokens_per_gen_step=Mock(return_value=1),
                    )
                    if case == "speculative_nonzero_runtime_draft"
                    else None,
                )
                spec_state = {}
                if case == "graphs_disabled":
                    runner.enabled = False
                elif case == "speculative_nonzero_runtime_draft":
                    spec_state = dict(enable_spec_decode=True, runtime_draft_len=1)
                elif case == "extra_inputs":
                    engine._graphs._context_promotion_supported = False
                elif case == "ple_recurrent_state":
                    engine._model.has_ple = True
                elif case == "nested_ple_recurrent_state":
                    engine._model.model = SimpleNamespace(
                        llm=SimpleNamespace(model=SimpleNamespace(has_ple=True))
                    )
                elif case == "context_parallel":
                    engine._mapping.cp_size = 2
                elif case == "context_only_cache":
                    resource_manager.get_resource_manager.side_effect = lambda _: SimpleNamespace(
                        context_swa_layer_limit=20
                    )

                batch = ScheduledRequests()
                batch.context_requests_last_chunk = [_make_request_stub(1)]
                with patch(
                    "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs.torch.cuda.Event",
                    return_value=Mock(),
                ):
                    engine.forward(
                        ScheduledInputs(batch=batch, **spec_state),
                        resource_manager=resource_manager,
                    )

                lookup = runner.maybe_get_cuda_graph.call_args
                self.assertIs(lookup.args[0], batch)
                self.assertEqual(lookup.kwargs["promoted_context_request_ids"], frozenset())
                self.assertIs(engine._inputs.prepare_inputs.call_args.args[0], batch)

    def test_forward_threads_call_state_and_returns_prepared_length(self) -> None:
        engine, runner, resource_manager, _, outputs = _make_decoder_runner(
            None,
            spec_config=SimpleNamespace(
                draft_len_schedule=None,
                spec_dec_mode=_TWO_MODEL_SPEC_MODE,
                is_linear_tree=False,
                tokens_per_gen_step=8,
                get_runtime_tokens_per_gen_step=Mock(return_value=1),
            ),
        )
        # A non-linear tree widens the length to its total draft tokens during
        # preparation, after graph selection.
        prepared = engine._inputs.prepare_inputs.return_value
        engine._inputs.prepare_inputs.return_value = replace(prepared, runtime_draft_len=7)
        batch = ScheduledRequests()
        batch.generation_requests = [_make_request_stub(2)]

        with patch(
            "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs.torch.cuda.Event",
            return_value=Mock(),
        ):
            actual_outputs = engine.forward(
                ScheduledInputs(batch=batch, enable_spec_decode=True, runtime_draft_len=2),
                resource_manager=resource_manager,
            )

        self.assertEqual(actual_outputs, {**outputs, "runtime_draft_len": 7})
        self.assertEqual(runner.pad_batch.call_args.args[2], 2)
        self.assertTrue(runner.maybe_get_cuda_graph.call_args.kwargs["enable_spec_decode"])
        prepare_kwargs = engine._inputs.prepare_inputs.call_args.kwargs
        self.assertEqual(
            (
                prepare_kwargs["enable_spec_decode"],
                prepare_kwargs["runtime_draft_len"],
                prepare_kwargs["is_dummy"],
            ),
            (True, 2, False),
        )
        step_kwargs = engine._forward_step.call_args.kwargs
        self.assertEqual(
            (
                step_kwargs["enable_spec_decode"],
                step_kwargs["runtime_draft_len"],
                step_kwargs["is_dummy"],
            ),
            (True, 7, False),
        )


@pytest.mark.cpu_only
class TestDecodeGraphMetadataPreparation(unittest.TestCase):
    @staticmethod
    def _metadata() -> CSA2TrtllmMetadata:
        from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

        metadata = object.__new__(CSA2TrtllmMetadata)
        metadata.is_cuda_graph = True
        return metadata

    def test_scope_requires_selected_ordinary_decode_graph(self) -> None:
        key = KeyType(batch_size=1, draft_len=0)
        cases = (
            "ordinary",
            "ordinary_overlap",
            "graph_miss",
            "eager_metadata",
            "context",
            "promoted_context",
            "speculative",
            "beam",
            "encoder_decoder",
            "context_parallel",
            "draft_key",
        )
        for case in cases:
            with self.subTest(case=case):
                engine, _, _, _, _ = _make_decoder_runner(
                    key,
                    disable_overlap_scheduler=case != "ordinary_overlap",
                    max_beam_width=2 if case == "beam" else 1,
                )
                enable_spec_decode = False
                metadata = self._metadata()
                batch = ScheduledRequests()
                batch.generation_requests = [_make_request_stub(1)]
                selected = key
                promoted = frozenset()
                if case == "graph_miss":
                    selected = None
                elif case == "eager_metadata":
                    metadata.is_cuda_graph = False
                elif case == "context":
                    batch.context_requests_last_chunk = [_make_request_stub(2)]
                elif case == "promoted_context":
                    promoted = frozenset({1})
                elif case == "speculative":
                    enable_spec_decode = True
                elif case == "encoder_decoder":
                    engine._deferred_decode_outputs_supported = False
                elif case == "context_parallel":
                    engine._mapping.cp_size = 2
                elif case == "draft_key":
                    selected = key._replace(draft_len=1)
                with engine._cuda_graph_metadata_prepare_scope(
                    metadata, selected, batch, promoted, enable_spec_decode=enable_spec_decode
                ):
                    self.assertEqual(
                        metadata._csa2_defer_decode_outputs,
                        case in ("ordinary", "ordinary_overlap"),
                    )
                self.assertFalse(metadata._csa2_defer_decode_outputs)

    def test_forward_scopes_proof_to_prepare_for_capture_replay_and_fallback(self) -> None:
        key = KeyType(batch_size=1, draft_len=0)
        for mode in ("warmup", "capture", "replay", "fallback", "prepare_error"):
            with self.subTest(mode=mode):
                selected = None if mode == "fallback" else key
                engine, runner, resource_manager, _, outputs = _make_decoder_runner(selected)
                metadata = self._metadata()
                engine.attn_metadata = metadata
                engine._metadata.setup_attention.return_value = metadata
                runner.maybe_get_cuda_graph.return_value = (metadata, None, selected)
                runner.needs_capture.return_value = mode in ("warmup", "capture")
                runner.is_warmup_only = mode == "warmup"
                runner.capture.return_value = outputs
                batch = ScheduledRequests()
                batch.generation_requests = [_make_request_stub(1)]
                prepared = engine._inputs.prepare_inputs.return_value

                def prepare(*args: object, **kwargs: object) -> PreparedInputs:
                    self.assertIs(metadata.trtllm_gen_jit_warmup, False)
                    self.assertEqual(metadata._csa2_defer_decode_outputs, mode != "fallback")
                    if mode == "prepare_error":
                        raise RuntimeError("prepare failed")
                    return prepared

                engine._inputs.prepare_inputs.side_effect = prepare
                with patch(
                    "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs.torch.cuda.Event",
                    return_value=Mock(),
                ):
                    if mode == "prepare_error":
                        with self.assertRaisesRegex(RuntimeError, "prepare failed"):
                            engine.forward(
                                ScheduledInputs(batch=batch), resource_manager=resource_manager
                            )
                    else:
                        actual_outputs = engine.forward(
                            ScheduledInputs(batch=batch), resource_manager=resource_manager
                        )
                        self.assertEqual(actual_outputs, {**outputs, "runtime_draft_len": 0})
                        self.assertIs(actual_outputs["logits"], outputs["logits"])
                self.assertFalse(metadata._csa2_defer_decode_outputs)
                if mode in ("warmup", "capture"):
                    runner.capture.assert_called_once()
                if mode in ("capture", "replay"):
                    runner.replay.assert_called_once()
                if mode == "fallback":
                    engine._forward_step.assert_called_once()

    def test_jit_warmup_flag_marks_only_the_warmup_forward(self) -> None:
        engine, _, resource_manager, _, _ = _make_decoder_runner(None)
        runner = engine
        metadata = self._metadata()
        runner.attn_metadata = metadata
        runner._metadata.setup_attention.return_value = metadata
        prepared = runner._inputs.prepare_inputs.return_value
        observed = []

        def prepare(*args: object, **kwargs: object) -> PreparedInputs:
            observed.append(metadata.trtllm_gen_jit_warmup)
            return prepared

        runner._inputs.prepare_inputs.side_effect = prepare
        batch = ScheduledRequests()
        batch.generation_requests = [_make_request_stub(1)]
        for jit_warmup in (True, False):
            runner._forward_warmup(
                batch,
                resource_manager,
                enable_spec_decode=False,
                runtime_draft_len=0,
                trtllm_gen_jit_warmup=jit_warmup,
            )
        self.assertEqual(observed, [True, False])
        self.assertFalse(runner._trtllm_gen_jit_warmup)


@pytest.mark.cpu_only
@pytest.mark.parametrize("local_contexts", [0, 1])
def test_prefill_compile_uses_all_rank_prefill_decision(
    monkeypatch: pytest.MonkeyPatch,
    local_contexts: int,
) -> None:
    """Route using the global prefill decision, regardless of local contexts."""
    eager = torch.nn.Linear(4, 4)
    compiled = torch.nn.Module()
    compiled.shared = eager
    expected = torch.ones((2, 4))
    eager.forward = Mock(return_value=expected)
    compiled.forward = Mock(return_value=expected)
    router = PrefillCompiledModel(eager, compiled)
    assert router.weight is eager.weight
    assert list(router.parameters()) == list(eager.parameters())
    # Local decode-only ranks participate when another attention-DP rank
    # prefills; an over-ceiling local context batch uses eager instead.
    metadata = SimpleNamespace(num_contexts=local_contexts)
    for eligible in (True, False):
        monkeypatch.setattr(
            decoder_runner_module, "get_per_request_prefill_cuda_graph_flag", lambda: eligible
        )
        assert router(expected, attn_metadata=metadata) is expected
        selected = compiled if eligible else eager
        selected.forward.assert_called_once_with(expected, attn_metadata=metadata)
    assert compiled.forward.call_count == eager.forward.call_count == 1


@pytest.mark.cpu_only
def test_prefill_compile_preserves_partial_weight_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reload selected weights by original names into both model entry points."""
    import tensorrt_llm._torch.models.modeling_utils as modeling_utils

    eager = torch.nn.Sequential(OrderedDict(layers=torch.nn.Sequential(torch.nn.Linear(4, 4))))
    compiled = torch.compile(eager, backend="eager")
    model = DecoderModelForCausalLM.__new__(DecoderModelForCausalLM)
    torch.nn.Module.__init__(model)
    model.model_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(tie_word_embeddings=False)
    )
    model.model = PrefillCompiledModel(eager, compiled)
    mapper = HfWeightMapper()
    mapper._model = model
    loader = ModelLoader.__new__(ModelLoader)
    loader.weight_mapper = mapper
    monkeypatch.setenv("TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL", "True")
    monkeypatch.setattr(modeling_utils, "local_mpi_rank", lambda: 0)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    weight = eager.layers[0].weight
    old_bias = eager.layers[0].bias.detach().clone()
    replacement = torch.full_like(weight, 3)
    for remove_duplicate in (False, True):
        names = dict(model.named_modules(remove_duplicate=remove_duplicate))
        assert names["model.layers.0"] is eager.layers[0]
        assert not any("eager_model" in name or "compiled_model" in name for name in names)
    assert list(model.model.children()) == [eager]
    assert list(model.model.state_dict()) == [
        "eager_model.layers.0.weight",
        "eager_model.layers.0.bias",
    ]
    apply_tensor = Mock(side_effect=lambda tensor: tensor)
    model.model._apply(apply_tensor)
    assert apply_tensor.call_count == 2
    loader.reload(model, {"model.layers.0.weight": replacement}, allow_partial_loading=True)

    assert eager.layers[0].weight is weight
    assert compiled.layers[0].weight is weight
    torch.testing.assert_close(weight, replacement)
    torch.testing.assert_close(eager.layers[0].bias, old_bias)
    inputs = torch.ones(2, 4)
    for eligible in (False, True):
        monkeypatch.setattr(
            decoder_runner_module, "get_per_request_prefill_cuda_graph_flag", lambda: eligible
        )
        torch.testing.assert_close(model.model(inputs), inputs @ replacement.t() + old_bias)


@pytest.mark.cpu_only
@pytest.mark.parametrize("prefill_only", [False, True])
@pytest.mark.parametrize("eligible", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_prefill_compile_scopes_whole_model_forward(
    monkeypatch: pytest.MonkeyPatch,
    prefill_only: bool,
    eligible: bool,
    bypass: bool,
    raises: bool,
) -> None:
    """Scope compile dispatch by phase independently of the model wrapper."""
    observed = []

    def forward(**kwargs: object) -> str:
        """Record compile state as a stand-in for a model-specific epilogue."""
        observed.append(is_torch_compiling())
        # This represents work after the transformer, such as Eagle3 drafting.
        if raises:
            raise RuntimeError("epilogue failure")
        observed.append(is_torch_compiling())
        return "done"

    eager_model = torch.nn.Identity()
    eager_model.model_config = SimpleNamespace(extra_attrs={})
    eager_model.forward = forward
    compiled_model = torch.nn.Identity()
    compiled_model.forward = forward
    prefill_compiled_model = PrefillCompiledModel(eager_model, compiled_model)
    model = prefill_compiled_model if prefill_only else eager_model
    engine = SimpleNamespace(
        _model_caller=ModelCaller(model, prefill_compile_only=prefill_only),
        _eager_workspace_reclaimer=None,
        _moe_workspace_reclaimer=None,
    )
    monkeypatch.setattr(model_call_module, "get_model_extra_attrs", lambda: {})
    monkeypatch.setattr(
        model_call_module, "get_per_request_prefill_cuda_graph_flag", lambda: eligible
    )
    monkeypatch.setattr(model_call_module, "is_trace_enabled", lambda name: False)
    bypass_scope = (
        prefill_compiled_model.bypass() if prefill_only and bypass else contextlib.nullcontext()
    )
    with torch_compiling(True), bypass_scope:
        if raises:
            with pytest.raises(RuntimeError, match="epilogue failure"):
                DecoderRunner._model_forward(engine, is_dummy=False, attn_metadata=Mock())
        else:
            assert (
                DecoderRunner._model_forward(engine, is_dummy=False, attn_metadata=Mock()) == "done"
            )
        assert is_torch_compiling()
    expected = eligible if prefill_only else True
    assert observed == [expected] * (1 if raises else 2)


@pytest.mark.cpu_only
@pytest.mark.parametrize("is_dummy", [False, True])
def test_decoder_runner_passes_call_state_without_engine_writes(is_dummy: bool) -> None:
    engine, _, resources, _, _ = _make_decoder_runner(None)
    # The dummy flag comes from the caller, not from the warmup scope.
    inputs = ScheduledInputs(
        batch=ScheduledRequests(), enable_spec_decode=True, runtime_draft_len=2
    )
    expected = {"logits": object(), "runtime_draft_len": 4}
    engine._forward_decoder = Mock(return_value=expected)

    outputs = engine.forward(inputs, resource_manager=resources, is_dummy=is_dummy)

    assert outputs is expected
    engine._forward_decoder.assert_called_once_with(inputs, resources, is_dummy=is_dummy)
    assert (inputs.enable_spec_decode, inputs.runtime_draft_len) == (True, 2)


@pytest.mark.cpu_only
def test_forward_warmup_runs_local_call_state_in_forward_context() -> None:
    engine, _, resources, _, _ = _make_decoder_runner(None)
    batch = ScheduledRequests()

    def decoder_forward(
        inputs: ScheduledInputs, resource_manager: ResourceManager, *, is_dummy: bool
    ) -> dict[str, object]:
        assert torch.is_inference_mode_enabled()
        assert get_model_extra_attrs() is engine._model.extra_attrs
        return {"logits": object(), "runtime_draft_len": 3}

    engine._forward_decoder = Mock(side_effect=decoder_forward)

    outputs = engine._forward_warmup(batch, resources, enable_spec_decode=True, runtime_draft_len=1)

    inputs = engine._forward_decoder.call_args.args[0]
    assert inputs.batch is batch
    assert inputs.new_tensors_device is None
    assert inputs.cache_indirection_buffer is None
    assert (inputs.enable_spec_decode, inputs.runtime_draft_len) == (True, 1)
    assert engine._forward_decoder.call_args.kwargs == {"is_dummy": True}
    assert outputs["runtime_draft_len"] == 3
    assert get_model_extra_attrs() is None


@pytest.mark.cpu_only
def test_decoder_runner_forward_enters_model_context() -> None:
    engine, _, resources, _, _ = _make_decoder_runner(None)
    inputs = ScheduledInputs(batch=ScheduledRequests())

    def decoder_forward(
        inputs: ScheduledInputs, resource_manager: ResourceManager, *, is_dummy: bool
    ) -> dict[str, object]:
        assert torch.is_inference_mode_enabled()
        assert get_model_extra_attrs() is engine._model.extra_attrs
        return {"logits": object()}

    engine._forward_decoder = Mock(side_effect=decoder_forward)

    engine.forward(inputs, resource_manager=resources)

    engine._forward_decoder.assert_called_once_with(inputs, resources, is_dummy=False)
    assert get_model_extra_attrs() is None
