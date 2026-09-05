# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the LoRA parameter builder.

These cover the eager construction path, which has no test coverage upstream:
the union-set padding across a mixed batch, and the speculative-decoding
re-labelling. The latter is the reason the builder takes ``enable_spec_decode``
and ``runtime_draft_len`` per call instead of snapshotting them -- warmup and
CUDA-graph capture rewrite both repeatedly, and a snapshot would produce wrong
token counts silently.
"""

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.engine.lora import LoraParamBuilder
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

pytestmark = pytest.mark.cpu_only

# kCONTEXT / kGENERATION as the LoRA op's C++ kernel sees them.
K_CONTEXT = 0
K_GENERATION = 1


def _request(request_id: int, lora_task_id: int | None) -> SimpleNamespace:
    return SimpleNamespace(py_request_id=request_id, lora_task_id=lora_task_id)


def _module(
    *,
    layer_id: int,
    module_id: int,
    adapter_size: int,
    weights_in: int,
    weights_out: int,
    scaling_vec_pointer: int | None = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        layer_id=layer_id,
        module_id=module_id,
        adapter_size=adapter_size,
        weights_in_pointer=weights_in,
        weights_out_pointer=weights_out,
        scaling_vec_pointer=scaling_vec_pointer,
    )


def _batch(
    context: Sequence[SimpleNamespace] = (), generation: Sequence[SimpleNamespace] = ()
) -> ScheduledRequests:
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = list(context)
    batch.generation_requests = list(generation)
    return batch


def _attn_metadata(
    *, num_contexts: int = 1, num_generations: int = 0, prompt_lens: list[int] | None = None
) -> SimpleNamespace:
    num_seqs = num_contexts + num_generations
    request_types = [K_CONTEXT] * num_contexts + [K_GENERATION] * num_generations
    if prompt_lens is None:
        prompt_lens = [5] * num_contexts + [1] * num_generations
    return SimpleNamespace(
        host_request_types=torch.tensor(request_types, dtype=torch.int32),
        prompt_lens_cpu=torch.tensor(prompt_lens, dtype=torch.int32),
        num_seqs=num_seqs,
        num_contexts=num_contexts,
        num_generations=num_generations,
    )


def _peft_cache_manager(
    peft_table: dict[int, list[SimpleNamespace]] | None, data_type: torch.dtype = torch.float16
) -> SimpleNamespace:
    return SimpleNamespace(
        get_and_reset_batch_peft_table=lambda: peft_table,
        data_type=data_type,
    )


def _builder(*, is_linear_tree: bool = True, extend_ctx: bool = False) -> LoraParamBuilder:
    spec_config = SimpleNamespace(
        is_linear_tree=is_linear_tree,
        spec_dec_mode=SimpleNamespace(extend_ctx=lambda backend: extend_ctx),
    )
    return LoraParamBuilder(spec_config=spec_config, attn_backend=object())


def _build(
    builder: LoraParamBuilder,
    batch: ScheduledRequests,
    attn_metadata: SimpleNamespace,
    peft_table: dict[int, list[SimpleNamespace]] | None,
    **kwargs: Any,
) -> dict | None:
    """Drive the eager path: no CUDA-graph manager, so ``maybe_graph`` is moot."""
    kwargs.setdefault("enable_spec_decode", False)
    kwargs.setdefault("runtime_draft_len", 0)
    return builder.build(
        batch,
        attn_metadata,
        cuda_graph_lora_manager=None,
        peft_cache_manager=_peft_cache_manager(peft_table),
        **kwargs,
    )


def test_single_request_produces_tensors_of_the_declared_dtypes() -> None:
    batch = _batch(context=[_request(0, lora_task_id=7)])
    peft_table = {
        7: [_module(layer_id=3, module_id=1, adapter_size=4, weights_in=11, weights_out=22)]
    }

    params = _build(_builder(), batch, _attn_metadata(), peft_table)

    entry = params[3][1]
    assert entry["adapter_size"].dtype == torch.int32
    assert entry["weight_pointers"].dtype == torch.int64
    assert entry["adapter_size"].tolist() == [4]
    assert entry["weight_pointers"].tolist() == [11, 22, 0]
    assert params["data_type"] is torch.float16


def test_request_without_an_adapter_is_padded_in_batch_order() -> None:
    """The zero rows must land at the non-LoRA request's position, not at the end."""
    batch = _batch(
        context=[_request(0, lora_task_id=7)], generation=[_request(1, lora_task_id=None)]
    )
    peft_table = {
        7: [_module(layer_id=0, module_id=1, adapter_size=4, weights_in=11, weights_out=22)]
    }

    params = _build(_builder(), batch, _attn_metadata(num_generations=1), peft_table)

    entry = params[0][1]
    assert entry["adapter_size"].tolist() == [4, 0]
    assert entry["weight_pointers"].tolist() == [11, 22, 0, 0, 0, 0]


def test_requests_touching_different_modules_take_the_union() -> None:
    """Each request must get a row in every module the batch touches."""
    batch = _batch(context=[_request(0, lora_task_id=7), _request(1, lora_task_id=8)])
    peft_table = {
        7: [_module(layer_id=0, module_id=1, adapter_size=4, weights_in=11, weights_out=22)],
        8: [_module(layer_id=0, module_id=2, adapter_size=6, weights_in=33, weights_out=44)],
    }

    params = _build(_builder(), batch, _attn_metadata(num_contexts=2), peft_table)

    assert sorted(params[0]) == [1, 2]
    assert params[0][1]["adapter_size"].tolist() == [4, 0]
    assert params[0][1]["weight_pointers"].tolist() == [11, 22, 0, 0, 0, 0]
    assert params[0][2]["adapter_size"].tolist() == [0, 6]
    assert params[0][2]["weight_pointers"].tolist() == [0, 0, 0, 33, 44, 0]


def test_absent_scaling_vector_becomes_a_null_pointer() -> None:
    batch = _batch(context=[_request(0, lora_task_id=7)])
    peft_table = {
        7: [
            _module(
                layer_id=0,
                module_id=1,
                adapter_size=4,
                weights_in=11,
                weights_out=22,
                scaling_vec_pointer=None,
            )
        ]
    }

    params = _build(_builder(), batch, _attn_metadata(), peft_table)

    assert params[0][1]["weight_pointers"].tolist() == [11, 22, 0]


def test_spec_decode_relabels_generation_requests_as_context() -> None:
    """One generation request covers runtime_draft_len + 1 tokens, so it is fed as context."""
    batch = _batch(context=[_request(0, lora_task_id=7)], generation=[_request(1, lora_task_id=7)])
    peft_table = {
        7: [_module(layer_id=0, module_id=1, adapter_size=4, weights_in=11, weights_out=22)]
    }
    attn_metadata = _attn_metadata(num_generations=1)

    params = _build(
        _builder(),
        batch,
        attn_metadata,
        peft_table,
        enable_spec_decode=True,
        runtime_draft_len=3,
    )

    assert params["host_request_types"].tolist() == [K_CONTEXT, K_CONTEXT]
    assert params["prompt_lens_cpu"].tolist() == [5, 4]
    assert params["num_seqs"] == 2
    # The caller's metadata is shared with the rest of input preparation.
    assert attn_metadata.host_request_types.tolist() == [K_CONTEXT, K_GENERATION]
    assert attn_metadata.prompt_lens_cpu.tolist() == [5, 1]


def test_spec_decode_state_is_read_per_call_not_snapshotted() -> None:
    """The same builder must relabel or not purely on this call's arguments.

    Warmup and CUDA-graph capture rewrite ``enable_spec_decode`` and
    ``runtime_draft_len`` between forwards; a builder that captured them at
    construction would keep using stale token counts without erroring.
    """
    builder = _builder()
    batch = _batch(context=[_request(0, lora_task_id=7)], generation=[_request(1, lora_task_id=7)])
    peft_table = {
        7: [_module(layer_id=0, module_id=1, adapter_size=4, weights_in=11, weights_out=22)]
    }

    spec_on = _build(
        builder,
        batch,
        _attn_metadata(num_generations=1),
        peft_table,
        enable_spec_decode=True,
        runtime_draft_len=3,
    )
    spec_off = _build(
        builder,
        batch,
        _attn_metadata(num_generations=1),
        peft_table,
        enable_spec_decode=False,
        runtime_draft_len=3,
    )

    assert spec_on["host_request_types"].tolist() == [K_CONTEXT, K_CONTEXT]
    assert spec_off["host_request_types"].tolist() == [K_CONTEXT, K_GENERATION]
    assert spec_off["prompt_lens_cpu"].tolist() == [5, 1]


@pytest.mark.parametrize(
    "builder_kwargs",
    [{"is_linear_tree": False}, {"extend_ctx": True}],
    ids=["not-linear-tree", "extend-ctx"],
)
def test_spec_decode_relabelling_needs_the_whole_predicate(builder_kwargs: dict[str, bool]) -> None:
    """Either construction-time condition alone suppresses the re-labelling."""
    batch = _batch(context=[_request(0, lora_task_id=7)], generation=[_request(1, lora_task_id=7)])
    peft_table = {
        7: [_module(layer_id=0, module_id=1, adapter_size=4, weights_in=11, weights_out=22)]
    }

    params = _build(
        _builder(**builder_kwargs),
        batch,
        _attn_metadata(num_generations=1),
        peft_table,
        enable_spec_decode=True,
        runtime_draft_len=3,
    )

    assert params["host_request_types"].tolist() == [K_CONTEXT, K_GENERATION]


def test_missing_peft_table_yields_none_and_an_empty_one_yields_an_empty_dict() -> None:
    """``None`` and ``{}`` are distinct results; callers pass both to the model."""
    batch = _batch(context=[_request(0, lora_task_id=7)])
    builder = _builder()

    no_manager = builder.build(
        batch,
        _attn_metadata(),
        cuda_graph_lora_manager=None,
        enable_spec_decode=False,
        runtime_draft_len=0,
        peft_cache_manager=None,
    )
    empty_table = _build(builder, batch, _attn_metadata(), {})

    assert no_manager is None
    assert empty_table == {}


def test_batch_without_any_adapter_gets_no_metadata_keys() -> None:
    """No LoRA request means no parameters at all -- not zero-filled ones."""
    batch = _batch(context=[_request(0, lora_task_id=None)])
    peft_table = {
        7: [_module(layer_id=0, module_id=1, adapter_size=4, weights_in=11, weights_out=22)]
    }

    params = _build(_builder(), batch, _attn_metadata(), peft_table)

    assert params == {}


class _FakeCudaGraphLoraManager:
    """Records the calls `build` makes on a CUDA-graph LoRA manager."""

    def __init__(self, graph_params: dict | None = None) -> None:
        self._graph_params = graph_params
        self.base_only_calls: list[object] = []
        self.tokens_per_seq_calls: list[int] = []
        self.evicted_for: list[object] = []
        self.adapter_slot_manager = SimpleNamespace(
            remove_evicted_slots_in_cpp=self.evicted_for.append
        )

    def prepare_base_only_batch(self, peft_cache_manager: object) -> None:
        self.base_only_calls.append(peft_cache_manager)

    def prepare_cuda_graph_lora_params(
        self,
        scheduled_requests: ScheduledRequests,
        attn_metadata: SimpleNamespace,
        peft_cache_manager: object,
        tokens_per_seq: int,
    ) -> dict | None:
        self.tokens_per_seq_calls.append(tokens_per_seq)
        return self._graph_params


def test_base_only_batch_short_circuits_before_any_parameter_build() -> None:
    """A graph batch with no LoRA request reserves base-only slots and returns nothing."""
    manager = _FakeCudaGraphLoraManager()
    peft_cache_manager = _peft_cache_manager({})

    params = _builder().build(
        _batch(context=[_request(0, lora_task_id=None)]),
        _attn_metadata(),
        cuda_graph_lora_manager=manager,
        enable_spec_decode=False,
        runtime_draft_len=0,
        peft_cache_manager=peft_cache_manager,
        maybe_graph=True,
        use_lora_graph=False,
    )

    assert params is None
    assert manager.base_only_calls == [peft_cache_manager]
    assert manager.tokens_per_seq_calls == []


@pytest.mark.parametrize(
    ("enable_spec_decode", "runtime_draft_len", "expected_tokens_per_seq"),
    [(False, 3, 1), (True, 0, 1), (True, 3, 4)],
    ids=["spec-decode-off", "no-draft-tokens", "spec-decode-on"],
)
def test_cuda_graph_path_sizes_tokens_per_seq(
    enable_spec_decode: bool, runtime_draft_len: int, expected_tokens_per_seq: int
) -> None:
    """Only a spec-decode batch with draft tokens covers more than one token per sequence."""
    graph_params = {"source": "cuda-graph"}
    manager = _FakeCudaGraphLoraManager(graph_params=graph_params)

    params = _builder().build(
        _batch(generation=[_request(0, lora_task_id=7)]),
        _attn_metadata(num_contexts=0, num_generations=1),
        cuda_graph_lora_manager=manager,
        enable_spec_decode=enable_spec_decode,
        runtime_draft_len=runtime_draft_len,
        peft_cache_manager=_peft_cache_manager({}),
        maybe_graph=True,
        use_lora_graph=True,
    )

    assert params is graph_params
    assert manager.tokens_per_seq_calls == [expected_tokens_per_seq]


def test_eager_fallback_releases_evicted_adapter_slots() -> None:
    """Falling back to eager with a live manager must still drop slots the cache evicted."""
    manager = _FakeCudaGraphLoraManager()
    peft_cache_manager = _peft_cache_manager({})

    _builder().build(
        _batch(context=[_request(0, lora_task_id=None)]),
        _attn_metadata(),
        cuda_graph_lora_manager=manager,
        enable_spec_decode=False,
        runtime_draft_len=0,
        peft_cache_manager=peft_cache_manager,
        maybe_graph=False,
    )

    assert manager.evicted_for == [peft_cache_manager]
    assert manager.base_only_calls == []
    assert manager.tokens_per_seq_calls == []
