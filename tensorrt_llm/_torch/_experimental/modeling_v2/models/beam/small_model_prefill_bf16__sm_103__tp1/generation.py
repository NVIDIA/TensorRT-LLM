# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stateful greedy generation for the Beam six-block model / sm_103 / tp1.

A thin serial controller that drives the stateless six-block model
(``BeamSmallModelPrefillBf16Sm103Tp1``) through TensorRT-LLM's own request, cache and
sampler objects: one ``LlmRequest`` per request, one long-lived ``KVCacheManagerV2`` with
block reuse for all of them, a ``SeqSlotManager`` for sampler slots and the existing
``TorchSampler``. It owns no cache allocator, prefix table or sampler of its own; it calls
those objects in the order the non-overlap executor does -- forward, ``sample_async``,
context-cursor advance, ``update_requests``, ``update_context_resources``, finish
handling, ``update_resources``. It is a component, not a serving engine: one request
executes at a time, at most ``GENERATOR_CACHE["max_batch_size"]`` (two) request states
are live, and there is no scheduler, batching, streaming or LLM API registration.

Lifecycle::

    with BeamSmallModelGenerator(model) as gen:  # binds the loaded model
        state = gen.create_request(prompt_ids, max_new_tokens=8, end_id=None)
        first = gen.prefill(state)  # prompt tokens not already cached; first token
        while state.lifecycle is not Lifecycle.FINISHED:
            step = gen.decode(state)  # consumes the pending sampled token
        tokens, reason = state.output_token_ids, state.finish_reason
        gen.release(state)

``generate(state)`` runs the prefill and decode steps of a new sampling request to its
end. A compute-only request (``sample=False``) runs the same steps without the sampler:
``prefill`` returns the last prompt token's logits and ``decode(state, token_id)``
consumes an explicit (teacher-forced) token; its token budget is ``max_new_tokens``
explicit tokens.

Semantics.

* Positions are absolute. A prefill computes prompt tokens ``reused .. P - 1`` at those
  positions after ``reused`` cached ones; a decode consumes one token at position
  ``consumed`` (the number of tokens whose keys and values the cache holds) and leaves
  ``consumed + 1``. Every model call is checked by the model against the manager before
  it runs (``model.check_request``).
* Prefix reuse is the manager's. A prefill first asks the manager for the request's
  cache (``prepare_context``): the manager matches the prompt against what earlier
  requests committed and claims the longest reusable full-page prefix -- at most
  ``P - 1`` tokens, since the last prompt token is always recomputed for its logits;
  with partial reuse off, only whole 32-token pages. The prefill computes the rest and the
  manager commits it (``update_context_resources``), so later requests can reuse it,
  also after this request is released. ``reuse=False`` claims nothing
  (``prepare_context_cache(reuse_limit=0)``) and computes the whole prompt: the
  reuse-disabled control. Generated tokens are not committed for reuse.
* Sampling is greedy: ``temperature=0``, ``top_k=1``, beam width 1, no penalties, stop
  words, logits processors or draft tokens. ``TorchSampler`` chooses each token from the
  step's FP32 logits (the lowest vocabulary index among exact maxima).
* Stopping follows the sampler: ``FinishReason.END_ID`` when the sampled token equals
  ``end_id``, ``FinishReason.LENGTH`` after ``max_new_tokens`` sampled tokens; END_ID wins
  when both hold. The last sampled token is returned and never consumed: after P prompt
  tokens and G sampled tokens the cache holds P + G - 1 tokens, and no model call follows
  the finishing step. An explicit token never counts as sampled.
* Isolation: each request owns its ``LlmRequest``, cache, sampler slot and ledger; the
  model holds no request state. Immutable committed prefix pages may be shared between
  requests at the same layer; every page a request writes is its own.
* Identity: a generator binds one model and its current weights -- each of the model's
  checkpoint tensors as the very tensor object it is, with its version counter, storage
  address, shape and strides. While it is open the model refuses ``load_weights``, and
  every step first verifies that the model's checkpoint tensors are still exactly those
  bound ones, unmodified: a tensor changed in place, replaced by another tensor (for
  example ``load_state_dict(..., assign=True)`` or attribute assignment) or given other
  storage is refused before the step looks up a prefix, allocates or runs a kernel, for as
  long as the change persists. ``close`` releases every request and shuts the manager
  down, after which a new generator starts from an empty cache and binds the weights the
  model then holds.

Limits: requests of 1 .. ``max_seq_len - max_new_tokens`` prompt tokens, ``P +
max_new_tokens <= 1024``; at most two live states; one GB300 (sm_103), eager execution only
(no CUDA graph, no overlap scheduler): every step synchronizes the host several times.
Requests are validated before any kernel runs; a request whose step failed after it began
is FAILED and can only be released.
"""

from __future__ import annotations

import enum
import itertools
import weakref
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp
from tensorrt_llm._torch.pyexecutor.sampler import TorchSampler
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.seq_slot_manager import SeqSlotManager
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.executor import FinishReason
from tensorrt_llm.bindings.executor import SamplingConfig as ExecutorSamplingConfig
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping as ParallelMapping

from .model import ATTENTION_WINDOWS, MAX_TOKENS, NUM_LAYERS, BeamSmallModelPrefillBf16Sm103Tp1

#: The generator's cache: bf16, 32-token pages, a 1024-token sequence capacity, two live
#: requests, block reuse of whole pages (partial reuse off), no host tier and no
#: sliding-window scratch pages. ``max_tokens`` sizes the device quota so that two live
#: requests at full capacity fit beside the reusable pages of released ones.
GENERATOR_CACHE = {
    "tokens_per_block": 32,
    "max_seq_len": 1024,
    "max_batch_size": 2,
    "max_tokens": 4096,
}

#: The sampler: greedy, beam width 1, no draft tokens, no overlap scheduler or async worker.
SAMPLER_ARGS = TorchSampler.Args(
    max_seq_len=GENERATOR_CACHE["max_seq_len"],
    max_draft_len=0,
    max_num_sequences=GENERATOR_CACHE["max_batch_size"],
    max_beam_width=1,
    max_total_draft_tokens=0,
    disable_overlap_scheduler=True,
    enable_async_worker=False,
)


def build_cache_manager(vocab_size: int = 151936) -> KVCacheManagerV2:
    """The generator's six-layer ``KVCacheManagerV2`` (``GENERATOR_CACHE``), block reuse on.

    Logical layers 0-5 in slots 0-5, 8 bf16 kv heads of 128, per-layer windows
    ``ATTENTION_WINDOWS`` (V2 keeps the five sliding layers and layer 5 in separate pools),
    whole-page reuse under the ``all_reusable`` policy, no host tier, no scratch pages,
    no overlap scheduler. The caller shuts it down.
    """
    cache = GENERATOR_CACHE
    return KVCacheManagerV2(
        KvCacheConfig(
            max_tokens=cache["max_tokens"],
            max_attention_window=list(ATTENTION_WINDOWS),
            enable_block_reuse=True,
            enable_partial_reuse=False,
            enable_swa_scratch_reuse=False,
            host_cache_size=0,
        ),
        CacheTypeCpp.SELF,
        num_layers=NUM_LAYERS,
        num_kv_heads=8,
        head_dim=128,
        tokens_per_block=cache["tokens_per_block"],
        max_seq_len=cache["max_seq_len"],
        max_batch_size=cache["max_batch_size"],
        mapping=ParallelMapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.BF16,
        vocab_size=vocab_size,
        disable_overlap_scheduler=True,
    )


def _tensor_state(tensor: torch.Tensor) -> tuple:
    """What changes when a tensor's contents or storage change: version, address, shape, strides."""
    return tensor._version, tensor.data_ptr(), tuple(tensor.shape), tensor.stride()


def greedy_sampling_config() -> ExecutorSamplingConfig:
    """The request sampling config: temperature 0 and top_k 1 at beam width 1, nothing else set.

    ``TorchSampler`` resolves it to ``SamplingMode.GREEDY``.
    """
    return ExecutorSamplingConfig(beam_width=1, top_k=1, temperature=0.0)


def sample_step(
    sampler: TorchSampler, request: LlmRequest, logits: torch.Tensor, *, is_context: bool
) -> None:
    """Choose ``request``'s next token from ``logits`` (``[vocab]`` FP32) with ``sampler``.

    Runs ``sample_async`` on a batch holding only ``request`` -- as its last context chunk
    or as its one generation row -- then advances a context request's cursor to the end of
    its prompt as the executor does, then ``update_requests``, which appends the chosen
    token to ``request`` and finishes it when the sampler's stopping rules hold. The
    request must already hold a sampler slot.
    """
    batch = ScheduledRequests()
    if is_context:
        batch.context_requests_last_chunk = [request]
        prefix_sum = [0, 1]
    else:
        batch.generation_requests = [request]
        prefix_sum = [0]
    sample_state = sampler.sample_async(batch, {"logits": logits.view(1, -1)}, prefix_sum)
    if is_context:
        request.py_last_context_chunk = (
            request.context_current_position,
            request.context_current_position + request.context_chunk_size,
        )
        request.move_to_next_context_chunk()
        if request.context_remaining_length == 0:
            request.state = LlmRequestState.GENERATION_IN_PROGRESS
    sampler.update_requests(sample_state)


class _Request(LlmRequest):
    """An ``LlmRequest`` that keeps the finish reason the sampler records on it."""

    py_finish_reason: Optional[FinishReason] = None

    def finish_by(self, reason: FinishReason, beam: int) -> None:
        super().finish_by(reason, beam)
        self.py_finish_reason = reason


class Lifecycle(enum.Enum):
    NEW = "new"
    PREFILLED = "prefilled"
    DECODING = "decoding"
    FINISHED = "finished"
    RELEASED = "released"
    FAILED = "failed"


@dataclass(frozen=True)
class StepOutput:
    """One model step of one request.

    ``logits`` are the FP32 ``[151936]`` softcapped logits of the step's last token, at
    absolute ``position``. ``token_id`` is the token the sampler chose from them (None for
    a compute-only request) and ``finish_reason`` is set on the step that finished it.
    """

    logits: torch.Tensor
    position: int
    num_computed: int
    token_id: Optional[int]
    finish_reason: Optional[FinishReason]


@dataclass(frozen=True)
class GenerationOutput:
    token_ids: tuple
    finish_reason: FinishReason
    steps: tuple


class RequestState:
    """A handle to one request of one generator; read-only to callers.

    ``consumed``: tokens whose keys and values the cache holds. ``reused``: prompt tokens
    the manager claimed from earlier requests. ``computed_prompt_tokens``: prompt tokens
    the prefill computed. ``output_token_ids``: sampled tokens, in order.
    ``ledger``: one record per step (positions, token source, manager claim, capacity,
    history, committed tokens and page indices after the step).
    """

    def __init__(
        self,
        generator,
        request: LlmRequest,
        prompt: tuple,
        *,
        sample: bool,
        reuse: bool,
        max_new_tokens: int,
        end_id: Optional[int],
    ):
        self._generator = weakref.ref(generator)
        self._request = request
        self.request_id = request.py_request_id
        self.prompt = prompt
        self.sample = sample
        self.reuse = reuse
        self.max_new_tokens = max_new_tokens
        self.end_id = end_id
        self.lifecycle = Lifecycle.NEW
        self.consumed = 0
        self.reused = 0
        self.computed_prompt_tokens = 0
        self.explicit_tokens = 0
        self.output_token_ids: list = []
        self.pending_token: Optional[int] = None
        self.finish_reason: Optional[FinishReason] = None
        self.ledger: list = []

    @property
    def prompt_len(self) -> int:
        return len(self.prompt)

    def __repr__(self) -> str:
        return (
            f"RequestState(id={self.request_id}, {self.lifecycle.value}, consumed={self.consumed}, "
            f"reused={self.reused}, output={self.output_token_ids})"
        )


class BeamSmallModelGenerator:
    """Greedy cached generation and cross-request prefix reuse for one loaded six-block model.

    See the module docstring for the lifecycle, semantics and limits. ``model`` must be a
    loaded ``BeamSmallModelPrefillBf16Sm103Tp1``; the generator builds its own
    ``KVCacheManagerV2`` (``GENERATOR_CACHE``), ``SeqSlotManager`` and ``TorchSampler``
    (``SAMPLER_ARGS``) and releases them in ``close``.
    """

    _request_ids = itertools.count(1)

    def __init__(self, model: BeamSmallModelPrefillBf16Sm103Tp1):
        if not isinstance(model, BeamSmallModelPrefillBf16Sm103Tp1):
            raise TypeError(
                f"model must be BeamSmallModelPrefillBf16Sm103Tp1, got {type(model).__name__}"
            )
        if not model._loaded:
            raise ValueError("load_weights must complete before a generator binds the model")
        self.model = model
        self.closed = False
        self._weights = {
            name: (tensor, _tensor_state(tensor))
            for name, tensor in model.checkpoint_tensors().items()
        }
        self.manager = build_cache_manager(model.vocab_size)
        try:
            self.seq_slots = SeqSlotManager(GENERATOR_CACHE["max_batch_size"])
            self.sampler = TorchSampler(SAMPLER_ARGS)
        except BaseException:
            self.manager.shutdown()
            raise
        self._live: dict = {}
        model.register_cache_holder(self)

    # ------------------------------------------------------------------ lifecycle

    def __enter__(self) -> "BeamSmallModelGenerator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def live_requests(self) -> tuple:
        return tuple(self._live.values())

    def create_request(
        self,
        prompt_ids,
        *,
        max_new_tokens: int,
        end_id: Optional[int] = None,
        sample: bool = True,
        reuse: bool = True,
    ) -> RequestState:
        """Admit a new request; nothing runs on the GPU yet.

        ``prompt_ids``: a non-empty sequence (or 1-D integer tensor) of token ids in
        ``[0, vocab)``, copied. ``max_new_tokens >= 1``: tokens to sample (a sampling
        request) or explicit tokens to consume (``sample=False``), with
        ``P + max_new_tokens <= max_seq_len``. ``end_id``: a token id that stops sampling,
        or None. ``reuse``: whether the prefill may claim a cached prefix. Raises
        ValueError, without side effects, on anything else or when two states are live.
        """
        self._check_open()
        if isinstance(prompt_ids, torch.Tensor):
            if (
                prompt_ids.dim() != 1
                or prompt_ids.is_floating_point()
                or prompt_ids.dtype == torch.bool
            ):
                raise ValueError("prompt_ids must be a 1-D integer tensor")
            prompt = tuple(int(t) for t in prompt_ids.tolist())
        else:
            prompt = tuple(prompt_ids)
            if not all(
                isinstance(t, (int, np.integer)) and not isinstance(t, bool) for t in prompt
            ):
                raise ValueError("prompt_ids must hold integer token ids")
            prompt = tuple(int(t) for t in prompt)
        vocab = self.model.vocab_size
        capacity = self.manager.max_seq_len
        if not prompt:
            raise ValueError("the prompt must hold at least one token")
        if any(not 0 <= t < vocab for t in prompt):
            raise ValueError(f"prompt token ids must be in [0, {vocab})")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be a positive int, got {max_new_tokens!r}")
        if len(prompt) + max_new_tokens > capacity:
            raise ValueError(
                f"{len(prompt)} prompt tokens + {max_new_tokens} new tokens exceed the "
                f"{capacity}-token sequence capacity"
            )
        if end_id is not None and (type(end_id) is not int or not 0 <= end_id < vocab):
            raise ValueError(f"end_id must be None or a token id in [0, {vocab}), got {end_id!r}")
        if end_id is not None and not sample:
            raise ValueError("end_id stops sampling; a compute-only request samples nothing")
        if len(self._live) >= GENERATOR_CACHE["max_batch_size"]:
            raise ValueError(
                f"{len(self._live)} requests are live; release one before creating another"
            )
        request = _Request(
            request_id=next(self._request_ids),
            max_new_tokens=max_new_tokens,
            input_tokens=list(prompt),
            sampling_config=greedy_sampling_config(),
            is_streaming=False,
            end_id=end_id,
            exclude_input_from_output=True,
        )
        state = RequestState(
            self,
            request,
            prompt,
            sample=sample,
            reuse=reuse,
            max_new_tokens=max_new_tokens,
            end_id=end_id,
        )
        self._live[state.request_id] = state
        return state

    def release(self, state: RequestState) -> None:
        """Free the request's cache and sampler slot; its handle can no longer be used.

        Waits for issued work first. Committed pages stay reusable by later requests until
        the manager evicts them. Releasing a released or foreign handle raises ValueError.
        """
        self._check_open()
        self._check_state(state, tuple(Lifecycle))
        torch.cuda.current_stream().synchronize()
        request = state._request
        try:
            self.manager.free_resources(request)
            if request.py_seq_slot is not None:
                self.seq_slots.free_resources(request)
        finally:
            state.lifecycle = Lifecycle.RELEASED
            del self._live[state.request_id]

    def close(self) -> None:
        """Release every live request and shut the manager down. Idempotent."""
        if self.closed:
            return
        try:
            torch.cuda.current_stream().synchronize()
            for state in list(self._live.values()):
                try:
                    self.manager.free_resources(state._request)
                finally:
                    state.lifecycle = Lifecycle.RELEASED
            self._live.clear()
        finally:
            self.closed = True
            self.manager.shutdown()

    # ------------------------------------------------------------------ steps

    def prefill(self, state: RequestState) -> StepOutput:
        """Compute the prompt tokens the cache does not hold yet, after any reused prefix.

        Claims the reusable prefix through the manager (none with ``reuse=False``), sizes
        the cache for the prompt, runs the model on tokens ``reused .. P - 1`` and commits
        them for reuse. A sampling request then samples its first token, which may finish
        it. Returns the last prompt token's logits.
        """
        self._check_open()
        self._check_state(state, (Lifecycle.NEW,))
        self._check_weights()
        request = state._request
        try:
            batch = ScheduledRequests()
            batch.context_requests_last_chunk = [request]
            self.seq_slots.prepare_resources(batch)
            if not state.reuse:
                claim = self.manager.prepare_context_cache(request, reuse_limit=0)
                cache = self.manager.kv_cache_map.get(state.request_id)
                if claim != 0 or cache is None or cache.num_committed_tokens != 0:
                    raise RuntimeError(f"the reuse-disabled control claimed {claim} tokens")
            if not self.manager.prepare_context(request):
                raise RuntimeError(f"the manager could not admit request {state.request_id}")
            reused = request.context_current_position
            num_tokens = request.context_remaining_length
            if reused + num_tokens != state.prompt_len or num_tokens < 1:
                raise RuntimeError(
                    f"the manager claimed {reused} of {state.prompt_len} prompt tokens and left "
                    f"{num_tokens} to compute"
                )
            if not self.manager.resize_context(request, num_tokens):
                raise RuntimeError(f"no cache capacity for request {state.request_id}'s prompt")
            self.manager.prepare_resources(batch)
            state.reused = reused
            state.computed_prompt_tokens = num_tokens
            logits = self._forward(state, state.prompt[reused:], reused, is_context=True)
            row = logits[0, -1]
            out_logits = row.clone()
            token = reason = None
            if state.sample:
                sample_step(self.sampler, request, row, is_context=True)
            else:
                request.py_last_context_chunk = (reused, reused + num_tokens)
                request.move_to_next_context_chunk()
                request.state = LlmRequestState.GENERATION_IN_PROGRESS
            self.manager.update_context_resources(batch)
            state.consumed = state.prompt_len
            if state.sample:
                token, reason = self._after_sampling(state)
            state.lifecycle = Lifecycle.FINISHED if reason is not None else Lifecycle.PREFILLED
            self._record(
                state, "prefill", reused, num_tokens, "prompt", token, reason, claimed=reused
            )
            return StepOutput(out_logits, state.prompt_len - 1, num_tokens, token, reason)
        except Exception:
            state.lifecycle = Lifecycle.FAILED
            raise

    def decode(self, state: RequestState, token_id: Optional[int] = None) -> StepOutput:
        """Consume one token at position ``consumed`` and return the next token's logits.

        A sampling request consumes its pending sampled token (``token_id`` must be None or
        equal to it) and samples the next one; a compute-only request consumes the explicit
        ``token_id``. The cache grows by one token, which the step writes in every layer.
        """
        self._check_open()
        self._check_state(state, (Lifecycle.PREFILLED, Lifecycle.DECODING))
        request = state._request
        if state.sample:
            if state.pending_token is None:
                raise ValueError("a sampling request has no pending token to consume")
            if token_id is not None and token_id != state.pending_token:
                raise ValueError(
                    f"a sampling request consumes its sampled token {state.pending_token}, "
                    f"not {token_id}"
                )
            token, source = state.pending_token, "sampled"
        else:
            if (
                token_id is None
                or type(token_id) is not int
                or not 0 <= token_id < self.model.vocab_size
            ):
                raise ValueError(
                    f"a compute-only request needs an explicit token id, got {token_id!r}"
                )
            if state.explicit_tokens >= state.max_new_tokens:
                raise ValueError(f"the request's budget of {state.max_new_tokens} tokens is spent")
            token, source = token_id, "teacher_forced"
        if state.consumed + 1 > self.manager.max_seq_len:
            raise ValueError(f"the request already holds {state.consumed} tokens, the capacity")
        self._check_weights()
        try:
            if not state.sample:
                request.add_new_token(token, 0)
                state.explicit_tokens += 1
            past = state.consumed
            if request.max_beam_num_tokens - 1 != past or request.get_last_tokens(0) != token:
                raise RuntimeError(
                    f"request {state.request_id} holds {request.max_beam_num_tokens} tokens "
                    f"ending in {request.get_last_tokens(0)}; the step consumes {token} at {past}"
                )
            if not self.manager.try_allocate_generation(request):
                raise RuntimeError(f"no cache capacity for request {state.request_id}'s next token")
            batch = ScheduledRequests()
            batch.generation_requests = [request]
            self.manager.prepare_resources(batch)
            logits = self._forward(state, (token,), past, is_context=False)
            row = logits[0, -1]
            out_logits = row.clone()
            state.consumed = past + 1
            state.pending_token = None
            reason = sampled = None
            if state.sample:
                sample_step(self.sampler, request, row, is_context=False)
                sampled, reason = self._after_sampling(state)
                self.manager.update_resources(batch)
            else:
                self.manager.update_resources(batch)
                cache = self.manager.kv_cache_map[state.request_id]
                if not cache.resize(None, state.consumed) or cache.history_length != state.consumed:
                    raise RuntimeError(f"the manager did not record {state.consumed} cached tokens")
            state.lifecycle = Lifecycle.FINISHED if reason is not None else Lifecycle.DECODING
            self._record(state, "decode", past, 1, source, sampled, reason)
            return StepOutput(out_logits, past, 1, sampled, reason)
        except Exception:
            state.lifecycle = Lifecycle.FAILED
            raise

    def generate(self, state: RequestState) -> GenerationOutput:
        """Prefill a new sampling request and decode until the sampler finishes it."""
        self._check_open()
        self._check_state(state, (Lifecycle.NEW,))
        if not state.sample:
            raise ValueError("generate needs a sampling request")
        steps = [self.prefill(state)]
        while state.lifecycle is not Lifecycle.FINISHED:
            steps.append(self.decode(state))
        return GenerationOutput(tuple(state.output_token_ids), state.finish_reason, tuple(steps))

    # ------------------------------------------------------------------ internals

    def _check_open(self) -> None:
        if self.closed:
            raise ValueError("the generator is closed")

    def _check_state(self, state, allowed) -> None:
        if not isinstance(state, RequestState) or state._generator() is not self:
            raise ValueError("the request belongs to another generator")
        if self._live.get(state.request_id) is not state:
            raise ValueError(f"request {state.request_id} was released")
        if state.lifecycle not in allowed:
            raise ValueError(
                f"request {state.request_id} is {state.lifecycle.value}; this call needs "
                f"{', '.join(s.value for s in allowed)}"
            )

    def _check_weights(self) -> None:
        """Raise ValueError unless the model's checkpoint tensors are the bound, unmodified ones."""
        current = self.model.checkpoint_tensors()
        if current.keys() != self._weights.keys():
            raise ValueError(
                "the model's tensors changed after the generator bound it; its cache no longer "
                "matches the weights. Close the generator and build a new one"
            )
        for name, tensor in current.items():
            bound, state = self._weights[name]
            if tensor is not bound or _tensor_state(tensor) != state:
                raise ValueError(
                    f"{name} changed after the generator bound the model; its cache no longer "
                    "matches the weights. Close the generator and build a new one"
                )

    def _metadata(self, state: RequestState, num_tokens: int, past: int, is_context: bool):
        md = TrtllmAttentionMetadata(
            max_num_requests=1,
            max_num_tokens=MAX_TOKENS,
            kv_cache_manager=self.manager,
            num_contexts=1 if is_context else 0,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[past]),
            seq_lens=torch.tensor([num_tokens], dtype=torch.int),
            request_ids=[state.request_id],
            prompt_lens=[num_tokens if is_context else state.prompt_len],
            runtime_features=AttentionRuntimeFeatures(cache_reuse=True),
        )
        md.prepare()
        return md

    def _forward(self, state: RequestState, tokens: Sequence[int], past: int, is_context: bool):
        num_tokens = len(tokens)
        device = self.model.device
        md = self._metadata(state, num_tokens, past, is_context)
        ids = torch.tensor([list(tokens)], dtype=torch.int64, device=device)
        positions = torch.arange(past, past + num_tokens, dtype=torch.int64, device=device).view(
            1, -1
        )
        return self.model(ids, positions, md)

    def _after_sampling(self, state: RequestState):
        """Read the sampler's choice and stopping decision off the request."""
        request = state._request
        token = int(request.get_last_tokens(0))
        state.output_token_ids.append(token)
        reason = None
        if request.is_finished:
            reason = request.py_finish_reason
            if reason not in (FinishReason.END_ID, FinishReason.LENGTH):
                raise RuntimeError(f"request {state.request_id} finished with reason {reason}")
            state.finish_reason = reason
        else:
            state.pending_token = token
        return token, reason

    def _record(self, state, step, past, num_tokens, source, token, reason, claimed=None) -> None:
        cache = self.manager.kv_cache_map.get(state.request_id)
        record = {
            "step": step,
            "past": past,
            "tokens": num_tokens,
            "positions": [past, past + num_tokens - 1],
            "input": source,
            "consumed": state.consumed,
            "sampled": token,
            "finish_reason": None if reason is None else reason.name,
            "seq_slot": state._request.py_seq_slot,
        }
        if claimed is not None:
            record["claimed"] = claimed
        if cache is not None:
            num_blocks = int(cache.num_blocks)
            record.update(
                capacity=int(cache.capacity),
                history=int(cache.history_length),
                committed=int(cache.num_committed_tokens),
                pages=[
                    np.frombuffer(
                        cache.get_base_page_indices(pool), dtype=np.int32, count=num_blocks
                    ).tolist()
                    for pool in range(len(self.manager.impl.layer_grouping))
                ],
            )
        state.ledger.append(record)
