# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dummy requests for decoder warmup and CUDA graph capture."""

import contextlib
import math
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    KVCacheManager,
    ResourceManager,
    ResourceManagerType,
    is_context_only_kv_cache,
)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.speculative import get_draft_kv_cache_manager, get_num_extra_kv_tokens
from tensorrt_llm._torch.speculative.utils import get_static_draft_len, resolve_draft_len
from tensorrt_llm.logger import logger
from tensorrt_llm.sampling_params import SamplingParams

from .config import DecoderRunnerConfig

# Arbitrary non-greedy params used to force the advanced-sampling CUDA graph
# warmup capture path.
NON_GREEDY_CAPTURE_SAMPLING_PARAMS = SamplingParams(
    temperature=0.7, top_k=50, top_p=0.9, min_p=0.05
)

# Temperature-only params (no top-k / top-p / min-p, no seed) for the capture pass of the one-engine
# temperature-only graph variant.
TEMPERATURE_ONLY_CAPTURE_SAMPLING_PARAMS = SamplingParams(temperature=0.7)


class DummyRequestFactory:
    """Builds the dummy batches that warmup and CUDA graph capture run.

    A model family whose dummy requests need more than the KV caches
    overrides the ``_dummy_*`` methods.
    """

    # Prompt tokens of the filler CUDA graph dummy requests, also the floor of
    # the longest one; None keeps the cache manager's default.
    _dummy_request_tokens: int | None = None

    def __init__(
        self,
        *,
        config: DecoderRunnerConfig,
        model: torch.nn.Module,
        get_runtime_tokens_per_gen_step: Callable[[int], int],
        release_request_state: Callable[[int], None] | None,
    ) -> None:
        self._config = config
        self.model = model
        self.get_runtime_tokens_per_gen_step = get_runtime_tokens_per_gen_step
        self._release_request_state = release_request_state

    def _dummy_request_kwargs(
        self, resource_manager: ResourceManager, num_requests: int
    ) -> dict[str, Any]:
        """Extra ``add_dummy_requests`` arguments for ``num_requests`` dummy requests."""
        return {}

    def _dummy_position_limit(self) -> int | None:
        """Return the largest position a dummy request may reach."""
        model_config = self.model.model_config.pretrained_config
        return getattr(model_config, "max_position_embeddings", None)

    def _add_dummy_request_resources(
        self, requests: list[LlmRequest], resource_manager: ResourceManager
    ) -> bool:
        """Allocate resources dummy requests need beyond the KV caches."""
        return True

    def _free_dummy_request_resources(
        self, requests: list[LlmRequest], resource_manager: ResourceManager
    ) -> None:
        """Free what ``_add_dummy_request_resources`` allocated."""

    @staticmethod
    def _apply_chunk_alignment(
        ctx_token_nums: list[int], alignment: int, aligned: bool
    ) -> list[int] | None:
        """Rewrite warmup context lengths onto one side of ``alignment``.

        Returns lengths that are all multiples of ``alignment`` (``aligned``) or
        that include at least one non-multiple, or None if that would leave the
        batch empty. Only shrinks or drops sequences.
        """
        if aligned:
            floored = [n - n % alignment for n in ctx_token_nums]
            kept = [n for n in floored if n > 0]
            return kept or None
        if any(n % alignment != 0 for n in ctx_token_nums):
            return list(ctx_token_nums)
        # Every length is a multiple; one token off the last breaks it.
        if ctx_token_nums[-1] <= 1:
            return None
        return ctx_token_nums[:-1] + [ctx_token_nums[-1] - 1]

    def create_warmup_request(
        self,
        resource_manager: ResourceManager,
        num_tokens: int,
        num_gen_requests: int,
        least_requests: bool = True,
        chunk_alignment: int | None = None,
        chunk_aligned: bool = False,
        num_context_requests: int | None = None,
    ) -> ScheduledRequests | None:
        """Creates a generic dummy ScheduledRequests object for warmup.

        ``num_context_requests``, when set, splits the context tokens evenly
        over that many requests instead of following ``least_requests``.

        ``chunk_alignment``, when set, rewrites the context lengths onto one
        side of it, so a kernel specializing on chunk alignment can be primed
        for both variants instead of whichever one the split happens to hit.
        Only shrinks or drops sequences, so the block estimate below stays a
        safe over-estimate.
        """
        kv_cache_manager = resource_manager.get_resource_manager(
            ResourceManagerType.KV_CACHE_MANAGER
        )
        if num_gen_requests and is_context_only_kv_cache(kv_cache_manager):
            # A disaggregated context server keeps no generation-only cache
            # state (KDA draft-verify buffers, FP4 MLA V scales), so
            # generation shapes cannot run there.
            return None
        draft_kv_cache_manager = get_draft_kv_cache_manager(
            self._config.spec_config, resource_manager
        )

        spec_resource_manager = resource_manager.get_resource_manager(
            ResourceManagerType.SPEC_RESOURCE_MANAGER
        )

        available_tokens = kv_cache_manager.get_num_available_tokens(
            token_num_upper_bound=num_tokens,
            max_num_draft_tokens=self._config.max_total_draft_tokens,
        )
        available_blocks = kv_cache_manager.get_num_free_blocks()
        if num_tokens > self._config.max_num_tokens or num_tokens > available_tokens:
            return None

        if num_gen_requests > self._config.max_batch_size:
            return None
        if (
            num_gen_requests > 0
            and getattr(kv_cache_manager, "context_swa_layer_limit", None) is not None
        ):
            # Generic warmup also requests decode/mixed shapes. Exercise their
            # token and batch bounds as fresh context rows on an encoder-only
            # cache; actual generation requests remain invalid. Packing many
            # requests preserves N one-token rows for the max-batch shape.
            num_gen_requests = 0
            least_requests = False
        num_gen_tokens = num_gen_requests * (1 + self._config.max_total_draft_tokens)
        if num_gen_tokens > self._config.max_num_tokens:
            return None

        num_ctx_tokens = num_tokens - num_gen_tokens
        num_ctx_requests = 0
        ctx_requests = []
        gen_requests = []

        # Leave room for at least one decode token per request.
        max_seq_len = self._config.max_seq_len - 1
        if max_seq_len < 1:
            return None
        ctx_token_nums = []

        max_context_requests = self._config.max_batch_size - num_gen_requests
        if max_context_requests * max_seq_len < num_ctx_tokens:
            return None

        if num_ctx_tokens > 0:
            if num_context_requests is not None:
                if not 0 < num_context_requests <= min(num_ctx_tokens, max_context_requests):
                    return None
                base_length, remainder = divmod(num_ctx_tokens, num_context_requests)
                ctx_token_nums = [base_length + 1] * remainder + [base_length] * (
                    num_context_requests - remainder
                )
                if ctx_token_nums[0] > max_seq_len:
                    return None
            elif least_requests:
                num_full_seqs, remainder = divmod(num_ctx_tokens, max_seq_len)
                ctx_token_nums = [max_seq_len] * num_full_seqs
                if remainder:
                    ctx_token_nums.append(remainder)
            else:
                max_bs = min(num_ctx_tokens, max_context_requests)
                # Keep every admitted request row while balancing lengths.
                # This covers the largest batch without exceeding max_seq_len
                # on uneven totals (five tokens in two rows become [3, 2]).
                base_length, remainder = divmod(num_ctx_tokens, max_bs)
                ctx_token_nums = [base_length + 1] * remainder + [base_length] * (
                    max_bs - remainder
                )
            num_ctx_requests = len(ctx_token_nums)

        if num_ctx_requests + num_gen_requests > self._config.max_batch_size:
            return None  # Not enough batch size to fill the request

        # Mirror add_dummy_requests' actual allocation: on top of the raw
        # token count, every sequence gets num_extra_kv_tokens add_token
        # calls, and generation dummies additionally reserve
        # max_draft_loop_tokens for the draft loop.
        # In one-engine spec modes that is (max_draft_len - 1) extra KV
        # tokens plus max_draft_len draft-loop tokens per gen dummy, i.e.
        # 2 * max_draft_len - 1 on top of the single prompt token.
        # Under-counting these let warmup start an allocation that fails
        # midway and, before the partial-allocation cleanup existed,
        # permanently leaked most of the estimation-sized KV pool
        # (TRTLLM-14903).
        def blocks_for_seq(num_tokens: int) -> int:
            return math.ceil(num_tokens / kv_cache_manager.tokens_per_block)

        extra_ctx_tokens = getattr(kv_cache_manager, "num_extra_kv_tokens", 0) or 0
        extra_gen_tokens = extra_ctx_tokens + self._config.max_draft_loop_tokens
        blocks_to_use = sum(
            blocks_for_seq(token_num + extra_ctx_tokens) for token_num in ctx_token_nums
        )
        blocks_to_use += (
            num_gen_requests * self._config.max_beam_width * blocks_for_seq(1 + extra_gen_tokens)
        )

        if blocks_to_use > available_blocks and isinstance(kv_cache_manager, KVCacheManager):
            return None

        if num_ctx_tokens > 0:
            if chunk_alignment:
                adjusted = self._apply_chunk_alignment(
                    ctx_token_nums, chunk_alignment, chunk_aligned
                )
                if adjusted is None:
                    logger.debug(
                        f"Warmup batch of {ctx_token_nums} cannot be made "
                        f"{'aligned' if chunk_aligned else 'ragged'} modulo "
                        f"{chunk_alignment}; that variant will compile on the "
                        f"first request needing it."
                    )
                else:
                    ctx_token_nums = adjusted
                    num_ctx_requests = len(ctx_token_nums)

            ctx_requests = kv_cache_manager.add_dummy_requests(
                list(range(num_ctx_requests)),
                token_nums=ctx_token_nums,
                is_gen=False,
                max_num_draft_tokens=self._config.max_total_draft_tokens,
                kv_reserve_draft_tokens=self._config.max_draft_loop_tokens,
                use_mrope=self._config.use_mrope,
                draft_kv_cache_manager=draft_kv_cache_manager,
            )

            if ctx_requests is None:
                return None

            if spec_resource_manager is not None:
                spec_resource_manager.add_dummy_requests(request_ids=list(range(num_ctx_requests)))

        if num_gen_requests > 0:
            gen_requests = kv_cache_manager.add_dummy_requests(
                list(range(num_ctx_requests, num_ctx_requests + num_gen_requests)),
                token_nums=[1] * num_gen_requests,
                is_gen=True,
                max_num_draft_tokens=self._config.max_total_draft_tokens,
                kv_reserve_draft_tokens=self._config.max_draft_loop_tokens,
                use_mrope=self._config.use_mrope,
                max_beam_width=self._config.max_beam_width,
                draft_kv_cache_manager=draft_kv_cache_manager,
            )

            if gen_requests is None:
                for r in ctx_requests:
                    kv_cache_manager.free_resources(r)
                    if draft_kv_cache_manager is not None:
                        draft_kv_cache_manager.free_resources(r)
                return None

            if spec_resource_manager is not None:
                spec_resource_manager.add_dummy_requests(
                    request_ids=list(range(num_ctx_requests, num_ctx_requests + num_gen_requests))
                )

        result = ScheduledRequests()
        result.reset_context_requests(ctx_requests)
        result.generation_requests = gen_requests
        static_draft_len = get_static_draft_len(self._config.spec_config)
        resolve_draft_len(
            self._config.spec_config,
            result,
            max_draft_len=self._config.max_draft_len,
            static_draft_len=static_draft_len,
            draft_len=static_draft_len,
        )
        return result

    def create_cuda_graph_warmup_request(
        self,
        resource_manager: ResourceManager,
        batch_size: int,
        draft_len: int,
        max_seq_len: int = None,
        force_non_greedy: bool = False,
        context_token_nums: Sequence[int] = (),
        context_request_kwargs: dict[str, Any] | None = None,
        temperature_only: bool = False,
    ) -> ScheduledRequests | None:
        """Creates a dummy ScheduledRequests tailored for CUDA graph capture.

        With ``context_token_nums``, the first rows are context requests of
        those lengths, created with ``context_request_kwargs``; the rest are
        generation requests. ``temperature_only`` (with ``force_non_greedy``)
        gives the dummies temperature-only sampling params, for the capture
        pass of the temperature-only variant.
        """
        capture_sampling_params = NON_GREEDY_CAPTURE_SAMPLING_PARAMS if force_non_greedy else None
        if force_non_greedy and temperature_only:
            capture_sampling_params = TEMPERATURE_ONLY_CAPTURE_SAMPLING_PARAMS
        kv_cache_manager = resource_manager.get_resource_manager(
            ResourceManagerType.KV_CACHE_MANAGER
        )
        if getattr(
            kv_cache_manager, "context_swa_layer_limit", None
        ) is not None or is_context_only_kv_cache(kv_cache_manager):
            return None
        draft_kv_cache_manager = get_draft_kv_cache_manager(
            self._config.spec_config, resource_manager
        )

        available_blocks = kv_cache_manager.get_num_free_blocks() // self._config.max_beam_width
        if available_blocks < batch_size:
            return None

        result = ScheduledRequests()
        runtime_tokens_per_gen_step = self.get_runtime_tokens_per_gen_step(draft_len)
        runtime_draft_token_buffer_width = runtime_tokens_per_gen_step - 1
        num_contexts = len(context_token_nums)
        if num_contexts >= batch_size:
            return None

        context_requests = []
        if num_contexts:
            # Mixed capture must create its context rows as real context
            # requests; converting generation dummies afterward leaves their
            # native prompt/context bookkeeping at one token.
            context_requests = kv_cache_manager.add_dummy_requests(
                list(range(num_contexts)),
                token_nums=list(context_token_nums),
                is_gen=False,
                max_num_draft_tokens=runtime_draft_token_buffer_width,
                kv_reserve_draft_tokens=self._config.max_draft_loop_tokens,
                use_mrope=self._config.use_mrope,
                max_beam_width=self._config.max_beam_width,
                draft_kv_cache_manager=draft_kv_cache_manager,
                **(context_request_kwargs or {}),
            )
            if context_requests is None:
                return None

        # Fill all rows but the longest with minimal-length generation requests.
        num_fillers = batch_size - 1 - num_contexts
        requests = kv_cache_manager.add_dummy_requests(
            list(range(num_contexts, batch_size - 1)),
            token_nums=(
                None
                if self._dummy_request_tokens is None
                else [self._dummy_request_tokens] * num_fillers
            ),
            is_gen=True,
            max_num_draft_tokens=runtime_draft_token_buffer_width,
            kv_reserve_draft_tokens=self._config.max_draft_loop_tokens,
            use_mrope=self._config.use_mrope,
            max_beam_width=self._config.max_beam_width,
            draft_kv_cache_manager=draft_kv_cache_manager,
            capture_sampling_params=capture_sampling_params,
            **self._dummy_request_kwargs(resource_manager, num_fillers),
        )
        if requests is None:
            for request in context_requests:
                kv_cache_manager.free_resources(request)
                if draft_kv_cache_manager is not None:
                    draft_kv_cache_manager.free_resources(request)
            return None
        requests = context_requests + requests

        max_seq_len_request = self.add_longest_dummy_request(
            resource_manager,
            requests,
            batch_size,
            max_seq_len,
            runtime_draft_token_buffer_width,
            capture_sampling_params,
        )
        if max_seq_len_request is None:
            return None

        if num_contexts:
            requests.append(max_seq_len_request)
            for request, token_num in zip(requests, context_token_nums):
                request.state = LlmRequestState.CONTEXT_INIT
                request.context_current_position = 0
                request.context_chunk_size = token_num
                request.cached_tokens = 0
                request.py_batch_idx = None
            result.context_requests_last_chunk = requests[:num_contexts]
            result.generation_requests = requests[num_contexts:]
        else:
            # Insert the longest request first to simulate padding for the CUDA
            # graph.
            requests.insert(0, max_seq_len_request)
            result.generation_requests = requests
        return self.finish_cuda_graph_warmup_request(
            result, resource_manager, batch_size, draft_len
        )

    def add_longest_dummy_request(
        self,
        resource_manager: ResourceManager,
        requests: list[LlmRequest],
        batch_size: int,
        max_seq_len: int | None,
        runtime_draft_token_buffer_width: int,
        capture_sampling_params: SamplingParams | None,
    ) -> LlmRequest | None:
        """Add the dummy request with the longest sequence; free ``requests`` if it does not fit."""
        kv_cache_manager = resource_manager.get_resource_manager(
            ResourceManagerType.KV_CACHE_MANAGER
        )
        draft_kv_cache_manager = get_draft_kv_cache_manager(
            self._config.spec_config, resource_manager
        )

        def free_warmup_requests() -> None:
            for r in requests:
                kv_cache_manager.free_resources(r)
                if draft_kv_cache_manager is not None:
                    draft_kv_cache_manager.free_resources(r)

        # Add one dummy request with the maximum possible sequence length.
        max_seq_len = min(
            self._config.max_seq_len if max_seq_len is None else max_seq_len,
            kv_cache_manager.max_seq_len,
        )

        # Use max_draft_loop_tokens for capacity estimation to account
        # for the actual KV reservation per request.
        _kv_draft = self._config.max_draft_loop_tokens
        available_tokens = kv_cache_manager.get_num_available_tokens(
            token_num_upper_bound=max_seq_len, batch_size=batch_size, max_num_draft_tokens=_kv_draft
        )

        # Also consider draft KV cache capacity when it exists
        if draft_kv_cache_manager is not None:
            draft_available_tokens = draft_kv_cache_manager.get_num_available_tokens(
                batch_size=batch_size,
                token_num_upper_bound=max_seq_len,
                max_num_draft_tokens=_kv_draft,
            )
            available_tokens = min(available_tokens, draft_available_tokens)

        token_num = max(
            self._dummy_request_tokens or 1,
            min(
                available_tokens,
                max_seq_len - 1 - get_num_extra_kv_tokens(self._config.spec_config) - _kv_draft,
            ),
        )
        max_position_embeddings = self._dummy_position_limit()
        if max_position_embeddings is not None:
            token_num = min(token_num, max_position_embeddings - _kv_draft)

        token_num = int(token_num)  # Ensure int for range() in add_dummy_requests

        max_seq_len_request = kv_cache_manager.add_dummy_requests(
            request_ids=[batch_size - 1],
            token_nums=[token_num],
            is_gen=True,
            max_num_draft_tokens=runtime_draft_token_buffer_width,
            kv_reserve_draft_tokens=self._config.max_draft_loop_tokens,
            use_mrope=self._config.use_mrope,
            max_beam_width=self._config.max_beam_width,
            draft_kv_cache_manager=draft_kv_cache_manager,
            capture_sampling_params=capture_sampling_params,
            **self._dummy_request_kwargs(resource_manager, 1),
        )

        if max_seq_len_request is None:
            free_warmup_requests()
            return None
        return max_seq_len_request[0]

    def finish_cuda_graph_warmup_request(
        self,
        result: ScheduledRequests,
        resource_manager: ResourceManager,
        batch_size: int,
        draft_len: int,
    ) -> ScheduledRequests | None:
        spec_resource_manager = resource_manager.get_resource_manager(
            ResourceManagerType.SPEC_RESOURCE_MANAGER
        )
        if spec_resource_manager is not None:
            spec_resource_manager.add_dummy_requests(request_ids=list(range(batch_size)))
        if not self._add_dummy_request_resources(result.all_requests(), resource_manager):
            self._free_requests(result.all_requests(), resource_manager)
            return None
        resolve_draft_len(
            self._config.spec_config,
            result,
            max_draft_len=self._config.max_draft_len,
            static_draft_len=get_static_draft_len(self._config.spec_config),
            draft_len=draft_len,
        )
        return result

    @contextlib.contextmanager
    def release_batch_context(
        self, batch: ScheduledRequests | None, resource_manager: ResourceManager
    ) -> Iterator[ScheduledRequests | None]:
        """A context manager to automatically free resources of a dummy batch."""
        try:
            yield batch
        finally:
            if batch is not None:
                requests = batch.all_requests()
                self._free_dummy_request_resources(requests, resource_manager)
                self._free_requests(requests, resource_manager)

    def _free_requests(self, requests: list[LlmRequest], resource_manager: ResourceManager) -> None:
        kv_cache_manager = resource_manager.get_resource_manager(
            ResourceManagerType.KV_CACHE_MANAGER
        )
        if kv_cache_manager is None:
            return
        draft_kv_cache_manager = get_draft_kv_cache_manager(
            self._config.spec_config, resource_manager
        )
        spec_resource_manager = resource_manager.get_resource_manager(
            ResourceManagerType.SPEC_RESOURCE_MANAGER
        )
        for req in requests:
            kv_cache_manager.free_resources(req)
            if draft_kv_cache_manager is not None:
                draft_kv_cache_manager.free_resources(req)
            if spec_resource_manager is not None:
                spec_resource_manager.free_resources(req)
            if self._release_request_state is not None:
                self._release_request_state(req.py_request_id)
