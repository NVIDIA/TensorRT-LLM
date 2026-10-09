# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decoder CUDA graph capture and replay state."""

import contextlib
import os
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch

from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.compilation.piecewise_optimizer import PiecewiseRunner
from tensorrt_llm._torch.compilation.utils import capture_piecewise_cuda_graph
from tensorrt_llm._torch.models.modeling_utils import timing_metric
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph_runner import BreakableCUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType
from tensorrt_llm._torch.pyexecutor.sampler.sampler_common import SampleType
from tensorrt_llm._torch.speculative.utils import get_static_draft_len
from tensorrt_llm.llmapi.llm_args import (
    BaseSparseAttentionConfig,
    PrefillCudaGraphBackend,
    SeqLenAwareSparseAttentionConfig,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ...lora import LoraParamBuilder
from .config import DecoderRunnerConfig
from .dummy_requests import DummyRequestFactory
from .metadata import DecoderMetadata
from .warmup import WarmupAgreement


class DecoderCudaGraphs:
    """Owns the decoder's CUDA graph runners and captures their graphs."""

    def __init__(
        self,
        cuda_graph_runner: CUDAGraphRunner | None,
        breakable_cuda_graph_runner: BreakableCUDAGraphRunner | None,
        config: DecoderRunnerConfig,
        mapping: Mapping,
        model: torch.nn.Module,
        sparse_attention_config: BaseSparseAttentionConfig | None,
        metrics: dict[str, float],
        kv_cache_manager_key: ResourceManagerType,
        dummy_requests: DummyRequestFactory,
        agreement: WarmupAgreement,
        metadata: DecoderMetadata,
        get_runtime_tokens_per_gen_step: Callable[[int], int],
        dynamic_draft_len_mapping: dict[int, int] | None,
        lora: LoraParamBuilder,
        *,
        forward_warmup: Callable[..., Any],
        prepare_capture_batch: Callable[..., Any],
        capture_additional_graphs: Callable[[ResourceManager], None],
        release_batch_context: Callable[..., Any],
    ) -> None:
        self.cuda_graph_runner = cuda_graph_runner
        self.breakable_cuda_graph_runner = breakable_cuda_graph_runner
        self._config = config
        self.mapping = mapping
        self.model = model
        self.sparse_attention_config = sparse_attention_config
        self._metrics = metrics
        self.kv_cache_manager_key = kv_cache_manager_key
        self._dummy_requests = dummy_requests
        self._agreement = agreement
        self._metadata = metadata
        self.get_runtime_tokens_per_gen_step = get_runtime_tokens_per_gen_step
        self._dynamic_draft_len_mapping = dynamic_draft_len_mapping
        self._lora = lora
        self._forward_warmup = forward_warmup
        self._prepare_capture_batch = prepare_capture_batch
        self._capture_additional_cuda_graphs = capture_additional_graphs
        self._release_batch_context = release_batch_context
        self._force_lora_graph_for_capture: bool | None = None

    @property
    def force_lora_graph_for_capture(self) -> bool | None:
        return self._force_lora_graph_for_capture

    def set_lora(self, lora: LoraParamBuilder) -> None:
        self._lora = lora

    @contextlib.contextmanager
    def no_cuda_graph(self):
        if self.cuda_graph_runner is None:
            yield
            return
        _run_cuda_graphs = self.cuda_graph_runner.enabled
        self.cuda_graph_runner.enabled = False
        try:
            yield
        finally:
            self.cuda_graph_runner.enabled = _run_cuda_graphs

    @contextmanager
    def maybe_autotune_lora(self):
        """Enable autotuning while warming up CUDA-graph LoRA kernels."""
        if not (self._config.enable_autotuner and self._lora.cuda_graph_manager is not None):
            yield
            return

        cache_path = os.environ.get("TLLM_AUTOTUNER_CACHE_PATH", None)
        with autotune(cache_path=cache_path):
            try:
                yield
            finally:
                # Complete the PP cache hand-off even on ranks without a
                # CUDA-graph-only tunable op.
                autotuner = AutoTuner.get()
                autotuner.cache_pp_recv()
                autotuner.cache_pp_send()
                autotuner.clean_pp_flag()

    def _get_graphs_to_capture(self, cuda_graph_batch_sizes: list[int]) -> list[tuple[int, int]]:
        """Determine which (batch_size, draft_len) graphs to capture.

        Returns:
            List of (batch_size, draft_len) tuples for CUDA graph capture.
        """
        # Case 1: One-model with dynamic draft length
        if (
            self._config.spec_config is not None
            and self._config.spec_config.draft_len_schedule is not None
            and self._config.spec_config.spec_dec_mode.support_dynamic_draft_len()
        ):
            graphs = [
                (graph_bs, draft_len)
                for graph_bs, draft_len in self._dynamic_draft_len_mapping.items()
            ]
            # Workaround for dynamic draft length:
            # capture the maximum speculative graph shape up front. Dynamic draft length
            # breaks the previous assumption that attention workspace demand can be safely
            # ordered by batch size alone; a later graph shape may require a larger shared
            # graph workspace, and resizing that workspace can change its data_ptr and
            # invalidate pointers captured by earlier graphs, causing illegal memory access
            # on replay.
            #
            # This adds the overhead of one extra captured graph, and that graph is not
            # expected to be used by the normal schedule-driven dynamic draft-length path.
            #
            # Follow-up first-principles fix:
            # query or precompute the exact attention workspace requirement for all
            # reachable graph shapes, pre-size the shared graph workspace once without
            # capturing an extra graph, and avoid resizing it in graph mode afterward.
            max_spec_graph = (max(cuda_graph_batch_sizes), self._config.original_max_draft_len)
            if max_spec_graph not in graphs:
                graphs.append(max_spec_graph)
            logger.info(
                f"Dynamic draft length enabled for one-model path. "
                f"Capturing {len(graphs)} graphs: {graphs}"
            )
            return graphs

        # Case 2: Static draft length
        # Match the runtime_draft_len semantics enforced in resolve_runtime_draft_len:
        # logical K for linear-tree modes, total tree tokens for tree decoding.
        # spec_config is None for non-spec models — fall back to max_draft_len (= 0).
        draft_lengths = [get_static_draft_len(self._config.spec_config)]
        should_capture_no_spec = (
            self._config.max_total_draft_tokens > 0
            and not self._config.spec_config.spec_dec_mode.use_one_engine()
            # Assume speculation is always on if no max_concurrency set (saves memory)
            and self._config.spec_config.max_concurrency is not None
        )
        if should_capture_no_spec:
            draft_lengths.append(0)
        return [(bs, draft_len) for bs in cuda_graph_batch_sizes for draft_len in draft_lengths]

    def run_cuda_graph_warmup(self, resource_manager: ResourceManager):
        """Warm up or capture CUDA graphs for the configured graph shapes."""
        is_warmup_only = self.cuda_graph_runner.is_warmup_only
        metric_name = (
            "gen_cuda_graph_warmup_seconds" if is_warmup_only else "gen_cuda_graph_capture_seconds"
        )
        with timing_metric(metric_name, self._metrics):
            # Include LoRA autotuning and its PP cache hand-off/cleanup in
            # warmup timing, including when this rank has no graph shapes.
            lora_context = (
                self.maybe_autotune_lora() if is_warmup_only else contextlib.nullcontext()
            )
            with lora_context:
                if not (
                    self.cuda_graph_runner.enabled
                    or self._config.prefill_cuda_graph_backend != PrefillCudaGraphBackend.DISABLED
                ):
                    return

                kv_cache_manager = resource_manager.get_resource_manager(self.kv_cache_manager_key)
                context_only = (
                    getattr(kv_cache_manager, "context_swa_layer_limit", None) is not None
                )
                # Context-only workers only use the separate prefill capture.
                if not context_only:
                    from tensorrt_llm._torch.modules.linear import (
                        MXFP8LinearMethod,
                        flashinfer_mxfp8_autotune,
                        flashinfer_mxfp8_decode_graph_capture,
                    )

                    # The automatic MiniMax-M3 MXFP8 selection is decode-graph-only.
                    # Tune every generation graph shape during the warmup-only pass.
                    # Keep piecewise context/prefill capture on the native backend.
                    flashinfer_methods = [
                        quant_method
                        for module in self.model.modules()
                        if isinstance(
                            (quant_method := getattr(module, "quant_method", None)),
                            MXFP8LinearMethod,
                        )
                        and quant_method.needs_flashinfer_autotune
                    ]
                    flashinfer_autotune_context = (
                        flashinfer_mxfp8_autotune()
                        if is_warmup_only and flashinfer_methods
                        else contextlib.nullcontext()
                    )
                    with flashinfer_autotune_context, flashinfer_mxfp8_decode_graph_capture():
                        self._capture_generation_cuda_graphs(resource_manager)
                    self._capture_additional_cuda_graphs(resource_manager)
        # Piecewise graphs have separate capture machinery and do not use the
        # whole-model attention workspace. Capture them only on the second pass.
        if not is_warmup_only:
            self._capture_prefill_cuda_graphs(resource_manager)

    def _capture_generation_cuda_graphs(self, resource_manager: ResourceManager):
        """Warm up or capture pure-generation CUDA graph shapes."""
        if not self.cuda_graph_runner.enabled:
            return

        operation = "warmup" if self.cuda_graph_runner.is_warmup_only else "capture"
        logger.info(
            f"Running CUDA graph {operation} for {len(self._config.cuda_graph_batch_sizes)} batch sizes."
        )

        # Reverse order so smaller graphs can reuse memory from larger ones
        cuda_graph_batch_sizes = sorted(self._config.cuda_graph_batch_sizes, reverse=True)

        # Determine which graph shapes to process.
        graphs_to_capture = self._get_graphs_to_capture(cuda_graph_batch_sizes)
        graphs_to_capture = sorted(graphs_to_capture, reverse=True)
        # Create CUDA graphs for short and long sequences separately for sparse attention.
        # self.max_seq_len is the global max sequence length. For Helix CP each
        # rank only holds max_seq_len / cp_size tokens, so scale accordingly to
        # avoid creating warmup requests whose position_ids exceed the RoPE
        # table (max_position_embeddings).
        effective_max_seq_len = self._config.max_seq_len
        if self.mapping is not None and self.mapping.has_cp_helix():
            effective_max_seq_len = self._config.max_seq_len // self.mapping.cp_size

        sparse_config = self.sparse_attention_config
        if (
            isinstance(sparse_config, SeqLenAwareSparseAttentionConfig)
            and sparse_config.needs_separate_short_long_cuda_graphs()
        ):
            # For short sequences, subtract the maximum runtime tokens consumed
            # by a generation step so all current-step tokens stay within the
            # sequence length threshold. PARD uses 2K tokens here, not K+1.
            max_runtime_tokens_per_gen_step = self.get_runtime_tokens_per_gen_step(
                self._config.max_draft_len
            )
            # For long sequences, use the default maximum sequence length.
            max_seq_len = sparse_config.seq_len_threshold - max_runtime_tokens_per_gen_step
            if max_seq_len < effective_max_seq_len:
                max_seq_len_list = [effective_max_seq_len, max_seq_len]
            else:
                max_seq_len_list = [effective_max_seq_len]
        else:
            max_seq_len_list = [effective_max_seq_len]

        def _run_capture_pass(
            force_non_greedy: bool,
            label: str,
            force_lora_graph: bool,
            sample_type: SampleType | None = None,
        ) -> None:
            assert self._force_lora_graph_for_capture is None
            self._force_lora_graph_for_capture = force_lora_graph
            # Pin the sampling tier for this pass. maybe_get_cuda_graph reads
            # it to build the graph key, so every graph captured below records
            # the kernels of this tier and nothing else. Passes that do not name
            # a tier capture FULL graphs -- ones carrying no sampling at all --
            # which is what a batch resolving to FULL replays.
            pinned_tier = sample_type or SampleType.FULL
            self.cuda_graph_runner.set_capture_sample_type(pinned_tier)
            try:
                for bs, draft_len in graphs_to_capture:
                    if bs > self._config.max_batch_size:
                        continue

                    for max_seq_len in max_seq_len_list:
                        warmup_request = self._dummy_requests.create_cuda_graph_warmup_request(
                            resource_manager,
                            bs,
                            draft_len,
                            max_seq_len,
                            force_non_greedy=force_non_greedy,
                        )
                        with self._release_batch_context(warmup_request, resource_manager) as batch:
                            if batch is None:
                                # No KV cache space for this batch size. During KV
                                # cache estimation this makes the profiling peak
                                # unrepresentative (the final executor still
                                # captures this graph), so don't skip silently.
                                logger.warning(
                                    f"Skipping CUDA graph warmup ({label}) for "
                                    f"batch size={bs}, draft_len={draft_len}: "
                                    f"not enough KV cache space."
                                )
                                continue
                            logger.info(
                                f"Run generation-only CUDA graph {operation} ({label}) "
                                f"for batch size={bs}, draft_len={draft_len}, "
                                f"max_seq_len={max_seq_len}"
                            )
                            enable_spec_decode = draft_len > 0 or (
                                self._config.spec_config is not None
                                and self._config.spec_config.spec_dec_mode.use_one_engine()
                            )
                            self._prepare_capture_batch(
                                batch,
                                resource_manager,
                                enable_spec_decode=enable_spec_decode,
                                runtime_draft_len=draft_len,
                            )
                            self._forward_warmup(
                                batch,
                                resource_manager,
                                enable_spec_decode=enable_spec_decode,
                                runtime_draft_len=draft_len,
                            )
                            torch.cuda.synchronize()
            finally:
                self._force_lora_graph_for_capture = None
                self.cuda_graph_runner.set_capture_sample_type(None)

        if self._lora.cuda_graph_manager is None:
            lora_graph_cases = [False]
        elif self._config.cuda_graph_specialize_lora:
            # Capture the larger LoRA graph first so the base-only graph can
            # reuse its CUDA graph memory-pool allocations.
            lora_graph_cases = [True, False]
        else:
            lora_graph_cases = [True]

        # Which variants to capture depends on which sampler this engine will
        # actually run, and the two are mutually exclusive: a spec-decoding mode
        # in a one-engine mode samples inside its worker and gets a dedicated
        # sampler from get_spec_decoder, so TorchSampler -- the only sampler
        # implementing in-graph sampling -- never runs. Capturing the other
        # branch's variants would spend warmup on graphs whose keys no batch can
        # ever produce.
        #
        # The predicate is use_one_engine() rather than has_spec_decoder(): the
        # two-model eagle3 / mtp_eagle modes also "have a spec decoder", but
        # get_spec_decoder hands them a plain TorchSampler, so they belong on
        # the TorchSampler branch and do need the FAST graphs.
        uses_own_spec_decoder = (
            self._config.spec_config is not None
            and self._config.spec_config.spec_dec_mode.use_one_engine()
        )

        def _capture_variant(
            label: str, force_non_greedy: bool = False, sample_type: SampleType | None = None
        ) -> None:
            for use_lora_graph in lora_graph_cases:
                variant_label = label
                if self._lora.cuda_graph_manager is not None:
                    variant_label += ", LoRA" if use_lora_graph else ", base-only"
                _run_capture_pass(
                    force_non_greedy=force_non_greedy,
                    label=variant_label,
                    force_lora_graph=use_lora_graph,
                    sample_type=sample_type,
                )

        if uses_own_spec_decoder:
            # One-engine spec branch: the greedy argmax fast path plus the
            # advanced-sampling variant. The latter is needed because on-the-fly
            # capture is disabled outside warmup, so a batch containing a
            # non-greedy request would otherwise fall back to eager.
            #
            # Dummy warmup requests carry no sampling params, so the greedy pass
            # needs no override while the advanced one has to force the flag.
            _capture_variant("greedy")
            _capture_variant("advanced sampling", force_non_greedy=True)
        else:
            # TorchSampler branch: FULL carries no sampling in the graph and is
            # what every batch replays unless in-graph sampling is enabled, so it
            # is always captured. FAST adds the in-graph sampling kernels and is
            # captured only when opted in, since it costs extra warmup time and
            # memory.
            _capture_variant("full", sample_type=SampleType.FULL)
            if self._config.enable_in_graph_sampling:
                # Give the warmup requests real non-greedy sampling params:
                # dummies otherwise carry none, resolve to greedy, and the
                # capture would record argmax rather than the fast tier's
                # top-k/top-p kernels. Same substitution the advanced-sampling
                # pass relies on.
                _capture_variant("fast", force_non_greedy=True, sample_type=SampleType.FAST)

        # update_is_all_greedy_sample inside each forward call during the
        # non-greedy capture pass leaves is_all_greedy_sample=False on
        # spec_metadata. Reset it so the first real iteration starts clean;
        # update_is_all_greedy_sample will refresh it on every iteration anyway.
        # This is a defensive guard.
        if self._metadata.spec_metadata is not None:
            self._metadata.spec_metadata.is_all_greedy_sample = True

    def _capture_prefill_cuda_graphs(self, resource_manager: ResourceManager):
        """Warm up and capture prefill CUDA graphs, timing each phase separately.

        After capture, run prefill batches with many requests to warm up
        logits buffers outside the captured model body. This post-capture
        warmup runs for both piecewise and breakable prefill CUDA graphs.
        """
        if self._config.prefill_cuda_graph_backend == PrefillCudaGraphBackend.DISABLED or (
            self._config.prefill_cuda_graph_backend == PrefillCudaGraphBackend.PIECEWISE
            and not self._config.torch_compile_enabled
        ):
            return

        logger.info("Running prefill CUDA graph warmup...")
        prefill_cuda_graph_num_tokens = sorted(
            self._config.prefill_cuda_graph_num_tokens, reverse=True
        )

        capture_context = (
            capture_piecewise_cuda_graph(True)
            if self._config.torch_compile_piecewise_cuda_graph
            else contextlib.nullcontext()
        )
        enable_spec_decode = self._config.is_spec_decode
        runtime_draft_len = get_static_draft_len(self._config.spec_config)
        with capture_context, self.no_cuda_graph():
            for num_tokens in prefill_cuda_graph_num_tokens:
                warmup_request = self._dummy_requests.create_warmup_request(
                    resource_manager, num_tokens, 0
                )
                with self._release_batch_context(warmup_request, resource_manager) as batch:
                    self._agreement.assert_all_tp_ranks_have_warmup_batch(batch, num_tokens)
                    if batch is None:
                        continue

                    logger.info(f"Run prefill CUDA graph capture for num tokens={num_tokens}")
                    if self.breakable_cuda_graph_runner is not None:
                        runner = self.breakable_cuda_graph_runner
                        try:
                            runner.capture(
                                num_tokens,
                                lambda: self._forward_warmup(
                                    batch,
                                    resource_manager,
                                    enable_spec_decode=enable_spec_decode,
                                    runtime_draft_len=runtime_draft_len,
                                ),
                            )
                        finally:
                            self._metrics["ctx_cuda_graph_warmup_seconds"] += runner.metrics.get(
                                BreakableCUDAGraphRunner.CUDA_GRAPH_WARMUP_METRIC, 0.0
                            )
                            self._metrics["ctx_cuda_graph_capture_seconds"] += runner.metrics.get(
                                BreakableCUDAGraphRunner.CUDA_GRAPH_CAPTURE_METRIC, 0.0
                            )
                    else:
                        with timing_metric("ctx_cuda_graph_warmup_seconds", self._metrics):
                            for _ in range(PiecewiseRunner.WARMUP_STEPS):
                                self._forward_warmup(
                                    batch,
                                    resource_manager,
                                    enable_spec_decode=enable_spec_decode,
                                    runtime_draft_len=runtime_draft_len,
                                )
                            torch.cuda.synchronize()
                        with timing_metric("ctx_cuda_graph_capture_seconds", self._metrics):
                            self._forward_warmup(
                                batch,
                                resource_manager,
                                enable_spec_decode=enable_spec_decode,
                                runtime_draft_len=runtime_draft_len,
                            )
                            torch.cuda.synchronize()

        # The logits allocations grow with the number of requests and are not
        # part of the captured model body. Warm up the largest request count so
        # those allocations can be reused during stable inference.
        with timing_metric("post_ctx_cuda_graph_capture_warmup_seconds", self._metrics):
            for num_tokens in prefill_cuda_graph_num_tokens:
                warmup_request = self._dummy_requests.create_warmup_request(
                    resource_manager, num_tokens, 0, least_requests=False
                )
                with self._release_batch_context(warmup_request, resource_manager) as batch:
                    self._agreement.assert_all_tp_ranks_have_warmup_batch(batch, num_tokens)
                    if batch is None:
                        continue
                    logger.info(
                        f"Run prefill CUDA graph warmup for num tokens={num_tokens} with most requests"
                    )
                    if self.breakable_cuda_graph_runner is not None:
                        with self.no_cuda_graph():
                            self.breakable_cuda_graph_runner.warmup(
                                lambda: self._forward_warmup(
                                    batch,
                                    resource_manager,
                                    enable_spec_decode=enable_spec_decode,
                                    runtime_draft_len=runtime_draft_len,
                                ),
                                steps=1,
                            )
                    else:
                        self._forward_warmup(
                            batch,
                            resource_manager,
                            enable_spec_decode=enable_spec_decode,
                            runtime_draft_len=runtime_draft_len,
                        )
                    torch.cuda.synchronize()

    def release(self) -> None:
        if hasattr(self, "cuda_graph_runner") and self.cuda_graph_runner is not None:
            self.cuda_graph_runner.clear()
        if (
            hasattr(self, "breakable_cuda_graph_runner")
            and self.breakable_cuda_graph_runner is not None
        ):
            self.breakable_cuda_graph_runner.clear()
