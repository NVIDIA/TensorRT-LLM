# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decoder warmup."""

from __future__ import annotations

import contextlib
import gc
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.distributed import Distributed
from tensorrt_llm._torch.distributed.ops import warmup_named_allreduce_topologies
from tensorrt_llm._torch.memory_buffer_utils import clear_memory_buffers
from tensorrt_llm._torch.models.modeling_utils import timing_metric
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._torch.nccl_window_tensor_scope import discard_nccl_window_tensor_outputs
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManager
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType
from tensorrt_llm._torch.pyexecutor.runtime_hooks import runtime_hooks_of
from tensorrt_llm._torch.pyexecutor.sampler.ops.flashinfer import (
    warmup_sampling_module,
    warmup_topk_module,
)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.trace_log_utils import log_mem_snapshot
from tensorrt_llm._torch.pyexecutor.workspace import EagerWorkspaceReclaimer
from tensorrt_llm._torch.speculative import get_draft_kv_cache_manager
from tensorrt_llm._torch.speculative.utils import get_static_draft_len
from tensorrt_llm._utils import global_mpi_rank
from tensorrt_llm.llmapi.llm_args import MiniMaxM3SparseAttentionConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import CpType, Mapping

from .metadata import resolve_mamba_metadata_cls

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.guided_decoder import CapturableGuidedDecoder
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
    from tensorrt_llm._torch.pyexecutor.warmup_timer import _WarmupTimer
    from tensorrt_llm.llmapi.llm_args import BaseSparseAttentionConfig

    from .config import DecoderRunnerConfig
    from .dummy_requests import DummyRequestFactory
    from .graphs import DecoderCudaGraphs, ForwardWarmup
    from .inputs import DecoderInputPreparer
    from .metadata import DecoderMetadata


class WarmupAgreement:
    """Keeps every rank of a warmup forward on the same shapes and phases."""

    def __init__(self, dist: Distributed | None, mapping: Mapping) -> None:
        self.dist = dist
        self.mapping = mapping

    def is_distributed_forward(self) -> bool:
        """Return whether model forward can communicate with peer workers.

        ``dist`` is optional. An engine built without a communicator cannot
        enter a collective at all, so it has no peers to strand and every
        warmup failure stays rank-local.
        """
        if self.dist is None:
            return False
        return self.dist.world_size > 1 or self.mapping.dwdp_enabled

    def allgather(self) -> Callable[[int], list[int]] | None:
        """Return an allgather over the ranks this warmup forward synchronizes with.

        ``None`` means agreement cannot be established here, so a missing batch
        stays fatal rather than being skipped unilaterally.

        DWDP is the case that cannot be answered: its peers are reached through
        a ``COMM_WORLD``-derived subgroup built in ``dwdp.py``, not through
        ``self.dist``, so a ``self.dist`` allgather would report a unanimity it
        never observed.
        """
        if self.dist is None or self.mapping.dwdp_enabled:
            return None
        if self.dist.world_size <= 1:
            return None
        return self.dist.allgather

    def agree_flag(self, flag: bool) -> bool:
        """Reduce a phase-entry decision to one the whole forward group shares.

        Several predicates that gate a warmup phase are rank-local. The
        capturable guided decoder is installed only on the last pipeline rank,
        so ``guided_decoder is None`` -- and through it
        ``can_run_general_warmup`` -- differs across pipeline stages. Mamba's
        entry test reads this rank's free KV capacity. Letting one rank enter a
        phase its peers skip strands whoever enters the collective, and it also
        unbalances the per-shape agreement below.

        Any rank opting out takes the whole group out with it.
        """
        allgather = self.allgather()
        if allgather is None:
            return flag
        return all(allgather(int(flag)))

    def agree_shapes(self, configs: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Reduce warmup shapes to the ones every rank in the group proposed.

        ``_get_max_shape_warmup_requests`` derives shapes from this rank's free
        KV capacity, so under attention-DP the values -- and, once
        ``dict.fromkeys`` drops a collision, the length -- differ per rank.
        Ranks would then walk different loops and meet in different forwards.

        The intersection keeps rank 0's ordering, which matters because the
        general warmup list is ordered for torch.compile specialization.
        """
        allgather = self.allgather()
        if allgather is None:
            return configs
        per_rank = allgather([list(config) for config in configs])
        shared = set.intersection(
            *({tuple(config) for config in rank_configs} for rank_configs in per_rank)
        )
        agreed = [config for config in configs if config in shared]
        dropped = [config for config in configs if config not in shared]
        if dropped:
            logger.warning(
                f"Dropping warmup shapes {dropped} that not every rank could "
                f"propose; per-rank KV capacity differs. Remaining: {agreed}."
            )
        return agreed

    def should_run_warmup_batch(
        self, batch: ScheduledRequests | None, num_tokens: int, shape: str
    ) -> bool:
        """Decide whether this warmup shape runs, is skipped, or fails the rank.

        A rank that skips a shape its peers run leaves them blocked in that
        forward's collectives for the rest of the job. Skipping is therefore
        safe exactly when every rank in the forward group skips too, and an
        allgather establishes that before any rank enters the forward.

        The plan agreed by ``_agree_warmup_plan`` is what makes that allgather
        safe: every rank walks the same shape list, so this runs the same
        number of times everywhere.
        """
        if not self.is_distributed_forward():
            if batch is not None:
                return True
            # Safe to skip, but never silently: a skip during KV cache
            # estimation makes the profiling peak unrepresentative of
            # this shape.
            logger.warning(f"Skipping warmup shape ({shape}): not enough KV cache space.")
            return False

        allgather = self.allgather()
        if allgather is None:
            # No reachable group to agree with. Keep the TP-only check, which
            # still catches the attention-DP asymmetry it was written for.
            self.assert_all_tp_ranks_have_warmup_batch(batch, num_tokens)
            if batch is None:
                raise RuntimeError(
                    f"Warmup batch creation failed for shape ({shape}) on "
                    f"global_rank={global_mpi_rank()}, "
                    f"model_rank={self.dist.rank}, and this topology offers no "
                    f"way to confirm that peers are skipping it too. They may "
                    f"already be inside the matching forward, so this rank "
                    f"cannot skip the shape without stranding them."
                )
            return True

        flags = list(allgather(int(batch is not None)))
        if all(flags):
            return True
        if not any(flags):
            # Every rank in the forward group is skipping, so none of them is
            # left inside a collective. This is the ordinary outcome for a
            # shape that does not fit the configuration at all, such as a
            # mixed context+generation shape under ``max_batch_size=1``.
            logger.info(
                f"Skipping warmup shape ({shape}) on all "
                f"{len(flags)} ranks: not enough KV cache space."
            )
            return False

        all_tokens = list(allgather(num_tokens))
        failed_ranks = [i for i, flag in enumerate(flags) if not flag]
        raise RuntimeError(
            f"Warmup batch creation failed for shape ({shape}) on rank(s) "
            f"{failed_ranks} but succeeded on others, so entering this forward "
            f"would deadlock the ranks that still hold a batch. Per-rank "
            f"curr_max_num_tokens: {all_tokens}. This indicates asymmetric KV "
            f"cache capacity across ranks. Consider increasing "
            f"--kv_cache_free_gpu_mem_fraction."
        )

    def assert_all_tp_ranks_have_warmup_batch(self, batch, num_tokens: int) -> None:
        """Assert every TP rank has a valid warmup batch, or raise with diagnostics.

        Under attention-DP, each rank's KV cache available capacity can differ at
        runtime, causing _create_warmup_request to return None on some ranks while
        others proceed into forward() with tp_comm collectives — deadlocking the
        job. This check prevents the deadlock by failing early with diagnostic info.

        ``tp_size`` alone does not establish that peers are reachable: ``dist``
        is optional, and without a communicator there is no tp_comm collective
        to deadlock in.
        """
        if self.mapping.tp_size <= 1 or self.dist is None:
            return
        has_batch = int(batch is not None)
        all_flags = list(self.dist.tp_allgather(has_batch))
        if any(all_flags) and not all(all_flags):
            # Gather token counts for diagnostics
            all_tokens = list(self.dist.tp_allgather(num_tokens))
            failed_ranks = [i for i, f in enumerate(all_flags) if not f]
            raise RuntimeError(
                f"Warmup batch creation failed on TP rank(s) {failed_ranks} "
                f"but succeeded on others. This would cause a collective "
                f"deadlock. Per-rank curr_max_num_tokens: {all_tokens}. "
                f"This indicates asymmetric KV cache capacity across TP ranks. "
                f"Consider increasing --kv_cache_free_gpu_mem_fraction."
            )


class DecoderWarmup:
    """Warms up decoder execution before serving.

    Runs the kernel JIT and autotuning passes, specializes torch.compile
    shapes, pre-populates the memory pool and captures the CUDA graphs.
    """

    def __init__(
        self,
        *,
        config: DecoderRunnerConfig,
        mapping: Mapping,
        dist: Distributed | None,
        model: torch.nn.Module,
        sparse_attention_config: BaseSparseAttentionConfig | None,
        warmup_timer: _WarmupTimer,
        metrics: dict[str, float],
        agreement: WarmupAgreement,
        metadata: DecoderMetadata,
        dummy_requests: DummyRequestFactory,
        graphs: DecoderCudaGraphs,
        inputs: DecoderInputPreparer,
        forward_warmup: ForwardWarmup,
        torch_compile_bypass: Callable[[], contextlib.AbstractContextManager[None]],
        context_warmups_supported: bool,
        eager_workspace_reclaimable: bool,
        warmup_inputs: torch.Tensor | None = None,
    ) -> None:
        self._config = config
        self.mapping = mapping
        self.dist = dist
        self.model = model
        self.sparse_attention_config = sparse_attention_config
        self._warmup_timer = warmup_timer
        self._metrics = metrics
        self._agreement = agreement
        self._metadata = metadata
        self._dummy_requests = dummy_requests
        self._graphs = graphs
        self._inputs = inputs
        self._forward_warmup = forward_warmup
        self._warmup_inputs = warmup_inputs
        self._torch_compile_bypass = torch_compile_bypass
        # Warmups that synthesize context batches.
        self._context_warmups_supported = context_warmups_supported
        self._eager_workspace_reclaimable = eager_workspace_reclaimable
        # Token shape of the last autotuner warmup that ran a forward pass.
        self._completed_autotuner_warmup_num_tokens: int | None = None

    @property
    def guided_decoder(self) -> CapturableGuidedDecoder | None:
        return self._inputs.guided_decoder

    def run(self, resource_manager: ResourceManager) -> EagerWorkspaceReclaimer | None:
        """Warm up and capture graphs; return the eager workspace reclaimer to serve with."""
        kv_cache_manager = resource_manager.get_resource_manager(
            ResourceManagerType.KV_CACHE_MANAGER
        )
        # Warm here, not in AllReduce.__init__, to scope it to the LLM executor's graph capture.
        warmup_named_allreduce_topologies(self.model)
        # Retain a partial summary when a phase raises.
        with self._warmup_timer:
            return self._warmup_scheduled(resource_manager, kv_cache_manager)

    def _get_max_shape_warmup_requests(
        self, resource_manager: ResourceManager
    ) -> list[tuple[int, int]]:
        """
        Returns warmup configs covering the maximum context and generation shapes.
        """

        kv_cache_manager = resource_manager.get_resource_manager(
            ResourceManagerType.KV_CACHE_MANAGER
        )
        token_num_upper_bound = min(
            self._config.max_num_tokens,
            self._config.max_batch_size * (self._config.max_seq_len - 1),
        )
        curr_max_num_tokens = kv_cache_manager.get_num_available_tokens(
            token_num_upper_bound=token_num_upper_bound,
            max_num_draft_tokens=self._config.original_max_draft_len,
        )
        max_batch_size = min(
            self._config.max_batch_size,
            curr_max_num_tokens
            // (1 + self._config.max_draft_loop_tokens)
            // self._config.max_beam_width,
        )

        warmup_requests_configs = [
            (curr_max_num_tokens, 0),  # max_num_tokens, pure context
            (max_batch_size, max_batch_size),  # max_batch_size, pure generation
        ]

        return warmup_requests_configs

    def _get_full_general_warmup_requests(
        self, resource_manager: ResourceManager
    ) -> list[tuple[int, int]]:
        """
        Returns the ordered warmup configs for torch.compile specialization.

        Covers 1-token (0-1 graph specialization), max-shape (best triton autotuning),
        and small-context (2-token path) cases.
        """
        max_configs = self._get_max_shape_warmup_requests(resource_manager)
        # Specialize for 1 token pure ctx and pure gen
        one_token_configs = [(1, 0), (1, 1)]
        # Small ctx specialization
        small_ctx_configs = [(2, 0)]

        # Ordering matters for torch.compile graph specialization:
        # 1-token first to capture the 0→1 transition graph; max-shape next to seed
        # triton autotuning with the largest inputs; 2-token last for the small-ctx path.
        warmup_configs = one_token_configs + max_configs + small_ctx_configs
        # Deduplicate the warmup_configs while keeping the order.
        return list(dict.fromkeys(warmup_configs))

    def _warmup_scheduled(
        self, resource_manager: ResourceManager, kv_cache_manager
    ) -> EagerWorkspaceReclaimer | None:
        """Body of ``warmup`` for the scheduled (KV-cache-backed) path."""
        # Ahead of the legacy early returns below: only the advanced-sampling
        # CUDA graph capture pass exercises the non-greedy sampler, so with
        # cuda_graph_config=None flashinfer's sampling kernels would be
        # JIT-built mid-serving.
        with self._warmup_timer.phase(
            "sampling_module_prewarm", metrics=self._metrics, metric_name="sampling_warmup_seconds"
        ):
            warmup_sampling_module()
            if self._config.max_beam_width > 1:
                warmup_topk_module()
            self._prewarm_engram_kernels(kv_cache_manager)
            self._prewarm_csa2_sparse_indexer(kv_cache_manager)
            self._prewarm_minimax_m3_msa_indexer(kv_cache_manager)

        if kv_cache_manager is None:
            logger.info("Skipping warm up as no KV Cache manager allocated.")
            return

        # The lifetime of model engine and kv cache manager can be different.
        # Reset the global cuda graph dummy requests in warmup.
        self._graphs.reset_padding_dummies()

        if self.mapping.cp_size > 1:
            cp_type = self.mapping.cp_config.get("cp_type", None)
            if cp_type != CpType.HELIX:
                logger.info(
                    f"[ModelEngine::warmup] Skipping warmup for cp_type: {None if cp_type is None else cp_type.name}."
                )
                return

        # Create AutoTuner singleton in eager context before any compiled forward.
        # Otherwise the first get() can happen inside torch.compile tracing and
        # trigger non-traceable code (time.time(), torch.cuda.*) in the cache.
        AutoTuner.get()

        # ``guided_decoder`` is installed only on the last pipeline rank, so
        # this predicate is not rank-uniform on its own. Agree it before it
        # gates either the attention or the general phase.
        can_run_general_warmup = self._agreement.agree_flag(
            self._context_warmups_supported
            and not self.mapping.has_cp_helix()
            and self.guided_decoder is None
            and not isinstance(kv_cache_manager, MambaHybridCacheManager)
        )

        log_mem_snapshot("warmup/before_warmup")
        # Compile the DSv4 indexer-Q CuTe DSL kernels before the first
        # collective-bearing forward, so their JIT cost is not charged against the
        # MoE all-to-all completion-flag deadline.
        with self._warmup_timer.phase("cute_dsl_indexer_q"):
            self._prewarm_cute_dsl_indexer_q()
        log_mem_snapshot("warmup/after_cute_dsl_indexer_q")
        if self._context_warmups_supported:
            with (
                self._warmup_timer.phase(
                    "attention_jit", metrics=self._metrics, metric_name="attention_warmup_seconds"
                ),
                self._torch_compile_bypass(),
            ):
                self._run_attention_warmup(resource_manager, can_run_general_warmup)

        if can_run_general_warmup:
            # Specialize torch.compile graphs across the key input shapes before CUDA graph capture.
            with self._warmup_timer.phase(
                "general", metrics=self._metrics, metric_name="general_warmup_seconds"
            ):
                warmup_requests_configs = self._agreement.agree_shapes(
                    self._get_full_general_warmup_requests(resource_manager)
                )
                # Currently graph has not been captured, disable cuda graph for this warmup.
                with self._graphs.no_cuda_graph():
                    self._general_warmup(resource_manager, warmup_requests_configs)
                    # Release C++ MoE workspace buffers so the autotuner can
                    # reclaim the memory.  They will be re-allocated on next use.
                    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner

                    MoERunner.clear_all_workspaces()
                    # Clear Cache now as autotuner may use additional memory.
                    # Memory pool will be warmed up later.
                    gc.collect()
                    torch.cuda.empty_cache()

        # Helix CP is decode-only, so it tunes on generation requests.
        if self._context_warmups_supported and self.mapping.has_cp_helix():
            with (
                self._warmup_timer.phase(
                    "autotuner", metrics=self._metrics, metric_name="autotuner_warmup_seconds"
                ),
                self._torch_compile_bypass(),
            ):
                self._run_autotuner_warmup(resource_manager, generation_only=True)
            log_mem_snapshot("warmup/after_autotuner")
            gc.collect()
            torch.cuda.empty_cache()
        elif self._context_warmups_supported:
            with (
                self._warmup_timer.phase(
                    "autotuner", metrics=self._metrics, metric_name="autotuner_warmup_seconds"
                ),
                self._torch_compile_bypass(),
            ):
                self._run_autotuner_warmup(resource_manager)
            log_mem_snapshot("warmup/after_autotuner")
            # Pre-JIT Mamba SSD multi-seq + HAS_INITSTATES=True Triton kernels
            # for Mamba hybrid models. Runs regardless of enable_autotuner,
            # since MambaHybridCacheManager skips _general_warmup and the
            # default autotuner shape is single-seq / no-initstates. Safe
            # no-op for non-Mamba models.
            with (
                self._warmup_timer.phase(
                    "mamba_hybrid", metrics=self._metrics, metric_name="mamba_hybrid_warmup_seconds"
                ),
                self._torch_compile_bypass(),
            ):
                self._run_mamba_hybrid_warmup(resource_manager)
            log_mem_snapshot("warmup/after_mamba_hybrid")
            # Release the autotuner's exploration-mode intermediates. The
            # exploration leftovers are pure waste that hide tens of GiB from
            # non-torch allocators (cuBLAS handle workspace, UCX/NIXL,
            # NVSHMEM).
            gc.collect()
            torch.cuda.empty_cache()
        with self._warmup_timer.phase("cuda_graph_capture"):
            self._graphs.warmup_and_capture(resource_manager)
        log_mem_snapshot("warmup/after_cuda_graph_capture")
        # Pre-compile DeepGEMM paged_mqa_logits_metadata for every 32-aligned
        # batch bucket the runtime can produce (max_batch_size scaled by the
        # MTP / DSL expansion factor when applicable). CUDA-graph warmup only
        # exercises the batch sizes in cuda_graph_batch_sizes, which round
        # up to a subset of buckets; any inference iter whose
        # context_lens.size(0) lands on an uncovered bucket triggers an
        # nvcc-driven JIT compile (~3s stall inside _prepare_inputs) on
        # first touch. Pre-touching every bucket funnels that cost into
        # warmup. No-op on non-DSA models.
        # Both DSA hooks read attn_metadata, which only a warmup forward
        # creates; build it when every forward above was skipped.
        with self._warmup_timer.phase("dsa_prewarm"):
            self._ensure_dsa_attn_metadata_for_warmup(resource_manager)
            if getattr(self._inputs, "_dspark_confidence_enabled", False):
                self._metadata._warmup_dspark_ragged_compressor_metadata()
            with timing_metric("dg_paged_mqa_warmup_seconds", self._metrics):
                self._warmup_dg_paged_mqa_logits_metadata()
            log_mem_snapshot("warmup/after_dg_paged_mqa_logits_metadata")
            with timing_metric("cute_dsl_radix_topk_warmup_seconds", self._metrics):
                self._warmup_cute_dsl_radix_topk()
        log_mem_snapshot("warmup/after_cute_dsl_radix_topk")
        if can_run_general_warmup:
            # Pre-populate the memory pool with max-shape allocations to reduce
            # fragmentation at runtime.
            with (
                self._warmup_timer.phase(
                    "memory_pool_prepop",
                    metrics=self._metrics,
                    metric_name="memory_pool_prepopulation_seconds",
                ),
                self._torch_compile_bypass(),
            ):
                warmup_requests_configs = self._get_max_shape_warmup_requests(resource_manager)
                self._general_warmup(resource_manager, warmup_requests_configs)
            log_mem_snapshot("warmup/after_memory_pool_prepop")

        # Allocate the CUDA graph padding dummies now, while the KV cache is
        # empty. Waiting for the first padded step can race KV saturation:
        # once the cache is full, the lazy allocation in _get_padded_batch
        # fails every step and padded batches silently run eager.
        if getattr(kv_cache_manager, "context_swa_layer_limit", None) is None:
            with self._warmup_timer.phase("preallocate_padding_dummies"):
                self._graphs.preallocate_padding_dummies(resource_manager)
        log_mem_snapshot("warmup/after_preallocate_padding_dummies")

        # If this is a BOLT-instrumented build (the profile-gen job sets
        # TLLM_BOLT_CLEAR_COUNTERS=1), reset the instrumentation counters now
        # that all startup JIT/autotune/graph-capture is done, so the emitted
        # .fdata reflects steady-state serving only. No-op on normal builds.
        from tensorrt_llm._torch.bolt_profiling import maybe_bolt_clear_counters

        maybe_bolt_clear_counters()

        return self._freeze_eager_workspace_floor()

    def _freeze_eager_workspace_floor(self) -> EagerWorkspaceReclaimer | None:
        if not self._eager_workspace_reclaimable:
            return None
        if os.environ.get("TRTLLM_RECLAIM_WORKSPACE", "1") == "0":
            return None
        metadata = self._metadata.attn_metadata
        if (
            self._config.is_spec_decode
            or self.mapping.cp_size != 1
            or self.sparse_attention_config is not None
            or self._config.torch_compile_enabled
            or self._graphs.breakable_cuda_graph_runner is not None
        ):
            logger.info(
                "Eager workspace reclamation is disabled for speculative, "
                "CP, sparse, compiled, or breakable-graph models"
            )
            return None
        if (
            type(metadata) is not TrtllmAttentionMetadata
            or not metadata.workspace_reclaimable
            or metadata.workspace is None
            or metadata.workspace.untyped_storage().nbytes() == 0
        ):
            logger.info("Eager workspace reclamation requires warmup of pure fallback scratch")
            return None
        return EagerWorkspaceReclaimer(metadata)

    def _warmup_dg_paged_mqa_logits_metadata(self) -> None:
        """Pre-compile DeepGEMM's `get_paged_mqa_logits_metadata` helper for
        every 32-aligned batch bucket the runtime can produce.

        DSA's `Indexer.prepare_scheduler_metadata` calls
        `deep_gemm.get_paged_mqa_logits_metadata(context_lens, block_kv,
        num_sms)` inside `_prepare_inputs` every iteration. The underlying
        kernel is templated on `<kAlignedBatchSize, split_kv, num_sms>`
        where `kAlignedBatchSize = align(context_lens.size(0), 32)` and
        `split_kv` / `num_sms` are fixed for a given (block_kv, device).
        deep_gemm's Python-side JIT compiles a fresh cubin (spawning
        nvcc/cicc/ptxas, ~3s on GB300) the first time each `aligned_bs`
        is requested. CUDA-graph warmup exercises only the batch sizes in
        `cuda_graph_batch_sizes`, which round up to a subset of the 32-
        aligned buckets; every uncovered bucket that the inference
        workload later touches produces a 3s stall on that iteration.
        Pre-touching every bucket here funnels those compiles into the
        deterministic warmup phase.

        `context_lens.size(0)` is not always `num_generations`. For MTP
        with `use_expanded_buffers_for_mtp=True` the expanded call passes
        `num_generations * (1 + max_draft_tokens)`. For DSL expansion the
        call passes `num_generations * dsl_expand_factor`, where
        `dsl_expand_factor = next_n // eff` (`eff in kernel_atoms`, see
        `_pick_dsl_expand` in `dsa.py`); its worst case is
        `next_n = 1 + max_draft_tokens` when `eff == 1`. Reading the
        current `dsl_expand_factor` off the metadata would under-estimate
        the eventual max (it defaults to 1 before any prepare() has run,
        and per-iter picks can differ across iters when CUDA graph is
        off), so we use the static upper bound `1 + max_draft_tokens`
        for both expansion paths. Bucket range is also scaled by
        `max_beam_width` as a defense-in-depth ceiling for future beam
        support (no-op today — DSA does not use beam). No-op on non-DSA
        models.

        Best-effort: per-bucket JIT failures are logged and skipped so a
        single broken bucket does not abort PyExecutor startup.
        """
        attn_meta = self._metadata.attn_metadata
        if attn_meta is None:
            return
        try:
            from tensorrt_llm._torch.attention.backends.sparse.dsa import (
                _DG_SCHEDULE_BLOCK_KV,
                DSAtrtllmAttentionMetadata,
            )
        except ImportError:
            return
        if not isinstance(attn_meta, DSAtrtllmAttentionMetadata):
            return
        try:
            from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata
        except ImportError:
            logger.info(
                "[DG warmup] deep_gemm.get_paged_mqa_logits_metadata not "
                "available; skipping paged_mqa_logits_metadata prewarm."
            )
            return

        num_sms = attn_meta.num_sms
        max_bs = max(1, int(self._config.max_batch_size))
        beam_width = max(1, self._config.max_beam_width)
        # Static upper bound on the row-count multiplier applied to
        # `context_lens`. Both MTP-expanded and DSL-expanded call sites
        # are bounded above by `(1 + max_draft_tokens)`; see the
        # docstring for why we don't read the runtime `dsl_expand_factor`
        # here.
        max_draft_tokens = int(getattr(attn_meta, "max_draft_tokens", 0) or 0)
        expands_batch = getattr(attn_meta, "use_expanded_buffers_for_mtp", False) or getattr(
            attn_meta, "expand_for_dsl", False
        )
        expand_factor = 1 + max_draft_tokens if expands_batch else 1
        max_aligned = ((max_bs * beam_width * expand_factor + 31) // 32) * 32
        buckets = list(range(32, max_aligned + 32, 32))
        logger.info(
            f"[DG warmup] Pre-compiling paged_mqa_logits_metadata for "
            f"{len(buckets)} aligned batch buckets up to {max_aligned} "
            f"(block_kv={_DG_SCHEDULE_BLOCK_KV}, num_sms={num_sms}, "
            f"max_bs={max_bs}, beam_width={beam_width}, "
            f"expand_factor={expand_factor})"
        )
        for aligned_bs in buckets:
            # Kernel scans `context_lens` and prefix-sums schedules; a
            # zero-filled 2D tensor of shape (aligned_bs, 1) is enough to
            # trigger dispatch and compile — the metadata output is
            # discarded.
            dummy = torch.zeros(aligned_bs, 1, dtype=torch.int32, device="cuda")
            try:
                _ = get_paged_mqa_logits_metadata(dummy, _DG_SCHEDULE_BLOCK_KV, num_sms)
            except RuntimeError as e:
                # Narrow to RuntimeError so signature drifts in
                # get_paged_mqa_logits_metadata (TypeError / ValueError)
                # surface loudly instead of silently degrading perf.
                logger.warning(
                    f"[DG warmup] paged_mqa_logits_metadata prewarm failed "
                    f"for aligned_bs={aligned_bs} "
                    f"(block_kv={_DG_SCHEDULE_BLOCK_KV}, num_sms={num_sms}); "
                    f"skipping bucket. {type(e).__name__}: {e}"
                )
        torch.cuda.synchronize()

    def _prewarm_cute_dsl_indexer_q(self) -> None:
        """Pre-compile the DSv4 indexer-Q CuTe DSL kernels, then barrier.

        Runs before any collective-bearing forward so this op's first-touch
        ``cute.compile`` is not charged against the MoE all-to-all
        completion-flag deadline. It is a partial mitigation only: other
        first-touch compiles remain inside collective-bearing forwards, and some
        sit on the all-to-all path itself and cannot be pre-compiled this way.
        The warmup MoE communication timeout (see ``moe_comm_timeout_guard.py``)
        covers the general case.

        Only the fallback tactics are compiled -- what an eager, cache-miss
        forward selects. The runner's kernel cache key excludes m/n/k, so one
        compile per tactic covers every shape. Uses the real module and weights,
        so it cannot drift from what the model runs.

        No-op unless this rank holds the weights of a DSv4 indexer; only an
        indexer whose Q projection takes the fused CuTe DSL path is compiled.
        See nvbugs/6482566.
        """
        try:
            from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import DeepseekV4Indexer
        except ImportError:
            return

        # Under pipeline parallelism the layers of other stages keep their
        # modules but not their parameters, so require the wq_b weight itself.
        local_indexers = [
            m
            for m in self.model.modules()
            if isinstance(m, DeepseekV4Indexer)
            and getattr(getattr(m, "wq_b", None), "weight", None) is not None
        ]
        if not local_indexers:
            return

        indexer = next(
            (m for m in local_indexers if m._is_fused_project_mxfp4_enabled(torch.bfloat16)),
            None,
        )
        compiled = 0
        if indexer is not None:
            weight = indexer.wq_b.weight
            # _fallback_tactic() branches on m at 4 and 8, so these three token
            # counts cover every fallback tactic it can return.
            with torch.inference_mode():
                for num_tokens in (4, 8, 16):
                    try:
                        qr = torch.zeros(
                            (num_tokens, weight.shape[1]),
                            dtype=torch.bfloat16,
                            device=weight.device,
                        )
                        position_ids = torch.zeros(
                            (num_tokens,), dtype=torch.int32, device=weight.device
                        )
                        indexer._project_and_quantize_q(qr, position_ids)
                        compiled += 1
                    except Exception as e:
                        # A prewarm miss only defers this compile to the first
                        # forward that needs it, so it must not fail startup.
                        logger.warning(
                            f"indexer-Q CuTe DSL prewarm skipped for {num_tokens} "
                            f"tokens. {type(e).__name__}: {e}"
                        )
            torch.cuda.synchronize()

        # Hold the TP group here until its slowest rank has finished these
        # compiles, so they add no skew between TP peers at the first MoE
        # all-to-all dispatch. Whether a rank enters depends only on the model
        # layout, not on the rank-local fused-path check above, so the whole TP
        # group always enters together.
        if self.mapping.tp_size > 1 and self.dist is not None:
            self.dist.tp_allgather(1)
        if compiled:
            logger.info(f"indexer-Q CuTe DSL prewarm complete ({compiled}/3 token counts)")

    def _ensure_dsa_attn_metadata_for_warmup(self, resource_manager: ResourceManager) -> None:
        """Build the DSA attention metadata if no warmup forward created it, so
        the top-K pre-compile hooks still run (draft engine, guided decoder or
        context-only server without general warmup). No-op unless DSA."""
        if self._metadata.attn_metadata is not None:
            return
        try:
            from tensorrt_llm._torch.attention.backends.sparse.dsa import DSAtrtllmAttentionMetadata
        except ImportError:
            return
        metadata_cls = getattr(self._config.attention_backend, "Metadata", None)
        if metadata_cls is None or not issubclass(metadata_cls, DSAtrtllmAttentionMetadata):
            return
        kv_cache_manager = resource_manager.get_resource_manager(
            ResourceManagerType.KV_CACHE_MANAGER
        )
        if kv_cache_manager is None:
            return
        self._metadata.setup_attention(
            kv_cache_manager, get_draft_kv_cache_manager(self._config.spec_config, resource_manager)
        )

    def _warmup_cute_dsl_radix_topk(self) -> None:
        """Pre-compile the DSA radix-filter CuTe DSL decode top-k for every
        cluster_size band during warmup, before serving.

        Captured geometries are already compiled by the warmup-step forwards;
        this fills in the bands the eager (non-captured) decode path can still
        hit (mixed prefill+decode batch, or cuda_graph disabled) so they do
        not pay a first-touch JIT stall on a live request. DSA-specific params
        live on the metadata, so delegate to it. No-op on non-DSA models.
        """
        attn_meta = self._metadata.attn_metadata
        if attn_meta is None:
            return
        try:
            from tensorrt_llm._torch.attention.backends.sparse.dsa import DSAtrtllmAttentionMetadata
        except ImportError:
            return
        if isinstance(attn_meta, DSAtrtllmAttentionMetadata):
            next_n = 1 + self._config.original_max_draft_len
            attn_meta.warmup_cute_dsl_radix_topk(next_n)
            if hasattr(attn_meta, "warmup_selfsampling_topk"):
                attn_meta.warmup_selfsampling_topk(
                    next_n, batch_sizes=self._config.cuda_graph_batch_sizes
                )

    def _general_warmup(
        self, resource_manager: ResourceManager, warmup_requests_configs: list[tuple[int, int]]
    ) -> None:
        """
        Runs forward passes for each config in warmup_requests_configs.

        Serves both torch.compile graph specialization and memory pool pre-population.
        """
        # Disable CUDA graph replay during general warmup to avoid replaying
        # graphs with stale KV cache block offsets from capture time.
        with self._graphs.no_cuda_graph():
            self._general_warmup_impl(resource_manager, warmup_requests_configs)

    def _general_warmup_impl(
        self, resource_manager: ResourceManager, warmup_requests_configs: list[tuple[int, int]]
    ) -> None:
        for num_tokens, num_gen_tokens in warmup_requests_configs:
            # Helix CP does not support warmup with context requests.
            if self.mapping.has_cp_helix() and num_tokens != num_gen_tokens:
                continue
            try:
                with self._dummy_requests.release_batch_context(
                    self._dummy_requests.create_warmup_request(
                        resource_manager, num_tokens, num_gen_tokens
                    ),
                    resource_manager,
                ) as batch:
                    if not self._agreement.should_run_warmup_batch(
                        batch,
                        num_tokens,
                        f"general, num_tokens={num_tokens}, num_gen_tokens={num_gen_tokens}",
                    ):
                        continue
                    logger.info(
                        f"Run warmup with {num_tokens} tokens, include {num_gen_tokens} generation tokens"
                    )
                    with self._warmup_timer.phase(
                        f"general shape num_tokens={num_tokens}, num_gen_tokens={num_gen_tokens}",
                        record=False,
                        log_start=False,
                    ):
                        with discard_nccl_window_tensor_outputs(
                            (getattr(self, "_warmup_inputs", None), batch)
                        ):
                            self._forward_warmup(
                                batch,
                                resource_manager,
                                enable_spec_decode=self._config.is_spec_decode,
                                runtime_draft_len=get_static_draft_len(self._config.spec_config),
                            )
                        torch.cuda.synchronize()
            except torch.OutOfMemoryError:
                if self._agreement.is_distributed_forward():
                    # Peers are inside the same forward's collectives and
                    # cannot follow a rank-local skip.
                    raise
                logger.warning(
                    f"OOM during general warmup with {num_tokens} tokens, "
                    f"{num_gen_tokens} generation tokens. Skipping."
                )
                # If the OOM aborted the forward between dispatch() and
                # combine(), the MoE A2A state machines are stuck in
                # ``dispatched`` and the next warmup will hit
                # ``dispatch called twice``. Reset them before retrying a
                # smaller shape.
                self._reset_moe_alltoall_state()
                torch.cuda.empty_cache()

    def _reset_moe_alltoall_state(self) -> None:
        """Reset all MoE all-to-all state machines reachable from ``self.model``.

        Each MoE backend keeps a small dispatch/combine phase state per layer
        (``MoeAlltoAll`` or ``NVLinkOneSided``). A forward that calls
        ``dispatch`` but raises before reaching ``combine`` (e.g., a warmup
        OOM mid-MoE) leaves that state in ``dispatched``, which fails the
        invariant on the next ``dispatch`` call. This helper walks the model
        and resets any A2A state found, so subsequent forwards start clean.
        """
        for module in self.model.modules():
            for attr_name in ("moe_a2a", "comm"):
                obj = getattr(module, attr_name, None)
                reset = getattr(obj, "reset_state", None)
                if callable(reset):
                    try:
                        reset()
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            f"Failed to reset MoE A2A state on {type(module).__name__}.{attr_name}: {e}"
                        )

    def _run_attention_warmup(
        self, resource_manager: ResourceManager, can_run_general_warmup: bool = True
    ) -> None:
        if not issubclass(self._config.attention_backend.Metadata, TrtllmAttentionMetadata):
            return

        logger.info("Running TRTLLM-Gen FMHA JIT warmup")

        warmup_requests_configs = []
        if self.guided_decoder is None:
            # doesn't support guided decoding
            warmup_requests_configs.append(
                (1 + self._config.max_total_draft_tokens, 1)
            )  # one generation request
        else:
            logger.debug("Skipped TRTLLM-Gen FMHA JIT warmup for Gen kernels")

        if can_run_general_warmup:
            warmup_requests_configs.append((1, 0))  # one context token
        else:
            logger.debug("Skipped TRTLLM-Gen FMHA JIT warmup for Ctx kernels")

        # Kimi K3 / Kimi Linear are warmed by ``_run_mamba_hybrid_warmup``, not
        # here: they always run on a ``MambaHybridCacheManager``, so
        # ``can_run_general_warmup`` is False and the configs above never run.

        if self.guided_decoder is None and can_run_general_warmup:
            # The cute_dsl_mla FMHA lib now only support the generation-only batch, we need to
            # warmup the TRTLLM-Gen FMHA lib for the mixed context+generation batch.
            # One MIXED context+generation batch (1 ctx token + 1 gen request).
            warmup_requests_configs.append((1 + self._config.max_total_draft_tokens + 1, 1))
        else:
            logger.debug(
                "Skipped TRTLLM-Gen flashinfer_trtllm_gen FMHA lib JIT warmup When enable cute_dsl_mla FMHA lib"
            )

        for num_tokens, num_gen_requests in warmup_requests_configs:
            warmup_request = self._dummy_requests.create_warmup_request(
                resource_manager, num_tokens=num_tokens, num_gen_requests=num_gen_requests
            )

            with (
                self._graphs.no_cuda_graph(),
                self._dummy_requests.release_batch_context(
                    warmup_request, resource_manager
                ) as batch,
            ):
                if not self._agreement.should_run_warmup_batch(
                    batch,
                    num_tokens,
                    f"attention, num_tokens={num_tokens}, num_gen_requests={num_gen_requests}",
                ):
                    continue
                # The first forward of the process lands here, so this shape
                # also absorbs every first-touch JIT (CuTe DSL GEMM, FMHA
                # NVRTC grid, ...). Time it per shape so a slow startup can be
                # attributed without a debugger.
                with self._warmup_timer.phase(
                    f"attention shape num_tokens={num_tokens}, num_gen_requests={num_gen_requests}",
                    record=False,
                    log_start=True,
                ):
                    with discard_nccl_window_tensor_outputs(getattr(self, "_warmup_inputs", None)):
                        self._forward_warmup(
                            batch,
                            resource_manager,
                            enable_spec_decode=self._config.is_spec_decode,
                            runtime_draft_len=get_static_draft_len(self._config.spec_config),
                            trtllm_gen_jit_warmup=True,
                        )
                    torch.cuda.synchronize()

    @staticmethod
    def _release_megamoe_profiling_scratch() -> None:
        # MegaMoE tuning resources are shared across layers, so only the engine
        # can release them after its full autotune warmup and before graph
        # capture. Later eviction could invalidate a captured workspace pointer.
        from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as _megamoe_op

        release_megamoe_scratch = getattr(_megamoe_op, "release_megamoe_profiling_scratch", None)
        if release_megamoe_scratch is not None:
            release_megamoe_scratch()

    def _run_autotuner_warmup(
        self, resource_manager: ResourceManager, generation_only: bool = False
    ) -> None:
        """Runs forward passes to populate the autotuner cache.

        ``generation_only`` tunes on one batch of ``max_batch_size`` generation
        requests instead of a context batch, for Helix CP generation workers.
        """
        from tensorrt_llm._torch.custom_ops.torch_custom_ops import (
            IS_FLASHINFER_MXFP8_CUTE_DSL_AVAILABLE,
            MXFP8GemmRunner,
        )
        from tensorrt_llm._torch.modules.linear import (
            MXFP8LinearMethod,
            flashinfer_mxfp8_autotune,
            mxfp8_gemm_backend_override,
        )

        enable_trtllm_autotuner = self._config.enable_autotuner
        if not enable_trtllm_autotuner:
            return

        mxfp8_methods = []
        for module in self.model.modules():
            quant_method = getattr(module, "quant_method", None)
            if isinstance(quant_method, MXFP8LinearMethod):
                mxfp8_methods.append(quant_method)

        # This engine owns startup warmup, so it explicitly opts its MXFP8
        # methods into native tuning. Standalone modules and engine paths that
        # skip this warmup remain on the direct native op.
        for method in mxfp8_methods:
            method.enable_native_autotune()

        # Native and FlashInfer tuning are independent. Capture native
        # eligibility before enabling graph-only FlashInfer dispatch.
        native_mxfp8_methods = [method for method in mxfp8_methods if method.needs_native_autotune]
        compile_all_batches = (
            self._config.torch_compile_enabled and not self._config.torch_compile_prefill_only
        )
        if compile_all_batches and mxfp8_gemm_backend_override() is None:
            # Compiled auto dispatch uses native; do not tune unused backends.
            # Prefill-only compile retains the eager generation-graph policy.
            for method in mxfp8_methods:
                method.disable_flashinfer_auto()
        use_mxfp8_flashinfer_graph_default = (
            self._graphs.enabled
            and not compile_all_batches
            and mxfp8_gemm_backend_override() is None
            and any(
                getattr(module, "_use_flashinfer_mxfp8_decode_graph_default", False)
                for module in self.model.modules()
            )
        )
        if use_mxfp8_flashinfer_graph_default:
            # CuTeDSL is the alternative backend; PP has no graph-pass handoff.
            tune_with_cute_dsl = (
                IS_FLASHINFER_MXFP8_CUTE_DSL_AVAILABLE and not self.mapping.has_pp()
            )
            for quant_method in mxfp8_methods:
                quant_method.enable_flashinfer_auto()
                quant_method.tune_decode_graph_backends = (
                    tune_with_cute_dsl and quant_method.uses_flashinfer
                )
        # An explicit auto setting stays intact, but compiled auto dispatch
        # uses native GEMM. Do not run or mark an unused FlashInfer tuning pass.
        flashinfer_mxfp8_methods = [
            method
            for method in mxfp8_methods
            if method.needs_flashinfer_autotune
            and (not compile_all_batches or method.backend == "flashinfer")
        ]

        # Every TP and PP rank must make the same backend decision before any
        # rank returns or enters a tuning forward with model collectives.
        if self.mapping.tp_size > 1 or self.mapping.has_pp():
            local_flashinfer_enabled = int(bool(flashinfer_mxfp8_methods))
            all_flashinfer_enabled = [local_flashinfer_enabled]
            if self.mapping.tp_size > 1:
                all_flashinfer_enabled = list(self.dist.tp_allgather(local_flashinfer_enabled))
            if self.mapping.has_pp():
                all_flashinfer_enabled = [
                    enabled
                    for stage_flags in self.dist.pp_allgather(all_flashinfer_enabled)
                    for enabled in stage_flags
                ]
            if any(all_flashinfer_enabled) and not all(all_flashinfer_enabled):
                forced_flashinfer = any(method.backend == "flashinfer" for method in mxfp8_methods)
                for method in mxfp8_methods:
                    method.disable_flashinfer_auto()
                flashinfer_mxfp8_methods = []
                if forced_flashinfer:
                    raise RuntimeError(
                        "FlashInfer MXFP8 was explicitly requested but is not "
                        "available on every TP/PP rank"
                    )
                logger.warning(
                    "FlashInfer MXFP8 availability differs across TP/PP ranks; "
                    "using the native TensorRT-LLM GEMM backend on every rank."
                )

        enable_flashinfer_mxfp8_autotuner = bool(flashinfer_mxfp8_methods)
        enable_native_mxfp8_autotuner = bool(native_mxfp8_methods)

        kv_cache_manager = resource_manager.get_resource_manager(
            ResourceManagerType.KV_CACHE_MANAGER
        )
        if generation_only:
            tokens_per_gen_request = 1 + self._config.max_total_draft_tokens
            num_gen_requests = min(
                self._config.max_batch_size,
                self._config.max_num_tokens // tokens_per_gen_request,
            )
            curr_max_num_tokens = num_gen_requests * tokens_per_gen_request
        else:
            token_num_upper_bound = min(
                self._config.max_num_tokens,
                self._config.max_batch_size * (self._config.max_seq_len - 1),
            )
            curr_max_num_tokens = kv_cache_manager.get_num_available_tokens(
                token_num_upper_bound=token_num_upper_bound,
                max_num_draft_tokens=self._config.original_max_draft_len,
            )

        # The warmup runs again after the KV cache is rebuilt; when the token
        # shape is unchanged and no MXFP8 tuning is pending, the profiling
        # cache already holds every tactic this pass would tune. The token
        # shape comes from this rank's free KV capacity, so the whole forward
        # group must agree before any rank skips the collective-bearing pass.
        skip_completed_warmup = self._agreement.agree_flag(
            self._completed_autotuner_warmup_num_tokens == curr_max_num_tokens
            and not enable_flashinfer_mxfp8_autotuner
            and not enable_native_mxfp8_autotuner
        )
        if skip_completed_warmup:
            logger.info(
                "Skipping duplicate autotuner warmup for already completed "
                f"num_tokens={curr_max_num_tokens}"
            )
            # Release what earlier warmup phases cached, as the full pass does
            # below, so it does not inflate the memory baseline.
            clear_memory_buffers()
            torch.cuda.empty_cache()
            return

        AutoTuner.get().setup_distributed_state(self.mapping, self.dist)
        logger.info(
            f"Running autotuner warmup (TRT-LLM={enable_trtllm_autotuner}, "
            f"native MXFP8={enable_native_mxfp8_autotuner}, "
            f"FlashInfer MXFP8={enable_flashinfer_mxfp8_autotuner})..."
        )

        if generation_only:
            warmup_configs = [(curr_max_num_tokens, num_gen_requests)]
        else:
            warmup_configs = [(curr_max_num_tokens, 0)]
            if self.guided_decoder is None and not self.mapping.has_pp():
                # Add generation request to warmup the autotuner cache.
                warmup_configs.append((1 + self._config.max_total_draft_tokens, 1))

        def run_autotuner_pass(autotune_context: Any, synchronize_trtllm_cache: bool) -> bool:
            """Run one isolated tuning pass with fresh synthetic batches."""
            ran_forward = False
            with self._graphs.no_cuda_graph(), autotune_context:
                for num_tokens, num_gen_requests in warmup_configs:
                    warmup_request = self._dummy_requests.create_warmup_request(
                        resource_manager, num_tokens, num_gen_requests
                    )
                    with self._dummy_requests.release_batch_context(
                        warmup_request, resource_manager
                    ) as batch:
                        if not self._agreement.should_run_warmup_batch(
                            batch,
                            num_tokens,
                            f"autotuner, num_tokens={num_tokens}, "
                            f"num_gen_requests={num_gen_requests}",
                        ):
                            continue
                        with self._warmup_timer.phase(
                            f"autotuner shape num_tokens={num_tokens}, "
                            f"num_gen_requests={num_gen_requests}",
                            record=False,
                            log_start=True,
                        ):
                            with discard_nccl_window_tensor_outputs(
                                (getattr(self, "_warmup_inputs", None), batch)
                            ):
                                self._forward_warmup(
                                    batch,
                                    resource_manager,
                                    enable_spec_decode=self._config.is_spec_decode,
                                    runtime_draft_len=get_static_draft_len(
                                        self._config.spec_config
                                    ),
                                )
                            ran_forward = True
                            torch.cuda.synchronize()

                if ran_forward and synchronize_trtllm_cache:
                    # pp_recv in AutoTuner choose_one will never be called if there is no tuning op
                    # during the forward pass.
                    # So we need to make an extra call to consume the previous rank's pp_send to
                    # guarantee that the previous rank's pp_send is released.
                    AutoTuner.get().cache_pp_recv()
                    # Send the cache after the tuning process to the next PP rank
                    AutoTuner.get().cache_pp_send()
                    # Clean the pp flag to avoid deadlock with synchronous send/recv
                    AutoTuner.get().clean_pp_flag()
            return ran_forward

        cache_path = os.environ.get("TLLM_AUTOTUNER_CACHE_PATH", None)
        ran_native_forward = run_autotuner_pass(
            autotune(cache_path=cache_path), synchronize_trtllm_cache=True
        )
        ran_flashinfer_forward = False
        if enable_flashinfer_mxfp8_autotuner:
            ran_flashinfer_forward = run_autotuner_pass(
                flashinfer_mxfp8_autotune(), synchronize_trtllm_cache=False
            )
        # The model's declared hooks (the all-off default when it declares none): the warmup passes also run on
        # engines built without the runner's constructor.
        if runtime_hooks_of(self.model).merge_tuned_tactics_across_ranks:
            # Rank-local profiling noise must not leave tensor-parallel peers
            # serving different tactics for one shape: keep the fastest everywhere.
            # Collective; before the native MXFP8 cache below registers the tactics.
            AutoTuner.get().post_tune_merge_tactics()

        if enable_flashinfer_mxfp8_autotuner:
            if ran_flashinfer_forward:
                for method in flashinfer_mxfp8_methods:
                    method.mark_flashinfer_autotuned()
            else:
                forced_flashinfer = any(
                    method.backend == "flashinfer" for method in flashinfer_mxfp8_methods
                )
                for method in flashinfer_mxfp8_methods:
                    method.disable_flashinfer_auto()
                if forced_flashinfer:
                    raise RuntimeError(
                        "FlashInfer MXFP8 was explicitly requested but its autotuner "
                        "warmup forward could not run"
                    )
                logger.warning(
                    "FlashInfer MXFP8 autotuning could not run; using the native "
                    "TensorRT-LLM GEMM backend."
                )

        if enable_native_mxfp8_autotuner:
            if ran_native_forward:
                MXFP8GemmRunner.sync_all_tactic_caches(AutoTuner.get())
                for method in native_mxfp8_methods:
                    method.mark_native_autotuned()
            else:
                for method in native_mxfp8_methods:
                    method.disable_native_autotune()
                logger.warning(
                    "Native MXFP8 autotuning had no runnable warmup batch; "
                    "using the default native GEMM tactic."
                )

        logger.info(
            f"[Autotuner] Cache size after warmup is {len(AutoTuner.get().profiling_cache)}"
        )
        AutoTuner.get().print_profiling_cache()

        self._release_megamoe_profiling_scratch()

        # Clear workspace buffers allocated during the autotuner forward pass.
        # The autotuner runs a context-only forward with max_num_tokens, which
        # causes the global Buffers pool to cache large MoE/GEMM workspaces.
        # If not cleared, these inflate the memory baseline seen by the KV cache
        # profiler, reducing memory available for activations during inference.
        clear_memory_buffers()
        torch.cuda.empty_cache()
        if ran_native_forward:
            self._completed_autotuner_warmup_num_tokens = curr_max_num_tokens

    def _run_mamba_hybrid_warmup(self, resource_manager: ResourceManager) -> None:
        """Pre-JIT the Mamba SSD multi-seq + HAS_INITSTATES=True Triton kernels.

        Mamba hybrid models (e.g. Nemotron 3 Super 120B, Nemotron-Nano-12B-v2)
        skip ``_general_warmup`` because ``can_run_general_warmup`` is False
        when the KV cache manager is a ``MambaHybridCacheManager``. The default
        ``_run_autotuner_warmup`` then issues a single ``least_requests=True``
        prefill = 1 sequence with ``num_cached_tokens_per_seq = 0``, which only
        compiles the ``num_seqs == 1`` / ``HAS_INITSTATES=False`` variants of
        the SSD kernels. The first real serve iteration with chunked prefill
        and multiple context requests then triggers autotune of the missing
        variants mid-inference, producing a ~30 s stall / large P99 spike.

        This method runs two extra forward passes to compile those variants
        during warmup:

        1. ``least_requests=False``, chunk-ragged — splits
           ``curr_max_num_tokens`` into many short sequences, forcing the
           multi-seq path of
           ``cu_seqlens_to_chunk_indices_offsets_triton`` and its
           ``_cu_seqlens_triton_kernel``.
        2. ``least_requests=False`` inside
           ``Mamba2Metadata.force_initial_states_for_warmup()`` — same as (1)
           plus the ``HAS_INITSTATES=True`` variants of
           ``_state_passing_fwd_kernel``, ``_chunk_scan_fwd_kernel``, and
           ``_chunk_state_varlen_kernel``.

        Runs regardless of ``enable_autotuner``. Wraps in ``autotune()`` when
        the autotuner is enabled so op-level (M,N,K) caches also get primed
        for these shapes. Set ``TLLM_MAMBA_MULTISEQ_WARMUP=0`` to disable.
        """
        if os.environ.get("TLLM_MAMBA_MULTISEQ_WARMUP", "1") != "1":
            return
        kv_cache_manager = resource_manager.get_resource_manager(
            ResourceManagerType.KV_CACHE_MANAGER
        )
        if kv_cache_manager is None or not isinstance(kv_cache_manager, MambaHybridCacheManager):
            return

        token_num_upper_bound = min(
            self._config.max_num_tokens,
            self._config.max_batch_size * (self._config.max_seq_len - 1),
        )
        curr_max_num_tokens = kv_cache_manager.get_num_available_tokens(
            token_num_upper_bound=token_num_upper_bound,
            max_num_draft_tokens=self._config.original_max_draft_len,
        )
        # Rank-local capacity, so peers can disagree. Leaving the phase alone
        # would unbalance the per-shape agreement inside it.
        if not self._agreement.agree_flag(curr_max_num_tokens >= 4):
            return

        # Cap the multi-seq warmup token count so we don't fill the KV cache
        # to the brim. The autotuner warmup that ran just before this uses
        # ``least_requests=True`` (few long sequences) which fits comfortably
        # even when ``curr_max_num_tokens`` is close to the block ceiling.
        # ``least_requests=False`` instead spreads the token budget across
        # ``batch_size`` short sequences; when each sequence's length lands
        # exactly on a block boundary AND the KV cache has
        # ``num_extra_kv_tokens`` > 0 (e.g. spec decoding cases),
        # ``add_token`` needs to allocate one extra block per sequence, which
        # ``_create_warmup_request``'s ``blocks_to_use`` estimate doesn't
        # account for. On a small KV pool (e.g. Qwen3.5 hybrid with DFlash spec
        # decoding on a single H100: 259 blocks total, ``max_num_tokens=8192``
        # nearly saturates it), that extra per-sequence block overflows the
        # pool and crashes with "Can't allocate new blocks for window size N".
        # The point of this warmup is only to trigger ``num_seqs > 1`` +
        # ``HAS_INITSTATES=True`` kernel variants — a modest token budget
        # achieves that with plenty of block headroom.
        WARMUP_TOKEN_CAP = 4096
        capped_num_tokens = min(curr_max_num_tokens, WARMUP_TOKEN_CAP)

        logger.info("Running Mamba hybrid warmup (multi-seq + HAS_INITSTATES=True)...")

        # A model whose prefill kernel specializes on chunk alignment declares
        # it on its Mamba metadata class; the two passes then cover both
        # variants, for free since alignment is independent of HAS_INITSTATES.
        # Resolved the same way the runtime does, so warmup can't prime the
        # alignment variant of a class the runtime never instantiates.
        metadata_cls = resolve_mamba_metadata_cls(self.model)
        chunk_alignment = metadata_cls.prefill_chunk_alignment

        # (num_tokens, num_gen_requests, least_requests, force_initstates,
        #  chunk_aligned)
        mamba_warmup_shapes = [
            (capped_num_tokens, 0, False, False, False),
            (capped_num_tokens, 0, False, True, True),
        ]

        autotuner_enabled = self._config.enable_autotuner
        cache_path = os.environ.get("TLLM_AUTOTUNER_CACHE_PATH", None)
        autotune_ctx = (
            autotune(cache_path=cache_path) if autotuner_enabled else contextlib.nullcontext()
        )

        with self._graphs.no_cuda_graph(), autotune_ctx:
            for (
                num_tokens_i,
                num_gen_requests_i,
                least_req_i,
                force_init_i,
                chunk_aligned_i,
            ) in mamba_warmup_shapes:
                init_ctx = (
                    Mamba2Metadata.force_initial_states_for_warmup()
                    if force_init_i
                    else contextlib.nullcontext()
                )
                shape = (
                    f"Mamba hybrid, num_tokens={num_tokens_i}, "
                    f"num_gen_requests={num_gen_requests_i}, "
                    f"force_initstates={force_init_i}"
                )
                with init_ctx:
                    try:
                        warmup_request = self._dummy_requests.create_warmup_request(
                            resource_manager,
                            num_tokens_i,
                            num_gen_requests_i,
                            least_requests=least_req_i,
                            chunk_alignment=chunk_alignment,
                            chunk_aligned=chunk_aligned_i,
                        )
                    except torch.OutOfMemoryError as e:
                        if self._agreement.is_distributed_forward():
                            raise
                        logger.warning(
                            f"Warmup skipped for shape ({shape}): {type(e).__name__}: {e}"
                        )
                        torch.cuda.empty_cache()
                        continue
                    except RuntimeError as e:
                        # The known KV allocation failure happens before
                        # forward and is recoverable only when no peer worker
                        # can advance independently. Any other RuntimeError is
                        # a defect, not a capacity limit, and is fatal.
                        if self._agreement.is_distributed_forward():
                            raise
                        if "Can't allocate new blocks for window size" not in str(e):
                            raise
                        logger.warning(
                            f"Warmup skipped for shape ({shape}): {type(e).__name__}: {e}"
                        )
                        torch.cuda.empty_cache()
                        continue

                    try:
                        with self._dummy_requests.release_batch_context(
                            warmup_request, resource_manager
                        ) as batch:
                            if not self._agreement.should_run_warmup_batch(
                                batch, num_tokens_i, shape
                            ):
                                continue
                            with discard_nccl_window_tensor_outputs(
                                (getattr(self, "_warmup_inputs", None), batch)
                            ):
                                self._forward_warmup(
                                    batch,
                                    resource_manager,
                                    enable_spec_decode=self._config.is_spec_decode,
                                    runtime_draft_len=get_static_draft_len(
                                        self._config.spec_config
                                    ),
                                )

                            if autotuner_enabled:
                                AutoTuner.get().cache_pp_recv()
                                AutoTuner.get().cache_pp_send()
                                AutoTuner.get().clean_pp_flag()

                            torch.cuda.synchronize()
                    # Once peers can enter warmup synchronization or forward,
                    # any rank-local exception can strand them in a collective.
                    except Exception as e:  # noqa: BLE001
                        if self._agreement.is_distributed_forward():
                            raise
                        # ``torch.OutOfMemoryError`` is a ``RuntimeError``
                        # subclass; anything outside that hierarchy is a defect
                        # rather than a capacity limit.
                        if not isinstance(e, RuntimeError):
                            raise
                        # A single-rank warmup is a pure perf optimization. If
                        # a forward shape does not fit, it can be compiled
                        # lazily on the first real request.
                        logger.warning(
                            f"Warmup skipped for shape ({shape}): {type(e).__name__}: {e}"
                        )
                        # An OOM between dispatch() and combine() leaves the
                        # local MoE A2A state in ``dispatched``.
                        self._reset_moe_alltoall_state()
                        torch.cuda.empty_cache()

        clear_memory_buffers()
        torch.cuda.empty_cache()

    def _prewarm_engram_kernels(
        self, kv_cache_manager: KVCacheManager | KVCacheManagerV2 | None
    ) -> None:
        """Compile Engram kernels before conditional model warmup."""
        from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import (
            CSA2CacheManager,
        )

        if not isinstance(kv_cache_manager, CSA2CacheManager):
            return

        from tensorrt_llm._torch.models.modeling_deepseekv41 import DeepseekV41Engram

        with torch.inference_mode():
            for module in self.model.modules():
                if isinstance(module, DeepseekV41Engram):
                    module.warmup_kernels()
        torch.cuda.synchronize()
        # Idle attention-DP ranks also finish JIT before collective forwards.
        if self.mapping.tp_size > 1 and self.dist is not None:
            self.dist.tp_allgather(1)
        logger.info("Engram kernel prewarm complete")

    def _prewarm_csa2_sparse_indexer(
        self, kv_cache_manager: KVCacheManager | KVCacheManagerV2 | None
    ) -> None:
        """Compile the DeepGEMM sparse MQA-logits kernels of the CSA2 candidate
        consumers before the first collective-bearing forward.

        Like `_warmup_dg_paged_mqa_logits_metadata` for DSA, this moves
        deep_gemm's JIT (nvcc/ptxas, several seconds per kernel) out of the
        warm-up forwards, where it would count against the MoE all-to-all
        completion deadline. The kernels are specialized on the candidate
        geometry only, so one compile per kernel covers every layer and batch.
        Best-effort: a failure here only defers the compile to the first use.
        """
        from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import (
            CSA2CacheManager,
        )

        if not isinstance(kv_cache_manager, CSA2CacheManager):
            return

        from tensorrt_llm._torch.attention.backends.sparse.csa2.indexer import (
            prewarm_sparse_candidate_kernels,
        )

        config = self.model.model_config.pretrained_config
        with torch.inference_mode():
            try:
                compiled = prewarm_sparse_candidate_kernels(
                    kv_cache_manager.layout,
                    config.index_n_heads,
                    config.index_head_dim,
                    torch.device("cuda"),
                )
            except RuntimeError as e:
                logger.warning(f"CSA2 sparse indexer prewarm skipped. {type(e).__name__}: {e}")
                compiled = False
        torch.cuda.synchronize()
        if self.mapping.tp_size > 1 and self.dist is not None:
            self.dist.tp_allgather(1)
        if compiled:
            logger.info("CSA2 sparse indexer prewarm complete")

    def _prewarm_minimax_m3_msa_indexer(
        self, kv_cache_manager: KVCacheManager | KVCacheManagerV2 | None
    ) -> None:
        """Compile the MSA indexer's KV-split proxy variants, which no warmup forward plans."""
        from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager import (
            MiniMaxM3KVCacheManagerV2,
        )

        if not isinstance(kv_cache_manager, MiniMaxM3KVCacheManagerV2):
            return
        sparse_config = self.sparse_attention_config
        if (
            not isinstance(sparse_config, MiniMaxM3SparseAttentionConfig)
            or sparse_config.implementation != "msa"
        ):
            return

        from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_indexer import (
            prewarm_split_proxy_variants,
        )

        params = sparse_config.to_sparse_metadata_params(
            pretrained_config=self.model.model_config.pretrained_config
        )
        index_dtype = (
            torch.float8_e4m3fn if kv_cache_manager.indexer_kv_dtype == "fp8" else torch.bfloat16
        )
        with torch.inference_mode():
            prewarm_split_proxy_variants(
                index_dtype,
                params.sharded_index_head_count(self.mapping),
                kv_cache_manager.tokens_per_block,
            )
        torch.cuda.synchronize()
        if self.mapping.tp_size > 1 and self.dist is not None:
            self.dist.tp_allgather(1)
        logger.info("MiniMax-M3 MSA indexer prewarm complete")
