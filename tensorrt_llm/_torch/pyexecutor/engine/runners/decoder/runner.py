# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decoder execution moved out of ``ModelEngine``."""

import contextlib
import inspect
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionMetadata,
    AttentionRuntimeFeatures,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.distributed import Distributed
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import (
    MoeLoadBalancer,
    MoeLoadBalancerIterContext,
)
from tensorrt_llm._torch.moe.workspace import CutlassWorkspaceReclaimer
from tensorrt_llm._torch.peft.lora.config import LoraConfig
from tensorrt_llm._torch.peft.lora.manager import LoraModelConfig
from tensorrt_llm._torch.pyexecutor import h2d_cache, step_inputs
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph_runner import BreakableCUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import (
    CUDAGraphRunner,
    CUDAGraphRunnerConfig,
    DSparkCUDAGraphRunner,
    KeyType,
)
from tensorrt_llm._torch.pyexecutor.guided_decoder import CapturableGuidedDecoder
from tensorrt_llm._torch.pyexecutor.h2d_cache import H2DSkipCache
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType
from tensorrt_llm._torch.pyexecutor.runtime_hooks import (
    DEFAULT_RUNTIME_HOOKS,
    RuntimeHooks,
    runtime_hooks_of,
)
from tensorrt_llm._torch.pyexecutor.sampler.beam_search import PendingBeamHistories
from tensorrt_llm._torch.pyexecutor.sampler.sampler import (
    GREEDY_HIDDEN_STATES_KEY,
    GREEDY_TOKENS_KEY,
    greedy_tokens_usable,
)
from tensorrt_llm._torch.pyexecutor.sampler.sampler_common import SampleType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.step_inputs import StepInputPack
from tensorrt_llm._torch.pyexecutor.warmup_timer import _WarmupTimer
from tensorrt_llm._torch.pyexecutor.workspace import EagerWorkspaceReclaimer
from tensorrt_llm._torch.speculative import SpecMetadata, get_draft_kv_cache_manager
from tensorrt_llm._torch.speculative.dspark import DSparkSpecMetadata
from tensorrt_llm._torch.utils import (
    get_per_request_prefill_cuda_graph_flag,
    with_model_extra_attrs,
)
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.llmapi.llm_args import (
    BaseSparseAttentionConfig,
    MiniMaxM3SparseAttentionConfig,
    PrefillCudaGraphBackend,
    SeqLenAwareSparseAttentionConfig,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ...lora import LoraParamBuilder, make_cuda_graph_lora_manager
from ...model_call import ModelCaller
from ...multimodal import MultimodalItemScheduler
from ..common import make_scheduled_inputs
from ..interface import ScheduledInputs, ScheduledModelRunner
from .config import DecoderRunnerConfig
from .dummy_requests import DummyRequestFactory
from .graphs import DecoderCudaGraphs, compute_dynamic_draft_len_mapping
from .inputs import DecoderInputBuffers, DecoderInputPreparer
from .metadata import (
    DecoderMetadata,
    get_spec_managers,
    sync_group_all_greedy_sample,
    update_spec_metadata,
)
from .warmup import DecoderWarmup, WarmupAgreement


class PrefillCompiledModel(torch.nn.Module):
    """Share weights between eligible prefill/mixed and original eager paths.

    The prefill flag includes the all-rank attention-DP decision and capture
    ceiling. A decode-only rank must still compile when another rank prefills.
    """

    def __init__(self, eager_model: torch.nn.Module, compiled_model: torch.nn.Module) -> None:
        """Keep eager and compiled entry points sharing the same model weights."""
        super().__init__()
        self.eager_model = eager_model
        self._bypass_active = False
        # The compiled callable references the same weights. Register only the
        # eager tree so state_dict(), children() and _apply() visit it once.
        object.__setattr__(self, "compiled_model", compiled_model)

    def named_modules(
        self,
        memo: set[torch.nn.Module] | None = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, torch.nn.Module]]:
        """Expose checkpoint-compatible module names for partial weight reloads."""
        # Weight reloads match checkpoint prefixes against this traversal.
        # Hide the eager_model prefix even with remove_duplicate=False.
        yield from self.eager_model.named_modules(memo, prefix, remove_duplicate)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Use the compiled path only for globally eligible prefill batches."""
        model = self.compiled_model if self.use_compiled_forward() else self.eager_model
        return model(*args, **kwargs)

    def use_compiled_forward(self) -> bool:
        """Return whether the current forward should use torch.compile."""
        return not self._bypass_active and get_per_request_prefill_cuda_graph_flag()

    @contextmanager
    def bypass(self) -> Iterator[None]:
        """Temporarily force eager execution regardless of graph eligibility."""
        previous = self._bypass_active
        self._bypass_active = True
        try:
            yield
        finally:
            self._bypass_active = previous

    def __getattr__(self, name: str) -> Any:
        """Delegate model-specific attributes to the original eager model."""
        # Epilogues can access transformer attributes after forward returns.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("eager_model"), name)


class DecoderRunner(ScheduledModelRunner):
    """Run scheduled decoder generation with its own buffers, graphs and metadata."""

    # Step-loop behaviour the model opted into (RuntimeHooks); all off unless it declares otherwise.
    runtime_hooks: RuntimeHooks = DEFAULT_RUNTIME_HOOKS
    # This engine's skip cache and step-input pack (None: the hook is off), active only during its own forward.
    _h2d_cache: H2DSkipCache | None = None
    _step_pack: StepInputPack | None = None
    # Whether a batch may keep the greedy tokens a forward sampled in place of the logits. False when the executor
    # runs a guided decoder, which reads the logits of every batch.
    keep_greedy_only_outputs = True

    # Optional execution paths that a subclass can turn off.
    _context_warmups_supported = True
    _context_graph_promotion_supported = True
    _steady_gen_cache_supported = True
    _eager_workspace_reclaim_supported = True
    _deferred_decode_outputs_supported = True

    def __init__(
        self,
        model: torch.nn.Module,
        config: DecoderRunnerConfig,
        *,
        model_caller: ModelCaller,
        mapping: Mapping,
        dist: Distributed | None,
        moe_load_balancer: MoeLoadBalancer | None,
        sparse_attention_config: BaseSparseAttentionConfig | None,
        attention_runtime_features: AttentionRuntimeFeatures,
        get_runtime_tokens_per_gen_step: Callable[[int], int],
        warmup_timer: _WarmupTimer,
        iter_states: dict[str, Any],
        metrics: dict[str, float],
        mm_item_scheduler: MultimodalItemScheduler | None = None,
    ) -> None:
        self._model = model
        if config.compile_only_piecewise_graphs:
            assert isinstance(model, DecoderModelForCausalLM), (
                "compile_only_piecewise_graphs requires DecoderModelForCausalLM"
            )
            assert isinstance(model.model, PrefillCompiledModel), (
                "compile_only_piecewise_graphs requires a PrefillCompiledModel decoder body"
            )
        self._model_caller = model_caller
        self._mapping = mapping
        self._dist = dist
        self._agreement = WarmupAgreement(dist, mapping)
        self._moe_load_balancer = moe_load_balancer
        self._sparse_attention_config = sparse_attention_config
        self._attention_runtime_features = attention_runtime_features
        self._get_runtime_tokens_per_gen_step = get_runtime_tokens_per_gen_step
        self._warmup_timer = warmup_timer
        self._iter_states = iter_states
        self._metrics = metrics

        self._config = config
        self.runtime_hooks = runtime_hooks_of(model)
        self._h2d_cache = H2DSkipCache() if self.runtime_hooks.skip_unchanged_h2d else None
        self._step_pack = (
            StepInputPack(torch.device("cuda", torch.cuda.current_device()))
            if self.runtime_hooks.pack_step_inputs
            else None
        )

        self.forward_pass_callable = None
        # Optional in-graph sampling hook, registered by PyExecutor. Called at
        # the tail of _forward_step -- i.e. right after the LM head, and inside
        # capture_forward_fn -- so that for graph-capturable sampling tiers the
        # sampling lands at the end of the captured forward graph instead of
        # costing a separate launch. Left None when the sampler does not opt in.
        self.sample_in_graph_callable = None
        # Starts rebuilding each live beam's generated tokens for beam-search
        # requests, registered by PyExecutor from the sampler. Logits
        # post-processors read the result instead of the per-slot token lists
        # (see _start_beam_histories). Left None when the sampler has none.
        self.start_beam_histories_callable = None
        # Stages (or clears) in-graph sampling state once the batch is settled.
        self._stage_in_graph_sampling = None
        # Set by warmup: whether a forward is the TRTLLM-Gen FMHA JIT warmup,
        # and the eager workspace reclaimer that serving forwards run under.
        self._trtllm_gen_jit_warmup = False
        self._eager_workspace_reclaimer: EagerWorkspaceReclaimer | None = None
        self._moe_workspace_reclaimer: CutlassWorkspaceReclaimer | None = None

        self._buffers = DecoderInputBuffers.for_config(self._config, self._use_beam_search)
        if self._config.is_spec_decode and self.runtime_hooks.spec_step_fold is not None:
            self.runtime_hooks.spec_step_fold.bind_engine_buffers(
                self._buffers.input_ids_cuda,
                self._buffers.draft_tokens_cuda,
                self._buffers.previous_pos_id_offsets_cuda,
                self._buffers.previous_kv_lens_offsets_cuda,
                self._config.max_batch_size,
            )
        self._metadata = DecoderMetadata(
            config=self._config,
            mapping=mapping,
            model=model,
            buffers=self._buffers,
            attention_runtime_features=attention_runtime_features,
        )
        self._dummy_requests = self._make_dummy_requests()
        self._init_decoder_state()
        self._inputs = self._make_input_preparer(mm_item_scheduler)
        self._inputs.decode_prologue = self.runtime_hooks.decode_prologue
        self._graphs = self._make_cuda_graphs()
        self._inputs.attach_cuda_graph_runners(
            self._graphs.cuda_graph_runner, self._graphs.breakable_cuda_graph_runner
        )
        self._initialize_moe_workspace_reclaimer()
        self._warmup = DecoderWarmup(
            config=self._config,
            mapping=mapping,
            dist=dist,
            model=model,
            sparse_attention_config=sparse_attention_config,
            warmup_timer=warmup_timer,
            metrics=metrics,
            agreement=self._agreement,
            metadata=self._metadata,
            dummy_requests=self._dummy_requests,
            graphs=self._graphs,
            inputs=self._inputs,
            forward_warmup=self._forward_warmup,
            torch_compile_bypass=self._bypass_torch_compile,
            context_warmups_supported=self._context_warmups_supported,
            eager_workspace_reclaimable=self._eager_workspace_reclaim_supported,
            warmup_inputs=self._buffers.input_ids_cuda,
        )

    def _make_dummy_requests(self) -> DummyRequestFactory:
        """Build the factory of warmup and capture dummy requests."""
        return DummyRequestFactory(**self._dummy_requests_kwargs())

    def _make_input_preparer(
        self, mm_item_scheduler: MultimodalItemScheduler | None
    ) -> DecoderInputPreparer:
        """Build the input preparer over this runner's buffers."""
        return DecoderInputPreparer(**self._input_preparer_kwargs(mm_item_scheduler))

    def _make_cuda_graphs(self) -> DecoderCudaGraphs:
        """Build the CUDA graph component; call after the input preparer."""
        return DecoderCudaGraphs(**self._cuda_graphs_kwargs(), inputs=self._inputs)

    def _dummy_requests_kwargs(self) -> dict[str, Any]:
        return dict(
            config=self._config,
            model=self._model,
            get_runtime_tokens_per_gen_step=self._get_runtime_tokens_per_gen_step,
            release_request_state=getattr(self._model, "release_request_state", None),
        )

    def _input_preparer_kwargs(
        self, mm_item_scheduler: MultimodalItemScheduler | None
    ) -> dict[str, Any]:
        return dict(
            buffers=self._buffers,
            config=self._config,
            mapping=self._mapping,
            dist=self._dist,
            model=self._model,
            get_runtime_tokens_per_gen_step=self._get_runtime_tokens_per_gen_step,
            iter_states=self._iter_states,
            lora=self._lora,
            use_beam_search=self._use_beam_search,
            steady_gen_cache_supported=self._steady_gen_cache_supported,
            mm_item_scheduler=mm_item_scheduler,
        )

    def _cuda_graphs_kwargs(self) -> dict[str, Any]:
        cuda_graph_runner = self._initialize_cuda_graph_runner()
        return dict(
            cuda_graph_runner=cuda_graph_runner,
            breakable_cuda_graph_runner=self._initialize_breakable_cuda_graph_runner(),
            config=self._config,
            mapping=self._mapping,
            model=self._model,
            sparse_attention_config=self._sparse_attention_config,
            metrics=self._metrics,
            dummy_requests=self._dummy_requests,
            agreement=self._agreement,
            metadata=self._metadata,
            get_runtime_tokens_per_gen_step=self._get_runtime_tokens_per_gen_step,
            dynamic_draft_len_mapping=self._dynamic_draft_len_mapping,
            lora=self._lora,
            forward_warmup=self._forward_warmup,
            torch_compile_bypass=self._bypass_torch_compile,
            context_promotion_supported=self._context_graph_promotion_supported,
            warmup_inputs=self._buffers.input_ids_cuda,
        )

    @property
    def attn_metadata(self) -> AttentionMetadata | None:
        return self._metadata.attn_metadata

    @attn_metadata.setter
    def attn_metadata(self, value: AttentionMetadata | None) -> None:
        self._metadata.attn_metadata = value

    @property
    def spec_metadata(self) -> SpecMetadata | None:
        return self._metadata.spec_metadata

    @spec_metadata.setter
    def spec_metadata(self, value: SpecMetadata | None) -> None:
        self._metadata.spec_metadata = value

    @property
    def guided_decoder(self) -> CapturableGuidedDecoder | None:
        return self._inputs.guided_decoder

    @guided_decoder.setter
    def guided_decoder(self, value: CapturableGuidedDecoder | None) -> None:
        self._inputs.guided_decoder = value

    @property
    def cuda_graph_runner(self) -> CUDAGraphRunner:
        return self._graphs.cuda_graph_runner

    @torch.inference_mode()
    @with_model_extra_attrs(lambda self: self._model.extra_attrs)
    def forward(
        self,
        inputs: ScheduledInputs,
        *,
        resource_manager: ResourceManager,
        is_dummy: bool = False,
    ) -> Any:
        with h2d_cache.active(self._h2d_cache), step_inputs.active(self._step_pack):
            try:
                return self._forward_decoder(inputs, resource_manager, is_dummy=is_dummy)
            except BaseException:
                # A failed step issues none of its packed copies, so forget the shadows recorded for them too.
                step_inputs.discard()
                if self._h2d_cache is not None:
                    self._h2d_cache.clear()
                raise

    def _initialize_moe_workspace_reclaimer(self) -> None:
        """Create the native workspace owner for this runner's lifetime."""
        self._moe_workspace_reclaimer = None
        if (
            os.environ.get("TRTLLM_RECLAIM_WORKSPACE", "1") != "0"
            and not self._config.is_spec_decode
            and self._mapping.cp_size == 1
            and self._eager_workspace_reclaim_supported
            and not self._config.torch_compile_enabled
            and self._graphs.breakable_cuda_graph_runner is None
        ):
            self._moe_workspace_reclaimer = CutlassWorkspaceReclaimer()

    def warmup(self, resource_manager: ResourceManager) -> None:
        self._eager_workspace_reclaimer = None
        self._eager_workspace_reclaimer = self._warmup.run(resource_manager)

    def release_graphs(self, *, release_nccl_window_owners: bool = True) -> None:
        self._graphs.release(release_nccl_window_owners=release_nccl_window_owners)

    def wait_for_input_copy(self) -> None:
        self._inputs.wait_for_input_copy()

    def register_sample_type_resolver(
        self, resolver: Callable | None, stage: Callable | None = None
    ) -> None:
        """Register how a batch maps to its sampling tier, for the graph key."""
        self._graphs.register_sample_type_resolver(resolver)
        self._stage_in_graph_sampling = stage

    def init_cuda_graph_lora_manager(
        self, lora_config: LoraConfig, lora_model: torch.nn.Module
    ) -> None:
        """Build LoRA preparation state for the current executor resources."""
        cuda_graph_manager = None
        if self._graphs.enabled:
            # For spec decode, each generation request contributes
            # max_draft_len + 1 tokens per forward pass.
            max_tokens_per_seq = (
                (self._config.original_max_draft_len + 1) if self._config.is_spec_decode else 1
            )
            cuda_graph_manager = make_cuda_graph_lora_manager(
                lora_model,
                lora_config,
                self.lora_model_config,
                self._config.max_batch_size,
                max_tokens_per_seq,
                self._config.max_num_tokens,
            )
        # Resource creation runs again after capacity estimation, once the old
        # executor has released its graphs, and replaces the graph manager.
        self._lora.cuda_graph_manager = cuda_graph_manager

    @property
    def lora_model_config(self) -> LoraModelConfig | None:
        return self._lora.model_config

    @lora_model_config.setter
    def lora_model_config(self, lora_model_config: LoraModelConfig | None) -> None:
        self._lora.model_config = lora_model_config

    def _init_decoder_state(self) -> None:
        """Initialize the mutable state read only by engine decoder execution."""
        self._lora = LoraParamBuilder(
            spec_config=self._config.spec_config,
            attn_backend=self._config.attention_backend,
            cuda_graph_manager=None,
        )
        self._dynamic_draft_len_mapping = compute_dynamic_draft_len_mapping(self._config)

    def _initialize_cuda_graph_runner(self) -> CUDAGraphRunner:
        runner_type = DSparkCUDAGraphRunner if self._dspark_confidence_enabled else CUDAGraphRunner
        cuda_graph_runner = runner_type(self._cuda_graph_runner_config())
        register_hooks = getattr(self._model, "register_cuda_graph_pre_replay_hooks", None)
        if register_hooks is not None:
            register_hooks(cuda_graph_runner)
        return cuda_graph_runner

    def _cuda_graph_runner_config(self) -> CUDAGraphRunnerConfig:
        return CUDAGraphRunnerConfig(
            use_cuda_graph=self._config.cuda_graph_config is not None,
            cuda_graph_padding_enabled=self._config.cuda_graph_padding_enabled,
            cuda_graph_batch_sizes=self._config.cuda_graph_batch_sizes,
            max_cuda_graph_batch_size=self._config.max_cuda_graph_batch_size,
            max_beam_width=self._config.max_beam_width,
            spec_config=self._config.spec_config,
            dynamic_draft_len_mapping=self._dynamic_draft_len_mapping,
            max_num_tokens=self._config.max_num_tokens,
            use_mrope=self._config.use_mrope,
            original_max_total_draft_tokens=self._config.original_max_total_draft_tokens,
            enable_attention_dp=self._config.enable_attention_dp,
            is_encoder_decoder=False,
            batch_size=self._config.max_batch_size,
            mapping=self._mapping,
            dist=self._dist,
            sparse_attention_config=self._sparse_attention_config,
            short_long_seq_len_threshold=self._short_long_seq_len_threshold(),
            private_metadata_buffers=self.runtime_hooks.skips_unchanged_metadata_uploads,
            enable_in_graph_sampling=self._config.enable_in_graph_sampling,
            # Speculative steps rewrite every input token and position, so
            # captured graphs can read the persistent buffers directly.
            static_input_ids=(
                self._buffers.input_ids_cuda
                if self._config.is_spec_decode and not self._config.use_mrope
                else None
            ),
            static_position_ids=(
                self._buffers.position_ids_cuda
                if self._config.is_spec_decode and not self._config.use_mrope
                else None
            ),
        )

    def _short_long_seq_len_threshold(self) -> int | None:
        """The total sequence length that splits the short and long decode graph families, or None for one family.

        Set by a sparse attention config that asks for the split, or by a model whose MSA
        policy stages identity selections for short sequences: a sequence within the
        top-k blocks attends every valid block.
        """
        sparse_config = self._sparse_attention_config
        if (
            isinstance(sparse_config, SeqLenAwareSparseAttentionConfig)
            and sparse_config.needs_separate_short_long_cuda_graphs()
        ):
            return sparse_config.seq_len_threshold
        if self.runtime_hooks.msa.short_sequence_identity:
            if not isinstance(sparse_config, MiniMaxM3SparseAttentionConfig):
                raise ValueError(
                    "MsaRuntimePolicy.short_sequence_identity requires a MiniMax-M3 sparse "
                    f"attention config, got {type(sparse_config).__name__}"
                )
            return sparse_config.sparse_topk_blocks * sparse_config.sparse_block_size
        return None

    def _initialize_breakable_cuda_graph_runner(self) -> BreakableCUDAGraphRunner | None:
        if self._config.prefill_cuda_graph_backend != PrefillCudaGraphBackend.BREAKABLE:
            return None

        decoder_model = (
            self._model
            if isinstance(self._model, DecoderModelForCausalLM)
            else getattr(self._model, "llm", None)
        )
        if not isinstance(decoder_model, DecoderModelForCausalLM):
            raise ValueError("breakable prefill CUDA graph requires a decoder model body")
        logger.info(
            f"Breakable prefill CUDA graphs on {type(decoder_model.model).__name__} "
            f"for num_tokens={self._config.prefill_cuda_graph_num_tokens}"
        )
        return BreakableCUDAGraphRunner(decoder_model.model)

    @property
    def _use_beam_search(self) -> bool:
        return self._config.max_beam_width > 1

    def _forward_warmup(
        self,
        batch: ScheduledRequests,
        resource_manager: ResourceManager,
        *,
        enable_spec_decode: bool,
        runtime_draft_len: int,
        trtllm_gen_jit_warmup: bool = False,
    ) -> Any:
        """Run a dummy scheduled pass with this call's speculation values."""
        inputs = make_scheduled_inputs(
            batch,
            None,
            None,
            enable_spec_decode=enable_spec_decode,
            runtime_draft_len=runtime_draft_len,
        )
        previous = self._trtllm_gen_jit_warmup
        self._trtllm_gen_jit_warmup = trtllm_gen_jit_warmup
        try:
            return self.forward(inputs, resource_manager=resource_manager, is_dummy=True)
        finally:
            self._trtllm_gen_jit_warmup = previous

    def _forward_decoder(
        self,
        forward_inputs: ScheduledInputs,
        resource_manager: ResourceManager,
        *,
        is_dummy: bool,
    ) -> Any:
        scheduled_requests = forward_inputs.batch
        new_tensors_device = forward_inputs.new_tensors_device
        cache_indirection_buffer = forward_inputs.cache_indirection_buffer
        gather_context_logits = forward_inputs.gather_context_logits
        enable_spec_decode = forward_inputs.enable_spec_decode
        runtime_draft_len = forward_inputs.runtime_draft_len
        kv_cache_manager = resource_manager.get_resource_manager(
            ResourceManagerType.KV_CACHE_MANAGER
        )
        assert kv_cache_manager is not None, "the legacy runner requires a KV cache manager"
        semantic_requests = scheduled_requests
        scheduled_requests = self._graphs.attention_dp_execution_batch(scheduled_requests)
        draft_kv_cache_manager = get_draft_kv_cache_manager(
            self._config.spec_config, resource_manager
        )

        attn_metadata = self._metadata.setup_attention(kv_cache_manager, draft_kv_cache_manager)
        if isinstance(attn_metadata, TrtllmAttentionMetadata):
            attn_metadata.trtllm_gen_jit_warmup = self._trtllm_gen_jit_warmup
        if enable_spec_decode:
            spec_resource_manager, spec_tree_manager = get_spec_managers(resource_manager)
            spec_metadata = self._metadata.setup_speculative(spec_resource_manager)
            assert spec_metadata is not None
            update_spec_metadata(
                spec_metadata,
                self._config,
                scheduled_requests,
                attn_metadata,
                spec_tree_manager=spec_tree_manager,
                runtime_draft_len=runtime_draft_len,
            )
        else:
            spec_resource_manager = None
            spec_metadata = None

        moe_load_balancer = self._moe_load_balancer
        from tensorrt_llm._torch.pyexecutor.ced_replay import uses_private_decoder_cache

        ced_enabled = uses_private_decoder_cache(
            getattr(self._model, "model", None), kv_cache_manager
        )
        graph_requests, promoted_context_request_ids = self._graphs.promote_final_contexts(
            scheduled_requests,
            kv_cache_manager,
            enable_spec_decode=enable_spec_decode,
            runtime_draft_len=runtime_draft_len,
            ced_enabled=ced_enabled,
        )

        with self._graphs.pad_batch(
            graph_requests, resource_manager, runtime_draft_len
        ) as padded_graph_requests:
            # Callee already no-ops when use_mrope=False, but the Python call /
            # frame setup itself is non-trivial under high concurrency. Gating
            # at the caller avoids that overhead for non-mrope models.
            if self._config.use_mrope:
                self._inputs.pad_batch_seed_mrope_delta_cache(padded_graph_requests)

            # Refresh is_all_greedy_sample for the *current* batch BEFORE the
            # CUDA graph key is built below. The key includes this flag to pick
            # the argmax vs advanced-sampling graph variant; populate (inside
            # _prepare_inputs) runs later and fills the matching GPU buffers.
            # Without this pre-scan the key would use the previous iteration's
            # stale value and could replay the advanced graph against
            # unpopulated (greedy) buffers, hanging the run (e.g. MTP nextn>=2).
            if spec_metadata is not None:
                spec_metadata.update_is_all_greedy_sample(padded_graph_requests.all_requests())
                sync_group_all_greedy_sample(spec_metadata, self._mapping, self._dist)

            selection = self._graphs.select(
                padded_graph_requests,
                resource_manager,
                enable_spec_decode=enable_spec_decode,
                attn_metadata=attn_metadata,
                spec_metadata=spec_metadata,
                draft_tokens_cuda=self._buffers.draft_tokens_cuda
                if self._config.is_spec_decode
                else None,
                new_tensors_device=new_tensors_device,
                spec_resource_manager=spec_resource_manager,
                promoted_context_request_ids=promoted_context_request_ids,
            )
            key = selection.key
            use_lora_graph = selection.use_lora_graph

            can_run_graph = key is not None and not (
                ced_enabled and scheduled_requests.num_context_requests > 0
            )
            if can_run_graph:
                attn_metadata = selection.attn_metadata
                spec_metadata = selection.spec_metadata
                execution_requests = padded_graph_requests
                execution_promoted_context_ids = promoted_context_request_ids
            else:
                attn_metadata = self._metadata.attn_metadata
                if enable_spec_decode:
                    spec_metadata = self._metadata.spec_metadata
                else:
                    spec_metadata = None
                execution_requests = scheduled_requests
                execution_promoted_context_ids = frozenset()

            # Stage in-graph sampling now that the batch is settled: the staged
            # scatter width has to match the batch the forward actually runs on,
            # which differs between the graph (padded) and eager (unpadded)
            # branches above. Falling back to eager also means no graph replays,
            # so the tier is dropped and the sampler runs after the forward.
            #
            # Promoted one-token contexts join the graph batch's generation
            # requests, but sample_async is handed the original scheduled batch,
            # which excludes them. Staging a tier here would sample rows that
            # are then discarded and advance those requests' Philox streams an
            # extra time, so keep those steps on the eager path.
            if self._stage_in_graph_sampling is not None:
                staged_sample_type = (
                    key.sample_type
                    if can_run_graph and not execution_promoted_context_ids
                    else SampleType.FULL
                )
                self._stage_in_graph_sampling(execution_requests, staged_sample_type)

            # Fill slot-ID buffer for scatter inside draft loop
            if enable_spec_decode and spec_tree_manager is not None:
                spec_tree_manager.slot_storage.fill_all_slot_ids(
                    execution_requests.context_requests,
                    execution_requests.generation_requests,
                )

            # Pack the step's small input copies into one upload and one scatter at the
            # graph replay. Only a replay of an already captured graph, which always
            # reaches the flush, may open a pack.
            if (
                self.runtime_hooks.pack_step_inputs
                and can_run_graph
                and not self.cuda_graph_runner.is_warmup_only
                and not self.cuda_graph_runner.needs_capture(key)
            ):
                step_inputs.begin()
            with self._cuda_graph_metadata_prepare_scope(
                attn_metadata,
                key,
                execution_requests,
                execution_promoted_context_ids,
                enable_spec_decode=enable_spec_decode,
            ):
                prepared = self._inputs.prepare_inputs(
                    execution_requests,
                    kv_cache_manager,
                    attn_metadata,
                    spec_metadata,
                    new_tensors_device,
                    cache_indirection_buffer,
                    resource_manager,
                    can_run_graph,
                    execution_promoted_context_ids,
                    use_lora_graph=use_lora_graph,
                    enable_spec_decode=enable_spec_decode,
                    runtime_draft_len=runtime_draft_len,
                    is_dummy=is_dummy,
                    gather_context_logits=gather_context_logits,
                )
            inputs = prepared.kwargs
            runtime_draft_len = prepared.runtime_draft_len
            if execution_promoted_context_ids:
                self._inputs.record_promoted_iter_states(scheduled_requests)
            self._inputs.record_input_copy()

            if (
                can_run_graph
                and not is_dummy
                and getattr(self._inputs, "_dspark_device_budget", None) is not None
            ):
                self._inputs._apply_device_window_prologue(inputs, new_tensors_device)

            def run_model(model_inputs: dict[str, Any]) -> Any:
                with MoeLoadBalancerIterContext(moe_load_balancer):
                    return self._forward_step(
                        model_inputs,
                        enable_spec_decode=enable_spec_decode,
                        runtime_draft_len=runtime_draft_len,
                        is_dummy=is_dummy,
                        gather_context_logits=gather_context_logits,
                    )

            def postprocess_inputs(model_inputs: dict[str, Any]) -> None:
                self._inputs.postprocess_inputs(
                    model_inputs,
                    enable_spec_decode=enable_spec_decode,
                    runtime_draft_len=runtime_draft_len,
                )

            pending_beam_histories = self._start_beam_histories(semantic_requests)

            graph_key = key if can_run_graph else None
            captures_model_body = self._graphs.captures_model_body(graph_key)
            outputs = self._graphs.run(
                graph_key,
                inputs,
                run_model,
                postprocess_inputs,
                attn_metadata=attn_metadata,
                draft_kv_cache_manager=draft_kv_cache_manager,
                moe_load_balancer=moe_load_balancer,
                enable_spec_decode=enable_spec_decode,
                gather_context_logits=gather_context_logits,
                static_inputs_staged=prepared.static_inputs_staged,
                position_id_offsets=(
                    self._buffers.previous_pos_id_offsets_cuda
                    if enable_spec_decode
                    and not self._config.disable_overlap_scheduler
                    and attn_metadata.kv_cache_manager is not None
                    else None
                ),
            )
            # No-op unless a pack was opened above and the replay did not issue it.
            step_inputs.flush()
            if captures_model_body:
                return outputs

            if self.forward_pass_callable is not None:
                self.forward_pass_callable()

            if isinstance(outputs, dict) and outputs.get(GREEDY_TOKENS_KEY) is not None:
                outputs = self._logits_for_greedy_tokens(semantic_requests, outputs)
            self._execute_logit_post_processors(semantic_requests, outputs, pending_beam_histories)

            if not isinstance(outputs, dict):
                return outputs
            outputs = self._restore_attention_dp_semantic_outputs(semantic_requests, outputs)
            return {**outputs, "runtime_draft_len": runtime_draft_len}

    @contextmanager
    def _maybe_bypass_torch_compile(self, bypass: bool = False) -> Iterator[None]:
        """Force eager execution for auxiliary work and ordinary CUDA graphs."""
        if not self._config.compile_only_piecewise_graphs or not bypass:
            yield
            return
        with self._model.model.bypass():
            yield

    def _bypass_torch_compile(self) -> contextlib.AbstractContextManager[None]:
        """Run auxiliary warmups and ordinary CUDA graph capture eagerly."""
        return self._maybe_bypass_torch_compile(bypass=True)

    def _model_forward(self, *, is_dummy: bool, **kwargs: Any) -> Any:
        assert self._model_caller is not None
        reclaimer = self._eager_workspace_reclaimer
        metadata = kwargs["attn_metadata"]
        reclaim_scope = (
            reclaimer.forward(metadata)
            if reclaimer is not None
            and not is_dummy
            and isinstance(metadata, TrtllmAttentionMetadata)
            else contextlib.nullcontext()
        )
        moe_reclaimer = self._moe_workspace_reclaimer
        moe_scope = contextlib.nullcontext()
        if moe_reclaimer is not None:
            moe_scope = moe_reclaimer.forward(warmup=is_dummy, num_tokens=metadata.num_tokens)
        with reclaim_scope, moe_scope:
            return self._model_caller(**kwargs)

    @nvtx_range("_forward_step")
    def _forward_step(
        self,
        inputs: dict[str, Any],
        *,
        enable_spec_decode: bool,
        runtime_draft_len: int,
        is_dummy: bool,
        gather_context_logits: bool = False,
    ) -> dict[str, Any]:
        inputs = self._inputs.preprocess_inputs(
            inputs, enable_spec_decode=enable_spec_decode, runtime_draft_len=runtime_draft_len
        )
        spec_metadata = inputs.get("spec_metadata")
        gather_ids = spec_metadata.gather_ids if spec_metadata is not None else None

        # For simplicity, just return all the the logits if we have special gather_ids
        # from speculative decoding.
        dspark_metadata = inputs.get("spec_metadata")
        if isinstance(dspark_metadata, DSparkSpecMetadata):
            # CED retains the final context row and all verification rows.
            # Distinguish that sparse gather from a real all-context request.
            dspark_metadata._requires_all_context_logits = bool(gather_context_logits)
        try:
            outputs = self._model_forward(
                **inputs,
                is_dummy=is_dummy,
                return_context_logits=gather_ids is not None or gather_context_logits,
            )
        finally:
            if isinstance(dspark_metadata, DSparkSpecMetadata):
                # Do not let a later direct caller inherit engine-only proof
                # that dropping non-final context rows is safe.
                dspark_metadata._requires_all_context_logits = None

        if self._config.without_logits:
            return outputs

        if isinstance(outputs, dict):
            # If the model returns a dict, get the logits from it. All other keys are kept.
            logits = outputs.get("logits", None)
            # If the logits are not found, no further processing is needed.
            if logits is None:
                return outputs
        else:
            # If the model returns a single tensor, assume it is the logits and wrap it in a dict.
            logits = outputs
            outputs = {"logits": logits}

        # If we have special gather_ids, gather the logits
        if gather_ids is not None:
            outputs["logits"] = logits[gather_ids]

        # Sample at the tail of the forward pass, so that under CUDA graph
        # capture the sampling kernels are recorded as part of this graph. The
        # hook is a no-op unless the sampler staged a graph-capturable tier for
        # this batch.
        #
        # Only the last PP rank runs the LM head. The others get a placeholder
        # that is the right shape but never filled, so a shape check waves it
        # through and they would sample garbage every step -- discarded, but
        # not free. _execute_logit_post_processors skips them for this reason.
        if self.sample_in_graph_callable is not None and self._mapping.is_last_pp_rank():
            self.sample_in_graph_callable(outputs)

        return outputs

    @staticmethod
    def _apply_logits_processors(
        request, logits_processors, logits_tensor, beam_width, token_ids, logits_row_offset
    ) -> None:
        logits_rows = logits_tensor[logits_row_offset : logits_row_offset + beam_width]
        # Reshape to align w/ the shape used in the TRT backend,
        # so the same logit processors can be used across both backends.
        logits_rows = logits_rows.view(beam_width, 1, -1)
        for lp in logits_processors:
            lp_params = inspect.signature(lp).parameters

            assert 4 <= len(lp_params) <= 5, (
                "Logit post processor signature must match the `LogitsProcessor` interface "
                "defined in `tensorrtllm.sampling_params`."
            )
            lp(request.py_request_id, logits_rows, token_ids, None, None)

        # logits_rows is a view into logits_tensor (narrow + view never
        # copy), so the processors already mutated it in place. Writing it
        # back would be a self-assignment, which torch rejects for the
        # non-contiguous slices a TP-padded vocab produces.

    def _logits_for_greedy_tokens(
        self, scheduled_requests: ScheduledRequests, outputs: dict
    ) -> dict:
        """Replace greedy tokens sampled by the forward with logits when a request needs logits.

        A forward that samples greedily in place of the LM head returns the head's input
        alongside the tokens. When any generation request needs more than its greedy token,
        the logits come from that input eagerly, followed by the in-graph sampling and the
        logits post-processing the forward skipped. Returns the outputs to use: ``outputs``
        itself, or a new dict (a replayed graph's outputs dict is reused by later replays).
        """
        if (
            self.keep_greedy_only_outputs
            and self._config.max_beam_width == 1
            and scheduled_requests.num_context_requests == 0
            # An idle attention-DP rank runs a dummy row for an empty batch, whose outputs are restored from logits.
            and len(scheduled_requests.generation_requests) > 0
            and all(
                greedy_tokens_usable(request) for request in scheduled_requests.generation_requests
            )
        ):
            return outputs
        outputs = dict(outputs)
        outputs.pop(GREEDY_TOKENS_KEY)
        hidden_states = outputs.pop(GREEDY_HIDDEN_STATES_KEY)
        # One row per generation request: every row is a last token.
        outputs["logits"] = self._model.logits_processor.forward(
            hidden_states, self._model.lm_head, None, True
        )
        if self.sample_in_graph_callable is not None and self._mapping.is_last_pp_rank():
            self.sample_in_graph_callable(outputs)
        return outputs

    def _start_beam_histories(
        self, scheduled_requests: ScheduledRequests
    ) -> tuple[list[LlmRequest], PendingBeamHistories] | None:
        """Start copying the live beam histories that logits post-processors will read.

        Under beam search, request.get_tokens(beam_idx) is the list of tokens
        appended to slot beam_idx, which stops matching the hypothesis that slot
        holds as soon as the search reorders its beams; the corrected paths are
        only written back when the request finishes. Processors get each live
        beam's real history instead, gathered for all such requests in one pass.
        The gather and its copy are queued before the forward pass, so the
        processors wait for the copy rather than for the forward pass.
        """
        if self.start_beam_histories_callable is None or not self._mapping.is_last_pp_rank():
            return None
        beam_requests = [
            request
            for request in scheduled_requests.generation_requests
            if getattr(request, "py_logits_post_processors", None)
            and request.get_beam_width_by_iter(for_next_iteration=False) > 1
        ]
        if not beam_requests:
            return None
        return beam_requests, self.start_beam_histories_callable(beam_requests)

    def _execute_logit_post_processors(
        self,
        scheduled_requests: ScheduledRequests,
        outputs: dict,
        pending_beam_histories: tuple[list[LlmRequest], PendingBeamHistories] | None = None,
    ) -> None:
        """Apply logit post processors (in-place modify outputs Tensors) if any.

        pending_beam_histories is what _start_beam_histories returned for this
        step.
        """

        if not (self._mapping.is_last_pp_rank()):
            return

        if not isinstance(outputs, dict) or "logits" not in outputs:
            # TODO: support models that don't return outputs as dict
            return

        logits_tensor = outputs["logits"]
        if logits_tensor is None:
            # Greedy tokens sampled by the forward; _logits_for_greedy_tokens supplied
            # logits instead whenever a request needed them.
            return

        if outputs.get("logits_vocab_shard", False):
            # The spec worker kept the logits vocabulary-sharded (this TP
            # rank's columns). A post-processor sees the whole vocabulary, so
            # gather them first, only when a request has one. Every TP rank
            # schedules the same requests, so all of them take the gather;
            # under attention DP they would not.
            if self._mapping.enable_attention_dp:
                raise RuntimeError(
                    "vocabulary-sharded spec-worker logits need every TP rank "
                    "to schedule the same requests, which attention DP does "
                    "not"
                )
            if not any(
                getattr(request, "py_logits_post_processors", None)
                for request in scheduled_requests.all_requests()
            ):
                return
            from tensorrt_llm._torch.distributed import allgather

            logits_tensor = allgather(logits_tensor, self._mapping, dim=-1).float()
            # Drop the vocabulary's TP padding, as the LM head's own gather
            # does.
            vocab_size = getattr(getattr(self._model, "lm_head", None), "num_embeddings", None)
            if vocab_size is not None:
                logits_tensor = logits_tensor[..., :vocab_size]

        beam_histories: dict[int, list[list[int]]] = {}
        if pending_beam_histories is not None:
            beam_requests, pending = pending_beam_histories
            beam_histories = {
                id(request): histories for request, histories in zip(beam_requests, pending.wait())
            }

        logits_row_offset = 0
        request_groups = (
            (scheduled_requests.context_requests, True),
            (scheduled_requests.generation_requests, False),
        )

        for requests, is_context_request in request_groups:
            for request in requests:
                if is_context_request:
                    beam_width = 1
                    row_stride = 1
                else:
                    # Generation rows are laid out at the static admission
                    # width, so that is the stride between requests, while
                    # only the leading beam_width rows hold live beams under
                    # a variable beam width array. Advancing the offset by the
                    # narrower width would make every request after the first
                    # rewrite another request's logits rows in place.
                    beam_width = request.get_beam_width_by_iter(for_next_iteration=False)
                    row_stride = request.py_beam_width

                logits_processors = getattr(request, "py_logits_post_processors", None)
                if logits_processors:
                    if (
                        not is_context_request
                        and getattr(request, "py_verify_len", None) is not None
                    ):
                        raise RuntimeError(
                            "per-request logits post-processors are not supported with DSpark "
                            "ragged verification: the processor interface assumes one uniform "
                            "row stride and would modify another request's packed logits after "
                            "a short verify window"
                        )
                    if is_context_request:
                        token_ids = [request.get_tokens(0)]
                    elif (histories := beam_histories.get(id(request))) is not None:
                        prompt = request.get_tokens(0)[: request.py_prompt_len]
                        token_ids = [prompt + history for history in histories]
                    else:
                        token_ids = [request.get_tokens(beam_idx) for beam_idx in range(beam_width)]
                    if is_context_request and request.py_orig_prompt_len < len(token_ids[0]):
                        # Skip as we only need to apply logit processor on the last context request
                        logits_row_offset += row_stride
                        continue

                    self._apply_logits_processors(
                        request,
                        logits_processors,
                        logits_tensor,
                        beam_width,
                        token_ids,
                        logits_row_offset,
                    )
                logits_row_offset += row_stride

    def _cuda_graph_metadata_prepare_scope(
        self,
        attn_metadata: AttentionMetadata,
        key: KeyType | None,
        scheduled_requests: ScheduledRequests,
        promoted_context_request_ids: frozenset[int],
        *,
        enable_spec_decode: bool,
    ) -> contextlib.AbstractContextManager[Any]:
        """Allow derived outputs only when this forward owns a decode graph.

        Both capture/warmup and replay execute _preprocess_inputs before model
        consumers. Non-speculative overlap has the same ordering; speculative
        acceptance corrections and draft-manager swaps keep full preparation.
        """
        defer = getattr(attn_metadata, "defer_cuda_graph_decode_outputs", None)
        if (
            defer is not None
            and isinstance(key, KeyType)
            and attn_metadata.is_cuda_graph is True
            and key.num_contexts == 0
            and key.draft_len == 0
            and scheduled_requests.num_context_requests == 0
            and scheduled_requests.num_generation_requests > 0
            and not promoted_context_request_ids
            and not enable_spec_decode
            and not self._use_beam_search
            and self._deferred_decode_outputs_supported
            and self._mapping.cp_size == 1
        ):
            return defer()
        return contextlib.nullcontext()

    @staticmethod
    def _restore_attention_dp_semantic_outputs(
        scheduled_requests: ScheduledRequests, outputs: dict[str, Any]
    ) -> dict[str, Any]:
        if not scheduled_requests.is_attention_dp_phase_idle:
            return outputs
        # The sampler sees the original empty batch, never the execution row.
        # Copy the map because CUDA graph outputs are retained across replays.
        return {**outputs, "logits": outputs["logits"][:0]}

    # Scheduler-facing state stays on the input owner that stages each layout.

    @property
    def _dspark_confidence_enabled(self):
        return self._inputs._dspark_confidence_enabled

    @property
    def _dspark_trims_submitted_tokens(self):
        return self._inputs._dspark_trims_submitted_tokens

    @property
    def _dspark_sps_cost_table(self):
        return self._inputs._dspark_sps_cost_table

    @property
    def _dspark_exact_candidate_cells(self):
        return self._inputs._dspark_exact_candidate_cells

    @property
    def _dspark_exact_identity_words(self):
        return self._inputs._dspark_exact_identity_words

    @property
    def _dspark_device_budget(self):
        return getattr(self._inputs, "_dspark_device_budget", None)

    @_dspark_device_budget.setter
    def _dspark_device_budget(self, value) -> None:
        self._inputs._dspark_device_budget = value

    @property
    def previous_request_ids(self):
        return self._inputs.previous_request_ids

    @property
    def previous_verify_lens(self):
        return self._inputs.previous_verify_lens

    @property
    def has_previous_device_draft(self):
        return self._inputs.has_previous_device_draft

    def ragged_verify_token_buckets(self, padded_bs: int):
        return self._inputs.ragged_verify_token_buckets(padded_bs)

    def fit_ragged_verify_lens(self, *args, **kwargs):
        return self._inputs.fit_ragged_verify_lens(*args, **kwargs)

    def _get_spec_worker(self):
        """Return the one-model speculative worker when it is initialized."""
        return getattr(self._model, "spec_worker", None)
