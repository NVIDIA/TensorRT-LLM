# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model setup shared by every executor: process setup, loading, capacity and compilation."""

import math
import os
import traceback
from dataclasses import dataclass

import torch
import torch._dynamo.config

import tensorrt_llm.bindings.internal.userbuffers as ub
from tensorrt_llm._torch.attention.backends.interface import AttentionBackend
from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
from tensorrt_llm._torch.compilation.backend import Backend
from tensorrt_llm._torch.distributed import Distributed
from tensorrt_llm._torch.distributed.communicator import init_pp_comm
from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce
from tensorrt_llm._torch.memory.monitor.startup import region as monitor_region
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm._torch.moe.expert_statistic import ExpertStatistic
from tensorrt_llm._torch.nccl_window_tensor_scope import install_eager_nccl_window_tensor_scopes
from tensorrt_llm._torch.pyexecutor.engine.model_call import ModelCaller
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner import PrefillCompiledModel
from tensorrt_llm._torch.pyexecutor.layerwise_nvtx_marker import LayerwiseNvtxMarker
from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader
from tensorrt_llm._torch.utils import set_torch_compiling
from tensorrt_llm._utils import mpi_disabled
from tensorrt_llm.llmapi.llm_args import (
    DecodingBaseConfig,
    PrefillCudaGraphBackend,
    TorchCompileConfig,
    TorchLlmArgs,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping


def _infer_max_seq_len_from_config(model: torch.nn.Module) -> int:
    if hasattr(model, "model_config") and model.model_config:
        model_config = model.model_config.pretrained_config
        rope_scaling = getattr(model_config, "rope_scaling", None)
        rope_factor = 1
        if rope_scaling is not None:
            rope_type = rope_scaling.get("type", rope_scaling.get("rope_type"))
            if rope_type not in ("su", "longrope", "llama3", "yarn"):
                rope_factor = rope_scaling.get("factor", 1.0)

        # Step 1: Find the upper bound of max_seq_len
        inferred_max_seq_len = 2048
        max_position_embeddings = getattr(model_config, "max_position_embeddings", None)
        if max_position_embeddings is None and hasattr(model_config, "text_config"):
            max_position_embeddings = getattr(
                model_config.text_config, "max_position_embeddings", None
            )
        if max_position_embeddings is not None:
            inferred_max_seq_len = max_position_embeddings

        # Step 2: Scale max_seq_len with rotary scaling
        if rope_factor != 1:
            inferred_max_seq_len = int(math.ceil(inferred_max_seq_len * rope_factor))
            logger.warning(
                f"max_seq_len is scaled to {inferred_max_seq_len} by rope scaling {rope_factor}"
            )

        return inferred_max_seq_len

    default_max_seq_len = 8192
    logger.warning(
        f"Could not infer max_seq_len from model config, using default value: {default_max_seq_len}"
    )
    return default_max_seq_len


def infer_max_seq_len(model: torch.nn.Module) -> int:
    """The longest sequence the loaded model supports."""
    # Vision encoders may expose their own sequence-length inference.
    if hasattr(model, "infer_max_seq_len"):
        return model.infer_max_seq_len()
    return _infer_max_seq_len_from_config(model)


def resolve_max_seq_len(model: torch.nn.Module, max_seq_len: int | None) -> int:
    """Clamp the requested max_seq_len to what the loaded model supports."""
    # Allow user to override the inferred max_seq_len with a warning.
    allow_long_max_model_len = os.getenv("TLLM_ALLOW_LONG_MAX_MODEL_LEN", "0").lower() in [
        "1",
        "true",
        "yes",
        "y",
    ]
    inferred_max_seq_len = infer_max_seq_len(model)

    if max_seq_len is None:
        logger.info(f"max_seq_len is not specified, using inferred value {inferred_max_seq_len}")
        return inferred_max_seq_len
    if inferred_max_seq_len < max_seq_len:
        if allow_long_max_model_len:
            logger.warning(
                f"User specified max_seq_len is larger than the config in the model config file "
                f"({inferred_max_seq_len}). Setting max_seq_len to user's specified value {max_seq_len}. "
            )
        else:
            # NOTE: py_executor_creator makes sure that the executor uses this
            # smaller value as its max_seq_len too.
            logger.warning(
                f"Specified {max_seq_len=} is larger than what the model can support "
                f"({inferred_max_seq_len}). Setting max_seq_len to {inferred_max_seq_len}. "
            )
            return inferred_max_seq_len
    return max_seq_len


def resolve_max_num_tokens(
    max_num_tokens: int | None, *, max_seq_len: int, max_batch_size: int
) -> int:
    """Clamp the requested max_num_tokens to max_seq_len * max_batch_size."""
    # Modified from tensorrt_llm/_bootstrap.py check_max_num_tokens
    if max_num_tokens is None:
        max_num_tokens = max_seq_len * max_batch_size
    if max_num_tokens > max_seq_len * max_batch_size:
        logger.warning(
            f"max_num_tokens ({max_num_tokens}) shouldn't be greater than "
            f"max_seq_len * max_batch_size ({max_seq_len * max_batch_size}), "
            f"specifying to max_seq_len * max_batch_size ({max_seq_len * max_batch_size})."
        )
        max_num_tokens = max_seq_len * max_batch_size
    return max_num_tokens


@dataclass
class CompiledModel:
    """A model after compilation, and the caller that runs it."""

    model: torch.nn.Module
    caller: ModelCaller
    backend: Backend | None
    prefill_only: bool
    nccl_window_tensor_scope_hooks: list


def _eager_decoder_scope(model: torch.nn.Module) -> torch.nn.Module:
    """Return the eager decoder scope under any compile or prefill wrapper."""
    scope = model.model
    if isinstance(scope, PrefillCompiledModel):
        scope = scope.eager_model
    return getattr(scope, "_orig_mod", scope)


def apply_torch_compile(
    model: torch.nn.Module, backend: Backend, *, llm_args: TorchLlmArgs, fullgraph: bool
) -> tuple[torch.nn.Module, bool]:
    """Compile the eager model scope; return the model and whether only prefill is compiled."""
    prefill_only = False
    apply_llm_torch_compile = getattr(model, "apply_llm_torch_compile", None)
    if isinstance(model, DecoderModelForCausalLM):
        eager_model = _eager_decoder_scope(model)
        compiled_model = torch.compile(eager_model, backend=backend, fullgraph=fullgraph)
        prefill_only = (
            llm_args.prefill_cuda_graph_backend == PrefillCudaGraphBackend.PIECEWISE
            and (
                llm_args.torch_compile_config.compile_only_piecewise_graphs
                or not model.use_fx_for_pcg_fallback
            )
        )
        model.model = (
            PrefillCompiledModel(eager_model, compiled_model) if prefill_only else compiled_model
        )
    elif callable(apply_llm_torch_compile):
        # Avoid nesting a new wrapper around the previous compile.
        llm = getattr(model, "llm", None)
        compiled_model = getattr(llm, "model", None)
        if hasattr(compiled_model, "_orig_mod"):
            llm.model = compiled_model._orig_mod
        # TODO: Move this contract to MultimodalModelMixin once
        # multimodal models consistently expose their LLM compile scope.
        apply_llm_torch_compile(backend=backend, fullgraph=fullgraph)
    else:
        eager_model = getattr(model, "_orig_mod", model)
        model = torch.compile(eager_model, backend=backend, fullgraph=fullgraph)
    return model, prefill_only


def remove_torch_compile(model: torch.nn.Module) -> torch.nn.Module:
    """Restore the eager model scope; return the model."""
    if isinstance(model, DecoderModelForCausalLM):
        model.model = _eager_decoder_scope(model)
        return model
    apply_llm_torch_compile = getattr(model, "apply_llm_torch_compile", None)
    if callable(apply_llm_torch_compile):
        llm = getattr(model, "llm", None)
        compiled_model = getattr(llm, "model", None)
        if hasattr(compiled_model, "_orig_mod"):
            llm.model = compiled_model._orig_mod
        return model
    return getattr(model, "_orig_mod", model)


def _init_userbuffers(
    mapping: Mapping, llm_args: TorchLlmArgs, *, hidden_size: int, max_num_tokens: int
) -> bool:
    if mapping.tp_size <= 1 or mapping.pp_size > 1:
        return False

    # The userbuffers bootstrap exchanges CUDA memory handles over raw
    # MPI_COMM_WORLD collectives (userbuffers-host.cpp). Under non-MPI
    # orchestrators (Ray sets TLLM_DISABLE_MPI) every worker holds a
    # 1-rank MPI world, so importer ranks would receive uninitialized
    # bytes instead of peer handles and crash in
    # cuMemImportFromShareableHandle with 'operation not supported'.
    if mpi_disabled():
        logger.info(
            "Disabling userbuffers: its bootstrap requires MPI, which is "
            "disabled under this orchestrator."
        )
        return False

    # Disable UB for unsupported platforms
    if not ub.ub_supported():
        return False
    # NCCL_SYMMETRIC strategy no longer requires UserBuffer allocator initialization.
    # It uses NCCLWindowAllocator from ncclUtils directly.
    if llm_args.allreduce_strategy == "NCCL_SYMMETRIC":
        # Skip UB initialization for NCCL_SYMMETRIC - it uses NCCLWindowAllocator directly
        return False
    # Allocated outside PyTorch with no size exposed to Python; the memory
    # monitor names it by measuring.
    with monitor_region("userbuffers"):
        ub.initialize_userbuffers_manager(
            mapping.tp_size,
            mapping.pp_size,
            mapping.cp_size,
            mapping.rank,
            mapping.gpus_per_node,
            hidden_size * max_num_tokens * 2,
        )

    return True


def compile_model(
    model: torch.nn.Module,
    *,
    llm_args: TorchLlmArgs,
    mapping: Mapping,
    dist: Distributed | None,
    max_num_tokens: int,
    prefill_capture_num_tokens: list[int],
) -> CompiledModel:
    """Compile the model, or install the eager NCCL window tensor scopes, and
    wrap the result in a model caller."""
    torch_compile_config = llm_args.torch_compile_config
    prefill_cuda_graph_backend = llm_args.prefill_cuda_graph_backend
    torch_compile_enabled = bool(torch_compile_config is not None)
    torch_compile_fullgraph = (
        torch_compile_config.enable_fullgraph
        if torch_compile_config is not None
        else TorchCompileConfig.model_fields["enable_fullgraph"].default
    )
    compile_only_piecewise_graphs = (
        torch_compile_config.compile_only_piecewise_graphs
        if torch_compile_config is not None
        else TorchCompileConfig.model_fields["compile_only_piecewise_graphs"].default
    )
    torch_compile_inductor_enabled = (
        torch_compile_config.enable_inductor
        if torch_compile_config is not None
        else TorchCompileConfig.model_fields["enable_inductor"].default
    )
    torch_compile_piecewise_cuda_graph = (
        prefill_cuda_graph_backend == PrefillCudaGraphBackend.PIECEWISE
    )
    torch_compile_enable_userbuffers = (
        torch_compile_config.enable_userbuffers
        if torch_compile_config is not None
        else TorchCompileConfig.model_fields["enable_userbuffers"].default
    )
    torch_compile_max_num_streams = (
        torch_compile_config.max_num_streams
        if torch_compile_config is not None
        else TorchCompileConfig.model_fields["max_num_streams"].default
    )

    backend = None
    prefill_only = False
    nccl_window_tensor_scope_hooks = []
    # A forward without peers cannot allocate NCCL window leases. DWDP
    # peers are separate from the model-parallel communicator's world size.
    window_scopes_enabled = dist is not None and (dist.world_size > 1 or mapping.dwdp_enabled)

    backend_num_streams = None
    try:
        if torch_compile_enabled:
            set_torch_compiling(True)
            if mpi_disabled() and mapping.world_size > 1:
                # Resolve the device-mesh dims the compiled forward reads
                # (mapping.tp_group & friends) while still eager: the
                # DeviceMesh resolver is compiler-disabled and must never
                # run under tracing; once memoized these reads trace as
                # constants.
                _ = mapping.tp_group
                _ = mapping.pp_group
                _ = mapping.cp_group
                if mapping.moe_ep_size > 1:
                    _ = mapping.moe_tp_group
                    _ = mapping.moe_ep_group
            # torch.compile traces AllReduce.forward, whose lazy MNNVL
            # workspace rescale cannot allocate inside the traced region -
            # grow the workspaces to the engine maximum up front.
            num_mnnvl_allreduces = 0
            for module in model.modules():
                if isinstance(module, MNNVLAllReduce):
                    module.prescale_workspace(max_num_tokens, model.config.hidden_size)
                    num_mnnvl_allreduces += 1
            if num_mnnvl_allreduces:
                logger.info(
                    f"Pre-scaled the MNNVL allreduce workspace for "
                    f"{num_mnnvl_allreduces} modules to cover "
                    f"max_num_tokens={max_num_tokens} before "
                    "torch.compile."
                )
            use_ub = torch_compile_enable_userbuffers and _init_userbuffers(
                mapping,
                llm_args,
                hidden_size=model.config.hidden_size,
                max_num_tokens=max_num_tokens,
            )
            backend_num_streams = Backend.Streams(
                [torch.cuda.Stream() for _ in range(torch_compile_max_num_streams - 1)]
            )
            backend = Backend(
                torch_compile_inductor_enabled,
                enable_userbuffers=use_ub,
                enable_piecewise_cuda_graph=(torch_compile_piecewise_cuda_graph),
                capture_num_tokens=prefill_capture_num_tokens,
                max_num_streams=torch_compile_max_num_streams,
                mapping=mapping,
            )
            if compile_only_piecewise_graphs and not isinstance(model, DecoderModelForCausalLM):
                raise ValueError(
                    "TorchCompileConfig."
                    "compile_only_piecewise_graphs=True is "
                    "only supported for DecoderModelForCausalLM models."
                )

            model, prefill_only = apply_torch_compile(
                model, backend, llm_args=llm_args, fullgraph=torch_compile_fullgraph
            )
            torch._dynamo.config.cache_size_limit = 16
        else:
            set_torch_compiling(False)
            if window_scopes_enabled:
                nccl_window_tensor_scope_hooks = install_eager_nccl_window_tensor_scopes(model)
    except Exception as e:
        traceback.print_exception(Exception, e, e.__traceback__)
        raise e
    caller = ModelCaller(
        model,
        compile_backend=backend,
        aux_streams=backend_num_streams,
        prefill_compile_only=prefill_only,
        nccl_window_scopes_enabled=window_scopes_enabled,
    )
    return CompiledModel(
        model=model,
        caller=caller,
        backend=backend,
        prefill_only=prefill_only,
        nccl_window_tensor_scope_hooks=nccl_window_tensor_scope_hooks,
    )


_DEEP_GEMM_PDL_CONFIGURED = False


def _configure_deep_gemm_pdl() -> None:
    global _DEEP_GEMM_PDL_CONFIGURED
    if _DEEP_GEMM_PDL_CONFIGURED:
        return

    from tensorrt_llm import deep_gemm

    deep_gemm.set_pdl(os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1")
    _DEEP_GEMM_PDL_CONFIGURED = True


def init_process_setup(mapping: Mapping, dist: Distributed | None) -> None:
    """Configure the process-wide state a model executor needs before loading."""
    _configure_deep_gemm_pdl()
    if mapping.has_pp():
        init_pp_comm(mapping)
    if dist is not None:
        ExpertStatistic.create(dist.rank)


def create_model_loader(
    llm_args: TorchLlmArgs,
    mapping: Mapping,
    *,
    spec_config: DecodingBaseConfig | None,
    max_num_tokens: int,
    max_seq_len: int | None,
    model_weights_memory_tag: str | None = None,
    model_weights_restore_mode=None,
) -> ModelLoader:
    """Create the loader that opens the checkpoint and can reload weights later."""
    return ModelLoader(
        llm_args=llm_args,
        mapping=mapping,
        spec_config=spec_config,
        sparse_attention_config=llm_args.sparse_attention_config,
        max_num_tokens=max_num_tokens,
        max_seq_len=max_seq_len,
        lora_config=llm_args.lora_config,
        model_weights_memory_tag=model_weights_memory_tag,
        model_weights_restore_mode=model_weights_restore_mode,
    )


def prepare_loaded_model(model: torch.nn.Module, llm_args: TorchLlmArgs) -> None:
    """Give the model its extra attributes and the requested NVTX markers."""
    # AutoModelForCausalLM sets extra_attrs; a model passed in through
    # `model=`, such as a test stub, may not have it.
    if not hasattr(model, "extra_attrs"):
        model.extra_attrs = {}
    if llm_args.enable_layerwise_nvtx_marker:
        layerwise_nvtx_marker = LayerwiseNvtxMarker()
        module_prefix = "Model"
        if (
            model.model_config
            and model.model_config.pretrained_config
            and model.model_config.pretrained_config.architectures
        ):
            module_prefix = "|".join(model.model_config.pretrained_config.architectures)
        layerwise_nvtx_marker.register_hooks(model, module_prefix)


def resolve_attention_backend(
    model: torch.nn.Module, llm_args: TorchLlmArgs
) -> type[AttentionBackend]:
    """Select the attention backend, with the model's sparse attention parameters."""
    sparse_attention_config = model.model_config.sparse_attention_config
    sparse_params = (
        sparse_attention_config.to_sparse_params(
            pretrained_config=model.model_config.pretrained_config
        )
        if sparse_attention_config is not None
        else None
    )
    return get_attention_backend(llm_args.attn_backend, sparse_params=sparse_params)
