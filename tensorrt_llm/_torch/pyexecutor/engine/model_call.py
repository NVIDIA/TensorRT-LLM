# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared token-model invocation with execution metadata and compile streams."""

from __future__ import annotations

import contextlib
import weakref
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from tensorrt_llm._torch.nccl_window_tensor_scope import nccl_window_tensor_scope
from tensorrt_llm._torch.utils import (
    get_model_extra_attrs,
    get_per_request_prefill_cuda_graph_flag,
    torch_compiling,
)
from tensorrt_llm._utils import is_trace_enabled, trace_func

if TYPE_CHECKING:
    from tensorrt_llm._torch.compilation.backend import Backend


class ModelCaller:
    """Call a token model inside the caller's model-extra-attrs context.

    Args:
        model: The loaded model after compilation setup.
        compile_backend: Optional backend owning compilation events.
        aux_streams: The backend's mutable stream container, retained by reference.
        prefill_compile_only: Whether torch.compile covers only eligible prefill
            batches, so each call must scope ``is_torch_compiling()`` to that decision.
        nccl_window_scopes_enabled: Whether this forward can allocate NCCL window
            leases through communication with peers.

    Encoder stacks and multimodal encoders with different invocation contracts
    call their models directly.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        compile_backend: Backend | None = None,
        aux_streams: Backend.Streams | None = None,
        prefill_compile_only: bool = False,
        nccl_window_scopes_enabled: bool = True,
    ) -> None:
        if compile_backend is not None and aux_streams is None:
            raise ValueError("A compile backend requires its auxiliary stream container.")
        self._model = model
        self._compile_backend = compile_backend
        self._aux_streams = aux_streams
        self._prefill_compile_only = prefill_compile_only
        self._nccl_window_scopes_enabled = nccl_window_scopes_enabled

    def __call__(self, **kwargs: Any) -> Any:
        attrs = get_model_extra_attrs()
        assert attrs is not None, "Model extra attrs is not set"
        # Attention and compiled custom ops dereference these borrowed resources.
        attrs["attention_metadata"] = weakref.ref(kwargs["attn_metadata"])
        attrs.update(self._model.model_config.extra_attrs)
        attrs["spec_metadata"] = kwargs.get("spec_metadata", None)

        if self._compile_backend is not None:
            # Compilation can update the stream and event containers in place.
            assert self._aux_streams is not None
            attrs["aux_streams"] = weakref.ref(self._aux_streams)
            attrs["events"] = weakref.ref(self._compile_backend.events)
            attrs["global_stream"] = torch.cuda.current_stream()

        # Scope the entire top-level forward, including Eagle3's epilogue, so
        # eager decode and over-ceiling prefill do not select compile-only ops.
        compile_scope = (
            torch_compiling(get_per_request_prefill_cuda_graph_flag())
            if self._prefill_compile_only
            else contextlib.nullcontext()
        )
        # This boundary stays outside torch.compile, so one lease scope covers
        # eager, compiled, and CUDA-graph model execution.
        window_scope = (
            nccl_window_tensor_scope(kwargs) if self._nccl_window_scopes_enabled else None
        )
        with compile_scope, window_scope if window_scope is not None else contextlib.nullcontext():
            if is_trace_enabled("TLLM_TRACE_MODEL_FORWARD"):
                outputs = trace_func(self._model.forward)(**kwargs)
            else:
                outputs = self._model.forward(**kwargs)
            return window_scope.escape(outputs) if window_scope is not None else outputs
