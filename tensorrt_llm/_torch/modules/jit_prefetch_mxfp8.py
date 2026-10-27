# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""JIT compile ahead and tune ahead for the FlashInfer MXFP8 GEMM of the
MXFP8 Linear layers (``MXFP8LinearMethod``; see ``jit_prefetch.py``).

The autotuner warmup's FlashInfer MXFP8 pass builds FlashInfer's CUTLASS
MXFP8 GEMM module the first time it tunes ``mm_mxfp8`` (DeepSeek-V4.1's
Engram WKV, MiniMax-M3). Whether that pass runs is predicted here from the
model and the engine's settings (``flashinfer_mxfp8_pass_selected``), so its
modules build in a CPU-only helper while the weights load
(``start_mxfp8_tune_builds``). ``FlashInferMxfp8GraphTuning`` tunes the
decode-graph backends of each generation CUDA graph shape once instead of in
every warmup-only forward.
"""

from __future__ import annotations

import contextlib
import os
from typing import Iterator

import torch

# FlashInfer modules the autotuner warmup's FlashInfer MXFP8 pass builds the
# first time it tunes ``mm_mxfp8`` on an SM100-family GPU, in the order it
# needs them: the delay kernel FlashInfer's autotuner times with (seconds),
# then the CUTLASS MXFP8 GEMM (over a minute).
FLASHINFER_MXFP8_TUNE_MODULES = ("trtllm_utils", "mxfp8_gemm_cutlass")


def flashinfer_mxfp8_pass_selected(
    model: torch.nn.Module,
    mapping,
    *,
    enable_autotuner: bool,
    context_warmups_supported: bool,
    cuda_graphs: bool,
    compile_all_batches: bool,
) -> bool:
    """Whether a decoder runner's warmup, on ``model``, runs the autotuner
    warmup's FlashInfer MXFP8 pass. Mirrors the pass's selection in
    ``DecoderWarmup._run_autotuner_warmup`` without changing any layer's
    backend; the autotuner warmup still decides.

    ``context_warmups_supported`` is the runner class's own flag,
    ``cuda_graphs`` whether the decode CUDA graph runner is enabled and
    ``compile_all_batches`` whether ``torch.compile`` runs the decode batches
    as well as the prefill ones.
    """
    from .linear import MXFP8LinearMethod

    # The autotuner warmup runs only with the autotuner on, for a runner with
    # context warmups; context parallelism other than Helix skips warmup
    # altogether.
    if (
        not enable_autotuner
        or not context_warmups_supported
        or (mapping.cp_size > 1 and not mapping.has_cp_helix())
    ):
        return False
    methods = [
        quant_method
        for module in model.modules()
        if isinstance((quant_method := getattr(module, "quant_method", None)), MXFP8LinearMethod)
    ]
    # FlashInfer MXFP8 runs on the CUTLASS-layout operands through the
    # trtllm::flashinfer_mm_mxfp8 op (MXFP8LinearMethod._load_flashinfer).
    if not any(method.use_cutlass for method in methods):
        return False
    if getattr(torch.ops.trtllm, "flashinfer_mm_mxfp8", None) is None:
        return False
    if "TRTLLM_MXFP8_GEMM_BACKEND" in os.environ:
        return any(
            method.backend == "flashinfer" or (method.backend == "auto" and not compile_all_batches)
            for method in methods
        )
    return (
        cuda_graphs
        and not compile_all_batches
        and any(
            getattr(module, "_use_flashinfer_mxfp8_decode_graph_default", False)
            for module in model.modules()
        )
    )


def decode_batches_compiled(model, llm_args) -> bool:
    """Whether ``torch.compile`` runs the decode batches of ``model`` in the
    engine ``llm_args`` describes, as ``PyTorchModelEngine`` decides it: with
    a ``torch_compile_config``, unless piecewise prefill CUDA graphs confine
    a decoder model's compiled graph to prefill (the engine's
    ``_torch_compile_prefill_only``)."""
    from tensorrt_llm.llmapi.llm_args import PrefillCudaGraphBackend

    from ..models.modeling_utils import DecoderModelForCausalLM

    compile_cfg = llm_args.torch_compile_config
    if compile_cfg is None:
        return False
    return not (
        isinstance(model, DecoderModelForCausalLM)
        and llm_args.prefill_cuda_graph_backend == PrefillCudaGraphBackend.PIECEWISE
        and (compile_cfg.compile_only_piecewise_graphs or not model.use_fx_for_pcg_fallback)
    )


def tunes_flashinfer_mxfp8_at_load(model, llm_args) -> bool:
    """Whether the engine ``llm_args`` describes will run the autotuner
    warmup's FlashInfer MXFP8 pass on ``model``, known once the model is
    constructed: ``flashinfer_mxfp8_pass_selected`` on the settings
    ``PyTorchModelEngine`` derives from ``llm_args``. False if the engine
    runs another runner than a decoder runner (encoder, pooling), which has
    no such pass."""
    from ..pyexecutor.engine.runners import DecoderRunner, resolve_runner_type

    runner_cls = resolve_runner_type(model, llm_args)
    if not issubclass(runner_cls, DecoderRunner):
        return False
    return flashinfer_mxfp8_pass_selected(
        model,
        model.model_config.mapping,
        enable_autotuner=llm_args.enable_autotuner,
        context_warmups_supported=runner_cls._context_warmups_supported,
        cuda_graphs=llm_args.cuda_graph_config is not None,
        compile_all_batches=decode_batches_compiled(model, llm_args),
    )


def start_mxfp8_tune_builds(tunes_flashinfer_mxfp8: bool) -> bool:
    """Start building, in a CPU-only helper, the FlashInfer modules the
    autotuner warmup's FlashInfer MXFP8 pass builds; called with whether this
    engine runs that pass, once the model is constructed (predicted from the
    engine's arguments, ``tunes_flashinfer_mxfp8_at_load``) and again at the
    start of warmup, where the engine decides it. The GEMM module is a
    ~75-100 s nvcc build, which then overlaps the weight load, or at least the
    warmup forwards before the autotuner; the pass waits on FlashInfer's file
    lock, held by the helper, instead of compiling it itself. Returns False if
    prefetch is off, the pass does not run, or there is nothing to build (the
    modules are built, or a helper is on them)."""
    from ..jit_prefetch import prefetch_enabled, start_flashinfer_build

    if not prefetch_enabled() or not tunes_flashinfer_mxfp8:
        return False
    from tensorrt_llm._utils import is_sm_100f

    if not is_sm_100f():
        return False
    return start_flashinfer_build(FLASHINFER_MXFP8_TUNE_MODULES)


class FlashInferMxfp8GraphTuning:
    """Tunes the FlashInfer MXFP8 decode-graph backends of one CUDA graph
    runner around its warmup-only forwards, once per generation CUDA graph
    shape.

    In the warmup-only pass, MXFP8 layers tune their per-bucket quantize and
    GEMM backends (``mxfp8_quantize_gemm_autotuned``), and FlashInfer's
    autotuner runs in tuning mode for every GEMM. A tuning-mode lookup of a
    cached tactic still rebuilds every tuning profile of the op, seconds per
    forward on a model with hundreds of MXFP8 GEMMs. A shape's tactics depend
    only on its GEMM shapes, which follow from the batch size and draft
    length, and stay in the process-wide autotuner caches. So with JIT compile
    ahead on (``warmup_jit_coverage_enabled`` false) a shape is tuned once per
    runner: a repeat (the advanced-sampling and LoRA variants, the final
    executor's warmup) runs outside tuning mode and looks the same tactics
    up, as the capture pass does. ``TLLM_WARMUP_JIT_COVERAGE=1``, the default
    without compile ahead, tunes every warmup-only forward. So does
    ``TLLM_AUTOTUNER_CACHE_PATH``: the autotuner warmup of the next pass
    reloads that file, which replaces the in-memory tactics with the file's.
    """

    def __init__(self) -> None:
        # Whether the current pass tunes (``tuning_pass``).
        self.active = False
        # Generation CUDA graph shapes (batch size, draft length) whose
        # warmup-only forward ran under the tuning.
        self.tuned_shapes: set[tuple[int, int]] = set()

    @contextlib.contextmanager
    def tuning_pass(self, enabled: bool) -> Iterator[None]:
        """A pass over the generation CUDA graph shapes, which tunes its
        forwards if ``enabled`` (a warmup-only pass of a model with MXFP8
        layers on FlashInfer)."""
        self.active = enabled
        try:
            yield
        finally:
            self.active = False

    @contextlib.contextmanager
    def forward(self, batch_size: int, draft_len: int) -> Iterator[None]:
        """Around one forward of a generation CUDA graph shape: tune it, unless
        the pass does not tune or this runner tuned the shape before."""
        from ..jit_prefetch import warmup_jit_coverage_enabled

        shape = (batch_size, draft_len)
        if not self.active or (
            shape in self.tuned_shapes
            and not warmup_jit_coverage_enabled()
            and os.environ.get("TLLM_AUTOTUNER_CACHE_PATH") is None
        ):
            yield
            return
        from .linear import flashinfer_mxfp8_autotune

        with flashinfer_mxfp8_autotune():
            yield
        self.tuned_shapes.add(shape)
