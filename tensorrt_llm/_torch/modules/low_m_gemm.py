# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
from typing import Any, List, Optional, Tuple

import torch

from tensorrt_llm._utils import is_sm_100f
from tensorrt_llm.logger import logger

from ..autotuner import AutoTuner, DynamicTensorSpec, TunableRunner, TuningConfig
from ..flashinfer_utils import get_env_enable_pdl

_BACKEND_ENV = "TRTLLM_LOW_M_GEMM_BACKEND"

# M-dimension tuning buckets fed to the TRT-LLM AutoTuner.  The AutoTuner
# generates one optimization profile per bucket value even if only a single
# M was seen during warmup, ensuring that all serving M values are covered.
_M_TUNING_BUCKETS: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)

_M_DIM_SPEC = DynamicTensorSpec(
    input_idx=0,
    dim_idx=0,
    gen_tuning_buckets=_M_TUNING_BUCKETS,
    map_to_tuning_buckets=lambda x: next(
        (b for b in _M_TUNING_BUCKETS if b >= x), _M_TUNING_BUCKETS[-1]
    ),
)

# Authoritative M upper bound, duplicated from low_m_bf16_splitk.MAX_M to avoid
# importing nvidia-cutlass-dsl on every import of this module.
_MAX_M = 32

_AUTOTUNER_OP_NAME = "trtllm::low_m_bf16_gemm"

# Env-var strings that activate the low-M GEMM path.
# "cublas" / "cublaslt" were legacy aliases meaning "disabled"; anything
# unrecognised raises ValueError.
_ENABLED_VALUES = frozenset({"auto", "cute-dsl", "flashinfer", "flashinfer-cute-dsl"})
_DISABLED_VALUES = frozenset({"", "0", "none", "off", "cublas", "cublaslt"})


def _parse_enabled() -> bool:
    """Return True when ``TRTLLM_LOW_M_GEMM_BACKEND`` activates the low-M GEMM path."""
    raw = os.environ.get(_BACKEND_ENV, "off").strip().lower().replace("_", "-")
    if raw in _ENABLED_VALUES:
        return True
    if raw in _DISABLED_VALUES:
        return False
    raise ValueError(
        f"Invalid {_BACKEND_ENV}={os.environ.get(_BACKEND_ENV)!r}; "
        f"expected one of: off, auto, cute-dsl."
    )


@functools.lru_cache(maxsize=None)
def _current_sm(device: torch.device) -> int:
    """Return the SM version (e.g. 100 for B200, 103 for GB300) for *device*.

    Kept as a named lru_cache'd function so unit tests can monkeypatch it without
    touching torch.cuda internals.
    """
    major, minor = torch.cuda.get_device_capability(device)
    return major * 10 + minor


# ---------------------------------------------------------------------------
# TunableRunner for cuBLAS (baseline / fallback competitor)
# ---------------------------------------------------------------------------


class _CuBLASGemmRunner(TunableRunner):
    """AutoTuner runner that delegates to cuBLAS via ``torch.mm`` / ``torch.addmm``.

    Included in every AutoTuner race so that low-M kernels are only chosen
    when they actually beat cuBLAS on the observed (M, K, N) shape.  For shapes
    where the custom kernels are slower (e.g. large-N / small-K on certain
    hardware), this runner wins and the low-M path is silently skipped —
    no performance regression regardless of env-var setting.

    The sentinel tactic ``("cublas",)`` is JSON-serialisable so the AutoTuner
    cache persists it correctly across warmup calls.

    inputs: ``[a [M,K] BF16, b_t [K,N] BF16]``; optional ``bias [N]`` kwarg.
    """

    _TACTIC = ("cublas",)

    def __init__(self, *, has_bias: bool) -> None:
        self.has_bias = has_bias

    def unique_id(self) -> tuple:
        return ("cublas", bool(self.has_bias))

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: Any,
        **kwargs: Any,
    ) -> List[Any]:
        return [self._TACTIC]

    def forward(
        self,
        inputs: List[torch.Tensor],
        *,
        tactic: Any = None,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        a, b_t = inputs[0], inputs[1]  # a=[M,K], b_t=[K,N]
        bias = kwargs.get("bias")
        out = kwargs.get("out")
        if out is None:
            out = torch.empty((int(a.shape[0]), int(b_t.shape[1])), dtype=a.dtype, device=a.device)
        # torch.mm / torch.addmm write directly into ``out`` with no extra copy.
        if bias is not None:
            torch.addmm(bias, a, b_t, out=out)
        else:
            torch.mm(a, b_t, out=out)
        return out


# ---------------------------------------------------------------------------
# TunableRunner for the SM10x direct (SIMT register-prefetch) BF16 GEMM
# ---------------------------------------------------------------------------


class _DirectGemmRunner(TunableRunner):
    """AutoTuner runner for the SM10x CuTe-DSL direct (SIMT) BF16 kernel.

    The direct kernel prefetches the entire K dimension of each weight row into
    registers and accumulates with scalar FP32, avoiding TMA and TMEM.  It
    outperforms the split-K kernel at ``M=1, K=8192, N≤4608`` and a few other
    narrow shapes; the AutoTuner selects it only when it wins the timed race.

    **Bias is not supported** — ``get_valid_tactics`` returns ``[]`` when
    ``bias`` is present so the AutoTuner never records a direct-kernel win for
    biased calls.

    inputs: ``[a [M,K] BF16, b_t [K,N] col-major BF16]``.
    """

    def __init__(self, *, pdl: bool) -> None:
        self.pdl = pdl

    def unique_id(self) -> tuple:
        return ("direct", bool(self.pdl))

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: Any,
        **kwargs: Any,
    ) -> List[Any]:
        """Return JSON-serialisable ``(block_size, outputs_per_block, rows_per_block)`` tuples.

        Returns an empty list when *bias* is present — the direct kernel has no
        bias support so it should never win an autotuned bias call.
        """
        import dataclasses

        if kwargs.get("bias") is not None:
            return []

        from tensorrt_llm._torch.cute_dsl_kernels.blackwell.low_m_bf16_direct import (
            autotune_tactics,
            default_tactic,
        )

        a, b_t = inputs[0], inputs[1]
        m, k, n = int(a.shape[0]), int(a.shape[1]), int(b_t.shape[1])
        seen: dict = {}
        try:
            for t in (default_tactic(m, n, k), *autotune_tactics(m, n, k)):
                seen.setdefault(dataclasses.astuple(t), None)
        except ValueError:
            pass
        return list(seen)

    def forward(
        self,
        inputs: List[torch.Tensor],
        *,
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        from tensorrt_llm._torch.cute_dsl_kernels.blackwell.low_m_bf16_direct import (
            DirectTactic,
            default_tactic,
            run_direct_dense,
        )

        a, b_t = inputs[0], inputs[1]
        m, k, n = int(a.shape[0]), int(a.shape[1]), int(b_t.shape[1])
        actual_tactic = default_tactic(m, n, k) if tactic == -1 else DirectTactic(*tactic)
        out = kwargs.get("out")
        if out is None:
            out = torch.empty((m, n), dtype=a.dtype, device=a.device)
        # b_t is [K, N] col-major (weight.t()), which is what run_direct_dense
        # expects as its 'b' argument (b.T will be [N, K] row-major for the kernel).
        return run_direct_dense(a, b_t, out, self.pdl, actual_tactic)


# ---------------------------------------------------------------------------
# TunableRunner for the built-in SM10x split-K BF16 GEMM
# ---------------------------------------------------------------------------


class _SplitKGemmRunner(TunableRunner):
    """AutoTuner runner for the built-in SM10x (Blackwell) CuTe-DSL split-K BF16 kernel.

    One instance per ``(has_bias, pdl)`` so the AutoTuner keeps separate
    best-tactic cache entries for each variant.

    inputs: ``[a [M,K] BF16, b_t [K,N] BF16]``; ``bias`` passed as a kwarg.
    """

    def __init__(self, *, has_bias: bool, pdl: bool) -> None:
        self.has_bias = has_bias
        self.pdl = pdl

    def unique_id(self) -> tuple:
        return (bool(self.has_bias), bool(self.pdl))

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: Any,
        **kwargs: Any,
    ) -> List[Any]:
        """Return JSON-serialisable ``(mma_m, mma_n, split_k, ab_stages)`` 4-tuples."""
        import dataclasses

        from tensorrt_llm._torch.cute_dsl_kernels.blackwell.low_m_bf16_splitk import (
            autotune_tactics,
            default_tactic,
        )

        a, b_t = inputs[0], inputs[1]
        m, k, n = int(a.shape[0]), int(a.shape[1]), int(b_t.shape[1])
        # Guard against zero-width outputs: default_tactic raises ValueError for n<=0,
        # which would propagate past AutoTuner's per-tactic exception handling and
        # bypass the cuBLAS fallback.
        if n <= 0:
            return []
        # Deduplicate: heuristic default first, then enumerated candidates.
        seen: dict = {}
        for t in (default_tactic(m, n, k), *autotune_tactics(m, n, k)):
            seen.setdefault(dataclasses.astuple(t), None)
        return list(seen)

    def forward(
        self,
        inputs: List[torch.Tensor],
        *,
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        from tensorrt_llm._torch.cute_dsl_kernels.blackwell.low_m_bf16_splitk import (
            SplitKTactic,
            default_tactic,
            run_splitk_dense,
        )

        a, b_t = inputs[0], inputs[1]
        bias = kwargs.get("bias")
        m, k, n = int(a.shape[0]), int(a.shape[1]), int(b_t.shape[1])
        actual_tactic = default_tactic(m, n, k) if tactic == -1 else SplitKTactic(*tactic)
        # Prefer a pre-allocated buffer supplied by the caller so the kernel
        # writes to a stable device address that a surrounding CUDA graph can
        # safely replay.  Fall back to per-call allocation when no buffer is
        # provided (e.g. during AutoTuner profiling).
        out = kwargs.get("out")
        if out is None:
            out = torch.empty((m, n), dtype=a.dtype, device=a.device)
        return run_splitk_dense(a, b_t, bias, out, self.pdl, actual_tactic)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

# ``LOW_M_GEMM_ACTIVE`` is read at import time from the env var so that
# ``linear.py`` can gate the hot path with a single bool check.
LOW_M_GEMM_ACTIVE = _parse_enabled()


class LowMGemmDispatcher:
    """Route eligible BF16 GEMMs to the best available kernel for each shape.

    On every forward call the TRT-LLM :class:`AutoTuner` races two low-M
    SM10x (Blackwell) CuTe-DSL kernels (split-K tensor-core and SIMT direct)
    against cuBLAS (``torch.mm``) and caches the winner per ``(M-bucket, K, N)``
    shape.  Custom kernels are only chosen when they are strictly faster than
    cuBLAS for the observed tensor geometry — shapes where neither custom kernel
    wins automatically fall back to cuBLAS without any manual tuning.

    Lifecycle
    ---------
    1. **Model loading**: call :meth:`attach` once per model to label every
       ``Linear`` submodule and bind this dispatcher to it.  This is lightweight
       (no GPU resources allocated).
    2. **First forward / warmup**: :meth:`apply` calls :meth:`_init_runners`
       lazily on the first invocation.  The AutoTuner then profiles each
       (M-bucket, K, N) shape it encounters and caches the winner.
    """

    def __init__(self) -> None:
        # _prepared: runners + TuningConfig have been created (set by _init_runners).
        self._prepared = False
        # _attached: modules have been labelled (set by attach).
        self._attached = False
        self._runner_no_bias: Optional[_SplitKGemmRunner] = None
        self._runner_with_bias: Optional[_SplitKGemmRunner] = None
        # Direct (SIMT register-prefetch) runner — no-bias only; cuBLAS handles
        # biased calls when neither low-M kernel wins.
        self._direct: Optional[_DirectGemmRunner] = None
        # cuBLAS runners compete in every AutoTuner race as a safety baseline.
        self._cublas_no_bias: Optional[_CuBLASGemmRunner] = None
        self._cublas_with_bias: Optional[_CuBLASGemmRunner] = None
        self._tuning_config: Optional[TuningConfig] = None
        # Pre-allocated output buffers keyed by ``(m_bucket, n)``.  Populated
        # lazily outside of CUDA-graph capture so the kernel always writes to a
        # stable device address that a surrounding CUDA graph can safely replay.
        self._output_buffers: dict = {}
        # Saved ``force`` flag from :meth:`attach` so :meth:`_init_runners` can
        # honour it without requiring an extra argument.
        self._force: bool = False

    def attach(
        self,
        model: torch.nn.Module,
        force: bool = False,
    ) -> None:
        """Label every ``Linear`` submodule and bind this dispatcher to it.

        This is the **model-loading phase**: it iterates the module tree once,
        assigns ``_low_m_gemm_name`` and ``_low_m_gemm_dispatcher`` on each
        eligible submodule, and returns immediately.  No GPU resources are
        allocated; runners are created lazily on the first :meth:`apply` call.

        Args:
            model: The model whose Linear submodules should be labelled.
            force: Attach even when ``TRTLLM_LOW_M_GEMM_BACKEND`` is not set.
                Reserved for callers that activate the dispatcher via a
                mechanism other than the env var (e.g. a future LLM arg).
        """
        if self._attached:
            return
        self._attached = True
        self._force = force
        if not LOW_M_GEMM_ACTIVE and not force:
            return
        for name, module in model.named_modules():
            if module.__class__.__name__.endswith("Linear"):
                module._low_m_gemm_name = name
                # Bind this dispatcher so apply_low_m_gemm can reach the
                # per-engine instance without a global singleton.
                module._low_m_gemm_dispatcher = self

    def _init_runners(self) -> None:
        """Create AutoTuner runners and the tuning config (lazy, first-use init).

        Called automatically by :meth:`apply` on the first forward pass.
        Separated from :meth:`attach` so that GPU-resource allocation happens
        during the warmup phase rather than at model-load time.
        """
        if self._prepared:
            return
        if not LOW_M_GEMM_ACTIVE and not self._force:
            # Mark prepared *before* returning so repeated apply() calls in the
            # inactive-but-not-forced path skip the check without creating runners
            # that would remain None.  apply() re-checks LOW_M_GEMM_ACTIVE / force_active
            # before reaching here, so the None-runner state is never observed by callers.
            self._prepared = True
            return
        pdl = get_env_enable_pdl()
        self._runner_no_bias = _SplitKGemmRunner(has_bias=False, pdl=pdl)
        self._runner_with_bias = _SplitKGemmRunner(has_bias=True, pdl=pdl)
        # Direct SIMT runner: no-bias only; wins for M=1 with narrow N on K=8192.
        self._direct = _DirectGemmRunner(pdl=pdl)
        # cuBLAS runners — always included as competitors so the AutoTuner can
        # fall back to cuBLAS for shapes where neither custom kernel is fastest.
        self._cublas_no_bias = _CuBLASGemmRunner(has_bias=False)
        self._cublas_with_bias = _CuBLASGemmRunner(has_bias=True)
        self._tuning_config = TuningConfig(
            dynamic_tensor_specs=(_M_DIM_SPEC,),
            use_cold_l2_cache=True,
            # Use CUDA-graph-based profiling when PDL is off; PDL and CUDA-graph
            # capture conflict so we fall back to standard timing in that case.
            use_cuda_graph=not pdl,
        )
        # Mark prepared only after all runners and the tuning config exist so the
        # invariant "_prepared implies runners are non-None" always holds.
        self._prepared = True
        logger.info(
            "Low-M BF16 GEMM dispatcher initialised "
            "(split-K tensor-core + SIMT direct vs cuBLAS AutoTuner; best kernel per shape)."
        )

    def prepare(
        self,
        model: torch.nn.Module,
        force: bool = False,
    ) -> None:
        """Attach the dispatcher and eagerly initialise runners.

        Convenience wrapper that calls :meth:`attach` followed by
        :meth:`_init_runners`.  Prefer calling :meth:`attach` at model-load
        time and letting :meth:`_init_runners` trigger lazily on the first
        forward pass; use this method only when immediate runner initialisation
        is required (e.g. in unit tests).

        Args:
            model: The model whose Linear submodules should be labelled.
            force: Initialise runners even when ``TRTLLM_LOW_M_GEMM_BACKEND``
                is not set.  Reserved for callers that activate the low-M
                GEMM path via a mechanism other than the env var (e.g. a future
                LLM arg).
        """
        self.attach(model, force=force)
        self._init_runners()

    @staticmethod
    def _is_candidate_shape(
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> bool:
        if (
            torch.is_grad_enabled()
            or input_tensor.ndim < 1
            or weight.ndim != 2
            or not input_tensor.is_cuda
            or not weight.is_cuda
            or input_tensor.device != weight.device
            or input_tensor.dtype != torch.bfloat16
            or weight.dtype != torch.bfloat16
            or not input_tensor.is_contiguous()
            or not weight.is_contiguous()
        ):
            return False
        k = int(input_tensor.shape[-1])
        if k <= 0 or int(weight.shape[1]) != k or k % 128:
            return False
        if not 1 <= input_tensor.numel() // k <= _MAX_M:
            return False
        if input_tensor.data_ptr() % 32 or weight.data_ptr() % 32:
            return False
        if bias is not None and (
            not bias.is_cuda
            or bias.device != input_tensor.device
            or bias.dtype != torch.bfloat16
            or bias.shape != (weight.shape[0],)
            or not bias.is_contiguous()
        ):
            return False
        return is_sm_100f(_current_sm(input_tensor.device))

    def apply(
        self,
        module: torch.nn.Module,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        force_active: bool = False,
    ) -> Optional[torch.Tensor]:
        """Return a low-M GEMM result, or ``None`` to fall through to the normal path.

        Args:
            force_active: When ``True`` the low-M path is attempted even if
                ``TRTLLM_LOW_M_GEMM_BACKEND`` is not set.  Reserved for callers
                that activate it via a mechanism other than the env var (e.g. a
                future ``use_low_m_low_latency_gemm`` LLM arg).
        """
        if not LOW_M_GEMM_ACTIVE and not force_active:
            return None
        if not self._prepared:
            # Unit tests and direct callers that bypass ModelEngine / attach().
            # Save force so _init_runners() can honour it.
            self._force = force_active
            self._init_runners()
        if not self._is_candidate_shape(input_tensor, weight, bias):
            return None
        # _is_candidate_shape() confirmed grad is disabled; detach weight only
        # (it may carry requires_grad=True even in inference mode).
        input_2d = input_tensor.view(-1, input_tensor.shape[-1])
        weight_t = weight.detach().t()  # [N, K] → [K, N] col-major view
        bias_d = bias.detach() if bias is not None else None
        has_bias = bias_d is not None
        # Race cuBLAS, split-K tensor-core, and SIMT direct; the AutoTuner caches
        # the winner per shape.  cuBLAS is listed first so that the cache-miss
        # fallback (runners[0]) always lands on the safe baseline rather than
        # running a custom kernel on an untuned shape.
        splitk_runner = self._runner_with_bias if has_bias else self._runner_no_bias
        cublas_runner = self._cublas_with_bias if has_bias else self._cublas_no_bias
        runners: List = [cublas_runner, splitk_runner]
        if not has_bias:
            # Include the direct SIMT runner for unbiased calls; it silently
            # returns [] for biased inputs so it is safe to always pass, but
            # excluding it for biased calls keeps the race concise.
            runners.append(self._direct)
        best_runner, best_tactic = AutoTuner.get().choose_one(
            _AUTOTUNER_OP_NAME,
            runners,
            self._tuning_config,
            [input_2d, weight_t],
            bias=bias_d,
        )
        # Fetch or lazily create a per-(module, M-bucket, N) output buffer so
        # the kernel always writes to a stable device address compatible with
        # TRT-LLM's model-level CUDA-graph replay.  Allocation is guarded
        # against happening inside a live CUDA-graph capture window.
        #
        # The module name (set by prepare()) is included in the key to prevent
        # two Linear modules that share the same output shape — e.g. gate_proj
        # and up_proj in a SwiGLU MLP, or Q/K/V projections with equal head
        # dimensions — from aliasing each other's output buffer.
        m = int(input_2d.shape[0])
        n = int(weight.shape[0])
        m_bucket = next((b for b in _M_TUNING_BUCKETS if b >= m), _M_TUNING_BUCKETS[-1])
        module_key = getattr(module, "_low_m_gemm_name", str(id(module)))
        buf_key = (module_key, m_bucket, n)
        if buf_key not in self._output_buffers:
            if torch.cuda.is_current_stream_capturing():
                # Allocation inside a CUDA-graph capture corrupts the graph.
                # Use a per-call tensor so the capture can complete; the next
                # non-capturing call will populate the cache.
                out_buf = torch.empty((m, n), dtype=input_2d.dtype, device=input_2d.device)
            else:
                self._output_buffers[buf_key] = torch.empty(
                    (m_bucket, n), dtype=torch.bfloat16, device=input_2d.device
                )
                out_buf = self._output_buffers[buf_key][:m, :]
        else:
            out_buf = self._output_buffers[buf_key][:m, :]
        out = best_runner([input_2d, weight_t], tactic=best_tactic, bias=bias_d, out=out_buf)
        out_view = out.view(*input_tensor.shape[:-1], weight.shape[0])
        # Clone when using the shared cached buffer (i.e. outside CUDA-graph capture)
        # so that callers retaining the result across repeated forward passes —
        # e.g. the speculative-decoding drafting loop that appends logits before
        # torch.stack() — receive an independent copy rather than a view that the
        # next call silently overwrites.  Inside a capture window out_buf is already
        # a fresh per-call allocation, so no clone is needed there.
        if not torch.cuda.is_current_stream_capturing():
            return out_view.clone()
        return out_view


# Process-global fallback dispatcher used by callers that bypass
# ``prepare_low_m_gemm`` (e.g. unit tests or direct kernel benchmarks).
# Production code paths receive a *per-engine* dispatcher created by
# ``prepare_low_m_gemm`` and stored on each Linear module.
_DISPATCHER = LowMGemmDispatcher()


def prepare_low_m_gemm(
    model: torch.nn.Module,
    force: bool = False,
) -> LowMGemmDispatcher:
    """Create a per-engine dispatcher and attach it to every Linear submodule.

    Each call creates a fresh :class:`LowMGemmDispatcher` so that two model
    engines running in the same process (e.g. target + draft in speculative
    decoding) own independent dispatcher state and never share output buffers.

    This function performs only the lightweight **attach** phase: it labels every
    ``Linear`` submodule with ``_low_m_gemm_name`` and ``_low_m_gemm_dispatcher``
    so :func:`apply_low_m_gemm` can reach the per-engine instance without a
    global reference.  AutoTuner runners are initialised lazily on the first
    forward pass (warmup), not here.

    Args:
        model: Model whose Linear submodules will be labelled.
        force: Attach even when ``TRTLLM_LOW_M_GEMM_BACKEND`` is not set
            (reserved for callers that activate the dispatcher via a future
            LLM arg).

    Returns:
        The newly created :class:`LowMGemmDispatcher` for this engine.
    """
    dispatcher = LowMGemmDispatcher()
    dispatcher.attach(model, force=force)
    return dispatcher


def apply_low_m_gemm(
    module: torch.nn.Module,
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    force_active: bool = False,
) -> Optional[torch.Tensor]:
    """Return a low-M BF16 result, or ``None`` for the normal path.

    Uses the per-engine :class:`LowMGemmDispatcher` stored on *module* by
    :func:`prepare_low_m_gemm`, falling back to the global ``_DISPATCHER``
    for callers that bypass preparation (e.g. unit tests).

    Args:
        force_active: Attempt the low-M path even when
            ``TRTLLM_LOW_M_GEMM_BACKEND`` is not set (passed through to
            :meth:`LowMGemmDispatcher.apply`).
    """
    dispatcher: LowMGemmDispatcher = getattr(module, "_low_m_gemm_dispatcher", _DISPATCHER)
    return dispatcher.apply(module, input_tensor, weight, bias, force_active=force_active)


__all__ = [
    "LOW_M_GEMM_ACTIVE",
    "_MAX_M",
    "apply_low_m_gemm",
    "prepare_low_m_gemm",
]
