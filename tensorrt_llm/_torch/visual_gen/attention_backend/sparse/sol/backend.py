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
"""VisualGen SOL attention backends.

SOL (Sol-Attn, https://arxiv.org/abs/2607.24027) routes attention blocks
dynamically and corrects the approximation of the blocks it skips. Two backends
serve one ``SolParams``:

* ``SOLTrtllmAttention`` runs SOL in two stages through the generic TRTLLM
  sparse lifecycle: one graph-visible predictor operator derives the exact
  block bitmask and K/V proxy summaries, then the shared PrimTS block-sparse
  FMHA executes that route.
* ``SOLCuTeDSLAttention`` runs the fused kernel vendored from the reference
  implementation (https://github.com/NVlabs/Sana, branch ``sol-engine``, pinned
  in ``cute_dsl_kernels/blackwell/sol_attn/THIRD_PARTY_NOTICES.md``), which
  folds routing, sparse computation and correction into one online-softmax
  pass. Only the sm100 kernels are carried; see
  ``cute_dsl_kernels/blackwell/sol_attn_backend.py`` for the shape and dtype
  guard around them.

``disabled_until_timestep`` is the dense-prefix control shared with skip
softmax: the layer runs dense while the normalized timestep is at or above the
cutoff and switches to SOL below it. The TRTLLM wrapper prepares that timestep
as a host value before CUDA Graph capture; the CuTeDSL backend reads the phase
the CUDA Graph runner resolved for the graph key.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from tensorrt_llm._torch.attention.backends.fmha.prims_ts_block_sparse import PrimsTSBlockSparseFmha
from tensorrt_llm._torch.attention.backends.fmha.utils import get_bmm1_scale
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.attention.backends.sparse.params import BlockSparseForwardInputs
from tensorrt_llm._torch.attention.backends.sparse.timestep_phase import graph_phase_for_timestep
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.visual_gen.cuda_graph_runner import resolved_extra_key
from tensorrt_llm.logger import logger

from ...interface import AttentionBackend, AttentionTensorLayout
from ...trtllm import TrtllmAttention
from ...vanilla import VanillaAttention
from . import predictor as sol_predictor
from .params import SolParams
from .predictor import BLOCK_SIZE

_sol_attn_import_error = None
try:
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.sol_attn_backend import (
        _run_sol_attn_bthd as _sol_attn_run,
    )
except (ImportError, OSError) as e:
    _sol_attn_run = None
    _sol_attn_import_error = e


def _cute_dense_available() -> bool:
    """Whether `cute_dsl_fmha_fwd` can run on the current device.

    Checked once at construction. SOL and the dense CuTe DSL kernel now
    cover the same set (sm_100a/sm_103a), so in practice this is always true
    wherever SOL runs; the negative branch exists so an unsupported device
    degrades to SDPA instead of raising.
    """
    try:
        from ...cute_dsl.fmha import _check_cute_runtime_available, _get_gpu_arch

        _check_cute_runtime_available()
        _get_gpu_arch()
    except Exception:
        return False
    return True


class SOLCuTeDSLAttention(AttentionBackend):
    """Fused SOL sparse attention (CuTeDSL, sm100/sm103).

    Inputs the kernel cannot serve (shape, dtype, architecture) are delegated to
    dense attention up front by ``_run_sol_attn_bthd``; a kernel error is not
    recovered from and propagates. This class adds the ``dense_layers`` guard
    (evaluated at construction time, no external plumbing needed) and forwards
    the routing knobs from config.
    """

    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 8,
        head_dim: int = 128,
        num_kv_heads: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        sparse_params: SolParams | None = None,
        **kwargs,
    ):
        if _sol_attn_run is None:
            raise ImportError(
                "SOLCuTeDSLAttention requires the vendored sol_attn kernel "
                f"package; import failed: {_sol_attn_import_error}"
            )
        if sparse_params is None:
            sparse_params = SolParams()
        if not isinstance(sparse_params, SolParams):
            raise TypeError("SOLCuTeDSLAttention requires SolParams")
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        if self.num_kv_heads != self.num_heads:
            # Not an assert: `python -O` strips those, and the kernel wrapper
            # would then see unequal Q/K shapes and quietly take its dense
            # fallback instead of rejecting an unsupported configuration.
            raise ValueError(
                f"SOL is MHA-only (num_kv_heads == num_heads), got "
                f"num_kv_heads={self.num_kv_heads}, num_heads={self.num_heads}. "
                f"GQA/MQA is not supported."
            )
        self.dtype = dtype
        self.tau = sparse_params.tau
        self.thresh_type = sparse_params.thresh_type
        self.disabled_until_timestep = sparse_params.disabled_until_timestep
        self.dense_layers = sparse_params.dense_layers

        # SOL's dense steps must run the backend the user selected. Without
        # this they ran torch SDPA while a `backend: CUTEDSL` baseline ran
        # cute_dsl_fmha_fwd, so candidate and reference differed on the dense
        # steps too -- measured at LPIPS 0.214 on Wan2.2-T2V-A14B with sparsity
        # switched off entirely, against a 0.25 gate.
        from ...cute_dsl.fmha import CuTeDSLAttention

        # The only backend that consumes `key_padding_mask` (CuTeDSL's `_fwd`
        # swallows it via **kwargs, silently). Masked self-attention is routed
        # here rather than to the mask-blind sparse kernel or to `_inner`.
        self._vanilla = VanillaAttention(
            layer_idx=layer_idx,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=self.num_kv_heads,
            dtype=dtype,
        )
        self._inner = CuTeDSLAttention(
            layer_idx=layer_idx,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=self.num_kv_heads,
            dtype=dtype,
        )
        # Resolve availability once at construction to keep capability checks
        # and attribute initialization out of `_dense` during torch.compile tracing.
        self._cute_dense_ok = _cute_dense_available()
        if not self._cute_dense_ok:
            logger.warning_once(
                "[sol-attn] the CuTe DSL FMHA kernel cannot serve this device; dense "
                "steps will use torch SDPA. Numerics will differ from a `backend: "
                "CUTEDSL` dense baseline.",
                key="sol_attn_dense_backend_unavailable",
            )

    # The `.item()` in here would graph-break the enclosing block once per
    # attention layer, so keep it in eager (as cute_dsl/fmha.py and VSA's
    # `_get_vsa_inputs` do). Returns a host-side bool, so the dense and sparse
    # phases still compile as separate graphs -- they run different kernels.
    @torch.compiler.disable
    def _dense_by_step(self, timestep: Any) -> bool:
        # Under CUDA-graph capture the runner has already resolved the phase
        # host-side (it is part of the graph key); reading the tensor here
        # would `.item()` inside capture, which CUDA forbids.
        phase = resolved_extra_key("sparse_attn_phase")
        if phase is None:
            phase = graph_phase_for_timestep(
                timestep,
                disabled_until_timestep=self.disabled_until_timestep,
            )
        if phase is None:
            # Fail open, matching the CuTeDSL skip-softmax path: without a
            # timestep we cannot tell which phase we are in, so run the
            # sparse kernel rather than silently forcing dense forever.
            # This degrades quality rather than raising, so say so once.
            logger.warning_once(
                "SolAttentionConfig.disabled_until_timestep="
                f"{self.disabled_until_timestep} is set, but no `timestep` reached "
                "the SOL forward call. The dense prefix it requests will not "
                "be applied. Ensure the pipeline passes a normalized timestep, or "
                "unset disabled_until_timestep.",
                key="sol_attn_missing_timestep",
            )
            return False
        return phase == 0

    @staticmethod
    def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Dense attention via torch SDPA, for devices CuTe DSL cannot serve.

        Honors the same two mask kwargs the backends accept: ``attention_mask``
        (``CAUSAL``) and ``key_padding_mask`` (``[B, S_kv]`` bool, True = valid).
        """
        is_causal = kwargs.get("attention_mask", PredefinedAttentionMask.FULL) == (
            PredefinedAttentionMask.CAUSAL
        )
        key_padding_mask = kwargs.get("key_padding_mask")
        attn_mask = None
        if key_padding_mask is not None:
            # SDPA wants a broadcastable bool mask where True = attend.
            attn_mask = key_padding_mask.to(torch.bool)[:, None, None, :]
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attn_mask,
            is_causal=is_causal and attn_mask is None,
        ).transpose(1, 2)

    def _delegate(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """Hand the call to the dense backend of the same family.

        ``_cute_dense_ok`` answers "can this *device* run the kernel", decided at
        construction; ``q.is_cuda`` answers "is this *tensor* on it". Both are
        needed: the construction-time probe inspects the current CUDA device, so
        it says yes on a GPU host even when a caller passes CPU tensors.
        """
        if kwargs.get("key_padding_mask") is not None:
            # CuTeDSL does not consume `key_padding_mask`, so staying in-family
            # here would silently attend to padded tokens; VANILLA honors it.
            # VANILLA works in HND ([B, H, S, D]); this backend is NHD.
            if self._vanilla.preferred_layout == AttentionTensorLayout.HND:
                out = self._vanilla.forward(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), **kwargs
                )
                return out.transpose(1, 2)
            return self._vanilla.forward(q, k, v, **kwargs)
        if self._cute_dense_ok and q.is_cuda:
            # CAUSAL is fine here: CuTeDSL handles it via `_prepare_inputs`.
            return self._inner.forward(q, k, v, **kwargs)
        return self._sdpa(q, k, v, **kwargs)

    def _can_serve(self, q: torch.Tensor, k: torch.Tensor, **kwargs: Any) -> bool:
        """Whether the sparse kernel applies to this particular call.

        Everything false here is delegated (see ``_delegate``). Deciding it from the
        tensors, per call, is deliberate, and it is why ``modules/attention.py``
        has no ``SEPARATE_QKV`` rule for SOL: ``qkv_mode`` describes how
        Q/K/V are *projected*, not whether K/V come from another sequence, so a
        construction-time rule keyed on it mistakes self-attention for
        cross-attention wherever that mode is chosen for other reasons --
        Qwen-Image always, and WAN's ``attn1`` under async Ulysses -- silently
        costing those modules their configured backend.
        """
        # Cross-attention: K/V come from another sequence, and SOL's
        # routing assumes one self-attending sequence. Unequal Q/K lengths are
        # a heuristic for that, not a definition: a cross-attention call whose
        # context happens to match the query length is not caught here. The
        # `Attention` module knows the answer (`encoder_hidden_states`), but the
        # backend `forward` kwargs carry no such flag yet; TRTLLM-16475 tracks
        # threading an explicit signal through and retiring this check.
        if k.shape[1] != q.shape[1]:
            return False
        # Masks: the sparse kernel is noncausal and takes no mask, so a masked
        # call -- HunyuanVideo1.5 / GLM-Image `key_padding_mask`, Cosmos3
        # `CAUSAL` -- must go to a backend that honors it. Equal Q/K lengths do
        # not imply "unmasked self-attention".
        if kwargs.get("key_padding_mask") is not None:
            return False
        if (
            kwargs.get("attention_mask", PredefinedAttentionMask.FULL)
            != PredefinedAttentionMask.FULL
        ):
            return False
        if self.layer_idx in self.dense_layers:
            return False
        if self.disabled_until_timestep is not None and self._dense_by_step(kwargs.get("timestep")):
            return False
        return True

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """q, k, v: [B, S, H, D] (NHD), same original token order in and out."""
        if not self._can_serve(q, k, **kwargs):
            return self._delegate(q, k, v, **kwargs)
        return _sol_attn_run(
            q,
            k,
            v,
            tau=self.tau,
            thresh_type=self.thresh_type,
            # Shape/dtype/arch ineligibility is only detectable inside the
            # wrapper, so that last delegation happens through this hook.
            dense_fn=lambda a, b, c: self._delegate(a, b, c, **kwargs),
        )

    @classmethod
    def support_lse(cls) -> bool:
        return False

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        return AttentionTensorLayout.NHD

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return False


class SOLTrtllmAttention(TrtllmAttention):
    """Predict SOL routes inside the core prediction hook, then execute them
    through the generic block-sparse FMHA."""

    def __init__(self, *, sparse_params: SolParams | None = None, **kwargs) -> None:
        if not isinstance(sparse_params, SolParams):
            raise TypeError("SOLTrtllmAttention requires SolParams")
        self.sol_params = sparse_params
        super().__init__(sparse_params=None, **kwargs)

    @property
    def timestep_cutoff(self) -> Optional[float]:
        """SOL keeps its parameters outside the core ``sparse_params`` slot."""

        return self.sol_params.disabled_until_timestep

    @property
    def dense_layers(self) -> frozenset[int]:
        return self.sol_params.dense_layers

    def block_sparse_attn_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: TrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> BlockSparseForwardInputs | None:
        """Return SOL routes for sparse calls and ``None`` for dense calls.

        ``q``, ``k``, and ``v`` arrive in the flattened ``[B*S, H*D]`` core
        layout; the batch layout comes from ``metadata`` and the timestep,
        already prepared by the wrapper forward, from ``forward_args``.
        """

        if not self.should_use_sparse(forward_args.timestep):
            return None

        if self.quant_attention_config is not None:
            raise ValueError("SOL sparse execution does not support quant_attention_config")
        if not any(
            isinstance(fmha, PrimsTSBlockSparseFmha) for fmha in self._fmha_manager.fmha_libs
        ):
            raise RuntimeError("SOL sparse execution requires PrimTS block-sparse FMHA")
        if forward_args.attention_mask != PredefinedAttentionMask.FULL:
            raise ValueError("SOL sparse execution requires a full attention mask")
        if k is None or v is None:
            raise ValueError("SOL sparse execution requires separate q, k, and v tensors")

        batch_size = metadata.num_seqs
        seq_len = metadata.max_seq_len
        num_tokens = batch_size * seq_len
        if q.shape[0] != num_tokens or k.shape[0] != num_tokens or v.shape[0] != num_tokens:
            raise ValueError(
                "SOL sparse execution supports only uniform-length self-attention; "
                f"got {q.shape[0]} query and {k.shape[0]} key tokens for "
                f"{batch_size} sequences of length {seq_len}"
            )

        # The VisualGen wrapper compacts the flattened tensors once; these views
        # are shared between prediction and the generic block-sparse FMHA.
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        unsupported_reason = sol_predictor.support_reason(q, k, v)
        if unsupported_reason is not None:
            raise ValueError(unsupported_reason)

        outputs = sol_predictor.predict(
            q,
            k,
            v,
            tau=self.sol_params.tau,
            sm_scale=get_bmm1_scale(self),
            thresh_type=self.sol_params.thresh_type,
        )
        return BlockSparseForwardInputs(
            q_block_size=BLOCK_SIZE,
            kv_block_size=BLOCK_SIZE,
            exact_block_bits=outputs.exact_block_bits,
            k_summary=outputs.k_summary,
            v_summary=outputs.v_summary,
        )

    @classmethod
    def support_fused_qkv(cls) -> bool:
        """SOL prediction requires separate Q, K, and V tensors."""

        return False


__all__ = ["SOLCuTeDSLAttention", "SOLTrtllmAttention"]
