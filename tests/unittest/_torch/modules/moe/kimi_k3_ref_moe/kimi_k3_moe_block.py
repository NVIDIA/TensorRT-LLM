# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KimiK3SparseMoeBlock — test-only Kimi K3 sparse MoE reference.

Used by ``test_kimi_k3_situ_moe.py`` as the HF-parity reference for the
native SiTU kernel path; the serving runtime (``KimiK3MoERuntime`` in
``modeling_kimi_linear.py``) does not use it. Structural mirror of HF
``KimiSparseMoeBlock`` at ``modeling_kimi.py:806-918`` end to end:

* :class:`KimiK3ReferenceMoEGate` for eager reference routing.
* :class:`KimiK3RoutedExpertBank` — per-expert MXFP4-packed
  ``w1 / w2 / w3`` linear weights (group_size=32), dequantized on the
  fly during the Python fallback path.
* :class:`KimiK3MLP` shared expert stack (``num_shared_experts`` fused
  into one KimiMLP with ``intermediate_size = moe_intermediate_size *
  num_shared_experts``).
* Latent projections ``routed_expert_down_proj`` /
  ``routed_expert_up_proj`` around the routed compute when
  ``routed_expert_hidden_size`` is set, plus optional
  ``routed_expert_norm`` (:class:`KimiK3RMSNorm`).

Two mutually exclusive kernel paths coexist:

* ``use_fused_cubin=False`` (default) — Python fallback with MXFP4
  bank + activation. Byte-exact HF parity under random weights when
  weights are canonicalized via :func:`copy_hf_moe_block_weights`.
* ``use_fused_cubin=True`` — native in-tree SiTU path through
  ``torch.ops.trtllm.mxe4m3_mxe2m1_block_scale_moe_runner``
  (``act_type=SiTu``) on checkpoint-derived MXFP4 weights. The same
  MXFP4 bank is the quantization source of truth for both paths, so
  fused-vs-fallback comparisons differ only by activation quantization
  (MXFP8) and kernel arithmetic. Routing uses the K3 gate's real
  top-k via the op's ``topk_weights``/``topk_ids`` bypass; weights must
  be loaded through :func:`copy_hf_moe_block_weights` (or
  :meth:`KimiK3SparseMoeBlock.build_fused_weights`) before forward.

Two mutation flags cover the negative controls required by AC4:

* ``missing_shared_experts_mutation`` — skip the
  ``+ shared_experts(identity)`` addition.
* ``non_situ_activation_mutation`` — replace SiTU with a SiLU-based
  activation in both routed and shared experts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
from _torch.modules.moe.kimi_k3_ref_moe._moe_kernels import (
    assert_native_situ_supported,
    invoke_native_situ_moe,
    make_situ_alpha_beta,
    pack_routed_expert_weights,
)
from _torch.modules.moe.kimi_k3_ref_moe._mxfp4 import (
    DEFAULT_GROUP_SIZE,
    dequantize_last_dim_mxfp4,
    quantize_last_dim_mxfp4,
)
from _torch.modules.moe.kimi_k3_ref_moe.kimi_k3_mlp_test_utils import KimiK3MLP, NonSituActivation
from torch import nn

from tensorrt_llm._torch.models.modeling_kimi_linear import KimiK3MoEGate, KimiK3RMSNorm
from tensorrt_llm._torch.modules.situ import SituAndMul


class KimiK3ReferenceMoEGate(KimiK3MoEGate):
    """Eager HF-compatible Kimi K3 routing reference."""

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.compute_logits(hidden_states)
        num_tokens = logits.shape[0]
        if self.moe_router_activation_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            scores = logits.softmax(dim=1)
        scores = scores.view(num_tokens, -1)

        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        if self.num_expert_group > 1 and self.num_expert_group > self.topk_group:
            group_scores = (
                scores_for_choice.view(num_tokens, self.num_expert_group, -1)
                .topk(2, dim=-1)[0]
                .sum(dim=-1)
            )
            group_indices = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_indices, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    num_tokens,
                    self.num_expert_group,
                    self.num_experts // self.num_expert_group,
                )
                .reshape(num_tokens, -1)
            )
            scores_for_selection = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        else:
            scores_for_selection = scores_for_choice

        topk_indices = torch.topk(scores_for_selection, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_indices)
        if self.top_k > 1 and self.moe_renormalize:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_indices, topk_weights * self.routed_scaling_factor


class KimiK3RoutedExpertBank(nn.Module):
    """MXFP4-packed routed expert weight bank.

    Stores per-expert ``w1``, ``w2``, ``w3`` (matching HF
    ``KimiBlockSparseMLP``'s naming) in ``mxfp4-pack-quantized`` form
    with ``group_size=32``.

    Shapes (all ``num_experts`` on the leading dim):

    * ``w1_packed`` / ``w3_packed``: ``uint8 [E, intermediate, hidden // 2]``
    * ``w2_packed``: ``uint8 [E, hidden, intermediate // 2]``
    * ``w1_scales`` / ``w3_scales``: ``uint8 [E, intermediate, hidden // group_size]``
    * ``w2_scales``: ``uint8 [E, hidden, intermediate // group_size]``

    ``hidden`` is the effective routed-expert input size (i.e.
    ``moe_hidden_size = routed_expert_hidden_size`` when the latent
    path is active, else ``config.hidden_size``) and ``intermediate`` is
    ``moe_intermediate_size``.

    All tensors are registered as buffers (not Parameters) since MXFP4
    packed weights are integer-typed and not differentiable inputs for
    the module tests. This module never trains.
    """

    def __init__(
        self,
        *,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        group_size: int = DEFAULT_GROUP_SIZE,
        activation: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        assert hidden_size % group_size == 0, (
            f"hidden_size {hidden_size} not divisible by group_size {group_size}"
        )
        assert intermediate_size % group_size == 0, (
            f"intermediate_size {intermediate_size} not divisible by group_size {group_size}"
        )
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.group_size = group_size
        self.activation = (
            activation if activation is not None else SituAndMul(beta=4.0, linear_beta=25.0)
        )

        h = hidden_size
        i = intermediate_size
        g = group_size
        dev = device

        # w1, w3: [I, H]. w2: [H, I].
        # NOTE: allocated with torch.empty (not zeros) so the buffers stay on
        # the meta device under the PyTorch backend's MetaInitMode and get
        # materialized directly on CUDA; real checkpoints overwrite every
        # element, and tests populate via store_expert before any forward.
        self.register_buffer(
            "w1_packed",
            torch.empty(num_experts, i, h // 2, dtype=torch.uint8, device=dev),
        )
        self.register_buffer(
            "w1_scales",
            torch.empty(num_experts, i, h // g, dtype=torch.uint8, device=dev),
        )
        self.register_buffer(
            "w3_packed",
            torch.empty(num_experts, i, h // 2, dtype=torch.uint8, device=dev),
        )
        self.register_buffer(
            "w3_scales",
            torch.empty(num_experts, i, h // g, dtype=torch.uint8, device=dev),
        )
        self.register_buffer(
            "w2_packed",
            torch.empty(num_experts, h, i // 2, dtype=torch.uint8, device=dev),
        )
        self.register_buffer(
            "w2_scales",
            torch.empty(num_experts, h, i // g, dtype=torch.uint8, device=dev),
        )

    def store_expert(
        self,
        expert_idx: int,
        w1_fp32: torch.Tensor,
        w2_fp32: torch.Tensor,
        w3_fp32: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize fp32 ``[out, in]`` weights and store them.

        Returns the fp32 round-tripped (canonical) values for each
        weight so callers can initialise a reference module with the
        exact numerical values the bank now holds.
        """
        assert w1_fp32.shape == (self.intermediate_size, self.hidden_size), (
            w1_fp32.shape,
            (self.intermediate_size, self.hidden_size),
        )
        assert w2_fp32.shape == (self.hidden_size, self.intermediate_size), (
            w2_fp32.shape,
            (self.hidden_size, self.intermediate_size),
        )
        assert w3_fp32.shape == w1_fp32.shape

        w1_packed, w1_scales = quantize_last_dim_mxfp4(w1_fp32, self.group_size)
        w2_packed, w2_scales = quantize_last_dim_mxfp4(w2_fp32, self.group_size)
        w3_packed, w3_scales = quantize_last_dim_mxfp4(w3_fp32, self.group_size)
        self.w1_packed[expert_idx].copy_(w1_packed)
        self.w1_scales[expert_idx].copy_(w1_scales)
        self.w3_packed[expert_idx].copy_(w3_packed)
        self.w3_scales[expert_idx].copy_(w3_scales)
        self.w2_packed[expert_idx].copy_(w2_packed)
        self.w2_scales[expert_idx].copy_(w2_scales)

        w1_canon = dequantize_last_dim_mxfp4(w1_packed, w1_scales, self.group_size)
        w2_canon = dequantize_last_dim_mxfp4(w2_packed, w2_scales, self.group_size)
        w3_canon = dequantize_last_dim_mxfp4(w3_packed, w3_scales, self.group_size)
        return w1_canon, w2_canon, w3_canon

    def dequantize_expert(self, expert_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w1 = dequantize_last_dim_mxfp4(
            self.w1_packed[expert_idx], self.w1_scales[expert_idx], self.group_size
        )
        w2 = dequantize_last_dim_mxfp4(
            self.w2_packed[expert_idx], self.w2_scales[expert_idx], self.group_size
        )
        w3 = dequantize_last_dim_mxfp4(
            self.w3_packed[expert_idx], self.w3_scales[expert_idx], self.group_size
        )
        return w1, w2, w3

    def forward_expert(self, expert_idx: int, tokens: torch.Tensor) -> torch.Tensor:
        """Run a single expert on ``tokens`` ``[N, H]``.

        Matches HF ``KimiBlockSparseMLP.forward`` with the ``situ``
        branch: ``cat[w1(x), w3(x)] → SituAndMul → w2``. Compute is in
        ``tokens.dtype``; weights are dequantized to canonical fp32 then
        cast down. Because canonical MXFP4 magnitudes are exactly
        representable in bf16, the cast is lossless.
        """
        w1_f, w2_f, w3_f = self.dequantize_expert(expert_idx)
        dt = tokens.dtype
        w1 = w1_f.to(dt) if w1_f.dtype != dt else w1_f
        w2 = w2_f.to(dt) if w2_f.dtype != dt else w2_f
        w3 = w3_f.to(dt) if w3_f.dtype != dt else w3_f
        gate = tokens @ w1.t()
        up = tokens @ w3.t()
        gate_up = torch.cat([gate, up], dim=-1)
        act = self.activation(gate_up)
        y = act @ w2.t()
        return y


@dataclass
class MoEBlockProvenance:
    """Provenance record returned by :func:`copy_hf_moe_block_weights`."""

    n_experts: int
    shared_expert_names: List[str]
    routed_expert_layout: Tuple[int, int]
    latent: bool
    latent_use_norm: bool
    canonicalized: bool


class KimiK3SparseMoeBlock(nn.Module):
    """Kimi K3 sparse MoE block — mirrors HF ``KimiSparseMoeBlock``.

    Parameters
    ----------
    config
        ``KimiLinearConfig``-like — provides ``hidden_size``,
        ``num_experts``, ``num_experts_per_token``, ``moe_intermediate_size``,
        ``moe_renormalize``, ``routed_expert_hidden_size`` (optional;
        enables the latent path), ``latent_moe_use_norm``,
        ``num_shared_experts``, ``activation_situ_beta``,
        ``activation_situ_linear_beta``, ``rms_norm_eps``.
    missing_shared_experts_mutation
        Mutation control — skip the ``+ shared_experts(identity)`` add.
    non_situ_activation_mutation
        Mutation control — replace SiTU with SiLU-based activation in
        both routed and shared experts.
    use_fused_cubin
        When True, ``forward()`` dispatches routed compute through the
        in-tree ``torch.ops.trtllm.mxe4m3_mxe2m1_block_scale_moe_runner``
        custom op with ``act_type=SiTu``. The MXFP4 ``expert_bank`` is
        still allocated (it is the checkpoint-quantization source of
        truth); the fused device buffers are derived from it by
        :meth:`build_fused_weights`, which
        :func:`copy_hf_moe_block_weights` calls automatically. Forward
        raises if the fused weights have not been built — there is no
        random-weight fallback.
    dtype
        Dtype used for latent projections, shared experts, RMSNorm
        weight. Default fp32.
        Set to bf16 to match HF's bf16 forward at real K3 dims.
    """

    def __init__(
        self,
        config: Any,
        *,
        missing_shared_experts_mutation: bool = False,
        non_situ_activation_mutation: bool = False,
        use_fused_cubin: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.moe_renormalize = config.moe_renormalize

        self.use_latent_moe = getattr(config, "routed_expert_hidden_size", None) is not None
        self.moe_hidden_size = (
            config.routed_expert_hidden_size if self.use_latent_moe else config.hidden_size
        )
        self.latent_moe_use_norm = getattr(config, "latent_moe_use_norm", False)

        # EP sharding trivialized — module tests run on one GPU.
        self.ep_size = 1
        self.experts_per_rank = config.num_experts
        self.ep_rank = 0

        self.missing_shared_experts_mutation = missing_shared_experts_mutation
        self.non_situ_activation_mutation = non_situ_activation_mutation
        self.use_fused_cubin = use_fused_cubin
        self._proj_dtype = dtype
        self._cubin_call_count = 0

        situ_beta = getattr(config, "activation_situ_beta", 4.0)
        situ_linear_beta = getattr(config, "activation_situ_linear_beta", 25.0)
        self._situ_beta = situ_beta
        self._situ_linear_beta = situ_linear_beta

        if non_situ_activation_mutation:
            routed_activation: nn.Module = NonSituActivation()
        else:
            routed_activation = SituAndMul(beta=situ_beta, linear_beta=situ_linear_beta)

        self.gate = KimiK3ReferenceMoEGate(config, device=device)

        # Routed expert storage — the MXFP4 bank is always the checkpoint
        # quantization source of truth. The fused path derives its packed
        # and shuffled device buffers from the same bank so both paths see
        # identical canonical weights.
        self.expert_bank = KimiK3RoutedExpertBank(
            num_experts=config.num_experts,
            hidden_size=self.moe_hidden_size,
            intermediate_size=config.moe_intermediate_size,
            activation=routed_activation,
            device=device,
        )
        self._fused_bank_ready = False
        self.gemm1_weights: Optional[torch.Tensor] = None
        self.gemm1_weights_scale: Optional[torch.Tensor] = None
        self.gemm2_weights: Optional[torch.Tensor] = None
        self.gemm2_weights_scale: Optional[torch.Tensor] = None
        self._gemm1_alpha: Optional[torch.Tensor] = None
        self._gemm1_beta: Optional[torch.Tensor] = None
        if use_fused_cubin:
            # Fail before any weight processing when the platform cannot run
            # the fused path at all.
            assert_native_situ_supported(
                hidden_size=self.moe_hidden_size,
                intermediate_size=config.moe_intermediate_size,
            )
            if non_situ_activation_mutation:
                raise RuntimeError(
                    "non_situ_activation_mutation is a Python-reference mutation "
                    "control; it cannot be combined with use_fused_cubin=True"
                )

        # Shared experts — HF fuses ``num_shared_experts`` KimiMLPs into
        # one, with ``intermediate_size = moe_intermediate_size *
        # num_shared_experts``. Unquantized Linear in ``_proj_dtype``.
        self.num_shared_experts = getattr(config, "num_shared_experts", None)
        if self.num_shared_experts is not None:
            shared_activation: nn.Module = (
                NonSituActivation()
                if non_situ_activation_mutation
                else SituAndMul(beta=situ_beta, linear_beta=situ_linear_beta)
            )
            self.shared_experts = KimiK3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=(config.moe_intermediate_size * self.num_shared_experts),
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
                activation=shared_activation,
                dtype=dtype,
                device=device,
            )
        else:
            self.shared_experts = None

        # Latent projections around routed compute.
        if self.use_latent_moe:
            self.routed_expert_down_proj = nn.Linear(
                config.hidden_size,
                self.moe_hidden_size,
                bias=False,
                dtype=dtype,
                device=device,
            )
            self.routed_expert_up_proj = nn.Linear(
                self.moe_hidden_size,
                config.hidden_size,
                bias=False,
                dtype=dtype,
                device=device,
            )
            if self.latent_moe_use_norm:
                self.routed_expert_norm = KimiK3RMSNorm(
                    self.moe_hidden_size,
                    eps=getattr(config, "rms_norm_eps", 1e-5),
                    dtype=dtype,
                    device=device,
                )
            else:
                self.routed_expert_norm = None
        else:
            self.routed_expert_down_proj = None
            self.routed_expert_up_proj = None
            self.routed_expert_norm = None

    # ------------------------------------------------------------------
    # Fused path — native in-tree TRTLLM-Gen SiTU dispatch.
    # ------------------------------------------------------------------

    def build_fused_weights(self) -> None:
        """Derive the fused TRTLLM-Gen device buffers from ``expert_bank``.

        Packs the bank's per-expert MXFP4 tensors into the padded and
        shuffled ``gemm1_*``/``gemm2_*`` layout expected by
        ``mxe4m3_mxe2m1_block_scale_moe_runner`` (w3 first / w1 second,
        opposite of the HF gate-first order) and materializes the
        per-expert alpha/beta CUDA buffers. Must be called after the bank
        holds checkpoint weights (``copy_hf_moe_block_weights`` does this
        automatically for fused-mode blocks).
        """
        assert self.use_fused_cubin, "build_fused_weights requires use_fused_cubin=True"
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        packed = pack_routed_expert_weights(
            w1_packed=self.expert_bank.w1_packed,
            w1_scales=self.expert_bank.w1_scales,
            w3_packed=self.expert_bank.w3_packed,
            w3_scales=self.expert_bank.w3_scales,
            w2_packed=self.expert_bank.w2_packed,
            w2_scales=self.expert_bank.w2_scales,
            device=device,
        )
        self.gemm1_weights = packed["gemm1_weights"]
        self.gemm1_weights_scale = packed["gemm1_weights_scale"]
        self.gemm2_weights = packed["gemm2_weights"]
        self.gemm2_weights_scale = packed["gemm2_weights_scale"]
        self._gemm1_alpha, self._gemm1_beta = make_situ_alpha_beta(
            local_num_experts=self.num_experts,
            situ_beta=self._situ_beta,
            situ_linear_beta=self._situ_linear_beta,
            device=device,
        )
        self._fused_bank_ready = True

    def _moe_infer_fused(
        self,
        routed_in: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Fused variant of :meth:`_moe_infer` via the in-tree SiTU op.

        ``topk_ids``/``topk_weights`` come from the K3 gate (bypass
        contract; weights already renormalized and scaled).
        """
        if not (self.use_fused_cubin and self._fused_bank_ready):
            raise RuntimeError(
                "fused SiTU path invoked, but fused weights were never built. "
                "Load checkpoint weights via copy_hf_moe_block_weights() or call "
                "build_fused_weights(); there is no random-weight fallback."
            )
        result = invoke_native_situ_moe(
            hidden_states=routed_in,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            gemm1_weights=self.gemm1_weights,
            gemm1_weights_scale=self.gemm1_weights_scale,
            gemm2_weights=self.gemm2_weights,
            gemm2_weights_scale=self.gemm2_weights_scale,
            gemm1_alpha=self._gemm1_alpha,
            gemm1_beta=self._gemm1_beta,
            num_experts=self.num_experts,
            top_k=self.top_k,
            valid_hidden_size=self.moe_hidden_size,
            valid_intermediate_size=self.config.moe_intermediate_size,
        )
        self._cubin_call_count += 1
        return result

    # ------------------------------------------------------------------
    # Forward + Python fallback MoE inference.
    # ------------------------------------------------------------------

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Reproduce HF ``KimiSparseMoeBlock.forward``.

        Python-fallback path:

        1. ``identity = hidden_states`` (used for shared expert input).
        2. Gate → ``topk_idx, topk_weight`` from raw ``hidden_states``.
        3. Reshape hidden to ``[T, hidden_size]``.
        4. If latent, ``hidden = routed_expert_down_proj(hidden)``.
        5. :meth:`_moe_infer` — dispatch tokens across selected experts,
           weight-mix outputs, reshape back to ``[..., moe_hidden]``.
        6. If latent, apply ``routed_expert_norm`` (optional) and
           ``routed_expert_up_proj``.
        7. Reshape back to ``hidden_states`` shape.
        8. Add ``shared_experts(identity)`` if configured (mutation:
           skip).

        Fused path (``use_fused_cubin=True``): same skeleton, but step 5
        is :meth:`_moe_infer_fused`, which invokes the in-tree
        ``mxe4m3_mxe2m1_block_scale_moe_runner`` op with
        ``act_type=SiTu``. The gate at step 2 runs in both modes — the
        fused path feeds its real top-k through the op's
        ``topk_weights``/``topk_ids`` bypass. Steps 4/6/8 still run
        (outside the fused kernel's scope).
        """
        identity = hidden_states
        orig_shape = hidden_states.shape

        topk_idx, topk_weight = self.gate(hidden_states)

        flat = hidden_states.view(-1, self.hidden_size)

        if self.use_latent_moe:
            routed_in = self.routed_expert_down_proj(flat.to(self._proj_dtype))
        else:
            routed_in = flat.to(self._proj_dtype)

        if self.use_fused_cubin:
            routed_in_bf16 = (
                routed_in if routed_in.dtype == torch.bfloat16 else routed_in.to(torch.bfloat16)
            )
            y = self._moe_infer_fused(routed_in_bf16, topk_idx, topk_weight)
            if y.dtype != self._proj_dtype:
                y = y.to(self._proj_dtype)
        else:
            y = self._moe_infer(routed_in, topk_idx, topk_weight)

        if self.use_latent_moe:
            if self.routed_expert_norm is not None:
                y = self.routed_expert_norm(y)
            y = self.routed_expert_up_proj(y)

        y = y.view(*orig_shape)

        if self.shared_experts is not None and not self.missing_shared_experts_mutation:
            shared = self.shared_experts(identity.to(self._proj_dtype))
            y = y + shared.to(y.dtype)
        return y.to(hidden_states.dtype)

    def _moe_infer(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Byte-for-byte port of HF ``KimiSparseMoeBlock.moe_infer``.

        HF does per-expert compute in the token dtype (the expert Linear
        layers inherit HF's module dtype), and casts to
        ``topk_weight.dtype`` for the weighted sum. We mirror that so
        both fp32 and bf16 HF modes stay byte-exact.
        """
        cnts = topk_ids.new_zeros((topk_ids.shape[0], self.num_experts))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]

        tokens_per_expert_cpu = tokens_per_expert.cpu().tolist()

        outputs: List[torch.Tensor] = []
        start = 0
        for i, n_tokens in enumerate(tokens_per_expert_cpu):
            end = start + int(n_tokens)
            if n_tokens == 0:
                continue
            tokens_for_i = sorted_tokens[start:end]
            expert_out = self.expert_bank.forward_expert(i, tokens_for_i)
            outputs.append(expert_out)
            start = end

        outs = torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out


# ---------------------------------------------------------------------------
# HF → K3 copy helper.
# ---------------------------------------------------------------------------


def copy_hf_moe_block_weights(
    hf: nn.Module,
    k3: KimiK3SparseMoeBlock,
) -> MoEBlockProvenance:
    """Copy weights from HF ``KimiSparseMoeBlock`` to a K3 block.

    Steps:

    1. Copy gate params (identity name mapping).
    2. Copy latent down/up projections and optional RMSNorm.
    3. Copy ``shared_experts`` (fused KimiMLP) parameters.
    4. For each routed expert:
       a. Read HF ``experts[i].{w1,w2,w3}.weight`` fp32.
       b. Quantize and store in ``k3.expert_bank``.
       c. Retrieve canonical fp32 (quantize→dequantize) values.
       d. Overwrite HF's Linear weights with the canonical values so
          both modules see byte-identical numbers on forward.
    5. When ``k3.use_fused_cubin=True``, additionally derive the fused
       TRTLLM-Gen device buffers from the freshly loaded bank via
       :meth:`KimiK3SparseMoeBlock.build_fused_weights`. Both paths then
       hold the same canonical checkpoint weights.

    Returns provenance metadata for logging.
    """
    with torch.no_grad():
        k3.gate.weight.data.copy_(hf.gate.weight.data.to(k3.gate.weight.dtype))
        k3.gate.e_score_correction_bias.data.copy_(
            hf.gate.e_score_correction_bias.data.to(k3.gate.e_score_correction_bias.dtype)
        )

    if k3.use_latent_moe:
        with torch.no_grad():
            k3.routed_expert_down_proj.weight.data.copy_(
                hf.routed_expert_down_proj.weight.data.to(k3.routed_expert_down_proj.weight.dtype)
            )
            k3.routed_expert_up_proj.weight.data.copy_(
                hf.routed_expert_up_proj.weight.data.to(k3.routed_expert_up_proj.weight.dtype)
            )
        if k3.routed_expert_norm is not None:
            assert hasattr(hf, "routed_expert_norm"), (
                "latent_moe_use_norm=True but HF has no routed_expert_norm"
            )
            with torch.no_grad():
                k3.routed_expert_norm.weight.data.copy_(
                    hf.routed_expert_norm.weight.data.to(k3.routed_expert_norm.weight.dtype)
                )

    shared_names: List[str] = []
    hf_shared_experts = getattr(hf, "shared_experts", None)
    if k3.shared_experts is None:
        assert hf_shared_experts is None, (
            "HF block has shared_experts but K3 block has none; "
            "shared-expert weights would be silently dropped"
        )
    else:
        assert hf_shared_experts is not None, (
            "K3 block has shared_experts but HF block has none; "
            "shared-expert weights would stay randomly initialized"
        )
        with torch.no_grad():
            gate_up_fused = torch.cat(
                [
                    hf.shared_experts.gate_proj.weight.data,
                    hf.shared_experts.up_proj.weight.data,
                ],
                dim=0,
            ).to(k3.shared_experts.gate_up_proj.weight.dtype)
            k3.shared_experts.gate_up_proj.weight.data.copy_(gate_up_fused)
            k3.shared_experts.down_proj.weight.data.copy_(
                hf.shared_experts.down_proj.weight.data.to(k3.shared_experts.down_proj.weight.dtype)
            )
        shared_names = ["gate_up_proj.weight", "down_proj.weight"]

    for i in range(k3.num_experts):
        expert = hf.experts[i]
        w1 = expert.w1.weight.data.to(torch.float32)
        w2 = expert.w2.weight.data.to(torch.float32)
        w3 = expert.w3.weight.data.to(torch.float32)
        w1c, w2c, w3c = k3.expert_bank.store_expert(i, w1, w2, w3)
        # bf16 exactly represents every MXFP4 magnitude
        # {0, 0.5, 1, 1.5, 2, 3, 4, 6} scaled by 2^e, so the cast to HF's
        # own weight dtype is lossless and byte-parity holds under bf16
        # HF too.
        with torch.no_grad():
            expert.w1.weight.data.copy_(w1c.to(expert.w1.weight.dtype))
            expert.w2.weight.data.copy_(w2c.to(expert.w2.weight.dtype))
            expert.w3.weight.data.copy_(w3c.to(expert.w3.weight.dtype))

    if k3.use_fused_cubin:
        k3.build_fused_weights()

    return MoEBlockProvenance(
        n_experts=k3.num_experts,
        shared_expert_names=shared_names,
        routed_expert_layout=(k3.moe_hidden_size, k3.config.moe_intermediate_size),
        latent=k3.use_latent_moe,
        latent_use_norm=k3.routed_expert_norm is not None,
        canonicalized=True,
    )
