# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KimiK3SparseMoeBlock — in-tree Kimi K3 sparse MoE module.

Structural mirror of HF ``KimiSparseMoeBlock`` at
``modeling_kimi.py:806-918`` end to end:

* :class:`KimiK3MoEGate` for routing (see :mod:`kimi_k3_moe_gate`).
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
* ``use_fused_cubin=True`` — Private-SiTU
  ``trtllm_mxint4_block_scale_moe(is_private=True,
  activation_type=Situ)`` invocation on a locally-allocated MXINT4
  bank. Increments ``_cubin_call_count`` on every dispatch so
  module-path AC5 evidence can distinguish a real cubin invocation
  from a silent no-op.

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
from torch import nn

from ._mlp import KimiK3MLP, KimiK3RMSNorm, NonSituActivation, SituAndMul
from ._moe_kernels import invoke_private_situ_moe
from ._mxfp4 import DEFAULT_GROUP_SIZE, dequantize_last_dim_mxfp4, quantize_last_dim_mxfp4
from .kimi_k3_moe_gate import KimiK3MoEGate


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
        When True, ``forward()`` dispatches routed compute through
        ``flashinfer.fused_moe.trtllm_mxint4_block_scale_moe(is_private=True,
        activation_type=ActivationType.Situ.value)`` on a locally-allocated
        MXINT4 bank (random init; suitable for cubin-invocation evidence,
        not HF parity). The MXFP4 ``expert_bank`` is not allocated in
        this mode.
    dtype
        Dtype used for latent projections, shared experts, RMSNorm
        weight, and the MXINT4 fused-cubin bank scales. Default fp32.
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

        self.gate = KimiK3MoEGate(config)

        # Routed expert storage — MXFP4 bank (Python fallback) OR MXINT4
        # bank (fused SiTU cubin invocation for AC5 evidence). Only one
        # is allocated per block.
        if use_fused_cubin:
            self.expert_bank = None
            self._init_fused_cubin_bank(device=device)
        else:
            self.expert_bank = KimiK3RoutedExpertBank(
                num_experts=config.num_experts,
                hidden_size=self.moe_hidden_size,
                intermediate_size=config.moe_intermediate_size,
                activation=routed_activation,
                device=device,
            )
            self._fused_bank_ready = False

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
    # Fused-cubin path — private MXINT4 bank + SiTU dispatch.
    # ------------------------------------------------------------------

    def _init_fused_cubin_bank(self, device: Optional[torch.device]) -> None:
        """Allocate MXINT4 packed weight + bf16 scale buffers matching
        the ``trtllm_mxint4_block_scale_moe`` contract.

        Buffer shapes copied from
        ``exisiting_optimization_work/trtllmgen_MOE/tests/moe/test_trtllm_gen_routed_fused_moe.py::test_trtllm_gen_mxint4_routed_fused_moe``:

        * ``gemm1_weights``: ``uint8 [E, 2*I, H // 2]``
        * ``gemm1_weights_scale``: ``bf16 [E, 2*I, H // 32]``
        * ``gemm2_weights``: ``uint8 [E, H, I // 2]``
        * ``gemm2_weights_scale``: ``bf16 [E, H, I // 32]``

        Random-initialised on construction so the block is self-
        contained (no HF load required for cubin-invocation evidence).
        Two fp4 codes per byte in ``uint8``; bf16 scales per 32-element
        group.
        """
        E = self.num_experts
        H = self.moe_hidden_size
        II = self.config.moe_intermediate_size
        g = 32

        assert H % 2 == 0 and H % g == 0, (
            f"moe_hidden_size {H} must be divisible by 2 and 32 for MXINT4"
        )
        assert II % 2 == 0 and II % g == 0, (
            f"moe_intermediate_size {II} must be divisible by 2 and 32 for MXINT4"
        )

        gemm1_weights = torch.randint(0, 256, (E, 2 * II, H // 2), dtype=torch.uint8, device=device)
        gemm1_weights_scale = torch.randn((E, 2 * II, H // g), dtype=torch.bfloat16, device=device)
        gemm2_weights = torch.randint(0, 256, (E, H, II // 2), dtype=torch.uint8, device=device)
        gemm2_weights_scale = torch.randn((E, H, II // g), dtype=torch.bfloat16, device=device)

        self.register_buffer("gemm1_weights", gemm1_weights)
        self.register_buffer("gemm1_weights_scale", gemm1_weights_scale)
        self.register_buffer("gemm2_weights", gemm2_weights)
        self.register_buffer("gemm2_weights_scale", gemm2_weights_scale)
        self._fused_bank_ready = True

    def _invoke_fused_moe_kernel(
        self,
        routed_in: torch.Tensor,
        *,
        activation_type_value: Optional[int] = None,
        output: Optional[torch.Tensor] = None,
        routing_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Invoke ``trtllm_mxint4_block_scale_moe`` on the private SiTU MXINT4 bank.

        Callers use this method for both the normal module path (via
        :meth:`_moe_infer_fused_cubin`) and for AC5 evidence checks that
        pass a pre-filled sentinel ``output`` and/or a specific
        ``activation_type_value`` (e.g. ``Swiglu`` for a negative control
        that must raise on the SiTU-only private pool).

        Parameters
        ----------
        routed_in
            bf16 ``[num_tokens, moe_hidden_size]``.
        activation_type_value
            Passed to the kernel's ``activation_type`` slot. Defaults
            to ``ActivationType.Situ.value``.
        output
            Pre-allocated output buffer (bf16 ``[num_tokens,
            moe_hidden_size]``). Written in place — smoke tests pre-fill
            it with a sentinel to detect silent no-ops.
        routing_logits
            fp32 ``[num_tokens, num_experts]``. Defaults to random
            uniform per the reference test.
        """
        assert self.use_fused_cubin and self._fused_bank_ready, (
            "fused-cubin path invoked but bank was not allocated (use_fused_cubin=False)"
        )
        assert routed_in.dtype == torch.bfloat16, (
            f"private SiTU MxInt4/Bfloat16 kernel requires bf16 "
            f"hidden_states, got {routed_in.dtype}"
        )
        assert routed_in.is_cuda, "fused kernel requires CUDA hidden_states"

        num_tokens = routed_in.shape[0]
        E = self.num_experts
        II = self.config.moe_intermediate_size

        if routing_logits is None:
            routing_logits = torch.rand(num_tokens, E, device=routed_in.device, dtype=torch.float32)

        result = invoke_private_situ_moe(
            routing_logits=routing_logits,
            hidden_states=routed_in,
            gemm1_weights=self.gemm1_weights,
            gemm1_weights_scale=self.gemm1_weights_scale,
            gemm2_weights=self.gemm2_weights,
            gemm2_weights_scale=self.gemm2_weights_scale,
            num_experts=E,
            top_k=self.top_k,
            intermediate_size=II,
            routed_scaling_factor=1.0,
            activation_type_value=activation_type_value,
            output=output,
        )
        self._cubin_call_count += 1
        return result

    def _moe_infer_fused_cubin(self, routed_in: torch.Tensor) -> torch.Tensor:
        """Fused-cubin variant of :meth:`_moe_infer`.

        Wraps :meth:`_invoke_fused_moe_kernel` with default args for the
        normal forward path (no sentinel, activation=Situ).
        """
        return self._invoke_fused_moe_kernel(routed_in)

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

        Fused-cubin path (``use_fused_cubin=True``): same skeleton, but
        step 5 is :meth:`_moe_infer_fused_cubin`, which invokes the
        private SiTU ``trtllm_mxint4_block_scale_moe`` cubin. Steps 4/6/8
        still run (outside the fused kernel's scope). The gate call at
        step 2 is skipped in fused mode since the kernel does its own
        routing internally.
        """
        identity = hidden_states
        orig_shape = hidden_states.shape

        if self.use_fused_cubin:
            topk_idx = None
            topk_weight = None
        else:
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
            y = self._moe_infer_fused_cubin(routed_in_bf16)
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

    Not applicable when ``k3.use_fused_cubin=True`` (that mode uses
    random MXINT4 data for AC5 cubin-invocation evidence; HF parity is
    out of scope for the fused path).

    Returns provenance metadata for logging.
    """
    if k3.use_fused_cubin:
        raise RuntimeError(
            "copy_hf_moe_block_weights: k3 block is in fused-cubin mode "
            "(use_fused_cubin=True). HF parity is not defined for the "
            "random MXINT4 bank; skip this copy in fused-cubin tests."
        )

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
    if k3.shared_experts is not None and hasattr(hf, "shared_experts"):
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

    return MoEBlockProvenance(
        n_experts=k3.num_experts,
        shared_expert_names=shared_names,
        routed_expert_layout=(k3.moe_hidden_size, k3.config.moe_intermediate_size),
        latent=k3.use_latent_moe,
        latent_use_norm=k3.routed_expert_norm is not None,
        canonicalized=True,
    )
