# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Thin shell hosting the Kimi K3 native SiTU MoE under ``ConfigurableMoE``.

``KimiK3SituBackend`` adapts the in-tree TRTLLM-Gen SiTU op
(:func:`._moe_kernels.invoke_native_situ_moe`) to the backend contract of
``ConfigurableMoE``'s ``ExternalCommMoEScheduler``:

* it subclasses plain :class:`MoE` (never ``TRTLLMGenFusedMoE`` — the
  scheduler dispatches extra kwargs by exact ``__class__`` match, so an
  unknown class receives exactly ``(x, token_selected_experts,
  token_final_scales, x_sf)`` and must always finalize);
* ``_supports_load_balancer() -> True`` turns on separated routing, so the
  wrapper's routing method recomputes top-k per chunk and ``run_moe``
  receives precomputed slot ids;
* weights stay OUTSIDE the parameter framework: the model-side
  ``KimiK3MoERuntime`` owns the MXFP4 expert bank and lazily packs it on
  the first ``run_moe`` via the injected closures. ``create_weights`` is a
  flag-setting no-op and the model-loader post-load walks hit no-op stubs.

``KimiK3ConfigurableMoE`` overrides ``_create_and_sync_backend`` to build
this backend directly, bypassing ``create_moe``'s hardcoded class tables
while inheriting the comm factory, chunking and EPLB attr sync unchanged.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch

from ...model_config import ModelConfig
from ..fused_moe.configurable_moe import _BACKEND_SYNC_ATTRS, ConfigurableMoE
from ..fused_moe.interface import MoE, MoEWeightLoadingMode
from ..fused_moe.routing import BaseMoeRoutingMethod
from ._moe_kernels import invoke_native_situ_moe


class KimiK3SituBackend(MoE):
    """Backend half of the shell; only usable behind ``KimiK3ConfigurableMoE``."""

    def __init__(
        self,
        *,
        routing_method: BaseMoeRoutingMethod,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: Optional[torch.dtype],
        reduce_results: bool,
        model_config: ModelConfig,
        ensure_weights_fn: Callable[[], None],
        get_weights_fn: Callable[[], Dict[str, torch.Tensor]],
        layer_idx: Optional[int] = None,
        init_load_balancer: bool = False,
    ):
        super().__init__(
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=reduce_results,
            model_config=model_config,
            weight_loading_mode=MoEWeightLoadingMode.VANILLA,
            layer_idx=layer_idx,
            init_load_balancer=init_load_balancer,
        )
        # The closures close over the model-side runtime (an nn.Module);
        # regular attribute assignment would register it as a submodule and
        # duplicate its state_dict keys.
        object.__setattr__(self, "_ensure_weights_fn", ensure_weights_fn)
        object.__setattr__(self, "_get_weights_fn", get_weights_fn)
        self.quant_method = None
        self._weights_created = False
        # Chunks never exceed moe_max_num_tokens rows; cap the autotune
        # profile at the op's own default so warmup cost cannot regress.
        self._tune_max_num_tokens = min(
            model_config.moe_max_num_tokens or 8192, 8192)
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

    @classmethod
    def can_implement(cls, *args, **kwargs):
        # Never consulted on the shell path (resolve_moe_cls is bypassed);
        # the op itself asserts SM100/SM103 support at first use.
        return True, None

    def _supports_load_balancer(self) -> bool:
        # Forces separated routing: routing_method.apply runs per chunk and
        # run_moe receives precomputed (slot) ids — the K3 topk bypass.
        return True

    # --- weights: bank-owned, outside the parameter framework --------------

    def create_weights(self):
        # Called by _create_and_sync_backend, by the wrapper delegation, AND
        # unconditionally by DecoderModelForCausalLM.__post_init__ on every
        # module — must stay an idempotent, allocation-free flag.
        self._weights_created = True

    def load_weights(self, weights, allow_partial_loading: bool = False):
        raise NotImplementedError(
            "KimiK3SituBackend weights are loaded via the model-side "
            "expert bank, not the module load_weights flow")

    def transform_weights(self):
        # model_loader's post-load walks visit the backend child module
        # (the wrapper skips itself via _weights_removed); the base impl
        # would dereference quant_method.
        self._weights_transformed = True

    def cache_derived_state(self):
        pass

    # --- scheduler touchpoints ---------------------------------------------

    def quantize_input(self, x, post_quant_comm: bool = True):
        # bf16 latent passes through; invoke_native_situ_moe runs its own
        # MXFP8 activation quantization after dispatch.
        return x, None

    def run_moe(self, x, token_selected_experts, token_final_scales,
                x_sf=None, **kwargs):
        if x.shape[0] == 0:
            return torch.zeros_like(x)
        self._ensure_weights_fn()
        fw = self._get_weights_fn()
        return invoke_native_situ_moe(
            hidden_states=x,
            topk_ids=token_selected_experts,
            topk_weights=token_final_scales,
            gemm1_weights=fw["gemm1_weights"],
            gemm1_weights_scale=fw["gemm1_weights_scale"],
            gemm2_weights=fw["gemm2_weights"],
            gemm2_weights_scale=fw["gemm2_weights_scale"],
            gemm1_alpha=fw["gemm1_alpha"],
            gemm1_beta=fw["gemm1_beta"],
            # SYNCED wrapper attrs, not the model-side contiguous slice:
            # under EPLB the scheduler delivers slot ids. Out-of-local-range
            # ids (incl. NVLinkOneSided's -1 filler rows) are skipped by the
            # op and contribute zeros.
            num_experts=self.num_slots,
            top_k=token_selected_experts.shape[-1],
            valid_hidden_size=self.hidden_size,
            valid_intermediate_size=self.intermediate_size,
            local_expert_offset=self.slot_start,
            local_num_experts=self.expert_size_per_partition,
            tune_max_num_tokens=self._tune_max_num_tokens,
        )

    def forward_impl(self, *args, **kwargs):
        raise NotImplementedError(
            "KimiK3SituBackend runs behind KimiK3ConfigurableMoE's "
            "scheduler; it has no standalone forward")

    def validate_configurable_moe(self, moe) -> None:
        assert not moe._using_dynamic_load_balancer(), (
            "dynamic EPLB needs registered per-slot nn.Parameters; the K3 "
            "expert bank is packed out-of-framework and freed after packing")
        assert moe.initial_local_expert_ids == list(
            range(moe.slot_start, moe.slot_end)), (
                "the K3 expert bank loads a contiguous expert slice; "
                "non-contiguous (EPLB) placements need a bank loader keyed "
                "by initial_local_expert_ids")


class KimiK3ConfigurableMoE(ConfigurableMoE):
    """``ConfigurableMoE`` wrapper that hosts :class:`KimiK3SituBackend`."""

    def __init__(
        self,
        *,
        ensure_weights_fn: Callable[[], None],
        get_weights_fn: Callable[[], Dict[str, torch.Tensor]],
        **kwargs,
    ):
        # _create_and_sync_backend runs inside super().__init__ and needs
        # the closures; object.__setattr__ both bypasses nn.Module's
        # pre-__init__ guard and avoids registering the closed-over runtime
        # as a submodule.
        object.__setattr__(self, "_ensure_weights_fn", ensure_weights_fn)
        object.__setattr__(self, "_get_weights_fn", get_weights_fn)
        kwargs.setdefault("weight_loading_mode", MoEWeightLoadingMode.VANILLA)
        super().__init__(**kwargs)

    def _create_and_sync_backend(
        self,
        *,
        model_config: ModelConfig,
        routing_method: BaseMoeRoutingMethod,
        override_quant_config=None,
        **kwargs,
    ) -> None:
        # Mirror the stock dance (skip-weights bracket -> construct ->
        # _BACKEND_SYNC_ATTRS mirror -> conditional create_weights) with the
        # K3 backend constructed directly; create_moe's hardcoded class
        # tables are bypassed, everything else — comm factory, scheduler,
        # chunking, EPLB attr sync — is inherited unchanged.
        with self._temporarily_skip_weight_creation(model_config):
            backend = KimiK3SituBackend(
                routing_method=routing_method,
                num_experts=self.num_experts,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                dtype=self.dtype,
                reduce_results=self.reduce_results,
                model_config=model_config,
                ensure_weights_fn=self._ensure_weights_fn,
                get_weights_fn=self._get_weights_fn,
                layer_idx=None,
                init_load_balancer=False,
            )
        self.backend = backend
        self.use_flashinfer = False
        for attr in _BACKEND_SYNC_ATTRS:
            setattr(self.backend, attr, getattr(self, attr))
        if not model_config.skip_create_weights_in_init:
            self.backend.create_weights()
