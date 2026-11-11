# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: Nemotron-3 Ultra 550B-A55B BF16 / sm_100 / dep16 -- self-contained modeling code.

A 108-block hybrid stack (48 Mamba2, 12 attention, 48 LatentMoE blocks, the order
read from the checkpoint's `layers_block_type`), every block pre-normed with its
own RMSNorm and added back to a bf16 residual stream. Every weight and every
activation is bf16 (the router's logits and the SSM state are fp32):

* Mamba2: `in_proj` -> depthwise causal conv + SiLU over x|B|C -> SSD scan
  (chunked on context rows, one recurrence step on generation rows) -> gated
  grouped RMSNorm (gate before the norm) -> `out_proj`.
* attention: GQA with no positional encoding at all, bf16 KV cache.
* LatentMoE: an fp32 sigmoid router with a selection-only correction bias over
  the full hidden state; the 512 relu2 experts run in a 2048-wide latent space
  between two projections; a relu2 shared expert runs on the full hidden state
  beside them.

dep16: attention data parallel over 16 GB200 GPUs (four trays of four). Every
rank runs its own requests through a whole copy of the mamba, attention and
shared-expert blocks, the router and both latent projections; only the routed
experts are expert-parallel, 32 of the 512 per rank. There is no all-reduce
anywhere: a block's output is this rank's own, added straight to the residual.

The routed experts run in the engine's MoE layer, built with Modeling V1's
`create_moe` on its unquantized Relu2 leaf, for every `moe_config.backend` routing
admits: CUTLASS (AUTO resolves to it), TRTLLM (FlashInfer's TRTLLM-Gen BF16 MoE) or
MEGAMOE_CUTEDSL (CuTe DSL MegaMoE). That layer owns the exchange of tokens between
ranks, which a target-local expert path would have to rebuild: CUTLASS and TRTLLM
dispatch and combine the 2048-wide latent with the engine's all-to-all (NVLinkOneSided
where the system has it, the strategy Modeling V1 runs at this layout; AllGather +
ReduceScatter measured about 10% slower end to end there), and MegaMoE fuses dispatch,
experts and combine in one kernel. Each layer routes the fp32 logits itself (the
checkpoint's noaux_tc selection) and returns the routed sum for this rank's own tokens.
A request the leaf turns down fails at construction; it is never served by another
kernel. `UNCERTIFIED_GENERIC_CALLS` names that stock code.

Tuning knobs (read once, at load): `TRTLLM_NEMOTRON_ULTRA_DEP16_LOWM_MAX_ROWS` and
`TRTLLM_NEMOTRON_ULTRA_DEP16_FI_SSU_MIN_ROWS` (see below); the MoE layer's own knobs
are the engine's (`moe_config.max_num_tokens`, `TRTLLM_FORCE_COMM_METHOD`,
`MEGAMOE_TACTIC_AUTOTUNE`, `MEGAMOE_COMBINE_FORMAT`).

Each target is self-contained: it holds the per-layer weight views it reads,
passes every argument to every op at the call site, and shares no code with the
other targets.
"""

import os
from typing import NamedTuple

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import step_contract_enabled
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (
    thop_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_fused_add_rmsnorm import (  # noqa: E501
    flashinfer_fused_add_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron_h.ops.gemm.flashinfer_direct_bf16_gemm import (  # noqa: E501
    flashinfer_direct_bf16_gemm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron_h.ops.norm.rms_norm_gated_grouped import (  # noqa: E501
    rms_norm_gated_grouped,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron_h.ops.ssm.causal_conv1d_fwd import (  # noqa: E501
    causal_conv1d_fwd,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron_h.ops.ssm.causal_conv1d_update import (  # noqa: E501
    causal_conv1d_update,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron_h.ops.ssm.flashinfer_selective_state_update import (  # noqa: E501
    flashinfer_selective_state_update,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron_h.ops.ssm.mamba_chunk_scan_combined import (  # noqa: E501
    mamba_chunk_scan_combined,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron_h.ops.ssm.selective_state_update import (  # noqa: E501
    selective_state_update,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm.bindings import DataType
from tensorrt_llm.logger import logger

from . import weights as _weights

# The certification is per GPU architecture: this code was gated on sm_100 only.
_SM = (10, 0)

#: Stock code this target runs outside the catalog, declared so it is not consumed silently: every
#: tensorrt_llm import of this module that computes. An entry leaves the list when a catalog entry
#: or a family op replaces it.
UNCERTIFIED_GENERIC_CALLS = (
    # The engine hooks this target inherits.
    "tensorrt_llm._torch.models.modeling_utils.DecoderModel",
    "tensorrt_llm._torch.models.modeling_utils.DecoderModelForCausalLM",
    # The routed experts and their exchange between ranks: the engine's MoE layer, its
    # communication strategy and the routing it applies.
    "tensorrt_llm._torch.moe.fused_moe.create_moe",
    "tensorrt_llm._torch.moe.fused_moe.ConfigurableMoE",
    "tensorrt_llm._torch.moe.fused_moe.DeepSeekV3MoeRoutingMethod",
)

# The engine MoE leaf each backend must resolve to. create_moe runs with backend degradation off,
# so a backend that declines the layer raises with its rejection trail; this also pins the leaf
# within the backend, so the request cannot be served by another kernel.
_ENGINE_MOE_LEAVES = {
    "CUTLASS": "trtllm.cutlass.grouped_gemm.none",
    "MEGAMOE_CUTEDSL": "trtllm.cutedsl.mega_moe.none",
    "TRTLLM": "flashinfer.trtllm_gen.fused_moe.none",
}
# The backends whose kernel owns the exchange between ranks (no host-side communication).
_FUSED_COMM_BACKENDS = frozenset({"MEGAMOE_CUTEDSL"})

# An empty slot in the recurrent-state pools (padding rows of a captured batch).
_PAD_SLOT_ID = -1

#: Decode steps of up to this many rows on a rank run the K = 8192 projections (mamba in_proj,
#: qkv, shared-expert up, fc1_latent) through FlashInfer's register-prefetch kernel instead of
#: cublas_mm. Its cells cover one and two rows, so the knob takes 0 (off), 1 or 2. At tep8's
#: per-rank shapes it was 1.3-2.3x faster at one row and still ahead at two; dep16's projections
#: are the whole ones (in_proj 8192 -> 35072, qkv -> 8704, shared up -> 10240), not measured.
LOWM_MAX_ROWS_ENV = "TRTLLM_NEMOTRON_ULTRA_DEP16_LOWM_MAX_ROWS"
_LOWM_MAX_ROWS_DEFAULT = 2
_LOWM_MAX_ROWS_CERTIFIED = 2
#: The decode step runs FlashInfer's Mamba2 kernel from this many rows on and the Triton kernel
#: below. At tep8's 32 heads per rank FlashInfer's was the faster from 32 rows (1.17x there, 1.36x
#: at 256) and the slower below (0.73x at one row); dep16 runs all 256 heads on every rank, eight
#: times the work per row, so the crossover is likely lower but not measured. A mixed step's
#: generation rows stay on the Triton kernel: FlashInfer's reads x, B and C through TMA, which needs
#: them 128-byte aligned, and those rows start past the context tokens' in_proj rows.
FI_SSU_MIN_ROWS_ENV = "TRTLLM_NEMOTRON_ULTRA_DEP16_FI_SSU_MIN_ROWS"
_FI_SSU_MIN_ROWS_DEFAULT = 32
# The checkpoint's time_step_limit (0, inf): softplus(dt + dt_bias) is not clamped.
_DT_LIMIT = (0.0, float("inf"))

_CALL_CONSTANTS = dict(
    output_sf=None,
    k=None,
    v=None,
    is_fused_qkv=True,
    update_kv_cache=True,
    predicted_tokens_per_seq=1,
    attention_input_type=0,
    is_mla_enable=False,
    mask_type=1,
    q_scaling=1.0,
    # bf16 KV cache: no quantization of Q, K, V or the cache, so no scale tensors.
    quant_mode=0,
    kv_scale_orig_quant=None,
    kv_scale_quant_orig=None,
    attention_sinks=None,
    # No positional encoding: in-kernel RoPE is disabled; the rope_* values below are
    # inert placeholders.
    position_embedding_type=0,
    rotary_inv_freq=None,
    rotary_cos_sin=None,
    rope_dim=0,
    rope_base=10000.0,
    rope_scale_type=0,
    rope_scale=1.0,
    rope_short_m_scale=1.0,
    rope_long_m_scale=1.0,
    rope_max_positions=1024,
    rope_original_max_positions=1024,
    out_scale=None,
    latent_cache=None,
    q_pe=None,
    q_lora_rank=None,
    kv_lora_rank=None,
    qk_nope_head_dim=None,
    qk_rope_head_dim=None,
    v_head_dim=None,
    rope_append=None,
    chunked_prefill_buffer_batch_size=1,
    attention_chunk_size=None,
    softmax_stats_tensor=None,
    sparse_kv_indices=None,
    sparse_kv_offsets=None,
    sparse_attn_indices=None,
    sparse_attn_offsets=None,
    sparse_attn_indices_block_size=0,
    mrope_rotary_cos_sin=None,
    mrope_position_deltas=None,
    helix_position_offsets=None,
    helix_is_inactive_rank=None,
    cross_kv=None,
    relative_attention_bias=None,
    relative_attention_max_distance=0,
    kv_norm_weight=None,
    kv_norm_eps=1e-6,
    skip_correction_threshold=0.0,
    # Routing refuses speculative decoding: no draft positions.
    spec_decoding_position_offsets=None,
)


def _env_rows(name: str, default: int, lo: int, hi: int | None) -> int:
    """A row-count knob from the environment: `default` when unset, else an int in [lo, hi]."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r}: expected an integer") from None
    if value < lo or (hi is not None and value > hi):
        bound = f"[{lo}, {hi}]" if hi is not None else f">= {lo}"
        raise ValueError(f"{name}={value}: expected {bound}")
    return value


class _Mamba(NamedTuple):
    """One Mamba2 block's weights, as the views the ops read."""

    in_w: torch.Tensor
    conv_w: torch.Tensor
    conv_b: torch.Tensor
    A: torch.Tensor
    A_hd: torch.Tensor
    D: torch.Tensor
    D_hd: torch.Tensor
    D_hd16: torch.Tensor
    dt_bias: torch.Tensor
    dt_bias_hd: torch.Tensor
    dt_bias_hd16: torch.Tensor
    gnorm: torch.Tensor
    out_w: torch.Tensor


class _Attention(NamedTuple):
    qkv: torch.Tensor
    o: torch.Tensor


class _Moe(NamedTuple):
    gate: torch.Tensor
    fc1_lat: torch.Tensor
    fc2_lat: torch.Tensor
    sh_up: torch.Tensor
    sh_down: torch.Tensor
    # The engine's MoE layer: routing, the exchange between ranks and the routed experts.
    experts: nn.Module


class _Bound(NamedTuple):
    """Everything taken off one KV cache manager: the attention layers' pool rows and the
    mamba layers' conv and SSM state pools."""

    cm: object
    pool_row: dict
    conv: dict
    ssm: dict


def _layer_views(core) -> list:
    """Per block, the views its forward reads, and the *next* block's input norm (the
    final norm after the last block) as the second element."""
    w = core.w
    d = core.dims
    hd = d.head_dim_ssm
    st = d.ssm_state
    views = []
    for i, kind in enumerate(d.kinds):
        if kind == "mamba":
            A = w[f"l{i}_A"]
            D = w[f"l{i}_D"]
            dtb = w[f"l{i}_dt_bias"]
            L = _Mamba(
                in_w=w[f"l{i}_in_proj_w"].t(),
                conv_w=w[f"l{i}_conv_w"],
                conv_b=w[f"l{i}_conv_b"],
                A=A,
                # The decode step reads per-(head, dim, state) A and per-(head, dim) D and
                # dt_bias; broadcast views, no copies.
                A_hd=A[:, None, None].expand(d.heads_r, hd, st),
                D=D,
                D_hd=D[:, None].expand(d.heads_r, hd),
                # FlashInfer's decode kernel takes D and dt_bias only in dt's dtype: the
                # checkpoint's bf16 values, which the fp32 copies hold exactly.
                D_hd16=D.to(torch.bfloat16)[:, None].expand(d.heads_r, hd),
                dt_bias=dtb,
                dt_bias_hd=dtb[:, None].expand(d.heads_r, hd),
                dt_bias_hd16=dtb.to(torch.bfloat16)[:, None].expand(d.heads_r, hd),
                gnorm=w[f"l{i}_gnorm"],
                out_w=w[f"l{i}_out_proj_w"].t(),
            )
        elif kind == "attention":
            L = _Attention(qkv=w[f"l{i}_qkv_w"].t(), o=w[f"l{i}_o_w"].t())
        else:
            assert kind == "moe", kind
            L = _Moe(
                gate=w[f"l{i}_gate_w"].t(),
                fc1_lat=w[f"l{i}_fc1_lat_w"].t(),
                fc2_lat=w[f"l{i}_fc2_lat_w"].t(),
                sh_up=w[f"l{i}_sh_up_w"].t(),
                sh_down=w[f"l{i}_sh_down_w"].t(),
                experts=core.routed_experts[str(i)],
            )
        nxt = w[f"l{i + 1}_norm"] if i + 1 < d.num_layers else w["final_norm"]
        views.append((L, nxt))
    return views


def _routed_moe_kwargs(model_config: ModelConfig, d, w: nn.ParameterDict, layer_idx: int) -> dict:
    """`create_moe`'s arguments for MoE block `layer_idx`: the checkpoint's routed experts in
    the latent space (512 x [5120, 2048], non-gated Relu2, no bias) behind its router (sigmoid
    + correction bias selection, one group, top-22, renormalized and scaled), expert-parallel
    by the engine's mapping (moe_tp 1, moe_ep 16: 32-expert windows) under attention DP."""
    from tensorrt_llm._torch.moe.fused_moe import (
        DeepSeekV3MoeRoutingMethod,
        MoEWeightLoadingMode,
        SimpleActivation,
    )
    from tensorrt_llm._torch.utils import ActivationType

    bias_key = f"l{layer_idx}_gate_b"
    return dict(
        routing_method=DeepSeekV3MoeRoutingMethod(
            top_k=d.top_k,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=d.routed_scale,
            # Looked up on every call: the loader replaces the meta parameter with the
            # loaded one. fp32, as the checkpoint stores it.
            callable_e_score_correction_bias=lambda w=w, k=bias_key: w[k],
            is_fused=True,
        ),
        num_experts=d.num_experts,
        hidden_size=d.latent,
        intermediate_size=d.moe_inter,
        dtype=torch.bfloat16,
        # Under attention DP the layer returns the routed sum of this rank's own tokens: the
        # combine of its exchange completes it, nothing is left to reduce.
        reduce_results=False,
        model_config=model_config,
        weight_loading_mode=MoEWeightLoadingMode.VANILLA,
        bias=False,
        layer_idx=layer_idx,
        activation=SimpleActivation(kind=ActivationType.Relu2),
        # A request the leaf declines raises with the rejection trail instead of running on
        # another backend and being reported as the one asked for.
        allow_backend_degradation=False,
    )


def _routed_experts(model_config: ModelConfig, d, w: nn.ParameterDict) -> nn.ModuleDict:
    """The engine's MoE layer per MoE block.

    Fails loudly on anything but the requested leaf over this rank's contiguous 32-expert
    window under attention DP, with an exchange between ranks: the kernel's own for MegaMoE,
    a host-side communication strategy for the others. Without one, every rank would run
    its own tokens through its own 32 experts only.
    """
    from tensorrt_llm._torch.moe.fused_moe import ConfigurableMoE, create_moe

    backend = d.engine_moe
    assert model_config.moe_backend == backend, (
        f"moe_config.backend={model_config.moe_backend_requested!r} resolved to "
        f"{model_config.moe_backend!r}, not {backend!r}"
    )
    leaf = _ENGINE_MOE_LEAVES[backend]
    window = list(range(d.expert_offset, d.expert_offset + d.local_experts))
    layers = nn.ModuleDict()
    for i in d.moe_layers:
        moe = create_moe(**_routed_moe_kwargs(model_config, d, w, i))
        if not isinstance(moe, ConfigurableMoE):
            raise RuntimeError(
                f"moe_config.backend={backend!r} built {type(moe).__name__}, not ConfigurableMoE; "
                "ENABLE_CONFIGURABLE_MOE must not be disabled for this target"
            )
        got = str(type(moe.backend).descriptor.identity)
        if got != leaf:
            raise RuntimeError(
                f"moe_config.backend={backend!r} resolved layer {i} to {got}; this target runs "
                f"that request only on {leaf}"
            )
        if list(moe.backend.initial_local_expert_ids) != window:
            raise RuntimeError(
                f"layer {i}: the MoE layer holds experts {moe.backend.initial_local_expert_ids}, "
                f"not this rank's window [{window[0]}, {window[-1] + 1})"
            )
        if not moe.use_dp:
            raise RuntimeError(f"layer {i}: the MoE layer was built without attention DP")
        if backend not in _FUSED_COMM_BACKENDS and moe.comm is None:
            raise RuntimeError(
                f"layer {i}: moe_config.backend={backend!r} built no communication strategy; "
                "every rank would run only its own tokens through its own experts"
            )
        layers[str(i)] = moe
    return layers


def _exchange(moe: nn.Module) -> str:
    """How a built MoE layer moves tokens between ranks, for the log."""
    comm = getattr(moe, "comm", None)
    return "fused in the kernel" if comm is None else type(comm).__name__


class NemotronUltraBf16Dep16ModelingV2(DecoderModel):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.dims = _weights.MODEL_WEIGHTS.dims(self)
        self.w = _weights.MODEL_WEIGHTS.declare(self.dims)
        # The engine's MoE layers, which own the routed experts' weights.
        self.routed_experts: nn.ModuleDict = _routed_experts(model_config, self.dims, self.w)
        self._bound: "_Bound | None" = None
        self._binds = 0
        # Set by post_load_weights, once the weights are real.
        self.side_stream: "torch.cuda.Stream | None" = None
        self.lowm_max_rows = _LOWM_MAX_ROWS_DEFAULT
        self.fi_ssu_min_rows = _FI_SSU_MIN_ROWS_DEFAULT
        self._prefill: "PrefillTarget | None" = None
        self._decode: "DecodeTarget | None" = None

    def post_load_weights(self) -> None:
        """Construct the two targets, now that the weights are real.

        Called twice by the engine (the shell's hook, then the loader's module
        walk), so it rebuilds rather than accumulates. Must never run from
        `__init__`, where the containers are still meta.
        """
        mapping = self.model_config.mapping
        self.lowm_max_rows = _env_rows(
            LOWM_MAX_ROWS_ENV, _LOWM_MAX_ROWS_DEFAULT, 0, _LOWM_MAX_ROWS_CERTIFIED
        )
        self.fi_ssu_min_rows = _env_rows(FI_SSU_MIN_ROWS_ENV, _FI_SSU_MIN_ROWS_DEFAULT, 1, None)
        if mapping.rank == 0:
            first = self.routed_experts[str(self.dims.moe_layers[0])]
            logger.info(
                f"nemotron3_ultra_bf16__sm_100__dep16: routed experts on "
                f"{_ENGINE_MOE_LEAVES[self.dims.engine_moe]} (moe_config.backend="
                f"{self.model_config.moe_backend_requested}), exchange between the "
                f"{mapping.moe_ep_size} ranks {_exchange(first)}, moe_max_num_tokens "
                f"{first.moe_max_num_tokens}; decode low-M GEMMs up to {self.lowm_max_rows} "
                f"rows, FlashInfer Mamba2 step from {self.fi_ssu_min_rows} rows"
            )
        # Whether kernels that take the choice launch as programmatic dependents.
        self.pdl = get_env_enable_pdl()
        # The shared expert runs on this stream beside the routed experts.
        self.side_stream = torch.cuda.Stream()
        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)

    def _bind_cache_manager(self, md) -> None:
        """Take every pool view this model reads from one KV cache manager.

        Called once per manager rather than once per model: with capacity
        estimation the engine builds a throwaway manager, warms up on it and then
        builds the real one under the same module, and a view held across the swap
        would dangle silently.
        """
        cm = md.kv_cache_manager
        assert cm is not None, "a paged KV cache is required"
        assert not (self._binds and torch.cuda.is_current_stream_capturing()), (
            f"the KV cache manager changed to {type(cm).__name__} during CUDA graph capture"
        )
        self._binds += 1
        d = self.dims
        for name in ("get_conv_states", "get_ssm_states", "layer_offsets"):
            assert hasattr(cm, name), (
                f"{type(cm).__name__} has no {name}: the runtime allocated no hybrid "
                "recurrent-state pool for this model"
            )
        # thop_attention addresses the KV pool by the layer's pool row; counting
        # attention layers instead would alias layers with finite, plausible output.
        pool_row = {i: cm.layer_offsets[i] for i in d.attn_layers}
        conv = {i: cm.get_conv_states(i) for i in d.mamba_layers}
        ssm = {i: cm.get_ssm_states(i) for i in d.mamba_layers}
        c0, s0 = conv[d.mamba_layers[0]], ssm[d.mamba_layers[0]]
        assert c0.shape[1:] == (d.conv_r, d.conv_kernel - 1), (
            f"conv state pool {tuple(c0.shape)}: expected [slots, {d.conv_r}, {d.conv_kernel - 1}]"
        )
        assert s0.shape[1:] == (d.heads_r, d.head_dim_ssm, d.ssm_state), (
            f"ssm state pool {tuple(s0.shape)}: expected "
            f"[slots, {d.heads_r}, {d.head_dim_ssm}, {d.ssm_state}]"
        )
        # The pools hold what the runtime was configured with, not what this target was gated on:
        # `kv_cache_config.dtype` or `mamba_ssm_cache_dtype` can change them, and the kernels
        # would read and write bf16 KV (no scales, quant_mode 0) and an fp32 SSM state regardless.
        assert getattr(cm, "dtype", DataType.BF16) == DataType.BF16, (
            f"KV cache dtype {cm.dtype}: this target reads and writes a bf16 KV cache"
        )
        assert c0.dtype == torch.bfloat16 and s0.dtype == torch.float32, (
            f"conv state {c0.dtype} / ssm state {s0.dtype}: this target needs bf16 / fp32"
        )
        # Every rank holds all 256 heads, so one slot of the fp32 state spans 48 layers x 2M
        # elements and the slot offsets the state kernels index pass 2^31 elements from slot 22.
        if self.model_config.mapping.rank == 0:
            logger.info(
                f"nemotron3_ultra_bf16__sm_100__dep16: recurrent-state pools of {s0.shape[0]} "
                f"slots, ssm slot stride {s0.stride(0)} elements ({s0.dtype}), conv slot stride "
                f"{c0.stride(0)} ({c0.dtype})"
            )
        assert md.effective_beam_width == 1, "beam search is not implemented"
        assert md.cache_indirection is None, "beam search is not implemented"
        assert not md.is_cross, "cross attention is not implemented"
        self._bound = _Bound(cm, pool_row, conv, ssm)

    def _select_target(self, attn_metadata) -> "PrefillTarget | DecodeTarget":
        """One host-side integer read: a mixed batch goes to prefill, so prefill is the
        general case and decode the specialization that may assume no context rows."""
        return self._decode if attn_metadata.num_contexts == 0 else self._prefill

    def forward(self, attn_metadata: AttentionMetadata, *args, **kwargs) -> torch.Tensor:
        """Bind the step's cache manager if it changed, then route the step."""
        if self._bound is None or attn_metadata.kv_cache_manager is not self._bound.cm:
            self._bind_cache_manager(attn_metadata)
        return self._select_target(attn_metadata).forward(attn_metadata, *args, **kwargs)


class PrefillTarget:
    """The general case: context rows, and possibly generation rows after them.

    Under attention DP each rank picks its own target every step: one rank's prefill can run
    beside another's decode. Both call the 48 MoE layers in the same order with the same
    `all_rank_num_tokens`, which is all the exchange between ranks needs to stay in step.
    """

    def __init__(self, core: NemotronUltraBf16Dep16ModelingV2) -> None:
        """Take the views and constants this target's forward reads, once, off the real weights."""
        self.core = core
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks for it; read once here.
        self._contract_pending = step_contract_enabled()
        cfg = core.model_config.pretrained_config
        self._eps = cfg.layer_norm_epsilon
        self._norm0 = core.w["l0_norm"]
        self._layers = _layer_views(core)

    def _check_step_contract(self, md) -> None:
        """Opt-in first-forward fail-fast: the metadata fields this target consumes
        must exist (they are private trtllm surface). Calling `step_args` and reading
        the mamba fields is the check."""
        if not self._contract_pending:
            return
        try:
            self.step_args(md)
            mm = md.mamba_metadata
            for name in ("state_indices", "has_initial_states", "query_start_loc", "seq_idx"):
                getattr(mm, name)
            mm.use_initial_states, mm.chunk_size  # noqa: B018 — read for the AttributeError
        except AttributeError as exc:
            raise AssertionError(f"attention metadata surface drifted: {exc}") from exc
        # The MoE layers' exchange reads one token count per rank.
        n = md.all_rank_num_tokens
        assert n is not None and len(n) == self.core.model_config.mapping.world_size, (
            f"all_rank_num_tokens {n}: the MoE exchange needs one count per rank"
        )
        self._contract_pending = False

    def step_args(self, md: TrtllmAttentionMetadata) -> dict:
        """Project the prepared metadata onto thop_attention's explicit batch state."""
        return dict(
            sequence_length=md.kv_lens_cuda_runtime,
            host_past_key_value_lengths=md.kv_lens_runtime,
            host_total_kv_lens=md.host_total_kv_lens,
            context_lengths=md.prompt_lens_cuda_runtime,
            host_context_lengths=md.prompt_lens_cpu_runtime,
            host_request_types=md.host_request_types_runtime,
            kv_cache_block_offsets=md.kv_cache_block_offsets,
            host_kv_cache_pool_pointers=md.host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=md.host_kv_cache_pool_mapping,
            workspace=md.effective_workspace,
            tokens_per_block=md.tokens_per_block,
            max_num_requests=md.max_num_requests,
            max_context_length=md.max_context_length,
            max_seq_len=md.max_seq_len,
            num_contexts=md.num_contexts,
            num_ctx_tokens=md.num_ctx_tokens,
            trtllm_gen_jit_warmup=md.trtllm_gen_jit_warmup,
            use_paged_context_fmha=md.use_paged_context_fmha,
            beam_width=md.effective_beam_width,
            cache_indirection=md.cache_indirection,
            block_ids_per_seq=md.block_ids_per_seq,
            max_context_q_len_override=md.max_context_q_len_override,
            is_cross=md.is_cross,
            is_spec_decoding_enabled=md.is_spec_decoding_enabled,
            use_spec_decoding=md.use_spec_decoding,
            is_spec_dec_tree=md.is_spec_dec_tree,
            spec_decoding_generation_lengths=md.spec_decoding_generation_lengths,
            spec_decoding_packed_mask=md.spec_decoding_packed_mask,
            spec_decoding_bl_tree_mask_offset=md.spec_decoding_bl_tree_mask_offset,
            spec_decoding_bl_tree_mask=md.spec_decoding_bl_tree_mask,
            spec_decoding_target_max_draft_tokens=md.max_total_draft_tokens,
            spec_bl_tree_first_sparse_mask_offset_kv=md.spec_bl_tree_first_sparse_mask_offset_kv,
            num_sparse_topk=md.num_sparse_topk,
            flash_mla_tile_scheduler_metadata=md.flash_mla_tile_scheduler_metadata,
            flash_mla_num_splits=md.flash_mla_num_splits,
            max_num_sequences=md.max_num_sequences,
            force_prepare_spec_dec_tree_mask=md.force_prepare_spec_dec_tree_mask,
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        core = self.core
        d = core.dims
        md = attn_metadata
        eps = self._eps
        bound = core._bound

        self._check_step_contract(md)
        step = self.step_args(md)
        window = md.max_seq_len
        # One token count per rank (host ints): every MoE layer's exchange reads them.
        all_rank_num_tokens = md.all_rank_num_tokens

        # Host-side batch split: context rows first, then single-token generation rows.
        nc = md.num_contexts
        nct = md.num_ctx_tokens
        ns = md.num_seqs
        mm = md.mamba_metadata
        si = mm.state_indices
        qsl = mm.query_start_loc
        hi = mm.has_initial_states
        use_init = mm.use_initial_states
        seq_idx = mm.seq_idx
        chunk_indices = mm.chunk_indices
        chunk_offsets = mm.chunk_offsets
        chunk = mm.chunk_size
        si_ctx = si[:nc]
        si_gen = si[nc:ns]
        has_gen = ns > nc

        if inputs_embeds is None:
            h = nn.functional.embedding(input_ids, core.w["embed"])
        else:
            h = inputs_embeds
        num_tokens = h.shape[0]
        attn_out = torch.empty([num_tokens, d.q_width_r], dtype=h.dtype, device=h.device)

        din, conv_c = d.d_inner_r, d.conv_r
        heads, hd, gn = d.heads_r, d.head_dim_ssm, d.gn_r
        groups, st = d.groups_r, d.ssm_state

        x = flashinfer_rmsnorm(h, self._norm0, eps)
        residual = h
        for i, (L, norm_next) in enumerate(self._layers):
            if isinstance(L, _Mamba):
                conv_state = bound.conv[i]
                ssm_state = bound.ssm[i]
                zxbcdt = cublas_mm(x, L.in_w)
                z = zxbcdt[:, :din]
                xbc = zxbcdt[:, din : din + conv_c]
                dt = zxbcdt[:, din + conv_c :]
                y = torch.empty([num_tokens, heads, hd], dtype=h.dtype, device=h.device)

                # Context rows: varlen conv (seeding from and writing back the conv
                # state), then the chunked scan seeded from the SSM state where the
                # sequence has one, with each sequence's final state written back.
                # Channel-last views on both sides: the result is token-major
                # [nct, x|B|C], so x, B and C are column views of it.
                xbc_ctx = causal_conv1d_fwd(
                    xbc[:nct].t(),
                    L.conv_w,
                    L.conv_b,
                    conv_state,
                    qsl[: nc + 1],
                    si_ctx,
                    hi[:nc],
                    True,
                    _PAD_SLOT_ID,
                    torch.empty([nct, conv_c], dtype=h.dtype, device=h.device).t(),
                ).t()
                xs = xbc_ctx[:, :din].reshape(1, nct, heads, hd)
                Bm = xbc_ctx[:, din : din + gn].reshape(1, nct, groups, st)
                Cm = xbc_ctx[:, din + gn :].reshape(1, nct, groups, st)
                # A fresh row's slot still holds the previous request's state, which
                # the scan cannot tell apart from a carried one: zero it here.
                init = None
                if use_init:
                    init = ssm_state[si_ctx].to(torch.float32)
                    init.masked_fill_(~hi[:nc].view(nc, 1, 1, 1), 0)
                final = mamba_chunk_scan_combined(
                    xs,
                    dt[:nct].unsqueeze(0),
                    L.A,
                    Bm,
                    Cm,
                    chunk,
                    L.D,
                    None,
                    L.dt_bias,
                    init,
                    seq_idx,
                    chunk_indices,
                    chunk_offsets,
                    qsl[: nc + 1],
                    True,
                    _DT_LIMIT,
                    y[:nct].view(1, nct, heads, hd),
                    False,
                    True,
                    torch.float32,
                )
                ssm_state[si_ctx] = final.to(ssm_state.dtype)

                # Generation rows: one conv step and one recurrence step each.
                if has_gen:
                    # Activates the generation rows of xBC in place inside zxbcdt.
                    xbc_gen = causal_conv1d_update(
                        xbc[nct:], conv_state, L.conv_w, L.conv_b, True, None, si_gen, _PAD_SLOT_ID
                    )
                    ng = num_tokens - nct
                    selective_state_update(
                        ssm_state,
                        xbc_gen[:, :din].reshape(ng, heads, hd),
                        dt[nct:][:, :, None].expand(ng, heads, hd),
                        L.A_hd,
                        xbc_gen[:, din : din + gn].reshape(ng, groups, st),
                        xbc_gen[:, din + gn :].reshape(ng, groups, st),
                        D=L.D_hd,
                        z=None,
                        dt_bias=L.dt_bias_hd,
                        dt_softplus=True,
                        state_batch_indices=si_gen,
                        pad_slot_id=_PAD_SLOT_ID,
                        out=y[nct:],
                    )

                yn = rms_norm_gated_grouped(
                    y.reshape(num_tokens, din), z, L.gnorm, eps, d.norm_group
                )
                out = cublas_mm(yn, L.out_w)
            elif isinstance(L, _Attention):
                qkv = cublas_mm(x, L.qkv)
                attn_out = thop_attention(
                    q=qkv,
                    output=attn_out,
                    local_layer_idx=bound.pool_row[i],
                    num_heads=d.q_heads_r,
                    num_kv_heads=d.kv_heads,
                    head_size=d.head_dim,
                    attention_window_size=window,
                    **step,
                    **_CALL_CONSTANTS,
                )
                out = cublas_mm(attn_out, L.o)
            else:
                # The shared expert reads only x: run it on the side stream while the
                # routed experts run on the main one, joined before the sum.
                side = core.side_stream
                main = torch.cuda.current_stream()
                side.wait_stream(main)
                with torch.cuda.stream(side):
                    u = cublas_mm(x, L.sh_up)
                    a = torch.square(torch.relu(u))
                    shared = cublas_mm(a, L.sh_down)
                # Router: fp32 logits over the full hidden state; the MoE layer selects on
                # sigmoid + correction bias and weights from the unbiased sigmoid, renormalized
                # and scaled.
                logits = cublas_mm(x, L.gate, out_dtype=torch.float32)
                # Routed experts in the latent space: the MoE layer sends each of this rank's
                # rows to the ranks holding its 22 experts and returns their weighted sum, for
                # this rank's rows only. Its output may be a view of a workspace the next MoE
                # launch rewrites (MegaMoE), consumed by fc2_lat before that.
                xl = cublas_mm(x, L.fc1_lat)
                routed = L.experts(
                    xl,
                    logits,
                    all_rank_num_tokens=all_rank_num_tokens,
                    use_dp_padding=False,
                )
                out = cublas_mm(routed, L.fc2_lat)
                main.wait_stream(side)
                out = out + shared
            x, residual = flashinfer_fused_add_rmsnorm(out, residual, norm_next, eps)
        return x


class DecodeTarget:
    """The specialization: routed to only when there are no context rows.

    Under CUDA graphs with attention DP the engine pads every rank's batch to the busiest
    rank's row count (rounded up to a captured size), so a replay sees the same
    `all_rank_num_tokens` on every rank, frozen at capture; the padding rows are dummy
    requests with their own state slots and run the whole stack like real ones.
    """

    def __init__(self, core: NemotronUltraBf16Dep16ModelingV2) -> None:
        """Take the views and constants this target's forward reads, once, off the real weights."""
        self.core = core
        self._contract_pending = step_contract_enabled()
        cfg = core.model_config.pretrained_config
        self._eps = cfg.layer_norm_epsilon
        self._norm0 = core.w["l0_norm"]
        self._layers = _layer_views(core)
        self._lowm_max_rows = core.lowm_max_rows
        self._fi_ssu_min_rows = core.fi_ssu_min_rows

    def _check_step_contract(self, md) -> None:
        """Opt-in first-forward fail-fast: the metadata fields this target consumes
        must exist (they are private trtllm surface)."""
        if not self._contract_pending:
            return
        try:
            self.step_args(md)
            md.mamba_metadata.state_indices  # noqa: B018 — read for the AttributeError
        except AttributeError as exc:
            raise AssertionError(f"attention metadata surface drifted: {exc}") from exc
        n = md.all_rank_num_tokens
        assert n is not None and len(n) == self.core.model_config.mapping.world_size, (
            f"all_rank_num_tokens {n}: the MoE exchange needs one count per rank"
        )
        self._contract_pending = False

    def step_args(self, md: TrtllmAttentionMetadata) -> dict:
        """Project the prepared metadata onto thop_attention's explicit batch state."""
        return dict(
            sequence_length=md.kv_lens_cuda_runtime,
            host_past_key_value_lengths=md.kv_lens_runtime,
            host_total_kv_lens=md.host_total_kv_lens,
            context_lengths=md.prompt_lens_cuda_runtime,
            host_context_lengths=md.prompt_lens_cpu_runtime,
            host_request_types=md.host_request_types_runtime,
            kv_cache_block_offsets=md.kv_cache_block_offsets,
            host_kv_cache_pool_pointers=md.host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=md.host_kv_cache_pool_mapping,
            workspace=md.effective_workspace,
            tokens_per_block=md.tokens_per_block,
            max_num_requests=md.max_num_requests,
            max_context_length=md.max_context_length,
            max_seq_len=md.max_seq_len,
            num_contexts=0,
            num_ctx_tokens=0,
            trtllm_gen_jit_warmup=md.trtllm_gen_jit_warmup,
            use_paged_context_fmha=md.use_paged_context_fmha,
            beam_width=md.effective_beam_width,
            cache_indirection=md.cache_indirection,
            block_ids_per_seq=md.block_ids_per_seq,
            max_context_q_len_override=md.max_context_q_len_override,
            is_cross=md.is_cross,
            is_spec_decoding_enabled=md.is_spec_decoding_enabled,
            use_spec_decoding=md.use_spec_decoding,
            is_spec_dec_tree=md.is_spec_dec_tree,
            spec_decoding_generation_lengths=md.spec_decoding_generation_lengths,
            spec_decoding_packed_mask=md.spec_decoding_packed_mask,
            spec_decoding_bl_tree_mask_offset=md.spec_decoding_bl_tree_mask_offset,
            spec_decoding_bl_tree_mask=md.spec_decoding_bl_tree_mask,
            spec_decoding_target_max_draft_tokens=md.max_total_draft_tokens,
            spec_bl_tree_first_sparse_mask_offset_kv=md.spec_bl_tree_first_sparse_mask_offset_kv,
            num_sparse_topk=md.num_sparse_topk,
            flash_mla_tile_scheduler_metadata=md.flash_mla_tile_scheduler_metadata,
            flash_mla_num_splits=md.flash_mla_num_splits,
            max_num_sequences=md.max_num_sequences,
            force_prepare_spec_dec_tree_mask=md.force_prepare_spec_dec_tree_mask,
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        core = self.core
        d = core.dims
        md = attn_metadata
        eps = self._eps
        bound = core._bound

        self._check_step_contract(md)
        step = self.step_args(md)
        window = md.max_seq_len
        all_rank_num_tokens = md.all_rank_num_tokens

        # Every row is one generation token of its own sequence.
        ns = md.num_seqs
        si = md.mamba_metadata.state_indices[:ns]

        if inputs_embeds is None:
            h = nn.functional.embedding(input_ids, core.w["embed"])
        else:
            h = inputs_embeds
        num_tokens = h.shape[0]
        attn_out = torch.empty([num_tokens, d.q_width_r], dtype=h.dtype, device=h.device)
        lowm = num_tokens <= self._lowm_max_rows
        fi_ssu = num_tokens >= self._fi_ssu_min_rows

        din, conv_c = d.d_inner_r, d.conv_r
        heads, hd, gn = d.heads_r, d.head_dim_ssm, d.gn_r
        groups, st = d.groups_r, d.ssm_state

        x = flashinfer_rmsnorm(h, self._norm0, eps)
        residual = h
        for i, (L, norm_next) in enumerate(self._layers):
            if isinstance(L, _Mamba):
                conv_state = bound.conv[i]
                ssm_state = bound.ssm[i]
                if lowm:
                    zxbcdt = flashinfer_direct_bf16_gemm(x, L.in_w, core.pdl)
                else:
                    zxbcdt = cublas_mm(x, L.in_w)
                z = zxbcdt[:, :din]
                xbc = zxbcdt[:, din : din + conv_c]
                dt = zxbcdt[:, din + conv_c :]
                # Activates xBC in place inside zxbcdt.
                xbc = causal_conv1d_update(
                    xbc, conv_state, L.conv_w, L.conv_b, True, None, si, _PAD_SLOT_ID
                )
                y = torch.empty([num_tokens, heads, hd], dtype=h.dtype, device=h.device)
                if fi_ssu:
                    flashinfer_selective_state_update(
                        ssm_state,
                        xbc[:, :din].reshape(num_tokens, heads, hd),
                        dt[:, :, None].expand(num_tokens, heads, hd),
                        L.A_hd,
                        xbc[:, din : din + gn].reshape(num_tokens, groups, st),
                        xbc[:, din + gn :].reshape(num_tokens, groups, st),
                        D=L.D_hd16,
                        z=None,
                        dt_bias=L.dt_bias_hd16,
                        dt_softplus=True,
                        state_batch_indices=si,
                        pad_slot_id=_PAD_SLOT_ID,
                        out=y,
                    )
                else:
                    selective_state_update(
                        ssm_state,
                        xbc[:, :din].reshape(num_tokens, heads, hd),
                        dt[:, :, None].expand(num_tokens, heads, hd),
                        L.A_hd,
                        xbc[:, din : din + gn].reshape(num_tokens, groups, st),
                        xbc[:, din + gn :].reshape(num_tokens, groups, st),
                        D=L.D_hd,
                        z=None,
                        dt_bias=L.dt_bias_hd,
                        dt_softplus=True,
                        state_batch_indices=si,
                        pad_slot_id=_PAD_SLOT_ID,
                        out=y,
                    )
                yn = rms_norm_gated_grouped(
                    y.reshape(num_tokens, din), z, L.gnorm, eps, d.norm_group
                )
                out = cublas_mm(yn, L.out_w)
            elif isinstance(L, _Attention):
                if lowm:
                    qkv = flashinfer_direct_bf16_gemm(x, L.qkv, core.pdl)
                else:
                    qkv = cublas_mm(x, L.qkv)
                attn_out = thop_attention(
                    q=qkv,
                    output=attn_out,
                    local_layer_idx=bound.pool_row[i],
                    num_heads=d.q_heads_r,
                    num_kv_heads=d.kv_heads,
                    head_size=d.head_dim,
                    attention_window_size=window,
                    **step,
                    **_CALL_CONSTANTS,
                )
                out = cublas_mm(attn_out, L.o)
            else:
                # The shared expert reads only x: run it on the side stream while the
                # routed experts run on the main one, joined before the sum.
                side = core.side_stream
                main = torch.cuda.current_stream()
                side.wait_stream(main)
                with torch.cuda.stream(side):
                    if lowm:
                        u = flashinfer_direct_bf16_gemm(x, L.sh_up, core.pdl)
                    else:
                        u = cublas_mm(x, L.sh_up)
                    a = torch.square(torch.relu(u))
                    shared = cublas_mm(a, L.sh_down)
                logits = cublas_mm(x, L.gate, out_dtype=torch.float32)
                if lowm:
                    xl = flashinfer_direct_bf16_gemm(x, L.fc1_lat, core.pdl)
                else:
                    xl = cublas_mm(x, L.fc1_lat)
                routed = L.experts(
                    xl,
                    logits,
                    all_rank_num_tokens=all_rank_num_tokens,
                    use_dp_padding=False,
                )
                out = cublas_mm(routed, L.fc2_lat)
                main.wait_stream(side)
                out = out + shared
            x, residual = flashinfer_fused_add_rmsnorm(out, residual, norm_next, eps)
        return x


class ModelingV2Nemotron3UltraBf16Sm100Dep16(
    DecoderModelForCausalLM[NemotronUltraBf16Dep16ModelingV2, PretrainedConfig]
):
    def __init__(self, model_config: ModelConfig):
        cap = torch.cuda.get_device_capability()
        assert cap == _SM, (
            f"this target is certified on sm_{_SM[0]}{_SM[1]} only; the device is "
            f"sm_{cap[0]}{cap[1]}"
        )
        cfg = model_config.pretrained_config
        super().__init__(
            NemotronUltraBf16Dep16ModelingV2(model_config),
            config=model_config,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )

    def load_weights(self, weights, *args, **kwargs):
        _weights.MODEL_WEIGHTS.load(self, weights)

    def post_load_weights(self):
        super().post_load_weights()
        self.model.post_load_weights()
