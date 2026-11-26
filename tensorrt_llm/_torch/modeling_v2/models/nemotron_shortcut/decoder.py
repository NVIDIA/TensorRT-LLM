# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed text decoder with wide residual connections, GDP and shortcut MoE."""

import os
from functools import partial

import torch
from torch import nn

from tensorrt_llm._torch.distributed.allreduce_helper import force_all_reduce_deterministic
from tensorrt_llm._torch.modeling_v2.catalog.attention.thop_attention import thop_attention
from tensorrt_llm._torch.modeling_v2.catalog.comm.allgather import allgather
from tensorrt_llm._torch.modeling_v2.catalog.comm.allreduce import allreduce, allreduce_group_args
from tensorrt_llm._torch.modeling_v2.catalog.comm.reducescatter import reducescatter
from tensorrt_llm._torch.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.peft.lora.layer import LoraLayer, LoraModuleType
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceStrategy

from .attention_metadata import attention_step_args
from .gdp import gdp_mixer
from .indexer import IndexState
from .lora import ShortcutLoraOwner
from .moe import clamped_relu2, grouped_experts, routing_weights

_CALL_CONSTANTS = dict(
    output_sf=None,
    k=None,
    v=None,
    is_fused_qkv=True,
    update_kv_cache=True,
    attention_input_type=0,
    is_mla_enable=False,
    mask_type=1,
    q_scaling=1.0,
    quant_mode=0,
    kv_scale_orig_quant=None,
    kv_scale_quant_orig=None,
    # position_embedding_type=0 applies no rotation; the rope_* values are
    # the contract's inert placeholders.
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
    attention_sinks=None,
)

_GDP = "linear_attention"
_MLP = "mlp"
_ATTN_MOE = "shortcut_moe_attn"
_GDP_MOE = "shortcut_moe_gdp"
# Residual-connection names in the checkpoint, per block kind.
_CONNECTIONS = {
    _GDP: ("res",),
    _MLP: ("res",),
    _ATTN_MOE: ("mix", "moe"),
    _GDP_MOE: ("mix", "moe"),
}


def rms_norm(x, weight, eps):
    wide = x.float()
    normalized = wide * torch.rsqrt(wide.square().mean(-1, keepdim=True) + eps)
    return normalized.to(x.dtype) * weight


def _apply_candidate_rows(num_ctx, width, op, x, *args):
    """Run a row-local operation at ordinary decode batch size for each MTP position."""
    candidates = x[num_ctx:].unflatten(0, (-1, width)).unbind(1)
    rows = torch.stack([op(group, *args) for group in candidates], dim=1).flatten(0, 1)
    return torch.cat((op(x[:num_ctx], *args), rows)) if num_ctx else rows


class ShortcutDecoder(DecoderModel):
    # Sparse prefill constructs per-request plans from host context lengths.
    supports_prefill_cuda_graph = False
    nvfp4_kv = False

    def __init__(self, model_config):
        super().__init__(model_config)
        cfg = model_config.pretrained_config
        mapping = model_config.mapping
        dt = model_config.torch_dtype
        self.attn_dp = mapping.enable_attention_dp
        tp = 1 if self.attn_dp else mapping.tp_size
        tp_rank = 0 if self.attn_dp else mapping.tp_rank
        self.tp_size = tp
        self.dp_size = mapping.tp_size if self.attn_dp else 1
        self.tp_group = list(mapping.tp_group)
        self.deterministic_allreduce = force_all_reduce_deterministic()
        # A request's output is independent of the requests batched with it only under
        # FORCE_DETERMINISTIC=1. The fixed launch shapes this takes cost throughput.
        self.batch_invariant = os.getenv("FORCE_DETERMINISTIC", "0") == "1"
        self.allreduce_args = (
            allreduce_group_args(mapping, strategy=int(AllReduceStrategy.NCCL))
            if mapping.tp_size > 1
            else {}
        )
        self.process_group = self.allreduce_args.get("process_group")
        self.dp_rank = mapping.tp_rank
        self.kinds = list(cfg.block_types)
        self.num_layers = len(self.kinds)
        spec = getattr(model_config, "spec_config", None)
        self.mtp_enabled = spec is not None and spec.spec_dec_mode.is_mtp_eagle_one_model()
        self.hidden = cfg.hidden_size
        self.eps = cfg.rms_norm_eps
        self.streams = cfg.wide_residual_num_streams
        self.max_forget = cfg.wide_residual_retention_max_forget

        self.clamp = float(cfg.activation_clamp_value)
        # Split query heads across ranks and replicate a single KV head.
        self.head_dim = cfg.head_dim
        self.heads_q = cfg.num_attention_heads // tp
        self.heads_kv_global = cfg.num_key_value_heads
        self.heads_kv = max(1, cfg.num_key_value_heads // tp)
        # Each sparse layer replicates the index projection across ranks.
        self.index_topk = cfg.index_topk
        self.index_dim = cfg.index_head_dim

        self.m = cfg.gdp_num_householder
        self.num_sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        self.gdp_heads = cfg.gdp_num_heads // tp
        self.gdp_groups = cfg.gdp_num_groups // tp
        self.gdp_dv = cfg.gdp_head_dim
        self.gdp_dk = cfg.gdp_key_head_dim
        self.gdp_eps = cfg.gdp_norm_eps
        self.gdp_g_clamp_value = cfg.gdp_g_clamp_value
        self.gdp_kernel = cfg.gdp_conv_kernel
        self.gdp_v = self.gdp_heads * self.gdp_dv
        self.gdp_k = self.gdp_groups * self.gdp_dk
        self.gdp_conv = self.m * self.gdp_v + (self.m + 1) * self.gdp_k
        self.gdp_proj = self.gdp_v + self.gdp_conv + (self.m + 1) * self.gdp_heads

        self.inter = cfg.intermediate_size // tp
        self.shared_inter = cfg.moe_shared_expert_intermediate_size // tp

        # Routed experts: split by expert over EP, by column over MoE-TP.
        ep, moe_tp = mapping.moe_ep_size, mapping.moe_tp_size
        self.num_experts = cfg.n_routed_experts
        self.topk = cfg.num_experts_per_tok
        self.routed_scale = float(cfg.routed_scaling_factor)
        self.latent = cfg.moe_latent_size
        self.expert_inter = cfg.moe_intermediate_size // moe_tp
        self.local_experts = self.num_experts // ep
        self.moe_ep_rank = mapping.moe_ep_rank
        self.expert_placements = {}
        if getattr(model_config, "moe_load_balancer", None) is not None:
            from .expert_placement import create_placements

            self.expert_placements = create_placements(
                model_config, self.kinds, (_ATTN_MOE, _GDP_MOE), self.num_experts, self.topk
            )

        attention_layers = [i for i, kind in enumerate(self.kinds) if kind == _ATTN_MOE]
        if self.mtp_enabled:
            attention_layers.append(self.num_layers)
        self.index = (
            IndexState(
                self.index_topk,
                attention_layers,
                self.num_sms,
                index_dim=self.index_dim,
                batch_invariant=self.batch_invariant,
                nvfp4_kv=self.nvfp4_kv,
            )
            if self.index_topk is not None
            else None
        )
        self.slices = dict(
            tp_rank=tp_rank, moe_ep_rank=mapping.moe_ep_rank, moe_tp_rank=mapping.moe_tp_rank
        )

        def P(*shape, dtype=dt):
            return nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)

        H, f32 = self.hidden, torch.float32
        w = nn.ParameterDict()
        for i, kind in enumerate(self.kinds):
            for conn in _CONNECTIONS[kind]:
                w[f"l{i}_{conn}_read"] = P(self.streams, dtype=f32)
                w[f"l{i}_{conn}_write"] = P(self.streams, dtype=f32)
                w[f"l{i}_{conn}_retain"] = P(self.streams, dtype=f32)
            w[f"l{i}_norm_in"] = P(H)
            if kind in (_GDP, _GDP_MOE):
                self._declare_matrix(w, f"l{i}_gdp_in", (self.gdp_proj, H), P)
                w[f"l{i}_gdp_conv"] = P(self.gdp_conv, self.gdp_kernel)
                w[f"l{i}_gdp_a_log"] = P(self.gdp_heads, dtype=f32)
                w[f"l{i}_gdp_dt_bias"] = P(self.gdp_heads, dtype=f32)
                w[f"l{i}_gdp_norm"] = P(self.gdp_v)
                self._declare_matrix(w, f"l{i}_gdp_out", (H, self.gdp_v), P)
            if kind == _ATTN_MOE:
                self._declare_attention(w, f"l{i}_", P)
            if kind == _MLP:
                self._declare_matrix(w, f"l{i}_up", (self.inter, H), P)
                self._declare_matrix(w, f"l{i}_down", (H, self.inter), P)
            if kind in (_ATTN_MOE, _GDP_MOE):
                w[f"l{i}_shortcut_read"] = P(self.streams, dtype=f32)
                w[f"l{i}_shortcut_norm"] = P(H)
                self._declare_moe(w, f"l{i}_", P)
        if self.mtp_enabled:
            for name in ("enorm", "hnorm", "norm_in", "final_norm"):
                w["mtp_" + name] = P(H)
            self._declare_matrix(w, "mtp_eh", (H, 2 * H), P)
            self._declare_attention(w, "mtp_", P)
            self._declare_moe(w, "mtp_", P)
        w["readout"] = P(self.streams, dtype=f32)
        w["final_norm"] = P(H)
        w["embed"] = P(cfg.vocab_size, H)

        self.adapters = nn.ModuleDict()
        if model_config.lora_config is not None:
            for i, kind in enumerate(self.kinds):
                if kind == _ATTN_MOE:
                    slots = {
                        "q": (LoraModuleType.ATTENTION_Q, self.heads_q * self.head_dim),
                        "k": (LoraModuleType.ATTENTION_K, self.heads_kv * self.head_dim),
                        "v": (LoraModuleType.ATTENTION_V, self.heads_kv * self.head_dim),
                        "o": (LoraModuleType.ATTENTION_DENSE, H),
                    }
                elif kind == _MLP:
                    slots = {
                        "up": (LoraModuleType.MLP_H_TO_4H, self.inter),
                        "down": (LoraModuleType.MLP_4H_TO_H, H),
                    }
                else:
                    continue
                group = nn.ModuleDict(
                    {name: LoraLayer([module], [size]) for name, (module, size) in slots.items()}
                )
                group.layer_idx = i
                self.adapters[str(i)] = group

        self.w = w
        self._layers = None
        self._kv_layer = {}

    def _declare_matrix(self, weights, name, shape, P):
        weights[name] = P(*shape)

    def _matrix_view(self, name):
        weight = self.w[name]
        return weight.t() if weight.ndim == 2 else weight

    def _declare_attention(self, w: nn.ParameterDict, prefix: str, P) -> None:
        qkv_rows = (self.heads_q + 2 * self.heads_kv) * self.head_dim
        self._declare_matrix(w, f"{prefix}qkv", (qkv_rows, self.hidden), P)
        self._declare_matrix(w, f"{prefix}o", (self.hidden, self.heads_q * self.head_dim), P)
        if self.index_topk is not None:
            self._declare_matrix(w, f"{prefix}idx", (2 * self.index_dim, self.hidden), P)

    def _declare_moe(self, w: nn.ParameterDict, prefix: str, P) -> None:
        """The post-mixer norm, router and latent / shared experts."""
        placement = self.expert_placements.get(prefix)
        slots = len(placement.initial_local_expert_ids) if placement else self.local_experts
        H = self.hidden
        w[f"{prefix}pre_moe_norm"] = P(H)
        self._declare_matrix(w, f"{prefix}router", (self.num_experts, H), P)
        w[f"{prefix}router_bias"] = P(self.num_experts, dtype=torch.float32)
        self._declare_matrix(w, f"{prefix}fc1_latent", (self.latent, H), P)
        w[f"{prefix}fc2_norm"] = P(self.latent)
        self._declare_matrix(w, f"{prefix}fc2_latent", (H, self.latent), P)
        self._declare_matrix(w, f"{prefix}experts_up", (slots, self.expert_inter, self.latent), P)
        self._declare_matrix(w, f"{prefix}experts_down", (slots, self.latent, self.expert_inter), P)
        self._declare_matrix(w, f"{prefix}shared_up", (self.shared_inter, H), P)
        self._declare_matrix(w, f"{prefix}shared_down", (H, self.shared_inter), P)

    def build_layer_views(self) -> None:
        """Build matrix views and FP32 stream factors after loading the parameters."""
        w = self.w
        layers = []
        for i, kind in enumerate(self.kinds):
            p = {"kind": kind, "norm_in": w[f"l{i}_norm_in"]}
            for conn in _CONNECTIONS[kind]:
                p[f"{conn}_read"] = torch.sigmoid(w[f"l{i}_{conn}_read"])
                p[f"{conn}_write"] = 2.0 * torch.sigmoid(w[f"l{i}_{conn}_write"])
                p[f"{conn}_retain"] = 1.0 - self.max_forget * torch.sigmoid(
                    -w[f"l{i}_{conn}_retain"]
                )
            if kind in (_GDP, _GDP_MOE):
                p["gdp_in"] = self._matrix_view(f"l{i}_gdp_in")
                p["gdp_conv"] = w[f"l{i}_gdp_conv"]
                p["gdp_a_log"] = w[f"l{i}_gdp_a_log"]
                p["gdp_dt_bias"] = w[f"l{i}_gdp_dt_bias"]
                p["gdp_norm"] = w[f"l{i}_gdp_norm"]
                p["gdp_out"] = self._matrix_view(f"l{i}_gdp_out")
            if kind == _ATTN_MOE:
                p.update(self._attention_views(f"l{i}_"))
            if kind == _MLP:
                p["up"] = self._matrix_view(f"l{i}_up")
                p["down"] = self._matrix_view(f"l{i}_down")
            if kind in (_ATTN_MOE, _GDP_MOE):
                p["shortcut_read"] = torch.sigmoid(w[f"l{i}_shortcut_read"])
                p["shortcut_norm"] = w[f"l{i}_shortcut_norm"]
                p.update(self._moe_views(f"l{i}_"))
            layers.append(p)
        self._layers = layers
        self._readout = torch.sigmoid(w["readout"])
        if self.mtp_enabled:
            self.mtp = {
                name: w["mtp_" + name] for name in ("enorm", "hnorm", "norm_in", "final_norm")
            }
            self.mtp["eh"] = self._matrix_view("mtp_eh")
            self.mtp.update(self._attention_views("mtp_"))
            self.mtp.update(self._moe_views("mtp_"))

    def _attention_views(self, prefix: str) -> dict:
        views = {"qkv": self._matrix_view(f"{prefix}qkv"), "o": self._matrix_view(f"{prefix}o")}
        if self.index_topk is not None:
            views["idx"] = self._matrix_view(f"{prefix}idx")
        return views

    def _moe_views(self, prefix: str) -> dict:
        w = self.w
        views = {"pre_moe_norm": w[f"{prefix}pre_moe_norm"]}
        views["placement"] = self.expert_placements.get(prefix)
        for name in (
            "router",
            "fc1_latent",
            "fc2_latent",
            "shared_up",
            "shared_down",
            "experts_up",
            "experts_down",
        ):
            views[name] = self._matrix_view(f"{prefix}{name}")
        for name in ("router_bias", "fc2_norm"):
            views[name] = w[f"{prefix}{name}"]
        return views

    def _bind_cache_layout(self, md) -> None:
        """Record where each attention block's KV lives in the pool, once.

        The hybrid cache manager is built after the model, so the layer
        offsets are first readable on the first forward; they are fixed for
        the engine's life. It also has to be the hybrid manager at all: a
        manager without per-layer recurrent state means the engine did not
        treat this checkpoint as a hybrid linear-attention model.
        """
        manager = md.kv_cache_manager
        assert hasattr(manager, "mamba_layer_cache"), (
            f"{type(manager).__name__} holds no recurrent state; the engine did "
            "not treat this checkpoint as a hybrid linear-attention model"
        )
        kv_layer = {
            i: manager.layer_offsets[i] for i, kind in enumerate(self.kinds) if kind == _ATTN_MOE
        }
        # Every attention block is full (no window), so all of them address one
        # pool; the rows of GDP and MLP blocks are the manager's business.
        mapping = md.host_kv_cache_pool_mapping.tolist()
        pools = {mapping[local][0] for local in kv_layer.values()}
        assert len(pools) == 1, f"one attention window class expected; pool ids {sorted(pools)}"
        self._kv_layer = kv_layer
        if self.index is not None:
            self.index.bind(md, self.w["final_norm"].device)

    @staticmethod
    def _project(x, weight, *, out_dtype=None):
        if x.shape[0] == 0:
            return x.new_empty((0, weight.shape[1]), dtype=out_dtype or x.dtype)
        return cublas_mm(x, weight, out_dtype=out_dtype)

    def _apply_rows(self, op, x, *args, **kwargs):
        if not self.batch_invariant or x.shape[0] == 0:
            return op(x, *args, **kwargs)
        # GEMM and RMS reductions both change their reduction tree with M.
        # Keep their launch shape identical for single and co-batched tokens.
        rows_per_launch = 128
        outputs = []
        for chunk in x.split(rows_per_launch):
            count = chunk.shape[0]
            if count < rows_per_launch:
                chunk = nn.functional.pad(chunk, (0, 0, 0, rows_per_launch - count))
            outputs.append(op(chunk.contiguous(), *args, **kwargs)[:count])
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs)

    def mm(self, x, weight, *, out_dtype=None):
        return self._apply_rows(self._project, x, weight, out_dtype=out_dtype)

    def _norm(self, x, weight):
        return self._apply_rows(rms_norm, x, weight, self.eps)

    _grouped_experts = staticmethod(grouped_experts)

    def tp_reduce(self, x):
        if self.tp_size == 1:
            return x
        x = x.contiguous()
        if self.deterministic_allreduce:
            # NCCL can change the reduction tree with the packed token count.
            # Gather in rank order and use the same FP32 additions for every row.
            gathered = allgather(x, None, self.tp_group, self.process_group)
            ranks = gathered.view(self.tp_size, *x.shape)
            total = ranks[0].float()
            for contribution in ranks[1:]:
                total = total + contribution.float()
            return total.to(x.dtype)
        return allreduce(
            x,
            residual=None,
            norm_weight=None,
            scale=None,
            bias=None,
            op=int(AllReduceFusionOp.NONE),
            eps=0.0,
            trigger_completion_at_end=False,
            **self.allreduce_args,
        )

    def _adapted_mm(self, x, weight, name, layer, lora_params):
        if not lora_params:
            return self.mm(x, weight)
        return LoraLayer.forward_with_base(
            lambda: self.mm(x, weight), (self.adapters[str(layer)][name],), x, lora_params, layer
        )

    def attention(
        self,
        local_layer: int,
        p: dict,
        x: torch.Tensor,
        step: dict,
        md,
        tp_reduce,
        plan=None,
        layer=None,
        lora_params=None,
    ) -> torch.Tensor:
        """Store every index key and use the selected causal keys on sparse steps."""
        qkv = self.mm(x, p["qkv"])
        if lora_params:
            adapters = self.adapters[str(layer)]
            start = 0
            for name in ("q", "k", "v"):
                adapter = adapters[name]
                width = adapter.output_hidden_sizes[0]
                delta = adapter(x, lora_params, layer)
                if delta is not None:
                    qkv[:, start : start + width].add_(delta)
                start += width
        iq = None
        if plan is not None:
            iq, ik = self.mm(x, p["idx"]).split(self.index_dim, dim=-1)
            plan.store(layer, ik)
        if self.nvfp4_kv:
            out = plan.attend_nvfp4(layer, qkv, iq, p["kv_scales"], p["inv_kv_scales"])
        else:
            out = torch.empty(
                x.shape[0], self.heads_q * self.head_dim, dtype=x.dtype, device=x.device
            )
            args = dict(
                local_layer_idx=local_layer,
                num_heads=self.heads_q,
                num_kv_heads=self.heads_kv,
                head_size=self.head_dim,
                attention_window_size=md.max_seq_len,
                **step,
                **_CALL_CONSTANTS,
            )
            if plan is not None and plan.sparse:
                plan.attend(qkv, out, plan.select(layer, iq), args)
            else:
                thop_attention(q=qkv, output=out, **args)
        return tp_reduce(self._adapted_mm(out, p["o"], "o", layer, lora_params))

    def _mlp(self, x, up, down, tp_reduce, layer=None, lora_params=None):
        h = clamped_relu2(self._adapted_mm(x, up, "up", layer, lora_params), self.clamp)
        return tp_reduce(self._adapted_mm(h, down, "down", layer, lora_params))

    def _dp_sizes(self, all_rank_num_tokens, rows: int) -> list[int] | None:
        """Per-rank token counts for the attention-DP exchange, or None when
        uniform. The engine publishes them as host ints identical on every
        rank; a decode graph is captured and replayed at one batch size on
        every rank, so the ragged form never reaches a capture."""
        sizes = [int(n) for n in all_rank_num_tokens]
        assert len(sizes) == self.dp_size and sizes[self.dp_rank] == rows, (
            f"rank {self.dp_rank} holds {rows} rows; all_rank_num_tokens says {sizes}"
        )
        return None if len(set(sizes)) == 1 else sizes

    def moe(self, p, routed_in, shared_in, all_rank_num_tokens, tp_reduce):
        logits = self.mm(routed_in, p["router"], out_dtype=torch.float32)
        weights, ids = routing_weights(logits, p["router_bias"], self.topk, self.routed_scale)
        latent = self.mm(routed_in, p["fc1_latent"])
        sizes = None
        if self.attn_dp:
            sizes = self._dp_sizes(all_rank_num_tokens, latent.shape[0])
            latent, ids, weights = [
                allgather(x, sizes, self.tp_group, self.process_group)
                for x in (latent, ids, weights)
            ]
        # Some recipes keep the repeated draft expert bank in BF16.
        expert_op = (
            grouped_experts if isinstance(p["experts_up"], torch.Tensor) else self._grouped_experts
        )
        placement = p.get("placement")
        if placement is not None:
            ids = placement.route(ids)
        mixed = expert_op(
            latent, ids, weights, p["experts_up"], p["experts_down"], self.moe_ep_rank, self.clamp
        )
        if placement is not None:
            placement.finish()
        mixed = (
            reducescatter(mixed, sizes, self.tp_group, self.process_group)
            if self.attn_dp
            else tp_reduce(mixed)
        )
        routed = self.mm(self._norm(mixed.to(latent.dtype), p["fc2_norm"]), p["fc2_latent"])
        return routed + self._mlp(shared_in, p["shared_up"], p["shared_down"], tp_reduce)

    def _read_streams(self, streams, factors, weight):
        return self._norm(torch.matmul(factors, streams).to(weight.dtype), weight)

    def _norm_reads(self, streams, pending, reads, read_streams):
        if pending is not None:
            update, write, retain = pending
            streams = torch.addcmul(
                streams * retain[:, None], update.float()[:, None], write[:, None]
            )
        return streams, [read_streams(streams, factors, weight) for factors, weight in reads]

    def forward(
        self,
        attn_metadata,
        input_ids=None,
        position_ids=None,
        inputs_embeds=None,
        spec_metadata=None,
        lora_params=None,
    ):
        if spec_metadata is not None:
            raise ValueError("ShortcutForCausalLM handles speculative decoding metadata.")
        md = attn_metadata
        embedding = (
            nn.functional.embedding(input_ids.long(), self.w["embed"])
            if inputs_embeds is None
            else inputs_embeds
        )
        if embedding.shape[0] == 0:
            # An idle attention-DP rank still owns experts needed by its peers.
            if self.attn_dp and any(md.all_rank_num_tokens):
                for p in self._layers:
                    if p["kind"] in (_ATTN_MOE, _GDP_MOE):
                        self.moe(p, embedding, embedding, md.all_rank_num_tokens, self.tp_reduce)
            return embedding
        if not self._kv_layer:
            self._bind_cache_layout(md)
        decode = md.num_contexts == 0
        step = attention_step_args(md, n_rows=embedding.shape[0], decode=decode)
        read_streams, tp_reduce = self._read_streams, self.tp_reduce
        width = step["predicted_tokens_per_seq"]
        if self.nvfp4_kv and width > 1:
            # Both the FP32 stream contraction and the NCCL sum can change their
            # reduction order with row count. Match ordinary decode before KV quantization.
            per_candidate = partial(_apply_candidate_rows, step["num_ctx_tokens"], width)
            read_streams = partial(per_candidate, read_streams)
            if self.tp_size > 1 and not self.deterministic_allreduce:
                tp_reduce = partial(per_candidate, tp_reduce)
        streams = embedding.float()[:, None, :].expand(-1, self.streams, -1)
        plan = None
        if self.index is not None:
            plan = self.index.plan(md, embedding.shape[0])
        pending = None

        for i, p in enumerate(self._layers):
            kind = p["kind"]
            if kind in (_GDP, _MLP):
                streams, (x,) = self._norm_reads(
                    streams, pending, ((p["res_read"], p["norm_in"]),), read_streams
                )
                if kind == _GDP:
                    u = tp_reduce(self.mm(gdp_mixer(self, p, x, md, i), p["gdp_out"]))
                else:
                    u = self._mlp(x, p["up"], p["down"], tp_reduce, i, lora_params)
                pending = (u, p["res_write"], p["res_retain"])
            else:
                # The router reads the block input, before the mixer writes.
                streams, (routed_in, x) = self._norm_reads(
                    streams,
                    pending,
                    ((p["shortcut_read"], p["shortcut_norm"]), (p["mix_read"], p["norm_in"])),
                    read_streams,
                )
                if kind == _ATTN_MOE:
                    u = self.attention(
                        self._kv_layer[i], p, x, step, md, tp_reduce, plan, i, lora_params
                    )
                else:
                    u = tp_reduce(self.mm(gdp_mixer(self, p, x, md, i), p["gdp_out"]))
                streams, (shared_in,) = self._norm_reads(
                    streams,
                    (u, p["mix_write"], p["mix_retain"]),
                    ((p["moe_read"], p["pre_moe_norm"]),),
                    read_streams,
                )
                u = self.moe(p, routed_in, shared_in, md.all_rank_num_tokens, tp_reduce)
                pending = (u, p["moe_write"], p["moe_retain"])

        _, (out,) = self._norm_reads(
            streams, pending, ((self._readout, self.w["final_norm"]),), read_streams
        )
        return out


class ShortcutForCausalLM(DecoderModelForCausalLM):
    resets_recurrent_state_in_context = True

    @classmethod
    def get_preferred_kv_cache_manager_version(cls, pretrained_config=None):
        return "V2"

    def __init__(self, model_config):
        cfg = model_config.pretrained_config
        super().__init__(
            self.decoder_cls(model_config),
            config=model_config,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )
        self.lora_owner = ShortcutLoraOwner(
            module_path="model",
            head_size=self.model.head_dim,
            attention_tp=self.model.tp_size,
        )
        self.draft_model = None
        self.spec_worker = None
        if self.model.mtp_enabled:
            from tensorrt_llm._torch.speculative import (
                get_spec_worker,
                should_use_separate_draft_kv_cache,
            )

            from .mtp import DraftModel

            spec = model_config.spec_config
            self.draft_model = DraftModel(self.model, self.lm_head, self.logits_processor)
            self.spec_worker = get_spec_worker(
                spec,
                model_config,
                model_config.mapping,
                use_separate_draft_kv_cache=should_use_separate_draft_kv_cache(spec),
            )

    def set_guided_decoder(self, guided_decoder) -> bool:
        return self.spec_worker is not None and self.spec_worker.set_guided_decoder(guided_decoder)

    def forward(
        self,
        attn_metadata,
        input_ids=None,
        position_ids=None,
        inputs_embeds=None,
        return_context_logits=False,
        spec_metadata=None,
        lora_params=None,
        **kwargs,
    ):
        if self.spec_worker is None:
            return super().forward(
                attn_metadata,
                input_ids,
                position_ids,
                inputs_embeds,
                return_context_logits,
                spec_metadata,
                lora_params,
                **kwargs,
            )
        if spec_metadata is None:
            raise ValueError("MTP requires speculative decoding metadata.")
        hidden = self.model(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            lora_params=lora_params,
        )
        logits = self.logits_processor(
            hidden[spec_metadata.gather_ids], self.lm_head, attn_metadata, True
        )
        return self.spec_worker(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden,
            logits=logits,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=self.draft_model,
            resource_manager=kwargs.get("resource_manager"),
        )

    def load_weights(self, weights, *args, **kwargs):
        self.weight_loader(self, weights)

    def cache_derived_state(self):
        super().cache_derived_state()
        self.model.build_layer_views()
