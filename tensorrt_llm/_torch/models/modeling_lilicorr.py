# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFlash drafter with LiLiCorr candidate scoring and checkpoint-owned logits."""

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from ..model_config import ModelConfig
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import Linear
from .checkpoints.base_weight_mapper import BaseWeightMapper
from .modeling_dflash import DFlashForCausalLM


class LiLiCorrRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = hidden.float() * torch.rsqrt(
            hidden.float().square().mean(-1, keepdim=True) + self.eps
        )
        return normalized.to(hidden.dtype) * self.weight


class LiLiCorrAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * hidden_size))
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden.shape
        qkv = F.linear(hidden, self.in_proj_weight, self.in_proj_bias)
        q, k, v = (
            x.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
            for x in qkv.chunk(3, dim=-1)
        )
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self.out_proj(attended.transpose(1, 2).reshape(batch, length, width))


class LiLiCorrMLP(nn.Module):
    def __init__(self, input_size: int, intermediate_size: int, output_size: int) -> None:
        super().__init__()
        self.up_proj = nn.Linear(input_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, output_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.up_proj(hidden)))


class LiLiCorrLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, eps: float) -> None:
        super().__init__()
        self.attn_norm = LiLiCorrRMSNorm(hidden_size, eps)
        self.attn = LiLiCorrAttention(hidden_size, num_heads)
        self.mlp_norm = LiLiCorrRMSNorm(hidden_size, eps)
        self.mlp = LiLiCorrMLP(hidden_size, int(hidden_size * mlp_ratio), hidden_size)

    def forward(self, hidden: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attn(self.attn_norm(hidden), bias)
        return hidden + self.mlp(self.mlp_norm(hidden))


class LiLiCorrHead(nn.Module):
    """Score adjacent candidates using all candidates and the accepted target context."""

    def __init__(
        self,
        *,
        model_hidden_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        block_size: int,
        candidate_topk: int,
        factor_dim: int,
        rms_norm_eps: float,
        vector_eps: float,
        logit_scale: float,
    ) -> None:
        super().__init__()
        if (
            min(model_hidden_size, hidden_size, num_layers, num_heads, candidate_topk, factor_dim)
            < 1
        ):
            raise ValueError("LiLiCorr dimensions must be positive")
        if hidden_size % num_heads or block_size < 2 or mlp_ratio <= 0:
            raise ValueError("Invalid LiLiCorr attention, block or MLP dimensions")
        if not 0 < vector_eps < 0.5 or logit_scale <= 0:
            raise ValueError("Invalid LiLiCorr normalization epsilon or logit scale")
        self.block_size = block_size
        self.candidate_topk = candidate_topk
        self.hidden_size = hidden_size
        self.vector_eps = vector_eps
        self.logit_scale = logit_scale
        self.token_proj = (
            nn.Identity()
            if model_hidden_size == hidden_size
            else nn.Linear(model_hidden_size, hidden_size)
        )
        self.pass_hidden_proj = nn.Linear(model_hidden_size, hidden_size)
        self.feature_norm = nn.LayerNorm(5)
        self.feature_mlp = LiLiCorrMLP(5, hidden_size, hidden_size)
        self.slot_embedding = nn.Parameter(torch.zeros(1, block_size - 1, 1, hidden_size))
        self.rank_embedding = nn.Parameter(torch.zeros(1, 1, candidate_topk, hidden_size))
        self.relative_slot_bias = nn.Parameter(torch.zeros(num_heads, 2 * block_size - 1))
        self.same_slot_bias = nn.Parameter(torch.zeros(num_heads))
        self.context_proj = nn.Linear(model_hidden_size, hidden_size)
        self.layers = nn.ModuleList(
            LiLiCorrLayer(hidden_size, num_heads, mlp_ratio, rms_norm_eps)
            for _ in range(num_layers)
        )
        self.output_norm = LiLiCorrRMSNorm(hidden_size, rms_norm_eps)
        self.anchor_norm = LiLiCorrRMSNorm(hidden_size, rms_norm_eps)
        self.factor_input_proj = nn.Linear(3 * hidden_size, hidden_size)
        self.out_head = nn.Linear(hidden_size, factor_dim)
        self.in_head = nn.Linear(hidden_size, factor_dim)
        self.anchor_out_head = nn.Linear(hidden_size, factor_dim)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        candidate_log_probs: torch.Tensor,
        pass_hidden: torch.Tensor,
        anchor_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return start [B, C] and transition [B, S-1, C, C] logits.

        Inputs are candidate embeddings [B, S, C, H], vocabulary-normalized
        log probabilities [B, S, C], draft states [B, S, H], and the projected
        target row that predicted the anchor token [B, H]. S excludes the anchor.
        """
        batch, slots, topk = candidate_log_probs.shape
        if not 1 <= slots < self.block_size or topk != self.candidate_topk:
            raise ValueError("LiLiCorr candidate lattice does not match its trained dimensions")
        log_probs = candidate_log_probs.float()
        ranks = torch.arange(topk, device=log_probs.device, dtype=torch.float32)
        rank_fraction = (ranks / max(topk - 1, 1)).expand_as(log_probs)
        is_top1 = (ranks == 0).float().expand_as(log_probs)
        features = torch.stack(
            (
                log_probs,
                log_probs.exp(),
                log_probs - log_probs.amax(-1, keepdim=True),
                rank_fraction,
                is_top1,
            ),
            dim=-1,
        ).to(pass_hidden.dtype)
        hidden = self.token_proj(token_embeddings) + self.pass_hidden_proj(pass_hidden).unsqueeze(
            -2
        )
        hidden = hidden + self.feature_mlp(self.feature_norm(features))
        hidden = hidden + self.slot_embedding[:, :slots] + self.rank_embedding
        hidden = hidden.reshape(batch, slots * topk, self.hidden_size)
        slot_ids = torch.arange(slots, device=hidden.device).repeat_interleave(topk)
        relative = slot_ids[:, None] - slot_ids[None, :]
        bias = self.relative_slot_bias[:, relative + self.block_size - 1]
        bias = bias + (relative == 0).unsqueeze(0) * self.same_slot_bias[:, None, None]
        for layer in self.layers:
            hidden = layer(hidden, bias.to(hidden.dtype).unsqueeze(0))
        hidden = self.output_norm(hidden).reshape(batch, slots, topk, self.hidden_size)
        anchor = self.anchor_norm(self.context_proj(anchor_hidden))
        anchor_rows = anchor[:, None, None, :].expand_as(hidden)
        factors = F.silu(
            self.factor_input_proj(torch.cat((hidden, anchor_rows, hidden * anchor_rows), dim=-1))
        )
        outgoing = F.normalize(self.out_head(factors), dim=-1, eps=self.vector_eps)
        incoming = F.normalize(self.in_head(factors), dim=-1, eps=self.vector_eps)
        anchor_out = F.normalize(self.anchor_out_head(anchor), dim=-1, eps=self.vector_eps)
        start = (anchor_out[:, None, :] * incoming[:, 0]).sum(-1)
        pairs = outgoing[:, :-1] @ incoming[:, 1:].transpose(-1, -2)
        return start.float() * self.logit_scale, pairs.float() * self.logit_scale


def lilicorr_proposal_logits(start: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """Return [B, S, C] proposal rows along the greedy candidate path.

    Subsequent sampling must use these same rows for both drawing and verification.
    """
    rows = [start]
    previous = start.argmax(-1)
    for slot in range(pairs.shape[1]):
        row = (
            pairs[:, slot]
            .gather(1, previous[:, None, None].expand(-1, 1, pairs.shape[-1]))
            .squeeze(1)
        )
        rows.append(row)
        previous = row.argmax(-1)
    return torch.stack(rows, dim=1)


def _linear_quant_config(model_config: ModelConfig, name: str) -> QuantConfig | None:
    """Resolve checkpoint module metadata, with global exclusions taking precedence."""
    candidates = (name, "model." + name)
    config = model_config.quant_config
    if config is not None and any(
        config.is_module_excluded_from_quantization(n) for n in candidates
    ):
        return None
    if model_config.quant_config_dict is not None:
        return next(
            (
                model_config.quant_config_dict[n]
                for n in candidates
                if n in model_config.quant_config_dict
            ),
            None,
        )
    return config


def _load_linear(
    weights: dict[str, torch.Tensor],
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
    *,
    bias: bool,
    quant_config: QuantConfig | None,
    device: torch.device,
) -> nn.Module:
    algo = quant_config.quant_algo if quant_config is not None else None
    scales = {
        None: (),
        QuantAlgo.FP8: ("weight_scale", "input_scale"),
        QuantAlgo.NVFP4: ("weight_scale", "weight_scale_2", "input_scale"),
        QuantAlgo.W4A16_NVFP4: ("weight_scale", "weight_scale_2"),
    }
    if algo not in scales:
        raise NotImplementedError(f"Unsupported LiLiCorr projection quantization: {algo}")
    required = {"weight", *scales[algo]}
    if bias:
        required.add("bias")
    missing = required - weights.keys()
    if missing:
        raise ValueError(f"LiLiCorr linear layer is missing {sorted(missing)}")
    if algo is None:
        if weights["weight"].dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            raise ValueError("Packed or FP8 LiLiCorr weights require quantization metadata")
        layer = nn.Linear(in_features, out_features, bias=bias, dtype=dtype, device=device)
        layer.load_state_dict(weights, strict=True)
    else:
        layer = Linear(
            in_features, out_features, bias=bias, dtype=dtype, quant_config=quant_config
        ).to(device)
        layer.load_weights([weights])
    return layer


def _normalize_lilicorr_name(name: str) -> str:
    name = name.replace("feature_mlp.0.", "feature_norm.")
    name = name.replace("feature_mlp.1.", "feature_mlp.up_proj.")
    name = name.replace("feature_mlp.3.", "feature_mlp.down_proj.")
    return name.replace(".mlp.0.", ".mlp.up_proj.").replace(".mlp.2.", ".mlp.down_proj.")


def _normalize_lilicorr_weights(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Accept sequential and named-MLP exports without changing their tensor semantics."""
    normalized = {}
    for name, tensor in weights.items():
        name = _normalize_lilicorr_name(name)
        if name in ("slot_embedding", "rank_embedding") and tensor.ndim == 5:
            tensor = tensor.squeeze(1)
        if name in normalized:
            raise ValueError(f"Duplicate LiLiCorr parameter {name}")
        normalized[name] = tensor
    return normalized


class LiLiCorrForCausalLM(DFlashForCausalLM):
    _uses_lilicorr = True

    def __init__(
        self, draft_config: ModelConfig, *, dflash_attention_backend: str = "AUTO"
    ) -> None:
        super().__init__(draft_config, dflash_attention_backend=dflash_attention_backend)
        # Gate and up have independently calibrated FP4 scales. Keep their
        # packed projections separate so neither calibration is discarded.
        for layer_idx, layer in enumerate(self.model.layers):
            mlp = layer.mlp
            if isinstance(mlp, GatedMLP) and not mlp.split_gate_up:
                layer.mlp = GatedMLP(
                    hidden_size=mlp.hidden_size,
                    intermediate_size=mlp.intermediate_size,
                    bias=mlp.gate_up_proj.has_bias,
                    activation=mlp.activation,
                    dtype=self.config.torch_dtype,
                    config=self.model_config,
                    layer_idx=layer_idx,
                    split_gate_up=True,
                    swiglu_limit=mlp.swiglu_limit,
                    swiglu_alpha=mlp.swiglu_alpha,
                    swiglu_beta=mlp.swiglu_beta,
                )
                for name, module in layer.mlp.named_modules():
                    if isinstance(module, Linear):
                        module.quant_config = _linear_quant_config(
                            self.model_config, f"layers.{layer_idx}.mlp.{name}"
                        )
                        module._weights_created = False
                        module.create_weights()
        settings = getattr(self.config, "dflash_config", {})
        fields = (
            "hidden_size",
            "num_layers",
            "num_heads",
            "mlp_ratio",
            "candidate_topk",
            "factor_dim",
            "vector_eps",
            "logit_scale",
        )
        missing = [f"lilicorr_{field}" for field in fields if f"lilicorr_{field}" not in settings]
        if missing:
            raise ValueError(f"LiLiCorr checkpoint is missing config fields {missing}")
        if self.block_size is None:
            raise ValueError("LiLiCorr checkpoint requires dflash_config.block_size or block_size")
        if getattr(self.config, "is_causal", False) or settings.get("causal", False):
            raise ValueError("LiLiCorr requires non-causal draft attention")
        self.lilicorr = LiLiCorrHead(
            model_hidden_size=self.config.hidden_size,
            block_size=self.block_size,
            rms_norm_eps=self.config.rms_norm_eps,
            **{field: settings[f"lilicorr_{field}"] for field in fields},
        )
        self.has_own_lm_head = bool(getattr(self.config, "has_own_lm_head", False))
        if not 0 <= self._dflash2_conv_taps <= self.block_size:
            raise ValueError("LiLiCorr conv_kernel_size must be between 0 and block_size")
        if self._dflash2_conv_taps and self._dflash2_conv_group_size < 1:
            raise ValueError("LiLiCorr grouped convolution requires conv_group_size > 0")
        logger.info(
            f"LiLiCorr enabled: block_size={self.block_size}, "
            f"candidate_topk={self.lilicorr.candidate_topk}, own_lm_head={self.has_own_lm_head}"
        )

    def load_weights(
        self,
        weights: dict[str, torch.Tensor],
        weight_mapper: BaseWeightMapper | None = None,
        **kwargs: object,
    ) -> None:
        weights = {name.removeprefix("model."): value for name, value in weights.items()}
        raw_head_weights = {
            name.removeprefix("lilicorr."): value
            for name, value in weights.items()
            if name.startswith("lilicorr.")
        }
        head_weights = _normalize_lilicorr_weights(raw_head_weights)
        source_names = {
            _normalize_lilicorr_name(name): "lilicorr." + name.rsplit(".", 1)[0]
            for name in raw_head_weights
        }
        device = self.model.norm.weight.device
        self.lilicorr.to(device=device, dtype=self.config.torch_dtype)
        consumed = set()
        for name, layer in list(self.lilicorr.named_modules()):
            if isinstance(layer, nn.Linear):
                prefix = name + "."
                values = {
                    key[len(prefix) :]: value
                    for key, value in head_weights.items()
                    if key.startswith(prefix)
                }
                loaded = _load_linear(
                    values,
                    layer.in_features,
                    layer.out_features,
                    self.config.torch_dtype,
                    bias=layer.bias is not None,
                    quant_config=_linear_quant_config(
                        self.model_config, source_names.get(name + ".weight", "lilicorr." + name)
                    ),
                    device=device,
                )
                self.lilicorr.set_submodule(name, loaded)
                consumed.update(prefix + key for key in values)
        remaining = {name: value for name, value in head_weights.items() if name not in consumed}
        expected = {
            name
            for name, _ in self.lilicorr.named_parameters()
            if not any(name.startswith(key.rsplit(".", 1)[0] + ".") for key in consumed)
        }
        if remaining.keys() != expected:
            raise ValueError(
                f"LiLiCorr head weight mismatch: missing={sorted(expected - remaining.keys())}, "
                f"unexpected={sorted(remaining.keys() - expected)}"
            )
        self.lilicorr.load_state_dict(remaining, strict=False)
        weights = {
            name: value for name, value in weights.items() if not name.startswith("lilicorr.")
        }
        if self.has_own_lm_head:
            head = {
                name.removeprefix("lm_head."): value
                for name, value in weights.items()
                if name.startswith("lm_head.")
            }
            if "weight" not in head:
                raise ValueError("LiLiCorr has_own_lm_head requires checkpoint lm_head weights")
            self.lm_head.load_weights([head])
            weights = {
                name: value for name, value in weights.items() if not name.startswith("lm_head.")
            }
        if self._dflash2_conv_taps:
            weights = self._load_dflash2_weights(weights, load_selector=False)
        super().load_weights(weights, weight_mapper=weight_mapper, **kwargs)
        # The backbone loads partially to leave shared embeddings and the output
        # head alone. Complete its accumulated quantization scales before use.
        for module in self.model.modules():
            if isinstance(module, Linear):
                module.process_weights_after_loading()

    def _load_target_projection(self, weights: dict[str, torch.Tensor]) -> None:
        projection = {
            name.removeprefix("fc."): value
            for name, value in weights.items()
            if name.startswith("fc.")
        }
        if "weight" not in projection:
            raise ValueError("LiLiCorr target projection is missing fc.weight")
        weight = projection["weight"]
        quant_config = _linear_quant_config(self.model_config, "fc")
        in_features = weight.shape[1]
        if (
            weight.dtype == torch.uint8
            and quant_config is not None
            and quant_config.quant_algo in (QuantAlgo.NVFP4, QuantAlgo.W4A16_NVFP4)
        ):
            in_features *= 2
        self.fc = _load_linear(
            projection,
            in_features,
            self.config.hidden_size,
            self.config.torch_dtype,
            bias=False,
            quant_config=quant_config,
            device=self.model.norm.weight.device,
        )
        for name in projection:
            del weights["fc." + name]

    def project_target_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.hidden_norm(self.fc(hidden_states.to(self.config.torch_dtype)))

    def load_weights_from_target_model(self, target_model: nn.Module) -> None:
        embedding = target_model.model.embed_tokens
        if (
            embedding.embedding_dim != self.config.hidden_size
            or embedding.num_embeddings != self.config.vocab_size
        ):
            raise ValueError("LiLiCorr requires matching target embedding width and vocabulary")
        self.draft_model_full.model.embed_tokens = embedding
        if not self.has_own_lm_head:
            self.draft_model_full.lm_head = target_model.lm_head
            self.lm_head = target_model.lm_head

    def select_lilicorr_path(
        self,
        candidate_ids: torch.Tensor,
        candidate_log_probs: torch.Tensor,
        draft_hidden: torch.Tensor,
        anchor_hidden: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.model.embed_tokens(candidate_ids.reshape(-1)).reshape(
            *candidate_ids.shape, self.config.hidden_size
        )
        start, pairs = self.lilicorr(embeddings, candidate_log_probs, draft_hidden, anchor_hidden)
        return lilicorr_proposal_logits(start, pairs)
