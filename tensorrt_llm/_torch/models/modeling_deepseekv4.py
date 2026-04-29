# --------------------------------------------------
# Portions of this code were derived from DeepSeek‑V3:
#   https://github.com/deepseek-ai/DeepSeek-V3
#
# MIT License

# Copyright (c) 2023 DeepSeek

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# --------------------------------------------------

import copy
import math
import os
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

import torch
import triton
import triton.language as tl
from torch import nn
from tqdm import tqdm
from transformers import PretrainedConfig

import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from tensorrt_llm._ipc_utils import can_access_peer
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..attention_backend.sparse.deepseek_v4.deepseek_v4 import DeepseekV4TrtllmAttentionMetadata
from ..distributed import (
    AllReduce,
    AllReduceFusionOp,
    AllReduceParams,
    MoEAllReduce,
    MoEAllReduceParams,
    allgather,
)
from ..model_config import ModelConfig
from ..modules.attention import MLA
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.engram import Engram, EngramConfig, EngramHashProvider
from ..modules.fused_moe import (
    CutlassFusedMoE,
    DeepSeekV4MoeRoutingMethod,
    MoE,
    MoEWeightLoadingMode,
    TritonFusedMoE,
    TRTLLMGenFusedMoE,
    create_moe,
    get_moe_cls,
)
from ..modules.fused_moe.fused_moe_wide_ep import WideEPMoE
from ..modules.linear import Linear
from ..modules.mhc.hyper_connection import HCHead, HCState, mHC

# isort: off
from ..modules.fused_moe.routing import (
    get_cached_perfect_router_logits,
    precompute_common_perfect_router_logits,
)

# isort: on
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import TensorParallelMode, WeightsLoadingConfig
from ..modules.multi_stream_utils import maybe_execute_in_parallel
from ..modules.rms_norm import RMSNorm
from ..peft.lora.layer import LoraLayer
from ..speculative import SpecMetadata
from ..utils import AuxStreamType, EventType, Fp4QuantizedTensor, create_lm_head_tp_mapping
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import DecoderModel, EagerFusionConfig, filter_weights, register_auto_model


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    Dequantizes weights using the provided scaling factors and stores the result.

    Args:
        x_ptr (tl.pointer): Pointer to the quantized weights.
        s_ptr (tl.pointer): Pointer to the scaling factors.
        y_ptr (tl.pointer): Pointer to the output buffer for dequantized weights.
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): Size of the block for tiling.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M, N).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    assert x.is_contiguous() and s.is_contiguous(), "Input tensors must be contiguous"
    assert x.dim() == 2 and s.dim() == 2, "Input tensors must have 2 dimensions"
    if s.dtype != torch.float32:
        s = s.float()
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE"]), triton.cdiv(N, meta["BLOCK_SIZE"]))  # noqa: E731
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


def _deepseek_v4_pos_embd_params(
    config: PretrainedConfig, model_config: ModelConfig, layer_idx: Optional[int]
) -> PositionalEmbeddingParams:
    rope_params = RopeParams.from_config(config)

    compress_ratios = None
    if model_config.sparse_attention_config is not None:
        compress_ratios = getattr(model_config.sparse_attention_config, "compress_ratios", None)
    if not compress_ratios:
        compress_ratios = getattr(config, "compress_ratios", None)

    compress_ratio = 0
    if compress_ratios and layer_idx is not None:
        compress_ratio = compress_ratios[min(layer_idx, len(compress_ratios) - 1)]

    if compress_ratio > 1:
        rope_params.theta = getattr(config, "compress_rope_theta", rope_params.theta)
        rope_params.scale_type = RotaryScalingType.yarn
        # DeepSeek-V4 reference applies YaRN frequency interpolation but does
        # not scale the cos/sin amplitudes.
        rope_params.mscale = 0.0
        rope_params.mscale_all_dim = 0.0
        pos_type = PositionEmbeddingType.yarn
    else:
        # DeepSeek-V4 reference uses base RoPE, without YaRN, for SWA-only
        # layers. Internal configs normalize checkpoint ratio 0 to 1, so both
        # values are treated as uncompressed here.
        rope_params.theta = getattr(config, "rope_theta", rope_params.theta)
        rope_params.scale_type = RotaryScalingType.none
        rope_params.scale = 1.0
        pos_type = PositionEmbeddingType.rope_gptj

    return PositionalEmbeddingParams(
        type=pos_type,
        rope=rope_params,
        is_neox=False,
    )


@torch.compile(dynamic=True)
def moe_reduce_add_shared_output(routed_output, shared_output):
    routed_output = torch.sum(routed_output, dim=1, keepdim=False)
    return shared_output + routed_output


# Per-attention parameter renames inside `attn.<X>` / `mtp.0.attn.<X>`.
# Maps checkpoint name component → model name component.
_ATTN_PARAM_RENAME = {
    "wq_a": "q_a_proj",
    "wq_b": "q_b_proj",
    "wkv": "kv_a_proj_with_mqa",
    "wo_b": "o_b_proj",
    "q_norm": "q_a_layernorm",
    "kv_norm": "kv_a_layernorm",
}

# Shared expert leaf rename: checkpoint w1/w3/w2 → model gate/up/down. The
# loader's `params_map` then fuses `gate_proj` + `up_proj` into `gate_up_proj`.
_SHARED_EXPERT_RENAME = {
    "w1": "gate_proj",
    "w3": "up_proj",
    "w2": "down_proj",
}


def _resolve_enable_fused_hc(config: PretrainedConfig) -> bool:
    """Resolve the DeepSeek-V4 fused HC boundary-fusion knob."""
    env = os.environ.get("TRTLLM_MHC_ENABLE_FUSED_HC")
    if env is not None:
        return env not in ("0", "false", "False")
    return bool(getattr(config, "enable_fused_hc", True))


def _remap_deepseek_v4_checkpoint_keys(
    weights: Dict, num_hidden_layers: int, kv_lora_rank: int = 448
) -> Dict:
    """Convert DeepSeek-V4 checkpoint keys to model named-parameter keys.

    Why: the upstream DS-V4 release uses keys like ``layers.X.attn.wkv.weight``,
    ``mtp.0.ffn.experts.0.w1.scale``, ``embed.weight``, ``head.weight``. The
    TRT-LLM model exposes them as ``model.layers.X.self_attn.kv_a_proj_with_mqa.weight``,
    ``model.layers.{N}.mlp.experts.0.w1.weight_scale_inv``, ``model.embed_tokens.weight``,
    ``lm_head.weight``. This pass renames keys, fuses split projections that the
    model represents as one tensor, and synthesizes default values for params
    the model has but the checkpoint does not (so the existing loader's strict
    key lookup does not raise).

    Caveats — limitations carried as TODOs (kept here so future readers can
    audit them in one place rather than chasing comments):
      * ``self_attn.indexer.wk.weight`` and ``self_attn.indexer.k_norm.{weight,bias}``
        are zero-filled — V4 indexer's k path is served by the compressor, so
        the base ``Indexer.wk`` / ``k_norm`` are unused at forward time.
      * Routed experts can use either FP8 block scales or the packed MXFP4
        layout. The first routed expert weight determines the scale suffix used
        for all routed expert tensors in the shard.
      * ``self_attn.o_a_proj`` is loaded by the DeepSeek-V4 loader because it
        is a direct MLA parameter, not a child Linear module.
      * ``mtp.0.head.weight`` is dropped — DeepSeekV4MTP reuses the main
        ``lm_head`` via ``shared_head``. Flash omits this key entirely; Flash-Base
        carries it but matches the main head, so we let the main head win.
    """
    mtp_layer_prefix = f"model.layers.{num_hidden_layers}"
    routed_moe_scale_name = "weight_scale_inv"
    for key, value in weights.items():
        if (
            key.startswith("layers.")
            and ".ffn.experts." in key
            and key.endswith(".weight")
            and getattr(value, "ndim", 0) == 2
            and value.dtype in (torch.int8, torch.uint8)
        ):
            routed_moe_scale_name = "weight_scale"
            break

    def _rename_attn_subkey(rest: str) -> Optional[str]:
        # rest examples: "wq_a.weight", "wq_a.scale", "wo_a.weight",
        # "attn_sink", "compressor.wkv.weight", "indexer.wq_b.scale",
        # "kv_norm.weight"
        # ``attn_sink`` is loaded by the ``mqa`` branch in the per-module
        # loader, which reads it under the parent ``self_attn.attn_sink``
        # key. Pass through unchanged.
        if rest == "attn_sink":
            return "attn_sink"
        # `wo_a` is an nn.Parameter on the model side (not a Linear), so
        # `wo_a.weight` carries the value directly into `o_a_proj` without
        # a trailing ``.weight``. Retain `.scale` so the loader can dequantize
        # FP8 block-scaled checkpoints before assigning the bf16 parameter.
        if rest == "wo_a.weight":
            return "o_a_proj"
        if rest == "wo_a.scale":
            return "o_a_proj.weight_scale_inv"
        # Compressor / indexer paths — pass through with .scale rename, plus
        # wkv+wgate fusion handled separately below.
        if rest.startswith("compressor.") or rest.startswith("indexer."):
            return rest.replace(".scale", ".weight_scale_inv")
        head, sep, tail = rest.partition(".")
        new_head = _ATTN_PARAM_RENAME.get(head, head)
        if tail == "scale":
            tail = "weight_scale_inv"
        return f"{new_head}.{tail}" if sep else new_head

    def _rename_ffn_subkey(rest: str) -> str:
        # Examples:
        #   gate.weight / gate.tid2eid → gate.weight / gate.tid2eid
        #   gate.bias → gate.e_score_correction_bias
        #   experts.<i>.<w1|w2|w3>.<weight|scale> → experts.<i>.<w1|w2|w3>.<weight|weight_scale*>
        #   shared_experts.<w1|w3|w2>.<weight|scale> → shared_experts.<gate|up|down>_proj.<weight|weight_scale_inv>
        if rest == "gate.bias":
            return "gate.e_score_correction_bias"
        if rest.startswith("experts.") and rest.endswith(".scale"):
            return f"{rest[: -len('.scale')]}.{routed_moe_scale_name}"
        rest = rest.replace(".scale", ".weight_scale_inv")
        # Non-hashed layers carry the routing logit bias as `gate.bias`; the
        # model wires it through `DeepseekV4Gate.e_score_correction_bias`.
        if rest == "gate.bias":
            return "gate.e_score_correction_bias"
        if rest.startswith("shared_experts."):
            parts = rest.split(".")
            if len(parts) >= 2 and parts[1] in _SHARED_EXPERT_RENAME:
                parts[1] = _SHARED_EXPERT_RENAME[parts[1]]
            rest = ".".join(parts)
        return rest

    def _rename_layer_subkey(rest: str) -> Optional[str]:
        # rest examples: "attn_norm.weight", "ffn_norm.weight",
        # "hc_attn_fn", "attn.wkv.weight", "ffn.experts.0.w1.weight"
        if rest == "attn_norm.weight":
            return "input_layernorm.weight"
        if rest == "ffn_norm.weight":
            return "post_attention_layernorm.weight"
        # ``hc_attn_*`` and ``hc_ffn_*`` are loaded by ``load_flat_hc_weights``
        # which builds the lookup key as ``f"{stem}_{a}"`` with the parent
        # module path as the stem (e.g. ``model.layers.0.hc_attn_fn``). Pass
        # the flat-underscore form through unchanged so that lookup succeeds.
        if rest.startswith("hc_attn_") or rest.startswith("hc_ffn_"):
            return rest
        if rest.startswith("attn."):
            new_sub = _rename_attn_subkey(rest[len("attn.") :])
            return None if new_sub is None else f"self_attn.{new_sub}"
        if rest.startswith("ffn."):
            return f"mlp.{_rename_ffn_subkey(rest[len('ffn.') :])}"
        return rest

    out: Dict[str, torch.Tensor] = {}
    # Pending fusions: collected first, materialized at the end so we don't
    # depend on iteration order.
    compressor_split: Dict[str, Dict[str, torch.Tensor]] = {}

    def _record_compressor_part(model_key: str, part: str, tensor: torch.Tensor):
        # model_key looks like "...self_attn.compressor.<part>.weight" or with
        # ".indexer.compressor.". Strip trailing ".<part>.weight" → ".<part>"
        # is wgate or wkv; we want a stable "fusion bucket" key that ends at
        # the parent compressor.
        bucket = model_key.rsplit(f".{part}.", 1)[0]
        compressor_split.setdefault(bucket, {})[part] = tensor

    def _emit_or_collect(model_key: str, tensor: torch.Tensor):
        """Route a (model_key, tensor) pair and collect compressor fusions."""
        if ".compressor." in model_key and (
            model_key.endswith(".wkv.weight") or model_key.endswith(".wgate.weight")
        ):
            part = "wkv" if model_key.endswith(".wkv.weight") else "wgate"
            _record_compressor_part(model_key, part, tensor)
            return
        if (
            routed_moe_scale_name == "weight_scale"
            and ".mlp.experts." in model_key
            and (model_key.endswith(".weight") or model_key.endswith(".weight_scale"))
            and tensor.dtype != torch.uint8
        ):
            tensor = tensor.view(torch.uint8)
        out[model_key] = tensor

    for k, v in weights.items():
        # Top-level keys that don't go through the layer/mtp branches.
        if k == "embed.weight":
            out["model.embed_tokens.weight"] = v
            continue
        if k == "head.weight":
            out["lm_head.weight"] = v
            continue
        if k == "norm.weight":
            out["model.norm.weight"] = v
            continue
        if k.startswith("hc_head_"):
            # ``load_flat_hc_weights`` looks up ``f"{stem}_{a}"`` with stem set
            # to the module's full name path (``model.hc_head``). Emit the flat
            # checkpoint key under the parent prefix so the lookup matches.
            out[f"model.{k}"] = v
            continue

        # mtp.0.head.weight is intentionally dropped (Flash-Base only); see
        # docstring caveats.
        if k == "mtp.0.head.weight":
            continue

        # mtp.0.* — route to model.layers.{num_hidden_layers}.*
        if k.startswith("mtp.0."):
            rest = k[len("mtp.0.") :]
            # MTP-only keys: enorm, hnorm map directly; norm maps to
            # shared_head.norm; hc_head_* maps under shared_head; e_proj /
            # h_proj are loaded as two separate Linear modules (no fused
            # eh_proj on the MTP layer).
            if rest in ("enorm.weight", "hnorm.weight"):
                out[f"{mtp_layer_prefix}.{rest}"] = v
                continue
            if rest == "norm.weight":
                out[f"{mtp_layer_prefix}.shared_head.norm.weight"] = v
                continue
            if rest.startswith("hc_head_"):
                # The MTP HCHead is wired at ``...shared_head.hc_head``, so
                # ``load_flat_hc_weights`` matches ``names[-1] == "hc_head"``
                # first and computes ``stem = ".".join(names)`` with the full
                # module path. Emit at that full path with the flat suffix
                # so the lookup matches.
                out[f"{mtp_layer_prefix}.shared_head.{rest}"] = v
                continue
            for proj in ("e_proj", "h_proj"):
                if rest.startswith(f"{proj}."):
                    suffix = rest[len(f"{proj}.") :]
                    if suffix == "scale":
                        suffix = "weight_scale_inv"
                    out[f"{mtp_layer_prefix}.{proj}.{suffix}"] = v
                    break
            else:
                # General per-layer transform reused for the MTP layer.
                new_rest = _rename_layer_subkey(rest)
                if new_rest is None:
                    continue
                _emit_or_collect(f"{mtp_layer_prefix}.{new_rest}", v)
            continue

        # layers.<i>.* — route to model.layers.<i>.*
        if k.startswith("layers."):
            parts = k.split(".", 2)
            if len(parts) < 3:
                continue
            layer_idx, rest = parts[1], parts[2]
            new_rest = _rename_layer_subkey(rest)
            if new_rest is None:
                continue
            _emit_or_collect(f"model.layers.{layer_idx}.{new_rest}", v)
            continue

        # Anything else: pass through. This catches future top-level keys
        # added by the upstream release without silently dropping them.
        out[k] = v

    # Materialize compressor wkv_gate fusion: model has a single Linear with
    # out_features = state_dim * 2; checkpoint splits it as `wkv` (kv half)
    # and `wgate` (gate half). Concatenating wkv first matches the order the
    # compressor kernels expect (kv_score = [kv | gate]).
    for bucket, parts in compressor_split.items():
        if "wkv" not in parts or "wgate" not in parts:
            # Partial — emit what we have so the loader fails loudly with a
            # specific missing key rather than a silent shape mismatch.
            for name, tensor in parts.items():
                out[f"{bucket}.{name}.weight"] = tensor
            continue
        out[f"{bucket}.wkv_gate.weight"] = torch.cat([parts["wkv"], parts["wgate"]], dim=0)

    return out


class DeepseekV4WeightLoader:
    def __init__(self, model, is_draft_model: bool = False):
        self.model = model
        self.config = model.config
        self.model_config = model.model_config
        self.is_draft_model = is_draft_model

    def load_weights(self, weights: Dict, skip_modules: List[str] = []):
        # If the checkpoint uses raw DS-V4 keys (layers.X.attn.wkv.weight,
        # mtp.0.*, embed.weight, head.weight), rewrite them to the model's
        # named-parameter keys before iterating modules. The detection is by
        # presence of any top-level "layers." key; HF-style checkpoints use
        # "model.layers." and skip this branch.
        if any(k == "embed.weight" or k.startswith("layers.") for k in weights):
            weights = _remap_deepseek_v4_checkpoint_keys(
                weights,
                num_hidden_layers=self.config.num_hidden_layers,
                kv_lora_rank=self.config.kv_lora_rank,
            )
            # Synthesize defaults (with correct shape pulled from the model)
            # for parameters the model has but the V4 checkpoint omits. We do
            # this in one place vs scattering zero-fills through the per-
            # module branches so the missing-key contract is auditable here.
            #   indexer.k_norm.weight     → ones
            #   indexer.k_norm.bias       → zeros
            #   indexer.wk.weight         → zeros (V4 indexer's k path is
            #                                served by compressor; wk unused)
            _ones_suffixes = ("self_attn.indexer.k_norm.weight",)
            _zeros_suffixes = (
                "self_attn.indexer.k_norm.bias",
                "self_attn.indexer.wk.weight",
            )
            model_params = dict(self.model.named_parameters())
            for pname, p in model_params.items():
                if pname in weights:
                    continue
                if any(pname.endswith(s) for s in _ones_suffixes):
                    weights[pname] = torch.ones_like(p, device="cpu")
                elif any(pname.endswith(s) for s in _zeros_suffixes):
                    weights[pname] = torch.zeros_like(p, device="cpu")

        def requantize_weight_with_new_scale(
            weight, weight_scale, old_scale_2, new_scale_2, device
        ):
            """
            Dequantize FP4 weights and requantize with a new scale.

            Args:
                weight: FP4 quantized weight tensor 2D [,]
                weight_scale: FP8 per-block scaling factors
                old_scale_2: original global scale (amax/(448*6))
                new_scale_2: new global scale (amax/(448*6))
                device: target device for computation

            Returns:
                (requantized_weight, new_weight_scale)
            """
            # Remember original dtype of weight_scale
            original_scale_dtype = weight_scale.dtype
            original_scale_shape = weight_scale.shape

            # Dequantize
            dequant_shape = (weight.shape[0], weight.shape[1] * 2)
            weight_dequant = (
                torch.ops.tensorrt_llm.e2m1_and_ufp8sf_scale_to_float_v2(
                    weight.contiguous(),
                    weight_scale.flatten().view(fp4_utils.float4_sf_dtype).contiguous(),
                    old_scale_2,
                    16,
                    1,
                    True,
                )
                .to(dtype=torch.bfloat16)
                .reshape(dequant_shape)
            )

            # Requantize using the new_scale_2
            weight_requant, weight_scale_requant = torch.ops.trtllm.fp4_quantize(
                weight_dequant.to(device),
                1.0 / new_scale_2.to(device),
                16,  # scaling_vector_size
                False,
            )

            # Ensure the returned scale has the same dtype as the input scale
            return weight_requant.cpu(), weight_scale_requant.reshape(original_scale_shape).view(
                original_scale_dtype
            ).cpu()

        def rename_moe_weight(weights: Dict, rename_rules: Dict):
            result = {}
            for key, value in weights.items():
                new_key = key
                for old, new in rename_rules.items():
                    new_key = new_key.replace(old, new)
                result[new_key] = value
            return result

        ## Prepare weights for TP
        def split(v, tp_size, idx, dim=0):
            if tp_size == 1:
                return v
            if len(v.shape) == 1:
                return torch.chunk(v, tp_size)[idx].contiguous()
            return torch.chunk(v, tp_size, dim=dim)[idx].contiguous()

        def split_matrix_tp(v, tensor_parallel, rank, dim):
            return split(v, tensor_parallel, rank, dim=dim)

        def load_kv_b_proj_and_k_b_proj_trans(module_name: str, is_scale: bool) -> torch.Tensor:
            weight_name = "weight" if not is_scale else "weight_scale_inv"
            local_qk_nope_head_dim = qk_nope_head_dim if not is_scale else qk_nope_head_dim // 128
            local_v_head_dim = v_head_dim if not is_scale else v_head_dim // 128
            local_kv_lora_rank = kv_lora_rank if not is_scale else kv_lora_rank // 128

            kv_b_proj = weights[f"{module_name}.{weight_name}"][:].unflatten(
                0,
                [
                    num_heads,
                    local_qk_nope_head_dim + local_v_head_dim,
                ],
            )

            if not self.model_config.mapping.enable_attention_dp:
                kv_b_proj = split_matrix_tp(kv_b_proj, tp_size, tp_rank, 0)
            k_nope_weight, v_weight = kv_b_proj.split(
                [local_qk_nope_head_dim, local_v_head_dim],
                dim=1,
            )
            weight_divisor = 1 if self.model_config.mapping.enable_attention_dp else tp_size
            local_num_heads = num_heads // weight_divisor

            k_nope_weight_trans = k_nope_weight.transpose(2, 1).contiguous()

            kv_b_proj = torch.concat(
                [
                    k_nope_weight.reshape(
                        local_num_heads * local_qk_nope_head_dim, local_kv_lora_rank
                    ),
                    v_weight.reshape(local_num_heads * local_v_head_dim, local_kv_lora_rank),
                ],
                dim=0,
            )

            return kv_b_proj, k_nope_weight_trans

        def load_kv_b_proj_and_k_b_proj_trans_dequant(module_name: str) -> torch.Tensor:
            weight_name = "weight"
            local_qk_nope_head_dim = qk_nope_head_dim
            local_v_head_dim = v_head_dim
            local_kv_lora_rank = kv_lora_rank

            kv_b_proj = weights[f"{module_name}.{weight_name}"][:].cuda()

            weight_name = "weight_scale_inv"
            kv_b_proj_scale = weights[f"{module_name}.{weight_name}"][:].cuda()

            kv_b_proj = weight_dequant(kv_b_proj, kv_b_proj_scale)
            kv_b_proj = kv_b_proj.unflatten(
                0,
                [
                    num_heads,
                    local_qk_nope_head_dim + local_v_head_dim,
                ],
            )
            if not self.model_config.mapping.enable_attention_dp:
                kv_b_proj = split_matrix_tp(kv_b_proj, tp_size, tp_rank, 0)
            k_nope_weight, v_weight = kv_b_proj.split(
                [local_qk_nope_head_dim, local_v_head_dim],
                dim=1,
            )
            weight_divisor = 1 if self.model_config.mapping.enable_attention_dp else tp_size
            local_num_heads = num_heads // weight_divisor

            k_nope_weight_trans = k_nope_weight.transpose(2, 1).contiguous()

            kv_b_proj = torch.concat(
                [
                    k_nope_weight.reshape(
                        local_num_heads * local_qk_nope_head_dim, local_kv_lora_rank
                    ),
                    v_weight.reshape(local_num_heads * local_v_head_dim, local_kv_lora_rank),
                ],
                dim=0,
            )

            return kv_b_proj, k_nope_weight_trans

        def load_o_a_proj(module_name: str, module) -> None:
            weight_name = f"{module_name}.o_a_proj"
            o_a_proj = weights[weight_name][:].unflatten(
                0,
                [
                    module.num_groups,
                    module.o_lora_rank,
                ],
            )

            scale_name = f"{weight_name}.weight_scale_inv"
            o_a_proj_scale = weights.get(scale_name)
            if o_a_proj_scale is not None:
                o_a_proj_scale = o_a_proj_scale[:].unflatten(
                    0,
                    [
                        module.num_groups,
                        module.o_lora_rank // 128,
                    ],
                )
            elif o_a_proj.dtype == torch.float8_e4m3fn:
                raise KeyError(f"Missing FP8 block scale for {weight_name}")

            if not self.model_config.mapping.enable_attention_dp:
                o_a_proj = split_matrix_tp(o_a_proj, tp_size, tp_rank, 0)
                if o_a_proj_scale is not None:
                    o_a_proj_scale = split_matrix_tp(o_a_proj_scale, tp_size, tp_rank, 0)

            if o_a_proj_scale is not None:
                o_a_proj = weight_dequant(
                    o_a_proj.reshape(-1, o_a_proj.shape[-1]).contiguous().cuda(),
                    o_a_proj_scale.reshape(-1, o_a_proj_scale.shape[-1]).contiguous().cuda(),
                ).view(o_a_proj.shape)

            module.o_a_proj.data.copy_(
                o_a_proj.reshape(module.o_a_proj.shape).to(module.o_a_proj.dtype)
            )

            if getattr(module, "o_a_proj_scale", None) is not None and o_a_proj_scale is not None:
                module.o_a_proj_scale.data.copy_(
                    o_a_proj_scale.reshape(module.o_a_proj_scale.shape)
                )

            if getattr(module, "o_a_proj_dequant", None) is not None and o_a_proj_scale is not None:
                module.o_a_proj_dequant.data.copy_(
                    o_a_proj.reshape(module.o_a_proj_dequant.shape).to(
                        module.o_a_proj_dequant.dtype
                    )
                )

        def split_kv_b_proj(kv_b_proj: torch.Tensor, is_scale: bool) -> torch.Tensor:
            local_qk_nope_head_dim = qk_nope_head_dim if not is_scale else qk_nope_head_dim // 128
            local_v_head_dim = v_head_dim if not is_scale else v_head_dim // 128

            weight_divisor = 1 if self.model_config.mapping.enable_attention_dp else tp_size
            local_num_heads = num_heads // weight_divisor

            k_b_proj, v_b_proj = kv_b_proj.split(
                [local_num_heads * local_qk_nope_head_dim, local_num_heads * local_v_head_dim],
                dim=0,
            )
            k_b_proj = k_b_proj.view([local_num_heads, local_qk_nope_head_dim, -1])
            v_b_proj = v_b_proj.view([local_num_heads, local_v_head_dim, -1])

            if cp_size > 1:
                local_cp_heads = local_num_heads // cp_size
                k_b_proj = k_b_proj[cp_rank * local_cp_heads : (cp_rank + 1) * local_cp_heads]
                v_b_proj = v_b_proj[cp_rank * local_cp_heads : (cp_rank + 1) * local_cp_heads]

            return k_b_proj, v_b_proj

        is_lite = self.config.q_lora_rank is None
        num_heads = self.config.num_attention_heads
        qk_nope_head_dim = getattr(self.config, "qk_nope_head_dim", None)
        v_head_dim = getattr(self.config, "v_head_dim", getattr(self.config, "head_dim", None))
        kv_lora_rank = getattr(self.config, "kv_lora_rank", None)

        tp_rank = self.model_config.mapping.tp_rank
        tp_size = self.model_config.mapping.tp_size
        cp_rank = self.model_config.mapping.cp_rank
        cp_size = self.model_config.mapping.cp_size

        params_map = {"gate_up_proj": ["gate_proj", "up_proj"]}
        all_named_modules = dict(self.model.named_modules())

        def load_flat_hc_weights(module, names: List[str]) -> bool:
            """Load mHC / HCHead from flat ckpt keys: ``<stem>_{fn,base,scale}``.

            V4 / DeepSeek-V4 checkpoints store these as flat names (e.g.
            ``hc_attn_fn``) rather than structured (``hc_attn.fn``).
            ``shared_head.hc_head`` (MTP) is rewritten to flat ``hc_head_*``
            under the same parent.
            """
            if names[-1] in ("hc_attn", "hc_ffn", "hc_head"):
                stem = ".".join(names)
            elif names[-2:] == ["shared_head", "hc_head"]:
                stem = ".".join(names[:-2] + ["hc_head"])
            else:
                return False
            keys = {a: f"{stem}_{a}" for a in ("fn", "base", "scale")}
            if not all(k in weights for k in keys.values()):
                return False
            for attr, key in keys.items():
                getattr(module, attr).data.copy_(weights[key][:])
            return True

        for name, module in tqdm(all_named_modules.items(), desc="Loading weights"):
            if name.startswith("draft_model"):
                continue
            names = name.split(".")
            parent_module_name = ".".join(names[:-1])

            if names[-1] == "mqa":
                parent_attn_name = ".".join(names[:-1])
                attn_sink_key = f"{parent_attn_name}.attn_sink"
                if attn_sink_key in weights:
                    sink_full = weights[attn_sink_key][:]
                    if not self.model_config.mapping.enable_attention_dp:
                        sink_full = split(sink_full, tp_size, tp_rank)
                    sink_full = sink_full.to(torch.float32).cuda()
                    module.attn_sink = nn.Parameter(sink_full, requires_grad=False)
                continue

            if len(module._parameters) <= 0:
                continue
            elif any(skip_module in name for skip_module in skip_modules):
                continue
            else:
                if "model.layers" in name and int(names[2]) >= self.config.num_hidden_layers:
                    mtp_layer_idx = int(names[2]) - self.config.num_hidden_layers
                    names[2] = str(
                        mtp_layer_idx % self.config.num_nextn_predict_layers
                        + self.config.num_hidden_layers
                    )
                    name = ".".join(names)
                if names[-1] == "kv_b_proj":
                    # TODO: remove weight_dequant after enabling fp8_bmm
                    dequant_kv_b_proj = (
                        self.model_config.quant_config.is_module_excluded_from_quantization(
                            names[-1]
                        )
                    )
                    if dequant_kv_b_proj:
                        kv_b_proj, k_b_proj_trans = load_kv_b_proj_and_k_b_proj_trans_dequant(name)
                    else:
                        kv_b_proj, k_b_proj_trans = load_kv_b_proj_and_k_b_proj_trans(
                            name, is_scale=False
                        )
                    module.weight.data.copy_(kv_b_proj.reshape(module.weight.shape))

                    attn_module = all_named_modules[parent_module_name]
                    _, v_b_proj = split_kv_b_proj(module.weight.data, is_scale=False)
                    attn_module.v_b_proj = nn.Parameter(v_b_proj, requires_grad=False)

                    attn_module.k_b_proj_trans.data.copy_(
                        k_b_proj_trans.reshape(attn_module.k_b_proj_trans.shape)
                    )

                    if getattr(module, "weight_scale", None) is not None and not dequant_kv_b_proj:
                        kv_b_proj_scale, k_b_proj_trans_scale = load_kv_b_proj_and_k_b_proj_trans(
                            name, is_scale=True
                        )
                        module.weight_scale.copy_(
                            kv_b_proj_scale.reshape(module.weight_scale.shape)
                        )
                        attn_module.k_b_proj_trans_scale.copy_(
                            k_b_proj_trans_scale.reshape(attn_module.k_b_proj_trans_scale.shape)
                        )

                        _, v_b_proj_scale = split_kv_b_proj(module.weight_scale.data, is_scale=True)
                        attn_module.v_b_proj_scale = nn.Parameter(
                            v_b_proj_scale, requires_grad=False
                        )

                        if attn_module.k_b_proj_trans_dequant is not None:
                            attn_module.k_b_proj_trans_dequant.data.copy_(
                                weight_dequant(
                                    k_b_proj_trans.view(-1, k_b_proj_trans.shape[-1]).cuda(),
                                    k_b_proj_trans_scale.view(
                                        -1, k_b_proj_trans_scale.shape[-1]
                                    ).cuda(),
                                )
                                .view(*attn_module.k_b_proj_trans_dequant.shape)
                                .to(attn_module.k_b_proj_trans_dequant.dtype)
                            )
                        if attn_module.v_b_proj_dequant is not None:
                            attn_module.v_b_proj_dequant.data.copy_(
                                weight_dequant(
                                    v_b_proj.view(-1, v_b_proj.shape[-1]).cuda(),
                                    v_b_proj_scale.view(-1, v_b_proj_scale.shape[-1]).cuda(),
                                )
                                .view(*attn_module.v_b_proj_dequant.shape)
                                .to(attn_module.v_b_proj_dequant.dtype)
                            )
                elif names[-1] == "kv_a_proj_with_mqa":
                    nvfp4_fused_a = (
                        self.model_config.get_quant_config().layer_quant_mode.has_nvfp4()
                        and weights[f"{'.'.join(names[:-1])}.kv_a_proj_with_mqa.weight"].dtype
                        == fp4_utils.float4_e2m1x2
                        and weights[f"{'.'.join(names[:-1])}.q_a_proj.weight"].dtype
                        == fp4_utils.float4_e2m1x2
                    )
                    if nvfp4_fused_a:
                        ########### input_scale
                        kv_a_proj_with_mqa_input_scale = weights[
                            f"{'.'.join(names[:-1])}.kv_a_proj_with_mqa.input_scale"
                        ]
                        if not is_lite:
                            q_a_proj_input_scale = weights[
                                f"{'.'.join(names[:-1])}.q_a_proj.input_scale"
                            ]
                            assert kv_a_proj_with_mqa_input_scale == q_a_proj_input_scale, (
                                "kv_a_proj_with_mqa.input_scale and q_a_proj.input_scale should be the same"
                            )
                        # modelopt ckpt stores amax/(448*6), convert to (448*6)/amax
                        shared_input_scale = kv_a_proj_with_mqa_input_scale
                        module.input_scale.data.copy_(1.0 / shared_input_scale)
                        E2M1_MAX = 6.0
                        module.inv_input_scale.data.copy_(module.input_scale / E2M1_MAX)
                        ########### weight_scale_2
                        need_requant_kv_a_proj_with_mqa = False
                        need_requant_q_a_proj = False
                        kv_a_proj_with_mqa_scale_2 = weights[
                            f"{'.'.join(names[:-1])}.kv_a_proj_with_mqa.weight_scale_2"
                        ]
                        shared_weight_scale_2 = kv_a_proj_with_mqa_scale_2
                        if not is_lite:
                            q_a_proj_scale_2 = weights[
                                f"{'.'.join(names[:-1])}.q_a_proj.weight_scale_2"
                            ]
                            if kv_a_proj_with_mqa_scale_2 < q_a_proj_scale_2:
                                shared_weight_scale_2 = q_a_proj_scale_2
                                need_requant_kv_a_proj_with_mqa = True
                            elif q_a_proj_scale_2 < kv_a_proj_with_mqa_scale_2:
                                need_requant_q_a_proj = True

                        ########### alpha
                        alpha = shared_input_scale.float() * shared_weight_scale_2.float()
                        module.alpha.data.copy_(alpha)
                        module.scalar_alpha = alpha.item()

                        ########### weights
                        kv_a_proj_with_mqa = weights[
                            f"{'.'.join(names[:-1])}.kv_a_proj_with_mqa.weight"
                        ][:]

                        if not is_lite:
                            q_a_proj = weights[f"{'.'.join(names[:-1])}.q_a_proj.weight"][:]

                        ########### weight_scale
                        kv_a_proj_with_mqa_scale = weights[
                            f"{'.'.join(names[:-1])}.kv_a_proj_with_mqa.weight_scale"
                        ][:]
                        kv_a_proj_with_mqa_scale = torch.ops.trtllm.block_scale_interleave(
                            kv_a_proj_with_mqa_scale.view(fp4_utils.float4_sf_dtype)
                        )
                        if not is_lite:
                            q_a_proj_scale = weights[
                                f"{'.'.join(names[:-1])}.q_a_proj.weight_scale"
                            ][:]
                            q_a_proj_scale = torch.ops.trtllm.block_scale_interleave(
                                q_a_proj_scale.view(fp4_utils.float4_sf_dtype)
                            )

                        ########### requantize
                        if need_requant_kv_a_proj_with_mqa:
                            # requant kv_a_proj_with_mqa
                            kv_a_proj_with_mqa, kv_a_proj_with_mqa_scale = (
                                requantize_weight_with_new_scale(
                                    kv_a_proj_with_mqa,
                                    kv_a_proj_with_mqa_scale,
                                    kv_a_proj_with_mqa_scale_2,
                                    shared_weight_scale_2,
                                    device=module.weight.device,
                                )
                            )
                        if need_requant_q_a_proj:
                            # requant q_a_proj
                            q_a_proj, q_a_proj_scale = requantize_weight_with_new_scale(
                                q_a_proj,
                                q_a_proj_scale,
                                q_a_proj_scale_2,
                                shared_weight_scale_2,
                                device=module.weight.device,
                            )

                        ########### fuse and load weights
                        if not is_lite:
                            fused_a = torch.cat([q_a_proj, kv_a_proj_with_mqa], dim=0)
                        else:
                            fused_a = kv_a_proj_with_mqa

                        # For DeepseekV32: kv_a_proj_with_mqa is oversized
                        # to include indexer k weights, which is filled in post_load_weights.
                        module.weight.data[0 : fused_a.shape[0]].copy_(fused_a)

                        ########### fuse weight_scale
                        if not is_lite:
                            fused_a_scale = torch.cat(
                                [q_a_proj_scale, kv_a_proj_with_mqa_scale], dim=0
                            )
                        else:
                            fused_a_scale = kv_a_proj_with_mqa_scale
                        # For DeepseekV32: kv_a_proj_with_mqa is oversized
                        # to include indexer k weights, which is filled in post_load_weights.
                        module.weight_scale.data[0 : fused_a_scale.shape[0]].copy_(fused_a_scale)
                    else:
                        fused_a = weights[f"{'.'.join(names[:-1])}.kv_a_proj_with_mqa.weight"][:]
                        if not is_lite:
                            q_a_proj = weights[f"{'.'.join(names[:-1])}.q_a_proj.weight"][:]
                            fused_a = torch.cat([q_a_proj, fused_a], dim=0)

                        if f"{'.'.join(names[:-1])}.kv_a_proj_with_mqa.weight_scale_inv" in weights:
                            fused_a_scale = weights[
                                f"{'.'.join(names[:-1])}.kv_a_proj_with_mqa.weight_scale_inv"
                            ]
                            if not is_lite:
                                q_a_proj_scale = weights[
                                    f"{'.'.join(names[:-1])}.q_a_proj.weight_scale_inv"
                                ][:]
                                fused_a_scale = torch.cat([q_a_proj_scale, fused_a_scale], dim=0)

                            module.weight_scale.data[0 : fused_a_scale.shape[0]].copy_(
                                fused_a_scale
                            )
                        # For DeepseekV32: kv_a_proj_with_mqa is oversized
                        # to include indexer k weights, which is filled in post_load_weights.
                        module.weight.data[0 : fused_a.shape[0]].copy_(fused_a)
                elif names[-1] in params_map:
                    module_weights = []
                    for new_name in params_map[names[-1]]:
                        module_weights.append(
                            filter_weights(".".join(names[:-1] + [new_name]), weights)
                        )
                    module.load_weights(weights=module_weights)
                elif names[-1] == "experts":
                    module_weights = filter_weights(name, weights)
                    module_weights = rename_moe_weight(
                        module_weights,
                        {
                            "down_proj": "w2",
                            "up_proj": "w3",
                            "gate_proj": "w1",
                        },
                    )
                    module.load_weights(weights=[module_weights])
                elif names[-1] == "backend" and isinstance(module, MoE):
                    # Special case: ConfigurableMoE.backend (TRTLLMGenFusedMoE)
                    # Currently saved MoE weights don't include 'backend' in their names.
                    # After MoE refactoring, ConfigurableMoE now has a backend submodule,
                    # and weights loading is done in the backend, so module name includes '.backend'.
                    # We need to use parent module name (without .backend) to match saved weight names.
                    # After MoE refactoring is fully complete, all paths will follow this branch.
                    parent_name = ".".join(names[:-1])
                    module_weights = filter_weights(parent_name, weights)
                    module_weights = rename_moe_weight(
                        module_weights,
                        {
                            "down_proj": "w2",
                            "up_proj": "w3",
                            "gate_proj": "w1",
                        },
                    )
                    module.load_weights(weights=[module_weights])
                elif names[-1] == "self_attn":
                    if f"{name}.o_a_proj" in weights:
                        load_o_a_proj(name, module)
                    attn_sink_key = f"{name}.attn_sink"
                    if attn_sink_key in weights:
                        sink_full = weights[attn_sink_key][:]
                        if not self.model_config.mapping.enable_attention_dp:
                            sink_full = split(sink_full, tp_size, tp_rank)
                        sink_full = sink_full.to(torch.float32).cuda()
                        module.mqa.attn_sink = nn.Parameter(sink_full, requires_grad=False)
                    continue
                elif names[-1] == "mqa":
                    # DeepseekV4TrtllmAttention owns the optional attn_sink
                    # (per-head fp32, already TP-sharded). The checkpoint key
                    # uses the parent attention module name, not the .mqa
                    # suffix. When the key is absent we leave module.attn_sink
                    # as None so DeepseekV4TrtllmAttention.forward does not pass
                    # attention_sinks to the kernel.
                    parent_attn_name = ".".join(names[:-1])
                    attn_sink_key = f"{parent_attn_name}.attn_sink"
                    if attn_sink_key in weights:
                        sink_full = weights[attn_sink_key][:]
                        if not self.model_config.mapping.enable_attention_dp:
                            sink_full = split(sink_full, tp_size, tp_rank)
                        sink_full = sink_full.to(torch.float32).cuda()
                        module.attn_sink = nn.Parameter(sink_full, requires_grad=False)
                    continue
                elif names[-1] == "next_layer_layernorm":
                    continue
                elif isinstance(module, (mHC, HCHead)) and load_flat_hc_weights(module, names):
                    continue
                elif names[-1] in ("engram",):
                    # Engram is a container module with no direct parameters;
                    # its leaf sub-modules (multi_head_embedding, kv_proj,
                    # short_conv) and direct parameters (key_norm_weight,
                    # query_norm_weight) are loaded via the generic path.
                    continue
                else:
                    module_weights = filter_weights(name, weights)
                    if hasattr(module, "load_weights"):
                        module.load_weights(weights=[module_weights])
                    else:
                        for n, p in module.named_parameters():
                            p.data.copy_(module_weights[n][:])


@torch.compile(options={"max-autotune": True})
def _get_last_token_states(hidden_states, attn_metadata):
    last_tokens = (
        torch.cumsum(
            attn_metadata.seq_lens_cuda,
            dim=0,
            dtype=torch.long,
        )
        - 1
    )
    return hidden_states[last_tokens]


class DeepseekV4MTPHead(nn.Module):
    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__()
        config = model_config.pretrained_config
        self.model_config = model_config

        self.norm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )
        self.hc_head = HCHead(config.hc_mult, config.hidden_size)
        self.hc_mult = config.hc_mult
        self.hidden_dim = config.hidden_size

        self.mapping_lm_head_tp = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        lm_head: Linear,
        attn_metadata: DeepseekV4TrtllmAttentionMetadata,
        return_context_logits: bool = False,
    ) -> torch.Tensor:
        if not return_context_logits:
            if attn_metadata is not None:
                hidden_states = _get_last_token_states(hidden_states, attn_metadata)
            else:
                hidden_states = hidden_states[-1].unsqueeze(0)

        hidden_states = hidden_states.reshape(-1, self.hc_mult, self.hidden_dim)
        hidden_states = self.hc_head(hidden_states)
        hidden_states = self.norm(hidden_states)

        enable_attention_dp = self.model_config.mapping.enable_attention_dp
        enable_lm_head_tp_in_adp = (
            enable_attention_dp and self.model_config.mapping.enable_lm_head_tp_in_adp
        )

        # Add pre-lm gather logic
        if enable_lm_head_tp_in_adp:
            # ADP + LM TP mode: perform All-Gather before LM_head
            self.mapping_lm_head_tp = create_lm_head_tp_mapping(
                self.model_config.mapping, hidden_states.shape[0]
            )
            hidden_states = allgather(hidden_states, self.mapping_lm_head_tp, dim=0)

        # Temporarily disable gather_output when not in ADP mode or (in ADP mode and LM TP is enabled)
        if not enable_attention_dp or enable_lm_head_tp_in_adp:
            lm_head.gather_output = False
        logits = lm_head(
            hidden_states, mapping_lm_head_tp=self.mapping_lm_head_tp, is_spec_decoding_head=True
        )
        if not enable_attention_dp or enable_lm_head_tp_in_adp:
            lm_head.gather_output = True
        return logits


class DeepseekV4LogitsProcessor(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        hc_head: HCHead,
        norm: RMSNorm,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.model_config = model_config
        self.hc_mult = config.hc_mult
        self.hidden_dim = config.hidden_size
        # Keep HCHead and final norm owned by DeepseekV4Model. This processor only
        # borrows them, so checkpoint loading and PP weight removal still happen
        # through the model's normal module tree.
        object.__setattr__(self, "_hc_head", hc_head)
        object.__setattr__(self, "_norm", norm)

    def forward(
        self,
        hidden_states: torch.Tensor,
        lm_head: Linear,
        attn_metadata: DeepseekV4TrtllmAttentionMetadata,
        return_context_logits: bool = False,
    ) -> torch.Tensor:
        if not self.model_config.mapping.is_last_pp_rank():
            return lm_head(hidden_states).float()

        if not return_context_logits:
            if attn_metadata is not None:
                hidden_states = _get_last_token_states(hidden_states, attn_metadata)
            else:
                hidden_states = hidden_states[-1]

        hidden_states = hidden_states.reshape(-1, self.hc_mult, self.hidden_dim)
        hidden_states = self._hc_head(hidden_states)
        hidden_states = self._norm(hidden_states)
        return lm_head(hidden_states).float()


class DeepseekV4Linear(Linear):
    """
    A wrapper around Linear because we may optionally use min-latency kernels depending on input shapes.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = None,
        mapping: Optional[Mapping] = None,
        tensor_parallel_mode: Optional[TensorParallelMode] = None,
        gather_output: bool = False,  # COLUMN parallel only
        quant_config: Optional[QuantConfig] = None,
        weights_loading_config: Optional[WeightsLoadingConfig] = None,
        reduce_output: bool = True,  # ROW parallel only
        skip_create_weights_in_init: bool = False,
        use_custom_cublas_mm: bool = False,
        lora: Optional[LoraLayer] = None,
    ):
        super().__init__(
            in_features,
            out_features,
            bias,
            dtype,
            mapping,
            tensor_parallel_mode,
            gather_output,
            quant_config,
            weights_loading_config,
            reduce_output,
            skip_create_weights_in_init,
            use_custom_cublas_mm,
            lora,
        )

    def apply_linear(
        self,
        input,
        bias,
        lora_params: Optional[dict] | None = None,
        layer_idx: Optional[int] | None = None,
    ):
        num_tokens = input.shape[0]
        if not self.has_any_quant and 1 <= num_tokens <= 16 and get_sm_version() not in [120, 121]:
            output = torch.ops.trtllm.dsv3_fused_a_gemm_op(input, self.weight.t(), bias, None)
        else:
            output = super().apply_linear(input, bias, lora_params, layer_idx)
        return output


class DeepseekV4Attention(MLA):
    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: Optional[int] = None,
        aux_stream: Optional[torch.cuda.Stream] = None,
        mapping_with_cp: Optional[Mapping] = None,
        reduce_output: bool = True,
    ):
        config = model_config.pretrained_config
        assert config.qk_rope_head_dim == 64, (
            "DeepseekV4Attention only supports qk_rope_head_dim=64"
        )
        assert config.kv_lora_rank == 448, "DeepseekV4Attention only supports kv_lora_rank=448"
        predicted_tokens_per_seq = (
            model_config.spec_config.tokens_per_gen_step
            if model_config.spec_config is not None
            else 1
        )
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            predicted_tokens_per_seq=predicted_tokens_per_seq,
            max_position_embeddings=config.max_position_embeddings,
            bias=False,
            pos_embd_params=_deepseek_v4_pos_embd_params(config, model_config, layer_idx),
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
            aux_stream=aux_stream,
            num_groups=config.o_groups,
            o_lora_rank=config.o_lora_rank,
            mapping_with_cp=mapping_with_cp,
            reduce_output=reduce_output,
        )

        self.indexer = getattr(self.mqa, "indexer", None)
        self.compressor = getattr(self.mqa, "compressor", None)


class DeepseekV4Gate(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        n_group: int,
        topk_group: int,
        routed_scaling_factor: float,
        is_hashed: bool,
        vocab_size: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        fuse_routing_kernel: bool = True,
        apply_routing: bool = False,
        moe_backend: str = "CUTLASS",
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((num_experts, hidden_size), dtype=dtype), requires_grad=False
        )
        self.moe_backend = moe_backend
        bias_dtype = torch.float32

        self.is_hashed = is_hashed
        self.top_k = top_k
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor

        if self.is_hashed:
            # self.tid2eid = nn.Parameter(
            #     torch.empty(vocab_size, top_k, dtype=torch.int32), requires_grad=False
            # )
            # WAR to avoid illegal expert indexes in hashed gating
            self.tid2eid = nn.Parameter(
                torch.stack([torch.randperm(num_experts)[:top_k] for _ in range(vocab_size)])
                .to(torch.int32)
                .contiguous(),
                requires_grad=False,
            )
            self.e_score_correction_bias = None
        else:
            self.e_score_correction_bias = nn.Parameter(
                torch.empty(num_experts, dtype=bias_dtype), requires_grad=False
            )

        assert not apply_routing, "DeepseekV4Gate routing is called inside MoE"

        def fetch_e_score_correction_bias():
            if not self.is_hashed:
                return self.e_score_correction_bias.to(torch.float32)
            else:
                return None

        self._routing_method = DeepSeekV4MoeRoutingMethod(
            top_k=self.top_k,
            n_group=self.n_group,
            topk_group=self.topk_group,
            routed_scaling_factor=self.routed_scaling_factor,
            # Pass a callable to fetch the tensor from DeepseekV4Gate at runtime, ensuring it is on the correct device
            callable_e_score_correction_bias=fetch_e_score_correction_bias,
            callable_tid2eid=lambda: self.tid2eid,
            is_hashed=self.is_hashed,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(hidden_states.float(), self.weight.float())

    def load_weights(self, weights: List[Dict]):
        assert len(weights) == 1

        self.weight.copy_(weights[0]["weight"][:])

        if self.is_hashed:
            self.tid2eid.copy_(weights[0]["tid2eid"][:].to(self.tid2eid.dtype))
        else:
            self.e_score_correction_bias.copy_(
                weights[0]["e_score_correction_bias"][:].to(self.e_score_correction_bias.dtype)
            )

    @property
    def routing_method(self) -> DeepSeekV4MoeRoutingMethod:
        return self._routing_method

    def apply(
        self, logits: torch.Tensor, input_ids: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # topk routing
        return self.routing_method.apply(logits, input_ids)

    def get_experts_per_token(self):
        return self.routing_method.top_k


class DeepseekV4MoE(nn.Module):
    def __init__(
        self,
        *,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        shared_expert_intermediate_size: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
        dtype: Optional[torch.dtype] = None,
        model_config: ModelConfig = ModelConfig(),
        override_quant_config: Optional[QuantConfig] = None,
        layer_idx: Optional[int] = None,
    ):
        from ..distributed import AllReduce

        super().__init__()
        config = model_config.pretrained_config
        self.top_k = top_k
        self.use_dp = model_config.mapping.enable_attention_dp
        self.gate = DeepseekV4Gate(
            hidden_size,
            num_experts,
            top_k=top_k,
            n_group=config.n_group,
            topk_group=config.topk_group,
            routed_scaling_factor=config.routed_scaling_factor,
            is_hashed=layer_idx < config.n_hash_layers,
            vocab_size=config.vocab_size,
            dtype=dtype,
            fuse_routing_kernel=True,
            apply_routing=False,
            moe_backend=model_config.moe_backend,
        )
        experts_quant_config = self._get_experts_quant_config(model_config, layer_idx)
        if override_quant_config is not None and experts_quant_config is model_config.quant_config:
            experts_quant_config = override_quant_config

        swiglu_limit = getattr(config, "swiglu_limit", None)
        moe_swiglu_limit = None
        if swiglu_limit is not None:
            # `create_moe` only accepts swiglu_limit for these MoE classes;
            # resolve via get_moe_cls so backend-string fallbacks (e.g.
            # TRTLLM/CUTEDSL/DENSEGEMM dropping back to CutlassFusedMoE on
            # unsupported quant) are handled correctly.
            moe_cls = get_moe_cls(model_config, override_quant_config=experts_quant_config)
            supports_swiglu_limit = moe_cls in (
                CutlassFusedMoE,
                TritonFusedMoE,
                TRTLLMGenFusedMoE,
            )
            if supports_swiglu_limit:
                moe_load_balancer_config = getattr(model_config, "moe_load_balancer", None)
                num_slots = (
                    moe_load_balancer_config.num_slots
                    if moe_load_balancer_config and moe_load_balancer_config.num_slots
                    else num_experts
                )
                local_num_slots = num_slots // model_config.mapping.moe_ep_size
                device = "cuda" if torch.cuda.is_available() else "cpu"
                moe_swiglu_limit = torch.full(
                    (local_num_slots,), float(swiglu_limit), dtype=torch.float32, device=device
                )

        self.experts = create_moe(
            num_experts=num_experts,
            routing_method=self.gate.routing_method,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=False,  # In both low‑latency and attention‑DP modes, FusedMoE skips the in‑op all‑reduce.
            model_config=model_config,
            override_quant_config=experts_quant_config,
            aux_stream_dict=aux_stream_dict,
            layer_idx=layer_idx,
            # DS-R1 W4A8 is only supported through custom quantization script from
            # examples/quantization/quantize_mixed_precision_moe.py
            weight_loading_mode=(
                MoEWeightLoadingMode.W4A8_CUSTOM
                if experts_quant_config.layer_quant_mode.is_int4_weight_only_per_group()
                else MoEWeightLoadingMode.VANILLA
            ),
            swiglu_limit=moe_swiglu_limit,
        )

        self.mapping = model_config.mapping

        # FIXME: incompatible with mixed quantization mode (including excluding modules from quantization)
        block_size = 1
        if model_config.quant_config and model_config.quant_config.group_size is not None:
            block_size = model_config.quant_config.group_size

        shared_tp_size, self.shared_output_scale = self._compute_shared_expert_tp_size(
            shared_expert_intermediate_size, block_size
        )

        self.shared_experts = GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=shared_expert_intermediate_size,
            bias=False,
            dtype=dtype,
            config=model_config,
            overridden_tp_size=shared_tp_size,
            reduce_output=False,
            swiglu_limit=swiglu_limit,
        )

        self.allreduce = None
        if not self.use_dp and self.mapping.tp_size > 1:
            self.allreduce = AllReduce(
                mapping=model_config.mapping, strategy=model_config.allreduce_strategy
            )
        self.aux_stream = aux_stream_dict[AuxStreamType.MoeShared]
        self.event_dict = {key: torch.cuda.Event() for key in [EventType.Main, EventType.MoeShared]}

        # Store config values for perfect routing.
        self.model_config = model_config
        self.dtype = dtype

        # Perfect router caching - precompute common logits if enabled.
        if os.environ.get("ENABLE_PERFECT_ROUTER", "0") == "1":
            precompute_common_perfect_router_logits(
                num_experts=num_experts,
                experts_per_token=top_k,
                moe_ep_size=model_config.mapping.moe_ep_size,
                dtype=dtype,
            )

    def _compute_shared_expert_tp_size(
        self, intermediate_size: int, block_size: int
    ) -> tuple[int, float | None]:
        """
        In the case of Deepseek-R1, the TP size of MLP is capped by intermediate_size // block_size.
        For example, when the intermediate_size is 2048 and block scaling size is 128,
        TP sizes are limited to {1, 2, 4, 8, 16} because of 2048/128 = 16.

        Args:
            intermediate_size (int): MLP intermediate size.
            block_size (int): The quantization block scale size. In the case of Deepseek FP8 recipe,
                it's 128. For NVFP4, it's 16.

        Returns:
            tuple[int, float | None]: A tuple containing (shared_tp_size, shared_output_scale).
                - shared_tp_size: The computed TP size.
                - shared_output_scale: The output scale factor, or None if not needed.
        """

        assert intermediate_size % block_size == 0, (
            "intermediate_size must be divisible by block_size."
        )

        shared_output_scale = None
        # The block scale size is 128, which requires shared_expert_intermediate_size to be divisible by 128.
        if self.use_dp:
            # If using attention DP, the shared experts also use DP instead of TP.
            shared_tp_size = 1
        else:
            # Due to the restriction of block scale size (i.e., 128), the supported
            # TP sizes only include 1, 2, 4, 8, and 16. The math.gcd operation ensures that
            # shared_tp_size falls in the supported TP sizes.
            shared_tp_size = math.gcd(
                intermediate_size // block_size,
                self.mapping.tp_size,
            )
            # If shared_tp_size has been overridden, the output of shared experts needs
            # to be scaled down accordingly before all-reduce.
            if shared_tp_size != self.mapping.tp_size:
                shared_output_scale = shared_tp_size / self.mapping.tp_size

        return shared_tp_size, shared_output_scale

    @staticmethod
    def _get_experts_quant_config(model_config, layer_idx: int) -> QuantConfig:
        if getattr(model_config, "quant_config_dict", None) is None:
            return model_config.quant_config
        return model_config.quant_config_dict.get(
            f"model.layers.{layer_idx}.mlp.experts", model_config.quant_config
        )

    def _create_ideal_expert_load_balanced_logits(
        self, num_tokens: int, num_experts: int, device: torch.device
    ) -> torch.Tensor:
        """
        Create ideal logits that produce GPU-aware load balanced expert assignment.
        This method uses the global cache to access precomputed logits to optimize performance.
        """
        # Use global cached logits.
        return get_cached_perfect_router_logits(
            num_tokens=num_tokens,
            num_experts=num_experts,
            experts_per_token=self.top_k,
            moe_ep_size=self.model_config.mapping.moe_ep_size,
            device=device,
            dtype=self.dtype,
        )

    def compute_routed_output(
        self, hidden_states, hidden_states_fp4, input_ids, all_rank_num_tokens, do_finalize
    ):
        # max-throughput
        use_dp_padding = False
        # Add DP padding on SM120 for context comm performance
        # TODO: Move this model-agonostic part to MoE
        if self.use_dp and self.mapping.tp_size > 1 and get_sm_version() == 120:
            use_dp_padding = True
            hidden_states = torch.nn.functional.pad(
                hidden_states, (0, 0, 0, max(all_rank_num_tokens) - hidden_states.shape[0])
            )
            input_ids = torch.nn.functional.pad(
                input_ids,
                (0, max(all_rank_num_tokens) - input_ids.shape[0]),
                value=self.model_config.pretrained_config.pad_token_id,
            )

        router_logits = self.gate(hidden_states)

        # Use ideal load balanced logits if enabled, otherwise use gate output.
        if os.environ.get("ENABLE_PERFECT_ROUTER", "0") == "1":
            # WARNING: This discards the learned gate output and uses ideal logits for perfect load balancing.
            # Only use this for testing load balancing strategies, not for actual inference.
            # The gate is still computed to maintain realistic performance measurement.
            num_tokens, num_experts = router_logits.shape
            router_logits = self._create_ideal_expert_load_balanced_logits(
                num_tokens=num_tokens, num_experts=num_experts, device=hidden_states.device
            )

        routed_output = self.experts(
            hidden_states_fp4 if hidden_states_fp4 is not None else hidden_states,
            router_logits,
            input_ids=input_ids,
            do_finalize=do_finalize,
            output_dtype=hidden_states.dtype,
            all_rank_num_tokens=all_rank_num_tokens,
            use_dp_padding=use_dp_padding,
            **({"alltoall_result_do_sum": False} if isinstance(self.experts, WideEPMoE) else {}),
        )

        return routed_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_fp4: Optional[Fp4QuantizedTensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        all_rank_num_tokens: Optional[list[int]] = None,
        final_all_reduce_params: Optional[AllReduceParams] = None,
        do_finalize: Optional[bool] = True,
    ) -> torch.Tensor:
        if not do_finalize:
            assert not self.use_dp

        def _compute_shared_output():
            shared_output = self.shared_experts(
                hidden_states_fp4 if hidden_states_fp4 is not None else hidden_states
            )
            if self.shared_output_scale is not None:
                shared_output *= self.shared_output_scale
            return shared_output

        def _compute_routed_output():
            routed_output = self.compute_routed_output(
                hidden_states, hidden_states_fp4, input_ids, all_rank_num_tokens, do_finalize
            )
            return routed_output

        # NOTE: define compiled helpers at module scope to avoid defining decorators inside compiled frames

        routed_output, shared_output = maybe_execute_in_parallel(
            _compute_routed_output,
            _compute_shared_output,
            self.event_dict[EventType.Main],
            self.event_dict[EventType.MoeShared],
            self.aux_stream,
        )

        if not do_finalize:
            return [shared_output, *routed_output]
        else:
            if routed_output.dim() == 3:
                assert shared_output.numel() * self.top_k == routed_output.numel(), (
                    "unmatched tensor shape"
                )
                final_hidden_states = moe_reduce_add_shared_output(routed_output, shared_output)
            else:
                assert shared_output.size() == routed_output.size(), "unmatched tensor shape"
                final_hidden_states = shared_output + routed_output

            if not self.use_dp and self.mapping.tp_size > 1:
                final_hidden_states = self.allreduce(
                    final_hidden_states, all_reduce_params=final_all_reduce_params
                )

            return final_hidden_states


class DeepseekV4DecoderLayer(DecoderLayer):
    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
        is_separate_draft_engine: bool = False,
        mapping_with_cp: Optional[Mapping] = None,
        disable_post_moe_fusion: bool = False,
    ):
        super().__init__()
        self.model_config = model_config
        self.config = model_config.pretrained_config
        config = self.config

        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.num_experts = config.n_routed_experts
        self.num_shared_experts = config.n_shared_experts
        self.top_k = config.num_experts_per_tok

        self.mapping = model_config.mapping
        mapping = self.mapping
        self.enable_attention_dp = mapping.enable_attention_dp
        self.mlp_tp_size = mapping.tp_size
        self.is_p2p_supported = can_access_peer(mapping)

        self.hc_attn = mHC(
            config.hc_mult,
            config.hidden_size,
            config.hc_sinkhorn_iters,
            dtype=torch.float32,
            post_mult_value=2.0,
        )

        layer_idx_for_attention = layer_idx
        if is_separate_draft_engine:
            # KVCacheManager only support 1 layer for separate draft engine
            layer_idx_for_attention = layer_idx - model_config.pretrained_config.num_hidden_layers

        self.self_attn = DeepseekV4Attention(
            model_config,
            layer_idx=layer_idx_for_attention,
            aux_stream=aux_stream_dict[AuxStreamType.Attention],
            reduce_output=not self.enable_attention_dp and self.mapping.tp_size > 1,
        )

        self.fusion_config = EagerFusionConfig()
        self.enable_fusion = os.environ.get("TRTLLM_DEEPSEEK_EAGER_FUSION_DISABLED", "0") == "0"
        self.enable_fusion &= not self.enable_attention_dp

        self.hc_ffn = mHC(
            config.hc_mult,
            config.hidden_size,
            config.hc_sinkhorn_iters,
            dtype=torch.float32,
            post_mult_value=2.0,
        )

        # FIXME: incompatible with mixed quantization mode
        quant_config = self._get_decoder_layer_quant_config(model_config, layer_idx)
        self.is_nvfp4 = quant_config.layer_quant_mode.has_nvfp4()
        assert quant_config.quant_algo is not QuantAlgo.MIXED_PRECISION, (
            "MIXED_PRECISION is ambiguous"
        )

        self.allreduce = None
        self.moe_allreduce = None
        if not self.enable_attention_dp and self.mapping.tp_size > 1:
            self.allreduce = AllReduce(
                mapping=model_config.mapping,
                strategy=model_config.allreduce_strategy,
                dtype=config.torch_dtype,
            )
            self.moe_allreduce = MoEAllReduce(self.mapping)

        has_tp = mapping.has_tp()
        self.fusion_config.PRE_MOE_FUSION = self.enable_fusion and has_tp
        # DeepSeek-V4 applies the next RMSNorm after mHC post_mapping and the
        # next layer's mHC pre_mapping. Fusing the post-MoE all-reduce with the
        # next RMSNorm would normalize the raw MoE output before mHC post_mapping,
        # which is not equivalent.
        self.fusion_config.POST_MOE_FUSION = False

        self.mlp = DeepseekV4MoE(
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=self.hidden_size,
            intermediate_size=self.moe_intermediate_size,
            shared_expert_intermediate_size=self.moe_intermediate_size * self.num_shared_experts,
            dtype=config.torch_dtype,
            model_config=model_config,
            override_quant_config=quant_config,
            aux_stream_dict=aux_stream_dict,
            layer_idx=layer_idx,
        )

        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )

        # When enable_attention_dp is True, we normally skip attention all-reduce since each
        # DP rank works on different batch elements. However, with CP > 1, attention is split
        # across CP ranks for the SAME batch element, so all-reduce is still needed.
        has_cp = mapping_with_cp is not None and mapping_with_cp.cp_size > 1
        can_skip_for_attention_dp = self.enable_attention_dp and not has_cp
        self.disable_attn_allreduce = (
            self.fusion_config.PRE_MOE_FUSION
            or self.fusion_config.PRE_MLP_FUSION
            or self.mapping.tp_size == 1
            or can_skip_for_attention_dp
        )

        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )
        self.layer_idx = layer_idx
        # is_first_layer is baked in at __init__ time so the Python-side branch in
        # forward() resolves at CUDA-graph capture time.
        self.is_first_layer = layer_idx == 0
        # fused_hc knob: pretrained-config attr `enable_fused_hc` controls whether
        # the MHC boundary fusion (`mHC.fused_hc`) is used. When False, fall back
        # to the unfused `post_mapping → pre_mapping` chain (same path engram
        # layers already take). Env var TRTLLM_MHC_ENABLE_FUSED_HC overrides the
        # config attr (set to "0" to force-disable for validation/rollback).
        self.enable_fused_hc = _resolve_enable_fused_hc(config)
        self.next_layer_layernorm: RMSNorm = None
        # Finalized in DeepseekV4ForCausalLM.post_load_weights once the full layer
        # list is visible: a layer may defer its hc_ffn.post_mapping only if
        # the next layer is able to absorb it via fused_hc (i.e. the next
        # layer has fused_hc enabled and no engram at its entry). Last layer
        # never defers — hc_head consumes the residual directly.
        self.defer_post_mapping: bool = False

        # Engram module (optional, for n-gram context augmentation)
        self.engram: Optional[Engram] = None
        _engram_config = getattr(config, "engram_config", None)
        _engram_vocab_sizes_by_layer = getattr(config, "engram_vocab_sizes_by_layer", {})
        if _engram_config is not None and layer_idx in _engram_vocab_sizes_by_layer:
            self.engram = Engram(
                layer_id=layer_idx,
                config=_engram_config,
                vocab_sizes_flat=_engram_vocab_sizes_by_layer[layer_idx],
                stream=aux_stream_dict[AuxStreamType.EngramPrecompute],
            )

    def _get_decoder_layer_quant_config(
        self, model_config: ModelConfig[PretrainedConfig], layer_idx: int
    ):
        """
        The MTP layer in the nvfp4 checkpoint is unquantized. Because the TRTLLM
        moe_backend only supports fp8/fp4 quantization, we need to override
        the quant_config for the MTP layer.
        """
        quant_config = model_config.quant_config

        layer_name = f"model.layers.{layer_idx}"
        if quant_config.is_module_excluded_from_quantization(layer_name):
            return QuantConfig(
                quant_algo=None,
                kv_cache_quant_algo=quant_config.kv_cache_quant_algo,
            )
        else:
            return model_config.quant_config

    def _compute_mlp_tp_size(self, intermediate_size: int, block_size: int) -> int:
        """
        For DeepSeek‑R1, MLP TP size is limited by intermediate_size // block_size
        and must also be multiples of gpus_per_node to avoid expensive inter‑node allreduce.

        Args:
            intermediate_size (int): MLP intermediate size.
            block_size (int): The quantization block scale size. In the case of Deepseek FP8 recipe,
                it's 128. For NVFP4, it's 16.

        Returns:
            int: The computed tp_size.
        """

        assert intermediate_size % block_size == 0, (
            "intermediate_size must be divisible by block_size."
        )
        if self.enable_attention_dp:
            # If using attention DP, the MLP also uses DP instead of TP.
            mlp_tp_size = 1
        else:
            # The two math.gcd operations ensure that mlp_tp_size falls in the candidate TP sizes.
            tp = math.gcd(
                intermediate_size // block_size,
                self.mapping.tp_size,
            )

            if tp > self.mapping.gpus_per_node:
                mlp_tp_size = math.gcd(
                    tp,
                    self.mapping.gpus_per_node,
                )  # Avoid costly inter-node TP
            else:
                mlp_tp_size = tp
        return mlp_tp_size

    def forward(
        self,
        position_ids: torch.IntTensor,
        hc_state,
        attn_metadata: DeepseekV4TrtllmAttentionMetadata,
        spec_metadata: Optional[SpecMetadata] = None,
        input_ids: Optional[torch.IntTensor] = None,
        engram_embeddings=None,
        **kwargs,
    ):
        """mHC-aware decoder layer with boundary fusion.

        ``hc_state`` carries the mHC pipeline state across layers:

        - ``is_first_layer=True``: ``hc_state`` is the initial residual tensor
          ``[B, HC_MULT, hidden]`` (bf16). This layer bootstraps the stream
          with ``hc_attn.pre_mapping`` (aka the standalone ``hc_pre``).
        - ``is_first_layer=False``: ``hc_state`` is an ``HCState``. If
          ``is_deferred`` (fused path), the 4 tensors feed the current
          ``hc_attn.fused_hc``. Otherwise ``residual`` is already post-mapped
          by the prior layer and this layer just runs ``pre_mapping``.

        Returns an ``HCState``. Fused mode returns a deferred state carrying
        this layer's ``hc_ffn`` inputs, so the next layer absorbs
        ``hc_ffn.post_mapping`` via its own ``fused_hc``. Engram / unfused
        mode resolves the post_mapping in-layer and returns a resolved state.

        Engram (when enabled on this layer) injects a residual modification
        between the previous block's post_mapping and this block's pre_mapping.
        That mutation breaks the algebraic assumptions of ``fused_hc``, so
        engram-enabled layers fall back to the unfused
        ``post_mapping → +engram → pre_mapping`` chain at the entry boundary.
        The mid-layer ``attn → MoE`` boundary is always safe to fuse.
        """
        has_engram = self.engram is not None and engram_embeddings is not None

        # -------------------------------------------------------------------
        # Entry boundary: hc_pre (layer 0) or fused_hc / unfused chain.
        # -------------------------------------------------------------------
        residual, post_mix, comb_mix, layer_input = self._entry_boundary(
            hc_state, engram_embeddings, has_engram
        )

        # -------------------------------------------------------------------
        # Attention block
        # -------------------------------------------------------------------
        x_attn = self.input_layernorm(layer_input)
        x_attn = self.self_attn(
            position_ids=position_ids,
            hidden_states=x_attn,
            attn_metadata=attn_metadata,
            all_reduce_params=AllReduceParams(enable_allreduce=not (self.disable_attn_allreduce)),
            **kwargs,
        )

        # -------------------------------------------------------------------
        # Mid-layer boundary: fuse hc_attn.post_mapping + hc_ffn.pre_mapping.
        # No engram concern here because engram only fires at layer entry.
        # When enable_fused_hc=False, fall back to the unfused chain.
        # -------------------------------------------------------------------
        if spec_metadata is not None and spec_metadata.is_layer_capture(self.layer_idx):
            self.fusion_config.POST_MOE_FUSION = False
        if self.enable_fused_hc:
            residual, post_mix, comb_mix, layer_input = self.hc_ffn.fused_hc(
                x_prev=x_attn,
                residual_prev=residual,
                post_mix_prev=post_mix,
                comb_mix_prev=comb_mix,
            )
        else:
            # Break fused_hc into post_mapping and pre_mapping as separate ops.
            residual = self.hc_attn.post_mapping(
                x=x_attn,
                residual=residual,
                post_layer_mix=post_mix,
                comb_res_mix=comb_mix,
            )
            post_mix, comb_mix, layer_input = self.hc_ffn.pre_mapping(residual)

        # -------------------------------------------------------------------
        # MoE block — returns x_ffn (post-MoE, already normed by
        # next_layer_layernorm when POST_MOE_FUSION is on).
        # -------------------------------------------------------------------
        x_ffn = self.forward_MoE(
            hidden_states=layer_input,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            input_ids=input_ids,
        )

        # Defer this layer's hc_ffn.post_mapping only when the NEXT layer can
        # absorb it via fused_hc (see post_load_weights). Otherwise resolve it
        # here and hand the next layer a fully post-mapped residual.
        if self.defer_post_mapping:
            return HCState.deferred(
                residual=residual, post_mix=post_mix, comb_mix=comb_mix, x_prev=x_ffn
            )
        resolved_residual = self.hc_ffn.post_mapping(
            x=x_ffn,
            residual=residual,
            post_layer_mix=post_mix,
            comb_res_mix=comb_mix,
        )
        return HCState.resolved(resolved_residual)

    def _entry_boundary(self, hc_state, engram_embeddings, has_engram):
        """Resolve the per-layer entry into (residual, post_mix, comb_mix, layer_input).

        Two code paths:
          1. fused: previous layer deferred its post_mapping; fold it into this
             layer's pre_mapping via hc_attn.fused_hc. The prev layer only
             defers when post_load_weights has proved we can absorb it here
             (no engram, fused_hc enabled), so a deferred state at entry is
             guaranteed fusable.
          2. unfused: previous layer already resolved its post_mapping (or
             this is layer 0); run pre_mapping on the residual directly.
        """
        # Fused entry: prev layer deferred its post_mapping; fold it into
        # hc_attn.fused_hc. By construction (post_load_weights), a deferred
        # state at entry is guaranteed fusable. Layer 0 receives the raw
        # residual tensor and has no HCState, so short-circuit on it.
        if not self.is_first_layer and hc_state.is_deferred:
            return self.hc_attn.fused_hc(
                x_prev=hc_state.x_prev,
                residual_prev=hc_state.residual,
                post_mix_prev=hc_state.post_mix,
                comb_mix_prev=hc_state.comb_mix,
            )

        # Unfused entry: layer 0 hands us the initial residual tensor
        # [B, HC_MULT, hidden] directly; post-layer-0 hands us a resolved
        # HCState. Both collapse to "apply engram delta (if any) then run
        # pre_mapping".
        residual = hc_state if self.is_first_layer else hc_state.residual
        if has_engram:
            residual = residual + self.engram(residual, engram_embeddings)
        post_mix, comb_mix, layer_input = self.hc_attn.pre_mapping(residual)
        return residual, post_mix, comb_mix, layer_input

    def forward_MoE(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: DeepseekV4TrtllmAttentionMetadata,
        spec_metadata: Optional[SpecMetadata] = None,
        input_ids: Optional[torch.IntTensor] = None,
    ) -> torch.Tensor:
        def _run_MoE(hidden_states, hidden_states_fp4, do_finalize, input_ids):
            return self.mlp(
                hidden_states,
                hidden_states_fp4,
                all_rank_num_tokens=attn_metadata.all_rank_num_tokens,
                final_all_reduce_params=AllReduceParams(
                    enable_allreduce=not (
                        self.fusion_config.POST_MOE_FUSION or self.mapping.tp_size == 1
                    )
                ),
                do_finalize=do_finalize,
                input_ids=input_ids,
            )

        if self.fusion_config.PRE_MOE_FUSION:
            # In DeepSeek-V4 the external residual connection is handled by mHC
            # (hc_ffn.post_mapping), so there is no residual to add here.
            # Use fused allreduce + RMSNorm (no residual addition).
            hidden_states = self.allreduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RMS_NORM,
                    norm_weight=self.post_attention_layernorm.weight,
                    eps=self.post_attention_layernorm.variance_epsilon,
                    trigger_completion_at_end=False,
                ),
            )
        else:
            # No fusion: just normalize.
            hidden_states = self.post_attention_layernorm(hidden_states)

        # Note: this fusion pattern is only supported for single-node TRTLLM-nvfp4 backend now
        do_finalize = self.mapping.is_multi_node() or (
            not (
                self.fusion_config.POST_MOE_FUSION
                and hidden_states.shape[0] <= self.moe_allreduce.max_token
                and self.model_config.moe_backend == "TRTLLM"
                and self.mlp.experts.has_nvfp4
                and self.is_p2p_supported
            )
        )

        hidden_states = _run_MoE(
            hidden_states, hidden_states_fp4=None, input_ids=input_ids, do_finalize=do_finalize
        )

        if self.fusion_config.POST_MOE_FUSION:
            if do_finalize:
                # Fused allreduce + RMSNorm, no residual needed (mHC handles it).
                hidden_states = self.allreduce(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RMS_NORM,
                        norm_weight=self.next_layer_layernorm.weight,
                        eps=self.next_layer_layernorm.variance_epsilon,
                        trigger_completion_at_end=False,
                    ),
                )
            else:
                assert len(hidden_states) == 4, "hidden_states must have 4 elements"

                shared_output = hidden_states[0]
                fc2_output = hidden_states[1]
                expert_scale_factor = hidden_states[2]
                expanded_idx_to_permuted_idx = hidden_states[3]

                moe_all_reduce_params = MoEAllReduceParams(
                    expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                    expert_scale_factor=expert_scale_factor,
                    shared_expert_output=shared_output,
                    residual=None,
                    norm_weight=self.next_layer_layernorm.weight,
                    eps=self.next_layer_layernorm.variance_epsilon,
                    is_cutlass_min_latency=False,
                )
                (hidden_states,) = self.moe_allreduce(
                    fc2_output, all_reduce_params=moe_all_reduce_params
                )
        else:
            if spec_metadata is not None and spec_metadata.is_layer_capture(self.layer_idx):
                spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states, None)

        return hidden_states


class DeepseekV4MTP(DeepseekV4DecoderLayer):
    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
        is_separate_draft_engine: bool = False,
    ):
        super().__init__(
            model_config,
            layer_idx,
            aux_stream_dict,
            is_separate_draft_engine,
            disable_post_moe_fusion=True,
        )
        config = model_config.pretrained_config
        self.hidden_dim = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.num_experts = config.n_routed_experts
        self.num_shared_experts = config.n_shared_experts
        self.top_k = config.num_experts_per_tok

        self.aux_stream = aux_stream_dict[AuxStreamType.MoeShared]
        self.event_dict = {key: torch.cuda.Event() for key in [EventType.Main, EventType.MoeShared]}

        self.enorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )

        self.hnorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )
        self.hc_mult = config.hc_mult
        if model_config.mapping.enable_attention_dp:
            self.e_proj = Linear(
                config.hidden_size,
                config.hidden_size,
                bias=False,
                dtype=config.torch_dtype,
                quant_config=model_config.get_quant_config(),
                skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            )
            self.h_proj = Linear(
                config.hidden_size,
                config.hidden_size,
                bias=False,
                dtype=config.torch_dtype,
                quant_config=model_config.get_quant_config(),
                skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            )
        else:
            self.e_proj = Linear(
                config.hidden_size,
                config.hidden_size,
                bias=False,
                dtype=config.torch_dtype,
                tensor_parallel_mode=TensorParallelMode.ROW,
                mapping=model_config.mapping,
                reduce_output=True,
                quant_config=model_config.get_quant_config(),
                skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            )
            self.h_proj = Linear(
                config.hidden_size,
                config.hidden_size,
                bias=False,
                dtype=config.torch_dtype,
                tensor_parallel_mode=TensorParallelMode.ROW,
                mapping=model_config.mapping,
                reduce_output=True,
                quant_config=model_config.get_quant_config(),
                skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            )

        self.shared_head = DeepseekV4MTPHead(model_config)

    def forward(
        self,
        input_ids: torch.IntTensor,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        embed_tokens: Embedding,
        attn_metadata: DeepseekV4TrtllmAttentionMetadata,
        all_rank_num_tokens: Optional[List[int]] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run an MTP layer.

        ``embed_tokens`` is injected by the one-model draft path and shared
        with the target model. ``hidden_states`` is the flattened mHC residual
        from the target or previous MTP layer: [num_tokens, hc_mult * hidden].
        """

        def norm_embeds():
            return self.enorm(embed_tokens(input_ids))  # emdedding

        def norm_hidden():
            return self.hnorm(hidden_states.reshape(-1, self.hc_mult, self.hidden_dim))

        inputs_embeds, hidden_states = maybe_execute_in_parallel(
            norm_embeds,
            norm_hidden,
            self.event_dict[EventType.Main],
            self.event_dict[EventType.MoeShared],
            self.aux_stream,
        )

        # Split hidden_states columnwise based on TP
        tp_size = self.model_config.mapping.tp_size
        tp_rank = self.model_config.mapping.tp_rank
        if tp_size > 1 and not (self.model_config.mapping.enable_attention_dp):
            inputs_embeds = torch.chunk(inputs_embeds, tp_size, dim=-1)[tp_rank].contiguous()
            hidden_states = torch.chunk(hidden_states, tp_size, dim=-1)[tp_rank].contiguous()

        inputs_embeds = self.e_proj(inputs_embeds).unsqueeze(1)
        hidden_states = self.h_proj(hidden_states)
        hidden_states = inputs_embeds + hidden_states

        original_all_rank_num_tokens = attn_metadata.all_rank_num_tokens
        if all_rank_num_tokens is not None:
            attn_metadata.all_rank_num_tokens = all_rank_num_tokens
        try:
            hc_state = super().forward(
                position_ids=position_ids,
                hc_state=HCState.resolved(hidden_states),
                attn_metadata=attn_metadata,
                spec_metadata=spec_metadata,
                input_ids=input_ids,
                **kwargs,
            )
        finally:
            attn_metadata.all_rank_num_tokens = original_all_rank_num_tokens
        hidden_states = hc_state.residual.flatten(1)

        return hidden_states


class DeepseekV4Model(DecoderModel):
    def __init__(
        self, model_config: ModelConfig[PretrainedConfig], mapping_with_cp: Optional[Mapping] = None
    ):
        super().__init__(model_config)
        config = model_config.pretrained_config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.hc_mult = config.hc_mult
        aux_stream_list = [torch.cuda.Stream() for _ in range(5)]
        self.aux_stream_dict = {
            AuxStreamType.Attention: aux_stream_list[0],
            AuxStreamType.MoeShared: aux_stream_list[0],
            AuxStreamType.MoeChunkingOverlap: aux_stream_list[1],
            AuxStreamType.MoeBalancer: aux_stream_list[2],
            AuxStreamType.MoeOutputMemset: aux_stream_list[3],
            AuxStreamType.EngramPrecompute: aux_stream_list[4],
        }

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
        )

        self.hc_head = HCHead(config.hc_mult, config.hidden_size)

        # Engram hash provider (optional, for n-gram context augmentation)
        # Must be created before layers so vocab sizes can be stored on config.
        self.engram_hash_provider: Optional[EngramHashProvider] = None
        self.use_engram = getattr(config, "has_engram", False)
        if self.use_engram:
            engram_config = EngramConfig(
                tokenizer_name_or_path=getattr(
                    config, "tokenizer_name_or_path", "deepseek-ai/DeepSeek-V3"
                ),
                engram_vocab_size=config.engram_vocab_size,
                max_ngram_size=config.engram_max_ngram_size,
                n_embed_per_ngram=config.engram_n_embed_per_ngram,
                n_head_per_ngram=config.engram_n_head_per_ngram,
                layer_ids=config.engram_layer_ids,
                pad_id=config.engram_pad_id,
                seed=config.engram_seed,
                kernel_size=config.engram_kernel_size,
                hidden_size=config.hidden_size,
                hc_mult=config.hc_mult,
                norm_eps=config.rms_norm_eps,
                dtype=config.torch_dtype,
            )
            self.engram_hash_provider = EngramHashProvider(engram_config)
            self.engram_layer_ids = engram_config.layer_ids
            # Store engram config and per-layer vocab sizes on the pretrained config so
            # DeepseekV4DecoderLayer can read them directly from model_config without extra params.
            config.engram_config = engram_config
            config.engram_vocab_sizes_by_layer = {
                layer_id: [
                    x
                    for y in self.engram_hash_provider.vocab_size_across_layers[layer_id]
                    for x in y
                ]
                for layer_id in engram_config.layer_ids
            }

        self.layers = nn.ModuleList(
            [
                DeepseekV4DecoderLayer(
                    model_config,
                    layer_idx,
                    self.aux_stream_dict,
                    mapping_with_cp=mapping_with_cp,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )

    def __pp_init__(self):
        self.epilogue.append(self.hc_head)
        super().__pp_init__()

    def forward(
        self,
        attn_metadata: DeepseekV4TrtllmAttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # -----------------------------------------------------------------
        # Engram pre-computation (overlapped with main-stream layer forward)
        #
        # Hash provider internally applies CompressedTokenizer to input_ids,
        # so no separate engram tokenizer input is needed.
        #
        # All precompute() calls are dispatched onto a dedicated engram CUDA
        # stream so they run concurrently with the main stream processing the
        # earlier transformer layers.  A per-layer Event is recorded after
        # each precompute; the main stream waits on the event just before it
        # needs the result for that specific layer.
        # -----------------------------------------------------------------
        engram_embeddings_cache: Optional[Dict] = None
        engram_events: Dict[int, torch.cuda.Event] = {}
        if self.use_engram and self.engram_hash_provider is not None and input_ids is not None:
            hash_cache = self.engram_hash_provider.compute_hashes(
                input_ids.view(-1),
                seq_lens=attn_metadata.seq_lens_cuda,
            )
            engram_embeddings_cache = {}
            for layer_id in self.engram_layer_ids:
                engram_mod = self.layers[layer_id].engram
                if engram_mod is not None:
                    # precompute() dispatches onto the engram stream internally and
                    # records sync_event; the main stream will wait on it before use.
                    engram_embeddings_cache[layer_id] = engram_mod.precompute(
                        hash_cache[layer_id], dtype=inputs_embeds.dtype
                    )
                    engram_events[layer_id] = engram_mod.sync_event

        mapping = self.model_config.mapping
        if mapping.has_pp() and not mapping.is_first_pp_rank():
            hidden_states = torch.empty(
                inputs_embeds.shape[0],
                self.hc_mult,
                inputs_embeds.shape[1],
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )
        else:
            hidden_states = inputs_embeds
            hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)

        # ``hc_state`` carries the mHC pipeline state across layers. Layer 0
        # receives the initial residual tensor and bootstraps via
        # hc_attn.pre_mapping ("hc_pre"). Every later layer receives an
        # ``HCState``. In fused mode the state is "deferred" (prior layer's
        # hc_ffn.post_mapping is folded into this layer's hc_attn.fused_hc);
        # in unfused mode the state is "resolved" (residual already
        # post-mapped). After the last layer, a deferred state is closed with
        # a standalone hc_post; a resolved state feeds hc_head directly.
        hc_state = hidden_states

        for idx, decoder_layer in enumerate(self.layers[: self.num_hidden_layers]):
            engram_embeddings = None
            if engram_embeddings_cache is not None and idx in engram_embeddings_cache:
                # Sync: ensure the engram stream has finished precompute for this layer
                # before the main stream reads the result.
                engram_events[idx].wait(torch.cuda.current_stream())
                engram_embeddings = engram_embeddings_cache[idx]

            hc_state = decoder_layer(
                position_ids=position_ids,
                hc_state=hc_state,
                attn_metadata=attn_metadata,
                spec_metadata=spec_metadata,
                input_ids=input_ids,
                engram_embeddings=engram_embeddings,
            )

        hidden_states = hc_state.residual.flatten(1)

        return hidden_states


@register_auto_model("DeepseekV4ForCausalLM")
class DeepseekV4ForCausalLM(SpecDecOneEngineForCausalLM[DeepseekV4Model, PretrainedConfig]):
    @classmethod
    def get_model_defaults(cls, llm_args: "TorchLlmArgs") -> dict:
        return {"kv_cache_config": {"tokens_per_block": 128}}

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        self.mapping_with_cp = None
        # Note: Currently the usage of mapping is all over the place making its usage brittle
        # in this file. As a temporary WAR, we hold on to an original copy of mapping when CP
        # is in action. This shall be passed on to attention which is the only layer that's
        # affected by CP. For other layers, CP ranks are repurposed to TP. This shall be undone
        # at the end of __init__.
        if model_config.mapping.has_cp_helix():
            print(
                "[DeepseekV4ForCausalLM::__init__] Repurposing KVP ranks to TP while keeping other details the same."
            )
            self.mapping_with_cp = copy.deepcopy(model_config.mapping)
            # Repurpose KVP ranks to TP while keeping other details the same.
            model_config._frozen = False
            model_config.mapping = model_config.mapping.repurpose_helix_cp_to_tp()
            model_config._frozen = True

        # Rename some keys of quant_config_dict to support legacy checkpoints
        if model_config.quant_config_dict is not None:
            model_config = copy.deepcopy(model_config)
            quant_config_dict = {}
            for key, val in model_config.quant_config_dict.items():
                key_split = key.split(".")
                if key_split[-1] == "fused_a":
                    key = ".".join(key_split[:-1] + ["kv_a_proj_with_mqa"])
                quant_config_dict[key] = val
            model_config._frozen = False
            model_config.quant_config_dict = quant_config_dict
            model_config._frozen = True

        super().__init__(
            model=DeepseekV4Model(model_config, mapping_with_cp=self.mapping_with_cp),
            model_config=model_config,
        )
        self.logits_processor = DeepseekV4LogitsProcessor(
            model_config, self.model.hc_head, self.model.norm
        )

        # Exclude Engram weights from quantization.  Engram embedding tables
        # and small linear projections are not suited for NVFP4/FP8 quant.
        if getattr(self.config, "has_engram", False):
            if model_config.quant_config.exclude_modules is None:
                model_config.quant_config.exclude_modules = []
            model_config.quant_config.exclude_modules.append("*engram*")

        self.model_nextn = 0
        if (
            model_config.spec_config is not None
            and model_config.spec_config.spec_dec_mode.is_mtp_one_model()
        ):
            self.model_nextn = model_config.spec_config.num_nextn_predict_layers
            ckpt_nextn = self.config.num_nextn_predict_layers
            self.num_hidden_layers = self.config.num_hidden_layers
            assert ckpt_nextn > 0, "There is not MTP modules in the checkpoint."
            if ckpt_nextn == 1 and not model_config.spec_config.use_mtp_vanilla:
                pass
            else:
                # modify the QuantConfig to support duplicated mtp layers
                if model_config.quant_config.exclude_modules is not None:
                    extend_exclude_modules = []
                    for model_mtp_idx in range(
                        self.num_hidden_layers, self.num_hidden_layers + self.model_nextn
                    ):
                        ckpt_mtp_idx = (
                            model_mtp_idx - self.num_hidden_layers
                        ) % ckpt_nextn + self.num_hidden_layers
                        model_prefix = f"model.layers.{model_mtp_idx}"
                        ckpt_prefix = f"model.layers.{ckpt_mtp_idx}"
                        for exclude_module in model_config.quant_config.exclude_modules:
                            if ckpt_prefix in exclude_module and model_prefix not in exclude_module:
                                extend_exclude_modules.append(
                                    exclude_module.replace(ckpt_prefix, model_prefix)
                                )
                    self.model_config.quant_config.exclude_modules.extend(extend_exclude_modules)
            self.model.layers.extend(self.draft_model.mtp_layers)

        # Undo any manipulations done to mapping.
        if self.mapping_with_cp is not None:
            print("[DeepseekV4ForCausalLM::__init__] Restoring original mapping.")
            model_config._frozen = False
            model_config.mapping = self.mapping_with_cp
            model_config._frozen = True

    def forward(
        self,
        attn_metadata: DeepseekV4TrtllmAttentionMetadata,
        input_ids: torch.IntTensor = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        return super().forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            return_context_logits=return_context_logits,
            **kwargs,
        )

    def load_weights(self, weights: Dict):
        weight_loader = DeepseekV4WeightLoader(self)
        weight_loader.load_weights(weights)

    def post_load_weights(self):
        layers = self.model.layers[: self.config.num_hidden_layers]
        last_idx = self.config.num_hidden_layers - 1
        for idx, layer in enumerate(layers):
            if idx == last_idx:
                # The V4 logits path is HCHead -> model.norm -> lm_head, so
                # the final decoder layer must not fold model.norm into MoE.
                layer.next_layer_layernorm = None
                layer.fusion_config.POST_MOE_FUSION = False
                layer.defer_post_mapping = False
            else:
                next_layer = layers[idx + 1]
                layer.next_layer_layernorm = next_layer.input_layernorm
                # Defer this layer's hc_ffn.post_mapping into the next layer's
                # hc_attn.fused_hc only if that next layer can actually absorb
                # it: fused_hc enabled on both sides, and no engram at the
                # next layer's entry (engram needs the materialized residual
                # before pre_mapping runs).
                layer.defer_post_mapping = (
                    layer.enable_fused_hc
                    and next_layer.enable_fused_hc
                    and next_layer.engram is None
                )
