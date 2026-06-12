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
"""Checkpoint conversion for Qwen3.5/Qwen3.6 hybrid MoE.

Maps the HF safetensors (text decoder) to the TensorRT-LLM engine checkpoint,
including the per-layer mixed-precision quant scoping (attention projections =
FP8, MoE/shared-expert/lm_head = W4A16_NVFP4 group_size 16). The pre-quantized
ModelOpt scale tensors (``weight_scale``, ``weight_scale_2``, ``input_scale``)
are renamed to the TRT-LLM names consumed by ``preprocess_perlayer_weights`` so
the engine builds real NVFP4 weights (not a bf16 fallback).
"""

from typing import TYPE_CHECKING, Callable, Tuple

from ...quantization.mode import QuantAlgo
from ..modeling_utils import LayerQuantConfig, QuantConfig

if TYPE_CHECKING:
    import torch


def split_attn_output_gate(
    q_proj_weight: "torch.Tensor", num_heads: int, head_dim: int
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """De-interleave a gate-doubled ``q_proj`` into real-Q and gate rows.

    For a Qwen3-Next / Qwen3.5-MoE full-attention layer, split the doubled
    ``q_proj`` into its real-Q rows and its ``attn_output_gate`` rows.

    In the HF checkpoint the attention output gate is fused into a single
    doubled projection ``q_proj.weight`` of shape
    ``[2 * num_heads * head_dim, hidden]``. The doubling is **per head and
    interleaved**: viewing the rows as ``[num_heads, 2 * head_dim, hidden]``,
    the first ``head_dim`` rows of each head are the real query projection and
    the next ``head_dim`` rows are the gate projection. (This matches HF
    ``Qwen3_5MoeAttention``: ``q_proj(h).view(..., num_heads, head_dim*2)`` then
    ``chunk(2, dim=-1)``.)

    A naive split at the tensor midpoint (``[:num_heads*head_dim]`` /
    ``[num_heads*head_dim:]``) would be WRONG — it would interleave heads. The
    per-head de-interleave below produces **head-major** real-Q rows (feeding
    the fused ``attention.qkv`` Q-section) and **head-major** gate rows (feeding
    ``attention.gate.weight``), the order the engine ``Attention`` and its
    gpt_attention context expect.

    Matches the per-head ``q_proj(h).view(..., num_heads, head_dim*2).chunk(2,
    dim=-1)`` split in HF ``Qwen3_5MoeAttention``.

    Args:
        q_proj_weight: HF ``self_attn.q_proj.weight``,
            shape ``[2 * num_heads * head_dim, hidden]``.
        num_heads: number of query heads (e.g. 16).
        head_dim: per-head dimension (e.g. 256).

    Returns:
        ``(q_weight, gate_weight)`` each shape ``[num_heads * head_dim, hidden]``,
        head-major and contiguous.

    Note: this operates on the raw weight rows; for the FP8-quantized
    checkpoint the per-tensor ``weight_scale``/``input_scale`` apply uniformly
    to both halves (a single scale on the whole ``q_proj``), so the caller may
    split before or after dequantization. The fused-QKV FP8 re-scaling across
    q/k/v is handled by the weight loader, not here.
    """
    hidden = q_proj_weight.shape[-1]
    expected = 2 * num_heads * head_dim
    assert q_proj_weight.shape[0] == expected, (
        f"q_proj rows {q_proj_weight.shape[0]} != 2*num_heads*head_dim "
        f"({expected}); attn_output_gate expects a doubled q projection"
    )
    qp = q_proj_weight.view(num_heads, 2 * head_dim, hidden)
    q_weight = qp[:, :head_dim, :].reshape(num_heads * head_dim, hidden)
    gate_weight = qp[:, head_dim:, :].reshape(num_heads * head_dim, hidden)
    return q_weight.contiguous(), gate_weight.contiguous()


# HF (modelopt) ``quant_algo`` strings -> TRT-LLM ``QuantAlgo``. The MoE/lm_head
# pieces are tagged ``W4A16_NVFP4`` in the checkpoint, which is *not* a TRT-LLM
# ``QuantAlgo`` member; the engine path expresses NVFP4 (4-bit weights, FP8
# group scales at group_size 16, FP32 global scale) as ``QuantAlgo.NVFP4`` +
# ``group_size=16`` (verified: ``QuantAlgo`` has NVFP4 but no W4A16_NVFP4).
_HF_ALGO_TO_TRTLLM = {
    "FP8": QuantAlgo.FP8,
    "W4A16_NVFP4": QuantAlgo.NVFP4,
    "NVFP4": QuantAlgo.NVFP4,
}


def _hf_layer_quant_config(hf_quant_config: dict) -> dict:
    """Return the HF ``quantized_layers`` map (module name -> spec dict).

    The Qwen3.5/3.6 ``config.json.quantization_config`` carries both a
    ``config_groups`` block and a flat ``quantized_layers`` dict; the latter is
    a clean per-module ``{name: {quant_algo, group_size?}}`` map and is the
    authoritative source used here.
    """
    layers = hf_quant_config.get("quantized_layers")
    if not isinstance(layers, dict) or not layers:
        raise ValueError(
            "quantization_config.quantized_layers is missing or empty; expected "
            "the modelopt per-module quant map for a MIXED_PRECISION checkpoint."
        )
    return layers


def _to_quant_config(hf_spec: dict, hf_name: str) -> QuantConfig:
    """Translate one HF ``quantized_layers`` entry into a ``QuantConfig``."""
    algo = hf_spec.get("quant_algo")
    if algo not in _HF_ALGO_TO_TRTLLM:
        raise ValueError(
            f"Unsupported quant_algo {algo!r} for module {hf_name!r}; expected "
            f"one of {sorted(_HF_ALGO_TO_TRTLLM)}."
        )
    trtllm_algo = _HF_ALGO_TO_TRTLLM[algo]
    kwargs = {"quant_algo": trtllm_algo}
    if trtllm_algo == QuantAlgo.NVFP4:
        # NVFP4 is group-wise; the checkpoint uses group_size 16.
        kwargs["group_size"] = hf_spec.get("group_size", 16)
    return QuantConfig(**kwargs)


def build_layer_quant_config(
    hf_quant_config: dict, num_hidden_layers: int, is_linear_attention: Callable[[int], bool]
) -> "LayerQuantConfig":
    """Translate the HF ``quantization_config`` into a TRT-LLM ``LayerQuantConfig``.

    The checkpoint is ``MIXED_PRECISION``: the attention
    projections are FP8, the MoE experts / shared expert / ``lm_head`` are
    NVFP4 (group_size 16), and MTP + vision stay bf16. The returned
    ``LayerQuantConfig`` keys are **TRT-LLM module-name globs** (matched by
    ``fnmatch`` against the names produced by ``named_modules_with_parent`` in
    :func:`tensorrt_llm.quantization.quantize.quantize`), NOT the HF tensor
    names — so the per-layer assignment must be expressed in engine namespace:

    =====================================  =========================================
    HF module (checkpoint)                 TRT-LLM module (engine)
    =====================================  =========================================
    ``self_attn.{q,k,v}_proj``             ``self_attn.attention.qkv`` (fused)
    ``self_attn.q_proj`` (gate half)       ``self_attn.attention.gate``
    ``self_attn.o_proj``                   ``self_attn.attention.dense``
    ``linear_attn.{in_proj_qkv,...}``      ``linear_attn.{in_proj_qkv,...}`` (same)
    ``mlp.experts`` / ``mlp.shared_*``     ``mlp`` (the whole SharedMoE, see below)
    ``lm_head``                            ``lm_head``
    =====================================  =========================================

    (The decoder wraps the engine ``Attention`` module under a ``self_attn``
    attribute, so the full-attention sub-linears live at
    ``transformer.layers.{i}.self_attn.attention.{qkv,gate,dense}``.)

    The MoE assignment targets only the parent ``...mlp`` module: ``quantize``
    re-creates the whole ``SharedMoE`` (rebuilding its expert + shared-expert
    sub-linears as NVFP4 in the constructor, while internally excluding the
    router and ``shared_expert_gate``), and ``named_modules_with_parent``
    re-reads children from the parent after replacement, so the rebuilt
    children must NOT also match a glob (mirrors Mixtral's
    ``PretrainedConfig.to_layer_quant_config``). The unquantized GDN params
    (``in_proj_a/b``, ``conv1d``, ``A_log``, ``dt_bias``, ``norm``), routers,
    and norms get no glob -> ``quantize`` leaves them at the model dtype.

    Args:
        hf_quant_config: the HF ``config.json.quantization_config`` dict.
        num_hidden_layers: number of text-decoder layers (40 here).
        is_linear_attention: ``layer_idx -> bool`` (True for Gated DeltaNet
            layers, False for full-attention layers).

    Returns:
        A ``LayerQuantConfig(quant_algo=MIXED_PRECISION, quantized_layers=...)``.
    """
    hf_layers = _hf_layer_quant_config(hf_quant_config)

    def hf_spec(name: str) -> dict:
        if name not in hf_layers:
            raise ValueError(
                f"Expected quantized module {name!r} in "
                f"quantization_config.quantized_layers but it is absent."
            )
        return hf_layers[name]

    quantized_layers: dict = {}
    for i in range(num_hidden_layers):
        prefix = f"transformer.layers.{i}"
        hf_prefix = f"model.language_model.layers.{i}"
        if is_linear_attention(i):
            # Gated DeltaNet: in_proj_qkv / in_proj_z / out_proj are FP8 (same
            # name in both namespaces); in_proj_a/b, conv1d, A_log, dt_bias,
            # norm are unquantized and intentionally omitted.
            for sub in ("in_proj_qkv", "in_proj_z", "out_proj"):
                cfg = _to_quant_config(
                    hf_spec(f"{hf_prefix}.linear_attn.{sub}"), f"{hf_prefix}.linear_attn.{sub}"
                )
                quantized_layers[f"{prefix}.linear_attn.{sub}"] = cfg
        else:
            # Full attention: q/k/v/o_proj are FP8. q and o map to the fused
            # ``attention.qkv`` and ``attention.dense``; the gate half of the
            # doubled q_proj feeds ``attention.gate`` (same FP8 per-tensor
            # scale as q_proj).
            q_cfg = _to_quant_config(
                hf_spec(f"{hf_prefix}.self_attn.q_proj"), f"{hf_prefix}.self_attn.q_proj"
            )
            o_cfg = _to_quant_config(
                hf_spec(f"{hf_prefix}.self_attn.o_proj"), f"{hf_prefix}.self_attn.o_proj"
            )
            quantized_layers[f"{prefix}.self_attn.attention.qkv"] = q_cfg
            quantized_layers[f"{prefix}.self_attn.attention.gate"] = q_cfg
            quantized_layers[f"{prefix}.self_attn.attention.dense"] = o_cfg
        # MoE (every layer) -> NVFP4. ``quantize`` re-creates the whole
        # ``SharedMoE`` from the parent ``...mlp`` glob (which rebuilds the 256
        # routed experts as an NVFP4 ``MOEWeightWrapper`` and internally excludes
        # the router + shared_expert_gate). BUT the rebuilt ``SharedMoE``'s
        # shared-expert ``MLP`` constructs plain ``ColumnLinear``/``RowLinear``
        # (its ctor does not quantize), so the shared-expert sub-linears need
        # their OWN globs -- ``named_modules_with_parent`` re-reads children from
        # the parent after replacement, so the walk reaches the rebuilt
        # ``shared_expert.fc/proj`` and swaps them to FP4. These names do NOT
        # collide with the expert wrapper (``mlp.fc``/``mlp.proj``), so the
        # experts are NOT doubly wrapped.
        moe_cfg = _to_quant_config(hf_spec(f"{hf_prefix}.mlp.experts"), f"{hf_prefix}.mlp.experts")
        quantized_layers[f"{prefix}.mlp"] = moe_cfg
        for sub in ("gate_proj", "up_proj", "down_proj"):
            # the shared expert's three HF projections share the experts' NVFP4
            # format in this checkpoint; assert presence + matching algo.
            _to_quant_config(
                hf_spec(f"{hf_prefix}.mlp.shared_expert.{sub}"),
                f"{hf_prefix}.mlp.shared_expert.{sub}",
            )
        quantized_layers[f"{prefix}.mlp.shared_expert.fc"] = moe_cfg
        quantized_layers[f"{prefix}.mlp.shared_expert.proj"] = moe_cfg

    # lm_head -> NVFP4.
    quantized_layers["lm_head"] = _to_quant_config(hf_spec("lm_head"), "lm_head")

    return LayerQuantConfig(quant_algo=QuantAlgo.MIXED_PRECISION, quantized_layers=quantized_layers)


def load_weights_from_hf_model(hf_model_dir: str, model) -> None:
    """Load the HF checkpoint into ``model`` via the unified ``ModelWeightsLoader``.

    Mirrors the modern qwen MoE path
    (``tensorrt_llm/models/qwen/model.py``): the unified loader translates each
    TRT-LLM parameter name to its HF tensor name(s) and dispatches to the
    module's own ``postprocess``, so the FP8 qkv-scale fusion, NVFP4 block-scale
    interleave + per-expert ``alpha``, and 256-expert stacking are all handled by
    the framework. This converter only supplies the Qwen3-Next-specific remaps
    the framework cannot infer:

      * the gate-doubled ``q_proj`` → fused ``attention.qkv`` Q-section +
        ``attention.gate`` (de-interleaved split, FP8 per-tensor scale shared);
      * the MoE ``w1/w3/w2`` → ``gate_proj/up_proj/down_proj`` rename and the
        router ``mlp.gate`` mapping (mirrors qwen);
      * the NVFP4 ``shared_expert.fc`` gate+up fusion (the fused FP4 ColumnLinear
        has no list-concat in its postprocess, so it is concatenated here).

    The GDN ``linear_attn`` params keep their HF names and load through the
    default path (FP8 for ``in_proj_qkv/in_proj_z/out_proj``; bf16/fp32 for
    ``in_proj_a/b``, ``conv1d``, ``A_log``, ``dt_bias``, ``norm``).

    Loads with ``skip_tp=True`` on the split projections (single-GPU,
    ``tp_size==1`` for the text-only token-exact check). MTP + vision are not
    constructed in the engine model, so the loader never requests their weights.

    Args:
        hf_model_dir: path to the HF checkpoint directory (safetensors).
        model: the constructed (and already ``quantize``-d) ``Qwen3NextForCausalLM``.
    """
    import torch
    from tqdm import tqdm

    from ...layers.moe import MOEWeightWrapper
    from ..model_weights_loader import ModelWeightsLoader

    config = model.config
    num_heads = config.num_attention_heads
    head_dim = config.head_size

    # Base key remaps the framework cannot infer from module names alone.
    #   * transformer -> model.language_model: the checkpoint nests the text
    #     decoder under ``model.language_model.*`` (multimodal wrapper), while
    #     ``lm_head`` stays top-level. The base loader maps ``transformer ->
    #     model``, which misses the ``language_model`` infix.
    #   * attention -> "": the decoder nests the engine ``Attention`` module under
    #     a ``self_attn`` attribute, so the TRT-LLM path is
    #     ``self_attn.attention.{qkv,gate,dense}``. The base loader maps
    #     ``attention -> self_attn``; collapse our extra ``attention`` section to
    #     empty so it resolves to the HF ``self_attn.{q,k,v,o}_proj``.
    #   * fc -> [up_proj, gate_proj]: the fused gate+up ColumnLinear in every MLP
    #     (incl. the shared expert) is loaded from the two HF projections, in the
    #     order the engine's gated-activation GEMM expects (up first, then gate),
    #     matching qwen3_moe.
    #   * q/k layernorm names.
    custom_dict = {
        "transformer": "model.language_model",
        "attention": "",
        "fc": ["up_proj", "gate_proj"],
        "q_layernorm": "q_norm",
        "k_layernorm": "k_norm",
    }
    loader = ModelWeightsLoader(hf_model_dir, custom_dict)
    loader.update_key_mapping(model)

    def split_q_half(weights):
        """Replace the q_proj weight (index 0) with its real-Q half.

        For the fused FP8 qkv, ``weight`` translates to the 6-tensor list
        ``[q_w, q_ws, k_w, k_ws, v_w, v_ws]`` (``FP8Linear`` is_qkv dict). Only
        the q weight is gate-doubled; de-interleave it to the real-Q rows and
        leave the per-tensor scales untouched (they apply to both halves).
        """
        if weights is None:
            return weights
        wlist = weights if isinstance(weights, list) else [weights]
        if wlist[0] is not None:
            wlist[0] = split_attn_output_gate(wlist[0].contiguous(), num_heads, head_dim)[0]
        return wlist if isinstance(weights, list) else wlist[0]

    def take_gate_half(weights):
        """Take the gate half of the gate-doubled q_proj weight."""
        if weights is None:
            return weights
        return split_attn_output_gate(weights.contiguous(), num_heads, head_dim)[1]

    # Patch per-module key dicts: MoE experts (w1/w3/w2 -> gate/up/down_proj),
    # router (mlp.gate), and the attention output gate (sources from q_proj).
    for tllm_key, _ in model.named_parameters():
        sub_module = model
        for attr in tllm_key.split(".")[:-1]:
            sub_module = getattr(sub_module, attr)

        if "router" in tllm_key or isinstance(sub_module, MOEWeightWrapper):
            d = sub_module.tllm_to_externel_key_dict
            d["mlp"] = "mlp"
            if "fc" in d:
                d["fc"] = [k.replace("w1", "gate_proj").replace("w3", "up_proj") for k in d["fc"]]
            if "proj" in d:
                d["proj"] = [k.replace("w2", "down_proj") for k in d["proj"]]
            sub_module.tllm_to_externel_key_dict = d
        elif ".attention.gate." in tllm_key:
            # The output gate has no direct HF tensor: source the weight from the
            # gate half of the gate-doubled q_proj, and reuse q_proj's per-tensor
            # FP8 scales (they apply to both halves).
            sub_module.tllm_to_externel_key_dict = {
                "gate": "q_proj",
                "weights_scaling_factor": "weight_scale",
                "activation_scaling_factor": "input_scale",
            }

    def fuse_shared_expert_fc(tllm_key, weights):
        """Fuse the [up, gate] halves of the NVFP4 ``shared_expert.fc``.

        ``fc -> [up_proj, gate_proj]`` makes every key a 2-element list; the
        fused FP4 ColumnLinear (non-qkv) ``postprocess`` does not concat lists.
        The per-row tensors (FP4 ``weight``, FP8 ``weights_block_scaling_factor``)
        concat along the output-feature dim 0. The global/activation scales are
        per-tensor; verified identical across up/gate in this checkpoint (modelopt
        used a shared global), so collapse the list to a single scalar — which
        lets the default FP4 postprocess (interleave / ``.float()`` /
        ``alpha = w_global * a_global``) produce the correct fused result.
        """
        if weights is None or not isinstance(weights, list):
            return weights
        if any(w is None for w in weights):
            raise ValueError(f"Missing an up/gate half while fusing {tllm_key}: {weights}")
        if (
            tllm_key.endswith("weight")
            or tllm_key.endswith("weights_block_scaling_factor")
            or tllm_key.endswith("weights_block_scaling_factor_interleaved")
        ):
            # per-output-row tensors -> concat along dim 0 (up rows then gate rows)
            return torch.cat(weights, dim=0)
        if tllm_key.endswith("alpha"):
            # translated to [up_w2, up_in, gate_w2, gate_in]; up==gate, keep one pair
            return [weights[0], weights[1]]
        # per-tensor global/activation scales: up == gate, collapse to one.
        return weights[0]

    def conv1d_weight_4d(weights):
        """Reshape the depthwise ``conv1d.weight`` to the engine's 4D layout.

        HF depthwise ``conv1d.weight`` is ``[conv_dim, 1, kernel]``; the engine
        ``MambaConv1d`` expects ``[conv_dim, 1, kernel, 1]`` (mirrors
        ``mamba/convert.py``). ``loader.fill`` would otherwise pad the missing
        dim and silently misshape the conv.
        """
        if weights is None:
            return weights
        return weights.unsqueeze(3)

    def gemma_norm_offset(weights):
        """Fold the Gemma-style RMSNorm ``+1`` offset into the loaded weight.

        Qwen3.5/3.6 uses the Gemma-style ``Qwen3_5MoeRMSNorm`` whose forward is
        ``x_normed * (1 + weight)`` for the decoder layernorms (input_layernorm,
        post_attention_layernorm), q/k norms, and the final norm. The engine's
        ``RmsNorm`` computes the standard ``x_normed * weight``, so fold the ``+1``
        into the loaded weight here. (The GDN gated norm uses the plain-weight
        ``Qwen3_5MoeRMSNormGated`` and is intentionally NOT offset.)
        """
        if weights is None:
            return weights
        return weights.float() + 1.0

    # TRT-LLM param-name suffixes whose RMSNorm uses the (1 + weight) convention.
    GEMMA_NORM_SUFFIXES = (
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".attention.q_layernorm.weight",
        ".attention.k_layernorm.weight",
    )

    tllm_weights = {}
    for tllm_key, _ in tqdm(model.named_parameters(), desc="Loading Qwen3-Next weights"):
        if ".attention.qkv." in tllm_key and tllm_key.endswith("weight"):
            tllm_weights.update(loader.load(tllm_key, preprocess=split_q_half, skip_tp=True))
        elif tllm_key.endswith(".attention.gate.weight"):
            tllm_weights.update(loader.load(tllm_key, preprocess=take_gate_half, skip_tp=True))
        elif ".mlp.shared_expert.fc." in tllm_key:
            tllm_weights.update(
                loader.load(tllm_key, preprocess=lambda w, k=tllm_key: fuse_shared_expert_fc(k, w))
            )
        elif tllm_key.endswith(".linear_attn.conv1d.weight"):
            tllm_weights.update(loader.load(tllm_key, preprocess=conv1d_weight_4d))
        elif tllm_key.endswith(".linear_attn.conv1d.bias"):
            # HF Qwen3-Next conv1d uses bias=False, but the engine MambaConv1d
            # always allocates a bias and uses it in the conv -> supply zeros.
            param = model
            for attr in tllm_key.split("."):
                param = getattr(param, attr)
            from ..._utils import trt_dtype_to_torch

            tllm_weights[tllm_key] = torch.zeros(
                tuple(param.shape), dtype=trt_dtype_to_torch(param.dtype)
            )
        elif tllm_key.endswith(GEMMA_NORM_SUFFIXES) or tllm_key == "transformer.norm.weight":
            # Gemma-style (1 + weight) RMSNorm (NOT the GDN gated norm).
            tllm_weights.update(loader.load(tllm_key, preprocess=gemma_norm_offset))
        else:
            tllm_weights.update(loader.load(tllm_key))

    loader.fill(tllm_weights)
