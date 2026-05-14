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
import re

from torch import nn

from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper
from tensorrt_llm._torch.modules.linear import W4A16_AWQ_LinearMethod

_LANG_PREFIX = "model.language_model."
_MODEL_PREFIX = "model."
_LAYER_IDX_RE = re.compile(r"layers\.(\d+)$")


@register_mapper("HF", "Gemma4ForCausalLM")
@register_mapper("HF", "Gemma4ForConditionalGeneration")
class Gemma4HfWeightMapper(HfWeightMapper):
    @property
    def _is_vlm(self) -> bool:
        """Check if the model is a VLM (has vision/audio towers) or text-only."""
        return hasattr(self.model, "vision_tower")

    def apply_callbacks(
        self,
        module: nn.Module,
        module_name: str,
        module_names_breakdown: list[str],
        weights: dict,
    ) -> list[dict]:
        """Thread-safe override that inlines ``_duplicate_kv_weights`` with
        per-layer ``head_dim``.

        Gemma4 uses per-layer head_dim (256 for sliding, 512 for full).  The
        base class's single ``self._head_dim`` miscomputes ``num_kv_heads``
        for full-attention layers under TP>1, and the weight mapper is
        called concurrently from a ``ThreadPoolExecutor`` so stashing the
        value on ``self`` would race across threads.  Instead, compute the
        correct head_dim locally here and pass it through a per-call
        callback bound to that layer.
        """
        layer_head_dim = self._resolve_layer_head_dim(module_names_breakdown)
        if layer_head_dim is None:
            return super().apply_callbacks(module, module_name, module_names_breakdown, weights)

        module_weights: list[dict] = []
        for new_name in self._mapping[module_name]:
            fw = self.filter_weights(".".join(module_names_breakdown + [new_name]), weights)
            fw = self._duplicate_kv_weights_with_head_dim(module, new_name, fw, layer_head_dim)
            module_weights.append(fw)
        return module_weights

    def _resolve_layer_head_dim(self, module_names_breakdown: list[str]):
        """Return the ``head_dim`` of the attention layer addressed by
        ``module_names_breakdown`` (e.g. ``["model", "layers", "4",
        "self_attn"]``) or ``None`` if the breakdown does not name an
        attention layer.
        """
        for i in range(len(module_names_breakdown) - 1):
            if module_names_breakdown[i] == "layers" and module_names_breakdown[i + 1].isdigit():
                layer_idx = int(module_names_breakdown[i + 1])
                layers = self._find_decoder_layers()
                if layers is not None and 0 <= layer_idx < len(layers):
                    attn = getattr(layers[layer_idx], "self_attn", None)
                    hd = getattr(attn, "head_dim", None)
                    if isinstance(hd, int) and hd > 0:
                        return hd
        return None

    def _find_decoder_layers(self):
        root = self.model
        if hasattr(root, "llm"):  # Gemma4ForConditionalGeneration wrapper
            root = root.llm
        inner = getattr(root, "model", None)
        return getattr(inner, "layers", None)

    def _duplicate_kv_weights_with_head_dim(
        self,
        module: nn.Module,
        new_name: str,
        weights: dict,
        layer_head_dim: int,
    ):
        """Local, thread-safe replacement for the base ``_duplicate_kv_weights``
        that uses the per-layer ``head_dim`` supplied by ``apply_callbacks``.
        """
        if new_name not in ("k_proj", "v_proj"):
            return weights
        if "weight" not in weights and "bias" not in weights:
            return weights

        kv_shape = weights["weight"].shape[0] if "weight" in weights else weights["bias"].shape[0]
        if isinstance(module.quant_method, W4A16_AWQ_LinearMethod):
            num_kv_heads = kv_shape * 2 // layer_head_dim
        else:
            num_kv_heads = kv_shape // layer_head_dim

        duplicated_keys = ["weight", "bias"]
        if module.quant_config is not None and module.quant_config.quant_mode.has_nvfp4():
            duplicated_keys.append("weight_scale")

        return {
            k: self._duplicate_kv(
                weight=v[:],
                num_kv_heads=num_kv_heads,
                tensor_parallel_size=self._tp_size,
            )
            if k in duplicated_keys
            else v
            for k, v in weights.items()
        }

    def preprocess_weights(self, weights: dict) -> dict:
        """Rename HF checkpoint keys to TRT-LLM module names and handle
        buffers / special tensors that the generic loader cannot handle.

        For the text-only model (Gemma4ForCausalLM):
        1. Strip ``model.language_model.`` -> ``model.`` prefix.

        For the VLM (Gemma4ForConditionalGeneration):
        1. Strip ``model.`` prefix from all keys so that the VLM's
           ``load_weights`` can use ``filter_weights("language_model", ...)``
           to split by component: ``language_model.*``, ``vision_tower.*``,
           ``embed_vision.*``, etc.

        This method is idempotent: if the weights have already been processed
        (no key starts with the expected raw-checkpoint prefix), it returns
        them unchanged.  This is important because ``Gemma4ForCausalLM.load_weights``
        calls ``preprocess_weights`` again when invoked as a sub-model of the VLM.

        Common transformations:
        2. Load ``layer_scalar`` buffers directly into the model.
        3. For ``attention_k_eq_v`` layers, duplicate ``k_proj`` into ``v_proj``.
        """
        # Detect if any key still has the raw checkpoint prefix.
        # Raw Gemma4 checkpoint keys always start with "model.language_model.",
        # "model.vision_tower.", or "model.embed_vision.".  Keys like
        # "model.layers.*" (from the LLM after filter_weights) are already
        # processed and should not be re-transformed.
        _RAW_PREFIXES = (
            _LANG_PREFIX,
            "model.vision_tower.",
            "model.embed_vision.",
            "model.embed_audio.",
            "model.audio_tower.",
        )
        sample_keys = list(weights.keys())[:20]
        has_raw_prefix = any(any(k.startswith(rp) for rp in _RAW_PREFIXES) for k in sample_keys)
        if not has_raw_prefix:
            # Already preprocessed or text-only LLM sub-model call from VLM.
            # Still need MoE remap, layer_scalar, and k_eq_v.
            weights = self._remap_moe_keys(weights)
            return self._handle_buffers_and_kvdup(weights)

        new_weights: dict = {}
        if self._is_vlm:
            # VLM: strip top-level "model." prefix and re-nest the
            # language_model sub-keys so that filter_weights("language_model")
            # produces "model.X" (matching the LLM's internal structure).
            #
            # Checkpoint:  model.language_model.layers.0.X
            # After strip: language_model.layers.0.X
            # Re-nest:     language_model.model.layers.0.X
            # After filter_weights("language_model"): model.layers.0.X  ← correct for LLM
            #
            # Non-language-model keys (vision_tower, embed_vision) just lose "model.".
            _LANG_COMP = "language_model."
            for key in list(weights.keys()):
                new_key = key
                if new_key.startswith(_MODEL_PREFIX):
                    new_key = new_key[len(_MODEL_PREFIX) :]
                # Re-nest language model keys: language_model.X → language_model.model.X
                if new_key.startswith(_LANG_COMP):
                    new_key = _LANG_COMP + "model." + new_key[len(_LANG_COMP) :]
                new_weights[new_key] = weights[key]
        else:
            # Text-only: strip "model.language_model." -> "model."
            for key in list(weights.keys()):
                new_key = key
                if new_key.startswith(_LANG_PREFIX):
                    new_key = "model." + new_key[len(_LANG_PREFIX) :]
                new_weights[new_key] = weights[key]

        # Remap MoE keys: HF uses layers.N.experts/router, TRT-LLM uses layers.N.moe.experts/router
        new_weights = self._remap_moe_keys(new_weights)

        return self._handle_buffers_and_kvdup(new_weights)

    def _remap_moe_keys(self, weights: dict) -> dict:
        """Remap HF Gemma4 MoE keys to TRT-LLM VANILLA format.

        HF stores MoE as 3D tensors:
          experts.gate_up_proj [E, 2*I, H] → per-expert {id}.w1.weight + {id}.w3.weight
          experts.down_proj    [E, H, I]   → per-expert {id}.w2.weight
          router.per_expert_scale          → moe.per_expert_scale
          router.*                         → moe.router.*

        Paths are also adjusted: layers.N.{experts,router} → layers.N.moe.{...}
        """
        _layer_re = r"((?:model\.|language_model\.model\.)?layers\.\d+)\."
        remapped = {}
        for key, val in weights.items():
            # per_expert_scale: HF router.per_expert_scale → TRT moe.per_expert_scale
            m_pes = re.match(_layer_re + r"router\.per_expert_scale$", key)
            if m_pes:
                prefix = m_pes.group(1)
                remapped[f"{prefix}.moe.per_expert_scale"] = val
                continue

            # experts.gate_up_proj [E, 2*I, H] → per-expert w1 + w3
            m_gup = re.match(_layer_re + r"experts\.gate_up_proj$", key)
            if m_gup:
                prefix = m_gup.group(1)
                for eid in range(val.shape[0]):
                    gate, up = val[eid].chunk(2, dim=0)  # each [I, H]
                    remapped[f"{prefix}.moe.experts.{eid}.w1.weight"] = gate
                    remapped[f"{prefix}.moe.experts.{eid}.w3.weight"] = up
                continue

            # experts.down_proj [E, H, I] → per-expert w2
            m_dp = re.match(_layer_re + r"experts\.down_proj$", key)
            if m_dp:
                prefix = m_dp.group(1)
                for eid in range(val.shape[0]):
                    remapped[f"{prefix}.moe.experts.{eid}.w2.weight"] = val[eid]
                continue

            # router.* → moe.router.*
            m_r = re.match(_layer_re + r"router\.", key)
            if m_r:
                new_key = re.sub(_layer_re + r"router\.", r"\1.moe.router.", key)
                remapped[new_key] = val
                continue

            # Non-MoE key: pass through
            remapped[key] = val
        return remapped

    def _handle_buffers_and_kvdup(self, weights: dict) -> dict:
        """Load layer_scalar buffers and duplicate k_proj for k_eq_v layers."""
        # Determine the layer scalar key pattern and accessor based on
        # whether any key starts with "language_model." (VLM sub-model
        # weights after filter_weights) or "model." (text-only).
        # Navigate to decoder layers regardless of model structure
        # (multimodal wrapper has .llm.model.layers, text-only has .model.layers)
        _root = self.model
        if hasattr(_root, "llm"):  # Gemma4ForConditionalGeneration
            _layers = _root.llm.model.layers
        elif hasattr(_root, "model") and hasattr(_root.model, "layers"):
            _layers = _root.model.layers
        else:
            _layers = None

        def get_layer(idx):
            return _layers[idx] if _layers else None

        sample = next(iter(weights), "")
        if sample.startswith("language_model.model."):
            scalar_pattern = r"language_model\.model\.layers\.(\d+)\.layer_scalar"
            key_tmpl = "language_model.model.layers.{}.self_attn.{}_proj.weight"
        else:
            scalar_pattern = r"model\.layers\.(\d+)\.layer_scalar"
            key_tmpl = "model.layers.{}.self_attn.{}_proj.weight"

        layer_scalar_keys = [k for k in weights if k.endswith(".layer_scalar")]
        for key in layer_scalar_keys:
            m = re.match(scalar_pattern, key)
            if m:
                layer_idx = int(m.group(1))
                try:
                    layer = get_layer(layer_idx)
                    layer.layer_scalar.copy_(weights[key])
                except (AttributeError, IndexError):
                    pass
            del weights[key]

        config = self.model.config
        if getattr(config, "attention_k_eq_v", False):
            layer_types = getattr(config, "layer_types", [])
            for layer_idx, lt in enumerate(layer_types):
                if lt == "full_attention":
                    k_key = key_tmpl.format(layer_idx, "k")
                    v_key = key_tmpl.format(layer_idx, "v")
                    if k_key in weights and v_key not in weights:
                        weights[v_key] = weights[k_key]

        # KV shared layers: HF omits k_proj/v_proj for shared layers.
        # The model uses Q-only projection for these layers, so no dummy
        # weights needed — q_proj maps directly to the Q-only Linear.

        return weights

    def should_skip_module(self, module_name: str) -> bool:
        config = self.model.config
        if getattr(config, "tie_word_embeddings", False) and module_name.startswith("lm_head"):
            return True

        # Determine the "inner" model (for VLM: self.model.llm, else: self.model)
        inner = self.model.llm if hasattr(self.model, "llm") else self.model

        if (
            hasattr(inner, "model")
            and hasattr(inner.model, "has_custom_embed_tokens")
            and inner.model.has_custom_embed_tokens
            and module_name == "model.embed_tokens"
        ):
            return True
        if (
            hasattr(inner, "has_custom_lm_head")
            and inner.has_custom_lm_head
            and module_name == "lm_head"
        ):
            return True

        return any(skip_module in module_name for skip_module in self._skip_modules)

    def handle_manual_copy(
        self,
        module_name: str,
        module_weights: dict,
        n: str,
        p: nn.Parameter,
        allow_partial_loading: bool = False,
    ) -> None:
        # Unlike Gemma2/Gemma3, Gemma4 does NOT use the weight+1 convention
        # for RMSNorm. HF Gemma4RMSNorm initializes weights to ones and
        # multiplies directly (no "+1" in forward), so we copy as-is.
        super().handle_manual_copy(
            module_name, module_weights, n, p, allow_partial_loading=allow_partial_loading
        )
