# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Map GLM-5.3-Flash checkpoint keys to text/MTP destinations and loading owners.

Mapping and audit use key names and config only; the model loader places tensors.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper
from tensorrt_llm._torch.pyexecutor.config_utils import unwrap_glm5_next_text_config
from tensorrt_llm.quantization.mode import QuantAlgo

# Match dotted namespaces exactly so unrelated decoder weights are not ignored.
_VISION_PREFIX = "model.visual."
_LANGUAGE_PREFIX = "model.language_model."

# ---------------------------------------------------------------------------
# Checkpoint key mapping
# ---------------------------------------------------------------------------

#: The two hyper-connection sites are published as flat per-layer tensors
#: (``hc_attn_fn``) while the runtime holds each site as one ``mHC`` submodule
#: whose parameters are ``fn`` / ``base`` / ``scale``. Only the separator moves;
#: no tensor is reshaped, split, or fused.
_HC_RE = re.compile(r"^(model\.layers\.\d+\.)hc_(attn|ffn)_(fn|base|scale)$")


@dataclass
class Glm5NextWeightAudit:
    """Checkpoint destinations, explicitly ignored keys and unresolved keys."""

    # Runtime destination -> checkpoint source key.
    destinations: dict[str, str] = field(default_factory=dict)
    ignored: set[str] = field(default_factory=set)
    unresolved: list[str] = field(default_factory=list)


def remap_glm5_next_key(key: str) -> str | None:
    """Map one text-decoder checkpoint key to its runtime destination.

    Returns ``None`` for keys outside the text decoder (the caller decides
    whether that is an allowlisted namespace or an error).
    """
    if key == "lm_head.weight":
        return key
    if not key.startswith(_LANGUAGE_PREFIX):
        return None
    # model.language_model.<rest> -> model.<rest>: the runtime decoder is not
    # nested inside a multimodal wrapper.
    dest = "model." + key[len(_LANGUAGE_PREFIX) :]

    hc = _HC_RE.match(dest)
    if hc is not None:
        prefix, site, param = hc.groups()
        return f"{prefix}hc_{site}.{param}"
    return dest


def audit_glm5_next_checkpoint(
    keys: Iterable[str],
    config: PretrainedConfig,
    *,
    num_mtp_layers: int = 0,
) -> Glm5NextWeightAudit:
    """Classify checkpoint keys and resolve text/MTP destinations without loading tensors.

    num_mtp_layers controls how many appended checkpoint layers are instantiated.
    Remaining MTP layers and the separately loaded vision namespace are ignored;
    unrecognized keys are reported as unresolved.
    """
    text = unwrap_glm5_next_text_config(config)
    num_layers = int(text.num_hidden_layers)
    num_nextn = int(getattr(text, "num_nextn_predict_layers", 0) or 0)
    if num_mtp_layers < 0 or num_mtp_layers > num_nextn:
        raise ValueError(
            f"glm5_next cannot load {num_mtp_layers} MTP layers; the checkpoint "
            f"declares num_nextn_predict_layers={num_nextn}"
        )
    # MTP layers are appended immediately after the decoder stack; the first
    # ``num_mtp_layers`` of them are real destinations, the rest are ignored.
    mtp_prefixes = tuple(
        f"{_LANGUAGE_PREFIX}layers.{num_layers + i}." for i in range(num_mtp_layers, num_nextn)
    )

    audit = Glm5NextWeightAudit()

    for key in keys:
        if key.startswith((_VISION_PREFIX, *mtp_prefixes)):
            audit.ignored.add(key)
            continue
        dest = remap_glm5_next_key(key)
        if dest is None:
            audit.unresolved.append(key)
        else:
            audit.destinations[dest] = key

    return audit


# ---------------------------------------------------------------------------
# Checkpoint quantization
# ---------------------------------------------------------------------------


def glm5_next_is_quantized(model_config: ModelConfig[PretrainedConfig]) -> bool:
    """Read block-FP8 construction from ModelConfig, rejecting other quantization modes.

    Unquantized modules are also supported for component construction.
    """
    quant = model_config.quant_config
    if quant is None or quant.quant_algo is None:
        return False
    if quant.quant_algo != QuantAlgo.FP8_BLOCK_SCALES:
        raise ValueError(
            "glm5_next supports only the published FP8_BLOCK_SCALES checkpoint "
            f"form or unquantized bf16 modules; got quant_algo={quant.quant_algo}"
        )
    return True


def glm5_next_weight_owner(dest: str, num_layers: int) -> Any:
    """Which materialization unit owns ``dest``: a layer index or a named part."""
    match = re.match(r"^model\.layers\.(\d+)\.", dest)
    if match is not None:
        index = int(match.group(1))
        return index if index < num_layers else None
    if dest.startswith("model.embed_tokens."):
        return "embed"
    if dest.startswith("model.norm."):
        return "norm"
    if dest.startswith("lm_head."):
        return "head"
    return None


@register_mapper("HF", "Glm5NextForConditionalGeneration")
@register_mapper("HF", "Glm5NextForCausalLM")
class Glm5NextHfWeightMapper(HfWeightMapper):
    """Resolve checkpoint destinations and owners for the model's custom loader.

    The text decoder loads model.language_model.* and optional MTP layers; vision
    weights are handled separately. Generic module-name callbacks are not used.
    """

    def audit(self, keys: Iterable[str]) -> Glm5NextWeightAudit:
        """Resolve every checkpoint key against the bound model's destinations."""
        num_mtp_layers = len(getattr(self.model, "mtp_layers", ()))
        return audit_glm5_next_checkpoint(
            keys, self.config.pretrained_config, num_mtp_layers=num_mtp_layers
        )
