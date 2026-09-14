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
"""Checkpoint-key mapping for GLM-5.3-Flash (``glm5_next``).

Everything here is a pure function of checkpoint key names and the HF config:
the per-key audit (loaded / transformed / ignored), the key remap onto the
runtime's parameter names and destination-owner routing. The tensors are placed by
``Glm5NextForCausalLM.load_weights``.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper
from tensorrt_llm._torch.pyexecutor.config_utils import unwrap_glm5_next_text_config
from tensorrt_llm.quantization.mode import QuantAlgo

#: Checkpoint namespaces this bring-up deliberately does not load. These are
#: matched as exact dotted-component prefixes, never as substrings or globs: a
#: pattern like ``*visual*`` would also swallow a decoder weight that merely
#: contained the word, and the whole point of the audit is that nothing is
#: dropped by accident.
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


class Disposition:
    """How a checkpoint tensor reaches (or does not reach) the runtime."""

    #: Placed on a destination parameter unchanged.
    LOADED = "loaded"
    #: Placed after a shape/dtype/layout transformation (conv fusion, expert
    #: stacking, block-FP8 dequantization, or a companion scale tensor).
    TRANSFORMED = "transformed"
    #: Deliberately not loaded, under an exact allowlisted namespace.
    IGNORED = "ignored"


@dataclass
class Glm5NextWeightAudit:
    """Exhaustive per-key accounting for a GLM-5.3-Flash checkpoint."""

    #: destination module/parameter name -> source key
    destinations: dict[str, str] = field(default_factory=dict)
    #: source key -> disposition
    disposition: dict[str, str] = field(default_factory=dict)
    #: source key -> why it was transformed / ignored
    reason: dict[str, str] = field(default_factory=dict)
    #: keys that could not be placed at all -- always a hard error
    unresolved: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return dict(Counter(self.disposition.values()))

    def keys_with(self, disposition: str) -> list[str]:
        return sorted(k for k, d in self.disposition.items() if d == disposition)

    def reasons(self) -> dict[str, int]:
        return dict(Counter(self.reason.values()))


def _ignored_reason(key: str, mtp_prefixes: Sequence[str]) -> str | None:
    if key.startswith(_VISION_PREFIX):
        return "vision tower weights are loaded by Glm5NextVLM, not the text decoder"
    for prefix in mtp_prefixes:
        if key.startswith(prefix):
            return "MTP / next-n prediction layer is not enabled (no speculative_config)"
    return None


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
    """Resolve every checkpoint key to exactly one destination and disposition.

    This is the Goal-1.2 contract in executable form. It is deliberately
    analytic -- it needs only the safetensors index and the config, not 328 GB
    of materialized weights -- so it can gate every later loading change
    cheaply.

    ``num_mtp_layers`` is how many of the checkpoint's appended next-n
    prediction (MTP) layers the model actually instantiates -- ``0`` for the
    plain text model, ``1`` under one-model MTP speculative decoding. Those
    layers' keys are placed on ``model.layers.{num_hidden_layers + i}.*``
    (the alias the speculative base class appends the draft layers under);
    any remaining MTP layer stays an allowlisted ignore.
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
        ignored = _ignored_reason(key, mtp_prefixes)
        if ignored is not None:
            audit.disposition[key] = Disposition.IGNORED
            audit.reason[key] = ignored
            continue

        dest = remap_glm5_next_key(key)
        if dest is None:
            audit.unresolved.append(key)
            continue

        if dest.endswith(".weight_scale_inv"):
            audit.disposition[key] = Disposition.TRANSFORMED
            audit.reason[key] = "block-FP8 128x128 weight scale"
            audit.destinations[dest] = key
            continue

        if ".mlp.experts." in dest:
            audit.disposition[key] = Disposition.TRANSFORMED
            audit.reason[key] = "routed expert stacked into the fused MoE layout"
            audit.destinations[dest] = key
            continue

        audit.disposition[key] = Disposition.LOADED
        audit.destinations[dest] = key

    return audit


# ---------------------------------------------------------------------------
# Checkpoint quantization
# ---------------------------------------------------------------------------


def glm5_next_is_quantized(model_config: ModelConfig[PretrainedConfig]) -> bool:
    """Whether construction uses the checkpoint's block-FP8 form.

    The runtime constructs models through ``AutoModelForCausalLM.from_config``,
    which calls ``cls(model_config)`` with no further arguments -- so the
    quantization decision must live on the ``ModelConfig`` itself, exactly
    where ``ModelConfig.from_pretrained`` puts it when it reads the
    checkpoint's ``quantization_config`` (``weight_block_size=[128,128]`` maps
    to ``FP8_BLOCK_SCALES``). A constructor flag that defaulted to bf16 would
    make the runtime path build a model the loader must reject.

    This checkpoint is published in exactly one quantized form; any other
    non-None algorithm on the config is a configuration error, not a request
    for a different build.
    """
    quant = getattr(model_config, "quant_config", None)
    if quant is None or quant.quant_algo is None:
        return False
    if quant.quant_algo != QuantAlgo.FP8_BLOCK_SCALES:
        raise ValueError(
            "glm5_next supports only the published FP8_BLOCK_SCALES checkpoint "
            f"form or unquantized bf16 modules; got quant_algo={quant.quant_algo}"
        )
    return True


def _destination_owner(dest: str, num_layers: int) -> Any:
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
    """Checkpoint-key mapper for GLM-5.3-Flash (``glm5_next``).

    The HF checkpoint is a multimodal ``Glm5NextForConditionalGeneration``
    tree; the text model loads its ``model.language_model.*`` subtree with an
    audited 1:1 placement (see :func:`audit_glm5_next_checkpoint`). This class
    is the mapping half of that loader: which keys are ignored (vision tower,
    surplus MTP layers), where each remaining key lands (:meth:`destination`),
    and which materialization unit owns it (:meth:`owner`). The model's
    ``load_weights`` owns the placement itself, so the generic module-name
    callbacks of :class:`HfWeightMapper` are not used for this architecture.
    """

    def audit(self, keys: Iterable[str]) -> Glm5NextWeightAudit:
        """Resolve every checkpoint key against the bound model's destinations."""
        num_mtp_layers = len(getattr(self.model, "mtp_layers", ()))
        return audit_glm5_next_checkpoint(
            keys, self.config.pretrained_config, num_mtp_layers=num_mtp_layers
        )

    @staticmethod
    def destination(key: str) -> str | None:
        """Runtime parameter name for a checkpoint key (``None``: not loaded)."""
        return remap_glm5_next_key(key)

    @staticmethod
    def owner(dest: str, num_layers: int) -> Any:
        """Materialization owner of a destination: a layer index or a named part."""
        return _destination_owner(dest, num_layers)
