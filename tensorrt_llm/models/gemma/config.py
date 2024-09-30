# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import TYPE_CHECKING, Optional, Union

import torch
from typing_extensions import Literal

from tensorrt_llm._utils import torch_dtype_to_str
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import (Gemma2ConfigGroup,
                                                PretrainedConfig, QuantConfig)

if TYPE_CHECKING:
    from os import PathLike

    import transformers

    HfConfigOrDir = Union[str, PathLike, transformers.PretrainedConfig]

GEMMA_ARCHITECTURE = "GemmaForCausalLM"
GEMMA2_ARCHITECTURE = "Gemma2ForCausalLM"


class GemmaConfig(PretrainedConfig):

    def __init__(
        self,
        *,
        architecture: str,
        rotary_base: float = 10000.0,
        rotary_scaling: Optional[dict] = None,
        attn_bias: bool = False,
        mlp_bias: bool = False,
        share_embedding_table: Literal[True] = True,
        position_embedding_type: PositionEmbeddingType = PositionEmbeddingType.
        rope_gpt_neox,
        query_pre_attn_scalar: Optional[int] = None,
        final_logit_softcapping: Optional[float] = None,
        attn_logit_softcapping: Optional[float] = None,
        mapping: Optional[Union[Mapping, dict]] = None,
        **kwargs,
    ):
        if not share_embedding_table:
            """
            We always pass `True` - the passed value is `False` by default, and ignored either way.
            We can't just raise an exception here, because this will force the user to explicitly pass `LLM(share_embedding_table=False)`.
            """
            logger.debug("Using `share_embedding_table=True` for Gemma")

        use_parallel_embedding = False
        if mapping:
            use_parallel_embedding = mapping.tp_size > 1 if isinstance(
                mapping, Mapping) else mapping["tp_size"] > 1
        if use_parallel_embedding != kwargs.pop("use_parallel_embedding", None):
            """
            We always pass `bool(mapping.tp_size > 1)` - the passed value is `False` by default, and ignored either way.
            We can't just raise an exception here, because this will force the user to explicitly pass `LLM(use_parallel_embedding=True)`.
            """
            logger.debug(
                f"Using `use_parallel_embedding={use_parallel_embedding}` for Gemma"
            )

        super().__init__(
            architecture=architecture,
            share_embedding_table=True,
            use_parallel_embedding=use_parallel_embedding,
            rotary_base=rotary_base,
            attn_bias=attn_bias,
            mlp_bias=mlp_bias,
            position_embedding_type=position_embedding_type,
            mapping=mapping,
            **kwargs,
        )
        self.rotary_base = rotary_base
        self.rotary_scaling = rotary_scaling
        self.attn_bias = attn_bias
        self.mlp_bias = mlp_bias

        self.inter_layernorms = False
        if self.is_gemma_2:
            self.inter_layernorms = True
            assert query_pre_attn_scalar is not None, "Gemma2 models must configure `query_pre_attn_scalar`"
            self.query_pre_attn_scalar = query_pre_attn_scalar
            self.final_logit_softcapping = final_logit_softcapping
            self.attn_logit_softcapping = attn_logit_softcapping

    GEMMA_ADDED_FIELDS = {
        "rotary_base", "rotary_scaling", "attn_bias", "mlp_bias",
        "inter_layernorms"
    }
    GEMMA2_ADDED_FIELDS = Gemma2ConfigGroup.keys()
    VERBATIM = {
        "num_hidden_layers", "num_attention_heads", "hidden_size",
        "intermediate_size", "vocab_size", "max_position_embeddings",
        "hidden_act", "use_parallel_embedding"
    } | GEMMA2_ADDED_FIELDS

    @property
    def is_gemma_2(self) -> bool:
        return self.architecture == GEMMA2_ARCHITECTURE

    def gemma2_config(self):
        if self.is_gemma_2:
            return self.get_config_group(Gemma2ConfigGroup)
        return None

    def to_dict(self):
        """Serialize the fields added in GemmaConfig"""

        return {
            **super().to_dict(),
            **{f: getattr(self, f)
               for f in self.GEMMA_ADDED_FIELDS},
            **({f: getattr(self, f)
                for f in self.GEMMA2_ADDED_FIELDS} if self.is_gemma_2 else {})
        }

    @classmethod
    def from_hugging_face(
        cls,
        hf_config_or_dir: "HfConfigOrDir",
        dtype: str = "auto",
        mapping: Optional[Mapping] = None,
        quant_config: Optional[QuantConfig] = None,
        **kwargs,
    ) -> "GemmaConfig":
        import transformers
        if isinstance(hf_config_or_dir, transformers.PretrainedConfig):
            hf_config = hf_config_or_dir
        else:
            hf_config = transformers.GemmaConfig.from_pretrained(
                hf_config_or_dir)

        if dtype == "auto":
            dtype = getattr(hf_config, "torch_dtype", None)
            if dtype is None:
                dtype = "float16"
            if isinstance(dtype, torch.dtype):
                dtype = torch_dtype_to_str(dtype)
            if dtype == "float32":
                dtype = "float16"
        assert isinstance(quant_config, QuantConfig) or quant_config is None
        assert isinstance(mapping, Mapping) or mapping is None
        return cls(
            architecture=hf_config.architectures[0],
            dtype=dtype,
            head_size=hf_config.head_dim,
            norm_epsilon=hf_config.rms_norm_eps,
            num_key_value_heads=getattr(hf_config, "num_key_value_heads",
                                        hf_config.num_attention_heads),
            rotary_scaling=getattr(hf_config, "rotary_scaling", None),
            quantization=quant_config,
            mapping=mapping,
            **{
                k: v
                for k, v in hf_config.to_dict().items() if k in cls.VERBATIM
            },
            **kwargs,
        )
