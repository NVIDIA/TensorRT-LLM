import json
import os
from argparse import ArgumentParser
from typing import Literal, Optional

from pydantic import AliasChoices, AliasPath, BaseModel, Field, model_validator
from transformers import AutoConfig
from utils import VALID_QUANT_ALGOS
from utils.enums import ComputeDtypeEnum, KVCacheDtypeEnum

PET_dict = {
    "tiiuae/falcon-7b": "rope_gpt_neox",
    "tiiuae/falcon-40b": "rope_gpt_neox",
    "tiiuae/falcon-180B": "rope_gpt_neox",
    "meta-llama/Llama-2-7b-hf": "rope_gpt_neox",
    "meta-llama/Llama-2-13b-hf": "rope_gpt_neox",
    "meta-llama/Llama-2-70b-hf": "rope_gpt_neox",
    "EleutherAI/gpt-j-6b": "rope_gptj",
    "bigscience/bloom-560m": "alibi",
    "mistralai/Mistral-7B-v0.1": "rope_gpt_neox",
    "01-ai/Yi-6B": "rope_gpt_neox",
    "01-ai/Yi-34B": "rope_gpt_neox",
}
HA_dict = {
    "tiiuae/falcon-7b": "gelu",
    "tiiuae/falcon-40b": "gelu",
    "tiiuae/falcon-180B": "gelu",
    "bigscience/bloom-560m": "gelu",
}


class TRTLLM_Mapping(BaseModel):
    world_size: int = 1
    tp_size: int = 1
    pp_size: int = 1

    @model_validator(mode="after")
    def check_world_size(self) -> "TRTLLM_Mapping":
        self.world_size = self.tp_size * self.pp_size
        return self


class TRTLLM_Quantization(BaseModel):
    quant_algo: Optional[VALID_QUANT_ALGOS] = None

    kv_cache_quant_algo: Optional[Literal[None, "FP8", "INT8"]] = None

    group_size: int = 128
    has_zero_point: bool = False
    pre_quant_scale: bool = False
    exclude_modules: Optional[list] = None


class TRTLLM_CheckpointConfig(BaseModel):
    """Dataclass for building TRT-LLM model configurations."""

    _VALID_EMBED_TYPE = Literal["learned_absolute", "rope_gptj",
                                "rope_gpt_neox", "alibi", "alibi_with_scale",
                                "relative", "chatglm", ]

    architecture: str = Field(validation_alias=AliasPath("architectures", 0))
    num_hidden_layers: int = Field(validation_alias=AliasChoices(
        "num_hidden_layers", "n_layer", "n_layers"))
    num_attention_heads: int = Field(validation_alias=AliasChoices(
        "num_attention_heads", "n_head", "n_heads"))
    num_key_value_heads: int = Field(
        default=None,
        validation_alias=AliasChoices("num_key_value_heads", "num_kv_heads"),
    )

    hidden_size: int = Field(
        validation_alias=AliasChoices("hidden_size", "n_embd", "d_model"))
    norm_epsilon: float = Field(
        default=1e-5,
        validation_alias=AliasChoices("norm_epsilon", "layer_norm_epsilon"),
    )
    vocab_size: int
    max_position_embeddings: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("max_position_embeddings", "n_positions"),
    )
    hidden_act: str = Field(
        validation_alias=AliasChoices("hidden_act", "activation_function"))
    # falcon options
    bias: Optional[bool] = None
    parallel_attention: Optional[bool] = Field(
        default=None, validation_alias=AliasChoices("parallel_attn"))
    new_decoder_architecture: Optional[bool] = None
    # opt options
    do_layer_norm_before: Optional[bool] = None
    # gptj options
    rotary_dim: Optional[int] = None

    # dtype has priority over torch_dtype, the latter of which is usually defined in the HF config
    dtype: Literal["float16", "bfloat16"] = Field(
        validation_alias=AliasChoices("dtype", "torch_dtype"))
    logits_dtype: str = "float32"
    position_embedding_type: _VALID_EMBED_TYPE = "learned_absolute"
    use_parallel_embedding: bool = False
    embedding_sharding_dim: int = 0
    share_embedding_table: bool = False
    intermediate_size: int = None
    use_prompt_tuning: bool = False

    mapping: TRTLLM_Mapping
    quantization: TRTLLM_Quantization

    @model_validator(mode="after")
    def set_kv_head_default_value(self) -> "TRTLLM_CheckpointConfig":
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        return self


class TRTLLMConfig:

    def __init__(self, trtllm_config, hf_config=None) -> None:
        self.trtllm_config = trtllm_config
        self.hf_config = hf_config
        # self.nemo_config = nemo_config

    @classmethod
    def from_hf(
        cls,
        hf_model_name,
        tp,
        pp,
        dtype=None,
        quant_dtype=None,
        kv_cache_quant_dtype=None,
    ):
        build_config = {
            "mapping": {
                "tp_size": tp,
                "pp_size": pp,
            },
            "quantization": {},
        }
        if dtype:
            build_config["dtype"] = ComputeDtypeEnum(dtype).value
        if quant_dtype:
            if not kv_cache_quant_dtype:
                # will throw errors during validation if the type is invalid
                kv_cache_quant_dtype = KVCacheDtypeEnum(quant_dtype).value
            build_config["quantization"] = {
                "quant_algo": quant_dtype,
                "kv_cache_quant_algo":
                KVCacheDtypeEnum(kv_cache_quant_dtype).value,
            }
        build_config["position_embedding_type"] = PET_dict[hf_model_name]
        if hf_model_name in HA_dict:
            build_config["hidden_act"] = HA_dict[hf_model_name]
        hf_config = AutoConfig.from_pretrained(hf_model_name).to_dict()
        trtllm_config = TRTLLM_CheckpointConfig(**hf_config,
                                                **build_config).model_dump()
        return cls(trtllm_config, hf_config)

    def to_json(self, output_dir):
        with open(os.path.join(output_dir, "generated_config.json"), "w") as f:
            json.dump(self.trtllm_config, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="HF model name",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="TP degree",
    )
    parser.add_argument(
        "--pp_size",
        type=int,
        default=1,
        help="PP degree",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        help="Datatype",
    )
    parser.add_argument(
        "--quant_dtype",
        type=str,
        help="Quantization datatype",
    )
    parser.add_argument(
        "--kv_cache_quant_dtype",
        type=str,
        help="KV cache datatype",
    )
    parser.add_argument(
        "--position_embedding_type",
        type=str,
        help="TRT-LLM argument",
    )
    parser.add_argument(
        "--hidden_act",
        type=str,
        help="TRT-LLM argument",
    )
    args = parser.parse_args()

    trtllm_config = TRTLLMConfig.from_hf(
        args.model,
        args.tp_size,
        args.pp_size,
        args.dtype,
        args.quant_dtype,
        args.kv_cache_quant_dtype,
    )
    trtllm_config.to_json(os.getcwd())
