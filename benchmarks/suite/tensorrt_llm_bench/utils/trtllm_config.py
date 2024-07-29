import json
import os
from argparse import ArgumentParser
from typing import Literal, Optional

from pydantic import AliasChoices, AliasPath, BaseModel, Field, model_validator
from transformers import AutoConfig
from utils import VALID_QUANT_ALGOS

PET_dict = {
    "tiiuae/falcon-7b": "rope_gpt_neox",
    "tiiuae/falcon-40b": "rope_gpt_neox",
    "tiiuae/falcon-180B": "rope_gpt_neox",
    "meta-llama/Llama-2-7b-hf": "rope_gpt_neox",
    "meta-llama/Llama-2-13b-hf": "rope_gpt_neox",
    "meta-llama/Llama-2-70b-hf": "rope_gpt_neox",
    "meta-llama/Meta-Llama-3-8B": "rope_gpt_neox",
    "meta-llama/Meta-Llama-3-70B": "rope_gpt_neox",
    "EleutherAI/gpt-j-6b": "rope_gptj",
    "bigscience/bloom-560m": "alibi",
    "mistralai/Mistral-7B-v0.1": "rope_gpt_neox",
    "mistralai/Mixtral-8x7B-v0.1": "rope_gpt_neox",
    "mistralai/Mixtral-8x22B-v0.1": "rope_gpt_neox",
    "01-ai/Yi-6B": "rope_gpt_neox",
    "01-ai/Yi-34B": "rope_gpt_neox",
    "codellama/CodeLlama-7b-hf": "rope_gpt_neox",
    "codellama/CodeLlama-13b-hf": "rope_gpt_neox",
    "codellama/CodeLlama-34b-hf": "rope_gpt_neox",
    "codellama/CodeLlama-70b-hf": "rope_gpt_neox",
    "facebook/opt-125m": "learned_absolute",
    "facebook/opt-350m": "learned_absolute",
    "facebook/opt-1.3b": "learned_absolute",
    "facebook/opt-2.7b": "learned_absolute",
    "facebook/opt-13b": "learned_absolute",
    "facebook/opt-30b": "learned_absolute",
    "facebook/opt-66b": "learned_absolute",
    "google/gemma-7b": "rope_gpt_neox",
    "google/gemma-2b": "rope_gpt_neox",
}
HA_dict = {
    "tiiuae/falcon-7b": "gelu",
    "tiiuae/falcon-40b": "gelu",
    "tiiuae/falcon-180B": "gelu",
    "bigscience/bloom-560m": "gelu",
    "mistralai/Mixtral-8x7B-v0.1": "swiglu",
}
ALLOWED_MODELS = list(PET_dict.keys())


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


class TRTLLMConfig(BaseModel):
    _VALID_EMBED_TYPE = Literal["learned_absolute", "rope_gptj",
                                "rope_gpt_neox", "alibi", "alibi_with_scale",
                                "relative", "chatglm", ]

    architecture: str = Field(validation_alias=AliasChoices(
        'architecture', AliasPath("architectures", 0)))
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
        validation_alias=AliasChoices("norm_epsilon", "layer_norm_epsilon",
                                      "rms_norm_eps"),
    )
    vocab_size: int
    max_position_embeddings: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("max_position_embeddings", "n_positions"),
    )
    head_size: Optional[int] = None
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

    sliding_window: Optional[int] = None

    moe_num_experts: Optional[int] = Field(
        default=0, validation_alias=AliasChoices("num_local_experts"))
    moe_top_k: Optional[int] = Field(
        default=0, validation_alias=AliasChoices("num_experts_per_tok"))
    rotary_base: Optional[float] = Field(
        default=None, validation_alias=AliasChoices("rope_theta"))

    mapping: TRTLLM_Mapping
    quantization: TRTLLM_Quantization

    @property
    def kv_dtype(self) -> str:
        if self.quantization.kv_cache_quant_algo == "FP8":
            return "fp8"
        elif self.quantization.kv_cache_quant_algo == "INT8":
            return "int8"
        else:
            return self.dtype

    @model_validator(mode="after")
    def set_values_if_none(self) -> "TRTLLM_CheckpointConfig":
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_size is None:
            self.head_size = self.hidden_size // self.num_attention_heads
        return self

    @classmethod
    def populate_build_config(cls,
                              model_name,
                              tp,
                              pp,
                              dtype=None,
                              quant_dtype=None,
                              kv_cache_quant_dtype=None):
        """
        Common function to populate build parameters, regardless of network
        """
        build_config = {
            "mapping": {
                "tp_size": tp,
                "pp_size": pp,
            },
            "quantization": {},
        }
        if dtype:
            build_config["dtype"] = dtype
        if quant_dtype:
            if not kv_cache_quant_dtype:
                # will throw errors during validation if the type is invalid
                kv_cache_quant_dtype = quant_dtype
            build_config["quantization"] = {
                "quant_algo": quant_dtype,
                "kv_cache_quant_algo": kv_cache_quant_dtype,
            }
        if model_name in PET_dict:
            build_config["position_embedding_type"] = PET_dict.get(model_name)
        return build_config

    @classmethod
    def from_hf(cls,
                hf_model_name,
                tp,
                pp,
                dtype=None,
                quant_dtype=None,
                kv_cache_quant_dtype=None):
        """
        Use transformers.AutoConfig to load a model's config from a HF name
        """
        build_config = cls.populate_build_config(hf_model_name, tp, pp, dtype,
                                                 quant_dtype,
                                                 kv_cache_quant_dtype)
        hf_config = AutoConfig.from_pretrained(hf_model_name).to_dict()
        if hf_model_name in HA_dict:
            hf_config["hidden_act"] = HA_dict[hf_model_name]
        return cls(**hf_config, **build_config)

    @classmethod
    def from_json(cls,
                  model_name,
                  tp,
                  pp,
                  dtype=None,
                  quant_dtype=None,
                  kv_cache_quant_dtype=None):
        """
        Load model parameters from a custom json file
        A full path can be specified. Otherwise, look for ./trtllm_configs/(model_name).json
        """
        build_config = cls.populate_build_config(model_name, tp, pp, dtype,
                                                 quant_dtype,
                                                 kv_cache_quant_dtype)
        if os.path.exists(model_name):
            path_to_json = model_name
        else:
            path_to_json = os.path.join(os.path.dirname(__file__),
                                        f"trtllm_configs/{model_name}.json")
            if not os.path.exists(path_to_json):
                raise FileNotFoundError(f"{path_to_json} not found")
        json_config = json.load(open(path_to_json))
        return cls(**json_config, **build_config)

    @classmethod
    def from_name(cls,
                  model,
                  tp,
                  pp,
                  dtype=None,
                  quant_dtype=None,
                  kv_cache_quant_dtype=None):
        """
        Attempts to create a config based on model name. Performs the following steps:
        1. Tries to load the HF config using AutoConfig. This will only work if the network name exists on HF.
        2. If this fails, try to load a custom config stored on $HF_HOME/custom/*.json
        """
        try:
            trtllm_config = cls.from_hf(model, tp, pp, dtype, quant_dtype,
                                        kv_cache_quant_dtype)
        except EnvironmentError:
            try:
                trtllm_config = cls.from_json(model, tp, pp, dtype, quant_dtype,
                                              kv_cache_quant_dtype)
            except FileNotFoundError as e:
                raise NameError(
                    f"Unable to create PretrainedConfig from {model} due to {e}"
                )

        return trtllm_config

    # future possibilities
    # def from_nemo_config (self, nemo_model_name)

    def to_json(self, output_dir):
        with open(os.path.join(output_dir, "generated_config.json"), "w") as f:
            json.dump(self.model_dump(), f, indent=4)


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
    parser.add_argument(
        "--populate_hf_cache",
        action='store_true',
        help="Populate the HF cache with all the supported networks",
    )
    args = parser.parse_args()

    if args.populate_hf_cache:
        for net in PET_dict.keys():
            _ = AutoConfig.from_pretrained(net)
    else:
        trtllm_config = TRTLLMConfig.from_name(args.model, args.tp_size,
                                               args.pp_size, args.dtype,
                                               args.quant_dtype,
                                               args.kv_cache_quant_dtype)
        trtllm_config.to_json(os.getcwd())
