from transformers import AutoConfig
from typing import Optional, Literal
from pydantic import BaseModel, Field, AliasChoices, model_validator
from huggingface_hub import get_safetensors_metadata


class ModelConfig(BaseModel):
    """ Model specific configurations. The parameters are needed in engine
        setting calculation.
    """
    name: str
    param_count: int
    num_hidden_layers: int = Field(
        validation_alias=AliasChoices("num_hidden_layers", "n_layer"))
    num_attention_heads: int = Field(
        validation_alias=AliasChoices("num_attention_heads", "n_head"))
    num_key_value_heads: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("num_key_value_heads", "num_kv_heads"),
    )
    hidden_size: int = Field(
        validation_alias=AliasChoices("hidden_size", "n_embd"))
    head_size: Optional[int] = Field(default=None,
                                     validation_alias=AliasChoices(
                                         "head_size", "head_dim"))
    max_position_embeddings: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("max_position_embeddings", "n_positions"),
    )
    dtype: Literal["float16", "bfloat16",
                   None] = Field(default="float16",
                                 validation_alias=AliasChoices(
                                     "dtype", "torch_dtype"))

    @model_validator(mode="after")
    def set_values_if_none(self):
        """ Set the values if cannot get values from HF config.json. """
        if not self.dtype:  # for GPT-J
            self.dtype = "float16"
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_size is None:
            self.head_size = self.hidden_size // self.num_attention_heads
        return self

    @classmethod
    def get_safetensor_metadata(cls, model_hf_name):
        """ Read the parameter count from HF safetensor metadata. """
        if model_hf_name == "EleutherAI/gpt-j-6b":  # GPT-J repo doesn't use safetensor format.
            param_count = 6053381344
        elif model_hf_name == "meta-llama/Llama-3.1-8B":
            param_count = 8030261248
        else:
            # TODO: This function requires HF token to access the metadata.
            metadata = get_safetensors_metadata(model_hf_name)
            param_count = metadata.parameter_count.get(
                'F16', metadata.parameter_count.get('BF16', None))
        assert param_count, f"Can't get valid parameter count for model: {model_hf_name}."

        return param_count

    @classmethod
    def from_hf(cls, model_hf_name, hf_model_path):
        try:
            model_path = hf_model_path or model_hf_name
            hf_config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=True).to_dict()
        except EnvironmentError as e:
            raise e

        param_count = cls.get_safetensor_metadata(model_hf_name)

        return cls(name=model_hf_name, param_count=param_count, **hf_config)
