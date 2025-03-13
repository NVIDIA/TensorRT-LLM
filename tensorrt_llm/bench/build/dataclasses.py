from transformers import AutoConfig
from typing import Optional, Literal
from pydantic import BaseModel, Field, AliasChoices, model_validator
import huggingface_hub
from huggingface_hub.constants import (
    SAFETENSORS_INDEX_FILE,
    SAFETENSORS_MAX_HEADER_LENGTH,
    SAFETENSORS_SINGLE_FILE,
)
from huggingface_hub.utils import SafetensorsRepoMetadata, SafetensorsFileMetadata, TensorInfo
from huggingface_hub.utils import tqdm as hf_tqdm
from tqdm.contrib.concurrent import thread_map
import os
import json
import struct


def parse_safetensors_file_metadata(model_path, filename):

    with open(os.path.join(model_path, filename), "rb") as f:
        metadata_size = f.read(8)
        metadata_size = struct.unpack("<Q", metadata_size)[0]

        if metadata_size > SAFETENSORS_MAX_HEADER_LENGTH:
            raise RuntimeError(
                f"Failed to parse safetensors header for '{filename}' (model_path '{model_path}'): "
                f"safetensors header is too big. Maximum supported size is "
                f"{SAFETENSORS_MAX_HEADER_LENGTH} bytes (got {metadata_size}).")

        metadata_as_bytes = f.read(metadata_size)

    try:
        metadata_as_dict = json.loads(metadata_as_bytes.decode(errors="ignore"))
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Failed to parse safetensors header for '{filename}' (model_path '{model_path}'): "
            "header format not recognized. Please make sure this is a correctly formatted safetensors file."
        ) from e

    try:
        return SafetensorsFileMetadata(
            metadata=metadata_as_dict.get("__metadata__", {}),
            tensors={
                key:
                TensorInfo(
                    dtype=tensor["dtype"],
                    shape=tensor["shape"],
                    data_offsets=tuple(tensor["data_offsets"]),  # type: ignore
                )
                for key, tensor in metadata_as_dict.items()
                if key != "__metadata__"
            },
        )
    except (KeyError, IndexError) as e:
        raise RuntimeError(
            f"Failed to parse safetensors header for '{filename}' (model_path '{model_path}'): "
            "header format not recognized. Please make sure this is a correctly formatted safetensors file."
        ) from e


def get_safetensors_metadata(model_name_or_path):
    """ Read the safetensors metadata from HF model. """
    if os.path.isdir(model_name_or_path):
        if os.path.exists(
                os.path.join(model_name_or_path, SAFETENSORS_SINGLE_FILE)):
            file_metadata = parse_safetensors_file_metadata(
                model_path=model_name_or_path, filename=SAFETENSORS_SINGLE_FILE)
            return SafetensorsRepoMetadata(
                metadata=None,
                sharded=False,
                weight_map={
                    tensor_name: SAFETENSORS_SINGLE_FILE
                    for tensor_name in file_metadata.tensors.keys()
                },
                files_metadata={SAFETENSORS_SINGLE_FILE: file_metadata},
            )
        elif os.path.exists(
                os.path.join(model_name_or_path, SAFETENSORS_INDEX_FILE)):
            with open(os.path.join(model_name_or_path,
                                   SAFETENSORS_INDEX_FILE)) as f:
                index = json.load(f)

            weight_map = index.get("weight_map", {})

            # Fetch metadata per shard
            files_metadata = {}

            def _parse(filename: str) -> None:
                files_metadata[filename] = parse_safetensors_file_metadata(
                    model_path=model_name_or_path, filename=filename)

            thread_map(
                _parse,
                set(weight_map.values()),
                desc="Parse safetensors files",
                tqdm_class=hf_tqdm,
            )

            return SafetensorsRepoMetadata(
                metadata=index.get("metadata", None),
                sharded=True,
                weight_map=weight_map,
                files_metadata=files_metadata,
            )
        else:
            # Not a safetensors repo
            raise RuntimeError(
                f"'{model_name_or_path}' is not a safetensors repo. Couldn't find '{SAFETENSORS_INDEX_FILE}' or '{SAFETENSORS_SINGLE_FILE}' files."
            )
    else:
        return huggingface_hub.get_safetensors_metadata(model_name_or_path)


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
    def get_param_count(cls, model_hf_name, hf_model_path):
        """ Read the parameter count from HF safetensor metadata. """
        if model_hf_name == "EleutherAI/gpt-j-6b":  # GPT-J repo doesn't use safetensor format.
            param_count = 6053381344
        else:
            model_name_or_path = hf_model_path or model_hf_name
            metadata = get_safetensors_metadata(model_name_or_path)
            param_count = sum(metadata.parameter_count.values())
        assert param_count, f"Can't get valid parameter count for model: {model_name_or_path}."

        return param_count

    @classmethod
    def from_hf(cls, model_hf_name, hf_model_path):
        model_name_or_path = hf_model_path or model_hf_name
        hf_config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=True).to_dict()

        param_count = cls.get_param_count(model_hf_name, hf_model_path)

        return cls(name=model_hf_name, param_count=param_count, **hf_config)
