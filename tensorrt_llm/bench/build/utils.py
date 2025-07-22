import pynvml
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

DEFAULT_HF_MODEL_DIRS = {
    'BaichuanForCausalLM': 'baichuan-inc/Baichuan-13B-Chat',
    'BloomForCausalLM': 'bigscience/bloom-560m',
    'GLMModel': 'THUDM/glm-10b',
    'ChatGLMModel': 'THUDM/chatglm3-6b',
    'ChatGLMForCausalLM': 'THUDM/chatglm3-6b',
    'FalconForCausalLM': 'tiiuae/falcon-rw-1b',
    'GPTForCausalLM': 'gpt2-medium',
    'GPTJForCausalLM': 'EleutherAI/gpt-j-6b',
    'GPTNeoXForCausalLM': 'EleutherAI/gpt-neox-20b',
    'InternLMForCausalLM': 'internlm/internlm-chat-7b',
    'InternLM2ForCausalLM': 'internlm/internlm2-chat-7b',
    'LlamaForCausalLM': 'meta-llama/Llama-2-7b-hf',
    'MPTForCausalLM': 'mosaicml/mpt-7b',
    'PhiForCausalLM': 'microsoft/phi-2',
    'OPTForCausalLM': 'facebook/opt-350m',
    'QWenLMHeadModel': 'Qwen/Qwen-7B',
    'QWenForCausalLM': 'Qwen/Qwen-7B',
    'Qwen2ForCausalLM': 'Qwen/Qwen1.5-7B',
    'Qwen2MoeForCausalLM': 'Qwen/Qwen1.5-MoE-A2.7B',
    'RecurrentGemmaForCausalLM': 'google/recurrentgemma-2b',
}


def get_device_memory():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total_memory = mem_info.total / (1024**3)
    pynvml.nvmlShutdown()

    return total_memory


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
