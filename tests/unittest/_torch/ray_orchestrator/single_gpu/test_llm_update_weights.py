import base64
import pickle
import json
import os
import re
import shutil
from tempfile import TemporaryDirectory
from typing import Callable, List, Optional, Tuple

import pytest
import torch
from torch.multiprocessing.reductions import reduce_tensor
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from utils.llm_data import llm_models_root
from utils.torch_ref import RefHFModel

from tensorrt_llm import LLM
from tensorrt_llm._torch.utils import get_device_uuid
from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams


class RefHFModelWithIPCHandles(RefHFModel):
    def __init__(self, model_dir: str, device_id: int = 0, fp8: bool = False):
        if fp8:
            super().__init__(model_dir, device_id)
        else:
            self.device_id = device_id
            config = AutoConfig.from_pretrained(model_dir)
            config.num_hidden_layers = 4
            self.model = AutoModelForCausalLM.from_config(config=config).to(f"cuda:{device_id}")
        self.all_weights = {}
        self.device_uuid = [get_device_uuid(i) for i in range(torch.cuda.device_count())]
        self._replicate_weights()

    def _replicate_weights(self):
        model_weights = []
        for n, p in self.model.named_parameters():
            model_weights.append((n, p.detach().clone()))

        self.all_weights[self.device_id] = model_weights
        for i in range(torch.cuda.device_count()):
            if i != self.device_id:
                cur_weights = []
                for n, p in self.all_weights[self.device_id]:
                    cur_weights.append((n, p.to("cuda:" + str(i))))
                self.all_weights[i] = cur_weights

    def get_weight_ipc_handles(
        self,
        device_ids: Optional[List[int]] = None,
        weight_filter: Optional[Callable[[str], bool]] = None,
    ):
        """
        Get IPC handles for model weights with flexible filtering.

        Args:
            device_ids: List of device indices to get weights from
            weight_filter: Optional function that takes weight name and returns True if weight should be included

        Returns:
            ret: Dictionary containing weight handles
        """
        ret = {}
        device_list = list(range(torch.cuda.device_count())) if device_ids is None else device_ids

        for device in device_list:
            all_handles = []
            for item in self.all_weights[device]:
                name, p = item
                # Apply filter if provided
                if weight_filter is not None and not weight_filter(name):
                    continue
                handle = reduce_tensor(p)
                all_handles.append((name, handle))
            ret[self.device_uuid[device]] = all_handles

        return ret

    def get_weight_ipc_handles_serialized(
        self,
        cuda_device: Optional[List[int]] = None,
        weight_filter: Optional[Callable[[str], bool]] = None,
    ):
        """
        Get base64-encoded serialized IPC handles for model weights.

        Args:
            cuda_device: List of CUDA device indices to get weights from
            weight_filter: Optional function that takes weight name and returns True if weight should be included

        Returns:
            ret: Dictionary mapping device UUIDs to base64-encoded pickled handles
        """
        ret = {}
        device_list = list(range(torch.cuda.device_count())) if cuda_device is None else cuda_device

        for device in device_list:
            all_handles = []
            for item in self.all_weights[device]:
                name, p = item
                # Apply filter if provided
                if weight_filter is not None and not weight_filter(name):
                    continue
                handle = reduce_tensor(p)
                all_handles.append((name, handle))

            # Serialize with base64-encoded pickle
            serialized = base64.b64encode(pickle.dumps(all_handles)).decode("utf-8")
            ret[self.device_uuid[device]] = serialized

        return ret

    def generate_batch_incremental(
        self, original_prompts: List[List[int]], generated_token_ids_list: List[List[int]]
    ):
        """
        Generate tokens incrementally for each prompt in the batch: [prompt, prompt+token0, prompt+token0+token1, ...]
        """
        logits_list = []

        for i in range(len(original_prompts)):
            base_token_ids = torch.tensor(
                original_prompts[i], dtype=torch.long, device=f"cuda:{self.device_id}"
            )
            cur_logits = []
            for j in range(len(generated_token_ids_list[i])):
                if j > 0:
                    cur_gen_tokens = torch.tensor(generated_token_ids_list[i][:j]).to("cuda")
                    cur_token_ids = torch.cat([base_token_ids, cur_gen_tokens], dim=-1)
                else:
                    cur_token_ids = base_token_ids

                ret = self.model.generate(
                    input_ids=cur_token_ids.unsqueeze(0).cuda(),
                    max_new_tokens=1,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

                cur_logits.append(ret["scores"][0])
            cur_logits = torch.stack(cur_logits, dim=0)
            logits_list.append(cur_logits.squeeze(1))

        return logits_list


def compare_logits(
    logits_list: List[torch.Tensor],
    ref_logits_list: List[torch.Tensor],
    topk: int = 20,
    threshold: float = 0.9,
):
    assert len(logits_list) == len(ref_logits_list)
    for i in range(len(logits_list)):
        assert logits_list[i].shape == ref_logits_list[i].shape
        lhs_idx = torch.topk(logits_list[i], topk, dim=-1).indices
        rhs_idx = torch.topk(ref_logits_list[i], topk, dim=-1).indices
        # Token wise comparison
        ratios = []
        for j in range(lhs_idx.shape[0]):
            lhs_idx_j = lhs_idx[j].tolist()
            rhs_idx_j = rhs_idx[j].tolist()
            overlap = set(lhs_idx_j) & set(rhs_idx_j)
            ratios.append(len(overlap) / len(lhs_idx_j))

        mean_ratio = sum(ratios) / len(ratios)
        assert mean_ratio > threshold, (
            f"Prompt {i}: overlap ratio: {mean_ratio:.2%} is less than {threshold:.2%}"
        )


def run_generate(
    llm: LLM,
    hf_model: RefHFModel,
    prompts: List[List[int]],
    sampling_params: SamplingParams,
    fp8: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    llm_responses = []
    llm_logits = []
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        llm_logits.append(output.outputs[0].generation_logits)
        llm_responses.append(output.outputs[0].token_ids)
    if fp8:
        ref_logits = hf_model.generate_batch_incremental(prompts, llm_responses)
    else:
        input_ids, attention_mask, position_ids = RefHFModel.pad_data(prompts, llm_responses)
        ref_logits = hf_model.generate_batch_with_padding(
            input_ids, attention_mask, position_ids, llm_responses, return_logits=True
        )
    return llm_logits, ref_logits


def process_and_copy_folder(src_folder, dst_folder):
    if os.path.exists(dst_folder):
        shutil.rmtree(dst_folder)
    os.makedirs(dst_folder)

    for root, dirs, files in os.walk(src_folder):
        rel_path = os.path.relpath(root, src_folder)
        dest_dir = os.path.join(dst_folder, rel_path)

        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        num_hidden_layers = 4
        for file in files:
            src_path = os.path.join(root, file)
            dest_path = os.path.join(dest_dir, file)
            if "safetensor" in file:
                continue

            if file == "config.json":
                with open(src_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                config["num_hidden_layers"] = num_hidden_layers
                with open(dest_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
            else:
                shutil.copy2(src_path, dest_path)


def process_and_copy_folder(src_folder, dst_folder):
    if os.path.exists(dst_folder):
        shutil.rmtree(dst_folder)
    os.makedirs(dst_folder)

    for root, dirs, files in os.walk(src_folder):
        rel_path = os.path.relpath(root, src_folder)
        dest_dir = os.path.join(dst_folder, rel_path)

        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        num_hidden_layers = 4
        for file in files:
            src_path = os.path.join(root, file)
            dest_path = os.path.join(dest_dir, file)
            if "safetensor" in file:
                continue

            if file == "config.json":
                with open(src_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                config["num_hidden_layers"] = num_hidden_layers
                with open(dest_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
            else:
                shutil.copy2(src_path, dest_path)


@pytest.mark.parametrize("use_serialized_handles", [True, False])
@pytest.mark.parametrize(
    "model_dir",
    [
        "llama-models-v2/TinyLlama-1.1B-Chat-v1.0",
        "Qwen2.5-0.5B-Instruct",
        "Qwen3/Qwen3-8B",
        "Qwen3/Qwen3-30B-A3B",
        "Qwen3/Qwen3-8B-FP8",
    ],
)
def test_llm_update_weights(model_dir, use_serialized_handles):
    """Test LLM update_weights with both serialized and direct IPC handle formats."""
    model_dir = str(llm_models_root() / model_dir)
    with TemporaryDirectory() as tmp_model_dir:
        if "FP8" in model_dir:
            test_model_dir = model_dir
            fp8 = True
        else:
            process_and_copy_folder(model_dir, tmp_model_dir)
            test_model_dir = tmp_model_dir
            fp8 = False
        hf_model = RefHFModelWithIPCHandles(test_model_dir, fp8=fp8)
        tokenizer = AutoTokenizer.from_pretrained(test_model_dir)
        kv_cache_config = KvCacheConfig(enable_block_reuse=True, free_gpu_memory_fraction=0.1)
        llm = LLM(
            model=test_model_dir,
            ray_worker_extension_cls="tensorrt_llm.llmapi.rlhf_utils.WorkerExtension",
            tensor_parallel_size=1,
            load_format="dummy",
            pipeline_parallel_size=1,
            kv_cache_config=kv_cache_config,
        )

        # Generate texts from the prompts.
        prompts_texts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        prompts = [tokenizer.encode(prompt) for prompt in prompts_texts]
        del tokenizer
        sampling_params = SamplingParams(
            temperature=0, return_generation_logits=True, max_tokens=1024
        )

    # Get IPC handles in either serialized or direct format
    if use_serialized_handles:
        ipc_handles = hf_model.get_weight_ipc_handles_serialized([0])
    else:
        ipc_handles = hf_model.get_weight_ipc_handles([0])

        llm._collective_rpc("update_weights", (ipc_handles,))
        # Finalize the update weights
        llm._collective_rpc("update_weights", (None,))

        llm_logits, ref_logits = run_generate(llm, hf_model, prompts, sampling_params, fp8=fp8)
        compare_logits(llm_logits, ref_logits)


@pytest.mark.parametrize(
    "model_dir",
    [
        "llama-models-v2/TinyLlama-1.1B-Chat-v1.0",
        "Qwen2.5-0.5B-Instruct",
        "Qwen3/Qwen3-8B",
        "Qwen3/Qwen3-30B-A3B",
        "Qwen3/Qwen3-8B-FP8",
    ],
)
def test_llm_partial_update_weights(model_dir):
    model_dir = str(llm_models_root() / model_dir)
    with TemporaryDirectory() as tmp_model_dir:
        if "FP8" in model_dir:
            test_model_dir = model_dir
            fp8 = True
        else:
            process_and_copy_folder(model_dir, tmp_model_dir)
            test_model_dir = tmp_model_dir
            fp8 = False
        hf_model = RefHFModelWithIPCHandles(test_model_dir, fp8=fp8)
        tokenizer = AutoTokenizer.from_pretrained(test_model_dir)
        kv_cache_config = KvCacheConfig(enable_block_reuse=True, free_gpu_memory_fraction=0.1)

        llm = LLM(
            model=test_model_dir,
            ray_worker_extension_cls="tensorrt_llm.llmapi.rlhf_utils.WorkerExtension",
            tensor_parallel_size=1,
            load_format="dummy",
            pipeline_parallel_size=1,
            kv_cache_config=kv_cache_config,
        )

        # Generate texts from the prompts.
        prompts_texts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        prompts = [tokenizer.encode(prompt) for prompt in prompts_texts]
        del tokenizer

        sampling_params = SamplingParams(
            temperature=0, return_generation_logits=True, max_tokens=1024
        )

        def common_filter(filter_name: str) -> Callable[[str], bool]:
            def filter_fn(name: str) -> bool:
                return name.endswith(filter_name)

            return filter_fn

        # Generate filter_list from model weight keys by removing layer prefix
        # e.g., "model.layers.41.input_layernorm.weight" -> "input_layernorm.weight"
        layer_prefix_pattern = re.compile(r"^model\.layers\.\d+\.")
        filter_set = set()
        for name, _ in hf_model.all_weights[hf_model.device_id]:
            suffix = layer_prefix_pattern.sub("", name)
            filter_set.add(suffix)
        filter_list = list(filter_set)

        for filter_name in filter_list:
            weight_filter = common_filter(filter_name=filter_name)
            ipc_handles = hf_model.get_weight_ipc_handles([0], weight_filter=weight_filter)
            llm._collective_rpc("update_weights", (ipc_handles,), non_block=True)
        # Finalize the update weights
        llm._collective_rpc("update_weights", (None,))

        llm_logits, ref_logits = run_generate(llm, hf_model, prompts, sampling_params, fp8=fp8)
        compare_logits(llm_logits, ref_logits)
