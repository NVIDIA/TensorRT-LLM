import json
import os
import shutil
from pathlib import Path

import pytest
from utils.llm_data import llm_models_root
from utils.util import getSMVersion

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig, MTPDecodingConfig
from tensorrt_llm.llmapi.utils import get_total_gpu_memory


def process_and_copy_folder(src_folder,
                            dst_folder,
                            exclude_mpt_from_quant=False):
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
            if 'safetensor' in file:
                continue

            if file == 'config.json':
                with open(src_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                config['num_hidden_layers'] = num_hidden_layers
                with open(dest_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
            elif file == 'hf_quant_config.json':
                with open(src_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    if exclude_mpt_from_quant:
                        config['exclude_modules'] = ['model.layers.4*']
                with open(dest_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
            else:
                shutil.copy2(src_path, dest_path)


@pytest.mark.parametrize("model_name", ["DeepSeek-R1", "DeepSeek-R1-FP4"],
                         ids=["deepseekr1", "deepseekr1_fp4"])
def test_deepseek_trtllmgen(model_name):

    if getSMVersion() < 100:
        pytest.skip(f"FP4 is not supported in this SM version {getSMVersion()}")

    if get_total_gpu_memory(0) < 60 * 1024**3:
        pytest.skip(f"Not enough GPU memory to run. {get_total_gpu_memory(0)}")

    prompts = [
        "The president of the United States is",
    ] * 4

    spec_config = MTPDecodingConfig(num_nextn_predict_layers=1)

    pytorch_config = dict(
        disable_overlap_scheduler=True,
        use_cuda_graph=False,
        kv_cache_dtype="auto",
        attn_backend="TRTLLM",
        load_format="dummy",
        moe_backend="TRTLLM",
    )

    model_dir = str(llm_models_root() / Path(f"DeepSeek-R1/{model_name}"))
    assert Path(model_dir).exists()
    tmp_model_dir = f"/tmp/{model_name}"
    exclude_mpt_from_quant = model_name == "DeepSeek-R1-FP4"

    process_and_copy_folder(model_dir, tmp_model_dir, exclude_mpt_from_quant)

    llm = LLM(model=tmp_model_dir,
              tensor_parallel_size=1,
              enable_chunked_prefill=False,
              **pytorch_config,
              moe_expert_parallel_size=-1,
              moe_tensor_parallel_size=-1,
              enable_attention_dp=False,
              speculative_config=spec_config,
              kv_cache_config=KvCacheConfig(enable_block_reuse=False,
                                            free_gpu_memory_fraction=0.4))

    sampling_params = SamplingParams(max_tokens=20)

    try:
        llm.generate(prompts, sampling_params)
    except Exception as e:
        raise e
