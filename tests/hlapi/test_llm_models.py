import subprocess
from typing import List, Optional

import pytest
import torch

from tensorrt_llm import LLM, BuildConfig, SamplingParams
from tensorrt_llm.hlapi import QuantAlgo, QuantConfig

try:
    from .test_llm import get_model_path
except ImportError:
    from test_llm import get_model_path

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import force_ampere, similar, skip_pre_hopper

gptj_model_path = get_model_path('gpt-j-6b')
gpt2_model_path = get_model_path('gpt2-medium')
starcoder2_model_path = get_model_path('starcoder2-3b')
phi_1_5_model_path = get_model_path('phi-1_5')
phi_2_model_path = get_model_path('phi-2')
phi_3_mini_4k_model_path = get_model_path('Phi-3/Phi-3-mini-4k-instruct')
phi_3_small_8k_model_path = get_model_path('Phi-3/Phi-3-small-8k-instruct')
phi_3_medium_4k_model_path = get_model_path('Phi-3/Phi-3-medium-4k-instruct')
falcon_model_path = get_model_path('falcon-rw-1b')

sampling_params = SamplingParams(max_new_tokens=10)


def llm_test_harness(model_dir: str,
                     prompts: List[str],
                     references: List[str],
                     *,
                     sampling_params: Optional[SamplingParams] = None,
                     similar_threshold: float = 0.8,
                     **llm_kwargs):

    # skip if no enough GPUs
    tp_size = llm_kwargs.get('tensor_parallel_size', 1)
    pp_size = llm_kwargs.get('pipeline_parallel_size', 1)
    world_size = tp_size * pp_size
    if world_size > torch.cuda.device_count():
        pytest.skip(
            f"world_size ({world_size}) is greater than available GPUs ({torch.cuda.device_count()})"
        )

    llm = LLM(model_dir, tokenizer=model_dir, **llm_kwargs)
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    print(outputs)
    for out, ref in zip(outputs, references):
        assert similar(out.outputs[0].text, ref, threshold=similar_threshold)


@force_ampere
def test_llm_gptj():
    llm_test_harness(gptj_model_path,
                     prompts=["A B C"],
                     references=["D E F G H I J K L M"],
                     sampling_params=sampling_params)


@force_ampere
def test_llm_gptj_int4_weight_only():
    quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16)
    llm_test_harness(gptj_model_path,
                     prompts=["A B C"],
                     references=["D E F G H I J K L M"],
                     sampling_params=sampling_params,
                     quant_config=quant_config)


@force_ampere
def test_llm_gptj_tp2():
    llm_test_harness(gptj_model_path,
                     prompts=["A B C"],
                     references=["D E F G H I J K L M"],
                     sampling_params=sampling_params,
                     tensor_parallel_size=2)


@force_ampere
def test_llm_gpt2():
    llm_test_harness(gpt2_model_path,
                     prompts=["A B C"],
                     references=["D E F G H I J K L M"],
                     sampling_params=sampling_params)


@skip_pre_hopper
def test_llm_gpt2_fp8():
    quant_config = QuantConfig(quant_algo=QuantAlgo.FP8)
    llm_test_harness(gpt2_model_path,
                     prompts=["A B C"],
                     references=["D E F G H I J K L M"],
                     sampling_params=sampling_params,
                     quant_config=quant_config)


@force_ampere
def test_llm_starcoder2():
    llm_test_harness(starcoder2_model_path,
                     prompts=["def print_hello_world():"],
                     references=['\n    print("Hello World")\n\ndef print'],
                     sampling_params=sampling_params)


@skip_pre_hopper
def test_llm_starcoder2_fp8():
    quant_config = QuantConfig(quant_algo=QuantAlgo.FP8)
    llm_test_harness(starcoder2_model_path,
                     prompts=["def print_hello_world():"],
                     references=['\n    print("Hello World")\n\ndef print'],
                     sampling_params=sampling_params,
                     quant_config=quant_config)


def test_llm_phi_1_5():
    llm_test_harness(phi_1_5_model_path,
                     prompts=['A B C'],
                     references=[' D E F G H I J K L M'],
                     sampling_params=sampling_params)


def test_llm_phi_2():
    llm_test_harness(phi_2_model_path,
                     prompts=['A B C'],
                     references=[' D E F G H I J K L M'],
                     sampling_params=sampling_params)


def test_llm_phi_3_mini_4k():
    phi_requirement_path = os.path.join(os.getenv("LLM_ROOT"),
                                        "examples/phi/requirements.txt")
    command = f"pip install -r {phi_requirement_path}"
    subprocess.run(command, shell=True, check=True, env=os.environ)
    llm_test_harness(phi_3_mini_4k_model_path,
                     prompts=['A B C'],
                     references=[' D E F G H I J K L M'],
                     sampling_params=sampling_params)


@force_ampere
def test_llm_phi_3_small_8k():
    phi_requirement_path = os.path.join(os.getenv("LLM_ROOT"),
                                        "examples/phi/requirements.txt")
    command = f"pip install -r {phi_requirement_path}"
    subprocess.run(command, shell=True, check=True, env=os.environ)
    build_config = BuildConfig()
    build_config.plugin_config._gemm_plugin = 'auto'
    llm_test_harness(
        phi_3_small_8k_model_path,
        prompts=["where is France's capital?"],
        references=[' Paris is the capital of France. It is known'],
        sampling_params=sampling_params,
        build_config=build_config)


@force_ampere
def test_llm_falcon():
    llm_test_harness(falcon_model_path,
                     prompts=['A B C'],
                     references=['D E F G H I J K L M'],
                     sampling_params=sampling_params)


@force_ampere
def test_llm_falcon_int4_weight_only():
    quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16)
    llm_test_harness(falcon_model_path,
                     prompts=['A B C'],
                     references=['D E F G H I J K L M'],
                     sampling_params=sampling_params,
                     quant_config=quant_config)


@force_ampere
def test_llm_falcon_tp2():
    llm_test_harness(falcon_model_path,
                     prompts=['A B C'],
                     references=['D E F G H I J K L M'],
                     sampling_params=sampling_params,
                     tensor_parallel_size=2)


if __name__ == '__main__':
    test_llm_gptj()
    test_llm_phi_1_5()
    test_llm_phi_2()
    test_llm_phi_3_mini_4k()
    test_llm_phi_3_small_8k()
