import os
import subprocess

import pytest

from tensorrt_llm import BuildConfig, SamplingParams
from tensorrt_llm.llmapi import CalibConfig, QuantAlgo, QuantConfig

# isort: off
from .test_llm import cnn_dailymail_path, get_model_path, llm_test_harness
from utils.util import (force_ampere, skip_pre_hopper)
# isort: on

gpt2_model_path = get_model_path('gpt2-medium')
starcoder2_model_path = get_model_path('starcoder2-3b')
phi_3_mini_4k_model_path = get_model_path('Phi-3/Phi-3-mini-4k-instruct')
phi_3_small_8k_model_path = get_model_path('Phi-3/Phi-3-small-8k-instruct')
phi_3_medium_4k_model_path = get_model_path('Phi-3/Phi-3-medium-4k-instruct')
gemma_2_9b_it_model_path = get_model_path('gemma/gemma-2-9b-it')
qwen2_model_path = get_model_path('Qwen2-7B-Instruct')
qwen2_5_model_path = get_model_path('Qwen2.5-0.5B-Instruct')
mamba2_370m_model_path = get_model_path('mamba2/mamba2-370m')
gpt_neox_20b_model_path = get_model_path('gpt-neox-20b')
sampling_params = SamplingParams(max_tokens=10, end_id=-1)


@force_ampere
def test_llm_gpt2():
    llm_test_harness(gpt2_model_path,
                     inputs=["A B C"],
                     references=["D E F G H I J K L M"],
                     sampling_params=sampling_params)


@force_ampere
@pytest.mark.part1
def test_llm_gpt2_sq():
    quant_config = QuantConfig(
        quant_algo=QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN,
        kv_cache_quant_algo=QuantAlgo.INT8)
    calib_config = CalibConfig(calib_dataset=cnn_dailymail_path)
    llm_test_harness(gpt2_model_path,
                     inputs=["A B C"],
                     references=["D E F G H I J K L M"],
                     sampling_params=sampling_params,
                     quant_config=quant_config,
                     calib_config=calib_config)


@force_ampere
@pytest.mark.part1
def test_llm_gpt2_int8_weight_only():
    quant_config = QuantConfig(quant_algo=QuantAlgo.W8A16,
                               kv_cache_quant_algo=QuantAlgo.INT8)
    calib_config = CalibConfig(calib_dataset=cnn_dailymail_path)
    llm_test_harness(gpt2_model_path,
                     inputs=["A B C"],
                     references=["D E F G H I J K L M"],
                     sampling_params=sampling_params,
                     quant_config=quant_config,
                     calib_config=calib_config)


@skip_pre_hopper
@pytest.mark.part1
def test_llm_gpt2_fp8():
    quant_config = QuantConfig(quant_algo=QuantAlgo.FP8)
    calib_config = CalibConfig(calib_dataset=cnn_dailymail_path)
    llm_test_harness(gpt2_model_path,
                     inputs=["A B C"],
                     references=["D E F G H I J K L M"],
                     sampling_params=sampling_params,
                     quant_config=quant_config,
                     calib_config=calib_config)


@force_ampere
@pytest.mark.part0
def test_llm_starcoder2():
    llm_test_harness(starcoder2_model_path,
                     inputs=["def print_hello_world():"],
                     references=['\n    print("Hello World")\n\ndef print'],
                     sampling_params=sampling_params)


@skip_pre_hopper
@pytest.mark.part0
def test_llm_starcoder2_fp8():
    quant_config = QuantConfig(quant_algo=QuantAlgo.FP8)
    calib_config = CalibConfig(calib_dataset=cnn_dailymail_path)
    llm_test_harness(starcoder2_model_path,
                     inputs=["def print_hello_world():"],
                     references=['\n    print("Hello World")\n\ndef print'],
                     sampling_params=sampling_params,
                     quant_config=quant_config,
                     calib_config=calib_config)


def test_llm_phi_3_mini_4k():
    phi_requirement_path = os.path.join(
        os.getenv("LLM_ROOT"), "examples/models/core/phi/requirements.txt")
    command = f"pip install -r {phi_requirement_path}"
    subprocess.run(command, shell=True, check=True, env=os.environ)
    phi3_mini_4k_sampling_params = SamplingParams(max_tokens=13)

    llm_test_harness(
        phi_3_mini_4k_model_path,
        inputs=["I am going to Paris, what should I see?"],
        references=["\n\nAssistant: Paris is a city rich in history,"],
        sampling_params=phi3_mini_4k_sampling_params)


@force_ampere
def test_llm_phi_3_small_8k():
    phi_requirement_path = os.path.join(
        os.getenv("LLM_ROOT"), "examples/models/core/phi/requirements.txt")
    command = f"pip install -r {phi_requirement_path}"
    subprocess.run(command, shell=True, check=True, env=os.environ)
    build_config = BuildConfig()
    build_config.plugin_config._gemm_plugin = 'auto'
    llm_test_harness(
        phi_3_small_8k_model_path,
        inputs=["where is France's capital?"],
        references=[' Paris is the capital of France. It is known'],
        sampling_params=sampling_params,
        build_config=build_config,
        trust_remote_code=True)


@force_ampere
@pytest.mark.part1
def test_llm_gemma_2_9b_it():
    build_config = BuildConfig()
    build_config.max_batch_size = 512
    llm_test_harness(gemma_2_9b_it_model_path,
                     inputs=['A B C'],
                     references=['D E F G H I J K L M'],
                     build_config=build_config,
                     sampling_params=sampling_params)


@pytest.mark.skip(
    reason=
    "Require further transformers update https://github.com/THUDM/ChatGLM3/issues/1324"
)
def test_llm_qwen2():
    build_config = BuildConfig()
    build_config.max_batch_size = 512
    llm_test_harness(qwen2_model_path,
                     inputs=['A B C'],
                     references=['D E F G H I J K L M'],
                     sampling_params=sampling_params,
                     build_config=build_config,
                     trust_remote_code=True)


def test_llm_qwen2_5():
    build_config = BuildConfig()
    build_config.max_batch_size = 512
    llm_test_harness(qwen2_5_model_path,
                     inputs=['A B C'],
                     references=['D E F G H I J K L M'],
                     sampling_params=sampling_params,
                     build_config=build_config,
                     trust_remote_code=True)


def test_llm_qwen2_int4_weight_only():
    quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16)
    calib_config = CalibConfig(calib_dataset=cnn_dailymail_path)
    llm_test_harness(qwen2_model_path,
                     inputs=['A B C'],
                     references=['D E F G H I J K L M'],
                     sampling_params=sampling_params,
                     quant_config=quant_config,
                     calib_config=calib_config,
                     trust_remote_code=True)


@skip_pre_hopper
def test_llm_qwen2_fp8():
    quant_config = QuantConfig(quant_algo=QuantAlgo.FP8)
    calib_config = CalibConfig(calib_dataset=cnn_dailymail_path)
    llm_test_harness(qwen2_model_path,
                     inputs=['A B C'],
                     references=['D E F G H I J K L M'],
                     sampling_params=sampling_params,
                     quant_config=quant_config,
                     calib_config=calib_config,
                     trust_remote_code=True)


def test_llm_mamba2_370m():
    build_config = BuildConfig()
    build_config.plugin_config._paged_kv_cache = False
    build_config.max_batch_size = 8
    llm_test_harness(mamba2_370m_model_path,
                     inputs=['A B C'],
                     references=['D E F G H I J K L M'],
                     sampling_params=sampling_params,
                     tokenizer=gpt_neox_20b_model_path,
                     build_config=build_config,
                     trust_remote_code=True)
