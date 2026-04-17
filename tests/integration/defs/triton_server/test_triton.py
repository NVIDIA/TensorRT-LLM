import os
import subprocess

import pytest


def run_shell_command(command, llm_root):
    env = os.environ.copy()
    env["LLM_ROOT"] = llm_root
    env["LLM_BACKEND_ROOT"] = os.path.join(llm_root, "triton_backend")
    env["NVIDIA_TRITON_SERVER_VERSION"] = os.environ.get(
        "NVIDIA_TRITON_SERVER_VERSION", "25.03")
    subprocess.run(command, env=env, check=True, shell=True)


def build_model(model_name, llm_root, tritonserver_test_root):
    """Build the model required for the test."""
    env = os.environ.copy()
    env["LLM_ROOT"] = llm_root
    env["LLM_BACKEND_ROOT"] = os.path.join(llm_root, "triton_backend")
    env["LLM_MODELS_ROOT"] = os.environ.get("LLM_MODELS_ROOT",
                                            "/scratch.trt_llm_data/llm-models")
    subprocess.run(f"bash {tritonserver_test_root}/build_model.sh {model_name}",
                   env=env,
                   check=True,
                   shell=True)


@pytest.fixture
def test_name(request):
    return request.param


@pytest.fixture
def model_path(test_name):
    """Returns the appropriate model path based on the test name."""
    model_mapping = {
        "gpt": "gpt2",
        "opt": "opt-125m",
        "llama": "llama-models/llama-7b-hf",
        "mistral": "mistral-7b-v0.1",
        "mistral-ib": "mistral-7b-v0.1",
        "mistral-ib-streaming": "mistral-7b-v0.1",
        "mistral-ib-mm": "mistral-7b-v0.1",
        "gptj": "gpt-j-6b",
        "gpt-ib": "gpt2",
        "gpt-ib-streaming": "gpt2",
        "gpt-ib-ptuning": "gpt2",
        "gpt-ib-lad": "gpt2",
        "gpt-speculative-decoding": "gpt2",
        "gpt-ib-speculative-decoding-bls": "gpt2",
        "gpt-2b-ib-lora": "gpt-next/gpt-next-tokenizer-hf-v2",
        "gpt-gather-logits": "gpt2",
        "medusa": "vicuna-7b-v1.3",
        "eagle": "vicuna-7b-v1.3",
        "bart-ib": "bart-large-cnn",
        "t5-ib": "t5-small",
        "blip2-opt": "blip2-opt-2.7b",
        "mllama": "llama-3.2-models/Llama-3.2-11B-Vision-Instruct",
        "whisper": "whisper-large-v3",
        "gpt-disaggregated-serving-bls": "gpt2",
        "llava_onevision": "llava-onevision-qwen2-7b-ov-hf",
        "qwen2_vl": "Qwen2-VL-7B-Instruct",
        "llava": "llava-1.5-7b-hf",
        "llava_fp8": "llava-1.5-7b-hf"
    }
    model_cache_root = os.environ.get("LLM_MODELS_ROOT",
                                      "/scratch.trt_llm_data/llm-models")
    return os.path.join(model_cache_root, model_mapping.get(test_name, ""))


@pytest.fixture
def engine_dir(test_name, llm_root):
    """Returns the appropriate engine directory based on the test name."""
    engine_mapping = {
        "gpt": "models/core/gpt/trt_engine/gpt2/fp16/1-gpu/",
        "opt": "models/contrib/opt/trt_engine/opt-125m/fp16/1-gpu/",
        "llama": "models/core/llama/llama_outputs",
        "mistral": "models/core/llama/mistral_7b_outputs",
        "mistral-ib": "models/core/llama/ib_mistral_7b_outputs",
        "mistral-ib-streaming": "models/core/llama/ib_mistral_7b_outputs",
        "mistral-ib-mm": "models/core/llama/ib_mistral_7b_outputs",
        "gptj": "models/contrib/gptj/gptj_outputs",
        "gpt-ib": "models/core/gpt/trt_engine/gpt2-ib/fp16/1-gpu/",
        "gpt-ib-streaming": "models/core/gpt/trt_engine/gpt2-ib/fp16/1-gpu/",
        "gpt-ib-ptuning":
        "models/core/gpt/trt_engine/email_composition/fp16/1-gpu/",
        "gpt-ib-lad": "models/core/gpt/trt_engine/gpt2-ib-lad/fp16/1-gpu/",
        "gpt-2b-ib-lora":
        "models/core/gpt/trt_engine/gpt-2b-lora-ib/fp16/1-gpu/",
        "medusa": "medusa/tmp/medusa/7B/trt_engines/fp16/1-gpu/",
        "eagle": "eagle/tmp/eagle/7B/trt_engines/fp16/1-gpu/",
        "bart-ib": "models/core/enc_dec/trt_engine/bart-ib/fp16/1-gpu/",
        "t5-ib": "models/core/enc_dec/trt_engine/t5-ib/fp16/1-gpu/",
        "blip2-opt": "models/core/multimodal/trt_engines/opt-2.7b/fp16/1-gpu",
        "mllama":
        "models/core/multimodal/trt_engines/Llama-3.2-11B-Vision-Instruct/bf16/1-gpu",
        "whisper": "models/core/whisper/trt_engine/whisper",
        "gpt-disaggregated-serving-bls":
        "models/core/gpt/trt_engine/gpt2/fp16/1-gpu/",
        "llava_onevision":
        "models/core/multimodal/trt_engines/llava-onevision-7b/fp16/1-gpu",
        "qwen2_vl": "models/core/multimodal/trt_engines/qwen2-vl-7b/fp16/1-gpu",
        "llava":
        "models/core/multimodal/trt_engines/llava-1.5-7b-hf/fp16/1-gpu",
        "llava_fp8":
        "models/core/multimodal/trt_engines/llava-1.5-7b-hf/fp8/1-gpu"
    }
    return os.path.join(llm_root, "examples/",
                        engine_mapping.get(test_name, ""))


@pytest.mark.parametrize("test_name", ["gpt"], indirect=True)
def test_gpt(tritonserver_test_root, test_name, llm_root, model_path,
             engine_dir):
    # Build the model
    build_model(test_name, llm_root, tritonserver_test_root)

    # Run the test
    tokenizer_type = "auto"
    run_shell_command(
        f"cd {tritonserver_test_root} && ./test.sh {test_name} {engine_dir} {model_path} {tokenizer_type}",
        llm_root)


@pytest.mark.parametrize("test_name", ["llama"], indirect=True)
def test_llama(tritonserver_test_root, test_name, llm_root, model_path,
               engine_dir):
    build_model(test_name, llm_root, tritonserver_test_root)
    tokenizer_type = "llama"
    run_shell_command(
        f"cd {tritonserver_test_root} && ./test.sh {test_name} {engine_dir} {model_path} {tokenizer_type}",
        llm_root)


@pytest.mark.parametrize("test_name", ["mistral"], indirect=True)
def test_mistral(tritonserver_test_root, test_name, llm_root, model_path,
                 engine_dir):
    build_model(test_name, llm_root, tritonserver_test_root)
    tokenizer_type = "llama"
    run_shell_command(
        f"cd {tritonserver_test_root} && ./test.sh {test_name} {engine_dir} {model_path} {tokenizer_type}",
        llm_root)


@pytest.mark.parametrize("test_name", ["gptj"], indirect=True)
def test_gptj(tritonserver_test_root, test_name, llm_root, model_path,
              engine_dir):
    build_model(test_name, llm_root, tritonserver_test_root)
    tokenizer_type = "auto"
    run_shell_command(
        f"cd {tritonserver_test_root} && ./test.sh {test_name} {engine_dir} {model_path} {tokenizer_type}",
        llm_root)


@pytest.mark.parametrize("test_name", ["mistral-ib-streaming"], indirect=True)
def test_mistral_ib_streaming(tritonserver_test_root, test_name, llm_root,
                              model_path, engine_dir):
    build_model("mistral-ib", llm_root, tritonserver_test_root)
    tokenizer_type = "llama"
    run_shell_command(
        f"cd {tritonserver_test_root} && ./test.sh {test_name} {engine_dir} {model_path} {tokenizer_type}",
        llm_root)


@pytest.mark.parametrize("test_name", ["whisper"], indirect=True)
def test_whisper(tritonserver_test_root, test_name, llm_root, model_path,
                 engine_dir):
    build_model(test_name, llm_root, tritonserver_test_root)
    tokenizer_type = "auto"
    decoder_path = f"{engine_dir}/decoder"
    encoder_path = f"{engine_dir}/encoder"
    run_shell_command(
        f"cd {tritonserver_test_root} && ./test.sh {test_name} {decoder_path} {model_path} {tokenizer_type} skip skip {encoder_path}",
        llm_root)


@pytest.mark.parametrize("test_name", ["t5-ib"], indirect=True)
def test_t5_ib(tritonserver_test_root, test_name, llm_root, model_path,
               engine_dir):
    build_model(test_name, llm_root, tritonserver_test_root)
    tokenizer_type = "auto"
    decoder_path = f"{engine_dir}/decoder"
    encoder_path = f"{engine_dir}/encoder"
    run_shell_command(
        f"cd {tritonserver_test_root} && ./test.sh {test_name} {decoder_path} {model_path} {tokenizer_type} skip skip {encoder_path}",
        llm_root)


@pytest.mark.parametrize("test_name", ["bart-ib"], indirect=True)
def test_bart_ib(tritonserver_test_root, test_name, llm_root, model_path,
                 engine_dir):
    build_model(test_name, llm_root, tritonserver_test_root)
    tokenizer_type = "auto"
    decoder_path = f"{engine_dir}/decoder"
    encoder_path = f"{engine_dir}/encoder"
    run_shell_command(
        f"cd {tritonserver_test_root} && ./test.sh {test_name} {decoder_path} {model_path} {tokenizer_type} skip skip {encoder_path}",
        llm_root)


@pytest.mark.parametrize("test_name", ["python-preproc-unit-tests"],
                         indirect=True)
def test_python_preproc_unit_tests(tritonserver_test_root, test_name, llm_root):
    run_shell_command(
        f"cd {llm_root}/triton_backend && PYTHONPATH=all_models/inflight_batcher_llm/preprocessing/1 "
        "python3 -m pytest all_models/tests/test_multi_image_preprocess.py",
        llm_root)


@pytest.mark.parametrize("test_name", ["fill-template"], indirect=True)
def test_fill_template(tritonserver_test_root, test_name, llm_root):
    run_shell_command(
        f"cd {llm_root}/triton_backend && PYTHONPATH=tools/ python3 -m pytest tools/tests/test_fill_template.py",
        llm_root)


@pytest.mark.parametrize("test_name", ["llmapi-unit-tests"], indirect=True)
def test_llmapi_unit_tests(tritonserver_test_root, test_name, llm_root):
    run_shell_command(
        f"cd {llm_root}/triton_backend && PYTHONPATH=all_models/llmapi/tensorrt_llm/1 "
        "python3 -m pytest all_models/tests/test_llmapi_python_backend.py",
        llm_root)
