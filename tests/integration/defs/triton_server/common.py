import json
import os
import re
import subprocess
import sys
import tempfile
import time
from difflib import SequenceMatcher

import pytest

from .conftest import venv_check_call, venv_check_output
from .trt_test_alternative import (check_call, check_output, print_error,
                                   print_info)

try:
    import psutil
except ModuleNotFoundError:
    check_call(f"pip3 install psutil", shell=True)


def install_venv_custom_package(package_name):
    pip_command = [sys.executable, "-m", "pip", "install", package_name]

    try:
        subprocess.check_call(pip_command)
        print(f"Successfully installed {package_name}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package_name}. Error: {e}")


# Install cuda-python in venv
install_venv_custom_package("cuda-python")

from cuda.bindings import driver as cuda_driver


def getSMVersion():
    # Init
    err_tuple = cuda_driver.cuInit(0)
    err = err_tuple[0] if isinstance(err_tuple, tuple) else err_tuple
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA initialization failed with error code {err}")

    # Device
    err, cuDevice = cuda_driver.cuDeviceGet(0)
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to get CUDA device with error code {err}")

    # Get target architecture
    err, sm_major = cuda_driver.cuDeviceGetAttribute(
        cuda_driver.CUdevice_attribute.
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice)
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(
            f"Failed to get compute capability major with error code {err}")

    err, sm_minor = cuda_driver.cuDeviceGetAttribute(
        cuda_driver.CUdevice_attribute.
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice)
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(
            f"Failed to get compute capability minor with error code {err}")

    return sm_major * 10 + sm_minor


skip_pre_ada = pytest.mark.skipif(
    getSMVersion() < 89,
    reason="This test is not supported in pre-Ada architecture")


def query_gpu_name():
    cmd = r"nvidia-smi --query-gpu=name --format=csv,noheader | head -n1"
    gpu_name = check_output(f"{cmd}", shell=True).strip()

    return gpu_name


def get_gpu_full_name():
    cmd = r"nvidia-smi -L | head -n1"
    gpu_info = check_output(f"{cmd}", shell=True).strip()

    # Extract GPU name using regex pattern
    pattern = r"GPU \d+: (.*?) \(UUID:"
    match = re.search(pattern, gpu_info)
    assert match is not None, f"Failed to extract GPU name from: {gpu_info}"

    return match.group(1).strip()


def check_server_ready(http_port="8000", timeout_timer=None, sleep_interval=5):
    env_timeout = int(os.getenv('TRITON_SERVER_LAUNCH_TIMEOUT', '300'))
    if timeout_timer is None:
        timeout = env_timeout
    else:
        timeout = max(timeout_timer, env_timeout)
    timer = 0
    while True:
        if http_port == "8000":
            status = check_output(
                r"curl -s -w %{http_code} 0.0.0.0:8000/v2/health/ready || true",
                shell=True).strip()
        elif http_port == "8003":
            status = check_output(
                r"curl -s -w %{http_code} 0.0.0.0:8003/v2/health/ready || true",
                shell=True).strip()
        if status == "200":
            break
        elif timer <= timeout:
            time.sleep(sleep_interval)
            timer += sleep_interval
        elif timer > timeout:
            raise TimeoutError(
                f"Error: Launch Triton server timed out, timer is {timeout} seconds."
            )

    print_info(
        f"Triton server launched successfully! Cost {timer} seconds to launch server."
    )


def assert_pattern_match_target(pattern, content, target_value):
    match = re.search(pattern, content)
    assert match is not None, f"'{pattern}' has no matches."
    num_match = int(match.group(1))
    assert num_match == target_value, f"'{pattern}' check failed, {num_match} does not equal to target {target_value}"


def check_server_metrics(metrics_port="8002",
                         batching_strategy="",
                         kv_cache_reuse=""):
    metrics = check_output(f"curl 0.0.0.0:{metrics_port}/metrics 2>&1",
                           shell=True).strip()
    print_info(metrics)

    pattern_request_success = r'nv_inference_request_success\{model="tensorrt_llm",version="1"\} (\d)'
    assert_pattern_match_target(pattern_request_success, metrics, 1)
    pattern_inference_count = r'nv_inference_count\{model="tensorrt_llm",version="1"\} (\d)'
    assert_pattern_match_target(pattern_inference_count, metrics, 1)
    pattern_exec_count = r'nv_inference_exec_count\{model="tensorrt_llm",version="1"\} (\d)'
    assert_pattern_match_target(pattern_exec_count, metrics, 1)
    if kv_cache_reuse == "False":
        pattern_kv_cache_block_used = r'nv_trt_llm_kv_cache_block_metrics\{kv_cache_block_type="used",model="tensorrt_llm",version="1"\} (\d)'
        assert_pattern_match_target(pattern_kv_cache_block_used, metrics, 0)
    if batching_strategy == "inflight_fused_batching":
        pattern_generation_requests = (
            r'nv_trt_llm_inflight_batcher_metrics'
            r'\{inflight_batcher_specific_metric="generation_requests",model="tensorrt_llm",version="1"\} (\d)'
        )
        assert_pattern_match_target(pattern_generation_requests, metrics, 0)


def search_and_replace(file_path, search_words, replace_words):
    with open(file_path, 'r') as file:
        original_contents = file.read()
        updated_contents = re.sub(search_words, replace_words,
                                  original_contents)
    with open(file_path, 'w') as file:
        file.write(updated_contents)


def prepare_ib_model_repo(llm_backend_repo_root, new_model_repo, model_name=""):
    origin_model_repo = os.path.join(llm_backend_repo_root, "all_models",
                                     "inflight_batcher_llm")
    check_call(f"rm -rf {new_model_repo}", shell=True)
    check_call(f"cp -R {origin_model_repo} {new_model_repo}", shell=True)

    if model_name == "whisper":
        whisper_model_repo = os.path.join(llm_backend_repo_root, "all_models",
                                          "whisper", "whisper_bls")
        check_call(f"cp -R {whisper_model_repo} {new_model_repo}", shell=True)


def prepare_custom_config(llm_backend_repo_root, new_model_repo,
                          new_config_name):
    tensorrt_llm_config = os.path.join(llm_backend_repo_root, "all_models",
                                       "inflight_batcher_llm", "tensorrt_llm")
    new_config = os.path.join(new_model_repo, new_config_name)
    check_call(f"cp -R {tensorrt_llm_config} {new_config}", shell=True)


def prepare_multimodal_model_repo(llm_backend_repo_root, new_model_repo,
                                  dir_name):
    origin_model_repo = os.path.join(llm_backend_repo_root, "all_models",
                                     "multimodal", dir_name)
    check_call(f"cp -R {origin_model_repo} {new_model_repo}", shell=True)


def prepare_disaggregated_serving_model_repo(llm_backend_repo_root,
                                             new_model_repo):
    origin_model_repo = os.path.join(llm_backend_repo_root, "all_models",
                                     "disaggregated_serving",
                                     "disaggregated_serving_bls")
    check_call(f"cp -R {origin_model_repo} {new_model_repo}", shell=True)


def prepare_llmapi_model_repo(llm_backend_repo_root, new_model_repo):
    origin_model_repo = os.path.join(llm_backend_repo_root, "all_models",
                                     "llmapi")
    check_call(f"rm -rf {new_model_repo}", shell=True)
    check_call(f"cp -R {origin_model_repo} {new_model_repo}", shell=True)


def modify_ib_config_pbtxt(REPO_PATH,
                           DECODER_ENGINE_PATH,
                           TOKENIZER_PATH,
                           llm_backend_repo_root,
                           DECOUPLED_MODE,
                           MAX_TOKENS_IN_KV_CACHE,
                           MAX_ATTENTION_WINDOW_SIZE,
                           BATCH_SCHEDULER_POLICY,
                           BATCHING_STRATEGY,
                           KV_CACHE_FREE_GPU_MEM_FRACTION,
                           EXCLUDE_INPUT_IN_OUTPUT,
                           ENABLE_TRT_OVERLAP,
                           TRITON_MAX_BATCH_SIZE,
                           MAX_QUEUE_DELAY_MICROSECONDS,
                           MAX_BEAM_WIDTH,
                           ENABLE_KV_CACHE_REUSE,
                           NORMALIZE_LOG_PROBS,
                           ENABLE_CHUNKED_CONTEXT,
                           GPU_DEVICE_IDS,
                           DECODING_MODE,
                           PREPROCESSING_INSTANCE_COUNT,
                           POSTPROCESSING_INSTANCE_COUNT,
                           ACCUMULATE_TOKEN,
                           BLS_INSTANCE_COUNT,
                           TENSORRT_LLM_TARGET_MODEL_NAME="tensorrt_llm_target",
                           TENSORRT_LLM_DRAFT_MODEL_NAME="tensorrt_llm_draft",
                           BACKEND="tensorrtllm",
                           GPU_WEIGHTS_PERCENT="1.0",
                           ENCODER_ENGINE_PATH="",
                           MULTIMODAL_ENGINE_PATH="",
                           DRAFT_ENGINE_PATH="",
                           TARGET_ENGINE_PATH="",
                           MAX_QUEUE_SIZE="0",
                           ENABLE_CONTEXT_FMHA_FP32_ACC="False",
                           PARTICIPANT_IDS="",
                           PARTICIPANT_IDS_DRAFT="",
                           PARTICIPANT_IDS_TARGET="",
                           SPEC_DEC_FAST_LOGITS="0",
                           EXECUTOR_LOOKAHEAD_WINDOW="",
                           EXECUTOR_LOOKAHEAD_NGRAM="",
                           EXECUTOR_LOOKAHEAD_VERIFICATION_SET="",
                           MAX_NUM_IMAGES="1",
                           CROSS_KV_CACHE_FRACTION="",
                           ENCODER_INPUT_FEATURES_DTYPE="TYPE_FP16",
                           GUIDED_DECODING_BACKEND="",
                           XGRAMMAR_TOKENIZER_INFO_PATH="",
                           PROMPT_EMBEDDING_TABLE_DTYPE="TYPE_FP16"):
    fill_template_py = os.path.join(llm_backend_repo_root, "tools",
                                    "fill_template.py")
    tensorrt_llm_config = os.path.join(llm_backend_repo_root, REPO_PATH,
                                       "tensorrt_llm", "config.pbtxt")
    preprocessing_config = os.path.join(llm_backend_repo_root, REPO_PATH,
                                        "preprocessing", "config.pbtxt")
    postprocessing_config = os.path.join(llm_backend_repo_root, REPO_PATH,
                                         "postprocessing", "config.pbtxt")
    ensemble_config = os.path.join(llm_backend_repo_root, REPO_PATH, "ensemble",
                                   "config.pbtxt")
    tensorrt_llm_bls_config = os.path.join(llm_backend_repo_root, REPO_PATH,
                                           "tensorrt_llm_bls", "config.pbtxt")
    whisper_bls_config = os.path.join(llm_backend_repo_root, REPO_PATH,
                                      "whisper_bls", "config.pbtxt")
    disaggregated_serving_bls_config = os.path.join(
        llm_backend_repo_root, REPO_PATH, "disaggregated_serving_bls",
        "config.pbtxt")

    if MULTIMODAL_ENGINE_PATH != "":
        multimodal_enc_config = os.path.join(llm_backend_repo_root, REPO_PATH,
                                             "multimodal_encoders",
                                             "config.pbtxt")

        check_call(
            f"python3 {fill_template_py} -i {multimodal_enc_config} triton_max_batch_size:{TRITON_MAX_BATCH_SIZE}," \
            f"multimodal_model_path:{MULTIMODAL_ENGINE_PATH},encoder_input_features_data_type:{ENCODER_INPUT_FEATURES_DTYPE}," \
            f"prompt_embedding_table_data_type:{PROMPT_EMBEDDING_TABLE_DTYPE}," \
            f"hf_model_path:{TOKENIZER_PATH}",
            shell=True)
        check_call(
            f"python3 {fill_template_py} -i {tensorrt_llm_bls_config} tensorrt_llm_model_name:tensorrt_llm," \
            f"multimodal_encoders_name:multimodal_encoders",
            shell=True)
        check_call(
            f"python3 {fill_template_py} -i {preprocessing_config} max_num_images:{MAX_NUM_IMAGES}",
            shell=True)

    if DRAFT_ENGINE_PATH != "":
        llm_draft_config = os.path.join(llm_backend_repo_root, REPO_PATH,
                                        "tensorrt_llm_draft", "config.pbtxt")
        search_words = 'name: "tensorrt_llm"'
        replace_words = 'name: "tensorrt_llm_draft"'
        search_and_replace(llm_draft_config, search_words, replace_words)
        check_call(
            f"python3 {fill_template_py} -i {llm_draft_config} 'triton_backend:{BACKEND},engine_dir:{DRAFT_ENGINE_PATH},decoupled_mode:{DECOUPLED_MODE}," \
            f"max_tokens_in_paged_kv_cache:{MAX_TOKENS_IN_KV_CACHE},max_attention_window_size:{MAX_ATTENTION_WINDOW_SIZE},batch_scheduler_policy:{BATCH_SCHEDULER_POLICY}," \
            f"batching_strategy:{BATCHING_STRATEGY}," \
            f"kv_cache_free_gpu_mem_fraction:{KV_CACHE_FREE_GPU_MEM_FRACTION},enable_trt_overlap:{ENABLE_TRT_OVERLAP}," \
            f"exclude_input_in_output:{EXCLUDE_INPUT_IN_OUTPUT},triton_max_batch_size:{TRITON_MAX_BATCH_SIZE}," \
            f"max_queue_delay_microseconds:{MAX_QUEUE_DELAY_MICROSECONDS},max_beam_width:{MAX_BEAM_WIDTH}," \
            f"enable_kv_cache_reuse:{ENABLE_KV_CACHE_REUSE},normalize_log_probs:{NORMALIZE_LOG_PROBS}," \
            f"enable_chunked_context:{ENABLE_CHUNKED_CONTEXT},gpu_device_ids:{GPU_DEVICE_IDS},decoding_mode:{DECODING_MODE}," \
            f"gpu_weights_percent:{GPU_WEIGHTS_PERCENT},encoder_engine_dir:{ENCODER_ENGINE_PATH},max_queue_size:{MAX_QUEUE_SIZE}," \
            f"speculative_decoding_fast_logits:{SPEC_DEC_FAST_LOGITS}," \
            f"lookahead_window_size:{EXECUTOR_LOOKAHEAD_WINDOW}," \
            f"lookahead_ngram_size:{EXECUTOR_LOOKAHEAD_NGRAM}," \
            f"lookahead_verification_set_size:{EXECUTOR_LOOKAHEAD_VERIFICATION_SET}," \
            f"encoder_input_features_data_type:{ENCODER_INPUT_FEATURES_DTYPE}," \
            f"prompt_embedding_table_data_type:{PROMPT_EMBEDDING_TABLE_DTYPE}," \
            f"participant_ids:{PARTICIPANT_IDS_DRAFT}," \
            f"logits_datatype:TYPE_FP32'",
            shell=True)
    if TARGET_ENGINE_PATH != "":
        llm_target_config = os.path.join(llm_backend_repo_root, REPO_PATH,
                                         "tensorrt_llm_target", "config.pbtxt")
        search_words = 'name: "tensorrt_llm"'
        replace_words = 'name: "tensorrt_llm_target"'
        search_and_replace(llm_target_config, search_words, replace_words)
        check_call(
            f"python3 {fill_template_py} -i {llm_target_config} 'triton_backend:{BACKEND},engine_dir:{TARGET_ENGINE_PATH},decoupled_mode:{DECOUPLED_MODE}," \
            f"max_tokens_in_paged_kv_cache:{MAX_TOKENS_IN_KV_CACHE},max_attention_window_size:{MAX_ATTENTION_WINDOW_SIZE},batch_scheduler_policy:{BATCH_SCHEDULER_POLICY}," \
            f"batching_strategy:{BATCHING_STRATEGY}," \
            f"kv_cache_free_gpu_mem_fraction:{KV_CACHE_FREE_GPU_MEM_FRACTION},enable_trt_overlap:{ENABLE_TRT_OVERLAP}," \
            f"exclude_input_in_output:{EXCLUDE_INPUT_IN_OUTPUT},triton_max_batch_size:{TRITON_MAX_BATCH_SIZE}," \
            f"max_queue_delay_microseconds:{MAX_QUEUE_DELAY_MICROSECONDS},max_beam_width:{MAX_BEAM_WIDTH}," \
            f"enable_kv_cache_reuse:true,normalize_log_probs:{NORMALIZE_LOG_PROBS}," \
            f"enable_chunked_context:{ENABLE_CHUNKED_CONTEXT},gpu_device_ids:{GPU_DEVICE_IDS},decoding_mode:{DECODING_MODE}," \
            f"gpu_weights_percent:{GPU_WEIGHTS_PERCENT},encoder_engine_dir:{ENCODER_ENGINE_PATH},max_queue_size:{MAX_QUEUE_SIZE}," \
            f"speculative_decoding_fast_logits:{SPEC_DEC_FAST_LOGITS}," \
            f"lookahead_window_size:{EXECUTOR_LOOKAHEAD_WINDOW}," \
            f"lookahead_ngram_size:{EXECUTOR_LOOKAHEAD_NGRAM}," \
            f"lookahead_verification_set_size:{EXECUTOR_LOOKAHEAD_VERIFICATION_SET}," \
            f"encoder_input_features_data_type:{ENCODER_INPUT_FEATURES_DTYPE}," \
            f"prompt_embedding_table_data_type:{PROMPT_EMBEDDING_TABLE_DTYPE}," \
            f"participant_ids:{PARTICIPANT_IDS_TARGET}," \
            f"logits_datatype:TYPE_FP32'",
            shell=True)

    check_call(
        f"python3 {fill_template_py} -i {preprocessing_config} tokenizer_dir:{TOKENIZER_PATH}," \
        f"triton_max_batch_size:{TRITON_MAX_BATCH_SIZE},preprocessing_instance_count:{PREPROCESSING_INSTANCE_COUNT}," \
        f"multimodal_model_path:{MULTIMODAL_ENGINE_PATH},engine_dir:{DECODER_ENGINE_PATH}",
        shell=True)
    check_call(
        f"python3 {fill_template_py} -i {postprocessing_config} tokenizer_dir:{TOKENIZER_PATH}," \
        f"triton_max_batch_size:{TRITON_MAX_BATCH_SIZE},postprocessing_instance_count:{POSTPROCESSING_INSTANCE_COUNT}",
        shell=True)
    check_call(
        f"python3 {fill_template_py} -i {ensemble_config} triton_max_batch_size:{TRITON_MAX_BATCH_SIZE},logits_datatype:TYPE_FP32",
        shell=True)
    check_call(
        f"python3 {fill_template_py} -i {tensorrt_llm_bls_config} triton_max_batch_size:{TRITON_MAX_BATCH_SIZE}," \
        f"decoupled_mode:{DECOUPLED_MODE},accumulate_tokens:{ACCUMULATE_TOKEN},bls_instance_count:{BLS_INSTANCE_COUNT}," \
        f"tensorrt_llm_model_name:{TENSORRT_LLM_TARGET_MODEL_NAME},tensorrt_llm_draft_model_name:{TENSORRT_LLM_DRAFT_MODEL_NAME},logits_datatype:TYPE_FP32," \
        f"prompt_embedding_table_data_type:{PROMPT_EMBEDDING_TABLE_DTYPE}",
        shell=True)

    check_call(
        f"python3 {fill_template_py} -i {tensorrt_llm_config} 'triton_backend:{BACKEND},engine_dir:{DECODER_ENGINE_PATH},decoupled_mode:{DECOUPLED_MODE}," \
        f"max_tokens_in_paged_kv_cache:{MAX_TOKENS_IN_KV_CACHE},max_attention_window_size:{MAX_ATTENTION_WINDOW_SIZE},batch_scheduler_policy:{BATCH_SCHEDULER_POLICY}," \
        f"batching_strategy:{BATCHING_STRATEGY}," \
        f"kv_cache_free_gpu_mem_fraction:{KV_CACHE_FREE_GPU_MEM_FRACTION},cross_kv_cache_fraction:{CROSS_KV_CACHE_FRACTION},enable_trt_overlap:{ENABLE_TRT_OVERLAP}," \
        f"exclude_input_in_output:{EXCLUDE_INPUT_IN_OUTPUT},triton_max_batch_size:{TRITON_MAX_BATCH_SIZE}," \
        f"max_queue_delay_microseconds:{MAX_QUEUE_DELAY_MICROSECONDS},max_beam_width:{MAX_BEAM_WIDTH}," \
        f"enable_kv_cache_reuse:{ENABLE_KV_CACHE_REUSE},normalize_log_probs:{NORMALIZE_LOG_PROBS}," \
        f"enable_chunked_context:{ENABLE_CHUNKED_CONTEXT},gpu_device_ids:{GPU_DEVICE_IDS},decoding_mode:{DECODING_MODE}," \
        f"gpu_weights_percent:{GPU_WEIGHTS_PERCENT},encoder_engine_dir:{ENCODER_ENGINE_PATH},max_queue_size:{MAX_QUEUE_SIZE}," \
        f"enable_context_fmha_fp32_acc:{ENABLE_CONTEXT_FMHA_FP32_ACC}," \
        f"encoder_input_features_data_type:{ENCODER_INPUT_FEATURES_DTYPE}," \
        f"prompt_embedding_table_data_type:{PROMPT_EMBEDDING_TABLE_DTYPE}," \
        f"participant_ids:{PARTICIPANT_IDS}," \
        f"logits_datatype:TYPE_FP32,guided_decoding_backend:{GUIDED_DECODING_BACKEND},tokenizer_dir:{TOKENIZER_PATH},xgrammar_tokenizer_info_path:{XGRAMMAR_TOKENIZER_INFO_PATH}'",
        shell=True)

    if os.path.exists(whisper_bls_config):
        check_call(
            f"python3 {fill_template_py} -i {whisper_bls_config} engine_dir:{ENCODER_ENGINE_PATH}," \
            f"n_mels:128,zero_pad:false,triton_max_batch_size:{TRITON_MAX_BATCH_SIZE},decoupled_mode:{DECOUPLED_MODE}",
            shell=True)
    if os.path.exists(disaggregated_serving_bls_config):
        check_call(
            f"python3 {fill_template_py} -i {disaggregated_serving_bls_config} 'triton_max_batch_size:{TRITON_MAX_BATCH_SIZE}," \
            f"decoupled_mode:{DECOUPLED_MODE},disaggregated_serving_bls_count:{BLS_INSTANCE_COUNT}," \
            "context_model_name:context,generation_model_name:generation,logits_datatype:TYPE_FP32'",
            shell=True)


def modify_disaggregated_serving_config_pbtxt(llm_backend_repo_root, REPO_PATH):
    check_call(f"cp -R {REPO_PATH}/tensorrt_llm {REPO_PATH}/generation",
               shell=True)
    check_call(f"mv {REPO_PATH}/tensorrt_llm {REPO_PATH}/context", shell=True)
    check_call(
        f"mv {REPO_PATH}/disaggregated_serving_bls {REPO_PATH}/tensorrt_llm",
        shell=True)

    tensorrt_llm_config = os.path.join(llm_backend_repo_root, REPO_PATH,
                                       "tensorrt_llm", "config.pbtxt")
    search_and_replace(tensorrt_llm_config, 'name: "disaggregated_serving_bls"',
                       'name: "tensorrt_llm"')
    context_config = os.path.join(llm_backend_repo_root, REPO_PATH, "context",
                                  "config.pbtxt")
    search_and_replace(context_config, 'name: "tensorrt_llm"',
                       'name: "context"')
    generation_config = os.path.join(llm_backend_repo_root, REPO_PATH,
                                     "generation", "config.pbtxt")
    search_and_replace(generation_config, 'name: "tensorrt_llm"',
                       'name: "generation"')


def validate_by_sequence_matcher(output_result, golden_results, threshold):
    rankings = {}
    for golden_result in golden_results:
        output_result = output_result.strip()
        golden_result = golden_result.strip()
        matcher = SequenceMatcher(None, output_result, golden_result)
        # Get the similarity ratio and populate rankings dict
        similarity_ratio = matcher.ratio()
        rankings[str(similarity_ratio)] = golden_result

    # Find out the highest_similarity_ratio
    highest_similarity_ratio, golden_result = max(rankings.items(),
                                                  key=lambda x: float(x[0]))
    print_info(f"output_result: {output_result}")
    print_info(
        f"rankings(similarity_ratio:golden_result):\n{json.dumps(rankings, indent=4)}"
    )

    if float(highest_similarity_ratio) < threshold:
        pytest.fail(
            f"highest_similarity_ratio {highest_similarity_ratio} is less than {threshold}"
        )


def validate_by_keyword(output_result, keyword):
    if keyword not in output_result:
        pytest.fail(f"FAIL! \"{keyword}\" not in output:\n{output_result}")
    else:
        print_info(f"PASS! \"{keyword}\" in output:\n{output_result}")


def run_cpp_backend_tests(feature_name, llm_backend_venv,
                          inflight_batcher_llm_client_root, tokenizer_dir):
    # Chooses script
    script_name = ""
    if feature_name in [
            "test_basic", "test_log_probs", "test_request_id", "test_n_returns"
    ]:
        script_name = f"{inflight_batcher_llm_client_root}/inflight_batcher_llm_client.py"
    elif feature_name in ["test_stop_words", "test_embedding_bias"]:
        script_name = f"{inflight_batcher_llm_client_root}/end_to_end_grpc_client.py"

    # Run command
    if "inflight_batcher_llm_client.py" in script_name:
        run_cmd = [
            f"{script_name}",
            f"--tokenizer-dir={tokenizer_dir}",
        ]

        if feature_name == "test_log_probs":
            run_cmd += [
                "--request-output-len=10",
                "--return-log-probs",
                "--top-k=2",
            ]
        elif feature_name == "test_request_id":
            run_cmd += [
                "--request-id=my_request",
            ]
        elif feature_name == "test_n_returns":
            run_cmd += [
                "--num-return-sequences=3",
            ]

        venv_check_call(llm_backend_venv, run_cmd)
    elif "end_to_end_grpc_client.py" in script_name:
        if feature_name == "test_stop_words":
            run_cmd = [
                f"{script_name}",
                f"-o=10",
                "-p=\"The only thing we have to fear is\"",
                "--stop-words=\" government\"",
            ]
            output = venv_check_output(llm_backend_venv, run_cmd)
            print_info(f"The test output is:\n{output}")
            with tempfile.NamedTemporaryFile(
                    dir=llm_backend_venv.get_working_directory(),
                    mode='w',
                    delete=False) as temp_file:
                temp_file.write(output)
                temp_file.close()
                check_call(
                    f"grep -v \"that the government will\" {temp_file.name}",
                    shell=True)
        if feature_name == "test_embedding_bias":
            run_cmd = [
                f"{script_name}",
                f"-o=10",
                "-p=\"The only thing we have to fear is\"",
                "--embedding-bias-words=\" government\"",
                "--embedding-bias-weights=-20",
            ]
            output = venv_check_output(llm_backend_venv, run_cmd)
            print_info(f"The test output is:\n{output}")
            with tempfile.NamedTemporaryFile(
                    dir=llm_backend_venv.get_working_directory(),
                    mode='w',
                    delete=False) as temp_file:
                temp_file.write(output)
                temp_file.close()
                check_call(
                    f"grep -v \"that the government will\" {temp_file.name}",
                    shell=True)


def run_cpp_streaming_backend_tests(feature_name,
                                    llm_backend_venv,
                                    inflight_batcher_llm_client_root,
                                    tokenizer_dir,
                                    model_name="",
                                    e2e_model=""):
    # Chooses script
    script_name = ""
    if feature_name in ["test_basic"]:
        script_name = f"{inflight_batcher_llm_client_root}/inflight_batcher_llm_client.py"
    elif feature_name in ["batched_inputs"] and e2e_model == "tensorrt_llm_bls":
        script_name = f"{inflight_batcher_llm_client_root}/end_to_end_grpc_client.py"

    # Run command
    if "inflight_batcher_llm_client.py" in script_name:
        run_cmd = [
            f"{script_name}",
            "--streaming",
            f"--tokenizer-dir={tokenizer_dir}",
        ]

        if feature_name == "test_basic":
            venv_check_call(llm_backend_venv, run_cmd)
    elif "end_to_end_grpc_client.py" in script_name:
        raw_input = """["This is a test","I want you to","The cat is"]"""
        raw_output = ""
        gpu_name = query_gpu_name()
        run_cmd = [
            f"{script_name}",
            "--streaming",
            "-o=5",
            f"--model-name={e2e_model}",
            f"-p={raw_input}",
            "--batch-inputs",
            "--overwrite-output-text",
        ]
        if "H100" in gpu_name:
            if "gpt" in model_name.lower():
                raw_output = """[" of the power of the"," know that I am not"," a very good cat."]"""
            elif "llama" in model_name.lower():
                raw_output = """["of the emergency alert","know that I am not", "out of the bag."]"""
            if raw_output != "":
                run_cmd += [
                    f"--expected-outputs={raw_output}",
                    "--check-outputs",
                ]
        if feature_name == "batched_inputs":
            venv_check_call(llm_backend_venv, run_cmd)


def retrieve_latency_value(log):
    m = re.search(r"Latency: (\d+\.\d+) ms", log)
    latency_value = None
    if m is not None:
        latency_value = m.group(1).strip()

    assert latency_value is not None, f"Did not find latency value in log: {log}."
    return float(latency_value)


def get_pid_by_name(process_name):
    proc_pid = None
    for proc in psutil.process_iter(['pid', 'name']):
        # Skip zombie process.
        if proc.info['name'] == process_name and proc.status(
        ) != psutil.STATUS_ZOMBIE:
            proc_pid = proc.info['pid']
            break

    assert proc_pid, f"Fail to get a valid process pid of {process_name}."
    return proc_pid


def get_rss_usage_bytes_by_pid(pid):
    rss = None
    try:
        process = psutil.Process(pid)
        rss = process.memory_info().rss
    except psutil.NoSuchProcess:
        print_error(f"Process with PID {pid} no longer exists.")
    except psutil.AccessDenied:
        print_error(f"Access denied to process with PID {pid}.")
    except Exception as e:
        print_error(f"An error occurred: {e}")

    assert rss is not None, f"Fail to get RSS usage of pid {pid}."
    return rss


def check_avg_rss_increasement(llm_backend_venv,
                               process_name,
                               inference_cmd,
                               rss_increase_bytes_threshold=64,
                               warm_up_times=10,
                               total_run_times=20):
    pid = get_pid_by_name(process_name)
    rss_usage_before_inference = get_rss_usage_bytes_by_pid(pid)

    # Warm-up.
    time = 1
    for _ in range(warm_up_times):
        venv_check_call(llm_backend_venv, inference_cmd)
        current_rss_usage = get_rss_usage_bytes_by_pid(pid)
        print_info(
            f"The RSS usage after {time} inference request is: {current_rss_usage} bytes."
        )
        time += 1

    rss_usage_after_warmup = get_rss_usage_bytes_by_pid(pid)

    # Calculate average RSS increasement.
    if total_run_times <= warm_up_times:
        raise ValueError(f"total_run_times must larger than {warm_up_times}.")
    for _ in range(total_run_times - warm_up_times):
        venv_check_call(llm_backend_venv, inference_cmd)
        current_rss_usage = get_rss_usage_bytes_by_pid(pid)
        print_info(
            f"The RSS usage after {time} inference request is: {current_rss_usage} bytes."
        )
        time += 1

    rss_usage_final_run = get_rss_usage_bytes_by_pid(pid)
    avg_rss_increasement = (rss_usage_final_run - rss_usage_after_warmup) // (
        total_run_times - warm_up_times)

    print_info(f"Checking RSS usage of process: {process_name}.")
    print_info(
        f"The RSS usage before inference is: {rss_usage_before_inference} bytes."
    )
    print_info(
        f"The RSS usage after {warm_up_times} times warm-up run is: {rss_usage_after_warmup} bytes."
    )
    print_info(
        f"The RSS usage after {total_run_times} times run is: {rss_usage_final_run} bytes."
    )
    print_info(
        f"The average RSS increasement after warm-up is: {avg_rss_increasement} bytes."
    )

    if avg_rss_increasement > rss_increase_bytes_threshold:
        pytest.fail(
            f"The average RSS increasement: {avg_rss_increasement} bytes > threshold: {rss_increase_bytes_threshold} bytes."
        )
    else:
        print_info(
            f"The average RSS increasement: {avg_rss_increasement} bytes <= threshold: {rss_increase_bytes_threshold} bytes."
        )


def parse_endpoint_generated_outputs(output_text,
                                     max_tokens,
                                     stream,
                                     count_tokens=False,
                                     check_repetition=False):
    print_info("Analyzing the outputs...")
    pattern = r'"text_output"\s*:\s*"(.*)"'
    matches = re.findall(pattern, output_text)
    assert matches is not None, "No matching outputs."
    print_info(f"The matched output tokens are:\n{matches}")

    if count_tokens:
        num_tokens = max_tokens if stream is True else 1
        num_matches = len(matches)
        assert num_tokens == num_matches, f"The output token amount: {num_matches} is not matching expected: {num_tokens}."

    if check_repetition:
        from collections import Counter
        match_counts = Counter(matches)
        total_matches = len(matches)
        for match, count in match_counts.items():
            repetition_rate = (count / total_matches) * 100
            assert repetition_rate <= 50, f"Repetition rate of '{match}' is {repetition_rate}%, which is beyond the allowed threshold."


def parse_endpoint_generated_json_outputs(output_text, check_repetition=False):
    print_info("Analyzing the outputs...")
    try:
        # Parse the JSON string
        output_json = json.loads(output_text)

        # Extract the text_output field
        text_output = output_json.get("text_output", [])

        # Print the output tokens
        print_info(f"The matched output tokens are:\n{text_output}")

        # Check for repetition in text_output
        if check_repetition:
            if isinstance(text_output, list) and len(text_output) > 0:
                from collections import Counter
                item_counts = Counter(text_output)
                duplicates = {
                    item: count
                    for item, count in item_counts.items() if count > 1
                }

                assert not duplicates, f"Repetition found in text_output: {duplicates}"
            else:
                print_error("text_output is not a list or is empty.")
    except json.JSONDecodeError as e:
        print_error(f"Error parsing JSON: {e}")
