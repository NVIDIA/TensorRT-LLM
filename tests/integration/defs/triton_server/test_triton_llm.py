import os
import re
import sys

import pytest
import torch
import yaml

from .build_engines import *
from .common import *
from .conftest import find_repo_root, venv_check_call, venv_check_output
from .trt_test_alternative import call, check_call, print_info

LLM_ROOT = os.environ.get("LLM_ROOT", find_repo_root())
sys.path.append(os.path.join(LLM_ROOT, "triton_backend"))


@pytest.fixture(autouse=True)
def stop_triton_server():
    # Make sure Triton server are killed before each test.
    call(f"pkill -9 -f tritonserver", shell=True)
    call(f"pkill -9 -f trtllmExecutorWorker", shell=True)
    time.sleep(2)
    yield
    # Gracefully terminate Triton Server after each test.
    call(f"pkill -f tritonserver", shell=True)
    call(f"pkill -f trtllmExecutorWorker", shell=True)
    time.sleep(8)


@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble", "tensorrt_llm_bls"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["True", "False"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", [""])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY",
                         ["max_utilization", "guaranteed_no_evict"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", [""])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY", ["inflight_fused_batching"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["True", "False"],
                         ids=["enableDecoupleMode", "disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["False"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("ENABLE_CHUNKED_CONTEXT", ["False"])
@pytest.mark.parametrize("GPU_DEVICE_IDS", [""])
@pytest.mark.parametrize("DECODING_MODE", [""])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
@pytest.mark.parametrize("FEATURE_NAME", [
    "test_basic", "batched_inputs", "test_log_probs", "test_request_id",
    "test_stop_words", "test_embedding_bias", "test_n_returns"
])
def test_llama_v2_7b_ifb(
    E2E_MODEL_NAME,
    FEATURE_NAME,
    MAX_TOKENS_IN_KV_CACHE,
    MAX_ATTENTION_WINDOW_SIZE,
    BATCH_SCHEDULER_POLICY,
    KV_CACHE_FREE_GPU_MEM_FRACTION,
    ENABLE_TRT_OVERLAP,
    BATCHING_STRATEGY,
    DECOUPLED_MODE,
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
    EXCLUDE_INPUT_IN_OUTPUT,
    inflight_batcher_llm_client_root,
    tensorrt_llm_llama_example_root,
    llama_v2_tokenizer_model_root,
    llm_backend_venv,
):
    if BATCHING_STRATEGY == "V1" and BATCH_SCHEDULER_POLICY == "max_utilization":
        pytest.skip("Skipping. V1 doesn't support max_utilization.")

    if BATCHING_STRATEGY == "V1" and FEATURE_NAME == "test_embedding_bias":
        pytest.skip("Skipping. V1 doesn't support embedding_bias tensor yet.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.path.join(LLM_ROOT, "triton_backend")
    # Build engine
    ENGINE_PATH = prepare_llama_v2_7b_engine("ifb",
                                             tensorrt_llm_llama_example_root,
                                             llama_v2_tokenizer_model_root)

    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)

    # Modify config.pbtxt
    TOKENIZER_PATH = llama_v2_tokenizer_model_root
    modify_ib_config_pbtxt(
        new_model_repo,
        ENGINE_PATH,
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
        TENSORRT_LLM_TARGET_MODEL_NAME="tensorrt_llm",
        TENSORRT_LLM_DRAFT_MODEL_NAME="",
    )

    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()
    # Run Test
    feature_name = f"{FEATURE_NAME}"
    tokenizer_dir = f"{llama_v2_tokenizer_model_root}"

    if DECOUPLED_MODE == "False":
        run_cpp_backend_tests(feature_name, llm_backend_venv,
                              inflight_batcher_llm_client_root, tokenizer_dir)
    else:
        test_model_name = ""
        if ACCUMULATE_TOKEN == "True" and E2E_MODEL_NAME == "tensorrt_llm_bls":
            test_model_name = "llama_v2_7b"

        run_cpp_streaming_backend_tests(feature_name,
                                        llm_backend_venv,
                                        inflight_batcher_llm_client_root,
                                        tokenizer_dir,
                                        model_name=test_model_name,
                                        e2e_model=E2E_MODEL_NAME)


@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["False"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", ["4096"])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY",
                         ["max_utilization", "guaranteed_no_evict"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", [""])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY", ["inflight_fused_batching"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["True", "False"],
                         ids=["enableDecoupleMode", "disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["False"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("ENABLE_CHUNKED_CONTEXT", ["False"])
@pytest.mark.parametrize("GPU_DEVICE_IDS", [""])
@pytest.mark.parametrize("DECODING_MODE", [""])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
def test_mistral_v1_7b_ifb(
    E2E_MODEL_NAME,
    MAX_TOKENS_IN_KV_CACHE,
    MAX_ATTENTION_WINDOW_SIZE,
    BATCH_SCHEDULER_POLICY,
    KV_CACHE_FREE_GPU_MEM_FRACTION,
    ENABLE_TRT_OVERLAP,
    BATCHING_STRATEGY,
    DECOUPLED_MODE,
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
    EXCLUDE_INPUT_IN_OUTPUT,
    inflight_batcher_llm_client_root,
    tensorrt_llm_llama_example_root,
    mistral_v1_tokenizer_model_root,
    llm_backend_venv,
):
    if BATCHING_STRATEGY == "V1" and BATCH_SCHEDULER_POLICY == "max_utilization":
        pytest.skip("Skipping. V1 doesn't support max_utilization.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.path.join(LLM_ROOT, "triton_backend")
    # Build Engine
    ENGINE_PATH = prepare_mistral_v1_7b_engine("ifb",
                                               tensorrt_llm_llama_example_root,
                                               mistral_v1_tokenizer_model_root)

    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)

    # Modify config.pbtxt
    TOKENIZER_PATH = mistral_v1_tokenizer_model_root
    modify_ib_config_pbtxt(
        new_model_repo,
        ENGINE_PATH,
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
    )

    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --force --world_size 1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()
    # Run Test
    run_cmd = [
        f"{inflight_batcher_llm_client_root}/inflight_batcher_llm_client.py",
        f"--tokenizer-dir={mistral_v1_tokenizer_model_root}",
        "--tokenizer-type=llama",
    ]
    if DECOUPLED_MODE == "True":
        run_cmd += [
            "--streaming",
        ]

    venv_check_call(llm_backend_venv, run_cmd)


@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["False"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", ["4096"])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY",
                         ["max_utilization", "guaranteed_no_evict"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", [""])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY", ["inflight_fused_batching"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["True", "False"],
                         ids=["enableDecoupleMode", "disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["False"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("ENABLE_CHUNKED_CONTEXT", ["False"])
@pytest.mark.parametrize("GPU_DEVICE_IDS", [""])
@pytest.mark.parametrize("DECODING_MODE", [""])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
def test_mistral_v1_multi_models(
    E2E_MODEL_NAME,
    MAX_TOKENS_IN_KV_CACHE,
    MAX_ATTENTION_WINDOW_SIZE,
    BATCH_SCHEDULER_POLICY,
    KV_CACHE_FREE_GPU_MEM_FRACTION,
    ENABLE_TRT_OVERLAP,
    BATCHING_STRATEGY,
    DECOUPLED_MODE,
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
    EXCLUDE_INPUT_IN_OUTPUT,
    inflight_batcher_llm_client_root,
    tensorrt_llm_llama_example_root,
    mistral_v1_tokenizer_model_root,
    llm_backend_venv,
):
    if BATCHING_STRATEGY == "V1" and BATCH_SCHEDULER_POLICY == "max_utilization":
        pytest.skip("Skipping. V1 doesn't support max_utilization.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.path.join(LLM_ROOT, "triton_backend")
    # Build Engine
    ENGINE_PATH = prepare_mistral_v1_7b_engine("ifb",
                                               tensorrt_llm_llama_example_root,
                                               mistral_v1_tokenizer_model_root)

    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)

    # Modify config.pbtxt
    TOKENIZER_PATH = mistral_v1_tokenizer_model_root
    modify_ib_config_pbtxt(
        new_model_repo,
        ENGINE_PATH,
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
    )

    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call((f"python3 {launch_server_py} --force --world_size 1 "
                f"--model_repo={new_model_repo} --multi-model"),
               shell=True)
    check_server_ready()
    # Run Test
    run_cmd = [
        f"{inflight_batcher_llm_client_root}/inflight_batcher_llm_client.py",
        f"--tokenizer-dir={mistral_v1_tokenizer_model_root}",
        "--tokenizer-type=llama",
        "--model-name=tensorrt_llm",
    ]
    if DECOUPLED_MODE == "True":
        run_cmd += [
            "--streaming",
        ]

    venv_check_call(llm_backend_venv, run_cmd)


@pytest.mark.parametrize("TEST_TYPE", ["e2e", "accuracy"])
def test_mistral_v1_7b_python_backend(
    TEST_TYPE,
    llm_backend_gpt_example_root,
    mistral_v1_tokenizer_model_root,
    tensorrt_llm_llama_example_root,
    llm_backend_venv,
):
    llm_backend_repo_root = os.path.join(LLM_ROOT, "triton_backend")
    # Build Engine
    ENGINE_PATH = prepare_mistral_v1_7b_engine("python_backend",
                                               tensorrt_llm_llama_example_root,
                                               mistral_v1_tokenizer_model_root)
    # Prepare model repo
    origin_model_repo = os.path.join(llm_backend_repo_root, "all_models", "gpt")
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    check_call(f"rm -rf {new_model_repo}", shell=True)
    check_call(f"cp -R {origin_model_repo} {new_model_repo}", shell=True)

    # Modify config.pbtxt
    TOKENIZER_PATH = mistral_v1_tokenizer_model_root
    fill_template_py = os.path.join(llm_backend_repo_root, "tools",
                                    "fill_template.py")
    llm_config = os.path.join(llm_backend_repo_root, "triton_repo",
                              "tensorrt_llm", "config.pbtxt")
    preprocessing_config = os.path.join(llm_backend_repo_root, "triton_repo",
                                        "preprocessing", "config.pbtxt")
    postprocessing_config = os.path.join(llm_backend_repo_root, "triton_repo",
                                         "postprocessing", "config.pbtxt")
    check_call(
        f"python3 {fill_template_py} -i {llm_config} engine_dir:{ENGINE_PATH}",
        shell=True)
    check_call(
        f"python3 {fill_template_py} -i {preprocessing_config} tokenizer_dir:{TOKENIZER_PATH}",
        shell=True)
    check_call(
        f"python3 {fill_template_py} -i {postprocessing_config} tokenizer_dir:{TOKENIZER_PATH}",
        shell=True)

    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()
    # Run Test
    if TEST_TYPE == "e2e":
        run_cmd = [
            f"{llm_backend_gpt_example_root}/end_to_end_test.py",
            f"--tokenizer_dir={TOKENIZER_PATH}",
        ]
        venv_check_call(llm_backend_venv, run_cmd)
    elif TEST_TYPE == "accuracy":
        run_cmd = [
            f"{llm_backend_gpt_example_root}/client.py",
            "--text=Born in north-east France, Soyer trained as a",
            "--output_len=10",
            f"--tokenizer_dir={TOKENIZER_PATH}",
        ]

        output = venv_check_output(llm_backend_venv,
                                   run_cmd).strip().split("\n")[-1]

        print_info(output)


@pytest.mark.skip_less_device(8)
@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["False"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", [""])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY",
                         ["max_utilization", "guaranteed_no_evict"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", [""])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY", ["inflight_fused_batching"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["True", "False"],
                         ids=["enableDecoupleMode", "disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["False"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("ENABLE_CHUNKED_CONTEXT", ["False"])
@pytest.mark.parametrize("GPU_DEVICE_IDS", [""])
@pytest.mark.parametrize("DECODING_MODE", [""])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
def test_llama_v2_70b_ifb(
    E2E_MODEL_NAME,
    MAX_TOKENS_IN_KV_CACHE,
    MAX_ATTENTION_WINDOW_SIZE,
    BATCH_SCHEDULER_POLICY,
    KV_CACHE_FREE_GPU_MEM_FRACTION,
    ENABLE_TRT_OVERLAP,
    BATCHING_STRATEGY,
    DECOUPLED_MODE,
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
    EXCLUDE_INPUT_IN_OUTPUT,
    inflight_batcher_llm_client_root,
    tensorrt_llm_llama_example_root,
    llama_v2_tokenizer_model_root,
    llm_backend_venv,
):
    if BATCHING_STRATEGY == "V1" and BATCH_SCHEDULER_POLICY == "max_utilization":
        pytest.skip("Skipping. V1 doesn't support max_utilization.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.path.join(LLM_ROOT, "triton_backend")
    # Build Engine
    ENGINE_PATH = prepare_llama_v2_70b_engine("ifb",
                                              tensorrt_llm_llama_example_root,
                                              llama_v2_tokenizer_model_root)
    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)

    # Modify config.pbtxt
    TOKENIZER_PATH = llama_v2_tokenizer_model_root
    modify_ib_config_pbtxt(
        new_model_repo,
        ENGINE_PATH,
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
    )

    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --world_size=8 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()
    # Run Test
    run_cmd = [
        f"{inflight_batcher_llm_client_root}/inflight_batcher_llm_client.py",
        f"--tokenizer-dir={llama_v2_tokenizer_model_root}",
        "--tokenizer-type=llama",
    ]
    if DECOUPLED_MODE == "True":
        run_cmd += [
            "--streaming",
        ]

    venv_check_call(llm_backend_venv, run_cmd)


@pytest.mark.skip_less_device(8)
@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["False"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", [""])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY",
                         ["max_utilization", "guaranteed_no_evict"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", [""])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY", ["inflight_fused_batching"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["True", "False"],
                         ids=["enableDecoupleMode", "disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["False"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("ENABLE_CHUNKED_CONTEXT", ["False"])
@pytest.mark.parametrize("GPU_DEVICE_IDS", [""])
@pytest.mark.parametrize("DECODING_MODE", ["lookahead"])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
@pytest.mark.parametrize("EXECUTOR_LOOKAHEAD_WINDOW", ["7"])
@pytest.mark.parametrize("EXECUTOR_LOOKAHEAD_NGRAM", ["7"])
@pytest.mark.parametrize("EXECUTOR_LOOKAHEAD_VERIFICATION_SET", ["7"])
def test_llama_v2_70b_ifb_lad(
    E2E_MODEL_NAME,
    MAX_TOKENS_IN_KV_CACHE,
    MAX_ATTENTION_WINDOW_SIZE,
    BATCH_SCHEDULER_POLICY,
    KV_CACHE_FREE_GPU_MEM_FRACTION,
    ENABLE_TRT_OVERLAP,
    BATCHING_STRATEGY,
    DECOUPLED_MODE,
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
    EXCLUDE_INPUT_IN_OUTPUT,
    EXECUTOR_LOOKAHEAD_WINDOW,
    EXECUTOR_LOOKAHEAD_NGRAM,
    EXECUTOR_LOOKAHEAD_VERIFICATION_SET,
    inflight_batcher_llm_client_root,
    tensorrt_llm_llama_example_root,
    llama_v2_tokenizer_model_root,
    llm_backend_venv,
):
    if BATCHING_STRATEGY == "V1" and BATCH_SCHEDULER_POLICY == "max_utilization":
        pytest.skip("Skipping. V1 doesn't support max_utilization.")

    llm_backend_repo_root = os.path.join(LLM_ROOT, "triton_backend")

    # Build Engine
    ENGINE_PATH = prepare_llama_v2_70b_engine("ifb",
                                              tensorrt_llm_llama_example_root,
                                              llama_v2_tokenizer_model_root,
                                              use_lad=True)
    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)

    # Modify config.pbtxt
    TOKENIZER_PATH = llama_v2_tokenizer_model_root
    modify_ib_config_pbtxt(
        new_model_repo,
        ENGINE_PATH,
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
        EXECUTOR_LOOKAHEAD_WINDOW=EXECUTOR_LOOKAHEAD_WINDOW,
        EXECUTOR_LOOKAHEAD_NGRAM=EXECUTOR_LOOKAHEAD_NGRAM,
        EXECUTOR_LOOKAHEAD_VERIFICATION_SET=EXECUTOR_LOOKAHEAD_VERIFICATION_SET,
    )

    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --world_size=8 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()
    # Run Test
    run_cmd = [
        f"{inflight_batcher_llm_client_root}/inflight_batcher_llm_client.py",
        f"--tokenizer-dir={llama_v2_tokenizer_model_root}",
        "--tokenizer-type=llama",
        f"--lookahead_config=[{EXECUTOR_LOOKAHEAD_WINDOW}, {EXECUTOR_LOOKAHEAD_NGRAM}, {EXECUTOR_LOOKAHEAD_VERIFICATION_SET}]"
    ]
    if DECOUPLED_MODE == "True":
        run_cmd += [
            "--streaming",
        ]

    venv_check_call(llm_backend_venv, run_cmd)


@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["False"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", [""])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY",
                         ["max_utilization", "guaranteed_no_evict"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", [""])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY", ["inflight_fused_batching"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["True", "False"],
                         ids=["enableDecoupleMode", "disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["False"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("ENABLE_CHUNKED_CONTEXT", ["False"])
@pytest.mark.parametrize("GPU_DEVICE_IDS", [""])
@pytest.mark.parametrize("DECODING_MODE", ["medusa"])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
def test_medusa_vicuna_7b_ifb(
    E2E_MODEL_NAME,
    MAX_TOKENS_IN_KV_CACHE,
    MAX_ATTENTION_WINDOW_SIZE,
    BATCH_SCHEDULER_POLICY,
    KV_CACHE_FREE_GPU_MEM_FRACTION,
    ENABLE_TRT_OVERLAP,
    BATCHING_STRATEGY,
    DECOUPLED_MODE,
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
    EXCLUDE_INPUT_IN_OUTPUT,
    inflight_batcher_llm_client_root,
    tensorrt_llm_medusa_example_root,
    vicuna_7b_model_root,
    medusa_vicuna_7b_model_root,
    llama_v2_tokenizer_model_root,
    llm_backend_dataset_root,
    llm_backend_venv,
):
    if BATCHING_STRATEGY == "V1" and BATCH_SCHEDULER_POLICY == "max_utilization":
        pytest.skip("Skipping. V1 doesn't support max_utilization.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.path.join(LLM_ROOT, "triton_backend")
    # Build Engine
    ENGINE_PATH = prepare_medusa_vicuna_7b_engine(
        tensorrt_llm_medusa_example_root, vicuna_7b_model_root,
        medusa_vicuna_7b_model_root)
    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)

    # Modify config.pbtxt
    TOKENIZER_PATH = llama_v2_tokenizer_model_root
    modify_ib_config_pbtxt(
        new_model_repo,
        ENGINE_PATH,
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
    )
    # Allow the output of the medusa model to be somewhat different from the output of the base model
    # This is a known issue, because starting medusa may select a different kernel
    correctness_threshold = 0.7

    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()
    # Run Test
    run_cmd = [
        f"{inflight_batcher_llm_client_root}/inflight_batcher_llm_client.py",
        "--request-output-len=128", "--end-id=1284", "--request-id=1",
        f"--tokenizer-dir={llama_v2_tokenizer_model_root}",
        f"--input-tokens-csv={llm_backend_dataset_root}/short_input_end_id_medusa.csv",
        f"--output-tokens-csv={llm_backend_dataset_root}/short_output_end_id_medusa.csv",
        "--check-output", f"--correctness-threshold={correctness_threshold}"
    ]
    if DECOUPLED_MODE == "True":
        run_cmd += [
            "--streaming",
        ]

    venv_check_call(llm_backend_venv, run_cmd)


@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["False"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", [""])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY",
                         ["max_utilization", "guaranteed_no_evict"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", [""])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY", ["inflight_fused_batching"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["True", "False"],
                         ids=["enableDecoupleMode", "disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["False"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("ENABLE_CHUNKED_CONTEXT", ["False"])
@pytest.mark.parametrize("GPU_DEVICE_IDS", [""])
@pytest.mark.parametrize("DECODING_MODE", ["eagle"])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
def test_eagle_vicuna_7b_ifb(
    E2E_MODEL_NAME,
    MAX_TOKENS_IN_KV_CACHE,
    MAX_ATTENTION_WINDOW_SIZE,
    BATCH_SCHEDULER_POLICY,
    KV_CACHE_FREE_GPU_MEM_FRACTION,
    ENABLE_TRT_OVERLAP,
    BATCHING_STRATEGY,
    DECOUPLED_MODE,
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
    EXCLUDE_INPUT_IN_OUTPUT,
    inflight_batcher_llm_client_root,
    tensorrt_llm_eagle_example_root,
    vicuna_7b_model_root,
    eagle_vicuna_7b_model_root,
    llama_v2_tokenizer_model_root,
    llm_backend_dataset_root,
    llm_backend_venv,
):
    if BATCHING_STRATEGY == "V1" and BATCH_SCHEDULER_POLICY == "max_utilization":
        pytest.skip("Skipping. V1 doesn't support max_utilization.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.path.join(LLM_ROOT, "triton_backend")
    # Build Engine
    ENGINE_PATH = prepare_eagle_vicuna_7b_engine(
        tensorrt_llm_eagle_example_root, vicuna_7b_model_root,
        eagle_vicuna_7b_model_root)
    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)

    # Modify config.pbtxt
    TOKENIZER_PATH = llama_v2_tokenizer_model_root
    modify_ib_config_pbtxt(
        new_model_repo,
        ENGINE_PATH,
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
    )
    # Allow the output of the eagle model to be somewhat different from the output of the base model
    # This is a known issue, because starting eagle may select a different kernel
    correctness_threshold = 0.7

    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()
    # Run Test
    run_cmd = [
        f"{inflight_batcher_llm_client_root}/inflight_batcher_llm_client.py",
        "--request-output-len=128",
        "--end-id=1284",
        "--request-id=1",
        f"--tokenizer-dir={llama_v2_tokenizer_model_root}",
        # We use the same i/o as medusa here as eagle is based on the same vicuna-1.3-7b model
        f"--input-tokens-csv={llm_backend_dataset_root}/short_input_end_id_medusa.csv",
        f"--output-tokens-csv={llm_backend_dataset_root}/short_output_end_id_medusa.csv",
        "--check-output",
        f"--correctness-threshold={correctness_threshold}"
    ]
    if DECOUPLED_MODE == "True":
        run_cmd += [
            "--streaming",
        ]

    venv_check_call(llm_backend_venv, run_cmd)


@pytest.mark.parametrize("TEST_TYPE", ["e2e", "accuracy"])
def test_gpt_350m_python_backend(
    TEST_TYPE,
    llm_backend_gpt_example_root,
    tensorrt_llm_gpt_example_root,
    gpt_tokenizer_model_root,
    llm_backend_venv,
):
    llm_backend_repo_root = os.path.join(LLM_ROOT, "triton_backend")
    # Build engine
    ENGINE_PATH = prepare_gpt_350m_engine(
        "python_backend",
        tensorrt_llm_gpt_example_root,
        gpt_tokenizer_model_root,
    )

    # Prepare model repo
    origin_model_repo = os.path.join(llm_backend_repo_root, "all_models", "gpt")
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    check_call(f"rm -rf {new_model_repo}", shell=True)
    check_call(f"cp -R {origin_model_repo} {new_model_repo}", shell=True)

    # Modify config.pbtxt
    TOKENIZER_PATH = gpt_tokenizer_model_root
    fill_template_py = os.path.join(llm_backend_repo_root, "tools",
                                    "fill_template.py")
    llm_config = os.path.join(llm_backend_repo_root, "triton_repo",
                              "tensorrt_llm", "config.pbtxt")
    preprocessing_config = os.path.join(llm_backend_repo_root, "triton_repo",
                                        "preprocessing", "config.pbtxt")
    postprocessing_config = os.path.join(llm_backend_repo_root, "triton_repo",
                                         "postprocessing", "config.pbtxt")
    check_call(
        f"python3 {fill_template_py} -i {llm_config} engine_dir:{ENGINE_PATH}",
        shell=True)
    check_call(
        f"python3 {fill_template_py} -i {preprocessing_config} tokenizer_dir:{TOKENIZER_PATH}",
        shell=True)
    check_call(
        f"python3 {fill_template_py} -i {postprocessing_config} tokenizer_dir:{TOKENIZER_PATH}",
        shell=True)
    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()
    # Run Test
    if TEST_TYPE == "e2e":
        run_cmd = [
            f"{llm_backend_gpt_example_root}/end_to_end_test.py",
            f"--tokenizer_dir={TOKENIZER_PATH}",
        ]
        venv_check_call(llm_backend_venv, run_cmd)
    elif TEST_TYPE == "accuracy":
        run_cmd = [
            f"{llm_backend_gpt_example_root}/client.py",
            "--text=Born in north-east France, Soyer trained as a",
            "--output_len=10",
            f"--tokenizer_dir={TOKENIZER_PATH}",
        ]

        output = venv_check_output(llm_backend_venv,
                                   run_cmd).strip().split("\n")[-1]

        print_info(output)
        check_server_metrics()

        # Validate Accuracy -ToDo


@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble", "tensorrt_llm_bls"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["True", "False"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", [""])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY",
                         ["max_utilization", "guaranteed_no_evict"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", [""])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY", ["inflight_fused_batching"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["True", "False"],
                         ids=["enableDecoupleMode", "disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["False"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("ENABLE_CHUNKED_CONTEXT", ["False"])
@pytest.mark.parametrize("GPU_DEVICE_IDS", [""])
@pytest.mark.parametrize("DECODING_MODE", ["", "top_k_top_p"])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
@pytest.mark.parametrize("FEATURE_NAME", [
    "test_basic", "batched_inputs", "test_log_probs", "test_request_id",
    "test_stop_words", "test_embedding_bias"
])
def test_gpt_350m_ifb(
    E2E_MODEL_NAME,
    FEATURE_NAME,
    MAX_TOKENS_IN_KV_CACHE,
    MAX_ATTENTION_WINDOW_SIZE,
    BATCH_SCHEDULER_POLICY,
    KV_CACHE_FREE_GPU_MEM_FRACTION,
    ENABLE_TRT_OVERLAP,
    BATCHING_STRATEGY,
    DECOUPLED_MODE,
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
    EXCLUDE_INPUT_IN_OUTPUT,
    inflight_batcher_llm_client_root,
    tensorrt_llm_gpt_example_root,
    gpt_tokenizer_model_root,
    llm_backend_venv,
):
    if BATCHING_STRATEGY == "V1" and BATCH_SCHEDULER_POLICY == "max_utilization":
        pytest.skip("Skipping. V1 doesn't support max_utilization.")

    if BATCHING_STRATEGY == "V1" and FEATURE_NAME == "test_embedding_bias":
        pytest.skip("Skipping. V1 doesn't support embedding_bias tensor yet.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.path.join(LLM_ROOT, "triton_backend")
    # Build engine
    ENGINE_PATH = prepare_gpt_350m_engine(
        "ifb",
        tensorrt_llm_gpt_example_root,
        gpt_tokenizer_model_root,
    )
    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)

    # Modify config.pbtxt
    TOKENIZER_PATH = gpt_tokenizer_model_root
    modify_ib_config_pbtxt(
        new_model_repo,
        ENGINE_PATH,
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
        TENSORRT_LLM_TARGET_MODEL_NAME="tensorrt_llm",
        TENSORRT_LLM_DRAFT_MODEL_NAME="",
    )

    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()
    # Run Test
    feature_name = f"{FEATURE_NAME}"
    tokenizer_dir = f"{gpt_tokenizer_model_root}"

    if DECOUPLED_MODE == "False":
        run_cpp_backend_tests(feature_name, llm_backend_venv,
                              inflight_batcher_llm_client_root, tokenizer_dir)
    else:
        test_model_name = ""
        if ACCUMULATE_TOKEN == "True" and E2E_MODEL_NAME == "tensorrt_llm_bls":
            test_model_name = "gpt_350m"

        run_cpp_streaming_backend_tests(feature_name,
                                        llm_backend_venv,
                                        inflight_batcher_llm_client_root,
                                        tokenizer_dir,
                                        model_name=test_model_name,
                                        e2e_model=E2E_MODEL_NAME)

    if feature_name == "test_basic":
        check_server_metrics(batching_strategy=BATCHING_STRATEGY,
                             kv_cache_reuse=ENABLE_KV_CACHE_REUSE)


@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble", "tensorrt_llm_bls"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["True", "False"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", ["4096"])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", [""])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY", ["guaranteed_no_evict"])
@pytest.mark.parametrize("CROSS_KV_CACHE_FRACTION", [""])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", [""])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY", ["inflight_fused_batching"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["False"],
                         ids=["disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["False"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("ENABLE_CHUNKED_CONTEXT", ["False"])
@pytest.mark.parametrize("GPU_DEVICE_IDS", [""])
@pytest.mark.parametrize("DECODING_MODE", ["", "top_k_top_p"])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["True", "False"])
@pytest.mark.parametrize("FEATURE_NAME", ["test_basic"])
def test_t5_small_enc_dec_ifb(
    E2E_MODEL_NAME,
    FEATURE_NAME,
    MAX_TOKENS_IN_KV_CACHE,
    MAX_ATTENTION_WINDOW_SIZE,
    BATCH_SCHEDULER_POLICY,
    KV_CACHE_FREE_GPU_MEM_FRACTION,
    CROSS_KV_CACHE_FRACTION,
    ENABLE_TRT_OVERLAP,
    BATCHING_STRATEGY,
    DECOUPLED_MODE,
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
    EXCLUDE_INPUT_IN_OUTPUT,
    inflight_batcher_llm_client_root,
    tensorrt_llm_enc_dec_example_root,
    t5_small_model_root,
    llm_backend_venv,
):
    if BATCHING_STRATEGY == "V1" and BATCH_SCHEDULER_POLICY == "max_utilization":
        pytest.skip("Skipping. V1 doesn't support max_utilization.")

    if BATCHING_STRATEGY == "V1" and FEATURE_NAME == "test_embedding_bias":
        pytest.skip("Skipping. V1 doesn't support embedding_bias tensor yet.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.path.join(LLM_ROOT, "triton_backend")
    # Build engine
    ENCODER_ENGINE_DIR, ENGINE_DIR = prepare_t5_small_engine(
        tensorrt_llm_enc_dec_example_root, t5_small_model_root)
    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)

    # Modify config.pbtxt
    TOKENIZER_PATH = t5_small_model_root
    if CROSS_KV_CACHE_FRACTION == "":
        CROSS_KV_CACHE_FRACTION = "0.5"
    modify_ib_config_pbtxt(
        new_model_repo,
        ENGINE_DIR,
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
        ENCODER_ENGINE_PATH=ENCODER_ENGINE_DIR,
        CROSS_KV_CACHE_FRACTION=CROSS_KV_CACHE_FRACTION,
    )

    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()
    # Run Test
    feature_name = f"{FEATURE_NAME}"

    if DECOUPLED_MODE == "False":
        run_cpp_backend_tests(feature_name, llm_backend_venv,
                              inflight_batcher_llm_client_root, TOKENIZER_PATH)
    else:
        run_cpp_streaming_backend_tests(feature_name, llm_backend_venv,
                                        inflight_batcher_llm_client_root,
                                        TOKENIZER_PATH)


@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["True", "False"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", ["24000"])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", [""])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY", ["guaranteed_no_evict"])
@pytest.mark.parametrize("CROSS_KV_CACHE_FRACTION", ["0.5"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", ["0.2"])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY", ["inflight_fused_batching"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["True", "False"],
                         ids=["enableDecoupleMode", "disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["False"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("ENABLE_CHUNKED_CONTEXT", ["False"])
@pytest.mark.parametrize("GPU_DEVICE_IDS", [""])
@pytest.mark.parametrize("DECODING_MODE", ["top_k_top_p"])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["True"])
def test_whisper_large_v3_ifb(
    E2E_MODEL_NAME,
    MAX_TOKENS_IN_KV_CACHE,
    MAX_ATTENTION_WINDOW_SIZE,
    BATCH_SCHEDULER_POLICY,
    KV_CACHE_FREE_GPU_MEM_FRACTION,
    CROSS_KV_CACHE_FRACTION,
    ENABLE_TRT_OVERLAP,
    BATCHING_STRATEGY,
    DECOUPLED_MODE,
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
    EXCLUDE_INPUT_IN_OUTPUT,
    llm_backend_whisper_example_root,
    tensorrt_llm_whisper_example_root,
    whisper_large_model_root,
    llm_backend_venv,
):
    if BATCHING_STRATEGY == "V1" and BATCH_SCHEDULER_POLICY == "max_utilization":
        pytest.skip("Skipping. V1 doesn't support max_utilization.")

    if BATCHING_STRATEGY == "V1" and FEATURE_NAME == "test_embedding_bias":
        pytest.skip("Skipping. V1 doesn't support embedding_bias tensor yet.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.path.join(LLM_ROOT, "triton_backend")
    # Build engine
    ENCODER_ENGINE_DIR, ENGINE_DIR = prepare_whisper_large_engine(
        tensorrt_llm_whisper_example_root, whisper_large_model_root)
    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root,
                          new_model_repo,
                          model_name="whisper")

    # Modify config.pbtxt
    TOKENIZER_PATH = whisper_large_model_root

    modify_ib_config_pbtxt(
        new_model_repo,
        ENGINE_DIR,
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
        ENCODER_ENGINE_PATH=ENCODER_ENGINE_DIR,
        CROSS_KV_CACHE_FRACTION=CROSS_KV_CACHE_FRACTION,
    )

    #####Whisper Specific#####
    # Delete useless triton repo
    check_call(f"rm -rf {new_model_repo}/preprocessing", shell=True)
    check_call(f"rm -rf {new_model_repo}/postprocessing", shell=True)
    check_call(f"rm -rf {new_model_repo}/ensemble", shell=True)
    check_call(f"rm -rf {new_model_repo}/tensorrt_llm_bls", shell=True)

    # Copy tiktoken and npz to triton repo
    check_call(
        f"cp -vf {whisper_large_model_root}/multilingual.tiktoken {new_model_repo}/whisper_bls/1",
        shell=True)
    check_call(
        f"cp -vf {whisper_large_model_root}/mel_filters.npz {new_model_repo}/whisper_bls/1",
        shell=True)

    # Install 3rd party libs
    check_call(f"pip3 install tiktoken soundfile", shell=True)
    ##########################

    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()
    # Run Test
    run_cmd = [
        f"{llm_backend_whisper_example_root}/client.py",
        f"--audio-path={whisper_large_model_root}/1221-135766-0002.wav",
    ]
    if DECOUPLED_MODE == "True":
        run_cmd += [
            "--streaming",
        ]
    venv_check_call(llm_backend_venv, run_cmd)


@pytest.mark.parametrize("TEST_TYPE", ["e2e", "client"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["True", "False"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", [""])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY",
                         ["max_utilization", "guaranteed_no_evict"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", ["0.2"])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY", ["inflight_fused_batching"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["False"],
                         ids=["disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["False"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("ENABLE_CHUNKED_CONTEXT", ["False"])
@pytest.mark.parametrize("GPU_DEVICE_IDS", [""])
@pytest.mark.parametrize("DECODING_MODE", [""])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
def test_gpt_gather_logits_ifb(
    TEST_TYPE,
    MAX_TOKENS_IN_KV_CACHE,
    MAX_ATTENTION_WINDOW_SIZE,
    BATCH_SCHEDULER_POLICY,
    KV_CACHE_FREE_GPU_MEM_FRACTION,
    ENABLE_TRT_OVERLAP,
    BATCHING_STRATEGY,
    DECOUPLED_MODE,
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
    EXCLUDE_INPUT_IN_OUTPUT,
    inflight_batcher_llm_client_root,
    llm_backend_inflight_batcher_llm_root,
    llm_backend_dataset_root,
    tensorrt_llm_gpt_example_root,
    gpt_tokenizer_model_root,
    llm_backend_venv,
):
    if BATCHING_STRATEGY == "V1" and BATCH_SCHEDULER_POLICY == "max_utilization":
        pytest.skip("Skipping. V1 doesn't support max_utilization.")

    llm_backend_repo_root = os.path.join(LLM_ROOT, "triton_backend")
    # Build engine
    ENGINE_PATH = prepare_gpt_gather_logits_engine(
        "ifb",
        tensorrt_llm_gpt_example_root,
        gpt_tokenizer_model_root,
    )
    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)

    # Modify config.pbtxt
    TOKENIZER_PATH = gpt_tokenizer_model_root
    modify_ib_config_pbtxt(
        new_model_repo,
        ENGINE_PATH,
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
    )

    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    print(
        f"Launching Triton Server with command: {launch_server_py} --world_size=1 --model_repo={new_model_repo}"
    )
    check_call(
        f"python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()
    # Run Test
    if TEST_TYPE == "client":
        run_cmd = [
            f"{inflight_batcher_llm_client_root}/inflight_batcher_llm_client.py",
            f"--tokenizer-dir={gpt_tokenizer_model_root}",
            "--return-context-logits", "--return-generation-logits"
        ]
    elif TEST_TYPE == "e2e":
        run_cmd = [
            f"{llm_backend_inflight_batcher_llm_root}/end_to_end_test.py",
            "-i=http",
            "--max-input-len=192",
            f"--dataset={llm_backend_dataset_root}/mini_cnn_eval.json",
        ]

    venv_check_call(llm_backend_venv, run_cmd)


@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["False"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", [""])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY",
                         ["max_utilization", "guaranteed_no_evict"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", ["0.2"])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY", ["inflight_fused_batching"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["False"],
                         ids=["disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["True"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("ENABLE_CHUNKED_CONTEXT", ["False"])
@pytest.mark.parametrize("GPU_DEVICE_IDS", [""])
@pytest.mark.parametrize("DECODING_MODE", [""])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
def test_gpt_350m_speculative_decoding(
    E2E_MODEL_NAME,
    MAX_TOKENS_IN_KV_CACHE,
    MAX_ATTENTION_WINDOW_SIZE,
    BATCH_SCHEDULER_POLICY,
    KV_CACHE_FREE_GPU_MEM_FRACTION,
    ENABLE_TRT_OVERLAP,
    BATCHING_STRATEGY,
    DECOUPLED_MODE,
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
    EXCLUDE_INPUT_IN_OUTPUT,
    tensorrt_llm_gpt_example_root,
    gpt_tokenizer_model_root,
    gpt2_medium_tokenizer_model_root,
    llm_backend_inflight_batcher_llm_root,
    llm_backend_dataset_root,
    llm_backend_venv,
):
    if BATCHING_STRATEGY == "V1":
        pytest.skip("Skipping. Speculative decoding is not supported in V1.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.path.join(LLM_ROOT, "triton_backend")
    # Build engine
    CONTROL_ENGINE_DIR = prepare_gpt_350m_engine(
        "medium_control_ifb",
        tensorrt_llm_gpt_example_root,
        gpt2_medium_tokenizer_model_root,
    )
    TARGET_ENGINE_DIR = prepare_gpt_350m_engine(
        "medium_target_ifb",
        tensorrt_llm_gpt_example_root,
        gpt2_medium_tokenizer_model_root,
    )
    DRAFT_ENGINE_DIR = prepare_gpt_350m_engine(
        "ifb",
        tensorrt_llm_gpt_example_root,
        gpt_tokenizer_model_root,
    )
    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)
    prepare_custom_config(llm_backend_repo_root, new_model_repo,
                          "tensorrt_llm_draft")
    prepare_custom_config(llm_backend_repo_root, new_model_repo,
                          "tensorrt_llm_target")

    # Modify config.pbtxt
    ENABLE_KV_CACHE_REUSE = "True"
    TOKENIZER_PATH = gpt_tokenizer_model_root
    modify_ib_config_pbtxt(
        new_model_repo,
        CONTROL_ENGINE_DIR,
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
        DRAFT_ENGINE_PATH=DRAFT_ENGINE_DIR,
        TARGET_ENGINE_PATH=TARGET_ENGINE_DIR,
    )

    # Launch First server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready(http_port="8000")

    ## second suit
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)
    prepare_custom_config(llm_backend_repo_root, new_model_repo,
                          "tensorrt_llm_draft")
    prepare_custom_config(llm_backend_repo_root, new_model_repo,
                          "tensorrt_llm_target")

    ENABLE_KV_CACHE_REUSE = "False"

    modify_ib_config_pbtxt(
        new_model_repo,
        CONTROL_ENGINE_DIR,
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
        DRAFT_ENGINE_PATH=DRAFT_ENGINE_DIR,
        TARGET_ENGINE_PATH=TARGET_ENGINE_DIR,
    )

    ## Launch second server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo} " \
        f"--grpc_port=8004 --http_port=8003 --metrics_port=8005",
        shell=True)
    check_server_ready(http_port="8003")

    # Run Test
    TENSORRT_LLM_DRAFT_MODEL_NAME = "tensorrt_llm_draft"
    TENSORRT_LLM_TARGET_MODEL_NAME = "tensorrt_llm_target"

    run_cmd = [
        f"{llm_backend_inflight_batcher_llm_root}/speculative_decoding_test.py",
        "--max-input-len=200",
        f"--dataset={llm_backend_dataset_root}/mini_cnn_eval_spec_decoding.json",
        "--url-draft=0.0.0.0:8004",
        "--url-target=0.0.0.0:8001",
        "--url-control=0.0.0.0:8001",
        f"--draft-tensorrt-llm-model-name={TENSORRT_LLM_DRAFT_MODEL_NAME}",
        f"--target-tensorrt-llm-model-name={TENSORRT_LLM_TARGET_MODEL_NAME}",
    ]

    venv_check_call(llm_backend_venv, run_cmd)


@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["False"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", [""])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY",
                         ["max_utilization", "guaranteed_no_evict"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", ["0.2"])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY", ["inflight_fused_batching"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["False"],
                         ids=["disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["True"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("ENABLE_CHUNKED_CONTEXT", ["False"])
@pytest.mark.parametrize("GPU_DEVICE_IDS", [""])
@pytest.mark.parametrize("DECODING_MODE", [""])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
def test_gpt_350m_speculative_decoding_return_logits(
    E2E_MODEL_NAME,
    MAX_TOKENS_IN_KV_CACHE,
    MAX_ATTENTION_WINDOW_SIZE,
    BATCH_SCHEDULER_POLICY,
    KV_CACHE_FREE_GPU_MEM_FRACTION,
    ENABLE_TRT_OVERLAP,
    BATCHING_STRATEGY,
    DECOUPLED_MODE,
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
    EXCLUDE_INPUT_IN_OUTPUT,
    tensorrt_llm_gpt_example_root,
    gpt_tokenizer_model_root,
    gpt2_medium_tokenizer_model_root,
    llm_backend_inflight_batcher_llm_root,
    llm_backend_dataset_root,
    llm_backend_venv,
):
    if BATCHING_STRATEGY == "V1":
        pytest.skip("Skipping. Speculative decoding is not supported in V1.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.path.join(LLM_ROOT, "triton_backend")
    # Build engine
    CONTROL_ENGINE_DIR = prepare_gpt_350m_engine(
        "medium_control_ifb",
        tensorrt_llm_gpt_example_root,
        gpt2_medium_tokenizer_model_root,
    )
    TARGET_ENGINE_DIR = prepare_gpt_350m_engine(
        "medium_target_ifb",
        tensorrt_llm_gpt_example_root,
        gpt2_medium_tokenizer_model_root,
    )
    DRAFT_ENGINE_DIR = prepare_gpt_350m_engine(
        "ifb",
        tensorrt_llm_gpt_example_root,
        gpt_tokenizer_model_root,
    )
    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)
    prepare_custom_config(llm_backend_repo_root, new_model_repo,
                          "tensorrt_llm_draft")
    prepare_custom_config(llm_backend_repo_root, new_model_repo,
                          "tensorrt_llm_target")

    # Modify config.pbtxt
    ENABLE_KV_CACHE_REUSE = "True"
    TOKENIZER_PATH = gpt_tokenizer_model_root
    modify_ib_config_pbtxt(
        new_model_repo,
        CONTROL_ENGINE_DIR,
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
        DRAFT_ENGINE_PATH=DRAFT_ENGINE_DIR,
        TARGET_ENGINE_PATH=TARGET_ENGINE_DIR,
    )

    # Launch First server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready(http_port="8000")

    ## second suit
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)
    prepare_custom_config(llm_backend_repo_root, new_model_repo,
                          "tensorrt_llm_draft")
    prepare_custom_config(llm_backend_repo_root, new_model_repo,
                          "tensorrt_llm_target")

    ENABLE_KV_CACHE_REUSE = "False"

    modify_ib_config_pbtxt(
        new_model_repo,
        CONTROL_ENGINE_DIR,
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
        DRAFT_ENGINE_PATH=DRAFT_ENGINE_DIR,
        TARGET_ENGINE_PATH=TARGET_ENGINE_DIR,
    )

    ## Launch second server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo} " \
        f"--grpc_port=8004 --http_port=8003 --metrics_port=8005",
        shell=True)
    check_server_ready(http_port="8003")
    # Run Test
    TENSORRT_LLM_DRAFT_MODEL_NAME = "tensorrt_llm_draft"
    TENSORRT_LLM_TARGET_MODEL_NAME = "tensorrt_llm_target"
    run_cmd = [
        f"{llm_backend_inflight_batcher_llm_root}/speculative_decoding_test.py",
        "--max-input-len=128",
        f"--dataset={llm_backend_dataset_root}/mini_cnn_eval_spec_decoding.json",
        "--url-draft=0.0.0.0:8004",
        "--url-target=0.0.0.0:8001",
        "--url-control=0.0.0.0:8001",
        "--num-draft-tokens=5",
        "--return-target-model-accepted-token-logits",
        "--return-draft-model-draft-logits",
        "--verbose",
        f"--draft-tensorrt-llm-model-name={TENSORRT_LLM_DRAFT_MODEL_NAME}",
        f"--target-tensorrt-llm-model-name={TENSORRT_LLM_TARGET_MODEL_NAME}",
    ]

    venv_check_call(llm_backend_venv, run_cmd)


@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["False"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", [""])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY",
                         ["guaranteed_no_evict", "max_utilization"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", ["0.2"])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY", ["inflight_fused_batching"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["False"],
                         ids=["disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["True"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("ENABLE_CHUNKED_CONTEXT", ["False"])
@pytest.mark.parametrize("GPU_DEVICE_IDS", [""])
@pytest.mark.parametrize("DECODING_MODE", [""])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
@pytest.mark.parametrize("USE_DRAFT_LOGITS_VALUES", ["True", "False"])
def test_gpt_speculative_decoding_bls(
    E2E_MODEL_NAME,
    MAX_TOKENS_IN_KV_CACHE,
    MAX_ATTENTION_WINDOW_SIZE,
    BATCH_SCHEDULER_POLICY,
    KV_CACHE_FREE_GPU_MEM_FRACTION,
    ENABLE_TRT_OVERLAP,
    BATCHING_STRATEGY,
    DECOUPLED_MODE,
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
    EXCLUDE_INPUT_IN_OUTPUT,
    USE_DRAFT_LOGITS_VALUES,
    tensorrt_llm_gpt_example_root,
    gpt_tokenizer_model_root,
    gpt2_medium_tokenizer_model_root,
    llm_backend_inflight_batcher_llm_root,
    llm_backend_dataset_root,
    llm_backend_venv,
):
    if BATCHING_STRATEGY == "V1":
        pytest.skip("Skipping. Speculative decoding is not supported in V1.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.path.join(LLM_ROOT, "triton_backend")
    # Build engine
    CONTROL_ENGINE_DIR = prepare_gpt_350m_engine(
        "medium_control_ifb",
        tensorrt_llm_gpt_example_root,
        gpt2_medium_tokenizer_model_root,
    )
    TARGET_ENGINE_DIR = prepare_gpt_350m_engine(
        "medium_target_ifb",
        tensorrt_llm_gpt_example_root,
        gpt2_medium_tokenizer_model_root,
    )
    DRAFT_ENGINE_DIR = prepare_gpt_350m_engine(
        "ifb",
        tensorrt_llm_gpt_example_root,
        gpt_tokenizer_model_root,
    )
    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)
    prepare_custom_config(llm_backend_repo_root, new_model_repo,
                          "tensorrt_llm_draft")
    prepare_custom_config(llm_backend_repo_root, new_model_repo,
                          "tensorrt_llm_target")

    # Modify config.pbtxt
    ENABLE_KV_CACHE_REUSE = "True"
    TOKENIZER_PATH = gpt_tokenizer_model_root
    modify_ib_config_pbtxt(
        new_model_repo,
        CONTROL_ENGINE_DIR,
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
        DRAFT_ENGINE_PATH=DRAFT_ENGINE_DIR,
        TARGET_ENGINE_PATH=TARGET_ENGINE_DIR,
    )

    # Launch Triton server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready(http_port="8000")

    # Run Test
    TENSORRT_LLM_DRAFT_MODEL_NAME = "tensorrt_llm_draft"
    TENSORRT_LLM_TARGET_MODEL_NAME = "tensorrt_llm_target"
    run_cmd = [
        f"{llm_backend_inflight_batcher_llm_root}/speculative_decoding_test.py",
        "--max-input-len=200",
        f"--dataset={llm_backend_dataset_root}/mini_cnn_eval_spec_decoding.json",
        "--url-target=0.0.0.0:8001",
        "--url-draft=0.0.0.0:8001",
        "--url-control=0.0.0.0:8001",
        f"--draft-tensorrt-llm-model-name={TENSORRT_LLM_DRAFT_MODEL_NAME}",
        f"--target-tensorrt-llm-model-name={TENSORRT_LLM_TARGET_MODEL_NAME}",
        "--bls-speculative-tensorrt-llm-model-name=tensorrt_llm_bls",
        "--execute-bls-speculative-decoding",
        "--num-draft-tokens=5",
        "--verbose",
    ]

    if USE_DRAFT_LOGITS_VALUES == "True":
        run_cmd += [
            "--return-generation-logits",
            "--use-draft-logits",
            "--disable-output-comparison",
        ]
    venv_check_call(llm_backend_venv, run_cmd)


@pytest.mark.skip_less_device(8)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["False"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", [""])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY", ["guaranteed_no_evict"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", ["0.2"])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY", ["inflight_fused_batching"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["False"],
                         ids=["disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["True"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("ENABLE_CHUNKED_CONTEXT", ["False"])
@pytest.mark.parametrize("GPU_DEVICE_IDS", [""])
@pytest.mark.parametrize("DECODING_MODE", [""])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
@pytest.mark.parametrize("USE_DRAFT_LOGITS_VALUES", ["True", "False"])
@pytest.mark.parametrize("DATA_TYPE", ["fp8", "bfloat16"])
def test_llama_v3_speculative_decoding_bls(
    E2E_MODEL_NAME,
    MAX_TOKENS_IN_KV_CACHE,
    MAX_ATTENTION_WINDOW_SIZE,
    BATCH_SCHEDULER_POLICY,
    KV_CACHE_FREE_GPU_MEM_FRACTION,
    ENABLE_TRT_OVERLAP,
    BATCHING_STRATEGY,
    DECOUPLED_MODE,
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
    EXCLUDE_INPUT_IN_OUTPUT,
    USE_DRAFT_LOGITS_VALUES,
    DATA_TYPE,
    tensorrt_llm_llama_example_root,
    llama_v3_8b_model_root,
    llama_v3_70b_model_root,
    tensorrt_llm_example_root,
    llm_backend_inflight_batcher_llm_root,
    llm_backend_dataset_root,
    llm_backend_venv,
):
    if DATA_TYPE == "fp8" and getSMVersion() < 89:
        pytest.skip("Skipping fp8 test on pre-Ada architecture")

    if BATCHING_STRATEGY == "V1":
        pytest.skip("Skipping. Speculative decoding is not supported in V1.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.path.join(LLM_ROOT, "triton_backend")
    # Build engine
    DRAFT_ENGINE_DIR = prepare_llama_v3_8b_engine(
        tensorrt_llm_example_root,
        tensorrt_llm_llama_example_root,
        llama_v3_8b_model_root,
        data_type=DATA_TYPE)
    CONTROL_ENGINE_DIR = prepare_llama_v3_70b_engine(
        "control_ifb",
        tensorrt_llm_example_root,
        tensorrt_llm_llama_example_root,
        llama_v3_70b_model_root,
        data_type=DATA_TYPE)
    TARGET_ENGINE_DIR = prepare_llama_v3_70b_engine(
        "target_ifb",
        tensorrt_llm_example_root,
        tensorrt_llm_llama_example_root,
        llama_v3_70b_model_root,
        data_type=DATA_TYPE)

    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)
    prepare_custom_config(llm_backend_repo_root, new_model_repo,
                          "tensorrt_llm_draft")
    prepare_custom_config(llm_backend_repo_root, new_model_repo,
                          "tensorrt_llm_target")

    # Modify config.pbtxt
    ENABLE_KV_CACHE_REUSE = "True"
    PARTICIPANT_IDS_DRAFT = "1\\,2\\,3\\,4\\,5\\,6\\,7\\,8"
    PARTICIPANT_IDS_TARGET = "9\\,10\\,11\\,12\\,13\\,14\\,15\\,16"
    PARTICIPANT_IDS = "17\\,18\\,19\\,20\\,21\\,22\\,23\\,24"
    SPEC_DEC_FAST_LOGITS = "1"
    TOKENIZER_PATH = llama_v3_8b_model_root
    modify_ib_config_pbtxt(
        new_model_repo,
        CONTROL_ENGINE_DIR,
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
        DRAFT_ENGINE_PATH=DRAFT_ENGINE_DIR,
        TARGET_ENGINE_PATH=TARGET_ENGINE_DIR,
        PARTICIPANT_IDS_DRAFT=PARTICIPANT_IDS_DRAFT,
        PARTICIPANT_IDS_TARGET=PARTICIPANT_IDS_TARGET,
        PARTICIPANT_IDS=PARTICIPANT_IDS,
        SPEC_DEC_FAST_LOGITS=SPEC_DEC_FAST_LOGITS,
    )

    # Launch Triton server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    model_names = "tensorrt_llm,tensorrt_llm_draft,tensorrt_llm_target"
    check_call(
        f"python3 {launch_server_py} --model_repo={new_model_repo} --tensorrt_llm_model_name {model_names} --multi-model --disable-spawn-processes --world_size=25",
        shell=True)
    check_server_ready(http_port="8000")

    # Run Test
    TENSORRT_LLM_DRAFT_MODEL_NAME = "tensorrt_llm_draft"
    TENSORRT_LLM_TARGET_MODEL_NAME = "tensorrt_llm_target"
    run_cmd = [
        f"{llm_backend_inflight_batcher_llm_root}/speculative_decoding_test.py",
        "--max-input-len=200",
        f"--dataset={llm_backend_dataset_root}/mini_cnn_eval_spec_decoding.json",
        "--url-target=0.0.0.0:8001",
        "--url-draft=0.0.0.0:8001",
        "--url-control=0.0.0.0:8001",
        f"--draft-tensorrt-llm-model-name={TENSORRT_LLM_DRAFT_MODEL_NAME}",
        f"--target-tensorrt-llm-model-name={TENSORRT_LLM_TARGET_MODEL_NAME}",
        "--bls-speculative-tensorrt-llm-model-name=tensorrt_llm_bls",
        "--execute-bls-speculative-decoding",
        "--num-draft-tokens=5",
        "--disable-output-comparison",
        "--verbose",
    ]

    if USE_DRAFT_LOGITS_VALUES == "True":
        run_cmd += [
            "--return-generation-logits",
            "--use-draft-logits",
            "--disable-output-comparison",
        ]
    venv_check_call(llm_backend_venv, run_cmd)


@pytest.mark.skip_less_device(8)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["False"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", [""])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY",
                         ["max_utilization", "guaranteed_no_evict"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", [""])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY", ["inflight_fused_batching"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["True", "False"],
                         ids=["enableDecoupleMode", "disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["False"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("ENABLE_CHUNKED_CONTEXT", ["False"])
@pytest.mark.parametrize("GPU_DEVICE_IDS", [""])
@pytest.mark.parametrize("DECODING_MODE", [""])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
def test_gpt_175b_dummyWeights_ifb(
    E2E_MODEL_NAME,
    MAX_TOKENS_IN_KV_CACHE,
    MAX_ATTENTION_WINDOW_SIZE,
    BATCH_SCHEDULER_POLICY,
    KV_CACHE_FREE_GPU_MEM_FRACTION,
    ENABLE_TRT_OVERLAP,
    BATCHING_STRATEGY,
    DECOUPLED_MODE,
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
    EXCLUDE_INPUT_IN_OUTPUT,
    inflight_batcher_llm_client_root,
    tensorrt_llm_gpt_example_root,
    tensorrt_llm_example_root,
    gpt_tokenizer_model_root,
    llm_backend_venv,
):
    if BATCHING_STRATEGY == "V1" and BATCH_SCHEDULER_POLICY == "max_utilization":
        pytest.skip("Skipping. V1 doesn't support max_utilization.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.path.join(LLM_ROOT, "triton_backend")
    # Build Engine
    ENGINE_PATH = prepare_gpt_175b_engine("ifb", tensorrt_llm_gpt_example_root,
                                          tensorrt_llm_example_root)
    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)

    # Modify config.pbtxt
    TOKENIZER_PATH = gpt_tokenizer_model_root
    modify_ib_config_pbtxt(
        new_model_repo,
        ENGINE_PATH,
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
    )

    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --world_size=8 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()
    # Run Test
    run_cmd = [
        f"{inflight_batcher_llm_client_root}/inflight_batcher_llm_client.py",
        f"--tokenizer-dir={gpt_tokenizer_model_root}",
        "--tokenizer-type=auto",
    ]
    if DECOUPLED_MODE == "True":
        run_cmd += [
            "--streaming",
        ]

    venv_check_call(llm_backend_venv, run_cmd)


@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble", "tensorrt_llm_bls"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["False"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", [""])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY",
                         ["max_utilization", "guaranteed_no_evict"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", ["0.7"])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY", ["inflight_fused_batching"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["True", "False"],
                         ids=["enableDecoupleMode", "disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["False"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("ENABLE_CHUNKED_CONTEXT", ["False"])
@pytest.mark.parametrize("GPU_DEVICE_IDS", [""])
@pytest.mark.parametrize("DECODING_MODE", [""])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
def test_llava(
    E2E_MODEL_NAME,
    MAX_TOKENS_IN_KV_CACHE,
    MAX_ATTENTION_WINDOW_SIZE,
    BATCH_SCHEDULER_POLICY,
    KV_CACHE_FREE_GPU_MEM_FRACTION,
    ENABLE_TRT_OVERLAP,
    BATCHING_STRATEGY,
    DECOUPLED_MODE,
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
    EXCLUDE_INPUT_IN_OUTPUT,
    tensorrt_llm_multimodal_example_root,
    tensorrt_llm_llama_example_root,
    llava_model_root,
    llm_backend_multimodal_example_root,
    llm_backend_venv,
):
    if BATCHING_STRATEGY == "V1" and BATCH_SCHEDULER_POLICY == "max_utilization":
        pytest.skip("Skipping. V1 doesn't support max_utilization.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.path.join(LLM_ROOT, "triton_backend")
    # Build Engine
    ENGINE_PATH, MULTIMODAL_ENGINE_DIR = prepare_llava_engine(
        tensorrt_llm_multimodal_example_root, tensorrt_llm_llama_example_root,
        llava_model_root)
    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)

    # Prepare multimodal specific repo
    prepare_multimodal_model_repo(llm_backend_repo_root, new_model_repo,
                                  "ensemble")
    prepare_multimodal_model_repo(llm_backend_repo_root, new_model_repo,
                                  "multimodal_encoders")

    # Modify config.pbtxt
    TOKENIZER_PATH = llava_model_root
    modify_ib_config_pbtxt(
        new_model_repo,
        ENGINE_PATH,
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
        MULTIMODAL_ENGINE_PATH=MULTIMODAL_ENGINE_DIR,
    )

    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")

    # NOTE
    # Due to mpi init error, manually set PMIX_MCA_gds=hash (ref: https://github.com/open-mpi/ompi/issues/6981)
    check_call(
        f"PMIX_MCA_gds=hash python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo} --exit_timeout=300",
        shell=True)
    check_server_ready()
    # Run Test
    run_cmd = [
        f"{llm_backend_multimodal_example_root}/client.py",
        "--model_type=llava",
        f"--hf_model_dir={llava_model_root}",
    ]
    if DECOUPLED_MODE == "True":
        run_cmd += [
            "--streaming",
        ]

        if E2E_MODEL_NAME == "tensorrt_llm_bls":
            run_cmd += [
                "--use_bls",
            ]

    venv_check_call(llm_backend_venv, run_cmd)


@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble", "tensorrt_llm_bls"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["False"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", [""])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY",
                         ["max_utilization", "guaranteed_no_evict"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", ["0.2"])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY", ["inflight_fused_batching"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["True", "False"],
                         ids=["enableDecoupleMode", "disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["False"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("ENABLE_CHUNKED_CONTEXT", ["False"])
@pytest.mark.parametrize("GPU_DEVICE_IDS", [""])
@pytest.mark.parametrize("DECODING_MODE", [""])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
@pytest.mark.parametrize("FEATURE_NAME", ["test_basic", "test_video"])
def test_llava_onevision(
    E2E_MODEL_NAME,
    MAX_TOKENS_IN_KV_CACHE,
    MAX_ATTENTION_WINDOW_SIZE,
    BATCH_SCHEDULER_POLICY,
    KV_CACHE_FREE_GPU_MEM_FRACTION,
    ENABLE_TRT_OVERLAP,
    BATCHING_STRATEGY,
    DECOUPLED_MODE,
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
    EXCLUDE_INPUT_IN_OUTPUT,
    FEATURE_NAME,
    tensorrt_llm_multimodal_example_root,
    tensorrt_llm_qwen_example_root,
    llava_onevision_model_root,
    llm_backend_all_models_root,
    llm_backend_multimodal_example_root,
    test_video_root,
    llm_backend_venv,
):
    if BATCHING_STRATEGY == "V1" and BATCH_SCHEDULER_POLICY == "max_utilization":
        pytest.skip("Skipping. V1 doesn't support max_utilization.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.path.join(LLM_ROOT, "triton_backend")
    # Build Engine
    ENGINE_PATH, MULTIMODAL_ENGINE_DIR = prepare_llava_onevision_engine(
        tensorrt_llm_multimodal_example_root, tensorrt_llm_qwen_example_root,
        llava_onevision_model_root, llm_backend_all_models_root)
    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)

    # Prepare multimodal specific repo
    prepare_multimodal_model_repo(llm_backend_repo_root, new_model_repo,
                                  "ensemble")
    prepare_multimodal_model_repo(llm_backend_repo_root, new_model_repo,
                                  "multimodal_encoders")

    # Modify config.pbtxt
    TOKENIZER_PATH = llava_onevision_model_root
    modify_ib_config_pbtxt(
        new_model_repo,
        ENGINE_PATH,
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
        MULTIMODAL_ENGINE_PATH=MULTIMODAL_ENGINE_DIR,
    )

    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")

    # NOTE
    # Due to mpi init error, manually set PMIX_MCA_gds=hash (ref: https://github.com/open-mpi/ompi/issues/6981)
    check_call(
        f"PMIX_MCA_gds=hash python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()
    # Run Test
    if FEATURE_NAME == "test_basic":
        keyword = "singapore"
        run_cmd = [
            f"{llm_backend_multimodal_example_root}/client.py",
            "--model_type=llava_onevision",
            "--end-id=151645",
            "--pad-id=151643",
        ]
        if DECOUPLED_MODE == "True":
            keyword = "sing"
            run_cmd += [
                "--streaming",
            ]
    elif FEATURE_NAME == "test_video":
        keyword = "robotic"
        run_cmd = [
            f"{llm_backend_multimodal_example_root}/client.py",
            "--model_type=llava_onevision",
            "--end-id=151645",
            "--pad-id=151643",
            "--text=What is in this video?",
            f"--video={test_video_root}/video_test.mp4",
            "--video_num_frames=8",
        ]
        if DECOUPLED_MODE == "True":
            run_cmd += [
                "--streaming",
            ]
    output_result = venv_check_output(llm_backend_venv, run_cmd)
    validate_by_keyword(output_result, keyword)


@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["False"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", [""])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY",
                         ["max_utilization", "guaranteed_no_evict"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", ["0.7"])
@pytest.mark.parametrize("CROSS_KV_CACHE_FRACTION", [""])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY", ["inflight_fused_batching"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["True", "False"],
                         ids=["enableDecoupleMode", "disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["False"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("ENABLE_CHUNKED_CONTEXT", ["False"])
@pytest.mark.parametrize("GPU_DEVICE_IDS", [""])
@pytest.mark.parametrize("DECODING_MODE", [""])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
@pytest.mark.parametrize("IMAGE_TYPE", ["URL", "BASE64"])
@pytest.mark.parametrize("ENCODER_INPUT_FEATURES_DTYPE", ["TYPE_BF16"])
def test_mllama(
    E2E_MODEL_NAME,
    MAX_TOKENS_IN_KV_CACHE,
    MAX_ATTENTION_WINDOW_SIZE,
    BATCH_SCHEDULER_POLICY,
    KV_CACHE_FREE_GPU_MEM_FRACTION,
    CROSS_KV_CACHE_FRACTION,
    ENABLE_TRT_OVERLAP,
    BATCHING_STRATEGY,
    DECOUPLED_MODE,
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
    EXCLUDE_INPUT_IN_OUTPUT,
    IMAGE_TYPE,
    ENCODER_INPUT_FEATURES_DTYPE,
    tensorrt_llm_multimodal_example_root,
    tensorrt_llm_mllama_example_root,
    mllama_model_root,
    llm_backend_root,
    llm_backend_multimodal_example_root,
    llm_backend_venv,
):
    if BATCHING_STRATEGY == "V1":
        pytest.skip("Skipping. mllama is not supported with V1.")

    if BATCH_SCHEDULER_POLICY == "max_utilization":
        pytest.skip(
            "Skipping. models with crossAttention not supported with max_utilization."
        )

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.path.join(LLM_ROOT, "triton_backend")
    # Build Engine
    ENGINE_PATH, MULTIMODAL_ENGINE_DIR = prepare_mllama_engine(
        tensorrt_llm_multimodal_example_root, tensorrt_llm_mllama_example_root,
        mllama_model_root, llm_backend_root)
    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)

    # Prepare multimodal specific repo
    prepare_multimodal_model_repo(llm_backend_repo_root, new_model_repo,
                                  "ensemble")
    prepare_multimodal_model_repo(llm_backend_repo_root, new_model_repo,
                                  "multimodal_encoders")

    # Modify config.pbtxt
    TOKENIZER_PATH = mllama_model_root
    if CROSS_KV_CACHE_FRACTION == "":
        CROSS_KV_CACHE_FRACTION = "0.5"
    modify_ib_config_pbtxt(
        new_model_repo,
        ENGINE_PATH,
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
        MULTIMODAL_ENGINE_PATH=MULTIMODAL_ENGINE_DIR,
        CROSS_KV_CACHE_FRACTION=CROSS_KV_CACHE_FRACTION,
        ENCODER_INPUT_FEATURES_DTYPE=ENCODER_INPUT_FEATURES_DTYPE,
    )

    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    # NOTE
    # Due to mpi init error, manually set PMIX_MCA_gds=hash (ref: https://github.com/open-mpi/ompi/issues/6981)
    check_call(
        f"PMIX_MCA_gds=hash python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo} --tensorrt_llm_model_name tensorrt_llm,multimodal_encoders",
        shell=True)
    check_server_ready()

    # Run Test
    if IMAGE_TYPE == 'URL':
        IMAGE_URL = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png"
    elif IMAGE_TYPE == 'BASE64':
        IMAGE_URL = (
            "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/"
            "2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/"
            "2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/"
            "wAARCAAKAAoDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/"
            "8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoK"
            "So0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztL"
            "W2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQF"
            "BgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8R"
            "cYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqK"
            "mqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwCK3trSL7TD5CqBmVP"
            "tcUR2jYm2PlmHIUNjhgEJwdxFX1msWUNLG5kIyxF1FGCe/wAm75fp26VqzQxS6rEskSOrxKWDKCGJksgSfXIJH4mvPNTuJ4tWvI"
            "45pERZ3VVViAAGOABW2GviN9NL6f1ff+uprVwLxEb05cr06KXTz2P/2Q==")

    text_prompt = "<|image|>\nPlease elaborate what you see in the image?"
    run_cmd = [
        f"{llm_backend_multimodal_example_root}/client.py",
        "--model_type=mllama",
        f"--hf_model_dir={mllama_model_root}",
        f"--text='{text_prompt}'",
        f"--image={IMAGE_URL}",
    ]
    if DECOUPLED_MODE == "True":
        run_cmd += [
            "--streaming",
        ]

        if E2E_MODEL_NAME == "tensorrt_llm_bls":
            run_cmd += [
                "--use_bls",
            ]
    venv_check_call(llm_backend_venv, run_cmd)

    payload_str = json.dumps({
        "id": "42",
        "text_input": f"{text_prompt}",
        "image_url_input": f"{IMAGE_URL}",
        "parameters": {
            "max_tokens": 16,
            "top_k": 1,
            "top_p": 0,
            "stream": (DECOUPLED_MODE == "True"),
            "temperature": 0
        }
    })
    curl_cmd = f"curl -m 10 -X POST localhost:8000/v2/models/{E2E_MODEL_NAME}/generate_stream -d '{payload_str}'"
    check_call(curl_cmd, shell=True)


@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["False"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", [""])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY", ["guaranteed_no_evict"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", [""])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY", ["inflight_fused_batching"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["False"],
                         ids=["disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["False"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("ENABLE_CHUNKED_CONTEXT", ["False"])
@pytest.mark.parametrize("GPU_DEVICE_IDS", [""])
@pytest.mark.parametrize("DECODING_MODE", [""])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
@pytest.mark.parametrize("VIRTUAL_TOKENS", ["True", "False"],
                         ids=["withVirtualTokens", "withoutVirtualTokens"])
@pytest.mark.parametrize("ENABLE_CONTEXT_FMHA_FP32_ACC", ["True", "False"])
def test_gpt_next_ptuning_ifb(
    E2E_MODEL_NAME,
    MAX_TOKENS_IN_KV_CACHE,
    MAX_ATTENTION_WINDOW_SIZE,
    BATCH_SCHEDULER_POLICY,
    KV_CACHE_FREE_GPU_MEM_FRACTION,
    ENABLE_TRT_OVERLAP,
    BATCHING_STRATEGY,
    DECOUPLED_MODE,
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
    EXCLUDE_INPUT_IN_OUTPUT,
    VIRTUAL_TOKENS,
    ENABLE_CONTEXT_FMHA_FP32_ACC,
    inflight_batcher_llm_client_root,
    gpt_tokenizer_model_root,
    tensorrt_llm_example_root,
    tensorrt_llm_gpt_example_root,
    gpt_next_ptuning_model_root,
    llm_backend_venv,
):
    if BATCHING_STRATEGY == "V1" and BATCH_SCHEDULER_POLICY == "max_utilization":
        pytest.skip("Skipping. V1 doesn't support max_utilization.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.path.join(LLM_ROOT, "triton_backend")
    # Build engine
    ENGINE_PATH, output_model_dir = prepare_gpt_next_ptuning_engine(
        "ifb", tensorrt_llm_gpt_example_root, gpt_next_ptuning_model_root)
    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)

    # Modify config.pbtxt
    TOKENIZER_PATH = gpt_tokenizer_model_root
    modify_ib_config_pbtxt(
        new_model_repo,
        ENGINE_PATH,
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
        ENABLE_CONTEXT_FMHA_FP32_ACC=ENABLE_CONTEXT_FMHA_FP32_ACC,
    )
    # WAR for https://nvbugspro.nvidia.com/bug/4742149
    gpu_name = query_gpu_name()
    if "NVIDIA H20" == gpu_name:
        check_call("pip3 install -U nvidia-cublas-cu12", shell=True)

    # Generate reference output
    run_py_path = os.path.join(tensorrt_llm_example_root, "run.py")
    vocab_file = os.path.join(output_model_dir, "tokenizer.model")
    # 1. Input with virtual tokens:
    if VIRTUAL_TOKENS == "True":
        prompt_table = os.path.join(tensorrt_llm_gpt_example_root,
                                    "email_composition.npy")
        input_tokens = os.path.join(tensorrt_llm_gpt_example_root, "input.csv")
        run_cmd = [
            f"{run_py_path}",
            "--max_output_len=8",
            f"--vocab_file={vocab_file}",
            f"--prompt_table_path={prompt_table}",
            f"--input_file={input_tokens}",
            f"--engine_dir={ENGINE_PATH}",
            f"--output_csv=output_w_prompt.csv",
            "--no_add_special_tokens",
            "--no-kv_cache_enable_block_reuse",
        ]
        if ENABLE_CONTEXT_FMHA_FP32_ACC == "True":
            run_cmd += [
                "--enable_context_fmha_fp32_acc",
            ]

        venv_check_call(llm_backend_venv, run_cmd)
    # 2. Input w/o virtual tokens:
    elif VIRTUAL_TOKENS == "False":
        input_wo_prompt_csv = os.path.join(
            llm_backend_venv.get_working_directory(), "input_wo_prompt.csv")
        check_call(
            f"echo \"25229,291,7379,251522,39854,5754,251514,315,32906,14297,398,261\" > {input_wo_prompt_csv}",
            shell=True)
        run_cmd = [
            f"{run_py_path}",
            "--max_output_len=8",
            f"--vocab_file={vocab_file}",
            f"--input_file={input_wo_prompt_csv}",
            f"--engine_dir={ENGINE_PATH}",
            f"--output_csv=output_wo_prompt.csv",
            "--no_add_special_tokens",
        ]
        if ENABLE_CONTEXT_FMHA_FP32_ACC == "True":
            run_cmd += [
                "--enable_context_fmha_fp32_acc",
            ]

        venv_check_call(llm_backend_venv, run_cmd)

    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()

    # Run Test
    if VIRTUAL_TOKENS == "True":
        run_cmd = [
            f"{inflight_batcher_llm_client_root}/inflight_batcher_llm_client.py",
            f"--prompt-embedding-table={prompt_table}", "--prompt-task-id=0",
            f"--input-tokens-csv={input_tokens}",
            "--output-tokens-csv=output_w_prompt.csv", "--request-output-len=8",
            "--check-output"
        ]
        venv_check_call(llm_backend_venv, run_cmd)
    elif VIRTUAL_TOKENS == "False":
        run_cmd = [
            f"{inflight_batcher_llm_client_root}/inflight_batcher_llm_client.py",
            f"--input-tokens-csv={input_wo_prompt_csv}",
            "--output-tokens-csv=output_wo_prompt.csv",
            "--request-output-len=8", "--check-output"
        ]
        venv_check_call(llm_backend_venv, run_cmd)


@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["False"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", [""])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY", ["guaranteed_no_evict"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", [""])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY", ["inflight_fused_batching"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["False"],
                         ids=["disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["False"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("ENABLE_CHUNKED_CONTEXT", ["False"])
@pytest.mark.parametrize("GPU_DEVICE_IDS", [""])
@pytest.mark.parametrize("DECODING_MODE", [""])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
@pytest.mark.parametrize("GPU_WEIGHTS_PERCENT", ["0.5", "1.0"])
def test_gpt_2b_lora_ifb(
    E2E_MODEL_NAME,
    MAX_TOKENS_IN_KV_CACHE,
    MAX_ATTENTION_WINDOW_SIZE,
    BATCH_SCHEDULER_POLICY,
    KV_CACHE_FREE_GPU_MEM_FRACTION,
    ENABLE_TRT_OVERLAP,
    BATCHING_STRATEGY,
    DECOUPLED_MODE,
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
    EXCLUDE_INPUT_IN_OUTPUT,
    GPU_WEIGHTS_PERCENT,
    inflight_batcher_llm_client_root,
    tensorrt_llm_example_root,
    tensorrt_llm_gpt_example_root,
    gpt_2b_lora_model_root,
    models_root,
    llm_backend_venv,
):
    if BATCHING_STRATEGY == "V1":
        pytest.skip("Skipping. LoRA is not supported in V1.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.path.join(LLM_ROOT, "triton_backend")
    # Build engine
    weight_streaming = float(GPU_WEIGHTS_PERCENT) < 1.0
    ENGINE_PATH = prepare_gpt_2b_lora_engine("ifb",
                                             tensorrt_llm_gpt_example_root,
                                             gpt_2b_lora_model_root,
                                             models_root, weight_streaming)
    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)

    # Modify config.pbtxt
    TOKENIZER_PATH = os.path.join(models_root, "gpt-next",
                                  "gpt-next-tokenizer-hf-v2")
    modify_ib_config_pbtxt(new_model_repo,
                           ENGINE_PATH,
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
                           GPU_WEIGHTS_PERCENT=GPU_WEIGHTS_PERCENT)

    # Generate reference output
    run_py_path = os.path.join(tensorrt_llm_example_root, "run.py")
    # Input with virtual tokens:
    input_tokens = os.path.join(tensorrt_llm_gpt_example_root, "input.csv")
    output_tokens = os.path.join(tensorrt_llm_gpt_example_root, "output.csv")
    lora_path = os.path.join(tensorrt_llm_gpt_example_root,
                             "gpt-2b-lora-train-900")
    lora_nemo_path = os.path.join(tensorrt_llm_gpt_example_root,
                                  "gpt2b_lora-900.nemo")
    run_cmd = [
        f"{run_py_path}", "--max_output_len=8", f"--lora_dir={lora_nemo_path}",
        "--lora_ckpt_source=nemo", "--lora_task_uids=0",
        f"--input_file={input_tokens}", f"--output_csv={output_tokens}",
        f"--engine_dir={ENGINE_PATH}", "--use_py_session",
        f"--gpu_weights_percent={GPU_WEIGHTS_PERCENT}"
    ]
    venv_check_call(llm_backend_venv, run_cmd)

    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()

    # Run Test
    gen_cache_cmd = [
        f"{inflight_batcher_llm_client_root}/inflight_batcher_llm_client.py",
        f"--input-tokens-csv={input_tokens}",
        f"--output-tokens-csv={output_tokens}",
        "--request-output-len=8",
        "--check-output",
        f"--lora-path={lora_path}",
        "--lora-task-id=12345",
    ]
    venv_check_call(llm_backend_venv, gen_cache_cmd)

    # Test GPU cache
    run_cmd = [
        f"{inflight_batcher_llm_client_root}/inflight_batcher_llm_client.py",
        f"--input-tokens-csv={input_tokens}",
        f"--output-tokens-csv={output_tokens}",
        "--request-output-len=8",
        "--check-output",
        "--lora-task-id=12345",
    ]
    venv_check_call(llm_backend_venv, run_cmd)


@pytest.mark.parametrize("TEST_TYPE", ["accuracy"])
@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["False"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", [""])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY", ["guaranteed_no_evict"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", [""])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY", ["inflight_fused_batching"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["False"],
                         ids=["disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["False"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("ENABLE_CHUNKED_CONTEXT", ["False"])
@pytest.mark.parametrize("GPU_DEVICE_IDS", [""])
@pytest.mark.parametrize("DECODING_MODE", [""])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["True"])
@pytest.mark.parametrize("BACKEND", ["tensorrtllm", "python"])
@pytest.mark.parametrize("GUIDED_DECODING_BACKEND", ["xgrammar"])
def test_tiny_llama_1b_guided_decoding(
    TEST_TYPE,
    E2E_MODEL_NAME,
    MAX_TOKENS_IN_KV_CACHE,
    MAX_ATTENTION_WINDOW_SIZE,
    BATCH_SCHEDULER_POLICY,
    KV_CACHE_FREE_GPU_MEM_FRACTION,
    ENABLE_TRT_OVERLAP,
    BATCHING_STRATEGY,
    DECOUPLED_MODE,
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
    EXCLUDE_INPUT_IN_OUTPUT,
    BACKEND,
    GUIDED_DECODING_BACKEND,
    inflight_batcher_llm_client_root,
    tensorrt_llm_example_root,
    tensorrt_llm_llama_example_root,
    tiny_llama_model_root,
    llm_backend_venv,
):

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.path.join(LLM_ROOT, "triton_backend")

    # Build engine
    ENGINE_PATH, XGRAMMAR_TOKENIZER_INFO_PATH = prepare_tiny_llama_1b_engine(
        type=BACKEND,
        tensorrt_llm_llama_example_root=tensorrt_llm_llama_example_root,
        tiny_llama_model_root=tiny_llama_model_root,
        tensorrt_llm_example_root=tensorrt_llm_example_root)

    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)

    # Modify config.pbtxt
    TOKENIZER_PATH = tiny_llama_model_root
    modify_ib_config_pbtxt(
        new_model_repo,
        ENGINE_PATH,
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
        BACKEND=BACKEND,
        GUIDED_DECODING_BACKEND=GUIDED_DECODING_BACKEND,
        XGRAMMAR_TOKENIZER_INFO_PATH=XGRAMMAR_TOKENIZER_INFO_PATH)

    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()
    # Run Test
    if TEST_TYPE == "accuracy":
        prompt = "What is the year after 2024? Answer:"
        guide_type_lists = [
            None, "json", "json_schema", "regex", "ebnf_grammar"
        ]
        guide_lists = [
            None, None,
            '{"properties": {"answer": {"title": "Answer", "type": "integer"}}, "required": ["answer"], "title": "Answer", "type": "object"}',
            r'\d+', 'root ::= [0-9]+'
        ]
        keywords = ['Answer: 2026', '[2025]', '{"answer":2028}', '2025', '2025']
        for guide_type, guide, keyword in zip(guide_type_lists, guide_lists,
                                              keywords):
            run_cmd = [
                f"{inflight_batcher_llm_client_root}/end_to_end_grpc_client.py",
                f"--prompt={prompt}",
                "--output-len=30",
                "--exclude-input-in-output",
                "--verbose",
            ]

            if guide_type is not None:
                run_cmd += [f"--guided-decoding-guide-type={guide_type}"]

            if guide is not None:
                run_cmd += [f"--guided-decoding-guide={guide}"]

            output = venv_check_output(llm_backend_venv, run_cmd)
            validate_by_keyword(output, keyword)


@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble", "tensorrt_llm_bls"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["True"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", [""])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY",
                         ["max_utilization", "guaranteed_no_evict"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", ["0.2"])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY", ["inflight_fused_batching"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["True", "False"],
                         ids=["enableDecoupleMode", "disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["True"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("ENABLE_CHUNKED_CONTEXT", ["False"])
@pytest.mark.parametrize("GPU_DEVICE_IDS", [""])
@pytest.mark.parametrize("DECODING_MODE", ["top_k_top_p"])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
@pytest.mark.parametrize("FEATURE_NAME", ["test_basic"])
def test_gpt_disaggregated_serving_bls(
    E2E_MODEL_NAME,
    FEATURE_NAME,
    MAX_TOKENS_IN_KV_CACHE,
    MAX_ATTENTION_WINDOW_SIZE,
    BATCH_SCHEDULER_POLICY,
    KV_CACHE_FREE_GPU_MEM_FRACTION,
    ENABLE_TRT_OVERLAP,
    BATCHING_STRATEGY,
    DECOUPLED_MODE,
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
    EXCLUDE_INPUT_IN_OUTPUT,
    inflight_batcher_llm_client_root,
    tensorrt_llm_gpt_example_root,
    gpt_tokenizer_model_root,
    llm_backend_venv,
    monkeypatch,
):
    # Enable disaggregated serving.
    monkeypatch.setenv("TRTLLM_USE_MPI_KVCACHE", "1")

    if BATCHING_STRATEGY == "V1" and BATCH_SCHEDULER_POLICY == "max_utilization":
        pytest.skip("Skipping. V1 doesn't support max_utilization.")

    if BATCHING_STRATEGY == "V1" and FEATURE_NAME == "test_embedding_bias":
        pytest.skip("Skipping. V1 doesn't support embedding_bias tensor yet.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.path.join(LLM_ROOT, "triton_backend")
    # Build engine
    ENGINE_PATH = prepare_gpt_350m_engine(
        "ifb",
        tensorrt_llm_gpt_example_root,
        gpt_tokenizer_model_root,
    )
    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)
    prepare_disaggregated_serving_model_repo(llm_backend_repo_root,
                                             new_model_repo)

    # Modify config.pbtxt
    TOKENIZER_PATH = gpt_tokenizer_model_root
    modify_ib_config_pbtxt(
        new_model_repo,
        ENGINE_PATH,
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
        TENSORRT_LLM_TARGET_MODEL_NAME="tensorrt_llm",
        TENSORRT_LLM_DRAFT_MODEL_NAME="",
    )
    modify_disaggregated_serving_config_pbtxt(llm_backend_repo_root,
                                              new_model_repo)

    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()
    # Run Test
    feature_name = f"{FEATURE_NAME}"
    tokenizer_dir = f"{gpt_tokenizer_model_root}"

    if DECOUPLED_MODE == "False":
        run_cpp_backend_tests(feature_name, llm_backend_venv,
                              inflight_batcher_llm_client_root, tokenizer_dir)
    else:
        test_model_name = ""
        if ACCUMULATE_TOKEN == "True" and E2E_MODEL_NAME == "tensorrt_llm_bls":
            test_model_name = "gpt_350m"

        run_cpp_streaming_backend_tests(feature_name,
                                        llm_backend_venv,
                                        inflight_batcher_llm_client_root,
                                        tokenizer_dir,
                                        model_name=test_model_name,
                                        e2e_model=E2E_MODEL_NAME)


# Define model configurations as a dictionary
MODEL_CONFIGS = {
    "llama_v2_7b": {
        "example_root_fixture": "tensorrt_llm_llama_example_root",
        "tokenizer_path_fixture": "llama_v2_tokenizer_model_root",
        "prepare_engine_fn": prepare_llama_v2_7b_engine
    },
    "gptj_6b": {
        "example_root_fixture": "tensorrt_llm_gptj_example_root",
        "tokenizer_path_fixture": "gptj_tokenizer_model_root",
        "prepare_engine_fn": prepare_gptj_6b_engine
    },
}

# Latency thresholds for different GPUs and models
LATENCY_THRESHOLDS = {
    "NVIDIA H100 PCIe": {
        "gptj_6b": 1300,  # Threshold in milliseconds
        "llama_v2_7b": 1200,
        # Can add more models here with their thresholds
    },
    # Can add more GPU types here
}


# Fixture to handle model configuration
@pytest.fixture
def model_setup(request):
    model_name = request.param
    config = MODEL_CONFIGS[model_name]

    # Get the actual fixture values
    example_root = request.getfixturevalue(config["example_root_fixture"])
    tokenizer_path = request.getfixturevalue(config["tokenizer_path_fixture"])

    return {
        "name": model_name,
        "example_root": example_root,
        "tokenizer_path": tokenizer_path,
        "prepare_engine_fn": config["prepare_engine_fn"]
    }


@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["False"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", ["4096"])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY", ["max_utilization"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", [""])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY", ["inflight_fused_batching"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["False"],
                         ids=["disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["False"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("ENABLE_CHUNKED_CONTEXT", ["False"])
@pytest.mark.parametrize("GPU_DEVICE_IDS", [""])
@pytest.mark.parametrize("DECODING_MODE", [""])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
@pytest.mark.parametrize("model_setup",
                         list(MODEL_CONFIGS.keys()),
                         indirect=True)
def test_benchmark_core_model(
    model_setup,
    MAX_TOKENS_IN_KV_CACHE,
    MAX_ATTENTION_WINDOW_SIZE,
    BATCH_SCHEDULER_POLICY,
    KV_CACHE_FREE_GPU_MEM_FRACTION,
    ENABLE_TRT_OVERLAP,
    BATCHING_STRATEGY,
    DECOUPLED_MODE,
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
    EXCLUDE_INPUT_IN_OUTPUT,
    llm_backend_inflight_batcher_llm_root,
    llm_backend_dataset_root,
    llm_backend_venv,
):

    llm_backend_repo_root = os.path.join(LLM_ROOT, "triton_backend")
    # Build Engine
    ENGINE_PATH = model_setup["prepare_engine_fn"](
        "ifb", model_setup["example_root"], model_setup["tokenizer_path"])
    TOKENIZER_PATH = model_setup["tokenizer_path"]
    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)

    # Modify config.pbtxt
    modify_ib_config_pbtxt(
        new_model_repo,
        ENGINE_PATH,
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
    )

    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --force --world_size 1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()
    # Run Test
    run_cmd = [
        f"{llm_backend_inflight_batcher_llm_root}/benchmark_core_model.py",
        "--concurrency=8",
        "--max-input-len=300",
        "dataset",
        f"--dataset={llm_backend_dataset_root}/mini_cnn_eval.json",
        f"--tokenizer-dir={TOKENIZER_PATH}",
    ]

    output = venv_check_output(llm_backend_venv, run_cmd)
    print(output)
    latency = retrieve_latency_value(output)
    print(f"Extracted latency: {latency} ms")

    gpu_full_name = get_gpu_full_name()
    if gpu_full_name in LATENCY_THRESHOLDS:
        latency_threshold = LATENCY_THRESHOLDS["NVIDIA H100 PCIe"][
            model_setup["name"]]
        assert latency < latency_threshold, f"Latency {latency} ms is greater than the threshold {latency_threshold} ms"


@pytest.mark.parametrize("E2E_MODEL_NAME", ["tensorrt_llm"])
@pytest.mark.parametrize("DECOUPLED_MODE", [False, True],
                         ids=["disableDecoupleMode", "enableDecoupleMode"])
# TODO: [JIRA-4496] Add batch support in llmapi backend and add tests here.
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["0"])
@pytest.mark.parametrize("TENSOR_PARALLEL_SIZE", ["1", "4"])
def test_llmapi_backend(E2E_MODEL_NAME, DECOUPLED_MODE, TRITON_MAX_BATCH_SIZE,
                        TENSOR_PARALLEL_SIZE,
                        llm_backend_inflight_batcher_llm_root, llm_backend_venv,
                        llm_backend_dataset_root, tiny_llama_model_root):
    llm_backend_repo_root = os.path.join(LLM_ROOT, "triton_backend")

    if torch.cuda.device_count() < int(TENSOR_PARALLEL_SIZE):
        pytest.skip("Skipping. Not enough GPUs.")

    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_llmapi_model_repo(llm_backend_repo_root, new_model_repo)
    model_config_path = os.path.join(new_model_repo, "tensorrt_llm", "1",
                                     "model.yaml")
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)
    model_config["triton_config"]["decoupled"] = DECOUPLED_MODE
    model_config["triton_config"]["max_batch_size"] = int(TRITON_MAX_BATCH_SIZE)
    model_config["tensor_parallel_size"] = int(TENSOR_PARALLEL_SIZE)
    model_config["kv_cache_config"] = {"free_gpu_memory_fraction": 0.8}
    model_config["model"] = tiny_llama_model_root
    with open(model_config_path, "w") as f:
        yaml.dump(model_config, f)

    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)
    print_info(f"DEBUG:: model_config: {model_config}")
    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    cmd = f"python3 {launch_server_py} --world_size={TENSOR_PARALLEL_SIZE} --model_repo={new_model_repo}"
    if TENSOR_PARALLEL_SIZE == "4":
        cmd += " --trtllm_llmapi_launch"
        cmd += " --oversubscribe"
    else:
        cmd += " --no-mpi"
    print_info(f"DEBUG:: launch_server with args: {cmd}")
    check_call(cmd, shell=True)
    check_server_ready()

    # Speed up the test by running multiple tests with different configurations sharing the same triton server.
    protocols = ["http", "grpc"]
    STREAMS = [False, True]
    if DECOUPLED_MODE:
        protocols = ['grpc']  # Triton only support grpc in decoupled mode
        STREAMS = [True]  # Triton only support non-streaming in decoupled mode
    else:
        STREAMS = [False
                   ]  # Triton only support non-streaming in non-decoupled mode

    for protocol in protocols:
        for STREAM in STREAMS:
            print_info(
                f"DEBUG:: protocol: {protocol}, STREAM: {STREAM}, DECOUPLED_MODE: {DECOUPLED_MODE}"
            )
            run_cmd = [
                f"{llm_backend_inflight_batcher_llm_root}/end_to_end_test.py",
                f"--protocol={protocol}",
                f"--test-llmapi",
                f"--model-name={E2E_MODEL_NAME}",
                f"--max-input-len=192",
                f"--dataset={os.path.join(llm_backend_dataset_root, 'mini_cnn_eval.json')}",
            ]
            if STREAM:
                run_cmd += [
                    "--streaming",
                ]

            print_info("DEBUG:: run_cmd: python3 " + " ".join(run_cmd))
            venv_check_call(llm_backend_venv, run_cmd)

            run_cmd = [
                f"{llm_backend_inflight_batcher_llm_root}/benchmark_core_model.py",
                f"--max-input-len=300",
                f"--tensorrt-llm-model-name={E2E_MODEL_NAME}",
                f"--protocol={protocol}",
                f"--test-llmapi",
                'dataset',
                f"--dataset={os.path.join(llm_backend_dataset_root, 'mini_cnn_eval.json')}",
                f"--tokenizer-dir=TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            ]

            print_info("DEBUG:: run_cmd: python3 " + " ".join(run_cmd))
            venv_check_call(llm_backend_venv, run_cmd)

            # Test request cancellation with stop request
            run_cmd = [
                f"{llm_backend_repo_root}/tools/llmapi_client.py",
                "--request-output-len=200", '--stop-after-ms=25'
            ]

            output = venv_check_output(llm_backend_venv, run_cmd)
            assert 'Request is cancelled' in output

            # Test request cancellation with  request cancel
            run_cmd += ['--stop-via-request-cancel']
            output = venv_check_output(llm_backend_venv, run_cmd)
            assert 'Request is cancelled' in output

            # Test request cancellation for non-existing request and completed request
            run_cmd = [
                f"{llm_backend_repo_root}/tools/tests/test_llmapi_cancel.py"
            ]
            output = venv_check_output(llm_backend_venv, run_cmd)


@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble", "tensorrt_llm_bls"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["False"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", [""])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY", ["guaranteed_no_evict"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", [""])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY", ["inflight_fused_batching"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["True", "False"],
                         ids=["enableDecoupleMode", "disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["False"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("ENABLE_CHUNKED_CONTEXT", ["False"])
@pytest.mark.parametrize("GPU_DEVICE_IDS", [""])
@pytest.mark.parametrize("DECODING_MODE", [""])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
@pytest.mark.parametrize("TOKEN_COUNT_TEST",
                         ["input_only", "output_only", "both"])
@pytest.mark.parametrize("BACKEND", ["tensorrtllm", "python"])
def test_tiny_llama_ifb_token_counts(
    E2E_MODEL_NAME,
    MAX_TOKENS_IN_KV_CACHE,
    MAX_ATTENTION_WINDOW_SIZE,
    BATCH_SCHEDULER_POLICY,
    KV_CACHE_FREE_GPU_MEM_FRACTION,
    ENABLE_TRT_OVERLAP,
    BATCHING_STRATEGY,
    DECOUPLED_MODE,
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
    EXCLUDE_INPUT_IN_OUTPUT,
    TOKEN_COUNT_TEST,
    BACKEND,
    inflight_batcher_llm_client_root,
    tensorrt_llm_llama_example_root,
    tiny_llama_model_root,
    llm_backend_venv,
):
    """Test that the TRT-LLM inflight batcher backend can return input and output token counts."""
    if BATCHING_STRATEGY == "V1" and BATCH_SCHEDULER_POLICY == "max_utilization":
        pytest.skip("Skipping. V1 doesn't support max_utilization.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.path.join(LLM_ROOT, "triton_backend")
    # Build engine
    ENGINE_PATH, _ = prepare_tiny_llama_1b_engine(
        type="ifb",
        tensorrt_llm_llama_example_root=tensorrt_llm_llama_example_root,
        tiny_llama_model_root=tiny_llama_model_root,
        tensorrt_llm_example_root=None,
    )
    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)

    # Modify config.pbtxt
    TOKENIZER_PATH = tiny_llama_model_root
    modify_ib_config_pbtxt(new_model_repo,
                           ENGINE_PATH,
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
                           TENSORRT_LLM_TARGET_MODEL_NAME="tensorrt_llm",
                           TENSORRT_LLM_DRAFT_MODEL_NAME="",
                           BACKEND=BACKEND)

    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()

    # Test token count functionality based on the test type
    tokenizer_dir = f"{tiny_llama_model_root}"

    # Prepare different test commands based on token count test type
    if TOKEN_COUNT_TEST == "input_only":
        test_args = ["--return-num-input-tokens"]
    elif TOKEN_COUNT_TEST == "output_only":
        test_args = ["--return-num-output-tokens"]
    elif TOKEN_COUNT_TEST == "both":
        test_args = ["--return-num-input-tokens", "--return-num-output-tokens"]

    if DECOUPLED_MODE == "False":
        run_cmd = [
            f"{inflight_batcher_llm_client_root}/inflight_batcher_llm_client.py",
            f"--tokenizer-dir={tokenizer_dir}",
            "--tokenizer-type=auto",
            "--request-output-len=20",
        ] + test_args

        output = venv_check_output(llm_backend_venv, run_cmd)
    else:
        run_cmd = [
            f"{inflight_batcher_llm_client_root}/inflight_batcher_llm_client.py",
            f"--tokenizer-dir={tokenizer_dir}",
            "--tokenizer-type=auto",
            "--request-output-len=20",
            "--streaming",
        ] + test_args

        output = venv_check_output(llm_backend_venv, run_cmd)

    print(output)
    if TOKEN_COUNT_TEST == "input_only":
        assert "Input token count: [[13]]" in output
    elif TOKEN_COUNT_TEST == "output_only":
        if DECOUPLED_MODE == "False":
            assert "Output token count: [[33]]" in output
        else:
            assert "Output token count: [[1]]" in output and not "Output token count: [[20]]" in output
    elif TOKEN_COUNT_TEST == "both":
        assert "Input token count: [[13]]" in output
        if DECOUPLED_MODE == "False":
            assert "Output token count: [[33]]" in output
        else:
            assert "Output token count: [[1]]" in output and not "Output token count: [[20]]" in output
    print_info(
        f"Successfully tested token count functionality for {TOKEN_COUNT_TEST} mode"
    )


@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble", "tensorrt_llm_bls"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["False"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", [""])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY",
                         ["max_utilization", "guaranteed_no_evict"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", ["0.7"])
@pytest.mark.parametrize("CROSS_KV_CACHE_FRACTION", [""])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY", ["inflight_fused_batching"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["True", "False"],
                         ids=["enableDecoupleMode", "disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["1"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["False"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("ENABLE_CHUNKED_CONTEXT", ["False"])
@pytest.mark.parametrize("GPU_DEVICE_IDS", [""])
@pytest.mark.parametrize("DECODING_MODE", [""])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
@pytest.mark.parametrize("PROMPT_EMBEDDING_TABLE_DTYPE",
                         ["TYPE_BF16"])  # allow override later
@pytest.mark.parametrize("ENCODER_INPUT_FEATURES_DTYPE",
                         ["TYPE_FP16"])  # pixtral uses fp16 vision by default
def test_mistral_small_3_1_24b_pixtral(
    E2E_MODEL_NAME,
    MAX_TOKENS_IN_KV_CACHE,
    MAX_ATTENTION_WINDOW_SIZE,
    BATCH_SCHEDULER_POLICY,
    KV_CACHE_FREE_GPU_MEM_FRACTION,
    CROSS_KV_CACHE_FRACTION,
    ENABLE_TRT_OVERLAP,
    BATCHING_STRATEGY,
    DECOUPLED_MODE,
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
    EXCLUDE_INPUT_IN_OUTPUT,
    PROMPT_EMBEDDING_TABLE_DTYPE,
    ENCODER_INPUT_FEATURES_DTYPE,
    tensorrt_llm_multimodal_example_root,
    tensorrt_llm_llama_example_root,
    mistral_small_3_1_24b_model_root,
    llm_backend_multimodal_example_root,
    llm_backend_venv,
    llm_root,
):
    if BATCHING_STRATEGY == "V1" and BATCH_SCHEDULER_POLICY == "max_utilization":
        pytest.skip("Skipping. V1 doesn't support max_utilization.")

    llm_backend_repo_root = os.environ["LLM_BACKEND_ROOT"]

    # Build Engines (LLM + vision)
    ENGINE_PATH, MULTIMODAL_ENGINE_DIR = prepare_mistral3_pixtral_engine(
        tensorrt_llm_multimodal_example_root, tensorrt_llm_llama_example_root,
        mistral_small_3_1_24b_model_root)

    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)

    # Prepare multimodal specific repo
    prepare_multimodal_model_repo(llm_backend_repo_root, new_model_repo,
                                  "ensemble")
    prepare_multimodal_model_repo(llm_backend_repo_root, new_model_repo,
                                  "multimodal_encoders")

    # Modify config.pbtxt
    TOKENIZER_PATH = mistral_small_3_1_24b_model_root
    modify_ib_config_pbtxt(
        new_model_repo,
        ENGINE_PATH,
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
        MULTIMODAL_ENGINE_PATH=MULTIMODAL_ENGINE_DIR,
        ENCODER_INPUT_FEATURES_DTYPE=ENCODER_INPUT_FEATURES_DTYPE,
        PROMPT_EMBEDDING_TABLE_DTYPE=PROMPT_EMBEDDING_TABLE_DTYPE,
    )

    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    check_call(
        f"PMIX_MCA_gds=hash python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()

    image_merlion = os.path.join(
        llm_root,
        "tests/integration/test_input_files/merlion.png",
    )
    image_football = os.path.join(
        llm_root,
        "tests/integration/test_input_files/pexels-franco-monsalvo-252430633-32285228.jpg",
    )
    image_hockey = os.path.join(
        llm_root,
        "tests/integration/test_input_files/pexels-ron-lach-8975010.jpg",
    )
    image_basketball = os.path.join(
        llm_root,
        "tests/integration/test_input_files/pexels-maxim-shklyaev-1511525-2914194.jpg",
    )

    test_cases = [
        {
            "text": "What is the capital of England?",
            "image": "",
            "match": re.compile("london", re.IGNORECASE)
        },
        {
            "text": "In as few words as possible, what city is this?",
            "image": image_merlion,
            "match": re.compile("singapore", re.IGNORECASE)
        },
        {
            "text":
            "In as few words as possible, what sports are depicted in the images?",
            "image":
            ",".join([image_football, image_hockey]),
            "match":
            re.compile("(football|soccer).*hockey", re.IGNORECASE | re.DOTALL)
        },
        {
            "text":
            "In as few words as possible, what sports are depicted in the images?",
            "image":
            ",".join([image_football, image_hockey, image_basketball]),
            "match":
            re.compile("(football|soccer).*hockey.*basket",
                       re.IGNORECASE | re.DOTALL)
        },
    ]

    for test_case in test_cases:
        TEXT = test_case["text"]
        IMAGE = test_case["image"]
        MATCH = test_case["match"]

        # Run Test: use multimodal client; set model_type to pixtral
        run_cmd = [
            f"{llm_backend_multimodal_example_root}/client.py",
            "--model_type=pixtral",
            f"--text={TEXT}",
            f"--image={IMAGE}",
            "--request-output-len=128",
            "--end-id=2",
        ]
        if DECOUPLED_MODE == "True":
            run_cmd += ["--streaming"]

            if E2E_MODEL_NAME == "tensorrt_llm_bls":
                run_cmd += ["--use_bls"]

        output = venv_check_output(llm_backend_venv, run_cmd)

        assert MATCH.search(
            output), f"Test failed for input: {TEXT=}, {IMAGE=}, {output=}"
