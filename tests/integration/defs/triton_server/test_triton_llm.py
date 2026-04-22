import os
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
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False", "True"],
                         ids=["disableTrtOverlap", "enableTrtOverlap"])
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
                f"--tokenizer-dir={tiny_llama_model_root}",
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


def test_llmapi_backend_multi_instance(llm_backend_inflight_batcher_llm_root,
                                       llm_backend_venv,
                                       llm_backend_dataset_root,
                                       tiny_llama_model_root):
    llm_backend_repo_root = os.path.join(LLM_ROOT, "triton_backend")

    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_llmapi_model_repo(llm_backend_repo_root, new_model_repo)

    # Modify model.yaml
    model_config_path = os.path.join(new_model_repo, "tensorrt_llm", "1",
                                     "model.yaml")
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)
    model_config["triton_config"]["decoupled"] = True
    model_config["triton_config"]["max_batch_size"] = 0
    model_config["tensor_parallel_size"] = 1
    # Low KV cache to ensure both instances fit on GPU 0
    model_config["kv_cache_config"] = {"free_gpu_memory_fraction": 0.3}
    model_config["model"] = tiny_llama_model_root
    with open(model_config_path, "w") as f:
        yaml.dump(model_config, f)

    # Modify config.pbtxt for 2 instances on GPU 0
    config_pbtxt_path = os.path.join(new_model_repo, "tensorrt_llm",
                                     "config.pbtxt")
    with open(config_pbtxt_path, "r") as f:
        config_content = f.read()
    # Replace instance_group to have 2 instances
    original_instance_group = "instance_group [\n  {\n    count: 1\n    kind : KIND_CPU\n  }\n]"
    assert original_instance_group in config_content, (
        f"Expected instance_group block not found in config.pbtxt. "
        f"The config.pbtxt format may have changed. Content:\n{config_content[:500]}"
    )
    config_content = config_content.replace(
        original_instance_group,
        "instance_group [\n  {\n    count: 2\n    kind : KIND_CPU\n  }\n]\n\nparameters {\n  key: \"gpu_device_ids\"\n  value: { string_value: \"0;0\" }\n}"
    )
    with open(config_pbtxt_path, "w") as f:
        f.write(config_content)

    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)
    print_info(f"DEBUG:: model_config: {model_config}")
    with open(config_pbtxt_path, "r") as f:
        print_info(f"DEBUG:: config.pbtxt:\n{f.read()}")

    # Launch Triton Server with --no-mpi (required for multi-instance)
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    cmd = f"python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo} --no-mpi"
    print_info(f"DEBUG:: launch_server with args: {cmd}")
    check_call(cmd, shell=True)
    check_server_ready()

    # Test with grpc protocol and streaming (decoupled mode)
    protocol = "grpc"

    # Run end_to_end_test
    run_cmd = [
        f"{llm_backend_inflight_batcher_llm_root}/end_to_end_test.py",
        f"--protocol={protocol}",
        "--test-llmapi",
        "--model-name=tensorrt_llm",
        "--max-input-len=192",
        f"--dataset={os.path.join(llm_backend_dataset_root, 'mini_cnn_eval.json')}",
        "--streaming",
    ]
    print_info("DEBUG:: run_cmd: python3 " + " ".join(run_cmd))
    venv_check_call(llm_backend_venv, run_cmd)

    # Run benchmark_core_model
    run_cmd = [
        f"{llm_backend_inflight_batcher_llm_root}/benchmark_core_model.py",
        "--max-input-len=300",
        "--tensorrt-llm-model-name=tensorrt_llm",
        f"--protocol={protocol}",
        "--test-llmapi",
        "dataset",
        f"--dataset={os.path.join(llm_backend_dataset_root, 'mini_cnn_eval.json')}",
        f"--tokenizer-dir={tiny_llama_model_root}",
    ]
    print_info("DEBUG:: run_cmd: python3 " + " ".join(run_cmd))
    venv_check_call(llm_backend_venv, run_cmd)


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
            assert "Output token count: [[1]]" in output and "Output token count: [[20]]" not in output
    elif TOKEN_COUNT_TEST == "both":
        assert "Input token count: [[13]]" in output
        if DECOUPLED_MODE == "False":
            assert "Output token count: [[33]]" in output
        else:
            assert "Output token count: [[1]]" in output and "Output token count: [[20]]" not in output
    print_info(
        f"Successfully tested token count functionality for {TOKEN_COUNT_TEST} mode"
    )
