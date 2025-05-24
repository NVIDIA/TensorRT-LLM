import os

import pytest

from .build_engines import *
from .common import *
from .conftest import venv_check_call
from .trt_test_alternative import call, check_call


@pytest.fixture(autouse=True)
def stop_triton_server():
    # Make sure Triton server are killed before each test.
    call(f"pkill -9 tritonserver", shell=True)
    call(f"pkill -9 trtllmExecutorWorker", shell=True)
    call(f"pkill -9 mpirun", shell=True)
    time.sleep(2)
    yield
    # Gracefully terminate Triton Server after each test.
    call(f"pkill tritonserver", shell=True)
    call(f"pkill trtllmExecutorWorker", shell=True)
    time.sleep(8)


@pytest.mark.skip_less_device(2)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble"])
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
@pytest.mark.parametrize("DECOUPLED_MODE", ["True"], ids=["enableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["True"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("ENABLE_CHUNKED_CONTEXT", ["False"])
@pytest.mark.parametrize("GPU_DEVICE_IDS", [""])
@pytest.mark.parametrize("DECODING_MODE", [""])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["True"])
@pytest.mark.parametrize("FEATURE_NAME", ["test_basic"])
def test_valgrind_llama_v2_13b(
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
    if BATCH_SCHEDULER_POLICY == "static_batch" and FEATURE_NAME == "test_embedding_bias":
        pytest.skip(
            "Skipping. static batch doesn't support embedding_bias tensor yet.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.environ["LLM_BACKEND_ROOT"]
    # Build engine
    ENGINE_PATH = prepare_llama_v2_13b_engine(tensorrt_llm_example_root,
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
    workspace = llm_backend_venv.get_working_directory()
    valgrind_log = os.path.join(workspace, "valgrind_log_llama13b.txt")
    check_call(
        "valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all --track-origins=yes --trace-children=yes " \
        f"--log-file={valgrind_log} python3 {launch_server_py} --world_size=2 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready(timeout_timer=3000, sleep_interval=60)
    # Run Test
    tokenizer_dir = f"{llama_v2_tokenizer_model_root}"
    run_cmd = [
        f"{inflight_batcher_llm_client_root}/inflight_batcher_llm_client.py",
        f"--tokenizer-dir={tokenizer_dir}",
        "--request-output-len=1000",
        "--streaming",
    ]

    venv_check_call(llm_backend_venv, run_cmd)


@pytest.mark.parametrize("E2E_MODEL_NAME", ["tensorrt_llm_bls"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["True"])
@pytest.mark.parametrize("BLS_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("PREPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("POSTPROCESSING_INSTANCE_COUNT", ["1"])
@pytest.mark.parametrize("MAX_TOKENS_IN_KV_CACHE", [""])
@pytest.mark.parametrize("MAX_ATTENTION_WINDOW_SIZE", [""])
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY", ["max_utilization"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", [""])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY", ["inflight_fused_batching"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["True"], ids=["enableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["False"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("ENABLE_CHUNKED_CONTEXT", ["False"])
@pytest.mark.parametrize("GPU_DEVICE_IDS", [""])
@pytest.mark.parametrize("DECODING_MODE", ["top_k_top_p"])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
def test_valgrind_gpt_350m(
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
    gpt_tokenizer_model_root,
    llm_backend_venv,
):
    if BATCH_SCHEDULER_POLICY == "static_batch" and FEATURE_NAME == "test_embedding_bias":
        pytest.skip(
            "Skipping. static batch doesn't support embedding_bias tensor yet.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.environ["LLM_BACKEND_ROOT"]
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
    workspace = llm_backend_venv.get_working_directory()
    valgrind_log = os.path.join(workspace, "valgrind_log_gpt350m.txt")
    check_call(
        "valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all --track-origins=yes --trace-children=yes " \
        f"--log-file={valgrind_log} python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready(timeout_timer=3000, sleep_interval=60)
    # Run Test
    tokenizer_dir = f"{gpt_tokenizer_model_root}"
    run_cmd = [
        f"{inflight_batcher_llm_client_root}/inflight_batcher_llm_client.py",
        f"--tokenizer-dir={tokenizer_dir}",
        "--request-output-len=1000",
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
@pytest.mark.parametrize("BATCH_SCHEDULER_POLICY", ["max_utilization"])
@pytest.mark.parametrize("KV_CACHE_FREE_GPU_MEM_FRACTION", [""])
@pytest.mark.parametrize("ENABLE_TRT_OVERLAP", ["False"],
                         ids=["disableTrtOverlap"])
@pytest.mark.parametrize("BATCHING_STRATEGY", ["inflight_fused_batching"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["True"], ids=["enableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["True"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("ENABLE_CHUNKED_CONTEXT", ["False"])
@pytest.mark.parametrize("GPU_DEVICE_IDS", [""])
@pytest.mark.parametrize("DECODING_MODE", [""])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["True"])
@pytest.mark.parametrize("FEATURE_NAME", ["test_basic"])
def test_llama_v3_8b_rss_increasement(
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
    llama_v3_8b_model_root,
    llm_backend_venv,
):
    if BATCH_SCHEDULER_POLICY == "static_batch" and FEATURE_NAME == "test_embedding_bias":
        pytest.skip(
            "Skipping. static batch doesn't support embedding_bias tensor yet.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.environ["LLM_BACKEND_ROOT"]
    # Build engine
    ENGINE_PATH = prepare_llama_v3_8b_engine(tensorrt_llm_llama_example_root,
                                             llama_v3_8b_model_root,
                                             workers=1)

    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)

    # Modify config.pbtxt
    TOKENIZER_PATH = llama_v3_8b_model_root
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
    tokenizer_dir = f"{llama_v3_8b_model_root}"
    run_cmd = [
        f"{inflight_batcher_llm_client_root}/inflight_batcher_llm_client.py",
        f"--tokenizer-dir={tokenizer_dir}",
        "--request-output-len=10",
        "--streaming",
    ]

    check_avg_rss_increasement(llm_backend_venv,
                               process_name="tritonserver",
                               inference_cmd=run_cmd,
                               rss_increase_bytes_threshold=64,
                               warm_up_times=50,
                               total_run_times=60)
