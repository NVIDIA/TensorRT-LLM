import json
import os
import re
import time

import pytest
import requests

from .build_engines import *
from .common import *
from .conftest import venv_check_call, venv_check_output
from .trt_test_alternative import call, check_call, print_info


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


def get_rcca_path():
    cur_path = os.path.abspath(os.path.dirname(__file__))
    rcca_path = os.path.join(cur_path, "rcca")
    return rcca_path


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
def test_rcca_bug_4323566(
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
    llm_backend_venv,
):
    if BATCHING_STRATEGY == "V1" and BATCH_SCHEDULER_POLICY == "max_utilization":
        pytest.skip("Skipping. V1 doesn't support max_utilization.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.environ["LLM_BACKEND_ROOT"]
    # Build Engine
    ENGINE_PATH = prepare_rcca_nvbug_4323566_engine(
        "ifb", tensorrt_llm_gpt_example_root, gpt_tokenizer_model_root)
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
        f"python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()
    # Run Test
    script_path = os.path.join(get_rcca_path(), "bug_4323566",
                               "inflight_batcher_llm_client_with_end_id.py")
    run_cmd = [
        f"{script_path}",
        f"--request-output-len=200",
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
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1", "4"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
def test_rcca_bug_4342666(
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
    total_gpu_memory_mib,
    llm_backend_venv,
):
    if BATCHING_STRATEGY == "V1" and BATCH_SCHEDULER_POLICY == "max_utilization":
        pytest.skip("Skipping. V1 doesn't support max_utilization.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    if (BATCHING_STRATEGY == "inflight_fused_batching"
            and int(MAX_BEAM_WIDTH) > 1 and min(total_gpu_memory_mib) < 60000):
        pytest.skip("Skipping due to insufficient GPU memory.")

    llm_backend_repo_root = os.environ["LLM_BACKEND_ROOT"]
    # Build Engine
    ENGINE_PATH = prepare_rcca_nvbug_4342666_engine(
        "ifb", tensorrt_llm_llama_example_root, llama_v2_tokenizer_model_root)
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
        f"python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()
    # Run Test
    TEXT = """
Input: Summarize the following conversation that took place between UberEats customer support......
Output:
Summarize the following conversation that took place between UberEats customer support and a customer:
Customer: Hi, I ordered food from UberEats an hour ago, but I still haven't received my order. Can you help me with this?
UberEats Support: Sorry to hear that. Can you please provide me with your order number so I can look into this for you?
Customer: Sure, it's #1234.
UberEats Support: Thank you. I've checked on your order, and it looks like the delivery partner is running a bit behind schedule. However, they should be arriving within the next 20 minutes. Would you like me to provide you with live updates on the status of your order?
Customer: Yes, that would be great. Can you also give me a discount on my order since it's taking so long?
UberEats Support: I understand your frustration. I can offer you a 10% discount on your order. Would you like me to apply that now?
Customer: Yes, that would be great. Thank you for your help.
UberEats Support: You're welcome. I've applied the discount to your order, and I'll make sure to provide you with live updates on the status of your delivery. Your order should arrive within the next 15 minutes. Is there anything else I can assist you with today?
Customer: No, that's all. Thank you for your help.
UberEats Support: You're welcome. Enjoy your meal!!
"""
    run_cmd = [
        f"{inflight_batcher_llm_client_root}/inflight_batcher_llm_client.py",
        f"--tokenizer-dir={llama_v2_tokenizer_model_root}",
        "--tokenizer-type=llama",
        "--request-output-len=500",
        f"--beam-width={MAX_BEAM_WIDTH}",
        f"--text={TEXT}",
    ]
    output_log = venv_check_output(llm_backend_venv, run_cmd)
    print_info(f"{output_log}")
    # Get output sentence from log
    m = re.search(r"Output beam 0:\s*(.*)\s*?\n", output_log)
    output_result = ""
    if m is not None:
        output_result = m.group(1).strip()

    # Golden output sentences
    golden_result_0 = """
In this conversation, the customer support representative was able to resolve the customer's issue by providing them with a discount on their order and keeping them updated on the status of their delivery. The representative was professional and courteous throughout the conversation, and the customer was satisfied with the resolution provided.
"""
    golden_result_1 = """
In this conversation, the UberEats customer support representative was able to resolve the customer's issue by providing a 10% discount on their order and offering live updates on the status of their delivery. The customer was satisfied with the resolution and thanked the representative for their help.
"""
    golden_result_2 = """
In this conversation, the customer support representative was able to resolve the customer's issue by providing them with a discount on their order and keeping them updated on the status of their delivery.
"""
    golden_results = [golden_result_0, golden_result_1, golden_result_2]
    # Validate Accuracy
    threshold = 0.8
    validate_by_sequence_matcher(output_result, golden_results, threshold)


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
@pytest.mark.parametrize("BATCHING_STRATEGY", ["inflight_fused_batching", "V1"])
@pytest.mark.parametrize("DECOUPLED_MODE", ["True"], ids=["enableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["False"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("ENABLE_CHUNKED_CONTEXT", ["False"])
@pytest.mark.parametrize("GPU_DEVICE_IDS", [""])
@pytest.mark.parametrize("DECODING_MODE", [""])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
def test_rcca_bug_4895566(
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

    llm_backend_repo_root = os.environ["LLM_BACKEND_ROOT"]
    # Build Engine
    ENGINE_PATH = prepare_rcca_nvbug_4895566_engine(
        tensorrt_llm_llama_example_root, mistral_v1_tokenizer_model_root)

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
        "--text=this is",  # mPromptLen=3
        "--request-output-len=2",  # mMaxNewTokens=2
    ]
    if DECOUPLED_MODE == "True":
        run_cmd += [
            "--streaming",
        ]

    # Test should pass.
    # maxRestartLen = mPromptLen + mMaxNewTokens -1 = 3 + 2 - 1 = 4
    # mModel->getMaxInputLen() = max_seq_len - 1 = 5 - 1 = 4
    # So maxRestartLen <= mModel->getMaxInputLen() is true.
    venv_check_call(llm_backend_venv, run_cmd)


@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["False"])
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
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["2048"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["False"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("ENABLE_CHUNKED_CONTEXT", ["False"])
@pytest.mark.parametrize("GPU_DEVICE_IDS", [""])
@pytest.mark.parametrize("DECODING_MODE", [""])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
@pytest.mark.parametrize("TOP_K", [0, 10], ids=lambda n: f"TOP_K:{n}")
@pytest.mark.parametrize("TOP_P", [0, 0.95], ids=lambda n: f"TOP_P:{n}")
@pytest.mark.parametrize("TEMPERATURE", [0, 0.5],
                         ids=lambda n: f"Temperature:{n}")
def test_rcca_bug_4934893(
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
    TOP_K,
    TOP_P,
    TEMPERATURE,
    tensorrt_llm_llama_example_root,
    llama3_v1_8b_model_root,
):
    if BATCHING_STRATEGY == "V1" and BATCH_SCHEDULER_POLICY == "max_utilization":
        pytest.skip("Skipping. V1 doesn't support max_utilization.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.environ["LLM_BACKEND_ROOT"]
    # Build engine
    ENGINE_PATH = prepare_llama3_v1_8b_engine(tensorrt_llm_llama_example_root,
                                              llama3_v1_8b_model_root)

    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)

    # Modify config.pbtxt
    TOKENIZER_PATH = llama3_v1_8b_model_root
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
    text_prompt = "Once upon a time"
    max_tokens = 20
    stream = (DECOUPLED_MODE == "True")
    payload_str = json.dumps({
        "id": "42",
        "text_input": f"{text_prompt}",
        "parameters": {
            "max_tokens": max_tokens,
            "repetition_penalty": 5,
            "presence_penalty": 5,
            "stream": stream,
            "top_k": int(TOP_K),
            "top_p": int(TOP_P),
            "temperature": int(TEMPERATURE),
        }
    })

    # Print curl cmd for manual debug purpose.
    url_base = f"localhost:8000/v2/models/{E2E_MODEL_NAME}/generate_stream"
    curl_cmd = f"curl -m 10 -X POST {url_base} -d '{payload_str}'"
    print_info(f"Running `{curl_cmd}`")

    # Run Test.
    url = f"http://{url_base}"
    # The payload is in JSON format
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url,
                                 data=payload_str,
                                 headers=headers,
                                 timeout=10)
        # Raises an HTTPError for bad responses
        response.raise_for_status()
        output_text = response.text
        print_info(f"The outputs are: \n{output_text}")
        parse_endpoint_generated_outputs(output_text,
                                         max_tokens,
                                         stream,
                                         count_tokens=True)
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Error occurred when send request: {e}")


@skip_pre_ada
@pytest.mark.skip_less_device(2)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("E2E_MODEL_NAME", ["ensemble"])
@pytest.mark.parametrize("ACCUMULATE_TOKEN", ["False"])
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
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["2048"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["False"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("ENABLE_CHUNKED_CONTEXT", ["False"])
@pytest.mark.parametrize("GPU_DEVICE_IDS", [""])
@pytest.mark.parametrize("DECODING_MODE", [""])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["1"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["False"])
@pytest.mark.parametrize("TOP_K", [3], ids=lambda n: f"TOP_K:{n}")
@pytest.mark.parametrize("TOP_P", [0.95], ids=lambda n: f"TOP_P:{n}")
@pytest.mark.parametrize("TEMPERATURE", [0.5], ids=lambda n: f"Temperature:{n}")
def test_rcca_bug_4714193(
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
    TOP_K,
    TOP_P,
    TEMPERATURE,
    tensorrt_llm_example_root,
    tensorrt_llm_mixtral_example_root,
    mixtral_8x7b_v0_1_model_root,
    llm_backend_root,
):
    if BATCHING_STRATEGY == "V1" and BATCH_SCHEDULER_POLICY == "max_utilization":
        pytest.skip("Skipping. V1 doesn't support max_utilization.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.environ["LLM_BACKEND_ROOT"]
    # Build engine
    ENGINE_PATH = prepare_rcca_nvbug_4714193_engine(
        tensorrt_llm_example_root, tensorrt_llm_mixtral_example_root,
        mixtral_8x7b_v0_1_model_root, llm_backend_root)

    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_ib_model_repo(llm_backend_repo_root, new_model_repo)

    # Modify config.pbtxt
    TOKENIZER_PATH = mixtral_8x7b_v0_1_model_root
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
        f"python3 {launch_server_py} --world_size=2 --model_repo={new_model_repo}",
        shell=True)
    check_server_ready()
    # Run Test
    text_prompt = "The capital of France is Paris." * 900
    max_tokens = 2048
    stream = (DECOUPLED_MODE == "True")
    payload_str = json.dumps({
        "id": "42",
        "text_input": f"{text_prompt}",
        "parameters": {
            "max_tokens": max_tokens,
            "repetition_penalty": 5,
            "presence_penalty": 5,
            "stream": stream,
            "top_k": int(TOP_K),
            "top_p": int(TOP_P),
            "temperature": int(TEMPERATURE),
        }
    })

    # Print curl cmd for manual debug purpose.
    url_base = f"localhost:8000/v2/models/{E2E_MODEL_NAME}/generate_stream"
    curl_cmd = f"curl -X POST {url_base} -d '{payload_str}'"
    print_info(f"Running `{curl_cmd}`")

    # Run Test.
    url = f"http://{url_base}"
    # The payload is in JSON format
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, data=payload_str, headers=headers)
        # Raises an HTTPError for bad responses
        response.raise_for_status()
        output_text = response.text
        print_info(f"The outputs are: \n{output_text}")
        parse_endpoint_generated_outputs(output_text,
                                         max_tokens,
                                         stream,
                                         check_repetition=True)
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Error occurred when send request: {e}")


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
@pytest.mark.parametrize("DECOUPLED_MODE", ["False"],
                         ids=["disableDecoupleMode"])
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["128"])
@pytest.mark.parametrize("MAX_QUEUE_DELAY_MICROSECONDS", ["0"])
@pytest.mark.parametrize("ENABLE_KV_CACHE_REUSE", ["False"])
@pytest.mark.parametrize("NORMALIZE_LOG_PROBS", ["True"])
@pytest.mark.parametrize("ENABLE_CHUNKED_CONTEXT", ["False"])
@pytest.mark.parametrize("GPU_DEVICE_IDS", [""])
@pytest.mark.parametrize("DECODING_MODE", [""])
@pytest.mark.parametrize("MAX_BEAM_WIDTH", ["10"])
@pytest.mark.parametrize("EXCLUDE_INPUT_IN_OUTPUT", ["True"])
@pytest.mark.parametrize("RCCA", ["rcca_5077106", "rcca_4714407"])
def test_mistral_beam_search(
    RCCA,
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
    tensorrt_llm_llama_example_root,
    mistral_v1_tokenizer_model_root,
):
    if BATCHING_STRATEGY == "V1" and BATCH_SCHEDULER_POLICY == "max_utilization":
        pytest.skip("Skipping. V1 doesn't support max_utilization.")

    if E2E_MODEL_NAME == "ensemble" and ACCUMULATE_TOKEN == "True":
        pytest.skip("Skipping.")

    llm_backend_repo_root = os.environ["LLM_BACKEND_ROOT"]
    # Build Engine
    ENGINE_PATH = prepare_mistral_v1_7b_engine("beam_search",
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
    if RCCA == "rcca_4714407":
        # Create a long prompt with approximately 5000 tokens
        text_prompt = """Summarize the following text: Machine learning is a subfield of artificial intelligence that focuses on developing systems that can learn from data without explicit programming.

Machine learning algorithms build models based on sample data to make predictions or decisions. The field encompasses supervised learning with labeled data, unsupervised learning for pattern discovery, and reinforcement learning where agents learn through environment interaction.

Deep learning, a subset of machine learning, uses neural networks with many layers to process complex patterns. These networks excel at tasks like image recognition, natural language processing, and game playing.

Machine learning applications are widespread across industries. In healthcare, algorithms predict disease outbreaks and assist in diagnosis. Financial institutions use them for fraud detection and algorithmic trading. Retail companies employ recommendation systems to personalize shopping experiences. Transportation benefits from route optimization and autonomous vehicle development.

The machine learning workflow typically involves data collection, preprocessing, feature engineering, model selection, training, evaluation, and deployment. Feature selection identifies the most relevant data attributes. Models must be regularized to prevent overfitting to training data.

Recent advances include transfer learning, where models trained on one task are adapted for another, and federated learning, which trains models across decentralized devices without exchanging data. Researchers continue exploring explainable AI to make model decisions more transparent and interpretable.

Ethical considerations in machine learning include data privacy, algorithmic bias, and the socioeconomic impact of automation. Ensuring fairness across demographic groups remains challenging but crucial.

As computing power increases and algorithms improve, machine learning will likely transform more aspects of society, from personalized education to climate modeling. However, human oversight remains essential to guide these systems toward beneficial outcomes.

The history of machine learning dates back to the 1950s when Arthur Samuel created programs that could play checkers and improve through experience. Early pattern recognition systems emerged in the 1960s, but limitations in computing power restricted progress.

The 1980s saw the development of decision tree algorithms and the rediscovery of backpropagation for neural network training. The 1990s brought support vector machines and ensemble methods like random forests.

The field experienced dramatic growth in the 2000s with improved algorithms, increased computational resources, and the availability of larger datasets. ImageNet, containing millions of labeled images, catalyzed breakthroughs in computer vision when combined with convolutional neural networks.

Natural language processing has evolved from rule-based systems to statistical approaches and now transformer-based models like BERT and GPT, which capture contextual relationships in text. These models demonstrate remarkable abilities in translation, summarization, and even creative writing.

Reinforcement learning achieved milestones with systems that mastered complex games like Go, Starcraft, sometimes developing strategies that surprised human experts. These approaches combine tree search algorithms with neural networks that evaluate positions and predict outcomes.

Generative models have progressed from basic Markov models to sophisticated architectures like variational autoencoders and generative adversarial networks. These can create realistic images, music, and text that blur the line between human and machine creativity.

Self-supervised learning reduces the need for labeled data by creating supervisory signals from the data itself. Models predict masked portions of images or text, learning representations that transfer well to downstream tasks.

The hardware landscape has evolved alongside algorithms. Graphics processing units (GPUs) accelerated neural network training, while specialized AI chips like tensor processing units (TPUs) offered further efficiency gains. Quantum computing promises future breakthroughs for certain classes of machine learning problems.

Interpretability research addresses the "black box" nature of complex models. Techniques like LIME and SHAP provide local explanations for individual predictions, while attention mechanisms offer insights into which inputs influence specific outputs.

Robotics combines machine learning with physical systems, enabling machines to manipulate objects, navigate environments, and interact naturally with humans. Challenges include sample efficiency and bridging the reality gap between simulations and the physical world.

Multimodal learning integrates information across different types of data such as text, images, and audio. This enables applications like image captioning, visual question answering, and cross-modal retrieval systems.

Meta-learning or "learning to learn" develops algorithms that adapt quickly to new tasks with minimal examples. Few-shot learning systems recognize new objects from just a handful of samples, approaching human-like flexibility.

Active learning strategies select the most informative samples for labeling, reducing annotation costs. This is particularly valuable in domains where labeling requires expensive expert knowledge, such as medical imaging or scientific research.

Online learning adapts models continuously as new data arrives, crucial for dynamic environments where patterns evolve over time. Applications include stock market prediction, user preference modeling, and anomaly detection in network security.

Distributed learning frameworks enable model training across multiple machines, handling datasets too large for single systems. Techniques like parameter servers and gradient compression optimize communication between nodes.

Probabilistic programming languages like PyMC and Stan make it easier to express and solve complex statistical models. These tools help quantify uncertainty, essential for high-stakes applications like medical diagnosis or autonomous driving.

Evolutionary algorithms draw inspiration from biological evolution, using mechanisms like mutation, recombination, and selection to optimize solutions. These approaches excel at complex problems with rugged fitness landscapes where gradient-based methods struggle.

Causal inference extends beyond correlation to understand cause-effect relationships in data. Techniques include propensity score matching, instrumental variables, and structural equation modeling, helping predict the effects of interventions.

Graph neural networks process data represented as networks of nodes and edges. Applications include social network analysis, molecular property prediction, and recommendation systems that capture complex relationships between entities.

Ensemble methods combine multiple models to improve performance and robustness. Techniques range from simple averaging to sophisticated stacking approaches that learn optimal combinations of base models.

The democratization of machine learning through user-friendly tools and cloud services has expanded access beyond specialized researchers. Automated machine learning (AutoML) platforms optimize model selection and hyperparameters, making powerful techniques accessible to non-experts.

Ethical frameworks for responsible AI development include principles like fairness, accountability, transparency, and safety. Regulations like GDPR and proposed AI acts establish legal requirements for data protection and algorithmic impact assessment.

The future of machine learning likely includes more human-in-the-loop systems that leverage both machine efficiency and human intuition. Hybrid approaches may overcome current limitations in common sense reasoning and long-term planning."""
    else:
        text_prompt = "What is machine learning?"
    max_tokens = 20
    stream = (DECOUPLED_MODE == "True")
    payload_str = json.dumps({
        "id": "42",
        "text_input": f"{text_prompt}",
        "parameters": {
            "max_tokens": max_tokens,
            "pad_id": 2,
            "end_id": 774,
            "stream": stream,
            "beam_width": 10,
        }
    })
    # Print curl cmd for manual debug purpose.
    url_base = f"localhost:8000/v2/models/{E2E_MODEL_NAME}/generate"
    curl_cmd = f"curl -m 10 -X POST {url_base} -d '{payload_str}'"
    print_info(f"Running `{curl_cmd}`")

    # Run Test.
    url = f"http://{url_base}"
    # The payload is in JSON format
    headers = {"Content-Type": "application/json"}
    try:

        def send_request_and_parse_response(url,
                                            payload_str,
                                            headers,
                                            timeout=10):
            """Send a POST request to the endpoint and parse the response."""
            response = requests.post(url,
                                     data=payload_str,
                                     headers=headers,
                                     timeout=timeout)
            # Raises an HTTPError for bad responses
            response.raise_for_status()
            output_text = response.text
            print_info(f"The raw text_output is: \n{output_text}")
            parse_endpoint_generated_json_outputs(output_text,
                                                  check_repetition=True)

        if RCCA == "rcca_5077106":
            # First request
            send_request_and_parse_response(url, payload_str, headers)
            print_info("Post 2nd time and check diversity.")
            send_request_and_parse_response(url, payload_str, headers)

        elif RCCA == "rcca_4714407":
            # Test cancellation of a request and verify it doesn't affect subsequent requests
            print_info("Testing request cancellation...")
            try:
                # Create a session for the cancellable request
                with requests.Session() as session:
                    # Start a request with a very short timeout to simulate cancellation
                    session.post(url,
                                 data=payload_str,
                                 headers=headers,
                                 timeout=0.2)
                # If we reached here, the timeout didn't occur
                pytest.fail(
                    "Expected timeout didn't occur in cancellation test")
            except requests.exceptions.Timeout:
                print_info(
                    "Request timed out as expected (simulating cancellation)")
            except Exception as e:
                print_info(
                    f"Unexpected exception during cancellation test: {e}")
            # Give server a moment to clean up the cancelled request
            time.sleep(2)
            # Send another request after cancellation to verify server is still functional
            print_info(
                "Sending request after cancellation to verify server state...")
            send_request_and_parse_response(url, payload_str, headers)
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Error occurred when send request: {e}")
