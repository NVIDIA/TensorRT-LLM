import os

import pytest

from .build_engines import *
from .common import *


@pytest.mark.skip_less_device(8)
@pytest.mark.skip_less_device_memory(80000)
def test_gpt175b_dummyWeights_multi_node_engine_config(
    tensorrt_llm_gpt_example_root,
    tensorrt_llm_example_root,
    gpt_tokenizer_model_root,
):
    ACCUMULATE_TOKEN = "False"
    BLS_INSTANCE_COUNT = "1"
    PREPROCESSING_INSTANCE_COUNT = "1"
    POSTPROCESSING_INSTANCE_COUNT = "1"
    MAX_TOKENS_IN_KV_CACHE = ""
    MAX_ATTENTION_WINDOW_SIZE = ""
    BATCH_SCHEDULER_POLICY = "max_utilization"
    KV_CACHE_FREE_GPU_MEM_FRACTION = ""
    ENABLE_TRT_OVERLAP = "False"
    BATCHING_STRATEGY = "inflight_fused_batching"
    DECOUPLED_MODE = "True"
    TRITON_MAX_BATCH_SIZE = "128"
    MAX_QUEUE_DELAY_MICROSECONDS = "0"
    ENABLE_KV_CACHE_REUSE = "False"
    NORMALIZE_LOG_PROBS = "True"
    ENABLE_CHUNKED_CONTEXT = "False"
    GPU_DEVICE_IDS = ""
    DECODING_MODE = ""
    MAX_BEAM_WIDTH = "1"
    EXCLUDE_INPUT_IN_OUTPUT = "False"

    llm_backend_repo_root = os.environ["LLM_BACKEND_ROOT"]
    # Build Engine
    ENGINE_PATH = prepare_gpt_multi_node_engine("ifb",
                                                tensorrt_llm_gpt_example_root,
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
