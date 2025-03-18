from test_llm_models import (gemma_2_9b_it_model_path, gptj_model_path,
                             llm_test_harness, qwen2_model_path,
                             sampling_params)
from utils.util import skip_single_gpu


@skip_single_gpu
def test_llm_gptj_tp2():
    llm_test_harness(gptj_model_path,
                     inputs=["A B C"],
                     references=["D E F G H I J K L M"],
                     sampling_params=sampling_params,
                     tensor_parallel_size=2)


@skip_single_gpu
def test_llm_gemma_2_9b_it_tp2():
    llm_test_harness(gemma_2_9b_it_model_path,
                     inputs=['A B C'],
                     references=['D E F G H I J K L M'],
                     sampling_params=sampling_params,
                     tensor_parallel_size=2)


@skip_single_gpu
def test_llm_qwen2_tp2():
    llm_test_harness(qwen2_model_path,
                     inputs=['A B C'],
                     references=['D E F G H I J K L M'],
                     sampling_params=sampling_params,
                     tensor_parallel_size=2)
