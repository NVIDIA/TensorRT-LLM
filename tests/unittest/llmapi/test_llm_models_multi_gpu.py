from test_llm_models import llm_test_harness, qwen2_model_path, sampling_params
from utils.util import skip_single_gpu


@skip_single_gpu
def test_llm_qwen2_tp2():
    llm_test_harness(qwen2_model_path,
                     inputs=['A B C'],
                     references=['D E F G H I J K L M'],
                     sampling_params=sampling_params,
                     tensor_parallel_size=2)
