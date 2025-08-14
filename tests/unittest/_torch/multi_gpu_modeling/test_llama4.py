from difflib import SequenceMatcher

import pytest
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig


@pytest.mark.parametrize(
    "model_name",
    ["Llama-4-Maverick-17B-128E-Instruct", "Llama-4-Scout-17B-16E-Instruct"],
    ids=['maverick', 'scout'])
@pytest.mark.parametrize("backend", ["TRTLLM", "FLASHINFER"],
                         ids=["trtllm", "flashinfer"])
@pytest.mark.parametrize("tp_size", [1, 8], ids=["tp1", "tp8"])
@pytest.mark.parametrize("use_cuda_graph", [True, False],
                         ids=["enable_graph", "disable_graph"])
@pytest.mark.parametrize("enable_attention_dp", [True, False],
                         ids=["enable_adp", "disable_adp"])
@pytest.mark.parametrize("ep_size", [4, 1], ids=["ep4", "ep1"])
@pytest.mark.parametrize("pp_size", [1, 8], ids=["pp1", "pp8"])
def test_llama4(model_name, backend, tp_size, use_cuda_graph,
                enable_attention_dp, ep_size, pp_size):
    if pp_size > 1 and (ep_size > 1 or tp_size > 1):
        return

    if pp_size == 1 and tp_size == 1:
        return

    if enable_attention_dp and not (tp_size == 8 and ep_size == 4
                                    and pp_size == 1):
        pytest.skip("Skip this attention DP test case to avoid too many tests")

    prompts = [
        {
            "prompt": "The president of the United States is"
        },
        {
            # NOTE: Long context accuracy testing (RULER) is not available in CI yet.
            # This test cannot be removed until long context is covered.
            "prompt":
            "This is a very long prompt to exercise long context. Count up to 10000 from 1, 2, 3,"
            + ", ".join(str(i) for i in range(4, 9000))
        },
        # TODO: Fix multimodal test.
        # {
        #     "prompt": "<|image|>This image is of color",
        #     "multi_modal_data": {
        #         "image": [torch.ones(3, 1024, 1024)]
        #     }
        # },
    ]

    expected_outputs = [
        " the head of state and head of government of the",
        ", 9000, 9001, ",
        # " white. What is the color of the background of"  # TODO: Fix multimodal test.
    ]

    pytorch_config = dict(attn_backend=backend)
    model_dir = str(llm_models_root() / "llama4-models" / model_name)

    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.25, )
    llm = LLM(
        model=model_dir,
        tensor_parallel_size=tp_size,
        moe_expert_parallel_size=ep_size,
        moe_tensor_parallel_size=tp_size // ep_size,
        cuda_graph_config=CudaGraphConfig() if use_cuda_graph else None,
        **pytorch_config,
        pipeline_parallel_size=pp_size,
        enable_attention_dp=enable_attention_dp,
        kv_cache_config=kv_cache_config,
        use_torch_sampler=True,
        enable_chunked_prefill=True,
    )
    with llm:
        outputs = llm.generate(
            prompts,
            sampling_params=SamplingParams(max_tokens=10),
        )

    assert len(outputs) == len(expected_outputs), "Output length mismatch"

    def similar(a, b, threshold=0.9):
        return SequenceMatcher(None, a, b).ratio() >= threshold

    for output, expected in zip(outputs, expected_outputs):
        output_text = output.outputs[0].text
        assert similar(
            output_text,
            expected), f"Expected '{expected}' but get '{output_text}'"
