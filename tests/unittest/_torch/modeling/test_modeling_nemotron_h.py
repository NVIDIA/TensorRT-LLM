import torch
from utils.llm_data import llm_models_root
from utils.util import skip_gpu_memory_less_than

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.llmapi.llm_args import CudaGraphConfig
from tensorrt_llm.sampling_params import SamplingParams


def get_logprobs(token_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    raw_probs = torch.softmax(logits, dim=-1)
    index = token_ids.unsqueeze(1).cuda()
    token_probs = torch.gather(raw_probs, dim=1, index=index).squeeze(-1)
    return torch.log(token_probs)


def extract_prefill_logprobs(result: RequestOutput) -> torch.Tensor:
    token_ids = torch.tensor(result.prompt_token_ids[1:])
    logits = result.context_logits[:-1, :]
    return get_logprobs(token_ids, logits)


def extract_decode_logprobs(result: RequestOutput,
                            gen_idx: int = 0) -> torch.Tensor:
    token_ids = torch.tensor(result.outputs[gen_idx].token_ids)
    logits = result.outputs[gen_idx].generation_logits
    return get_logprobs(token_ids, logits)


def create_nemotron_h_llm(use_cuda_graph, disable_overlap_scheduler,
                          max_batch_size):
    """Create LLM with specific overlap scheduler setting"""
    model_dir = f"{llm_models_root(check=True)}/Nemotron-H-8B-Base-8K"
    return LLM(
        model=model_dir,
        tensor_parallel_size=1,
        max_batch_size=max_batch_size,
        cuda_graph_config=CudaGraphConfig() if use_cuda_graph else None,
        disable_overlap_scheduler=disable_overlap_scheduler,
        kv_cache_config=KvCacheConfig(enable_block_reuse=False),
        enable_trtllm_sampler=True,
    )


@skip_gpu_memory_less_than(
    (2 * 8 + 1) * 2**30)  # 8B, bf16, plus 1 GB for good measure
def test_nemotron_h_correctness():
    # This test is close to memory limit on A30 (with 24GB), so empty cache first
    torch.cuda.empty_cache()

    text_prompts = [
        "The future of AI is",
        "The president of the United States is",
    ]
    num_prompts = len(text_prompts)

    nemotron_h = create_nemotron_h_llm(use_cuda_graph=False,
                                       disable_overlap_scheduler=False,
                                       max_batch_size=num_prompts)

    expected_completions = [
        " function-driven, not just data-driven. This",
        " functionary of the executive branch of the government",
    ]

    # reference logprobs for first prompt from mcore for prompt minus first token
    # TODO(oargov): generate a reference on-the-fly once we have confidence in the HF impl
    prefill_logprobs_ref_mcore = torch.tensor([
        -7.415980815887451, -0.36192911863327026, -2.8658294677734375,
        -2.316344738006592
    ])

    # reference logprobs from initial implementation (commit 5ce1102a02bd2938c0c8334138371f081f55fcc1 on single RTX 6000)
    initial_impl_atol = 0.2
    batching_atol = 0.2

    prefill_logprobs_ref_initial_no_batching = [
        torch.tensor([-7.4191, -0.3989, -2.9075, -2.2781]),
        torch.tensor([-8.7160, -1.6506, -0.5491, -1.7701, -0.0556, -1.4478])
    ]
    prefill_logprobs_ref_initial_with_batching = [
        torch.tensor([-7.4191, -0.3989, -2.9075, -2.2781]),
        torch.tensor([-8.7160, -1.6506, -0.5491, -1.7701, -0.0556, -1.4478])
    ]

    decode_logprobs_ref_initial_no_batching = [
        torch.tensor([
            -14.2367, -2.8060, -1.0944, -1.6960, -2.2636, -2.2669, -0.0893,
            -0.6623, -2.3737
        ]),
        torch.tensor([
            -14.1523, -1.1880, -1.9109, -0.5800, -1.5452, -0.2429, -0.7025,
            -0.6344, -1.0305
        ])
    ]
    decode_logprobs_ref_initial_with_batching = [
        torch.tensor([
            -14.2367, -2.8015, -1.0884, -1.6462, -2.2261, -2.2851, -0.0862,
            -0.7100, -2.3674
        ]),
        torch.tensor([
            -14.1523, -1.1964, -1.8843, -0.5719, -1.5365, -0.2451, -0.6931,
            -0.5795, -1.0162
        ])
    ]

    try:
        sampling_params = SamplingParams(max_tokens=9,
                                         temperature=0.0,
                                         add_special_tokens=False,
                                         return_context_logits=True,
                                         return_generation_logits=True)

        results_no_batching = [
            nemotron_h.generate(text_prompt, sampling_params)
            for text_prompt in text_prompts
        ]
        completions_no_batching = [
            result.outputs[0].text for result in results_no_batching
        ]
        prefill_logprobs_no_batching = [
            extract_prefill_logprobs(result).cpu()
            for result in results_no_batching
        ]
        decode_logprobs_no_batching = [
            extract_decode_logprobs(result).cpu()
            for result in results_no_batching
        ]

        results_batching = nemotron_h.generate(text_prompts, sampling_params)
        completions_batching = [
            result.outputs[0].text for result in results_batching
        ]
        prefill_logprobs_batching = [
            extract_prefill_logprobs(result).cpu()
            for result in results_batching
        ]
        decode_logprobs_batching = [
            extract_decode_logprobs(result).cpu() for result in results_batching
        ]

        # compare logprobs with mcore logprobs, check that the max error is less than 0.3
        mcore_atol = 0.3
        torch.testing.assert_close(torch.tensor(
            prefill_logprobs_no_batching[0]),
                                   prefill_logprobs_ref_mcore,
                                   atol=mcore_atol,
                                   rtol=0.0)

        for i in range(num_prompts):
            # compare prompt logprobs with initial implementation
            torch.testing.assert_close(
                prefill_logprobs_no_batching[i],
                prefill_logprobs_ref_initial_no_batching[i],
                atol=initial_impl_atol,
                rtol=0.0)
            torch.testing.assert_close(
                prefill_logprobs_batching[i],
                prefill_logprobs_ref_initial_with_batching[i],
                atol=initial_impl_atol,
                rtol=0.0)

            # compare expected completion
            assert completions_batching[i] == expected_completions[i]
            assert completions_no_batching[i] == expected_completions[i]

            # compare decode logprobs with initial implementation
            torch.testing.assert_close(
                decode_logprobs_no_batching[i],
                decode_logprobs_ref_initial_no_batching[i],
                atol=initial_impl_atol,
                rtol=0.0)
            torch.testing.assert_close(
                decode_logprobs_batching[i],
                decode_logprobs_ref_initial_with_batching[i],
                atol=initial_impl_atol,
                rtol=0.0)

            # compare logprobs with and without batching, tolerace by diff in initial implementation
            torch.testing.assert_close(prefill_logprobs_batching[i],
                                       prefill_logprobs_no_batching[i],
                                       atol=batching_atol,
                                       rtol=0.0)
            torch.testing.assert_close(decode_logprobs_batching[i],
                                       decode_logprobs_no_batching[i],
                                       atol=batching_atol,
                                       rtol=0.0)

        # now let's test that decodes match prefill logprobs
        text_prompts_with_completions = [
            f"{text_prompts[i]}{completions_batching[i]}"
            for i in range(num_prompts)
        ]

        sampling_params.max_tokens = 1
        full_sequence_results = nemotron_h.generate(
            text_prompts_with_completions, sampling_params)
        full_sequence_logprobs = [
            extract_prefill_logprobs(result).cpu()
            for result in full_sequence_results
        ]

        # compare full sequence logprobs with prefill+decode logprobs, tolerance like mcore tolerance
        for i in range(num_prompts):
            prefill_decode_logprobs = torch.cat(
                [prefill_logprobs_batching[i], decode_logprobs_batching[i]])
            torch.testing.assert_close(full_sequence_logprobs[i],
                                       prefill_decode_logprobs,
                                       atol=mcore_atol,
                                       rtol=0.0)

    finally:
        nemotron_h.shutdown()


def test_nemotron_h_cuda_graph_overlap_scheduler():
    prompts = [
        "Tell me something I don't know about the future of AI",
        "The president of the United States is",
        "The capital of France is",
        "Hello, this is a beautiful day and I'm eager to start my day and",
    ]
    sampling_config = SamplingParams(max_tokens=12,
                                     temperature=0.0,
                                     return_generation_logits=True)

    # Test without cg and overlap scheduler disabled
    with create_nemotron_h_llm(use_cuda_graph=False,
                               disable_overlap_scheduler=True,
                               max_batch_size=16) as llm:
        outputs_no_cg_no_overlap = llm.generate(prompts,
                                                sampling_params=sampling_config,
                                                use_tqdm=True)

    # Test with cg and overlap scheduler disabled
    with create_nemotron_h_llm(use_cuda_graph=True,
                               disable_overlap_scheduler=True,
                               max_batch_size=16) as llm:
        outputs_with_cg_no_overlap = llm.generate(
            prompts, sampling_params=sampling_config, use_tqdm=True)

    # Test with cg and overlap scheduler enabled
    with create_nemotron_h_llm(use_cuda_graph=True,
                               disable_overlap_scheduler=False,
                               max_batch_size=16) as llm:
        outputs_with_cg_with_overlap = llm.generate(
            prompts, sampling_params=sampling_config, use_tqdm=True)

    # Verify outputs are consistent
    for (no_cg_no_overlap, with_cg_no_overlap,
         with_cg_with_overlap) in zip(outputs_no_cg_no_overlap,
                                      outputs_with_cg_no_overlap,
                                      outputs_with_cg_with_overlap):

        assert (no_cg_no_overlap.outputs[0].text ==
                with_cg_no_overlap.outputs[0].text)
        assert (with_cg_no_overlap.outputs[0].text ==
                with_cg_with_overlap.outputs[0].text)

        # similar to other unittests comparing with / without CG, compare logits of first generation step (2nd generated token)
        torch.testing.assert_close(
            no_cg_no_overlap.outputs[0].generation_logits[1, :],
            with_cg_no_overlap.outputs[0].generation_logits[1, :],
            atol=0.2,
            rtol=0.2)

        # compare logprobs of all generated tokens
        torch.testing.assert_close(extract_decode_logprobs(no_cg_no_overlap),
                                   extract_decode_logprobs(with_cg_no_overlap),
                                   atol=0.2,
                                   rtol=0.2)

        # overlap scheduler should have no effect on all logits - low tolerance
        torch.testing.assert_close(
            with_cg_no_overlap.outputs[0].generation_logits,
            with_cg_with_overlap.outputs[0].generation_logits,
            atol=0.05,
            rtol=0.05)
