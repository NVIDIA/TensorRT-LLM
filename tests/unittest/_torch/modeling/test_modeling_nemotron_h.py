import torch
from utils.llm_data import llm_models_root
from utils.util import skip_gpu_memory_less_than

from tensorrt_llm._torch import LLM
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.llmapi.llm import RequestOutput
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


@skip_gpu_memory_less_than(
    (2 * 8 + 1) * 2**30)  # 8B, bf16, plus 1 GB for good measure
def test_nemotron_h_correctness():
    # This test is close to memory limit on A30 (with 24GB), so empty cache first
    torch.cuda.empty_cache()

    model_dir = f"{llm_models_root(check=True)}/Nemotron-H-8B-Base-8K"
    text_prompts = [
        "The future of AI is",
        "The president of the United States is",
    ]
    num_prompts = len(text_prompts)

    nemotron_h = LLM(
        model=model_dir,
        max_batch_size=num_prompts,
        use_cuda_graph=False,
        kv_cache_config=KvCacheConfig(enable_block_reuse=False),
        enable_trtllm_sampler=True,
    )

    expected_completions = [
        " bright, with endless possibilities for innovation and growth",
        " the head of state and head of government of",
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
        torch.tensor([
            -7.4359540939331055,
            -0.37661877274513245,
            -2.8925108909606934,
            -2.268364906311035,
        ]),
        torch.tensor([
            -8.759482383728027,
            -1.656238079071045,
            -0.5448741912841797,
            -1.7702054977416992,
            -0.05832016468048096,
            -1.460732102394104,
        ])
    ]
    prefill_logprobs_ref_initial_with_batching = [
        torch.tensor([
            -7.401950836181641, -0.38696032762527466, -2.8725428581237793,
            -2.2654521465301514
        ]),
        torch.tensor([
            -8.73007583618164, -1.6853574514389038, -0.5468529462814331,
            -1.7846013307571411, -0.053610533475875854, -1.4385275840759277
        ])
    ]

    decode_logprobs_ref_initial_no_batching = [
        torch.tensor([
            -2.2722280025482178, -0.5124826431274414, -0.7916123270988464,
            -2.1908130645751953, -0.059298671782016754, -0.5125972032546997,
            -0.3856367766857147, -0.055953752249479294, -1.1059765815734863
        ]),
        torch.tensor([
            -1.329713225364685, -1.5038213729858398, -0.021283088251948357,
            -0.38457369804382324, -0.3582419157028198, -0.16527847945690155,
            -0.0044861179776489735, -0.059462934732437134, -0.041099339723587036
        ])
    ]
    decode_logprobs_ref_initial_with_batching = [
        torch.tensor([
            -2.2877156734466553, -0.46699056029319763, -0.7909849286079407,
            -2.1276988983154297, -0.062114741653203964, -0.5291495323181152,
            -0.38685765862464905, -0.05595658719539642, -1.1020748615264893
        ]),
        torch.tensor([
            -1.3567769527435303, -1.5647790431976318, -0.022344056516885757,
            -0.38503751158714294, -0.3581986725330353, -0.18398350477218628,
            -0.004726295825093985, -0.05941498652100563, -0.04291720315814018
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
