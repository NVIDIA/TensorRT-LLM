import pytest
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
    index = token_ids.unsqueeze(1)
    assert index.device == raw_probs.device, f"index and raw_probs should be on the same device, but got index location: {index.device}, raw_probs location: {raw_probs.device}"
    token_probs = torch.gather(raw_probs, dim=1, index=index).squeeze(-1)
    return torch.log(token_probs)


def extract_prefill_logprobs(result: RequestOutput) -> torch.Tensor:
    token_ids = torch.tensor(result.prompt_token_ids[1:])
    logits = result.context_logits[:-1, :]
    return get_logprobs(token_ids.cuda(), logits)


def extract_decode_logprobs(result: RequestOutput,
                            gen_idx: int = 0) -> torch.Tensor:
    token_ids = torch.tensor(result.outputs[gen_idx].token_ids)
    logits = result.outputs[gen_idx].generation_logits
    return get_logprobs(token_ids, logits)


def create_nemotron_h_llm(use_cuda_graph,
                          disable_overlap_scheduler,
                          max_batch_size,
                          mamba_ssm_cache_dtype=None,
                          enable_chunked_prefill=False,
                          max_num_tokens=None):
    """Create LLM with specific overlap scheduler setting"""
    model_dir = f"{llm_models_root(check=True)}/Nemotron-H-8B-Base-8K"
    return LLM(
        model=model_dir,
        tensor_parallel_size=1,
        max_batch_size=max_batch_size,
        cuda_graph_config=CudaGraphConfig() if use_cuda_graph else None,
        disable_overlap_scheduler=disable_overlap_scheduler,
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False,
            mamba_ssm_cache_dtype="auto"
            if mamba_ssm_cache_dtype is None else mamba_ssm_cache_dtype),
        sampler_type="TRTLLMSampler",
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_tokens=max_num_tokens,
    )


@skip_gpu_memory_less_than(
    (2 * 8 + 1) * 2**30)  # 8B, bf16, plus 1 GB for good measure
@pytest.mark.parametrize("mamba_ssm_cache_dtype", [None, "float32"],
                         ids=lambda n: f"mamba_ssm_cache_dtype:{n}")
def test_nemotron_h_correctness(mamba_ssm_cache_dtype):
    # This test is close to memory limit on A30 (with 24GB), so empty cache first
    torch.cuda.empty_cache()

    text_prompts = [
        "The future of AI is",
        "The president of the United States is",
    ]
    num_prompts = len(text_prompts)

    nemotron_h = create_nemotron_h_llm(
        use_cuda_graph=False,
        disable_overlap_scheduler=False,
        max_batch_size=num_prompts,
        mamba_ssm_cache_dtype=mamba_ssm_cache_dtype)

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


def test_nemotron_h_cuda_graph_overlap_scheduler():
    prompts = [
        "The sky is blue because",
        "The sum of two and two is",
        "The largest mammal is the",
        "The chemical symbol for water is",
    ]

    sampling_config = SamplingParams(max_tokens=10,
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
    for i, (no_cg_no_overlap, with_cg_no_overlap,
            with_cg_with_overlap) in enumerate(
                zip(outputs_no_cg_no_overlap, outputs_with_cg_no_overlap,
                    outputs_with_cg_with_overlap)):

        assert (
            no_cg_no_overlap.outputs[0].text ==
            with_cg_no_overlap.outputs[0].text
        ), f"Prompt {i}: no CG no overlap generated text != with CG no overlap generated text"
        assert (
            with_cg_no_overlap.outputs[0].text ==
            with_cg_with_overlap.outputs[0].text
        ), f"Prompt {i}: with CG no overlap generated text != with CG with overlap generated text"

        # similar to other unittests comparing with / without CG, compare logits of first generation step (2nd generated token)
        torch.testing.assert_close(
            no_cg_no_overlap.outputs[0].generation_logits[1, :],
            with_cg_no_overlap.outputs[0].generation_logits[1, :],
            atol=0.2,
            rtol=0.2,
            msg=lambda x:
            f"Prompt {i}: with/without CG (no overlap) logits for first generated step {x}"
        )

        # compare logprobs of all generated tokens
        torch.testing.assert_close(
            extract_decode_logprobs(no_cg_no_overlap),
            extract_decode_logprobs(with_cg_no_overlap),
            atol=0.2,
            rtol=0.2,
            msg=lambda x:
            f"Prompt {i}: with/without CG (no overlap) logprobs for all selected tokens {x}"
        )

        # Similar comparison for with / without overlap scheduler, compare logits of first generation step (2nd generated token)
        # overlap scheduler should have no effect on all logits - low tolerance
        torch.testing.assert_close(
            with_cg_no_overlap.outputs[0].generation_logits[1, :],
            with_cg_with_overlap.outputs[0].generation_logits[1, :],
            atol=0.05,
            rtol=0.05,
            msg=lambda x:
            f"Prompt {i}: with/without overlap scheduler (with CG) logits for first generated step {x}"
        )

        # compare logprobs of all generated tokens
        torch.testing.assert_close(
            extract_decode_logprobs(with_cg_no_overlap),
            extract_decode_logprobs(with_cg_with_overlap),
            atol=0.05,
            rtol=0.05,
            msg=lambda x:
            f"Prompt {i}: with/without overlap scheduler (with CG) logprobs for all selected tokens {x}"
        )


def test_nemotron_h_chunked_prefill():
    # Long prompts (~100 tokens) to make sure chunked prefill is enabled
    # (At the time of development, tokens_per_block isn't configurable from the LLM API,
    # and max_tokens (i.e. chunk size) needs to be a multiple of tokens_per_block)
    prompts = [
        "Artificial Intelligence in Healthcare: Artificial intelligence (AI) is transforming healthcare by improving diagnostics, treatment plans, and patient care. AI algorithms can analyze medical images with high accuracy, assist in early disease detection, and personalize treatment plans based on patient data. Additionally, AI-powered chatbots and virtual assistants provide support to patients, enhancing accessibility and efficiency in healthcare services. As AI technology continues to advance, its integration into healthcare systems promises to deliver better outcomes and reduce costs. With continuous research and development, AI in healthcare is poised to",
        "The Role of Cloud Computing: Cloud computing has revolutionized the way businesses operate by providing scalable, on-demand access to computing resources. This technology allows organizations to store and process data remotely, reducing the need for physical infrastructure and enabling greater flexibility. Cloud services facilitate collaboration, enhance data security, and support the deployment of innovative applications. As businesses increasingly adopt cloud solutions, they benefit from improved efficiency, cost savings, and the ability to rapidly adapt to changing market conditions. Companies leveraging cloud computing are better positioned to",
        "Advancements in Renewable Energy: Renewable energy technologies, such as solar and wind power, are crucial for addressing climate change and reducing dependence on fossil fuels. Advances in energy storage, grid integration, and efficiency are making renewable energy sources more viable and cost-effective. Innovations in materials science and engineering are also driving the development of next-generation renewable technologies. As global efforts to combat climate change intensify, the continued advancement of renewable energy will play a pivotal role in achieving a sustainable future. Governments and industries are increasingly investing in",
        "The Importance of Cybersecurity: In today's digital age, cybersecurity has become essential to protect sensitive information and maintain the integrity of systems. With the rise of cyber threats such as hacking, phishing, and ransomware, organizations must implement robust security measures to safeguard their data. Cybersecurity involves a combination of technologies, processes, and practices designed to defend against unauthorized access and attacks. By staying vigilant and updating security protocols, businesses can mitigate risks and ensure the safety of their digital assets. Proactive cybersecurity strategies are crucial in",
        "The Impact of Artificial Intelligence on Education: Artificial intelligence is reshaping education by providing personalized learning experiences and automating administrative tasks. AI-driven educational tools can adapt to individual student needs, offering tailored feedback and resources to enhance learning outcomes. Additionally, AI can streamline administrative processes, allowing educators to focus more on teaching and student engagement. As AI continues to evolve, its role in education will expand, offering new opportunities for innovation and efficiency. The integration of AI in classrooms promises to revolutionize how students learn and how educators manage their",
    ]
    sampling_config = SamplingParams(max_tokens=10,
                                     temperature=0.0,
                                     return_context_logits=True,
                                     return_generation_logits=True)

    with create_nemotron_h_llm(use_cuda_graph=False,
                               disable_overlap_scheduler=True,
                               max_batch_size=16) as llm:
        outputs = llm.generate(prompts,
                               sampling_params=sampling_config,
                               use_tqdm=True)

    with create_nemotron_h_llm(use_cuda_graph=False,
                               disable_overlap_scheduler=True,
                               max_batch_size=16,
                               enable_chunked_prefill=True,
                               max_num_tokens=64) as llm:
        chunked_prefill_outputs = llm.generate(prompts,
                                               sampling_params=sampling_config,
                                               use_tqdm=True)

    for i, (output, chunked_prefill_output) in enumerate(
            zip(outputs, chunked_prefill_outputs)):
        assert output.outputs[0].text == chunked_prefill_output.outputs[0].text

        # assert same prefill logprobs. Same atol as diff between mcore and initial impl
        prefill_logprobs = extract_prefill_logprobs(output)
        chunked_prefill_logprobs = extract_prefill_logprobs(
            chunked_prefill_output)
        torch.testing.assert_close(
            prefill_logprobs,
            chunked_prefill_logprobs,
            atol=0.3,
            rtol=0.05,
            msg=lambda x: f"Prompt {i} prefill logprobs {x}")

        # Decode logprobs shouldn't be affected by chunked prefill - tolerance like batching tolerance
        decode_logprobs = extract_decode_logprobs(output)
        chunked_decode_logprobs = extract_decode_logprobs(
            chunked_prefill_output)
        torch.testing.assert_close(
            decode_logprobs,
            chunked_decode_logprobs,
            atol=0.2,
            rtol=0.05,
            msg=lambda x: f"Prompt {i} decode logprobs {x}")
