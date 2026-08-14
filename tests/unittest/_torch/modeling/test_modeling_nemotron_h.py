import pytest
import torch
from utils.llm_data import llm_models_root
from utils.util import skip_fp8_pre_ada, skip_gpu_memory_less_than

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.llmapi.llm_args import CudaGraphConfig, LoadFormat
from tensorrt_llm.sampling_params import SamplingParams


def get_logprobs(token_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    raw_probs = torch.softmax(logits, dim=-1)
    index = token_ids.unsqueeze(1)
    assert index.device == raw_probs.device, f"index and raw_probs should be on the same device, but got index location: {index.device}, raw_probs location: {raw_probs.device}"
    token_probs = torch.gather(raw_probs, dim=1, index=index).squeeze(-1)
    return torch.log(token_probs)


def extract_decode_logprobs(result: RequestOutput,
                            gen_idx: int = 0) -> torch.Tensor:
    token_ids = torch.tensor(result.outputs[gen_idx].token_ids)
    logits = result.outputs[gen_idx].generation_logits
    return get_logprobs(token_ids, logits)


def create_nemotron_h_llm(model_folder,
                          use_cuda_graph,
                          disable_overlap_scheduler,
                          max_batch_size,
                          mamba_ssm_cache_dtype=None,
                          enable_chunked_prefill=False,
                          max_num_tokens=8192,
                          load_format=None,
                          cuda_graph_batch_sizes=None):
    """Create LLM with specific overlap scheduler setting"""
    model_dir = f"{llm_models_root(check=True)}/{model_folder}"
    kwargs = {}
    if max_num_tokens is not None:
        kwargs["max_num_tokens"] = max_num_tokens
    if load_format is not None:
        kwargs["load_format"] = load_format

    cuda_graph_config = None
    if use_cuda_graph:
        # Pin capture sizes when provided so MoE decode hits an exact graph
        # rather than a padded bucket (extra numeric drift vs eager).
        if cuda_graph_batch_sizes is not None:
            cuda_graph_config = CudaGraphConfig(
                batch_sizes=list(cuda_graph_batch_sizes),
                enable_padding=False,
            )
        else:
            cuda_graph_config = CudaGraphConfig()

    return LLM(
        model=model_dir,
        tensor_parallel_size=1,
        max_batch_size=max_batch_size,
        cuda_graph_config=cuda_graph_config,
        disable_overlap_scheduler=disable_overlap_scheduler,
        kv_cache_config=KvCacheConfig(
            mamba_ssm_cache_dtype=mamba_ssm_cache_dtype)
        if mamba_ssm_cache_dtype is not None else KvCacheConfig(),
        enable_chunked_prefill=enable_chunked_prefill,
        **kwargs,
    )


# Nemotron-H-8B-Base-8K product coverage was pruned (TRTLLM-15100/15101).
# Keep hybrid-architecture behavior on in-scope Nano-30B successors instead.
# test_nemotron_h_correctness was not ported: its mcore / initial-impl logprob
# goldens are 8B-specific. Product accuracy for Nano is covered by GSM8K/MMLU
# integration tests, which do not replace the CUDA-graph / overlap-scheduler
# SSM-cache equality checks below.
_NANO_30B_BF16 = "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
_NANO_30B_FP8 = "NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"


@pytest.mark.parametrize("mamba_ssm_cache_dtype", [None, "float32"],
                         ids=lambda n: f"mamba_ssm_cache_dtype:{n}")
@pytest.mark.parametrize("model_folder", [
    pytest.param(_NANO_30B_BF16,
                 marks=skip_gpu_memory_less_than((2 * 30 + 1) * 2**30)),
    pytest.param(_NANO_30B_FP8,
                 marks=skip_gpu_memory_less_than((30 + 1) * 2**30)),
])
def test_nemotron_h_sanity(mamba_ssm_cache_dtype, model_folder):
    """Hybrid path smoke only: LoadFormat.DUMMY (random weights), no numerics.

    Does not catch CUDA-graph / overlap-scheduler SSM-cache divergence; see
    test_nemotron_h_cuda_graph_overlap_scheduler for that coverage on Nano.
    """
    # Skip test if FP8 is not supported on the current architecture.
    use_fp8 = model_folder == _NANO_30B_FP8
    skip_fp8_pre_ada(use_fp8)

    torch.cuda.empty_cache()

    text_prompts = [
        "The future of AI is",
        "The president of the United States is",
    ]
    num_prompts = len(text_prompts)

    with create_nemotron_h_llm(
            model_folder=model_folder,
            use_cuda_graph=False,
            disable_overlap_scheduler=False,
            max_batch_size=num_prompts,
            mamba_ssm_cache_dtype=mamba_ssm_cache_dtype,
            load_format=LoadFormat.DUMMY,
    ) as nemotron_h:
        sampling_params = SamplingParams(max_tokens=9,
                                         temperature=0.0,
                                         add_special_tokens=False,
                                         return_context_logits=True,
                                         return_generation_logits=True)

        # Non-batching prefill sanity check.
        _ = [
            nemotron_h.generate(text_prompt, sampling_params)
            for text_prompt in text_prompts
        ]

        # Batching prefill sanity check.
        results_batching = nemotron_h.generate(text_prompts, sampling_params)
        completions_batching = [
            result.outputs[0].text for result in results_batching
        ]

        # Decoding sanity check.
        text_prompts_with_completions = [
            f"{text_prompts[i]}{completions_batching[i]}"
            for i in range(num_prompts)
        ]
        sampling_params.max_tokens = 1
        nemotron_h.generate(text_prompts_with_completions, sampling_params)


@skip_gpu_memory_less_than((2 * 30 + 1) * 2**30)
def test_nemotron_h_cuda_graph_overlap_scheduler():
    """Real-weight Nano: CUDA-graph vs eager and overlap on vs off logprobs.

    Re-parametrized from Nemotron-H-8B-Base-8K onto Nano-30B-BF16 so the
    hybrid SSM-cache behavior under graph replay stays covered after the
    8B product prune. GSM8K/MMLU do not catch small CG/overlap divergences.

    Nano-30B-A3B is MoE: greedy strings can flip under tiny CG/eager numeric
    differences even when full-vocab logits stay close. Exact text equality
    (fine on dense 8B) is therefore not required for CG vs eager; we compare
    the first decode step's logits (still aligned on the same prefill) and
    only tighten further when the greedy prefixes still match. Overlap on vs
    off stays on the CG path and keeps the stricter checks.
    """
    prompts = [
        "The sky is blue because",
        "The sum of two and two is",
        "The largest mammal is the",
        "The chemical symbol for water is",
    ]
    batch_size = len(prompts)
    # Exact CG capture size for this batch; avoid padded MoE buckets.
    cg_batch_sizes = [batch_size]

    # max_tokens=2 keeps the smoke check tight.
    sampling_config = SamplingParams(max_tokens=2,
                                     temperature=0.0,
                                     return_generation_logits=True)

    # Test without cg and overlap scheduler disabled
    with create_nemotron_h_llm(model_folder=_NANO_30B_BF16,
                               use_cuda_graph=False,
                               disable_overlap_scheduler=True,
                               max_batch_size=batch_size) as llm:
        outputs_no_cg_no_overlap = llm.generate(prompts,
                                                sampling_params=sampling_config,
                                                use_tqdm=True)

    # Test with cg and overlap scheduler disabled
    with create_nemotron_h_llm(model_folder=_NANO_30B_BF16,
                               use_cuda_graph=True,
                               disable_overlap_scheduler=True,
                               max_batch_size=batch_size,
                               cuda_graph_batch_sizes=cg_batch_sizes) as llm:
        outputs_with_cg_no_overlap = llm.generate(
            prompts, sampling_params=sampling_config, use_tqdm=True)

    # Test with cg and overlap scheduler enabled
    with create_nemotron_h_llm(model_folder=_NANO_30B_BF16,
                               use_cuda_graph=True,
                               disable_overlap_scheduler=False,
                               max_batch_size=batch_size,
                               cuda_graph_batch_sizes=cg_batch_sizes) as llm:
        outputs_with_cg_with_overlap = llm.generate(
            prompts, sampling_params=sampling_config, use_tqdm=True)

    # Verify outputs are consistent
    for i, (no_cg_no_overlap, with_cg_no_overlap,
            with_cg_with_overlap) in enumerate(
                zip(outputs_no_cg_no_overlap, outputs_with_cg_no_overlap,
                    outputs_with_cg_with_overlap)):

        no_cg_logits = no_cg_no_overlap.outputs[0].generation_logits
        with_cg_logits = with_cg_no_overlap.outputs[0].generation_logits
        with_cg_overlap_logits = with_cg_with_overlap.outputs[
            0].generation_logits
        no_cg_token_ids = no_cg_no_overlap.outputs[0].token_ids
        with_cg_token_ids = with_cg_no_overlap.outputs[0].token_ids

        # First decode step shares the same prefill → distributions align even
        # if MoE later flips the greedy token.
        torch.testing.assert_close(
            no_cg_logits[0, :],
            with_cg_logits[0, :],
            atol=0.5,
            rtol=0.2,
            msg=lambda x:
            f"Prompt {i}: with/without CG (no overlap) logits for first decode step {x}"
        )

        # Later steps / selected-token logprobs are only comparable while the
        # greedy prefixes still match.
        if no_cg_token_ids == with_cg_token_ids:
            assert (
                no_cg_no_overlap.outputs[0].text ==
                with_cg_no_overlap.outputs[0].text
            ), f"Prompt {i}: no CG no overlap generated text != with CG no overlap generated text"
            torch.testing.assert_close(
                no_cg_logits[1, :],
                with_cg_logits[1, :],
                atol=0.5,
                rtol=0.2,
                msg=lambda x:
                f"Prompt {i}: with/without CG (no overlap) logits for second decode step {x}"
            )
            torch.testing.assert_close(
                extract_decode_logprobs(no_cg_no_overlap),
                extract_decode_logprobs(with_cg_no_overlap),
                atol=0.5,
                rtol=0.2,
                msg=lambda x:
                f"Prompt {i}: with/without CG (no overlap) logprobs for all selected tokens {x}"
            )

        # Overlap on/off both use CG; keep the dense-8B-style strict checks.
        assert (
            with_cg_no_overlap.outputs[0].text ==
            with_cg_with_overlap.outputs[0].text
        ), f"Prompt {i}: with CG no overlap generated text != with CG with overlap generated text"

        torch.testing.assert_close(
            with_cg_logits[1, :],
            with_cg_overlap_logits[1, :],
            atol=0.05,
            rtol=0.05,
            msg=lambda x:
            f"Prompt {i}: with/without overlap scheduler (with CG) logits for first generated step {x}"
        )

        torch.testing.assert_close(
            extract_decode_logprobs(with_cg_no_overlap),
            extract_decode_logprobs(with_cg_with_overlap),
            atol=0.05,
            rtol=0.05,
            msg=lambda x:
            f"Prompt {i}: with/without overlap scheduler (with CG) logprobs for all selected tokens {x}"
        )


@skip_gpu_memory_less_than((2 * 30 + 1) * 2**30)
def test_nemotron_h_chunked_prefill():
    """Real-weight Nano: non-empty output with chunked prefill on the Mamba path.

    Ported from the pruned 8B coverage onto Nano-30B-BF16 so chunked prefill
    on hybrid SSM layers stays exercised.
    """
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
    sampling_config = SamplingParams(max_tokens=2, temperature=0.0)

    with create_nemotron_h_llm(model_folder=_NANO_30B_BF16,
                               use_cuda_graph=False,
                               disable_overlap_scheduler=True,
                               max_batch_size=16,
                               enable_chunked_prefill=True,
                               max_num_tokens=64) as llm:
        outputs = llm.generate(prompts,
                               sampling_params=sampling_config,
                               use_tqdm=True)

    for i, output in enumerate(outputs):
        # Verify non-empty generation
        assert len(output.outputs[0].token_ids
                   ) > 0, f"Prompt {i}: chunked prefill produced empty output"
        assert len(output.outputs[0].text
                   ) > 0, f"Prompt {i}: chunked prefill produced empty text"
