import pytest
import torch
from utils.llm_data import llm_models_root
from utils.util import (skip_fp8_pre_ada, skip_gpu_memory_less_than,
                        skip_single_gpu)

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.llmapi.llm_args import (CudaGraphConfig, LoadFormat,
                                          PrefillCudaGraphBackend)
from tensorrt_llm.sampling_params import SamplingParams


def get_logprobs(token_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    raw_probs = torch.softmax(logits, dim=-1)
    index = token_ids.unsqueeze(1)
    assert index.device == raw_probs.device, (
        "index and raw_probs should be on the same device, "
        f"but got index location: {index.device}, raw_probs location: {raw_probs.device}"
    )
    token_probs = torch.gather(raw_probs, dim=1, index=index).squeeze(-1)
    return torch.log(token_probs)


def extract_decode_logprobs(result: RequestOutput,
                            gen_idx: int = 0) -> torch.Tensor:
    """Shared by test_modeling_nemotron_nano_v2_vl.py."""
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
        # rather than a padded bucket.
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
# Dense-8B CG/eager/overlap logprob equality was not ported: Nano-30B-A3B is
# MoE and flips greedy tokens (and fails absolute / cosine logit checks) under
# tiny numeric drift in L0. Product accuracy stays on GSM8K/MMLU; the tests
# below exercise the real-weight hybrid / CG / overlap / chunked-prefill paths.
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

    See test_nemotron_h_cuda_graph_overlap_scheduler for real-weight CG /
    overlap path coverage on Nano.
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
    """Real-weight Nano smoke: eager, CUDA-graph, and CG+overlap all generate.

    Ported from dense 8B equality checks onto Nano-30B-BF16. MoE greedy /
    logit equality is too noisy for L0 (CG vs eager and overlap on vs off
    both flip tokens), so this only verifies the hybrid path runs under each
    config. Distribution / SSM-cache numeric equality belongs in a follow-up
    with MoE-stable refs or a denser in-scope checkpoint.
    """
    prompts = ["The sky is blue because"]
    batch_size = len(prompts)
    cg_batch_sizes = [batch_size]
    sampling_config = SamplingParams(max_tokens=1, temperature=0.0)

    configs = (
        ("eager", False, True, None),
        ("cg", True, True, cg_batch_sizes),
        ("cg_overlap", True, False, cg_batch_sizes),
    )
    for label, use_cuda_graph, disable_overlap, cg_sizes in configs:
        with create_nemotron_h_llm(
                model_folder=_NANO_30B_BF16,
                use_cuda_graph=use_cuda_graph,
                disable_overlap_scheduler=disable_overlap,
                max_batch_size=batch_size,
                cuda_graph_batch_sizes=cg_sizes,
        ) as llm:
            outputs = llm.generate(prompts,
                                   sampling_params=sampling_config,
                                   use_tqdm=True)
        assert len(outputs) == 1, f"{label}: expected one response"
        assert len(outputs[0].outputs[0].token_ids
                   ) > 0, f"{label}: produced empty output"
        assert len(
            outputs[0].outputs[0].text) > 0, f"{label}: produced empty text"


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


# Breakable prefill CUDA graphs (BCG) on the hybrid path. Under BCG the Mamba2
# layers run through the eager_on_graph boundary op (mamba2_mixer.py), so this
# covers the NemotronH-specific bridge rather than the generic BCG runner
# (which unittest/_torch/executor/test_breakable_cuda_graph.py exercises).
_BCG_MAX_NUM_TOKENS = 256
_BCG_CAPTURE_NUM_TOKENS = [64, 128, 256]
# Probability-ratio bound shared with ray_orchestrator/multi_gpu/
# test_accuracy_with_allreduce_strategy.py::compare_logprobs (e^-2.30 ~ 0.1x).
_BCG_LOGPROB_TOLERANCE = 2.30
# Context batches: an exact capture bucket, one token past a bucket (padded
# replay), two context requests of unequal length in one batch, and a prompt
# longer than max_num_tokens (chunked prefill: 256 + 44).
_BCG_CONTEXT_BATCHES = [
    [[17] * 128],
    [[17] * 129],
    [[17] * 64, [23] * 65],
    [[31] * 300],
]
# The mixed-batch section adds the streaming decode request and the context
# request admitted while it decodes.
_BCG_MIXED_BATCH_REQUESTS = 2
# Decode length of that streaming request: with EOS ignored this is a ~1 s
# generation window, orders of magnitude longer than the client-side gap
# before the second request is submitted.
_BCG_STREAM_MAX_TOKENS = 64
# One (token_ids, logprobs) entry per request from each backend run.
_BCG_NUM_REQUESTS = (sum(len(batch) for batch in _BCG_CONTEXT_BATCHES) +
                     _BCG_MIXED_BATCH_REQUESTS)


def _first_step_logprobs(output) -> torch.Tensor:
    """Log-probabilities of the first generated position for one request."""
    return torch.log_softmax(
        output.outputs[0].generation_logits[0].float().cpu(), dim=-1)


def _assert_mixed_batch_overlap(decoding_out, admitted_out) -> None:
    """Prove that `admitted` was prefilled while `decoding` was still decoding.

    Both requests return per-request iteration metrics taken from the engine's
    iteration counter. With the overlap scheduler disabled, a batch capacity of
    4 and 65 + 1 tokens inside the token budget, the scheduler runs every active
    generation request each iteration, so admitted's first iteration (its
    context chunk) falling inside decoding's generation window means that
    iteration carried a context chunk and a decode token together.
    """
    dec = decoding_out.outputs[0].request_perf_metrics
    adm = admitted_out.outputs[0].request_perf_metrics
    assert dec is not None and adm is not None, "request_perf_metrics missing"
    assert None not in (dec.first_iter, dec.last_iter, adm.first_iter), (
        f"iteration metrics not populated: decoding {dec.first_iter}.."
        f"{dec.last_iter}, admitted {adm.first_iter}")
    assert dec.first_iter < adm.first_iter <= dec.last_iter, (
        f"no mixed batch: decoding ran iterations [{dec.first_iter}, "
        f"{dec.last_iter}] but admitted was prefilled at iteration "
        f"{adm.first_iter}")


def _run_nemotron_h_prefill_backend(backend: PrefillCudaGraphBackend,
                                    tp_size: int):
    """Run the fixed prompt schedule with one prefill CUDA graph backend.

    Returns one (token_ids, first_step_logprobs) pair per request, in a
    deterministic order shared by both backends.
    """
    # ignore_eos on every compared request: the two backends may legitimately
    # pick different greedy tokens after the first step (see the test
    # docstring), and if one of them is EOS the arms would stop at different
    # lengths for a difference the test otherwise accepts.
    sampling_params = SamplingParams(max_tokens=4,
                                     temperature=0.0,
                                     ignore_eos=True,
                                     return_generation_logits=True)
    per_request = []
    with LLM(
            model=f"{llm_models_root(check=True)}/{_NANO_30B_BF16}",
            tensor_parallel_size=tp_size,
            # Pin NCCL so the allreduce path under BCG is the plain one on
            # every platform (AUTO may pick MNNVL on multi-node NVLink).
            allreduce_strategy="NCCL",
            max_batch_size=4,
            max_num_tokens=_BCG_MAX_NUM_TOKENS,
            enable_chunked_prefill=True,
            disable_overlap_scheduler=True,
            gather_generation_logits=True,
            kv_cache_config=KvCacheConfig(mamba_ssm_cache_dtype="float32"),
            cuda_graph_config=CudaGraphConfig(enable_padding=True,
                                              max_batch_size=4),
            prefill_cuda_graph_backend=backend,
            prefill_capture_num_tokens=_BCG_CAPTURE_NUM_TOKENS,
    ) as llm:
        # The validator must not have downgraded the requested backend.
        assert llm.args.prefill_cuda_graph_backend == backend
        for batch in _BCG_CONTEXT_BATCHES:
            for output in llm.generate(batch, sampling_params=sampling_params):
                per_request.append((list(output.outputs[0].token_ids),
                                    _first_step_logprobs(output)))

        # Mixed batch: admit a context request while another request is
        # decoding, so BCG replays a batch that carries both a context chunk
        # and decode tokens. The overlap is proved afterwards from the two
        # requests' iteration metrics rather than assumed.
        decoding = llm.generate_async([17] * 128,
                                      sampling_params=SamplingParams(
                                          max_tokens=_BCG_STREAM_MAX_TOKENS,
                                          temperature=0.0,
                                          ignore_eos=True,
                                          return_generation_logits=True,
                                          return_perf_metrics=True),
                                      streaming=True)
        # In streaming mode every response carries only the newest step's
        # logits, so the first-step distribution must be read from the first
        # streamed response, before the stream moves on.
        next(decoding)
        decoding_first_step_logprobs = _first_step_logprobs(decoding)
        admitted = llm.generate_async([23] * 65,
                                      sampling_params=SamplingParams(
                                          max_tokens=4,
                                          temperature=0.0,
                                          ignore_eos=True,
                                          return_generation_logits=True,
                                          return_perf_metrics=True),
                                      streaming=False)
        decoding_out = decoding.result()
        admitted_out = admitted.result()
        _assert_mixed_batch_overlap(decoding_out, admitted_out)
        per_request.append((list(decoding_out.outputs[0].token_ids),
                            decoding_first_step_logprobs))
        per_request.append((list(admitted_out.outputs[0].token_ids),
                            _first_step_logprobs(admitted_out)))
    return per_request


@skip_gpu_memory_less_than((2 * 30 + 1) * 2**30)
@pytest.mark.mpi_ray_parity
@pytest.mark.parametrize("tp_size", [
    1,
    pytest.param(2, marks=skip_single_gpu),
],
                         ids=lambda n: f"tp{n}")
def test_nemotron_h_breakable_prefill_cuda_graph(tp_size):
    """Real-weight Nano: breakable prefill CUDA graphs vs eager prefill.

    Same context / chunked-prefill / mixed-admission schedule with
    prefill_cuda_graph_backend DISABLED and BREAKABLE. The first generated
    token is the direct product of the (captured) prefill, so its
    distribution is compared per request within the repo's accepted
    probability-ratio bound and BCG's greedy pick must be one of eager's top-2.
    Full-sequence greedy equality is reported but not required: Nano-30B-A3B
    is MoE and flips greedy tokens under padding-induced numeric drift (see
    the note above test_nemotron_h_sanity). NCCL allreduce is pinned: tp1 has
    no allreduce at all, tp2 (skipped below 2 GPUs) runs the NCCL allreduce inside the captured
    segments. Runs under the MPI executor and, with --run-ray, under the Ray
    executor (mpi_ray_parity). The mixed-admission overlap is proved from the
    two requests' iteration metrics, not assumed.
    """
    eager = _run_nemotron_h_prefill_backend(PrefillCudaGraphBackend.DISABLED,
                                            tp_size)
    bcg = _run_nemotron_h_prefill_backend(PrefillCudaGraphBackend.BREAKABLE,
                                          tp_size)
    assert len(eager) == len(bcg) == _BCG_NUM_REQUESTS

    identical = 0
    worst_diff = 0.0
    for i, ((eager_ids, eager_lp), (bcg_ids,
                                    bcg_lp)) in enumerate(zip(eager, bcg)):
        assert len(bcg_ids) == len(eager_ids) > 0, (
            f"request {i}: BCG produced {len(bcg_ids)} tokens, "
            f"eager {len(eager_ids)}")
        eager_top1 = eager_ids[0]
        diff = (eager_lp[eager_top1] - bcg_lp[eager_top1]).abs().item()
        worst_diff = max(worst_diff, diff)
        assert diff < _BCG_LOGPROB_TOLERANCE, (
            f"request {i}: BCG log-prob of eager's first token differs by "
            f"{diff:.3f} nats (bound {_BCG_LOGPROB_TOLERANCE})")
        eager_top2 = torch.topk(eager_lp, 2).indices.tolist()
        assert bcg_ids[0] in eager_top2, (
            f"request {i}: BCG first token {bcg_ids[0]} not in eager top-2 "
            f"{eager_top2}")
        identical += int(bcg_ids == eager_ids)
    print(f"BCG vs eager: {identical}/{len(eager)} sequences identical, "
          f"worst first-token log-prob diff {worst_diff:.3f} nats")
