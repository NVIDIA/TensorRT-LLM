import torch

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig

# isort: off
from tensorrt_llm._torch.models.modeling_nemotron_h import (NemotronHConfig,
                                                            NemotronHForCausalLM
                                                            )
# isort: on
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from utils.llm_data import llm_models_root
from utils.util import skip_gpu_memory_less_than

from tensorrt_llm._torch import LLM
from tensorrt_llm._torch.pyexecutor.model_engine import load_weights
from tensorrt_llm._torch.pyexecutor.resource_manager import \
    MambaHybridCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig as KvCacheConfigCpp
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.sampling_params import SamplingParams


def get_logprobs(token_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    raw_probs = torch.softmax(logits, dim=-1)
    index = token_ids.unsqueeze(1).cuda()
    token_probs = torch.gather(raw_probs, dim=1, index=index).squeeze(-1)
    return torch.log(token_probs)


def _generate(
    model: NemotronHForCausalLM, tokenizer: PreTrainedTokenizerBase,
    cache: MambaHybridCacheManager, text_prompts: list[str],
    tokens_to_generate: int, device: torch.device
) -> tuple[list[int], list[list[int]], list[list[float]]]:
    num_seqs = len(text_prompts)
    all_token_ids = [
        tokenizer.encode(prompt, add_special_tokens=False)
        for prompt in text_prompts
    ]
    input_ids = torch.cat([
        torch.tensor(token_ids, dtype=torch.int64, device=device)
        for token_ids in all_token_ids
    ],
                          dim=0)
    request_ids = list(range(1, num_seqs + 1))
    prompt_lens = [len(token_ids) for token_ids in all_token_ids]

    requests = cache.add_dummy_requests(request_ids, prompt_lens)
    cache.prepare_mamba_cache_blocks(request_ids)

    metadata_cls = get_attention_backend(
        model.model_config.attn_backend).Metadata
    attn_metadata = metadata_cls(
        seq_lens=torch.tensor(prompt_lens, dtype=torch.int),
        num_contexts=num_seqs,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[0] * num_seqs,
        ),
        max_num_requests=num_seqs,
        max_num_tokens=8192,
        kv_cache_manager=cache,
        request_ids=request_ids,
        prompt_lens=prompt_lens,
    )

    # prefill
    position_ids = [torch.arange(0, prompt_len) for prompt_len in prompt_lens]
    position_ids = torch.cat(position_ids).unsqueeze(0).cuda()
    with torch.inference_mode():
        attn_metadata.prepare()
        logits = model.forward(input_ids=input_ids,
                               position_ids=position_ids,
                               attn_metadata=attn_metadata,
                               return_context_logits=True)

    # compute logprobs from logits
    all_logits = logits.split(prompt_lens, dim=0)
    all_logprobs = [
        get_logprobs(
            torch.tensor(token_ids[1:], dtype=torch.int64, device=device),
            this_logits[:-1]).tolist()
        for token_ids, this_logits in zip(all_token_ids, all_logits)
    ]

    if tokens_to_generate > 0:
        # sample token greedily
        sampled_tokens = torch.cat([
            torch.argmax(this_logits[-1]).unsqueeze(0)
            for this_logits in all_logits
        ],
                                   dim=0)
        for i in range(num_seqs):
            all_token_ids[i].append(sampled_tokens[i].item())
            all_logprobs[i].append(
                get_logprobs(sampled_tokens[i].unsqueeze(0),
                             all_logits[i][-1:]).item())

        # one token already generated at prefill
        for i in range(tokens_to_generate - 1):
            num_cached_tokens_per_seq = [
                prompt_len + i for prompt_len in prompt_lens
            ]
            position_ids = torch.tensor([num_cached_tokens_per_seq],
                                        dtype=torch.int64,
                                        device=device)

            attn_metadata = metadata_cls(
                seq_lens=torch.tensor([1] * num_seqs, dtype=torch.int),
                num_contexts=0,
                kv_cache_params=KVCacheParams(
                    use_cache=True,
                    num_cached_tokens_per_seq=num_cached_tokens_per_seq,
                ),
                max_num_requests=num_seqs,
                max_num_tokens=8192,
                kv_cache_manager=cache,
                request_ids=request_ids,
                prompt_lens=prompt_lens,
            )

            with torch.inference_mode():
                attn_metadata.prepare()
                logits = model.forward(input_ids=sampled_tokens,
                                       position_ids=position_ids,
                                       attn_metadata=attn_metadata)

            # sample token greedily
            sampled_tokens = torch.argmax(logits, dim=-1, keepdim=False)
            for i in range(num_seqs):
                all_token_ids[i].append(sampled_tokens[i].item())
                all_logprobs[i].append(
                    get_logprobs(sampled_tokens[i].unsqueeze(0),
                                 logits[i].unsqueeze(0)).item())

    for req in requests:
        cache.free_resources(req)

    return prompt_lens, all_token_ids, all_logprobs


def generate(
    model: NemotronHForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    cache: MambaHybridCacheManager,
    text_prompts: list[str],
    tokens_to_generate: int,
    device: torch.device,
    one_by_one: bool = False
) -> tuple[list[int], list[list[int]], list[list[float]]]:
    """
    Generate `tokens_to_generate` tokens from the given prompts using the given model and cache.
    Return the prompt_lens along with the prefill+generated tokens and their logprobs, minus the first token in the prompt.
    """
    if one_by_one:
        num_prompts = len(text_prompts)
        prompt_lens, tokens, logprobs = [None] * num_prompts, [
            None
        ] * num_prompts, [None] * num_prompts
        for i in range(num_prompts):
            p, t, l = _generate(model, tokenizer, cache, [text_prompts[i]],
                                tokens_to_generate, device)
            prompt_lens[i], tokens[i], logprobs[i] = p[0], t[0], l[0]
        return prompt_lens, tokens, logprobs
    return _generate(model, tokenizer, cache, text_prompts, tokens_to_generate,
                     device)


@skip_gpu_memory_less_than(
    (2 * 8 + 1) * 2**30)  # 8B, bf16, plus 1 GB for good measure
def test_nemotron_h_correctness():
    model_dir = f"{llm_models_root(check=True)}/Nemotron-H-8B-Base-8K"
    nemotron_h_config = NemotronHConfig.from_pretrained(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    dtype = nemotron_h_config.torch_dtype
    device = torch.device('cuda')
    assert dtype == torch.bfloat16
    kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16

    model_config = ModelConfig(pretrained_config=nemotron_h_config)
    nemotron_h = NemotronHForCausalLM(model_config).to(device)

    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    weights = load_weights(model_dir)
    nemotron_h.load_weights(weights)

    text_prompts = [
        "The future of AI is",
        "The president of the United States is",
    ]
    num_prompts = len(text_prompts)

    num_blocks = 100
    tokens_per_block = 128
    head_dim = nemotron_h.config.hidden_size // nemotron_h.config.num_attention_heads
    num_layers = nemotron_h.config.hybrid_override_pattern.count("*")
    layer_mask = [
        char == "*" for char in nemotron_h.config.hybrid_override_pattern
    ]
    mamba_num_layers = nemotron_h.config.hybrid_override_pattern.count("M")
    mamba_layer_mask = [
        char == "M" for char in nemotron_h.config.hybrid_override_pattern
    ]
    num_kv_heads = nemotron_h.config.num_key_value_heads
    max_seq_len = num_blocks * tokens_per_block
    max_batch_size = num_prompts

    kv_cache_config = KvCacheConfigCpp(max_tokens=num_blocks * tokens_per_block,
                                       enable_block_reuse=False)
    kv_cache_manager = MambaHybridCacheManager(
        # mamba cache parameters
        nemotron_h.config.hidden_size,
        nemotron_h.config.ssm_state_size,
        nemotron_h.config.conv_kernel,
        nemotron_h.config.expand,
        nemotron_h.config.n_groups,
        nemotron_h.config.mamba_head_dim,
        mamba_num_layers,
        mamba_layer_mask,
        nemotron_h.config.torch_dtype,
        # kv cache parameters
        kv_cache_config,
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=num_layers,
        layer_mask=layer_mask,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        mapping=mapping,
        dtype=kv_cache_dtype,
    )

    prompt_lens, tokens_no_batching, logprobs_no_batching = generate(
        model=nemotron_h,
        tokenizer=tokenizer,
        cache=kv_cache_manager,
        text_prompts=text_prompts,
        tokens_to_generate=9,
        device=torch.device("cuda"),
        one_by_one=True)
    completions_no_batching = [
        tokenizer.decode(tokens_no_batching[i][prompt_lens[i]:])
        for i in range(num_prompts)
    ]

    _, tokens_batching, logprobs_batching = generate(
        model=nemotron_h,
        tokenizer=tokenizer,
        cache=kv_cache_manager,
        text_prompts=text_prompts,
        tokens_to_generate=9,
        device=torch.device("cuda"))
    completions_batching = [
        tokenizer.decode(tokens_batching[i][prompt_lens[i]:])
        for i in range(num_prompts)
    ]

    # reference logprobs for first prompt from mcore for prompt minus first token
    # TODO(oargov): generate a reference on-the-fly once we have confidence in the HF impl
    prefill_logprobs_ref_mcore = torch.tensor([
        -7.415980815887451, -0.36192911863327026, -2.8658294677734375,
        -2.316344738006592
    ])

    # compare logprobs with mcore logprobs, check that the max error is less than 0.3
    mcore_atol = 0.3
    torch.testing.assert_close(torch.tensor(
        logprobs_no_batching[0][:prompt_lens[0] - 1]),
                               prefill_logprobs_ref_mcore,
                               atol=mcore_atol,
                               rtol=0.0)
    print(
        f"max mcore prefill diff: {torch.max(torch.abs(torch.tensor(logprobs_no_batching[0][:prompt_lens[0] - 1]) - prefill_logprobs_ref_mcore))}"
    )
    print()

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

    expected_completions = [
        " bright, with endless possibilities for innovation and growth",
        " the head of state and head of government of",
    ]

    for i in range(num_prompts):
        prefill_logprobs_no_batching = torch.tensor(
            logprobs_no_batching[i][:prompt_lens[i] - 1])
        decode_logprobs_no_batching = torch.tensor(
            logprobs_no_batching[i][prompt_lens[i] - 1:])

        prefill_logprobs_batching = torch.tensor(
            logprobs_batching[i][:prompt_lens[i] - 1])
        decode_logprobs_batching = torch.tensor(
            logprobs_batching[i][prompt_lens[i] - 1:])

        # compare prompt logprobs with initial implementation
        torch.testing.assert_close(prefill_logprobs_no_batching,
                                   prefill_logprobs_ref_initial_no_batching[i],
                                   atol=initial_impl_atol,
                                   rtol=0.0)
        print(
            f"max prefill without batching diff: {torch.max(torch.abs(prefill_logprobs_no_batching - prefill_logprobs_ref_initial_no_batching[i]))}"
        )
        torch.testing.assert_close(
            prefill_logprobs_batching,
            prefill_logprobs_ref_initial_with_batching[i],
            atol=initial_impl_atol,
            rtol=0.0)
        print(
            f"max prefill with batching diff: {torch.max(torch.abs(prefill_logprobs_batching - prefill_logprobs_ref_initial_with_batching[i]))}"
        )

        # compare expected completion
        assert completions_batching[i] == expected_completions[i]
        assert completions_no_batching[i] == expected_completions[i]

        # compare decode logprobs with initial implementation
        torch.testing.assert_close(decode_logprobs_no_batching,
                                   decode_logprobs_ref_initial_no_batching[i],
                                   atol=initial_impl_atol,
                                   rtol=0.0)
        print(
            f"max decode without batching diff: {torch.max(torch.abs(decode_logprobs_no_batching - decode_logprobs_ref_initial_no_batching[i]))}"
        )
        torch.testing.assert_close(decode_logprobs_batching,
                                   decode_logprobs_ref_initial_with_batching[i],
                                   atol=initial_impl_atol,
                                   rtol=0.0)
        print(
            f"max decode with batching diff: {torch.max(torch.abs(decode_logprobs_batching - decode_logprobs_ref_initial_with_batching[i]))}"
        )

        # compare logprobs with and without batching, tolerace by diff in initial implementation
        torch.testing.assert_close(prefill_logprobs_batching,
                                   prefill_logprobs_no_batching,
                                   atol=batching_atol,
                                   rtol=0.0)
        print(
            f"max batching diff (prefill): {torch.max(torch.abs(prefill_logprobs_batching - prefill_logprobs_no_batching))}"
        )
        torch.testing.assert_close(decode_logprobs_batching,
                                   decode_logprobs_no_batching,
                                   atol=batching_atol,
                                   rtol=0.0)
        print(
            f"max batching diff (decode): {torch.max(torch.abs(decode_logprobs_batching - decode_logprobs_no_batching))}"
        )
        print()

    # now let's test that decodes match prefill logprobs
    text_prompts_with_completions = [
        f"{text_prompts[i]}{completions_batching[i]}"
        for i in range(num_prompts)
    ]
    _, _, full_sequence_logprobs = generate(
        model=nemotron_h,
        tokenizer=tokenizer,
        cache=kv_cache_manager,
        text_prompts=text_prompts_with_completions,
        tokens_to_generate=0,
        device=torch.device("cuda"))

    # compare full sequence logprobs with prefill+decode logprobs, tolerance like mcore tolerance
    for i in range(num_prompts):
        torch.testing.assert_close(torch.tensor(full_sequence_logprobs[i]),
                                   torch.tensor(logprobs_batching[i]),
                                   atol=mcore_atol,
                                   rtol=0.0)
        print(
            f"max full sequence diff: {torch.max(torch.abs(torch.tensor(full_sequence_logprobs[i]) - torch.tensor(logprobs_batching[i])))}"
        )

    kv_cache_manager.shutdown()

    # clear memory before next test
    del nemotron_h
    torch.cuda.empty_cache()


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
def test_nemotron_h_correctness_2():
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
        print(
            f"max mcore prefill diff: {torch.max(torch.abs(prefill_logprobs_no_batching[0] - prefill_logprobs_ref_mcore))}"
        )
        print()

        for i in range(num_prompts):
            # compare prompt logprobs with initial implementation
            torch.testing.assert_close(
                prefill_logprobs_no_batching[i],
                prefill_logprobs_ref_initial_no_batching[i],
                atol=initial_impl_atol,
                rtol=0.0)
            print(
                f"max prefill without batching diff: {torch.max(torch.abs(prefill_logprobs_no_batching[i] - prefill_logprobs_ref_initial_no_batching[i]))}"
            )
            torch.testing.assert_close(
                prefill_logprobs_batching[i],
                prefill_logprobs_ref_initial_with_batching[i],
                atol=initial_impl_atol,
                rtol=0.0)
            print(
                f"max prefill with batching diff: {torch.max(torch.abs(prefill_logprobs_batching[i] - prefill_logprobs_ref_initial_with_batching[i]))}"
            )

            # compare expected completion
            assert completions_batching[i] == expected_completions[i]
            assert completions_no_batching[i] == expected_completions[i]

            # compare decode logprobs with initial implementation
            torch.testing.assert_close(
                decode_logprobs_no_batching[i],
                decode_logprobs_ref_initial_no_batching[i],
                atol=initial_impl_atol,
                rtol=0.0)
            print(
                f"max decode without batching diff: {torch.max(torch.abs(decode_logprobs_no_batching[i] - decode_logprobs_ref_initial_no_batching[i]))}"
            )
            torch.testing.assert_close(
                decode_logprobs_batching[i],
                decode_logprobs_ref_initial_with_batching[i],
                atol=initial_impl_atol,
                rtol=0.0)
            print(
                f"max decode with batching diff: {torch.max(torch.abs(decode_logprobs_batching[i] - decode_logprobs_ref_initial_with_batching[i]))}"
            )

            # compare logprobs with and without batching, tolerace by diff in initial implementation
            torch.testing.assert_close(prefill_logprobs_batching[i],
                                       prefill_logprobs_no_batching[i],
                                       atol=batching_atol,
                                       rtol=0.0)
            print(
                f"max batching diff (prefill): {torch.max(torch.abs(prefill_logprobs_batching[i] - prefill_logprobs_no_batching[i]))}"
            )
            torch.testing.assert_close(decode_logprobs_batching[i],
                                       decode_logprobs_no_batching[i],
                                       atol=batching_atol,
                                       rtol=0.0)
            print(
                f"max batching diff (decode): {torch.max(torch.abs(decode_logprobs_batching[i] - decode_logprobs_no_batching[i]))}"
            )
            print()

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
            print(
                f"max full sequence diff: {torch.max(torch.abs(full_sequence_logprobs[i] - prefill_decode_logprobs))}"
            )
    finally:
        nemotron_h.shutdown()


# TODO: once LLM API supports context and generation logits, use it in above test and remove this one
@skip_gpu_memory_less_than(
    (2 * 8 + 1) * 2**30)  # 8B, bf16, plus 1 GB for good measure
def test_nemotron_h_llm_api():
    model_dir = f"{llm_models_root(check=True)}/Nemotron-H-8B-Base-8K"
    text_prompts = [
        "The future of AI is",
        "The president of the United States is",
    ]
    num_prompts = len(text_prompts)

    nemotron_h = LLM(
        model=model_dir,
        use_cuda_graph=False,
        max_batch_size=num_prompts,
        kv_cache_config=KvCacheConfig(enable_block_reuse=False),
    )

    expected_completions = [
        " bright, with endless possibilities for innovation and growth",
        " the head of state and head of government of",
    ]

    sampling_params = SamplingParams(max_tokens=9,
                                     temperature=0.0,
                                     add_special_tokens=False)

    try:
        results = nemotron_h.generate(text_prompts, sampling_params)
        for result, expected_completion in zip(results, expected_completions):
            assert result.outputs[0].text == expected_completion
    finally:
        nemotron_h.shutdown()
