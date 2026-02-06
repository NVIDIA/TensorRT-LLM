import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from test_common.llm_data import with_mocked_hf_download_for_single_gpu
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm.llmapi import (CudaGraphConfig, Eagle3DecodingConfig,
                                 KvCacheConfig)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture(scope="function")
def enforce_single_worker(monkeypatch):
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    yield


def test_kv_lens_runtime_with_eagle3_one_model():
    """
    Validates that kv_lens_runtime correctly excludes num_extra_kv_tokens when
    preparing attention metadata during EAGLE3 one-model speculative decoding.

    Background:
    - EAGLE3 reserves num_extra_kv_tokens = max_draft_len - 1 in KV cache for draft token management
    - kv_lens_runtime becomes host_past_key_value_lengths, which eventually becomes mMaxSeqLenKv in FMHA kernel
    - Bug: mMaxSeqLenKv was incorrectly set to actual_kv_length + num_extra_kv_tokens
    - Fix: mMaxSeqLenKv should be set to actual_kv_length only (without extra tokens)

    This test validates the fix by directly testing the prepare() logic.
    """

    # Test parameters
    num_seqs = 3
    num_extra_kv_tokens = 7  # e.g., max_draft_len = 8, so extra = 7
    prompt_lens = [50, 100, 75]  # These represent actual KV lengths
    seq_lens_q = [1, 1, 1]  # 1 token each in generation
    num_cached_tokens_per_seq = [
        prompt_lens[i] - seq_lens_q[i] for i in range(num_seqs)
    ]

    # Create a mock KV cache manager
    mock_kv_cache_manager = MagicMock()
    mock_kv_cache_manager.tokens_per_block = 32
    mock_kv_cache_manager.num_pools = 1
    mock_kv_cache_manager.max_blocks_per_seq = 16
    mock_kv_cache_manager.max_batch_size = num_seqs
    mock_kv_cache_manager.max_seq_len = 512  # Large enough to hold our test sequences
    mock_kv_cache_manager.copy_batch_block_offsets = MagicMock()

    attn_metadata = TrtllmAttentionMetadata(
        max_num_requests=num_seqs,
        max_num_tokens=sum(seq_lens_q),
        kv_cache_manager=mock_kv_cache_manager,
    )

    # Set required attributes
    attn_metadata.request_ids = list(range(1, num_seqs + 1))
    attn_metadata.prompt_lens = prompt_lens
    attn_metadata._seq_lens = torch.tensor(seq_lens_q, dtype=torch.int32)
    # seq_lens_kv is the number of new KV tokens being added in this step (for generation, same as seq_lens_q)
    attn_metadata._seq_lens_kv = torch.tensor(seq_lens_q, dtype=torch.int32)

    # Set KV cache params with num_extra_kv_tokens (EAGLE3 one-model case)
    attn_metadata.kv_cache_params = KVCacheParams(
        use_cache=True,
        num_cached_tokens_per_seq=num_cached_tokens_per_seq,
        num_extra_kv_tokens=num_extra_kv_tokens)

    attn_metadata.prepare()
    actual_kv_lengths = torch.tensor(prompt_lens, dtype=torch.int32)

    # kv_lens_runtime should equal actual KV lengths (without extra tokens)
    kv_lens_runtime = attn_metadata.kv_lens_runtime[:num_seqs]
    assert torch.equal(kv_lens_runtime, actual_kv_lengths), \
        f"kv_lens_runtime should be {actual_kv_lengths.tolist()}, but got {kv_lens_runtime.tolist()}"

    # Internal kv_lens should include extra tokens
    kv_lens_internal = attn_metadata.kv_lens[:num_seqs]
    expected_kv_lens_with_extra = actual_kv_lengths + num_extra_kv_tokens
    assert torch.equal(kv_lens_internal, expected_kv_lens_with_extra), \
        f"kv_lens should be {expected_kv_lens_with_extra.tolist()}, but got {kv_lens_internal.tolist()}"


@pytest.mark.skip(reason="https://nvbugs/5856637")
@pytest.mark.parametrize(
    "use_cuda_graph,attn_backend,disable_overlap_scheduler,enable_block_reuse,use_one_model,enable_chunked_prefill,use_chain_drafter,multi_batch,attention_dp,use_hf_speculative_model",
    [
        [True, "TRTLLM", True, False, False, False, True, False, False, False],
        [True, "TRTLLM", True, False, False, False, False, False, False, False],
        [False, "TRTLLM", True, False, False, False, True, False, False, False],
        [
            False, "TRTLLM", True, False, False, False, False, False, False,
            False
        ],
        [
            True, "FLASHINFER", True, False, False, False, True, False, False,
            False
        ],
        [
            False, "FLASHINFER", True, False, False, False, True, False, False,
            False
        ],
        [False, "TRTLLM", False, True, True, False, True, False, False, False],
        [True, "TRTLLM", False, True, True, False, True, False, False, False],
        [True, "TRTLLM", True, False, True, True, True, False, False, False],
        [True, "TRTLLM", True, False, True, False, True, False, False, False],
        [True, "TRTLLM", True, False, False, True, True, False, False, False],
        [True, "TRTLLM", False, False, False, False, True, False, False, False],
        [
            False, "TRTLLM", False, False, False, False, True, False, False,
            False
        ],
        [True, "TRTLLM", False, False, False, False, False, True, False, False],
        [True, "TRTLLM", False, False, False, False, False, True, True, False],
        [
            False, "TRTLLM", False, False, False, False, False, True, False,
            False
        ],
        [True, "TRTLLM", False, False, False, False, True, True, False, False],
        [False, "TRTLLM", False, False, False, False, True, True, False, False],
        [
            True, "TRTLLM", False, False, False, False, False, False, False,
            False
        ],
        [
            False, "TRTLLM", False, False, False, False, False, False, False,
            False
        ],
        [True, "TRTLLM", False, False, False, True, True, False, False, False],
        [True, "TRTLLM", False, False, False, True, False, False, False, False],
        [
            True, "FLASHINFER", False, False, False, False, True, False, False,
            False
        ],
        [
            False, "FLASHINFER", False, False, False, False, True, False, False,
            False
        ],
        # Tests (mocked) speculative model auto-download from HuggingFace
        [False, "TRTLLM", True, False, False, False, True, False, False, True],
    ])
@pytest.mark.high_cuda_memory
@with_mocked_hf_download_for_single_gpu
def test_llama_eagle3(use_cuda_graph: bool, attn_backend: str,
                      disable_overlap_scheduler: bool, enable_block_reuse: bool,
                      use_one_model: bool, enable_chunked_prefill: bool,
                      use_chain_drafter: bool, multi_batch: bool,
                      attention_dp: bool, use_hf_speculative_model: bool,
                      request):
    # Eagle3 one model works with overlap scheduler and block reuse.
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 35:
        pytest.skip("Not enough memory to load target + draft model")

    models_path = llm_models_root()
    target_model_dir = f"{models_path}/llama-3.1-model/Llama-3.1-8B-Instruct"

    if use_hf_speculative_model:
        eagle_model = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
    else:
        eagle_model = f"{models_path}/EAGLE3-LLaMA3.1-Instruct-8B"

    # bs > 1 gives non-deterministic when doing IFB. There are slight chances
    # that ref and spec does not match 100%
    max_batch_size = 4 if multi_batch else 1
    max_draft_len = 4
    kv_cache_config = KvCacheConfig(enable_block_reuse=enable_block_reuse,
                                    max_tokens=8192)
    cuda_graph_config = CudaGraphConfig(
        batch_sizes=[i for i in range(1, max_batch_size +
                                      1)]) if use_cuda_graph else None

    llm_common_config = dict(
        model=target_model_dir,
        attn_backend=attn_backend,
        disable_overlap_scheduler=disable_overlap_scheduler,
        cuda_graph_config=cuda_graph_config,
        max_batch_size=max_batch_size,
        kv_cache_config=kv_cache_config,
        enable_attention_dp=attention_dp,
        # This max_seq_len is larger than the one specified
        # in the llama 3 8B eagle's config. We want to make sure
        # that the draft model won't go above its max in warmup
        # in this test.
        max_seq_len=8192,
        enable_chunked_prefill=enable_chunked_prefill,
    )
    if enable_chunked_prefill:
        # Use a small max_num_tokens so that the chunked prefill path gets exercised.
        llm_common_config['max_num_tokens'] = 64

    spec_config = Eagle3DecodingConfig(
        max_draft_len=max_draft_len,
        speculative_model=eagle_model,
        # Llama 3 does not support one model eagle.
        eagle3_one_model=use_one_model,
    )
    spec_config._allow_chain_drafter = use_chain_drafter

    # Create the LLM instance
    llm_spec = LLM(**llm_common_config, speculative_config=spec_config)

    # Acceptance rate tests
    if enable_chunked_prefill:
        # Use a long prompt for chunked prefill tests.
        prompts = [
            "The capital of France is a city of romance, art, fashion, and cuisine. Paris is a must-visit destination for anyone who loves history, architecture, and culture. From the iconic Eiffel Tower to the world-famous Louvre Museum, Paris has something to offer for every interest and age.\nThe city is divided into 20 arrondissements, each with its own unique character and charm. The Latin Quarter is a popular area for students and young travelers, while the Champs-Élysées is a hub for shopping and dining. The Montmartre neighborhood is famous for its bohemian vibe and stunning views of the city.\nParis is also known for its beautiful parks and gardens, such as the Luxembourg Gardens and the Tuileries Garden. The city has a rich history, with landmarks like the Notre-Dame Cathedral and the Arc de Triomphe. Visitors can also explore the city's many museums, including the Musée d'Orsay and the Musée Rodin.\nIn addition to its cultural and historical attractions, Paris is also a great destination for foodies. The city is famous for its cuisine, including croissants, baguettes, and cheese. Visitors can sample the city's famous dishes at one of the many restaurants, cafes, and "
        ]
        tok_ids = [llm_spec.tokenizer.encode(prompts[0])]
    else:
        prompts = [
            "The capital of France is",
            "The president of the United States is",
        ]
        tok_ids = [llm_spec.tokenizer.encode("The future of AI is")]
        if multi_batch:
            tok_ids.append(llm_spec.tokenizer.encode(prompts))

    sampling_params = SamplingParams(max_tokens=128, temperature=0)

    for i in range(len(tok_ids)):
        num_tokens = 0
        num_drafted = 0
        num_accepted = 0

        for output in llm_spec.generate_async(tok_ids[i],
                                              sampling_params,
                                              streaming=True):
            new_tokens = output.outputs[0].token_ids
            num_drafted += max_draft_len
            num_accepted += len(new_tokens) - num_tokens - 1
            num_tokens = len(new_tokens)

        accept_rate = num_accepted / num_drafted
        assert accept_rate > 0.10

    # Output tests
    sampling_params = SamplingParams(max_tokens=10, temperature=0)

    results_spec = llm_spec.generate(prompts, sampling_params)
    generated_text_spec = [result.outputs[0].text for result in results_spec]
    llm_spec.shutdown()

    llm_ref = LLM(**llm_common_config)
    results_ref = llm_ref.generate(prompts, sampling_params)
    generated_text_ref = [result.outputs[0].text for result in results_ref]
    llm_ref.shutdown()

    for text_spec, text_ref in zip(generated_text_spec, generated_text_ref):
        # The spec decode algorithm currently guarantees identical results
        assert text_spec == text_ref


@pytest.mark.parametrize("eagle3_one_model", [True, False])
def test_eagle3_spec_decoding_stats(eagle3_one_model):
    """Test that specDecodingStats are correctly populated in metrics endpoint"""
    models_path = llm_models_root()
    eagle_model_dir = f"{models_path}/EAGLE3-LLaMA3.1-Instruct-8B"
    target_model_dir = f"{models_path}/llama-3.1-model/Llama-3.1-8B-Instruct"

    # Skip if models don't exist
    if not os.path.exists(target_model_dir) or not os.path.exists(
            eagle_model_dir):
        pytest.skip(f"Required models not found")

    kv_cache_config = KvCacheConfig(enable_block_reuse=False,
                                    free_gpu_memory_fraction=0.6)
    spec_config = Eagle3DecodingConfig(
        max_draft_len=3,
        speculative_model=eagle_model_dir,
        eagle3_one_model=eagle3_one_model,
    )

    with LLM(
            model=target_model_dir,
            speculative_config=spec_config,
            kv_cache_config=kv_cache_config,
            disable_overlap_scheduler=not eagle3_one_model,
            enable_iter_perf_stats=True,
            max_batch_size=4,
    ) as llm:
        # Generate some output to collect stats
        prompts = [
            "The capital of France is",
            "The president of the United States is",
        ]
        sampling_params = SamplingParams(max_tokens=20, temperature=0)
        llm.generate(prompts, sampling_params)

        # Get iteration stats
        stats = llm.get_stats(timeout=2)
        assert len(stats) > 0, "Should have iteration stats"

        # Find iterations with speculation (generation phase)
        iterations_with_spec = []
        for stat in stats:
            if 'specDecodingStats' in stat:
                spec_stats = stat['specDecodingStats']
                if spec_stats.get('numDraftTokens', 0) > 0:
                    iterations_with_spec.append(spec_stats)

        # Should have at least some iterations with spec decoding
        assert len(iterations_with_spec) > 0, \
            f"Should have iterations with specDecodingStats (found {len(iterations_with_spec)})"

        # Validate specDecodingStats structure and values
        for spec_stats in iterations_with_spec:
            # Check all fields are present
            assert 'numDraftTokens' in spec_stats
            assert 'numAcceptedTokens' in spec_stats
            assert 'numRequestsWithDraftTokens' in spec_stats
            assert 'acceptanceLength' in spec_stats
            assert 'iterLatencyMS' in spec_stats
            assert 'draftOverhead' in spec_stats

            # Validate value constraints
            assert spec_stats['numDraftTokens'] > 0
            assert 0 <= spec_stats['numAcceptedTokens'] <= spec_stats[
                'numDraftTokens']
            assert spec_stats['numRequestsWithDraftTokens'] > 0
            assert spec_stats['acceptanceLength'] >= 1.0
            assert spec_stats['iterLatencyMS'] >= 0.0
            assert 0.0 <= spec_stats['draftOverhead'] <= 1.0

        # Calculate overall acceptance rate
        total_draft = sum(s['numDraftTokens'] for s in iterations_with_spec)
        total_accepted = sum(s['numAcceptedTokens']
                             for s in iterations_with_spec)
        acceptance_rate = (total_accepted / total_draft *
                           100) if total_draft > 0 else 0

        # Should have reasonable acceptance rate for Eagle3
        assert acceptance_rate > 5.0, f"Acceptance rate too low: {acceptance_rate:.1f}%"


@pytest.mark.parametrize("use_cuda_graph", [True, False])
@pytest.mark.high_cuda_memory
def test_llama_eagle3_long_prompt(use_cuda_graph):
    # Eagle3 one model works with overlap scheduler and block reuse.
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 35:
        pytest.skip("Not enough memory to load target + draft model")

    models_path = llm_models_root()
    eagle_model_dir = f"{models_path}/EAGLE3-LLaMA3.1-Instruct-8B"
    target_model_dir = f"{models_path}/llama-3.1-model/Llama-3.1-8B-Instruct"

    spec_config = Eagle3DecodingConfig(
        max_draft_len=3,
        speculative_model=eagle_model_dir,
        eagle3_one_model=False,
    )

    if use_cuda_graph:
        cuda_graph_config = CudaGraphConfig(batch_sizes=[1])
    else:
        cuda_graph_config = None

    llm_spec = LLM(model=target_model_dir,
                   speculative_config=spec_config,
                   max_batch_size=1,
                   cuda_graph_config=cuda_graph_config,
                   disable_overlap_scheduler=True)

    prompt = [", ".join(str(i) for i in range(1000))]

    sampling_params = SamplingParams(max_tokens=10, temperature=0)
    results_spec = llm_spec.generate(prompt, sampling_params)

    generated_text_spec = [result.outputs[0].text for result in results_spec]
    llm_spec.shutdown()

    llm_ref = LLM(model=target_model_dir,
                  max_batch_size=1,
                  cuda_graph_config=None,
                  disable_overlap_scheduler=False)

    results_ref = llm_ref.generate(prompt, sampling_params)

    generated_text_ref = [result.outputs[0].text for result in results_ref]
    llm_ref.shutdown()

    # The LLM with speculation on should dynamically turn it off in this
    # test since it goes beyond the max seqlen. Thus, the text should be
    # _exactly_ the same, no need to use similarity scoring.
    assert generated_text_spec[0] == generated_text_ref[0]


def test_deepseek_eagle3():
    use_cuda_graph = True
    attn_backend = "TRTLLM"
    disable_overlap_scheduler = False
    enable_block_reuse = False
    use_one_model = False
    enable_chunked_prefill = False

    # Eagle3 one model works with overlap scheduler and block reuse.
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 150:
        pytest.skip("Not enough memory to load target + draft model")

    models_path = llm_models_root()
    eagle_config = {
        'architectures': ['LlamaForCausalLMEagle3'],
        'attention_bias': False,
        'attention_dropout': 0.0,
        'bos_token_id': 128000,
        'eos_token_id': [128001, 128008, 128009],
        'eagle_config': {
            'use_aux_hidden_state': False,
            'use_input_layernorm_in_first_layer': True,
            'use_last_layernorm': True,
            'use_mtp_layernorm': False
        },
        'head_dim': 128,
        'hidden_act': 'silu',
        'hidden_size': 2560,
        'initializer_range': 0.02,
        'intermediate_size': 16384,
        'max_position_embeddings': 4096,
        'mlp_bias': False,
        'model_type': 'llama',
        'num_attention_heads': 32,
        'num_eagle_features': 1,
        'num_hidden_layers': 1,
        'num_key_value_heads': 8,
        'pretraining_tp': 1,
        'rms_norm_eps': 1e-05,
        'rope_scaling': {
            'factor': 8.0,
            'high_freq_factor': 4.0,
            'low_freq_factor': 1.0,
            'original_max_position_embeddings': 8192,
            'rope_type': 'llama3'
        },
        'rope_theta': 500000.0,
        'tie_word_embeddings': False,
        'torch_dtype': 'bfloat16',
        'transformers_version': '4.52.4',
        'use_cache': True,
        'vocab_size': 129280,
        'draft_vocab_size': 129280,
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        eagle_model_dir = Path(temp_dir)
        config_path = eagle_model_dir / "config.json"
        with config_path.open("w") as f:
            json.dump(eagle_config, f, indent=2)
        target_model_dir = f"{models_path}/DeepSeek-V3-Lite/nvfp4_moe_only"

        # bs > 1 gives non-deterministic when doing IFB. There are slight chances
        # that ref and spec does not match 100%
        max_batch_size = 16
        max_draft_len = 3
        kv_cache_config = KvCacheConfig(enable_block_reuse=enable_block_reuse,
                                        max_tokens=8192)
        cuda_graph_config = CudaGraphConfig(
            batch_sizes=[1]) if use_cuda_graph else None

        llm_common_config = dict(
            model=target_model_dir,
            attn_backend=attn_backend,
            disable_overlap_scheduler=disable_overlap_scheduler,
            cuda_graph_config=cuda_graph_config,
            max_batch_size=max_batch_size,
            max_num_tokens=4096,
            max_seq_len=4096,
            kv_cache_config=kv_cache_config,
            enable_chunked_prefill=enable_chunked_prefill,
        )

        spec_config = Eagle3DecodingConfig(
            max_draft_len=max_draft_len,
            speculative_model=eagle_model_dir,
            # Llama 3 does not support one model eagle.
            eagle3_one_model=use_one_model,
            eagle3_layers_to_capture={29},
            load_format="dummy")

        llm_spec = LLM(**llm_common_config, speculative_config=spec_config)

        tok_ids = llm_spec.tokenizer.encode("The future of AI is")

        sampling_params = SamplingParams(max_tokens=32, temperature=0)
        for output in llm_spec.generate_async(tok_ids,
                                              sampling_params,
                                              streaming=True):
            pass


def test_deepseek_mla_eagle3():
    use_cuda_graph = True
    attn_backend = "TRTLLM"
    disable_overlap_scheduler = False
    enable_block_reuse = False
    use_one_model = True
    enable_chunked_prefill = False

    # Eagle3 one model works with overlap scheduler and block reuse.
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 150:
        pytest.skip("Not enough memory to load target + draft model")

    models_path = llm_models_root()
    eagle_config = {
        "architectures": ["Eagle3DeepseekV3ForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "first_k_dense_replace": 1,
        "hidden_act": "silu",
        "hidden_size": 2560,
        "intermediate_size": 8192,
        "kv_lora_rank": 512,
        "max_position_embeddings": 4096,
        "model_type": "kimi_k2",
        "num_attention_heads": 32,
        "num_hidden_layers": 1,
        "num_key_value_heads": 32,
        "num_nextn_predict_layers": 0,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "rms_norm_eps": 1e-05,
        "rope_scaling": {
            "beta_fast": 1.0,
            "beta_slow": 1.0,
            "factor": 64.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "type": "yarn"
        },
        "rope_theta": 50000.0,
        "routed_scaling_factor": 2.827,
        "scoring_func": "sigmoid",
        "seq_aux": True,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "torch_dtype": "bfloat16",
        "torchscript": False,
        "transformers_version": "4.51.3",
        "use_bfloat16": False,
        "use_cache": True,
        "v_head_dim": 128,
        "vocab_size": 129280,
        "draft_vocab_size": 129280,
        "eagle_config": {
            "use_aux_hidden_state": True,
            "use_input_layernorm_in_first_layer": True,
            "use_last_layernorm": True,
            "use_mtp_layernorm": False
        }
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        eagle_model_dir = Path(temp_dir)
        config_path = eagle_model_dir / "config.json"
        with config_path.open("w") as f:
            json.dump(eagle_config, f, indent=2)
        target_model_dir = f"{models_path}/DeepSeek-V3-Lite/nvfp4_moe_only"

        # bs > 1 gives non-deterministic when doing IFB. There are slight chances
        # that ref and spec does not match 100%
        max_batch_size = 16
        max_draft_len = 3
        kv_cache_config = KvCacheConfig(enable_block_reuse=enable_block_reuse,
                                        max_tokens=8192)
        cuda_graph_config = CudaGraphConfig(
            batch_sizes=[1]) if use_cuda_graph else None

        llm_common_config = dict(
            model=target_model_dir,
            attn_backend=attn_backend,
            disable_overlap_scheduler=disable_overlap_scheduler,
            cuda_graph_config=cuda_graph_config,
            max_batch_size=max_batch_size,
            max_num_tokens=4096,
            max_seq_len=4096,
            kv_cache_config=kv_cache_config,
            enable_chunked_prefill=enable_chunked_prefill,
            load_format="dummy",
        )

        spec_config = Eagle3DecodingConfig(max_draft_len=max_draft_len,
                                           speculative_model=eagle_model_dir,
                                           eagle3_one_model=use_one_model,
                                           load_format="dummy")

        llm_spec = LLM(**llm_common_config, speculative_config=spec_config)

        tok_ids = llm_spec.tokenizer.encode("The future of AI is")

        sampling_params = SamplingParams(max_tokens=32, temperature=0)
        for output in llm_spec.generate_async(tok_ids,
                                              sampling_params,
                                              streaming=True):
            pass


@pytest.mark.parametrize("use_one_model", [True, False])
def test_multi_eagle3(use_one_model: bool):
    use_cuda_graph = True
    attn_backend = "TRTLLM"
    disable_overlap_scheduler = False
    enable_block_reuse = False
    enable_chunked_prefill = False

    # Eagle3 one model works with overlap scheduler and block reuse.
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 150:
        pytest.skip("Not enough memory to load target + draft model")

    models_path = llm_models_root()
    eagle_config = {
        'architectures': ['LlamaForCausalLMEagle3'],
        'attention_bias': False,
        'attention_dropout': 0.0,
        'bos_token_id': 128000,
        'eos_token_id': [128001, 128008, 128009],
        'eagle_config': {
            'use_aux_hidden_state': False,
            'use_input_layernorm_in_first_layer': True,
            'use_last_layernorm': True,
            'use_mtp_layernorm': False
        },
        'head_dim': 128,
        'hidden_act': 'silu',
        'hidden_size': 4096,
        'initializer_range': 0.02,
        'intermediate_size': 16384,
        'max_position_embeddings': 131072,
        'mlp_bias': False,
        'model_type': 'llama',
        'num_attention_heads': 32,
        'num_eagle_features': 1,
        'num_hidden_layers': 2,
        'num_key_value_heads': 8,
        'pretraining_tp': 1,
        'rms_norm_eps': 1e-05,
        'rope_scaling': {
            'factor': 8.0,
            'high_freq_factor': 4.0,
            'low_freq_factor': 1.0,
            'original_max_position_embeddings': 8192,
            'rope_type': 'llama3'
        },
        'rope_theta': 500000.0,
        'tie_word_embeddings': False,
        'torch_dtype': 'bfloat16',
        'transformers_version': '4.52.4',
        'use_cache': True,
        'vocab_size': 128256,
        'draft_vocab_size': 128256,
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        eagle_model_dir = Path(temp_dir)
        config_path = eagle_model_dir / "config.json"
        with config_path.open("w") as f:
            json.dump(eagle_config, f, indent=2)
        target_model_dir = f"{models_path}/llama-3.1-model/Llama-3.1-8B-Instruct"

        # bs > 1 gives non-deterministic when doing IFB. There are slight chances
        # that ref and spec does not match 100%
        max_batch_size = 16
        max_draft_len = 3
        kv_cache_config = KvCacheConfig(enable_block_reuse=enable_block_reuse,
                                        free_gpu_memory_fraction=0.5)
        cuda_graph_config = CudaGraphConfig(
            batch_sizes=[1]) if use_cuda_graph else None

        llm_common_config = dict(
            model=target_model_dir,
            attn_backend=attn_backend,
            disable_overlap_scheduler=disable_overlap_scheduler,
            cuda_graph_config=cuda_graph_config,
            max_batch_size=max_batch_size,
            kv_cache_config=kv_cache_config,
            enable_chunked_prefill=enable_chunked_prefill,
            load_format="dummy",
        )

        spec_config = Eagle3DecodingConfig(
            max_draft_len=max_draft_len,
            speculative_model=eagle_model_dir,
            # Llama 3 does not support one model eagle.
            eagle3_one_model=use_one_model,
            num_eagle_layers=2,
            load_format="dummy")

        llm_spec = LLM(**llm_common_config, speculative_config=spec_config)

        tok_ids = llm_spec.tokenizer.encode("The future of AI is")

        sampling_params = SamplingParams(max_tokens=32, temperature=0)
        for output in llm_spec.generate_async(tok_ids,
                                              sampling_params,
                                              streaming=True):
            pass


@pytest.mark.parametrize("disable_overlap_scheduler", [True, False])
def test_eagle3_cuda_graph_padding(disable_overlap_scheduler: bool):
    """Test CUDA graph padding with 3 requests and max_batch_size=4.

    This test verifies that when using CUDA graph with padding enabled,
    the system properly reserves one additional slot for the padded dummy request.
    Without this fix, there would be errors caused by no free slot.
    """
    attn_backend = "TRTLLM"
    enable_block_reuse = False
    use_one_model = False
    enable_chunked_prefill = False

    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 35:
        pytest.skip("Not enough memory to load target + draft model")

    models_path = llm_models_root()
    eagle_model_dir = f"{models_path}/EAGLE3-LLaMA3.1-Instruct-8B"
    target_model_dir = f"{models_path}/llama-3.1-model/Llama-3.1-8B-Instruct"

    # Test with 3 requests and max_batch_size=4 to trigger padding
    max_batch_size = 4
    max_draft_len = 4
    kv_cache_config = KvCacheConfig(enable_block_reuse=enable_block_reuse,
                                    max_tokens=4096)
    cuda_graph_config = CudaGraphConfig(batch_sizes=[1, 2, 4],
                                        enable_padding=True)

    llm_common_config = dict(
        model=target_model_dir,
        attn_backend=attn_backend,
        disable_overlap_scheduler=disable_overlap_scheduler,
        cuda_graph_config=cuda_graph_config,
        max_batch_size=max_batch_size,
        kv_cache_config=kv_cache_config,
        max_seq_len=2048,
        enable_chunked_prefill=enable_chunked_prefill,
    )

    spec_config = Eagle3DecodingConfig(
        max_draft_len=max_draft_len,
        speculative_model=eagle_model_dir,
        eagle3_one_model=use_one_model,
    )

    # Create the LLM instance
    llm_spec = LLM(**llm_common_config, speculative_config=spec_config)

    prompts = [
        "The capital of France is", "The president of the United States is",
        "The future of AI is"
    ]

    sampling_params = SamplingParams(max_tokens=2048, temperature=0)
    llm_spec.generate(prompts, sampling_params)
    llm_spec.shutdown()


@pytest.mark.parametrize("disable_overlap_scheduler", [True, False])
def test_eagle3_cdl_sampling(disable_overlap_scheduler: bool):
    """Test CDL sampling with 2 requests and max_batch_size=2."""
    attn_backend = "TRTLLM"
    enable_block_reuse = False
    use_one_model = False
    enable_chunked_prefill = False

    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 35:
        pytest.skip("Not enough memory to load target + draft model")

    models_path = llm_models_root()
    eagle_model_dir = f"{models_path}/EAGLE3-LLaMA3.1-Instruct-8B"
    target_model_dir = f"{models_path}/llama-3.1-model/Llama-3.1-8B-Instruct"

    max_batch_size = 1
    max_draft_len = 4
    kv_cache_config = KvCacheConfig(enable_block_reuse=enable_block_reuse,
                                    max_tokens=8192)
    cuda_graph_config = CudaGraphConfig(batch_sizes=[1, 2, 4],
                                        enable_padding=True)

    llm_common_config = dict(
        model=target_model_dir,
        attn_backend=attn_backend,
        disable_overlap_scheduler=disable_overlap_scheduler,
        cuda_graph_config=cuda_graph_config,
        max_batch_size=max_batch_size,
        kv_cache_config=kv_cache_config,
        max_seq_len=8192,
        enable_chunked_prefill=enable_chunked_prefill,
    )

    spec_config = Eagle3DecodingConfig(
        max_draft_len=max_draft_len,
        speculative_model=eagle_model_dir,
        eagle3_one_model=use_one_model,
    )

    # Create the LLM instance
    llm_spec = LLM(**llm_common_config, speculative_config=spec_config)

    prompts = ["The president of the United States is"]

    sampling_params = SamplingParams(max_tokens=20, temperature=1.0, top_p=0.9)
    llm_spec.generate(prompts, sampling_params)
    llm_spec.shutdown()


if __name__ == "__main__":
    unittest.main()
