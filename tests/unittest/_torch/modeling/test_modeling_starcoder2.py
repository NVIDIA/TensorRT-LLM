import pytest
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
from transformers import Starcoder2Config
from transformers import Starcoder2ForCausalLM as HFStarcoder2ForCausalLM
from utils.util import default_dtype

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_starcoder2 import Starcoder2ForCausalLM
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping

# Base config for all StarCoder2 models (based on HuggingFace configs)
_STARCODER2_BASE_CONFIG = {
    "architectures": ["Starcoder2ForCausalLM"],
    "attention_dropout": 0.1,
    "residual_dropout": 0.1,
    "embedding_dropout": 0.1,
    "bos_token_id": 0,
    "eos_token_id": 0,
    "hidden_act": "gelu_pytorch_tanh",
    "max_position_embeddings": 16384,
    "mlp_type": "default",
    "model_type": "starcoder2",
    "norm_epsilon": 1e-5,
    "num_hidden_layers": 6,  # Reduced from 30/32/40 for testing
    "sliding_window": 4096,
    "transformers_version": "4.37.0.dev0",
    "use_bias": True,
    "use_cache": True,
    "vocab_size": 49152,
    "torch_dtype": "bfloat16",
}

# StarCoder2-3B config (reduced for testing)
STARCODER2_3B_CONFIG = {
    **_STARCODER2_BASE_CONFIG,
    "hidden_size": 3072,
    "initializer_range": 0.018042,
    "intermediate_size": 12288,
    "num_attention_heads": 24,
    "num_key_value_heads": 2,
    "rope_theta": 999999.4420358813,
}

# StarCoder2-7B config (reduced for testing)
STARCODER2_7B_CONFIG = {
    **_STARCODER2_BASE_CONFIG,
    "hidden_size": 4608,
    "initializer_range": 0.018042,
    "intermediate_size": 18432,
    "num_attention_heads": 36,
    "num_key_value_heads": 4,
    "rope_theta": 1000000,
}

# StarCoder2-15B config (reduced for testing)
STARCODER2_15B_CONFIG = {
    **_STARCODER2_BASE_CONFIG,
    "hidden_size": 6144,
    "initializer_range": 0.01275,
    "intermediate_size": 24576,
    "num_attention_heads": 48,
    "num_key_value_heads": 4,
    "rope_theta": 100000,
}


@dataclass(repr=False)
class Scenario:
    backend: str
    config_name: str
    use_cuda_graph: bool = False

    def __repr__(self) -> str:
        return f"backend:{self.backend.lower()}_config:{self.config_name.lower()}_cuda_graph:{self.use_cuda_graph}"


def get_kv_cache_manager(
    dtype: torch.dtype,
    config: Starcoder2Config,
    tokens_per_block: int,
    max_seq_len: int,
    batch_size: int,
    num_blocks: int,
):
    """Helper to create KV cache manager."""
    if dtype == torch.half:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
    elif dtype == torch.bfloat16:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
    else:
        raise ValueError(f"Invalid dtype: {dtype}")

    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=False,
        enable_partial_reuse=False,
        copy_on_partial_reuse=False,
        max_tokens=num_blocks * tokens_per_block,
    )

    head_dim = config.hidden_size // config.num_attention_heads
    kv_cache_manager = KVCacheManager(
        kv_cache_config,
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=kv_cache_dtype,
    )
    return kv_cache_manager

@pytest.mark.parametrize(
    "scenario",
    [
        # Test without CUDA graphs
        Scenario(backend="TRTLLM", config_name="3B", use_cuda_graph=False),
        Scenario(backend="TRTLLM", config_name="7B", use_cuda_graph=False),
        Scenario(backend="TRTLLM", config_name="15B", use_cuda_graph=False),
        # Test with CUDA graphs
        Scenario(backend="TRTLLM", config_name="3B", use_cuda_graph=True),
        Scenario(backend="TRTLLM", config_name="7B", use_cuda_graph=True),
        Scenario(backend="TRTLLM", config_name="15B", use_cuda_graph=True),
    ],
    ids=str,
)
@torch.no_grad()
def test_starcoder2_allclose_to_hf(scenario: Scenario) -> None:
    """
    Compare TensorRT-LLM StarCoder2 output to HuggingFace.

    Tests both context and generation phases using randomly initialized models.
    Optionally tests with CUDA graphs for generation phase optimization.
    """
    backend = scenario.backend
    config_name = scenario.config_name
    use_cuda_graph = scenario.use_cuda_graph
    metadata_cls = get_attention_backend(backend).Metadata

    torch.random.manual_seed(0)

    # Create config based on model size
    config_mapping = {
        "3B": STARCODER2_3B_CONFIG,
        "7B": STARCODER2_7B_CONFIG,
        "15B": STARCODER2_15B_CONFIG,
    }
    config_dict = deepcopy(config_mapping[config_name])

    # Create HuggingFace model from config with random weights
    hf_config = Starcoder2Config.from_dict(config_dict)
    hf_starcoder2 = HFStarcoder2ForCausalLM(hf_config)
    hf_starcoder2 = hf_starcoder2.to(dtype=torch.bfloat16, device="cuda")

    dtype = torch.bfloat16
    device = torch.device("cuda")

    # Build TRT-LLM model and copy the same random weights from HF model
    with torch.device(device), default_dtype(dtype):
        model_config = ModelConfig(pretrained_config=hf_config, attn_backend=backend)
        starcoder2 = Starcoder2ForCausalLM(model_config).to(dtype).to(device)
        starcoder2.load_weights(hf_starcoder2.state_dict())

    num_blocks = 1
    tokens_per_block = 128
    max_seq_len = num_blocks * tokens_per_block
    batch_size = 1

    kv_cache_manager = get_kv_cache_manager(
        dtype=dtype,
        config=hf_config,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        num_blocks=num_blocks,
    )

    # Context phase (no CUDA graphs for prefill)
    input_ids = torch.tensor(
        [100, 200, 300, 400, 500, 600, 700, 800],
        dtype=torch.long,
        device=device,
    )
    num_cached_tokens_per_seq = [0]
    request_ids = [1]
    token_nums = [input_ids.size(-1)]
    prompt_lens = [input_ids.size(-1)]
    kv_cache_manager.add_dummy_requests(request_ids, token_nums)

    attn_metadata = metadata_cls(
        seq_lens=torch.tensor([input_ids.size(-1)], dtype=torch.long),
        num_contexts=1,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=num_cached_tokens_per_seq,
        ),
        max_num_requests=1,
        max_num_tokens=8192,
        kv_cache_manager=kv_cache_manager,
        request_ids=request_ids,
        prompt_lens=prompt_lens,
    )

    position_ids = [torch.arange(0, input_ids.size(-1), dtype=torch.long)]
    position_ids = torch.cat(position_ids).unsqueeze(0).cuda()

    with torch.inference_mode():
        attn_metadata.prepare()
        logits = starcoder2.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attn_metadata=attn_metadata,
        )
        ref = hf_starcoder2.forward(
            input_ids=input_ids.unsqueeze(0),
            position_ids=position_ids,
            use_cache=True,
        )
        torch.testing.assert_close(logits, ref.logits[:, -1].float(), atol=0.4, rtol=0.4)

    # Generation phase (optionally with CUDA graphs)
    gen_input_ids = torch.tensor([900], dtype=torch.long, device=device)
    num_cached_tokens_per_seq = [input_ids.size(-1)]

    attn_metadata = metadata_cls(
        seq_lens=torch.tensor([gen_input_ids.size(-1)], dtype=torch.long),
        num_contexts=0,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=num_cached_tokens_per_seq,
        ),
        kv_cache_manager=kv_cache_manager,
        request_ids=request_ids,
        prompt_lens=prompt_lens,
        max_num_requests=1,
        max_num_tokens=8192,
    )

    gen_position_ids = [
        torch.arange(
            input_ids.size(-1), input_ids.size(-1) + gen_input_ids.size(-1), dtype=torch.long
        )
    ]
    gen_position_ids = torch.cat(gen_position_ids).unsqueeze(0).cuda()

    # Setup CUDA graph runner if requested
    graph_runner = None
    if use_cuda_graph:
        from _torch.helpers import create_mock_cuda_graph_runner

        graph_runner = create_mock_cuda_graph_runner(1)
        attn_metadata = attn_metadata.create_cuda_graph_metadata(1)

    # Run generation phase
    with torch.inference_mode():
        if not use_cuda_graph:
            attn_metadata.prepare()
            logits = starcoder2.forward(
                input_ids=gen_input_ids,
                position_ids=gen_position_ids,
                attn_metadata=attn_metadata,
            )
        else:
            # CUDA graph path
            inputs = {
                "input_ids": gen_input_ids,
                "position_ids": gen_position_ids,
                "attn_metadata": attn_metadata,
            }
            key = (1, 0, False)

            attn_metadata.prepare()
            graph_runner.capture(key, lambda inputs: starcoder2.forward(**inputs), inputs)

            # Run twice to catch buffer reallocation issues
            for _ in range(2):
                attn_metadata.prepare()
                logits = graph_runner.replay(key, inputs)

        # Compare with HuggingFace
        ref = hf_starcoder2.forward(
            input_ids=gen_input_ids.unsqueeze(0),
            position_ids=gen_position_ids,
            past_key_values=ref.past_key_values,
            use_cache=True,
        )
        torch.testing.assert_close(logits, ref.logits[:, -1].float(), atol=0.4, rtol=0.4)

    # Cleanup
    if graph_runner is not None:
        graph_runner.clear()
    kv_cache_manager.shutdown()


@pytest.mark.parametrize(
    "scenario",
    [
        # Test token-level generation for different model sizes
        Scenario(backend="TRTLLM", config_name="3B"),
        Scenario(backend="TRTLLM", config_name="7B"),
        Scenario(backend="TRTLLM", config_name="15B"),
    ],
    ids=str,
)
@torch.no_grad()
def test_starcoder2_generated_tokens_match_hf(scenario: Scenario) -> None:
    """
    Compare generated tokens from TRT-LLM PyTorch backend to HuggingFace.
    Uses randomly initialized models with identical weights.
    """
    backend = scenario.backend
    config_name = scenario.config_name

    torch.random.manual_seed(0)

    # Create config based on model size
    config_mapping = {
        "3B": STARCODER2_3B_CONFIG,
        "7B": STARCODER2_7B_CONFIG,
        "15B": STARCODER2_15B_CONFIG,
    }
    config_dict = deepcopy(config_mapping[config_name])

    # Create HuggingFace model from config with random weights
    hf_config = Starcoder2Config.from_dict(config_dict)
    hf_starcoder2 = HFStarcoder2ForCausalLM(hf_config)
    hf_starcoder2 = hf_starcoder2.to(dtype=torch.bfloat16, device="cuda")

    dtype = torch.bfloat16
    device = torch.device("cuda")

    # Build TRT-LLM model and copy the same random weights from HF model
    with torch.device(device), default_dtype(dtype):
        model_config = ModelConfig(pretrained_config=hf_config, attn_backend=backend)
        starcoder2 = Starcoder2ForCausalLM(model_config).to(dtype).to(device)
        starcoder2.load_weights(hf_starcoder2.state_dict())

    test_prompt = "def fibonacci(n):"
    # Create a simple tokenizer for the test (just split by characters for simplicity)
    # Use a fixed token mapping for deterministic testing
    input_ids = torch.tensor(
        [100, 200, 300, 400, 500],  # Fixed token IDs for testing
        dtype=torch.long,
        device=device,
    )

    # Setup KV cache for TRT-LLM generation
    num_blocks = 2
    tokens_per_block = 128
    max_seq_len = num_blocks * tokens_per_block
    batch_size = 1

    kv_cache_manager = get_kv_cache_manager(
        dtype=dtype,
        config=hf_config,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        num_blocks=num_blocks,
    )

    # Generate tokens with TRT-LLM (manual generation loop)
    max_new_tokens = 20
    trt_output_ids = []
    num_cached_tokens = 0
    request_ids = [1]
    prompt_lens = [input_ids.size(-1)]
    metadata_cls = get_attention_backend(backend).Metadata

    # Context phase - process initial prompt
    token_nums = [input_ids.size(-1)]
    kv_cache_manager.add_dummy_requests(request_ids, token_nums)

    attn_metadata = metadata_cls(
        seq_lens=torch.tensor([input_ids.size(-1)], dtype=torch.long),
        num_contexts=1,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[0],
        ),
        kv_cache_manager=kv_cache_manager,
        request_ids=request_ids,
        prompt_lens=prompt_lens,
        max_num_requests=1,
        max_num_tokens=8192,
    )

    position_ids = torch.arange(
        0, input_ids.size(-1), dtype=torch.long, device=device
    ).unsqueeze(0)

    with torch.inference_mode():
        attn_metadata.prepare()
        logits = starcoder2.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attn_metadata=attn_metadata,
        )

    # Get first token
    next_token_id = torch.argmax(logits, dim=-1).item()
    trt_output_ids.append(next_token_id)
    num_cached_tokens = input_ids.size(-1)

    # Generation phase - generate remaining tokens
    for step in range(1, max_new_tokens):
        gen_input_ids = torch.tensor([next_token_id], dtype=torch.long, device=device)

        attn_metadata = metadata_cls(
            seq_lens=torch.tensor([1], dtype=torch.long),
            num_contexts=0,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[num_cached_tokens],
            ),
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=prompt_lens,
            max_num_requests=1,
            max_num_tokens=8192,
        )

        gen_position_ids = torch.arange(
            num_cached_tokens, num_cached_tokens + 1, dtype=torch.long, device=device
        ).unsqueeze(0)

        with torch.inference_mode():
            attn_metadata.prepare()
            logits = starcoder2.forward(
                input_ids=gen_input_ids,
                position_ids=gen_position_ids,
                attn_metadata=attn_metadata,
            )

        # Greedy sampling: take argmax
        next_token_id = torch.argmax(logits, dim=-1).item()
        trt_output_ids.append(next_token_id)
        num_cached_tokens += 1

    # Generate with HuggingFace for comparison (manual loop for consistency)
    hf_output_ids = []
    hf_past_key_values = None
    hf_current_ids = input_ids.unsqueeze(0)

    with torch.inference_mode():
        for step in range(max_new_tokens):
            hf_output = hf_starcoder2.forward(
                input_ids=hf_current_ids,
                past_key_values=hf_past_key_values,
                use_cache=True,
            )
            # Greedy sampling: take argmax
            next_token_id = torch.argmax(hf_output.logits[:, -1, :], dim=-1).item()
            hf_output_ids.append(next_token_id)
            hf_past_key_values = hf_output.past_key_values
            hf_current_ids = torch.tensor([[next_token_id]], dtype=torch.long, device=device)

    # Compare outputs - both should match exactly with same random weights
    min_len = min(len(trt_output_ids), len(hf_output_ids))
    matches = sum(1 for i in range(min_len) if trt_output_ids[i] == hf_output_ids[i])
    match_ratio = matches / min_len if min_len > 0 else 0.0

    # Print for debugging
    print(f"\n{config_name}/{backend} TRT output tokens: {trt_output_ids}")
    print(f"{config_name}/{backend} HF output tokens:  {hf_output_ids}")
    print(f"Match ratio: {match_ratio:.2%} ({matches}/{min_len} tokens)")

    # Should match exactly with identical random weights
    assert match_ratio == 1.0, (
        f"TRT-LLM and HF token outputs should match exactly: {match_ratio:.2%} match"
    )

    kv_cache_manager.shutdown()
