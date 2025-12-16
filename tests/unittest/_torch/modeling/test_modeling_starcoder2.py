from copy import deepcopy
from dataclasses import dataclass

import pytest
import torch
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model
from transformers import AutoModelForCausalLM, Starcoder2Config
from transformers import Starcoder2ForCausalLM as HFStarcoder2ForCausalLM
from utils.llm_data import llm_models_root
from utils.util import default_dtype

import tensorrt_llm
from tensorrt_llm import LLM
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_starcoder2 import Starcoder2ForCausalLM
from tensorrt_llm._torch.modules.layer_norm import LayerNorm
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.lora_manager import LoraConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.sampling_params import SamplingParams

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
    hf_starcoder2 = hf_starcoder2.to(dtype=torch.bfloat16, device="cuda").eval()

    dtype = torch.bfloat16
    device = torch.device("cuda")

    # Build TRT-LLM model and copy the same random weights from HF model
    with torch.device(device), default_dtype(dtype):
        model_config = ModelConfig(pretrained_config=hf_config, attn_backend=backend)
        starcoder2 = Starcoder2ForCausalLM(model_config).to(dtype).to(device).eval()
        starcoder2.load_weights(hf_starcoder2.state_dict())

    # Convert LayerNorm random weights to FP32 for numerical stability
    for name, module in starcoder2.named_modules():
        if isinstance(module, LayerNorm):
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data = module.weight.data.to(torch.float32)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data = module.bias.data.to(torch.float32)

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
        dtype=torch.int,
        device=device,
    )
    num_cached_tokens_per_seq = [0]
    request_ids = [1]
    token_nums = [input_ids.size(-1)]
    prompt_lens = [input_ids.size(-1)]
    kv_cache_manager.add_dummy_requests(request_ids, token_nums)

    attn_metadata = metadata_cls(
        seq_lens=torch.tensor([input_ids.size(-1)], dtype=torch.int),
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

    position_ids = [torch.arange(0, input_ids.size(-1), dtype=torch.int)]
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
        torch.testing.assert_close(logits, ref.logits[:, -1].float(), atol=0.1, rtol=0.1)

    # Generation phase (optionally with CUDA graphs)
    gen_input_ids = torch.tensor([900], dtype=torch.int, device=device)
    num_cached_tokens_per_seq = [input_ids.size(-1)]

    attn_metadata = metadata_cls(
        seq_lens=torch.tensor([gen_input_ids.size(-1)], dtype=torch.int),
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
            input_ids.size(-1), input_ids.size(-1) + gen_input_ids.size(-1), dtype=torch.int
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
        torch.testing.assert_close(logits, ref.logits[:, -1].float(), atol=0.1, rtol=0.1)

    # Cleanup
    if graph_runner is not None:
        graph_runner.clear()
    kv_cache_manager.shutdown()


@torch.no_grad()
def test_starcoder2_multi_lora(tmp_path) -> None:
    """
    Test StarCoder2 3b model with multiple synthetic LoRA adapters created using PEFT.

    This test creates dummy LoRA adapters for StarCoder2 and verifies that:
    1. Multiple LoRA adapters can be loaded and used simultaneously
    2. Different requests can use different LoRA adapters
    3. The model produces reasonable outputs with LoRA adapters applied
    """

    # Check if we have enough GPU memory (need ~10GB for StarCoder2-3B + LoRA)
    _, total_mem = torch.cuda.mem_get_info()
    min_mem_required = 10 * (2**30)  # 10 GB
    if total_mem < min_mem_required:
        pytest.skip("Insufficient GPU memory for StarCoder2 with LoRA test")

    # Check for pretrained model
    model_path = f"{llm_models_root()}/starcoder2-3b"

    # Target modules for LoRA - attention projections
    target_modules = ["attn_q", "attn_k", "attn_v", "attn_dense"]

    # Load the pretrained model to create LoRA adapters
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    # HuggingFace module names for StarCoder2 attention
    hf_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    peft_lora_config = PeftLoraConfig(
        r=8,  # LoRA rank
        lora_alpha=16,
        target_modules=hf_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Create two synthetic LoRA adapters with zeroed weights
    lora_paths = []
    for i in range(2):
        lora_model = get_peft_model(model, peft_lora_config)

        # Zero out all LoRA parameters for deterministic testing
        for name, param in lora_model.named_parameters():
            if "lora_" in name:
                param.data.zero_()

        # Save the LoRA adapter
        lora_path = tmp_path / f"lora_{i}"
        lora_model.save_pretrained(lora_path)
        lora_paths.append(str(lora_path))

    del model
    del lora_model
    torch.cuda.empty_cache()

    # Configure TensorRT-LLM LoRA
    trtllm_lora_config = LoraConfig(
        lora_target_modules=target_modules, max_lora_rank=8, max_loras=2, max_cpu_loras=2
    )

    llm = LLM(
        model_path,
        lora_config=trtllm_lora_config,
        # Disable CUDA graph for LoRA (LoRA is not supported with CUDA graphs yet)
        cuda_graph_config=None,
    )

    with llm:
        prompts = [
            "def fibonacci(n):",
            "def quick_sort(arr):",
        ]

        lora_req1 = LoRARequest("lora-1", 0, lora_paths[0])
        lora_req2 = LoRARequest("lora-2", 1, lora_paths[1])
        lora_requests = [lora_req1, lora_req2]

        # Sampling parameters
        sampling_params = SamplingParams(
            max_tokens=50,
            temperature=0.0,  # Greedy decoding for deterministic output
        )

        outputs = llm.generate(prompts, sampling_params, lora_request=lora_requests)

        # Verify we got outputs for both prompts
        assert len(outputs) == 2, f"Expected 2 outputs, got {len(outputs)}"

        # Verify each output has text
        for i, output in enumerate(outputs):
            assert len(output.outputs) > 0, f"Output {i} has no results"
            assert len(output.outputs[0].text) > 0, f"Output {i} generated empty text"

        # Test without LoRA for comparison
        outputs_no_lora = llm.generate(prompts, sampling_params, lora_request=None)

        assert len(outputs_no_lora) == 2

        assert outputs[0].outputs[0].text == outputs_no_lora[0].outputs[0].text
        assert outputs[1].outputs[0].text == outputs_no_lora[1].outputs[0].text
