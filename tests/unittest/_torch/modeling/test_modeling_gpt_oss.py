import json
import os
import shutil

import pytest
import torch
from torch.profiler import ProfilerActivity
from transformers import AutoTokenizer, GptOssConfig
from utils.llm_data import llm_models_root
from utils.util import skip_no_hopper

import tensorrt_llm
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_gpt_oss import GptOssForCausalLM
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import \
    KvCacheConfig as BindingsKvCacheConfig
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MoeConfig
from tensorrt_llm.mapping import Mapping

configs = """
{
    "architectures": [
        "GptOssForCausalLM"
    ],
    "model_type": "gpt_oss",
    "torch_dtype": "bfloat16",
    "num_hidden_layers": 4,
    "num_experts": 128,
    "experts_per_token": 4,
    "vocab_size": 201088,
    "hidden_size": 2880,
    "intermediate_size": 2880,
    "head_dim": 64,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "sliding_window": 128,
    "initial_context_length": 4096,
    "rope_theta": 150000,
    "rope_scaling_factor": 32.0,
    "rope_ntk_alpha": 1,
    "rope_ntk_beta": 32
}
"""


def dump_config_json(dst_dir):
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir)

    dst_path = os.path.join(dst_dir, 'config.json')
    with open(dst_path, 'w', encoding='utf-8') as f:
        json_configs = json.loads(configs)
        json.dump(json_configs, f, indent=2, ensure_ascii=False)


@pytest.mark.parametrize(
    "moe_backend",
    ["CUTLASS", pytest.param("TRITON", marks=skip_no_hopper)])
def test_gpt_oss_trtllmgen(moe_backend):
    prompts = [
        "How are you?",
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    pytorch_config = dict(
        disable_overlap_scheduler=False,
        cuda_graph_config=CudaGraphConfig(),
        attn_backend="TRTLLM",
        load_format="dummy",
        moe_config=MoeConfig(backend=moe_backend),
    )

    tmp_model_dir = f"/tmp/test_model_trtllm"

    dump_config_json(tmp_model_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        f"{llm_models_root()}/gpt_oss/gpt-oss-20b")

    llm = LLM(model=tmp_model_dir,
              tokenizer=tokenizer,
              tensor_parallel_size=1,
              enable_chunked_prefill=False,
              **pytorch_config,
              max_batch_size=16,
              max_seq_len=1024,
              moe_expert_parallel_size=-1,
              moe_tensor_parallel_size=-1,
              enable_attention_dp=False,
              kv_cache_config=KvCacheConfig(enable_block_reuse=False,
                                            free_gpu_memory_fraction=0.4))

    sampling_params = SamplingParams(max_tokens=20)
    llm.generate(prompts, sampling_params)


@pytest.mark.skipif(not torch.cuda.is_available()
                    or "H200" not in torch.cuda.get_device_name(0),
                    reason="This test is only supported on H200 Hopper GPUs")
def test_gpt_oss_xqa_kernel_selection():
    """NVBug 5720470: GPT-OSS-20B must use XQA kernel (not MMHA) in decode.

    GPT-OSS-20B config: num_heads=64, num_kv_heads=8, head_dim=64, bfloat16.
    With batch_size=8 and sufficient decode history, the XQA heuristic
    (mayHavePerfGain) should select XQA over MMHA.

    Verification: use torch.profiler to capture CUDA kernels and assert that
    XQA kernel (kernel_mha) is launched instead of MMHA
    (masked_multihead_attention_kernel).
    """
    config_dict = json.loads(configs)
    # Use fewer layers for faster test
    config_dict["num_hidden_layers"] = 1
    gpt_oss_config = GptOssConfig.from_dict(config_dict)

    dtype = torch.bfloat16
    device = torch.device("cuda")

    model_config = ModelConfig(pretrained_config=gpt_oss_config,
                               attn_backend="TRTLLM")
    with torch.no_grad():
        model = GptOssForCausalLM(model_config).cuda()
        # Cast model weights to bfloat16 but keep float32 params (e.g. sinks).
        for name, param in model.named_parameters():
            if param.dtype == torch.float32 and 'sinks' not in name:
                param.data = param.data.to(dtype)
            elif param.dtype not in (torch.float32, dtype):
                param.data = param.data.to(dtype)

    # All-decode batch: 8 sequences in generation phase with past history.
    # This triggers the generation-phase attention where XQA/MMHA dispatch
    # happens. batch_size=8, past_kv >= 128 ensures the occupancy heuristic
    # selects XQA: num_kv_heads(8) * batch(8) * multi_block(>=1) * 4.0 >= SM_count.
    batch_size = 8
    context_sequence_length = []  # no context (prefill) sequences
    sequence_length = [1] * batch_size  # all decode, 1 new token each
    past_seen_tokens = [256] * batch_size  # enough history for multi-block
    request_ids = list(range(batch_size))
    token_nums = [p + s for p, s in zip(past_seen_tokens, sequence_length)]
    prompt_lens = past_seen_tokens

    num_blocks = 200
    tokens_per_block = 128
    head_dim = gpt_oss_config.head_dim
    num_layers = gpt_oss_config.num_hidden_layers
    num_kv_heads = gpt_oss_config.num_key_value_heads
    max_seq_len = num_blocks * tokens_per_block

    kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    kv_cache_config = BindingsKvCacheConfig(max_tokens=num_blocks *
                                            tokens_per_block)
    kv_cache_manager = KVCacheManager(
        kv_cache_config,
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=kv_cache_dtype,
    )
    kv_cache_manager.add_dummy_requests(request_ids, token_nums)

    metadata_cls = get_attention_backend(model_config.attn_backend).Metadata
    attn_metadata = metadata_cls(
        seq_lens=torch.tensor(sequence_length, dtype=torch.int32),
        num_contexts=len(context_sequence_length),
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=past_seen_tokens,
        ),
        kv_cache_manager=kv_cache_manager,
        request_ids=request_ids,
        prompt_lens=prompt_lens,
        max_num_requests=batch_size,
        max_num_tokens=8192,
    )

    # 1 token per decode sequence
    input_ids = torch.randint(0,
                              gpt_oss_config.vocab_size, (batch_size, ),
                              dtype=torch.int32,
                              device=device)
    position_ids = torch.tensor(past_seen_tokens,
                                dtype=torch.long,
                                device=device).unsqueeze(0)

    # Warm-up run (JIT compile XQA kernels, allocate buffers).
    with torch.inference_mode():
        attn_metadata.prepare()
        model.forward(input_ids=input_ids,
                      position_ids=position_ids,
                      attn_metadata=attn_metadata)

    # Profiled run: capture CUDA kernel names to verify XQA dispatch.
    kernel_names = []
    with torch.inference_mode(), \
         torch.profiler.profile(activities=[ProfilerActivity.CUDA]) as prof:
        attn_metadata.prepare()
        logits = model.forward(input_ids=input_ids,
                               position_ids=position_ids,
                               attn_metadata=attn_metadata)

    assert logits.shape[0] == batch_size, \
        f"Expected {batch_size} logits, got {logits.shape[0]}"

    kv_cache_manager.shutdown()

    # Collect CUDA kernel names from the profiler trace.
    kernel_names = [
        evt.key for evt in prof.key_averages()
        if evt.device_type == torch.autograd.DeviceType.CUDA
    ]
    all_kernels = " ".join(kernel_names)

    # XQA kernel: "kernel_mha" (from decoderXQA JIT).
    # MMHA kernel: "masked_multihead_attention_kernel".
    has_xqa = any("kernel_mha" in k for k in kernel_names)
    has_mmha = any("masked_multihead_attention_kernel" in k
                   for k in kernel_names)

    assert has_xqa, (
        "GPT-OSS-20B decode did not launch XQA kernel (kernel_mha). "
        f"See NVBug 5720470. Captured CUDA kernels: {all_kernels}")
    assert not has_mmha, (
        "GPT-OSS-20B decode launched MMHA kernel instead of XQA. "
        f"See NVBug 5720470. Captured CUDA kernels: {all_kernels}")
