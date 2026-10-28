# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch.profiler import ProfilerActivity
from transformers import AutoTokenizer, GptOssConfig
from utils.llm_data import llm_models_root
from utils.util import skip_no_hopper

import tensorrt_llm
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_gpt_oss import (
    GptOssForCausalLM, MLPBlock, TransformerBlock,
    _fused_mlp_nvfp4_input_enabled)
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.utils import Fp4QuantizedTensor
from tensorrt_llm.bindings.executor import \
    KvCacheConfig as BindingsKvCacheConfig
from tensorrt_llm.llmapi import (CudaGraphConfig, Eagle3DecodingConfig,
                                 KvCacheConfig, MoeConfig)
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.llmapi.llm_utils import _resolve_kv_cache_manager_v2_auto
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


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("prequantized", [False, True])
def test_gpt_oss_mlp_uses_bf16_router_view_and_fp4_expert_input(
        tp_size: int, prequantized: bool) -> None:
    mlp = MLPBlock.__new__(MLPBlock)
    torch.nn.Module.__init__(mlp)
    mlp.enable_attention_dp = tp_size > 1
    mlp.mapping = SimpleNamespace(tp_size=tp_size)
    counts = [3] + [7] * (tp_size - 1)
    metadata = SimpleNamespace(all_rank_num_tokens=counts)
    norm_out = torch.randn((3, 16), dtype=torch.bfloat16)
    carrier = Fp4QuantizedTensor(
        torch.zeros((3, 24), dtype=torch.uint8),
        torch.zeros((3, 3), dtype=torch.uint8),
        is_sf_swizzled=False,
        unquantized_hidden_states=norm_out,
    )
    router_logits = torch.randn((3, 4), dtype=torch.bfloat16)
    mlp.compute_gate_output = Mock(return_value=router_logits)
    mlp.experts = Mock(return_value=torch.zeros_like(norm_out))
    residual = torch.randn_like(norm_out)

    output, residual_out = mlp(carrier if prequantized else norm_out,
                               metadata,
                               residual=residual)

    torch.testing.assert_close(output, torch.zeros_like(norm_out))
    assert residual_out is residual
    router_call = mlp.compute_gate_output.call_args
    torch.testing.assert_close(router_call.args[0], norm_out)
    assert router_call.kwargs == {"lora_params": None}
    expert_call = mlp.experts.call_args
    if prequantized:
        assert expert_call.kwargs["x"] is carrier
        assert expert_call.kwargs["output_dtype"] == torch.bfloat16
    else:
        torch.testing.assert_close(expert_call.kwargs["x"], norm_out)
        assert "output_dtype" not in expert_call.kwargs
    assert expert_call.kwargs["router_logits"] is router_logits
    if mlp.enable_attention_dp:
        assert expert_call.kwargs["all_rank_num_tokens"] is counts
        assert expert_call.kwargs["use_dp_padding"] is False
    else:
        assert "all_rank_num_tokens" not in expert_call.kwargs

    first, second = carrier.split([1, 2])
    assert first.fp4_tensor.shape[0] == 1
    assert second.fp4_tensor.shape[0] == 2
    torch.testing.assert_close(first.unquantized_hidden_states, norm_out[:1])
    torch.testing.assert_close(second.unquantized_hidden_states, norm_out[1:])


@pytest.mark.parametrize("tp_size, enable_attn_dp", [(1, False), (2, True),
                                                     (4, True), (8, True)])
def test_gpt_oss_fused_nvfp4_mlp_input_kill_switch(
        tp_size: int, enable_attn_dp: bool) -> None:
    module = "tensorrt_llm._torch.models.modeling_gpt_oss"
    with patch(f"{module}._GPT_OSS_DISABLE_FUSED_MLP_NVFP4_INPUT", True), \
            patch(f"{module}.add_rmsnorm_fp4_quant_available") as available:
        assert not _fused_mlp_nvfp4_input_enabled(tp_size, enable_attn_dp, 103)

    available.assert_not_called()


@pytest.mark.parametrize("tp_size, enable_attn_dp, expected", [
    (1, False, True),
    (1, True, True),
    (2, True, True),
    (4, True, True),
    (8, True, True),
    (16, True, True),
    (2, False, False),
    (4, False, False),
    (8, False, False),
    (16, False, False),
])
def test_gpt_oss_fused_nvfp4_mlp_input_enabled_by_default(
        tp_size: int, enable_attn_dp: bool, expected: bool) -> None:
    module = "tensorrt_llm._torch.models.modeling_gpt_oss"
    with patch(f"{module}._GPT_OSS_DISABLE_FUSED_MLP_NVFP4_INPUT", False), \
            patch(f"{module}.add_rmsnorm_fp4_quant_available",
                  return_value=True) as available:
        assert _fused_mlp_nvfp4_input_enabled(tp_size, enable_attn_dp,
                                              103) is expected

    if expected:
        available.assert_called_once_with()
    else:
        available.assert_not_called()


@pytest.mark.parametrize("counts", [None, [2, 7], [2] + [7] * 7])
def test_gpt_oss_builds_unswizzled_fused_nvfp4_mlp_input(
        counts: list[int] | None) -> None:
    block = TransformerBlock.__new__(TransformerBlock)
    torch.nn.Module.__init__(block)
    block.enable_fused_mlp_nvfp4_input = True
    input_scale = torch.ones((1, ), dtype=torch.float32)
    block.mlp = SimpleNamespace(experts=SimpleNamespace(
        get_nvfp4_input_quant_params=Mock(return_value=(input_scale, 32))))
    weight = torch.ones((16, ), dtype=torch.bfloat16)
    block.post_attention_layernorm = SimpleNamespace(weight=weight,
                                                     variance_epsilon=1e-5)
    x = torch.randn((2, 16), dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    fp4 = torch.zeros((2, 16), dtype=torch.uint8)
    sf = torch.zeros((2, 2), dtype=torch.uint8)
    norm_out = torch.randn_like(x)

    with patch(
            "tensorrt_llm._torch.models.modeling_gpt_oss.add_rmsnorm_fp4_quant",
            return_value=(fp4, sf, norm_out)) as fused:
        result = block._try_fused_mlp_nvfp4_input(x, residual, None, counts)

    assert result is not None
    carrier, residual_out = result
    assert carrier.fp4_tensor is fp4
    assert carrier.scaling_factor is sf
    assert carrier.unquantized_hidden_states is norm_out
    assert not carrier.is_sf_swizzled
    assert residual_out is residual
    fused.assert_called_once_with(x, residual, weight, 1e-5, input_scale, 32)


@pytest.mark.parametrize("counts", [[0, 7], [7, 0], [0, 0], [7] * 7 + [0]])
def test_gpt_oss_fused_nvfp4_mlp_input_empty_rank_falls_back_on_all_ranks(
        counts: list[int]) -> None:
    block = TransformerBlock.__new__(TransformerBlock)
    torch.nn.Module.__init__(block)
    block.enable_fused_mlp_nvfp4_input = True
    block.mlp = SimpleNamespace(experts=Mock())

    with patch(
            "tensorrt_llm._torch.models.modeling_gpt_oss.add_rmsnorm_fp4_quant"
    ) as fused:
        for num_tokens in counts:
            x = torch.randn((num_tokens, 16), dtype=torch.bfloat16)
            residual = torch.randn_like(x)
            residual_before = residual.clone()
            assert block._try_fused_mlp_nvfp4_input(x, residual, None,
                                                    counts) is None
            torch.testing.assert_close(residual, residual_before)

    fused.assert_not_called()
    block.mlp.experts.get_nvfp4_input_quant_params.assert_not_called()


@pytest.mark.parametrize("counts", [[2, 7], [2, 0]])
def test_gpt_oss_fused_nvfp4_forward_passes_dp_token_counts(
        counts: list[int]) -> None:
    block = TransformerBlock.__new__(TransformerBlock)
    torch.nn.Module.__init__(block)
    block.enable_attn_dp = True
    block.is_tp = False
    x = torch.randn((2, 16), dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    metadata = SimpleNamespace(all_rank_num_tokens=counts)
    block.attn = Mock(return_value=(x, residual))
    block._try_fused_mlp_nvfp4_input = Mock(return_value=None)
    block.post_attention_layernorm = Mock(return_value=(x, residual))
    block.mlp = Mock(return_value=(x, residual))
    block.next_layer_layernorm = Mock(return_value=(x, residual))

    block(None, x, metadata, residual=residual)

    block._try_fused_mlp_nvfp4_input.assert_called_once_with(
        x, residual, None, counts)
    block.post_attention_layernorm.assert_called_once_with(x, residual)


def test_gpt_oss_fused_nvfp4_mlp_input_falls_back_for_lora() -> None:
    block = TransformerBlock.__new__(TransformerBlock)
    torch.nn.Module.__init__(block)
    block.enable_fused_mlp_nvfp4_input = True
    block.mlp = SimpleNamespace(experts=Mock())
    x = torch.randn((2, 16), dtype=torch.bfloat16)

    assert block._try_fused_mlp_nvfp4_input(x, torch.randn_like(x),
                                            {"adapter": object()}) is None
    block.mlp.experts.get_nvfp4_input_quant_params.assert_not_called()


def test_gpt_oss_fused_nvfp4_mlp_input_falls_back_for_other_backend() -> None:
    block = TransformerBlock.__new__(TransformerBlock)
    torch.nn.Module.__init__(block)
    block.enable_fused_mlp_nvfp4_input = True
    block.mlp = SimpleNamespace(experts=SimpleNamespace())
    x = torch.randn((2, 16), dtype=torch.bfloat16)

    assert block._try_fused_mlp_nvfp4_input(x, torch.randn_like(x),
                                            None) is None


def test_gpt_oss_prefers_python_transceiver() -> None:
    assert GptOssForCausalLM.get_preferred_transceiver_runtime() == "PYTHON"


def _resolve_gpt_oss_kv_cache_manager_v2(**llm_args_kwargs) -> bool:
    """Resolve the GPT-OSS preference through the model-loading path."""
    llm_args = TorchLlmArgs(model="/tmp/dummy_model", **llm_args_kwargs)
    return _resolve_kv_cache_manager_v2_auto(llm_args, GptOssForCausalLM)


def test_gpt_oss_model_preference_selects_v2():
    """GPT-OSS is VSWA, so "auto" resolves to KVCacheManagerV2."""
    assert _resolve_gpt_oss_kv_cache_manager_v2() is True


@pytest.mark.parametrize("user_setting", [False, True])
def test_gpt_oss_explicit_setting_wins(user_setting):
    """An explicit user value is never overridden by the model preference."""
    assert _resolve_gpt_oss_kv_cache_manager_v2(kv_cache_config=KvCacheConfig(
        use_kv_cache_manager_v2=user_setting)) is user_setting


def test_gpt_oss_one_model_eagle3_keeps_v2():
    """One-model Eagle3 shares the target engine, so V2 still applies."""
    assert _resolve_gpt_oss_kv_cache_manager_v2(
        speculative_config=Eagle3DecodingConfig(
            max_draft_len=3,
            speculative_model="/tmp/dummy_eagle_model")) is True


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
