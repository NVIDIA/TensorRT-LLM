# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest
import torch
import transformers
from _torch.modeling.test_modeling_gemma3 import GEMMA3_1B_CONFIG
from _torch.modeling.test_modeling_gemma4 import GEMMA4_SMALL_CONFIG

from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention, TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.hf.gemma3_weight_mapper import Gemma3HfWeightMapper
from tensorrt_llm._torch.models.checkpoints.hf.gemma4_weight_mapper import Gemma4HfWeightMapper
from tensorrt_llm._torch.models.modeling_gemma3 import Gemma3ForCausalLM
from tensorrt_llm._torch.models.modeling_gemma4 import Gemma4ForCausalLM
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo
from tensorrt_llm.sampling_params import SamplingParams


def _make_model(
    monkeypatch: pytest.MonkeyPatch, name: str, kv_dtype: str, *, trtllm_gen_only: bool = False
) -> tuple[
    Gemma3ForCausalLM | Gemma4ForCausalLM,
    transformers.Gemma3TextConfig | transformers.Gemma4TextConfig,
]:
    major, _ = torch.cuda.get_device_capability()
    if (name == "gemma4" or trtllm_gen_only) and major != 10:
        pytest.skip("Gemma4 H512 and trtllm-gen require SM100-family GPUs.")
    libraries = "flashinfer_trtllm_gen,fallback" if major == 10 else "fallback"
    monkeypatch.setenv("TLLM_FMHA_LIBS", "flashinfer_trtllm_gen" if trtllm_gen_only else libraries)
    torch.manual_seed(42)
    config_dict = deepcopy(GEMMA3_1B_CONFIG if name == "gemma3" else GEMMA4_SMALL_CONFIG)
    config_dict.update(
        vocab_size=1024,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=12,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=256,
        sliding_window=128,
        max_position_embeddings=65536 if kv_dtype == "nvfp4" else 32768,
    )
    if name == "gemma3":
        config = transformers.Gemma3TextConfig(**config_dict)
        hf_cls, trt_cls, mapper_cls = (
            transformers.Gemma3ForCausalLM,
            Gemma3ForCausalLM,
            Gemma3HfWeightMapper,
        )
    else:
        config_dict.update(
            global_head_dim=512,
            num_kv_shared_layers=2,
            layer_types=["sliding_attention", "sliding_attention", "full_attention"] * 4,
        )
        config = transformers.Gemma4TextConfig(**config_dict)
        hf_cls, trt_cls, mapper_cls = (
            transformers.Gemma4ForCausalLM,
            Gemma4ForCausalLM,
            Gemma4HfWeightMapper,
        )
    reference = hf_cls(config).to(dtype=torch.bfloat16, device="cuda").eval()
    model_config = ModelConfig(
        pretrained_config=config,
        attn_backend="TRTLLM",
        quant_config=QuantConfig(
            kv_cache_quant_algo={"bf16": None, "fp8": QuantAlgo.FP8, "nvfp4": QuantAlgo.NVFP4}[
                kv_dtype
            ]
        ),
    )
    model = trt_cls(model_config).to(device="cuda").eval()
    mapper = mapper_cls()
    mapper.init_model_and_config(model, model_config)
    model.load_weights(reference.state_dict(), mapper)
    for layer in model.model.layers:
        assert isinstance(layer.self_attn.attn, TrtllmAttention)
    return model, config


def _make_cache_manager(
    config: transformers.Gemma3TextConfig | transformers.Gemma4TextConfig,
    *,
    scratch: bool,
    kv_dtype: str,
    reuse: bool = False,
    max_seq_len: int = 32768,
    batch_size: int = 2,
) -> KVCacheManagerV2:
    if kv_dtype == "nvfp4":
        max_seq_len = max(max_seq_len, 65536)
    layer_types = config.layer_types
    windows = [128 if kind == "sliding_attention" else max_seq_len for kind in layer_types]
    head_dims = [
        config.head_dim
        if kind == "sliding_attention"
        else getattr(config, "global_head_dim", config.head_dim)
        for kind in layer_types
    ]
    return KVCacheManagerV2(
        KvCacheConfig(
            max_gpu_total_bytes=64 << 20,
            free_gpu_memory_fraction=0.9,
            pool_ratio=[0.5, 0.5],
            max_attention_window=windows,
            enable_block_reuse=reuse,
            enable_partial_reuse=False,
            enable_swa_scratch_reuse=scratch,
            dtype="auto" if kv_dtype == "bf16" else kv_dtype,
        ),
        CacheType.SELF,
        num_layers=config.num_hidden_layers,
        num_kv_heads=1,
        head_dim=head_dims,
        tokens_per_block=32,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        max_num_tokens=batch_size * 353,
        mapping=Mapping(world_size=1, rank=0, tp_size=1),
        dtype={"bf16": DataType.BF16, "fp8": DataType.FP8, "nvfp4": DataType.NVFP4}[kv_dtype],
        vocab_size=1024,
        disable_overlap_scheduler=True,
    )


def _run(
    model: Gemma3ForCausalLM | Gemma4ForCausalLM,
    config: transformers.Gemma3TextConfig | transformers.Gemma4TextConfig,
    *,
    scratch: bool,
    kv_dtype: str,
    chunk_size: int,
    capture_decode: bool = False,
    prefill_only: bool = False,
    batch_size: int = 2,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    prompt_len = 353
    manager = _make_cache_manager(config, scratch=scratch, kv_dtype=kv_dtype, batch_size=batch_size)
    active_ids = [11 * (index + 1) for index in range(batch_size)]
    requests = []
    hooks = []
    layer_outputs = []
    outputs = []
    saw_scratch = False
    try:
        for request_id in active_ids:
            request = LlmRequest(
                request_id=request_id,
                max_new_tokens=3,
                input_tokens=[1] * prompt_len,
                sampling_config=SamplingParams(temperature=0)._get_sampling_config(),
                is_streaming=False,
            )
            requests.append(request)
            assert manager.prepare_context(request)
        for layer in model.model.layers:
            hooks.append(
                layer.self_attn.register_forward_hook(
                    lambda _module, _args, output: layer_outputs.append(output.detach().clone())
                )
            )
        batches = []
        for start in range(0, prompt_len, chunk_size):
            batches.append((active_ids, start, min(start + chunk_size, prompt_len), True))
        if not prefill_only:
            batches.extend(
                [
                    (active_ids, prompt_len, prompt_len + 1, False),
                    (list(reversed(active_ids)), prompt_len + 1, prompt_len + 2, False),
                    ([active_ids[-1]], prompt_len + 2, prompt_len + 3, False),
                ]
            )
        for ids, start, end, context in batches:
            torch.cuda.synchronize()
            if ids == [active_ids[-1]]:
                while len(requests) > 1:
                    manager.free_resources(requests.pop(0))
            for request_id in ids:
                cache = manager.kv_cache_map[request_id]
                if not context:
                    assert cache.resize(None, start)
                    cache.enable_swa_scratch_reuse = False
                assert cache.resize(end, start)
                descs = [cache.get_scratch_desc(group) for group in range(manager.num_pools)]
                if context and scratch and end - start > 128 + 32:
                    assert any(descs), "Long prefill must allocate scratch slots."
                    saw_scratch = True
                else:
                    assert not any(descs)
            count = end - start
            metadata = TrtllmAttentionMetadata(
                seq_lens=torch.tensor([count] * len(ids), dtype=torch.int32),
                num_contexts=len(ids) if context else 0,
                kv_cache_params=KVCacheParams(
                    use_cache=True, num_cached_tokens_per_seq=[start] * len(ids)
                ),
                max_num_requests=batch_size,
                max_num_tokens=batch_size * prompt_len,
                kv_cache_manager=manager,
                request_ids=ids,
                prompt_lens=[count if context else prompt_len] * len(ids),
                runtime_features=AttentionRuntimeFeatures(chunked_prefill=True),
                is_cuda_graph=capture_decode and not context,
            )
            positions = torch.arange(start, end, dtype=torch.int32, device="cuda")
            input_ids = torch.cat([(positions + request_id * 7) % 1024 for request_id in ids])
            position_ids = positions.repeat(len(ids)).unsqueeze(0)
            metadata.prepare()

            def forward() -> torch.Tensor:
                return model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attn_metadata=metadata,
                    return_context_logits=True,
                )

            if metadata.is_cuda_graph:
                for _ in range(2):
                    forward()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    result = forward()
                input_ids.add_(1).remainder_(1024)
                graph.replay()
                torch.testing.assert_close(result, forward(), atol=0.005, rtol=0.005)
                outputs.append(result.detach().clone())
            else:
                outputs.append(forward().detach().clone())
        assert saw_scratch == (scratch and chunk_size > 160)
        for output in outputs + layer_outputs:
            assert torch.isfinite(output).all()
            assert torch.count_nonzero(output) > 0
        torch.cuda.synchronize()
        return outputs, layer_outputs
    finally:
        for hook in hooks:
            hook.remove()
        for request in requests:
            manager.free_resources(request)
        manager.shutdown()


@pytest.mark.parametrize(
    "name,kv_dtype,chunk_size,mode",
    [
        (name, dtype, chunk, "eager")
        for name in ("gemma3", "gemma4")
        for dtype in ("bf16", "fp8")
        for chunk in (353, 191, 32)
    ]
    + [
        (name, "bf16", 353, mode)
        for name in ("gemma3", "gemma4")
        for mode in ("trtllm_gen", "cuda_graph")
    ]
    + [("gemma3", "nvfp4", 353, "trtllm_gen")],
)
@torch.inference_mode()
def test_text_decoder_scratch_prefill_and_retained_decode(
    monkeypatch: pytest.MonkeyPatch, name: str, kv_dtype: str, chunk_size: int, mode: str
) -> None:
    model, config = _make_model(monkeypatch, name, kv_dtype, trtllm_gen_only=mode == "trtllm_gen")
    results = [
        _run(
            model,
            config,
            scratch=scratch,
            kv_dtype=kv_dtype,
            chunk_size=chunk_size,
            capture_decode=mode == "cuda_graph",
            prefill_only=kv_dtype == "nvfp4",
            batch_size=5 if mode == "trtllm_gen" and kv_dtype == "bf16" else 2,
        )
        for scratch in (False, True)
    ]
    tolerance = {"bf16": 0.005, "fp8": 0.02, "nvfp4": 0.05}[kv_dtype]
    for actual_group, reference_group in zip(results[1], results[0], strict=True):
        for observed, expected in zip(actual_group, reference_group, strict=True):
            torch.testing.assert_close(observed, expected, atol=tolerance, rtol=tolerance)


def _run_reused_prefix(
    model: Gemma3ForCausalLM | Gemma4ForCausalLM,
    config: transformers.Gemma3TextConfig | transformers.Gemma4TextConfig,
    scratch: bool,
) -> list[torch.Tensor]:
    manager = _make_cache_manager(
        config, scratch=scratch, kv_dtype="bf16", reuse=True, max_seq_len=1024
    )
    outputs = []
    requests = []
    try:
        for request_id, prompt_len, expected_reuse in ((11, 257, 0), (22, 610, 256)):
            tokens = list(range(prompt_len))
            request = LlmRequest(
                request_id=request_id,
                max_new_tokens=1,
                input_tokens=tokens,
                sampling_config=SamplingParams(temperature=0)._get_sampling_config(),
                is_streaming=False,
            )
            requests.append(request)
            assert manager.prepare_context(request)
            start = request.context_current_position
            assert start == expected_reuse
            assert manager.resize_context(request, prompt_len - start)
            cache = manager.kv_cache_map[request_id]
            assert (
                any(cache.get_scratch_desc(group) for group in range(manager.num_pools)) == scratch
            )
            metadata = TrtllmAttentionMetadata(
                seq_lens=torch.tensor([prompt_len - start], dtype=torch.int32),
                num_contexts=1,
                kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[start]),
                max_num_requests=1,
                max_num_tokens=610,
                kv_cache_manager=manager,
                request_ids=[request_id],
                prompt_lens=[prompt_len - start],
                runtime_features=AttentionRuntimeFeatures(cache_reuse=True),
            )
            positions = torch.arange(start, prompt_len, device="cuda", dtype=torch.int32)
            metadata.prepare()
            outputs.append(
                model(
                    input_ids=positions,
                    position_ids=positions.unsqueeze(0),
                    attn_metadata=metadata,
                    return_context_logits=True,
                )
                .detach()
                .clone()
            )
            request.context_current_position = prompt_len
            manager.try_commit_blocks(request)
            torch.cuda.synchronize()
            manager.free_resources(request)
            requests.remove(request)
        return outputs
    finally:
        for request in requests:
            manager.free_resources(request)
        manager.shutdown()


@pytest.mark.parametrize("name", ["gemma3", "gemma4"])
@torch.inference_mode()
def test_text_decoder_scratch_preserves_cached_prefix(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    model, config = _make_model(monkeypatch, name, "bf16")
    reference = _run_reused_prefix(model, config, False)
    actual = _run_reused_prefix(model, config, True)
    for observed, expected in zip(actual, reference, strict=True):
        torch.testing.assert_close(observed, expected, atol=0.005, rtol=0.005)
