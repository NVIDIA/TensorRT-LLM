# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.llmapi.llm_args import DecodeCudaGraphConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("use_v2", [False, True], ids=["v1", "v2"])
def test_int8_kv_checkpoint_generation_with_cuda_graph(tmp_path: Path, use_v2: bool) -> None:
    """Explicit INT8 generation is deterministic across eager and CUDA Graph runs."""
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=256,
        dtype="float16",
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config.architectures = ["LlamaForCausalLM"]
    config.save_pretrained(tmp_path)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260909)
        model = LlamaForCausalLM(config).half()
    weights = model.state_dict()
    # Known test scales for this small random checkpoint; no downloaded model.
    for layer in range(2):
        prefix = f"model.layers.{layer}.self_attn"
        weights[f"{prefix}.k_proj.k_scale"] = torch.tensor([1 / 64], dtype=torch.float32)
        weights[f"{prefix}.v_proj.v_scale"] = torch.tensor([1 / 32], dtype=torch.float32)
    save_file(weights, str(tmp_path / "model.safetensors"), metadata={"format": "pt"})
    del weights, model

    results = []
    resolved_dtypes = []
    for use_cuda_graph in (False, True):
        with LLM(
            model=tmp_path,
            backend="pytorch",
            skip_tokenizer_init=True,
            dtype="float16",
            attn_backend="TRTLLM",
            max_batch_size=2,
            max_num_tokens=128,
            max_seq_len=256,
            cuda_graph_config=DecodeCudaGraphConfig(batch_sizes=[1, 2]) if use_cuda_graph else None,
            enable_chunked_prefill=False,
            disable_overlap_scheduler=True,
            kv_cache_config=KvCacheConfig(
                dtype="int8",
                enable_block_reuse=False,
                free_gpu_memory_fraction=0.001,
                use_kv_cache_manager_v2=use_v2,
            ),
        ) as llm:
            resolved_dtypes.append(llm.args.quant_config.kv_cache_quant_algo)
            assert llm.args.kv_cache_config.dtype == "int8"
            assert llm.args.kv_cache_config.use_kv_cache_manager_v2 == use_v2
            assert (llm.args.cuda_graph_config is not None) == use_cuda_graph
            assert resolved_dtypes[-1] == QuantAlgo.INT8
            outputs = llm.generate(
                [[1] + list(range(10, 72)), [1] + list(range(100, 130))],
                SamplingParams(end_id=2, pad_id=0, max_tokens=8, temperature=0, ignore_eos=True),
            )
            ids = [output.outputs[0].token_ids for output in outputs]
            assert len(ids) == 2 and all(len(tokens) == 8 for tokens in ids)
            results.append(ids)
    assert resolved_dtypes == [QuantAlgo.INT8, QuantAlgo.INT8]
    assert results[0] == results[1]
