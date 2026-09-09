# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.llmapi.llm_args import DecodeCudaGraphConfig


def test_int8_kv_checkpoint_generation_with_cuda_graph(tmp_path: Path) -> None:
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
    for cache_dtype in ("auto", "int8"):
        with LLM(
            model=tmp_path,
            backend="pytorch",
            skip_tokenizer_init=True,
            dtype="float16",
            attn_backend="TRTLLM",
            max_batch_size=2,
            max_num_tokens=128,
            max_seq_len=256,
            cuda_graph_config=DecodeCudaGraphConfig(batch_sizes=[1, 2]),
            enable_chunked_prefill=False,
            disable_overlap_scheduler=True,
            kv_cache_config=KvCacheConfig(
                dtype=cache_dtype, enable_block_reuse=False, max_tokens=512
            ),
        ) as llm:
            outputs = llm.generate(
                [[1] + list(range(10, 72)), [1] + list(range(100, 130))],
                SamplingParams(end_id=2, pad_id=0, max_tokens=8, temperature=0, ignore_eos=True),
            )
            ids = [output.outputs[0].token_ids for output in outputs]
            assert len(ids) == 2 and all(len(tokens) == 8 for tokens in ids)
            results.append(ids)
    assert results[0] == results[1]
