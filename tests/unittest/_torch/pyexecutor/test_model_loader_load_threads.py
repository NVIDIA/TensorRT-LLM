# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TLLM_WEIGHT_LOAD_THREADS: a real checkpoint loads to the same bytes on more torch threads."""

import os

import pytest
import torch
from transformers import LlamaConfig
from transformers import LlamaForCausalLM as HFLlamaForCausalLM

from tensorrt_llm._torch.models.modeling_llama import LlamaForCausalLM
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader, construct_checkpoint_loader
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.mapping import Mapping


def _load(checkpoint_dir: str, monkeypatch: pytest.MonkeyPatch, load_threads: str | None):
    """Load the checkpoint; return every tensor's bytes and the thread counts the loads ran with."""
    seen = {"model": set(), "modules": set()}
    model_load = LlamaForCausalLM.load_weights
    module_load = Linear.load_weights

    def load_model(self, weights, weight_mapper=None, params_map=None, allow_partial_loading=False):
        seen["model"].add(torch.get_num_threads())
        model_load(
            self,
            weights,
            weight_mapper=weight_mapper,
            params_map=params_map,
            allow_partial_loading=allow_partial_loading,
        )

    def load_module(self, weights, allow_partial_loading=False):
        seen["modules"].add(torch.get_num_threads())
        module_load(self, weights, allow_partial_loading=allow_partial_loading)

    with monkeypatch.context() as patch:
        if load_threads is None:
            patch.delenv("TLLM_WEIGHT_LOAD_THREADS", raising=False)
        else:
            patch.setenv("TLLM_WEIGHT_LOAD_THREADS", load_threads)
        patch.setattr(LlamaForCausalLM, "load_weights", load_model)
        patch.setattr(Linear, "load_weights", load_module)
        loader = ModelLoader(
            llm_args=TorchLlmArgs(model=checkpoint_dir),
            mapping=Mapping(),
            spec_config=None,
            sparse_attention_config=None,
            max_num_tokens=64,
            max_seq_len=64,
        )
        model, _ = loader.load(checkpoint_dir, construct_checkpoint_loader(None, "HF"))
    tensors = dict(model.named_parameters())
    tensors.update((f"buffer:{name}", buf) for name, buf in model.named_buffers())
    data = {
        name: t.detach().cpu().contiguous().view(-1).view(torch.uint8)
        for name, t in tensors.items()
    }
    return data, seen


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_load_threads_same_bytes_and_restored(tmp_path, monkeypatch):
    """Unset keeps 1 thread; N raises model.load_weights only, and 1 is back after the load."""
    config = LlamaConfig(
        hidden_size=256,
        intermediate_size=768,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=1024,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    HFLlamaForCausalLM(config).to(torch.bfloat16).save_pretrained(tmp_path)
    monkeypatch.delenv("TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL", raising=False)
    raised = min(4, len(os.sched_getaffinity(0)))
    saved = torch.get_num_threads()
    # Start like a worker launched with OMP_NUM_THREADS=1.
    torch.set_num_threads(1)
    try:
        off, off_threads = _load(str(tmp_path), monkeypatch, None)
        assert torch.get_num_threads() == 1
        on, on_threads = _load(str(tmp_path), monkeypatch, "4")
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(saved)

    assert off_threads == {"model": {1}, "modules": {1}}
    assert on_threads == {"model": {raised}, "modules": {1}}
    assert off.keys() == on.keys()
    for name, data in off.items():
        assert torch.equal(data, on[name]), name
