# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A checkpoint load deferred past a KernelBuilder rehearsal ends where a direct load does."""

import pytest
import torch

from tensorrt_llm._torch import kernel_builder
from tensorrt_llm._torch.kernel_builder import KernelBuilder
from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import HfCheckpointLoader
from tensorrt_llm._torch.pyexecutor.model_loader import (
    ModelLoader,
    _ModuleTree,
    fill_rehearsal_values,
    runner_rehearses,
)
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.mapping import Mapping

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        torch_dtype="bfloat16",
    )
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("llama")
    transformers.LlamaForCausalLM(config).to(torch.bfloat16).save_pretrained(path)
    return str(path)


def _load(checkpoint, defer_checkpoint):
    loader = ModelLoader(
        llm_args=TorchLlmArgs(model=checkpoint),
        mapping=Mapping(),
        spec_config=None,
        sparse_attention_config=None,
        max_num_tokens=128,
        max_seq_len=128,
    )
    model, _ = loader.load(checkpoint, HfCheckpointLoader(), defer_checkpoint=defer_checkpoint)
    return loader, model


def _state(model):
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def _kernel_builder(monkeypatch, rehearse):
    # No op family: the families would queue real kernel builds.
    monkeypatch.setattr(kernel_builder, "_FAMILIES", {})
    monkeypatch.setattr(
        KernelBuilder, "_instance", KernelBuilder(threads=1, enabled=True, rehearse=rehearse)
    )


@cuda
def test_a_deferred_load_matches_a_direct_one(checkpoint, monkeypatch):
    _kernel_builder(monkeypatch, rehearse=True)
    _, direct = _load(checkpoint, defer_checkpoint=False)
    expected = _state(direct)
    del direct

    loader, model = _load(checkpoint, defer_checkpoint=True)
    assert loader.has_deferred_checkpoint
    rehearsal = _state(model)
    # The model holds dummy weights until the checkpoint loads.
    assert any(not torch.equal(rehearsal[name], expected[name]) for name in expected)
    loader.load_deferred_checkpoint(model)
    assert not loader.has_deferred_checkpoint
    loaded = _state(model)
    assert loaded.keys() == expected.keys()
    for name, tensor in expected.items():
        assert torch.equal(loaded[name], tensor), name


@cuda
def test_no_deferral_with_the_rehearsal_off(checkpoint, monkeypatch):
    _kernel_builder(monkeypatch, rehearse=False)
    loader, _ = _load(checkpoint, defer_checkpoint=True)
    assert not loader.has_deferred_checkpoint


@pytest.mark.cpu_only
def test_only_a_decoder_runner_with_context_warmups_rehearses():
    from types import SimpleNamespace

    def model(**config):
        config = {"is_generation": True, "is_encoder_decoder": False, **config}
        return SimpleNamespace(model_config=SimpleNamespace(**config))

    def args(**flags):
        return SimpleNamespace(**{"mm_encoder_only": False, "encode_only": False, **flags})

    assert runner_rehearses(model(), args())
    # Encoder-decoder: no context warmups. Pooling, encode-only, multimodal encoder: no decoder.
    assert not runner_rehearses(model(is_encoder_decoder=True), args())
    assert not runner_rehearses(model(is_generation=False), args())
    assert not runner_rehearses(model(), args(encode_only=True))
    assert not runner_rehearses(model(), args(mm_encoder_only=True))


@pytest.mark.cpu_only
def test_the_module_tree_goes_back_as_created():
    draft = torch.nn.Module()
    draft.embed_tokens = None  # an attribute, not a submodule
    draft.lm_head = torch.nn.Linear(2, 4)
    root = torch.nn.Module()
    root.draft = draft
    root.embed_tokens = torch.nn.Embedding(4, 2)
    tree = _ModuleTree(root)
    # Tying the draft to the target moves the name to the submodules.
    draft.embed_tokens = root.embed_tokens
    draft.lm_head = None
    tree.restore()
    assert draft.embed_tokens is None and "embed_tokens" not in draft._modules
    assert isinstance(draft.lm_head, torch.nn.Linear)


@cuda
def test_rehearsal_values_are_bounded_and_indices_zero():
    floats = torch.full((64,), float("nan"), device="cuda", dtype=torch.bfloat16)
    fp8 = torch.empty(64, device="cuda", dtype=torch.float8_e4m3fn)
    index = torch.full((64,), -7, device="cuda", dtype=torch.int32)
    fill_rehearsal_values([floats, fp8, index])
    assert floats.abs().max().item() <= 1e-3
    assert fp8.float().abs().max().item() <= 3.0
    assert torch.equal(index, torch.zeros_like(index))
