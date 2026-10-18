# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the tep8 target's drafter fast paths together, on the published DFlash 2 drafter.

``drafter_block_conv`` re-classes the drafter's block convolutions;
``drafter_block_decode`` replaces its block
attention and writes out the block decode around them. The written-out decode
reaches the convolutions through their modules (``prepare`` / ``finish``), so
with both selected the fused convolution runs inside it. The fused convolution
is bitwise the built-in one, so the ``drafter_block_conv`` + ``drafter_block_decode``
decode is bitwise the ``drafter_block_decode`` decode alone -- outputs and arena, eager and under CUDA-graph replay,
at contexts below and above the 2048-token window.

Needs ``MUSESPARK_REFERENCE_DRAFTER`` (the published drafter checkpoint) and
flash-attn (the drafter's built-in VANILLA attention).
"""

import os

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.drafter_block_decode import (  # noqa: E501
    block_decode as draft,
)

assert torch.cuda.is_available(), "the drafter features require a CUDA device"

_DRAFTER_DIR = os.environ.get("MUSESPARK_REFERENCE_DRAFTER")

#: Context lengths below, just past and well past the drafter's 2048-token window.
CONTEXTS = (1_000, 2_049, 3_000)
#: Block sizes K + 1 for K = 3, 5, 7.
BLOCKS = (4, 6, 8)
_SEEDS = (11, 12)
_SLOT = 1


def _has_flash_attn():
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _DRAFTER_DIR or not _has_flash_attn(),
    reason="needs MUSESPARK_REFERENCE_DRAFTER and flash_attn",
)


@pytest.fixture(scope="module")
def drafter():
    """The published drafter (TP1) with the block decode installed; the fused convolution toggled per test."""
    from safetensors.torch import load_file
    from transformers import AutoConfig

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM

    pretrained = AutoConfig.from_pretrained(_DRAFTER_DIR)
    with torch.device("cuda"):
        model = DFlashForCausalLM(ModelConfig(pretrained_config=pretrained, attn_backend="TRTLLM"))
    model.load_weights(load_file(os.path.join(_DRAFTER_DIR, "model.safetensors")))
    assert model.has_block_conv and model.dflash_attention_backend == "VANILLA"
    draft.install(model, False, True)
    assert isinstance(model._dflash_flash_attention, draft.WindowedBlockAttention)
    assert model.dflash_forward.__func__ is draft.windowed_block_decode
    return model


def _convs(model):
    return [conv for convs in (model.attention_convs, model.mlp_convs) for conv in convs]


def _set_block_conv(model, on: bool) -> None:
    from tensorrt_llm._torch.models.modeling_dflash import DFlash2BlockConv

    if on:
        draft.install(model, True, True)
        assert all(type(c) is draft.FusedDFlash2BlockConv for c in _convs(model))
    else:
        for conv in _convs(model):
            conv.__class__ = DFlash2BlockConv
    assert model.dflash_forward.__func__ is draft.windowed_block_decode


class _Step:
    """One request's arena and the static inputs of its block decodes."""

    def __init__(self, model, ctx_len: int, block: int) -> None:
        config = model.config
        captured = torch.randn(
            ctx_len,
            config.hidden_size * len(model.target_layer_ids),
            generator=torch.Generator(device="cuda").manual_seed(ctx_len),
            device="cuda",
        ).to(torch.bfloat16)
        k, v = model.precompute_context_kv(
            model.project_target_hidden(captured), torch.arange(ctx_len, device="cuda")
        )
        shape = (2, model._num_attn_layers, ctx_len + 256, model._num_kv_heads, model._head_dim)
        self.pool_k = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
        self.pool_v = torch.zeros_like(self.pool_k)
        self.pool_k[_SLOT, :, :ctx_len] = k.permute(1, 0, 2, 3)
        self.pool_v[_SLOT, :, :ctx_len] = v.permute(1, 0, 2, 3)
        self.ctx_len, self.block = ctx_len, block
        self.noise = torch.empty(1, block, config.hidden_size, dtype=torch.bfloat16, device="cuda")
        self.positions = torch.arange(ctx_len, ctx_len + block, device="cuda").unsqueeze(0)
        self.num_ctx = torch.tensor([ctx_len], device="cuda")
        self.batch_idx = torch.tensor([_SLOT], device="cuda")
        self.noises = {
            seed: torch.randn(
                1,
                block,
                config.hidden_size,
                generator=torch.Generator(device="cuda").manual_seed(seed),
                device="cuda",
            ).to(torch.bfloat16)
            for seed in _SEEDS
        }

    def reset_block_columns(self) -> None:
        """Poison the block's arena columns: the decode's append must overwrite them."""
        end = self.ctx_len + self.block
        self.pool_k[_SLOT, :, self.ctx_len : end].fill_(float("nan"))
        self.pool_v[_SLOT, :, self.ctx_len : end].fill_(float("nan"))

    def forward(self, model) -> torch.Tensor:
        return model.dflash_forward(
            noise_embedding=self.noise,
            query_positions=self.positions,
            num_ctx_per_req=self.num_ctx,
            ctx_k_cache=self.pool_k,
            ctx_v_cache=self.pool_v,
            ctx_cache_batch_idx=self.batch_idx,
        )

    def eager(self, model, seed: int):
        self.noise.copy_(self.noises[seed])
        self.reset_block_columns()
        out = self.forward(model).clone()
        return out, self.pool_k[_SLOT].clone(), self.pool_v[_SLOT].clone()


def _assert_same(got, expected, what: str) -> None:
    for name, a, b in zip(("output", "arena K", "arena V"), got, expected):
        assert torch.equal(a, b), (
            f"{what}: {name} differs in {(a != b).sum().item()} elements, max abs "
            f"{(a.float() - b.float()).abs().nan_to_num(float('inf')).max().item():.3e}"
        )


@pytest.mark.parametrize("block", BLOCKS)
@pytest.mark.parametrize("ctx_len", CONTEXTS)
@torch.inference_mode()
def test_block_conv_under_block_decode_is_bitwise_block_decode_alone_and_runs_the_fused_convolution(
    drafter, ctx_len, block, monkeypatch
):
    step = _Step(drafter, ctx_len, block)
    _set_block_conv(drafter, on=False)
    expected = {seed: step.eager(drafter, seed) for seed in _SEEDS}

    _set_block_conv(drafter, on=True)
    calls = {"conv": 0, "attention": 0}
    fused_conv, window_attention = draft.dflash2_block_conv, draft.dflash_window_attention

    def conv_spy(*args, **kwargs):
        calls["conv"] += 1
        return fused_conv(*args, **kwargs)

    def attention_spy(*args, **kwargs):
        calls["attention"] += 1
        return window_attention(*args, **kwargs)

    def builtin_conv(*args, **kwargs):
        raise AssertionError("the built-in block convolution ran under drafter_block_conv")

    import tensorrt_llm._torch.models.modeling_dflash as modeling_dflash

    monkeypatch.setattr(draft, "dflash2_block_conv", conv_spy)
    monkeypatch.setattr(draft, "dflash_window_attention", attention_spy)
    monkeypatch.setattr(modeling_dflash, "dflash2_grouped_conv", builtin_conv)
    layers = len(drafter.model.layers)
    for seed in _SEEDS:
        calls.update(conv=0, attention=0)
        got = step.eager(drafter, seed)
        # Per layer: prepare + finish on the attention and on the MLP convolution,
        # and one windowed attention with the append fused.
        assert calls == {"conv": 4 * layers, "attention": layers}, calls
        _assert_same(got, expected[seed], f"ctx {ctx_len} block {block} seed {seed}")


@pytest.mark.parametrize("block", BLOCKS)
@pytest.mark.parametrize("ctx_len", CONTEXTS)
@torch.inference_mode()
def test_block_conv_under_block_decode_graph_replay_is_bitwise_the_eager_block_decode_decode(
    drafter, ctx_len, block
):
    step = _Step(drafter, ctx_len, block)
    _set_block_conv(drafter, on=False)
    expected = {seed: step.eager(drafter, seed) for seed in _SEEDS}

    _set_block_conv(drafter, on=True)
    step.noise.copy_(step.noises[_SEEDS[0]])
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            step.forward(drafter)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = step.forward(drafter)

    for seed in reversed(_SEEDS):
        step.noise.copy_(step.noises[seed])
        step.reset_block_columns()
        graph.replay()
        torch.cuda.synchronize()
        got = (static_out.clone(), step.pool_k[_SLOT].clone(), step.pool_v[_SLOT].clone())
        _assert_same(got, expected[seed], f"graph: ctx {ctx_len} block {block} seed {seed}")
