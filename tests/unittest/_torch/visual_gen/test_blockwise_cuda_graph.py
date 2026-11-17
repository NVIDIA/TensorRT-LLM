# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-block CUDA graphs (``BlockCUDAGraphRunner``).

With Cache-DiT and CUDA graphs both enabled, VisualGen captures one graph per
transformer block instead of one per transformer forward, so Cache-DiT's
skip decision stays in Python between graph replays. A block replayed from a
graph must produce bit-identical output to the same block run eagerly.
"""

import inspect
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.visual_gen.cuda_graph_runner import (
    BlockCUDAGraphRunner,
    SharedGraphPool,
    wrap_blocks_with_cuda_graphs,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

_DIM = 64


class _RopeLikeBlock(nn.Module):
    """Takes a tensor tuple (like FLUX's ``image_rotary_emb``) and a Python scalar."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(_DIM, _DIM)
        self.norm = nn.LayerNorm(_DIM)

    def forward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb, scale=1.0):
        cos, sin = image_rotary_emb
        hidden_states = self.norm(self.proj(hidden_states) * cos + sin) * scale + temb[:, None]
        return encoder_hidden_states * 2.0, hidden_states


class _TwoBlockModel(nn.Module):
    """Calls its blocks the way FLUX does; optionally registers a timestep graph key."""

    def __init__(self, timestep_key: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList([_RopeLikeBlock(), _RopeLikeBlock()])
        self._timestep_key = timestep_key

    def register_cuda_graph_extra_key_fns(self, runner) -> None:
        if self._timestep_key:
            runner.register_extra_key_fn(
                "late_phase", lambda *args, **kwargs: bool(kwargs["timestep"] < 500)
            )

    def forward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb, timestep=0):
        for block in self.blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )
        return encoder_hidden_states, hidden_states


def _inputs(seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, device="cuda", generator=generator)

    return dict(
        hidden_states=rand(2, 16, _DIM),
        encoder_hidden_states=rand(2, 8, _DIM),
        temb=rand(2, _DIM),
        image_rotary_emb=(rand(16, _DIM), rand(16, _DIM)),
    )


def _assert_same(actual, expected):
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return _TwoBlockModel().cuda().eval()


@torch.inference_mode()
def test_replay_matches_eager_and_copies_nested_tensors(model):
    """Each call replays one graph per block and matches eager bit for bit,
    including when only the nested ``image_rotary_emb`` tensors change."""
    first = _inputs(1)
    rope_only_changed = dict(first, image_rotary_emb=_inputs(2)["image_rotary_emb"])
    calls = [first, _inputs(3), rope_only_changed, first]
    expected = [model(**call) for call in calls]

    runners = wrap_blocks_with_cuda_graphs(model, ["blocks"], SharedGraphPool())
    for call, reference in zip(calls, expected):
        _assert_same(model(**call), reference)

    assert sorted(runners) == ["transformer.blocks.0", "transformer.blocks.1"]
    assert all(len(runner.graphs) == 1 for runner in runners.values())


@torch.inference_mode()
def test_positional_and_keyword_calls_share_a_graph(model):
    block = model.blocks[0]
    call = _inputs(1)
    expected = block(**call)
    runners = wrap_blocks_with_cuda_graphs(model, ["blocks"])

    positional = block(
        call["hidden_states"], call["encoder_hidden_states"], call["temb"], call["image_rotary_emb"]
    )
    keyword = block(**call)

    _assert_same(positional, expected)
    _assert_same(keyword, expected)
    assert len(runners["transformer.blocks.0"].graphs) == 1


@torch.inference_mode()
def test_non_tensor_values_select_separate_graphs(model):
    block = model.blocks[0]
    call = _inputs(1)
    expected = {scale: block(**call, scale=scale) for scale in (1.0, 0.5)}
    runners = wrap_blocks_with_cuda_graphs(model, ["blocks"])

    for scale in (1.0, 0.5, 1.0):
        _assert_same(block(**call, scale=scale), expected[scale])
    assert len(runners["transformer.blocks.0"].graphs) == 2


@torch.inference_mode()
def test_model_extra_keys_follow_model_forward_inputs():
    """A model's graph-key callback reads ``model.forward`` kwargs that blocks
    never receive; each block must still get one graph per key value."""
    torch.manual_seed(0)
    model = _TwoBlockModel(timestep_key=True).cuda().eval()
    call = _inputs(1)
    expected = model(**call)
    runners = wrap_blocks_with_cuda_graphs(model, ["blocks"])

    for timestep in (900, 800, 100, 50):
        _assert_same(model(**call, timestep=timestep), expected)
    assert all(len(runner.graphs) == 2 for runner in runners.values())


@torch.inference_mode()
def test_capture_disallowed_or_disabled_runs_eagerly(model):
    call = _inputs(1)
    expected = model(**call)
    runners = wrap_blocks_with_cuda_graphs(model, ["blocks"])

    for runner in runners.values():
        runner.allow_capture = False
    _assert_same(model(**call), expected)
    assert not any(runner.graphs for runner in runners.values())

    for runner in runners.values():
        runner.enabled = False
        runner.allow_capture = True
    _assert_same(model(**call), expected)
    assert not any(runner.graphs for runner in runners.values())


def test_blocks_keep_identity_and_signature(model):
    """Cache-DiT holds references to the blocks and inspects ``block.forward``."""
    blocks_before = list(model.blocks)
    signature_before = inspect.signature(model.blocks[0].forward)

    runners = wrap_blocks_with_cuda_graphs(model, ["blocks"])

    assert list(model.blocks) == blocks_before
    assert inspect.signature(model.blocks[0].forward) == signature_before
    assert all(isinstance(runner, BlockCUDAGraphRunner) for runner in runners.values())


# ---------------------------------------------------------------------------
# Real FLUX.1 blocks: graph replay is bit-identical to eager
# ---------------------------------------------------------------------------

_FLUX1_TINY_CONFIG = {
    "attention_head_dim": 128,
    "guidance_embeds": True,
    "in_channels": 64,
    "joint_attention_dim": 512,
    "num_attention_heads": 4,
    "num_layers": 2,
    "num_single_layers": 2,
    "patch_size": 1,
    "pooled_projection_dim": 256,
    "torch_dtype": "bfloat16",
    "axes_dim": [16, 56, 56],
}


def _flux_inputs(seed: int, seq_len: int = 256, text_len: int = 64):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    config = _FLUX1_TINY_CONFIG

    def rand(*shape):
        return torch.randn(*shape, device="cuda", generator=generator, dtype=torch.bfloat16)

    side = int(seq_len**0.5)
    img_ids = torch.zeros(side, side, 3, device="cuda")
    img_ids[..., 1] = torch.arange(side, device="cuda")[:, None]
    img_ids[..., 2] = torch.arange(side, device="cuda")[None, :]
    return dict(
        hidden_states=rand(1, seq_len, config["in_channels"]),
        encoder_hidden_states=rand(1, text_len, config["joint_attention_dim"]),
        pooled_projections=rand(1, config["pooled_projection_dim"]),
        timestep=torch.tensor([0.5 + seed / 100], device="cuda", dtype=torch.bfloat16),
        guidance=torch.tensor([3.5], device="cuda", dtype=torch.bfloat16),
        img_ids=img_ids.reshape(seq_len, 3),
        txt_ids=torch.zeros(text_len, 3, device="cuda"),
    )


def _tiny_flux1(backend: str):
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
    from tensorrt_llm._torch.visual_gen.models.flux.transformer_flux import FluxTransformer2DModel
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.visual_gen.args import AttentionConfig

    model_config = DiffusionModelConfig(
        pretrained_config=SimpleNamespace(**deepcopy(_FLUX1_TINY_CONFIG)),
        quant_config=QuantConfig(),
        mapping=Mapping(),
        attention=AttentionConfig(backend=backend),
        skip_create_weights_in_init=False,
    )
    torch.manual_seed(0)
    model = FluxTransformer2DModel(model_config).to("cuda", dtype=torch.bfloat16).eval()
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0.0, 0.02)
    return model


@pytest.mark.parametrize(
    "backend,compiled",
    [("VANILLA", False), ("VANILLA", True), ("FA4", False)],
    ids=["vanilla-eager", "vanilla-compiled", "fa4-eager"],
)
@torch.inference_mode()
def test_flux1_blocks_replay_bit_identical(backend, compiled):
    from tensorrt_llm._utils import get_sm_version

    if backend == "FA4" and get_sm_version() < 100:
        pytest.skip("FA4 attention needs SM100 or newer")

    model = _tiny_flux1(backend)
    block_attrs = ["transformer_blocks", "single_transformer_blocks"]
    if compiled:
        for attr in block_attrs:
            setattr(model, attr, nn.ModuleList(torch.compile(b) for b in getattr(model, attr)))

    calls = [_flux_inputs(1), _flux_inputs(2), _flux_inputs(1)]
    expected = [model(**call)["sample"].clone() for call in calls]
    assert all(torch.isfinite(out).all() for out in expected)

    runners = wrap_blocks_with_cuda_graphs(model, block_attrs, SharedGraphPool())
    for call, reference in zip(calls, expected):
        torch.testing.assert_close(model(**call)["sample"], reference, rtol=0, atol=0)

    assert len(runners) == 4
    assert all(len(runner.graphs) == 1 for runner in runners.values())
    if compiled:
        torch._dynamo.reset()
