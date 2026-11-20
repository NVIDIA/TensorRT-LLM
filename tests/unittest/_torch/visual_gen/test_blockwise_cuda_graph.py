# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-block CUDA graphs (used with Cache-DiT) must replay bit-identically to eager blocks."""

import gc
import weakref
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils._pytree import tree_leaves

from tensorrt_llm._torch.visual_gen.cuda_graph_runner import (
    SharedGraphPool,
    resolved_extra_key,
    resolved_extra_keys_scope,
    wrap_blocks_with_cuda_graphs,
)
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline

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


class _InPlaceInputBlock(_RopeLikeBlock):
    def forward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb, scale=1.0):
        hidden_states.mul_(0.5)
        return super().forward(hidden_states, encoder_hidden_states, temb, image_rotary_emb, scale)


class _PhaseBlock(_RopeLikeBlock):
    """Reads the model's graph key on the host, as the sparse-attention backends do."""

    def forward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb, scale=1.0):
        if resolved_extra_key("late_phase"):
            scale = scale * 2.0
        return super().forward(hidden_states, encoder_hidden_states, temb, image_rotary_emb, scale)


class _FirstCallStateBlock(_RopeLikeBlock):
    """Keeps a value from its first call per shape, as VC-Attention keeps its token grouping."""

    def __init__(self):
        super().__init__()
        self.first_call_mean = {}

    def forward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb, scale=1.0):
        if hidden_states.shape not in self.first_call_mean:
            self.first_call_mean[hidden_states.shape] = hidden_states.mean()
        hidden_states = hidden_states + self.first_call_mean[hidden_states.shape]
        return super().forward(hidden_states, encoder_hidden_states, temb, image_rotary_emb, scale)


class _TwoBlockModel(nn.Module):
    """Calls its blocks the way FLUX does; optionally registers a timestep graph key."""

    def __init__(self, timestep_key: bool = False, block_cls=_RopeLikeBlock):
        super().__init__()
        self.blocks = nn.ModuleList([block_cls(), block_cls()])
        self._timestep_key = timestep_key

    def register_cuda_graph_extra_key_fns(self, runner) -> None:
        if self._timestep_key:
            runner.register_extra_key_fn(
                "late_phase", lambda *args, **kwargs: bool(kwargs["timestep"] < 500), max_graphs=4
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


class _TembChangingModel(_TwoBlockModel):
    """Changes temb in place between its two block calls."""

    def forward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb, timestep=0):
        encoder_hidden_states, hidden_states = self.blocks[0](
            hidden_states, encoder_hidden_states, temb, image_rotary_emb
        )
        temb.add_(1.0)
        return self.blocks[1](hidden_states, encoder_hidden_states, temb, image_rotary_emb)


class _PairBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(_DIM, _DIM)

    def forward(self, x, y):
        return self.proj(x) + 2.0 * y


class _PairModel(nn.Module):
    """Passes the first block's output to both inputs of the second block."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_PairBlock(), _PairBlock()])

    def forward(self, x, y):
        out = self.blocks[0](x, y)
        return self.blocks[1](out, out)


def _inputs(seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, device="cuda", generator=generator)

    return dict(
        hidden_states=rand(2, 16, _DIM),
        encoder_hidden_states=rand(2, 8, _DIM),
        temb=rand(2, _DIM),
        image_rotary_emb=(rand(1, _DIM).expand(16, -1), rand(1, _DIM).expand(16, -1)),
    )


def _clone_call(call):
    return {
        name: tuple(t.clone() for t in value) if isinstance(value, tuple) else value.clone()
        for name, value in call.items()
    }


def _assert_same(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return _TwoBlockModel().cuda().eval()


@torch.inference_mode()
def test_replay_matches_eager():
    """New inputs, nested broadcast rotary tensors and positional calls share a block's graph.

    The blocks keep state from their first call, which capture must build from the real inputs.
    """
    torch.manual_seed(0)
    model = _TwoBlockModel(block_cls=_FirstCallStateBlock).cuda().eval()
    graph_model = deepcopy(model)
    first = _inputs(1)
    rope_only_changed = dict(first, image_rotary_emb=_inputs(2)["image_rotary_emb"])
    cos = first["image_rotary_emb"][0]
    stepped = tuple(torch.randn(16, 2 * _DIM, device="cuda")[:, ::2] for _ in range(2))
    calls = [first, _inputs(3), rope_only_changed, first, dict(first, image_rotary_emb=stepped)]
    expected = [model(**call) for call in calls]
    expected_block = model.blocks[0](**first)
    runners = wrap_blocks_with_cuda_graphs(graph_model, ["blocks"], SharedGraphPool())

    rng_state = torch.cuda.get_rng_state()
    for call, reference in zip(calls, expected):
        _assert_same(graph_model(**call), reference)
    assert torch.equal(torch.cuda.get_rng_state(), rng_state)  # capture leaves the default RNG
    _assert_same(graph_model.blocks[0](*first.values()), expected_block)
    assert all(len(runner.graphs) == 2 for runner in runners.values())
    # Broadcast rotary is stored unexpanded; a stepped slice keeps its strides.
    (broadcast, _), (strided, _) = runners["transformer.blocks.0"].static_inputs.values()
    assert broadcast[3].untyped_storage().nbytes() == cos.untyped_storage().nbytes()
    assert strided[3].stride() == stepped[0].stride()


@torch.inference_mode()
def test_graph_keys():
    """Model graph keys select graphs and reach the block during capture; so do block args.

    Once capture stops (after warmup), a new key runs eagerly and adds no graph.
    """
    torch.manual_seed(0)
    model = _TwoBlockModel(timestep_key=True, block_cls=_PhaseBlock).cuda().eval()
    call = _inputs(1)
    early = model(**call)
    with resolved_extra_keys_scope({"late_phase": True}):
        late = model(**call)
    block_calls = [dict(call, scale=0.5), dict(call, scale=0.25)]  # differ only in a value
    block_calls += [dict(call, temb=call["temb"].half()), call]  # differ only in a dtype
    expected_blocks = [model.blocks[0](**block_call) for block_call in block_calls]
    rope = tuple(torch.randn(1, _DIM, device="cuda").expand(32, -1) for _ in range(2))
    longer = dict(
        call, hidden_states=torch.randn(2, 32, _DIM, device="cuda"), image_rotary_emb=rope
    )
    expected_longer = model(**longer)
    runners = wrap_blocks_with_cuda_graphs(model, ["blocks"])

    for timestep, expected in ((900, early), (100, late), (800, early), (50, late)):
        _assert_same(model(**call, timestep=timestep), expected)
    assert all(len(runner.graphs) == 2 for runner in runners.values())
    for block_call, expected in zip(block_calls * 2, expected_blocks * 2):  # capture, replay
        _assert_same(model.blocks[0](**block_call), expected)
    assert len(runners["transformer.blocks.0"].graphs) == 6

    BasePipeline._stop_block_graph_capture(SimpleNamespace(_cuda_graph_runners=runners))
    _assert_same(model(**longer, timestep=900), expected_longer)
    assert [len(runner.graphs) for runner in runners.values()] == [6, 2]


@torch.inference_mode()
def test_chained_and_shared_inputs(model):
    """Inputs read in place still take new tensors; the others take new contents."""
    call = _inputs(1)
    other = _inputs(2)
    expected = model(**call)
    expected_doubled_temb = model(**dict(call, temb=call["temb"] * 2))
    second_block_call = dict(
        call,
        hidden_states=other["hidden_states"],
        encoder_hidden_states=other["encoder_hidden_states"],
    )
    expected_second_block = model.blocks[1](**second_block_call)
    runners = wrap_blocks_with_cuda_graphs(model, ["blocks"])

    _assert_same(model(**call), expected)  # capture
    # The blocks keep one temb, cos and sin buffer between them.
    (first_args, _), (second_args, _) = (
        next(iter(runner.static_inputs.values())) for runner in runners.values()
    )
    assert all(a is b for a, b in zip(first_args[2:], second_args[2:], strict=True))
    _assert_same(model(**call), expected)  # replay
    # An eager tensor at a chained input (Cache-DiT's add-back result) is copied in.
    _assert_same(model.blocks[1](**second_block_call), expected_second_block)
    # The same temb tensor with new contents is copied again in the next forward.
    call["temb"].mul_(2)
    _assert_same(model(**call), expected_doubled_temb)


@torch.inference_mode()
def test_input_changed_in_place_between_blocks():
    torch.manual_seed(0)
    model = _TembChangingModel().cuda().eval()
    call = _inputs(1)
    expected = model(**_clone_call(call))
    runners = wrap_blocks_with_cuda_graphs(model, ["blocks"])

    for _ in range(2):  # capture, then replay
        _assert_same(model(**_clone_call(call)), expected)
    # Both blocks read temb from one buffer; each replay copies temb's current contents in.
    first_temb, second_temb = (
        next(iter(runner.static_inputs.values()))[0][2] for runner in runners.values()
    )
    assert first_temb is second_temb


@torch.inference_mode()
def test_block_writing_its_input_keeps_a_private_buffer():
    """Block 1 halves its hidden_states input, so it does not read block 0's output in place."""
    torch.manual_seed(0)
    model = _TwoBlockModel().cuda().eval()
    model.blocks[1] = _InPlaceInputBlock().cuda().eval()
    call = _inputs(2)
    expected = model(**_clone_call(call))
    expected_first_block = model.blocks[0](**_clone_call(call))
    runners = wrap_blocks_with_cuda_graphs(model, ["blocks"])

    model(**_clone_call(_inputs(1)))  # capture
    (first_block_output,) = runners["transformer.blocks.0"].graph_outputs.values()
    _assert_same(model(**_clone_call(call)), expected)
    # The first block's output buffer was not halved by the second block.
    _assert_same(first_block_output, expected_first_block)


@torch.inference_mode()
def test_input_aliased_at_capture_gets_its_own_buffer():
    """Captured with one tensor at both inputs, the graph still takes two different tensors."""
    torch.manual_seed(0)
    model = _PairModel().cuda().eval()
    x, y, z = (torch.randn(4, _DIM, device="cuda") for _ in range(3))
    expected = model.blocks[1](model.blocks[0](x, y), z)
    expected_reversed = model.blocks[1](z, model.blocks[0](x, y))
    expected_broadcast = model.blocks[1](z[:1], model.blocks[0](x[:1], y[:1]).expand(4, -1))
    wrap_blocks_with_cuda_graphs(model, ["blocks"])

    model(x, y)  # capture: block 1 receives block 0's output at both inputs
    actual = model.blocks[1](model.blocks[0](x, y), z)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Reversed: z goes into block 0's output buffer, which is also the second input's source.
    actual = model.blocks[1](z, model.blocks[0](x, y))
    torch.testing.assert_close(actual, expected_reversed, rtol=0, atol=0)
    # Same with a broadcast view of block 0's output as the second input.
    out = model.blocks[0](x[:1], y[:1])
    model.blocks[1](out, out.expand(4, -1))  # capture
    actual = model.blocks[1](z[:1], model.blocks[0](x[:1], y[:1]).expand(4, -1))
    torch.testing.assert_close(actual, expected_broadcast, rtol=0, atol=0)


@torch.inference_mode()
def test_clear_releases_graph_buffers(model):
    """Clearing the runners frees every graph input and output buffer; the next call recaptures."""
    call = _inputs(1)
    expected = model(**call)
    runners = wrap_blocks_with_cuda_graphs(model, ["blocks"])
    _assert_same(model(**call), expected)
    buffers = [
        weakref.ref(tensor)
        for runner in runners.values()
        for tensor in tree_leaves(
            [list(runner.static_inputs.values()), list(runner.graph_outputs.values())]
        )
        if isinstance(tensor, torch.Tensor)
    ]
    assert buffers

    for runner in runners.values():
        runner.clear()
    gc.collect()
    assert all(ref() is None for ref in buffers)
    _assert_same(model(**call), expected)
    runners["transformer.blocks.1"].clear()  # recaptured after block 0 replays, keeping its output
    _assert_same(model(**call), expected)
    assert all(len(runner.graphs) == 1 for runner in runners.values())


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
    [("VANILLA", True), ("FA4", False)],
    ids=["vanilla-compiled", "fa4-eager"],
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
