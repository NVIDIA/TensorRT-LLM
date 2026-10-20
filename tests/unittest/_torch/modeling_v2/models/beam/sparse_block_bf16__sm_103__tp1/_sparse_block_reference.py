# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plain-Torch reference for the sparse sliding-attention decoder block 3 of Beam.

An independent restatement of the source block for generated-input comparison: sandwich
RMSNorms (fp32 statistics and affine, one bf16 rounding); separate bias-free q/k/v/gate
projections in [out, in] layout; per-head q/k RMSNorm rounded to bf16 before an fp32 NeoX
RoPE rounded once; causal GQA attention, 6 q heads per kv head, scale 1/sqrt(128), window
512; the per-head sigmoid output gate in bf16 steps; the output projection; the MoE of
``../moe_bf16__sm_103__tp1/_moe_reference.py``; and block 3's depth-scaled residual adds,
(1/sqrt(8), sqrt(7)/sqrt(8)) after attention and (1/3, sqrt(8)/3) after the feed-forward,
each scaled term and each sum rounded to bf16.

Only torch and that MoE reference are imported. The attention core is torch's SDPA, which
the component never runs. State and input are generated on the GPU as the MoE reference
generates its own, one seeded generator per tensor; the block's 770 ``mlp.`` tensors are
exactly the MoE reference's state of the same seed.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn.functional as F

__extra_import_path__ = ["../moe_bf16__sm_103__tp1"]
import _moe_reference as moe_reference  # noqa: E402

HIDDEN = 5120
HEADS = 48
KV_HEADS = 8
HEAD_DIM = 128
EPS = 1e-6
ROPE_THETA = 10000.0
WINDOW = 512
LAYER_IDX = 3
MAX_TOKENS = 129
SEEDS = (17, 41)
LENGTHS = (1, 17, 129)
#: The MoE's tensors carry this prefix among the block's names.
MLP_PREFIX = "mlp."

STATE_RECIPE = {
    "seeding": "one torch.Generator on the device per call, re-seeded for every tensor with "
    "zlib.crc32(f'{seed}|{name}'); the mlp. tensors are the MoE reference's state of the "
    "seed, drawn under their names relative to mlp.",
    "*norm.weight": "1 + 0.25 * N(0, 1), drawn in fp32 and rounded once to bf16",
    "self_attn.gate_bias": "0.75 * N(0, 1), drawn in fp32 and rounded once to bf16",
    "projections": "N(0, 1) / sqrt(input width), drawn in fp32 and rounded once to bf16",
    "input": "N(0, 1) of shape [1, 129, 5120] under the name 'block_input', drawn in fp32 "
    "and rounded once to bf16; a T-token case uses the first T tokens, positions arange(T)",
}


def depth_coefs(layer_idx: int, is_attention: bool) -> tuple[float, float]:
    """(branch, stream) scales of the n-th residual add, 1 / sqrt(n) and sqrt(n - 1) /
    sqrt(n); block i's attention add is the 2(i + 1)-th, its feed-forward add the next."""
    num_adds = 2 * (layer_idx + 1) + (0 if is_attention else 1)
    denom = math.sqrt(num_adds)
    return 1.0 / denom, math.sqrt(num_adds - 1) / denom


ATTENTION_SCALES = depth_coefs(LAYER_IDX, True)
FEED_FORWARD_SCALES = depth_coefs(LAYER_IDX, False)


def _ignore(name: str, value: object) -> None:
    del name, value


# ----------------------------------------------------------------------------- state and inputs


def own_shapes() -> "OrderedDict[str, tuple[int, ...]]":
    """The block's twelve bf16 attention and norm tensors by checkpoint name, [out, in]."""
    q, kv = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
    return OrderedDict(
        [
            ("input_layernorm.weight", (HIDDEN,)),
            ("self_attn.gate_bias", (HEADS,)),
            ("self_attn.q_proj.weight", (q, HIDDEN)),
            ("self_attn.k_proj.weight", (kv, HIDDEN)),
            ("self_attn.v_proj.weight", (kv, HIDDEN)),
            ("self_attn.gate_proj.weight", (q, HIDDEN)),
            ("self_attn.o_proj.weight", (HIDDEN, q)),
            ("self_attn.q_norm.weight", (HEAD_DIM,)),
            ("self_attn.k_norm.weight", (HEAD_DIM,)),
            ("pre_feedforward_layernorm.weight", (HIDDEN,)),
            ("post_attention_layernorm.weight", (HIDDEN,)),
            ("post_feedforward_layernorm.weight", (HIDDEN,)),
        ]
    )


def param_shapes() -> "OrderedDict[str, tuple[tuple[int, ...], torch.dtype]]":
    """The block's 782 tensors by checkpoint name: its own twelve, then the MoE's under
    ``mlp.``, with their dtypes."""
    shapes = OrderedDict((name, (shape, torch.bfloat16)) for name, shape in own_shapes().items())
    for name, shape_dtype in moe_reference.param_shapes().items():
        shapes[MLP_PREFIX + name] = shape_dtype
    return shapes


def generate_state(
    seed: int, device: str | torch.device = "cuda"
) -> "OrderedDict[str, torch.Tensor]":
    """The seed's 782 tensors on ``device``, in checkpoint order (see `STATE_RECIPE`)."""
    gen = torch.Generator(device=device)
    state: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for name, shape in own_shapes().items():
        gen.manual_seed(moe_reference.tensor_seed(seed, name))
        draw = torch.randn(shape, generator=gen, dtype=torch.float32, device=device)
        if name.endswith("norm.weight"):
            value = 1.0 + 0.25 * draw
        elif name == "self_attn.gate_bias":
            value = 0.75 * draw
        else:
            value = draw / math.sqrt(shape[1])
        state[name] = value.to(torch.bfloat16).contiguous()
    for name, tensor in moe_reference.generate_state(seed, device).items():
        state[MLP_PREFIX + name] = tensor
    return state


def moe_state(state: "dict[str, torch.Tensor]") -> "OrderedDict[str, torch.Tensor]":
    """The MoE's tensors of a block state by name relative to ``mlp.``; nothing is copied."""
    return OrderedDict(
        (name[len(MLP_PREFIX) :], tensor)
        for name, tensor in state.items()
        if name.startswith(MLP_PREFIX)
    )


def generate_input(seed: int, device: str | torch.device = "cuda") -> torch.Tensor:
    """``[1, 129, 5120]`` bf16: the residual stream entering block 3 (see `STATE_RECIPE`)."""
    gen = torch.Generator(device=device)
    gen.manual_seed(moe_reference.tensor_seed(seed, "block_input"))
    x = torch.randn(1, MAX_TOKENS, HIDDEN, generator=gen, dtype=torch.float32, device=device)
    return x.to(torch.bfloat16)


def state_problems(state: "dict[str, torch.Tensor]", seed: int) -> list[str]:
    """Exact accounting plus the nontrivial-state properties the comparisons rely on."""
    shapes = param_shapes()
    if list(state) != list(shapes):
        return ["names or order differ from param_shapes()"]
    problems = []
    own = own_shapes()
    for name, shape in own.items():
        tensor = state[name]
        if tuple(tensor.shape) != shape or tensor.dtype != torch.bfloat16:
            problems.append(f"{name}: {tuple(tensor.shape)} {tensor.dtype}, expected {shape}")
        elif not bool(torch.isfinite(tensor).all()):
            problems.append(f"{name}: nonfinite")
    if problems:
        return problems
    names = list(own) + list(moe_reference.param_shapes())
    if len({moe_reference.tensor_seed(seed, name) for name in names}) != len(names):
        problems.append("two tensors share a generator seed")
    norms = [name for name in own if name.endswith("norm.weight")]
    for name in norms:
        if bool((state[name] == 1).all()):
            problems.append(f"{name}: unit affine")
    for i, a in enumerate(norms):
        for b in norms[i + 1 :]:
            if state[a].shape == state[b].shape and torch.equal(state[a], state[b]):
                problems.append(f"{a} and {b} are identical")
    bias = state["self_attn.gate_bias"]
    if bool((bias == bias[0]).all()):
        problems.append("self_attn.gate_bias does not vary across heads")
    problems.extend(f"MoE: {p}" for p in moe_reference.state_problems(moe_state(state), seed))
    return problems


# ----------------------------------------------------------------------------- the block


def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Over the last dim: fp32 statistics, normalization and affine, rounded once."""
    x32 = x.float()
    normed = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + EPS)
    return (normed * weight.float()).to(x.dtype)


def rope(
    q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """NeoX half rotation of ``q [T, 48, 128]`` and ``k [T, 8, 128]`` by their token's
    position: fp32 angles ``position / 10000^(2i / 128)``, ``x cos + rotate_half(x) sin``
    in fp32, rounded once."""
    exponent = torch.arange(0, HEAD_DIM, 2, dtype=torch.float32, device=positions.device)
    inv_freq = 1.0 / (ROPE_THETA ** (exponent / HEAD_DIM))
    angles = positions.float().unsqueeze(-1) * inv_freq
    emb = torch.cat((angles, angles), dim=-1)
    cos, sin = emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)
    half = HEAD_DIM // 2

    def rotate(x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        rotated = torch.cat((-x32[..., half:], x32[..., :half]), dim=-1)
        return (x32 * cos + rotated * sin).to(x.dtype)

    return rotate(q), rotate(k)


def attention_core(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Causal windowed GQA over one fresh context: ``q [T, 48, 128]``, ``k`` and ``v``
    ``[T, 8, 128]`` bf16, q head h reading kv head h // 6; returns ``[T, 48, 128]`` bf16."""
    tokens = q.shape[0]
    repeat = HEADS // KV_HEADS
    k = k.repeat_interleave(repeat, dim=1)
    v = v.repeat_interleave(repeat, dim=1)
    position = torch.arange(tokens, device=q.device)
    distance = position.unsqueeze(1) - position.unsqueeze(0)
    keep = (distance >= 0) & (distance < WINDOW)
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
        attn_mask=keep,
        dropout_p=0.0,
        scale=HEAD_DIM**-0.5,
    )
    return out.squeeze(0).transpose(0, 1)


def output_gate(core: torch.Tensor, gate: torch.Tensor, gate_bias: torch.Tensor) -> torch.Tensor:
    """``core * sigmoid(gate + gate_bias)`` per head, ``[T, 48, 128]``: the add, the
    sigmoid and the product each rounded to bf16."""
    biased = (gate.float() + gate_bias.float().view(1, HEADS, 1)).to(torch.bfloat16)
    weight = torch.sigmoid(biased.float()).to(torch.bfloat16)
    return (core.float() * weight.float()).to(torch.bfloat16)


def attention(
    h: torch.Tensor, positions: torch.Tensor, state: "dict[str, torch.Tensor]", record
) -> torch.Tensor:
    """Normalized input ``[T, 5120]`` -> attention output ``[T, 5120]``, before the
    post-attention norm."""
    tokens = h.shape[0]
    q = F.linear(h, state["self_attn.q_proj.weight"])
    k = F.linear(h, state["self_attn.k_proj.weight"])
    v = F.linear(h, state["self_attn.v_proj.weight"])
    gate = F.linear(h, state["self_attn.gate_proj.weight"])
    for name, value in (("q_proj", q), ("k_proj", k), ("v_proj", v), ("gate_proj", gate)):
        record(name, value)
    q = rms_norm(q.view(tokens, HEADS, HEAD_DIM), state["self_attn.q_norm.weight"])
    k = rms_norm(k.view(tokens, KV_HEADS, HEAD_DIM), state["self_attn.k_norm.weight"])
    record("q_norm", q)
    record("k_norm", k)
    q, k = rope(q, k, positions)
    record("q_rope", q)
    record("k_rope", k)
    core = attention_core(q, k, v.view(tokens, KV_HEADS, HEAD_DIM))
    record("attention_core", core)
    gated = output_gate(core, gate.view(tokens, HEADS, HEAD_DIM), state["self_attn.gate_bias"])
    record("gated", gated)
    out = F.linear(gated.reshape(tokens, HEADS * HEAD_DIM), state["self_attn.o_proj.weight"])
    record("attention_output", out)
    return out


def residual_add(
    branch: torch.Tensor, stream: torch.Tensor, scales: tuple[float, float]
) -> torch.Tensor:
    """``branch * scales[0] + stream * scales[1]``: each product and the sum computed in
    fp32 and rounded to bf16 in turn."""
    branch_scale, stream_scale = scales
    scaled_branch = (branch.float() * branch_scale).to(torch.bfloat16)
    scaled_stream = (stream.float() * stream_scale).to(torch.bfloat16)
    return (scaled_branch.float() + scaled_stream.float()).to(torch.bfloat16)


def block(
    x: torch.Tensor,
    positions: torch.Tensor,
    state: "dict[str, torch.Tensor]",
    record=None,
) -> torch.Tensor:
    """``x [1, T, 5120]`` bf16 and integer ``positions [1, T]`` -> ``[1, T, 5120]`` bf16.

    ``record(name, value)``, when given, receives every intermediate ``[T, ...]``, the
    MoE's under ``moe.`` (``moe.routing`` is its route), and ``block_output`` last.
    """
    keep = record if record is not None else _ignore
    tokens = x.shape[1]
    stream = x.reshape(tokens, HIDDEN)
    with moe_reference.reference_math():
        h = rms_norm(stream, state["input_layernorm.weight"])
        keep("input_layernorm", h)
        a = attention(h, positions.reshape(tokens), state, keep)
        a = rms_norm(a, state["post_attention_layernorm.weight"])
        keep("post_attention_layernorm", a)
        stream = residual_add(a, stream, ATTENTION_SCALES)
        keep("attention_residual", stream)
        h = rms_norm(stream, state["pre_feedforward_layernorm.weight"])
        keep("pre_feedforward_layernorm", h)
        m = moe_reference.moe(h, moe_state(state), lambda name, value: keep(f"moe.{name}", value))
        keep("moe_output", m)
        m = rms_norm(m, state["post_feedforward_layernorm.weight"])
        keep("post_feedforward_layernorm", m)
        out = residual_add(m, stream, FEED_FORWARD_SCALES)
        keep("block_output", out)
    return out.view(1, tokens, HIDDEN)
