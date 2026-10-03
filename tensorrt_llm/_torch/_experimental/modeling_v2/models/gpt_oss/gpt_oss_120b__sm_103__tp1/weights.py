# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight manifest and loader: gpt-oss-120b / sm_103 / tp1.

MANIFEST is a data table: target parameter -> the checkpoint keys that fill
it, each with an optional destination index into the parameter and an
optional source transform. tp1: no sharding — every checkpoint tensor
reaches exactly one parameter whole.

Storage layout is HF [out, in] row-major, so every attention/router/embed
copy is layout-preserving; the GEMM-side column-major views are derived
after load in modeling.build_layer_views(). Two families need a transform:

* **fp32 promotions.** The expert biases and the attention sink logits are
  bf16 on disk, but the MoE op rejects a bf16 bias and thop_attention reads
  the sink buffer as raw fp32.

* **The MXFP4 expert operands.** The checkpoint stores each layer's experts
  as `gate_up_proj_blocks` [E, 2I, H/32, 16] / `_scales` [E, 2I, H/32] and
  `down_proj_blocks` [E, H, I/32, 16] / `_scales` [E, H, I/32] — E2M1 codes
  two per byte (low nibble = even K index) with one E8M0 exponent per 32 K
  elements, already in [out, in] orientation. The MoE op wants them padded,
  row-permuted and (scales only) swizzled; `_prep_fc1` / `_prep_fc2` below
  are that recipe, applied on device, one layer at a time.

  The parity trap: this checkpoint's `2I` axis runs (gate, up, gate, up,
  ...) — HF reads `gate = gate_up[..., ::2]`, `up = gate_up[..., 1::2]` —
  while the kernel's interleave wants destination row `2i` = **up** `i` and
  `2i+1` = gate `i`. The halves are therefore split by parity and
  re-concatenated as [up ; gate] before the permutation. Getting it
  backwards is finite, plausibly scaled and invisible to a boot check.

Loading contract: `load(model, weights)` consumes the engine-provided
per-rank checkpoint dict (safetensors lazy slices), fills every declared
parameter exactly once, and asserts full bidirectional coverage — every
target parameter written, every checkpoint key consumed.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.nn as nn

SCALE_BLOCK = 32  # mxfp4: one E8M0 exponent per 32 elements along K


def _pad_rows_cols(t: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Zero-pad a `[E, R, C]` stack to `[E, rows, cols]`."""
    out = torch.zeros(t.shape[0], rows, cols, dtype=t.dtype, device=t.device)
    out[:, : t.shape[1], : t.shape[2]] = t
    return out


def _pad_cols(t: torch.Tensor, cols: int) -> torch.Tensor:
    """Zero-pad a `[E, C]` stack to `[E, cols]`."""
    out = torch.zeros(t.shape[0], cols, dtype=t.dtype, device=t.device)
    out[:, : t.shape[1]] = t
    return out


def _block32_perm(rows: int, device) -> torch.Tensor:
    """Gather index of the 32-row block shuffle: inside each aligned block of
    32 rows, source row `4u + v` (0<=u<8, 0<=v<4) lands at destination row
    `8v + u`."""
    assert rows % 32 == 0, rows
    src = torch.arange(32)
    dst = (src % 4) * 8 + src // 4
    idx = torch.empty(32, dtype=torch.long)
    idx[dst] = src
    blocks = rows // 32
    return (idx.repeat(blocks) + torch.arange(blocks).repeat_interleave(32) * 32).to(device)


def _interleave_perm(rows: int, device) -> torch.Tensor:
    """Gather index that interleaves the `[up | gate]` halves of a `2*I_pad`
    row stack into up0, gate0, up1, gate1, ..."""
    p = torch.empty(rows, dtype=torch.long)
    p[0::2] = torch.arange(0, rows // 2)
    p[1::2] = torch.arange(rows // 2, rows)
    return p.to(device)


def _swizzle_scales(s: torch.Tensor) -> torch.Tensor:
    """Per-expert 128x4 block-scale swizzle, viewed back as `[E, M, C]`: the
    byte at `(m, c)` moves to flat offset `(m//128)*512*(C//4) + (c//4)*512
    + (m%32)*16 + ((m%128)//32)*4 + (c%4)`."""
    e, m, c = s.shape
    assert m % 128 == 0 and c % 4 == 0, (m, c)
    v = s.reshape(e, m // 128, 4, 32, c // 4, 4)  # (e, m/128, (m%128)/32, m%32, c/4, c%4)
    return v.permute(0, 1, 4, 3, 2, 5).reshape(e, m, c).contiguous()


def _split_gate_up(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the checkpoint's interleaved `2I` output axis into (up, gate)."""
    return t[:, 1::2], t[:, 0::2]


def _fc1_perm(rows: int, device) -> torch.Tensor:
    """Interleave then 32-row block shuffle, composed into one gather."""
    return _interleave_perm(rows, device)[_block32_perm(rows, device)]


def _prep_fc1_weight(blocks: torch.Tensor, core) -> torch.Tensor:
    """[E, 2I, H/32, 16] uint8 blocks -> kernel FC1 operand."""
    cfg = core.model_config.pretrained_config
    b = blocks.reshape(cfg.num_local_experts, 2 * cfg.intermediate_size, cfg.hidden_size // 2)
    up, gate = _split_gate_up(b)
    cols = core.fc1_k_pad // 2
    w = torch.cat(
        [
            _pad_rows_cols(up, core.inter_pad, cols),
            _pad_rows_cols(gate, core.inter_pad, cols),
        ],
        dim=1,
    )
    return torch.index_select(w, 1, _fc1_perm(w.shape[1], w.device)).contiguous()


def _prep_fc1_scale(scales: torch.Tensor, core) -> torch.Tensor:
    """[E, 2I, H/32] uint8 E8M0 scales -> kernel FC1 scale operand."""
    up, gate = _split_gate_up(scales)
    cols = core.fc1_k_pad // SCALE_BLOCK
    s = torch.cat(
        [
            _pad_rows_cols(up, core.inter_pad, cols),
            _pad_rows_cols(gate, core.inter_pad, cols),
        ],
        dim=1,
    )
    return _swizzle_scales(torch.index_select(s, 1, _fc1_perm(s.shape[1], s.device)))


def _prep_fc1_bias(bias: torch.Tensor, core) -> torch.Tensor:
    """[E, 2I] bf16 biases -> fp32 kernel FC1 bias (same row permutation)."""
    up, gate = _split_gate_up(bias.float())
    b = torch.cat([_pad_cols(up, core.inter_pad), _pad_cols(gate, core.inter_pad)], dim=1)
    return torch.index_select(b, 1, _fc1_perm(b.shape[1], b.device)).contiguous()


def _prep_fc2_weight(blocks: torch.Tensor, core) -> torch.Tensor:
    """[E, H, I/32, 16] uint8 blocks -> kernel FC2 operand (no interleave)."""
    cfg = core.model_config.pretrained_config
    w = _pad_rows_cols(
        blocks.reshape(cfg.num_local_experts, cfg.hidden_size, cfg.intermediate_size // 2),
        core.fc2_rows_pad,
        core.inter_pad // 2,
    )
    return torch.index_select(w, 1, _block32_perm(core.fc2_rows_pad, w.device)).contiguous()


def _prep_fc2_scale(scales: torch.Tensor, core) -> torch.Tensor:
    """[E, H, I/32] uint8 E8M0 scales -> kernel FC2 scale operand."""
    s = _pad_rows_cols(scales, core.fc2_rows_pad, core.inter_pad // SCALE_BLOCK)
    return _swizzle_scales(torch.index_select(s, 1, _block32_perm(core.fc2_rows_pad, s.device)))


def _prep_fc2_bias(bias: torch.Tensor, core) -> torch.Tensor:
    """[E, H] bf16 biases -> fp32 kernel FC2 bias (same row permutation)."""
    b = _pad_cols(bias.float(), core.fc2_rows_pad)
    return torch.index_select(b, 1, _block32_perm(core.fc2_rows_pad, b.device)).contiguous()


def _to_fp32(t: torch.Tensor, core) -> torch.Tensor:
    """bf16 -> fp32 promotion for tensors the kernels require in fp32."""
    return t.float()


@dataclass(frozen=True)
class W:
    """One weight role: what it is, where its bytes come from, what happens on the way.

    Three facts that used to live in three places -- the shape in the model's
    declaration block, the checkpoint key in the manifest, the transform in a
    function far from either. A reader who wants to know what `fc1_w` is now
    reads one entry.

    `shape` is a callable rather than a constant because every shape here is
    derived from configuration: `fc1_k_pad` is a padded width, `q_width` is a
    product. It takes the dimension bundle `declare` is given.

    `src` is a checkpoint key template with `{p}` standing for the layer
    prefix, or a sequence of `(template, slice_fn)` pairs when one parameter is
    assembled from several checkpoint tensors -- `qkv` is three.

    `transform` runs on the source tensor before the copy, on the destination's
    device. `None` means the checkpoint tensor is already the declared shape.
    """

    name: str
    shape: Callable[[Any], tuple[int, ...]]
    src: str | tuple[tuple[str, Callable], ...]
    dtype: torch.dtype | None = None
    transform: Callable | None = None
    per_layer: bool = True


WEIGHTS: tuple[W, ...] = (
    W("norm1", shape=lambda d: (d.hidden,), src="{p}.input_layernorm.weight"),
    W(
        "qkv",
        shape=lambda d: (d.q_width + 2 * d.kv_width, d.hidden),
        src=(
            ("{p}.self_attn.q_proj.weight", lambda d: (slice(0, d.q_width),)),
            (
                "{p}.self_attn.k_proj.weight",
                lambda d: (slice(d.q_width, d.q_width + d.kv_width),),
            ),
            (
                "{p}.self_attn.v_proj.weight",
                lambda d: (slice(d.q_width + d.kv_width, d.q_width + 2 * d.kv_width),),
            ),
        ),
    ),
    W(
        "qkv_bias",
        shape=lambda d: (d.q_width + 2 * d.kv_width,),
        src=(
            ("{p}.self_attn.q_proj.bias", lambda d: (slice(0, d.q_width),)),
            (
                "{p}.self_attn.k_proj.bias",
                lambda d: (slice(d.q_width, d.q_width + d.kv_width),),
            ),
            (
                "{p}.self_attn.v_proj.bias",
                lambda d: (slice(d.q_width + d.kv_width, d.q_width + 2 * d.kv_width),),
            ),
        ),
    ),
    W(
        "sinks",
        shape=lambda d: (d.heads_q,),
        dtype=torch.float32,
        src="{p}.self_attn.sinks",
        transform=_to_fp32,
    ),
    W("o", shape=lambda d: (d.hidden, d.q_width), src="{p}.self_attn.o_proj.weight"),
    W("o_bias", shape=lambda d: (d.hidden,), src="{p}.self_attn.o_proj.bias"),
    W("norm2", shape=lambda d: (d.hidden,), src="{p}.post_attention_layernorm.weight"),
    W("router", shape=lambda d: (d.num_experts, d.hidden), src="{p}.mlp.router.weight"),
    W("router_bias", shape=lambda d: (d.num_experts,), src="{p}.mlp.router.bias"),
    W(
        "fc1_w",
        shape=lambda d: (d.num_experts, d.fc1_rows, d.fc1_k_pad // 2),
        dtype=torch.uint8,
        src="{p}.mlp.experts.gate_up_proj_blocks",
        transform=_prep_fc1_weight,
    ),
    W(
        "fc1_s",
        shape=lambda d: (d.num_experts, d.fc1_rows, d.fc1_k_pad // 32),
        dtype=torch.uint8,
        src="{p}.mlp.experts.gate_up_proj_scales",
        transform=_prep_fc1_scale,
    ),
    W(
        "fc1_b",
        shape=lambda d: (d.num_experts, d.fc1_rows),
        dtype=torch.float32,
        src="{p}.mlp.experts.gate_up_proj_bias",
        transform=_prep_fc1_bias,
    ),
    W(
        "fc2_w",
        shape=lambda d: (d.num_experts, d.fc2_rows_pad, d.inter_pad // 2),
        dtype=torch.uint8,
        src="{p}.mlp.experts.down_proj_blocks",
        transform=_prep_fc2_weight,
    ),
    W(
        "fc2_s",
        shape=lambda d: (d.num_experts, d.fc2_rows_pad, d.inter_pad // 32),
        dtype=torch.uint8,
        src="{p}.mlp.experts.down_proj_scales",
        transform=_prep_fc2_scale,
    ),
    W(
        "fc2_b",
        shape=lambda d: (d.num_experts, d.fc2_rows_pad),
        dtype=torch.float32,
        src="{p}.mlp.experts.down_proj_bias",
        transform=_prep_fc2_bias,
    ),
    W("final_norm", shape=lambda d: (d.hidden,), src="model.norm.weight", per_layer=False),
    W(
        "embed",
        shape=lambda d: (d.vocab, d.hidden),
        src="model.embed_tokens.weight",
        per_layer=False,
    ),
)


def declare(**dims: Any) -> nn.ParameterDict:
    """Allocate every weight the table declares.

    Takes the dimensions explicitly rather than reading them off the core: the
    core no longer carries them, having stopped forwarding configuration, and
    passing them is what keeps it that way.

    Meta-init intercepts `torch.empty` here -- real CUDA storage arrives when
    the engine materializes the registry.
    """
    d = SimpleNamespace(**dims)
    w = nn.ParameterDict()
    for entry in WEIGHTS:
        dtype = entry.dtype or d.dtype
        keys = (
            [f"l{i}_{entry.name}" for i in range(d.num_layers)] if entry.per_layer else [entry.name]
        )
        for key in keys:
            w[key] = nn.Parameter(torch.empty(*entry.shape(d), dtype=dtype), requires_grad=False)
    return w


def _manifest(core) -> dict:
    """target param key -> list of (ckpt key, index into the param | None,
    source transform | None)."""
    cfg = core.model_config.pretrained_config
    q_width = cfg.num_attention_heads * cfg.head_dim
    kv_width = cfg.num_key_value_heads * cfg.head_dim
    rows: dict = {}
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        rows[f"l{i}_norm1"] = [(f"{p}.input_layernorm.weight", None, None)]
        rows[f"l{i}_qkv"] = [
            (f"{p}.self_attn.q_proj.weight", (slice(0, q_width),), None),
            (
                f"{p}.self_attn.k_proj.weight",
                (slice(q_width, q_width + kv_width),),
                None,
            ),
            (
                f"{p}.self_attn.v_proj.weight",
                (slice(q_width + kv_width, q_width + 2 * kv_width),),
                None,
            ),
        ]
        rows[f"l{i}_qkv_bias"] = [
            (f"{p}.self_attn.q_proj.bias", (slice(0, q_width),), None),
            (f"{p}.self_attn.k_proj.bias", (slice(q_width, q_width + kv_width),), None),
            (
                f"{p}.self_attn.v_proj.bias",
                (slice(q_width + kv_width, q_width + 2 * kv_width),),
                None,
            ),
        ]
        rows[f"l{i}_sinks"] = [(f"{p}.self_attn.sinks", None, _to_fp32)]
        rows[f"l{i}_o"] = [(f"{p}.self_attn.o_proj.weight", None, None)]
        rows[f"l{i}_o_bias"] = [(f"{p}.self_attn.o_proj.bias", None, None)]
        rows[f"l{i}_norm2"] = [(f"{p}.post_attention_layernorm.weight", None, None)]
        rows[f"l{i}_router"] = [(f"{p}.mlp.router.weight", None, None)]
        rows[f"l{i}_router_bias"] = [(f"{p}.mlp.router.bias", None, None)]
        rows[f"l{i}_fc1_w"] = [(f"{p}.mlp.experts.gate_up_proj_blocks", None, _prep_fc1_weight)]
        rows[f"l{i}_fc1_s"] = [(f"{p}.mlp.experts.gate_up_proj_scales", None, _prep_fc1_scale)]
        rows[f"l{i}_fc1_b"] = [(f"{p}.mlp.experts.gate_up_proj_bias", None, _prep_fc1_bias)]
        rows[f"l{i}_fc2_w"] = [(f"{p}.mlp.experts.down_proj_blocks", None, _prep_fc2_weight)]
        rows[f"l{i}_fc2_s"] = [(f"{p}.mlp.experts.down_proj_scales", None, _prep_fc2_scale)]
        rows[f"l{i}_fc2_b"] = [(f"{p}.mlp.experts.down_proj_bias", None, _prep_fc2_bias)]
    rows["final_norm"] = [("model.norm.weight", None, None)]
    rows["embed"] = [("model.embed_tokens.weight", None, None)]
    return rows


def load(model, weights) -> None:
    core = model.model
    manifest = _manifest(core)
    consumed: set = set()

    def fill(param: torch.nn.Parameter, ckpt_key: str, index, transform) -> None:
        assert ckpt_key in weights, f"checkpoint key missing: {ckpt_key}"
        src = weights[ckpt_key][:]  # materialize the lazy slice
        dst = param.data if index is None else param.data[index]
        if transform is not None:
            # The expert transforms are heavy row gathers over ~1 GB of
            # blocks; run them where the destination lives.
            src = transform(src.to(dst.device, non_blocking=True), core)
        assert dst.shape == src.shape, (ckpt_key, tuple(dst.shape), tuple(src.shape))
        assert src.dtype == dst.dtype, (ckpt_key, src.dtype, dst.dtype)
        dst.copy_(src, non_blocking=True)
        consumed.add(ckpt_key)

    assert set(manifest.keys()) == set(core.w.keys()), (
        "manifest/parameter drift",
        set(manifest.keys()) ^ set(core.w.keys()),
    )
    for param_key, sources in manifest.items():
        for ckpt_key, index, transform in sources:
            fill(core.w[param_key], ckpt_key, index, transform)
        # The transforms allocate several GB of scratch per layer; release it
        # before the next one so peak load memory stays one layer deep.
        if param_key.endswith("_fc2_b"):
            torch.cuda.empty_cache()

    # Shell-registered exception: the base class owns lm_head (untied).
    fill(model.lm_head.weight, "lm_head.weight", None, None)

    torch.cuda.synchronize()
    leftover = set(weights.keys()) - consumed
    assert not leftover, f"unconsumed checkpoint keys: {sorted(leftover)[:8]}"
