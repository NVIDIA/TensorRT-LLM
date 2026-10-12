# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight table: gpt-oss-120b / sm_103 / tp1.

One `W` per weight role, carrying its shape, checkpoint source, dtype and
load-time transform. tp1: no sharding -- every checkpoint tensor reaches
exactly one parameter whole. Storage layout is HF [out, in] row-major, so the
attention, router and embed copies are layout-preserving.

The parity trap in the MXFP4 expert operands: this checkpoint's `2I` axis
runs (gate, up, gate, up, ...) -- HF reads `gate = gate_up[..., ::2]`, `up =
gate_up[..., 1::2]` -- while the kernel's interleave wants destination row
`2i` = **up** `i` and `2i+1` = gate `i`. Getting it backwards is finite,
plausibly scaled and invisible to a boot check.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.nn as nn

#: `layers=ALL_LAYERS` is the common case and the default. A sentinel rather
#: than a lambda so the table reads as data, and rather than `True` so the
#: field has one type of answer: which layers.
ALL_LAYERS = object()


@dataclass(frozen=True)
class W:
    """One weight role: what it is, where its bytes come from, what happens on the way.

    `shape` takes the dimension bundle and returns the declared shape; every
    shape in these models is derived from configuration rather than constant.

    `src` says where the bytes come from, in one of three forms:

    * a checkpoint key template, with `{p}` standing for the layer prefix;
    * a sequence of `(template, slice_fn)` pairs, when one parameter is
      assembled from several checkpoint tensors side by side -- `slice_fn`
      takes the dimension bundle and returns an index into the destination;
    * a callable `(d, i) -> [(key, index), ...]`, taking the dimension bundle
      and the layer index. The first two forms are special cases of it; this
      one exists for sources that are not a fixed template -- a per-rank
      expert window, or a prefix that differs between layer families. A `key`
      may itself be a tuple, meaning the transform is handed those tensors
      together.

    `layers` says which layers carry this weight: `ALL_LAYERS` for every one,
    `None` for a weight that is not per-layer at all, or a callable
    `d -> iterable[int]` when only some layers have it.

    `transform` runs on the source tensor before the copy, on the
    destination's device. `None` means the checkpoint tensor is already the
    declared shape.
    """

    name: str
    shape: Callable[[Any], tuple[int, ...]]
    src: str | tuple[tuple[str, Callable], ...] | Callable
    dtype: torch.dtype | None = None
    transform: Callable | None = None
    layers: object | None = ALL_LAYERS

    def layer_indices(self, d: SimpleNamespace) -> list[int] | None:
        """The layers this weight exists on, or None if it is not per-layer."""
        if self.layers is None:
            return None
        if self.layers is ALL_LAYERS:
            return list(range(d.num_layers))
        return list(self.layers(d))

    def sources(self, d: SimpleNamespace, i: int | None, prefix: str) -> list[tuple[Any, Any]]:
        """`[(checkpoint key or key tuple, index into the param | None), ...]`."""
        if callable(self.src):
            return list(self.src(d, i))
        if isinstance(self.src, str):
            return [(self.src.format(p=prefix), None)]
        return [(template.format(p=prefix), slice_fn(d)) for template, slice_fn in self.src]


SCALE_BLOCK = 32  # mxfp4: one E8M0 exponent per 32 elements along K

# FC1's K alignment for the trtllm-gen MXFP4 weight family. It sizes the
# declared expert operands here and is the alignment mxfp8_quantize pads the
# hidden states up to in modeling.py; the two must be the same number.
FC1_K_ALIGN = 512


def _pad_up(x: int, align: int) -> int:
    return (x + align - 1) // align * align


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


class GptOssWeights:
    """gpt-oss-120b / sm_103 / tp1.

    The table, and the passes over it: `declare` allocates every weight it names,
    `manifest` maps checkpoint keys onto them, `load` fills them. `declare` and
    `manifest` share one "which parameters exist" decision (`_targets`), so they
    cannot disagree about a weight.
    """

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
        W("final_norm", shape=lambda d: (d.hidden,), src="model.norm.weight", layers=None),
        W(
            "embed",
            shape=lambda d: (d.vocab, d.hidden),
            src="model.embed_tokens.weight",
            layers=None,
        ),
    )

    RELEASE_AFTER = "_fc2_b"

    def dims(self, core) -> SimpleNamespace:
        cfg = core.model_config.pretrained_config
        inter_pad = _pad_up(cfg.intermediate_size, 128)
        return SimpleNamespace(
            num_layers=cfg.num_hidden_layers,
            hidden=cfg.hidden_size,
            q_width=cfg.num_attention_heads * cfg.head_dim,
            kv_width=cfg.num_key_value_heads * cfg.head_dim,
            heads_q=cfg.num_attention_heads,
            num_experts=cfg.num_local_experts,
            vocab=cfg.vocab_size,
            fc1_rows=2 * inter_pad,
            fc1_k_pad=_pad_up(cfg.hidden_size, FC1_K_ALIGN),
            inter_pad=inter_pad,
            fc2_rows_pad=_pad_up(cfg.hidden_size, 128),
            dtype=core.model_config.torch_dtype,
        )

    def declare(self, d: SimpleNamespace) -> nn.ParameterDict:
        """Allocate every weight the table declares.

        Meta-init intercepts `torch.empty` here -- real CUDA storage arrives
        when the engine materializes the registry.
        """
        w = nn.ParameterDict()
        for entry in self.WEIGHTS:
            dtype = entry.dtype or d.dtype
            for key, _, _ in self._targets(entry, d):
                w[key] = nn.Parameter(
                    torch.empty(*entry.shape(d), dtype=dtype), requires_grad=False
                )
        return w

    def _targets(self, entry: W, d: SimpleNamespace) -> list[tuple[str, int | None, str]]:
        """`[(param key, layer index | None, checkpoint layer prefix), ...]`.

        One place decides which parameters an entry declares, so `declare` and
        `manifest` cannot disagree about whether a weight exists.
        """
        layers = entry.layer_indices(d)
        if layers is None:
            return [(entry.name, None, "")]
        return [(f"l{i}_{entry.name}", i, f"model.layers.{i}") for i in layers]

    def manifest(self, core) -> dict[str, list[tuple[Any, Any, Callable | None]]]:
        """param key -> [(checkpoint key, index into the param | None, transform | None)]."""
        d = self.dims(core)
        rows: dict[str, list[tuple[Any, Any, Callable | None]]] = {}
        for entry in self.WEIGHTS:
            for key, i, prefix in self._targets(entry, d):
                rows[key] = [
                    (ckpt, index, entry.transform) for ckpt, index in entry.sources(d, i, prefix)
                ]
        return rows

    def load(self, model, weights) -> None:
        core = model.model
        manifest = self.manifest(core)
        consumed: set[str] = set()

        def fill(param: nn.Parameter, ckpt_key: str, index, transform) -> None:
            assert ckpt_key in weights, f"checkpoint key missing: {ckpt_key}"
            src = weights[ckpt_key][:]  # materialize the lazy slice
            dst = param.data if index is None else param.data[index]
            if transform is not None:
                src = transform(src.to(dst.device, non_blocking=True), core)
            assert dst.shape == src.shape, (ckpt_key, tuple(dst.shape), tuple(src.shape))
            assert src.dtype == dst.dtype, (ckpt_key, src.dtype, dst.dtype)
            dst.copy_(src, non_blocking=True)
            consumed.add(ckpt_key)

        for param_key, sources in manifest.items():
            for ckpt_key, index, transform in sources:
                fill(core.w[param_key], ckpt_key, index, transform)
            if self.RELEASE_AFTER is not None and param_key.endswith(self.RELEASE_AFTER):
                torch.cuda.empty_cache()

        # Shell-registered exception: the base class owns lm_head (untied).
        fill(model.lm_head.weight, "lm_head.weight", None, None)

        torch.cuda.synchronize()
        leftover = set(weights.keys()) - consumed - self.expected_unconsumed(core, weights)
        assert not leftover, f"unconsumed checkpoint keys: {sorted(leftover)[:8]}"


MODEL_WEIGHTS = GptOssWeights()
