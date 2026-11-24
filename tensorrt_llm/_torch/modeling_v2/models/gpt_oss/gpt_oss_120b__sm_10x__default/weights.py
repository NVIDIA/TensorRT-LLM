# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight table: gpt-oss-120b / sm_10x / default.

One `W` per weight role, carrying its shape, checkpoint source, dtype and
load-time transform, declared at this rank's share. Every share is computed
once, in `dims`, from the engine's `Mapping`; on one rank every share is the
whole tensor. Storage layout is HF [out, in] row-major, so the attention,
router and embed copies are layout-preserving.

* **Attention heads** are split over the attention TP group, which is the
  whole TP group without attention DP and a group of one with it. A rank holds
  its q heads and its kv heads, the matching sink logits, and the matching
  o_proj input columns. o_proj's bias is loaded on every rank and added on
  attention TP rank 0 only (see modeling.py), as the built-in row-parallel
  Linear does.

* **Experts** are split over the MoE EP group: a rank holds the contiguous id
  window `[expert_offset, expert_offset + local_experts)`.

* **The expert intermediate** is split over the MoE TP group, along the FC1
  output rows and the FC2 input columns. The MXFP4 layout fixes how: one e8m0
  scale covers 32 intermediate values, and the kernel wants the padded width
  128-aligned. When `I / moe_tp` is a whole number of scale groups, each rank
  takes its logical slice and pads it locally (moe_tp=2: 1440 -> 1536). When it
  is not (moe_tp=4: 720 is 22.5 groups), the intermediate is first padded to a
  width every rank can split on group boundaries, then split (3072 / 4 = 768;
  the last rank holds 576 real columns). This is the same rule as the built-in
  `MXFP4WeightTRTLLMGenFusedMoEMethod`, so the shard boundaries match it.
  FC2's bias is divided by moe_tp: every MoE TP rank adds it, and the
  partial sums meet in the allreduce or reduce-scatter that follows.

Everything else -- embedding, norms, router -- is replicated. lm_head is the
shell's: vocab-parallel under TP, whole under attention DP; `load` below
slices it to whatever the shell built.

The parity trap in the MXFP4 expert operands: this checkpoint's `2I` axis
runs (gate, up, gate, up, ...) -- HF reads `gate = gate_up[..., ::2]`, `up =
gate_up[..., 1::2]` -- while the kernel's interleave wants destination row
`2i` = **up** `i` and `2i+1` = gate `i`. Getting it backwards is finite,
plausibly scaled and invisible to a boot check.
"""

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.nn as nn

from . import native_ops

#: `layers=ALL_LAYERS` is the common case and the default. A sentinel rather
#: than a lambda so the table reads as data, and rather than `True` so the
#: field has one type of answer: which layers.
ALL_LAYERS = object()


@dataclass(frozen=True)
class W:
    """One weight role: what it is, where its bytes come from, what happens on the way.

    `shape` takes the dimension bundle and returns the declared shape at this
    rank's share.

    `src` says where the bytes come from, in one of two forms:

    * a checkpoint key template, with `{p}` standing for the layer prefix;
    * a callable `(d, i) -> [(key, index), ...]`, taking the dimension bundle
      and the layer index. A `key` may itself be a tuple, meaning the
      transform is handed those tensors together (q, k and v feed one packed
      parameter through one transform).

    `layers` says which layers carry this weight: `ALL_LAYERS` for every one,
    or `None` for a weight that is not per-layer at all.

    `transform` runs on the source tensor before the copy, on the
    destination's device, and is where a rank cuts its share. `None` means
    the checkpoint tensor is already the declared shape (a replicated weight).
    """

    name: str
    shape: Callable[[Any], tuple[int, ...]]
    src: str | Callable
    dtype: torch.dtype | None = None
    transform: Callable | None = None
    layers: object | None = ALL_LAYERS

    def layer_indices(self, d: SimpleNamespace) -> list[int] | None:
        """The layers this weight exists on, or None if it is not per-layer."""
        if self.layers is None:
            return None
        return list(range(d.num_layers))

    def sources(self, d: SimpleNamespace, i: int | None, prefix: str) -> list[tuple[Any, Any]]:
        """`[(checkpoint key or key tuple, index into the param | None), ...]`."""
        if callable(self.src):
            return list(self.src(d, i))
        return [(self.src.format(p=prefix), None)]


SCALE_BLOCK = 32  # mxfp4: one E8M0 exponent per 32 elements along K

# FC1's K alignment for the trtllm-gen MXFP4 weight family. It sizes the
# declared expert operands here and is the alignment mxfp8_quantize pads the
# hidden states up to in modeling.py; the two must be the same number.
FC1_K_ALIGN = 512

# The kernel's row alignment for both GEMMs' intermediate axis.
WEIGHT_ALIGN = 128


def _pad_up(x: int, align: int) -> int:
    return (x + align - 1) // align * align


def intermediate_shard(inter: int, moe_tp: int, moe_tp_rank: int) -> tuple[int, int, int]:
    """`(start, valid, padded)` of this rank's slice of the expert intermediate.

    `start` and `valid` are in logical columns of the checkpoint's intermediate;
    `padded` is the width the kernel operands are declared at, the same on
    every rank. Mirrors `_requires_rank_local_tp_padding` and
    `_get_weight_alignment` in `_torch/moe/fused_moe/quantization.py`.
    """
    if moe_tp == 1:
        return 0, inter, _pad_up(inter, WEIGHT_ALIGN)
    if inter % moe_tp == 0 and (inter // moe_tp) % SCALE_BLOCK == 0:
        per = inter // moe_tp
        return moe_tp_rank * per, per, _pad_up(per, WEIGHT_ALIGN)
    align = math.lcm(WEIGHT_ALIGN, SCALE_BLOCK, moe_tp)
    per = _pad_up(inter, align) // moe_tp
    if per % WEIGHT_ALIGN != 0:
        per = _pad_up(per, WEIGHT_ALIGN)
    start = moe_tp_rank * per
    valid = max(0, min(inter, start + per) - start)
    return start, valid, per


def _split_gate_up(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the checkpoint's interleaved `2I` axis (dim 1) into (up, gate)."""
    return t[:, 1::2], t[:, 0::2]


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


def _fc1_perm(rows: int, device) -> torch.Tensor:
    """Interleave then 32-row block shuffle, composed into one gather."""
    return _interleave_perm(rows, device)[_block32_perm(rows, device)]


def _swizzle_scales(s: torch.Tensor) -> torch.Tensor:
    """Per-expert 128x4 block-scale swizzle, viewed back as `[E, M, C]`: the
    byte at `(m, c)` moves to flat offset `(m//128)*512*(C//4) + (c//4)*512
    + (m%32)*16 + ((m%128)//32)*4 + (c%4)`. On the device this target's own
    kernel (csrc/swizzleBlockScales.cu) moves it in one pass; on the host,
    a torch permutation does."""
    if s.is_cuda:
        return native_ops.swizzle_block_scales(s)
    e, m, c = s.shape
    assert m % 128 == 0 and c % 4 == 0, (m, c)
    v = s.reshape(e, m // 128, 4, 32, c // 4, 4)  # (e, m/128, (m%128)/32, m%32, c/4, c%4)
    return v.permute(0, 1, 4, 3, 2, 5).reshape(e, m, c).contiguous()


def _experts(t: torch.Tensor, d) -> torch.Tensor:
    """This rank's expert window, dim 0."""
    return t[d.expert_offset : d.expert_offset + d.local_experts]


def _inter_rows(t: torch.Tensor, d) -> torch.Tensor:
    """This rank's intermediate slice along dim 1, in logical rows."""
    return t[:, d.inter_start : d.inter_start + d.inter_valid]


def _prep_fc1_weight(blocks: torch.Tensor, core) -> torch.Tensor:
    """[E, 2I, H/32, 16] uint8 blocks -> this rank's kernel FC1 operand."""
    d = core.dims
    b = _experts(blocks, d).reshape(d.local_experts, 2 * d.inter, d.hidden // 2)
    up, gate = _split_gate_up(b)
    cols = d.fc1_k_pad // 2
    w = torch.cat(
        [
            _pad_rows_cols(_inter_rows(up, d), d.inter_pad, cols),
            _pad_rows_cols(_inter_rows(gate, d), d.inter_pad, cols),
        ],
        dim=1,
    )
    return torch.index_select(w, 1, _fc1_perm(w.shape[1], w.device)).contiguous()


def _prep_fc1_scale(scales: torch.Tensor, core) -> torch.Tensor:
    """[E, 2I, H/32] uint8 E8M0 scales -> this rank's kernel FC1 scale operand."""
    d = core.dims
    up, gate = _split_gate_up(_experts(scales, d))
    cols = d.fc1_k_pad // SCALE_BLOCK
    s = torch.cat(
        [
            _pad_rows_cols(_inter_rows(up, d), d.inter_pad, cols),
            _pad_rows_cols(_inter_rows(gate, d), d.inter_pad, cols),
        ],
        dim=1,
    )
    return _swizzle_scales(torch.index_select(s, 1, _fc1_perm(s.shape[1], s.device)))


def _prep_fc1_bias(bias: torch.Tensor, core) -> torch.Tensor:
    """[E, 2I] bf16 biases -> this rank's fp32 kernel FC1 bias."""
    d = core.dims
    up, gate = _split_gate_up(_experts(bias, d).float())
    b = torch.cat(
        [
            _pad_cols(_inter_rows(up, d), d.inter_pad),
            _pad_cols(_inter_rows(gate, d), d.inter_pad),
        ],
        dim=1,
    )
    return torch.index_select(b, 1, _fc1_perm(b.shape[1], b.device)).contiguous()


def _prep_fc2_weight(blocks: torch.Tensor, core) -> torch.Tensor:
    """[E, H, I/32, 16] uint8 blocks -> this rank's kernel FC2 operand.

    The intermediate is FC2's K axis, packed two values per byte, so the slice
    is taken in bytes."""
    d = core.dims
    w = _experts(blocks, d).reshape(d.local_experts, d.hidden, d.inter // 2)
    w = w[:, :, d.inter_start // 2 : (d.inter_start + d.inter_valid) // 2]
    w = _pad_rows_cols(w, d.fc2_rows_pad, d.inter_pad // 2)
    return torch.index_select(w, 1, _block32_perm(d.fc2_rows_pad, w.device)).contiguous()


def _prep_fc2_scale(scales: torch.Tensor, core) -> torch.Tensor:
    """[E, H, I/32] uint8 E8M0 scales -> this rank's kernel FC2 scale operand."""
    d = core.dims
    s = _experts(scales, d)
    s = s[:, :, d.inter_start // SCALE_BLOCK : (d.inter_start + d.inter_valid) // SCALE_BLOCK]
    s = _pad_rows_cols(s, d.fc2_rows_pad, d.inter_pad // SCALE_BLOCK)
    return _swizzle_scales(torch.index_select(s, 1, _block32_perm(d.fc2_rows_pad, s.device)))


def _prep_fc2_bias(bias: torch.Tensor, core) -> torch.Tensor:
    """[E, H] bf16 biases -> this rank's fp32 kernel FC2 bias, divided by
    moe_tp so the MoE TP ranks' partial sums add it exactly once."""
    d = core.dims
    b = _pad_cols(_experts(bias, d).float() / d.moe_tp, d.fc2_rows_pad)
    return torch.index_select(b, 1, _block32_perm(d.fc2_rows_pad, b.device)).contiguous()


def _qkv_rows(t: tuple[torch.Tensor, torch.Tensor, torch.Tensor], core) -> torch.Tensor:
    """(q, k, v) projections, weights or biases -> this rank's packed
    `[q_local; k_local; v_local]` along dim 0."""
    d = core.dims
    q, k, v = t
    hd = d.head_dim
    q_rows = slice(d.attn_tp_rank * d.q_width, (d.attn_tp_rank + 1) * d.q_width)
    kv_rows = slice(d.attn_tp_rank * d.kv_width, (d.attn_tp_rank + 1) * d.kv_width)
    assert q.shape[0] == d.heads_q_total * hd and k.shape[0] == d.heads_kv_total * hd
    return torch.cat([q[q_rows], k[kv_rows], v[kv_rows]], dim=0).contiguous()


def _sinks(t: torch.Tensor, core) -> torch.Tensor:
    """bf16 per-head sink logits -> this rank's heads, fp32 (the attention op
    reads the buffer as raw fp32)."""
    d = core.dims
    return t[d.attn_tp_rank * d.heads_q : (d.attn_tp_rank + 1) * d.heads_q].float().contiguous()


def _o_cols(t: torch.Tensor, core) -> torch.Tensor:
    """o_proj `[H, Hq * hd]` -> this rank's input columns."""
    d = core.dims
    return t[:, d.attn_tp_rank * d.q_width : (d.attn_tp_rank + 1) * d.q_width].contiguous()


def _qkv_src(kind: str):
    """Source for the packed qkv row: the three projections, handed to `_qkv_rows` together."""

    def src(d, i):
        p = f"model.layers.{i}.self_attn"
        return [((f"{p}.q_proj.{kind}", f"{p}.k_proj.{kind}", f"{p}.v_proj.{kind}"), None)]

    return src


class GptOssDefaultWeights:
    """gpt-oss-120b / sm_10x / default.

    The table, and the passes over it: `dims` computes this rank's share,
    `declare` allocates every weight the table names at that share,
    `manifest` maps checkpoint keys onto them, `load` fills them. `declare`
    and `manifest` share one "which parameters exist" decision (`_targets`),
    so they cannot disagree about a weight.
    """

    WEIGHTS: tuple[W, ...] = (
        W("norm1", shape=lambda d: (d.hidden,), src="{p}.input_layernorm.weight"),
        W(
            "qkv",
            shape=lambda d: (d.q_width + 2 * d.kv_width, d.hidden),
            src=_qkv_src("weight"),
            transform=_qkv_rows,
        ),
        W(
            "qkv_bias",
            shape=lambda d: (d.q_width + 2 * d.kv_width,),
            src=_qkv_src("bias"),
            transform=_qkv_rows,
        ),
        W(
            "sinks",
            shape=lambda d: (d.heads_q,),
            dtype=torch.float32,
            src="{p}.self_attn.sinks",
            transform=_sinks,
        ),
        W(
            "o",
            shape=lambda d: (d.hidden, d.q_width),
            src="{p}.self_attn.o_proj.weight",
            transform=_o_cols,
        ),
        W("o_bias", shape=lambda d: (d.hidden,), src="{p}.self_attn.o_proj.bias"),
        W("norm2", shape=lambda d: (d.hidden,), src="{p}.post_attention_layernorm.weight"),
        W("router", shape=lambda d: (d.num_experts, d.hidden), src="{p}.mlp.router.weight"),
        W("router_bias", shape=lambda d: (d.num_experts,), src="{p}.mlp.router.bias"),
        W(
            "fc1_w",
            shape=lambda d: (d.local_experts, d.fc1_rows, d.fc1_k_pad // 2),
            dtype=torch.uint8,
            src="{p}.mlp.experts.gate_up_proj_blocks",
            transform=_prep_fc1_weight,
        ),
        W(
            "fc1_s",
            shape=lambda d: (d.local_experts, d.fc1_rows, d.fc1_k_pad // 32),
            dtype=torch.uint8,
            src="{p}.mlp.experts.gate_up_proj_scales",
            transform=_prep_fc1_scale,
        ),
        W(
            "fc1_b",
            shape=lambda d: (d.local_experts, d.fc1_rows),
            dtype=torch.float32,
            src="{p}.mlp.experts.gate_up_proj_bias",
            transform=_prep_fc1_bias,
        ),
        W(
            "fc2_w",
            shape=lambda d: (d.local_experts, d.fc2_rows_pad, d.inter_pad // 2),
            dtype=torch.uint8,
            src="{p}.mlp.experts.down_proj_blocks",
            transform=_prep_fc2_weight,
        ),
        W(
            "fc2_s",
            shape=lambda d: (d.local_experts, d.fc2_rows_pad, d.inter_pad // 32),
            dtype=torch.uint8,
            src="{p}.mlp.experts.down_proj_scales",
            transform=_prep_fc2_scale,
        ),
        W(
            "fc2_b",
            shape=lambda d: (d.local_experts, d.fc2_rows_pad),
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
        """Every per-rank width, from the config and the engine's Mapping."""
        cfg = core.model_config.pretrained_config
        m = core.model_config.mapping
        attention_dp = bool(m.enable_attention_dp)
        attn_tp = 1 if attention_dp else m.tp_size
        attn_tp_rank = 0 if attention_dp else m.tp_rank
        assert cfg.num_attention_heads % attn_tp == 0, (cfg.num_attention_heads, attn_tp)
        assert cfg.num_key_value_heads % attn_tp == 0, (
            f"{cfg.num_key_value_heads} kv heads do not split over attention TP {attn_tp}; "
            "replicating kv heads is not implemented"
        )
        assert cfg.num_local_experts % m.moe_ep_size == 0, (cfg.num_local_experts, m.moe_ep_size)
        heads_q = cfg.num_attention_heads // attn_tp
        heads_kv = cfg.num_key_value_heads // attn_tp
        local_experts = cfg.num_local_experts // m.moe_ep_size
        inter_start, inter_valid, inter_pad = intermediate_shard(
            cfg.intermediate_size, m.moe_tp_size, m.moe_tp_rank
        )
        assert inter_valid > 0, (
            f"moe_tp rank {m.moe_tp_rank} of {m.moe_tp_size} holds only padding "
            f"of intermediate {cfg.intermediate_size}"
        )
        return SimpleNamespace(
            num_layers=cfg.num_hidden_layers,
            hidden=cfg.hidden_size,
            head_dim=cfg.head_dim,
            heads_q_total=cfg.num_attention_heads,
            heads_kv_total=cfg.num_key_value_heads,
            attention_dp=attention_dp,
            attn_tp=attn_tp,
            attn_tp_rank=attn_tp_rank,
            heads_q=heads_q,
            heads_kv=heads_kv,
            q_width=heads_q * cfg.head_dim,
            kv_width=heads_kv * cfg.head_dim,
            num_experts=cfg.num_local_experts,
            local_experts=local_experts,
            expert_offset=m.moe_ep_rank * local_experts,
            moe_tp=m.moe_tp_size,
            inter=cfg.intermediate_size,
            inter_start=inter_start,
            inter_valid=inter_valid,
            inter_pad=inter_pad,
            fc1_rows=2 * inter_pad,
            fc1_k_pad=_pad_up(cfg.hidden_size, FC1_K_ALIGN),
            fc2_rows_pad=_pad_up(cfg.hidden_size, WEIGHT_ALIGN),
            vocab=cfg.vocab_size,
            dtype=core.model_config.torch_dtype,
        )

    def declare(self, d: SimpleNamespace) -> nn.ParameterDict:
        """Allocate every weight the table declares, at this rank's share.

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
        """param key -> [(checkpoint key or key tuple, index | None, transform | None)]."""
        d = core.dims
        rows: dict[str, list[tuple[Any, Any, Callable | None]]] = {}
        for entry in self.WEIGHTS:
            for key, i, prefix in self._targets(entry, d):
                rows[key] = [
                    (ckpt, index, entry.transform) for ckpt, index in entry.sources(d, i, prefix)
                ]
        return rows

    def load(self, model, weights) -> None:
        """Fill every declared weight from the checkpoint, and lm_head.

        Beyond a one-to-one copy, two things: multi-key rows (q, k and v feed
        one parameter through one transform) and a vocab-parallel lm_head.

        Every rank reads every checkpoint tensor whole and slices on device.
        A rank that holds a quarter of the experts still reads all of them;
        reading only the window from the lazy slice would cut load time, and
        is left for later.
        """
        core = model.model
        manifest = self.manifest(core)
        consumed: set[str] = set()

        def fill(param: nn.Parameter, ckpt_key, index, transform) -> None:
            keys = ckpt_key if isinstance(ckpt_key, tuple) else (ckpt_key,)
            for key in keys:
                assert key in weights, f"checkpoint key missing: {key}"
            dst = param.data if index is None else param.data[index]
            # [:] materializes the lazy slice.
            src_all = tuple(weights[key][:].to(dst.device, non_blocking=True) for key in keys)
            if transform is not None:
                src = transform(src_all if isinstance(ckpt_key, tuple) else src_all[0], core)
            else:
                assert len(src_all) == 1, "a multi-key row needs a transform"
                src = src_all[0]
            assert dst.shape == src.shape, (ckpt_key, tuple(dst.shape), tuple(src.shape))
            assert src.dtype == dst.dtype, (ckpt_key, src.dtype, dst.dtype)
            dst.copy_(src, non_blocking=True)
            consumed.update(keys)

        assert set(manifest.keys()) == set(core.w.keys()), (
            "manifest/parameter drift",
            set(manifest.keys()) ^ set(core.w.keys()),
        )
        for param_key, sources in manifest.items():
            for ckpt_key, index, transform in sources:
                fill(core.w[param_key], ckpt_key, index, transform)
            if param_key.endswith(self.RELEASE_AFTER):
                torch.cuda.empty_cache()

        # The shell builds lm_head vocab-parallel under TP and whole under
        # attention DP; take the rows it built. A vocab that does not split
        # evenly leaves zero rows at the end of the last shard, as the
        # built-in LMHead pads them.
        lm_head = model.lm_head.weight
        rows = lm_head.shape[0]
        vocab = core.dims.vocab
        tp_rank = 0 if rows == vocab else core.model_config.mapping.tp_rank
        start = tp_rank * rows
        assert rows == vocab or rows * core.model_config.mapping.tp_size >= vocab, (rows, vocab)

        def _lm_head_rows(t: torch.Tensor, core) -> torch.Tensor:
            shard = t[start : min(start + rows, vocab)]
            if shard.shape[0] == rows:
                return shard.contiguous()
            out = torch.zeros(rows, t.shape[1], dtype=t.dtype, device=t.device)
            out[: shard.shape[0]] = shard
            return out

        fill(lm_head, "lm_head.weight", None, _lm_head_rows)

        torch.cuda.synchronize()
        # Every key in this checkpoint is one the table names; nothing is meant
        # to be left untouched, on any rank.
        leftover = set(weights.keys()) - consumed
        assert not leftover, f"unconsumed checkpoint keys: {sorted(leftover)[:8]}"


MODEL_WEIGHTS = GptOssDefaultWeights()
