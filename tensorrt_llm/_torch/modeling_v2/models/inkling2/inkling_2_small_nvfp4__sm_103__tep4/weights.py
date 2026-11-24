# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight table: Inkling-2-Small / sm_103 / attention TP4 + MoE EP4.

Every weight the target holds, at this rank's share, with the checkpoint tensors it comes from
and the transform on the way. `declare` allocates them (the engine materializes the storage),
`load` fills them from the engine's lazy checkpoint view.

* **Attention heads** are split over TP: a rank holds 12 of the 48 query heads and 2 of the 8 KV
  heads, the matching rows of the q, k, v, relative-state (16 per head), gate and sink
  projections, of the q and relative clip scales and of the k/v short convolutions (their
  channels are the KV heads' 128-wide slices, and a global layer's dynamic taps are 4 per KV
  head), and the matching input columns of the output projection, whose partial sums meet in an
  all-reduce. A sliding layer's six projections read one input and are stacked into one weight;
  a global layer reads three inputs (q, k and v each have their own residual mix and norm), so it
  stacks q, r, gate and sink, and appends each dynamic convolution's tap projection to its k or v
  projection.

* **Experts** are split over EP: a rank holds the 96 experts `[96 * rank, 96 * rank + 96)`, and
  reads only those rows of the checkpoint. NVFP4 layers (0-39 in the release) are stored for
  FlashInfer's CuTe DSL MoE: `w13` rows reordered from the checkpoint's per-row gate/up
  interleave (gate `j` at row `2j`, up `j` at `2j + 1`) into the kernel's order, block scales
  reordered the same way and swizzled into 128x4 atoms, and the per-expert global scale
  `amax / outer_scale_denominator` as the GEMM alpha. MXFP8 layers keep the checkpoint's bytes.

* **Short convolutions** are stored as lag-major fp32 taps `[4, C]` (`taps[0]` weights the
  current token): a checkpoint `nn.Conv1d` weight `[C, 1, 4]` is reversed along its last axis; a
  dynamic convolution's `static_w` is lag-major already.

* **Everything else** is replicated: the embedding (whole, so no all-reduce follows the lookup),
  the norms, the block-residual queries (fp32), the relative-logit profiles and the router (bf16
  weight, fp32 logits; fp32 bias). `lm_head` is the shell's vocab-parallel module; `load` fills
  its rows of `unembed`.

Only the text decoder's tensors (`model.llm.*`) are loaded: the target is text only and drafts nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from ..config import ExpertFormat, InklingDims
from ..ops.moe.inkling2_nvfp4_experts import fc1_row_order, swizzle_block_scales

#: Prefix of the text decoder's tensors in the checkpoint. The target loads every one of them and
#: nothing else: not the vision tower (`model.visual.*`) nor the DSpark drafter (`mtp.safetensors`).
LLM = "model.llm"


@dataclass(frozen=True)
class Shares:
    """This rank's share of every split dimension, from the config and the engine's Mapping."""

    model: InklingDims
    tp_size: int
    tp_rank: int
    ep_size: int
    ep_rank: int

    @property
    def heads(self) -> int:
        return self.model.num_heads // self.tp_size

    @property
    def kv_heads(self) -> int:
        return self.model.num_kv_heads // self.tp_size

    @property
    def q_width(self) -> int:
        return self.heads * self.model.head_dim

    @property
    def kv_width(self) -> int:
        return self.kv_heads * self.model.head_dim

    @property
    def r_width(self) -> int:
        return self.heads * self.model.d_rel

    @property
    def local_experts(self) -> int:
        return self.model.num_experts // self.ep_size

    @property
    def expert_offset(self) -> int:
        return self.ep_rank * self.local_experts

    def rows(self, width_total: int) -> slice:
        """This rank's rows of a projection whose output is split evenly over TP."""
        width = width_total // self.tp_size
        return slice(self.tp_rank * width, (self.tp_rank + 1) * width)

    def experts(self) -> slice:
        return slice(self.expert_offset, self.expert_offset + self.local_experts)


def shares_of(model: InklingDims, mapping) -> Shares:
    assert mapping.pp_size == 1 and mapping.cp_size == 1, "no pipeline or context parallelism"
    assert not mapping.enable_attention_dp, "attention DP is not implemented"
    assert mapping.moe_tp_size == 1, "the experts are split by EP only"
    assert mapping.moe_ep_size == mapping.tp_size, "EP must span the TP group"
    s = Shares(model, mapping.tp_size, mapping.tp_rank, mapping.moe_ep_size, mapping.moe_ep_rank)
    assert model.num_heads % s.tp_size == 0 and model.num_kv_heads % s.tp_size == 0
    assert model.num_experts % s.ep_size == 0
    return s


# ---------------------------------------------------------------------------------------------
# Transforms: (sources, shares) -> the tensor to copy in. Sources are the checkpoint tensors in
# the order the entry names them, already on the device.
# ---------------------------------------------------------------------------------------------


def _cat_rows(srcs, s: Shares) -> torch.Tensor:
    """Each source's TP rows, stacked in source order (projections that read one input)."""
    return torch.cat([t[s.rows(t.shape[0])] for t in srcs], dim=0).contiguous()


def _rows(srcs, s: Shares) -> torch.Tensor:
    (t,) = srcs
    return t[s.rows(t.shape[0])].contiguous()


def _o_cols(srcs, s: Shares) -> torch.Tensor:
    (t,) = srcs
    return t[:, s.rows(t.shape[1])].contiguous()


def _conv_taps(srcs, s: Shares) -> torch.Tensor:
    """`nn.Conv1d` weight `[C, 1, 4]` (oldest tap first) -> lag-major fp32 `[4, C]`."""
    (t,) = srcs
    return t[:, 0, :].flip(-1).t().float().contiguous()


def _kv_conv_taps(srcs, s: Shares) -> torch.Tensor:
    (t,) = srcs
    return _conv_taps((t[s.rows(t.shape[0])],), s)


def _static_taps(srcs, s: Shares) -> torch.Tensor:
    """A dynamic convolution's `static_w` `[4, C]` (lag-major): this rank's channels, fp32."""
    (t,) = srcs
    return t[:, s.rows(t.shape[1])].float().contiguous()


def _kv_with_taps(srcs, s: Shares) -> torch.Tensor:
    """k or v projection rows, then the dynamic convolution's tap projection rows (4 per head)."""
    proj, taps = srcs
    return torch.cat([proj[s.rows(proj.shape[0])], taps[s.rows(taps.shape[0])]], dim=0).contiguous()


def _fp32(srcs, s: Shares) -> torch.Tensor:
    (t,) = srcs
    return t.float().reshape(-1, t.shape[-1]).contiguous()


def _as_is(srcs, s: Shares) -> torch.Tensor:
    (t,) = srcs
    return t.contiguous()


def _nvfp4_w13(srcs, s: Shares) -> torch.Tensor:
    (t,) = srcs
    return t[:, fc1_row_order(t.shape[1] // 2, t.device)].contiguous()


def _nvfp4_w13_scale(srcs, s: Shares) -> torch.Tensor:
    (t,) = srcs
    rows = t.view(torch.uint8)[:, fc1_row_order(t.shape[1] // 2, t.device)]
    return swizzle_block_scales(rows)


def _nvfp4_scale(srcs, s: Shares) -> torch.Tensor:
    (t,) = srcs
    return swizzle_block_scales(t.view(torch.uint8))


def _nvfp4_alpha(srcs, s: Shares) -> torch.Tensor:
    """Per local expert: `amax / outer_scale_denominator`, fp32."""
    amax, denominator = srcs
    return (amax[s.experts()].float() / denominator.float().reshape(())).contiguous()


def _raw_u8(srcs, s: Shares) -> torch.Tensor:
    (t,) = srcs
    return t.view(torch.uint8).contiguous()


@dataclass(frozen=True)
class Entry:
    """One weight of one layer (or of the model): shape, dtype, checkpoint sources, transform.

    `sources` are checkpoint names; `expert_rows` marks sources read only at this rank's expert
    rows (the lazy view reads just those bytes).
    """

    name: str
    shape: tuple
    dtype: torch.dtype
    sources: tuple[str, ...]
    transform: Callable[[Any, Shares], torch.Tensor]
    expert_rows: tuple[bool, ...] = ()


def layer_entries(s: Shares, layer: int) -> list[Entry]:
    d = s.model
    p = f"{LLM}.layers.{layer}"
    a = f"{p}.attn"
    h, bf16, f32, u8 = d.hidden_size, torch.bfloat16, torch.float32, torch.uint8
    entries = [
        Entry("mlp_norm", (h,), bf16, (f"{p}.mlp_norm.weight",), _as_is),
        Entry("wo", (h, s.q_width), bf16, (f"{a}.wo_ud.weight",), _o_cols),
        Entry("q_scale", (s.heads, d.head_dim), bf16, (f"{a}.q_l2_clip.scale",), _rows),
        Entry("r_scale", (s.heads, d.d_rel), bf16, (f"{a}.rel_l2_clip.scale",), _rows),
        Entry(
            "rel_proj",
            (d.d_rel, d.rel_extent_of(layer)),
            bf16,
            (f"{a}.rel_logits_proj.proj",),
            _as_is,
        ),
        Entry("attn_conv", (4, h), f32, (f"{p}.attn_sconv.weight",), _conv_taps),
        Entry("mlp_conv", (4, h), f32, (f"{p}.mlp_sconv.weight",), _conv_taps),
        Entry("router", (d.num_experts, h), bf16, (f"{p}.mlp.gate.weight",), _as_is),
        Entry("router_bias", (d.num_experts,), f32, (f"{p}.mlp.gate.bias",), _as_is),
        Entry("mlp_query", (1, h), f32, (f"{LLM}.attn_res.mlp_queries.{layer}",), _fp32),
    ]
    stacked_q = s.q_width + s.r_width + s.q_width + s.heads
    if d.is_global(layer):
        dyn = s.kv_heads * 4
        entries += [
            Entry("q_norm", (h,), bf16, (f"{p}.q_input_norm.weight",), _as_is),
            Entry("k_norm", (h,), bf16, (f"{p}.k_input_norm.weight",), _as_is),
            Entry("v_norm", (h,), bf16, (f"{p}.v_input_norm.weight",), _as_is),
            Entry(
                "w_q",
                (stacked_q, h),
                bf16,
                (
                    f"{a}.wq_du.weight",
                    f"{a}.wr_du.weight",
                    f"{a}.gate_attn.weight",
                    f"{a}.attn_sink_proj.weight",
                ),
                _cat_rows,
            ),
            Entry(
                "w_k",
                (s.kv_width + dyn, h),
                bf16,
                (f"{a}.wk_dv.weight", f"{a}.k_sconv.weight_proj"),
                _kv_with_taps,
            ),
            Entry(
                "w_v",
                (s.kv_width + dyn, h),
                bf16,
                (f"{a}.wv_dv.weight", f"{a}.v_sconv.weight_proj"),
                _kv_with_taps,
            ),
            Entry("k_conv", (4, s.kv_width), f32, (f"{a}.k_sconv.static_w",), _static_taps),
            Entry("v_conv", (4, s.kv_width), f32, (f"{a}.v_sconv.static_w",), _static_taps),
        ]
    else:
        entries += [
            Entry("attn_norm", (h,), bf16, (f"{p}.attn_norm.weight",), _as_is),
            Entry(
                "w_in",
                (stacked_q + 2 * s.kv_width, h),
                bf16,
                (
                    f"{a}.wq_du.weight",
                    f"{a}.wr_du.weight",
                    f"{a}.gate_attn.weight",
                    f"{a}.attn_sink_proj.weight",
                    f"{a}.wk_dv.weight",
                    f"{a}.wv_dv.weight",
                ),
                _cat_rows,
            ),
            Entry("k_conv", (4, s.kv_width), f32, (f"{a}.k_sconv.weight",), _kv_conv_taps),
            Entry("v_conv", (4, s.kv_width), f32, (f"{a}.v_sconv.weight",), _kv_conv_taps),
        ]
    if layer > 0:
        queries = 3 if d.is_global(layer) else 1
        entries.append(
            Entry(
                "attn_query",
                (queries, h),
                f32,
                (f"{LLM}.attn_res.attn_queries.{layer - 1}",),
                _fp32,
            )
        )

    e, i = s.local_experts, d.expert_intermediate_size
    x = f"{p}.mlp.experts"
    fmt = d.expert_formats[layer]
    if fmt is ExpertFormat.NVFP4:
        entries += [
            Entry("w13", (e, 2 * i, h // 2), u8, (f"{x}.w13_weight",), _nvfp4_w13, (True,)),
            Entry(
                "w13_sf",
                (e, 2 * i, h // 16),
                u8,
                (f"{x}.w13_weight_scale",),
                _nvfp4_w13_scale,
                (True,),
            ),
            Entry(
                "w13_alpha",
                (e,),
                f32,
                (f"{x}.w13_weight_amax", f"{x}.w13_weight_outer_scale_denominator"),
                _nvfp4_alpha,
            ),
            Entry("w2", (e, h, i // 2), u8, (f"{x}.w2_weight",), _as_is, (True,)),
            Entry("w2_sf", (e, h, i // 16), u8, (f"{x}.w2_weight_scale",), _nvfp4_scale, (True,)),
            Entry(
                "w2_alpha",
                (e,),
                f32,
                (f"{x}.w2_weight_amax", f"{x}.w2_weight_outer_scale_denominator"),
                _nvfp4_alpha,
            ),
        ]
    elif fmt is ExpertFormat.MXFP8:
        e4m3 = torch.float8_e4m3fn
        entries += [
            Entry("w13", (e, 2 * i, h), e4m3, (f"{x}.w13_weight",), _as_is, (True,)),
            Entry("w13_s", (e, 2 * i, h // 32), u8, (f"{x}.w13_weight_scale",), _raw_u8, (True,)),
            Entry("w2", (e, h, i), e4m3, (f"{x}.w2_weight",), _as_is, (True,)),
            Entry("w2_s", (e, h, i // 32), u8, (f"{x}.w2_weight_scale",), _raw_u8, (True,)),
        ]
    else:
        raise NotImplementedError(
            f"layer {layer}: {fmt.value} experts are not supported by this target"
        )
    return entries


def model_entries(s: Shares) -> list[Entry]:
    d = s.model
    h, bf16, f32 = d.hidden_size, torch.bfloat16, torch.float32
    return [
        Entry("embed", (d.vocab_size, h), bf16, (f"{LLM}.embed.weight",), _as_is),
        Entry("embed_norm", (h,), bf16, (f"{LLM}.embed_norm.weight",), _as_is),
        Entry("final_query", (1, h), f32, (f"{LLM}.attn_res.output_query",), _fp32),
        Entry("final_norm", (h,), bf16, (f"{LLM}.norm.weight",), _as_is),
    ]


def _key(layer: Optional[int], name: str) -> str:
    return name if layer is None else f"l{layer}_{name}"


def all_entries(s: Shares) -> list[tuple[Optional[int], Entry]]:
    rows = [(None, entry) for entry in model_entries(s)]
    for layer in range(s.model.num_layers):
        rows += [(layer, entry) for entry in layer_entries(s, layer)]
    return rows


def declare(s: Shares) -> nn.ParameterDict:
    """Every weight at this rank's share; the engine's meta init gives them storage later."""
    w = nn.ParameterDict()
    for layer, entry in all_entries(s):
        w[_key(layer, entry.name)] = nn.Parameter(
            torch.empty(entry.shape, dtype=entry.dtype), requires_grad=False
        )
    return w


def layer_view(w: nn.ParameterDict, layer: int) -> SimpleNamespace:
    """Layer `layer`'s weights by name, `w.<name>`."""
    prefix = f"l{layer}_"
    return SimpleNamespace(**{k[len(prefix) :]: v for k, v in w.items() if k.startswith(prefix)})


def load(model, weights, s: Shares) -> None:
    """Fill every declared weight and the shell's lm_head rows from the checkpoint.

    `weights` is the engine's lazy view (`weights[name][a:b]` reads only those rows). Every
    text-decoder tensor of the checkpoint must be consumed; other tensors are not read.
    """
    core = model.model
    consumed: set[str] = set()
    expert_rows = s.experts()

    def read(name: str, rows: Optional[slice], device: torch.device) -> torch.Tensor:
        assert name in weights, f"checkpoint tensor missing: {name}"
        consumed.add(name)
        t = weights[name][rows] if rows is not None else weights[name][:]
        return t.to(device, non_blocking=True)

    for layer, entry in all_entries(s):
        dst = core.w[_key(layer, entry.name)].data
        flags = entry.expert_rows or (False,) * len(entry.sources)
        srcs = tuple(
            read(n, expert_rows if f else None, dst.device) for n, f in zip(entry.sources, flags)
        )
        src = entry.transform(srcs, s)
        if src.dtype != dst.dtype and src.dtype == torch.uint8 and dst.dtype.itemsize == 1:
            src = src.view(dst.dtype)
        assert src.shape == dst.shape, (layer, entry.name, tuple(src.shape), tuple(dst.shape))
        assert src.dtype == dst.dtype, (layer, entry.name, src.dtype, dst.dtype)
        dst.copy_(src, non_blocking=True)
        if entry.expert_rows:
            del srcs, src
            torch.cuda.empty_cache()

    lm_head = model.lm_head.weight
    vocab = s.model.vocab_size
    rows = lm_head.shape[0]
    start = 0 if rows == vocab else s.tp_rank * rows
    unembed = read(f"{LLM}.unembed.weight", slice(start, min(start + rows, vocab)), lm_head.device)
    if unembed.shape[0] < rows:
        lm_head.data.zero_()
    lm_head.data[: unembed.shape[0]].copy_(unembed)
    torch.cuda.synchronize()

    leftover = [
        name for name in weights.keys() if name.startswith(f"{LLM}.") and name not in consumed
    ]
    assert not leftover, f"text-decoder tensors the table does not name: {sorted(leftover)[:8]}"
