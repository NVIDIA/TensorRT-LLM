# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight tables: deepseek-v4.1-flash-nvfp4 / sm_103 / unified.

Two tables, one per weight pass of the engine:

* ``MODEL_WEIGHTS`` -- the trunk, the vision tower and the aligner, declared into
  the core's ``w`` and filled by the shell's ``load_weights``;
* ``DRAFT_WEIGHTS`` -- the three DSpark stages (``mtp.*``), declared into the
  core's ``w_ds`` only when the engine was built with a ``speculative_config``,
  and filled by the loader's second, lazily mapped pass over the same
  checkpoint (``load_draft_weights``).

Each states every weight once, as a ``W`` row: shape, checkpoint source, dtype,
load-time transform and the layers that carry it. ``dims`` is the one place a
per-rank width is derived, from the mapping alone.

**Per-rank sources.** The member (tep4 / dep4 / tp4, and tep8 / dep8 over two
nodes) decides the slices, read off the mapping, never off the path segment:

* attention heads: TP members hold heads ``[h r, h r + h)``, ``h = 64 / tp``
  (16 at tp4, 8 at tp8) -- ``wq_b`` rows, ``attn_sink``, ``wo_a`` groups
  ``[g r, g r + g)``, ``g = 8 / tp``, and ``wo_b`` columns split on the same
  boundary; attention DP holds all 64;
* routed experts: EP members hold experts ``[e r, e r + e)`` whole, ``e = 384 / tp``
  (96 at EP4, 48 at EP8); tp4 holds all 384 at intermediate ``[576r, 576r+576)``;
* Engram tables (``layers.{1,14}.engram.embed.{weight,scale}``): with
  host_offload on (the default) they are no parameter at all -- loaded whole,
  once per node, into the shared host copy of the sibling ``engram_host_table``
  (``core.attach_engram_host_table``) and checked against the checkpoint here,
  never copied to a device; with ``TRTLLM_DSV41_ENGRAM_HOST_OFFLOAD=0``, rows
  ``[ceil(rows/tp) r, ...)`` on the device of every member;
* everything else is replicated: hyper-connections, norms, ``wq_a``/``wkv``,
  compressor, indexer, router (``gate.bias`` and ``gate.bias_vl``), shared
  expert, embedding, and the whole vision tower + aligner.

A source that is a per-rank part of a checkpoint tensor is a ``Slice``: the
part is read **before** any transform, because a relayout is a function of the
per-rank row count, and a lazily mapped safetensors entry then reads only this
rank's bytes off disk.

Formats. Dense FP8 weights are E4M3 in 32x32 blocks with one E8M0 ``.scale``
per block: exactly MXFP8 once each block scale is repeated over its 32 rows
(``repeat_interleave`` -- tiling is silently wrong) and 128x4-swizzled, which
is what ``mxfp8_mxfp8_gemm`` reads. Fused projections expand each part's scales
and then concatenate (``wq_a || wkv``; the shared expert's ``[w1 ; w3]`` =
``[gate ; up]``, the half order ``silu_and_mul`` reads). ``wo_a`` is
dequantized to bf16 exactly as the checkpoint's own converter does. Routed
NVFP4 experts are packed for the trtllm-gen block-scale runner: FC1 =
``[w3 ; w1]`` = ``[up ; gate]`` (the OPPOSITE half order), interleaved, 32-row
block shuffled (weight and scale bytes), scales 128x4 swizzled; FC2 shuffled,
scales swizzled. The per-expert fp32 scalars are loaded whole (the layer-level
maxima need all 384) and folded in the core's ``post_load_weights``.

The vision tower (``vision.*``), the aligner (``aligner.*``) and the three span
delimiters are bf16 in the checkpoint and load as they are, with two
relayouts: ``aligner.w1``'s columns go from the reference unfold order
``(channel, kh, kw)`` to ``(kh, kw, channel)``, so the aligner input is a
plain row gather of the ViT output; ``image_start`` / ``image_end`` /
``image_newline`` are stacked into one ``[3, 5120]`` table in that order.

Predicted non-load sets of ``MODEL_WEIGHTS.load`` (asserted exactly, never relaxed):

* ``mtp.*`` (2401 keys) -- the three DSpark stages, ``DRAFT_WEIGHTS``' rows;
* EP members: the weight and weight_scale keys of every routed expert outside
  this rank's window (6 per expert per layer);
* pipeline stages: every key of another stage's layers, except the previous
  stage's last ``hc_ffn`` triple, and ``head.weight`` outside the last stage.

**The DSpark drafter** (``DRAFT_WEIGHTS``, layer ``s`` = stage ``mtp.{s}``):
replicated on every rank of every member. Dense stage weights are MXFP8
exactly like the trunk's; the 128 routed experts per stage are MXFP4 (E2M1 x
UE8M0/32) packed for the MXFP4 trtllm-gen runner with the NVFP4 runner's
byte-generic relayouts (FC1 = ``[w3 ; w1]`` interleaved and 32-row shuffled,
FC2 shuffled, scales 128x4 swizzled); ``wkv`` of the three stages is also
stacked into one ``[1536, 5120]`` operand for the ring write. Its predicted
non-load set is the confidence head (the worker drafts the whole block,
``confidence_threshold`` 0) and the all-zero ``gate.bias_vl``.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from tensorrt_llm._torch._experimental.modeling_v2._weights import ModelWeights, W

from . import engram_host_table as _engram_host

SF16 = 16  # NVFP4: one e4m3 scale per 16 along K
MX = 32  # MXFP8 / checkpoint FP8 block

# Hyper-connection mixes: 4 pre + 4 post + 16 comb per copy count of 4.
HC = 4
MIX = (2 + HC) * HC
IDX_HEADS = 32
IDX_DIM = 128
VIT_DIM = 1024
VIT_LAYERS = 32
VIT_HEADS = 16
VIT_INTER = 2816
VIT_PATCH_IN = 3 * 14 * 14  # a patch flattened (channel, kh, kw)
VIT_RATIO = 3  # the aligner's 3x3 block
_AL_TAPS = VIT_RATIO * VIT_RATIO
# The checkpoint's Engram layers (1 and 14). One row pair per slot, because the
# two tables have different row counts and a row's shape cannot depend on its
# layer index.
_ENGRAM_SLOTS = 2

# Parameter-key suffixes after which the load releases transform scratch.
_RELEASE_AFTER = ("_fc1_w", "_fc1_s", "_fc2_w", "_fc2_s", "_eng_table")

# Copy threshold above which an untransformed row is streamed in row chunks,
# so a 25 GB Engram shard never sits in host RAM at once.
_CHUNK_ELEMS = 1 << 28


@dataclass(frozen=True)
class Slice:
    """This rank's part of one checkpoint tensor: ``weights[key][rows]``.

    ``rows`` is an index tuple over the tensor's leading dimensions, applied
    before any transform.
    """

    key: str
    rows: tuple


def _split_source(ckpt) -> tuple[tuple[str, ...], tuple, bool]:
    """``(keys, per-key index tuples or None, multi)`` of one manifest source."""
    multi = isinstance(ckpt, tuple)
    parts = ckpt if multi else (ckpt,)
    keys = tuple(p.key if isinstance(p, Slice) else p for p in parts)
    srcs = tuple(p.rows if isinstance(p, Slice) else None for p in parts)
    return keys, srcs, multi


def _is_lazy(v) -> bool:
    return not isinstance(v, torch.Tensor)


def _materialize(v, src=None) -> torch.Tensor:
    """Realize a checkpoint entry (a lazy safetensors slice or a tensor),
    reading only ``src`` of it. 0-dim entries (NVFP4 scalars) reject ``[:]``."""
    if _is_lazy(v):
        if len(v.get_shape()) == 0:
            return v[()]
        return v[src] if src is not None else v[:]
    if v.ndim == 0:
        return v
    return v[src] if src is not None else v


def _shape(v) -> tuple:
    return tuple(v.get_shape()) if _is_lazy(v) else tuple(v.shape)


def mx_sf_numel(n: int, k: int) -> int:
    """Bytes of the 128x4-swizzled UE8M0 scales of an [n, k] MXFP8 weight."""
    return ((n + 127) // 128 * 128) * ((k // MX + 3) // 4 * 4)


# ---- transforms (run on the destination device) ---------------------------


def _swizzle(x: torch.Tensor) -> torch.Tensor:
    """128x4 block-scale swizzle of a [M, C] uint8 scale matrix, flat."""
    return torch.ops.trtllm.block_scale_interleave(x.contiguous()).reshape(-1)


def _expand_rows(scale: torch.Tensor, rows: int) -> torch.Tensor:
    """[ceil(N/32), K/32] E8M0 block scales -> [N, K/32] uint8 per-row scales."""
    return scale.view(torch.uint8).repeat_interleave(MX, dim=0)[:rows]


def _mx_sf(t, rows) -> torch.Tensor:
    """One fused MXFP8 weight's 128x4-swizzled scales: each part's 32x32 block
    scales expanded to its ``rows``, concatenated, then swizzled once."""
    scales = t if isinstance(t, tuple) else (t,)
    assert len(scales) == len(rows), (len(scales), rows)
    parts = [_expand_rows(s, n) for s, n in zip(scales, rows)]
    return _swizzle(torch.cat(parts, dim=0))


def _wqkv_a_sf(t, core) -> torch.Tensor:
    return _mx_sf(t, (core.q_lora, core.head_dim))


def _wq_b_sf(t, core) -> torch.Tensor:
    return _mx_sf(t, (core.h_loc * core.head_dim,))


def _hidden_sf(t, core) -> torch.Tensor:
    """Scales of a weight with ``hidden`` output rows (``wo_b``, ``sh_w2``)."""
    return _mx_sf(t, (core.hidden,))


def _idx_wq_b_sf(t, core) -> torch.Tensor:
    return _mx_sf(t, (IDX_HEADS * IDX_DIM,))


def _sh_w13_sf(t, core) -> torch.Tensor:
    return _mx_sf(t, (core.moe_inter, core.moe_inter))


def _eng_wkv_sf(t, core) -> torch.Tensor:
    return _mx_sf(t, (core.eng_wkv_n,))


def _cat_rows_f32(t, core) -> torch.Tensor:
    return torch.cat([x.float() for x in t], dim=0).contiguous()


def _to_f32(t, core) -> torch.Tensor:
    return t.float().contiguous()


def _u8(t, core) -> torch.Tensor:
    return t.view(torch.uint8).contiguous()


def _idx_wproj(t, core) -> torch.Tensor:
    """weights_proj with the indexer's 128^-0.5 * 32^-0.5 = 2^-6 folded in
    (a power of two: exact in bf16)."""
    return (t.float() * 2.0**-6).to(torch.bfloat16).contiguous()


def _dequant_wo_a(t, core) -> torch.Tensor:
    """FP8 E4M3 x E8M0 32x32 -> bf16, the checkpoint converter's own recipe."""
    w, s = t
    out = w.float().unflatten(0, (-1, MX)).unflatten(-1, (-1, MX)) * s.float()[:, None, :, None]
    return out.flatten(2, 3).flatten(0, 1).to(torch.bfloat16).contiguous()


def _block32_perm(rows: int, device) -> torch.Tensor:
    """Gather index of the 32-row block shuffle: inside each block of 32 rows,
    source row ``4u + v`` lands at destination row ``8v + u``."""
    assert rows % 32 == 0, rows
    j = torch.arange(32)
    dst = (j % 4) * 8 + j // 4
    idx = torch.empty(32, dtype=torch.long)
    idx[dst] = j
    blocks = rows // 32
    return (idx.repeat(blocks) + torch.arange(blocks).repeat_interleave(32) * 32).to(device)


def _interleave_perm(rows: int, device) -> torch.Tensor:
    """[up | gate] -> up0, gate0, up1, gate1, ..."""
    p = torch.empty(rows, dtype=torch.long)
    p[0::2] = torch.arange(0, rows // 2)
    p[1::2] = torch.arange(rows // 2, rows)
    return p.to(device)


def _fc1_perm(rows: int, device) -> torch.Tensor:
    return _interleave_perm(rows, device)[_block32_perm(rows, device)]


def _expert_fc1_w(t, core) -> torch.Tensor:
    """(w3 = up, w1 = gate) packed nibbles -> one expert's FC1 operand."""
    up, gate = t
    w = torch.cat([up, gate], dim=0)
    return torch.index_select(w, 0, _fc1_perm(w.shape[0], w.device)).contiguous()


def _expert_fc1_s(t, core) -> torch.Tensor:
    up, gate = t
    s = torch.cat([up.view(torch.uint8), gate.view(torch.uint8)], dim=0)
    rows, cols = s.shape
    s = torch.index_select(s, 0, _fc1_perm(rows, s.device))
    return _swizzle(s).reshape(rows, cols)


def _expert_fc2_w(t, core) -> torch.Tensor:
    return torch.index_select(t, 0, _block32_perm(t.shape[0], t.device)).contiguous()


def _expert_fc2_s(t, core) -> torch.Tensor:
    s = t.view(torch.uint8)
    rows, cols = s.shape
    s = torch.index_select(s, 0, _block32_perm(rows, s.device))
    return _swizzle(s).reshape(rows, cols)


def _aligner_w1(t, core) -> torch.Tensor:
    """[5120, 9216] columns ``c * 9 + (kh * 3 + kw)`` -> ``(kh * 3 + kw) * 1024 + c``."""
    n, k = t.shape
    assert k == VIT_DIM * _AL_TAPS, t.shape
    return t.view(n, VIT_DIM, _AL_TAPS).transpose(1, 2).reshape(n, k).contiguous()


def _stack_rows(t, core) -> torch.Tensor:
    return torch.stack(t, dim=0).contiguous()


# ---- sources ------------------------------------------------------------------


def _one(template):
    """A single checkpoint key, no index into the destination."""
    return lambda d, i, t=template: [(t.format(i=i), None)]


def _rows(template, slice_fn):
    """This rank's rows of one checkpoint tensor."""
    return lambda d, i, t=template: [(Slice(t.format(i=i), slice_fn(d)), None)]


def _pair(a, b):
    """Two checkpoint tensors handed to the transform together."""
    return lambda d, i, x=a, y=b: [((x.format(i=i), y.format(i=i)), None)]


def _fused(*parts):
    """One parameter assembled from several checkpoint tensors, each written to
    its own destination rows: ``parts`` are ``(template, d -> dst slice)``."""
    return lambda d, i, p=parts: [(t.format(i=i), fn(d)) for t, fn in p]


def _wo_a_src(d, i):
    a = f"layers.{i}.attn"
    return [
        ((Slice(f"{a}.wo_a.weight", (d.o_rows,)), Slice(f"{a}.wo_a.scale", (d.o_sf_rows,))), None)
    ]


def _expert_window(names, transform_pair: bool, cols=None):
    """This rank's routed experts, one source per local expert at ``(local,)``.

    ``names`` is ``(up, gate)`` for an FC1 row (w3, w1: the up half first) or
    ``(down,)`` for an FC2 row. Every source is this rank's part of the expert:
    its intermediate rows (FC1) or columns (FC2) under MoE TP, all of it under EP.
    """

    def build(d, i):
        out = []
        for local in range(d.e_loc):
            q = f"layers.{i}.ffn.experts.{d.e_off + local}"
            if transform_pair:
                rows = (slice(d.i_off, d.i_off + d.i_loc),)
                src = tuple(Slice(f"{q}.{n}", rows) for n in names)
            else:
                src = Slice(f"{q}.{names[0]}", cols(d))
            out.append((src, (local,)))
        return out

    return build


def _expert_scalars(proj, suffix):
    """A per-expert fp32 scalar over the whole routing space, not windowed: the
    layer-level maxima ``post_load_weights`` folds need all 384 on every rank."""
    return lambda d, i: [
        (f"layers.{i}.ffn.experts.{e}.{proj}.{suffix}", (e,)) for e in range(d.num_experts)
    ]


def _engram_shard(name):
    return lambda d, i: [
        (
            Slice(
                f"layers.{i}.engram.embed.{name}",
                (
                    slice(
                        d.eng_start[d.engram_layers.index(i)], d.eng_end[d.engram_layers.index(i)]
                    ),
                ),
            ),
            slice(0, d.eng_end[d.engram_layers.index(i)] - d.eng_start[d.engram_layers.index(i)]),
        )
    ]


# ---- layer sets ---------------------------------------------------------------


def _stage(d):
    """This pipeline stage's layers."""
    return d.pp_layer_ids


def _hc_ffn_layers(d):
    """The stage's layers, plus the previous stage's last layer: its FFN mixes
    collapse this stage's first attention input."""
    return list(d.pp_layer_ids) + ([d.pp_prev] if d.pp_prev is not None else [])


def _kv_sources_of_ratio(r):
    return lambda d: [i for i in d.pp_layer_ids if i in d.kv_sources and d.ratios[i] == r]


def _kv_sources(d):
    return [i for i in d.pp_layer_ids if i in d.kv_sources]


def _index_sources(d):
    return [i for i in d.pp_layer_ids if i in d.index_sources]


def _engram_layers(d):
    return [i for i in d.pp_layer_ids if i in d.engram_layers]


def _engram_device_slot(li):
    """Engram slot ``li``'s layer, when its table is a device row shard."""
    return lambda d: (
        [d.engram_layers[li]]
        if not d.eng_host and li < len(d.engram_layers) and d.engram_layers[li] in d.local
        else []
    )


def _mlp_scalar_rows():
    return tuple(
        W(
            f"e_{name}",
            shape=lambda d: (d.num_experts,),
            dtype=torch.float32,
            src=_expert_scalars(proj, suffix),
            layers=_stage,
        )
        for name, (proj, suffix) in {
            "isc1": ("w1", "input_scale"),
            "isc3": ("w3", "input_scale"),
            "isc2": ("w2", "input_scale"),
            "ws1": ("w1", "weight_scale_2"),
            "ws3": ("w3", "weight_scale_2"),
            "ws2": ("w2", "weight_scale_2"),
        }.items()
    )


def _hc_rows(sublayer, layers):
    return (
        W(
            f"hc_{sublayer}_fn",
            shape=lambda d: (MIX, d.hc_k),
            dtype=torch.float32,
            src=_one(f"layers.{{i}}.hc_{sublayer}_fn"),
            layers=layers,
        ),
        W(
            f"hc_{sublayer}_base",
            shape=lambda d: (MIX,),
            dtype=torch.float32,
            src=_one(f"layers.{{i}}.hc_{sublayer}_base"),
            layers=layers,
        ),
        W(
            f"hc_{sublayer}_scale",
            shape=lambda d: (3,),
            dtype=torch.float32,
            src=_one(f"layers.{{i}}.hc_{sublayer}_scale"),
            layers=layers,
        ),
    )


def _vision_rows():
    """The vision tower, aligner and span delimiters (replicated, bf16)."""
    rows = [
        W(
            "vis_patch_w",
            shape=lambda d: (VIT_DIM, VIT_PATCH_IN),
            src="vision.patch_embed.proj.weight",
            layers=None,
        ),
        W(
            "vis_patch_b",
            shape=lambda d: (VIT_DIM,),
            src="vision.patch_embed.proj.bias",
            layers=None,
        ),
        W("vis_norm", shape=lambda d: (VIT_DIM,), src="vision.norm.weight", layers=None),
        W(
            "vis_al_w1",
            shape=lambda d: (d.hidden, VIT_DIM * _AL_TAPS),
            src="aligner.w1.weight",
            transform=_aligner_w1,
            layers=None,
        ),
        W("vis_al_b1", shape=lambda d: (d.hidden,), src="aligner.w1.bias", layers=None),
        W("vis_al_w2", shape=lambda d: (d.hidden, d.hidden), src="aligner.w2.weight", layers=None),
        W("vis_al_b2", shape=lambda d: (d.hidden,), src="aligner.w2.bias", layers=None),
        W(
            "vis_span_tail",
            shape=lambda d: (3, d.hidden),
            src=lambda d, i: [(("image_start", "image_end", "image_newline"), None)],
            transform=_stack_rows,
            layers=None,
        ),
    ]
    for L in range(VIT_LAYERS):
        p, v = f"vis{L}_", f"vision.blocks.{L}."
        rows += [
            W(p + "norm1", shape=lambda d: (VIT_DIM,), src=v + "norm1.weight", layers=None),
            W(
                p + "wqkv_w",
                shape=lambda d: (3 * VIT_DIM, VIT_DIM),
                src=v + "attn.wqkv.weight",
                layers=None,
            ),
            W(p + "wqkv_b", shape=lambda d: (3 * VIT_DIM,), src=v + "attn.wqkv.bias", layers=None),
            W(
                p + "wo_w",
                shape=lambda d: (VIT_DIM, VIT_DIM),
                src=v + "attn.wo.weight",
                layers=None,
            ),
            W(p + "wo_b", shape=lambda d: (VIT_DIM,), src=v + "attn.wo.bias", layers=None),
            W(p + "norm2", shape=lambda d: (VIT_DIM,), src=v + "norm2.weight", layers=None),
            W(
                p + "w1",
                shape=lambda d: (2 * VIT_INTER, VIT_DIM),
                src=v + "mlp.w1.weight",
                layers=None,
            ),
            W(p + "w2", shape=lambda d: (VIT_DIM, VIT_INTER), src=v + "mlp.w2.weight", layers=None),
        ]
    return tuple(rows)


def _engram_vocab_size(tc) -> int:
    """The Engram bucket vocabulary as one number, from either form of the config:
    the checkpoint's integer, or a per-n-gram list of the same value."""
    bucket = getattr(tc, "engram_bucket_vocab_size", None)
    if bucket:
        return int(bucket)
    v = tc.engram_vocab_size
    if isinstance(v, (list, tuple)):
        assert len(set(v)) == 1, v
        v = v[0]
    return int(v)


def _text_config(core):
    cfg = core.model_config.pretrained_config
    tc = getattr(cfg, "text_config", None)
    return cfg if tc is None else tc


def _layout(core) -> SimpleNamespace:
    """The per-rank partition of the trunk, from the configuration and mapping alone.

    Shared by ``dims`` and the core: the core reads its head, group, expert and
    Engram windows from here rather than deriving them a second time.
    """
    tc = _text_config(core)
    mapping = core.model_config.mapping
    tp = mapping.tp_size
    dp = bool(mapping.enable_attention_dp)
    attn_tp = 1 if dp else tp
    attn_rank = 0 if dp else mapping.tp_rank
    heads, head_dim, o_groups, o_lora = (
        tc.num_attention_heads,
        tc.head_dim,
        tc.o_groups,
        tc.o_lora_rank,
    )
    h_loc = heads // attn_tp
    g_loc = o_groups // attn_tp
    h0 = attn_rank * h_loc
    g0 = attn_rank * g_loc
    q_rows = slice(h0 * head_dim, (h0 + h_loc) * head_dim)
    o_rows = slice(g0 * o_lora, (g0 + g_loc) * o_lora)
    num_experts, moe_inter = tc.n_routed_experts, tc.moe_intermediate_size
    if mapping.moe_ep_size == tp:
        e_loc = num_experts // tp
        e_off = e_loc * mapping.moe_ep_rank
        i_loc, i_off = moe_inter, 0
    else:
        e_loc, e_off = num_experts, 0
        i_loc = moe_inter // tp
        i_off = i_loc * mapping.moe_tp_rank
    num_layers = tc.num_hidden_layers
    pp_layer_ids = [int(i) for i in mapping.pp_layers(num_layers)]
    eng_rows = [int(n) for n in tc.engram_num_embeddings]
    eng_host = _engram_host.host_offload_enabled()
    if eng_host:
        eng_start, eng_end = [0] * len(eng_rows), list(eng_rows)
        eng_part = list(eng_rows)
    else:
        eng_part = [(n + tp - 1) // tp for n in eng_rows]
        eng_start = [p * mapping.tp_rank for p in eng_part]
        eng_end = [min(s + p, n) for s, p, n in zip(eng_start, eng_part, eng_rows)]
    eng_cols = (tc.engram_max_ngram_size - 1) * tc.engram_n_heads
    return SimpleNamespace(
        num_layers=num_layers,
        hidden=tc.hidden_size,
        vocab=tc.vocab_size,
        heads=heads,
        head_dim=head_dim,
        q_lora=tc.q_lora_rank,
        o_groups=o_groups,
        o_lora=o_lora,
        group_in=heads * head_dim // o_groups,
        attn_tp=attn_tp,
        attn_rank=attn_rank,
        h_loc=h_loc,
        g_loc=g_loc,
        q_rows=q_rows,
        q_sf_rows=slice(q_rows.start // MX, q_rows.stop // MX),
        sink_rows=slice(h0, h0 + h_loc),
        o_rows=o_rows,
        o_sf_rows=slice(o_rows.start // MX, o_rows.stop // MX),
        num_experts=num_experts,
        moe_inter=moe_inter,
        e_loc=e_loc,
        e_off=e_off,
        i_loc=i_loc,
        i_off=i_off,
        hc_k=HC * tc.hidden_size,
        ratios=list(tc.compress_ratios)[:num_layers],
        kv_sources=tuple(tc.kv_source_layer_ids),
        index_sources=tuple(tc.index_source_layer_ids),
        engram_layers=tuple(tc.engram_layer_ids),
        eng_rows=eng_rows,
        eng_part=eng_part,
        eng_start=eng_start,
        eng_end=eng_end,
        eng_host=eng_host,
        eng_dim=tc.engram_head_dim,
        eng_cols=eng_cols,
        eng_in=eng_cols * tc.engram_head_dim,
        eng_wkv_n=(HC + 1) * tc.hidden_size,
        pp_layer_ids=pp_layer_ids,
        local=frozenset(pp_layer_ids),
        pp_prev=pp_layer_ids[0] - 1 if pp_layer_ids[0] > 0 else None,
        last_pp=mapping.is_last_pp_rank(),
        dtype=core.model_config.torch_dtype,
    )


class DeepseekV41UnifiedWeights(ModelWeights):
    """deepseek-v4.1-flash-nvfp4 / sm_103 / unified: the trunk, vision tower and aligner."""

    LAYER_PREFIX = "layers.{i}"

    WEIGHTS: tuple[W, ...] = (
        W("embed", shape=lambda d: (d.vocab, d.hidden), src="embed.weight", layers=None),
        W("final_norm", shape=lambda d: (d.hidden,), src="norm.weight", layers=None),
        # --- hyper-connections: the stage's layers (and the previous stage's last FFN mixes) ---
        *_hc_rows("attn", _stage),
        *_hc_rows("ffn", _hc_ffn_layers),
        W("attn_norm", shape=lambda d: (d.hidden,), src="{p}.attn_norm.weight", layers=_stage),
        W("ffn_norm", shape=lambda d: (d.hidden,), src="{p}.ffn_norm.weight", layers=_stage),
        # --- attention: the fused low-rank query + window latent, MXFP8 ---
        W(
            "wqkv_a",
            shape=lambda d: (d.q_lora + d.head_dim, d.hidden),
            dtype=torch.float8_e4m3fn,
            src=_fused(
                ("layers.{i}.attn.wq_a.weight", lambda d: slice(0, d.q_lora)),
                ("layers.{i}.attn.wkv.weight", lambda d: slice(d.q_lora, d.q_lora + d.head_dim)),
            ),
            layers=_stage,
        ),
        W(
            "wqkv_a_sf",
            shape=lambda d: (mx_sf_numel(d.q_lora + d.head_dim, d.hidden),),
            dtype=torch.uint8,
            src=_pair("layers.{i}.attn.wq_a.scale", "layers.{i}.attn.wkv.scale"),
            transform=_wqkv_a_sf,
            layers=_stage,
        ),
        W("q_norm", shape=lambda d: (d.q_lora,), src="{p}.attn.q_norm.weight", layers=_stage),
        W("kv_norm", shape=lambda d: (d.head_dim,), src="{p}.attn.kv_norm.weight", layers=_stage),
        W(
            "wq_b",
            shape=lambda d: (d.h_loc * d.head_dim, d.q_lora),
            dtype=torch.float8_e4m3fn,
            src=_rows("layers.{i}.attn.wq_b.weight", lambda d: (d.q_rows,)),
            layers=_stage,
        ),
        W(
            "wq_b_sf",
            shape=lambda d: (mx_sf_numel(d.h_loc * d.head_dim, d.q_lora),),
            dtype=torch.uint8,
            src=_rows("layers.{i}.attn.wq_b.scale", lambda d: (d.q_sf_rows,)),
            transform=_wq_b_sf,
            layers=_stage,
        ),
        W(
            "sink",
            shape=lambda d: (d.h_loc,),
            dtype=torch.float32,
            src=_rows("layers.{i}.attn.attn_sink", lambda d: (d.sink_rows,)),
            layers=_stage,
        ),
        W(
            "wo_a",
            shape=lambda d: (d.g_loc * d.o_lora, d.group_in),
            src=_wo_a_src,
            transform=_dequant_wo_a,
            layers=_stage,
        ),
        W(
            "wo_b",
            shape=lambda d: (d.hidden, d.g_loc * d.o_lora),
            dtype=torch.float8_e4m3fn,
            src=_rows("layers.{i}.attn.wo_b.weight", lambda d: (slice(None), d.o_rows)),
            layers=_stage,
        ),
        W(
            "wo_b_sf",
            shape=lambda d: (mx_sf_numel(d.hidden, d.g_loc * d.o_lora),),
            dtype=torch.uint8,
            src=_rows("layers.{i}.attn.wo_b.scale", lambda d: (slice(None), d.o_sf_rows)),
            transform=_hidden_sf,
            layers=_stage,
        ),
        # --- compressor and indexer K: the KV sources. A ratio-2 compressor is
        # [wkv ; wgate] promoted to fp32 (the reference's fp32 pooling path), a
        # ratio-1 one stays the checkpoint's bf16: two rows over disjoint layers. ---
        W(
            "comp_w",
            shape=lambda d: (2 * d.head_dim, d.hidden),
            dtype=torch.float32,
            src=_pair(
                "layers.{i}.attn.compressor.wkv.weight", "layers.{i}.attn.compressor.wgate.weight"
            ),
            transform=_cat_rows_f32,
            layers=_kv_sources_of_ratio(2),
        ),
        W(
            "comp_w",
            shape=lambda d: (d.head_dim, d.hidden),
            src="{p}.attn.compressor.wkv.weight",
            layers=_kv_sources_of_ratio(1),
        ),
        W(
            "comp_norm",
            shape=lambda d: (d.head_dim,),
            src="{p}.attn.compressor.norm.weight",
            layers=_kv_sources,
        ),
        W(
            "idx_wk",
            shape=lambda d: (IDX_DIM, d.head_dim),
            src="{p}.attn.indexer.wk.weight",
            layers=_kv_sources,
        ),
        W(
            "idx_k_norm",
            shape=lambda d: (IDX_DIM,),
            src="{p}.attn.indexer.k_norm.weight",
            layers=_kv_sources,
        ),
        # --- indexer query: the index sources ---
        W(
            "idx_wq_b",
            shape=lambda d: (IDX_HEADS * IDX_DIM, d.q_lora),
            dtype=torch.float8_e4m3fn,
            src="{p}.attn.indexer.wq_b.weight",
            layers=_index_sources,
        ),
        W(
            "idx_wq_b_sf",
            shape=lambda d: (mx_sf_numel(IDX_HEADS * IDX_DIM, d.q_lora),),
            dtype=torch.uint8,
            src="{p}.attn.indexer.wq_b.scale",
            transform=_idx_wq_b_sf,
            layers=_index_sources,
        ),
        W(
            "idx_wproj",
            shape=lambda d: (IDX_HEADS, d.hidden),
            src="{p}.attn.indexer.weights_proj.weight",
            transform=_idx_wproj,
            layers=_index_sources,
        ),
        # --- MoE router: the checkpoint's bf16 gate weight, its bias, and the
        # bias that selects experts for image-span tokens ---
        W(
            "gate_w",
            shape=lambda d: (d.num_experts, d.hidden),
            src="{p}.ffn.gate.weight",
            layers=_stage,
        ),
        W(
            "gate_bias",
            shape=lambda d: (d.num_experts,),
            dtype=torch.float32,
            src="{p}.ffn.gate.bias",
            layers=_stage,
        ),
        W(
            "gate_bias_vl",
            shape=lambda d: (d.num_experts,),
            dtype=torch.float32,
            src="{p}.ffn.gate.bias_vl",
            layers=_stage,
        ),
        # --- this rank's routed experts, NVFP4 ---
        W(
            "fc1_w",
            shape=lambda d: (d.e_loc, 2 * d.i_loc, d.hidden // 2),
            dtype=torch.uint8,
            src=_expert_window(("w3.weight", "w1.weight"), True),
            transform=_expert_fc1_w,
            layers=_stage,
        ),
        W(
            "fc1_s",
            shape=lambda d: (d.e_loc, 2 * d.i_loc, d.hidden // SF16),
            dtype=torch.uint8,
            src=_expert_window(("w3.weight_scale", "w1.weight_scale"), True),
            transform=_expert_fc1_s,
            layers=_stage,
        ),
        W(
            "fc2_w",
            shape=lambda d: (d.e_loc, d.hidden, d.i_loc // 2),
            dtype=torch.uint8,
            src=_expert_window(
                ("w2.weight",),
                False,
                cols=lambda d: (slice(None), slice(d.i_off // 2, (d.i_off + d.i_loc) // 2)),
            ),
            transform=_expert_fc2_w,
            layers=_stage,
        ),
        W(
            "fc2_s",
            shape=lambda d: (d.e_loc, d.hidden, d.i_loc // SF16),
            dtype=torch.uint8,
            src=_expert_window(
                ("w2.weight_scale",),
                False,
                cols=lambda d: (slice(None), slice(d.i_off // SF16, (d.i_off + d.i_loc) // SF16)),
            ),
            transform=_expert_fc2_s,
            layers=_stage,
        ),
        *_mlp_scalar_rows(),
        # --- the shared expert, MXFP8, [gate ; up] packed ---
        W(
            "sh_w13",
            shape=lambda d: (2 * d.moe_inter, d.hidden),
            dtype=torch.float8_e4m3fn,
            src=_fused(
                ("layers.{i}.ffn.shared_experts.w1.weight", lambda d: slice(0, d.moe_inter)),
                (
                    "layers.{i}.ffn.shared_experts.w3.weight",
                    lambda d: slice(d.moe_inter, 2 * d.moe_inter),
                ),
            ),
            layers=_stage,
        ),
        W(
            "sh_w13_sf",
            shape=lambda d: (mx_sf_numel(2 * d.moe_inter, d.hidden),),
            dtype=torch.uint8,
            src=_pair(
                "layers.{i}.ffn.shared_experts.w1.scale", "layers.{i}.ffn.shared_experts.w3.scale"
            ),
            transform=_sh_w13_sf,
            layers=_stage,
        ),
        W(
            "sh_w2",
            shape=lambda d: (d.hidden, d.moe_inter),
            dtype=torch.float8_e4m3fn,
            src="{p}.ffn.shared_experts.w2.weight",
            layers=_stage,
        ),
        W(
            "sh_w2_sf",
            shape=lambda d: (mx_sf_numel(d.hidden, d.moe_inter),),
            dtype=torch.uint8,
            src="{p}.ffn.shared_experts.w2.scale",
            transform=_hidden_sf,
            layers=_stage,
        ),
        # --- Engram: the tables as a device row shard (host_offload off), and
        # the projections every member holds ---
        *(
            row
            for li in range(_ENGRAM_SLOTS)
            for row in (
                W(
                    "eng_table",
                    shape=lambda d, li=li: (d.eng_part[li], d.eng_dim),
                    dtype=torch.float8_e4m3fn,
                    src=_engram_shard("weight"),
                    layers=_engram_device_slot(li),
                ),
                W(
                    "eng_scale",
                    shape=lambda d, li=li: (d.eng_part[li], d.eng_dim // MX),
                    dtype=torch.uint8,
                    src=_engram_shard("scale"),
                    transform=_u8,
                    layers=_engram_device_slot(li),
                ),
            )
        ),
        W(
            "eng_wkv",
            shape=lambda d: (d.eng_wkv_n, d.eng_in),
            dtype=torch.float8_e4m3fn,
            src="{p}.engram.wkv.weight",
            layers=_engram_layers,
        ),
        W(
            "eng_wkv_sf",
            shape=lambda d: (mx_sf_numel(d.eng_wkv_n, d.eng_in),),
            dtype=torch.uint8,
            src="{p}.engram.wkv.scale",
            transform=_eng_wkv_sf,
            layers=_engram_layers,
        ),
        W(
            "eng_q",
            shape=lambda d: (HC, d.hidden),
            src="{p}.engram.q_weight",
            layers=_engram_layers,
        ),
        W(
            "eng_k",
            shape=lambda d: (HC, d.hidden),
            src="{p}.engram.k_weight",
            layers=_engram_layers,
        ),
        # --- the vision tower, aligner and span delimiters ---
        *_vision_rows(),
    )

    def dims(self, core) -> SimpleNamespace:
        return _layout(core)

    def host_keys(self, d) -> list[str]:
        """Checkpoint keys of the Engram tables held in the node's shared host copy."""
        if not d.eng_host:
            return []
        return [
            f"layers.{i}.engram.embed.{name}"
            for i in _engram_layers(d)
            for name in ("weight", "scale")
        ]

    def expected_unconsumed(self, core, weights) -> set:
        """Exactly the keys this rank does not consume (module docstring)."""
        keys = set(weights.keys())
        out = {k for k in keys if k.startswith("mtp.")}
        prev_hc = {f"layers.{core.pp_prev}.hc_ffn_{f}" for f in ("fn", "base", "scale")}
        out |= {
            k
            for k in keys
            if (i := _layer_of(k)) is not None and i not in core.local and k not in prev_hc
        }
        if not core.last_pp:
            out.add("head.weight")
        if core.e_loc != core.num_experts:
            lo, hi = core.e_off, core.e_off + core.e_loc
            for i in core.pp_layer_ids:
                for e in range(core.num_experts):
                    if lo <= e < hi:
                        continue
                    for w_ in ("w1", "w2", "w3"):
                        out.add(f"layers.{i}.ffn.experts.{e}.{w_}.weight")
                        out.add(f"layers.{i}.ffn.experts.{e}.{w_}.weight_scale")
        return out

    def load(self, model, weights) -> None:
        """Overrides the base: sources are per-rank parts of checkpoint tensors read
        before their transform, the per-tensor scalars are 0-dim, the Engram tables
        may live in the node's shared host copy, a huge untransformed row streams in
        chunks, and the vocab-parallel ``lm_head`` is a slice."""
        core = model.model
        d = self.dims(core)
        manifest = self.manifest(core)
        consumed: set = set()
        ckpt_dir = getattr(weights, "checkpoint_dir", None)
        if ckpt_dir is not None:
            core._ckpt_dir = ckpt_dir

        def fill(param, ckpt, dst_index, transform) -> None:
            keys, srcs, multi = _split_source(ckpt)
            for key in keys:
                assert key in weights, f"checkpoint key missing: {key}"
            dst = param.data if dst_index is None else param.data[dst_index]
            if transform is None and len(keys) == 1:
                v = weights[keys[0]]
                s0 = srcs[0]
                shp = _shape(v)
                n_elems = 1
                for d_ in shp:
                    n_elems *= d_
                if len(shp) >= 1 and n_elems > _CHUNK_ELEMS and (s0 is None or len(s0) == 1):
                    # Stream a huge row range in chunks.
                    r0 = 0 if s0 is None or s0[0].start is None else s0[0].start
                    r1 = shp[0] if s0 is None or s0[0].stop is None else s0[0].stop
                    assert dst.shape[0] == r1 - r0, (keys, tuple(dst.shape), r0, r1)
                    row_elems = n_elems // shp[0]
                    step = max(1, _CHUNK_ELEMS // row_elems)
                    for a in range(r0, r1, step):
                        b = min(a + step, r1)
                        part = _materialize(v, (slice(a, b),))
                        assert part.dtype == dst.dtype, (keys, part.dtype, dst.dtype)
                        dst[a - r0 : b - r0].copy_(part.to(dst.device))
                    consumed.update(keys)
                    return
            src_all = tuple(
                _materialize(weights[k], s).to(dst.device, non_blocking=True)
                for k, s in zip(keys, srcs)
            )
            if transform is not None:
                out = transform(src_all if multi else src_all[0], core)
            else:
                assert len(src_all) == 1, "a multi-key row needs a transform"
                out = src_all[0]
            assert tuple(dst.shape) == tuple(out.shape), (keys, tuple(dst.shape), tuple(out.shape))
            assert out.dtype == dst.dtype, (keys, out.dtype, dst.dtype)
            dst.copy_(out, non_blocking=True)
            consumed.update(keys)

        assert set(manifest.keys()) == set(core.w.keys()), (
            "manifest/parameter drift",
            sorted(set(manifest.keys()) ^ set(core.w.keys()))[:8],
        )
        if core.eng_attach:
            # Collective over the node's ranks (local rank 0 fills the shared
            # file), and first: it fails fast if the node lacks the host memory.
            core.attach_engram_host_table(core._ckpt_dir)
        host = self.host_keys(d)
        if host:
            held = {t["key"]: tuple(t["shape"]) for t in core._eng_table.manifest["tensors"]}
            for key in host:
                assert key in weights, f"checkpoint key missing: {key}"
                assert held.get(key) == _shape(weights[key]), (
                    key,
                    held.get(key),
                    _shape(weights[key]),
                )
                consumed.add(key)
        for param_key, sources in manifest.items():
            for ckpt, dst_index, transform in sources:
                fill(core.w[param_key], ckpt, dst_index, transform)
            # The expert relayouts allocate per-expert scratch e_loc times per
            # layer and an Engram shard is tens of GB: release before the next
            # parameter so peak load memory stays one parameter deep.
            if param_key.endswith(_RELEASE_AFTER):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        # Shell-registered exception: lm_head. The shell builds it vocab-parallel
        # ([V/tp, H], rows of this TP rank) on the TP members and whole under
        # attention DP; asserted, not adapted. Outside the last pipeline stage the
        # engine skips the lm_head and drops its weight.
        if core.last_pp:
            lm = model.lm_head.weight
            if tuple(lm.shape) == (core.vocab, core.hidden):
                assert core.dp, "a replicated lm_head on a head-split member"
                fill(lm, "head.weight", None, None)
            else:
                part = core.vocab // core.tp
                assert tuple(lm.shape) == (part, core.hidden) and not core.dp, tuple(lm.shape)
                fill(
                    lm,
                    Slice("head.weight", (slice(core.tp_rank * part, (core.tp_rank + 1) * part),)),
                    None,
                    None,
                )
        shell = {
            n for n, _ in model.named_parameters() if not n.startswith(("model.w.", "model.w_ds."))
        }
        want = {"lm_head.weight"} if core.last_pp else set()
        assert shell == want, f"unfed shell parameters: {sorted(shell)}"

        torch.cuda.synchronize()
        leftover = set(weights.keys()) - consumed
        expected = self.expected_unconsumed(core, weights)
        assert leftover == expected, (
            "checkpoint coverage: leftover keys are not exactly the predicted non-load set; "
            f"unexpected {sorted(leftover - expected)[:8]}, missed {sorted(expected - leftover)[:8]}"
        )


def _layer_of(key: str):
    """The backbone layer of a ``layers.{i}.`` checkpoint key, else None."""
    if not key.startswith("layers."):
        return None
    return int(key.split(".", 2)[1])


MODEL_WEIGHTS = DeepseekV41UnifiedWeights()


# ---- the DSpark drafter (the ``load_draft_weights`` pass) --------------------


def _fc1_w_u8(t, core) -> torch.Tensor:
    """MXFP4 (w3 = up, w1 = gate) packed nibbles -> one expert's FC1 operand
    (the NVFP4 relayout is byte-generic)."""
    up, gate = t
    return _expert_fc1_w((up.view(torch.uint8), gate.view(torch.uint8)), core)


def _fc2_w_u8(t, core) -> torch.Tensor:
    return _expert_fc2_w(t.view(torch.uint8), core)


def _ds_wqkv_a_sf(t, core) -> torch.Tensor:
    return _mx_sf(t, (core.ds.q_lora, core.ds.head_dim))


def _ds_wq_b_sf(t, core) -> torch.Tensor:
    return _mx_sf(t, (core.ds.heads * core.ds.head_dim,))


def _ds_hidden_sf(t, core) -> torch.Tensor:
    return _mx_sf(t, (core.ds.hidden,))


def _ds_sh_w13_sf(t, core) -> torch.Tensor:
    return _mx_sf(t, (core.ds.inter, core.ds.inter))


def _ds_wkv3_sf(t, core) -> torch.Tensor:
    return _mx_sf(t, (core.ds.head_dim,) * core.ds.stages)


def _ds_experts(names, transform_pair: bool):
    """All of a stage's routed experts, replicated, one source per expert at ``(e,)``."""

    def build(d, s):
        out = []
        for e in range(d.experts):
            q = f"mtp.{s}.ffn.experts.{e}"
            src = tuple(f"{q}.{n}" for n in names) if transform_pair else f"{q}.{names[0]}"
            out.append((src, (e,)))
        return out

    return build


def _ds_hc_rows(sublayer):
    return (
        W(
            f"hc_{sublayer}_fn",
            shape=lambda d: (MIX, HC * d.hidden),
            dtype=torch.float32,
            src=f"{{p}}.hc_{sublayer}_fn",
        ),
        W(
            f"hc_{sublayer}_base",
            shape=lambda d: (MIX,),
            dtype=torch.float32,
            src=f"{{p}}.hc_{sublayer}_base",
        ),
        W(
            f"hc_{sublayer}_scale",
            shape=lambda d: (3,),
            dtype=torch.float32,
            src=f"{{p}}.hc_{sublayer}_scale",
        ),
    )


class DeepseekV41DSparkWeights(ModelWeights):
    """The three DSpark stages of deepseek-v4.1-flash-nvfp4, replicated on every rank.

    Layer ``s`` is stage ``mtp.{s}``; ``d.num_layers`` is the stage count.
    """

    LAYER_PREFIX = "mtp.{i}"

    WEIGHTS: tuple[W, ...] = (
        *_ds_hc_rows("attn"),
        *_ds_hc_rows("ffn"),
        W("attn_norm", shape=lambda d: (d.hidden,), src="{p}.attn_norm.weight"),
        W("ffn_norm", shape=lambda d: (d.hidden,), src="{p}.ffn_norm.weight"),
        W(
            "wqkv_a",
            shape=lambda d: (d.q_lora + d.head_dim, d.hidden),
            dtype=torch.float8_e4m3fn,
            src=(
                ("{p}.attn.wq_a.weight", lambda d: slice(0, d.q_lora)),
                ("{p}.attn.wkv.weight", lambda d: slice(d.q_lora, d.q_lora + d.head_dim)),
            ),
        ),
        W(
            "wqkv_a_sf",
            shape=lambda d: (mx_sf_numel(d.q_lora + d.head_dim, d.hidden),),
            dtype=torch.uint8,
            src=lambda d, s: [((f"mtp.{s}.attn.wq_a.scale", f"mtp.{s}.attn.wkv.scale"), None)],
            transform=_ds_wqkv_a_sf,
        ),
        W("q_norm", shape=lambda d: (d.q_lora,), src="{p}.attn.q_norm.weight"),
        W("kv_norm", shape=lambda d: (d.head_dim,), src="{p}.attn.kv_norm.weight"),
        W(
            "wq_b",
            shape=lambda d: (d.heads * d.head_dim, d.q_lora),
            dtype=torch.float8_e4m3fn,
            src="{p}.attn.wq_b.weight",
        ),
        W(
            "wq_b_sf",
            shape=lambda d: (mx_sf_numel(d.heads * d.head_dim, d.q_lora),),
            dtype=torch.uint8,
            src="{p}.attn.wq_b.scale",
            transform=_ds_wq_b_sf,
        ),
        W("sink", shape=lambda d: (d.heads,), dtype=torch.float32, src="{p}.attn.attn_sink"),
        W(
            "wo_a",
            shape=lambda d: (d.o_groups * d.o_lora, d.heads * d.head_dim // d.o_groups),
            src=lambda d, s: [((f"mtp.{s}.attn.wo_a.weight", f"mtp.{s}.attn.wo_a.scale"), None)],
            transform=_dequant_wo_a,
        ),
        W(
            "wo_b",
            shape=lambda d: (d.hidden, d.o_groups * d.o_lora),
            dtype=torch.float8_e4m3fn,
            src="{p}.attn.wo_b.weight",
        ),
        W(
            "wo_b_sf",
            shape=lambda d: (mx_sf_numel(d.hidden, d.o_groups * d.o_lora),),
            dtype=torch.uint8,
            src="{p}.attn.wo_b.scale",
            transform=_ds_hidden_sf,
        ),
        W(
            "gate_w",
            shape=lambda d: (d.experts, d.hidden),
            dtype=torch.float32,
            src="{p}.ffn.gate.weight",
            transform=_to_f32,
        ),
        W("gate_bias", shape=lambda d: (d.experts,), dtype=torch.float32, src="{p}.ffn.gate.bias"),
        W(
            "fc1_w",
            shape=lambda d: (d.experts, 2 * d.inter, d.hidden // 2),
            dtype=torch.uint8,
            src=_ds_experts(("w3.weight", "w1.weight"), True),
            transform=_fc1_w_u8,
        ),
        W(
            "fc1_s",
            shape=lambda d: (d.experts, 2 * d.inter, d.hidden // MX),
            dtype=torch.uint8,
            src=_ds_experts(("w3.scale", "w1.scale"), True),
            transform=_expert_fc1_s,
        ),
        W(
            "fc2_w",
            shape=lambda d: (d.experts, d.hidden, d.inter // 2),
            dtype=torch.uint8,
            src=_ds_experts(("w2.weight",), False),
            transform=_fc2_w_u8,
        ),
        W(
            "fc2_s",
            shape=lambda d: (d.experts, d.hidden, d.inter // MX),
            dtype=torch.uint8,
            src=_ds_experts(("w2.scale",), False),
            transform=_expert_fc2_s,
        ),
        W(
            "sh_w13",
            shape=lambda d: (2 * d.inter, d.hidden),
            dtype=torch.float8_e4m3fn,
            src=(
                ("{p}.ffn.shared_experts.w1.weight", lambda d: slice(0, d.inter)),
                ("{p}.ffn.shared_experts.w3.weight", lambda d: slice(d.inter, 2 * d.inter)),
            ),
        ),
        W(
            "sh_w13_sf",
            shape=lambda d: (mx_sf_numel(2 * d.inter, d.hidden),),
            dtype=torch.uint8,
            src=lambda d, s: [
                (
                    (
                        f"mtp.{s}.ffn.shared_experts.w1.scale",
                        f"mtp.{s}.ffn.shared_experts.w3.scale",
                    ),
                    None,
                )
            ],
            transform=_ds_sh_w13_sf,
        ),
        W(
            "sh_w2",
            shape=lambda d: (d.hidden, d.inter),
            dtype=torch.float8_e4m3fn,
            src="{p}.ffn.shared_experts.w2.weight",
        ),
        W(
            "sh_w2_sf",
            shape=lambda d: (mx_sf_numel(d.hidden, d.inter),),
            dtype=torch.uint8,
            src="{p}.ffn.shared_experts.w2.scale",
            transform=_ds_hidden_sf,
        ),
        # --- shared by the stages: stage 0's main projection, the stacked ring
        # write operand, and the last stage's norm and Markov head ---
        W(
            "main_proj",
            shape=lambda d: (d.hidden, d.targets * d.hidden),
            dtype=torch.float8_e4m3fn,
            src="mtp.0.main_proj.weight",
            layers=None,
        ),
        W(
            "main_proj_sf",
            shape=lambda d: (mx_sf_numel(d.hidden, d.targets * d.hidden),),
            dtype=torch.uint8,
            src="mtp.0.main_proj.scale",
            transform=_ds_hidden_sf,
            layers=None,
        ),
        W("main_norm", shape=lambda d: (d.hidden,), src="mtp.0.main_norm.weight", layers=None),
        W(
            "wkv3",
            shape=lambda d: (d.num_layers * d.head_dim, d.hidden),
            dtype=torch.float8_e4m3fn,
            src=lambda d, i: [
                (f"mtp.{s}.attn.wkv.weight", slice(s * d.head_dim, (s + 1) * d.head_dim))
                for s in range(d.num_layers)
            ],
            layers=None,
        ),
        W(
            "wkv3_sf",
            shape=lambda d: (mx_sf_numel(d.num_layers * d.head_dim, d.hidden),),
            dtype=torch.uint8,
            src=lambda d, i: [
                (tuple(f"mtp.{s}.attn.wkv.scale" for s in range(d.num_layers)), None)
            ],
            transform=_ds_wkv3_sf,
            layers=None,
        ),
        W(
            "norm",
            shape=lambda d: (d.hidden,),
            src=lambda d, i: [(f"mtp.{d.num_layers - 1}.norm.weight", None)],
            layers=None,
        ),
        W(
            "mk_embed",
            shape=lambda d: (d.vocab, d.markov_rank),
            src=lambda d, i: [(f"mtp.{d.num_layers - 1}.markov_head.embed.weight", None)],
            layers=None,
        ),
        W(
            "mk_head",
            shape=lambda d: (d.vocab, d.markov_rank),
            dtype=torch.float32,
            src=lambda d, i: [(f"mtp.{d.num_layers - 1}.markov_head.head.weight", None)],
            transform=_to_f32,
            layers=None,
        ),
    )

    def dims(self, core) -> SimpleNamespace:
        tc = _text_config(core)
        return SimpleNamespace(
            num_layers=int(tc.num_nextn_predict_layers),
            hidden=int(tc.hidden_size),
            vocab=int(tc.vocab_size),
            heads=int(tc.num_attention_heads),
            head_dim=int(tc.head_dim),
            q_lora=int(tc.q_lora_rank),
            o_groups=int(tc.o_groups),
            o_lora=int(tc.o_lora_rank),
            inter=int(tc.moe_intermediate_size),
            experts=int(tc.dspark_n_routed_experts),
            markov_rank=int(tc.dspark_markov_rank),
            targets=len(tc.dspark_target_layer_ids),
            dtype=core.model_config.torch_dtype,
        )

    def expected_unconsumed(self, core, weights) -> set:
        """``mtp.*`` keys the drafter does not consume (module docstring)."""
        mtp = [k for k in weights.keys() if k.startswith("mtp.")]
        out = {k for k in mtp if ".confidence_head." in k}
        out |= {k for k in mtp if k.endswith(".ffn.gate.bias_vl")}
        return out

    def load(self, model, weights) -> None:
        """Fill the core's ``w_ds`` from the second weight session; asserts ``mtp.*``
        coverage exactly."""
        core = model.model
        assert core.ds is not None, "load_draft on a core built without the DSpark drafter"
        manifest = self.manifest(core)
        assert set(manifest.keys()) == set(core.w_ds.keys()), (
            "draft manifest/parameter drift",
            sorted(set(manifest.keys()) ^ set(core.w_ds.keys()))[:8],
        )
        consumed: set = set()

        def fill(param, ckpt, dst_index, transform) -> None:
            keys, srcs, multi = _split_source(ckpt)
            for key in keys:
                assert key in weights, f"checkpoint key missing: {key}"
            dst = param.data if dst_index is None else param.data[dst_index]
            src_all = tuple(
                _materialize(weights[k], s).to(dst.device, non_blocking=True)
                for k, s in zip(keys, srcs)
            )
            if transform is not None:
                out = transform(src_all if multi else src_all[0], core)
            else:
                assert len(src_all) == 1, "a multi-key row needs a transform"
                out = src_all[0]
            assert tuple(dst.shape) == tuple(out.shape), (keys, tuple(dst.shape), tuple(out.shape))
            assert out.dtype == dst.dtype, (keys, out.dtype, dst.dtype)
            dst.copy_(out, non_blocking=True)
            consumed.update(keys)

        for param_key, sources in manifest.items():
            for ckpt, dst_index, transform in sources:
                fill(core.w_ds[param_key], ckpt, dst_index, transform)
            if param_key.endswith("_fc2_s"):
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        mtp = {k for k in weights.keys() if k.startswith("mtp.")}
        leftover = mtp - consumed
        expected = self.expected_unconsumed(core, weights)
        assert leftover == expected, (
            "mtp coverage: leftover keys are not exactly the predicted non-load set; "
            f"unexpected {sorted(leftover - expected)[:8]}, missed {sorted(expected - leftover)[:8]}"
        )


DRAFT_WEIGHTS = DeepseekV41DSparkWeights()
