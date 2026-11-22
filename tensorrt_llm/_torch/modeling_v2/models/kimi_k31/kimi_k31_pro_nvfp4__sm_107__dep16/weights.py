# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight table and loader: Kimi K3.1 Pro NVFP4 (ModelOpt) / sm_107 / dep16.

Every weight the target reads is one ``W`` row at this rank's share, and every row fills a
``core.w`` parameter in the layout and dtype its kernel reads: nothing is relaid after the load.

dep16 shares: attention DP over the 16 ranks. A rank holds the whole attention and MLP
weights; only the routed experts are split, EP16 (56 whole experts per rank). The LM head is
the shell's, whole.

``load`` asserts coverage both ways: every core parameter (the ``core.w`` table) is written
exactly once, the shell's only parameter is the LM head, and every checkpoint key is consumed or
an expected leftover (the vision tower and projector on a server that builds no tower, the other
ranks' experts).
"""

from __future__ import annotations

import glob
import json
import os
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Iterator, NamedTuple

import torch
from safetensors import safe_open
from torch import nn

from ..config import modelopt_nvfp4_mismatch

#: The LM head's checkpoint key (the shell's lm_head; load fills it).
CKPT_LM_HEAD = "language_model.lm_head.weight"
#: Checkpoint key prefixes of the vision tower and projector: ``w4_rows`` reads them on a server
#: that builds the tower (``Dims.vision``), and they are expected leftovers everywhere else.
CKPT_VISION = ("vision_tower.", "mm_projector.")

#: ``core.w`` keys of the flat drafter (``w3_rows``) start with this; the drafter's checkpoint
#: fills them.
DRAFT_KEY_PREFIX = "d_"

#: ``layers=ALL_LAYERS``: the weight is on every layer; ``None``: it is not per-layer.
ALL_LAYERS = object()

#: An index into a checkpoint tensor: one slice per leading axis.
Cut = tuple[slice, ...]


class Cuts(tuple):
    """One cut per key of a multi-key source, in key order (None: that key whole)."""


# Every split intermediate (expert, shared expert, dense MLP) lands on whole 128-row tiles.
_ROW_ALIGN = 128


class Dims(NamedTuple):
    """Every per-rank width the table and the forward read; ``dims`` computes it once.

    ``kda_tp`` / ``kda_tp_rank`` also split the MLA output heads (``g_proj``, ``o_proj``) and the
    dense and shared MLP intermediates: one TP group serves all of them (a group of one under
    attention DP).
    """

    num_layers: int
    hidden: int
    vocab: int
    dtype: torch.dtype  # the model dtype
    role: str  # the server role the core serves (``core.role``): "context" or "generation"
    kda_layers: tuple[int, ...]  # global indices of the KDA layers
    mla_layers: tuple[int, ...]  # global indices of the MLA (DSA) layers
    moe_layers: tuple[int, ...]  # global indices of the latent-MoE layers
    dense_layers: tuple[int, ...]  # global indices of the dense-MLP layers
    attn_res_block: int  # layers per attention-residual block
    dp_size: int  # attention DP ranks, 1 without attention DP
    kda_tp: int  # ways the KDA heads are split
    kda_tp_rank: int
    kda_heads: int  # this rank's KDA heads
    kda_head_dim: int
    kda_conv: int  # short-convolution width
    mla_heads: int  # query heads this rank's MLA attention computes
    mla_o_heads: int  # heads of the g_proj / o_proj share this rank holds
    q_lora: int
    kv_lora: int
    nope: int
    rope: int
    v_dim: int
    helix_cp: int  # helix context-parallel ranks, 1 without helix
    helix_cp_rank: int
    index_heads: int
    index_dim: int
    num_experts: int
    moe_tp: int
    moe_tp_rank: int
    moe_ep: int
    moe_ep_rank: int
    local_experts: int
    expert_offset: int
    latent: int  # the routed experts' hidden width
    moe_inter: int  # this rank's slice of an expert's intermediate
    shared_inter: int  # this rank's slice of the shared experts' intermediate
    dense_inter: int  # this rank's slice of the dense MLP's intermediate
    # True on a disaggregated context server (``core.role`` ``context``) with a drafter: its drafter
    # (``w3_rows``) holds and builds only the context KV. False everywhere else.
    draft_context_kv_only: bool = False
    # The lever whose weights the table lays out (modeling.py's ``_Levers``, on for a generation
    # server): each drafter layer's q_a stacked with its kv_a.
    draft_q_a_kv_a_fused: bool = False
    # Whether this server builds the vision tower and projector (``w4_rows``): the checkpoint's
    # composite config has a vision config and the server is not text-only (``disable_mm_encoder``).
    # Both roles of this layout can run context steps.
    vision: bool = False


@dataclass(frozen=True)
class W:
    """One weight role: where its bytes come from, this rank's share, and where it lands.

    ``name``: the ``core.w`` key, ``l{i}_<name>`` per layer and ``<name>`` otherwise (the views in
    modeling.py read these keys). ``shape(d)``: this rank's share. ``src``: a checkpoint key
    template (``{i}`` the layer), or ``(d, i) -> [(key or key tuple, index into the parameter or
    None), ...]``, a key tuple handing those tensors to the transform together; None for a
    parameter its owner derives after the load (nothing is read). ``dtype``: None for the model
    dtype. ``cut(d)``: this rank's rows or columns of every source, read from the lazy entry; a
    ``Cuts`` gives each key of a key tuple its own cut. ``transform(src, core)``: the kernels'
    relayout, after the cut, on the destination's device; it must also run on meta tensors (the
    coverage test checks shapes that way). ``layers``: ``ALL_LAYERS``, None, or ``d -> layer
    indices``.
    """

    name: str
    shape: Callable[[Dims], tuple[int, ...]]
    src: str | Callable | None
    dtype: torch.dtype | None = None
    cut: Callable[[Dims], Cut | Cuts] | None = None
    transform: Callable | None = None
    layers: object | None = ALL_LAYERS

    def layer_indices(self, d: Dims) -> list[int] | None:
        """The layers this weight is on, or None when it is not per-layer."""
        if self.layers is None:
            return None
        if self.layers is ALL_LAYERS:
            return list(range(d.num_layers))
        return list(self.layers(d))

    def sources(self, d: Dims, i: int | None) -> list[tuple[Any, Any]]:
        """``[(checkpoint key or key tuple, index into the parameter or None), ...]``."""
        if self.src is None:
            return []
        if callable(self.src):
            return list(self.src(d, i))
        return [(self.src.format(i=i), None)]


# --- sources and cuts the rows compose ---


def src_keys(*templates: str) -> Callable:
    """Several checkpoint keys (``{i}`` the layer) handed to the transform together."""
    return lambda d, i, ts=templates: [(tuple(t.format(i=i) for t in ts), None)]


def cut_rows(block: Callable[[Dims], tuple[int, int]]) -> Callable[[Dims], Cut]:
    """Rows ``[rank * span, (rank + 1) * span)``, with ``block(d) -> (rank, span)``."""

    def cut(d: Dims) -> Cut:
        rank, span = block(d)
        return (slice(rank * span, (rank + 1) * span),)

    return cut


def cut_cols(block: Callable[[Dims], tuple[int, int]]) -> Callable[[Dims], Cut]:
    """Columns ``[rank * span, (rank + 1) * span)`` of every row."""

    def cut(d: Dims) -> Cut:
        rank, span = block(d)
        return (slice(None), slice(rank * span, (rank + 1) * span))

    return cut


# --- dims ---


def _text_config(pretrained_config: Any) -> Any:
    """The K3.1 text config, whether the model config holds the composite or the text config."""
    return getattr(pretrained_config, "text_config", None) or pretrained_config


def _dims(
    cfg: Any,
    dtype: torch.dtype,
    *,
    role: str,
    dp_size: int,
    tp: int,
    tp_rank: int,
    mla_heads: int,
    helix_cp: int,
    helix_cp_rank: int,
    moe: Any,
) -> Dims:
    """Dims of one layout.

    ``tp`` / ``tp_rank`` split the KDA heads, the MLA output heads and the MLP intermediates;
    ``moe`` is the mapping whose MoE TP x EP splits the routed experts.
    """
    lin = cfg.linear_attn_config
    n = cfg.num_hidden_layers
    heads, kda_heads = cfg.num_attention_heads, lin["num_heads"]
    assert heads % tp == 0 and kda_heads % tp == 0, (heads, kda_heads, tp)
    if getattr(cfg, "moe_layer_freq", 1) != 1:
        raise ValueError(
            f"moe_layer_freq {getattr(cfg, 'moe_layer_freq', 1)}: every layer past the dense ones is MoE"
        )
    assert cfg.num_experts % moe.moe_ep_size == 0, (cfg.num_experts, moe.moe_ep_size)
    shared = cfg.moe_intermediate_size * cfg.num_shared_experts
    # Layer 0's MLP splits over the whole TP group, as in the DS target, not over a TP capped at
    # the GPUs per node; modeling.py all-reduces it over that group (dense_all_reduce).
    for what, full, parts in (
        ("expert", cfg.moe_intermediate_size, moe.moe_tp_size),
        ("shared expert", shared, tp),
        ("dense MLP", cfg.intermediate_size, tp),
    ):
        assert full % parts == 0 and (full // parts) % _ROW_ALIGN == 0, (
            f"the {what} intermediate {full} does not split {parts} ways on {_ROW_ALIGN}-row tiles"
        )
    local = cfg.num_experts // moe.moe_ep_size
    return Dims(
        num_layers=n,
        hidden=cfg.hidden_size,
        vocab=cfg.vocab_size,
        dtype=dtype,
        role=role,
        # linear_attn_config numbers layers from 1.
        kda_layers=tuple(j - 1 for j in lin["kda_layers"] if j <= n),
        mla_layers=tuple(j - 1 for j in lin["full_attn_layers"] if j <= n),
        moe_layers=tuple(range(cfg.first_k_dense_replace, n)),
        dense_layers=tuple(range(min(cfg.first_k_dense_replace, n))),
        attn_res_block=cfg.attn_res_block_size,
        dp_size=dp_size,
        kda_tp=tp,
        kda_tp_rank=tp_rank,
        kda_heads=kda_heads // tp,
        kda_head_dim=lin["head_dim"],
        kda_conv=lin["short_conv_kernel_size"],
        mla_heads=mla_heads,
        mla_o_heads=heads // tp,
        q_lora=cfg.q_lora_rank,
        kv_lora=cfg.kv_lora_rank,
        nope=cfg.qk_nope_head_dim,
        rope=cfg.qk_rope_head_dim,
        v_dim=cfg.v_head_dim,
        helix_cp=helix_cp,
        helix_cp_rank=helix_cp_rank,
        index_heads=cfg.dsa_config["index_num_heads"],
        index_dim=cfg.dsa_config["index_head_dim"],
        num_experts=cfg.num_experts,
        moe_tp=moe.moe_tp_size,
        moe_tp_rank=moe.moe_tp_rank,
        moe_ep=moe.moe_ep_size,
        moe_ep_rank=moe.moe_ep_rank,
        local_experts=local,
        expert_offset=local * moe.moe_ep_rank,
        latent=cfg.routed_expert_hidden_size,
        moe_inter=cfg.moe_intermediate_size // moe.moe_tp_size,
        shared_inter=shared // tp,
        dense_inter=cfg.intermediate_size // tp,
    )


def _draft_context_kv_only(model_config: Any, role: str) -> bool:
    """Whether the drafter keeps only the context KV (``context_kv_only``).

    True on a disaggregated context server, whose drafter never drafts and only writes the
    context history.
    """
    return getattr(model_config, "spec_config", None) is not None and role == "context"


def _vision(core: Any) -> bool:
    """Whether this server builds the vision tower (``Dims.vision``): the composite config's
    vision config reached the core (``core.vision_config``) and the server is not text-only
    (``disable_mm_encoder``, which a disaggregated generation server sets: it never encodes)."""
    text_only = bool(getattr(core.model_config, "disable_mm_encoder", False))
    return getattr(core, "vision_config", None) is not None and not text_only


def dims(core: Any) -> Dims:
    """Compute every per-rank width from the config, the engine's ``Mapping`` and ``core.role``.

    dep16 is attention DP: every rank holds whole attention and MLP weights, and the routed
    experts are EP-split over the DP ranks; a server that builds the vision tower holds it whole.
    """
    mc = core.model_config
    m = mc.mapping
    assert m.enable_attention_dp and m.cp_size == 1, "dep16 runs attention DP without CP"
    cfg = _text_config(mc.pretrained_config)
    levers = getattr(core, "levers", None)  # modeling.py's _Levers: the stacked rows' layout
    return _dims(
        cfg,
        mc.torch_dtype,
        role=core.role,
        dp_size=m.tp_size,
        tp=1,
        tp_rank=0,
        mla_heads=cfg.num_attention_heads,
        helix_cp=1,
        helix_cp_rank=0,
        moe=m,
    )._replace(
        draft_context_kv_only=_draft_context_kv_only(mc, core.role),
        draft_q_a_kv_a_fused=bool(getattr(levers, "draft_q_a_kv_a_fused", False)),
        vision=_vision(core),
    )


# --- the table and the passes over it ---


def table(d: Dims) -> tuple[W, ...]:
    """Every row of the target: the dense weights, the routed experts, the drafter and the vision
    tower (none on a server that builds no tower)."""
    return (*w1_rows(d), *w2_rows(d), *w3_rows(d), *w4_rows(d))


def _w_keys(row: W, d: Dims) -> list[tuple[str, int | None]]:
    """``[(core.w key, layer or None), ...]`` of a row."""
    layers = row.layer_indices(d)
    if layers is None:
        return [(row.name, None)]
    return [(f"l{i}_{row.name}", i) for i in layers]


def _destinations(row: W, d: Dims) -> list[tuple[str, int | None]]:
    """``[(parameter name under the core, layer or None), ...]`` of a row: ``w.<key>``.

    ``_w_keys`` decides the keys; ``declare``, ``manifest`` and ``_declared`` all derive from it.
    """
    return [(f"w.{key}", i) for key, i in _w_keys(row, d)]


def declare(d: Dims, rows: tuple[W, ...] | None = None) -> nn.ParameterDict:
    """Allocate every ``core.w`` parameter the table declares, at this rank's share.

    Meta-init intercepts ``torch.empty``; the engine materializes the storage.
    """
    w = nn.ParameterDict()
    for row in table(d) if rows is None else rows:
        for key, _ in _w_keys(row, d):
            assert key not in w, f"two rows declare core.w[{key!r}]"
            w[key] = nn.Parameter(
                torch.empty(*row.shape(d), dtype=row.dtype or d.dtype), requires_grad=False
            )
    return w


def manifest(
    core: Any, rows: tuple[W, ...] | None = None
) -> dict[str, list[tuple[Any, Any, Cut | None, Callable | None]]]:
    """Map each destination to its sources: ``(checkpoint key, index, cut, transform)``.

    A destination is named as ``core.named_parameters()`` names it: ``w.<key>``. A derived
    parameter maps to ``[]``.
    """
    d = core.dims
    out: dict[str, list[tuple[Any, Any, Cut | None, Callable | None]]] = {}
    for row in table(d) if rows is None else rows:
        cut = row.cut(d) if row.cut is not None else None
        for name, i in _destinations(row, d):
            assert name not in out, f"two rows fill {name}"
            out[name] = [(key, index, cut, row.transform) for key, index in row.sources(d, i)]
    return out


def _read_entry(entry: Any, cut: Cut | None) -> torch.Tensor:
    """Realize one checkpoint entry (a lazy safetensors slice or a tensor), reading only ``cut``.

    Uncut, ``[:]`` reads it whole; a 0-dim entry (the NVFP4 per-tensor scales) takes ``[()]``.
    """
    if cut is not None:
        return entry[cut[0]] if len(cut) == 1 else entry[cut]
    shape = entry.get_shape() if hasattr(entry, "get_shape") else entry.shape
    return entry[()] if len(shape) == 0 else entry[:]


def read(
    weights: Any,
    ckpt_key: Any,
    cut: Cut | Cuts | None,
    transform: Callable | None,
    core: Any,
    device: Any = None,
) -> torch.Tensor:
    """Read one source through its cut, move it to ``device`` and relay it.

    Every key of a multi-key source is read through the same cut, or through its own with a
    ``Cuts``, and handed to the transform together. The one path ``load`` takes, so a test drives
    it directly.
    """
    keys = ckpt_key if isinstance(ckpt_key, tuple) else (ckpt_key,)
    for key in keys:
        if key not in weights:
            raise KeyError(f"checkpoint key missing: {key}")
    cuts = cut if isinstance(cut, Cuts) else (cut,) * len(keys)
    assert len(cuts) == len(keys), f"{ckpt_key}: {len(cuts)} cuts for {len(keys)} keys"
    src = tuple(_read_entry(weights[key], c) for key, c in zip(keys, cuts))
    if device is not None:
        src = tuple(t.to(device, non_blocking=True) for t in src)
    if transform is not None:
        return transform(src if isinstance(ckpt_key, tuple) else src[0], core)
    assert len(src) == 1, f"{ckpt_key}: a multi-key source needs a transform"
    return src[0]


def _check_slots(name: str, param: torch.Tensor, sources: list) -> None:
    """Check an indexed parameter's slots.

    Every source indexes it, no slot is filled twice, and first-axis slots cover it whole.
    """
    indices = [index for _, index, _, _ in sources]
    if all(index is None for index in indices):
        assert len(sources) <= 1, f"{name}: {len(sources)} unindexed sources"
        return
    assert None not in indices, f"{name}: indexed and whole sources mixed"
    assert len(set(indices)) == len(indices), f"{name}: a slot is filled twice"
    if all(len(ix) == 1 and isinstance(ix[0], int) for ix in indices):
        got = sorted(ix[0] for ix in indices)
        assert got == list(range(param.shape[0])), f"{name}: slots {got[:4]}... of {param.shape[0]}"


def _declared(core: Any, rows: tuple[W, ...]) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Each destination's ``(shape, dtype)`` as its row declares it."""
    d = core.dims
    return {
        name: (tuple(row.shape(d)), row.dtype or d.dtype)
        for row in rows
        for name, _ in _destinations(row, d)
    }


def fill_table(
    core: Any, weights: Any, rows: tuple[W, ...], params: dict[str, torch.Tensor]
) -> set[str]:
    """Fill ``params`` (name under the core -> parameter) through ``rows``; return the keys read.

    Asserts the rows fill exactly ``params``, each once (a derived parameter is named but not
    read), every parameter at its row's shape and dtype, every indexed parameter whole, and every
    source to its slot's shape and dtype.
    """
    mf = manifest(core, rows)
    unfed, unknown = sorted(set(params) - set(mf)), sorted(set(mf) - set(params))
    assert not unfed and not unknown, (
        f"parameters no row fills: {unfed[:8]}; rows naming no parameter: {unknown[:8]}"
    )
    declared = _declared(core, rows)
    consumed: set[str] = set()
    for name, sources in mf.items():
        param = params[name].data
        shape, dtype = declared[name]
        assert tuple(param.shape) == shape and param.dtype == dtype, (
            f"{name}: held as {tuple(param.shape)} {param.dtype}, declared {shape} {dtype}"
        )
        _check_slots(name, param, sources)
        for ckpt_key, index, cut, transform in sources:
            dst = param if index is None else param[index]
            src = read(weights, ckpt_key, cut, transform, core, dst.device)
            assert dst.shape == src.shape, (name, ckpt_key, tuple(dst.shape), tuple(src.shape))
            assert dst.dtype == src.dtype, (name, ckpt_key, dst.dtype, src.dtype)
            dst.copy_(src, non_blocking=True)
            consumed.update(ckpt_key if isinstance(ckpt_key, tuple) else (ckpt_key,))
    return consumed


def lm_head_share(rows: int, vocab: int, tp_rank: int) -> tuple[Cut, Callable]:
    """The cut and the pad that fill an LM head the shell built with ``rows`` rows.

    Whole when ``rows == vocab``; a vocab-parallel shard otherwise. A vocab that does not split
    evenly leaves the last shard short; its missing rows are zero, as the built-in LMHead pads.
    """
    start = 0 if rows == vocab else tp_rank * rows
    stop = min(start + rows, vocab)
    assert start < stop, (rows, vocab, tp_rank)

    def pad(t: torch.Tensor, core: Any) -> torch.Tensor:
        if t.shape[0] == rows:
            return t.contiguous()
        out = torch.zeros(rows, t.shape[1], dtype=t.dtype, device=t.device)
        out[: t.shape[0]] = t
        return out

    return (slice(start, stop),), pad


def expected_unconsumed(core: Any, weights: Any) -> set[str]:
    """The keys this rank's load leaves: the vision tower and projector on a server that builds no
    tower (``w4_rows`` reads them otherwise), the other ranks' experts."""
    vision = (
        set()
        if core.dims.vision
        else {key for key in weights.keys() if key.startswith(CKPT_VISION)}
    )
    return vision | w2_offwindow_keys(core, weights)


def check_checkpoint(model_config: Any) -> None:
    """Raise unless the checkpoint is the ModelOpt NVFP4 export the table reads as stored."""
    reason = modelopt_nvfp4_mismatch(model_config)
    if reason is not None:
        raise ValueError(f"the weight table reads the ModelOpt NVFP4 export as stored: {reason}")


def load(model: Any, weights: Any) -> None:
    """Fill every destination from the checkpoint; assert coverage both ways.

    ``model`` is the shell (its core is ``model.model``); the drafter's parameters (``core.w``
    keys ``d_*``) are the rows of ``w3_rows``, from its own checkpoint; the drafter holds no shell
    parameter. The vision tower's (``w4_rows``) come from this checkpoint.
    """
    core = model.model
    d = core.dims
    check_checkpoint(core.model_config)
    params = {
        name: p
        for name, p in core.named_parameters()
        if not name.startswith(f"w.{DRAFT_KEY_PREFIX}")
    }
    consumed = fill_table(core, weights, (*w1_rows(d), *w2_rows(d), *w4_rows(d)), params)

    # The shell's LM head: vocab-parallel over the TP group, or whole; read its rows only.
    head = model.lm_head.weight
    cut, pad = lm_head_share(head.shape[0], d.vocab, d.kda_tp_rank)
    src = read(weights, CKPT_LM_HEAD, cut, pad, core, head.device)
    assert src.shape == head.shape and src.dtype == head.dtype, (src.shape, head.shape)
    head.data.copy_(src, non_blocking=True)
    consumed.add(CKPT_LM_HEAD)
    shell = {name for name, _ in model.named_parameters() if not name.startswith("model.")}
    assert shell == {"lm_head.weight"}, f"shell parameters no row fills: {sorted(shell)}"

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    leftover = set(weights.keys()) - consumed
    expected = expected_unconsumed(core, weights)
    assert leftover == expected, (
        "checkpoint coverage: unexpected leftovers "
        f"{sorted(leftover - expected)[:8]}, expected but read {sorted(expected - leftover)[:8]}"
    )


class LazyCheckpoint(dict):
    """Checkpoint key -> lazy safetensors slice; ``checkpoint_dir`` names where it was opened."""

    checkpoint_dir: str = ""


@contextmanager
def open_checkpoint_weights(checkpoint_dir: str) -> Iterator[LazyCheckpoint]:
    """Open the checkpoint's safetensors lazily: key -> slice, read on demand through a cut.

    The files stay mapped until the load ends.
    """
    files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors files in {checkpoint_dir}")
    with ExitStack() as stack:
        weights = LazyCheckpoint()
        weights.checkpoint_dir = checkpoint_dir
        for path in files:
            handle = stack.enter_context(safe_open(path, framework="pt", device="cpu"))
            for key in handle.keys():
                assert key not in weights, f"{key} is in two files"
                weights[key] = handle.get_slice(key)
        try:
            yield weights
        finally:
            weights.clear()


# --------------------------------------------------------------------------------------------------
# ``w1_rows``: the dense weights, every ``core.w`` parameter modeling.py's views read, and the
# indexer.
#
# Parameter names are the views' (``l{i}_kda_*``, ``l{i}_mla_*``, ``l{i}_moe_*``, ``l{i}_dense_*``,
# ``l{i}_<norm>``; ``embed``, ``out_res_proj``, ``out_res_norm``, ``final_norm``); keys are the
# checkpoint's, under ``language_model.model.``.
#
# Shares: none; under attention DP every rank holds these weights whole.
#
# Layouts:
#
# * The six KDA input projections load as one GEMM operand, ``kda_qkvgfab``: the rows of q, k, v, g
#   (this rank's heads), f_a (whole) and b (this rank's heads), then zero rows to a multiple of 16.
#   Its scales are the swizzles of qkvg, f_a and b, stacked: qkvg and f_a are whole 128-row tiles,
#   and b's tile holds b's rows, then zeros, so tile t of the stack scales rows [128t, 128t + 128)
#   of the weight. Row stacks never cross a 1x32 group, so each projection's bytes are the stored
#   ones.
# * kv_b is regrouped to ``[every head's k_nope ; every head's v]`` (as the DS target does), so
#   k_b and v_b are views of it.
# * The FP32 conv and o_norm round once to the model dtype, as the engine's KDA does; A_log, dt_bias
#   and the router bias stay FP32.
#
# Not here: ``language_model.lm_head.weight`` is the shell's (``load`` fills it, as the DS target's
# ``lm_head_share`` does); the shared and routed experts are in ``w2_rows``; the vision tower and
# projector are leftovers.
# --------------------------------------------------------------------------------------------------

_W1_LAYER = "language_model.model.layers.{i}."
_W1_MODEL = "language_model.model."

# K3.1's full widths of the weights split below (the core asserts the config).
_W1_MLA_HEADS = 96
_W1_DENSE_INTER = 33792
_W1_MX_BLOCK = 32  # MXFP8: one UE8M0 scale per 32 values along K
# The fused KDA projection's rows are padded to a multiple of this.
_W1_KDA_FAB_ROW_ALIGN = 16
# The server roles whose router loads in FP32 (router_bf16 off).
_W1_ROUTER_FP32_ROLES: tuple[str, ...] = ("generation",)
# The KDA input projections, in the fused operand's row order.
_W1_KDA_QKVG = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.g_proj")
_W1_KDA_FA = "self_attn.f_a_proj"
_W1_KDA_B = "self_attn.b_proj"


def _w1_sf_shape(rows: int, k: int) -> tuple[int, int]:
    """Shape of a swizzled ``[rows, k / 32]`` scale: rows padded to 128, columns to 4."""
    return -(-rows // 128) * 128, -(-(k // _W1_MX_BLOCK) // 4) * 4


def _w1_sf_flat(rows: int, k: int) -> tuple[int]:
    """Shape of a swizzled ``[rows, k / 32]`` scale as the MXFP8 linear takes it: flat."""
    mp, cp = _w1_sf_shape(rows, k)
    return (mp * cp,)


# --- shares: (index, span) along one axis of a checkpoint tensor ---


def _w1_rank(d: Dims) -> int:
    """This rank's index in the attention TP group."""
    assert d.helix_cp == 1 or d.kda_tp_rank == d.helix_cp_rank, (d.kda_tp_rank, d.helix_cp_rank)
    return d.kda_tp_rank


def _w1_block(d: Dims, full: int, share: int) -> tuple[int, int]:
    """This rank's block of ``share`` out of ``full``; a share wider than full / group repeats."""
    assert share > 0 and full % share == 0, (full, share)
    return _w1_rank(d) % (full // share), share


def _w1_kda_heads(d: Dims) -> tuple[int, int]:
    """One entry per KDA head: b_proj rows, A_log."""
    return _w1_block(d, d.kda_heads * d.kda_tp, d.kda_heads)


def _w1_kda_channels(d: Dims) -> tuple[int, int]:
    """One entry per channel of this rank's KDA heads (P = heads x 128)."""
    index, heads = _w1_kda_heads(d)
    return index, heads * d.kda_head_dim


def _w1_kda_sf(d: Dims) -> tuple[int, int]:
    """o_proj scale columns of those channels."""
    index, span = _w1_kda_channels(d)
    return index, span // _W1_MX_BLOCK


def _w1_q_heads(d: Dims) -> tuple[int, int]:
    """q_b rows of the query heads this rank computes."""
    index, heads = _w1_block(d, _W1_MLA_HEADS, d.mla_heads)
    return index, heads * (d.nope + d.rope)


def _w1_kv_heads(d: Dims) -> tuple[int, int]:
    """kv_b rows of the same heads, ``[k_nope | v]`` per head."""
    index, heads = _w1_block(d, _W1_MLA_HEADS, d.mla_heads)
    return index, heads * (d.nope + d.v_dim)


def _w1_o_heads(d: Dims) -> tuple[int, int]:
    """g_proj rows / o_proj columns of the heads this rank projects."""
    index, heads = _w1_block(d, _W1_MLA_HEADS, d.mla_o_heads)
    return index, heads * d.v_dim


def _w1_dense(d: Dims) -> tuple[int, int]:
    """This rank's slice of layer 0's MLP intermediate."""
    return _w1_block(d, _W1_DENSE_INTER, d.dense_inter)


def _w1_rows(block: Callable[[Dims], tuple[int, int]]) -> Callable[[Dims], Cut]:
    """Cut: rows ``[index * span, (index + 1) * span)``."""

    def cut(d: Dims) -> Cut:
        index, span = block(d)
        return (slice(index * span, (index + 1) * span),)

    return cut


def _w1_cols(block: Callable[[Dims], tuple[int, int]]) -> Callable[[Dims], Cut]:
    """Cut: every row, columns ``[index * span, (index + 1) * span)``."""

    def cut(d: Dims) -> Cut:
        index, span = block(d)
        return (slice(None), slice(index * span, (index + 1) * span))

    return cut


# --- sources, layers, per-rank widths ---


def _w1_one(suffix: str) -> Callable:
    """Layer ``i``'s key ``suffix``."""
    return lambda d, i: [(_W1_LAYER.format(i=i) + suffix, None)]


def _w1_stack(*suffixes: str) -> Callable:
    """Layer ``i``'s keys, handed to the transform together, in this order."""
    return lambda d, i: [(tuple(_W1_LAYER.format(i=i) + s for s in suffixes), None)]


def _w1_kda_layers(d: Dims) -> tuple[int, ...]:
    return d.kda_layers


def _w1_mla_layers(d: Dims) -> tuple[int, ...]:
    return d.mla_layers


def _w1_moe_layers(d: Dims) -> tuple[int, ...]:
    return d.moe_layers


def _w1_dense_layers(d: Dims) -> tuple[int, ...]:
    return d.dense_layers


def _w1_p(d: Dims) -> int:
    """This rank's KDA channels: heads x head dim."""
    return d.kda_heads * d.kda_head_dim


def _w1_hidden(d: Dims) -> int:
    return d.hidden


# --- relayouts, on the rank's cut ---


def _w1_swizzle(x: torch.Tensor) -> torch.Tensor:
    """128x4 swizzle of a ``[rows, k / 32]`` UE8M0 scale matrix, zero-padded to ``_w1_sf_shape``.

    128-row x 4-column tiles in 32 x 4 x 4 atom order: the bytes of the engine's
    ``block_scale_interleave``, the K32 R128c4 layout of the dense block-scaled GEMM.
    """
    m, c = x.shape
    mp, cp = -(-m // 128) * 128, -(-c // 4) * 4
    if (mp, cp) != (m, c):
        x = torch.nn.functional.pad(x, (0, cp - c, 0, mp - m))
    return x.reshape(mp // 128, 4, 32, cp // 4, 4).permute(0, 3, 2, 1, 4).reshape(mp, cp)


def _w1_cat(t: tuple[torch.Tensor, ...], core: Any) -> torch.Tensor:
    """The sources' rows in order, bitwise (FP8 joined as bytes)."""
    if t[0].dtype == torch.float8_e4m3fn:
        return torch.cat([x.view(torch.uint8) for x in t]).view(torch.float8_e4m3fn)
    return torch.cat(t).contiguous()


def _w1_mx_scale(t: torch.Tensor, core: Any) -> torch.Tensor:
    """One MXFP8 scale matrix, swizzled, flat."""
    return _w1_swizzle(t.view(torch.uint8)).reshape(-1)


def _w1_mx_scale_stack(t: tuple[torch.Tensor, ...], core: Any) -> torch.Tensor:
    """Stacked MXFP8 scale rows, one swizzle, flat."""
    return _w1_swizzle(torch.cat([x.view(torch.uint8) for x in t])).reshape(-1)


def _w1_kda_fab_rows(d: Dims) -> int:
    """Rows of the fused KDA projection: q, k, v, g, f_a, b, then zero rows to a multiple of 16."""
    rows = 4 * _w1_p(d) + d.kda_head_dim + d.kda_heads
    return -(-rows // _W1_KDA_FAB_ROW_ALIGN) * _W1_KDA_FAB_ROW_ALIGN


def _w1_kda_fab_sf_rows(d: Dims) -> int:
    """Scale rows of the fused KDA projection: the three swizzled stacks' tiles, back to back."""
    qkvg, fa, b = 4 * _w1_p(d), d.kda_head_dim, d.kda_heads
    assert qkvg % 128 == 0 and fa == 128 and b <= 128, (qkvg, fa, b)
    rows = qkvg + fa + 128
    assert rows == _w1_sf_shape(_w1_kda_fab_rows(d), d.hidden)[0], (rows, _w1_kda_fab_rows(d))
    return rows


def _w1_kda_fab_cuts(d: Dims) -> Cuts:
    """This rank's rows of q, k, v, g (its channels), f_a (whole) and b (its heads)."""
    index, span = _w1_kda_channels(d)
    channels = (slice(index * span, (index + 1) * span),)
    index, span = _w1_kda_heads(d)
    heads = (slice(index * span, (index + 1) * span),)
    return Cuts((channels,) * len(_W1_KDA_QKVG) + (None, heads))


def _w1_kda_fab_weight(d: Dims) -> Callable:
    """The six projections' E4M3 rows in order, then zero rows (as bytes: bitwise)."""

    def stack(t: tuple[torch.Tensor, ...], core: Any) -> torch.Tensor:
        rows = [x.view(torch.uint8) for x in t]
        pad = _w1_kda_fab_rows(d) - sum(x.shape[0] for x in rows)
        rows.append(rows[-1].new_zeros(pad, rows[-1].shape[1]))
        return torch.cat(rows).view(torch.float8_e4m3fn)

    return stack


def _w1_kda_fab_scale(t: tuple[torch.Tensor, ...], core: Any) -> torch.Tensor:
    """The swizzles of the qkvg stack, f_a and b, back to back, flat (whole 128-row tiles each)."""
    qkvg = torch.cat([x.view(torch.uint8) for x in t[:4]])
    return torch.cat(
        [
            _w1_swizzle(qkvg),
            _w1_swizzle(t[4].view(torch.uint8)),
            _w1_swizzle(t[5].view(torch.uint8)),
        ]
    ).reshape(-1)


def _w1_router_dtype(d: Dims) -> torch.dtype | None:
    """FP32 on the roles whose router runs FP32 logits from an FP32 weight; else the model's."""
    return torch.float32 if d.role in _W1_ROUTER_FP32_ROLES else None


def _w1_cast(dtype: torch.dtype) -> Callable:
    """Round the source to ``dtype`` once."""
    return lambda t, core: t.to(dtype)


def _w1_conv(dtype: torch.dtype) -> Callable:
    """Stack the q / k / v ``[P, 1, width]`` weights as ``[3P, width]`` in ``dtype``."""
    return lambda t, core: torch.cat([x.squeeze(1) for x in t]).to(dtype)


def _w1_kv_b_regroup(d: Dims) -> Callable:
    """Per-head ``[k_nope | v]`` rows -> ``[every head's k_nope ; every head's v]``."""

    def regroup(t: torch.Tensor, core: Any) -> torch.Tensor:
        t3 = t.reshape(-1, d.nope + d.v_dim, d.kv_lora)
        k = t3[:, : d.nope].reshape(-1, d.kv_lora)
        return torch.cat([k, t3[:, d.nope :].reshape(-1, d.kv_lora)]).contiguous()

    return regroup


def _w1_wk_wp(t: tuple[torch.Tensor, torch.Tensor], core: Any) -> torch.Tensor:
    """The indexer's wk rows, then its weights_proj rows, in FP32 (an exact cast)."""
    return torch.cat(t).to(torch.float32)


def _w1_mxfp8(
    name: str,
    *,
    rows: Callable[[Dims], int],
    k: Callable[[Dims], int],
    src: tuple[str, ...],
    cut_w: Callable[[Dims], Cut] | None,
    cut_s: Callable[[Dims], Cut] | None,
    layers: Callable[[Dims], tuple[int, ...]],
) -> tuple[W, W]:
    """The ``{name}_w`` / ``{name}_s`` rows of one MXFP8 weight; several sources are stacked."""
    if len(src) == 1:
        w_src, s_src = _w1_one(f"{src[0]}.weight"), _w1_one(f"{src[0]}.weight_scale")
        w_fix, s_fix = None, _w1_mx_scale
    else:
        w_src = _w1_stack(*(f"{s}.weight" for s in src))
        s_src = _w1_stack(*(f"{s}.weight_scale" for s in src))
        w_fix, s_fix = _w1_cat, _w1_mx_scale_stack
    return (
        W(
            f"{name}_w",
            shape=lambda d: (rows(d), k(d)),
            src=w_src,
            dtype=torch.float8_e4m3fn,
            cut=cut_w,
            transform=w_fix,
            layers=layers,
        ),
        W(
            f"{name}_s",
            shape=lambda d: _w1_sf_flat(rows(d), k(d)),
            src=s_src,
            dtype=torch.uint8,
            cut=cut_s,
            transform=s_fix,
            layers=layers,
        ),
    )


def w1_rows(d: Dims) -> tuple[W, ...]:
    """Return the dense rows at this rank's share."""
    kda, mla, moe = _w1_kda_layers, _w1_mla_layers, _w1_moe_layers
    fp32 = torch.float32
    kda_rows = _w1_rows(_w1_kda_channels)
    kda_head_rows = _w1_rows(_w1_kda_heads)
    fab_src = (*_W1_KDA_QKVG, _W1_KDA_FA, _W1_KDA_B)
    return (
        # --- KDA layers ---
        W(
            "kda_qkvgfab_w",
            shape=lambda d: (_w1_kda_fab_rows(d), d.hidden),
            src=_w1_stack(*(f"{s}.weight" for s in fab_src)),
            dtype=torch.float8_e4m3fn,
            cut=_w1_kda_fab_cuts,
            transform=_w1_kda_fab_weight(d),
            layers=kda,
        ),
        W(
            "kda_qkvgfab_s",
            shape=lambda d: (_w1_kda_fab_sf_rows(d) * _w1_sf_shape(1, d.hidden)[1],),
            src=_w1_stack(*(f"{s}.weight_scale" for s in fab_src)),
            dtype=torch.uint8,
            cut=_w1_kda_fab_cuts,
            transform=_w1_kda_fab_scale,
            layers=kda,
        ),
        *_w1_mxfp8(
            "kda_o",
            rows=_w1_hidden,
            k=_w1_p,
            src=("self_attn.o_proj",),
            cut_w=_w1_cols(_w1_kda_channels),
            cut_s=_w1_cols(_w1_kda_sf),
            layers=kda,
        ),
        W(
            "kda_fb",
            shape=lambda d: (_w1_p(d), d.kda_head_dim),
            src=_w1_one("self_attn.f_b_proj.weight"),
            cut=kda_rows,
            layers=kda,
        ),
        W(
            "kda_conv",
            shape=lambda d: (3 * _w1_p(d), d.kda_conv),
            src=_w1_stack(
                "self_attn.q_conv1d.weight",
                "self_attn.k_conv1d.weight",
                "self_attn.v_conv1d.weight",
            ),
            cut=kda_rows,
            transform=_w1_conv(d.dtype),
            layers=kda,
        ),
        # The checkpoint pads A_log to 128 entries; the cut never reaches the zero tail.
        W(
            "kda_a_log",
            shape=lambda d: (d.kda_heads,),
            src=_w1_one("self_attn.A_log"),
            dtype=fp32,
            cut=kda_head_rows,
            layers=kda,
        ),
        W(
            "kda_dt_bias",
            shape=lambda d: (_w1_p(d),),
            src=_w1_one("self_attn.dt_bias"),
            dtype=fp32,
            cut=kda_rows,
            layers=kda,
        ),
        W(
            "kda_o_norm",
            shape=lambda d: (d.kda_head_dim,),
            src=_w1_one("self_attn.o_norm.weight"),
            transform=_w1_cast(d.dtype),
            layers=kda,
        ),
        # --- MLA layers ---
        W(
            "mla_qkv_a",
            shape=lambda d: (d.q_lora + d.kv_lora + d.rope, d.hidden),
            src=_w1_stack("self_attn.q_a_proj.weight", "self_attn.kv_a_proj_with_mqa.weight"),
            transform=_w1_cat,
            layers=mla,
        ),
        W(
            "mla_q_a_norm",
            shape=lambda d: (d.q_lora,),
            src=_w1_one("self_attn.q_a_layernorm.weight"),
            layers=mla,
        ),
        W(
            "mla_kv_a_norm",
            shape=lambda d: (d.kv_lora,),
            src=_w1_one("self_attn.kv_a_layernorm.weight"),
            layers=mla,
        ),
        W(
            "mla_q_b",
            shape=lambda d: (d.mla_heads * (d.nope + d.rope), d.q_lora),
            src=_w1_one("self_attn.q_b_proj.weight"),
            cut=_w1_rows(_w1_q_heads),
            layers=mla,
        ),
        W(
            "mla_kv_b",
            shape=lambda d: (d.mla_heads * (d.nope + d.v_dim), d.kv_lora),
            src=_w1_one("self_attn.kv_b_proj.weight"),
            cut=_w1_rows(_w1_kv_heads),
            transform=_w1_kv_b_regroup(d),
            layers=mla,
        ),
        W(
            "mla_g",
            shape=lambda d: (d.mla_o_heads * d.v_dim, d.hidden),
            src=_w1_one("self_attn.g_proj.weight"),
            cut=_w1_rows(_w1_o_heads),
            layers=mla,
        ),
        W(
            "mla_o",
            shape=lambda d: (d.hidden, d.mla_o_heads * d.v_dim),
            src=_w1_one("self_attn.o_proj.weight"),
            cut=_w1_cols(_w1_o_heads),
            layers=mla,
        ),
        # --- MLA layers: the DSA indexer ---
        W(
            "mla_idx_wq_b",
            shape=lambda d: (d.index_heads * d.index_dim, d.q_lora),
            src=_w1_one("self_attn.indexer.wq_b.weight"),
            layers=mla,
        ),
        W(
            "mla_idx_wk_wp",
            shape=lambda d: (d.index_dim + d.index_heads, d.hidden),
            src=_w1_stack("self_attn.indexer.wk.weight", "self_attn.indexer.weights_proj.weight"),
            dtype=fp32,
            transform=_w1_wk_wp,
            layers=mla,
        ),
        W(
            "mla_idx_k_norm_w",
            shape=lambda d: (d.index_dim,),
            src=_w1_one("self_attn.indexer.k_norm.weight"),
            dtype=fp32,
            transform=_w1_cast(fp32),
            layers=mla,
        ),
        W(
            "mla_idx_k_norm_b",
            shape=lambda d: (d.index_dim,),
            src=_w1_one("self_attn.indexer.k_norm.bias"),
            dtype=fp32,
            transform=_w1_cast(fp32),
            layers=mla,
        ),
        # --- MoE layers: router and latent projections (the experts are in ``w2_rows``) ---
        W(
            "moe_router",
            shape=lambda d: (d.num_experts, d.hidden),
            src=_w1_one("block_sparse_moe.gate.weight"),
            dtype=_w1_router_dtype(d),
            transform=_w1_cast(_w1_router_dtype(d) or d.dtype),
            layers=moe,
        ),
        W(
            "moe_router_bias",
            shape=lambda d: (d.num_experts,),
            src=_w1_one("block_sparse_moe.gate.e_score_correction_bias"),
            dtype=fp32,
            layers=moe,
        ),
        W(
            "moe_latent_down",
            shape=lambda d: (d.latent, d.hidden),
            src=_w1_one("block_sparse_moe.routed_expert_down_proj.weight"),
            layers=moe,
        ),
        W(
            "moe_latent_norm",
            shape=lambda d: (d.latent,),
            src=_w1_one("block_sparse_moe.routed_expert_norm.weight"),
            layers=moe,
        ),
        W(
            "moe_latent_up",
            shape=lambda d: (d.hidden, d.latent),
            src=_w1_one("block_sparse_moe.routed_expert_up_proj.weight"),
            layers=moe,
        ),
        # --- layer 0's MLP ---
        W(
            "dense_gu",
            shape=lambda d: (2 * d.dense_inter, d.hidden),
            src=_w1_stack("mlp.gate_proj.weight", "mlp.up_proj.weight"),
            cut=_w1_rows(_w1_dense),
            transform=_w1_cat,
            layers=_w1_dense_layers,
        ),
        W(
            "dense_dn",
            shape=lambda d: (d.hidden, d.dense_inter),
            src=_w1_one("mlp.down_proj.weight"),
            cut=_w1_cols(_w1_dense),
            layers=_w1_dense_layers,
        ),
        # --- every layer: norms and attention-residual mixers ---
        W("attn_norm", shape=lambda d: (d.hidden,), src=_w1_one("input_layernorm.weight")),
        W(
            "attn_res_proj",
            shape=lambda d: (1, d.hidden),
            src=_w1_one("self_attention_res_proj.weight"),
        ),
        W(
            "attn_res_norm",
            shape=lambda d: (d.hidden,),
            src=_w1_one("self_attention_res_norm.weight"),
        ),
        W("mlp_norm", shape=lambda d: (d.hidden,), src=_w1_one("post_attention_layernorm.weight")),
        W("mlp_res_proj", shape=lambda d: (1, d.hidden), src=_w1_one("mlp_res_proj.weight")),
        W("mlp_res_norm", shape=lambda d: (d.hidden,), src=_w1_one("mlp_res_norm.weight")),
        # --- the model's own ---
        W(
            "embed",
            shape=lambda d: (d.vocab, d.hidden),
            src=_W1_MODEL + "embed_tokens.weight",
            layers=None,
        ),
        W(
            "out_res_proj",
            shape=lambda d: (1, d.hidden),
            src=_W1_MODEL + "output_attn_res_proj.weight",
            layers=None,
        ),
        W(
            "out_res_norm",
            shape=lambda d: (d.hidden,),
            src=_W1_MODEL + "output_attn_res_norm.weight",
            layers=None,
        ),
        W("final_norm", shape=lambda d: (d.hidden,), src=_W1_MODEL + "norm.weight", layers=None),
    )


# --------------------------------------------------------------------------------------------------
# ``w2_rows``: the experts of every MoE layer, shared (MXFP8) and routed (NVFP4), into
# ``core.w``.
#
# Each row fills ``core.w["l{i}_moe_<name>"]`` for the global layer ``i``: the ``_MoeW`` view reads
# it there and the MoE block hands it to the expert entries. Checkpoint values are used as stored.
# Checkpoint keys under ``language_model.model.layers.{i}.block_sparse_moe.``:
#
# * ``shared_experts.{gate,up,down}_proj``: ``weight`` (E4M3) and ``weight_scale`` (UE8M0, one per
#   32 values of a row);
# * ``experts.{e}.{w1,w3,w2}`` (gate, up, down; their hidden is the 3584-wide latent): ``weight``
#   (E2M1 pairs in U8), ``weight_scale`` (E4M3, one per 16 values), ``weight_scale_2`` and
#   ``input_scale`` (FP32 scalars).
#
# Layout dep16: attention DP, so the shared experts are whole on every rank; the routed experts
# MoE EP16.
#
# * shared: every rank holds all of ``gate_proj``, ``up_proj`` and ``down_proj``.
# * routed: a rank holds the 56 experts ``[56 moe_ep_rank, 56 moe_ep_rank + 56)``, whole.
#
# The routed rows hold the expert kernels' final operands, the same bytes for MegaMoE and the
# CuteDSL kernels, so both server roles read the same rows. The load requires one ``input_scale``
# per layer (the export stores 1.0), so MegaMoE's per-expert FC2 input scale ``fc1_norm_const``
# equals the layer's ``fc2_in_scale``.
# --------------------------------------------------------------------------------------------------

#: A routed expert's checkpoint key: ``w1`` gate, ``w3`` up, ``w2`` down.
_EXPERT_KEY = "language_model.model.layers.{i}.block_sparse_moe.experts.{e}.{w}.{kind}"
#: A shared-expert projection's checkpoint key prefix.
_SHARED_EXPERT_KEY = "language_model.model.layers.{i}.block_sparse_moe.shared_experts.{proj}"
#: The routed-expert split this table is written for: (moe_tp, moe_ep).
_EXPERT_SPLIT = (1, 16)
#: Ranks the shared experts' intermediate is split over.
_SHARED_EXPERT_TP = 1
#: The shared experts' full intermediate (2 shared experts x 3072).
_SHARED_EXPERT_INTER = 6144
#: Values per block scale: NVFP4 (routed), MXFP8 (shared).
_EXPERT_NVFP4_BLOCK = 16
_EXPERT_MXFP8_BLOCK = 32
#: Gate rows, then as many up rows, per FC1 group in the SM107 kernels' row order.
_EXPERT_GATE_UP_ROWS = 16
#: What a routed expert outside this rank's window leaves unread; every ``input_scale`` is read.
_EXPERT_OFFWINDOW_KINDS = ("weight", "weight_scale", "weight_scale_2")
#: The server roles this table loads (both read the same expert rows).
_EXPERT_ROLES = ("context", "generation")


def _expert_key(i: int, e: int, w: str, kind: str) -> str:
    return _EXPERT_KEY.format(i=i, e=e, w=w, kind=kind)


def _expert_layers(d: Dims) -> tuple[int, ...]:
    return d.moe_layers


def _expert_check(d: Dims) -> None:
    """Fail unless ``d`` is the split this table is written for, at widths with no padding."""
    problems = []
    if d.role not in _EXPERT_ROLES:
        problems.append(f"role {d.role!r}, not one of {_EXPERT_ROLES}")
    split = (d.moe_tp, d.moe_ep)
    if split != _EXPERT_SPLIT:
        problems.append(f"routed split (moe_tp, moe_ep) {split}, not {_EXPERT_SPLIT}")
    if d.local_experts * d.moe_ep != d.num_experts:
        problems.append(f"{d.local_experts} experts x EP{d.moe_ep} != {d.num_experts}")
    if d.expert_offset != d.moe_ep_rank * d.local_experts:
        problems.append(f"expert window starts at {d.expert_offset}, not EP rank x local")
    if d.moe_inter % 64 or d.latent % 128:
        problems.append(f"routed widths {d.moe_inter} / {d.latent} need padding")
    if d.shared_inter * _SHARED_EXPERT_TP != _SHARED_EXPERT_INTER:
        problems.append(f"shared slice {d.shared_inter} x {_SHARED_EXPERT_TP} != 6144")
    if d.kda_tp != _SHARED_EXPERT_TP:
        problems.append(f"shared experts split {_SHARED_EXPERT_TP} ways, TP group {d.kda_tp}")
    if d.shared_inter % 128 or d.hidden % 128:
        problems.append(f"shared widths {d.shared_inter} / {d.hidden} need padding")
    if problems:
        raise ValueError("experts: " + "; ".join(problems))


# --- relayouts (after the cut, on the destination's device) ---


def _expert_swizzle(s: torch.Tensor) -> torch.Tensor:
    """128x4 swizzle of an ``[M, C]`` block-scale byte matrix, returned ``[M, C]``.

    128-row x 4-column tiles, each in 32 x 4 x 4 atom order: the layout of the engine's
    ``block_scale_interleave`` (NVFP4 and the MXFP8 GEMM's K32 R128c4 scales) and of MegaMoE's
    per-slot scale build. ``_expert_check`` leaves no padding.
    """
    m, c = s.shape
    return s.reshape(m // 128, 4, 32, c // 4, 4).permute(0, 3, 2, 1, 4).reshape(m, c)


def _expert_gate_up16(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """FC1 rows in the SM107 kernels' order: 16 gate rows, 16 up rows, and so on.

    The engine's ``interleave_gate_and_linear(cat([up, gate]), 16)`` and MegaMoE's 16-atom
    gate/up interleave give the same order.
    """
    g = _EXPERT_GATE_UP_ROWS
    rows, cols = gate.shape
    pairs = torch.stack((gate.reshape(rows // g, g, cols), up.reshape(rows // g, g, cols)), dim=1)
    return pairs.reshape(2 * rows, cols)


def _shared_expert_gu_w(t: tuple[torch.Tensor, torch.Tensor], core: Any) -> torch.Tensor:
    """(gate, up) E4M3 rows -> one ``[gate; up]`` operand."""
    return torch.cat([x.view(torch.uint8) for x in t]).view(torch.float8_e4m3fn)


def _shared_expert_gu_s(t: tuple[torch.Tensor, torch.Tensor], core: Any) -> torch.Tensor:
    """(gate, up) UE8M0 scales -> concatenated, then swizzled once, flat (the MXFP8 linear's form)."""
    return _expert_swizzle(torch.cat([x.view(torch.uint8) for x in t])).reshape(-1)


def _shared_expert_dn_s(t: torch.Tensor, core: Any) -> torch.Tensor:
    """The down projection's UE8M0 scales -> swizzled, flat (the MXFP8 linear's form)."""
    return _expert_swizzle(t.view(torch.uint8)).reshape(-1)


def _expert_swizzled(t: torch.Tensor, core: Any) -> torch.Tensor:
    """One block-scale matrix (UE8M0 or E4M3 bytes) -> swizzled."""
    return _expert_swizzle(t.view(torch.uint8))


def _expert_fc1_w(t: tuple[torch.Tensor, torch.Tensor], core: Any) -> torch.Tensor:
    """(gate, up) E2M1 rows of one expert -> its FC1 operand."""
    return _expert_gate_up16(*t)


def _expert_fc1_s(t: tuple[torch.Tensor, torch.Tensor], core: Any) -> torch.Tensor:
    """(gate, up) E4M3 block scales of one expert -> its FC1 block scales."""
    return _expert_swizzle(_expert_gate_up16(*(x.view(torch.uint8) for x in t)))


def _expert_scalar_parts(t: tuple[torch.Tensor, ...], parts: int) -> list[torch.Tensor]:
    """A scalar source cut into its ``parts``, each stacked into one vector."""
    n = len(t) // parts
    return [torch.stack([x.reshape([]) for x in t[k * n : (k + 1) * n]]) for k in range(parts)]


def _expert_one_input_scale(name: str, *values: torch.Tensor) -> None:
    """Fail unless every routed ``input_scale`` in ``values`` is the same value.

    Meta tensors pass: the table machinery runs the transforms on them to check shapes, and they
    hold no values.
    """
    if values[0].is_meta:
        return
    first = values[0][0]
    if not all(bool((v == first).all()) for v in values):
        raise ValueError(
            f"the routed experts' {name} input_scale differ within a layer; this table takes "
            "one value per layer (the ModelOpt export stores 1.0)"
        )


def _expert_fc1_input_scale(t: tuple[torch.Tensor, ...], core: Any) -> torch.Tensor:
    """Every routed expert's gate then up input_scale -> 1 / their max."""
    gate, up = _expert_scalar_parts(t, 2)
    _expert_one_input_scale("FC1", gate, up)
    return gate.max().reciprocal().reshape(1)


def _expert_fc2_input_scale(t: tuple[torch.Tensor, ...], core: Any) -> torch.Tensor:
    """Every routed expert's down input_scale -> 1 / their max."""
    (down,) = _expert_scalar_parts(t, 1)
    _expert_one_input_scale("FC2", down)
    return down.max().reciprocal().reshape(1)


def _expert_fc1_alpha(t: tuple[torch.Tensor, ...], core: Any) -> torch.Tensor:
    """This rank's experts' gate and up input_scale and weight_scale_2 -> FC1 alpha.

    ``1 / (fc31_input_scale * (1 / weight_scale_2))`` as the engine computes it, with the gate's
    weight_scale_2. The engine takes the larger of gate and up when they differ; this load
    refuses that case. Meta tensors pass (no values).
    """
    in_gate, in_up, ws2_gate, ws2_up = _expert_scalar_parts(t, 4)
    _expert_one_input_scale("FC1", in_gate, in_up)
    if not ws2_gate.is_meta and not torch.allclose(ws2_gate, ws2_up):
        raise ValueError("a routed expert's gate and up weight_scale_2 differ")
    return 1.0 / (in_gate.max().reciprocal() * (1.0 / ws2_gate))


def _expert_fc2_alpha(t: tuple[torch.Tensor, ...], core: Any) -> torch.Tensor:
    """This rank's experts' down input_scale and weight_scale_2 -> FC2 alpha."""
    in_down, ws2_down = _expert_scalar_parts(t, 2)
    _expert_one_input_scale("FC2", in_down)
    return 1.0 / (in_down.max().reciprocal() * (1.0 / ws2_down))


def _expert_fc1_norm_const(t: tuple[torch.Tensor, ...], core: Any) -> torch.Tensor:
    """This rank's experts' down input_scale -> MegaMoE's FC2 input scale, 1 / each."""
    (down,) = _expert_scalar_parts(t, 1)
    _expert_one_input_scale("FC2", down)
    return down.reciprocal()


# --- sources and cuts ---


def _shared_expert_pair(kind: str) -> Callable[[Dims, int], list]:
    """The gate and up projections' ``kind``, handed to the transform together."""

    def build(d: Dims, i: int) -> list:
        gate, up = (_SHARED_EXPERT_KEY.format(i=i, proj=p) for p in ("gate_proj", "up_proj"))
        return [((f"{gate}.{kind}", f"{up}.{kind}"), None)]

    return build


def _shared_expert_down(kind: str) -> Callable[[Dims, int], list]:
    """The down projection's ``kind``."""

    def build(d: Dims, i: int) -> list:
        return [(_SHARED_EXPERT_KEY.format(i=i, proj="down_proj") + f".{kind}", None)]

    return build


def _shared_expert_rank(d: Dims) -> int:
    """This rank's index in the TP group that splits the MLPs."""
    return d.kda_tp_rank


def _shared_expert_rows(d: Dims) -> Cut:
    """Gate and up rows of this rank's slice (weights and scales)."""
    r, s = _shared_expert_rank(d), d.shared_inter
    return (slice(r * s, (r + 1) * s),)


def _shared_expert_cols(per_col: int) -> Callable[[Dims], Cut]:
    """Down-projection input columns of the slice, ``per_col`` values per column."""

    def cut(d: Dims) -> Cut:
        r, s = _shared_expert_rank(d), d.shared_inter // per_col
        return (slice(None), slice(r * s, (r + 1) * s))

    return cut


def _expert_window(*parts: tuple[str, str]) -> Callable[[Dims, int], list]:
    """One source per expert of this rank's EP window; several ``parts`` go together."""

    def build(d: Dims, i: int) -> list:
        out = []
        for local in range(d.local_experts):
            keys = tuple(_expert_key(i, d.expert_offset + local, w, kind) for w, kind in parts)
            out.append((keys if len(keys) > 1 else keys[0], (local,)))
        return out

    return build


def _expert_scalars(*parts: tuple[str, str], every_expert: bool) -> Callable[[Dims, int], list]:
    """One source for the whole destination: the ``parts`` scalars, part by part.

    Of every routed expert, or of this rank's window.
    """

    def build(d: Dims, i: int) -> list:
        if every_expert:
            experts = range(d.num_experts)
        else:
            experts = range(d.expert_offset, d.expert_offset + d.local_experts)
        return [(tuple(_expert_key(i, e, w, kind) for w, kind in parts for e in experts), None)]

    return build


def _expert_fc1_rows(d: Dims) -> Cut:
    """FC1 output rows of this rank's MoE TP slice: the same for gate and up, weights and scales."""
    return (slice(d.moe_tp_rank * d.moe_inter, (d.moe_tp_rank + 1) * d.moe_inter),)


def _expert_fc2_cols(per_col: int) -> Callable[[Dims], Cut]:
    """FC2 input columns of the slice, ``per_col`` values per column (2 per byte, 16 per scale)."""

    def cut(d: Dims) -> Cut:
        span = d.moe_inter // per_col
        return (slice(None), slice(d.moe_tp_rank * span, (d.moe_tp_rank + 1) * span))

    return cut


# --- the rows ---


def _routed_expert_row(
    name: str,
    shape: Callable[[Dims], tuple[int, ...]],
    dtype: torch.dtype,
    src: Callable | None,
    cut: Callable[[Dims], Cut] | None = None,
    transform: Callable | None = None,
) -> W:
    """A routed-expert row: ``core.w["l{i}_<name>"]`` on every MoE layer."""
    return W(
        name,
        shape=shape,
        src=src,
        dtype=dtype,
        cut=cut,
        transform=transform,
        layers=_expert_layers,
    )


def w2_rows(d: Dims) -> tuple[W, ...]:
    """Return the expert rows at this rank's share, by global layer."""
    _expert_check(d)
    return (
        # Shared experts, MXFP8.
        W(
            "moe_shared_gu_w",
            shape=lambda d: (2 * d.shared_inter, d.hidden),
            src=_shared_expert_pair("weight"),
            dtype=torch.float8_e4m3fn,
            cut=_shared_expert_rows,
            transform=_shared_expert_gu_w,
            layers=_expert_layers,
        ),
        W(
            "moe_shared_gu_s",
            shape=lambda d: (2 * d.shared_inter * (d.hidden // _EXPERT_MXFP8_BLOCK),),
            src=_shared_expert_pair("weight_scale"),
            dtype=torch.uint8,
            cut=_shared_expert_rows,
            transform=_shared_expert_gu_s,
            layers=_expert_layers,
        ),
        W(
            "moe_shared_dn_w",
            shape=lambda d: (d.hidden, d.shared_inter),
            src=_shared_expert_down("weight"),
            dtype=torch.float8_e4m3fn,
            cut=_shared_expert_cols(1),
            layers=_expert_layers,
        ),
        W(
            "moe_shared_dn_s",
            shape=lambda d: (d.hidden * (d.shared_inter // _EXPERT_MXFP8_BLOCK),),
            src=_shared_expert_down("weight_scale"),
            dtype=torch.uint8,
            cut=_shared_expert_cols(_EXPERT_MXFP8_BLOCK),
            transform=_shared_expert_dn_s,
            layers=_expert_layers,
        ),
        # Routed experts, NVFP4: this rank's EP window at its MoE TP slice.
        _routed_expert_row(
            "moe_fc1_w",
            lambda d: (d.local_experts, 2 * d.moe_inter, d.latent // 2),
            torch.uint8,
            _expert_window(("w1", "weight"), ("w3", "weight")),
            _expert_fc1_rows,
            _expert_fc1_w,
        ),
        _routed_expert_row(
            "moe_fc1_s",
            lambda d: (d.local_experts, 2 * d.moe_inter, d.latent // _EXPERT_NVFP4_BLOCK),
            torch.uint8,
            _expert_window(("w1", "weight_scale"), ("w3", "weight_scale")),
            _expert_fc1_rows,
            _expert_fc1_s,
        ),
        _routed_expert_row(
            "moe_fc2_w",
            lambda d: (d.local_experts, d.latent, d.moe_inter // 2),
            torch.uint8,
            _expert_window(("w2", "weight")),
            _expert_fc2_cols(2),
        ),
        _routed_expert_row(
            "moe_fc2_s",
            lambda d: (d.local_experts, d.latent, d.moe_inter // _EXPERT_NVFP4_BLOCK),
            torch.uint8,
            _expert_window(("w2", "weight_scale")),
            _expert_fc2_cols(_EXPERT_NVFP4_BLOCK),
            _expert_swizzled,
        ),
        _routed_expert_row(
            "moe_fc31_in_scale",
            lambda d: (1,),
            torch.float32,
            _expert_scalars(("w1", "input_scale"), ("w3", "input_scale"), every_expert=True),
            transform=_expert_fc1_input_scale,
        ),
        _routed_expert_row(
            "moe_fc2_in_scale",
            lambda d: (1,),
            torch.float32,
            _expert_scalars(("w2", "input_scale"), every_expert=True),
            transform=_expert_fc2_input_scale,
        ),
        _routed_expert_row(
            "moe_fc31_alpha",
            lambda d: (d.local_experts,),
            torch.float32,
            _expert_scalars(
                ("w1", "input_scale"),
                ("w3", "input_scale"),
                ("w1", "weight_scale_2"),
                ("w3", "weight_scale_2"),
                every_expert=False,
            ),
            transform=_expert_fc1_alpha,
        ),
        _routed_expert_row(
            "moe_fc2_alpha",
            lambda d: (d.local_experts,),
            torch.float32,
            _expert_scalars(("w2", "input_scale"), ("w2", "weight_scale_2"), every_expert=False),
            transform=_expert_fc2_alpha,
        ),
        _routed_expert_row(
            "moe_fc1_norm_const",
            lambda d: (d.local_experts,),
            torch.float32,
            _expert_scalars(("w2", "input_scale"), every_expert=False),
            transform=_expert_fc1_norm_const,
        ),
    )


def w2_offwindow_keys(core: Any, weights: Any) -> set[str]:
    """Return the expert keys this rank leaves unread: the other ranks' experts.

    For each routed expert outside this rank's EP window, every key but ``input_scale`` (read on
    every rank). Predicted from the layout; ``weights`` is not read.
    """
    del weights
    d = dims(core)
    window = range(d.expert_offset, d.expert_offset + d.local_experts)
    return {
        _expert_key(i, e, w, kind)
        for i in d.moe_layers
        for e in range(d.num_experts)
        if e not in window
        for w in ("w1", "w2", "w3")
        for kind in _EXPERT_OFFWINDOW_KINDS
    }


# --------------------------------------------------------------------------------------------------
# ``w3_rows``: the DFlash2 drafter's checkpoint (its own checkpoint, BF16) into
# ``core.w["d_*"]``.
#
# Keys are the checkpoint's: ``fc.weight``, ``hidden_norm.weight``, ``norm.weight``,
# ``layers.{j}.*`` and ``candidate_selector.*``. The rows hold what modeling.py's drafter reads, in
# the layout it reads; ``d_kv_a`` stacks the five layers' ``kv_a_proj_with_mqa``, so one
# projection builds every layer's history latent.
#
# dep16 shares: attention DP, so every rank holds the whole drafter (the cuts below are whole).
#
# On a disaggregated context server (``Dims.draft_context_kv_only``) the drafter never drafts: it
# only builds the history, from ``fc``, ``hidden_norm`` and every layer's ``kv_a_proj_with_mqa``
# and ``kv_a_layernorm``. ``w3_rows`` then returns only those rows, so nothing else is declared or
# read, and ``w3_unread_keys`` names the checkpoint's other keys.
# --------------------------------------------------------------------------------------------------

#: The DFlash2 drafter this target runs; modeling.py asserts its config matches.
DRAFT_SHAPE = MappingProxyType(
    {
        "hidden_size": 7168,
        "num_hidden_layers": 5,
        "num_attention_heads": 96,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "intermediate_size": 14336,
        "vocab_size": 163840,
        "rms_norm_eps": 1e-5,
        "sliding_window": 1024,
    }
)
DRAFT_DFLASH = MappingProxyType(
    {
        "attention_mode": "mla",
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "selector_rank": 256,
        "selector_top_k": 32,
    }
)
#: Target layers whose captures ``fc`` projects (``dflash_config.target_layer_ids``).
DRAFT_CAPTURES = 6

_DL = DRAFT_SHAPE["num_hidden_layers"]
_DH = DRAFT_SHAPE["hidden_size"]
_DHEADS = DRAFT_SHAPE["num_attention_heads"]
_QL = DRAFT_SHAPE["q_lora_rank"]
_KVL = DRAFT_SHAPE["kv_lora_rank"]
_NOPE = DRAFT_SHAPE["qk_nope_head_dim"]
_ROPE = DRAFT_SHAPE["qk_rope_head_dim"]
_V = DRAFT_SHAPE["v_head_dim"]
_DI = DRAFT_SHAPE["intermediate_size"]
_TAPS = DRAFT_DFLASH["conv_kernel_size"]
_GROUPS = _DH // DRAFT_DFLASH["conv_group_size"]
_RANK = DRAFT_DFLASH["selector_rank"]


def _layer_key(j: int, name: str) -> str:
    return f"layers.{j}.{name}"


def _cat_rows(src: tuple, core: Any) -> torch.Tensor:
    return torch.cat(src, 0)


def _stack(src: tuple, core: Any) -> torch.Tensor:
    return torch.stack(src, 0)


def _kv_b_regroup(t: torch.Tensor, core: Any) -> torch.Tensor:
    """``[heads * (nope + v), kv_lora]`` head-major -> every head's ``k_nope`` rows, then ``v``."""
    w = t.view(-1, _NOPE + _V, _KVL)
    return torch.cat((w[:, :_NOPE].reshape(-1, _KVL), w[:, _NOPE:].reshape(-1, _KVL)), 0)


def _draft_heads(d: Dims) -> tuple[int, int]:
    """(rank, heads) of this rank's ``q_b`` / ``kv_b`` heads: all of them under attention DP."""
    return 0, _DHEADS


def _draft_o_heads(d: Dims) -> tuple[int, int]:
    """(rank, heads) of this rank's ``o_proj`` input columns: all of them."""
    return 0, _DHEADS


def _draft_mlp(d: Dims) -> tuple[int, int]:
    """(rank, rows) of this rank's MLP intermediate: all of it."""
    return 0, _DI


def _history_rows() -> tuple[W, ...]:
    return (
        W("d_fc", lambda d: (_DH, DRAFT_CAPTURES * _DH), "fc.weight", layers=None),
        W("d_hidden_norm", lambda d: (_DH,), "hidden_norm.weight", layers=None),
        W(
            "d_kv_a",
            lambda d: (_DL * (_KVL + _ROPE), _DH),
            src_keys(*(_layer_key(j, "self_attn.kv_a_proj_with_mqa.weight") for j in range(_DL))),
            transform=_cat_rows,
            layers=None,
        ),
        W(
            "d_kv_a_norm",
            lambda d: (_DL, _KVL),
            src_keys(*(_layer_key(j, "self_attn.kv_a_layernorm.weight") for j in range(_DL))),
            transform=_stack,
            layers=None,
        ),
    )


def _layer_rows(j: int, q_a_kv_a: bool = False) -> tuple[W, ...]:
    """Drafter layer ``j``'s rows; ``q_a_kv_a`` (draft_q_a_kv_a_fused): q_a's row stacked with the
    layer's kv_a rows (``d_l{j}_q_a_kv_a``), which ``d_kv_a`` also holds."""

    def key(name: str) -> str:
        return _layer_key(j, name)

    heads = cut_rows(lambda d: (_draft_heads(d)[0], _draft_heads(d)[1] * (_NOPE + _ROPE)))
    kv_heads = cut_rows(lambda d: (_draft_heads(d)[0], _draft_heads(d)[1] * (_NOPE + _V)))
    mlp_rows = cut_rows(_draft_mlp)
    return (
        W(f"d_l{j}_in_norm", lambda d: (_DH,), key("input_layernorm.weight"), layers=None),
        W(
            f"d_l{j}_post_norm",
            lambda d: (_DH,),
            key("post_attention_layernorm.weight"),
            layers=None,
        ),
        W(
            f"d_l{j}_q_a_kv_a",
            lambda d: (_QL + _KVL + _ROPE, _DH),
            src_keys(key("self_attn.q_a_proj.weight"), key("self_attn.kv_a_proj_with_mqa.weight")),
            transform=_cat_rows,
            layers=None,
        )
        if q_a_kv_a
        else W(f"d_l{j}_q_a", lambda d: (_QL, _DH), key("self_attn.q_a_proj.weight"), layers=None),
        W(f"d_l{j}_q_a_norm", lambda d: (_QL,), key("self_attn.q_a_layernorm.weight"), layers=None),
        W(
            f"d_l{j}_q_b",
            lambda d: (_draft_heads(d)[1] * (_NOPE + _ROPE), _QL),
            key("self_attn.q_b_proj.weight"),
            cut=heads,
            layers=None,
        ),
        W(
            f"d_l{j}_kv_b",
            lambda d: (_draft_heads(d)[1] * (_NOPE + _V), _KVL),
            key("self_attn.kv_b_proj.weight"),
            cut=kv_heads,
            transform=_kv_b_regroup,
            layers=None,
        ),
        W(
            f"d_l{j}_o",
            lambda d: (_DH, _draft_o_heads(d)[1] * _V),
            key("self_attn.o_proj.weight"),
            cut=cut_cols(lambda d: (_draft_o_heads(d)[0], _draft_o_heads(d)[1] * _V)),
            layers=None,
        ),
        W(
            f"d_l{j}_gate_up",
            lambda d: (2 * _draft_mlp(d)[1], _DH),
            src_keys(key("mlp.gate_proj.weight"), key("mlp.up_proj.weight")),
            cut=lambda d: Cuts((mlp_rows(d), mlp_rows(d))),
            transform=_cat_rows,
            layers=None,
        ),
        W(
            f"d_l{j}_down",
            lambda d: (_DH, _draft_mlp(d)[1]),
            key("mlp.down_proj.weight"),
            cut=cut_cols(_draft_mlp),
            layers=None,
        ),
        W(
            f"d_l{j}_attn_conv_base",
            lambda d: (2, _TAPS, _DH),
            key("attention_conv.base_kernel"),
            layers=None,
        ),
        W(
            f"d_l{j}_attn_conv_proj",
            lambda d: (2 * _TAPS * _GROUPS, _DH),
            key("attention_conv.kernel_projection.weight"),
            layers=None,
        ),
        W(
            f"d_l{j}_mlp_conv_base",
            lambda d: (2, _TAPS, _DH),
            key("mlp_conv.base_kernel"),
            layers=None,
        ),
        W(
            f"d_l{j}_mlp_conv_proj",
            lambda d: (2 * _TAPS * _GROUPS, _DH),
            key("mlp_conv.kernel_projection.weight"),
            layers=None,
        ),
    )


def w3_rows(d: Dims) -> tuple[W, ...]:
    """Return the drafter's rows; on a history-only drafter, only the history's."""
    rows = list(_history_rows())
    if d.draft_context_kv_only:
        return tuple(rows)
    rows.append(W("d_final_norm", lambda d: (_DH,), "norm.weight", layers=None))
    rows += [
        W(
            "d_sel_pred",
            lambda d: (d.vocab, _RANK),
            "candidate_selector.predecessor_codebook",
            layers=None,
        ),
        W(
            "d_sel_succ",
            lambda d: (d.vocab, _RANK),
            "candidate_selector.successor_codebook",
            layers=None,
        ),
        W(
            "d_sel_hidden_proj",
            lambda d: (_RANK, _DH),
            "candidate_selector.hidden_projection.weight",
            layers=None,
        ),
    ]
    for j in range(_DL):
        rows += _layer_rows(j, d.draft_q_a_kv_a_fused)
    return tuple(rows)


def _read_keys(d: Dims) -> set[str]:
    """Every checkpoint key the rows of ``d`` read."""
    keys: set[str] = set()
    for row in w3_rows(d):
        for key, _ in row.sources(d, None):
            keys.update(key if isinstance(key, tuple) else (key,))
    return keys


def w3_unread_keys(core: Any, weights: Any) -> set[str]:
    """Return the drafter checkpoint keys this rank leaves unread.

    None on a drafting server (every key is read, through this rank's cut); with
    ``core.dims.draft_context_kv_only``, every key outside the history's rows.
    """
    return set(weights.keys()) - _read_keys(core.dims)


def load_draft(model: Any, weights: Any, weight_mapper: Any = None) -> None:
    """Fill the drafter's ``core.w["d_*"]`` parameters of the shell ``model`` from its checkpoint.

    Asserts coverage both ways: every drafter parameter written once at its row's shape and dtype,
    and every checkpoint key read or a history-only drafter's expected leftover.
    """
    del weight_mapper
    core = model.model
    d = core.dims
    params = {
        name: p for name, p in core.named_parameters() if name.startswith(f"w.{DRAFT_KEY_PREFIX}")
    }
    consumed = fill_table(core, weights, w3_rows(d), params)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    leftover = set(weights.keys()) - consumed
    expected = w3_unread_keys(core, weights)
    assert leftover == expected, (
        "drafter checkpoint coverage: unexpected leftovers "
        f"{sorted(leftover - expected)[:8]}, expected but read {sorted(expected - leftover)[:8]}"
    )


# --------------------------------------------------------------------------------------------------
# ``w4_rows``: the vision tower (MoonViT3d) and its projector (PatchMergerMLPV2) into ``core.w["v_*"]``,
# on a server that builds the tower (``Dims.vision``); none elsewhere, where the keys are leftovers.
#
# Every tensor whole on every rank, BF16 as stored (the export excludes the tower from quantization):
# ``v_patch_proj``, ``v_pos_emb``, ``v_final_norm``, the 27 blocks' ``v{j}_{norm0, norm1, wqkv, wo,
# fc0, fc1}`` and the projector's ``v_merge_fc0``, ``v_merge_fc1``, ``v_post_norm``: the 168 keys under
# ``vision_tower.`` and ``mm_projector.``, the parameters of the generic target's ``K31VisionTower``
# under its checkpoint names. modeling.py's ``_vision_view`` reads them.
# --------------------------------------------------------------------------------------------------

#: MoonViT3d's widths: hidden, fused qkv width (12 heads x 128), MLP, blocks, patch side and the
#: position table; the projector merges 2 x 2 rows of the hidden width into the text model's.
_W4_HIDDEN = 1024
_W4_QKV = 1536
_W4_MLP = 4096
_W4_BLOCKS = 27
_W4_PATCH = 14
_W4_POS = (64, 64)
_W4_MERGED = 4 * _W4_HIDDEN
_W4_TEXT = 7168
#: The 27 blocks' tensors: ``core.w`` name suffix, checkpoint key suffix, shape.
_W4_BLOCK = (
    ("norm0", "norm0.weight", (_W4_HIDDEN,)),
    ("norm1", "norm1.weight", (_W4_HIDDEN,)),
    ("wqkv", "wqkv.weight", (3 * _W4_QKV, _W4_HIDDEN)),
    ("wo", "wo.weight", (_W4_HIDDEN, _W4_QKV)),
    ("fc0", "mlp.fc0.weight", (_W4_MLP, _W4_HIDDEN)),
    ("fc1", "mlp.fc1.weight", (_W4_HIDDEN, _W4_MLP)),
)


def _w4_row(name: str, shape: tuple[int, ...], key: str) -> W:
    """One whole BF16 tensor of the tower into ``core.w[name]``."""
    return W(name, lambda d, s=shape: s, key, dtype=torch.bfloat16, layers=None)


def w4_rows(d: Dims) -> tuple[W, ...]:
    """The vision tower's and projector's rows; none on a server that builds no tower."""
    if not d.vision:
        return ()
    rows = [
        _w4_row(
            "v_patch_proj",
            (_W4_HIDDEN, 3, _W4_PATCH, _W4_PATCH),
            "vision_tower.patch_embed.proj.weight",
        ),
        _w4_row("v_pos_emb", (*_W4_POS, _W4_HIDDEN), "vision_tower.patch_embed.pos_emb.weight"),
        _w4_row("v_final_norm", (_W4_HIDDEN,), "vision_tower.encoder.final_layernorm.weight"),
        _w4_row("v_merge_fc0", (_W4_MERGED, _W4_MERGED), "mm_projector.proj.0.weight"),
        _w4_row("v_merge_fc1", (_W4_TEXT, _W4_MERGED), "mm_projector.proj.2.weight"),
        _w4_row("v_post_norm", (_W4_TEXT,), "mm_projector.post_norm.weight"),
    ]
    for j in range(_W4_BLOCKS):
        rows += [
            _w4_row(f"v{j}_{name}", shape, f"vision_tower.encoder.blocks.{j}.{key}")
            for name, key, shape in _W4_BLOCK
        ]
    return tuple(rows)


def media_proc_cfg(weights: Any) -> dict | None:
    """The ``media_proc_cfg`` of the checkpoint's HF image processor (``preprocessor_config.json``
    beside the shards ``weights`` was opened from): the image limits the tower's largest grid comes
    from. None without the file (modeling.py then warns and takes the processor's default limits).
    """
    directory = getattr(weights, "checkpoint_dir", None)
    path = os.path.join(directory, "preprocessor_config.json") if directory else None
    if path is None or not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f).get("media_proc_cfg") or {}
