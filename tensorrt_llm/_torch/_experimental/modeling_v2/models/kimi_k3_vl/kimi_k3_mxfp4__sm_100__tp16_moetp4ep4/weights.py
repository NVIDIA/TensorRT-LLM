# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight manifest and loader: Kimi K3 (MXFP4) / sm_100 / tp16_moetp4ep4.

The checkpoint is the vision-language wrapper's. `language_model.*` holds the language model; `vision_tower.*` and
`mm_projector.*` hold the vision tower and its projector, which this text-only target never reads. Any other key fails
the load, naming it.

Every parameter of the text model is one manifest row (`manifest`): the checkpoint key or keys it is read from, this
rank's cut of them, and the transform on the way. Every share is computed once, in `dims`, from the config and the
engine's `Mapping`:

* **KDA** (`linear_attn`; `self_attn.*` in the checkpoint) is split by head over the TP group: a rank holds its heads'
  rows of q, k, v, g, the short convolutions, f_b, b, dt_bias and A_log, and the matching input columns of o_proj.
  f_a and o_norm are replicated. The six input projections are one parameter, `k3_proj_weight`
  `[q | k | v | g | f_a | b | pad]` (zero pad rows up to a multiple of 8), one row read from six keys. The checkpoint
  pads A_log to the head dim; the cut takes the rank's heads from it.
* **MLA** (`self_attn.mixer`; `self_attn.*` in the checkpoint) is split by head the same way: rows of q_b, kv_b and g,
  columns of o_proj. q_a and kv_a are replicated. q_a, kv_a and the rank's g rows are one parameter, `k3_ag_weight`
  `[q_a ; kv_a ; g]`, one row read from three keys. kv_b's per-head `[K | V]` rows are regrouped into
  `[all K ; all V]`.
* **The dense MLP** (layer 0) is split over the layer's own TP group (`mlp_tp_size`, which the layer narrows to fit a
  node), **the shared expert** over the TP group: rows of gate and up, concatenated gate first, and columns of down.
* **The routed experts** are this rank's EP window of whole experts, each at its MoE-TP slice of the intermediate:
  under moe_tp 4 x moe_ep 4 a rank holds a quarter of the width of a quarter of the experts.
* Everything else (embedding, norms, the attention-residual norms and projections, router and bias, the latent
  projections and norm) is replicated. lm_head is the shell's, vocab-parallel; the load reads the rows it built.

**A rank reads only its share.** A split row carries a `cut` (or one cut per key, `cuts`), applied to the engine's
lazy safetensors entry, so the bytes of other ranks' shares are never read. A row of several keys concatenates them
on the host, where the lazy entries are read, and copies the result into its parameter once. The routed experts are
the exception: each local expert is read whole and handed to the routed experts' relayout
(`moe.pack_mxfp4_expert`), which cuts the MoE-TP slice itself.

**The routed experts stream per shard file.** The engine's lazy weights keep every shard mapped for the whole load,
and the page cache cannot drop pages of an open mapping. The experts are about 90 GB of distinct pages per rank,
more than a node holds for 4 ranks. So the rank's expert tensors are grouped by shard file, and each
file is opened, read, closed and dropped from the page cache (`posix_fadvise(DONTNEED)`). Without a
`model.safetensors.index.json` the experts fall back to the shared lazy weights, with a warning.

**What the load derives.** After the experts, the KDA modules' decode views and constants (`_finalize_kda`: the
target's `K3DecodeKDA.finalize_decode_weights`). MLA's absorption weights (`k_b_proj_trans`, `v_b_proj`) and the row
views of `k3_ag_weight` are not the load's: the MLA module derives them in its `post_load_weights`.

Coverage is checked both ways before anything is read: every parameter is one row (`_check_rows`); every key the
rank reads is in the checkpoint; a `language_model.*` key outside `model.layers.` that nothing reads warns.

This file imports no tensorrt_llm code, and every module it writes into is the target's own (`kda.py`, `mla.py`,
`moe.py`, `norm.py`), except the shell's lm_head.
"""

from __future__ import annotations

import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable

import torch
from safetensors import safe_open
from torch import nn

#: The language model's key prefix in the checkpoint.
LANG_PREFIX = "language_model."

#: The vision tower's and the projector's key families: in the checkpoint, not loaded by this text-only target.
PREDICTED_NON_LOAD = ("vision_tower.", "mm_projector.")

#: Threads reading the rows, and threads streaming the routed experts (one shard file each).
TRUNK_WORKERS = 8
EXPERT_WORKERS = 4

#: Parameters that are not rows: the draft model's (its own checkpoint) and the routed experts' buffers (filled per
#: expert by `_pack_routed_expert`).
_NOT_ROW_PREFIXES = ("draft_model.",)
_NOT_ROW_INFIXES = (".routed_experts.backend.",)

#: One routed expert's packed MXFP4 tensors: the relayout's argument name -> the key suffix.
_EXPERT_TENSORS = {
    f"{w}_{arg}": f"{w}.{kind}"
    for w in ("w1", "w2", "w3")
    for arg, kind in (("weight", "weight_packed"), ("weight_scale", "weight_scale"))
}

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

#: An index into a checkpoint tensor: a tuple of slices, one per leading axis.
Cut = tuple[slice, ...]


@dataclass(frozen=True)
class Row:
    """One parameter of the module tree and where its bytes come from.

    `param` is the parameter's name in `model.named_parameters()`. `keys` are checkpoint keys without `LANG_PREFIX`;
    a row of several keys reads each through its cut and hands them to `transform` together. `cut` is this rank's
    rows or columns of every key, read from the lazy entry so nothing else is read; None reads the tensor whole.
    `cuts`, in place of `cut`, is one cut per key, for keys split differently. `transform(tensors, d)` relayouts the
    cut tensor(s) without slicing of its own; None means the one cut tensor is the parameter. The result is cast to
    the parameter's dtype.
    """

    param: str
    keys: tuple[str, ...]
    cut: Cut | None = None
    transform: Callable | None = None
    cuts: tuple[Cut | None, ...] | None = None

    def key_cuts(self) -> tuple[Cut | None, ...]:
        """The cut of each key, in `keys` order."""
        if self.cuts is None:
            return (self.cut,) * len(self.keys)
        assert self.cut is None and len(self.cuts) == len(self.keys), (
            f"{self.param}: give `cut` or one cut per key in `cuts`"
        )
        return self.cuts


# --- cuts: this rank's block of rows or columns ---


def _rows(rank: int, span: int) -> Cut:
    return (slice(rank * span, (rank + 1) * span),)


def _cols(rank: int, span: int) -> Cut:
    return (slice(None), slice(rank * span, (rank + 1) * span))


def _read(entry, cut: Cut | None) -> torch.Tensor:
    """Realize one checkpoint entry, reading only `cut` of it: on a lazy safetensors slice the index reads those rows
    or columns and nothing else; on a tensor it is a view. A 0-dim entry rejects `[:]`."""
    if cut is None:
        return entry if getattr(entry, "ndim", 1) == 0 else entry[:]
    return entry[cut[0]] if len(cut) == 1 else entry[cut]


# --- transforms, on the rank's cut ---


def _cat_rows(t: tuple[torch.Tensor, ...], d) -> torch.Tensor:
    """Row-concatenate: gate then up (`gate_up_proj`); q_a, kv_a then g (MLA's `k3_ag_weight`)."""
    return torch.cat(t, dim=0)


#: KDA's `k3_proj_weight` has zero rows up to a multiple of this.
_KDA_PROJ_ROW_MULTIPLE = 8


def _cat_rows_padded(t: tuple[torch.Tensor, ...], d) -> torch.Tensor:
    """Row-concatenate, then zero rows up to a multiple of `_KDA_PROJ_ROW_MULTIPLE`: KDA's `k3_proj_weight`
    `[q | k | v | g | f_a | b | pad]`."""
    pad = (-sum(x.shape[0] for x in t)) % _KDA_PROJ_ROW_MULTIPLE
    if pad:
        t = (*t, t[0].new_zeros(pad, t[0].shape[1]))
    return torch.cat(t, dim=0)


def _regroup_kv_b(t: torch.Tensor, d) -> torch.Tensor:
    """This rank's per-head `[K | V]` kv_b rows -> `[all K ; all V]`: context attention projects straight into the
    FMHA layout, and the absorbed decode takes zero-copy K and V views."""
    k, v = t.reshape(d.mla_heads, d.nope + d.v_dim, d.kv_lora).split([d.nope, d.v_dim], dim=1)
    return torch.cat(
        [k.reshape(d.mla_heads * d.nope, d.kv_lora), v.reshape(d.mla_heads * d.v_dim, d.kv_lora)]
    )


def _unpad_a_log(t: torch.Tensor, d) -> torch.Tensor:
    """A_log on a rank holding every head: the checkpoint pads it from the head count to the head dim, and the padding
    must be zeros."""
    tail = t[d.kda_heads :]
    if tail.numel() and tail.abs().max().item() != 0.0:
        raise ValueError(
            f"Kimi K3 A_log: expected zero padding beyond {d.kda_heads} entries, got a nonzero tail"
        )
    return t[: d.kda_heads]


def _lm_head_pad(rows: int) -> Callable:
    """Zero rows past the vocabulary on the last vocab-parallel shard, as the built-in LMHead pads them."""

    def pad(t: torch.Tensor, d) -> torch.Tensor:
        if t.shape[0] == rows:
            return t
        out = t.new_zeros(rows, t.shape[1])
        out[: t.shape[0]] = t
        return out

    return pad


# --- the manifest ---


def dims(model) -> SimpleNamespace:
    """Every per-rank width and layer set, from the text config and the engine's Mapping."""
    cfg = model.config
    m = model.model_config.mapping
    lin = cfg.linear_attn_config
    tp = m.tp_size
    n = cfg.num_hidden_layers
    moe_layers = [
        i
        for i in range(n)
        if cfg.num_experts is not None
        and i >= cfg.first_k_dense_replace
        and i % getattr(cfg, "moe_layer_freq", 1) == 0
    ]
    for what, full, parts in (
        ("KDA heads", lin["num_heads"], tp),
        ("MLA heads", cfg.num_attention_heads, tp),
        ("shared expert intermediate", cfg.moe_intermediate_size * cfg.num_shared_experts, tp),
        ("routed experts", cfg.num_experts, m.moe_ep_size),
    ):
        if full % parts:
            raise ValueError(f"Kimi K3 load: {what} ({full}) do not split {parts} ways")
    local_experts = cfg.num_experts // m.moe_ep_size
    return SimpleNamespace(
        num_layers=n,
        kda_layers=[i for i in range(n) if i + 1 in lin["kda_layers"]],
        mla_layers=[i for i in range(n) if i + 1 in lin["full_attn_layers"]],
        moe_layers=moe_layers,
        tp=tp,
        rank=m.tp_rank,
        vocab=cfg.vocab_size,
        kda_heads=lin["num_heads"] // tp,
        kda_proj=lin["num_heads"] // tp * lin["head_dim"],
        mla_heads=cfg.num_attention_heads // tp,
        nope=cfg.qk_nope_head_dim,
        qk_dim=cfg.qk_nope_head_dim + cfg.qk_rope_head_dim,
        v_dim=cfg.v_head_dim,
        kv_lora=cfg.kv_lora_rank,
        dense_inter=cfg.intermediate_size,
        shared_inter=cfg.moe_intermediate_size * cfg.num_shared_experts // tp,
        local_experts=local_experts,
        expert_offset=local_experts * m.moe_ep_rank,
    )


#: Replicated per-layer weights, under the same name in the module tree and the checkpoint.
_LAYER_REPLICATED = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attention_res_norm.weight",
    "mlp_res_norm.weight",
    "self_attention_res_proj.weight",
    "mlp_res_proj.weight",
)

#: KDA's input projections, in the row order of `k3_proj_weight`.
_KDA_FUSED_PROJ = ("q_proj", "k_proj", "v_proj", "g_proj", "f_a_proj", "b_proj")

#: KDA's head-split rows (a rank's heads times the head dim), outside `k3_proj_weight`.
_KDA_HEAD_ROWS = (
    "f_b_proj.weight",
    "q_conv1d.weight",
    "k_conv1d.weight",
    "v_conv1d.weight",
    "dt_bias",
)


def _kda_rows(i: int, d) -> list[Row]:
    p, c = f"model.layers.{i}.linear_attn.", f"model.layers.{i}.self_attn."
    proj, heads = _rows(d.rank, d.kda_proj), _rows(d.rank, d.kda_heads)
    rows = [
        # q, k, v, g: the rank's heads' rows; f_a: whole (replicated); b: the rank's heads.
        Row(
            p + "k3_proj_weight",
            tuple(c + name + ".weight" for name in _KDA_FUSED_PROJ),
            transform=_cat_rows_padded,
            cuts=(proj, proj, proj, proj, None, heads),
        )
    ]
    rows += [Row(p + name, (c + name,), proj) for name in _KDA_HEAD_ROWS]
    rows += [
        Row(p + "o_norm.weight", (c + "o_norm.weight",)),
        Row(p + "o_proj.weight", (c + "o_proj.weight",), _cols(d.rank, d.kda_proj)),
        # A rank holding every head reads the padded A_log whole and checks the padding.
        Row(p + "A_log", (c + "A_log",), _rows(d.rank, d.kda_heads))
        if d.tp > 1
        else Row(p + "A_log", (c + "A_log",), None, _unpad_a_log),
    ]
    return rows


def _mla_rows(i: int, d) -> list[Row]:
    p, c = f"model.layers.{i}.self_attn.mixer.", f"model.layers.{i}.self_attn."
    return [
        # q_a, kv_a: whole (replicated); g: the rank's heads' rows.
        Row(
            p + "k3_ag_weight",
            (c + "q_a_proj.weight", c + "kv_a_proj_with_mqa.weight", c + "g_proj.weight"),
            transform=_cat_rows,
            cuts=(None, None, _rows(d.rank, d.mla_heads * d.v_dim)),
        ),
        Row(p + "q_a_layernorm.weight", (c + "q_a_layernorm.weight",)),
        Row(p + "kv_a_layernorm.weight", (c + "kv_a_layernorm.weight",)),
        Row(p + "q_b_proj.weight", (c + "q_b_proj.weight",), _rows(d.rank, d.mla_heads * d.qk_dim)),
        Row(
            p + "kv_b_proj.weight",
            (c + "kv_b_proj.weight",),
            _rows(d.rank, d.mla_heads * (d.nope + d.v_dim)),
            _regroup_kv_b,
        ),
        Row(p + "o_proj.weight", (c + "o_proj.weight",), _cols(d.rank, d.mla_heads * d.v_dim)),
    ]


def _gated_mlp_rows(prefix: str, inter: int, rank: int) -> list[Row]:
    """A GatedMLP `inter` rows wide per rank, at share `rank`: gate and up rows concatenated, down columns."""
    return [
        Row(
            prefix + "gate_up_proj.weight",
            (prefix + "gate_proj.weight", prefix + "up_proj.weight"),
            _rows(rank, inter),
            _cat_rows,
        ),
        Row(prefix + "down_proj.weight", (prefix + "down_proj.weight",), _cols(rank, inter)),
    ]


def _moe_rows(i: int, d) -> list[Row]:
    p = f"model.layers.{i}.block_sparse_moe."
    replicated = (
        "gate.weight",
        "gate.e_score_correction_bias",
        "routed_expert_down_proj.weight",
        "routed_expert_up_proj.weight",
        "routed_expert_norm.weight",
    )
    return [Row(p + name, (p + name,)) for name in replicated] + _gated_mlp_rows(
        p + "shared_experts.", d.shared_inter, d.rank
    )


def _dense_rows(i: int, layer: nn.Module, d) -> list[Row]:
    """The dense MLP over the layer's own TP group (`mlp_tp_size`): a group smaller than TP repeats, so a rank's
    share is its TP rank modulo the group size."""
    mlp_tp = layer.mlp_tp_size
    return _gated_mlp_rows(f"model.layers.{i}.mlp.", d.dense_inter // mlp_tp, d.rank % mlp_tp)


def _resident(model, layers) -> list[int]:
    """The layers whose weights are resident (a layer-wise benchmark removes the others' weights)."""
    return [i for i in layers if not getattr(model.model.layers[i], "_weights_removed", False)]


def manifest(model, d: SimpleNamespace) -> list[Row]:
    """Every parameter of the text model and the shell's lm_head, one row each, at this rank's share."""
    rows = [
        Row(name, (name,))
        for name in (
            "model.embed_tokens.weight",
            "model.norm.weight",
            "model.output_attn_res_norm.weight",
            "model.output_attn_res_proj.weight",
        )
    ]
    lm_rows = model.lm_head.weight.shape[0]
    start = 0 if lm_rows == d.vocab else d.rank * lm_rows
    lm_cut = (slice(start, min(start + lm_rows, d.vocab)),)
    rows.append(Row("lm_head.weight", ("lm_head.weight",), lm_cut, _lm_head_pad(lm_rows)))
    for i in _resident(model, range(d.num_layers)):
        rows += [
            Row(f"model.layers.{i}.{name}", (f"model.layers.{i}.{name}",))
            for name in _LAYER_REPLICATED
        ]
        rows += _kda_rows(i, d) if i in d.kda_layers else _mla_rows(i, d)
        rows += _moe_rows(i, d) if i in d.moe_layers else _dense_rows(i, model.model.layers[i], d)
    return rows


def expert_keys(i: int, expert: int) -> dict[str, str]:
    """One routed expert's checkpoint keys (without `LANG_PREFIX`), by the relayout's argument name."""
    base = f"model.layers.{i}.block_sparse_moe.experts.{expert}."
    return {arg: base + suffix for arg, suffix in _EXPERT_TENSORS.items()}


def expected_keys(model, d: SimpleNamespace, rows: list[Row]) -> set[str]:
    """Every checkpoint key this rank reads, without `LANG_PREFIX`: the rows' and this rank's experts'."""
    keys = {key for row in rows for key in row.keys}
    for i in _resident(model, d.moe_layers):
        for e in range(d.expert_offset, d.expert_offset + d.local_experts):
            keys.update(expert_keys(i, e).values())
    return keys


# --- checks ---


def _trunk_params(model) -> dict[str, nn.Parameter]:
    """The parameters the rows fill: every parameter except the ones `_NOT_ROW_*` name."""
    return {
        name: param
        for name, param in model.named_parameters()
        if not name.startswith(_NOT_ROW_PREFIXES)
        and not any(infix in name for infix in _NOT_ROW_INFIXES)
    }


def _check_supported(model, weights, d, params: dict[str, nn.Parameter]) -> None:
    """The configurations this loader does not take, each with the reason."""
    m = model.model_config.mapping
    if m.cp_size > 1:
        raise NotImplementedError(
            "Kimi K3 load: helix context parallelism (cp_size > 1) is not supported"
        )
    if m.enable_attention_dp:
        raise NotImplementedError(
            "Kimi K3 load: attention DP is not supported; this target splits attention by head"
        )
    if m.pp_size > 1:
        raise NotImplementedError("Kimi K3 load: pipeline parallelism is not supported")
    fp8 = sorted(name for name, param in params.items() if param.dtype in _FP8_DTYPES)
    if fp8 or model.model_config.quant_config_dict:
        raise NotImplementedError(
            "Kimi K3 load: FP8 attention projections are not supported; this target loads bf16 attention "
            f"(FP8 parameters: {fp8[:5]}, per-layer quant declarations: {bool(model.model_config.quant_config_dict)})"
        )
    moe_layers = _resident(model, d.moe_layers)
    if moe_layers:
        key = LANG_PREFIX + expert_keys(moe_layers[0], d.expert_offset)["w1_weight"]
        if key not in weights:
            raise NotImplementedError(
                f"Kimi K3 load: the routed experts must be packed MXFP4 ({key}); NVFP4 and other expert layouts are "
                "not supported"
            )


def _check_rows(rows: list[Row], params: dict[str, nn.Parameter]) -> None:
    """Parameter-side coverage: the rows name every parameter once, and nothing else."""
    names = [row.param for row in rows]
    assert len(names) == len(set(names)), "a parameter has two manifest rows"
    assert set(names) == set(params), (
        "manifest/parameter drift: rows without a parameter "
        f"{sorted(set(names) - set(params))[:8]}, parameters without a row {sorted(set(params) - set(names))[:8]}"
    )


def _check_keys(weights, expected: set[str]) -> None:
    """Checkpoint-side coverage. Keys outside the language model must be predicted non-loads, and every expected key
    must be present. A `language_model.*` key that is not read warns, unless it is a routed expert or under
    `model.layers.` (another rank's experts, or a checkpoint with more layers than the config)."""
    keys = list(weights.keys())
    unknown = sorted(
        k for k in keys if not k.startswith(LANG_PREFIX) and not k.startswith(PREDICTED_NON_LOAD)
    )
    assert not unknown, (
        f"{len(unknown)} checkpoint key(s) are neither language-model weights nor a predicted non-load, "
        f"first {unknown[:5]}"
    )
    present = {k[len(LANG_PREFIX) :] for k in keys if k.startswith(LANG_PREFIX)}
    assert present, f"the checkpoint has no {LANG_PREFIX}* keys"
    missing = sorted(expected - present)
    if missing:
        raise KeyError(
            f"Kimi K3 load: {len(missing)} expected checkpoint keys are missing, e.g. {missing[:10]}"
        )
    surprising = sorted(
        k
        for k in present - expected
        if ".block_sparse_moe.experts." not in k and not k.startswith("model.layers.")
    )
    if surprising:
        warnings.warn(
            f"Kimi K3 load: {len(surprising)} unmatched checkpoint keys, e.g. {surprising[:10]}"
        )


# --- the load ---


def _set_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.set_device(device)


def _run(fn: Callable, jobs: list, workers: int) -> None:
    """Run `fn(*job)` for every job on a thread pool, re-raising the first failure."""
    if not jobs:
        return
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        for future in [pool.submit(fn, *job) for job in jobs]:
            future.result()


def _fill(param: nn.Parameter, row: Row, weights, d) -> None:
    """Read one row's keys through its cut, transform, cast to the parameter's dtype and copy into the parameter."""
    _set_device(param.device)
    tensors = tuple(
        _read(weights[LANG_PREFIX + key], cut) for key, cut in zip(row.keys, row.key_cuts())
    )
    for key, t in zip(row.keys, tensors):
        if t.dtype in _FP8_DTYPES:
            raise NotImplementedError(
                f"Kimi K3 load: {key} is {t.dtype}; this target loads bf16 attention and dense weights and does not "
                "dequantize FP8 checkpoint tensors"
            )
    if row.transform is not None:
        src = row.transform(tensors if len(tensors) > 1 else tensors[0], d)
    else:
        assert len(tensors) == 1, f"{row.param}: a row of several keys needs a transform"
        src = tensors[0]
    if src.shape != param.shape:
        raise ValueError(
            f"{row.param}: checkpoint share {tuple(src.shape)} != parameter {tuple(param.shape)}"
        )
    param.data.copy_(src.to(param.dtype))


def _pack_routed_expert(
    moe: nn.Module, expert: int, slot: int, tensors: dict[str, torch.Tensor]
) -> None:
    """One routed expert's packed MXFP4 tensors (whole, as the checkpoint stores them) into its slot of the routed
    experts' buffers: `moe.pack_mxfp4_expert`'s trtllm-gen relayout (MoE-TP cut, padding, shuffle, scale
    interleave), through the buffers' own `load_packed_mxfp4_expert`."""
    moe.routed_experts.backend.load_packed_mxfp4_expert(
        global_expert_id=expert, local_slot_id=slot, **tensors
    )


def _check_expert_window(moe: nn.Module, d) -> None:
    """The routed experts' buffers hold the manifest's expert window."""
    backend = moe.routed_experts.backend
    built = (backend.slot_start, backend.expert_size_per_partition)
    assert built == (d.expert_offset, d.local_experts), (
        f"the routed experts' buffers hold experts [{built[0]}, +{built[1]}), the manifest "
        f"[{d.expert_offset}, +{d.local_experts})"
    )


def _finish_routed_experts(moe: nn.Module, d) -> None:
    """Every slot of the layer filled (the buffers are final as written; their post-load fills only the SiTU caps)."""
    del d
    moe.routed_experts.backend.check_loaded()


def _drop_file_pages(path: str) -> None:
    """Drop a closed shard file's pages from the page cache."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass


def _load_routed_experts(model, weights, d) -> None:
    """Stream this rank's routed experts into the backend slots, one shard file at a time (module docstring)."""
    layers = _resident(model, d.moe_layers)
    moes = {i: model.model.layers[i].block_sparse_moe for i in layers}
    for moe in moes.values():
        _check_expert_window(moe, d)
    jobs = [(i, d.expert_offset + slot, slot) for i in layers for slot in range(d.local_experts)]
    device = model.model.embed_tokens.weight.device
    ckpt_dir = getattr(weights, "checkpoint_dir", None) or getattr(
        model.config, "_name_or_path", None
    )
    index_path = os.path.join(ckpt_dir or "", "model.safetensors.index.json")

    def pack(i: int, expert: int, slot: int, get: Callable) -> None:
        _set_device(device)
        _pack_routed_expert(
            moes[i], expert, slot, {arg: get(key) for arg, key in expert_keys(i, expert).items()}
        )

    if jobs and ckpt_dir and os.path.isfile(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        per_file: dict[str, list] = {}
        split_jobs = []
        for job in jobs:
            files = {weight_map[LANG_PREFIX + key] for key in expert_keys(job[0], job[1]).values()}
            if len(files) == 1:
                per_file.setdefault(files.pop(), []).append(job)
            else:
                split_jobs.append((job, sorted(files)))

        def load_file(name: str, file_jobs: list) -> None:
            path = os.path.join(ckpt_dir, name)
            with safe_open(path, framework="pt", device="cpu") as fh:
                for job in file_jobs:
                    pack(*job, lambda key: fh.get_tensor(LANG_PREFIX + key))
            # Closed, so unmapped: the drop takes effect.
            _drop_file_pages(path)

        def load_split(job: tuple, files: list) -> None:
            with ExitStack() as stack:
                handles = {
                    name: stack.enter_context(
                        safe_open(os.path.join(ckpt_dir, name), framework="pt", device="cpu")
                    )
                    for name in files
                }
                pack(
                    *job,
                    lambda key: handles[weight_map[LANG_PREFIX + key]].get_tensor(
                        LANG_PREFIX + key
                    ),
                )
            for name in files:
                _drop_file_pages(os.path.join(ckpt_dir, name))

        _run(load_file, sorted(per_file.items()), EXPERT_WORKERS)
        _run(load_split, split_jobs, EXPERT_WORKERS)
    elif jobs:
        warnings.warn(
            f"Kimi K3 load: no safetensors index at '{index_path}', so the routed experts are read from the shared "
            "lazy weights instead of being streamed per shard file; every shard stays mapped for the whole load, "
            "which runs a full-size load out of host memory"
        )

        def load_layer(i: int) -> None:
            for slot in range(d.local_experts):
                pack(
                    i,
                    d.expert_offset + slot,
                    slot,
                    lambda key: _read(weights[LANG_PREFIX + key], None),
                )

        _run(load_layer, [(i,) for i in layers], EXPERT_WORKERS)
    for moe in moes.values():
        _finish_routed_experts(moe, d)


def _finalize_kda(model, d) -> None:
    """The KDA modules' derived weights once their checkpoint weights are in (`K3DecodeKDA.finalize_decode_weights`):
    the `[q | k | v | g]` and `[f_a | b | pad]` row views of `k3_proj_weight`, the conv taps in each kernel's layout
    and the fp32 constants. At the end of the load, where the built-in loader builds them, so they
    exist before any post-load hook runs."""
    for i in _resident(model, d.kda_layers):
        model.model.layers[i].linear_attn.finalize_decode_weights()


def load(model, weights) -> None:
    """Load the language model's weights into `model`, the target shell, from the engine's checkpoint dict.

    Checks coverage both ways, fills every manifest row from this rank's cut of its keys, streams this rank's routed
    experts and builds the KDA modules' derived weights (`_finalize_kda`)."""
    d = dims(model)
    params = _trunk_params(model)
    _check_supported(model, weights, d, params)
    rows = manifest(model, d)
    _check_rows(rows, params)
    _check_keys(weights, expected_keys(model, d, rows))
    _run(
        lambda row: _fill(params[row.param], row, weights, d),
        [(row,) for row in rows],
        TRUNK_WORKERS,
    )
    _load_routed_experts(model, weights, d)
    _finalize_kda(model, d)
    device = model.model.embed_tokens.weight.device
    if device.type == "cuda":
        # The lazy mappings are load-scoped; finish the device copies before the caller releases them.
        torch.cuda.synchronize(device)
