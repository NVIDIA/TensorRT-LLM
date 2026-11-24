# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight loading: Kimi K3.1 Pro (MXFP4) / sm_10x / default.

The checkpoint is the vision-language wrapper's (Moonshot's safetensors as published). Its keys split three ways:

* ``language_model.*`` goes to the text model loader (:func:`load_text_model`) with the prefix stripped. That loader
  streams the routed experts and keeps this rank's slice (whole experts under expert parallelism, the intermediate
  shard of every expert under MoE TP), packs the MXFP4 experts for the TRTLLM-Gen runner, keeps the KDA ``A_log`` /
  ``dt_bias`` in FP32, derives the MLA absorption weights, and checks its own key coverage: every expected key must be
  present, and any other key is reported unless it is another rank's routed expert or belongs to a layer the model
  does not hold (past a truncated config's last layer, or a layer whose weights were removed).
* ``vision_tower.*`` and ``mm_projector.*`` go to the vision tower's loader (``modeling.K31VisionTower``),
  built here on the first load. Where the server runs without a local vision tower (``disable_mm_encoder``, or encoder
  disaggregation) they are a predicted non-load: listed here, never read.
* Any other key fails the load, naming it, rather than being dropped.

The engine's checkpoint loader reads this checkpoint lazily (model_type ``kimi_k31`` is one of the lazy safetensors
model types): about 1.5 TB never fits in host RAM, so every value is a safetensors slice
of the mapped file and the loaders read only this rank's part of it.

This target adds no step after the load.

The checkpoint helpers:

* :class:`CheckpointWeights`: checkpoint name -> tensor (or lazy safetensors slice) for one load, with the directory it
  was read from and the name prefix it was taken under.
* :func:`strip_prefix`: the entries under one name prefix (``language_model.``), named without it.
* :func:`tp_shard`: one tensor-parallel rank's part of a checkpoint tensor along one axis.
* :func:`run_in_threads`: one function over a list of jobs on a thread pool.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from typing import Any, Callable, Dict, Iterable, List, Mapping, NamedTuple, Optional, Set, Tuple

import torch
from safetensors import safe_open
from torch import nn

from tensorrt_llm.logger import logger
from tensorrt_llm.quantization import QuantAlgo

from ..ops.moe.mxfp4_expert_pack import load_packed_mxfp4_expert

#: Threads of the per-expert loader.
_EXPERT_LOAD_THREADS = 4

LANG_PREFIX = "language_model."
#: Key prefixes of the vision tower's checkpoint weights.
VISION_TOWER_PREFIX = "vision_tower."
PROJECTOR_PREFIX = "mm_projector."
VISION_PREFIXES = (VISION_TOWER_PREFIX, PROJECTOR_PREFIX)


# ---------------------------------------------------------------------------
# Checkpoint helpers.
# ---------------------------------------------------------------------------


class CheckpointWeights(dict):
    """Checkpoint name -> tensor (or lazy safetensors slice) for one load.

    ``checkpoint_dir`` is the directory the entries were read from: the routed-expert loader re-opens its shard files
    there. ``checkpoint_prefix`` is the prefix :func:`strip_prefix` removed from the names, which that loader puts back
    to look a name up in the checkpoint's index.
    """

    checkpoint_dir: Optional[str] = None
    checkpoint_prefix: str = ""


def strip_prefix(weights: Mapping[str, Any], prefix: str) -> CheckpointWeights:
    """The entries of ``weights`` whose name starts with ``prefix``, in their order, named without the prefix.

    The result records ``prefix`` and the ``checkpoint_dir`` of ``weights`` (if it has one).
    """
    out = CheckpointWeights(
        (name[len(prefix) :], value) for name, value in weights.items() if name.startswith(prefix)
    )
    out.checkpoint_prefix = prefix
    checkpoint_dir = getattr(weights, "checkpoint_dir", None)
    if checkpoint_dir is not None:
        out.checkpoint_dir = checkpoint_dir
    return out


def tp_shard(
    tensor: torch.Tensor, tp_size: int, tp_rank: int, dim: int, device: torch.device
) -> torch.Tensor:
    """Rank ``tp_rank``'s part of ``tensor`` split ``tp_size`` ways along ``dim``, moved to ``device``.

    Every part is ``ceil(width / tp_size)`` wide except the last, which keeps the remainder (ranks past the end get an
    empty part). The whole tensor is returned when ``tp_size <= 1``, when the tensor has no axis ``dim`` (a 1-D tensor
    split along dim 1), or when that axis has width 1. A whole tensor already on ``device`` is returned as is.
    """
    if tp_size <= 1 or tensor.dim() <= dim:
        return tensor.to(device)
    width = tensor.shape[dim]
    if width == 1:
        return tensor.to(device)
    part = -(-width // tp_size)
    index = [slice(None)] * tensor.dim()
    index[dim] = slice(tp_rank * part, min((tp_rank + 1) * part, width))
    return tensor[tuple(index)].to(device)


def _brief(value: Any, depth: int = 0) -> str:
    """A short description of a job argument for the failure log: plain values as such, containers by their first
    items, anything else (a parameter, a module) by its type."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return repr(value)
    if isinstance(value, (tuple, list, set, frozenset)) and depth < 2:
        items = list(value)
        shown = ", ".join(_brief(item, depth + 1) for item in items[:4])
        return f"({shown}{', ...' if len(items) > 4 else ''})"
    return type(value).__name__


def run_in_threads(func: Callable[..., Any], jobs: Iterable[tuple], num_workers: int) -> None:
    """Call ``func(*job)`` for every job on ``num_workers`` threads and wait for all of them.

    The first failure seen as the jobs finish is logged with its job and re-raised once the pool has drained.
    """
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(func, *job): job for job in jobs}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                logger.error(f"{func.__name__} failed on job {_brief(futures[future])}")
                raise


# ---------------------------------------------------------------------------
# The checkpoint read and its split.
# ---------------------------------------------------------------------------


def split_keys(keys) -> tuple[list[str], list[str], list[str]]:
    """``(language-model keys, vision keys, unknown keys)`` of a checkpoint's key set."""
    lang, vision, unknown = [], [], []
    for key in keys:
        if key.startswith(LANG_PREFIX):
            lang.append(key)
        elif key.startswith(VISION_PREFIXES):
            vision.append(key)
        else:
            unknown.append(key)
    return lang, vision, unknown


def load(model, weights) -> None:
    """Load the checkpoint into ``model`` (the target): the vision tower, then the text model."""
    lang, vision, unknown = split_keys(weights.keys())
    if unknown:
        unknown.sort()
        raise ValueError(
            f"Kimi K3.1 checkpoint: {len(unknown)} key(s) are neither language-model "
            f"({LANG_PREFIX}*) nor vision ({', '.join(p + '*' for p in VISION_PREFIXES)}) "
            f"weights, first {unknown[:5]}"
        )
    if not lang:
        raise ValueError(f"Kimi K3.1 checkpoint: no {LANG_PREFIX}* keys")

    if model._skip_mm_encoder:
        if vision:
            logger.info(
                f"Kimi K3.1: skipping {len(vision)} vision tower / projector weights "
                "because local image encoding is disabled."
            )
    else:
        if model.mm_encoder is None:
            # modeling.py imports this module, so the tower builder is imported when it is called.
            from .modeling import build_vision_tower

            # model_config.pretrained_config points at the text config once the model is built.
            model.mm_encoder = build_vision_tower(
                model._vlm_pretrained_config, getattr(weights, "checkpoint_dir", None)
            )
        if not model.mm_encoder.load_weights(weights):
            logger.warning(
                "Kimi K3.1: the checkpoint has no vision tower weights for local image prefill."
            )
            model.mm_encoder = None

    load_text_model(model.llm, strip_prefix(weights, LANG_PREFIX))


# ---------------------------------------------------------------------------
# Text model: checkpoint mapping and loading.
# ---------------------------------------------------------------------------


_MLA_DERIVED_PARAM_SUFFIXES = (
    ".self_attn.mixer.k_b_proj_trans",
    ".self_attn.mixer.v_b_proj",
)


# The dense and shared-expert MLPs (``K31SituMLP``) hold one fused [gate | up] parameter; the checkpoint stores
# separate ``gate_proj`` / ``up_proj`` tensors, which the loader row-concatenates (see ``_gate_up_ckpt_keys``).
_GATE_UP_FUSED_SUFFIX = ".gate_up_proj.weight"


def _gate_up_ckpt_keys(fused_key: str) -> Tuple[str, str]:
    """Checkpoint ``(gate_proj, up_proj)`` keys whose row-concat loads the
    fused ``gate_up_proj`` parameter named by ``fused_key``."""
    return (
        fused_key.replace(_GATE_UP_FUSED_SUFFIX, ".gate_proj.weight"),
        fused_key.replace(_GATE_UP_FUSED_SUFFIX, ".up_proj.weight"),
    )


def _shard_head_major_param(
    name: str,
    src: torch.Tensor,
    param: torch.nn.Parameter,
    *,
    kda_tp_size: int,
    kda_tp_rank: int,
    model_tp_rank: int,
) -> torch.Tensor:
    """TP-shard a KDA ``linear_attn`` or shared-expert / dense-MLP
    ``down_proj`` checkpoint tensor down to this rank's local slice.

    Returns ``src`` unchanged when it already matches ``param`` (replicated or
    single-rank tensors) and for every other name — the caller then routes
    MLA head-shards and the shape-match copy. The slicing is
    :func:`tp_shard`'s ceil-divide split.

    - ``.linear_attn.`` names are KDA head-major tensors (``linear_attn``
      exists only on KDA layers): ``o_proj`` shards its input columns
      (dim 1), every other projection its output rows (dim 0), by
      ``kda_tp_size``.
    - a shared-expert / dense-MLP ``down_proj`` is sharded on its input
      columns (dim 1); the fused ``gate_up_proj`` is row-concatenated and returns
      before this helper, so ``down_proj`` is the only half that reaches here.
      The TP factor comes from the checkpoint-vs-param shapes; a subgroup
      smaller than model TP repeats, so the shard index is ``model_tp_rank``
      modulo the parameter's shard count.
    """
    if src.shape == param.shape:
        return src
    if ".linear_attn." in name:
        dim = 1 if name.endswith(".o_proj.weight") else 0
        return tp_shard(src, kda_tp_size, kda_tp_rank, dim, param.device)
    if name.endswith(".down_proj.weight") and (".shared_experts." in name or ".mlp." in name):
        assert src.shape[1] % param.shape[1] == 0, (
            f"{name}: checkpoint input dim {src.shape[1]} is not "
            f"divisible by param input dim {param.shape[1]}"
        )
        tp = src.shape[1] // param.shape[1]
        return tp_shard(src, tp, model_tp_rank % tp, 1, param.device)
    return src


def _has_weights(module: nn.Module) -> bool:
    """False once ``modeling_utils.remove_weights()`` has stripped a module.

    Post-load finalization walks every decoder layer, so it must skip layers
    whose parameters were dropped — the layer-wise benchmarks keep only the
    profiled slice resident.
    """
    return not getattr(module, "_weights_removed", False)


def _load_packed_mxfp4_expert(
    experts,
    base,
    ckpt_expert_idx,
    slot_expert_idx,
    local_slot_id,
    get_tensor,
) -> None:
    """Pack checkpoint expert ``ckpt_expert_idx`` into local slot ``local_slot_id`` of the routed experts
    (``slot_expert_idx``: the expert id the slot serves)."""
    if slot_expert_idx != experts.initial_local_expert_ids[local_slot_id]:
        raise ValueError(
            f"local slot {local_slot_id} expects global expert "
            f"{experts.initial_local_expert_ids[local_slot_id]}, got {slot_expert_idx}."
        )
    prefix = f"{base}.{ckpt_expert_idx}"
    load_packed_mxfp4_expert(
        experts.w3_w1_weight,
        experts.w3_w1_weight_scale,
        experts.w2_weight,
        experts.w2_weight_scale,
        local_slot_id,
        experts._packed_mxfp4_loaded_slots,
        intermediate_size=experts.intermediate_size,
        moe_tp_size=experts.tp_size,
        moe_tp_rank=experts.tp_rank,
        w1_weight=get_tensor(f"{prefix}.w1.weight_packed"),
        w1_weight_scale=get_tensor(f"{prefix}.w1.weight_scale"),
        w2_weight=get_tensor(f"{prefix}.w2.weight_packed"),
        w2_weight_scale=get_tensor(f"{prefix}.w2.weight_scale"),
        w3_weight=get_tensor(f"{prefix}.w3.weight_packed"),
        w3_weight_scale=get_tensor(f"{prefix}.w3.weight_scale"),
    )


class ExpertCkptSpec(NamedTuple):
    """How one routed-expert quantization is spelled and loaded."""

    # Per-``w{1,2,3}`` checkpoint tensor suffixes this layout stores. Every
    # combination is required: a missing one fails the key plan.
    kinds: Tuple[str, ...]
    loader: Callable[..., None]
    # Set of filled slots the loader maintains, checked after the load.
    loaded_slots_attr: str


# The MXFP4 checkpoint's routed experts (``mxfp4-pack-quantized``), served by the TRTLLM-Gen
# W4A8_MXFP4_MXFP8 runner. The loader writes each expert straight into its routed-expert slot.
EXPERT_CKPT_SPECS = {
    QuantAlgo.W4A8_MXFP4_MXFP8: ExpertCkptSpec(
        kinds=("weight_packed", "weight_scale"),
        loader=_load_packed_mxfp4_expert,
        loaded_slots_attr="_packed_mxfp4_loaded_slots",
    ),
}


def expert_ckpt_spec(quant_algo: Optional[QuantAlgo]) -> ExpertCkptSpec:
    spec = EXPERT_CKPT_SPECS.get(quant_algo)
    if spec is None:
        raise NotImplementedError(
            f"Kimi K3.1 routed experts are quantized as {quant_algo}, for which "
            "no per-expert checkpoint layout is known. Supported: "
            f"{sorted(a.name for a in EXPERT_CKPT_SPECS)}."
        )
    return spec


def _read_unquantized(weights, ckpt_key: str) -> torch.Tensor:
    """Read trunk tensor ``ckpt_key``. The target loads BF16 / FP32 trunk weights, so an FP8 tensor fails
    the load rather than being cast without its scale."""
    src = _materialize(weights[ckpt_key])
    if src.dtype == torch.float8_e4m3fn:
        raise ValueError(
            f"Kimi K3.1: {ckpt_key} is FP8 E4M3; the target loads BF16 attention / MLP / LM head"
        )
    return src


def _materialize(value) -> torch.Tensor:
    """Materialize a (possibly lazy safetensors slice) weight value."""
    if isinstance(value, torch.Tensor):
        return value
    return value[:]


_SAFETENSORS_DTYPES = {
    "U8": torch.uint8,
    "I8": torch.int8,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn,
}


def read_safetensors_slices(
    path: str, names: Iterable[str], get_buffer: Callable[[int], torch.Tensor]
) -> Optional[Dict[str, torch.Tensor]]:
    """Read the named tensors of one safetensors file with a few large preadv calls.

    The tensors' byte ranges are merged across gaps under 1 MiB and read into the
    uint8 tensor ``get_buffer(nbytes)`` returns (at least ``nbytes`` long). The result
    maps every name found in the file to a view of that buffer, valid until the buffer
    is reused. Each range starts at a buffer offset congruent to its file offset mod
    64, so the views keep the file's alignment. Returns None when a requested tensor
    has a dtype this reader does not map or the file ends early.
    """
    with open(path, "rb") as fh:
        header_len = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(header_len))
    data_start = 8 + header_len
    entries = []
    for name in names:
        meta = header.get(name)
        if meta is None:
            continue
        dtype = _SAFETENSORS_DTYPES.get(meta["dtype"])
        if dtype is None:
            return None
        start, end = meta["data_offsets"]
        entries.append((start, end, name, dtype, meta["shape"]))
    entries.sort()
    ranges = []
    for start, end, *_ in entries:
        if ranges and start - ranges[-1][1] < (1 << 20):
            ranges[-1][1] = max(ranges[-1][1], end)
        else:
            ranges.append([start, end])
    placements = []
    buf_size = 0
    for start, end in ranges:
        buf_off = (buf_size + 63) // 64 * 64 + start % 64
        placements.append(buf_off)
        buf_size = buf_off + end - start
    buf = get_buffer(buf_size)
    view = memoryview(buf.numpy())
    fd = os.open(path, os.O_RDONLY)
    try:
        for (start, end), buf_off in zip(ranges, placements):
            done = 0
            while done < end - start:
                got = os.preadv(
                    fd,
                    [view[buf_off + done : buf_off + end - start]],
                    data_start + start + done,
                )
                if got <= 0:
                    return None
                done += got
    finally:
        os.close(fd)
    tensors = {}
    range_idx = 0
    for start, end, name, dtype, shape in entries:
        while end > ranges[range_idx][1]:
            range_idx += 1
        buf_off = placements[range_idx] + start - ranges[range_idx][0]
        tensors[name] = buf[buf_off : buf_off + end - start].view(dtype).view(shape)
    return tensors


# ------------------------------------------------------------------
# Weight loading (streams the 1.5TB checkpoint; only the rank-local
# expert slice of each MoE layer is kept: whole experts under MoE EP,
# the intra-expert intermediate shard of ALL experts under MoE TP —
# in the TP case every expert tensor is read and sliced, so expect a
# correspondingly longer load).
# ------------------------------------------------------------------


def trunk_parameters(llm) -> Dict[str, torch.nn.Parameter]:
    """Named parameters of the trunk only. Spec-dec draft modules
    (the DFlash2 drafter, ``draft_model``)
    live in a separate checkpoint loaded by
    ModelLoader.load_draft_weights, not in the target checkpoint. MLA
    K/V absorb Parameters are derived by the KV-B loader and are likewise
    excluded from checkpoint jobs."""
    return {
        name: param
        for name, param in llm.named_parameters()
        if not name.startswith("draft_model.") and not name.endswith(_MLA_DERIVED_PARAM_SUFFIXES)
    }


def checkpoint_name_plan(
    llm, prefix: str
) -> Tuple[Dict[str, str], Set[str], List[Tuple[nn.Module, str]]]:
    """Return ``(name_map, expected_keys, expert_jobs)``.

    ``name_map`` maps every model parameter name to its checkpoint key
    (for fused ``gate_up_proj`` parameters the mapped key is virtual;
    the two real per-half keys come from ``_gate_up_ckpt_keys``);
    ``expected_keys`` additionally covers the rank-local per-expert MXFP4
    tensors; ``expert_jobs`` lists ``(moe_module, key_base)`` for the
    routed-expert slots. Exposed separately so the weight-name
    mapping can be dry-run without touching any tensor data.
    """
    params = trunk_parameters(llm)
    expected_keys = set()
    name_map: Dict[str, str] = {}
    for name in params:
        # The routed experts hold their weights in the runner layout, packed
        # per expert slot by the expert loader below; they do not have
        # one-to-one checkpoint parameter names.
        if ".routed_experts." in name:
            continue
        if name == "lm_head.weight":
            ckpt_key = prefix + "lm_head.weight"
        else:
            if ".linear_attn." in name:
                ckpt_key = prefix + name.replace(".linear_attn.", ".self_attn.")
            else:
                # MLA retains its runtime/mixer hierarchy; the checkpoint
                # has no intermediate ``mixer`` scope. The DSA indexer is the
                # MLA attention's ``mqa.indexer`` but sits directly under
                # ``self_attn`` in the checkpoint.
                ckpt_key = prefix + name.replace(".self_attn.mixer.", ".self_attn.").replace(
                    ".self_attn.mqa.indexer.", ".self_attn.indexer."
                )
        name_map[name] = ckpt_key
        if name.endswith(_GATE_UP_FUSED_SUFFIX):
            # Fused [gate | up] MLP layout (dense mlp / shared_experts):
            # the checkpoint stores two separate tensors.
            expected_keys.update(_gate_up_ckpt_keys(ckpt_key))
        else:
            expected_keys.add(ckpt_key)

    # Routed-expert slots (per-expert checkpoint tensors; the
    # rank-local id range — an EP slice of whole experts, or ALL experts
    # when the routed MoE is TP-sharded (moe_ep=1 -> ids 0..num_experts)).
    expert_jobs = []
    for layer_idx, layer in enumerate(llm.model.layers):
        if not getattr(layer, "is_moe", False) or not _has_weights(layer):
            continue
        moe = layer.block_sparse_moe
        base = f"{prefix}model.layers.{layer_idx}.block_sparse_moe.experts"
        kinds = moe.expert_ckpt_spec.kinds
        for expert_idx in moe.ckpt_expert_ids:
            for w in ("w1", "w2", "w3"):
                for kind in kinds:
                    expected_keys.add(f"{base}.{expert_idx}.{w}.{kind}")
        expert_jobs.append((moe, base))
    # The MLA's q_a projection is fused into its kv_a_proj_with_mqa parameter (the MLA requires it), so q_a_proj has
    # no parameter of its own; the trunk loader reads it with kv_a.
    for index, layer in enumerate(llm.model.layers):
        if getattr(layer, "is_kda", True) or not _has_weights(layer):
            continue
        expected_keys.add(f"{prefix}model.layers.{index}.self_attn.q_a_proj.weight")
    return name_map, expected_keys, expert_jobs


def load_text_model(llm, weights: Dict[str, torch.Tensor]) -> None:
    """Load the text model ``llm`` from its checkpoint entries (names without the ``language_model.`` prefix).

    Once every value is in place, each KDA layer builds its derived decode weights.
    """
    prefix = "language_model." if any(k.startswith("language_model.") for k in weights) else ""
    params = trunk_parameters(llm)
    name_map, expected_keys, expert_jobs = checkpoint_name_plan(llm, prefix)

    _validate_checkpoint_keys(llm, weights, expected_keys, prefix)
    num_params = _load_trunk_params(llm, weights, params, name_map)
    _load_expert_slices(llm, weights, expert_jobs)
    _finalize_weight_load(llm, num_params, len(expert_jobs))
    device = next(llm.parameters()).device
    if device.type == "cuda":
        # Lazy source mappings are load-scoped; finish nonblocking H2D
        # work before the caller can release the weights container.
        torch.cuda.synchronize(device)


def _validate_checkpoint_keys(
    llm, weights: Dict[str, torch.Tensor], expected_keys: Set[str], prefix: str
) -> None:
    """Key-set validation (both directions): every expected key must be
    present; any other checkpoint key warns, except the expected leftovers:
    other ranks' routed experts and the keys of layers the model does not
    hold (past a truncated config's last layer, or whose weights were
    removed)."""
    ckpt_keys = set(weights.keys())
    relevant_ckpt_keys = {
        k for k in ckpt_keys if not (k.startswith("vision_tower.") or k.startswith("mm_projector."))
    }
    missing = sorted(expected_keys - ckpt_keys)
    if missing:
        raise KeyError(
            f"Kimi K3.1 load_weights: {len(missing)} expected checkpoint "
            f"keys are missing, e.g. {missing[:10]}"
        )
    unexpected = relevant_ckpt_keys - expected_keys
    layers = llm.model.layers
    layer_prefix = f"{prefix}model.layers."

    def held_layer(key: str) -> bool:
        index = key[len(layer_prefix) :].split(".", 1)[0]
        return index.isdigit() and int(index) < len(layers) and _has_weights(layers[int(index)])

    surprising = sorted(
        k
        for k in unexpected
        if ".block_sparse_moe.experts." not in k
        and (not k.startswith(layer_prefix) or held_layer(k))
    )
    if surprising:
        logger.warning(
            f"Kimi K3.1 load_weights: {len(surprising)} unmatched "
            f"checkpoint keys, e.g. {surprising[:10]}"
        )


def _load_trunk_params(
    llm,
    weights: Dict[str, torch.Tensor],
    params: Dict[str, torch.nn.Parameter],
    name_map: Dict[str, str],
) -> int:
    """Load every non-expert trunk parameter concurrently (with the
    per-parameter TP-shard / pad / fuse conversions) and return the
    number of parameters loaded."""
    # The checkpoint stores every MLA KV-B head as interleaved [K | V]
    # rows. Runtime keeps one DeepSeek-style [all K | all V] parameter
    # instead, so context can project directly into the FMHA layout and
    # absorbed decode can take zero-copy K/V views.
    mla_mixers = [
        layer.self_attn.mixer
        for layer in llm.model.layers
        if not getattr(layer, "is_kda", True) and _has_weights(layer)
    ]
    mla_kv_b_mixers = {id(mixer.kv_b_proj.weight): mixer for mixer in mla_mixers}
    mla_fused_a_mixers = {id(mixer.kv_a_proj_with_mqa.weight): mixer for mixer in mla_mixers}
    mla_head_shard_linears = {}
    for mixer in mla_mixers:
        mla_head_shard_linears[id(mixer.q_b_proj.weight)] = mixer.q_b_proj
        mla_head_shard_linears[id(mixer.o_proj.weight)] = mixer.o_proj
        g_proj = getattr(mixer, "g_proj", None)
        if g_proj is not None:
            mla_head_shard_linears[id(g_proj.weight)] = g_proj

    device = next(llm.parameters()).device

    # MLP TP shard index. A dense MLP whose intermediate size does not
    # divide model TP uses a smaller repeated TP subgroup, so its local
    # shard rank is model tp_rank modulo the parameter's shard count.
    # Under Helix it is the repurposed mapping's (CP ranks become TP
    # ranks); the restored CP mapping's tp_rank is 0 on every rank.
    model_tp_rank = (
        getattr(llm, "_repurposed_tp_mapping", None) or llm.model_config.mapping
    ).tp_rank
    # KDA head-shard (attention-DP off): rank r loads head rows/cols
    # [r*local : (r+1)*local] of every head-major KDA tensor.
    kda_tp_size, kda_tp_rank = 1, 0
    for layer in llm.model.layers:
        if getattr(layer, "is_kda", False):
            kda_tp_size = layer.linear_attn._kda_tp_size
            kda_tp_rank = layer.linear_attn._kda_tp_rank
            break

    def load_param(name: str, param: torch.nn.Parameter) -> None:
        if device.type == "cuda":
            torch.cuda.set_device(device)
        if name.endswith(_GATE_UP_FUSED_SUFFIX):
            # Row-concat the checkpoint's separate gate_proj / up_proj
            # tensors into the fused [gate | up] parameter.
            gate_key, up_key = _gate_up_ckpt_keys(name_map[name])
            inter = param.shape[0] // 2
            # Read each half, then column(TP)-shard and row-concat. TP factor
            # from the checkpoint-vs-param shapes; a subgroup smaller than
            # model TP repeats, so the shard index is model tp_rank modulo the
            # half's shard count.
            gate_full = _read_unquantized(weights, gate_key)
            tp = gate_full.shape[0] // inter
            # tp_shard returns the whole tensor when tp <= 1, so the rank
            # needs no tp > 1 guard. Both halves shard their output rows.
            rk = model_tp_rank % tp
            gate = tp_shard(gate_full, tp, rk, 0, param.device)
            up = tp_shard(_read_unquantized(weights, up_key), tp, rk, 0, param.device)
            if gate.shape != (inter, param.shape[1]) or up.shape != gate.shape:
                raise ValueError(
                    f"{name}: checkpoint gate/up shapes "
                    f"{tuple(gate.shape)} / {tuple(up.shape)} do not "
                    f"concat to param shape {tuple(param.shape)}"
                )
            param.data[:inter].copy_(gate.to(param.dtype))
            param.data[inter:].copy_(up.to(param.dtype))
            return
        src = _read_unquantized(weights, name_map[name])
        fused_a = mla_fused_a_mixers.get(id(param))
        if fused_a is not None:
            q_key = name_map[name].replace("kv_a_proj_with_mqa.weight", "q_a_proj.weight")
            q_weight = _read_unquantized(weights, q_key)
            if q_weight.shape != (fused_a.q_lora_rank, fused_a.hidden_size) or src.shape != (
                fused_a.kv_lora_rank + fused_a.qk_rope_head_dim,
                fused_a.hidden_size,
            ):
                raise ValueError(f"{name}: checkpoint Q-A/KV-A shapes do not match MLA dimensions")
            src = torch.cat((q_weight, src), dim=0)
        if name == "lm_head.weight":
            # LMHead is vocab-sharded (TP column) + gathered; its
            # load_weights shards the full checkpoint tensor.
            llm.lm_head.load_weights(weights=[{"weight": src}])
            return
        mla_mixer = mla_kv_b_mixers.get(id(param))
        if mla_mixer is not None:
            h = mla_mixer.num_heads_tp
            n = mla_mixer.qk_nope_head_dim
            v = mla_mixer.v_head_dim
            kv = mla_mixer.kv_lora_rank
            local = mla_mixer.kv_b_proj.load_shard(src, device=param.device).view(h, n + v, kv)
            k_weight, v_weight = local.split([n, v], dim=1)
            param.data.copy_(
                torch.cat(
                    [
                        k_weight.reshape(h * n, kv),
                        v_weight.reshape(h * v, kv),
                    ],
                    dim=0,
                ).to(param.dtype)
            )
            mla_mixer.k_b_proj_trans.data.copy_(k_weight.transpose(1, 2))
            # Share the loaded BF16 V rows with context GEMM instead of keeping a
            # separate copy. Wrap the view to retain v_b_proj's parameter registration.
            v_weight = param[h * n :].view(h, v, kv)
            if mla_mixer.mapping.has_cp_helix():
                # Helix: every rank holds every head's K-B / V-B rows; the V absorption
                # runs on this rank's 1/cp of the heads.
                heads = mla_mixer.num_heads_tp_cp
                start = mla_mixer.mapping.cp_rank * heads
                v_weight = v_weight[start : start + heads]
            mla_mixer.v_b_proj = nn.Parameter(v_weight, requires_grad=False)
            return
        if name.endswith(".A_log") and src.numel() != param.numel():
            # The checkpoint pads A_log from [num_heads] to [head_dim]
            # (e.g. [96] -> [128]); the tail must be zeros. Under KDA
            # head-shard TP the param holds this rank's head range
            # instead of the full [num_heads].
            assert src.numel() > param.numel(), (name, src.shape)
            if kda_tp_size > 1:
                lo = kda_tp_rank * param.numel()
                src = src[lo : lo + param.numel()]
            else:
                tail = src[param.numel() :]
                if tail.abs().max().item() != 0.0:
                    raise ValueError(
                        f"{name}: expected zero padding beyond "
                        f"{param.numel()} entries, got nonzero tail"
                    )
                src = src[: param.numel()]
        # KDA ``linear_attn`` (head-major) and shared-expert / dense-MLP
        # ``down_proj`` shards are shape-derived; replicated tensors and
        # every other name pass through unchanged. MLA head-shards and the
        # shape-match copy are handled by the block below.
        src = _shard_head_major_param(
            name,
            src,
            param,
            kda_tp_size=kda_tp_size,
            kda_tp_rank=kda_tp_rank,
            model_tp_rank=model_tp_rank,
        )

        if src.shape != param.shape:
            # MLA q_b/g/o head-shard: the projection's own K31Linear.load_shard
            # (its COLUMN/ROW split). KV-B is handled above; KDA and
            # shared-expert/MLP shards were resolved in the pre-block above.
            mla_sharded_linear = mla_head_shard_linears.get(id(param))
            if mla_sharded_linear is not None:
                shard = mla_sharded_linear.load_shard(src, device=param.device)
                if shard.shape != param.shape:
                    raise ValueError(
                        f"{name}: MLA shard shape {tuple(shard.shape)} does "
                        f"not match param shape {tuple(param.shape)}"
                    )
                param.data.copy_(shard.to(param.dtype))
                return
            raise ValueError(
                f"{name}: shard/pad result {tuple(src.shape)} != param shape {tuple(param.shape)}"
            )
        param.data.copy_(src.to(param.dtype))

    param_jobs = [(name, params[name]) for name in name_map]
    run_in_threads(load_param, param_jobs, num_workers=8)

    logger.info(
        f"Kimi K3.1: loaded {len(mla_mixers)} MLA KV-B projections in grouped runtime layout"
    )
    return len(param_jobs)


def _load_expert_slices(
    llm,
    weights: Dict[str, torch.Tensor],
    expert_jobs: List[Tuple[nn.Module, str]],
) -> None:
    """Load the rank-local MXFP4 expert slices of every MoE layer into
    the routed-expert slots, then verify every slot was filled."""
    device = next(llm.parameters()).device
    # Startup accounting, logged once below: checkpoint reads (bytes and
    # thread-summed time) and the per-expert loader (reads, H2D
    # copy, shuffle, scale interleave; thread-summed).
    load_threads = _EXPERT_LOAD_THREADS
    load_stats = {
        "read_s": 0.0,
        "read_bytes": 0,
        "expert_s": 0.0,
        "experts": 0,
    }
    stats_lock = threading.Lock()
    load_start = time.perf_counter()

    def timed_read(get):
        def read(key):
            start = time.perf_counter()
            tensor = get(key)
            elapsed = time.perf_counter() - start
            nbytes = tensor.numel() * tensor.element_size()
            with stats_lock:
                load_stats["read_s"] += elapsed
                load_stats["read_bytes"] += nbytes
            return tensor

        return read

    def load_expert(
        moe: nn.Module,
        base: str,
        local_slot_id: int,
        expert_idx: int,
        get_tensor,
    ):
        """Load checkpoint expert ``expert_idx`` into local slot ``local_slot_id``."""
        if device.type == "cuda":
            torch.cuda.set_device(device)
        start = time.perf_counter()
        moe.expert_ckpt_spec.loader(
            moe.routed_experts,
            base,
            expert_idx,
            moe.local_expert_ids[local_slot_id],
            local_slot_id,
            timed_read(get_tensor),
        )
        with stats_lock:
            load_stats["expert_s"] += time.perf_counter() - start
            load_stats["experts"] += 1

    def load_experts_from_weights(moe: nn.Module, base: str):
        for local_slot_id, expert_idx in enumerate(moe.ckpt_expert_ids):
            load_expert(
                moe,
                base,
                local_slot_id,
                expert_idx,
                lambda key: _materialize(weights[key]),
            )

    # ---- routed-expert slots: file-grouped streaming ----
    # The shared lazy ``weights`` dict keeps every shard mapped for the
    # whole load, so the pages it touches cannot be dropped until the load
    # ends (fadvise skips mapped pages). The expert slices are ~90 GB of
    # distinct pages per rank, which with 4 ranks per node overruns the
    # job's memory limit. So the rank-local expert tensors are grouped by
    # shard file and each file is streamed through a short-lived handle:
    # open -> copy -> close (unmap) -> fadvise(DONTNEED).
    # The directory is the one the lazy loader opened; the config's
    # ``_name_or_path`` (empty under transformers 5) is only a fallback.
    ckpt_dir = getattr(weights, "checkpoint_dir", None) or getattr(
        llm.model_config.pretrained_config, "_name_or_path", None
    )
    index_path = os.path.join(ckpt_dir or "", "model.safetensors.index.json")
    checkpoint_prefix = getattr(weights, "checkpoint_prefix", "")

    def checkpoint_key(key: str) -> str:
        return f"{checkpoint_prefix}{key}"

    if expert_jobs and ckpt_dir and os.path.isfile(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        per_file: Dict[str, list] = {}
        per_file_keys: Dict[str, list] = {}
        split_file_jobs = []
        for moe, base in expert_jobs:
            spec = moe.expert_ckpt_spec
            for local_slot_id, expert_idx in enumerate(moe.ckpt_expert_ids):
                keys = [
                    f"{base}.{expert_idx}.{w}.{kind}"
                    for w in ("w1", "w2", "w3")
                    for kind in spec.kinds
                ]
                files = {weight_map[checkpoint_key(key)] for key in keys}
                job = (moe, base, local_slot_id, expert_idx)
                if len(files) == 1:
                    file_name = files.pop()
                    per_file.setdefault(file_name, []).append(job)
                    per_file_keys.setdefault(file_name, []).extend(keys)
                else:
                    split_file_jobs.append((job, files))

        def drop_file_pages(file_name: str):
            path = os.path.join(ckpt_dir, file_name)
            try:
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
            except OSError:
                pass

        # Read this rank's byte ranges of a shard (merged across gaps under
        # 1 MiB) with large preadv calls into a per-thread pinned host buffer
        # and serve its expert tensors as views of it. Tensors are stored
        # sorted by name, so experts 0..N/ep-1 (rank 0) are spread over the
        # whole shard (experts 0, 1, 10-19, 100-...) while every other rank's
        # slice is one compact region; copying them to the GPU straight out of
        # the safetensors mapping would fault the file in page by page, which
        # is random access for rank 0. Any problem (an unknown dtype, a short
        # read) falls back to the mapping.
        pread_buffers = threading.local()

        def pread_buffer(nbytes: int) -> torch.Tensor:
            buf = getattr(pread_buffers, "buf", None)
            if buf is None or buf.numel() < nbytes:
                pread_buffers.buf = None
                buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=device.type == "cuda")
                pread_buffers.buf = buf
            return buf

        def read_file_tensors(file_name: str) -> Optional[Dict[str, torch.Tensor]]:
            return read_safetensors_slices(
                os.path.join(ckpt_dir, file_name),
                [checkpoint_key(key) for key in per_file_keys[file_name]],
                pread_buffer,
            )

        def load_expert_file(file_name: str, jobs: list):
            if device.type == "cuda":
                torch.cuda.set_device(device)
            read_start = time.perf_counter()
            try:
                tensors = read_file_tensors(file_name)
            except (OSError, ValueError, KeyError, RuntimeError):
                tensors = None
            if tensors is not None:
                # Bytes are counted when the loaders take the views.
                with stats_lock:
                    load_stats["read_s"] += time.perf_counter() - read_start
                for moe, base, local_slot_id, expert_idx in jobs:
                    load_expert(
                        moe,
                        base,
                        local_slot_id,
                        expert_idx,
                        lambda key: tensors[checkpoint_key(key)],
                    )
                if device.type == "cuda":
                    # Copies out of the buffer may still be in flight;
                    # the next shard overwrites it.
                    torch.cuda.current_stream(device).synchronize()
                drop_file_pages(file_name)
                return
            path = os.path.join(ckpt_dir, file_name)
            with safe_open(path, framework="pt", device="cpu") as fh:
                for moe, base, local_slot_id, expert_idx in jobs:
                    load_expert(
                        moe,
                        base,
                        local_slot_id,
                        expert_idx,
                        lambda key: fh.get_tensor(checkpoint_key(key)),
                    )
            # Handle closed -> pages unmapped -> the drop takes effect.
            drop_file_pages(file_name)

        def load_split_file_expert(job, files):
            if device.type == "cuda":
                torch.cuda.set_device(device)
            with ExitStack() as stack:
                handles = {
                    file_name: stack.enter_context(
                        safe_open(os.path.join(ckpt_dir, file_name), framework="pt", device="cpu")
                    )
                    for file_name in files
                }

                def get_tensor(key):
                    source_key = checkpoint_key(key)
                    return handles[weight_map[source_key]].get_tensor(source_key)

                load_expert(*job, get_tensor)
            for file_name in files:
                drop_file_pages(file_name)

        run_in_threads(load_expert_file, sorted(per_file.items()), num_workers=load_threads)
        run_in_threads(load_split_file_expert, split_file_jobs, num_workers=load_threads)
    else:
        # Falling back is a silent loss of the whole point of the block
        # above: the shared lazy dict keeps every shard mapped, which is
        # the OOM this streaming path exists to avoid. Say so.
        if expert_jobs:
            logger.warning(
                f"Kimi K3.1: no safetensors index at '{index_path}', so routed "
                "experts are loaded from the shared lazy weight dict instead "
                "of being streamed per shard file. Every shard stays mapped "
                "for the whole load, which can exhaust host memory with several ranks per node."
            )
        run_in_threads(load_experts_from_weights, expert_jobs, num_workers=load_threads)

    for moe, _ in expert_jobs:
        spec = moe.expert_ckpt_spec
        experts = moe.routed_experts
        loaded_slots = getattr(experts, spec.loaded_slots_attr, set())
        expected_slots = set(range(experts.expert_size_per_partition))
        if loaded_slots != expected_slots:
            missing_slots = sorted(expected_slots - loaded_slots)
            raise RuntimeError(
                "Kimi K3.1 streaming expert loading did not fill all routed-expert "
                f"slots; missing {missing_slots[:10]}."
            )
    wall = time.perf_counter() - load_start
    logger.info(
        f"Kimi K3.1: expert slices: {load_stats['experts']} experts, "
        f"{load_stats['read_bytes'] / 2**30:.1f} GiB read in {load_stats['read_s']:.1f}s, "
        f"per-expert load {load_stats['expert_s']:.1f}s (incl. reads, thread-summed), "
        f"wall {wall:.1f}s, "
        f"{load_threads} threads"
    )


def _finalize_weight_load(llm, num_params: int, num_moe_layers: int) -> None:
    """Build the KDA layers' derived decode weights, once the checkpoint values are in place."""
    for layer in llm.model.layers:
        if not _has_weights(layer):
            continue
        if getattr(layer, "is_kda", False):
            attention = layer.linear_attn
            attention.finalize_decode_weights()
    logger.info(
        f"Kimi K3.1: loaded {num_params} parameters and expert slices of {num_moe_layers} layers"
    )
