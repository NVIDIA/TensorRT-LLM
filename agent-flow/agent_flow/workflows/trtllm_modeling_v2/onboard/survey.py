# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What this checkpoint demands, and what this build can already supply.

- :func:`read_checkpoint_surface` reads ``config.json`` and reports what the
  model *structurally demands*. Pure Python, no GPU, no imports of the engine.
- :func:`run_inventory_probe` runs on the real host and reports what the
  build *supplies*.

**The join between them is judgement and stays in prose**: the survey lays
both columns in front of the surveyor and lets the surveyor join them.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..infra.environment import RunEnvironment

#: The inventory probe imports the engine, so it pays a real import cost, but
#: it runs no kernel and allocates nothing. Anything slower than this is not
#: the probe.
INVENTORY_TIMEOUT_S = 600

#: What the probe prints its JSON payload behind, so the parser can find it
#: in a log that also carries the engine's import-time chatter.
MARKER = "__SURVEY_JSON__"


# --------------------------------------------------------------------------
# half one: what the checkpoint demands
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Demand:
    """One structural property of the checkpoint, and what it obliges.

    ``keys`` is the evidence — the ``config.json`` keys this was read off.
    """

    name: str
    detail: str
    keys: tuple[str, ...] = ()

    def __str__(self) -> str:
        where = f" [{', '.join(self.keys)}]" if self.keys else ""
        return f"{self.name}: {self.detail}{where}"


@dataclass(frozen=True)
class CheckpointSurface:
    """Everything the config declares that a target has to answer for."""

    architecture: str
    model_type: str
    demands: tuple[Demand, ...] = ()
    unrecognized: tuple[str, ...] = ()
    submodels: tuple[str, ...] = ()

    @property
    def demand_names(self) -> tuple[str, ...]:
        return tuple(d.name for d in self.demands)


def _num(config: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = config.get(name)
        if value is not None:
            return value
    return None


def _present(config: Mapping[str, Any], *names: str) -> tuple[str, ...]:
    return tuple(n for n in names if config.get(n) is not None)


# Keys that carry no structural obligation: sizes, names, ids, bookkeeping.
# Anything outside this set and outside a probe's claim is reported as
# unrecognized.
_INERT_KEYS = frozenset(
    {
        "architectures",
        "attention_bias",
        "attention_dropout",
        "auto_map",
        "bos_token_id",
        "dtype",
        "eos_token_id",
        "hidden_size",
        "id2label",
        "initializer_range",
        "intermediate_size",
        "label2id",
        "max_position_embeddings",
        "model_type",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "pad_token_id",
        "tie_word_embeddings",
        "torch_dtype",
        "transformers_version",
        "unsloth_version",
        "use_cache",
        "use_return_dict",
        "vocab_size",
    }
)

# Prefixes whose keys a probe below has already accounted for. Listed as
# prefixes rather than exact names because config authors spell the same
# concept a dozen ways (`n_routed_experts`, `num_local_experts`,
# `moe_num_experts`).
_CLAIMED_PREFIXES = (
    "attn_",
    "conv_",
    "decoder_sparse_",
    "draft",
    "dspark_",
    "expert",
    "final_logit",
    "full_attention_",
    "full_attn",
    "head_dim",
    "local_",
    "hidden_act",
    "kv_lora",
    "layer_norm",
    "layer_types",
    "linear_",
    "mamba",
    "mlp_only",
    "moe",
    "mtp_num",
    "n_group",
    "n_routed",
    "n_shared",
    "norm_topk",
    "num_experts",
    "num_local_experts",
    "num_nextn",
    "partial_rotary",
    "q_lora",
    "quantization",
    "rms_norm",
    "rope",
    "routed_scaling",
    "scoring_func",
    "shared_expert",
    "sink",
    "sliding_window",
    "ssm_",
    "swa_",
    "topk_",
    "use_mrope",
    "use_qk_norm",
    "use_sliding_window",
    "v_head_dim",
)

# Keys a probe claims by PATTERN rather than by prefix, because the concept
# sits in the MIDDLE of the name (`add_swa_attention_sink_bias`).
_CLAIMED_SUBSTRINGS = ("attention_sink", "layer_pattern")

_SUBMODEL_KEYS = ("vision_config", "audio_config", "text_config", "speech_config")

# Per-layer attention pattern keys: a list with one entry per decoder layer
# saying which KIND of attention that layer runs. Named rather than matched by
# prefix because `moe_layer_freq` has the same shape and asks a different
# question; any `*_layer_pattern` spelling is accepted on top of these.
_HYBRID_PATTERN_KEYS = ("hybrid_layer_pattern", "layer_types", "full_attention_idx")

# The geometry a hybrid stack may redeclare per layer kind. When one of these
# exists under a `swa_` / `local_` prefix and disagrees with its bare
# counterpart, the two kinds are different SHAPES rather than one shape under
# two masks.
_HYBRID_GEOMETRY_KEYS = (
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "v_head_dim",
    "rope_theta",
)
_HYBRID_KIND_PREFIXES = ("swa_", "local_")


def _repeating_period(values: Sequence[Any]) -> int | None:
    """The smallest ``p`` with ``values[i] == values[i % p]`` for every ``i``.

    ``None`` when there is none. Candidates stop at half the length, because a
    "period" that occurs once is not a period.
    """
    for period in range(1, len(values) // 2 + 1):
        if all(value == values[index % period] for index, value in enumerate(values)):
            return period
    return None


def _sink_layer_kind(key: str) -> str:
    """``add_swa_attention_sink_bias`` -> ``swa``: which layers carry the sink."""
    head = key.split("attention_sink", 1)[0]
    if head.startswith("add_"):
        head = head[len("add_") :]
    # An unprefixed spelling names no kind, which is itself the answer: the
    # sink is on the whole stack.
    return head.strip("_") or "all"


def read_checkpoint_surface(config: Mapping[str, Any]) -> CheckpointSurface:
    """Report what this checkpoint structurally demands of a target."""
    # A multimodal checkpoint nests the language model. This workflow's scope
    # block usually names the text tower, and reading the outer config for
    # attention geometry would describe the vision encoder instead.
    submodels = tuple(k for k in _SUBMODEL_KEYS if isinstance(config.get(k), dict))
    inner = config.get("text_config")
    scoped: Mapping[str, Any] = inner if isinstance(inner, dict) else config

    demands: list[Demand] = []

    heads = _num(scoped, "num_attention_heads", "n_head")
    kv_heads = _num(scoped, "num_key_value_heads", "num_kv_heads")
    if heads and kv_heads:
        shape = "MHA" if kv_heads == heads else ("MQA" if kv_heads == 1 else "GQA")
        demands.append(
            Demand(
                "attention.shape",
                f"{shape}, {heads} query heads over {kv_heads} kv head(s)",
                ("num_attention_heads", "num_key_value_heads"),
            )
        )
    if scoped.get("head_dim"):
        head_dim_detail = f"{scoped['head_dim']}, declared not derived"
        head_dim_keys: tuple[str, ...] = ("head_dim",)
        # Asymmetric qk / v head dimensions, stated as its own clause because
        # it is a different question from "what is the head dim".
        v_head_dim = scoped.get("v_head_dim")
        if v_head_dim is not None and v_head_dim != scoped["head_dim"]:
            head_dim_detail += (
                f"; v_head_dim={v_head_dim} differs from head_dim={scoped['head_dim']} "
                f"— asymmetric qk/v head dimensions, which not every attention "
                f"kernel admits"
            )
            head_dim_keys = ("head_dim", "v_head_dim")
        demands.append(Demand("attention.head_dim", head_dim_detail, head_dim_keys))
    if keys := _present(scoped, "kv_lora_rank", "q_lora_rank"):
        demands.append(
            Demand(
                "attention.mla",
                "compressed latent KV — a different cache layout and a different "
                "op family from ordinary paged attention",
                keys,
            )
        )
    # Deliberately *not* keyed on `layer_types`: that key declares which kind
    # of layer sits where, which is a hybrid stack and not a window.
    if keys := _present(scoped, "sliding_window", "use_sliding_window", "sliding_window_pattern"):
        demands.append(
            Demand(
                "attention.windowing",
                ", ".join(f"{k}={scoped[k]!r}" for k in keys)[:200],
                keys,
            )
        )
    # A per-layer pattern is not a window: a hybrid stack can be layers of two
    # different kinds whose KV head counts and RoPE bases disagree.
    #
    # **The survey reports the pattern and refuses to interpret it.** Which
    # value means sliding-window and which means full attention is the config
    # author's private convention.
    pattern_keys = tuple(
        key
        for key in scoped
        if isinstance(key, str)
        and (key in _HYBRID_PATTERN_KEYS or key.endswith("_layer_pattern"))
        and isinstance(scoped[key], Sequence)
        and not isinstance(scoped[key], str)
        and len(scoped[key]) > 1
    )
    if pattern_keys:
        pattern_key = pattern_keys[0]
        values = list(scoped[pattern_key])
        if pattern_key.endswith("_idx"):
            # An INDEX list, not a per-layer list: it names the layers of one
            # kind and leaves the rest implied.
            layers = _num(scoped, "num_hidden_layers")
            hybrid_detail = (
                f"{layers} layers, {len(values)} of them named by `{pattern_key}` "
                f"and {layers - len(values)} not"
                if isinstance(layers, int) and layers >= len(values)
                else f"{len(values)} layer(s) named by `{pattern_key}`"
            )
        else:
            counts = Counter(str(value) for value in values)
            tally = ", ".join(f"{n} of `{value}`" for value, n in counts.most_common())
            period = _repeating_period(values)
            shape = (
                f"repeating every {period} layer(s)"
                if period is not None
                else "with no fixed period"
            )
            hybrid_detail = f"{len(values)} layers in a pattern {shape}: {tally}"
        hybrid_detail += (
            "; which value means which kind of attention is the config's own "
            "convention and is a question for the plan phase, not something this "
            "survey can know"
        )
        differing: list[str] = []
        agreeing: list[str] = []
        geometry_keys: list[str] = []
        for name in _HYBRID_GEOMETRY_KEYS:
            base = scoped.get(name)
            for prefix in _HYBRID_KIND_PREFIXES:
                counterpart = scoped.get(prefix + name)
                if counterpart is None:
                    continue
                geometry_keys.append(prefix + name)
                stated = f"{prefix}{name}={counterpart} against {name}={base}"
                (agreeing if counterpart == base else differing).append(stated)
        if differing:
            hybrid_detail += (
                f". The two kinds declare DIFFERENT geometry: {', '.join(differing)} "
                f"— two shapes in one stack, not one shape under two masks"
            )
        elif agreeing:
            hybrid_detail += f". Per-kind geometry is declared and agrees: {', '.join(agreeing)}"
        demands.append(
            Demand("attention.hybrid", hybrid_detail, pattern_keys + tuple(geometry_keys))
        )

    # Distinct from `attention.sinks` below, which reads the count-of-sink-tokens
    # spelling: this one is the per-layer-kind boolean.
    sink_keys = tuple(
        key for key in sorted(scoped) if isinstance(key, str) and "attention_sink" in key
    )
    if sink_keys:
        carried = ", ".join(f"{_sink_layer_kind(key)} layers={scoped[key]}" for key in sink_keys)
        demands.append(
            Demand(
                "attention.sink",
                f"a learned sink term in the softmax denominator, declared per layer "
                f"kind: {carried}",
                sink_keys,
            )
        )

    if keys := _present(scoped, "attn_logit_softcapping", "final_logit_softcapping"):
        demands.append(Demand("attention.softcap", "logit soft-capping in the softmax", keys))
    if keys := _present(scoped, "sinks", "attn_sinks", "num_sinks"):
        demands.append(Demand("attention.sinks", "extra softmax-denominator logits", keys))
    if keys := _present(
        scoped,
        "indexer_budget",
        "indexer_n_heads",
        "indexer_head_dim",
        "index_topk",
        "index_n_heads",
        "sparse_attention_config",
    ):
        budget = scoped.get("indexer_budget") or scoped.get("index_topk")
        detail = "a learned index selects which past positions each query attends to"
        if budget:
            detail += f"; budget={budget}"
        demands.append(Demand("attention.indexer", detail, keys))
    if keys := _present(scoped, "use_qk_norm", "qk_layernorm", "qk_norm"):
        demands.append(Demand("attention.qk_norm", "per-head q/k normalization before RoPE", keys))

    # transformers 5.x carries the whole RoPE description in `rope_parameters`;
    # earlier configs use `rope_scaling`. Accept both spellings, and remember
    # which one was read so the citation is not a lie.
    rope_key = "rope_parameters" if isinstance(scoped.get("rope_parameters"), Mapping) else None
    if rope_key is None and isinstance(scoped.get("rope_scaling"), Mapping):
        rope_key = "rope_scaling"
    rope_block: Mapping[str, Any] = scoped.get(rope_key) if rope_key else {}  # type: ignore[assignment]

    if rope_block:
        kind = rope_block.get("rope_type") or rope_block.get("type") or "unknown"
        theta = rope_block.get("rope_theta") or scoped.get("rope_theta")
        if str(kind) in ("default", "unknown") and theta is not None:
            demands.append(Demand("rope.plain", f"theta={theta}", (rope_key,)))
        else:
            demands.append(Demand("rope.scaling", f"{kind} — not plain RoPE", (rope_key,)))
    elif scoped.get("rope_theta") is not None:
        demands.append(Demand("rope.plain", f"theta={scoped['rope_theta']}", ("rope_theta",)))
    if scoped.get("partial_rotary_factor") is not None or rope_block.get("partial_rotary_factor"):
        factor = scoped.get("partial_rotary_factor") or rope_block.get("partial_rotary_factor")
        demands.append(
            Demand(
                "rope.partial",
                f"rotary applied to {factor} of each head",
                ("partial_rotary_factor",),
            )
        )
    if scoped.get("use_mrope") or rope_block.get("mrope_section"):
        section = rope_block.get("mrope_section")
        detail = "multimodal sectioned RoPE"
        if section:
            detail += f"; sections={section}"
            if rope_block.get("mrope_interleaved"):
                detail += ", interleaved"
        demands.append(Demand("rope.mrope", detail, (rope_key or "rope_scaling",)))

    act = (
        scoped.get("hidden_act")
        or scoped.get("hidden_activation")
        or scoped.get("activation_function")
    )
    if act:
        demands.append(Demand("mlp.activation", str(act), ("hidden_act",)))

    experts = _num(
        scoped, "num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts"
    )
    if experts:
        topk = _num(scoped, "num_experts_per_tok", "moe_topk", "topk", "n_group")
        shared = _num(scoped, "n_shared_experts", "shared_expert_intermediate_size")
        bits = [f"{experts} experts", f"top-{topk}" if topk else "top-k unstated"]
        if shared:
            bits.append("plus a shared expert")
        if scoped.get("scoring_func"):
            bits.append(f"scoring={scoped['scoring_func']}")
        # The scoring function says how an expert is SCORED; these three say
        # how the top-k is SELECTED, and they pick a different routing entry
        # rather than a different constant.
        if scoped.get("topk_method"):
            bits.append(f"topk_method={scoped['topk_method']}")
        if scoped.get("n_group") is not None:
            bits.append(f"n_group={scoped['n_group']}")
        if scoped.get("topk_group") is not None:
            bits.append(f"topk_group={scoped['topk_group']}")
        if scoped.get("norm_topk_prob") is not None:
            bits.append(f"norm_topk_prob={scoped['norm_topk_prob']}")
        demands.append(Demand("moe.routing", ", ".join(bits), ("num_experts",)))
    if scoped.get("mlp_only_layers") or scoped.get("decoder_sparse_step"):
        demands.append(
            Demand(
                "moe.interleave",
                "dense and sparse layers interleave — two MLP shapes in one stack",
                _present(scoped, "mlp_only_layers", "decoder_sparse_step"),
            )
        )

    if keys := _present(
        scoped,
        "mamba_d_state",
        "mamba_expand",
        "ssm_state_size",
        "conv_kernel",
        "linear_conv_kernel_dim",
    ):
        demands.append(
            Demand(
                "recurrent.state",
                "a recurrent/SSM state plane — needs a state cache manager, not a KV "
                "page pool, and the two are managed by different code",
                keys,
            )
        )
    layer_types = scoped.get("layer_types")
    if isinstance(layer_types, Sequence) and not isinstance(layer_types, str):
        kinds = sorted({str(t) for t in layer_types})
        if len(kinds) > 1:
            demands.append(
                Demand("stack.hybrid", f"layer kinds interleave: {kinds}", ("layer_types",))
            )

    nextn = _num(scoped, "num_nextn_predict_layers", "num_mtp_modules", "mtp_num_hidden_layers")
    # DSpark is read FIRST because a DSpark checkpoint can carry both
    # spellings: its DSpark stages live under `mtp.*` beside
    # `num_nextn_predict_layers`, and the MTP reading of that key builds the
    # wrong drafter.
    if dspark := _present(
        scoped,
        "dspark_block_size",
        "dspark_noise_token_id",
        "dspark_target_layer_ids",
        "dspark_markov_rank",
    ):
        demands.append(
            Demand(
                "speculation.heads",
                f"a DSpark block drafter (block {scoped.get('dspark_block_size', '?')}, "
                f"target layers {scoped.get('dspark_target_layer_ids', '?')}) is the "
                f"checkpoint's own, stored as mtp.* beside num_nextn_predict_layers="
                f"{nextn} — the feature is `dspark`, NOT `mtp`, which would build chained "
                f"one-token heads over weights trained as one block pass",
                dspark,
            )
        )
    elif nextn:
        demands.append(
            Demand(
                "speculation.heads",
                f"{nextn} in-checkpoint predictor layer(s) — weights exist for a "
                f"speculator, so MTP is the checkpoint's own and not a bolt-on",
                ("num_nextn_predict_layers",),
            )
        )

    quant = config.get("quantization_config") or scoped.get("quantization_config")
    if isinstance(quant, Mapping):
        algo = quant.get("quant_algo") or quant.get("quant_method") or "unstated"
        kv = quant.get("kv_cache_quant_algo")
        bits = [str(algo)]
        if kv:
            bits.append(f"kv={kv}")
        if quant.get("group_size"):
            bits.append(f"group={quant['group_size']}")
        excluded = (
            quant.get("exclude_modules")
            or quant.get("ignore")
            or quant.get("modules_to_not_convert")
        )
        if excluded:
            bits.append(f"{len(excluded)} module(s) held out")
        demands.append(Demand("quantization", ", ".join(bits), ("quantization_config",)))
    else:
        demands.append(
            Demand("quantization", "none declared — weights load at their stored dtype", ())
        )

    norm_keys = _present(scoped, "rms_norm_eps", "layer_norm_eps", "layer_norm_epsilon", "norm_eps")
    if norm_keys:
        style = "RMSNorm" if any("rms" in k for k in norm_keys) else "LayerNorm"
        demands.append(Demand("norm.style", style, norm_keys))

    if keys := _present(
        scoped, "hc_count", "hc_lowrank", "num_residual_streams", "hyper_connection_count"
    ):
        count = scoped.get("hc_count") or scoped.get("num_residual_streams")
        detail = "parallel residual streams mixed at every block boundary"
        if count:
            detail = f"{count} " + detail
        if scoped.get("hc_lowrank"):
            detail += f"; low-rank mixer width {scoped['hc_lowrank']}"
        demands.append(Demand("residual.hyper_connection", detail, keys))

    # Keyed on the OUTER config, because `scoped` is the text tower by then and
    # the vision config is its sibling, not its child.
    tower_keys = tuple(
        k for k in ("vision_config", "audio_config", "speech_config") if config.get(k)
    )
    token_keys = _present(
        config, "image_token_id", "video_token_id", "audio_token_id", "vision_start_token_id"
    )
    if tower_keys or token_keys:
        towers = ", ".join(k.replace("_config", "") for k in tower_keys) or "unnamed"
        detail = f"{towers} tower in front of the language model"
        if token_keys:
            detail += f"; placeholder tokens declared ({', '.join(token_keys)})"
        demands.append(Demand("multimodal.tower", detail, tower_keys + token_keys))

    # A secondary table whose size dwarfs the vocabulary: the signature of an
    # n-gram / product-key / memory-layer embedding.
    primary = _num(scoped, "vocab_size") or 0
    for key, value in sorted(scoped.items()):
        if key == "vocab_size" or "vocab_size" not in key:
            continue
        size = value if isinstance(value, int) and not isinstance(value, bool) else None
        if size is None or primary <= 0 or size < primary * 10:
            continue
        demands.append(
            Demand(
                "embedding.huge_table",
                f"{key}={size:,} against vocab_size={primary:,} — a secondary table "
                f"{size // primary}x the vocabulary, so residency is a decision and "
                f"not a default",
                (key, "vocab_size"),
            )
        )
        break

    unrecognized = tuple(
        sorted(
            key
            for key in scoped
            if isinstance(key, str)
            and key not in _INERT_KEYS
            and key not in _SUBMODEL_KEYS
            and not key.startswith("_")
            and not key.startswith(_CLAIMED_PREFIXES)
            and not any(claimed in key for claimed in _CLAIMED_SUBSTRINGS)
        )
    )

    architectures = config.get("architectures") or []
    return CheckpointSurface(
        architecture=str(architectures[0]) if architectures else "",
        model_type=str(config.get("model_type") or ""),
        demands=tuple(demands),
        unrecognized=unrecognized,
        submodels=submodels,
    )


# --------------------------------------------------------------------------
# half two: what the build supplies
# --------------------------------------------------------------------------

#: An entry's ``ARCHS`` names this architecture: its cells were driven here.
RECEIPT_PASSED = "passed"
#: This architecture is not in ``ARCHS`` (or the entry declares none): unknown.
RECEIPT_ABSENT = "absent"


@dataclass(frozen=True)
class CatalogEntry:
    """One catalog entry as the host actually holds it.

    ``scope`` is empty for the shared catalog and ``<family>/<target>`` for an
    op that target keeps in its own ``ops/``.
    """

    path: str
    impl: str = ""
    receipt: str = RECEIPT_ABSENT
    impl_resolves: bool = True
    scope: str = ""
    #: How many ``Cell`` s the entry certifies; 0 when it states none.
    cells: int = 0

    @property
    def name(self) -> str:
        return Path(self.path).stem

    @property
    def category(self) -> str:
        parts = Path(self.path).parts
        return parts[0] if len(parts) > 1 else ""

    @property
    def certifiable(self) -> bool:
        """Every shared entry is an ``OpWrapper`` with cells to certify."""
        return not self.scope

    @property
    def spendable(self) -> bool:
        """Usable without a kernel writer dispatch: shared, certified here, and still wired."""
        if not self.impl_resolves or self.scope:
            return False
        return self.receipt == RECEIPT_PASSED


@dataclass(frozen=True)
class Inventory:
    """What the host supplies, measured rather than assumed."""

    arch: str = ""
    trtllm_ops: tuple[str, ...] = ()
    catalog: tuple[CatalogEntry, ...] = ()
    modules: tuple[str, ...] = ()
    #: op name -> the source that registered it. Empty when the op inventory
    #: could not be taken; a name absent from the Python plane maps to
    #: ``cpp_thop``.
    op_sources: dict[str, str] = field(default_factory=dict)
    versions: dict[str, str] = field(default_factory=dict)
    error: str = ""
    #: FlashInfer's public callables, ``flashinfer.<module>.<name>``, read off
    #: the installed package's files. Empty when it is not installed.
    flashinfer_api: tuple[str, ...] = ()
    #: callable name -> the engine files that reference it.
    flashinfer_refs: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def spendable(self) -> tuple[CatalogEntry, ...]:
        return tuple(e for e in self.catalog if e.spendable)

    @property
    def uncertified(self) -> tuple[CatalogEntry, ...]:
        """Present, wired, certifiable, but carrying no pass for this arch."""
        return tuple(
            e for e in self.catalog if e.certifiable and e.impl_resolves and not e.spendable
        )

    @property
    def promotable(self) -> tuple[CatalogEntry, ...]:
        """Ops another target keeps in its own ``ops/``."""
        return tuple(e for e in self.catalog if e.scope)

    @property
    def rotted(self) -> tuple[CatalogEntry, ...]:
        """Entries whose ``impl`` does not resolve in this build."""
        return tuple(e for e in self.catalog if not e.impl_resolves)

    @property
    def unwrapped_ops(self) -> tuple[str, ...]:
        """Registered ops that no catalog entry wraps."""
        wrapped = {e.impl.rsplit(".", 1)[-1] for e in self.catalog if e.impl}
        return tuple(op for op in self.trtllm_ops if op not in wrapped)

    @property
    def flashinfer_unreferenced(self) -> tuple[str, ...]:
        """FlashInfer callables the engine references nowhere."""
        return tuple(
            path
            for path in self.flashinfer_api
            if path.rsplit(".", 1)[-1] not in self.flashinfer_refs
        )


#: Runs on the host, inside the run's own interpreter. Kept as a string
#: rather than a module so it can be shipped through any transport that
#: carries a shell command — ``srun``, a spool file, or a broker call.
PROBE_SOURCE = r"""
import ast, json, os, re, sys
from pathlib import Path

out = {"trtllm_ops": [], "op_sources": {}, "catalog": [], "modules": [], "versions": {}, "error": ""}
repo = Path(sys.argv[1])
arch = sys.argv[2]

try:
    import torch
    out["versions"]["torch"] = torch.__version__
    import tensorrt_llm
    out["versions"]["tensorrt_llm"] = getattr(tensorrt_llm, "__version__", "?")
    # `torch.ops.<ns>` is a LAZY proxy: `dir()` lists only the names something
    # has already resolved, not every registered name. The registry is the
    # truth; `dir()` stays as a fallback for a torch without the private accessor.
    # Import the Python registration plane before enumerating: an op whose
    # `custom_ops/*.py` module has not been imported has not registered its
    # schema either.
    try:
        import importlib

        importlib.import_module("tensorrt_llm._torch.custom_ops")
        out["versions"]["custom_ops_imported"] = "yes"
    except Exception as exc:
        out["versions"]["custom_ops_imported"] = "no: %s" % type(exc).__name__
    ops = getattr(torch.ops, "trtllm", None)
    names = set()
    try:
        for schema in torch._C._jit_get_all_schemas():
            if schema.name.startswith("trtllm::"):
                names.add(schema.name.split("::", 1)[1])
    except Exception:
        pass
    if ops is not None:
        names.update(n for n in dir(ops) if not n.startswith("_"))
    out["trtllm_ops"] = sorted(names)
    out["op_enumeration"] = "schema_registry" if len(names) > 200 else "dir_fallback"
except Exception as exc:
    out["error"] = "%s: %s" % (type(exc).__name__, exc)

CATALOG_PKG = "tensorrt_llm._torch._experimental.modeling_v2.catalog"

def resolves(dotted, entry_path=None):
    if not dotted:
        return True
    if dotted.startswith("torch.ops.trtllm."):
        name = dotted.rsplit(".", 1)[-1]
        if name in out["trtllm_ops"]:
            return True
        # Not in the registry YET. An op registers when the module holding
        # its `@torch.library.custom_op` is imported, and several live
        # outside `custom_ops/`. So ask the wrapper: importing it performs
        # whatever import the op needs.
        if entry_path:
            import importlib
            module = entry_path
            if "/" in module:
                module = CATALOG_PKG + "." + module[:-3].replace("/", ".")
            try:
                importlib.import_module(module)
            except Exception:
                return False
            try:
                for schema in torch._C._jit_get_all_schemas():
                    if schema.name.startswith("trtllm::"):
                        out["trtllm_ops"].append(schema.name.split("::", 1)[1])
                out["trtllm_ops"] = sorted(set(out["trtllm_ops"]))
            except Exception:
                pass
            return name in out["trtllm_ops"]
        return False
    import importlib
    parts = dotted.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except Exception:
            continue
        for attr in parts[cut:]:
            obj = getattr(obj, attr, None)
            if obj is None:
                return False
        return True
    return False

def entry_certification(py_path):
    # An entry is an OpWrapper subclass whose class attributes are its
    # contract: `ARCHS = frozenset({Arch.SM_103})` names the architectures
    # its `CELLS` were driven on. Read from the AST rather than by importing,
    # so an entry whose op does not import here still reports what it claims.
    # Returns (archs, cells, impl).
    try:
        text = py_path.read_text(encoding="utf-8")
        tree = ast.parse(text)
    except (OSError, SyntaxError):
        return [], 0, ""
    archs, cells, call = set(), 0, ""
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(
            (isinstance(b, ast.Name) and b.id == "OpWrapper")
            or (isinstance(b, ast.Attribute) and b.attr == "OpWrapper")
            for b in node.bases
        ):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef) and stmt.name == "__call__":
                call = ast.get_source_segment(text, stmt) or ""
                continue
            if isinstance(stmt, ast.Assign):
                names = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                names = [stmt.target.id]
            else:
                continue
            if stmt.value is None:
                continue
            for sub in ast.walk(stmt.value):
                if "ARCHS" in names:
                    if isinstance(sub, ast.Attribute) and sub.attr.startswith("SM_"):
                        archs.add(sub.attr.lower())
                    elif isinstance(sub, ast.Constant) and str(sub.value).startswith("sm_"):
                        archs.add(str(sub.value))
                if "CELLS" in names and isinstance(sub, ast.Call):
                    func = sub.func
                    if (isinstance(func, ast.Name) and func.id == "Cell") or (
                        isinstance(func, ast.Attribute) and func.attr == "Cell"
                    ):
                        cells += 1
    # The op `__call__` makes, not the first op the file mentions: an entry's
    # docstring and reference routinely name its neighbours. A tier-3 entry
    # calls a FlashInfer callable instead of a trtllm op.
    match = re.search(r"torch\.ops\.trtllm\.([A-Za-z0-9_]+)", call)
    if match:
        return sorted(archs), cells, "torch.ops.trtllm." + match.group(1)
    library = re.search(r"\b(flashinfer(?:\.[A-Za-z_][A-Za-z0-9_]*)+)\s*\(", call)
    if library:
        return sorted(archs), cells, library.group(1)
    match = re.search(r"torch\.ops\.trtllm\.([A-Za-z0-9_]+)", text)
    return sorted(archs), cells, ("torch.ops.trtllm." + match.group(1)) if match else ""

catalog = repo / "tensorrt_llm/_torch/_experimental/modeling_v2/catalog"
# The shared vocabulary is the entry files themselves -- there is no index: one
# OpWrapper per file under catalog/<category>/; underscore modules (_op.py) are
# the catalog's own infrastructure.
for py in sorted(catalog.glob("*/*.py")) if catalog.is_dir() else ():
    if py.name.startswith("_"):
        continue
    out["catalog"].append({
        "path": "%s/%s" % (py.parent.name, py.name),
        "impl": entry_certification(py)[2],
    })
# A target's own ops: certified for that target, promotable for another.
models = repo / "tensorrt_llm/_torch/_experimental/modeling_v2/models"
for py in sorted(models.glob("*/*/ops/*/*.py")) if models.is_dir() else ():
    if py.name == "__init__.py":
        continue
    target_dir = py.parents[2]
    out["catalog"].append({
        "path": "%s/%s" % (py.parent.name, py.name),
        "impl": entry_certification(py)[2],
        "scope": "%s/%s" % (target_dir.parent.name, target_dir.name),
    })
for entry in out["catalog"]:
    if entry.get("scope"):
        py = models / entry["scope"] / "ops" / entry["path"]
        module = "tensorrt_llm._torch._experimental.modeling_v2.models.%s.ops.%s" % (
            entry["scope"].replace("/", "."), entry["path"][:-3].replace("/", "."))
    else:
        py = catalog / entry["path"]
        module = entry["path"]
    archs, cells, _ = entry_certification(py)
    entry["cells"] = cells
    entry["receipt"] = "passed" if arch in archs and cells else "absent"
    # Only claim an entry is unwired when the op inventory was actually
    # taken.
    entry["impl_resolves"] = True if out["error"] else bool(resolves(entry["impl"], module))

# Provenance. Every op enters `torch.ops.trtllm` through one of two planes,
# and the Python plane names its source in the file name. An op matched by
# no file is registered from C++, which is itself the answer.
custom_ops = repo / "tensorrt_llm/_torch/custom_ops"
registrations = {}
for py in sorted(custom_ops.glob("*.py")):
    try:
        body = py.read_text(encoding="utf-8")
    except OSError:
        continue
    for name in re.findall(r'"trtllm::([a-zA-Z0-9_]+)', body):
        registrations.setdefault(name, py.stem)
out["op_sources"] = {op: registrations.get(op, "cpp_thop") for op in out["trtllm_ops"]}

modules_dir = repo / "tensorrt_llm/_torch/modules"
for py in sorted(modules_dir.rglob("*.py")):
    try:
        tree = ast.parse(py.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        continue
    rel = py.relative_to(modules_dir).with_suffix("").as_posix().replace("/", ".")
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and not node.name.startswith("_"):
            out["modules"].append("%s.%s" % (rel, node.name))
out["modules"] = sorted(set(out["modules"]))

# FlashInfer's public API. The engine wraps a handful of FlashInfer kernels as
# trtllm ops; the op plane above sees those and nothing else. Read the
# installed package's files rather than importing it: no torch, no GPU, and no
# import-time side effects in a probe that must stay read-only.
FLASHINFER_INFRA = {
    "aot", "api_logging", "artifacts", "autotuner", "collect_env", "compilation_context",
    "cuda_utils", "data", "fi_trace", "jit", "profiler", "tactics_blocklist",
    "tactics_blocklist_gen", "testing", "tllm_enums", "tllm_utils", "trace", "trace_apply",
    "tuning_configs", "utils", "version",
}

def _module_level(body):
    # Statements at module level, including those under a guard (`try`,
    # `if`, `with`): FlashInfer re-exports several optional kernels that way.
    for stmt in body:
        yield stmt
        for attr in ("body", "orelse", "finalbody"):
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                yield from _module_level(getattr(stmt, attr, None) or [])
        for handler in getattr(stmt, "handlers", None) or []:
            yield from _module_level(handler.body)

def _public_names(path, reexports):
    # (where, exposed, defined) for what a module defines and, for a package
    # __init__, re-exports; `where` is the relative module a re-export comes
    # from, "" for a def. A plain module's own imports are its helpers, not
    # its API, so only a package's re-exports count.
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    names, explicit = [], None
    for stmt in _module_level(tree.body):
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(("", stmt.name, stmt.name))
        elif reexports and isinstance(stmt, ast.ImportFrom) and stmt.level >= 1 and stmt.module:
            names.extend(
                (stmt.module, alias.asname or alias.name, alias.name) for alias in stmt.names
            )
        elif isinstance(stmt, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in stmt.targets
        ):
            try:
                explicit = set(ast.literal_eval(stmt.value))
            except ValueError:
                pass
    return [
        row for row in names
        if not row[1].startswith("_") and (explicit is None or row[1] in explicit)
    ]

try:
    import importlib.util
    fi_spec = importlib.util.find_spec("flashinfer")
except (ImportError, ValueError):
    fi_spec = None
if fi_spec is None or not fi_spec.submodule_search_locations:
    out["versions"]["flashinfer"] = "not installed"
else:
    try:
        from importlib import metadata
        out["versions"]["flashinfer"] = metadata.version("flashinfer-python")
    except Exception:
        out["versions"]["flashinfer"] = "installed, version unknown"
    pkg = Path(list(fi_spec.submodule_search_locations)[0])
    submodules = sorted(
        p.stem if p.suffix == ".py" else p.name
        for p in pkg.iterdir()
        if (p.suffix == ".py" or (p / "__init__.py").is_file())
        and not p.name.startswith("_")
        and (p.stem if p.suffix == ".py" else p.name) not in FLASHINFER_INFRA
    )
    api = {}

    def _add(group, name, path):
        key = (group, name)
        if key not in api or len(path) < len(api[key]):
            api[key] = path

    # What the package itself re-exports, then what each submodule offers.
    for where, name, defined in _public_names(pkg / "__init__.py", True):
        group = where.split(".")[0]
        if where and group in submodules and name not in submodules:
            _add(group, name, "flashinfer.%s.%s" % (where, defined))
    for sub in submodules:
        init = pkg / sub / "__init__.py"
        is_pkg = init.is_file()
        for _where, name, _defined in _public_names(init if is_pkg else pkg / (sub + ".py"), is_pkg):
            _add(sub, name, "flashinfer.%s.%s" % (sub, name))
    out["flashinfer_api"] = sorted(api.values())

# Which FlashInfer callables the engine already references, and where: a
# name referenced by a `custom_ops` file is probably a trtllm op already, and
# one referenced only by a module is reachable through the kernel plane.
fi_refs = {}
torch_tree = repo / "tensorrt_llm/_torch"
for py in sorted(torch_tree.rglob("*.py")) if torch_tree.is_dir() else ():
    try:
        text = py.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        continue
    if "flashinfer" not in text:
        continue
    try:
        tree = ast.parse(text)
    except SyntaxError:
        continue
    names, roots = set(), {"flashinfer"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "flashinfer":
                    roots.add(alias.asname or "flashinfer")
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and (
            (node.module or "").split(".")[0] == "flashinfer"
        ):
            for alias in node.names:
                names.add(alias.name)
                roots.add(alias.asname or alias.name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            base = node.value
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name) and base.id in roots:
                names.add(node.attr)
    rel = py.relative_to(repo).as_posix()
    for name in names:
        fi_refs.setdefault(name, []).append(rel)
out["flashinfer_refs"] = {name: sorted(files) for name, files in sorted(fi_refs.items())}

print("__SURVEY_JSON__" + json.dumps(out))
"""


def probe_command(environment: RunEnvironment, *, repo: str | Path, arch: str) -> str:
    """The literal command that takes the inventory, fully prefixed.

    Launched as **one** process whatever the run's world size: this is
    read-only introspection, so nothing in it wants a rank per device.
    """
    return environment.python_command(
        ["-c", PROBE_SOURCE, str(repo), str(arch)], single_process=True
    )


def parse_inventory(stdout: str, *, arch: str = "") -> Inventory:
    """Pull the probe's payload out of a log that also carries engine chatter."""
    match = re.search(rf"^{re.escape(MARKER)}(.*)$", stdout, re.MULTILINE)
    if not match:
        tail = "\n".join(stdout.splitlines()[-15:])
        return Inventory(arch=arch, error=f"probe printed no payload. last lines:\n{tail}")
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        return Inventory(arch=arch, error=f"probe payload is not JSON: {exc}")
    entries = tuple(
        CatalogEntry(
            path=str(e.get("path", "")),
            impl=str(e.get("impl", "")),
            receipt=str(e.get("receipt", RECEIPT_ABSENT)),
            impl_resolves=bool(e.get("impl_resolves", True)),
            scope=str(e.get("scope", "") or ""),
            cells=int(e.get("cells", 0) or 0),
        )
        for e in data.get("catalog", ())
    )
    return Inventory(
        arch=arch,
        trtllm_ops=tuple(str(o) for o in data.get("trtllm_ops", ())),
        catalog=entries,
        modules=tuple(str(m) for m in data.get("modules", ())),
        op_sources={str(k): str(v) for k, v in (data.get("op_sources") or {}).items()},
        versions={str(k): str(v) for k, v in (data.get("versions") or {}).items()},
        error=str(data.get("error", "")),
        flashinfer_api=tuple(str(p) for p in data.get("flashinfer_api", ())),
        flashinfer_refs={
            str(k): tuple(str(f) for f in v) for k, v in (data.get("flashinfer_refs") or {}).items()
        },
    )


def run_inventory_probe(
    task_data: Mapping[str, Any], environment: RunEnvironment, *, shell_runner: Any
) -> Inventory:
    """Take the inventory on the host that will run the build.

    A failure here is reported, not raised.
    """
    arch = str(task_data.get("arch", ""))
    command = probe_command(environment, repo=task_data["trtllm_repo_path"], arch=arch)
    result = shell_runner(command, None, INVENTORY_TIMEOUT_S)
    if getattr(result, "timed_out", False):
        return Inventory(arch=arch, error=f"inventory probe timed out after {INVENTORY_TIMEOUT_S}s")
    stdout = getattr(result, "stdout", "") + getattr(result, "stderr", "")
    inventory = parse_inventory(stdout, arch=arch)
    # A payload that landed is kept whatever the exit code says. The probe
    # catches its own import failures and still prints what it read off
    # disk.
    landed = bool(
        inventory.catalog or inventory.trtllm_ops or inventory.modules or inventory.flashinfer_api
    )
    if not landed and not getattr(result, "ok", True):
        return Inventory(
            arch=arch,
            error=f"{inventory.error} (probe exited {getattr(result, 'returncode', '?')})",
        )
    return inventory


# --------------------------------------------------------------------------
# the block the surveyor reads
# --------------------------------------------------------------------------


def format_surface(surface: CheckpointSurface, inventory: Inventory, *, limit: int = 40) -> str:
    """Render both columns for the plan prompt, joined by nothing; ``limit`` caps the long lists."""
    lines = ["## Surface survey", ""]
    # Whether the right-hand column is a reading or a blank is the first thing
    # this block has to say.
    taken = (
        "Measured before this phase started."
        if inventory.ok
        else (
            "The demand column was measured before this phase started. **The supply "
            "column was NOT.** The inventory probe failed, so the right-hand side "
            "below is an assumption and not a reading: nothing missing from it is "
            "evidence of absence, no row in it can carry a ceiling, and the plan must "
            "name the missing measurement as an open item."
        )
    )
    lines.append(
        f"{taken} The left column is what the checkpoint "
        f"demands; the right is what `{inventory.arch or 'this arch'}` supplies. "
        f"**Joining them is your job** — no function here guesses which entry answers "
        f"which demand."
    )
    lines.append("")

    lines.append(f"### Demands — `{surface.architecture or '?'}` (`{surface.model_type or '?'}`)")
    lines.append("")
    if surface.submodels:
        lines.append(
            f"Multimodal checkpoint: nested {', '.join(surface.submodels)}. Geometry below "
            f"is read from `text_config` where one exists.\n"
        )
    for demand in surface.demands:
        lines.append(f"- {demand}")
    if surface.unrecognized:
        lines.append("")
        lines.append(
            "**Unrecognized structural keys** — no probe here claims these, so treat each "
            "as a candidate for a layer this vocabulary has no word for. Read them before "
            "you size the queue; this is the cheapest moment in the run to discover a "
            "ceiling:"
        )
        lines.append("")
        lines.append(f"  `{'`, `'.join(surface.unrecognized[:limit])}`")
    lines.append("")

    lines.append("### Supply")
    lines.append("")
    # A probe can fail at the engine import and still have read the catalog
    # off disk: the error is stated, and whatever landed is still rendered.
    if not inventory.ok:
        lines.append(
            f"**The inventory probe reported an error**: `{inventory.error}`. Anything "
            f"below was still measured; anything absent was not. In particular, with no "
            f"op inventory nothing can be called rotted and nothing can be called "
            f"unwrapped — treat both as unknown and say so in the plan's open items."
        )
        lines.append("")
    if not inventory.catalog and not inventory.trtllm_ops and not inventory.flashinfer_api:
        lines.append(
            "Nothing landed at all. Plan on the assumption that you do not know what "
            "exists, and name that as an open item."
        )
        return "\n".join(lines) + "\n"

    spendable = inventory.spendable
    shared = [e for e in inventory.catalog if not e.scope]
    lines.append(
        f"{len(spendable)} of {len(shared)} shared catalog entries are spendable on "
        f"`{inventory.arch}` — their `ARCHS` name it, so their `CELLS` were driven here, "
        f"and they are still wired. {len(inventory.trtllm_ops)} ops are registered in this "
        f"build."
    )
    lines.append("")
    if spendable:
        lines.append("**Free vocabulary** — spend without a kernel writer dispatch:")
        lines.append("")
        for entry in spendable[:limit]:
            lines.append(f"- `{entry.category}/{entry.name}`")
        if len(spendable) > limit:
            lines.append(f"- … {len(spendable) - limit} more, see `surface.md`")
        lines.append("")
    unrun = list(inventory.uncertified)
    if unrun:
        lines.append(
            "**Present but un-run on this arch** — the entry exists and its `ARCHS` does "
            "not name this architecture, which reads as *unknown*. Driving its cells here "
            "is what is missing, and that is much cheaper than an onboard:"
        )
        lines.append("")
        for entry in unrun[:limit]:
            lines.append(f"- `{entry.category}/{entry.name}`")
        lines.append("")
    promotable = inventory.promotable
    if promotable:
        lines.append(
            "**Kept by another target in its own `ops/`** — certified for that target, "
            "importable by no other. A demand one of these answers costs a promotion into "
            "the shared catalog (the entry and its test move, the cells are re-driven "
            "here), not an onboard:"
        )
        lines.append("")
        for entry in promotable[:limit]:
            impl = f" → `{entry.impl}`" if entry.impl else ""
            lines.append(f"- `{entry.scope}/ops/{entry.category}/{entry.name}`{impl}")
        lines.append("")
    if inventory.rotted:
        lines.append(
            "**Rotted** — the entry names an `impl` that does not resolve in this build. "
            "This is worse than missing: it reads as vocabulary and fails at the call. "
            "Do not spend these:"
        )
        lines.append("")
        for entry in inventory.rotted:
            lines.append(f"- `{entry.category}/{entry.name}` → `{entry.impl}`")
        lines.append("")
    unwrapped = inventory.unwrapped_ops
    if unwrapped:
        lines.append(
            f"**{len(unwrapped)} registered ops carry no catalog entry.** A demand "
            f"answered by one of these is tier 1 — a wrapper and a receipt, not an "
            f"implementation. Search this list before you conclude anything is missing:"
        )
        lines.append("")
        lines.append(f"  `{'`, `'.join(unwrapped[:limit])}`")
        if len(unwrapped) > limit:
            lines.append(f"\n  … {len(unwrapped) - limit} more, see `surface.md`")
        lines.append("")
    if inventory.op_sources:
        tally = Counter(inventory.op_sources.values())
        breakdown = ", ".join(f"{n} {src}" for src, n in tally.most_common())
        lines.append(
            f"**Where those ops came from**: {breakdown}. Provenance decides what a "
            f"receipt must say — an architecture-gated JIT kernel and a compiled-in "
            f"C++ one are certified differently. See the kernel-source registry."
        )
        lines.append("")
    lines.append(
        f"The kernel plane (`_torch/modules/**`) holds {len(inventory.modules)} public "
        f"callables — the tier-2 rung. `surface.md` lists them."
    )
    lines.append("")
    lines.extend(_flashinfer_lines(inventory, limit))
    return "\n".join(lines) + "\n"


def _flashinfer_lines(inventory: Inventory, limit: int) -> list[str]:
    """The tier-3 rung: FlashInfer kernels no trtllm op wraps, grouped by module."""
    version = inventory.versions.get("flashinfer", "")
    if not inventory.flashinfer_api:
        return [
            f"**FlashInfer's API was not read** ({version or 'the probe did not report it'}). "
            f"Tier 3 is unmeasured, not empty: say so in the plan's open items, and do not "
            f"claim tier 4 or 5 or a ceiling without searching the installed package yourself."
        ]
    unreferenced = inventory.flashinfer_unreferenced
    lines = [
        f"**FlashInfer {version}: {len(inventory.flashinfer_api)} public callables, "
        f"{len(unreferenced)} referenced nowhere in the engine.** The op inventory cannot "
        f"see these: no trtllm op wraps them. A demand one of them answers is tier 3 — an "
        f"entry whose `__call__` calls `flashinfer.<module>.<name>` directly, which costs a "
        f"wrapper and its cells, not an implementation. Search them before you claim tier "
        f"4 or 5, and before any ceiling:",
        "",
    ]
    groups: dict[str, list[str]] = {}
    for path in unreferenced:
        group = path.split(".")[1] if path.count(".") >= 2 else ""
        groups.setdefault(group, []).append(path.rsplit(".", 1)[-1])
    per_group = max(1, limit // 5)
    for group, names in sorted(groups.items()):
        shown = ", ".join(f"`{name}`" for name in names[:per_group])
        more = f", … {len(names) - per_group} more" if len(names) > per_group else ""
        lines.append(f"- `flashinfer.{group}`: {shown}{more}")
    lines += [
        "",
        "`surface.md` lists every callable with its full path and the engine files that "
        "already reference it.",
    ]
    return lines


def write_surface_file(path: str | Path, surface: CheckpointSurface, inventory: Inventory) -> Path:
    """Write the unabridged survey to the workspace, for audit and for agents."""
    out = Path(path)
    body = [format_surface(surface, inventory, limit=10_000), "", "## Kernel plane", ""]
    body.extend(f"- `{m}`" for m in inventory.modules)
    body.extend(["", "## FlashInfer API", ""])
    for path in inventory.flashinfer_api:
        refs = inventory.flashinfer_refs.get(path.rsplit(".", 1)[-1], ())
        where = (
            "referenced in " + ", ".join(f"`{ref}`" for ref in refs)
            if refs
            else "referenced nowhere in the engine"
        )
        body.append(f"- `{path}` — {where}")
    if not inventory.flashinfer_api:
        body.append(f"- not read: {inventory.versions.get('flashinfer', 'not reported')}")
    body.extend(["", "## Versions", ""])
    body.extend(f"- {k}: {v}" for k, v in sorted(inventory.versions.items()))
    out.write_text("\n".join(body) + "\n", encoding="utf-8")
    return out


__all__ = [
    "CatalogEntry",
    "CheckpointSurface",
    "Demand",
    "INVENTORY_TIMEOUT_S",
    "Inventory",
    "PROBE_SOURCE",
    "RECEIPT_ABSENT",
    "RECEIPT_PASSED",
    "format_surface",
    "parse_inventory",
    "probe_command",
    "read_checkpoint_surface",
    "run_inventory_probe",
    "write_surface_file",
]
