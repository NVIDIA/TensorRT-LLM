# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What this checkpoint demands, and what this build can already supply.

The plan phase used to open with the drafter *guessing*: guessing which
``torch.ops.trtllm.*`` names exist in this build, guessing which catalog
entries are certified on this architecture, and guessing which of the
checkpoint's structural features have a home. Every guess that missed became
a queue item that a smith discovered was wrong, hours later, with a GPU
allocation held open behind it.

This module replaces the guessing with a measurement taken before the first
agent exists. It has two halves and they are deliberately not joined:

- :func:`read_checkpoint_surface` reads ``config.json`` and reports what the
  model *structurally demands* — attention geometry, positional scheme,
  activation, normalization, expert routing, recurrent state, speculation
  heads, quantization. Pure Python, no GPU, no imports of the engine.
- :func:`run_inventory_probe` runs on the real host and reports what the
  build *supplies* — which ops are registered, which catalog entries exist,
  which of those carry a receipt for this architecture, and which entries
  name an ``impl`` that no longer resolves.

**The join between them is judgement and stays in prose.** Deciding that
"this model wants SwiGLU" is answered by ``activation/flashinfer_silu_and_mul``
is a modeling call, not a string match, and a Python function that guesses it
would be confidently wrong on exactly the models this workflow exists to
onboard. So the survey lays both columns in front of the drafter and lets the
drafter join them.

The half that generalizes is the one that matters for a model nobody has
seen. :class:`CheckpointSurface` carries ``unrecognized``: structural keys in
``config.json`` that no probe here claims. A key nobody recognizes is the
earliest honest signal that this checkpoint has a layer the vocabulary has no
word for — which is the case that used to be discovered by a smith, at tier
4, with no budget left to absorb it.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .environment import RunEnvironment

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

    ``keys`` is the evidence — the ``config.json`` keys this was read off —
    so a drafter that disbelieves a demand can check it rather than argue
    with it.
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
# unrecognized, which is the point — the list is allowed to be incomplete,
# and its incompleteness costs a line of prose rather than a wrong plan.
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
# `moe_num_experts`), and an exact list would report a rename as a discovery.
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
# sits in the MIDDLE of the name. `add_swa_attention_sink_bias` has no prefix
# any list could carry without also claiming every `add_*` key, and the
# `attention.sink` probe that claims it globs on exactly this substring; the
# same is true of `<something>_layer_pattern` and `attention.hybrid`.
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
# two masks — which decides whether the cache manager is asked for one pool or
# two, and whether the weight manifest has one shard calculation or two.
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

    ``None`` when there is none. Reported rather than assumed: a stack whose
    global layers land every sixth position and a stack whose first block is
    short are different stacks, and the second one is what this was written
    against.

    Candidates stop at half the length, because a "period" that occurs once
    is not a period — a 48-entry list trivially satisfies ``p = 47`` whenever
    its last entry happens to equal its first, and reporting that as a
    repeat is worse than reporting none.
    """
    for period in range(1, len(values) // 2 + 1):
        if all(value == values[index % period] for index, value in enumerate(values)):
            return period
    return None


def _sink_layer_kind(key: str) -> str:
    """``add_swa_attention_sink_bias`` -> ``swa``: which layers carry the sink.

    The layer kind is in the key name and nowhere else, and it is the whole
    content of the demand — a sink on every layer and a sink on the local
    layers only are different models.
    """
    head = key.split("attention_sink", 1)[0]
    if head.startswith("add_"):
        head = head[len("add_") :]
    # An unprefixed spelling names no kind, which is itself the answer: the
    # sink is on the whole stack.
    return head.strip("_") or "all"


def read_checkpoint_surface(config: Mapping[str, Any]) -> CheckpointSurface:
    """Report what this checkpoint structurally demands of a target.

    Written against *concepts* rather than against one model family: each
    probe accepts the several spellings the ecosystem actually uses, because
    the alternative is a reader that silently reports "dense" for a MoE model
    whose config says ``n_routed_experts``.
    """
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
        # it is a different question from "what is the head dim". Every other
        # demand here is answered by choosing an entry; this one decides
        # whether any paged-attention entry can be reused at all, or whether
        # the target needs a padded-V fallback — so a drafter that reads only
        # `head_dim` sizes the queue without the hardest item in it.
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
    # of layer sits where, which is a hybrid stack and not a window. Reading
    # it here reported "per-layer or global window" for a checkpoint with no
    # window anywhere in it. `stack.hybrid` below is where it belongs.
    if keys := _present(scoped, "sliding_window", "use_sliding_window", "sliding_window_pattern"):
        demands.append(
            Demand(
                "attention.windowing",
                ", ".join(f"{k}={scoped[k]!r}" for k in keys)[:200],
                keys,
            )
        )
    # A per-layer pattern is not a window. `attention.windowing` above reports
    # `sliding_window=128` and nothing else, which reads as one window over a
    # uniform stack; the stack this was written against is 48 layers of two
    # different kinds whose KV head counts and RoPE bases disagree. That is a
    # demand on the cache manager and on the weight manifest, not a mask.
    #
    # **The survey reports the pattern and refuses to interpret it.** Which
    # value means sliding-window and which means full attention is the config
    # author's private convention — this checkpoint spells local as `1` and
    # full as `0`, and nothing in the file says so. Asserting a polarity the
    # survey cannot know would silently window the wrong 39 layers, which is
    # strictly worse than handing the plan phase the counts and the question.
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

    # A learned term in the softmax DENOMINATOR, and the layer kind that
    # carries it is in the key name and nowhere else. Distinct from
    # `attention.sinks` below, which reads the count-of-sink-tokens spelling:
    # this one is the per-layer-kind boolean, and a target that silently drops
    # it boots, generates plausible text and is wrong by a constant per head.
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
    # A LEARNED selector, not a mask. A window or a sink leaves cache
    # addressing alone; an indexer decides per query which past positions are
    # attended to, so the top-k selection is part of the forward and part of
    # what a receipt has to certify. Missing this probe once meant a
    # 48-layer checkpoint with `indexer_budget: 2048` surfaced no sparse
    # demand at all and the whole sparse_attention area stayed unread.
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

    # transformers 5.x moved the whole RoPE description into `rope_parameters`
    # and left `rope_scaling` behind. Reading only the old name made a
    # checkpoint with `rope_parameters.mrope_section` surface NO rope demand
    # except `rope.partial` -- so mrope, the thing a multimodal target is most
    # likely to get wrong, went unmentioned. Accept both spellings, and
    # remember which one was read so the citation is not a lie.
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
        # rather than a different constant. `noaux_tc` over `n_group` groups
        # with `topk_group` groups kept is grouped routing with a bias
        # correction, which is its own op — a target that reads only
        # `scoring_func` builds plain sigmoid top-k and routes to the wrong
        # experts while scoring close enough to pass a short gate.
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

    # `mtp_num_hidden_layers` is here because the first checkpoint this ran on
    # used it and the probe missed it — the key fell out in `unrecognized`
    # instead, which is how it was noticed. Left as evidence that the probe
    # list is allowed to be incomplete as long as the leftovers are reported.
    nextn = _num(scoped, "num_nextn_predict_layers", "num_mtp_modules", "mtp_num_hidden_layers")
    # DSpark is read FIRST because DeepSeek-V4.1 carries both spellings: its
    # DSpark stages live under `mtp.*` with `num_nextn_predict_layers: 3`, and
    # the MTP reading of that key is the one that builds the wrong drafter.
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

    # The residual stream is not one tensor. Hyper-connections carry `n`
    # parallel streams mixed at every block boundary, which changes the shape
    # of every residual add in the forward -- so a target that assumes one
    # stream is not slightly wrong, it is a different model.
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

    # A tower in front of the language model. Keyed on the OUTER config,
    # because `scoped` is the text tower by then and the vision config is its
    # sibling, not its child.
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

    # A secondary table whose size dwarfs the vocabulary. This is the
    # signature of an n-gram / product-key / memory-layer embedding, and the
    # reason it matters is residency: a table an order of magnitude larger
    # than the vocabulary can be most of the checkpoint, so whether it lives
    # on the device decides how much is left for KV.
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

RECEIPT_PASSED = "passed"
RECEIPT_FAILED = "failed"
RECEIPT_ABSENT = "absent"


@dataclass(frozen=True)
class CatalogEntry:
    """One catalog entry as the host actually holds it."""

    path: str
    impl: str = ""
    summary: str = ""
    receipt: str = RECEIPT_ABSENT
    impl_resolves: bool = True

    @property
    def name(self) -> str:
        return Path(self.path).stem

    @property
    def category(self) -> str:
        parts = Path(self.path).parts
        return parts[0] if len(parts) > 1 else ""

    @property
    def certifiable(self) -> bool:
        """Whether a receipt is a thing this entry can even have.

        ``torch/`` entries are thin mirrors of torch callables; upstream owns
        their correctness, so the catalog gives them no contract, no test and
        no architecture certification. For those, absence is *by design* and
        not the three-state "unknown" — listing them as "a receipt run is
        what is missing" points at work that does not exist.
        """
        return self.category != "torch"

    @property
    def spendable(self) -> bool:
        """Usable without a smith dispatch: certified here and still wired."""
        if not self.impl_resolves:
            return False
        return self.receipt == RECEIPT_PASSED or not self.certifiable


@dataclass(frozen=True)
class Inventory:
    """What the host supplies, measured rather than assumed."""

    arch: str = ""
    trtllm_ops: tuple[str, ...] = ()
    catalog: tuple[CatalogEntry, ...] = ()
    modules: tuple[str, ...] = ()
    #: op name -> the source that registered it. Empty when the op inventory
    #: could not be taken; a name absent from the Python plane maps to
    #: ``cpp_thop``, which is a conclusion rather than a default.
    op_sources: dict[str, str] = field(default_factory=dict)
    versions: dict[str, str] = field(default_factory=dict)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def spendable(self) -> tuple[CatalogEntry, ...]:
        return tuple(e for e in self.catalog if e.spendable)

    @property
    def uncertified(self) -> tuple[CatalogEntry, ...]:
        """Present, wired, certifiable, but carrying no pass for this arch.

        Includes the entries whose receipt says ``failed``, which are the
        most important rows here: a known-failing entry that reads as merely
        "not yet run" is how a plan spends vocabulary that is already known
        not to work.
        """
        return tuple(
            e for e in self.catalog if e.certifiable and e.impl_resolves and not e.spendable
        )

    @property
    def rotted(self) -> tuple[CatalogEntry, ...]:
        """Entries whose ``impl`` no longer resolves in this build.

        A rotted entry is worse than a missing one: it reads as vocabulary
        the plan may spend, and it fails at the first call rather than at
        selection.
        """
        return tuple(e for e in self.catalog if not e.impl_resolves)

    @property
    def unwrapped_ops(self) -> tuple[str, ...]:
        """Registered ops that no catalog entry wraps.

        This is the tier-1 rung made concrete: a gap answered by a name in
        here costs a wrapper and a receipt, not an implementation.
        """
        wrapped = {e.impl.rsplit(".", 1)[-1] for e in self.catalog if e.impl}
        return tuple(op for op in self.trtllm_ops if op not in wrapped)


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
    # has already resolved, not the names that are registered. On this build
    # that is 91 out of 273 -- so a supply survey built on `dir()` reports two
    # thirds of the op plane as missing, and the catalog entries wrapping them
    # as "rotted: do not spend these". Measured: `allgather`, `fp4_quantize`,
    # `fused_qk_norm_rope`, `causal_conv1d_fwd`, `noaux_tc_op` and eight more
    # were all reported absent and all resolve.
    #
    # The registry is the truth. `dir()` stays as a fallback for a torch
    # without the private accessor, where under-reporting beats no report.
    # Import the Python registration plane before enumerating. An op whose
    # `custom_ops/*.py` module has not been imported has not registered its
    # schema either, so it is absent from BOTH the registry and `dir()` --- and
    # the entry wrapping it is reported as rotted. Measured: two entries
    # (`fused_sigmoid_mul_inplace`, `rms_norm_gated_token_major`) read absent
    # before the import and resolved 27 seconds later with it. Importing
    # `tensorrt_llm` alone is not enough; the submodule has to be touched.
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
        # outside `custom_ops/` -- `rms_norm_gated_token_major` registers in
        # `_torch.modules.mamba.layernorm_gated`, and its catalog wrapper
        # says so in a comment at the top of the file. So ask the wrapper:
        # importing it performs whatever import the op needs, by
        # construction, because that is what makes the wrapper work.
        #
        # Without this, two live entries were reported as "rotted -- do not
        # spend these" and the plan phase had to argue the survey down.
        if entry_path:
            import importlib
            module = CATALOG_PKG + "." + entry_path[:-3].replace("/", ".")
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

def frontmatter(md_path):
    try:
        text = md_path.read_text(encoding="utf-8")
    except OSError:
        return ""
    if not text.startswith("---") or text.count("---") < 2:
        return ""
    return text.split("---", 2)[1]

def receipt_state(md_path, arch):
    # Three-state, keyed by arch: passed / failed / absent. Absent means
    # UNKNOWN, and conflating it with either other state is the whole reason
    # this is three-state.
    #
    # Two frontmatter shapes are in use -- `sm_103: {status: passed, tests: 9}`
    # and the block form with `status:` on its own indented line -- and an
    # earlier hand-rolled reader got both wrong in the same direction: it
    # reported the inline FAILED entry as passed, and the block FAILED entry
    # as absent. yaml is used when importable, which in the run's own
    # interpreter it always is; the fallback below reads `status:` rather
    # than guessing from the presence of a value.
    head = frontmatter(md_path)
    if not head:
        return "absent"
    try:
        import yaml
        receipts = (yaml.safe_load(head) or {}).get("receipts") or {}
        record = receipts.get(arch)
        if record is None:
            return "absent"
        if isinstance(record, dict):
            status = str(record.get("status", "")).lower()
        else:
            status = str(record).lower()
        if status.startswith(("pass", "true", "ok")):
            return "passed"
        if status.startswith(("fail", "false")):
            return "failed"
        return "absent"
    except ImportError:
        pass
    except Exception:
        return "absent"

    inside, base_indent, status = False, None, ""
    for line in head.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip())
        if stripped.startswith("receipts:"):
            inside, base_indent = True, indent
            continue
        if not inside:
            continue
        if indent <= base_indent:
            break
        if stripped.startswith(arch + ":"):
            rest = stripped.split(":", 1)[1].strip()
            if rest:
                # inline flow mapping, or a bare scalar
                marker = "status:"
                status = (
                    rest.split(marker, 1)[1].strip(" {}'\"").split(",")[0].strip()
                    if marker in rest
                    else rest.strip("{}'\"")
                )
            else:
                # block form: the first `status:` under this arch key
                status = "__block__"
            continue
        if status == "__block__":
            if stripped.startswith("status:"):
                status = stripped.split(":", 1)[1].strip().strip("'\"")
            elif indent <= base_indent + 2:
                status = ""  # left the arch block without ever seeing a status
                break
    status = status.lower()
    if status.startswith(("pass", "true", "ok")):
        return "passed"
    if status.startswith(("fail", "false")):
        return "failed"
    return "absent"

catalog = repo / "tensorrt_llm/_torch/_experimental/modeling_v2/catalog"
index = catalog / "index.yaml"
if index.is_file():
    # Same reason as above: a five-field list read without a YAML parser.
    entry = None
    # The list may sit at column 0 or be indented under `entries:` -- the
    # in-tree index indents it, and a column-0 match read that index as EMPTY,
    # so a dry run printed `0/0 entries certified` over 19 certified entries.
    for raw in index.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("- path:"):
            if entry:
                out["catalog"].append(entry)
            entry = {"path": line.split(":", 1)[1].strip(), "impl": "", "summary": ""}
        elif entry is not None and line.startswith("impl:"):
            entry["impl"] = line.split(":", 1)[1].strip()
        elif entry is not None and line.startswith("summary:"):
            entry["summary"] = line.split(":", 1)[1].strip().strip("'\"")[:160]
    if entry:
        out["catalog"].append(entry)
for entry in out["catalog"]:
    md = catalog / entry["path"].replace(".py", ".md")
    entry["receipt"] = receipt_state(md, arch)
    # Only claim an entry is unwired when the op inventory was actually
    # taken. Without that guard an engine that failed to import reports the
    # whole catalog as rotted, which is the most alarming possible way to
    # say "I could not measure this".
    entry["impl_resolves"] = True if out["error"] else bool(resolves(entry["impl"], entry.get("path")))

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

print("__SURVEY_JSON__" + json.dumps(out))
"""


def probe_command(environment: RunEnvironment, *, repo: str | Path, arch: str) -> str:
    """The literal command that takes the inventory, fully prefixed.

    Launched as **one** process whatever the run's world size. This is
    read-only introspection — it imports the build, walks ``torch.ops.trtllm``
    and parses catalog frontmatter off disk — so nothing in it wants a rank
    per device, and taking the engine's launch rules instead cost this
    workflow a whole survey once: at world size 4 the probe went out as
    ``mpirun -n 4 ... trtllm-llmapi-launch``, the run's own ``PATH`` did not
    carry the container's MPI, and the rc=127 arrived at the plan phase as an
    empty inventory.

    The distinction belongs to :meth:`RunEnvironment.command`, not here.
    Assembling a launcher line outside ``environment.py`` is exactly what
    that module exists to prevent.
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
            summary=str(e.get("summary", "")),
            receipt=str(e.get("receipt", RECEIPT_ABSENT)),
            impl_resolves=bool(e.get("impl_resolves", True)),
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
    )


def run_inventory_probe(
    task_data: Mapping[str, Any],
    environment: RunEnvironment,
    *,
    shell_runner: Any,
    timeout: int = INVENTORY_TIMEOUT_S,
) -> Inventory:
    """Take the inventory on the host that will run the build.

    A failure here is reported, not raised. The plan can be drafted without
    an inventory — it just gets drafted the old way, on assumption — and
    losing the whole run because a probe could not import is a worse trade
    than losing the measurement.
    """
    arch = str(task_data.get("arch", ""))
    command = probe_command(environment, repo=task_data["trtllm_repo_path"], arch=arch)
    result = shell_runner(command, None, timeout)
    if getattr(result, "timed_out", False):
        return Inventory(arch=arch, error=f"inventory probe timed out after {timeout}s")
    stdout = getattr(result, "stdout", "") + getattr(result, "stderr", "")
    inventory = parse_inventory(stdout, arch=arch)
    # A payload that landed is kept whatever the exit code says. The probe
    # catches its own import failures and still prints what it read off
    # disk, and the catalog half is most of the value.
    landed = bool(inventory.catalog or inventory.trtllm_ops or inventory.modules)
    if not landed and not getattr(result, "ok", True):
        return Inventory(
            arch=arch,
            error=f"{inventory.error} (probe exited {getattr(result, 'returncode', '?')})",
        )
    return inventory


# --------------------------------------------------------------------------
# the block the drafter reads
# --------------------------------------------------------------------------


def format_surface(surface: CheckpointSurface, inventory: Inventory, *, limit: int = 40) -> str:
    """Render both columns for the plan prompt, joined by nothing.

    ``limit`` caps the long lists. A drafter that needs the full inventory
    can read ``surface.md`` in the workspace, which is written unabridged.
    """
    lines = ["## Surface survey", ""]
    # Whether the right-hand column is a reading or a blank is the first thing
    # this block has to say. An inventory that failed to land renders with the
    # same shape as one that landed empty — "0 of 0 entries certified" — and a
    # drafter that reads the second as the first plans every entry as a gap.
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
    # off disk, which is most of the value. Reporting "nothing was measured"
    # when two thirds of it was is its own kind of wrong answer — so the
    # error is stated, and whatever landed is still rendered.
    if not inventory.ok:
        lines.append(
            f"**The inventory probe reported an error**: `{inventory.error}`. Anything "
            f"below was still measured; anything absent was not. In particular, with no "
            f"op inventory nothing can be called rotted and nothing can be called "
            f"unwrapped — treat both as unknown and say so in the plan's open items."
        )
        lines.append("")
    if not inventory.catalog and not inventory.trtllm_ops:
        lines.append(
            "Nothing landed at all. Plan on the assumption that you do not know what "
            "exists, and name that as an open item."
        )
        return "\n".join(lines) + "\n"

    spendable = inventory.spendable
    lines.append(
        f"{len(spendable)} of {len(inventory.catalog)} catalog entries are spendable on "
        f"`{inventory.arch}` — certified here and still wired, or a `torch/` mirror that "
        f"needs no certification. {len(inventory.trtllm_ops)} ops are registered in this "
        f"build."
    )
    lines.append("")
    if spendable:
        lines.append("**Free vocabulary** — spend without a smith dispatch:")
        lines.append("")
        for entry in spendable[:limit]:
            lines.append(f"- `{entry.category}/{entry.name}`")
        if len(spendable) > limit:
            lines.append(f"- … {len(spendable) - limit} more, see `surface.md`")
        lines.append("")
    # Split by state rather than lumped, because the two mean opposite
    # things about how much work is left. `failed` is not "not yet run".
    failed = [e for e in inventory.uncertified if e.receipt == RECEIPT_FAILED]
    unrun = [e for e in inventory.uncertified if e.receipt != RECEIPT_FAILED]
    if failed:
        lines.append(
            "**Known failing on this arch** — the entry's own receipt records a failure "
            "here. This is the opposite of un-run: someone took the measurement and it "
            "did not pass. A plan that spends one of these is spending vocabulary that "
            "is already known not to work, so budget for the failure or route around it:"
        )
        lines.append("")
        for entry in failed[:limit]:
            lines.append(f"- `{entry.category}/{entry.name}` — receipt: failed")
        lines.append("")
    if unrun:
        lines.append(
            "**Present but un-run on this arch** — the wrapper exists and the receipt is "
            "absent, which reads as *unknown*. A receipt run is what is missing, and that "
            "is much cheaper than an onboard:"
        )
        lines.append("")
        for entry in unrun[:limit]:
            lines.append(f"- `{entry.category}/{entry.name}`")
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
    return "\n".join(lines) + "\n"


def write_surface_file(path: str | Path, surface: CheckpointSurface, inventory: Inventory) -> Path:
    """Write the unabridged survey to the workspace, for audit and for agents."""
    out = Path(path)
    body = [format_surface(surface, inventory, limit=10_000), "", "## Kernel plane", ""]
    body.extend(f"- `{m}`" for m in inventory.modules)
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
    "RECEIPT_FAILED",
    "RECEIPT_PASSED",
    "format_surface",
    "parse_inventory",
    "probe_command",
    "read_checkpoint_surface",
    "run_inventory_probe",
    "write_surface_file",
]
