# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight loading: GLM-5.3-Flash (NVFP4) / sm_100 / default.

The checkpoint is the vision-language wrapper's. ``model.language_model.*`` holds the language model (its
``layers.45.*`` are the MTP layer), ``lm_head.*`` the output head and ``model.visual.*`` the vision tower. The
registered ``Glm5NextHfWeightMapper`` audits every key against the target's destinations: text and MTP keys
resolve to an owner (a decoder layer, the embedding, the final norm or the head), the vision namespace is a
predicted non-load, and any key that resolves nowhere fails the load, naming it. Each owner is then filled through
its modules' own loaders, which keep this rank's shard and the NVFP4 block-scale layout.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from tensorrt_llm._torch.models.checkpoints.hf.glm5_next_weight_mapper import (
    Glm5NextHfWeightMapper,
    glm5_next_weight_owner,
)
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

if TYPE_CHECKING:
    from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper

#: Routed-expert destinations: consumed by the fused MoE layer's loader rather
#: than placed by name.
_FUSED_MOE_RE = re.compile(
    r"^(?P<layer>model\.layers\.\d+\.mlp)\.experts\.(?P<expert>\d+)"
    r"\.(?P<proj>gate_proj|up_proj|down_proj)"
    r"\.(?P<tensor>weight|weight_scale_inv|weight_scale|weight_scale_2|input_scale)$"
)


def load(model: Any, weights: Any, weight_mapper: BaseWeightMapper | None = None) -> None:
    """Load checkpoint tensors owner by owner into the allocated model.

    The registered GLM mapper resolves keys and destination owners. FP8 payloads
    and FP32 block scales retain their checkpoint representation; module loaders
    handle sharding and fused projections. Unresolved or unfilled tensors raise.

    Args:
        weights: Mapping of checkpoint keys to tensors or lazy safetensors slices.
        weight_mapper: Initialized Glm5NextHfWeightMapper, or None to create one.
    """
    # The KDA layer class lives in modeling.py, which imports this module.
    from .modeling import Glm5NextLinearAttention

    if weight_mapper is None:
        weight_mapper = Glm5NextHfWeightMapper()
        weight_mapper.init_model_and_config(model, model.model_config)
    elif not isinstance(weight_mapper, Glm5NextHfWeightMapper):
        raise ValueError(
            "glm5_next needs its registered Glm5NextHfWeightMapper; got "
            f"{type(weight_mapper).__name__}"
        )
    if not model.quantized:
        raise ValueError(
            "glm5_next whole-model loading requires a block-FP8 or NVFP4 build "
            "(quantized=True). Dequantizing all 288 experts of 42 routed "
            "layers to bf16 would double the resident model to ~656 GB and "
            "exceed the supported loading footprint."
        )
    # Decoder stack plus the aliased MTP draft layer(s), if any: owners
    # ``45..`` are the draft layers, placed from the checkpoint's own
    # ``layers.45.*`` keys instead of being allowlisted away.
    num_layers = model.schedule.num_layers + len(model.mtp_layers)
    audit = weight_mapper.audit(list(weights.keys()))

    # Checkpoint tensors are grouped by owner (a decoder layer or a named
    # top-level part) and placed by that owner's modules.
    targets: dict[Any, tuple[nn.Module, str]] = {
        **{i: (model.model.layers[i], f"model.layers.{i}.") for i in range(num_layers)},
        "embed": (model.model.embed_tokens, "model.embed_tokens."),
        "norm": (model.model.norm, "model.norm."),
        "head": (model.lm_head, "lm_head."),
    }
    # Owners pruned by pipeline parallelism (`__pp_init__` cleared their
    # parameters): their checkpoint keys belong to another rank.
    remote = {
        owner
        for owner, (module, _) in targets.items()
        if getattr(module, "_weights_removed", False)
    }

    by_owner: dict[Any, list[tuple[str, str]]] = {}
    for dest, key in audit.destinations.items():
        owner = glm5_next_weight_owner(dest, num_layers)
        if owner is None:
            raise ValueError(f"glm5_next has no destination owner for {key!r} -> {dest!r}")
        if owner in remote:
            continue
        by_owner.setdefault(owner, []).append((key, dest))
    if audit.unresolved:
        raise ValueError(f"glm5_next cannot place {sorted(audit.unresolved)[:5]}")

    device = torch.device("cuda", torch.cuda.current_device())
    for owner, (module, prefix) in targets.items():
        if owner in remote:
            continue
        _fill_module(module, owner, prefix, by_owner.get(owner, []), weights, device)
    # The shared KDA mixer's post-load kernel constants (built from the
    # just-loaded local shards; the same step the Kimi K3 loader runs).
    for layer in model.model.layers:
        attn = getattr(layer, "self_attn", None)
        if isinstance(attn, Glm5NextLinearAttention) and not getattr(
            layer, "_weights_removed", False
        ):
            attn.finalize_weights()
    logger.info(f"glm5_next loaded {len(targets) - len(remote)} owners")


def _fill_module(
    module: nn.Module,
    owner: Any,
    prefix: str,
    entries: Sequence[tuple[str, str]],
    weights: Any,
    device: torch.device,
) -> None:
    """Place one materialized owner's tensors.

    * A TensorRT-LLM ``Linear`` destination receives its checkpoint tensors
      through the module's own ``load_weights`` (which owns the TP slicing
      and the block-scale layout). Projections the runtime keeps fused
      (``GatedMLP.gate_up_proj``, the sparse layers' ``[q_a | kv_a]`` and
      the indexer's ``[wk | gate | weights_proj]``) collect their separate
      checkpoint tensors first (:attr:`FUSED_LINEARS` on the owning module).
    * Production routed experts go to the fused MoE layer's loader,
      filtered to ``initial_local_expert_ids`` except for the scalar
      ``input_scale``, which is passed for every expert.
    * The KDA layer's tensors (the shared mixer's local-width modules) are
      sliced to this rank's head range by the module itself.
    * Everything else (norms, HC, router, embeddings) is an exact-shape
      replicated copy.
    """
    # The KDA layer class lives in modeling.py, which imports this module.
    from .modeling import Glm5NextLinearAttention

    params = dict(module.named_parameters())
    params.update(dict(module.named_buffers()))
    named_modules = dict(module.named_modules())
    kda = named_modules.get("self_attn")
    if not isinstance(kda, Glm5NextLinearAttention):
        kda = None
    filled: set[str] = set()

    # source module path -> (fused Linear path, position in the concatenation)
    fused_of: dict[str, tuple[str, int]] = {}
    for mod_name, sub in named_modules.items():
        base = f"{mod_name}." if mod_name else ""
        for fused_attr, sources in getattr(type(sub), "FUSED_LINEARS", ()):
            for pos, src in enumerate(sources):
                fused_of[base + src] = (base + fused_attr, pos)
        if isinstance(sub, GatedMLP):
            fused_of[base + "gate_proj"] = (base + "gate_up_proj", 0)
            fused_of[base + "up_proj"] = (base + "gate_up_proj", 1)
    linear_groups: dict[str, dict[str, Any]] = {}
    fused_groups: dict[str, dict[int, dict[str, Any]]] = {}

    experts = getattr(getattr(module, "mlp", None), "experts", None)
    local_expert_ids = set(experts.initial_local_expert_ids) if experts is not None else set()
    experts_quant = getattr(experts, "quant_config", None)
    quant_algo = experts_quant.quant_algo if experts_quant is not None else None
    # The loader reduces every expert's scalar input_scale into the
    # layer-wide activation scale, which must be identical on every rank
    # because tokens are quantized before dispatch.
    collect_all_expert_input_scales = quant_algo == QuantAlgo.NVFP4
    moe_weights: dict[str, torch.Tensor] = {}
    proj_to_w = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}

    def materialize(t: Any) -> torch.Tensor:
        # Ellipsis supports both scalar calibration scales and matrix weights.
        return t if torch.is_tensor(t) else t[...]

    for key, dest in sorted(entries):
        local = dest.removeprefix(prefix)
        lazy = weights[key]

        moe = _FUSED_MOE_RE.match(dest)
        if moe is not None:
            expert_id = int(moe.group("expert"))
            tensor = moe.group("tensor")
            # Other EP ranks' weights are never read.
            if expert_id in local_expert_ids or (
                collect_all_expert_input_scales and tensor == "input_scale"
            ):
                moe_weights[f"{expert_id}.{proj_to_w[moe.group('proj')]}.{tensor}"] = materialize(
                    lazy
                )
            continue

        if local.endswith(".weight_scale_inv"):
            mod_path, label = local[: -len(".weight_scale_inv")], "weight_scale_inv"
        else:
            mod_path, _, label = local.rpartition(".")
        if mod_path in fused_of:
            fused_path, pos = fused_of[mod_path]
            fused_groups.setdefault(fused_path, {}).setdefault(pos, {})[label] = materialize(lazy)
            continue
        if local in fused_of:  # a bare parameter (no ``.weight``) feeding a fused Linear
            fused_path, pos = fused_of[local]
            fused_groups.setdefault(fused_path, {}).setdefault(pos, {})["weight"] = materialize(
                lazy
            )
            continue
        if isinstance(named_modules.get(mod_path), Linear):
            linear_groups.setdefault(mod_path, {})[label] = materialize(lazy)
            continue

        tensor = materialize(lazy)
        if kda is not None and local.startswith("self_attn."):
            tensor = kda.shard_checkpoint_tensor(local[len("self_attn.") :], tensor)
        target = params.get(local)
        if target is None:
            raise KeyError(f"glm5_next has no parameter for {key!r} (destination {dest!r})")
        if tuple(target.shape) != tuple(tensor.shape):
            raise ValueError(
                f"glm5_next {dest!r}: checkpoint shape {tuple(tensor.shape)} does not "
                f"match parameter shape {tuple(target.shape)}"
            )
        with torch.no_grad():
            target.copy_(tensor.to(device=device, dtype=target.dtype))
        filled.add(local)

    def mark_linear(mod_path: str, labels: Iterable[str] = ()) -> None:
        # ``labels`` are extra tensors the Linear's load_weights consumed
        # (e.g. the router gate's e_score_correction_bias).
        linear = named_modules[mod_path]
        param_names = ["weight", "weight_scale", "bias", *labels]
        if linear.quant_config is not None and linear.quant_config.quant_algo == QuantAlgo.NVFP4:
            # The shared loader fills the calibrated scales and derives
            # alpha; the KV scales keep their construction values.
            param_names += [
                "input_scale",
                "inv_input_scale",
                "weight_scale_2",
                "alpha",
                "kv_scales",
                "inv_kv_scales",
            ]
        for param_name in param_names:
            # mod_path is "" when the owner *is* the Linear (the LMHead).
            name = f"{mod_path}.{param_name}" if mod_path else param_name
            if name in params:
                filled.add(name)

    def load_linear(
        mod_path: str, groups: list[dict[str, torch.Tensor]], labels: Iterable[str] = ()
    ) -> None:
        linear = named_modules[mod_path]
        if linear.quant_config is not None and linear.quant_config.quant_algo == QuantAlgo.NVFP4:
            required = {"weight", "weight_scale", "weight_scale_2", "input_scale"}
            for part, group in enumerate(groups):
                missing = required - group.keys()
                if missing:
                    raise ValueError(
                        f"glm5_next owner {owner!r}: NVFP4 projection {mod_path!r} "
                        f"part {part} is missing checkpoint tensors {sorted(missing)}"
                    )
        linear.load_weights(groups)
        mark_linear(mod_path, labels)

    for mod_path, group in linear_groups.items():
        load_linear(mod_path, [group], group)
    for fused_path, parts in fused_groups.items():
        dest_mod = named_modules[fused_path]
        groups = [parts[i] for i in sorted(parts)]
        if isinstance(dest_mod, Linear) and dest_mod.weights_loading_config.weight_mode.name == (
            "FUSED_GATE_UP_LINEAR"
        ):
            # GatedMLP: the Linear shards each half over this rank's
            # intermediate range and stacks them [gate; up] itself.
            load_linear(fused_path, groups)
        else:
            # Replicated fusions: one row-concatenated tensor (and block
            # scales), the order being the module's declared source order.
            fused_quant = getattr(dest_mod, "quant_config", None)
            if fused_quant is not None and fused_quant.quant_algo == QuantAlgo.NVFP4:
                # Each NVFP4 source has its own per-tensor scales; merging them is not
                # implemented.
                raise ValueError(
                    f"glm5_next owner {owner!r}: cannot fuse NVFP4 projections into "
                    f"{fused_path!r}: each source has its own per-tensor scales "
                    "(weight_scale_2, input_scale); exclude the sources from quantization"
                )
            merged = {"weight": torch.cat([g["weight"] for g in groups], dim=0)}
            if all("weight_scale_inv" in g for g in groups):
                merged["weight_scale_inv"] = torch.cat(
                    [g["weight_scale_inv"] for g in groups], dim=0
                )
            load_linear(fused_path, [merged])

    if experts is not None:
        tensors_per_expert = {
            QuantAlgo.FP8_BLOCK_SCALES: 6,
            QuantAlgo.NVFP4: 12,
        }.get(quant_algo, 3)
        expected = tensors_per_expert * len(local_expert_ids)
        if collect_all_expert_input_scales:
            # One input_scale per projection of every non-local expert.
            expected += len(proj_to_w) * (experts.num_experts - len(local_expert_ids))
        if len(moe_weights) != expected:
            raise ValueError(
                f"glm5_next owner {owner!r}: fused MoE collected {len(moe_weights)} expert "
                f"tensors, expected {expected} (local experts {len(local_expert_ids)})"
            )
        experts.load_weights([moe_weights])
        if hasattr(experts, "post_load_weights"):
            experts.post_load_weights()
        filled.update(n for n in params if n.startswith("mlp.experts."))

    # FP8_BLOCK_SCALES' static activation scales: this checkpoint is
    # activation_scheme='dynamic', so they are never read; zero them rather
    # than leave uninitialized memory behind.
    unused = {
        n for n in params if n not in filled and n.endswith(("input_scale", "inv_input_scale"))
    }
    with torch.no_grad():
        for name in unused:
            params[name].zero_()
    unfilled = sorted(set(params) - filled - unused)
    if unfilled:
        raise ValueError(f"glm5_next owner {owner!r}: no checkpoint tensor reached {unfilled}")
