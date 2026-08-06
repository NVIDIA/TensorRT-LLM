# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from bisect import bisect_left
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

import torch

from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig

from ..pyexecutor.guided_decoder import GuidedDecoder
from ..pyexecutor.sampler import TorchSampler
from ..pyexecutor.seq_slot_manager import SeqSlotManager
from ..speculative.interface import SpecMetadata
from .dflash import DFlashSpecMetadata, DFlashWorker
from .draft_target import (DraftTargetOneModelSpecMetadata,
                           DraftTargetOneModelWorker)
from .dspark import DSparkSpecMetadata, DSparkWorker, DSv4DSparkWorker
from .eagle3 import (Eagle3OneModelDynamicTreeResourceManager,
                     Eagle3OneModelSpecMetadata, Eagle3OneModelWorker,
                     Eagle3ResourceManager, Eagle3SpecMetadata, MTPEagleWorker)
from .eagle3_dynamic_tree import Eagle3OneModelDynamicTreeWorker
from .model_drafter import ModelDrafter
from .mtp import MTPHiddenStatesManager, MTPSpecMetadata, MTPWorker
from .mtp_dynamic_tree import (MTPEagleDynamicTreeResourceManager,
                               MTPEagleDynamicTreeWorker)
from .ngram import NGramDrafter, NGramPoolManager
from .pard import PARDSpecMetadata, PARDWorker
from .sa_worker import SASpecMetadata, SAWorker
from .save_hidden_state import (SaveHiddenStatesResourceManager,
                                SaveHiddenStatesSpecMetadata)
from .spec_sampler_base import SpecSampler
from .suffix_automaton import SuffixAutomatonManager

_GEMMA4_SHARED_KV_TARGET_ARCHITECTURES = (
    "Gemma4ForCausalLM",
    "Gemma4ForConditionalGeneration",
)

# MTP structure fields copied from a separate MTP-head checkpoint onto the
# target pretrained config when ``speculative_model`` is set.
# Prefer writable HF fields: NemotronHConfig exposes mtp_hybrid_override_pattern
# as a read-only property derived from mtp_layers_block_type.
_MTP_STRUCTURE_FIELDS_FROM_DRAFT = (
    "num_nextn_predict_layers",
    "mtp_layers_block_type",
    "mtp_block_configs",
)

_MTP_PATTERN_TO_LAYER = {
    "M": "mamba",
    "E": "moe",
    "*": "attention",
    "-": "mlp",
}


def _set_pretrained_config_attr(model_config,
                                name: str,
                                value,
                                *,
                                required: bool = True) -> bool:
    """Set a config field, tolerating read-only properties / strict dataclasses.

    Each write is verified by reading the value back: a class-level property
    shadows ``__dict__``, so writing through ``vars()`` can appear to succeed
    while the config keeps reporting its old value. When ``required`` is False,
    failures are logged and ignored (used for optional fields like
    ``mtp_block_configs``).
    """
    writes = (
        lambda: setattr(model_config, name, value),
        lambda: vars(model_config).__setitem__(name, value),
    )
    for write in writes:
        try:
            write()
        except (TypeError, AttributeError):
            continue
        if getattr(model_config, name, None) == value:
            return True

    message = (f"Unable to set MTP config field '{name}' on "
               f"{type(model_config).__name__}")
    if required:
        raise AttributeError(message)
    logger.warning("%s; keeping the target checkpoint's value.", message)
    return False


def _pattern_to_mtp_layers_block_type(pattern: str) -> list:
    try:
        return [_MTP_PATTERN_TO_LAYER[char] for char in pattern]
    except KeyError as exc:
        raise ValueError(
            f"Invalid mtp_hybrid_override_pattern {pattern!r}: "
            f"expected characters in {sorted(_MTP_PATTERN_TO_LAYER)}") from exc


def _is_mtp_checkpoint_weight_key(key: str) -> bool:
    """Return True for checkpoint keys that belong to MTP heads."""
    return key.startswith("mtp.") or key.startswith("mtp/")


def filter_mtp_checkpoint_weights(weights: dict) -> dict:
    """Drop ``mtp.*`` keys so embedded MTP heads do not override a separate MTP checkpoint."""
    return {
        k: v
        for k, v in weights.items() if not _is_mtp_checkpoint_weight_key(k)
    }


def select_mtp_checkpoint_weights(weights: dict) -> dict:
    """Keep only ``mtp.*`` keys from a (possibly full) checkpoint dict.

    Separate MTP-head checkpoints may still ship unrelated tensors (or a full
    target copy). Loading those into the one-engine model would overwrite the
    already-loaded target backbone and corrupt generation.
    """
    return {
        k: v
        for k, v in weights.items() if _is_mtp_checkpoint_weight_key(k)
    }


def remap_preprocessed_mtp_weights_for_draft_model(
    weights: dict,
    num_hidden_layers: int,
    num_mtp_layers: int,
) -> dict:
    """Map ``model.layers.{{N[+h]}}.*`` keys onto ``mtp_layers.{{h}}.*``.

    Nemotron preprocess rewrites ``mtp.layers.*`` onto the target module path
    ``model.layers.{{num_hidden_layers}}.*``. For a strict draft-only load we
    re-home those keys under ``draft_model.mtp_layers``.
    """
    remapped: dict = {}
    unused: list[str] = []
    for key, value in weights.items():
        matched = False
        for head_idx in range(num_mtp_layers):
            prefix = f"model.layers.{num_hidden_layers + head_idx}."
            if key.startswith(prefix):
                remapped[f"mtp_layers.{head_idx}.{key[len(prefix):]}"] = value
                matched = True
                break
        if not matched:
            unused.append(key)
    if unused:
        sample = ", ".join(unused[:8])
        more = "" if len(unused) <= 8 else f" (+{len(unused) - 8} more)"
        raise ValueError(
            "After MTP preprocess, expected keys under "
            f"'model.layers.{{{num_hidden_layers}+h}}.*' for "
            f"h in [0, {num_mtp_layers}), but found unmatched keys: "
            f"{sample}{more}")
    return remapped


def skip_modules_for_separate_mtp_checkpoint(weights: dict) -> list[str]:
    """Modules to skip when loading a separate MTP checkpoint into draft_model.

    ``shared_head`` is optional across MTP architectures:
    - Nemotron ``mtp.*`` checkpoints omit it (final_layernorm on the last
      sublayer is the trained norm; ``shared_head`` only wraps ``lm_head``).
    - DeepSeek / Qwen / Exaone / Step3 ship ``shared_head.norm`` (and Step3
      also ships a dedicated ``shared_head.output``).

    Skip only when the remapped weight dict has no ``shared_head`` keys so
    strict loading stays architecture-agnostic.
    """
    skip: list[str] = []
    if not any("shared_head" in key for key in weights):
        skip.append("shared_head")
    return skip


def uses_mtp_head_checkpoint(spec_config) -> bool:
    """True when `speculative_model` contains replacement MTP heads."""
    if spec_config is None:
        return False
    return spec_config.uses_replacement_heads


def _refers_to_same_checkpoint(lhs, rhs) -> bool:
    """True when two model references point at the same checkpoint."""
    if lhs is None or rhs is None:
        return False
    if str(lhs) == str(rhs):
        return True
    try:
        return os.path.samefile(str(lhs), str(rhs))
    except OSError:
        # At least one side is not an existing local directory (e.g. a Hub
        # model id), so string equality above was the only comparison.
        return False


def resolve_mtp_checkpoint_source(spec_config, checkpoint_dir) -> None:
    """Keep one-model MTP on the target checkpoint when both paths match.

    Before separate MTP checkpoints were supported, ``speculative_model`` was
    ignored for one-model MTP and the heads always came from the target
    weights. Configs that point ``speculative_model`` at the target checkpoint
    keep that behavior instead of switching to the separate-heads load path,
    which the target checkpoint's key layout may not even satisfy.
    """
    from tensorrt_llm.llmapi.llm_args import (MTPDecodingConfig,
                                              _MTPDraftCheckpointType)
    if not isinstance(spec_config, MTPDecodingConfig):
        return
    if spec_config.speculative_model is None:
        return
    if not _refers_to_same_checkpoint(spec_config.speculative_model,
                                      checkpoint_dir):
        return
    if (spec_config._mtp_draft_checkpoint_type
            != _MTPDraftCheckpointType.TARGET):
        logger.info(
            "speculative_model points at the target checkpoint "
            f"({checkpoint_dir}); loading MTP heads from the target weights.")
        spec_config._mtp_draft_checkpoint_type = _MTPDraftCheckpointType.TARGET


def _load_speculative_model_config_dict(spec_config) -> Optional[dict]:
    """Read ``config.json`` from ``spec_config.speculative_model``, if present."""
    draft_dir = getattr(spec_config, "speculative_model", None)
    if not draft_dir:
        return None
    try:
        cfg_path = os.path.join(str(draft_dir), "config.json")
        if not os.path.isfile(cfg_path):
            return None
        with open(cfg_path) as f:
            return json.load(f)
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        logger.warning(
            f"Unable to read speculative_model config from {draft_dir}: {exc}")
        return None


def _merge_mtp_fields_from_speculative_model(spec_config,
                                             model_config) -> Optional[int]:
    """Overlay MTP structure fields from ``speculative_model`` onto ``model_config``.

    Returns the MTP layer count from the draft checkpoint when available.
    """
    draft_cfg = _load_speculative_model_config_dict(spec_config)
    if not draft_cfg:
        return None

    draft_nextn = draft_cfg.get("num_nextn_predict_layers")
    if draft_nextn is None:
        draft_nextn = draft_cfg.get("mtp_num_hidden_layers")

    for field in _MTP_STRUCTURE_FIELDS_FROM_DRAFT:
        if field in draft_cfg and draft_cfg[field] is not None:
            _set_pretrained_config_attr(
                model_config,
                field,
                draft_cfg[field],
                required=(field != "mtp_block_configs"),
            )

    # HF NemotronHConfig: mtp_hybrid_override_pattern is a read-only property
    # derived from mtp_layers_block_type. Convert the pattern when the draft
    # checkpoint only provides the legacy string form.
    if (draft_cfg.get("mtp_layers_block_type") is None
            and draft_cfg.get("mtp_hybrid_override_pattern") is not None):
        _set_pretrained_config_attr(
            model_config,
            "mtp_layers_block_type",
            _pattern_to_mtp_layers_block_type(
                draft_cfg["mtp_hybrid_override_pattern"]),
        )

    if draft_nextn is not None:
        _set_pretrained_config_attr(model_config, "num_nextn_predict_layers",
                                    draft_nextn)
        return int(draft_nextn)
    return None


def _is_effective_dynamic_tree(spec_config) -> bool:
    # At dynamic_tree_max_topK == 1 the tree collapses to a linear chain; route
    # to the linear Eagle3 one-model path to avoid divergence in tree bookkeeping.
    return (getattr(spec_config, 'use_dynamic_tree', False)
            and getattr(spec_config, 'dynamic_tree_max_topK', 0) > 1)


def _get_draft_vocab_size(spec_config, target_vocab_size: int) -> int:
    """Draft-model vocab size, used to decide whether rejection sampling needs
    the d2t-expanded ``full_draft_probs`` buffer (only when it differs from the
    target vocab).

    Reads the draft model's ``config.json`` from ``spec_config.speculative_model``.
    Eagle3 configs store the target vocab in ``vocab_size`` and the reduced head
    width in ``draft_vocab_size``, so ``draft_vocab_size`` is read first, falling
    back to ``vocab_size`` (or a nested ``text_config.vocab_size``). Returns
    ``target_vocab_size`` (shared vocab, no buffer needed) when there is no
    separate draft model or the config cannot be read.
    """
    draft_dir = getattr(spec_config, "speculative_model", None)
    if not draft_dir:
        return target_vocab_size
    try:
        import json
        import os
        cfg_path = os.path.join(str(draft_dir), "config.json")
        if not os.path.isfile(cfg_path):
            return target_vocab_size
        with open(cfg_path) as f:
            cfg = json.load(f)
        vs = cfg.get("draft_vocab_size") or cfg.get("vocab_size")
        if vs is None:
            vs = (cfg.get("text_config") or {}).get("vocab_size")
        return int(vs) if vs else target_vocab_size
    except (OSError, ValueError, TypeError, AttributeError):
        return target_vocab_size


def get_spec_metadata(spec_config,
                      model_config,
                      max_num_requests,
                      max_num_tokens,
                      spec_resource_manager=None,
                      is_draft_model=False,
                      max_seq_len=262144,
                      num_seq_slots=None):
    metadata = _build_spec_metadata(spec_config,
                                    model_config,
                                    max_num_requests,
                                    max_num_tokens,
                                    spec_resource_manager=spec_resource_manager,
                                    is_draft_model=is_draft_model,
                                    max_seq_len=max_seq_len,
                                    num_seq_slots=num_seq_slots)
    # Set here rather than in each branch below: every one-model mode needs it and
    # the per-mode constructors are easy to miss one of.
    if metadata is not None:
        metadata.enable_penalty = getattr(spec_config, "enable_penalty", False)
    return metadata


def _build_spec_metadata(spec_config,
                         model_config,
                         max_num_requests,
                         max_num_tokens,
                         spec_resource_manager=None,
                         is_draft_model=False,
                         max_seq_len=262144,
                         num_seq_slots=None):
    use_rejection_sampling = getattr(spec_config, "use_rejection_sampling",
                                     False)
    # Slot-indexed buffers (draft_probs) must span the SeqSlotManager pool;
    # DeepSeek-V4 overlap can exceed max_num_requests.
    num_seq_slots = (num_seq_slots
                     if num_seq_slots is not None else max_num_requests)
    vocab_size = getattr(model_config, "vocab_size", 0)
    # Draft-model vocab size, used to gate the d2t-expanded full_draft_probs
    # buffer allocation (see SpecMetadata.prepare_rejection_sampling_buffers).
    draft_vocab_size = _get_draft_vocab_size(spec_config, vocab_size)
    if spec_config.spec_dec_mode.is_mtp_eagle_one_model():
        # MTP Eagle one-model reuses Eagle3 one-model metadata for the
        # unified worker/sampler/slot_ids plumbing, but skips per-layer
        # hidden-state capture: the worker feeds the target model's
        # hidden_states directly into the MTP layer, so we leave
        # layers_to_capture unset and let Eagle3OneModelSpecMetadata default
        # it to an empty tuple. This also keeps post-MLP/MoE fusion enabled
        # on models that gate it on is_layer_capture().
        return Eagle3OneModelSpecMetadata(
            max_draft_len=spec_config.max_draft_len,
            max_total_draft_tokens=spec_config.tokens_per_gen_step - 1,
            spec_dec_mode=spec_config.spec_dec_mode,
            max_num_requests=max_num_requests,
            num_layers=model_config.num_hidden_layers,
            hidden_size=model_config.hidden_size,
            max_num_tokens=max_num_tokens,
            use_rejection_sampling=use_rejection_sampling,
            advanced_sampling_mode=spec_config.advanced_sampling_mode,
            vocab_size=vocab_size,
            num_seq_slots=num_seq_slots,
            draft_vocab_size=draft_vocab_size,
            spec_resource_manager=spec_resource_manager,
            use_dynamic_tree=getattr(spec_config, 'use_dynamic_tree', False),
        )
    if spec_config.spec_dec_mode.is_mtp_vanilla():
        return MTPSpecMetadata(
            max_draft_len=spec_config.max_draft_len,
            max_total_draft_tokens=spec_config.tokens_per_gen_step - 1,
            spec_dec_mode=spec_config.spec_dec_mode,
            mtp_num_modules=spec_config.max_draft_len,
            max_num_requests=max_num_requests,
            mtp_hidden_states_manager=spec_resource_manager,
            use_rejection_sampling=use_rejection_sampling,
            vocab_size=vocab_size,
            draft_vocab_size=draft_vocab_size,
        )
    if spec_config.spec_dec_mode.is_mtp_eagle():
        return Eagle3SpecMetadata(
            max_draft_len=spec_config.max_draft_len,
            max_total_draft_tokens=spec_config.tokens_per_gen_step - 1,
            spec_dec_mode=spec_config.spec_dec_mode,
            max_num_requests=max_num_requests,
            num_layers=model_config.num_hidden_layers,
            hidden_size=model_config.hidden_size,
            max_num_tokens=max_num_tokens,
            dtype=model_config.torch_dtype,
            is_draft_model=is_draft_model,
            eagle3_resource_manager=spec_resource_manager,
            layers_to_capture=None,
            is_mtp_eagle=True,
        )
    if spec_config.spec_dec_mode.is_eagle3():
        effective_dynamic_tree = _is_effective_dynamic_tree(spec_config)
        return Eagle3SpecMetadata(
            max_draft_len=spec_config.max_draft_len,
            max_total_draft_tokens=spec_config.tokens_per_gen_step - 1,
            spec_dec_mode=spec_config.spec_dec_mode,
            max_num_requests=max_num_requests,
            num_layers=model_config.num_hidden_layers,
            hidden_size=model_config.hidden_size,
            max_num_tokens=max_num_tokens,
            dtype=model_config.torch_dtype,
            is_draft_model=is_draft_model,
            eagle3_resource_manager=spec_resource_manager,
            layers_to_capture=spec_config.eagle3_layers_to_capture,
            is_mtp_eagle=False,
            eagle_choices=spec_config.eagle_choices,
            is_spec_dec_tree=spec_config.eagle_choices is not None
            or effective_dynamic_tree,
            is_spec_dec_dynamic_tree=effective_dynamic_tree,
        )
    if spec_config.spec_dec_mode.is_eagle3_one_model():
        return Eagle3OneModelSpecMetadata(
            max_draft_len=spec_config.max_draft_len,
            max_total_draft_tokens=spec_config.tokens_per_gen_step - 1,
            spec_dec_mode=spec_config.spec_dec_mode,
            max_num_requests=max_num_requests,
            num_layers=model_config.num_hidden_layers,
            hidden_size=model_config.hidden_size,
            max_num_tokens=max_num_tokens,
            layers_to_capture=spec_config.eagle3_layers_to_capture,
            use_rejection_sampling=use_rejection_sampling,
            vocab_size=vocab_size,
            draft_vocab_size=draft_vocab_size,
            spec_resource_manager=spec_resource_manager,
            use_dynamic_tree=_is_effective_dynamic_tree(spec_config),
            eagle_choices=spec_config.eagle_choices,
        )
    if spec_config.spec_dec_mode.is_pard():
        return PARDSpecMetadata(
            max_draft_len=spec_config.max_draft_len,
            max_total_draft_tokens=spec_config.tokens_per_gen_step - 1,
            spec_dec_mode=spec_config.spec_dec_mode,
            max_num_requests=max_num_requests,
            spec_resource_manager=spec_resource_manager,
            use_rejection_sampling=use_rejection_sampling,
            vocab_size=vocab_size,
            draft_vocab_size=draft_vocab_size,
        )
    # A standalone DSpark drafter is drafted by DFlashWorker, so it needs the
    # DFlash metadata (paged draft KV, DFlash capture buffer). Only the
    # embedded DeepSeek-V4-Pro draft uses DSparkSpecMetadata and its rolling
    # window. See DSparkDecodingConfig.draft_is_embedded_in_target.
    if spec_config.spec_dec_mode.is_dflash() or (
            spec_config.spec_dec_mode.is_dspark()
            and not spec_config.draft_is_embedded_in_target):
        target_layer_ids = getattr(spec_config, 'target_layer_ids', None)
        return DFlashSpecMetadata(
            max_draft_len=spec_config.max_draft_len,
            max_total_draft_tokens=spec_config.tokens_per_gen_step - 1,
            spec_dec_mode=spec_config.spec_dec_mode,
            max_num_requests=max_num_requests,
            layers_to_capture=target_layer_ids,
            hidden_size=model_config.hidden_size,
            max_num_tokens=max_num_tokens,
            dtype=model_config.torch_dtype,
            use_rejection_sampling=use_rejection_sampling,
            advanced_sampling_mode=spec_config.advanced_sampling_mode,
            vocab_size=vocab_size,
            draft_vocab_size=draft_vocab_size,
        )
    if spec_config.spec_dec_mode.is_dspark():
        target_layer_ids = getattr(spec_config, 'target_layer_ids', None)
        return DSparkSpecMetadata(
            max_draft_len=spec_config.max_draft_len,
            max_total_draft_tokens=spec_config.tokens_per_gen_step - 1,
            spec_dec_mode=spec_config.spec_dec_mode,
            max_num_requests=max_num_requests,
            layers_to_capture=target_layer_ids,
            hidden_size=model_config.hidden_size,
            max_num_tokens=max_num_tokens,
            dtype=model_config.torch_dtype,
            use_rejection_sampling=use_rejection_sampling,
            vocab_size=vocab_size,
            draft_vocab_size=draft_vocab_size,
        )
    if spec_config.spec_dec_mode.is_draft_target_one_model():
        return DraftTargetOneModelSpecMetadata(
            max_draft_len=spec_config.max_draft_len,
            max_total_draft_tokens=spec_config.max_total_draft_tokens,
            spec_dec_mode=spec_config.spec_dec_mode,
            max_num_requests=max_num_requests,
            max_num_tokens=max_num_tokens,
            use_rejection_sampling=use_rejection_sampling,
            vocab_size=vocab_size,
            draft_vocab_size=draft_vocab_size,
        )
    if spec_config.spec_dec_mode.is_save_hidden_states():
        return SaveHiddenStatesSpecMetadata(
            max_draft_len=spec_config.max_draft_len,
            max_total_draft_tokens=1,
            spec_dec_mode=spec_config.spec_dec_mode,
            max_num_requests=max_num_requests,
            num_model_layers=model_config.num_hidden_layers,
            hidden_size=model_config.hidden_size,
            max_num_tokens=max_num_tokens,
            dtype=model_config.torch_dtype,
            resource_manager=spec_resource_manager,
            layers_to_capture=spec_config.eagle3_layers_to_capture,
        )
    if spec_config.spec_dec_mode.is_sa():
        return SASpecMetadata(
            max_draft_len=spec_config.max_draft_len,
            max_total_draft_tokens=spec_config.max_total_draft_tokens,
            spec_dec_mode=spec_config.spec_dec_mode,
            max_num_requests=max_num_requests,
            sa_manager=spec_resource_manager,
            max_matching_ngram_size=spec_config.max_matching_ngram_size,
        )
    if  spec_config.spec_dec_mode.is_draft_target() or \
        spec_config.spec_dec_mode.is_ngram() or \
        spec_config.spec_dec_mode.is_user_provided():
        return SpecMetadata(
            max_draft_len=spec_config.max_draft_len,
            max_total_draft_tokens=spec_config.tokens_per_gen_step - 1,
            spec_dec_mode=spec_config.spec_dec_mode,
            max_num_requests=max_num_requests,
        )
    return None


def get_mtp_hidden_size(model_config) -> int:
    pretrained_config = getattr(model_config, "pretrained_config", model_config)
    hidden_size = getattr(pretrained_config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(model_config, "hidden_size")
    if getattr(pretrained_config, "model_type", None) == "deepseek_v4":
        return hidden_size * getattr(pretrained_config, "hc_mult", 1)
    return hidden_size


def get_spec_resource_manager(model_engine, draft_model_engine=None):
    spec_config = model_engine.spec_config
    if spec_config is None:
        return None
    model_config = model_engine.model.config
    max_num_requests = model_engine.batch_size
    max_seq_len = model_engine.max_seq_len
    max_num_tokens = model_engine.max_num_tokens
    spec_dec_mode = spec_config.spec_dec_mode
    if spec_dec_mode.is_mtp_eagle_one_model():
        sa_manager = None
        sa_cfg = getattr(spec_config, 'sa_config', None)
        if sa_cfg is not None:
            sa_manager = SuffixAutomatonManager(sa_cfg, max_num_requests,
                                                max_seq_len)
        # Dynamic tree combines SpecTreeManager with MTP hidden-state slots.
        if getattr(spec_config, 'use_dynamic_tree', False):
            return MTPEagleDynamicTreeResourceManager(
                spec_config,
                model_config.torch_dtype,
                model_config.hidden_size,
                max_num_requests,
                sa_manager=sa_manager,
            )
        if spec_config.use_relaxed_acceptance_for_thinking or sa_manager is not None:
            # Unified resource manager: the unified worker reads
            # ``relaxed_delta_pool`` from ``Eagle3ResourceManager`` (mirrors the
            # pool ``MTPHiddenStatesManager`` used to provide).
            return Eagle3ResourceManager(
                spec_config,
                model_config.torch_dtype,
                get_mtp_hidden_size(model_config),
                max_num_requests,
                max_seq_len,
                max_num_tokens,
                sa_manager=sa_manager,
            )
        else:
            return None
    if spec_dec_mode.is_mtp_vanilla():
        sa_manager = None
        sa_cfg = getattr(spec_config, 'sa_config', None)
        if sa_cfg is not None:
            sa_manager = SuffixAutomatonManager(sa_cfg, max_num_requests,
                                                max_seq_len)
        return MTPHiddenStatesManager(
            spec_config,
            model_config.torch_dtype,
            get_mtp_hidden_size(model_config),
            max_num_requests,
            sa_manager=sa_manager,
        )
    if spec_dec_mode.is_eagle3_one_model() and _is_effective_dynamic_tree(
            spec_config):
        return Eagle3OneModelDynamicTreeResourceManager(spec_config,
                                                        max_num_requests)
    if spec_dec_mode.is_eagle3_one_model():
        sa_manager = None
        sa_cfg = getattr(spec_config, 'sa_config', None)
        if sa_cfg is not None:
            sa_manager = SuffixAutomatonManager(sa_cfg, max_num_requests,
                                                max_seq_len)
        return Eagle3ResourceManager(
            spec_config,
            model_config.torch_dtype,
            model_config.hidden_size,
            max_num_requests,
            max_seq_len,
            max_num_tokens,
            sa_manager=sa_manager,
        )
    if spec_dec_mode.is_eagle3() or spec_dec_mode.is_mtp_eagle():
        assert draft_model_engine is not None, "Draft model engine is required for Eagle3 and MTP Eagle two model flow."
        return Eagle3ResourceManager(
            spec_config,
            draft_model_engine.model.config.torch_dtype,
            model_config.hidden_size,
            max_num_requests,
            max_seq_len,
            max_num_tokens,
        )
    if spec_dec_mode.is_save_hidden_states():
        return SaveHiddenStatesResourceManager(
            spec_config,
            model_engine.model.config.torch_dtype,
            model_config.hidden_size,
            max_num_requests,
            max_num_tokens,
        )
    if spec_dec_mode.is_parallel_draft():
        sa_cfg = getattr(spec_config, 'sa_config', None)
        if sa_cfg is not None:
            return SuffixAutomatonManager(sa_cfg, max_num_requests, max_seq_len)
        return None
    if spec_dec_mode.is_ngram():
        return NGramPoolManager(spec_config, max_num_requests)
    if spec_dec_mode.is_sa():
        return SuffixAutomatonManager(spec_config, max_num_requests,
                                      max_seq_len)
    if spec_dec_mode.is_user_provided():
        return spec_config.resource_manager
    return None


def get_spec_decoder(
    sampler_args: TorchSampler.Args,
    spec_config: "DecodingBaseConfig",
):
    spec_dec_mode = spec_config.spec_dec_mode
    if spec_dec_mode.is_eagle3() or spec_dec_mode.is_mtp_eagle():
        # Two-model path: the target model emits logits, so the general-purpose
        # TorchSampler does the actual sampling (and folds in the d2t vocab
        # mapping). One-model modes below sample inside the worker kernel.
        return TorchSampler(sampler_args)
    if spec_dec_mode.use_one_engine():
        # One sampler for every one-model mode (use_one_engine covers MTP,
        # MTP Eagle, Eagle3, PARD/DFlash/DSpark, DraftTarget and SA): it only
        # moves the worker's pre-sampled output around, and its buffer shapes
        # derive from sampler_args alone.
        #
        # WORKAROUND (remove with eagle_choices in release 1.4): the static
        # tree is the one mode where a step can accept more than
        # max_draft_len + 1 tokens. The one-model drafter never builds the tree
        # -- _forward_draft_loop is linear over runtime_draft_len, which for a
        # non-linear tree is max_total_draft_tokens -- so max_draft_len only
        # describes a tree depth that is never used, and acceptance is bounded
        # by the wire width instead. Tree-aware acceptance only exists in the
        # two-model TorchSampler path, which is deprecated alongside this.
        accepted_path_len = None
        if getattr(spec_config, "eagle_choices", None):
            accepted_path_len = sampler_args.max_total_draft_tokens + 1
        # Occurrence penalties assume the linear row layout: one logits row per
        # speculative position, so a position's prefix is the positions before it.
        # A tree's rows are nodes whose prefix is their root path instead, and
        # sibling branches must not penalize each other -- so tree modes are not
        # supported yet and are rejected at admission rather than mispenalized.
        penalty_supported = not (getattr(spec_config, "eagle_choices", None)
                                 or _is_effective_dynamic_tree(spec_config))
        return SpecSampler(sampler_args,
                           accepted_path_len=accepted_path_len,
                           enable_penalty=spec_config.enable_penalty,
                           penalty_supported=penalty_supported)
    raise ValueError(
        f"Unsupported speculative decoding mode: {spec_config.spec_dec_mode}")


def get_spec_drafter(model_engine,
                     draft_model_engine,
                     sampler,
                     spec_resource_manager,
                     guided_decoder: Optional[GuidedDecoder] = None):
    spec_config = model_engine.spec_config
    if spec_config is None:
        return None

    if spec_config.spec_dec_mode.is_user_provided():
        return spec_config.drafter

    max_num_requests = model_engine.batch_size
    if spec_config.spec_dec_mode.is_draft_target(
    ) or spec_config.spec_dec_mode.is_eagle3(
    ) or spec_config.spec_dec_mode.is_mtp_eagle():
        return ModelDrafter(spec_config,
                            draft_model_engine,
                            spec_config.max_draft_len,
                            spec_config.tokens_per_gen_step - 1,
                            SeqSlotManager(max_num_requests),
                            sampler,
                            spec_resource_manager=spec_resource_manager,
                            guided_decoder=guided_decoder)

    if spec_config.spec_dec_mode.is_ngram():
        return NGramDrafter(spec_config, spec_resource_manager)

    return None


def get_num_spec_layers(spec_config):
    if getattr(spec_config, "_use_shared_kv_cache", False):
        return 0
    if spec_config.spec_dec_mode.is_mtp_eagle_one_model():
        return 1
    if spec_config.spec_dec_mode.is_mtp_vanilla():
        return spec_config.num_nextn_predict_layers
    if spec_config.spec_dec_mode.is_eagle3_one_model():
        num_draft_hidden_layers = spec_config._num_draft_hidden_layers
        return num_draft_hidden_layers if num_draft_hidden_layers is not None else 1
    return 0


def update_spec_config_from_draft_model_config(spec_config,
                                               draft_pretrained_config) -> None:
    """Populate Eagle draft-layer fields from the loaded draft model config."""
    from tensorrt_llm.llmapi.llm_args import EagleDecodingConfig

    if not isinstance(spec_config, EagleDecodingConfig):
        return

    num_layers = getattr(draft_pretrained_config, "num_hidden_layers", None)
    if num_layers is None:
        logger.warning(
            "Draft model pretrained config is missing num_hidden_layers; "
            "defaulting _num_draft_hidden_layers to 1.")
        num_layers = 1
    spec_config._num_draft_hidden_layers = num_layers


def get_spec_worker(spec_config,
                    model_config,
                    mapping,
                    use_separate_draft_kv_cache: bool = False):
    spec_dec_mode = spec_config.spec_dec_mode
    if spec_dec_mode.is_mtp_vanilla():
        return MTPWorker(spec_config,
                         model_config,
                         use_separate_draft_kv_cache,
                         mapping=mapping)
    if spec_dec_mode.is_mtp_eagle_one_model():
        if getattr(spec_config, 'use_dynamic_tree', False):
            return MTPEagleDynamicTreeWorker(spec_config,
                                             model_config,
                                             use_separate_draft_kv_cache,
                                             mapping=mapping)
        return MTPEagleWorker(spec_config,
                              model_config,
                              use_separate_draft_kv_cache,
                              mapping=mapping)
    if spec_dec_mode.is_eagle3_one_model():
        if _is_effective_dynamic_tree(spec_config):
            return Eagle3OneModelDynamicTreeWorker(spec_config, mapping,
                                                   use_separate_draft_kv_cache)
        return Eagle3OneModelWorker(
            spec_config,
            mapping=mapping,
            use_separate_draft_kv_cache=use_separate_draft_kv_cache)
    if spec_dec_mode.is_pard():
        return PARDWorker(spec_config, mapping, use_separate_draft_kv_cache)
    if spec_dec_mode.is_dflash():
        return DFlashWorker(spec_config, mapping, use_separate_draft_kv_cache)
    # DSpark splits by deployment form, mirroring the draft-model side. The
    # embedded DeepSeek-V4-Pro draft needs DSv4DSparkWorker, whose rolling-window
    # plumbing reads V4-draft-only attributes (num_stages, write_context_windows,
    # forward_batched). A standalone drafter is DFlash lineage and is served by
    # DSparkWorker, which adds only the Markov bias and the shift_label
    # slot convention on top of DFlashWorker.
    if spec_dec_mode.is_dspark():
        if spec_config.draft_is_embedded_in_target:
            return DSv4DSparkWorker(spec_config, mapping,
                                    use_separate_draft_kv_cache)
        return DSparkWorker(spec_config, mapping, use_separate_draft_kv_cache)
    if spec_dec_mode.is_sa():
        return SAWorker(spec_config, model_config)
    if spec_dec_mode.is_draft_target_one_model():
        return DraftTargetOneModelWorker(spec_config, mapping,
                                         use_separate_draft_kv_cache)
    return None


def get_num_extra_kv_tokens(spec_config):
    """
    Implementation detail for one model implementations of speculative decoding. Extra
    KV cache tokens are required.
    """
    if spec_config is None:
        return 0
    if getattr(spec_config, "_use_shared_kv_cache", False):
        return 0
    if spec_config.spec_dec_mode.use_one_engine():
        return spec_config.max_draft_len - 1
    return 0


def get_draft_kv_cache_manager(spec_config, resource_manager):
    """
    Returns the draft KV cache manager only in one-model speculative decoding
    mode where the target model manages a separate draft KV cache.
    """
    from ..pyexecutor.resource_manager import ResourceManagerType

    if spec_config is None:
        return None
    if not spec_config.spec_dec_mode.use_one_engine():
        return None
    return resource_manager.get_resource_manager(
        ResourceManagerType.DRAFT_KV_CACHE_MANAGER)


def update_spec_config_from_model_config(spec_config,
                                         model_config,
                                         target_model_cls=None):
    from tensorrt_llm.llmapi.llm_args import (MTPDecodingConfig,
                                              _MTPDraftCheckpointType)
    if not isinstance(spec_config, MTPDecodingConfig):
        return

    architectures = getattr(model_config, "architectures", None) or ()
    if (architectures
            and architectures[0] in _GEMMA4_SHARED_KV_TARGET_ARCHITECTURES):
        spec_config._use_shared_kv_cache = (
            spec_config.spec_dec_mode.is_mtp_eagle_one_model())

    # The target implementation owns the contract for its MTP drafter. Some one-model MTP
    # implementations construct `MTPForCausalLM` from the target config, and optionally load a
    # head replacement checkpoint (e.g. NemotronH).
    # Other implementations advertise an external assistant architecture, which must be
    # constructed from the assistant's own config.
    checkpoint_type = spec_config._mtp_draft_checkpoint_type
    if spec_config.speculative_model is None:
        checkpoint_type = _MTPDraftCheckpointType.TARGET
    elif checkpoint_type != _MTPDraftCheckpointType.TARGET:
        if target_model_cls is not None:
            checkpoint_type = (
                _MTPDraftCheckpointType.EXTERNAL_DRAFT_MODEL if getattr(
                    target_model_cls, "build_mtp_draft_model_from_config",
                    False) else _MTPDraftCheckpointType.HEAD_REPLACEMENT)
        elif checkpoint_type == _MTPDraftCheckpointType.UNRESOLVED:
            checkpoint_type = _MTPDraftCheckpointType.HEAD_REPLACEMENT
    spec_config._mtp_draft_checkpoint_type = checkpoint_type

    # When MTP heads live in a separate checkpoint, prefer that checkpoint's
    # layer count / pattern over the target model's (which may have no MTP or
    # an older embedded MTP head that will be overridden at weight load).
    draft_nextn = None
    if uses_mtp_head_checkpoint(spec_config):
        draft_nextn = _merge_mtp_fields_from_speculative_model(
            spec_config, model_config)

    # Read the MTP layer count from the model's pretrained config. This
    # determines the actual MTP layer count in the checkpoint and drives the
    # spec_dec_mode decision (EAGLE vs vanilla MTP). Different checkpoints expose
    # this under different names: DeepSeek-style configs use
    # `num_nextn_predict_layers`, while Qwen3Next-style configs (including
    # Qwen3.5) use `mtp_num_hidden_layers`. Fall back to a single shared MTP /
    # EAGLE layer when neither field is present.
    if draft_nextn is not None:
        num_nextn_predict_layers = draft_nextn
    else:
        num_nextn_predict_layers = getattr(model_config,
                                           "num_nextn_predict_layers", None)
        if num_nextn_predict_layers is None:
            num_nextn_predict_layers = getattr(model_config,
                                               "mtp_num_hidden_layers", None)
        if num_nextn_predict_layers is None:
            num_nextn_predict_layers = 1
    spec_config.num_nextn_predict_layers = num_nextn_predict_layers
    spec_config._validate_moe_backend_compatibility(model_config_resolved=True)
    is_vanilla = spec_config.spec_dec_mode.is_mtp_vanilla()

    # Resolve max_draft_len when the user didn't set it:
    #   vanilla MTP -> use all checkpoint MTP heads
    #   MTP-Eagle   -> replay the single head once
    if spec_config.max_draft_len is None:
        spec_config.max_draft_len = (spec_config.num_nextn_predict_layers
                                     if is_vanilla else 1)
    elif is_vanilla and spec_config.max_draft_len != spec_config.num_nextn_predict_layers:
        effective_draft_len = min(spec_config.max_draft_len,
                                  spec_config.num_nextn_predict_layers)
        logger.warning(
            f"MTP: max_draft_len ({spec_config.max_draft_len}) does not match "
            f"num_nextn_predict_layers ({spec_config.num_nextn_predict_layers}); "
            f"using max_draft_len={effective_draft_len} draft tokens.")
        spec_config.max_draft_len = effective_draft_len

    if not spec_config.use_dynamic_tree:
        spec_config.max_total_draft_tokens = spec_config.max_draft_len


def update_spec_config_from_loaded_model(spec_config, model) -> None:
    """Populate spec config fields from loaded target and draft model configs."""
    update_spec_config_from_model_config(spec_config, model.config)
    draft_config = getattr(model, 'draft_config', None)
    if draft_config is not None:
        update_spec_config_from_draft_model_config(
            spec_config, draft_config.pretrained_config)


@dataclass
class SpecDecodingTensor:
    """
    Container for speculative decoding tensor parameters.

    Attributes:
        position_offsets: Position offsets for speculative decoding
        packed_mask: Packed attention mask for speculative decoding
        generation_lengths: Optional generation lengths for speculative decoding
    """
    position_offsets: torch.Tensor
    packed_mask: torch.Tensor
    generation_lengths: Optional[torch.Tensor] = None


def get_draft_len_for_batch_size(draft_len_schedule: Dict[int, int],
                                 batch_size: int, max_draft_len: int) -> int:
    """
    Get the appropriate draft length for the given batch size using binary search.

    This is a standalone function that can be used by both the drafter (two-model path)
    and the model engine / spec workers (one-model path).

    New semantics: Keys represent specific batch sizes (transition points).
    Values represent draft_len to use for batch sizes UP TO that key.

    Args:
        draft_len_schedule: Mapping from batch size thresholds to draft lengths.
                            Example: {4: 4, 8: 2, 32: 1} means:
                            - batch size 1-4:   use draft_len=4 (up to key 4)
                            - batch size 5-8:   use draft_len=2 (up to key 8)
                            - batch size 9-32:  use draft_len=1 (up to key 32)
                            - batch size 33+:   use draft_len=0 (speculation disabled, implicit)
        batch_size: Current batch size.
        max_draft_len: Maximum draft length to use if no schedule is provided.

    Returns:
        The draft length to use for this batch size.
    """
    if draft_len_schedule is None:
        return max_draft_len

    # Binary search to find the first threshold >= batch_size
    # draft_len_schedule is already sorted by config validator
    schedule_batch_sizes = list(draft_len_schedule.keys())

    # bisect_left finds where to insert batch_size to keep list sorted
    # This gives us the index of the first key >= batch_size
    idx = bisect_left(schedule_batch_sizes, batch_size)

    if idx < len(schedule_batch_sizes):
        return draft_len_schedule[schedule_batch_sizes[idx]]

    # batch_size > all batch sizes in draft_len_schedule: speculation disabled (implicit)
    return 0
