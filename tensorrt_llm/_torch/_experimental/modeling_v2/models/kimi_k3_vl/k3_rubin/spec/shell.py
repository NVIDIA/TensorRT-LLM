# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Speculative shell of the Kimi K3 Rubin targets: the package's own DSpark drafter and worker.

The K3 text model (``k3_rubin/text_model.py``, ``KimiLinearForCausalLM``) is built by
main's one-engine shell, ``SpecDecOneEngineForCausalLM.__init__``, which builds main's
DSpark drafter through the draft-model registry (``get_draft_model`` ->
``_build_dspark_draft``) and main's ``DSparkWorker`` through ``get_spec_worker``.
:func:`install_k3_dspark` then replaces both with the package's own: the drafter is
``modeling_dspark_mla.MLADSparkForCausalLM`` (built by ``build_k3_dspark_draft``, no
registry entry) and the worker ``dspark.DSparkWorker`` on ``dflash.DFlashWorker``, which
carry the K3 Rubin DSpark fusions merged onto
main's engine contract.

Main's drafter is constructed and discarded. The model loader constructs models under
``MetaInitMode`` (``pyexecutor/model_loader.py``), so that costs no device memory; only
the loader's non-meta fallback allocates it once, and it is freed with the old modules.

Only one-engine DSpark with a standalone MLA drafter is replaced. SA, DFlash, a GQA
DSpark drafter and a draft embedded in the target keep main's drafter and worker, so the
target stays identical to main for them.

Contract with the text model (``KimiLinearForCausalLM._install_private_spec``): call
``install_k3_dspark(self, model_config)`` once, right after
``SpecDecOneEngineForCausalLM.__init__`` and before the helix CP mapping is restored, so
the package's drafter sees the mapping main's drafter was built under. It swaps
``draft_model`` / ``spec_worker`` in ``self.epilogue`` too, calls
``set_draft_model`` / ``set_target_model(self)`` on the new worker, leaves ``self.lm_head``
and ``self.logits_processor`` untouched (main's vocab-parallel selection requires an
``LMHead`` and exactly a ``LogitsProcessor``), and the package's worker keeps the attribute
contract main's ``DFlashSpecMetadata.prepare()`` reads (``_ctx_buf_inited``,
``_req_to_slot``, ``_req_ctx_pos``, ``_free_slots``, ``_write_ctx_len``,
``_graph_dummy_id_floor``, ``_assign_slot``, ``_dummy_slot``, ``_batch_to_slot_buffer``).
"""

from typing import Optional

from torch import nn

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor.config_utils import is_mla
from tensorrt_llm.logger import logger


def uses_k3_dspark(spec_config) -> bool:
    """Whether ``spec_config`` is one-engine DSpark with a separately shipped drafter.

    Whether that drafter is MLA-shaped, and so served by the package's classes, is only
    known from its loaded config (:func:`install_k3_dspark`).
    """
    return (
        spec_config is not None
        and spec_config.spec_dec_mode.use_one_engine()
        and spec_config.spec_dec_mode.is_dspark()
        and not bool(getattr(spec_config, "draft_is_embedded_in_target", False))
    )


def _replace_in_epilogue(target: nn.Module, old: Optional[nn.Module], new: nn.Module) -> None:
    """Put ``new`` where main's shell appended ``old`` (by identity), else append it."""
    for i, module in enumerate(target.epilogue):
        if module is old:
            target.epilogue[i] = new
            return
    target.epilogue.append(new)


def install_k3_dspark(target: nn.Module, model_config: ModelConfig) -> bool:
    """Replace main's DSpark drafter and worker on ``target`` with the package's own.

    Args:
        target: a ``SpecDecOneEngineForCausalLM`` whose ``__init__`` just ran (it reads
            ``draft_config``, ``draft_model``, ``spec_worker``, ``use_separate_draft_kv_cache``,
            ``epilogue``, ``lm_head``, ``logits_processor`` and ``model``).
        model_config: the target's ``ModelConfig``, with the mapping main's drafter was
            built under still in place.

    Returns:
        True when the package's drafter and worker were installed; False when the
        configuration keeps main's (no DSpark, embedded draft, GQA drafter).
    """
    spec_config = getattr(model_config, "spec_config", None)
    if not uses_k3_dspark(spec_config):
        return False
    # Main's shell built the drafter's own ModelConfig for the external drafter
    # (external_drafter_config_kwargs: mapping_for_full_rows(), the draft KV quant
    # algo, the target's extra_attrs); the package's drafter is built on the same one,
    # which the KV cache creator also reads.
    draft_config = target.draft_config
    if draft_config is None or not is_mla(draft_config.pretrained_config):
        logger.info("K3 DSpark: not an MLA drafter; main's DSpark drafter and worker serve it")
        return False

    # Imported here: only an MLA DSpark deployment pays for the package's drafter and
    # worker modules (and the kernel modules they import) at import time.
    from .dspark import DSparkWorker
    from .modeling_dspark_mla import build_k3_dspark_draft

    draft_model = build_k3_dspark_draft(model_config, draft_config, target.lm_head, target.model)
    if spec_config.spec_dec_mode.is_parallel_draft():
        draft_model.logits_processor = target.logits_processor
    # The drafter runs on every row: the mapping main hands get_spec_worker.
    spec_worker = DSparkWorker(
        spec_config,
        model_config.mapping.mapping_for_full_rows(),
        target.use_separate_draft_kv_cache,
    )

    _replace_in_epilogue(target, target.draft_model, draft_model)
    _replace_in_epilogue(target, target.spec_worker, spec_worker)
    target.draft_model = draft_model
    target.spec_worker = spec_worker
    # As SpecDecOneEngineForCausalLM.__init__: cache the drafter's static vocab map
    # and bind the worker to both models (vocab-parallel selection binds the heads
    # in setup_aliases / post_load_weights, after loading).
    spec_worker.set_draft_model(draft_model)
    spec_worker.set_target_model(target)
    logger.info(
        f"K3 DSpark: private drafter {type(draft_model).__module__}.{type(draft_model).__name__}"
        f" and worker {type(spec_worker).__module__}.{type(spec_worker).__name__}"
    )
    return True


__all__ = ["install_k3_dspark", "uses_k3_dspark"]
