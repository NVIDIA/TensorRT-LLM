# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``context_prep``: a stage's context steps run the pre-attention prep as one op per layer.

The stage's layers' attention is re-classed as the TEP target's ``MuseSparkOpenTepAttention``
(the class its ``preattn_prep`` install sets), so a layer call that carries ``context_prep``
-- only ``modeling.PrefillTarget`` passes it -- runs the ``preattn_prep`` op on the step's
QKV rows in place of the built-in ``preprocess_qkv`` chain, with the same bits (Q / K
RMSNorm, the interleaved RoPE of the sliding-window layers, the gate copy). The op reads the
head counts off the layer (``q_size`` / ``kv_size``), so it runs the stage's 128 query / 16
KV heads as it runs a TEP4 rank's 32 / 4. Every other call, and every decode step, runs the
built-in prep.

Each re-classed attention also zeroes the rows of its output that a replayed prefill CUDA
graph pads a step with (``_zero_padded_rows``), and, with ``context_prep_rows_vec``, takes
the op's vectorized rows kernel. Only the stage's own layers are re-classed: the others
were stripped by ``MuseSparkOpenModel.__pp_init__`` and never run.

Construction-time; agreed over the stages (``dispatch.agreed_install``): every stage checks
its layers, the outcome is agreed, and only then are the layers re-classed.
"""

from __future__ import annotations

from tensorrt_llm.logger import logger

from ... import dispatch as _dispatch
from ..._tep import module as _tep_module


def install(text_model, mapping, plan: "_dispatch.Plan") -> int:
    """Re-class every stage layer's attention for ``context_prep``; returns the layers re-classed.

    Every layer is checked, the outcome agreed over the stages, and only then is any layer
    re-classed, so a refusal on any stage changes nothing. Raises ``ValueError`` on every
    stage alike when some stage refuses.
    """
    attention_class = _tep_module("attention").MuseSparkOpenTepAttention
    layers = _dispatch.owned_layers(text_model)

    def check() -> None:
        for layer in layers:
            attention_class.check(layer.self_attn)

    _dispatch.agreed_install(mapping, check, (ValueError, TypeError))
    row_vec = plan.enabled("context_prep_rows_vec")
    for layer in layers:
        attention_class.adopt(layer.self_attn)
        layer.self_attn._zero_padded_rows = True
        if row_vec:
            layer.self_attn._context_prep_row_vec = True
    logger.info(
        f"{_dispatch.TAG} context_prep: context steps' pre-attention prep in one launch on the "
        f"{len(layers)} layers of stage {text_model._pp_stage}"
        + (" on the vectorized rows kernel" if row_vec else "")
        + "; a replayed prefill graph's padded rows get a zero attention output"
    )
    return len(layers)
