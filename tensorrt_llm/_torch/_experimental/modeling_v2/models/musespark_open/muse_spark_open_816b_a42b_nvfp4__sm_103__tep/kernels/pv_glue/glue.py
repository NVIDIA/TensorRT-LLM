# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The identity row select before the target LM head.

Part of the ``last_layer_boundary`` fast path. On a decode step the built-in
``SpecDecOneEngineForCausalLM.forward`` indexes ``hidden_states[spec_metadata.gather_ids]``
(an index-dtype copy and an index kernel) before the LM head. ``install_row_select``
re-classes the target decoder so that, on a step with no context rows whose gather ids
are as many as the rows, the LM head reads the rows themselves instead. The executor
builds the gather ids of a generation request as the consecutive range of its rows
(``engine/runners/decoder/runner.py``: ``gather_ids.extend(range(len(position_ids),
...))`` next to ``position_ids.extend`` of the same length), so on such a step they are
``0 .. rows - 1`` and the index is the identity: a copy of the same bytes. Every other
step runs the built-in body.
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.models import modeling_speculative as _spec
from tensorrt_llm.logger import logger


# ---- the identity row select ------------------------------------------------------
class PvRowSelectCausalLM(_spec.SpecDecOneEngineForCausalLM):
    """``SpecDecOneEngineForCausalLM`` whose generation-only steps skip the identity row select.

    Never constructed: ``install_row_select`` re-classes the decoder as a class derived
    from this one and its current class. ``forward`` is the built-in body statement for
    statement, except that the LM head reads ``hidden_states`` itself when
    ``identity_rows`` holds.
    """

    def forward(
        self,
        attn_metadata,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        spec_metadata=None,
        resource_manager=None,
        **kwargs,
    ) -> torch.Tensor:
        if self.spec_worker is None or spec_metadata is None:
            return super().forward(
                attn_metadata,
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                return_context_logits=return_context_logits,
                spec_metadata=spec_metadata,
                resource_manager=resource_manager,
                **kwargs,
            )
        hidden_states = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            **kwargs,
        )
        if spec_metadata is not None and spec_metadata.is_layer_capture(self.layer_idx):
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states)
        if attn_metadata.padded_num_tokens is not None:
            hidden_states = hidden_states[: attn_metadata.num_tokens]

        # get logits (the row select is the identity on a generation-only step)
        if identity_rows(attn_metadata, spec_metadata, hidden_states):
            selected = hidden_states
        else:
            selected = hidden_states[spec_metadata.gather_ids]
        logits = self.logits_processor.forward(
            selected,
            self.lm_head,
            attn_metadata,
            True,
        )

        spec_input_ids = input_ids if input_ids is not None else kwargs.get("orig_input_ids")
        spec_position_ids = position_ids
        if attn_metadata.padded_num_tokens is not None:
            if spec_input_ids is not None:
                spec_input_ids = spec_input_ids[: attn_metadata.num_tokens]
            if position_ids is not None:
                spec_position_ids = _spec._slice_spec_position_ids(
                    position_ids, attn_metadata.num_tokens
                )

        # get accepted tokens and next draft tokens
        return self.spec_worker(
            input_ids=spec_input_ids,
            position_ids=spec_position_ids,
            hidden_states=hidden_states,
            logits=logits,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=self.draft_model,
            resource_manager=resource_manager,
        )


def identity_rows(attn_metadata, spec_metadata, hidden_states: torch.Tensor) -> bool:
    """Whether ``hidden_states[spec_metadata.gather_ids]`` is ``hidden_states`` (host facts only)."""
    gather_ids = getattr(spec_metadata, "gather_ids", None)
    return (
        isinstance(gather_ids, torch.Tensor)
        and getattr(attn_metadata, "num_contexts", None) == 0
        and gather_ids.dim() == 1
        and hidden_states.dim() == 2
        and int(gather_ids.shape[0]) == int(hidden_states.shape[0])
    )


_COMPOSED: dict = {}


def row_select_refusal(llm) -> Optional[str]:
    """Why the row select cannot be skipped on this decoder, or None."""
    if not isinstance(llm, _spec.SpecDecOneEngineForCausalLM):
        return f"the decoder is {type(llm).__name__}, not a SpecDecOneEngineForCausalLM"
    if type(llm).forward is not _spec.SpecDecOneEngineForCausalLM.forward and not isinstance(
        llm, PvRowSelectCausalLM
    ):
        return f"{type(llm).__name__} overrides forward"
    if "forward" in vars(llm):
        return "forward is replaced on the decoder instance"
    return None


def install_row_select(llm) -> bool:
    """Re-class ``llm`` to skip the identity row select (idempotent); whether it is installed. Rank-local."""
    if isinstance(llm, PvRowSelectCausalLM):
        return True
    reason = row_select_refusal(llm)
    if reason is not None:
        logger.warning(f"[musespark_open] pv_glue: the row select stays ({reason})")
        return False
    cls = type(llm)
    composed = _COMPOSED.get(cls)
    if composed is None:
        composed = type(
            f"PvRowSelect{cls.__name__}",
            (PvRowSelectCausalLM, cls),
            {"__module__": __name__, "__doc__": f"``{cls.__qualname__}`` with pv's row select."},
        )
        _COMPOSED[cls] = composed
    llm.__class__ = composed
    logger.info(
        f"[musespark_open] pv_glue: identity row select skipped on generation-only steps ({composed.__name__})"
    )
    return True
