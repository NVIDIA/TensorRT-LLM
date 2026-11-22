# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 worker differences of the sm_107 targets and the NVFP4 default from the engine's DFlash worker.

``RubinDFlash2WorkerMixin`` goes before the engine worker in a target worker's bases:

* the TP ranks agree on the drafts (``agree_draft_tokens``, which the worker's ``_forward_impl``
  calls on its outputs). Under TP without attention DP every rank drafts the same requests itself,
  and the DFlash2 selector can split a near-tie between ranks. A rank whose draft differs accepts
  a different count, finishes the request a step apart and waits in a TP collective alone. The
  request-sharing TP group (Helix CP folded in) takes its rank 0's drafts
  (``rubin_draft_token_agreement``): one int32 all-gather, no host sync, capturable;
* an FP8 E4M3 drafter history (the drafter's ``context_kv_fp8``) stores the MLA latent
  saturated (``to_draft_fp8``), as its block decode quantizes the query that reads it; other
  histories store as the engine does. Under Helix a context chunk's latents go to this rank's
  pages only (``_helix_shard``). The engine's fused generation-step preparation
  (``_prepare_paged_latent_gen_inputs``) stores the latents the drafter returns and requires the
  pool's dtype: over an FP8 history it gets the drafter with its context latents saturated
  (``Fp8HistoryDrafter``);
* the draft step's KV-length prep refreshes only the host-side attention metadata when the target
  says its drafter reads none of it (``_draft_reads_target_metadata``), as the family DSA
  metadata allows (``update_for_spec_dec(refresh_device_metadata=False)``);
* forced acceptance (``TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS``) applies to the generation
  rows that verify drafts the drafter made. A request's first generation step after a context step
  or a disaggregated handoff verifies drafts that count as placeholders, as for the engine's
  parallel-draft modes, and keeps its own count (``placeholder_rows``, ``_keep_placeholder_counts``).
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from tensorrt_llm._torch.distributed.ops import allgather
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .draft_context_cache import to_draft_fp8


def rubin_draft_config_kwargs(model_config: Any, spec_config: Any) -> dict:
    """``ModelConfig.from_pretrained`` arguments of the DFlash2 drafter: its own checkpoint under
    the target's execution layout (every row of the TP group), the target's MoE backend unless the
    speculative config names one, and no speculative decoding of its own."""
    moe_backend = getattr(spec_config, "moe_backend", None)
    return dict(
        trust_remote_code=True,
        attn_backend=model_config.attn_backend,
        moe_backend=model_config.moe_backend if moe_backend is None else moe_backend,
        mapping=model_config.mapping.mapping_for_full_rows(),
        spec_config=None,
        max_num_tokens=model_config.max_num_tokens,
        moe_max_num_tokens=model_config.moe_max_num_tokens,
        use_cute_dsl_blockscaling_mm=model_config.use_cute_dsl_blockscaling_mm,
        # The user's value: the drafter's position table follows the positions the worker serves.
        max_seq_len=model_config.max_seq_len,
    )


def draft_agreement_mapping(mapping: Any) -> Optional[Any]:
    """The TP group (Helix CP folded in) when its ranks serve the same requests; None for one rank
    or under attention DP."""
    if mapping is None:
        return None
    if mapping.has_cp_helix():
        mapping = mapping.repurpose_helix_cp_to_tp()
    if mapping.tp_size <= 1 or not mapping.tp_group_shares_requests:
        return None
    return mapping


class RubinDraftTokenAgreement(OpWrapper):
    """The group rank 0's drafts ``[rows, K]`` int32 on every rank of ``mapping``'s TP group.

    Every rank passes its own drafts of the same rows; each gets rank 0's (a new tensor).
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(why="TEP8 generation, c1: one request, draft length 7", spec=dict(rows=1, k=7)),
        Cell(why="TEP8 generation, c8: draft length 7", spec=dict(rows=8, k=7)),
        Cell(why="helix8 generation, c16: draft length 5", spec=dict(rows=16, k=5)),
    )
    note = """
    One all-gather of the int32 drafts over the TP group, then rank 0's rows: no host sync, no
    allocation beyond the gathered tensor, capturable. Multi-rank: certified by the four-rank all-gather
    test on GB300 and VR200 and by the TEP8 and helix8 DFlash2 runs on VR200, not on one GPU.
    """

    def __call__(self, drafts: torch.Tensor, mapping: Any) -> torch.Tensor:
        rows = drafts.shape[0]
        return allgather(drafts.contiguous(), mapping, dim=0)[:rows]

    def reference(self, per_rank_drafts: list) -> torch.Tensor:
        """The group's drafts in rank order; every rank gets rank 0's."""
        return per_rank_drafts[0].clone()

    def is_valid(self, drafts: torch.Tensor, mapping: Any) -> None:
        assert drafts.dim() == 2, f"drafts must be [rows, draft_len], got {tuple(drafts.shape)}"
        assert drafts.dtype == torch.int32, f"drafts must be int32, got {drafts.dtype}"
        assert mapping.tp_size > 1, "agreement needs a TP group of more than one rank"


rubin_draft_token_agreement = RubinDraftTokenAgreement()


def agree_verified_tokens(worker, tokens, lengths):
    """Use rank zero's verification result before updating the per-request state."""
    group = draft_agreement_mapping(worker.mapping)
    if group is None:
        return tokens, lengths
    packed = torch.cat((tokens, lengths[:, None]), dim=1).to(torch.int32)
    agreed = rubin_draft_token_agreement(packed, group)
    return agreed[:, :-1].contiguous().to(tokens.dtype), agreed[:, -1].contiguous().to(
        lengths.dtype
    )


def agree_draft_tokens(outputs: Any, mapping: Any) -> None:
    """Give ``next_draft_tokens`` ``[rows, K]`` and the draft columns of ``next_new_tokens``
    ``[rows, 1 + K]`` the group rank 0's values (column 0, the last accepted token, already
    agrees). Nothing to do without a request-sharing TP group or without drafts."""
    group = draft_agreement_mapping(mapping)
    if group is None or not isinstance(outputs, dict):
        return
    drafts = outputs.get("next_draft_tokens")
    if drafts is None or drafts.numel() == 0:
        return
    if drafts.dim() != 2:
        raise ValueError(
            "next_draft_tokens must be [rows, draft_len] to agree across the TP group; got shape "
            f"{tuple(drafts.shape)}"
        )
    rows, width = drafts.shape
    new_tokens = outputs.get("next_new_tokens")
    if new_tokens is not None and tuple(new_tokens.shape) != (rows, width + 1):
        raise ValueError(
            "next_new_tokens must be [rows, 1 + draft_len] (the last accepted token, then the "
            f"drafts) to agree across the TP group; got {tuple(new_tokens.shape)} for drafts "
            f"{tuple(drafts.shape)}"
        )
    agreed = rubin_draft_token_agreement(drafts.to(torch.int32), group)
    outputs["next_draft_tokens"] = agreed.to(drafts.dtype)
    if new_tokens is not None:
        outputs["next_new_tokens"] = torch.cat(
            [new_tokens[:, :1], agreed.to(new_tokens.dtype)], dim=1
        )


class Fp8HistoryDrafter:
    """A drafter whose context latents come back saturated FP8 E4M3 at unit scale (``to_draft_fp8``),
    as an FP8 drafter history stores them; everything else is the drafter's own."""

    def __init__(self, drafter: Any):
        self._drafter = drafter

    def __getattr__(self, name: str) -> Any:
        return getattr(self._drafter, name)

    def precompute_context_kv(self, projected_hidden: torch.Tensor, positions: torch.Tensor):
        latent, value = self._drafter.precompute_context_kv(projected_hidden, positions)
        return to_draft_fp8(latent), value


def placeholder_rows(attn_metadata: Any, num_gens: int) -> Optional[torch.Tensor]:
    """Whether each of the step's ``num_gens`` generation rows verifies placeholder drafts, or None when the
    step's metadata carries no flags.

    The family DSA metadata uploads the flags before the forward
    (``rubin_dsa_metadata.prepare_generation_placeholder_rows``); a captured graph reads the replayed step's.
    """
    rows = getattr(attn_metadata, "generation_placeholder_rows", None)
    if rows is None or num_gens == 0:
        return None
    return rows[:num_gens]


class RubinDFlash2WorkerMixin:
    """The sm_107 targets' worker overrides of the engine DFlash worker: the FP8 and Helix history
    stores and the draft step's metadata refresh (see the module docstring).

    A target worker lists it before the engine worker, defines ``_draft_reads_target_metadata`` and
    calls ``agree_draft_tokens`` on the outputs of its ``_forward_impl`` (the engine worker's
    ``forward`` is not overridden: it guarantees the spec-decoding metadata cleanup).
    """

    def _set_placeholder_rows(self, attn_metadata: Any, num_gens: int) -> None:
        """Under forced acceptance, mark the step's rows that verify placeholder drafts (``placeholder_rows``)
        for ``_keep_placeholder_counts`` and the target's forced count; nothing otherwise."""
        self._placeholder_rows = (
            placeholder_rows(attn_metadata, num_gens)
            if self.force_num_accepted_tokens != 0.0
            else None
        )

    def _natural_generation_counts(
        self, num_accepted_tokens: torch.Tensor, num_contexts: int
    ) -> Optional[torch.Tensor]:
        """A copy of the generation rows' own counts when forced acceptance may replace some, else None."""
        if getattr(self, "_placeholder_rows", None) is None:
            return None
        return num_accepted_tokens[num_contexts:].clone()

    def _keep_placeholder_counts(
        self, out: torch.Tensor, natural: Optional[torch.Tensor], num_contexts: int
    ) -> None:
        """Give the rows that verify placeholder drafts back their own counts (``natural``) after the engine
        forced every generation row; a static-shape select, so a captured graph replays it."""
        if natural is None:
            return
        gen = out[num_contexts:]
        gen.copy_(torch.where(self._placeholder_rows[: gen.shape[0]], natural, gen))

    def _forced_rows(self, num_gens: int) -> Optional[torch.Tensor]:
        """Which of the ``num_gens`` generation rows forced acceptance set, or None for all of them."""
        placeholders = getattr(self, "_placeholder_rows", None)
        return None if placeholders is None else ~placeholders[:num_gens]

    def _draft_reads_target_metadata(self) -> bool:
        """Whether the draft forward reads the target's attention metadata (its device side)."""
        return True

    def _prepare_kv_for_draft_forward(
        self,
        attn_metadata,
        num_accepted_tokens: torch.Tensor,
        num_contexts: int,
        batch_size: int,
    ) -> None:
        """The engine's KV-length prep for the draft forward (RoPE positions), refreshing only the
        host-side metadata when the drafter reads none of the target's."""
        if hasattr(attn_metadata, "kv_lens_cuda"):
            self._kv_rewind_amount = 1 - num_accepted_tokens[num_contexts:batch_size]
            self._kv_rewind_nc = num_contexts
            self._kv_rewind_bs = batch_size
            if batch_size > num_contexts:
                attn_metadata.kv_lens_cuda[num_contexts:batch_size] += 1
            self._kv_rewind_pending = True
            if self._draft_reads_target_metadata():
                attn_metadata.update_for_spec_dec()
            else:
                attn_metadata.update_for_spec_dec(refresh_device_metadata=False)

    def _prepare_paged_latent_gen_inputs(
        self,
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        attn_metadata: Any,
        spec_metadata: Any,
        draft_model: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """The engine's fused generation-step preparation; over an FP8 history its drafter returns
        the context latents saturated (``Fp8HistoryDrafter``), the dtype its store requires."""
        if self._fp8_history():
            draft_model = Fp8HistoryDrafter(draft_model)
        return super()._prepare_paged_latent_gen_inputs(
            accepted_tokens,
            num_accepted_tokens,
            attn_metadata,
            spec_metadata,
            draft_model,
            *args,
            **kwargs,
        )

    def _fp8_history(self) -> bool:
        """Whether the drafter history pools hold FP8 E4M3."""
        pools = self._ctx_kv_buf
        return bool(pools) and pools[0].dtype == torch.float8_e4m3fn

    def _store_context_kv_paged(
        self,
        k: torch.Tensor,
        v: Optional[torch.Tensor],
        rows: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """A context chunk's latents: under Helix (``_helix_shard``) only those at positions on this
        rank's pages, at their rank-local positions (an eager selection); into an FP8 E4M3 history the
        MLA latent saturated, into any other history as the engine stores it."""
        shard = self._helix_shard()
        if shard is not None:
            positions, owned = shard.local_slots(positions)
            k, rows, positions = k[owned], rows[owned], positions[owned]
            v = None if v is None else v[owned]
        if v is not None or not self._fp8_history():
            super()._store_context_kv_paged(k, v, rows, positions)
            return
        table = (
            self._ctx_block_tables if self._ctx_block_tables is not None else self._ctx_page_table
        )
        page_size = self._ctx_page_size
        pages = table[rows.long(), positions.long() // page_size].long()
        offsets = positions.long() % page_size
        for layer_idx in range(k.size(1)):
            # Byte views of the FP8 pool and values: index_put support for FP8 varies across builds.
            pool = self._ctx_kv_buf[layer_idx].view(torch.uint8)
            pool[pages, 0, :, offsets] = to_draft_fp8(k[:, layer_idx]).view(torch.uint8)


__all__ = [
    "Fp8HistoryDrafter",
    "RubinDFlash2WorkerMixin",
    "agree_draft_tokens",
    "draft_agreement_mapping",
    "rubin_draft_config_kwargs",
    "rubin_draft_token_agreement",
]
