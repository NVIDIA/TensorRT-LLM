# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helix context-parallel state of the attention metadata."""

from typing import List, Optional

import torch

from tensorrt_llm._utils import prefer_pinned

from ....utils import helix_local_len_tensor
from ..extensions import MetadataExtension
from ..fp4_mla.metadata import populate_helix_spec_buffers


class HelixMetadata(MetadataExtension):
    """Per-token global positions and KV ownership under helix parallelism.

    Each helix rank holds a subset of a sequence's tokens. Only the active rank
    appends the KV of a new token and attends to it; inactive ranks attend to
    the previously cached tokens only.

    The per-sequence and per-token buffers are GENERATION-relative: the
    packing loops in the executor append only for extend and plain generation
    rows, so ``update_helix_param`` writes exactly ``[0, num_generations)``.
    """

    fields = frozenset(
        {
            "helix_position_offsets",
            "helix_position_offsets_cpu",
            "helix_is_inactive_rank",
            "helix_is_inactive_rank_cpu",
            "helix_kv_bounds",
            "helix_local_slots",
            "helix_owned_new_tokens_cpu",
            "_helix_spec_tokens_valid",
        }
    )
    methods = frozenset(
        {
            "helix_local_len_vec",
            "recompute_helix_spec_buffers",
            "update_helix_param",
        }
    )

    def __init__(self, metadata, *, source=None, **config) -> None:
        super().__init__(metadata, source=source, **config)
        # Global position id of each token of each sequence.
        self.helix_position_offsets: Optional[torch.Tensor] = None
        self.helix_position_offsets_cpu: Optional[torch.Tensor] = None
        # Whether this rank is inactive for each generation sequence.
        self.helix_is_inactive_rank: Optional[torch.Tensor] = None
        self.helix_is_inactive_rank_cpu: Optional[torch.Tensor] = None
        # Per-token state for speculative verify groups: a 1 + draft_len group
        # may straddle a ledger-page boundary onto two CP ranks, so the
        # per-sequence flag above is not expressive enough. See
        # recompute_helix_spec_buffers for the derivation.
        #   helix_local_slots[t]: rank-local KV write slot of generation token t on
        #     this rank, or -1 when another rank owns its position (consumed
        #     by the mla_rope_generation append kernel).
        #   helix_kv_bounds[t]: number of rank-local KV entries token t may attend
        #     to, i.e. local_len(pos_t + 1) (consumed by the CuTe DSL MLA
        #     decode mask and the helix stats identity).
        self.helix_local_slots: Optional[torch.Tensor] = None
        self.helix_kv_bounds: Optional[torch.Tensor] = None
        # Host per-sequence count of this step's new tokens owned by this rank
        # (spec path; the single-token path derives it from the boolean flag).
        self.helix_owned_new_tokens_cpu: Optional[torch.Tensor] = None
        self._helix_spec_tokens_valid = False
        # Width of this step's speculative verify groups (spec path).
        self._tokens_per_gen_seq = 0
        buffers = metadata.cuda_graph_buffers
        md = self.metadata
        capture_graph = md.is_cuda_graph
        self.helix_position_offsets = md.get_empty(
            buffers,
            (md.max_num_tokens,),
            cache_name="helix_position_offsets",
            dtype=torch.int,
            capture_graph=capture_graph,
        )
        self.helix_position_offsets_cpu = torch.empty_like(
            self.helix_position_offsets,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        self.helix_is_inactive_rank = md.get_empty(
            buffers,
            (md.max_num_sequences,),
            cache_name="helix_is_inactive_rank",
            dtype=torch.bool,
            capture_graph=capture_graph,
        )
        self.helix_is_inactive_rank_cpu = torch.empty_like(
            self.helix_is_inactive_rank,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        self.helix_local_slots = md.get_empty(
            buffers,
            (md.max_num_tokens,),
            cache_name="helix_local_slots",
            dtype=torch.int,
            capture_graph=capture_graph,
        )
        self.helix_kv_bounds = md.get_empty(
            buffers,
            (md.max_num_tokens,),
            cache_name="helix_kv_bounds",
            dtype=torch.int,
            capture_graph=capture_graph,
        )
        self.helix_owned_new_tokens_cpu = torch.zeros(
            (md.max_num_sequences,),
            device="cpu",
            dtype=torch.int,
            pin_memory=prefer_pinned(),
        )
        self._helix_spec_tokens_valid = False

    def update_helix_param(
        self,
        helix_position_offsets: List[int],
        helix_is_inactive_rank: List[bool],
        helix_owned_new_tokens: Optional[List[int]] = None,
        tokens_per_gen_seq: int = 1,
    ) -> None:
        """Copy this step's helix parameters into the static buffers.

        Args:
            helix_position_offsets: Global position of each token, shape (num_tokens,).
            helix_is_inactive_rank: Whether this rank is inactive for each
                generation sequence, shape (num_generations,).
            helix_owned_new_tokens: Per-sequence count of this step's new tokens
                owned by this rank (speculative verify groups; one group may
                straddle a page boundary onto two ranks). None on the
                single-token path, where the boolean flag carries it.
            tokens_per_gen_seq: Width of the speculative verify groups; read
                only with ``helix_owned_new_tokens``.
        """
        if helix_position_offsets is not None:
            num_tokens = len(helix_position_offsets)
            self.helix_position_offsets_cpu[:num_tokens].copy_(
                torch.tensor(helix_position_offsets, dtype=torch.int)
            )
            self.helix_position_offsets[:num_tokens].copy_(
                self.helix_position_offsets_cpu[:num_tokens], non_blocking=True
            )

        if helix_is_inactive_rank is not None:
            batch_size = len(helix_is_inactive_rank)
            self.helix_is_inactive_rank_cpu[:batch_size].copy_(
                torch.tensor(helix_is_inactive_rank, dtype=torch.bool)
            )
            self.helix_is_inactive_rank[:batch_size].copy_(
                self.helix_is_inactive_rank_cpu[:batch_size], non_blocking=True
            )

        self._helix_spec_tokens_valid = False
        if helix_owned_new_tokens is not None:
            batch_size = len(helix_owned_new_tokens)
            self.helix_owned_new_tokens_cpu[:batch_size].copy_(
                torch.tensor(helix_owned_new_tokens, dtype=torch.int)
            )
            self._helix_spec_tokens_valid = True
            self._tokens_per_gen_seq = tokens_per_gen_seq

    def prepare(self) -> None:
        """Count only the new tokens this rank appends for generation rows.

        Context rows are not part of a verify group and are not packed into
        the helix buffers; they append every one of their tokens. Generation
        rows pair the generation-relative helix buffers with the generation
        slice of the batch-indexed lengths. The core republishes the corrected
        KV lengths before the extensions after this one read them.

        With speculative verify groups the device KV lengths are then derived
        on device from the corrected positions (``recompute_helix_spec_buffers``).
        This override supersedes the core's ``kv_lens_offsets`` correction,
        which is not ownership-aware.
        """
        md = self.metadata
        assert md.kv_cache_params.use_cache, "helix requires a KV cache"
        kv_lens = md.kv_lens_runtime.clone()
        num_gen = md.num_generations
        gen = slice(md.num_contexts, md.num_seqs)
        cached = kv_lens[gen] - md.seq_lens_kv[gen]
        if self._helix_spec_tokens_valid:
            # Speculative verify groups: ownership of this step's new tokens is
            # a per-sequence COUNT, not a boolean. Provisional host values,
            # which the device derivation below overrides.
            kv_lens[gen] = cached + self.helix_owned_new_tokens_cpu[:num_gen]
        else:
            kv_lens[gen] = torch.where(
                self.helix_is_inactive_rank_cpu[:num_gen], cached, kv_lens[gen]
            )
        md.republish_kv_lens(kv_lens)
        if self._helix_spec_tokens_valid and md.kv_cache_manager is not None:
            num_gen_tokens = md.num_tokens - md.num_ctx_tokens
            if md.position_id_offsets is not None:
                # The host packed provisional positions from a stale base; apply
                # the overlap scheduler's device correction. Staged again from
                # the host values so that a repeated prepare does not apply it
                # twice.
                positions = self.helix_position_offsets[:num_gen_tokens]
                positions.copy_(self.helix_position_offsets_cpu[:num_gen_tokens], non_blocking=True)
                positions += md.position_id_offsets[:num_gen_tokens]
            self.recompute_helix_spec_buffers(num_gen_tokens, self._tokens_per_gen_seq)

    def helix_local_len_vec(self, global_lens: torch.Tensor) -> torch.Tensor:
        """Vectorized rank-local prefix length for helix round-robin pages.

        For each global sequence length g, returns the number of the first g
        tokens whose ledger page lives on this CP rank (page b -> rank
        b % cp_size). The rule and its scalar twin live in ``_torch.utils``;
        KVCacheManagerV2._helix_local_len and the host packing in the executor
        use the same definition.
        """
        md = self.metadata
        return helix_local_len_tensor(
            global_lens,
            md.kv_cache_manager.tokens_per_block,
            md.mapping.cp_size,
            md.mapping.cp_rank,
        )

    def recompute_helix_spec_buffers(self, num_gen_tokens: int, tokens_per_gen_seq: int) -> None:
        """Derive per-token helix buffers from (corrected) global positions.

        Called after the overlap-scheduler device correction has been applied
        to position_offsets, so every derived quantity reflects the real
        committed length even though the host packed provisional values.
        The generation rows' device KV lengths become the rank-local lengths.

        ``tokens_per_gen_seq`` is the uniform verify-group width. Non-uniform
        groups are rejected rather than silently mis-sliced: without the
        overlap scheduler the extend loop packs a per-request
        ``1 + get_draft_token_length(request)`` and a request entering with no
        draft tokens is packed as a single-token generation row instead, so a
        batch can arrive whose total happens to divide but whose rows do not
        line up.

        Two index bases meet here, and they are not the same:
          * helix_position_offsets / helix_local_slots / helix_kv_bounds are GENERATION-RELATIVE.
          * kv_lens_cuda is BATCH-indexed, hence the num_contexts offset on
            the write below.
        """
        md = self.metadata
        assert num_gen_tokens % tokens_per_gen_seq == 0, (
            f"helix spec expects uniform verify groups: {num_gen_tokens} gen "
            f"tokens not divisible by group size {tokens_per_gen_seq}"
        )
        num_gen_seqs = num_gen_tokens // tokens_per_gen_seq
        # Divisibility alone does not imply uniformity: a batch of mixed group
        # widths can still divide and would then write the wrong number of
        # kv_lens_cuda rows with values taken from the wrong tokens.
        assert num_gen_seqs == md.num_generations, (
            f"helix spec expects uniform verify groups: {num_gen_tokens} gen "
            f"tokens over {md.num_generations} generation rows do not all "
            f"have width {tokens_per_gen_seq}"
        )
        # One launch derives the write slots, the attention bounds and the
        # per-sequence rank-local kv length (the bound of the sequence's last
        # token: committed + owned in-flight tokens) from the positions; it
        # replaces a chain of elementwise ops on the step's critical path.
        local_kv_lens = torch.empty(
            (num_gen_seqs,), dtype=torch.int32, device=self.helix_position_offsets.device
        )
        populate_helix_spec_buffers(
            self.helix_position_offsets[:num_gen_tokens],
            self.helix_local_slots[:num_gen_tokens],
            self.helix_kv_bounds[:num_gen_tokens],
            local_kv_lens,
            num_tokens=num_gen_tokens,
            tokens_per_seq=tokens_per_gen_seq,
            tokens_per_block=md.kv_cache_manager.tokens_per_block,
            cp_size=md.mapping.cp_size,
            cp_rank=md.mapping.cp_rank,
        )
        md.set_kv_lens(slice(md.num_contexts, md.num_contexts + num_gen_seqs), local_kv_lens)
