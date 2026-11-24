# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
from typing import Optional

from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig


def setup_helix_mappings(
    self,
    model_config: ModelConfig,
    cfg: "PretrainedConfig",
    spec_config: Optional["DecodingBaseConfig"],
) -> None:
    """Validate helix preconditions and stage the dual-mapping swap.

    DeepseekV3 pattern: the MLA layers keep the CP original; everything
    else is built against the repurposed mapping (CP ranks become TP
    ranks). Sets ``mapping_with_cp`` (restored after construction) and
    ``_repurposed_tp_mapping`` (load_weights shard selection); both stay
    None outside helix.
    """
    self.mapping_with_cp = None
    self._repurposed_tp_mapping = None
    if not model_config.mapping.has_cp_helix():
        return
    # Helix + attention DP = helix_a2a: KDA keeps head-TP state across the
    # CP group (every CP rank holds the same requests) and only the MoE
    # layers run attention-DP on ceil(T / cp) token rows each.
    if model_config.mapping.enable_attention_dp and model_config.mapping.tp_size != 1:
        raise NotImplementedError(
            "Kimi K3 helix with enable_attention_dp requires tp_size=1 "
            f"(got tp_size={model_config.mapping.tp_size}): KDA and MLA are "
            "head-TP over the CP group, so there is no request-level DP to add."
        )
    if spec_config is not None:
        # Standalone block drafters use the V2 superblock ledger for
        # page ownership and CTX-to-GEN transfer.
        if not (spec_config.spec_dec_mode.is_dspark() or spec_config.spec_dec_mode.is_dflash()):
            raise ValueError(
                "Kimi K3 helix supports speculative decoding only with "
                f"DSpark or MLA DFlash2 (standalone drafter); got "
                f"{spec_config.decoding_type!r}."
            )
        # The SpeculationGate acceptance-rate trip permanently disables
        # speculation mid-flight while enable_spec_decode stays True;
        # in-flight helix requests then fall into the plain generation
        # loop whose position math (total_input_len_cp +
        # py_decoding_iter - 1) is stale once any draft token was
        # accepted -> silently wrong RoPE positions and KV slots. Reject
        # the trip wires until that loop is helix-group aware.
        if (
            spec_config.acceptance_rate_window_size is not None
            or spec_config.acceptance_rate_threshold is not None
        ):
            raise ValueError(
                "Kimi K3 helix does not support the speculation "
                "acceptance-rate gate (acceptance_rate_window_size / "
                "acceptance_rate_threshold): dynamically disabling "
                "speculation mid-flight leaves helix requests on a "
                "single-token position formula."
            )
        # max_concurrency is the same trip wire by another name: the
        # drafter re-evaluates should_use_spec_decode on every scheduling
        # iteration and flips enable_spec_decode off as soon as the active
        # batch exceeds the cap. In-flight helix requests then take the
        # plain generation loop, whose position formula counts ITERATIONS
        # (total_input_len_cp + py_decoding_iter - 1) rather than
        # committed tokens, so it is stale by however many draft tokens
        # were accepted -- a wrong RoPE position, and across a ledger page
        # boundary a KV write to the wrong CP rank. Mirror the drafter's
        # own "unset" test (Drafter.should_use_spec_decode returns True
        # when max_concurrency is None) so an unset value is not rejected.
        if spec_config.max_concurrency is not None:
            raise ValueError(
                "Kimi K3 helix does not support the speculation "
                "concurrency cutoff (max_concurrency): disabling "
                "speculation above the cap leaves in-flight helix "
                "requests on a position formula that assumes one "
                "committed token per iteration, which accepted draft "
                "tokens break."
            )
        # draft_len_schedule is the user-facing alternative to
        # max_concurrency (llm_args rejects setting both) and reaches the
        # same end by a route that does not go through
        # should_use_spec_decode at all: py_executor turns speculation off
        # directly once the schedule yields draft_len 0 for the active
        # batch size. Guarding only max_concurrency would leave this door
        # open. Skip the schedule that llm_args synthesized from
        # max_concurrency, so a config that set only that field raises the
        # message above naming the field the user actually wrote.
        if (
            spec_config.draft_len_schedule is not None
            and not spec_config._translated_from_max_concurrency
        ):
            raise ValueError(
                "Kimi K3 helix does not support the dynamic draft-length "
                "schedule (draft_len_schedule): a batch size past the "
                "last entry drops the draft length to 0 and turns "
                "speculation off mid-run, leaving in-flight helix "
                "requests on a position formula that assumes one "
                "committed token per iteration."
            )
    cp = model_config.mapping.cp_size
    repurposed_tp = model_config.mapping.tp_size * cp
    if cfg.num_attention_heads % repurposed_tp != 0:
        raise ValueError(
            f"Kimi K3 helix requires tp_size*cp_size ({repurposed_tp}) "
            f"to divide the MLA head count ({cfg.num_attention_heads})."
        )
    kda_heads = cfg.linear_attn_config["num_heads"]
    if kda_heads % repurposed_tp != 0:
        raise ValueError(
            f"Kimi K3 helix requires tp_size*cp_size ({repurposed_tp}) to "
            f"divide the KDA head count ({kda_heads})."
        )
    # MoE splits apply to the repurposed tp*cp group (helix
    # moe_world_size = tp*cp); default EP-only. The Mapping constructor
    # skips its product check when both sizes are 1, so validate here.
    moe_ep = repurposed_tp
    if model_config.mapping.moe_tp_ep_user_specified:
        moe_tp = model_config.mapping.moe_tp_size
        moe_ep = model_config.mapping.moe_ep_size
        if moe_tp * moe_ep != repurposed_tp:
            raise ValueError(
                f"Kimi K3 helix: moe_tensor_parallel_size ({moe_tp}) x "
                f"moe_expert_parallel_size ({moe_ep}) must equal "
                f"tp_size*cp_size ({repurposed_tp}): MoE runs on the "
                "repurposed tp*cp group."
            )
    if cfg.num_experts and cfg.num_experts % moe_ep != 0:
        raise ValueError(
            f"Kimi K3 helix requires the MoE EP size ({moe_ep}) to "
            f"divide the routed expert count ({cfg.num_experts}): each "
            "EP rank of the repurposed tp*cp group holds whole experts."
        )
    self.mapping_with_cp = copy.deepcopy(model_config.mapping)
    repurposed = model_config.mapping.repurpose_helix_cp_to_tp()
    # repurpose passes resolved moe sizes, which the Mapping constructor
    # mistakes for user-specified values; restore the flag.
    repurposed.moe_tp_ep_user_specified = self.mapping_with_cp.moe_tp_ep_user_specified
    # load_weights shard selection must use this tp_rank; the restored
    # CP original's tp_rank is 0 on every rank.
    self._repurposed_tp_mapping = repurposed
    model_config._frozen = False
    model_config.mapping = repurposed
    # Side-channel for the MLA layers (avoids threading a kwarg through
    # every intermediate signature).
    model_config._helix_mapping_with_cp = self.mapping_with_cp
    model_config._frozen = True
