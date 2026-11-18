# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuration checks for Wide-EP fault tolerance, and the rank identity from its launcher."""

import os
from typing import TYPE_CHECKING, NamedTuple, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from .llmapi.llm_args import TorchLlmArgs

# Set for every rank under Wide-EP fault tolerance to
# "<rank>,<world size>,<GPUs per node>,<replacement>".
FAULT_TOLERANCE_RANK_ENV = "TLLM_FAULT_TOLERANCE_RANK"
# The rendezvous store's host:port. The authority hosts the store.
RENDEZVOUS_ENV = "TLLM_FAULT_TOLERANCE_RENDEZVOUS"
# The hex secret that every deployment process holds. It derives the hello token.
SHARED_SECRET_ENV = "TLLM_FAULT_TOLERANCE_SHARED_SECRET"
# The model architectures that Wide-EP fault tolerance supports, as named in config.json.
SUPPORTED_ARCHITECTURES = ("DeepseekV3ForCausalLM",)
# The alternative that each error about an unmet fault tolerance requirement offers.
TURN_FAULT_TOLERANCE_OFF = "turn fault tolerance off (enable_wide_ep_fault_tolerance=False)"


def check_node_supervisor_launch(enabled: bool) -> None:
    """Check that only a fault-tolerant deployment starts ranks through a node supervisor.

    Raises:
        ValueError: A node supervisor started this process, but ``enabled`` is False.
    """
    if not enabled and FAULT_TOLERANCE_RANK_ENV in os.environ:
        raise ValueError(
            f"A node supervisor started this rank ({FAULT_TOLERANCE_RANK_ENV} is set), but "
            "enable_wide_ep_fault_tolerance is false"
        )


def check_fault_tolerance_requirements(checks: Sequence[Tuple[bool, str]]) -> None:
    """Raise ValueError naming every unmet ``(met, requirement)`` check."""
    missing = [requirement for met, requirement in checks if not met]
    if missing:
        raise ValueError(
            "Wide-EP fault tolerance requires "
            + "; ".join(missing)
            + f"; change the configuration or {TURN_FAULT_TOLERANCE_OFF}"
        )


def check_llm_args(llm_args: "TorchLlmArgs") -> None:
    """Check the LLM arguments that Wide-EP fault tolerance requires.

    Raises:
        ValueError: The arguments are not supported with fault tolerance.
    """
    adp_config = llm_args.attention_dp_config
    spec_config = llm_args.speculative_config
    spec_mode = None if spec_config is None else spec_config.spec_dec_mode
    decoding_type = None if spec_config is None else spec_config.decoding_type
    sa_enhanced = getattr(spec_config, "sa_config", None) is not None
    current_decoding = f"{decoding_type!r} with sa_config" if sa_enhanced else repr(decoding_type)
    # The options with which each rank picks its own draft length.
    rank_local_draft_len = []
    if spec_config is not None:
        if spec_config.max_concurrency is not None:
            rank_local_draft_len.append(f"max_concurrency={spec_config.max_concurrency}")
        elif spec_config.draft_len_schedule is not None:
            rank_local_draft_len.append(f"draft_len_schedule={spec_config.draft_len_schedule}")
        if (
            spec_config.acceptance_rate_window_size
            and spec_config.acceptance_rate_threshold is not None
        ):
            rank_local_draft_len.append(
                f"acceptance_rate_window_size={spec_config.acceptance_rate_window_size} with "
                f"acceptance_rate_threshold={spec_config.acceptance_rate_threshold}"
            )
    kv_events = llm_args.kv_cache_config.kv_events_config
    transceiver = llm_args.cache_transceiver_config
    transceiver_backend = None if transceiver is None else transceiver.backend
    load_balancer = llm_args.moe_config.load_balancer
    balancer_mode = None if load_balancer is None else load_balancer.mode
    warmup_placement = getattr(load_balancer, "warmup_placement", None)
    adp_routing = [
        name
        for name in (
            "enable_kv_balanced_admission",
            "enable_kv_cache_aware_routing",
            "kv_cache_routing_conversation_affinity",
        )
        if adp_config is not None and getattr(adp_config, name)
    ]
    compile_config = llm_args.torch_compile_config
    # An unset max_num_tokens means the default token budget.
    max_num_tokens = llm_args.max_num_tokens
    if max_num_tokens is None:
        max_num_tokens = type(llm_args).model_fields["max_num_tokens"].default
    current_max_seq_len = (
        "an unset max_seq_len (the model's maximum length)"
        if llm_args.max_seq_len is None
        else f"max_seq_len={llm_args.max_seq_len}"
    )
    check_fault_tolerance_requirements(
        [
            (
                llm_args.enable_attention_dp,
                "attention DP (enable_attention_dp=True), because a failed rank must take only its "
                "own requests with it",
            ),
            (
                llm_args.pipeline_parallel_size == 1,
                "no pipeline parallelism (pipeline_parallel_size=1, not "
                f"{llm_args.pipeline_parallel_size}), because a failed pipeline stage leaves its "
                "layers on no rank",
            ),
            (
                llm_args.max_beam_width in (None, 1),
                f"no beam search (max_beam_width=1, not {llm_args.max_beam_width}), because a "
                "request that a failure interrupts continues by recompute, which does not restore "
                "its beams",
            ),
            (
                not llm_args.disable_overlap_scheduler,
                "the overlap scheduler (disable_overlap_scheduler=False), because fault tolerance "
                "accepts or discards each step when the next one starts, which only the overlap "
                "scheduler's executor loop does yet",
            ),
            (
                load_balancer is not None,
                "EPLB (moe_config.load_balancer.num_slots, for example the number of routed "
                "experts), whose spare slots take a failed rank's experts",
            ),
            (
                balancer_mode in (None, "standard"),
                "standard EPLB (moe_config.load_balancer.mode='standard', not "
                f"{balancer_mode!r}), because per-layer dynamic EPLB refills its helper slots in "
                "every forward pass and has no spare slots for a failed rank's experts",
            ),
            (
                warmup_placement is None,
                "no warmup placement (moe_config.load_balancer.warmup_placement=None), because its "
                "placement rounds run collectives over the MPI world and an NCCL group of every EP "
                "rank, while each rank is its own MPI world and a failed rank would hang the group",
            ),
            (
                not adp_routing,
                "the default attention-DP routing and admission (no enable_kv_balanced_admission, "
                "enable_kv_cache_aware_routing or kv_cache_routing_conversation_affinity in "
                f"attention_dp_config, which sets {', '.join(adp_routing)}), because the others "
                "keep per-rank state that a replacement would not share",
            ),
            (
                llm_args.kv_cache_config.event_buffer_max_size == 0,
                "no KV cache events (kv_cache_config.event_buffer_max_size=0, not "
                f"{llm_args.kv_cache_config.event_buffer_max_size}), because the events would "
                "miss the removal of a failed rank's KV cache blocks",
            ),
            (
                kv_events is None or not kv_events.enable_kv_cache_events,
                "no streaming KV cache events (kv_cache_config.kv_events_config."
                "enable_kv_cache_events=False, not True), because the events would miss the "
                "removal of a failed rank's KV cache blocks",
            ),
            (
                not llm_args.kv_cache_config.host_cache_bind_to_gpu_numa_node,
                "no NUMA binding of the host cache tier (kv_cache_config."
                "host_cache_bind_to_gpu_numa_node=False, not True), because a replacement rank "
                "would find the host tiers of the running ranks on its NUMA node already allocated "
                "and fail its startup check that the node holds every tier bound to it",
            ),
            (
                transceiver_backend is None,
                "no disaggregated serving (no cache_transceiver_config.backend, not "
                f"{transceiver_backend!r}), because the KV cache transfers between serving "
                "instances do not follow a membership change",
            ),
            (
                llm_args.kv_connector_config is None,
                "no KV cache connector (kv_connector_config=None), because a connector saves the "
                "KV cache of a step before the step is accepted",
            ),
            (
                llm_args.kv_cache_compression_config is None,
                "no KV cache compression (kv_cache_compression_config=None), because the KV cache "
                "pool that a membership change resizes would also hold the compressed pages",
            ),
            (
                spec_mode is None or (spec_mode.is_mtp_one_model() and not sa_enhanced),
                "MTP or no speculative decoding (speculative_config: none, or decoding_type 'MTP' "
                f"without sa_config, not {current_decoding})",
            ),
            (
                not rank_local_draft_len,
                "a fixed draft length (no speculative_config.draft_len_schedule or max_concurrency, "
                "and no acceptance-rate gate, which is acceptance_rate_window_size with "
                f"acceptance_rate_threshold; not {', '.join(rank_local_draft_len)}), because "
                "ranks with different draft lengths run different numbers of MTP-layer "
                "all-to-alls",
            ),
            (
                llm_args.enable_chunked_prefill
                or (
                    llm_args.max_seq_len is not None
                    and max_num_tokens is not None
                    and llm_args.max_seq_len <= max_num_tokens
                ),
                "chunked prefill (enable_chunked_prefill=True) or a max_seq_len of at most "
                f"max_num_tokens, not {current_max_seq_len} with max_num_tokens={max_num_tokens}, "
                "because a request that a failure interrupts continues by recompute, with its "
                "prompt and output tokens as one context of up to max_seq_len tokens, and without "
                "chunked prefill the scheduler never schedules a context longer than "
                "max_num_tokens",
            ),
            (
                not llm_args.enable_lm_head_tp_in_adp,
                "no LM head TP in attention DP (enable_lm_head_tp_in_adp=False), because a failed "
                "rank would take its share of the vocabulary with it",
            ),
            (
                compile_config is None,
                "no torch.compile (torch_compile_config=None, and no piecewise prefill CUDA graphs, "
                "which run it), because fault tolerance does not support torch.compile yet (for "
                "example, the userbuffers that it can enable exchange memory handles over an MPI "
                "world of every rank, while each rank is its own MPI world)",
            ),
            (
                llm_args.modeling_v2 != "require",
                "the built-in model (modeling_v2='auto' or 'off', not 'require'), because no "
                "modeling_v2 target supports it yet: a target's MoE layers would need EPLB spare "
                "slots and the NVLink one-sided all-to-all, which take over a failed rank's "
                "experts and stop waiting on it",
            ),
            (
                llm_args.sleep_config is None,
                "no sleep mode (sleep_config=None), because sleep releases and restores the GPU "
                "memory of the weights, whose spare slots fault tolerance backs and unbacks itself",
            ),
            (
                not llm_args.enable_return_routed_experts,
                "no routed-expert capture (enable_return_routed_experts=False), because a step "
                "that a failure discards would leave its routes in the request's routes",
            ),
        ]
    )


def check_model_architectures(architectures: Optional[Sequence[str]]) -> None:
    """Check that Wide-EP fault tolerance supports the model's ``architectures``.

    Raises:
        ValueError: The model architecture is not supported with fault tolerance.
    """
    check_fault_tolerance_requirements(
        [
            (
                bool(architectures)
                and all(name in SUPPORTED_ARCHITECTURES for name in architectures),
                f"a model architecture in {list(SUPPORTED_ARCHITECTURES)} (DeepSeek-V3 and "
                f"DeepSeek-R1), not {architectures}",
            )
        ]
    )


class RankIdentity(NamedTuple):
    """A rank under Wide-EP fault tolerance, where each rank is its own MPI world."""

    rank: int
    world_size: int
    gpus_per_node: int
    replacement: bool

    def env_value(self) -> str:
        """Return the ``FAULT_TOLERANCE_RANK_ENV`` value that gives a process this identity."""
        return ",".join(str(int(field)) for field in self)


def wide_ep_fault_tolerance_rank() -> Optional[RankIdentity]:
    """Return the rank identity the node supervisor gave this process, if one started it."""
    value = os.environ.get(FAULT_TOLERANCE_RANK_ENV)
    if value is None:
        return None
    rank, world_size, gpus_per_node, replacement = (int(field) for field in value.split(","))
    return RankIdentity(rank, world_size, gpus_per_node, bool(replacement))
