# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model decoder-suffix replay over completed encoder GLOBAL cache.

By default the model retains the final prompt SWA window, enlarged for embedded
DSpark captures, after materializing each chunk's encoder GLOBAL cache. This is approximate across
stacked SWA layers. Set ``TRTLLM_V41_DECODER_BOUNDED_REPLAY=0`` for full decoder prefill.
SWA is request-private; query-row index/candidate publications
are gathered at the actual layer boundary, while GLOBAL cache pages stay whole.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch

from tensorrt_llm._torch.utils import get_per_request_prefill_cuda_graph_flag

from .....pyexecutor.ced_replay import (
    EncoderCheckpoint,
    EncoderReplay,
    requires_full_decoder_prefill,
)

if TYPE_CHECKING:
    from .....pyexecutor.llm_request import LlmRequest
    from .metadata import CSA2TrtllmMetadata

__all__ = [
    "DecoderReplayPlan",
    "enter_decoder_replay",
    "enter_remote_tail_decoder",
    "exit_decoder_replay",
    "gather_replayed_rows",
    "plan_decoder_replay",
    "scatter_replayed_rows",
]


@dataclass
class DecoderReplayPlan:
    """One forward's replay bookkeeping: what to run, and how to undo it.

    One record rather than loose locals because the "undo" half is not derivable
    from the metadata once :func:`enter_decoder_replay` has overwritten it -- the
    original ``seq_lens`` is the only place the encoder-pass row count survives,
    and ``rows`` is the only map back to it.
    """

    #: Global query rows, in the *encoder* pass's token space, that the decoder
    #: half replays. Packed per request in request order, so it doubles as the
    #: gather index for every per-token activation crossing the boundary.
    rows: torch.Tensor
    #: Per-request query count: the chunk's intersection with the final prompt
    #: window for eligible contexts, unchanged for generation requests.
    replay_seq_lens: torch.Tensor
    #: Per-request cached-token count for the replay pass, raised by exactly the
    #: number of query rows the pass drops.
    replay_num_cached: List[int]
    #: ``seq_lens`` as the encoder pass saw it. Restoring this restores
    #: ``num_tokens``, ``num_ctx_tokens`` and ``seq_lens_cuda`` with it, because
    #: they are all derived through the ``seq_lens`` setter.
    saved_seq_lens: torch.Tensor
    saved_num_cached: List[int]
    replay_prompt_lens: List[int]
    saved_prompt_lens: List[int]
    #: Row count of the encoder pass, i.e. the length every per-token activation
    #: crossing the boundary has on the way in and must have again on the way out.
    num_encoder_tokens: int
    #: Earliest valid Decoder SWA position for each request; zero adds no floor.
    swa_floors: list[int]
    updates_local_metadata: bool = True
    replay_all_rank_num_tokens: list[int] | None = None
    saved_all_rank_num_tokens: list[int] | None = None

    @property
    def replays_local_tokens(self) -> bool:
        return self.num_replay_tokens < self.num_encoder_tokens

    @property
    def num_replay_tokens(self) -> int:
        return int(self.rows.shape[0])


def plan_decoder_replay(
    metadata: "CSA2TrtllmMetadata",
    window: int,
    requests: Optional[list[LlmRequest]] = None,
    *,
    private_decoder: bool = False,
    allow_replay: bool = True,
) -> Optional[DecoderReplayPlan]:
    """Prepare local decoder rows before the input-preparation count exchange.

    ADP keeps identity plans so every peer can adopt the exchanged Decoder
    counts. This function performs no communication. Private decoder recovery
    still applies its SWA floor when all local rows are kept.
    """
    if metadata.is_cuda_graph or window <= 0:
        return None

    mapping = metadata.mapping
    attention_dp = mapping is not None and mapping.enable_attention_dp
    if attention_dp and get_per_request_prefill_cuda_graph_flag():
        return None
    local_unpadded = getattr(metadata, "padded_num_tokens", None) in (None, metadata.num_tokens)

    updates_private_metadata = private_decoder and local_unpadded and metadata.num_contexts > 0
    allow_replay = (
        allow_replay
        and local_unpadded
        and metadata.num_contexts > 0
        and (
            private_decoder
            or not (getattr(metadata, "num_chunked_ctx_requests", 0) or _swa_is_reused(metadata))
        )
    )
    if not attention_dp and not allow_replay and not updates_private_metadata:
        return None

    seq_lens = metadata.seq_lens
    num_requests = metadata.num_seqs
    num_contexts = metadata.num_contexts
    host_seq_lens = seq_lens[:num_requests].tolist()
    num_cached = list(metadata.kv_cache_params.num_cached_tokens_per_seq[:num_requests])

    context_ends = getattr(metadata, "decoder_context_ends", None)
    if context_ends is None and requests and all(hasattr(req, "prompt_len") for req in requests):
        context_ends = tuple(req.prompt_len for req in requests)
    replay_lens = list(host_seq_lens)
    floors = [0] * num_requests
    for i in range(num_contexts):
        req = requests[i] if requests and i < len(requests) else None
        if req is not None and requires_full_decoder_prefill(req):
            continue
        length, cached = host_seq_lens[i], num_cached[i]
        recovery = getattr(req, "py_ced_replay", None)
        recovery_floor = (
            recovery.start
            if isinstance(recovery, EncoderReplay)
            else recovery.global_end
            if isinstance(recovery, EncoderCheckpoint)
            else 0
        )
        if allow_replay:
            if context_ends is not None and i < len(context_ends):
                end = context_ends[i]
                if cached + length > end:
                    raise ValueError("Decoder context queries extend past the prepared prompt end")
                start = max(0, end - window)
                replay_lens[i] = max(0, cached + length - max(cached, start))
                # The same floor spans all chunks in this final window. Earlier
                # decoder KV inside it is valid, even when this chunk keeps all rows.
                floors[i] = max(start, recovery_floor)
            else:
                # Direct callers without whole-prompt endpoints retain a chunk
                # tail; they cannot establish that any chunk's outputs are unused.
                replay_lens[i] = min(length, window)
                floors[i] = (
                    cached + length - replay_lens[i] if replay_lens[i] < length else recovery_floor
                )
        else:
            floors[i] = recovery_floor
    # Generation and speculative verification rows keep their live endpoints
    # and every query; their cached counts must agree with the KV manager.
    compact_local = replay_lens != host_seq_lens
    updates_local_metadata = compact_local or updates_private_metadata
    if not attention_dp and not updates_local_metadata:
        return None

    rows: List[torch.Tensor] = []
    start = 0
    for length, kept in zip(host_seq_lens, replay_lens):
        rows.append(torch.arange(start + length - kept, start + length, dtype=torch.int64))
        start += length
    replay_rows = torch.cat(rows) if rows else torch.empty(0, dtype=torch.int64)

    replay_num_cached = [
        cached + (length - kept)
        for cached, length, kept in zip(num_cached, host_seq_lens, replay_lens)
    ]

    return DecoderReplayPlan(
        rows=(
            replay_rows.to(device=metadata.seq_lens_cuda.device, non_blocking=True)
            if updates_local_metadata
            else replay_rows
        ),
        replay_seq_lens=torch.tensor(replay_lens, dtype=seq_lens.dtype),
        replay_num_cached=replay_num_cached,
        saved_seq_lens=seq_lens,
        saved_num_cached=num_cached,
        replay_prompt_lens=[
            replay_lens[r] if r < num_contexts else int(metadata.prompt_lens[r])
            for r in range(num_requests)
        ],
        saved_prompt_lens=metadata.prompt_lens,
        num_encoder_tokens=metadata.num_tokens,
        swa_floors=floors,
        updates_local_metadata=updates_local_metadata,
    )
    # No cross-layer handoff is captured here on purpose. Planning happens before
    # the first layer runs, so at this point every one of those dicts is empty --
    # the publications the decoder half consumes are written *by* the encoder half.
    # enter_decoder_replay() reads them at the boundary instead, which is the only
    # moment they exist and the only moment they are about to be cleared.


def _swa_is_reused(metadata: "CSA2TrtllmMetadata") -> bool:
    """Whether a later request could read this pass's sliding-window KV.

    Read off the manager rather than the config so a manager that disabled the
    feature for its own reasons (a draft cache does) is believed over the config
    that asked for it. An unrecognized manager answers True: refusing to replay
    costs prefill time, while replaying into a reused window cache corrupts
    another request's prefix.
    """
    manager = metadata.kv_cache_manager
    return bool(getattr(manager, "enable_block_reuse", True))


def enter_decoder_replay(
    metadata: "CSA2TrtllmMetadata",
    plan: DecoderReplayPlan,
    first_layer: Optional[int] = None,
) -> None:
    """Rebuild query views and carry encoder publications across the boundary."""
    if plan.replay_all_rank_num_tokens is not None:
        metadata.all_rank_num_tokens = plan.replay_all_rank_num_tokens
    if not plan.updates_local_metadata:
        return
    saved_indices = dict(metadata.csa2_indices)
    saved_candidates = {
        name: dict(getattr(metadata, name))
        for name in ("csa2_candidates", "csa2_candidate_blocks", "csa2_candidate_counts")
    }
    saved_last_layer = metadata._csa2_last_layer
    precomputed = set(getattr(metadata, "csa2_precomputed_kv_layers", ()))
    live_kv_lens = metadata.kv_lens_cuda[: metadata.num_seqs].clone()
    metadata.kv_cache_params.num_cached_tokens_per_seq[: len(plan.replay_num_cached)] = (
        plan.replay_num_cached
    )
    metadata.prompt_lens = plan.replay_prompt_lens
    metadata.seq_lens = plan.replay_seq_lens
    if plan.num_replay_tokens == 0:
        # No local attention runs. Keep GLOBAL publication and live KV lengths,
        # but do not ask attention backends to prepare an empty query batch.
        metadata.reset_routing()
        metadata.csa2_precomputed_kv_layers = precomputed
        metadata.csa2_replay_query_rows = plan.rows
        return
    if first_layer is not None:
        metadata.set_decoder_query_boundary(first_layer)
    metadata.prepare()
    metadata.kv_lens_cuda[: metadata.num_seqs].copy_(live_kv_lens)
    metadata.on_update_kv_lens()
    metadata.csa2_precomputed_kv_layers = precomputed
    metadata.csa2_replay_query_rows = plan.rows
    for layer, indices in saved_indices.items():
        metadata.csa2_indices[layer] = indices.index_select(0, plan.rows)
    for name, published in saved_candidates.items():
        target = getattr(metadata, name)
        for layer, value in published.items():
            target[layer] = value.index_select(0, plan.rows)
    metadata._csa2_last_layer = saved_last_layer
    # Only context SWA is truncated. Generation rows and global visibility
    # retain their original logical positions and cache history.
    starts = metadata._copy_csa2_tensor(
        "decoder_starts",
        torch.tensor(
            plan.swa_floors,
            dtype=torch.int64,
            device="cpu",
        ),
    )
    _apply_decoder_swa_floor(metadata, starts)


def enter_remote_tail_decoder(metadata: "CSA2TrtllmMetadata", context_starts: List[int]) -> None:
    """Hide pre-handoff SWA when a generation worker enters decoder layers.

    Unlike ordinary decoder replay, the query rows are already exactly the
    remote tail. Encoder layers may read transferred SWA before the handoff;
    decoder layers must see only SWA written at or after each context's split.
    Generation rows retain their ordinary visibility.
    """
    if len(context_starts) != metadata.num_contexts:
        raise ValueError("remote-tail starts must match the context rows")
    starts = metadata._copy_csa2_tensor(
        "remote_tail_starts",
        torch.tensor(
            list(context_starts) + [0] * metadata.num_generations,
            dtype=torch.int64,
            device="cpu",
        ),
    )
    _apply_decoder_swa_floor(metadata, starts)


def _apply_decoder_swa_floor(metadata: "CSA2TrtllmMetadata", starts: torch.Tensor) -> None:
    """Hide cached SWA positions below one logical floor per request from reads.

    Writes keep the prepared floors: query rows below the floor still store
    their SWA. The next layer resolves its slots against the raised floors.
    """
    replay = metadata.csa2_replay_start_positions
    floors = metadata._get_csa2_buffer("swa_read_floors", tuple(replay.shape), torch.int64)
    previous = getattr(metadata, "_csa2_swa_read_floors", None)
    torch.maximum(replay if previous is None else previous, starts, out=floors)
    metadata._csa2_swa_read_floors = floors
    # Earlier layers of this forward may have resolved slots below the floors.
    metadata._csa2_swa_resolved = False
    # The steady fast path skips the full prepare that restores the floors.
    metadata._csa2_steady_snapshot = None


def exit_decoder_replay(
    metadata: "CSA2TrtllmMetadata",
    plan: DecoderReplayPlan,
) -> None:
    """Restore the encoder pass's shape.

    Only the *inputs* are restored -- ``seq_lens`` (and with it ``num_tokens``,
    ``num_ctx_tokens`` and ``seq_lens_cuda``), the cached-token counts, and the
    floor switch. The device index buffers are deliberately left holding replay
    values: every forward rebuilds them from these inputs in ``prepare()``, and
    copying them back would cost a second rebuild to no observable end.
    """
    if plan.saved_all_rank_num_tokens is not None:
        metadata.all_rank_num_tokens = plan.saved_all_rank_num_tokens
    if not plan.updates_local_metadata:
        return
    metadata.kv_cache_params.num_cached_tokens_per_seq[: len(plan.saved_num_cached)] = (
        plan.saved_num_cached
    )
    metadata.prompt_lens = plan.saved_prompt_lens
    metadata.seq_lens = plan.saved_seq_lens
    metadata.csa2_replay_query_rows = None
    # Publications are indexed in the compact query space and cannot survive
    # shape restoration.
    metadata.reset_routing()


def gather_replayed_rows(
    tensor: Optional[torch.Tensor],
    plan: DecoderReplayPlan,
) -> Optional[torch.Tensor]:
    """Slice a per-token tensor down to the replayed rows.

    The model body carries two layouts and must not have to tell them apart at
    each call site: activations are row-major ``[N, ...]``, while position ids
    arrive as ``[N]`` or ``[1, N]`` -- per-token along the *last* axis. The axis
    is identified by which one is ``N``, and a tensor that is ``N`` on neither is
    an error rather than something to pass through, because passing an
    encoder-length tensor into the replay pass is the silent-corruption case this
    whole module is built to avoid.
    """
    if tensor is None or not plan.replays_local_tokens:
        return tensor
    num_tokens = plan.num_encoder_tokens
    rows = plan.rows
    if tensor.shape[0] == num_tokens:
        return tensor.index_select(0, rows)
    if tensor.shape[-1] == num_tokens:
        return tensor.index_select(tensor.dim() - 1, rows)
    raise ValueError(
        f"tensor of shape {tuple(tensor.shape)} is not per-token for a pass of "
        f"{num_tokens} tokens, so the decoder replay boundary cannot re-index it."
    )


def scatter_replayed_rows(
    replayed: torch.Tensor,
    plan: DecoderReplayPlan,
) -> torch.Tensor:
    """Widen the replay pass's output back to the encoder pass's row count.

    Everything downstream of the model body indexes hidden states with
    encoder-space rows -- ``_get_last_token_states`` takes ``cumsum(seq_lens) -
    1``, and the padded-token slice takes ``[:num_tokens]``. Rather than teach
    those the row map, the replayed rows are scattered back into a full-height
    buffer. The rows the decoder half did not run are zero, which is correct only
    because replay is refused when any consumer needs them (see
    ``DeepseekV41Model._plan_bounded_replay``); zeros rather than uninitialized
    memory so that a future consumer that does read them is wrong reproducibly
    instead of wrong intermittently.
    """
    if not plan.replays_local_tokens:
        return replayed
    full = replayed.new_zeros((plan.num_encoder_tokens,) + tuple(replayed.shape[1:]))
    full.index_copy_(0, plan.rows, replayed)
    return full
