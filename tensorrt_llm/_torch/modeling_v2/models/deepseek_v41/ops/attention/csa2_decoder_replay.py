# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decoder bounded replay of a context step: the engine's planner (`attention/backends/sparse/csa2/
decoder_replay.py::plan_decoder_replay`) as an entry, and as setup helpers the engine's replay switches and request /
cache predicates (`configs/deepseek_v41.py`, `pyexecutor/ced_replay.py`) and stock's replay policy and step plan
(`DeepseekV41ForCausalLM._resolve_replay_policy` / `_make_bounded_replay_plan` / `_plan_bounded_replay`), which every
target calls. Each calls the engine's code unchanged; the targets read no replay switch themselves."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.sparse.csa2.decoder_replay import (
        DecoderReplayPlan,  # noqa: F401
    )


def csa2_decoder_replay_enabled() -> bool:
    """Whether decoder bounded replay is on (`TRTLLM_V41_DECODER_BOUNDED_REPLAY`, default on); a setup helper."""
    from tensorrt_llm._torch.configs.deepseek_v41 import decoder_bounded_replay_enabled

    return decoder_bounded_replay_enabled()


def csa2_bounded_replay_tokens(required_window: int) -> int:
    """The replayed rows per prompt (`TRTLLM_V41_BOUNDED_REPLAY_TOKENS`, above `required_window`); a setup helper."""
    from tensorrt_llm._torch.configs.deepseek_v41 import bounded_replay_tokens

    return bounded_replay_tokens(required_window)


def csa2_requires_full_decoder_prefill(request) -> bool:
    """Whether `request`'s consumers need every context row (context logits, additional outputs, multimodal data)."""
    from tensorrt_llm._torch.pyexecutor.ced_replay import requires_full_decoder_prefill

    return requires_full_decoder_prefill(request)


def csa2_uses_private_decoder_cache(model, cache_manager) -> bool:
    """Whether `model`'s replay boundary uses private decoder working pages of `cache_manager`."""
    from tensorrt_llm._torch.pyexecutor.ced_replay import uses_private_decoder_cache

    return uses_private_decoder_cache(model, cache_manager)


def csa2_encoder_replay_enabled(max_num_tokens, window: int, supported: bool) -> bool:
    """Whether the engine recovers evicted Encoder state by bounded replay (`TRTLLM_V41_ENCODER_REPLAY`, the engine's
    `encoder_replay_enabled` for this token budget, replay window and whether the model's replay topology supports it:
    a decoder replay split with the boundary's precompute); a setup helper."""
    from tensorrt_llm._torch.configs.deepseek_v41 import encoder_replay_enabled

    return encoder_replay_enabled(
        max_num_tokens=max_num_tokens, replay_window=window, supported=supported
    )


def csa2_encoder_replay_tokens(request) -> int:
    """The historical rows an unconsumed Encoder replay adds before `request`'s chunk (0 without one); a setup helper
    (`pyexecutor/ced_replay.py::encoder_replay_tokens`)."""
    from tensorrt_llm._torch.pyexecutor.ced_replay import encoder_replay_tokens

    return encoder_replay_tokens(request)


def csa2_encoder_recovery(requests, num_requests: int) -> Optional[tuple]:
    """Per request row of a step, from the context requests' Encoder recovery (`request.py_ced_replay`) as the
    engine's `CSA2TrtllmMetadata.prepare_context_replay` reads it; a setup helper. Returns `(hist_end, floor)` lists of
    `num_requests` ints, or None when no context request recovers its Encoder:

    * `hist_end`: an unconsumed `EncoderReplay`'s Global end. The step's rows of that request below it are historical
      Encoder rows: they rebuild the Encoder working state (SWA) and must not write GLOBAL / compressor rows, which the
      replay's Global claim already holds. 0 elsewhere.
    * `floor`: the first position the request's Encoder SWA rows may read (the replay start, or the restored
      checkpoint's `swa_read_floor`), consumed or not, since rows before it are not in the pool. 0 elsewhere."""
    from tensorrt_llm._torch.pyexecutor.ced_replay import EncoderReplay

    if not requests:
        return None
    hist_end, floor = [0] * num_requests, [0] * num_requests
    found = False
    for row, request in enumerate(requests[:num_requests]):
        plan = getattr(request, "py_ced_replay", None)
        if isinstance(plan, EncoderReplay):
            found = True
            # A replay that restores an earlier Encoder checkpoint may read the checkpoint's window rows before its
            # first recomputed query (`swa_read_floor`), as the engine's `prepare_context_replay` does.
            floor[row] = plan.swa_read_floor if plan.restores_checkpoint else plan.start
            if not plan.consumed:
                hist_end[row] = plan.global_end
    return (hist_end, floor) if found else None


def csa2_complete_encoder_replay(requests) -> None:
    """Mark the context requests' Encoder replays consumed, after the model forward succeeded (the engine's
    `attention/backends/sparse/csa2/ced.py::complete_encoder_replay`); a setup helper. Until then the next step's
    metadata preparation expects the replay's historical rows again."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.ced import complete_encoder_replay

    if requests:
        complete_encoder_replay(list(requests))


def csa2_replay_policy(cfg, model_config, captures: bool) -> Tuple[Optional[int], int]:
    """`(first decoder layer, window)` of the bounded decoder replay, or `(None, 0)` for full prefill: stock's
    `DeepseekV41ForCausalLM._resolve_replay_policy` on this family's layout; a setup helper.

    The last unpooled KV owner (layer 20 of the released checkpoint) publishes the compressed KV and index keys of every
    row of a context chunk; from that layer on only the rows inside the final `window` tokens of each prompt run
    (queries, index selection, attention, experts). The window is the SWA window, enlarged to an embedded DSpark draft's
    window, past `TRTLLM_V41_BOUNDED_REPLAY_TOKENS` (the engine's `bounded_replay_tokens`); the switch is the engine's
    `TRTLLM_V41_DECODER_BOUNDED_REPLAY`, whose decision the cache manager's pools follow too. `captures`: whether the
    target writes a speculative worker's captures under the replay (their row map); a target without keeps full prefill
    for a speculative configuration."""
    from tensorrt_llm._torch.configs.deepseek_v41 import (
        bounded_replay_tokens,
        decoder_bounded_replay_enabled,
    )

    if not decoder_bounded_replay_enabled():
        return None, 0

    def refuse(why):
        logger.warning(
            f"DeepSeek-V4.1 decoder bounded replay was requested but {why}; running full decoder prefill."
        )
        return None, 0

    mapping = model_config.mapping
    if mapping.pp_size != 1 or mapping.cp_size != 1:
        return refuse("the replay boundary supports only PP=1 and CP=1")
    if (getattr(model_config, "extra_attrs", None) or {}).get(
        "bounded_replay_on_generation", False
    ):
        return refuse("these targets have no disaggregated remote-tail replay")
    srcs = [int(i) for i in cfg.kv_source_layer_ids]
    if not srcs:
        return refuse("no kv_source_layer_ids establish a cache-independent decoder suffix")
    last = max(srcs)
    if cfg.compress_ratios[last] != 1:
        return refuse(
            f"the last KV source (layer {last}) is pooled, so the boundary cannot publish it ahead"
        )
    late = [i for i in (cfg.engram_layer_ids or ()) if i >= last]
    if late:
        return refuse(f"Engram layers {late} need unsupported replay re-indexing")
    window = int(cfg.sliding_window)
    spec = getattr(model_config, "spec_config", None)
    if spec is not None:
        if not captures:
            return refuse("this target writes no speculative captures under the replay")
        if not (spec.spec_dec_mode.is_dspark() and spec.draft_is_embedded_in_target):
            return refuse("this speculative worker has no proven bounded capture contract")
        layers = spec.target_layer_ids
        if not layers or any(
            not isinstance(c, int) or not 0 <= c < cfg.num_hidden_layers for c in layers
        ):
            return refuse(
                "the embedded DSpark window or validated entry-capture layers are unknown"
            )
        # stock's draft window is `model_config.pretrained_config.sliding_window`: the same config value the SWA window
        # (`CSA2Layout.window_size`) is read from, so the embedded draft never widens it (stock's and the cache
        # manager's `max(layout.window_size, pretrained_config.sliding_window)`)
        window = max(window, int(cfg.sliding_window))
    window = bounded_replay_tokens(window)
    logger.info(
        f"DeepSeek-V4.1 decoder bounded replay enabled: layers {last}-{cfg.num_hidden_layers - 1} replay over a "
        f"{window}-token window (approximate, kv sources {sorted(srcs)}). Set TRTLLM_V41_DECODER_BOUNDED_REPLAY=0 for "
        "full decoder prefill."
    )
    return last, window


def csa2_replay_plan(model, attn_metadata, all_token_states_required: bool, requests=None):
    """The step's bounded-replay plan, stock's `_make_bounded_replay_plan`: which rows of each context chunk run the
    decoder half (`csa2_decoder_replay_plan`), None when `model` does not replay; a setup helper. A caller that needs
    every row's hidden state (context logits) gets no compaction; with block reuse the replay needs the cache manager's
    private decoder SWA pages. `model` carries `decoder_replay_split`
    / `decoder_replay_window` (`csa2_replay_policy`)."""
    if model.decoder_replay_split is None:
        return None
    allow = not all_token_states_required or bool(
        requests and any(csa2_requires_full_decoder_prefill(r) for r in requests)
    )
    kvm = attn_metadata.kv_cache_manager
    private = csa2_uses_private_decoder_cache(model, kvm)
    if not private and getattr(kvm, "enable_block_reuse", False):
        if not getattr(model, "_replay_reuse_warned", False):
            model._replay_reuse_warned = True
            logger.warning(
                "DeepSeek-V4.1 decoder bounded replay is disabled because KV block reuse is enabled. Replayed decoder "
                "SWA is approximate and must not be published for prefix caching. Set "
                "kv_cache_config.enable_block_reuse=False to use bounded replay; running full decoder prefill."
            )
        allow = False
    return csa2_decoder_replay_plan(
        attn_metadata, model.decoder_replay_window, requests, private, allow
    )


def csa2_step_replay_plan(
    model, attn_metadata, all_token_states_required: bool = True, requests=None
):
    """The plan the layers from `model.decoder_replay_split` on follow this step, stock's `_plan_bounded_replay`: under
    attention DP the one `prepare_adp_inputs` made before the token-count exchange (checked against the exchanged
    counts), else made here; None under a CUDA graph (a captured forward keeps every padded row) and when nothing
    replays; a setup helper. A caller that does not say whether it needs every row's state is taken to need it."""
    from tensorrt_llm._torch.utils import get_per_request_prefill_cuda_graph_flag

    if model.decoder_replay_split is None:
        return None
    mapping = attn_metadata.mapping
    if mapping is not None and mapping.enable_attention_dp:
        if attn_metadata.is_cuda_graph or get_per_request_prefill_cuda_graph_flag():
            return None
        plan = getattr(attn_metadata, "decoder_replay_plan", None)
        if plan is not None:
            if plan.replay_all_rank_num_tokens is None:
                raise ValueError("Exchange ADP token counts after prepare_adp_inputs().")
            if (
                plan.saved_all_rank_num_tokens != attn_metadata.all_rank_num_tokens
                or plan.saved_all_rank_num_tokens[mapping.tp_rank]
                != (attn_metadata.padded_num_tokens or attn_metadata.num_tokens)
            ):
                raise ValueError("ADP input rows changed after exchanging token counts.")
            if (
                plan.replays_local_tokens
                and all_token_states_required
                and not (requests and any(csa2_requires_full_decoder_prefill(r) for r in requests))
            ):
                raise ValueError(
                    "Full token states must be requested in prepare_adp_inputs(), before exchanging ADP "
                    "token counts."
                )
    else:
        plan = getattr(attn_metadata, "decoder_replay_plan", None)
        if plan is None:
            plan = csa2_replay_plan(model, attn_metadata, all_token_states_required, requests)
    if (
        plan is not None
        and plan.replays_local_tokens
        and not getattr(model, "_replay_logged", False)
    ):
        model._replay_logged = True
        logger.info(
            f"DeepSeek-V4.1 decoder bounded replay engaged: {plan.num_replay_tokens} of {plan.num_encoder_tokens} "
            f"rows re-run from layer {model.decoder_replay_split}."
        )
    return plan


class Csa2DecoderReplayPlan(OpWrapper):
    """`(metadata, window, requests, private_decoder, allow_replay) -> DecoderReplayPlan or None`: which rows of each
    context chunk run the decoder half. With a prompt end `e` (the request's `prompt_len`), a context chunk of `L` rows
    after `c` cached tokens keeps its rows at positions `>= e - window` (`max(0, c + L - max(c, e - window))` rows, the
    last ones) and gets the SWA floor `e - window` (or its recovery floor); generation rows keep every row. `rows` are
    the kept rows' indices in the step, `replay_seq_lens` the kept counts. Host planning, no communication; None when
    nothing changes (and not attention DP), under a CUDA graph, or with no window."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="two prompts in one chunk, both longer than the window",
            spec=dict(window=140, ctx=((600, 0, 600), (300, 0, 300)), gen=0),
        ),
        Cell(
            why="a chunk ending before the window starts: no row kept",
            spec=dict(window=140, ctx=((512, 0, 2048),), gen=0),
        ),
        Cell(
            why="the last chunk of a prompt, the window straddling its start",
            spec=dict(window=140, ctx=((100, 1948, 2048),), gen=0),
        ),
        Cell(
            why="a short prompt inside the window, with two generation rows: nothing to compact",
            spec=dict(window=140, ctx=((90, 0, 90),), gen=2),
        ),
    )
    note = (
        "The test restates the kept-row rule for prompts with known "
        "ends, outside attention DP (no floors from recovery)."
    )

    def __call__(
        self,
        metadata,
        window: int,
        requests=None,
        private_decoder: bool = False,
        allow_replay: bool = True,
    ) -> "Optional[DecoderReplayPlan]":
        from tensorrt_llm._torch.attention.backends.sparse.csa2.decoder_replay import (
            plan_decoder_replay,
        )

        return plan_decoder_replay(
            metadata, window, requests, private_decoder=private_decoder, allow_replay=allow_replay
        )

    def reference(
        self,
        metadata,
        window: int,
        requests=None,
        private_decoder: bool = False,
        allow_replay: bool = True,
    ):
        """`(rows, replay_seq_lens, swa_floors)`, or None when no row is dropped and no floor is set, or with no
        window."""
        if window <= 0:  # no window disables the replay, as the engine's planner
            return None
        lens = metadata.seq_lens[: metadata.num_seqs].tolist()
        cached = list(metadata.kv_cache_params.num_cached_tokens_per_seq[: metadata.num_seqs])
        ends = metadata.decoder_context_ends
        kept, floors = list(lens), [0] * len(lens)
        if allow_replay:
            for i in range(metadata.num_contexts):
                start = max(0, ends[i] - window)
                kept[i] = max(0, cached[i] + lens[i] - max(cached[i], start))
                floors[i] = start
        if kept == lens and not any(floors):
            return None
        rows, s = [], 0
        for n, k in zip(lens, kept):
            rows += list(range(s + n - k, s + n))
            s += n
        return torch.tensor(rows, dtype=torch.int64), kept, floors


csa2_decoder_replay_plan = Csa2DecoderReplayPlan()
