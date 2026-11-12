# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The weights this target expects: the Muse Spark Open NVFP4 checkpoint as the built-in model loads it.

The target's module tree is the built-in model's, so the built-in loader maps
every checkpoint tensor -- the open-format renames, the stacked NVFP4 experts and
their scales, the head-interleaved query and output-gate projection, and the
perception towers when they are built. The drafter's weights come from the
``speculative_model`` checkpoint through the built-in draft loader, not from here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenModel
from tensorrt_llm._torch.models.modeling_musespark_open_omni import (
    MuseSparkOpenOmniForConditionalGeneration,
)

from . import dispatch as _tp_agreement

if TYPE_CHECKING:
    from . import dispatch as _dispatch
    from .modeling import ModelingV2MuseSparkOpen816bA42bNvfp4Sm103Tep


def load(
    model: "ModelingV2MuseSparkOpen816bA42bNvfp4Sm103Tep",
    weights: Any,
    weight_mapper: Optional[Any] = None,
) -> Dict[str, int]:
    """Load the checkpoint ``weights`` into ``model`` with the built-in loader.

    Returns the built-in loader's key-accounting summary.
    """
    return MuseSparkOpenOmniForConditionalGeneration.load_weights(model, weights, weight_mapper)


def install_loaded_features(
    text_model: MuseSparkOpenModel,
    mapping,
    plan: "_dispatch.Plan",
    compute_capability: Optional[Tuple[int, int]] = None,
) -> None:
    """Install the plan's expert paths on a text model whose weights are final.

    ``replicated_experts``: ``ring_buddy.install`` (collective over the TP
    group), balanced. Imports ``ring_buddy`` only when an expert path is planned;
    without one this installs no expert path. With ``persistent_experts`` the install
    routes the NVFP4 layers' stacked calls to the persistent expert kernel
    (``persistent=True``; the keyword is passed only then), with
    ``three_window_placement`` it stacks a third ring window on those
    layers (``windows=3``, likewise passed only then), and with
    ``bf16_routed_experts`` the BF16 layers' stacked calls take unpacked
    routing (``bf16_unpacked=True``; likewise passed only then). On a device and TP
    size with a row cap of their own (``ring_buddy.max_rows_for`` of
    ``compute_capability``: sm103 at tp 8) the stacked path also takes the decode calls
    up to that cap (``max_rows``, passed only then); elsewhere up to ``ring_buddy.MAX_ROWS``.

    Without ``replicated_experts``, ``persistent_experts`` and ``bf16_routed_experts``
    are installed on the owner-only window (``_install_owner_window``: no
    replication, every expert on its owner), at tp 4 or tp 8; with
    ``half_window_experts`` (tp 4) the NVFP4 layers also stack the ring predecessor's
    upper half, which the prologue's routed kernel places onto (``half=True``).

    Then ``hot_replicas`` (``hot_replicas.install``: hooks only, at tp 4; the copies are
    gathered on the first forward of a deployment that serves 33..512-row decode steps),
    then ``early_expert_ids`` (``early_ids.install``, on the layers just installed),
    then ``pool_into_route`` (``DirectPool.install_sites``, whose routed kernel
    variants carry the placement and the early expert ids), each collective over the
    TP group, then ``next_qkv_prefetch``'s wrapper around every hooked layer's
    expert forward (``prefetch.wrap_expert_forwards``, outermost, so it sees
    whichever expert call the paths above installed).
    """
    if plan.enabled("replicated_experts"):
        _install_ring_buddy(text_model, mapping, plan, compute_capability)
    elif plan.enabled("persistent_experts") or plan.enabled("bf16_routed_experts"):
        _install_owner_window(text_model, mapping, plan)
    if plan.enabled("hot_replicas"):
        from .kernels.replicated_experts import hot_replicas as _hot_replicas

        # After the owner-window install, whose expert and routing hooks it wraps.
        _hot_replicas.install(text_model, mapping)
    if plan.enabled("early_expert_ids"):
        _install_early_expert_ids(text_model, mapping)
    if plan.enabled("pool_into_route"):
        pool_direct = getattr(text_model, "_pool_direct", None)
        if pool_direct is None:
            raise ValueError(
                "[musespark_open] pool_into_route is planned but the model shell built no direct pool "
                "transport"
            )
        pool_direct.install_sites(text_model, mapping, wide=plan.enabled("tep8_wide_direct_pool"))
    if plan.enabled("next_qkv_prefetch"):
        from .kernels.next_qkv_prefetch import prefetch as _qkv_prefetch

        _qkv_prefetch.wrap_expert_forwards(text_model)


def _install_ring_buddy(
    text_model,
    mapping,
    plan: "_dispatch.Plan",
    compute_capability: Optional[Tuple[int, int]] = None,
) -> None:
    """``install_loaded_features``'s ``replicated_experts`` install."""
    from .kernels.replicated_experts import ring_buddy as _ring_buddy

    extra = {"persistent": True} if plan.enabled("persistent_experts") else {}
    if plan.enabled("three_window_placement"):
        extra["windows"] = 3
    if plan.enabled("bf16_routed_experts"):
        extra["bf16_unpacked"] = True
    max_rows = _ring_buddy.max_rows_for(compute_capability, getattr(mapping, "tp_size", None))
    if max_rows != _ring_buddy.MAX_ROWS:
        # sm103 tep8: the balanced placement also takes the 17..32-row decode steps
        # (three and four requests), one ring_buddy_assign program of 256 slots.
        extra["max_rows"] = max_rows
    batch_rows = _tep8_persistent_batch_rows(
        compute_capability,
        mapping,
        plan.enabled("persistent_experts"),
        max_rows,
        plan.param("tep8_persistent_batch_experts", "MAX_ROWS")
        if plan.enabled("tep8_persistent_batch_experts")
        else 0,
    )
    if batch_rows:
        extra["persistent_rows"] = batch_rows
    if _tep8_experts_switch(
        "tep8_experts_early_start",
        plan.enabled("tep8_experts_early_start"),
        batch_rows,
        EARLY_START_WHAT,
    ):
        extra["persistent_early_start"] = True
    if _tep8_experts_switch(
        "tep8_experts_in_kernel_placement",
        plan.enabled("tep8_experts_in_kernel_placement"),
        batch_rows,
        IN_KERNEL_PLACEMENT_WHAT,
    ):
        extra["persistent_in_kernel_placement"] = True
    _ring_buddy.install(
        text_model,
        mapping,
        balance=True,
        static_from=_ring_buddy.BLOCK,
        **extra,
    )


def _ring_buddy_row_cap(ring_buddy, compute_capability, mapping) -> int:
    """The largest stacked call the balanced ring-buddy path takes on this device.

    ``max_rows_for`` is the cap ``ring_buddy.install`` will itself apply, from
    ``tep8_ring_buddy_wide`` (else the module constant ``MAX_ROWS``, 16, so the batch builds
    are clamped to 16 rows here and only the 16-column build is reachable). It takes the ring
    size, which is what ``install`` passes it, not the compute capability: a tep8 mapping
    gives 8, a tep4 one 4, anything else None. Older trees carry neither function, and the cap
    is then ``MAX_ROWS``.
    """
    resolve = getattr(ring_buddy, "max_rows_for", None)
    ring_size = getattr(ring_buddy, "ring_size", None)
    if resolve is None or ring_size is None:
        return int(ring_buddy.MAX_ROWS)
    return int(resolve(ring_size(mapping)))


def _tep8_persistent_batch_rows(
    compute_capability, mapping, persistent: bool, max_rows: int, rows: int
) -> int:
    """``tep8_persistent_batch_experts``' row bound for this install, or 0 (off).

    ``rows`` is the lever's ``MAX_ROWS``: the largest stacked NVFP4 expert call (9..32 rows:
    decode steps of two to four requests at K = 7) the persistent expert kernel serves, on its
    batch builds (``ops/moe/persistent_experts.BATCH_NTOKS``), instead of TRTLLM-gen's
    pre-routed call. Off (0), TRTLLM-gen serves every stacked call above 8 rows.

    Resolved on every rank alike. It takes effect on sm103 at tp 8 with ``persistent_experts``
    planned; elsewhere it warns and does nothing. Calls above the ring-buddy cap (``max_rows``)
    never reach the stacked path, so the bound is clamped to it, with a warning.

    Raises:
        ValueError: a ``rows`` that is not in ``9 .. BATCH_MAX_ROWS``.
    """
    if not rows:
        return 0
    from tensorrt_llm.logger import logger

    from .ops.moe import persistent_experts as _persistent

    rows = int(rows)
    if not _persistent.MAX_ROWS < rows <= _persistent.BATCH_MAX_ROWS:
        raise ValueError(
            f"[musespark_open] tep8_persistent_batch_experts MAX_ROWS {rows}: the persistent "
            f"expert kernel's batch builds serve stacked calls of {_persistent.MAX_ROWS + 1}.."
            f"{_persistent.BATCH_MAX_ROWS} rows; give the largest such call"
        )
    tp = getattr(mapping, "tp_size", None)
    if _tp_agreement.sm_of(compute_capability) != 103 or tp != 8 or not persistent:
        logger.warning(
            "[musespark_open] tep8_persistent_batch_experts has no effect here: it serves "
            "sm103 tep8 with persistent_experts planned (got sm "
            f"{_tp_agreement.sm_of(compute_capability)}, tp {tp}, "
            f"persistent_experts {persistent})"
        )
        return 0
    if rows > max_rows:
        logger.warning(
            f"[musespark_open] tep8_persistent_batch_experts MAX_ROWS {rows}: the ring-buddy "
            f"stacked path takes calls of at most {max_rows} rows; the batch builds serve "
            f"9..{max_rows}"
        )
        rows = max_rows
        if rows <= _persistent.MAX_ROWS:
            return 0
    builds = [n for n in _persistent.BATCH_NTOKS if n < rows] + [_persistent.batch_ntok(rows)]
    logger.info(
        f"[musespark_open] tep8_persistent_batch_experts: the NVFP4 "
        f"layers' stacked calls of {_persistent.MAX_ROWS + 1}..{rows} rows on the 64-expert "
        f"ring-buddy window run the persistent expert kernel's batch builds "
        f"({', '.join(f'{n} columns' for n in sorted(set(builds)))}) instead of TRTLLM-gen"
    )
    return rows


#: sm103 tep8, default off; acts only with ``tep8_persistent_batch_experts``: a 9..N-row
#: stacked NVFP4 call launches its ``ring_buddy_assign`` with programmatic dependent launch and
#: an early trigger (after its own dependency wait), and the persistent kernel's batch build
#: behind it launches while the assignment runs and reads its epoch before its own wait.
#: Same rows.
EARLY_START_WHAT = (
    "ring_buddy_assign triggers the persistent expert kernel's batch builds right after its own "
    "dependency wait, and they read their epoch before theirs"
)


#: sm103 tep8, default off; acts only with ``tep8_persistent_batch_experts``, and never with
#: ``tep8_experts_early_start`` (the batch build serves one or the other). The stacked call
#: passes the step's global expert ids and the rank's own index, and each CTA derives the
#: balanced ring placement of ``ops/moe/ring_buddy_assign`` itself.
IN_KERNEL_PLACEMENT_WHAT = (
    "the persistent expert kernel's batch builds derive the balanced ring placement themselves "
    "from the step's global expert ids"
)


def _tep8_experts_switch(name: str, on: bool, batch_rows: int, what: str) -> bool:
    """A lever of the persistent kernel's batch builds, as the plan resolved it.

    Off: off. On where the batch builds serve (``batch_rows``, from
    ``tep8_persistent_batch_experts``), with one INFO line; elsewhere a warning and off.
    """
    if not on:
        return False
    from tensorrt_llm.logger import logger

    if not batch_rows:
        logger.warning(
            f"[musespark_open] {name} has no effect here: it acts on the persistent expert "
            "kernel's batch builds, which serve only with tep8_persistent_batch_experts on "
            "sm103 tep8 with persistent_experts planned"
        )
        return False
    logger.info(
        f"[musespark_open] {name}: on the NVFP4 layers' stacked calls of "
        f"9..{batch_rows} rows, {what}"
    )
    return True


def _install_owner_window(text_model, mapping, plan: "_dispatch.Plan") -> None:
    """``persistent_experts`` / ``bf16_routed_experts`` without ``replicated_experts``.

    ``ring_buddy.install(windows=1)``: every MoE layer's stacked calls run over the
    rank's own expert block (64 experts at tp 4, 32 at tp 8) with owner-only ids;
    nothing is replicated. Collective over the TP group (its refusals are agreed).

    Raises:
        ValueError: at a tensor-parallel size other than 4 or 8, or for any
            refusal of the install (on every rank alike).
    """
    tp = int(getattr(mapping, "tp_size", 0) or 0)
    if tp == 4:
        # One tray: the owner-only window is the served expert path (a second ring
        # window would cost about 113 GiB per rank).
        pass
    elif tp == 8:
        # The tp 8 composition serves replicated_experts; this window serves plans
        # that leave it out.
        pass
    else:
        raise ValueError(
            "[musespark_open] persistent_experts / bf16_routed_experts on the owner-only expert window "
            f"are written for tp 4 and tp 8, not tp {tp}"
        )
    from .kernels.replicated_experts import ring_buddy as _ring_buddy

    extra = {"persistent": True} if plan.enabled("persistent_experts") else {}
    if plan.enabled("bf16_routed_experts"):
        extra["bf16_unpacked"] = True
    if plan.enabled("half_window_experts"):
        extra["half"] = True
    _ring_buddy.install(text_model, mapping, balance=False, windows=1, **extra)


def _install_early_expert_ids(text_model, mapping) -> None:
    """``early_expert_ids``: the persistent expert kernel's early start, on the layers the
    expert install just patched (needs their stacks and the persistent kernel's workspace)."""
    from .kernels.early_expert_ids import early_ids as _early_experts

    def install_early_ids():
        if _early_experts.install(text_model) == 0:
            raise ValueError(
                "[musespark_open] early_expert_ids installed the early expert ids on no layer "
                "(see its warning above)"
            )

    # Agreed over the TP group (replicated_embedding's collectives follow): a
    # refusal, none installed (no layer runs the prologue's routed kernel into the
    # persistent expert kernel with PDL on), or a failed compile of the persistent
    # kernel's early-ids variant.
    _tp_agreement.agreed_install(mapping, install_early_ids, (ValueError, RuntimeError))


def install_embedding_features(
    text_model: MuseSparkOpenModel, mapping, plan: "_dispatch.Plan"
) -> None:
    """Install ``replicated_embedding`` on a text model whose weights are final.

    ``replicated_embedding.install`` is collective over the TP group. Imports
    ``replicated_embedding`` only when it is planned; otherwise this does nothing.
    """
    if plan.enabled("replicated_embedding"):
        from .kernels.replicated_embedding import embedding as _replicated_embedding

        _replicated_embedding.install(text_model, mapping)
    if plan.enabled("model_boundary"):
        # After replicated_embedding, whose table it gathers from. Collective (agreed refusal).
        from .kernels.boundary_early_release import triggers as _pdl_triggers
        from .kernels.model_boundary import boundary as _model_boundary

        _model_boundary.install(text_model, _pdl_triggers.capture_layers(text_model), mapping)
