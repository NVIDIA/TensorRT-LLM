# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The layer-level installs of a composition and the fused post-attention hop's layer pieces."""

from __future__ import annotations

import contextlib
import dataclasses
from typing import Optional

import torch

from tensorrt_llm._torch.models.modeling_musespark_open import (
    MuseSparkOpenDecoderLayer,
    MuseSparkOpenGatedResidual,
    MuseSparkOpenModel,
)
from tensorrt_llm.functional import AllReduceParams, AllReduceStrategy
from tensorrt_llm.logger import logger

from . import attention as _attention
from . import dispatch as _dispatch
from . import dispatch as _tp_agreement
from .attention import _PREATTN_PREP_MAX_ROWS, DecodeAttentionPlan, _attention_class
from .ops.allreduce import post_attn_hop as _hop


def _refuse_unfused_sites(layer_kwargs: dict, plan: "_dispatch.Plan") -> None:
    """Refuse a fast path whose unpool site ``projection_gathers.install`` kept on the built-in path.

    ``install`` returns no unpool keyword when it fused no unpool site.
    ``boundary_early_release`` moves the trigger of the fused unpool boundary,
    ``capture_fold`` writes the DFlash capture from it, and
    ``unpool_into_boundary`` publishes into its transport: without that site
    each would change nothing on its path.

    Raises:
        ValueError: naming the fast path and the site.
    """
    if layer_kwargs.get("fused_unpool_gather"):
        return
    for path, name, what in (
        ("boundary_early_release", "early_release", "moves the trigger of"),
        ("capture_fold", "capture_fold", "writes the DFlash capture from"),
        ("unpool_into_boundary", "unpool_direct", "publishes into"),
    ):
        if plan.enabled(path):
            raise ValueError(
                f"[musespark_open] {name} {what} the projection_gathers fused unpool boundary, which "
                "runs the built-in path on this installation (see the projection_gathers warning above)"
            )


def _refuse_unconverted_edges(converted: list, kept: list, num_layers: int) -> None:
    """Refuse ``boundary_early_release`` when it converted no boundary -> QKV edge it was to convert.

    Every layer but the last has one edge; the DFlash capture layers (``kept``;
    those ``capture_fold`` does not fold) keep theirs by design. Any other edge left unconverted
    with none converted means the fused boundary class was not installed where the install looks
    for it.

    Raises:
        ValueError: if edges outside the capture layers exist and none was converted.
    """
    open_edges = num_layers - 1 - len(kept)
    if not converted and open_edges > 0:
        raise ValueError(
            f"[musespark_open] boundary_early_release converted none of the {open_edges} fused "
            "boundary -> QKV edges outside the DFlash capture layers; the plan would run without its "
            "early launches"
        )


def _refuse_uninstalled_prologue(installed: int) -> None:
    """Refuse ``moe_prologue`` when ``moe_prologue.install`` re-classed no layer.

    ``install`` keeps every layer on its path, with a warning and on every TP
    rank alike, when the fused pool site runs the built-in path or built-in code
    the routed prologue reproduces has changed on some rank; the plan would
    then carry the fast path's name and none of its change.

    Raises:
        ValueError: naming the fast path.
    """
    if installed == 0:
        raise ValueError(
            "[musespark_open] moe_prologue installed the one-chain MoE prologue on no layer "
            "(see the moe_prologue warning above)"
        )


#: The layer keyword of a decode step the fused hop takes: o_proj returns its partial unreduced.
_NO_ALLREDUCE = AllReduceParams(enable_allreduce=False)


class DirectPublish:
    """``attention_output_into_hop``: per layer, the o_proj dispatcher that publishes and the slot it fills.

    On a decode step of ``ROWS`` rows, ``arm`` hands layer ``l``'s o_proj dispatcher this
    rank's ``[ROWS, 8192]`` slot of the direct transport's buffer ``l mod 2`` (on the
    multicast address) and tells layer ``l``'s join that slot and parity; ``disarm`` takes
    both back. The step's row count is the same on every rank, so every rank publishes and
    consumes on the same calls. The parity is fixed per layer because the GEMM cannot read
    the workspace's state; consecutive calls on the workspace alternate buffers because
    every layer runs the hop once per step, in layer order, and the layer count is even
    (``_build_post_attn_hop`` refuses an odd one).

    Publication and consumption are two launches with host code between them, so a step
    that raises in between (eager steps only: a replay runs no host code) leaves a buffer
    published and never consumed, and the fixed parity would hand its words to a later
    call. ``check_step`` therefore runs before anything is published, and
    ``PostAttnHopCall.armed`` calls ``recover`` on any exception after ``arm``: a
    collective reset of the transport. A reset that fails poisons this object: every
    later ``arm`` raises.
    """

    #: The one row count published directly: the K = 7 verify step. Other row counts keep
    #: the post-attention hop, on its own workspace.
    ROWS = 8

    def __init__(
        self,
        workspace: "_hop.PostAttnDirectWorkspace",
        dispatchers: tuple,
        joins: "tuple[MuseSparkOpenTepPostAttnJoin, ...]",
    ) -> None:
        if len(dispatchers) != len(joins) or len(joins) % 2:
            raise ValueError(
                f"attention_output_into_hop needs one o_proj dispatcher per join and an even layer "
                f"count, got "
                f"{len(dispatchers)} dispatchers and {len(joins)} joins"
            )
        self.op = _hop.post_attn_consumer_hop
        self.workspace = workspace
        self.dispatchers = tuple(dispatchers)
        self.joins = tuple(joins)
        self.slots = tuple(workspace.slot(index % 2, self.ROWS) for index in range(len(joins)))
        #: Why the transport could not be reset after a failed step, or ``None``.
        self.poisoned: Optional[str] = None

    def check_step(self, residual: torch.Tensor) -> None:
        """Before anything is published: the step's input residual is what the consumer takes.

        Every layer's consumer call reads the residual the previous layer returned, which
        has the input's dtype, shape and device (the consumer and the fused post-FFN boundary
        return FP32 ``[rows, 8192]`` tensors of their own); the coefficients and the norm weights
        were checked when the post-attention hop was built. So the step's input is the one runtime operand
        left to check, and it is checked before layer 0's o_proj stores anything.

        Raises:
            ValueError: if ``residual`` is not a contiguous FP32 ``[ROWS, 8192]`` tensor on
                the workspace's device, or the step is being compiled.
        """
        device = getattr(self.workspace, "device", residual.device)
        if (
            residual.dtype != torch.float32
            or tuple(residual.shape) != (self.ROWS, _hop.HIDDEN)
            or not residual.is_contiguous()
            or residual.device != device
            or torch.compiler.is_compiling()
        ):
            raise ValueError(
                f"[musespark_open] attention_output_into_hop: a {self.ROWS}-row step's residual is {residual.dtype} "
                f"{tuple(residual.shape)} on {residual.device} (contiguous="
                f"{residual.is_contiguous()}); the consumer hop takes a contiguous FP32 "
                f"[{self.ROWS}, {_hop.HIDDEN}] tensor on {device}, not compiled"
            )

    def arm(self) -> None:
        """Hand every layer's o_proj its slot and every join its slot and parity.

        Raises:
            RuntimeError: if an earlier failed step left the transport unreset.
        """
        if self.poisoned is not None:
            raise RuntimeError(
                "[musespark_open] attention_output_into_hop: the direct transport was not reset after "
                "a failed step "
                f"({self.poisoned}); its buffers may hold words no call consumed, so no "
                f"{self.ROWS}-row step may publish into it"
            )
        for index, (dispatcher, join) in enumerate(zip(self.dispatchers, self.joins)):
            dispatcher.publish = self.slots[index]
            join._direct = (self.slots[index], index % 2)

    def disarm(self) -> None:
        for dispatcher, join in zip(self.dispatchers, self.joins):
            dispatcher.publish = None
            join._direct = None

    def recover(self, error: BaseException) -> None:
        """After ``error`` ended an armed step: every slot back to the sentinel on every rank.

        Collective (``PostAttnDirectWorkspace.reset``); a step's exceptions come from what
        every rank shares (shapes, the configuration), so every rank arrives here. Inside a
        CUDA graph capture nothing ran, and nothing is reset. A reset that fails poisons
        this object (``arm`` then raises) and is logged; it never replaces ``error``.
        """
        try:
            if torch.cuda.is_current_stream_capturing():
                return
        except RuntimeError:
            pass
        logger.error(
            f"[musespark_open] attention_output_into_hop: a {self.ROWS}-row decode step raised while "
            "o_proj published into the "
            f"post-attention hop's direct transport ({error!r}); resetting the transport on "
            "every rank of the group (collective: a rank that did not raise leaves this one "
            "waiting here)"
        )
        try:
            self.workspace.reset()
        except Exception as reset_error:  # noqa: BLE001 - logged and kept; the step's error propagates
            self.poisoned = repr(reset_error)
            logger.error(
                "[musespark_open] attention_output_into_hop: the direct transport could not be reset "
                f"({reset_error!r}); every "
                f"later {self.ROWS}-row step raises"
            )
        else:
            logger.warning(
                "[musespark_open] attention_output_into_hop: direct transport reset on every rank "
                "after a failed step"
            )


class PostAttnHopCall:
    """What ``DecodeTarget`` holds under ``post_attention_hop``: the op, its workspace and the joins it arms.

    With ``attention_output_into_hop``, also the ``DirectPublish`` that steps of its row count arm.
    """

    def __init__(
        self,
        workspace: "_hop.PostAttnHopWorkspace",
        joins: "tuple[MuseSparkOpenTepPostAttnJoin, ...]" = (),
        direct: Optional[DirectPublish] = None,
    ) -> None:
        self.op = _hop.post_attn_hop
        self.workspace = workspace
        self.joins = tuple(joins)
        self.direct = direct

    @staticmethod
    def takes(rows: int) -> bool:
        """Whether a decode step of ``rows`` rows runs through the armed joins.

        Every rank of the group sees the same rows, so every rank arms or none does.
        """
        return 1 <= rows <= _hop.MAX_ROWS

    @contextlib.contextmanager
    def armed(self, rows: Optional[int] = None, residual: Optional[torch.Tensor] = None):
        """Arm every join for the duration of one decode forward; disarm on any exit.

        A step of ``DirectPublish.ROWS`` rows (``attention_output_into_hop``) also arms the
        direct publication,
        after ``DirectPublish.check_step`` on the step's input ``residual`` (when given);
        an exception after that hands the transport to ``DirectPublish.recover`` before it
        propagates.
        """
        direct = self.direct if self.direct is not None and rows == DirectPublish.ROWS else None
        for join in self.joins:
            join._armed = self
        direct_armed = False
        try:
            if direct is not None:
                if residual is not None:
                    direct.check_step(residual)
                direct.arm()
                direct_armed = True
            yield
        except Exception as error:
            if direct_armed:
                direct.recover(error)
            raise
        finally:
            if direct is not None:
                direct.disarm()
            for join in self.joins:
                join._armed = None


class MuseSparkOpenTepPostAttnJoin(MuseSparkOpenGatedResidual):
    """A layer's built-in attention-side residual join, which ``DecodeTarget`` can arm.

    Never constructed directly: ``adopt`` re-classes a layer's built-in
    ``add_attn`` in place and adds two plain attributes. It is the only module
    ``post_attention_hop`` re-classes. Armed (by ``PostAttnHopCall.armed``, for one decode
    forward whose layer calls carry ``all_reduce_params=_NO_ALLREDUCE``), the
    join receives the unreduced o_proj partial as ``branch`` and returns what
    the built-in all-reduce followed by the built-in join returns: through the
    fused hop when the call fits it, through those two built-in steps
    otherwise. Unarmed it is the built-in join.

    The fused hop and the built-in all-reduce are two different collectives, so
    the choice between them is made on what every rank of the group shares --
    the row count, dtypes, shapes and devices (``hop_takes``) and the layer's
    configuration -- never on a tensor's address or strides: a rank that chose
    the other one would leave every rank waiting. A rank-local layout the
    kernel cannot read in place is copied (``readable``) rather than declined.

    With ``attention_output_into_hop`` armed for the step (``DirectPublish.arm``) the join also holds
    ``_direct``, its layer's slot and parity: o_proj has already written the
    partial into that slot of every rank, so the join runs the consumer hop.
    That call cannot fall back to the built-in all-reduce (every peer polls this
    rank's slot, and nothing may read the slot's multicast address), so a call
    the hop would not take raises. A partial that is not the slot itself (a
    rank-local path that bypassed the publication) is copied into it first.
    """

    _armed: Optional[PostAttnHopCall]
    _o_proj_all_reduce: object
    _direct: Optional[tuple] = None

    @classmethod
    def adopt(
        cls, join: MuseSparkOpenGatedResidual, o_proj_all_reduce
    ) -> "MuseSparkOpenTepPostAttnJoin":
        """Re-class one built-in join, given the all-reduce of its layer's o_proj.

        Raises:
            TypeError: if ``join`` is not exactly the built-in ``MuseSparkOpenGatedResidual``.
        """
        if type(join) is not MuseSparkOpenGatedResidual:
            raise TypeError(
                f"{cls.__name__} adopts the built-in MuseSparkOpenGatedResidual, "
                f"got {type(join).__name__}"
            )
        join.__class__ = cls
        join._armed = None
        # A plain attribute, not a registered submodule: the all-reduce stays a
        # child of o_proj only, and the module tree stays the built-in model's.
        object.__setattr__(join, "_o_proj_all_reduce", o_proj_all_reduce)
        return join

    def _add_after_rmsnorm_and_norm(self, residual, branch, eps, next_norm, gain=None):
        call = self._armed
        if call is None:
            return super()._add_after_rmsnorm_and_norm(residual, branch, eps, next_norm, gain)
        if self._direct is not None:
            return self._consume_published(call, residual, branch, eps, next_norm, gain)
        alpha, beta = self.coefficients()
        weight = next_norm.weight
        if (
            gain is None
            and self._fused_enabled
            and next_norm.use_gemma
            and not torch.compiler.is_compiling()
            and _hop.hop_takes(branch, residual, alpha, beta, weight)
        ):
            branch, residual, alpha, beta, weight = _hop.readable(
                branch, residual, alpha, beta, weight
            )
            return call.op(
                branch,
                residual,
                alpha,
                beta,
                weight,
                eps,
                next_norm.variance_epsilon,
                self._preqkv_norm_fp32,
                call.workspace,
            )
        # A call the hop does not take: the built-in all-reduce, then the built-in join.
        return super()._add_after_rmsnorm_and_norm(
            residual, self._o_proj_all_reduce(branch), eps, next_norm, gain
        )

    def _consume_published(self, call, residual, branch, eps, next_norm, gain):
        """The consumer hop on the slot o_proj published into (see the class docstring)."""
        slot, parity = self._direct
        alpha, beta = self.coefficients()
        weight = next_norm.weight
        if not (
            gain is None
            and self._fused_enabled
            and next_norm.use_gemma
            and not torch.compiler.is_compiling()
            and _hop.hop_takes(slot, residual, alpha, beta, weight)
            and tuple(branch.shape) == tuple(slot.shape)
        ):
            raise RuntimeError(
                "[musespark_open] attention_output_into_hop: an armed direct-publication step reached "
                "a join the consumer hop "
                f"does not take (branch {tuple(branch.shape)}, slot {tuple(slot.shape)}, "
                f"gain={gain is not None}); every rank polls this call, so it cannot fall back"
            )
        if branch.data_ptr() != slot.data_ptr():
            logger.warning_once(
                "[musespark_open] attention_output_into_hop: an o_proj partial did not come from the "
                "publishing dispatcher; "
                "copied into the slot before the consumer hop",
                key="musespark_open.direct_publish.copied_partial",
            )
            _hop.publish_copy(slot, branch)
        residual, alpha, beta, weight = _hop.readable(residual, alpha, beta, weight)
        direct = call.direct
        return direct.op(
            slot,
            residual,
            alpha,
            beta,
            weight,
            eps,
            next_norm.variance_epsilon,
            self._preqkv_norm_fp32,
            parity,
            direct.workspace,
        )


def post_attn_hop_refusals(layer: MuseSparkOpenDecoderLayer) -> list:
    """Why the layer's built-in hop is not the pair the fused hop reproduces, if it is not.

    That pair is the MNNVL one-shot all-reduce on o_proj (not fused into the
    GEMM) followed by the fused residual join on the cluster boundary, into a
    BF16 Gemma pre-FFN norm of width 8192. Reads the layer and changes nothing;
    the layer may be any ``MuseSparkOpenDecoderLayer``, re-classed by another
    fast path or not.

    Raises:
        TypeError: if ``layer`` is not a ``MuseSparkOpenDecoderLayer``.
    """
    if not isinstance(layer, MuseSparkOpenDecoderLayer):
        raise TypeError(
            f"post_attention_hop reads a MuseSparkOpenDecoderLayer, got {type(layer).__name__}"
        )
    all_reduce = layer.self_attn.o_proj.all_reduce
    weight = getattr(layer.pre_ffn_norm, "weight", None)
    return [
        reason
        for failed, reason in (
            (
                type(layer.add_attn) is not MuseSparkOpenGatedResidual,
                "add_attn is not the built-in MuseSparkOpenGatedResidual (got "
                f"{type(layer.add_attn).__name__})",
            ),
            (all_reduce is None, "o_proj does not all-reduce"),
            (
                getattr(layer.self_attn.o_proj, "use_fused_gemm_allreduce", False),
                "o_proj fuses its all-reduce into the GEMM",
            ),
            (
                all_reduce is not None
                and (
                    all_reduce.strategy != AllReduceStrategy.MNNVL
                    or all_reduce.mnnvl_allreduce is None
                    or all_reduce.symm_mem_allreduce is not None
                ),
                "the o_proj all-reduce is not the MNNVL one-shot "
                "(MUSESPARK_OPEN_ALLREDUCE=MNNVL on an MNNVL-capable group)",
            ),
            (
                not layer.add_attn._fused_enabled,
                "MUSESPARK_OPEN_FUSED_RESIDUAL is off (the built-in join is not the "
                "cluster boundary)",
            ),
            (not layer.pre_ffn_norm.use_gemma, "the pre-FFN norm is not Gemma-style"),
            (
                weight is None or weight.dtype != torch.bfloat16 or weight.numel() != _hop.HIDDEN,
                f"the pre-FFN norm weight is not BF16 of width {_hop.HIDDEN}",
            ),
        )
        if failed
    ]


def _build_post_attn_hop(
    model: MuseSparkOpenModel, early_trigger: bool = False, stale_poll: bool = False
) -> PostAttnHopCall:
    """Take the group's one workspace (collective), then re-class every layer's ``add_attn``.

    All or nothing: every layer is checked before the workspace is built and any
    join is re-classed. The workspace comes from ``shared_workspace``, so a
    constructor that runs again reuses it. ``early_trigger`` places the hop's
    dependent-launch trigger after its last transport access; ``stale_poll`` has it
    re-read only the peer slots still stale.
    """
    if not _hop.HAS_CUTE_DSL:
        raise ValueError("[musespark_open] post_attention_hop needs the CuTe DSL")
    refused = [
        f"layer {layer.layer_idx}: " + "; ".join(reasons)
        for layer in model.layers
        for reasons in (post_attn_hop_refusals(layer),)
        if reasons
    ]
    if refused:
        raise ValueError(
            "[musespark_open] post_attention_hop: post_attn_hop cannot replace the hop of "
            + " | ".join(refused)
        )
    variants = sorted(
        {
            (
                float(layer.post_attn_norm.variance_epsilon),
                float(layer.pre_ffn_norm.variance_epsilon),
                bool(layer.add_attn._preqkv_norm_fp32),
            )
            for layer in model.layers
        }
    )
    workspace = _hop.shared_workspace(
        model.layers[0].mapping,
        variants=variants,
        early_trigger=early_trigger,
        stale_poll=stale_poll,
    )
    joins = tuple(
        MuseSparkOpenTepPostAttnJoin.adopt(layer.add_attn, layer.self_attn.o_proj.all_reduce)
        for layer in model.layers
    )
    _hop.log_once_enabled(len(model.layers), workspace.multicast, early_trigger, stale_poll)
    return PostAttnHopCall(workspace, joins)


def _build_direct_publish(
    model: MuseSparkOpenModel, call: PostAttnHopCall, early_trigger: bool = False
) -> DirectPublish:
    """``attention_output_into_hop``: take the group's direct workspace (collective) for ``call``'s joins.

    Checked first, on every layer, and agreed over the TP group before the workspace's
    collective construction (``tp_agreement.agreed_refusal``, itself collective): an even
    layer count (the per-layer parity), the post-attention hop's multicast transport, and
    every o_proj on the dense GEMM dispatcher with cuBLAS for ``DirectPublish.ROWS`` rows --
    the kernel whose epilogue stores are known to publish through the multicast address.
    ``early_trigger`` gives the consumer hop the hop's early trigger placement. Every o_proj site
    then lists ``DirectPublish.ROWS`` in its ``skip_prefetch_rows``. The kernel cuBLAS
    actually picks is checked once the weights are loaded (``verify_direct_publish``).

    Raises:
        ValueError: on every rank alike, naming what is missing on which ranks; nothing
            has been built then.
    """
    from .kernels.dense_gemms import dispatcher as dense_gemms
    from .ops.gemm.dense_gemm import CUBLAS, parse_kernel

    layers = list(model.layers)
    reasons = []
    if len(layers) % 2:
        reasons.append(f"{len(layers)} layers: the per-layer buffer parity needs an even count")
    if not call.workspace.multicast:
        reasons.append("the post-attention hop runs without the multicast transport")
    if len(call.joins) != len(layers):
        reasons.append(f"{len(call.joins)} post-attention hop joins for {len(layers)} layers")
    dispatchers = []
    for layer in layers:
        o_proj = layer.self_attn.o_proj
        if not isinstance(o_proj, dense_gemms.MuseSparkOpenTepDenseLinear):
            reasons.append(f"layer {layer.layer_idx}: o_proj is not on the dense GEMM dispatcher")
            continue
        dispatcher = o_proj._low_m_gemm_dispatcher
        kernel = dispatcher.site.plan.kernels[dispatcher.site.role][DirectPublish.ROWS - 1]
        if dispatcher.site.role != "o_proj" or parse_kernel(kernel)[0] != CUBLAS:
            reasons.append(
                f"layer {layer.layer_idx}: o_proj runs {kernel} at {DirectPublish.ROWS} rows, "
                "not cuBLAS"
            )
            continue
        dispatchers.append(dispatcher)
    refusal = _tp_agreement.agreed_refusal(
        layers[0].mapping,
        "[musespark_open] attention_output_into_hop cannot publish o_proj directly: "
        + " | ".join(reasons)
        if reasons
        else None,
    )
    if refusal is not None:
        raise ValueError(refusal)
    # The o_proj sites' producer policy: a cuBLAS call stores into the slot, any other
    # call's product is copied there.
    policy = dense_gemms.PublishPolicy("direct_publish", frozenset({CUBLAS}), _hop.publish_copy)
    for dispatcher in dispatchers:
        dense_gemms.claim_publish(dispatcher, policy)
    variants = sorted(
        {
            (
                float(layer.post_attn_norm.variance_epsilon),
                float(layer.pre_ffn_norm.variance_epsilon),
                bool(layer.add_attn._preqkv_norm_fp32),
            )
            for layer in layers
        }
    )
    workspace = _hop.shared_direct_workspace(
        layers[0].mapping, variants=variants, early_trigger=early_trigger
    )
    # The dense GEMM prefetch of the pool weight, forked after o_proj and joined by the pool GEMM
    # on its side stream, outlasts the shorter consumer hop and then gates the pool GEMM:
    # the published row count forks none.
    dropped = 0
    for dispatcher in dispatchers:
        site = dispatcher.site
        site.skip_prefetch_rows = frozenset(site.skip_prefetch_rows | {DirectPublish.ROWS})
        dropped += int(bool(site.plan.prefetch) and site.next_weight is not None)
    _hop.log_once_direct(len(layers), DirectPublish.ROWS, early_trigger, dropped)
    return DirectPublish(workspace, tuple(dispatchers), call.joins)


def verify_direct_publish(text_model: MuseSparkOpenModel) -> str:
    """``attention_output_into_hop``, once the weights are final: the kernel the published o_proj runs.

    The protocol needs the producer to store each output element exactly once, with no
    reduction through the output (``PostAttnDirectWorkspace``). cuBLAS picks its kernel by
    heuristic, so the build's check of the dense GEMM plan cannot see it. Here layer 0's real
    dispatcher publishes a ``DirectPublish.ROWS``-row product of its real weight into slot 0
    inside a CUDA graph capture -- as the decode graph does -- that is never replayed (so
    nothing is published), and ``PostAttnDirectWorkspace.check_producer`` reads what was
    captured. Collective; the shell's ``load_weights`` calls it before any graph is
    captured. Returns the kernel's name.

    Raises:
        ValueError: on every rank, if the direct publication was not built, or any rank's
            capture is not exactly one kernel of ``post_attn_hop.VALIDATED_PRODUCERS``.
    """
    call = getattr(text_model, "_post_attn_hop", None)
    direct = getattr(call, "direct", None)
    if direct is None:
        raise ValueError(
            "[musespark_open] attention_output_into_hop is planned but the model has no direct publication"
        )
    o_proj = text_model.layers[0].self_attn.o_proj
    dispatcher = direct.dispatchers[0]
    slot = direct.slots[0]
    weight = o_proj.weight
    x = torch.zeros(
        DirectPublish.ROWS, int(weight.shape[1]), dtype=weight.dtype, device=weight.device
    )

    def publish() -> None:
        dispatcher.publish = slot
        try:
            with torch.inference_mode():
                dispatcher.apply(o_proj, x, weight, None)
        finally:
            dispatcher.publish = None

    def warm() -> None:
        with torch.inference_mode():
            torch.mm(x, weight.t())

    return direct.workspace.check_producer(publish, warm)


@dataclasses.dataclass
class AdoptedLayers:
    """What ``adopt_model`` installed, which the core keeps as its attributes."""

    finalize_transport: Optional[object]
    decode_attention: Optional[DecodeAttentionPlan]
    dense_gemm_plan: Optional[object]
    layer_kwargs: dict
    post_attn_hop: Optional[PostAttnHopCall]
    norm_gate_trigger: Optional[str]
    swa_block: Optional[object]
    unpool_direct: Optional[object]
    pool_direct: Optional[object]
    qkv_prefetch: Optional[object]


def adopt_model(model: MuseSparkOpenModel, plan: "_dispatch.Plan") -> AdoptedLayers:
    """Install ``plan``'s layer-level fast paths on the built-in text model, in install order.

    Raises:
        ValueError: if a fast path of the plan cannot run on this model.
    """
    preattn_prep_on = plan.enabled("preattn_prep")
    decode_attention = (
        DecodeAttentionPlan(sliding_window=True, q_fp8=True)
        if plan.enabled("swa_paged_decode")
        else None
    )
    if preattn_prep_on:
        mismatch = _attention._builtin_prep_kernel_mismatch()
        if mismatch is not None:
            raise ValueError(
                "[musespark_open] preattn_prep reproduces flashinfer's CuTe-DSL RMSNorm and cos/sin-cache "
                f"RoPE, which the built-in prep does not run here: {mismatch}"
            )
    dense_gemm_plan = None
    if plan.enabled("dense_gemms"):
        # Imported only when planned: with no fast path the target imports
        # and runs nothing beyond the built-in modules. First of the re-classes, so
        # its configuration refusals leave the model untouched.
        from .kernels.dense_gemms import dispatcher as dense_gemms

        dense_gemm_plan = dense_gemms.install(model, prefetch=plan.enabled("dense_gemm_prefetch"))
    norm_gate_trigger = None
    early_release_on = plan.enabled("boundary_early_release")
    if early_release_on:
        # Imported only when planned, like the other installers.
        from .kernels.boundary_early_release import triggers as pdl_triggers

        norm_gate_trigger = pdl_triggers.NORM_GATE_TRIGGER
    # preattn_prep, swa_paged_decode and boundary_early_release all re-class the
    # attention: one class carries what is planned.
    attention_class = _attention_class(
        preattn_prep_on, decode_attention, norm_gate_trigger is not None
    )
    swa_block = None
    if plan.enabled("swa_block"):
        # Imported only when planned; swa_block wraps whatever class the attention
        # re-classes above picked.
        from .kernels.swa_block import attention as swa_attention_block

        swa_block = swa_attention_block.SwaBlockPlan(fused_prep=True, fused_gate=True)
        attention_class = swa_attention_block.attention_class(attention_class)
    if attention_class is not None:
        # Every layer is checked before any is re-classed: a refusal changes nothing.
        for layer in model.layers:
            attention_class.check(layer.self_attn)
        for layer in model.layers:
            attention_class.adopt(layer.self_attn)
    if plan.enabled("full_attention_prep_append"):
        # After every attention re-class: the prep append puts its mixin in front of
        # whatever class the full-attention layers carry by now.
        from .kernels.full_attention_prep_append import attention as full_attention

        full_attention.install(model)
        if plan.enabled("dense_gemm_prefetch"):
            # The full-attention layers' QKV site forks no attention output
            # prefetch behind the folded prep.
            full_attention.drop_o_proj_prefetch(model)
    if plan.enabled("full_attention_split_kv"):
        # After the prep append: the split-KV mixin goes in front of the prep append's on
        # the full-attention layers.
        # Its refusal is agreed over the TP group: collectives follow in this install.
        from .kernels.full_attention_split_kv import attention as split_kv_attention

        _tp_agreement.agreed_install(
            model.layers[0].mapping, lambda: split_kv_attention.install(model)
        )
        # The split-KV layers launch the kernel's spill-free build; agreed like the install.
        _tp_agreement.agreed_install(
            model.layers[0].mapping, lambda: split_kv_attention.install_spill_free(model)
        )
    if plan.enabled("full_attention_prep_fold"):
        # After split-KV: the prep fold's mixin goes in front of split-KV's on the
        # full-attention layers. Its refusal is agreed over the TP group, as split-KV's.
        from .kernels.full_attention_prep_fold import attention as prep_fold_attention

        _tp_agreement.agreed_install(
            model.layers[0].mapping, lambda: prep_fold_attention.install(model)
        )
    if preattn_prep_on:
        logger.info(
            f"[musespark_open] preattn_prep: fused pre-attention prep on the decode path of all "
            f"{len(model.layers)} layers (steps of at most {_PREATTN_PREP_MAX_ROWS} rows)"
        )
    if decode_attention is not None:
        sliding = sum(1 for layer in model.layers if layer.self_attn.is_sliding)
        logger.info(
            f"[musespark_open] swa_paged_decode: decode attention plan {decode_attention} over {sliding} "
            f"sliding-window and {len(model.layers) - sliding} full-attention layers"
        )
    if swa_block is not None:
        swa_attention_block.log_installed(model.layers, swa_block)
    layer_kwargs = {}
    if plan.enabled("projection_gathers"):
        # Imported only when planned: with no fast path the target imports
        # and runs nothing beyond the built-in modules.
        from .kernels.projection_gathers import gathers as projection_gathers

        layer_kwargs = projection_gathers.install(model, unpool=True, pool=True)
    # The site decision is agreed over the TP group, so every rank raises alike.
    _refuse_unfused_sites(layer_kwargs, plan)
    # After projection_gathers: moe_prologue re-classes its MoE as the one-chain prologue.
    if plan.enabled("moe_prologue"):
        from .kernels.moe_prologue import prologue as moe_prologue

        installed = moe_prologue.install(model)
        # None installed (the pool site not fused) is agreed over the TP group inside
        # install, before the prologue's inbox is built, so every rank raises alike.
        _refuse_uninstalled_prologue(installed)
        # The router fork: its refusals (PDL off, a stream not created) are agreed over
        # the TP group.
        moe_prologue.install_router_fork(model)
    # The capture layers whose boundary writes the DFlash capture (capture_fold): no
    # plain capture copy follows their boundary, so boundary_early_release converts
    # their QKV edge too.
    capture_folded = frozenset()
    if plan.enabled("capture_fold"):
        from .kernels.capture_fold import fold as capture_fold

        spec_config = getattr(getattr(model, "model_config", None), "spec_config", None)
        installed = capture_fold.install(model, spec_config)
        layer_kwargs = dict(layer_kwargs, **installed)
        if installed:
            capture_folded = capture_fold.folded_layers(model, spec_config)
    # post_attention_hop reads each layer (any MuseSparkOpenDecoderLayer, the
    # projection_gathers re-class included) and re-classes only add_attn, which no
    # other fast path touches.
    post_attn_hop = (
        _build_post_attn_hop(model, early_trigger=True, stale_poll=True)
        if plan.enabled("post_attention_hop")
        else None
    )
    if early_release_on:
        # After projection_gathers (whose boundary class it reads) and dense_gemms
        # (whose prefetch it trims).
        converted, kept = pdl_triggers.install_boundary_triggers(
            model, pdl_triggers.capture_layers(model) - capture_folded
        )
        _refuse_unconverted_edges(converted, kept, len(model.layers))
        dropped = pdl_triggers.drop_converted_prefetches(
            model, dense_gemm_plan, qkv=bool(converted)
        )
        pdl_triggers.log_installed(
            len(model.layers),
            converted,
            kept,
            norm_gate_trigger is not None,
            dropped,
        )
    qkv_prefetch = None
    if plan.enabled("next_qkv_prefetch"):
        # After dense_gemms (whose plan carries the fork) and boundary_early_release
        # (which converts the edges). Its refusal is agreed over the TP group: the
        # unpool_into_boundary, finalize_transport and attention_output_into_hop
        # collectives follow.
        from .kernels.next_qkv_prefetch import prefetch as _qkv_prefetch

        qkv_prefetch = _tp_agreement.agreed_install(
            model.layers[0].mapping, lambda: _qkv_prefetch.install(model, dense_gemm_plan)
        )
    unpool_direct = None
    if plan.enabled("unpool_into_boundary"):
        # After projection_gathers (whose join it re-classes), dense_gemms (whose
        # dispatcher publishes), boundary_early_release (whose trigger it keeps) and
        # capture_fold (whose capture it keeps).
        from .kernels.unpool_into_boundary import boundary as _unpool_direct

        unpool_direct = _unpool_direct.install(
            model, dense_gemm_plan, layer_kwargs, capture_folded, mapping=model.layers[0].mapping
        )
    pool_direct = None
    if plan.enabled("pool_into_route"):
        # After dense_gemms (whose pool dispatcher publishes) and the router fork
        # (whose side stream the router's publish twin runs on). Its refusal is agreed
        # over the TP group before the workspace's collective construction; the sites,
        # which need the loaded placement and the early expert ids, follow in
        # install_loaded_features.
        from .kernels.pool_into_route import route as _pool_direct

        pool_direct = _pool_direct.install(model, dense_gemm_plan, mapping=model.layers[0].mapping)
    # finalize_transport replaces only the object every MoE layer calls its fused
    # finalize on, whatever class the MoE has.
    finalize_transport = None
    if plan.enabled("finalize_transport"):
        # Imported only when planned: with no fast path the target imports
        # and runs nothing beyond the built-in modules.
        from .kernels.finalize_transport import transport as _finalize_transport

        finalize_transport = _finalize_transport.install(model)
    if plan.enabled("attention_output_into_hop"):
        # After dense_gemms (its o_proj dispatchers) and post_attention_hop (its joins).
        post_attn_hop.direct = _build_direct_publish(model, post_attn_hop, early_trigger=True)
    return AdoptedLayers(
        finalize_transport=finalize_transport,
        decode_attention=decode_attention,
        dense_gemm_plan=dense_gemm_plan,
        layer_kwargs=layer_kwargs,
        post_attn_hop=post_attn_hop,
        norm_gate_trigger=norm_gate_trigger,
        swa_block=swa_block,
        unpool_direct=unpool_direct,
        pool_direct=pool_direct,
        qkv_prefetch=qkv_prefetch,
    )
