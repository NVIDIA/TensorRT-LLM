# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prepared MoE provider: one model owner serializes its scratch, no tensor pointers
are rebound at execution, and the side stream joins before the tail."""

from contextlib import contextmanager, suppress

import torch
from cutlass import Int32

from ...shapes import Choice, Path, rows_policy
from ..common.policy import batch_table
from ..oproj.prepared import OprojMailbox
from .experts_runtime import scratch as expert_scratch
from .shared import scratch as shared_scratch
from .tail_rows import TailRowsBox
from .tail_runtime import CAP, experts_push, flat


class MoEScratch:
    def __init__(self, tp, device, batches, owner=None):
        """``owner`` (group, device, TP, rank): with batches above the tuned ``CAP``
        rows, the wide path's boxes at the largest of them (``rows``)."""
        self.tp, self.device, self.batches = tp, torch.device(device), tuple(batches)
        cap, h = max(batches), 6144 // tp
        with torch.accelerator.device_index(self.device.index):
            self.stream = torch.cuda.Stream(device=self.device)
            self.begin, self.shared_done = torch.cuda.Event(), torch.cuda.Event()
            self.gate_done = torch.cuda.Event()  # the split shared expert's front half
            f32, i32, bf16 = torch.float32, torch.int32, torch.bfloat16
            empty = lambda s, t=bf16: torch.empty(s, dtype=t, device=self.device)
            self.latent = empty((cap, 3584))
            self.front: tuple = (empty((cap, 896), f32), empty((cap, 2 * h), f32))
            self.front += (empty((cap, 16), i32), empty((cap, 16)), empty((cap, h)))
            self.front += (empty((cap, 7168)), self.latent)
            self.expanded = empty((cap * 16, 3584))
            # Wide batches: the wide experts' workspace and ceil(cap / 64) plan blocks.
            rows = dict(rows=cap) if cap > CAP else {}
            self.experts = expert_scratch(tp, self.device, **rows)
            # The experts' plan (MXFP8 x), written by the routing kernel at B>8.
            self.plan = self.experts["plan"]
            self.shared = shared_scratch(self.device)  # fused shared-expert counters
        self.views = {}
        for b in batches:
            self.views[b] = (self.latent[:b], self.expanded[: b * 16], self.experts)
        # Wide batches (``choose``): their front box (gemm64's wide front), the
        # experts' local partial and the shared partial (plain rows), and the wide
        # tail's mailboxes (``tail_rows.TailRowsBox``).
        self.rows = None
        if cap > CAP:
            if owner is None:
                raise ValueError(f"MoE batches above {CAP} rows need the TP group")
            self.rows = TailRowsBox(*owner, cap)
            self.front_rows = OprojMailbox(*owner, (1, cap, 3584 + 2 * 896))
            empty = lambda s: torch.empty(s, dtype=torch.bfloat16, device=self.device)
            self.latent_partial, self.shared_partial = (
                empty((cap, 3584)),
                empty((cap, 7168)),
            )


# TP8 B16: the experts start beside the split gate/up half (+1% E2E).
JOIN = {8: "9-16:0", 16: ""}
# Rubin-class TP16 batches whose routing releases the experts early
# (``SharedFront.release_early``; ``front.RELEASE_AFTER_FRONT_WIDE`` picks the release
# once the front kernel completed). M64 (B8 verify; SharedDown on its split SMs): the
# experts' launch and prologue overlap routing, -0.68 / -0.70 us per layer (release
# after the front) in a one-GPU MoE layer proxy (391 / 293 distinct experts),
# outputs bitwise identical.
RELEASE_EARLY_WIDE = "3-8:1,64:1"


class PreparedMoE:
    def __init__(
        self, front, experts, tail, state, *, prefetch=None, layer_idx=None,
        rows=None, module=None, front_rows=None,
    ):  # fmt: skip
        """``rows``: the wide batches' tail (``tail_rows.PreparedTailRows``), fed by
        the stock MoE module's router, latent-down and shared expert (``module``) and,
        with ``WIDE_EXPERTS``, v8moe's wide routed experts on the stock router's
        routes (``experts_runtime.WideExperts``, self-planned), else the stock layer's
        routed experts (``stock_partials``). ``front_rows``: the wide batches' front,
        routing and shared expert (``front_rows.FrontRows``, ``WIDE_FRONT``) in place
        of the stock router, latent-down and shared expert; its routing writes the
        wide experts' plan."""
        self.front, self.experts, self.tail, self.state = front, experts, tail, state
        self.prefetch, self.layer_idx = prefetch, layer_idx
        self.rows, self.module = rows, module
        self.front_rows = front_rows if rows is not None else None
        self.wide_experts = None
        if rows is not None and WIDE_EXPERTS:
            from .experts_runtime import WideExperts

            wide = [b for b in state.batches if b > CAP]
            partials = {b: state.latent_partial for b in wide}  # one base, every call
            planned = self.front_rows is not None  # FrontRows' routing writes the plan
            if self.front_rows is not None:
                # SharedDown's SMs (front_rows.SHARED_DOWN_SMS): the experts' grid
                # leaves them, so the down rows run beside the experts.
                props = torch.cuda.get_device_properties(state.device)
                for b, n in self.front_rows.down_sms.items():
                    sms = props.multi_processor_count
                    experts.grids[b] = min(experts.grid_for(b), sms - n)
            # the wide tail's EXPERTS_PUSH rows: the experts push into its slots
            pushed = {b: rows.experts_box for b in rows.experts_push if b in partials}
            self.wide_experts = WideExperts(
                experts, state.experts, partials, planned=planned, pushed=pushed
            )
            self.route = stock_router(module.experts.router)
        self.small_batch = None  # optional small-batch chain
        # Split batches whose routed experts launch beside the shared expert's
        # gate/up half instead of after it.
        self.join = batch_table(JOIN[state.tp])
        self.sms = torch.cuda.get_device_properties(state.device).multi_processor_count
        for b, sms in front.split.items():  # the split's down CTAs take the SMs left
            experts.grids[b] = min(experts.grid_for(b), self.sms - sms)
        if self.sms > 152 and state.tp == 16:
            # Rubin-class TP16 B3-8: routing releases the experts at entry (their
            # launch and prologue overlap it). The shared expert drains first (B5-8,
            # grid = SMs) or its down GEMM runs on the SMs the 192-CTA grid leaves
            # (B3-4): -1.0 to -2.8 us per layer in a one-GPU layer proxy.
            early = batch_table(RELEASE_EARLY_WIDE)
            front.release_early(b for b in state.batches if early.get(b))
        for b in (b for b in state.batches if b <= CAP):
            _, expanded, buffers = state.views[b]
            publish, planned = tail.publish.get(b, ()), front.planned(b)
            if publish and experts_push(state.tp, b, state.device):
                # The experts push the shared rows; they must launch after the shared
                # expert completed, i.e. join the side stream before (``run``): their
                # grid spans every SM (``experts_push`` keys on it).
                if experts.grid_for(b) < self.sms:
                    raise RuntimeError(f"B{b}: experts push before the shared expert")
                shared = state.front[5][:b]
                publish = (*publish, flat(shared), tail.peers, Int32(CAP))
            twoshot = b in tail.twoshots
            experts.prepare(b, buffers, expanded, publish, planned, twoshot, True)

    def run(self, x, out, *, consume=None):
        state, b = self.state, x.shape[0]
        if b > CAP:
            return self.run_rows(x, out, consume)
        if self.small_batch is not None and b in self.small_batch.batches:
            return self.small_batch.run(out, consume)
        main = torch.cuda.current_stream(state.device)
        # The shared expert runs on the side stream beside the routed experts, or, when
        # the persistent expert grid spans every SM, beside the front only. Split: its
        # gate/up half beside the front, its down rows beside the experts.
        before = self.experts.grid_for(b) >= self.sms
        if self.front.merged_batch(b):
            # One kernel publishes the routing logits and leaves h; the down rows then
            # run beside the experts.
            ids, weights, latent = self.front.route(x)
            state.begin.record(main)
            with torch.cuda.stream(state.stream):
                state.stream.wait_event(state.begin)
                shared, beside = self.front.shared_expert(x, self.tail)
                beside()
                state.shared_done.record(state.stream)
            self.experts.group(ids)
            return self.finish(b, out, latent, ids, weights, shared, consume)
        state.begin.record(main)
        with torch.cuda.stream(state.stream):
            state.stream.wait_event(state.begin)
            shared, beside = self.front.shared_expert(x, self.tail)
            if beside is not None:
                state.gate_done.record(state.stream)
                beside()
            state.shared_done.record(state.stream)
        ids, weights, latent = self.front.route(x)
        self.experts.group(ids)  # pregrouped unit table, beside the shared expert
        if before or (beside is not None and self.join.get(b, 1)):
            main.wait_event(state.shared_done if before else state.gate_done)
        return self.finish(b, out, latent, ids, weights, shared, consume, before)

    def finish(self, b, out, latent, ids, weights, shared, consume, before=False):
        """The routed experts, the next layer's prefetch fork, then the tail once the
        shared expert's side stream is done."""
        main = torch.cuda.current_stream(self.state.device)
        expanded = self.experts.run(latent, ids, weights)
        if self.prefetch is not None:
            self.prefetch.fork("next_attention", self.layer_idx, b)
        if not before:
            main.wait_event(self.state.shared_done)
        return self.tail.run(expanded, weights, shared, out, consume=consume)

    @contextmanager
    def beside(self, work):
        """``work`` on the side stream beside the block, joined after it, also when
        the work or the block raises: a capture left with unjoined side-stream work
        fails with cudaErrorStreamCaptureUnjoined over the error that aborted it.
        While an error unwinds, the record and the join are best effort (an
        invalidated capture refuses every stream op): that error stays the one
        raised. Yields the join, for a block that must wait for the work earlier
        (``PreparedTailRows.run``'s NormRows)."""
        state = self.state
        main = torch.cuda.current_stream(state.device)
        done = lambda: state.shared_done.record(state.stream)  # noqa: E731
        join = lambda: main.wait_event(state.shared_done)  # noqa: E731
        state.begin.record(main)
        try:
            with torch.cuda.stream(state.stream):
                state.stream.wait_event(state.begin)
                try:
                    work()
                except BaseException:
                    with suppress(Exception):
                        done()
                    raise
                done()
            yield join
        except BaseException:
            with suppress(Exception):
                join()
            raise
        join()

    def run_rows(self, x, out, consume):
        """A wide batch: the stock router, latent-down, shared and routed experts up to
        their unreduced partials, then the wide tail into the output ring, which the
        next AttnRes consumes (``consume``). The shared rows are pushed on the side
        stream beside the latent all-reduce: NormRows polls their RS slots, so only
        the join (before the tail returns) orders the push (4 ranks: M128 15.9 ->
        13.7 us per tail, M512 28.8 -> 25.4)."""
        b = x.shape[0]
        if self.wide_experts is not None:
            return self.run_wide_experts(x, out, consume)
        shared, partial = stock_partials(self.module, x)
        with self.beside(lambda: self.rows.push(shared)) as join:
            self.rows.run(partial, b, join)
        if consume(out) is not out:
            raise RuntimeError("The wide tail output must be consumed from its ring")
        return out

    def run_wide_experts(self, x, out, consume):
        """A wide batch on the wide routed experts: the stock shared expert and its
        RS push on the side stream; the stock router + latent-down and the stock
        router's routes (``stock_router``: BF16 weights, as the stock kernel takes
        them when it routes itself) on the main stream; the wide experts' partial
        into the wide tail."""
        b, module = x.shape[0], self.module
        if self.front_rows is not None:
            return self.run_front_rows(x, out, consume)
        with self.beside(lambda: self.rows.push(module.shared_experts(x))) as join:
            latent, logits, _ = module._maybe_overlap_router_and_down_proj(x)
            weights, ids = self.route(logits)
            weights = weights.to(torch.bfloat16)
            partial = self.wide_experts.run(latent.contiguous(), weights, ids)
            self.rows.run(partial, b, join)
        if consume(out) is not out:
            raise RuntimeError("The wide tail output must be consumed from its ring")
        return out

    def run_front_rows(self, x, out, consume):
        """A wide batch on FrontRows and the wide routed experts: the front GEMM and
        routing (ids, weights, latent copy and the experts' plan) on the main stream,
        then the wide experts (planned) and the wide tail; the shared expert's partial
        and its RS push on the side stream (``beside``), forked after routing where
        the front kernel also leaves h (``merged_batch``: the down rows alone), else
        before the front kernel, or after it / after routing (``fork``)."""
        state, b, front = self.state, x.shape[0], self.front_rows

        def shared_expert():
            if b in front.down_sms:  # SharedDown pushes the down rows itself
                front.push_down(x, self.rows.push_args)
            else:
                self.rows.push(front.shared_expert(x, state.shared_partial[:b]))

        merged = front.merged_batch(b)
        if early := front.early.get(b):
            return self.run_shared_early(x, out, consume, early)
        # Unmerged rows fork the whole shared expert before the front kernel, or later
        # (``front_rows.SHARED_FORK_ROWS``: 1 after the front kernel, 2 after routing).
        fork = front.fork.get(b, 0)
        if merged or fork == 2:
            ids, weights, latent = front.route(x)
        elif fork == 1:
            front.front(x)
        with self.beside(shared_expert) as join:
            if fork == 1:
                ids, weights, latent = front.route_only(x)
            elif not merged and not fork:
                ids, weights, latent = front.route(x)
            partial = self.wide_experts.run(latent, weights, ids)
            self.rows.run(partial, b, join)
        if consume(out) is not out:
            raise RuntimeError("The wide tail output must be consumed from its ring")
        return out

    def run_shared_early(self, x, out, consume, early):
        """``run_front_rows`` at ``front_rows.SHARED_EARLY_ROWS``: the shared down rows
        fork once the front kernel completes (h stored), beside routing; the routed
        experts (``early`` 2: after the down rows) and the wide tail on the main
        stream; the shared rows' RS push on the side stream after the experts, joined
        before NormRows as in ``run_front_rows``."""
        state, b, front = self.state, x.shape[0], self.front_rows
        shared = state.shared_partial[:b]
        front.front(x)
        with self.beside(lambda: front.shared_expert(x, shared)) as join:
            ids, weights, latent = front.route_only(x)
            if early == 2:
                join()
            partial = self.wide_experts.run(latent, weights, ids)
        with self.beside(lambda: self.rows.push(shared)) as join:
            self.rows.run(partial, b, join)
        if consume(out) is not out:
            raise RuntimeError("The wide tail output must be consumed from its ring")
        return out


def stock_router(router):
    """The stock router's routes, eager: ``route(logits) -> (FP32 weights, int32
    ids)``, the fused grouped top-k op its ``grouped_topk`` calls for K3's sigmoid +
    bias routing (``ops.grouped_topk``; same kernel and inputs, bitwise its routes).
    ``select_experts`` runs through ``torch.compile``, which guards on its arguments
    and compiles on a miss: the eager warmup calls it from the stock runner only,
    so a wide batch's first call compiled inside the capture and aborted it.
    Raises unless the router takes that op (no EPLB, no padding masks)."""
    from .... import _compat_custom_ops as ops
    from .... import _compat_envs as envs
    from ...._compat import (
        GroupedTopKRouter,
    )

    groups, top_k = router.num_expert_group, router.top_k
    experts = router.global_num_experts
    fused = (
        type(router) is GroupedTopKRouter
        and router.scoring_func == "sigmoid"
        and router.e_score_correction_bias is not None
        and router.eplb_state is None
        and envs.VLLM_USE_FUSED_MOE_GROUPED_TOPK
        and groups <= 32
        and top_k <= 32
        and experts > groups
        and experts % groups == 0
        and not (router.skip_padding and envs.VLLM_MOE_SKIP_PADDING)
    )
    if not fused:
        raise ValueError(f"wide MoE rows: no eager stock routes for {router!r}")
    args = (groups, router.topk_group, top_k, router.renormalize)
    args += (router.routed_scaling_factor, router.e_score_correction_bias, 1)

    def route(logits):
        if router.capture_fn is not None or router._routing_replay_out is not None:
            raise RuntimeError("wide MoE rows: the stock router records its routes")
        weights, ids = ops.grouped_topk(logits, *args)
        if ids.dtype != torch.int32:
            raise RuntimeError(f"wide MoE rows: {ids.dtype} routes")
        return weights, ids

    return route


def stock_partials(module, x):
    """The stock MoE layer up to its tail (seam S4 without the all-reduces): this
    rank's unreduced shared-expert output [M, 7168] and routed latent partial [M,
    3584] (``LatentMoERunner._fused_forward`` before its tail), or, where the runner
    defers the top-k finalize (its tier-0 tail fusion, M <= 128), the experts'
    ``UnfinalizedMoEOutput`` (the wide tail's LatentRows sums the top k)."""
    runner = module.experts
    routed, logits, _ = module._maybe_overlap_router_and_down_proj(x)
    hidden, padded, _ = runner._maybe_pad_hidden_states(x, routed)
    if padded is not None:
        raise RuntimeError("wide MoE rows: the stock runner pads its input")
    if runner.moe_config.should_defer_moe_finalize(x.shape[0]):
        from ...._compat import _unpack

        return _unpack(runner._forward_impl(hidden, logits, x, None))
    config = runner.moe_config
    unpadded = (
        config.hidden_dim_unpadded if runner._quant_method.has_unpadded_output else 0
    )
    name = runner._encode_layer_name()
    shared, fused = runner._forward_entry(hidden, logits, x, None, name, unpadded)
    return shared, fused


def stock_partials_ok(module, rows):
    """Whether ``stock_partials`` can run ``rows`` (the runner's fused path)."""
    runner = getattr(module, "experts", None)
    if runner is None or not hasattr(runner, "_use_fused_path"):
        return False
    return runner._use_fused_path()


def deferred_rows(module, batches):
    """(The ``batches`` whose top-k finalize the stock runner defers, top k)."""
    config = module.experts.moe_config
    rows = tuple(b for b in batches if config.should_defer_moe_finalize(b))
    return rows, config.experts_per_token


# Rubin (compute capability 10.7) TP16 decode shapes above the tuned rows whose MoE
# layer runs the wide tail (``tail_rows``: two-shot latent, RS shared rows, RMS +
# up-projection, output ring at the largest graph) behind the stock router,
# latent-down, shared and routed experts, instead of the whole stock layer (seam S4).
# Keyed per TP on token rows (``shapes.rows_policy``: every q).
WIDE_TAIL = {16: "65-768:1"}
# Wide batches' routed experts: v8moe's wide engines (``experts_runtime.WideExperts``)
# on the stock router's routes, instead of the stock layer's kernel.
WIDE_EXPERTS = True
# Wide batches' front, routing and shared expert: ``front_rows.FrontRows`` (the wide
# experts then read the plan its routing writes), instead of the stock router,
# latent-down and shared expert. 1-GPU chain with the wide experts on real routes, us
# per layer over the experts alone at M128 / 256: stock-free front + routing + shared
# expert +10.4 / +16.0, against ~30 / 33 for the stock front + route + shared in the
# 16-rank model ledger.
WIDE_FRONT = True


def choose(shape, ctx):
    """The MoE layer's specialized providers for a captured decode shape (the
    runtime's family hook), or None: the step runs the stock MoE layer (seam S4) and
    the next AttnRes reads its plain output. A Choice above 64 rows makes the runtime
    prepare these providers at that shape (scratch, front and tail boxes sized from
    ``ctx.max_rows``) and bind the next AttnRes to the tail's ring there. Today: the
    tail ring, the front box and the experts' capacity hold 64 rows; wide shapes on
    Rubin TP16 (``WIDE_TAIL``) run ``PreparedMoE.run_rows`` (the routed experts on
    the wide engines with ``WIDE_EXPERTS``, else the stock layer's)."""
    if shape.rows <= CAP:
        return Choice(Path.GENERIC)
    if ctx.arch != "rubin" or not rows_policy(WIDE_TAIL.get(ctx.tp, ""), shape, 0):
        return None
    # As the runtime selects them (runtime.decode: FrontRows with WIDE_EXPERTS and
    # WIDE_FRONT, else the stock router + latent-down and shared expert).
    front = "FrontRows front" if WIDE_EXPERTS and WIDE_FRONT else "stock front"
    experts = "wide experts" if WIDE_EXPERTS else "stock experts"
    return Choice(
        Path.GENERIC, ("rows", "stock-front"), f"{front}, {experts}, wide tail"
    )
