# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA graphs for the one-model MTP draft step of a context-only batch (CTX / prefill worker).

``TRTLLM_Q35_CTX_MTP_GRAPH=1`` (default off). Every forward of a disaggregated CTX worker with MTP ends with one eager
draft step (draft length 1): the target-token sample, the draft-input shift, the MTP layer (embedding, pre-fc norms,
fc + all-reduce, attention, MoE + all-reduce, norms), the draft LM head and the tensor-parallel greedy draft sample.
In eager PyTorch that is ~80 kernel launches behind ~4.6 ms of Python per forward, while the target layers run as
piecewise CUDA graphs. Apart from the attention call the step depends only on the token count, so it is captured the
same way as the target layers: per padded token bucket ``s`` one graph up to the MTP attention op (draft embedding,
norms, fc, input norm, QKV projection, fused QK-norm / RoPE / gate and the attention output buffer) and one graph
after it (output gate, o_proj, post-attention norm, MoE, shared_head norm, the requests' last-token rows, the LM head
and the greedy draft sample over the vocabulary shards). The attention op runs eagerly between the two replays with
the forward's own metadata, as in the target layers (inside ``draft_kv_cache_context``, like the stock draft loop:
with a separate draft KV cache, which non-disaggregated runs use, the MTP attention reads and writes that pool).
Inputs are copied into static buffers (draft token ids, target hidden states, positions, last-token row indices
padded to ``max_num_requests`` rows); a padded row reads a valid row and its draft token is dropped.

The context-only draft step with draft length 1 also skips the attention-metadata save / step-0 -> 1 update /
restore of the stock loop: no second draft pass reads them, and the next forward's prepare() rebuilds every field the
update touches (seq lens, kv lens, request types, number of contexts).

Captures happen only while the model engine runs its piecewise CUDA-graph warmup (``capture_piecewise_cuda_graph``),
for token buckets up to ``TRTLLM_Q35_CTX_MTP_GRAPH_MAX_TOKENS`` (default 4096), after three eager calls of the same
split path, into one private graph pool (buckets never run concurrently, so they share it). Nothing is captured while
serving. A forward that is not a pure context batch with draft length 1 and greedy drafting, or whose bucket has no
graph, runs the stock eager draft step.

Debug: ``TRTLLM_Q35_CTX_MTP_GRAPH_STATS=N`` logs the graph / fallback counters every N draft steps (rank 0);
``TRTLLM_Q35_CTX_MTP_GRAPH_CHECK=1`` runs the graph path and then the stock path on the same inputs (the stock outputs
and MTP KV win), and logs draft-token agreement plus each mismatch's top-2 draft-logit gap (greedy batches only).
"""

import os
from collections import Counter
from typing import Optional

import torch

from tensorrt_llm.logger import logger

from ..compilation.utils import get_capture_piecewise_cuda_graph_flag
from ..utils import get_model_extra_attrs, get_per_request_piecewise_cuda_graph_flag, make_weak_ref

ENV = "TRTLLM_Q35_CTX_MTP_GRAPH"
_WARMUP_CALLS = (
    3  # eager calls of the split path per bucket before its capture (as PiecewiseRunner)
)


def ctx_mtp_graph_enabled() -> bool:
    return os.environ.get(ENV, "0") == "1"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


class _Bucket:
    """Graphs and recorded attention call of one padded token count."""

    def __init__(self):
        self.calls = 0
        self.pre = None  # CUDAGraph up to the attention op
        self.post = None  # CUDAGraph after it, ends with the draft tokens
        self.attn_args = None  # (q, k, v, mask, ...) weak refs into the graph pool
        self.attn_kwargs = None
        self.draft = None  # [rows] int32 draft tokens (weak ref)
        self.logits = None  # [rows, vocab shard] draft logits (weak ref; check mode)


class _AttnSplit:
    """Instance-level stand-in for the MTP attention module's ``_attn_impl`` during a capture: ends the
    pre-attention capture, runs the real attention call (not captured), records its arguments for the replays and
    begins the post-attention capture."""

    def __init__(self, runner: "CtxMtpGraphs", orig):
        self.runner = runner
        self.orig = orig
        self.hit = False
        self.error = None

    def __call__(
        self,
        q,
        k,
        v,
        attn_metadata,
        attention_mask,
        mrope_rotary_cos_sin,
        mrope_position_deltas,
        attention_window_size,
        attention_mask_data,
        output=None,
        output_sf=None,
        **kwargs,
    ):
        r = self.runner
        if (
            self.hit
            or output is None
            or mrope_rotary_cos_sin is not None
            or mrope_position_deltas is not None
            or attention_mask_data is not None
            or kwargs.get("has_lora", False)
        ):
            # Only the custom-op path (output preallocated inside the pre graph) and a call without
            # per-forward side tensors can be replayed; anything else aborts the capture.
            self.error = "unsupported attention call"
            raise RuntimeError(self.error)
        self.hit = True
        r._cur_pre.capture_end()
        r._capturing = None

        def weak(t):
            return make_weak_ref(t) if isinstance(t, torch.Tensor) else t

        r._cur_bucket.attn_args = (weak(q), weak(k), weak(v), attention_mask, attention_window_size)
        r._cur_bucket.attn_kwargs = dict(
            output=weak(output),
            output_sf=weak(output_sf),
            **{key: weak(val) for key, val in kwargs.items()},
        )
        result = self.orig(
            q,
            k,
            v,
            attn_metadata,
            attention_mask,
            None,
            None,
            attention_window_size,
            None,
            output=output,
            output_sf=output_sf,
            **kwargs,
        )
        r._cur_post.capture_begin(pool=r.pool, capture_error_mode="thread_local")
        r._capturing = r._cur_post
        return result


class CtxMtpGraphs:
    """Per-bucket CUDA graphs of the MTP draft step of a context-only batch (one per draft model)."""

    def __init__(self, worker, draft_model, spec_metadata):
        self.max_tokens = _env_int("TRTLLM_Q35_CTX_MTP_GRAPH_MAX_TOKENS", 4096)
        self.stats_every = _env_int("TRTLLM_Q35_CTX_MTP_GRAPH_STATS", 0)
        self.check = os.environ.get("TRTLLM_Q35_CTX_MTP_GRAPH_CHECK", "0") == "1"
        self.rows = max(1, int(getattr(spec_metadata, "max_num_requests", 0) or 1))
        self.mtp = draft_model.mtp_layers[0]
        self.attn = self.mtp.self_attn
        self.buckets = {}
        self.pool = None
        self.stream = None
        self.disabled = None  # reason string once disabled
        self.ids = self.pos = self.hid = self.gid = None
        self.counts = Counter()
        self.steps = 0  # draft steps seen (the TRTLLM_Q35_CTX_MTP_GRAPH_STATS cadence)
        self.check_counts = Counter()
        self.check_gaps = []
        self.rank = getattr(worker.mapping, "rank", 0) if worker.mapping is not None else 0
        self._cur_bucket = self._cur_pre = self._cur_post = None
        self._capturing = None

    # ---- eligibility -------------------------------------------------------------------------------------------

    def ineligible(
        self, worker, attn_metadata, spec_metadata, draft_model, input_ids, hidden_states
    ):
        """Reason the stock eager step has to run for this forward, or None."""
        if self.disabled:
            return "disabled"
        if not worker.is_mtp_eagle:
            return "config"
        num_seqs = attn_metadata.num_seqs
        if attn_metadata.num_contexts != num_seqs or num_seqs <= 0:
            return "not_ctx_only"
        if spec_metadata.runtime_draft_len != 1:
            return "draft_len"
        if num_seqs > self.rows:
            return "rows"
        if worker.guided_decoder is not None or worker.sa_enhancer is not None:
            return "guided_or_sa"
        if worker._uses_external_shared_target_kv or worker._d2t is not None:
            return "draft_model_kind"
        if getattr(worker.spec_config, "use_relaxed_acceptance_for_thinking", False):
            return "relaxed_acceptance"
        if getattr(draft_model, "q35_draft_head", None) is not None:
            return "fp8_draft_head"
        mapping = worker.model_config.mapping if worker.model_config is not None else None
        if mapping is None or mapping.enable_attention_dp:
            return "attention_dp"
        if spec_metadata.wants_advanced_draft_sampling:
            return "sampling"
        if input_ids is None:
            return "no_input_ids"
        if attn_metadata.kv_cache_manager is None:
            return "no_kv_cache"
        n = attn_metadata.num_tokens
        if (
            hidden_states.shape[0] != n
            or input_ids.shape[0] != n
            or attn_metadata.num_ctx_tokens != n
        ):
            return "shape"
        return None

    # ---- static inputs -----------------------------------------------------------------------------------------

    def _alloc(self, positions, hidden_states):
        cap = self.max_tokens
        dev = hidden_states.device
        self.ids = torch.zeros((cap,), dtype=torch.int32, device=dev)
        self.pos = torch.zeros(
            tuple(positions.shape[:-1]) + (cap,), dtype=positions.dtype, device=dev
        )
        self.hid = torch.zeros(
            (cap, hidden_states.shape[-1]), dtype=hidden_states.dtype, device=dev
        )
        self.gid = torch.zeros((self.rows,), dtype=torch.long, device=dev)
        self.pool = torch.cuda.graph_pool_handle()
        self.stream = torch.cuda.Stream()

    def _stage(self, input_ids, positions, hidden_states, target_tokens, gather_ids, n, b):
        """The stock draft inputs of a context batch, in the static buffers: token ids shifted left by one with each
        request's sampled target token at its last position, the target hidden states, the positions, and the
        requests' last-token rows."""
        self.ids[: n - 1].copy_(input_ids[1:n])
        self.ids[gather_ids[:b]] = target_tokens
        self.hid[:n].copy_(hidden_states[:n])
        self.pos[..., :n].copy_(positions[..., :n])
        self.gid[:b].copy_(gather_ids[:b])

    # ---- the step body (eager warm-up, capture) ----------------------------------------------------------------

    def _greedy_draft(self, worker, logits, spec_metadata):
        """SpecWorkerBase.greedy_sample_draft_with_tp_gather without the torch.compile'd combine (same tokens:
        global argmax over the vocabulary shards, lowest index on ties)."""
        mapping = worker.mapping
        if (
            mapping is not None
            and getattr(mapping, "tp_size", 1) > 1
            and worker._draft_logits_are_sharded(logits, spec_metadata)
        ):
            from ..distributed.ops import allgather

            gathered = allgather(worker._get_local_max_and_combined(logits), mapping, dim=-1)
            best = torch.argmax(gathered[..., 1::2], dim=-1, keepdim=True)
            return torch.gather(gathered[..., 0::2], -1, best).squeeze(-1).type(torch.int32)
        return torch.argmax(logits, dim=-1).type(torch.int32)

    def _body(self, worker, s, attn_metadata, spec_metadata, draft_model):
        hidden = self.mtp(
            input_ids=self.ids[:s],
            position_ids=self.pos[..., :s],
            hidden_states=self.hid[:s],
            embed_tokens=draft_model.embed_tokens,
            attn_metadata=attn_metadata,
            all_rank_num_tokens=spec_metadata.all_rank_num_tokens,
            spec_metadata=spec_metadata,
        )
        rows = hidden[torch.clamp(self.gid, max=s - 1)]
        logits = self.mtp.shared_head(rows, draft_model.lm_head, attn_metadata, True)
        return self._greedy_draft(worker, logits, spec_metadata), logits

    def _capture(self, worker, bucket, s, attn_metadata, spec_metadata, draft_model):
        pre, post = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        split = _AttnSplit(self, self.attn._attn_impl)
        self._cur_bucket, self._cur_pre, self._cur_post = bucket, pre, post
        torch.cuda.synchronize()
        main = torch.cuda.current_stream()
        self.stream.wait_stream(main)
        extra_attrs = get_model_extra_attrs()
        self.attn._attn_impl = split  # instance attribute: only the MTP attention module
        failure = None
        try:
            with torch.cuda.stream(self.stream):
                if extra_attrs is not None:
                    extra_attrs["global_stream"] = torch.cuda.current_stream()
                try:
                    pre.capture_begin(pool=self.pool, capture_error_mode="thread_local")
                    self._capturing = pre
                    draft, logits = self._body(worker, s, attn_metadata, spec_metadata, draft_model)
                    if not split.hit:
                        raise RuntimeError("MTP attention op not reached")
                    post.capture_end()
                    self._capturing = None
                except Exception as error:  # noqa: BLE001 (any capture failure: the eager step from now on)
                    failure = f"{type(error).__name__}: {error}"
                    if self._capturing is not None:
                        # End the capture on the stream it began on (still current here).
                        try:
                            self._capturing.capture_end()
                        except Exception:  # noqa: BLE001
                            pass
                        self._capturing = None
        finally:
            del self.attn._attn_impl
            if extra_attrs is not None:
                extra_attrs["global_stream"] = main
            main.wait_stream(self.stream)
            self._cur_bucket = self._cur_pre = self._cur_post = None
        if failure is not None:
            self.disabled = f"capture failed at {s} tokens: {failure}"
            logger.warning(
                f"[q35 ctx mtp graph] rank {self.rank}: {self.disabled}; the eager draft step runs "
                "from now on"
            )
            return False
        bucket.pre, bucket.post = pre, post
        bucket.draft, bucket.logits = make_weak_ref(draft), make_weak_ref(logits)
        return True

    def _replay(self, bucket, attn_metadata):
        bucket.pre.replay()
        q, k, v, mask, window = bucket.attn_args
        self.attn._attn_impl(
            q, k, v, attn_metadata, mask, None, None, window, None, **bucket.attn_kwargs
        )
        bucket.post.replay()
        return bucket.draft

    # ---- entry ------------------------------------------------------------------------------------------------

    def usable(self, s: int, positions: torch.Tensor) -> Optional[str]:
        """None when this forward's bucket ``s`` has graphs or may be warmed up / captured now, else the reason."""
        if self.pos is not None and tuple(positions.shape[:-1]) != tuple(self.pos.shape[:-1]):
            return "positions"
        bucket = self.buckets.get(s)
        if bucket is not None and bucket.post is not None:
            return None
        if (
            get_capture_piecewise_cuda_graph_flag()
            and s <= self.max_tokens
            and get_per_request_piecewise_cuda_graph_flag()
        ):
            return None
        return "no_graph"

    def draft(
        self,
        worker,
        input_ids,
        positions,
        hidden_states,
        target_tokens,
        attn_metadata,
        spec_metadata,
        draft_model,
        draft_kv_cache_manager=None,
    ):
        """Draft tokens [rows] int32 of this context batch (rows >= num_seqs; usable() said yes), or None after a
        failed capture. With a separate draft KV cache (non-disaggregated runs) the MTP attention runs on it, as in
        the stock draft loop (draft_kv_cache_context swaps the manager and block offsets in place)."""
        n = attn_metadata.num_tokens
        b = attn_metadata.num_seqs
        s = attn_metadata.padded_num_tokens or n
        if self.ids is None:
            self._alloc(positions, hidden_states)
        bucket = self.buckets.setdefault(s, _Bucket())
        self._stage(
            input_ids, positions, hidden_states, target_tokens, spec_metadata.gather_ids, n, b
        )
        with worker.draft_kv_cache_context(attn_metadata, draft_kv_cache_manager) as meta:
            return self._draft_in_context(worker, bucket, s, meta, spec_metadata, draft_model)

    def _draft_in_context(self, worker, bucket, s, attn_metadata, spec_metadata, draft_model):
        if bucket.post is not None:
            return self._replay(bucket, attn_metadata)
        # Piecewise CUDA-graph warmup of this bucket: eager calls of the split path, then the capture.
        bucket.calls += 1
        if bucket.calls <= _WARMUP_CALLS:
            self.counts["warmup"] += 1
            return self._body(worker, s, attn_metadata, spec_metadata, draft_model)[0]
        ok = self._capture(worker, bucket, s, attn_metadata, spec_metadata, draft_model)
        # Every rank must take the same path (the graphs pad the collectives to s rows): a capture that failed on
        # any rank turns the graphs off on all of them. Warmup only, so a host all-gather is fine here.
        try:
            from tensorrt_llm._utils import mpi_allgather

            oks = mpi_allgather(ok)
            ok = all(oks) if isinstance(oks, list) else bool(oks)
        except Exception as error:  # noqa: BLE001 (no MPI: this rank's own result)
            logger.warning(f"[q35 ctx mtp graph] capture agreement skipped: {error}")
        if not ok:
            if not self.disabled:
                self.disabled = f"capture failed on another rank at {s} tokens"
                logger.warning(
                    f"[q35 ctx mtp graph] rank {self.rank}: {self.disabled}; the eager draft step runs "
                    "from now on"
                )
            return None
        self.counts["capture"] += 1
        logger.info(
            f"[q35 ctx mtp graph] rank {self.rank}: captured the draft step for {s} tokens "
            f"({sum(1 for v in self.buckets.values() if v.post is not None)} buckets, graph pool "
            f"{self._pool_mib():.0f} MiB)"
        )
        return self._replay(bucket, attn_metadata)

    def _pool_mib(self) -> float:
        """Reserved size of this runner's private graph pool (allocator segments of that pool; warmup only)."""
        try:
            pool = tuple(self.pool)
            return (
                sum(
                    seg["total_size"]
                    for seg in torch.cuda.memory_snapshot()
                    if tuple(seg.get("segment_pool_id", ())) == pool
                )
                / 2**20
            )
        except Exception:  # noqa: BLE001 (diagnostics only)
            return float("nan")

    def note(self, key: str) -> None:
        self.counts[key] += 1
        self.steps += 1
        if self.stats_every and self.rank == 0 and self.steps % self.stats_every == 0:
            logger.info(
                f"[q35 ctx mtp graph] {dict(self.counts)}; buckets {sorted(self.buckets)}"
                + (f"; disabled: {self.disabled}" if self.disabled else "")
            )

    def check_against(self, worker, draft_tokens, ref_next_draft, num_seqs, spec_metadata, s):
        """TRTLLM_Q35_CTX_MTP_GRAPH_CHECK: compare the graph draft tokens with the stock step's (same inputs)."""
        b = num_seqs
        eq = draft_tokens[:b] == ref_next_draft[:b, 0].to(torch.int32)
        bucket = self.buckets.get(s)
        gaps = None
        if bucket is not None and bucket.logits is not None:
            # Top-2 gap of the full-vocabulary draft logits (graph path) per request: all-gather each rank's
            # local top-2 values (collective; every rank checks the same requests).
            top2 = torch.topk(bucket.logits[:b].float(), 2, dim=-1).values
            mapping = worker.mapping
            if (
                mapping is not None
                and getattr(mapping, "tp_size", 1) > 1
                and worker._draft_logits_are_sharded(bucket.logits, spec_metadata)
            ):
                from ..distributed.ops import allgather

                top2 = allgather(top2.contiguous(), mapping, dim=-1)
            top2 = torch.topk(top2, 2, dim=-1).values
            gaps = (top2[:, 0] - top2[:, 1]).tolist()
        eq = eq.tolist()
        self.check_counts["rows"] += b
        self.check_counts["equal"] += sum(1 for e in eq if e)
        if gaps is not None:
            for e, g in zip(eq, gaps):
                if not e:
                    self.check_gaps.append(round(g, 4))
        if (
            self.rank == 0
            and (self.check_counts["rows"] - b) // 50 != self.check_counts["rows"] // 50
        ):
            logger.info(
                f"[q35 ctx mtp graph check] rows {self.check_counts['rows']}, equal draft tokens "
                f"{self.check_counts['equal']}; top-2 gaps of the mismatches (nats of bf16 logits): "
                f"{self.check_gaps[-20:]}"
            )


def q35_ctx_mtp_graph_step(
    worker,
    input_ids,
    position_ids,
    hidden_states,
    logits,
    attn_metadata,
    spec_metadata,
    draft_model,
    resource_manager=None,
):
    """The draft step of a context-only batch on the bucket's CUDA graphs. Returns the worker outputs, or None
    (nothing done: the caller runs the stock step)."""
    runner = draft_model.__dict__.get("_q35_ctx_mtp_graphs")
    if runner is None:
        runner = CtxMtpGraphs(worker, draft_model, spec_metadata)
        draft_model.__dict__["_q35_ctx_mtp_graphs"] = runner
    if getattr(worker, "_q35_ctx_mtp_in_check", False):
        return None
    reason = runner.ineligible(
        worker, attn_metadata, spec_metadata, draft_model, input_ids, hidden_states
    )
    positions = position_ids.squeeze(0)
    s = attn_metadata.padded_num_tokens or attn_metadata.num_tokens
    if reason is None:
        reason = runner.usable(s, positions)
    if reason is not None:
        runner.note("eager:" + reason)
        return None
    raw_logits = logits
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    b = attn_metadata.num_seqs
    # The stock context-request acceptance: the sampled target token, one accepted token per request.
    target_tokens = worker._sample_tokens_for_batch(logits, spec_metadata, b, b)
    accepted_tokens = torch.empty((b, 2), dtype=torch.int, device=logits.device)
    accepted_tokens[:, 0] = target_tokens
    num_accepted_tokens = torch.ones(b, dtype=torch.int, device=logits.device)
    draft_tokens = runner.draft(
        worker,
        input_ids,
        positions,
        hidden_states,
        accepted_tokens[:, 0],
        attn_metadata,
        spec_metadata,
        draft_model,
        worker.get_draft_kv_cache_manager(resource_manager),
    )
    if draft_tokens is None:
        # A failed capture (logged; graphs off from now on): this forward takes the stock step, which samples the
        # target tokens again (greedy: the same tokens).
        runner.note("eager:capture_failed")
        return None
    if runner.check:
        worker._q35_ctx_mtp_in_check = True
        try:
            ref = worker._forward_impl(
                input_ids,
                position_ids,
                hidden_states,
                logits,
                attn_metadata,
                spec_metadata,
                draft_model,
                resource_manager,
            )
        finally:
            worker._q35_ctx_mtp_in_check = False
        runner.check_against(worker, draft_tokens, ref["next_draft_tokens"], b, spec_metadata, s)
        runner.note("checked")
        return ref
    runner.note("graph")
    next_draft_tokens = draft_tokens[:b].unsqueeze(1).clone()
    next_new_tokens = torch.cat((accepted_tokens[:, :1], next_draft_tokens), dim=1)
    attn_metadata.use_spec_decoding = True
    return {
        "logits": raw_logits,
        "new_tokens": accepted_tokens,
        "new_tokens_lens": num_accepted_tokens,
        "next_draft_tokens": next_draft_tokens,
        "next_new_tokens": next_new_tokens,
    }
