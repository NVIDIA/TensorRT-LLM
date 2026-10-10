# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Admitted host-mixed allocation in the existing DSpark policy transaction.

There is no prescribed action, diagnostic pricing mode, new collective, or
decode-table conversion. A missing quote/owner/route keeps mixed steps native.
"""

import math
from dataclasses import dataclass, replace
from fractions import Fraction

from ..speculative.dspark_mixed_evidence import AdmittedMixedCosts, MixedGeometry
from ..speculative.dspark_planner import _canonical_json_sha256
from ..speculative.dspark_schedule import (
    DSparkScheduleConfig,
    compute_survival,
    schedule_verify_lens_topk,
)
from .dspark_mixed_host_capacity import allocate_candidate, prepare_local_curve
from .dspark_mixed_host_snapshot import AuthenticatedCurve, prepare_current_mixed_curve

BODY_BUCKETS = (96, 128, 192, 256, 384, 512)
WIRE_WORDS = 46
_MAGIC = 0x4D485031
_VERSION = 1
_K = 3
_WORLD = 8
_SCALE = 1_000_000


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value: int) -> int:
    _require(type(value) is int and 0 <= value < 2**63, "Invalid mixed geometry integer")
    return value


def _membership(batch) -> tuple:
    def rows(requests):
        return tuple((id(r), r.py_request_id, r.py_seq_slot, bool(r.is_dummy)) for r in requests)

    return rows(batch.context_requests), rows(batch.generation_requests)


def _geometry(batch) -> tuple:
    contexts = tuple(
        (
            id(r),
            r.context_chunk_size,
            r.context_current_position,
            r.py_num_compressed_tokens,
            r.py_max_new_tokens,
            bool(r.is_first_context_chunk),
        )
        for r in batch.context_requests
    )
    generations = tuple(
        (
            id(r),
            r.py_batch_idx,
            len(r.py_draft_tokens),
            r.get_num_tokens(0),
            getattr(r.state, "name", None),
            r.py_orig_prompt_len,
            r.py_max_new_tokens,
        )
        for r in batch.generation_requests
    )
    return contexts, generations, tuple(id(r) for r in batch.context_requests_last_chunk)


def _remaining(request, runner) -> int:
    return max(
        0,
        min(
            _integer(int(request.py_orig_prompt_len)) + _integer(int(request.py_max_new_tokens)),
            runner._config.max_seq_len,
        )
        - _integer(int(request.get_num_tokens(0))),
    )


def _greedy(batch) -> bool:
    from tensorrt_llm.sampling_params import SamplingParams

    from .sampler.sampler_common import _request_get_sampling_params

    return all(
        SamplingParams.params_imply_greedy_decoding(
            temperature=p.temperature,
            top_p=p.top_p,
            top_k=p.top_k,
            min_p=p.min_p,
            top_p_decay=p.top_p_decay,
            use_beam_search=p.use_beam_search,
        )
        for p in (_request_get_sampling_params(r) for r in batch.all_requests())
    )


@dataclass(frozen=True)
class _Ticket:
    batch: object
    membership: tuple
    geometry: tuple
    iteration: int
    rank: int
    eligible: tuple
    curve: AuthenticatedCurve
    vote: tuple[int, ...]


def stage_mixed_ownership(planner, worker, iteration: int, generation_requests) -> None:
    """Bind the existing asynchronous copy, without reading device storage.

    An attempted forward is not a write. At the next decision, the copied
    positive producer stamp must match that attempt and the current incarnation.
    """
    if not planner._snapshot_valid or planner._host_buffer is None:
        planner._mixed_current_snapshot_meta = None
        return
    previous = getattr(planner, "_mixed_current_snapshot_meta", None)
    sequence = int(worker._draft_seq_host)
    attempts = {}
    for request in generation_requests:
        if request.is_dummy:
            continue
        request_id = int(request.py_request_id)
        slot = worker._req_to_slot.get(request_id)
        incarnation = worker.confidence_incarnation_for(request_id)
        if type(slot) is int and type(incarnation) is int and incarnation > 0:
            attempts[request_id] = (request, slot, incarnation)
    producer_attempts = (
        previous.get("generation_attempts", {})
        if isinstance(previous, dict) and previous.get("staging_sequence") == sequence - 1
        else {}
    )
    planner._mixed_current_snapshot_meta = dict(
        buffer=planner._host_buffer,
        event=planner._copy_event,
        owners=dict(worker._req_to_slot),
        staged_iteration=int(iteration),
        staging_sequence=sequence,
        confidence_stamp=planner._host_confidence_stamp,
        generation_attempts=attempts,
        producer_attempts=producer_attempts,
    )


class MixedRuntime:
    """A separately bound optional mixed path; decode admission is untouched."""

    def __init__(self, runner, costs: AdmittedMixedCosts | None) -> None:
        self.runner = runner
        self.costs = costs
        self._ticket: _Ticket | None = None
        identity = (
            None
            if costs is None
            else {
                "live": costs.identity.payload(),
                "costs": costs.binding.costs_sha256,
                "receipt": costs.binding.admission_sha256,
                "protocol": costs.binding.protocol_sha256,
            }
        )
        digest = "0" * 64 if identity is None else _canonical_json_sha256(identity)
        self._identity_words = tuple(int(digest[n : n + 8], 16) for n in range(0, 64, 8))

    def reset_step(self) -> None:
        self._ticket = None
        self.runner._dspark_host_window_step = False
        self.runner.cuda_graph_runner._dspark_host_window_batch = None

    def _route_ready(self, batch, planner) -> bool:
        cfg, mapping = self.runner._config, self.runner.mapping
        spec, breakable = cfg.spec_config, self.runner.breakable_cuda_graph_runner
        return bool(
            self.costs is not None
            and self.runner._dspark_confidence_enabled
            and self.runner._dspark_trims_submitted_tokens
            and cfg.max_draft_len == spec.max_draft_len == spec.block_size == _K
            and spec.decoding_type == "DSpark"
            and spec.draft_is_embedded_in_target
            and not getattr(spec, "use_rejection_sampling", False)
            and mapping.tp_size == _WORLD
            and mapping.pp_size == mapping.cp_size == 1
            and cfg.enable_attention_dp
            and not cfg.torch_compile_enabled
            and not self.runner.use_beam_search
            and not cfg.is_multimodal
            and not getattr(batch, "encoder_requests", ())
            and getattr(cfg.prefill_cuda_graph_backend, "name", None) == "BREAKABLE"
            and tuple(sorted(cfg.prefill_cuda_graph_num_tokens)) == BODY_BUCKETS
            and breakable is not None
            and not breakable.is_capturing
            and not breakable.is_warming_up
            and all(breakable.has_graph(bucket) for bucket in BODY_BUCKETS)
            and planner.max_verify_len == _K
            and planner.cfg.min_verify_len == 1
            and _greedy(batch)
        )

    def local_vote(
        self, batch, worker, planner, iteration: int, rank: int, *, executor_eligible: bool
    ) -> list[int]:
        """Append a fixed suffix to the existing allgather; never wait for a copy."""
        out = [
            _MAGIC,
            _VERSION,
            int(rank),
            int(iteration),
            sum(not r.is_dummy for r in batch.context_requests),
            0,
            *self._identity_words,
            *([0] * 32),
        ]
        _require(len(out) == WIRE_WORDS, "Mixed vote extent differs")
        self._ticket = None
        try:
            _require(
                executor_eligible and self._route_ready(batch, planner), "Unsupported mixed route"
            )
            contexts, generations = list(batch.context_requests), list(batch.generation_requests)
            # First-chunk KV preparation can resolve prefix-cache hits and
            # change context position/chunk size after this policy transaction.
            _require(
                not any(r.is_first_context_chunk for r in contexts),
                "Mixed context geometry is not finalized",
            )
            native_rows = sum(_integer(r.context_chunk_size) for r in contexts)
            native_rows += len(generations) * (_K + 1)
            _require(
                native_rows <= _integer(self.runner._config.max_num_tokens),
                "Mixed native rows exceed the prepared token budget",
            )
            real = [r for r in generations if not r.is_dummy]
            _require(
                all(
                    getattr(r.state, "name", None) not in (None, "GENERATION_COMPLETE")
                    and _remaining(r, self.runner) > 0
                    for r in real
                ),
                "Completed generation",
            )
            meta = getattr(planner, "_mixed_current_snapshot_meta", None)
            owners = meta.get("owners") if isinstance(meta, dict) else None
            prospective = [
                r for r in real if r.py_batch_idx is not None and len(r.py_draft_tokens) >= _K
            ]
            _require(
                len({r.py_request_id for r in prospective}) == len(prospective), "Duplicate owner"
            )
            eligible = [
                r for r in prospective if not isinstance(owners, dict) or r.py_request_id in owners
            ]
            fixed = [r for r in generations if id(r) not in {id(e) for e in eligible}]
            chunks = tuple(_integer(r.context_chunk_size) for r in contexts)
            _require(all(chunks), "Empty context chunk")
            curve = prepare_current_mixed_curve(
                planner=planner,
                request_ids=tuple(int(r.py_request_id) for r in eligible),
                rows=tuple(worker.confidence_row_for(r.py_request_id) for r in eligible),
                iteration=iteration,
                staging_sequence=int(worker._draft_seq_host),
                context_rows=sum(chunks),
                fixed_generation_rows=len(fixed) * (_K + 1),
                body_capacities=BODY_BUCKETS,
                original_compute_survival=compute_survival,
                requests=tuple(eligible),
                incarnations=tuple(
                    worker.confidence_incarnation_for(r.py_request_id) for r in eligible
                ),
            )
            _require(curve is not None, "Current owned snapshot unavailable")
            eligible = [r for r in eligible if r.py_request_id in curve.curve.request_ids]
            fixed = [r for r in generations if id(r) not in {id(e) for e in eligible}]
            useful = curve.curve.survival.clone()
            for row, request in enumerate(eligible):
                useful[row, max(0, _remaining(request, self.runner) - 1) :] = 0
            curve = replace(
                curve,
                curve=prepare_local_curve(
                    survival=useful,
                    request_ids=curve.curve.request_ids,
                    context_rows=sum(chunks),
                    fixed_generation_rows=len(fixed) * (_K + 1),
                    physical_k=_K,
                    min_verify_len=1,
                    body_capacities=BODY_BUCKETS,
                ),
            )
            real_fixed = [r for r in fixed if not r.is_dummy]
            last = {id(r) for r in batch.context_requests_last_chunk if not r.is_dummy}
            _require(
                last <= {id(r) for r in contexts if not r.is_dummy},
                "Final-context ownership differs",
            )
            context_progress = sum(r.py_max_new_tokens > 0 for r in contexts if id(r) in last)
            lower = context_progress + len(real_fixed)
            upper = context_progress + sum(
                min(_K + 1, _remaining(r, self.runner)) for r in real_fixed
            )
            histories = [_integer(int(r.get_num_tokens(0))) for r in real]
            out[14:26] = [
                sum(chunks),
                sum(n * n for n in chunks),
                sum(n * _integer(r.context_current_position) for n, r in zip(chunks, contexts)),
                sum(n * _integer(r.py_num_compressed_tokens) for n, r in zip(chunks, contexts)),
                sum(histories),
                max(histories, default=0),
                len(real_fixed),
                out[4],
                len(eligible),
                len(contexts),
                len(generations),
                len(fixed) * (_K + 1),
            ]
            out[26:28] = [
                math.ceil((curve.curve.native_expected_yield + upper) * _SCALE),
                native_rows,
            ]
            for index, bucket in enumerate(BODY_BUCKETS):
                try:
                    cell = curve.curve.candidate(bucket)
                except ValueError:
                    continue
                out[28 + 3 * index : 31 + 3 * index] = [
                    1,
                    math.floor((cell.expected_yield + lower) * _SCALE),
                    cell.total_real_rows,
                ]
            out[5] = 1
            _require(all(type(n) is int and 0 <= n < 2**63 for n in out), "Vote integer overflow")
            self._ticket = _Ticket(
                batch,
                _membership(batch),
                _geometry(batch),
                iteration,
                rank,
                tuple(eligible),
                curve,
                tuple(out),
            )
        except (
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            RuntimeError,
            IndexError,
            OverflowError,
        ):
            out[5] = 0
            self._ticket = None
            # Invalid optional geometry cannot poison the collective's int64
            # extent. Its real-context count still selects group-native mixed.
            out[14:] = [0] * (WIRE_WORDS - 14)
        return out

    def split_votes(self, payloads, legacy_words: int) -> tuple[list, list]:
        _require(
            len(payloads) == _WORLD and all(len(p) == legacy_words + WIRE_WORDS for p in payloads),
            "Mixed policy suffix shape differs",
        )
        peers = [list(p[legacy_words:]) for p in payloads]
        for rank, row in enumerate(peers):
            _require(
                all(type(n) is int and 0 <= n < 2**63 for n in row)
                and row[:3] == [_MAGIC, _VERSION, rank]
                and row[5] in (0, 1),
                "Mixed policy header differs",
            )
            for index, bucket in enumerate(BODY_BUCKETS):
                feasible, amount, total = row[28 + 3 * index : 31 + 3 * index]
                _require(
                    feasible in (0, 1)
                    and (feasible or amount == total == 0)
                    and (not feasible or total <= bucket and amount <= row[26]),
                    "Mixed capacity vote differs",
                )
        _require(len({r[3] for r in peers}) == 1, "Mixed iteration differs")
        return [list(p[:legacy_words]) for p in payloads], peers

    def apply(self, batch, worker, planner, peers, iteration: int) -> bool:
        """Return False only for all-decode; preserve its original exact policy."""
        if not sum(row[4] for row in peers):
            return False
        _require(all(row[3] == iteration for row in peers), "Mixed publication iteration differs")
        chosen, windows = None, {}
        if self.costs is not None and all(
            row[5] and tuple(row[6:14]) == self._identity_words for row in peers
        ):
            features = tuple(tuple(row[14:26]) for row in peers)
            native_rows = tuple(row[27] for row in peers)
            native_yield = sum(row[26] for row in peers)
            for index, capacity in enumerate(BODY_BUCKETS):
                cells = [row[28 + 3 * index : 31 + 3 * index] for row in peers]
                if not all(cell[0] for cell in cells):
                    continue
                try:
                    geometry = MixedGeometry(
                        features, capacity, tuple(c[2] for c in cells), native_rows
                    )
                except ValueError:
                    continue
                quote = self.costs.lookup(geometry)
                if quote is None:
                    continue
                candidate = Fraction(sum(cell[1] for cell in cells), 1) / Fraction(
                    str(quote.candidate_upper_ms)
                )
                native = Fraction(native_yield, 1) / Fraction(str(quote.native_lower_ms))
                margin = 1 + Fraction(str(self.costs.minimum_predicted_gain))
                if candidate > native and candidate >= native * margin:
                    item = (candidate, -capacity, geometry)
                    if chosen is None or item[:2] > chosen[:2]:
                        chosen = item
        if chosen is not None:
            ticket = self._ticket
            _require(
                ticket is not None
                and ticket.batch is batch
                and ticket.iteration == iteration
                and ticket.membership == _membership(batch)
                and ticket.geometry == _geometry(batch)
                and ticket.vote == tuple(peers[ticket.rank]),
                "Mixed request/geometry ticket changed",
            )
            _require(
                worker._draft_seq_host == ticket.curve.staging_sequence
                and tuple(worker.confidence_row_for(r.py_request_id) for r in ticket.eligible)
                == tuple(row for _, row in ticket.curve.request_slots)
                and tuple(
                    worker.confidence_incarnation_for(r.py_request_id) for r in ticket.eligible
                )
                == ticket.curve.request_incarnations,
                "Mixed snapshot/slot incarnation changed",
            )

            def original_allocate(survival, budget, k, floor):
                return schedule_verify_lens_topk(
                    survival=survival,
                    budget=budget,
                    cfg=DSparkScheduleConfig(k, floor, survival_eps=0.0),
                )

            windows = dict(
                allocate_candidate(ticket.curve.curve, chosen[2].capacity, original_allocate)
            )
            actual = ticket.curve.curve.context_rows + ticket.curve.curve.fixed_generation_rows
            actual += sum(value + 1 for value in windows.values())
            _require(
                actual == chosen[2].actual_rank_rows[ticket.rank], "Mixed allocated rows differ"
            )
        values = [
            _K if r.is_dummy else windows.get(r.py_request_id, _K)
            for r in batch.generation_requests
        ]
        _require(all(type(v) is int and 1 <= v <= _K for v in values), "Mixed window bounds differ")
        for request, value in zip(batch.generation_requests, values):
            request.py_verify_len = value
        graph = self.runner.cuda_graph_runner
        graph.agreed_ragged_bucket = None
        graph.ragged_pad_verify_len = 0
        graph.ragged_zero_real_high_rows = 0
        graph._dspark_host_window_batch = batch
        self.runner._dspark_device_budget = None
        self.runner._dspark_host_window_step = True
        return True
