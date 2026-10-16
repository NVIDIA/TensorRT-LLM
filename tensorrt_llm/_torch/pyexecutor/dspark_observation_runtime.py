# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit offline observation factory over normal native/automatic execution.

No fixed-cell/profile-only engine is used here. Calibration alone evaluates the
confidence head under native full-K verification and copies observations before
the SAME normal sampler event; benchmark arms add no calibration copies.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from tensorrt_llm.tools.dspark_observations import NativePairs, ProposalSnapshot


class ObservationExecutorMixin:
    """Thin lifecycle joins; every forward, sampler/update and error delegates."""

    def __init__(self, *args, **kwargs):
        self._observation_inputs = {}
        self._observation_states = {}
        self._observation_pending = {}
        self._observation_forward = None
        self._observation_failures = []
        super().__init__(*args, **kwargs)
        self._native_pairs = NativePairs(self.model_engine.spec_config.max_draft_len)

    def _handle_errors(
        self,
        error_msg=None,
        *,
        requests=None,
        charge_budget=True,
        fatal_is_collective_aligned=False,
    ):
        self._observation_failures.append(str(error_msg or "Original executor error"))
        return super()._handle_errors(
            error_msg,
            requests=requests,
            charge_budget=charge_budget,
            fatal_is_collective_aligned=fatal_is_collective_aligned,
        )

    def _forward_step(self, batch, new_tensors_device=None):
        if self.is_warmup or not self.model_engine._collect_calibration:
            return super()._forward_step(batch, new_tensors_device)
        if self._observation_forward is not None:
            raise RuntimeError("Overlapping unpaired observation forwards")
        requests = tuple(r for r in batch.generation_requests if not r.is_dummy)
        source = self._observation_inputs.get(id(new_tensors_device))
        input_host = None
        if source is not None:
            if source.state.device is not new_tensors_device:
                raise RuntimeError("Original device input incarnation changed")
            # Issued before original forward can reuse persistent output storage.
            # The next original sample_async event covers this same-stream copy.
            input_host = self.sampler._copy_to_host(new_tensors_device.next_draft_tokens)
        elif requests:
            self._native_pairs.stats["bootstrap_or_no_bound_input"] += len(requests)
        # Existing source stamp scalar is observation identity only. It does
        # not alter proposal/acceptance math or construct a scheduling policy.
        draft_sequence = self.model_engine.model.spec_worker.bump_draft_seq() + 1
        self._observation_forward = (
            self.iter_counter,
            requests,
            source,
            input_host,
            draft_sequence,
        )
        result = super()._forward_step(batch, new_tensors_device)
        if result is None:
            raise RuntimeError("Original forward failed; observations cannot be promoted")
        return result

    def _sample_async(self, batch, output):
        if self.is_warmup or not self.model_engine._collect_calibration:
            return super()._sample_async(batch, output)
        worker = self.model_engine.model.spec_worker
        confidence = worker.staged_confidence_buffer()
        if confidence is None or self._observation_forward is None:
            raise RuntimeError("Offline full-K collector has no produced confidence/forward")
        iteration, requests, source, input_host, draft_sequence = self._observation_forward
        rows = {}
        for request in requests:
            row = worker.confidence_row_for(request.py_request_id)
            if row is None or request.py_seq_slot is None:
                raise RuntimeError("Original request has no worker confidence/sampler slot")
            rows[request.py_request_id] = (request, int(request.py_seq_slot), int(row))
        # Already-produced full-K logits are copied with the original sampler's
        # helper BEFORE it records its normal event. No extra event or wait.
        host = self.sampler._copy_to_host(confidence)
        stamps = self.sampler._copy_to_host(worker.confidence_stamp_buffer())
        state = super()._sample_async(batch, output)
        if state is None or id(state) in self._observation_pending:
            raise RuntimeError("Missing or reused original sampler state")
        snapshot = ProposalSnapshot(iteration, state, host, rows, stamps, draft_sequence)
        self._observation_inputs[id(state.device)] = snapshot
        self._observation_states[id(state)] = snapshot
        self._observation_pending[id(state)] = (iteration, requests, source, input_host)
        self._observation_forward = None
        return state

    def _update_requests(self, state, resource_manager=None):
        if self.is_warmup or not self.model_engine._collect_calibration:
            return super()._update_requests(state, resource_manager)
        pending = self._observation_pending.get(id(state))
        snapshot = self._observation_states.get(id(state))
        if pending is None or snapshot is None or snapshot.state is not state:
            raise RuntimeError("Unknown or consumed observation sampler state")
        before = {id(request): request.py_decoding_iter for request in state.requests}
        result = super()._update_requests(state, resource_manager)
        # Original SpecSampler.update_requests performed its existing synchronize
        # and updates. Parent errors are fatal to publication, not waived here.
        if self._observation_failures:
            raise RuntimeError("Original sampler/update error; no paired-data publication")
        snapshot.updated = True
        iteration, requests, source, input_host = pending
        by_object = {id(request): index for index, request in enumerate(state.requests)}
        for request in requests:
            index = by_object.get(id(request))
            if index is None:
                raise RuntimeError("Original generation request absent from sampler state")
            if request.py_decoding_iter <= before[id(request)]:
                self._native_pairs.stats["retired_or_not_advanced"] += 1
                continue
            if source is None:
                continue
            if request.py_request_id not in source.requests:
                # A finished-context output has only placeholder draft rows,
                # not a confidence-scored block from a generation forward.
                self._native_pairs.stats["prefill_placeholder_no_scored_proposal"] += 1
                continue
            slot = int(request.py_seq_slot)
            token_window = (
                int(state.host.verify_lens[index])
                if state.host.verify_lens is not None
                else state.runtime_draft_len + 1
            )
            verified = min(int(state.draft_lens[index]), token_window - 1)
            if verified != self._native_pairs.physical_k:
                # First-draft bootstrap has no scored full-K proposal. Never
                # reinterpret its placeholder confidence as calibration data.
                self._native_pairs.stats["bootstrap_or_partial"] += 1
                continue
            self._native_pairs.record(
                source=source,
                request=request,
                target_iteration=iteration,
                input_proposal=input_host[slot],
                target_slot=slot,
                accepted=int(state.host.new_tokens_lens[slot]) - 1,
                verified=verified,
            )
        del self._observation_pending[id(state)]
        return result

    def _observation_drain_predicates(self):
        previous = self.previous_batch
        state = previous.sample_state if previous is not None else None
        snapshot = self._observation_states.get(id(state)) if state is not None else None
        return {
            "failures": len(self._observation_failures),
            "active_requests": len(self.active_requests),
            "waiting_requests": len(self.waiting_queue),
            "pending_samples": len(self._observation_pending),
            "forward_pending": self._observation_forward is not None,
            "previous_batch_present": previous is not None,
            "pending_matches_previous": (
                state is not None
                and len(self._observation_pending) == 1
                and id(state) in self._observation_pending
                and snapshot is not None
                and snapshot.state is state
                and not snapshot.updated
            ),
        }

    def observation_receipt(self, *, reset=False, control=None):
        with self.control_action(drain=True):
            before = self._observation_drain_predicates()
            consumed = False
            if self.model_engine._collect_calibration:
                if control is None:
                    raise RuntimeError("Offline drain needs the original gang-control communicator")
                peers = control.allgather(before)
                quiescent = all(
                    row["failures"] == 0
                    and row["active_requests"] == 0
                    and row["waiting_requests"] == 0
                    and not row["forward_pending"]
                    for row in peers
                )
                no_tail = all(row["pending_samples"] == 0 for row in peers)
                exact_tail = all(
                    row["pending_samples"] == 1 and row["pending_matches_previous"] for row in peers
                )
                if not quiescent or not (no_tail or exact_tail):
                    raise RuntimeError(
                        "Original requests/updates did not naturally drain: "
                        f"gang_predicates={peers!r}"
                    )
                if exact_tail:
                    # Every rank is paused at the original exclusive boundary.
                    # Complete the same existing sampler/update/resource tail
                    # before checking the observer's stronger publication gate.
                    self._consume_previous_batch_for_rebalance()
                    consumed = True
            after = self._observation_drain_predicates()
            if (
                self._observation_failures
                or self.active_requests
                or len(self.waiting_queue)
                or self._observation_pending
                or self._observation_forward is not None
            ):
                raise RuntimeError(
                    "Original requests/updates did not naturally drain: "
                    f"before={before!r}, after={after!r}"
                )
            runner = self.model_engine.cuda_graph_runner
            result = {
                "drain": {
                    "before": before,
                    "after": after,
                    "original_previous_batch_consumed": consumed,
                },
                "routing": dict(runner.observed_routes),
                "graph_keys": list(runner.observed_keys.values()),
                "resolved_confidence_enabled": bool(
                    self.model_engine.spec_config.enable_confidence_scheduling
                ),
                "runtime_confidence_enabled": bool(
                    getattr(self.model_engine, "_dspark_confidence_enabled", False)
                ),
            }
            if reset:
                runner.observed_routes = {"native_replays": 0, "compact_replays": 0}
                runner.observed_keys.clear()
            return result


def create_observation_executor(
    *, llm_args, collect_calibration=False, checkpoint_dir=None, tokenizer=None
):
    """Only this public offline caller gets observation hooks; normal serving is unchanged."""
    from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import (
        CUDAGraphRunner,
        DSparkCUDAGraphRunner,
        KeyType,
    )
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner import DecoderRunner
    from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
    from tensorrt_llm._torch.pyexecutor.py_executor_creator import (
        _RuntimeConstructors,
        create_py_executor,
    )
    from tensorrt_llm._torch.speculative.dspark import DSv4DSparkWorker, resolve_confidence_head
    from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig

    spec = llm_args.speculative_config
    if not isinstance(spec, DSparkDecodingConfig) or not spec.draft_is_embedded_in_target:
        raise ValueError("The DSpark adapter requires embedded chain drafting")
    if collect_calibration and (spec.enable_confidence_scheduling or spec.draft_len_schedule):
        raise ValueError(
            "Offline calibration requires native full-K, not policy-trimmed verification"
        )

    # Retain PR5's ordinary native-OFF versus DSpark-ON graph implementation.
    graph_base = DSparkCUDAGraphRunner if spec.enable_confidence_scheduling else CUDAGraphRunner

    class ObservedGraphs(graph_base):
        def __init__(self, *args, **kwargs):
            self.observed_routes = {"native_replays": 0, "compact_replays": 0}
            self.observed_keys = {}
            super().__init__(*args, **kwargs)

        def replay(self, key, inputs, *, position_id_offsets=None):
            result = super().replay(key, inputs, position_id_offsets=position_id_offsets)
            key = KeyType(*key)  # same accepted tuple/KeyType contract as the original runner
            if key.is_first_draft or key.draft_len != spec.max_draft_len:
                return result
            raw = dict(key._asdict())
            # Existing KeyType carries a SampleType enum; no arbitrary objects
            # or environment values are recorded by this finite graph witness.
            raw["sample_type"] = str(raw["sample_type"])
            budget = getattr(key, "ragged_verify_bucket", None)
            compact = budget is not None and 0 < budget < key.batch_size * (spec.max_draft_len + 1)
            self.observed_routes["compact_replays" if compact else "native_replays"] += 1
            self.observed_keys[repr(key)] = raw
            return result

    class ObservationDecoderRunner(DecoderRunner):
        def _initialize_cuda_graph_runner(self):
            if collect_calibration:
                worker = self.model.spec_worker
                if type(worker) is not DSv4DSparkWorker or worker._win_inited:
                    raise RuntimeError(
                        "Calibrated head must be enabled before original graph/storage setup"
                    )
                if resolve_confidence_head(self.model.draft_model) is None:
                    raise ValueError("Actual drafter has no confidence head")
                worker.return_confidence = True  # head evaluation only, no planner/selection
            return super()._initialize_cuda_graph_runner()

        def _create_cuda_graph_runner(self, config):
            return ObservedGraphs(config)

    class ObservationEngine(PyTorchModelEngine):
        def __init__(self, **kwargs: Any):
            self._collect_calibration = collect_calibration
            super().__init__(**kwargs)

        def _initialize_decoder_runner(self, runner_cls, runner_config=None, **runner_kwargs):
            if runner_cls is not DecoderRunner:
                raise ValueError("Offline DSpark observation requires the original decoder runner")
            return super()._initialize_decoder_runner(
                ObservationDecoderRunner, runner_config, **runner_kwargs
            )

    class ObservationExecutor(ObservationExecutorMixin, PyExecutor):
        pass

    constructors = _RuntimeConstructors(
        model_engine=ObservationEngine,
        executor=ObservationExecutor,
        kv_cache_creator=KvCacheCreator,
    )
    return create_py_executor(
        llm_args=llm_args,
        checkpoint_dir=checkpoint_dir,
        tokenizer=tokenizer,
        runtime_constructors=constructors,
    )


def observation_factory(*, collect_calibration=False):
    return partial(create_observation_executor, collect_calibration=collect_calibration)
