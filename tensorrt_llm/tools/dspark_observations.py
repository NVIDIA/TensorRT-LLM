# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU joins for offline DSpark observations; no timing or policy decisions."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections import Counter
from typing import Any

import torch


def token_digest(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


@dataclasses.dataclass
class ProposalSnapshot:
    """One genuine forward's proposals and confidence under the same sampler event.

    Host tensors must not be read until the ORIGINAL sampler has updated this
    state. ``device`` ownership identifies which proposal the later forward
    consumed; row order or the latest live confidence never establishes pairing.
    """

    iteration: int
    state: Any
    confidence_host: torch.Tensor
    requests: dict[int, tuple[Any, int, int]]  # original object, sampler slot, confidence row
    stamp_host: torch.Tensor
    draft_sequence: int
    updated: bool = False


class NativePairs:
    """Accept only uncensored, incarnation- and proposal-exact full-K pairs."""

    def __init__(self, physical_k: int) -> None:
        if type(physical_k) is not int or physical_k < 1:
            raise ValueError("Positive physical K required")
        self.physical_k = physical_k
        self.logits: list[torch.Tensor] = []
        self.accepted: list[int] = []
        self.joins: list[dict] = []
        self._pair_keys: set[tuple[int, int, int, int]] = set()
        self.stats: Counter = Counter()

    def record(
        self,
        *,
        source: ProposalSnapshot,
        request: Any,
        target_iteration: int,
        input_proposal: torch.Tensor,
        accepted: int,
        verified: int,
        target_slot: int,
    ) -> None:
        if not source.updated:
            raise RuntimeError("Original source sampler update has not completed")
        if target_iteration <= source.iteration:
            raise ValueError("A proposal must be verified after its producing forward")
        record = source.requests.get(request.py_request_id)
        if record is None or record[0] is not request:
            raise ValueError("Proposal belongs to another request incarnation")
        if type(accepted) is not int or not 0 <= accepted <= self.physical_k:
            raise ValueError("Full-K accepted-prefix count outside physical K")
        if verified != self.physical_k:
            raise ValueError("Compact or partial verification censors calibration labels")
        _, source_slot, confidence_row = record
        if (
            source.stamp_host.device.type != "cpu"
            or int(source.stamp_host[confidence_row]) != source.draft_sequence
        ):
            raise ValueError("Confidence row was not written by the bound producing draft")
        source_tokens = source.state.host.next_draft_tokens[source_slot]
        if (
            source_tokens.device.type != "cpu"
            or input_proposal.device.type != "cpu"
            or source_tokens.numel() != self.physical_k
            or input_proposal.numel() != self.physical_k
            or not torch.equal(source_tokens, input_proposal)
        ):
            raise ValueError("Actual verification input differs from the bound source proposal")
        logits = source.confidence_host[confidence_row]
        if (
            logits.device.type != "cpu"
            or logits.shape != (self.physical_k,)
            or not logits.is_floating_point()
            or not torch.isfinite(logits).all()
        ):
            raise ValueError("Bound confidence must contain finite full-K logits")
        key = (request.py_request_id, id(request), source.iteration, target_iteration)
        if key in self._pair_keys:
            raise ValueError("Repeated calibration pair")
        self._pair_keys.add(key)
        self.logits.append(logits.clone())
        self.accepted.append(accepted)
        self.joins.append(
            {
                "pair_key": list(key),
                "source_slot": source_slot,
                "target_slot": target_slot,
                "confidence_row": confidence_row,
                "proposal_sha256": token_digest(source_tokens.tolist()),
                "accepted_draft_prefix": accepted,
                "verified_drafts": verified,
                "producing_draft_sequence": source.draft_sequence,
            }
        )
        self.stats["recorded"] += 1

    def payload(self) -> dict:
        if not self.logits:
            raise ValueError("No genuine paired full-K observations")
        counts = torch.tensor(self.accepted, dtype=torch.int64).view(-1, 1)
        return {
            "logits": torch.stack(self.logits),
            "prefix_mask": (torch.arange(self.physical_k).view(1, -1) < counts).float(),
            "meta": {
                "pairing": "draft_seq_ring",
                "pairing_protocol": "original-input-state-object-and-proposal-v1",
                "block_size": self.physical_k,
                "stats": dict(self.stats),
                "joins": self.joins,
                "native_full_k": True,
                "task_accuracy_qualified": False,
            },
        }


def validate_responses(rows: list[dict], inputs: list[dict], output_tokens: int) -> None:
    if len(rows) != len(inputs) or len({row["input_id"] for row in rows}) != len(rows):
        raise ValueError("Missing, duplicated or additional original response")
    expected = {row["input_id"]: row["prompt_token_ids"] for row in inputs}
    for row in rows:
        tokens = row["output_token_ids"]
        if (
            row["input_id"] not in expected
            or row["prompt_token_ids"] != expected[row["input_id"]]
            or row["finish_reason"] != "length"
            or len(tokens) != output_tokens
            or any(type(token) is not int or token < 0 for token in tokens)
        ):
            raise ValueError("Original prompt, output cardinality or token contract failed")


def compare_runs(records: list[dict]) -> dict:
    """Matched fixed-cohort E2E comparison, not task accuracy or a best-K claim."""
    arms: dict[str, list[dict]] = {"static": [], "dynamic": []}
    contract = None
    for record in records:
        if (
            record.get("schema") != "dspark-public-run-v1"
            or not record.get("run_complete")
            or not record.get("model_shutdown_completed")
        ):
            raise ValueError("Only naturally completed, closed public runs can be compared")
        mode = record["mode"]
        if mode not in arms or record.get("calibration_collection"):
            raise ValueError("Calibration observation is not a serving benchmark arm")
        match = record["comparison_contract"]
        if contract is None:
            contract = match
        if match != contract:
            raise ValueError("Inputs, physical K, BS or numerical runtime differ")
        if bool(record["resolved_confidence_enabled"]) != (mode == "dynamic") or bool(
            record["runtime_confidence_enabled"]
        ) != (mode == "dynamic"):
            raise ValueError("Requested dynamic execution fell back to static")
        routes = record["routing"]
        if not routes["native_replays"] and not routes["compact_replays"]:
            raise ValueError("No real generation graph replay")
        if mode == "static" and routes["compact_replays"]:
            raise ValueError("Static arm unexpectedly used compact verification")
        if mode == "dynamic" and routes["compact_replays"] <= 0:
            raise ValueError("Enabled policy never executed compact; no dynamic gain claim")
        value = record["elapsed_seconds"]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("Positive real E2E elapsed time required")
        validate_responses(record["responses"], record["inputs"], contract["output_tokens"])
        arms[mode].append(record)
    if not all(arms.values()) or len(arms["static"]) != len(arms["dynamic"]):
        raise ValueError("Equal numbers of matched static and dynamic lifetimes required")
    token_work = contract["batch_size"] * contract["output_tokens"]
    prices = {
        arm: token_work * len(runs) / sum(run["elapsed_seconds"] for run in runs)
        for arm, runs in arms.items()
    }
    native_outputs = {
        row["input_id"]: row["output_token_ids"] for row in arms["static"][0]["responses"]
    }
    equal = all(
        {row["input_id"]: row["output_token_ids"] for row in run["responses"]} == native_outputs
        for runs in arms.values()
        for run in runs
    )
    return {
        "schema": "dspark-public-comparison-v1",
        "comparison_contract": contract,
        "timing_scope": "fixed-cohort-request-enqueue-to-final-response-including-prefill-and-drain",
        "tps": prices,
        "dynamic_over_static_pct": 100 * (prices["dynamic"] / prices["static"] - 1),
        "exact_output_tokens_equal": equal,
        "task_accuracy_qualified": False,
        "default_production_gain_qualified": False,
        "confidence_interval": None,
        "best_static_or_dynamic_k_qualified": False,
    }
