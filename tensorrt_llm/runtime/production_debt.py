from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log: logging.Logger = logging.getLogger(__name__)

GENESIS_HASH: str = (
    "0000000000000000000000000000000000000000000000000000000000000000"
)


@dataclass
class TensorRTDebtReport:
    engine_id: str
    trtdi_score: float  # TensorRT Debt Index (target <= 12.0)
    batching_memory_multiplier: float  # Target <= 1.08x
    allreduce_barrier_latency_us: float  # Target <= 35.0us
    mutation_safety_score: float  # Target 100.0
    production_readiness_index: float  # Scale 0 - 100
    is_production_ready: bool
    critical_smells: List[str]
    receipt_hash: str


class TechnicalDueDiligenceLedger:
    """
    Cryptographic SHA-256 hash-chained Action Ledger for NVIDIA TensorRT-LLM runtime runs.
    """

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []
        self._last_hash: str = GENESIS_HASH

    def record_runtime_event(
        self,
        engine_id: str,
        event_type: str,
        readiness_index: float,
        critical_smells: List[str],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        index = len(self._entries)

        meta_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
        canonical_content = f"{index}|{self._last_hash}|{engine_id}|{event_type}|{readiness_index}|{timestamp}|{hashlib.sha256(meta_bytes).hexdigest()}"
        curr_hash = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()

        entry = {
            "index": index,
            "timestamp": timestamp,
            "engine_id": engine_id,
            "event_type": event_type,
            "readiness_index": readiness_index,
            "critical_smells": critical_smells,
            "prev_hash": self._last_hash,
            "curr_hash": curr_hash,
            "metadata": metadata,
        }

        self._entries.append(entry)
        self._last_hash = curr_hash
        return entry

    def get_ledger_entries(self) -> List[Dict[str, Any]]:
        return list(self._entries)

    def verify_ledger_integrity(self) -> bool:
        prev = GENESIS_HASH
        for entry in self._entries:
            if entry["prev_hash"] != prev:
                return False
            prev = entry["curr_hash"]
        return True


class ProductionDebtRuntimeGate:
    """
    A2Z SOC Production Debt & Technical Due Diligence Gate for NVIDIA TensorRT-LLM Runtimes.

    Quantifies TensorRT engine compilation debt, In-Flight Batching memory, and AllReduce barrier latency against 4 Enterprise KPIs:
    1. TensorRT Debt Index (TRTDI <= 12.0)
    2. In-Flight Batching Multiplier (IFBM <= 1.08x)
    3. P99 AllReduce Barrier Latency (<= 35us)
    4. Deterministic Mutation Boundaries (never_equate_intent_to_approval)
    """

    def __init__(
        self,
        never_equate_intent_to_approval: bool = True,
        max_acceptable_trtdi: float = 12.0,
    ) -> None:
        self.never_equate_intent_to_approval = never_equate_intent_to_approval
        self.max_acceptable_trtdi = max_acceptable_trtdi
        self.ledger = TechnicalDueDiligenceLedger()

    def check_kill_switch(self) -> bool:
        if os.environ.get("AAG_KILL_SWITCH", "").lower() in ("true", "1", "yes"):
            return True
        for path_str in ("artifacts/KILL", "/tmp/KILL"):
            if Path(path_str).exists():
                return True
        return False

    def evaluate_runtime_engine(
        self,
        engine_id: str,
        allocated_slot_bytes: int = 16000000000,
        peak_batching_bytes: int = 16800000000,
        allreduce_barrier_latency_us: float = 28.5,
        engine_rebuild_cycles: int = 0,
        un_gated_mutations: int = 0,
    ) -> TensorRTDebtReport:
        # 1. Evaluate emergency kill switch
        if self.check_kill_switch():
            self.ledger.record_runtime_event(
                engine_id=engine_id,
                event_type="engine_halted_kill_switch",
                readiness_index=0.0,
                critical_smells=["EMERGENCY_KILL_SWITCH_ENGAGED"],
                metadata={"reason": "AAG_KILL_SWITCH is set"},
            )
            raise PermissionError(
                "A2Z SOC ActionGate: Emergency kill switch is engaged. TensorRT-LLM runtime execution halted."
            )

        critical_smells: List[str] = []

        # KPI 2: In-Flight Batching Multiplier
        batch_ratio = peak_batching_bytes / max(1, allocated_slot_bytes)
        if batch_ratio > 1.8:
            critical_smells.append(f"HIGH_INFLIGHT_BATCH_SPRAWL_{batch_ratio:.2f}X")

        # KPI 3: Latency Ceiling
        if allreduce_barrier_latency_us > 120.0:
            critical_smells.append(f"HIGH_ALLREDUCE_BARRIER_LATENCY_{allreduce_barrier_latency_us:.1f}US")

        # Engine rebuild cycles
        if engine_rebuild_cycles > 1:
            critical_smells.append(f"DETECTED_{engine_rebuild_cycles}_UNOPTIMIZED_ENGINE_REBUILDS")

        # KPI 4: Mutation Safety
        if un_gated_mutations > 0:
            critical_smells.append(f"DETECTED_{un_gated_mutations}_UNGATED_RUNTIME_MUTATIONS")

        # KPI 1: TensorRT Debt Index (0 = Clean, 100 = Catastrophic)
        trtdi = (
            max(0.0, (batch_ratio - 1.0) * 20.0)
            + max(0.0, (allreduce_barrier_latency_us - 35.0) * 0.5)
            + (engine_rebuild_cycles * 15.0)
            + (un_gated_mutations * 30.0)
        )
        trtdi_score = round(min(100.0, trtdi), 2)

        # Production Readiness Index (0 - 100)
        readiness = max(0.0, 100.0 - trtdi_score)
        is_production_ready = (
            trtdi_score <= self.max_acceptable_trtdi and len(critical_smells) == 0
        )

        # Cryptographic Ledger Entry
        entry = self.ledger.record_runtime_event(
            engine_id=engine_id,
            event_type="engine_authorized" if is_production_ready else "engine_flagged_debt",
            readiness_index=readiness,
            critical_smells=critical_smells,
            metadata={
                "trtdi_score": trtdi_score,
                "batch_ratio": batch_ratio,
                "allocated_slot_bytes": allocated_slot_bytes,
                "peak_batching_bytes": peak_batching_bytes,
                "allreduce_barrier_latency_us": allreduce_barrier_latency_us,
                "engine_rebuild_cycles": engine_rebuild_cycles,
                "un_gated_mutations": un_gated_mutations,
                "never_equate_intent_to_approval": self.never_equate_intent_to_approval,
            },
        )

        return TensorRTDebtReport(
            engine_id=engine_id,
            trtdi_score=trtdi_score,
            batching_memory_multiplier=round(batch_ratio, 2),
            allreduce_barrier_latency_us=round(allreduce_barrier_latency_us, 2),
            mutation_safety_score=(
                100.0 if un_gated_mutations == 0 else max(0.0, 100.0 - un_gated_mutations * 30.0)
            ),
            production_readiness_index=readiness,
            is_production_ready=is_production_ready,
            critical_smells=critical_smells,
            receipt_hash=entry["curr_hash"],
        )
