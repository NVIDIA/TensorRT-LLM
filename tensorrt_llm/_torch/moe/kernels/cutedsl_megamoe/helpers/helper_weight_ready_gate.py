"""Shared Blackwell/Rubin kernel gate for dynamically installed helper weights.

The gate is deliberately optional at code generation time.  A specialization
with ``S == 0`` constructs no gate and therefore carries neither of the two
runtime values.  A helper-bearing specialization derives ``H = M - S`` and
waits once per scheduler CTA, immediately before its first physical helper
work tile is published.

``terminal_flags`` is the local ``uint64[EP]`` view published by the weight
transport.  Bit 63 is reserved.  A source is terminal-ready when that bit is
clear and its low-63-bit generation is greater than or equal to the expected
generation.  The expected generation is a CPU-captured launch-time scalar, not
device READY storage.  After READY publication, callers keep the weights
immutable until the MegaMoE launch completes.
"""

import dataclasses
from enum import IntEnum
from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import AddressSpace, Pointer
from cutlass.cutlass_dsl import Int32, extract_mlir_values, new_from_mlir_values

from .ptx_helpers import nanosleep, read_globaltimer_ns


HELPER_WEIGHT_READY_TERMINAL_ERROR_BIT = 1 << 63
HELPER_WEIGHT_READY_GENERATION_MASK = HELPER_WEIGHT_READY_TERMINAL_ERROR_BIT - 1
HELPER_WEIGHT_READY_POLL_SLEEP_NS = 100
# Elapsed time after the first unsuccessful READY sweep, independent of SM clocks.
# This is deliberately compile-time policy, not another production ABI knob.
HELPER_WEIGHT_READY_TIMEOUT_NS = 120_000_000_000


class HelperWeightReadyStatus(IntEnum):
    """Warp-uniform result of one terminal-flags sweep."""

    WAITING = 0
    READY = 1
    TERMINAL_ERROR = 2
    INVALID_EXPECTED_GENERATION = 3
    TIMEOUT = 4


@dataclasses.dataclass(frozen=True)
class HelperWeightReadyTopology:
    """Compile-time ``[H home slots][S helper slots]`` topology."""

    memory_slot_count: int
    helper_count: int
    home_expert_count: int
    gate_required: bool

def derive_helper_weight_ready_topology(
    memory_slot_count: int,
    helper_count: int,
) -> HelperWeightReadyTopology:
    """Validate static ``M``/``S`` and derive the only helper boundary ``H``."""

    if type(memory_slot_count) is not int or memory_slot_count <= 0:
        raise ValueError("memory_slot_count (M) must be a positive exact int")
    if type(helper_count) is not int or helper_count < 0:
        raise ValueError("helper_count (S) must be a non-negative exact int")
    if helper_count >= memory_slot_count:
        raise ValueError("helper_count (S) must leave at least one home slot")

    home_expert_count = memory_slot_count - helper_count
    return HelperWeightReadyTopology(
        memory_slot_count=memory_slot_count,
        helper_count=helper_count,
        home_expert_count=home_expert_count,
        gate_required=helper_count > 0,
    )


def _validate_helper_weight_ready_binding(
    *,
    topology: HelperWeightReadyTopology,
    source_count: Optional[int],
    terminal_flags: Optional[Pointer],
    expected_generation: Optional[object],
) -> None:
    """Enforce exact runtime-bundle presence from the compile-time topology."""

    if not topology.gate_required:
        if terminal_flags is not None or expected_generation is not None:
            raise ValueError("S=0 must not carry READY runtime ABI values")
        return

    if type(source_count) is not int or source_count <= 0:
        raise ValueError("source_count (EP) must be a positive exact int when S>0")
    if terminal_flags is None or expected_generation is None:
        raise ValueError("S>0 requires terminal_flags and expected_generation")
    if isinstance(expected_generation, int):
        if type(expected_generation) is not int or not (
            1 <= expected_generation <= HELPER_WEIGHT_READY_GENERATION_MASK
        ):
            raise ValueError("expected_generation must be in [1, 2**63)")


@cute.jit
def _helper_weight_ready_fail_closed_trap() -> None:
    """Prevent publication after a terminal, invalid, or timed-out wait."""

    llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="trap;",
        constraints="",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class HelperWeightReadyGenerationGate:
    """One scheduler CTA's wait-all gate with an explicit caller-owned READY latch."""

    def __init__(
        self,
        *,
        topology: HelperWeightReadyTopology,
        source_count: int,
        terminal_flags: Pointer,
        expected_generation,
    ) -> None:
        _validate_helper_weight_ready_binding(
            topology=topology,
            source_count=source_count,
            terminal_flags=terminal_flags,
            expected_generation=expected_generation,
        )
        if not topology.gate_required:
            raise ValueError("construct no HelperWeightReadyGenerationGate when S=0")

        self.topology = topology
        self.source_count = source_count
        self.terminal_flags = terminal_flags
        self.expected_generation = expected_generation

    def __extract_mlir_values__(self) -> list:
        values = []
        for field in (
            self.terminal_flags,
            self.expected_generation,
        ):
            values.extend(extract_mlir_values(field))
        return values

    def __new_from_mlir_values__(self, values: list) -> "HelperWeightReadyGenerationGate":
        value_index = 0

        def rebuild(field):
            nonlocal value_index
            field_value_count = len(extract_mlir_values(field))
            result = new_from_mlir_values(
                field,
                values[value_index : value_index + field_value_count],
            )
            value_index += field_value_count
            return result

        result = type(self)(
            topology=self.topology,
            source_count=self.source_count,
            terminal_flags=rebuild(self.terminal_flags),
            expected_generation=rebuild(self.expected_generation),
        )
        if value_index != len(values):
            raise ValueError(
                "HelperWeightReadyGenerationGate MLIR value count mismatch: "
                f"consumed {value_index}, got {len(values)}."
            )
        return result

    @cute.jit
    def _typed_terminal_flags(self):
        return cute.make_ptr(
            cutlass.Uint64,
            self.terminal_flags.toint(),
            AddressSpace.gmem,
            assumed_align=8,
        )

    @cute.jit
    def _sweep_terminal_flags(self):
        """Read every source cell; acquire all releases before admitting READY."""

        # The launch generation is a uniform scalar.  This terminal pointer is
        # the only global-memory READY authority loaded by the spin loop.
        flags = self._typed_terminal_flags()
        all_sources_geq = Int32(1)
        terminal_error_seen = Int32(0)
        generation_mask = cutlass.Uint64(HELPER_WEIGHT_READY_GENERATION_MASK)

        for source_rank in cutlass.range_constexpr(self.source_count):
            observed = cute.arch.load(
                flags + Int32(source_rank),
                cutlass.Uint64,
                sem="relaxed",
                scope="sys",
            )
            observed_generation = observed & generation_mask
            if observed > generation_mask:
                terminal_error_seen = Int32(1)
            if observed_generation < self.expected_generation:
                all_sources_geq = Int32(0)

        status = Int32(HelperWeightReadyStatus.WAITING)
        if terminal_error_seen != Int32(0):
            status = Int32(HelperWeightReadyStatus.TERMINAL_ERROR)
        elif all_sources_geq != Int32(0):
            # Each system-scope strong read plus this acquire fence forms an
            # acquire pattern for its source's release (PTX ISA, section 8.8).
            llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string="fence.acquire.sys;",
                constraints="~{memory}",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
            status = Int32(HelperWeightReadyStatus.READY)
        return status

    @cute.jit
    def _wait_all(self):
        """Lane 0 polls; fence the READY acquire before returning success."""

        status = Int32(HelperWeightReadyStatus.WAITING)
        if cute.arch.lane_idx() == Int32(0):
            generation_mask = cutlass.Uint64(HELPER_WEIGHT_READY_GENERATION_MASK)
            if (self.expected_generation == cutlass.Uint64(0)) | (
                self.expected_generation > generation_mask
            ):
                status = Int32(HelperWeightReadyStatus.INVALID_EXPECTED_GENERATION)
            else:
                status = self._sweep_terminal_flags()
                if status == Int32(HelperWeightReadyStatus.WAITING):
                    # Already-ready weights take no timer reads. Unsigned elapsed
                    # time also handles wraparound of the 64-bit nanosecond timer.
                    wait_started_ns = read_globaltimer_ns()
                    while status == Int32(HelperWeightReadyStatus.WAITING):
                        elapsed_ns = read_globaltimer_ns() - wait_started_ns
                        if elapsed_ns >= cutlass.Uint64(HELPER_WEIGHT_READY_TIMEOUT_NS):
                            status = Int32(HelperWeightReadyStatus.TIMEOUT)
                        else:
                            nanosleep(HELPER_WEIGHT_READY_POLL_SLEEP_NS)
                            status = self._sweep_terminal_flags()

        status = Int32(cute.arch.shuffle_sync(status, offset=0))
        if status == Int32(HelperWeightReadyStatus.READY):
            if cute.arch.lane_idx() == Int32(0):
                cute.arch.fence_proxy("async.global")
        return status

    @cute.jit
    def wait_before_first_helper_publish(self, expert_idx, ready_latched: Int32) -> Int32:
        """Return the next CTA latch after admitting physical ``m in [H,M)``.

        The scheduler must carry this scalar through its work loop. Mutating a
        gate nested in a frozen extension does not produce a CuTe loop result.
        """

        needs_admission = (
            (expert_idx >= Int32(self.topology.home_expert_count))
            & (expert_idx < Int32(self.topology.memory_slot_count))
            & (ready_latched == Int32(0))
        )
        if needs_admission:
            status = self._wait_all()
            if status != Int32(HelperWeightReadyStatus.READY):
                _helper_weight_ready_fail_closed_trap()
            ready_latched = Int32(1)
        return ready_latched


def make_helper_weight_ready_generation_gate(
    *,
    memory_slot_count: int,
    helper_count: int,
    source_count: Optional[int] = None,
    terminal_flags: Optional[Pointer] = None,
    expected_generation: Optional[object] = None,
) -> Optional[HelperWeightReadyGenerationGate]:
    """Create the mandatory ``S>0`` gate or the ABI-free ``S==0`` sentinel."""

    topology = derive_helper_weight_ready_topology(memory_slot_count, helper_count)
    _validate_helper_weight_ready_binding(
        topology=topology,
        source_count=source_count,
        terminal_flags=terminal_flags,
        expected_generation=expected_generation,
    )
    if not topology.gate_required:
        return None
    return HelperWeightReadyGenerationGate(
        topology=topology,
        source_count=source_count,
        terminal_flags=terminal_flags,
        expected_generation=expected_generation,
    )


__all__ = [
    "HELPER_WEIGHT_READY_GENERATION_MASK",
    "HELPER_WEIGHT_READY_TIMEOUT_NS",
    "HELPER_WEIGHT_READY_TERMINAL_ERROR_BIT",
    "HelperWeightReadyGenerationGate",
    "HelperWeightReadyStatus",
    "HelperWeightReadyTopology",
    "derive_helper_weight_ready_topology",
    "make_helper_weight_ready_generation_gate",
]
