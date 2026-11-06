# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Python mirror of ``rs_common.cuh``: the q record layout, kinds, flags, diagnostics indices and RNG constants; and
the forced-acceptance policy every RS reader shares.

The constants have no logic: the host code of the RS kernels (record views, the diagnostics read-out, test
references) uses these names, and the rs_sampler unit test checks every value against the header compiled by nvcc.

Forced acceptance (``TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS`` set, read through the speculative interface's
parser) on a deployment the policy applies to (``policy_applies``: sm107 TP8, the composition the
``rejection_sampler`` fast path is built for, with that lever not turned off): rejection sampling is inert.
``apply_forced_policy_to_spec_config`` (``routing.route``, when the model loader resolves this target) switches
``use_rejection_sampling`` off on the engine's speculative config, the object the model engine, its runner, the sampler
and the spec metadata hold, before the loader deep-copies the model config for the model; so the steps run the argmax
drafts and the strict verify. ``apply_forced_policy`` (first statement of ``drafter.adopt_drafter``) applies it to the
model's copy when the model was built without that route. The ``rejection_sampler`` install reads the flag after the
policy ran, and two guards (``check_forced_inert_metadata``, ``check_forced_inert_step``) raise if a step still sees the
flag on. Elsewhere the policy changes nothing: the engine's own rejection sampling runs and the forced count applies
after it.
"""

import dataclasses
from enum import IntEnum, IntFlag
from pathlib import Path
from typing import Any, Mapping, Optional

from tensorrt_llm.logger import logger

#: Directory of ``rs_common.cuh`` (add it to ``extra_include_paths`` of every library that includes the header).
INCLUDE_DIR = Path(__file__).resolve().parent
HEADER = INCLUDE_DIR / "rs_common.cuh"
#: Flags of every library that includes the header: no fast math (``logf`` / ``log1pf`` must be the accurate ones).
NVCC_FLAGS = ("-O3", "-lineinfo")

# ---- the q record ----------------------------------------------------------------------------------------------------

#: Draft positions per record (K = 7), candidate lanes per position, record size in bytes.
POS = 7
LANES = 16
RECORD_BYTES = 1024
RECORD_ALIGN = 16
#: ``valid`` after a writer (one bit per position).
VALID_ALL = (1 << POS) - 1

#: Byte offsets of the record fields; ``ids`` int32 [POS][LANES], ``qf`` fp32 [POS][LANES], ``draft`` int32 [POS],
#: ``valid`` uint32, ``kind`` / ``mode`` uint8, ``pos_flags`` uint8 [POS], ``pad0`` uint8, ``reserved`` uint32 [21].
OFFSET_IDS = 0
OFFSET_QF = 448
OFFSET_DRAFT = 896
OFFSET_VALID = 924
OFFSET_KIND = 928
OFFSET_MODE = 929
OFFSET_POS_FLAGS = 930
OFFSET_PAD0 = 937
OFFSET_RESERVED = 940
RESERVED_WORDS = 21


class Kind(IntEnum):
    """``QRecord.kind``; SAMPLED, ONEHOT0 and FROM_DENSE are the usable kinds."""

    NONE = 0
    SAMPLED = 1
    ONEHOT0 = 2
    FROM_DENSE = 3


USABLE_KINDS = (Kind.SAMPLED, Kind.ONEHOT0, Kind.FROM_DENSE)


class PosFlag(IntFlag):
    """``QRecord.posFlags[t]`` bits."""

    POISONED = 1
    Q_INVALID = 2
    MASS_OFF = 4
    FILTERED = 8


class Transform(IntEnum):
    """The walk's q transform (``rs.KNOBS.q_transform``: ``unary``), ``QRecord.mode`` bits 1..2.

    RAW: q is the softmax of the walk score R = unary + edge; SCORE: of f(R); UNARY: the walk itself runs on
    f(unary) + edge, the draft's own sampling law (f(x) = cap tanh(x m / cap), or x m when cap = 0). With RAW or SCORE
    the walk's greedy path is the untransformed walk's; with UNARY it is the greedy path of f(unary) + edge.
    """

    RAW = 0
    SCORE = 1
    UNARY = 2


class FilterMode(IntEnum):
    """Filtered rows of a request with top-k / top-p / min-p (``rs.KNOBS.filter``: ``exact``).

    EXACT: exact filtered rejection sampling; STRICT: the strict body (acceptance against the filtered target sample).
    """

    EXACT = 0
    STRICT = 1


#: The worker's forced-acceptance setting (``speculative.interface.FORCE_NUM_ACCEPTED_TOKENS_ENV_VAR``), named in
#: messages; its value is read through ``speculative.interface.get_force_num_accepted_tokens_float``.
FORCE_ENV = "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS"

#: ``QRecord.mode`` bit 0: the chain follows the sampled lane (the default chain); bits 1..2: ``Transform``.
MODE_CHAIN_SAMPLED = 1
MODE_TRANSFORM_SHIFT = 1

# ---- diagnostics (``rs.diag``, int64 [DIAG_WORDS]) -------------------------------------------------------------------

DIAG_WORDS = 48


class Diag(IntEnum):
    """Index of each ``rs.diag`` counter.

    FILT_OWNER_ROWS, FILT_LIST_ENTRIES, FILT_LIST_MAX (a maximum) and FILT_SCREEN_FALLBACK count on the rank that owns
    the filtered row's list; every other counter is equal on all ranks.
    """

    RS_ROWS = 0
    ACCEPTS = 1
    REJECTS = 2
    STRICT_KIND = 3
    STRICT_DRAFTS = 4
    STRICT_KD = 5
    STRICT_DUP = 6
    STRICT_QZERO = 7
    STRICT_POISON = 8
    STRICT_QDIV = 9
    STRICT_QINVALID = 10
    FILTERED_ROWS = 11
    DRAFT_POISONED = 12
    DENSIFY_ROWS = 13
    SPARSIFY_SLOTS = 14
    MASS_OFF = 15
    ONEHOT_WRITES = 16
    SLOT_OOB = 17
    DUMMY_ROWS = 18
    WALK_RS_CALLS = 19
    RS_DRAW_CALLS = 20
    DENSIFY_CALLS = 21
    SPARSIFY_NONE = 22
    CONSUMED_UNUSED = 23
    WALK_DUP_IDS = 24
    TRAP_RANK_HINT = 29
    TRAP_CODE = 30
    TRAP_DETAIL = 31
    FILT_RS_ROWS = 32
    FILT_ACCEPTS = 33
    FILT_REJECTS = 34
    FILT_D_UNKEPT = 35
    FILT_PLAIN_ROWS = 36
    FILT_OWNER_ROWS = 37
    FILT_LIST_ENTRIES = 38
    FILT_LIST_MAX = 39
    FILT_SCREEN_FALLBACK = 40
    FILT_RESID_FALLBACK = 41
    WALK_FILTERED_POSITIONS = 42
    WALK_FILTER_REMOVED_LANES = 43
    FILT_STRICT_LAST_RESORT = 44
    FILT_STRICT_ROWS = 45


class TrapCode(IntEnum):
    """Value of ``diag[Diag.TRAP_CODE]`` before a trap."""

    CLAIMS = 1
    DIGEST_D = 2
    CHECK_STRICT = 3
    CHECK_REREAD = 4
    CHECK_SLOT = 5
    CHECK_DUP_IDS = 6
    FILTER_PARAMS = 7


# ---- fabric words, hash, RNG -----------------------------------------------------------------------------------------

#: Candidate-slot claim word: CLAIM_OWNER | id from the owner, the id from a non-owner, CLAIM_EMPTY for id < 0.
CLAIM_OWNER = 0x80000000
CLAIM_EMPTY = 0x7FFFFFFF
#: H(w_0 .. w_{n-1}) starts from HASH_INIT and folds h = fmix32(h ^ w) (murmur3 finalizer).
HASH_INIT = 0x9E3779B9

#: Bit 63 marks the RS Philox subsequences; row j of request b uses off[8b + j] = B + ROW_STRIDE j.
SUBSEQ_FLAG = 1 << 63
ROW_STRIDE = 32
#: Draft position t: SUBSEQ_FLAG | (off[8b] + ROW_STRIDE (DRAFT_SLOT0 + t)), word .x of counter 0.
DRAFT_SLOT0 = 8
#: Race noise of row j: SUBSEQ_FLAG | (off[8b + j] + RACE_SUB), counters 2v / 2v + 1 for vector v = id >> 2.
RACE_SUB = 1
#: Accept test of row j: SUBSEQ_FLAG | (off[8b + j] + ACCEPT_SUB), word .x of counter 0.
ACCEPT_SUB = 2
#: Cap of an exponential variate.
EXPO_MAX = 23.0
#: Philox4x32-10 multipliers and Weyl increments (curand's generator).
PHILOX_M0 = 0xD2511F53
PHILOX_M1 = 0xCD9E8D57
PHILOX_W0 = 0x9E3779B9
PHILOX_W1 = 0xBB67AE85

# ---- forced acceptance -----------------------------------------------------------------------------------------------


@dataclasses.dataclass
class ForcedInert:
    """What the forced-acceptance policy switched off in this process.

    Attributes:
        force: the forced acceptance count (``get_force_num_accepted_tokens_float``).
        spec_config: the speculative config whose ``use_rejection_sampling`` it set False.
        step_checked: whether ``check_forced_inert_step`` has passed a forward with spec metadata.
    """

    force: float
    spec_config: Any
    step_checked: bool = False


#: This process's forced-inert record (set by ``apply_forced_policy_to_spec_config`` / ``apply_forced_policy``); None:
#: rejection sampling was not switched off.
FORCED_INERT: Optional[ForcedInert] = None


def _forced_count() -> float:
    """The forced acceptance count as the speculative worker parses it (0.0: no forced acceptance)."""
    from tensorrt_llm._torch.speculative.interface import get_force_num_accepted_tokens_float

    return get_force_num_accepted_tokens_float()


def forced_mode() -> str:
    """``"off"`` without forced acceptance (the speculative worker's own parser reads 0), else ``"inert"``:
    rejection sampling is inert under a forced acceptance count."""
    return "off" if _forced_count() == 0.0 else "inert"


#: The architecture (``10 * major + minor``) and tensor-parallel size of the one composition the policy applies to: the
#: one the ``rejection_sampler`` fast path is built for (``dispatch.SM107_TEP8_ONLY``).
POLICY_SM = 107
POLICY_TP = 8


def policy_applies(sm: int, tp: int, snapshot: Optional[Mapping[str, str]] = None) -> bool:
    """Whether the forced-acceptance policy acts on a deployment of architecture ``sm`` (``10 * major + minor``) at
    tensor-parallel size ``tp``.

    True only at ``POLICY_SM`` / ``POLICY_TP`` with the ``rejection_sampler`` lever not turned off: its knob is not
    ``off``, and an unset knob under ``MUSESPARK_OPEN_LEVER__DEFAULT=baseline`` is off (the resolver's own order;
    ``levers.environment``, or ``snapshot`` in its place). Elsewhere a forced acceptance count leaves the engine's own
    rejection sampling running and forces the count after it. The engine facts (``max_batch_size``, the server's role)
    are not known where the route calls this, so a forced sm107 TP8 deployment outside the lever's single-request band
    is switched off too.
    """
    if (int(sm), int(tp)) != (POLICY_SM, POLICY_TP):
        return False
    from ... import levers as _levers

    snapshot = _levers.environment() if snapshot is None else snapshot
    mode = snapshot.get(_levers.knob_name("rejection_sampler"), "").strip().lower()
    if mode:
        return mode != _levers.OFF
    return snapshot.get(_levers.DEFAULT_KNOB, "").strip().lower() != _levers.BASELINE


def _switch_off(spec) -> ForcedInert:
    """Set ``use_rejection_sampling`` False on ``spec``, record it as this process's forced-inert config, log it."""
    global FORCED_INERT
    spec.use_rejection_sampling = False
    FORCED_INERT = ForcedInert(_forced_count(), spec)
    logger.warning(
        f"[musespark_open] rs: forced acceptance ({FORCE_ENV}={FORCED_INERT.force}): rejection sampling inert, the "
        "argmax drafts and the strict verify run"
    )
    return FORCED_INERT


def apply_forced_policy_to_spec_config(spec_config, sm: int, tp: int) -> Optional[ForcedInert]:
    """Under ``forced_mode() == "inert"`` on a deployment ``policy_applies(sm, tp)``: set ``use_rejection_sampling``
    False on ``spec_config``.

    ``spec_config`` is the speculative config the model loader routes with (``routing.route``): the engine's own
    object, which the model engine, its runner (and the spec metadata it builds at the first forward, warmup
    included), the CUDA-graph runner, the sampler and the executor hold. The loader deep-copies the model config
    after routing, so the model's copy (``llm.spec_config``) inherits the flag off. Idempotent.

    Returns:
        This process's forced-inert record, or None when nothing was switched off.
    """
    if forced_mode() != "inert" or not policy_applies(sm, tp):
        return FORCED_INERT
    if spec_config is None or not getattr(spec_config, "use_rejection_sampling", False):
        return FORCED_INERT
    return _switch_off(spec_config)


def apply_forced_policy(shell, deployment) -> Optional[ForcedInert]:
    """Under ``forced_mode() == "inert"`` on a ``deployment`` the policy applies to: rejection sampling off on the
    model's speculative config.

    Call before any install (``drafter.adopt_drafter``). ``deployment`` is the plan's ``levers.Deployment`` (its
    ``sm`` and ``tp`` go to ``policy_applies``), or None for a plan without a lever table, which switches nothing. With
    the target routed by the model loader the flag is already off (``apply_forced_policy_to_spec_config``) and this
    only records the forced-inert state on ``shell.llm``. A model built without that route still has it on: this
    switches ``shell.llm.spec_config`` off, and if the engine holds another object, the step guard
    (``check_forced_inert_step``) refuses the first forward. Idempotent.

    Returns:
        This process's forced-inert record, or None when nothing was switched off.

    Raises:
        RuntimeError: ``shell.llm.spec_config`` has the flag on and is not ``shell.llm.model_config.spec_config``
            (switching one copy would leave readers of the other on).
    """
    if forced_mode() != "inert":
        return FORCED_INERT
    llm = shell.llm
    spec = getattr(llm, "spec_config", None)
    applies = deployment is not None and policy_applies(deployment.sm, deployment.tp)
    if applies and spec is not None and getattr(spec, "use_rejection_sampling", False):
        if spec is not getattr(getattr(llm, "model_config", None), "spec_config", None):
            raise RuntimeError(
                "[musespark_open] rs: forced acceptance: llm.spec_config is not llm.model_config.spec_config, so "
                "switching rejection sampling off on it would not reach every reader"
            )
        _switch_off(spec)
    if FORCED_INERT is not None:
        llm.__dict__["_rs_forced_inert"] = FORCED_INERT
    return FORCED_INERT


def check_forced_inert_metadata(spec_metadata, where: str) -> None:
    """Fail-closed guard of the verify (eager and capture calls): in a forced-inert process, the step's spec metadata
    must have rejection sampling off.

    Raises:
        RuntimeError: it has it on.
    """
    inert = FORCED_INERT
    if inert is not None and getattr(spec_metadata, "use_rejection_sampling", False):
        raise RuntimeError(
            f"[musespark_open] rs: {where}: forced acceptance ({FORCE_ENV}={inert.force}) switched rejection "
            "sampling off, but this step's spec metadata has use_rejection_sampling on"
        )


def check_forced_inert_step(runner, spec_metadata) -> None:
    """Fail-closed guard of the first forwards of a forced-inert process (``rs.on_step``, every forward from the
    first warmup one, before its prepare and any CUDA-graph capture): the runner's engine ``spec_config`` must be the
    switched object with the flag off, and the step's own spec metadata must have it off. Passes once with spec
    metadata present, then checks nothing.

    Raises:
        RuntimeError: either does not hold.
    """
    inert = FORCED_INERT
    if inert is None or inert.step_checked:
        return
    engine = getattr(getattr(runner, "_config", None), "spec_config", None)
    if engine is not inert.spec_config or getattr(engine, "use_rejection_sampling", False):
        raise RuntimeError(
            f"[musespark_open] rs: forced acceptance ({FORCE_ENV}={inert.force}): the runner's spec_config is not "
            "the switched config, or has use_rejection_sampling on"
        )
    if spec_metadata is None:
        return
    check_forced_inert_metadata(spec_metadata, "first forward")
    inert.step_checked = True
