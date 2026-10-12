# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema validation for staircase-onboard's ``task.yaml`` input.

A staircase target's identity is the ``(checkpoint, GPU arch, parallel
config)`` triple, so those three fields are the spec's core and the run's
slug is derived from them. ``arch`` is **required and never inferred**:
catalog receipts are per-architecture, so a different arch means the whole
certification is redone, and guessing it wrong means gating against
receipts that describe another machine.

The ``environment`` block is the part that earns its keep. It declares —
once, up front — how GPU work is launched, and the workflow resolves it
into a single command template that every agent prompt carries verbatim.
Agents never construct a launcher command, never probe for one, and never
call a test-runner skill. That is deliberate: the failure mode this
workflow exists to avoid is a run that spends its budget deciding how to
submit a job instead of onboarding a model.

Validation runs at the CLI boundary so an invalid spec aborts before any
agent is constructed, and every detected problem is batched into one
message so the user fixes them in a single pass rather than one per run.
:func:`load_and_validate_task_yaml` also fills in the workflow's defaults,
so the resolved spec copied into the workspace is fully explicit.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from agent_flow.agent_runtime import AGENTS_FIELD, validate_agents

from . import stats
from .committer import COMMIT_POLICY_FIELD, validate_commit_policy_block
from .embargo import PACKAGE_PREFIX, package_entry_problem, resolve_entry
from .roles import ROLES, validate_role_defaults

REQUIRED_PATH_FIELDS: tuple[str, ...] = (
    "checkpoint_path",
    "trtllm_repo_path",
)

ARCH_FIELD = "arch"
PARALLEL_FIELD = "parallel"
FEATURES_FIELD = "features"
ENVIRONMENT_FIELD = "environment"
ACCURACY_FIELD = "accuracy"

# ``sm_100`` (B200), ``sm_103`` (GB300), ... Matched rather than
# enumerated: a new architecture should not need a code change here, and
# preflight asserts the value against the host anyway, which is the check
# that actually protects the receipts.
ARCH_PATTERN = re.compile(r"^sm_\d+$")

# ``tp4`` / ``tep4`` / ``dep4`` carry their world size in the trailing
# number; ``unified`` means one modeling codebase over a family of
# parallel configurations and takes its world size from the member set.
PARALLEL_PATTERN = re.compile(r"^(tp|tep|dep)(\d+)$")
PARALLEL_UNIFIED = "unified"
DEFAULT_PARALLEL = PARALLEL_UNIFIED

# Which concrete configurations a ``unified`` segment is actually gated over.
MEMBERS_FIELD = "members"

# The default member set, and the reason there is one: ``unified`` is
# ``DEFAULT_PARALLEL``, so a spec that omits ``parallel`` gets the form with
# the broadest claim, and before this it also got the narrowest gating — one
# build, gated at ``--tp_size=N`` and nothing else, with the expert-parallel
# and attention-DP branches that only a unified codebase carries never
# executed by any gate.
#
# TEP and DEP at the world size are the baseline pair
# ``skills/staircase-modeling-parallel/references/unified.md`` names; TP is
# the degenerate path, and it is not free — it is the only member that folds
# every branch off, so it runs the general path with nothing switched on and
# no other member covers that.
DEFAULT_MEMBER_KINDS: tuple[str, ...] = ("tp", "tep", "dep")

# Feature name -> the skill the assembler loads for it. This mapping is
# the reason the feature set is closed: a feature with no skill is an
# assembler that has been told to build something nobody wrote down how
# to build, which surfaces as a confused mid-run report rather than as an
# error here.
FEATURE_SKILLS: dict[str, str] = {
    "mtp": "staircase-modeling-speculator",
    "eagle3": "staircase-modeling-speculator",
    "dflash": "staircase-modeling-speculator",
    # A block drafter: one 3-stage pass proposes a 5-token block, where MTP
    # chains one-token heads. Its own kind rather than a spelling of `mtp`
    # because the two read the SAME checkpoint namespace: DeepSeek-V4.1 ships
    # its DSpark stages as `mtp.*` with `num_nextn_predict_layers: 3`, and
    # `--spec_decode_algo=MTP` over that builds three chained heads that either
    # fail to load or draft garbage the verifier silently rejects.
    "dspark": "staircase-modeling-speculator",
    # DeepSeek-V4.1's "SWA Bounded Replay", decoder and encoder halves. NOT
    # output-identical -- both deliberately truncate sliding-window state -- so
    # accuracy is a real check here; but a unit that implements nothing passes
    # it too, which is why each carries a work witness. See
    # `witness.run_replay_witness`.
    "decoder_replay": "staircase-modeling-bounded-replay",
    "encoder_replay": "staircase-modeling-bounded-replay",
    # KV pools switch from bf16 containers holding the reference's
    # fake-quantized values (exact, and what core ships) to the checkpoint's
    # native storage formats. Output-identical by construction when attention
    # dequantizes exactly, so boot and accuracy cannot see whether it happened;
    # its witness counts stored bytes per token against a declared figure.
    "kv_quant": "staircase-modeling-kv-quant",
    "eplb": "staircase-modeling-eplb",
    "dwdp": "staircase-modeling-dwdp",
    "multimodal": "staircase-modeling-multimodal",
    # A weight tensor too large to keep resident lives in pinned host memory
    # and the forward gathers what it needs. Named for the mechanism, not for
    # the first tensor it was needed for (Qwen3.8-Flash-Next's 53.7 GB PLE
    # n-gram table), because the mechanism is what generalizes.
    "host_offload": "staircase-modeling-host-offload",
    # Prefill and decode on separate engines. A modeling feature rather than a
    # serving one: only per-sequence state the transceiver enumerates survives
    # the hop, so a target that keeps state elsewhere passes every
    # single-engine gate and is wrong across it.
    "disagg": "staircase-modeling-disagg",
}

# Features that read the checkpoint's `mtp.*` namespace, which holds MTP
# modules OR DSpark stages and never both. Listing two of them is not a richer
# target; it is two readings of one set of weights, one of which is wrong.
MTP_NAMESPACE_FEATURES: tuple[str, ...] = ("mtp", "dspark")

# The engine batch limit BOTH legs of a token-comparing witness (residency,
# kv_quant, equivalence) run at. It replaces the runtime block's
# `--max_batch_size` in leg A's quickstart argv, and a `contrast_command` leg B
# is handed it as STAIRCASE_WITNESS_MAX_BATCH_SIZE. `1` is the value that
# gives both legs the same schedule. A row's bits depend on the rows beside it
# (fp32 GEMMs, sparse attention at a large verify batch, the all-reduce
# strategy by token count), and batched admission follows arrival timing, so
# two batched legs can pack rows differently and flip a near-tie that neither
# leg's feature caused. Unset keeps the runtime block's limit. The replay
# witnesses count work rather than compare tokens, and do not take it.
WITNESS_BATCH_SIZE_KEY = "max_batch_size"

# The keys every witness block may carry. `expected_mib` is the residency
# witness's prediction, `contrast_command` the equivalence witness's second
# launcher. `timeout` is seconds PER LEG, where the one-engine boot budget does
# not fit -- a disagg leg boots two engines in sequence.
# The workload a token-comparing witness (residency, kv_quant, equivalence)
# declares, as the acceptance gate does: `prompts` or `prompts_file`,
# `max_tokens`, and `min_generated_tokens` (default 32) -- a witness whose
# continuations are shorter fails as too short to compare instead of passing
# on two-token answers. With none, the declared acceptance prompts, else the
# boot prompts.
WITNESS_WORKLOAD_KEYS: tuple[str, ...] = (
    "prompts",
    "prompts_file",
    "max_tokens",
    "min_generated_tokens",
)
WITNESS_KEYS: tuple[str, ...] = (
    "contrast_env",
    "expected_mib",
    "contrast_command",
    "timeout",
    WITNESS_BATCH_SIZE_KEY,
    *WITNESS_WORKLOAD_KEYS,
)

# Features gated by the EQUIVALENCE witness (token identity across a hop).
#
#   contrast_command  REQUIRED. Leg B's launcher: a context engine, a generation
#                     engine, a proxy and a client, which no environment variable
#                     over the aggregated command turns into. Without it leg B is
#                     leg A again and the witness compares a run to itself
#   prompt_tokens     optional list: one generated prompt of at least that many
#                     tokens each, added to the boot prompts on BOTH legs, for the
#                     lengths a hop has edges at (over a prefill chunk, odd, one
#                     past a page) that hand-written prompts rarely land on
#   legs_env          optional mapping, see WITNESS_LEGS_ENV_KEY
EQUIVALENCE_WITNESS_FEATURES: tuple[str, ...] = ("disagg",)

# The environment BOTH legs of an equivalence witness run under: leg A's
# process, and leg B's launcher, which is handed it as
# STAIRCASE_WITNESS_LEGS_ENV to apply to every engine it starts. A witness
# holds when its legs differ in exactly one variable, and leg B's launcher is a
# different program from leg A's: a code path a disaggregated engine refuses or
# guards off (a replay that cannot run across the hop) stays on in the
# aggregated leg, so the legs differ in it as well as in the hop and token
# identity no longer holds by construction. Pinning it here on both legs takes
# it out of the comparison; its own witness covers it. A variable may not be in
# both this and `contrast_env`: there it is the contrast. The residency and
# kv_quant witnesses run the same aggregated command on both legs, so every
# code path other than `contrast_env` is already alike and they do not take it.
WITNESS_LEGS_ENV_KEY = "legs_env"
EQUIVALENCE_WITNESS_KEYS: tuple[str, ...] = (
    "contrast_env",
    "contrast_command",
    "prompt_tokens",
    "timeout",
    WITNESS_BATCH_SIZE_KEY,
    WITNESS_LEGS_ENV_KEY,
    *WITNESS_WORKLOAD_KEYS,
)

# The bounded-replay witness parameters, per feature, each REQUIRED. Declared
# in the spec because a work witness that sizes its workload from the run it is
# judging can only agree with itself -- and because both verdicts are
# comparisons against `window`, which is a property of the checkpoint, not of
# anything the target prints.
#
#   window         the checkpoint's sliding window (128 for DeepSeek-V4.1)
#   prompt_tokens  decoder_replay: the long prompt's length, at least. Must
#                  exceed `window`, or replay-on and replay-off do the same work
#   prefix_tokens  encoder_replay: the shared prefix's length. Must exceed
#                  `window`, or a prefix hit replays everything it reused and
#                  "re-forwarded the whole prefix" looks like a pass
REPLAY_WITNESS_REQUIRED: dict[str, tuple[str, ...]] = {
    "decoder_replay": ("contrast_env", "window", "prompt_tokens"),
    "encoder_replay": ("contrast_env", "window", "prefix_tokens"),
}
# `reuse_alignment`: the reuse boundary must be a multiple of this. DeepSeek-V4.1
# compresses global KV in pairs (compress ratio 2), so a boundary that splits a
# pair leaves one half of it nowhere; SGLang refuses an odd boundary for that
# reason. Default 1, i.e. no constraint.
REPLAY_WITNESS_OPTIONAL: dict[str, tuple[str, ...]] = {
    "decoder_replay": ("timeout",),
    "encoder_replay": ("reuse_alignment", "timeout"),
}

# The witness builds its long prompt as a quickstart `--prompt` argument, one
# short word per token at least. Linux caps ONE argv string at 128 KiB
# (MAX_ARG_STRLEN) and fails the exec with E2BIG, which surfaces as a leg that
# "did not complete" -- so the cap is stated here, where it costs nothing.
# 16384 words of at most six bytes each is ~96 KiB.
MAX_WITNESS_PROMPT_WORDS = 16384

# The `kv_quant` witness block.
#
#   expected_bytes_per_token  REQUIRED. Persistent KV bytes one rank stores per
#                             token, computed from the checkpoint's formats
#                             before the run (890 for DeepSeek-V4.1 NVFP4)
#   exact                     default true: attention dequantizes the native
#                             values exactly, so output must be token-identical
#                             to bf16 storage of the same values. `false` names
#                             a lossy attention path (an FP8-query kernel) and
#                             drops the identity leg, leaving accuracy as the
#                             only output check -- a weaker claim, recorded so
#   contrast_env              the leg WITH bf16 storage; required when exact
#   bytes_tolerance           relative, default 0.02. Enough for row padding,
#                             not for a different layout: padding the 4-byte
#                             indexer scale to 16 moves 890 to 920 (+3.4%)
KV_QUANT_WITNESS_KEYS: tuple[str, ...] = (
    "contrast_env",
    "expected_bytes_per_token",
    "exact",
    "bytes_tolerance",
    "timeout",
    WITNESS_BATCH_SIZE_KEY,
    *WITNESS_WORKLOAD_KEYS,
)
DEFAULT_BYTES_TOLERANCE = 0.02

# Every feature whose witness block is a DECLARATION -- a workload or a figure
# the verdict is judged against -- and is therefore required at spec time.
DECLARED_WITNESS_FEATURES: tuple[str, ...] = (*REPLAY_WITNESS_REQUIRED, "kv_quant")

# The acceptance gate's WORKLOAD, declared in `accuracy.acceptance` beside the
# per-unit floors: at the block's top level for every unit, or inside one
# unit's entry for that unit alone. With none declared, the gate takes the
# first `num_prompts` (default 32) natural-text questions of the accuracy
# benchmark's dataset -- a local path, or a Hugging Face id read from the local
# cache only. Never the boot prompts: a 2-token answer ends in EOS before a
# drafter is asked anything, and the rate reads 1.00 for every drafter.
#
#   prompts               a list of prompt strings, passed as-is (no chat
#                         template is applied: write them the way the boot
#                         prompts are written)
#   prompts_file          or a file holding them: a YAML/JSON list of strings,
#                         or a mapping with `prompts:` (boot.yaml's shape).
#                         Relative paths resolve against the spec's directory
#   dataset               or a dataset: a local file or directory (JSON, JSON
#                         lines, YAML, Parquet, Arrow) or a cached dataset id
#   num_prompts           questions taken from the dataset (default 32)
#   text_field            the dataset field holding the question (default: the
#                         first of question/prompt/problem/input/text/...)
#   prompt_template       how a question becomes a prompt, `{text}` naming it;
#                         write the model's chat markers here if it needs them
#   split, dataset_config which split and config of a dataset id
#   max_tokens            tokens each prompt may generate (default 256)
#   min_draft_iterations  generation iterations the whole workload must reach
#                         before its rate is read (default 32); fewer fails as
#                         "workload too short" rather than as a low number
#   min_prompts           prompts the workload must hold (default 32); fewer
#                         fails as "workload too small": four prompts read a
#                         change in generated text as an acceptance regression
ACCEPTANCE_WORKLOAD_KEYS: tuple[str, ...] = (
    "prompts",
    "prompts_file",
    "dataset",
    "num_prompts",
    "text_field",
    "prompt_template",
    "split",
    "dataset_config",
    "max_tokens",
    "min_draft_iterations",
    "min_prompts",
)
# The paired comparison against the previous tree's verdict, at the block's top
# level only: `test` (t or wilcoxon, default t), `alpha` (one-sided, default
# 0.05), `min_effect` (accepted tokens per iteration, default 0.1). A drop must
# be both significant and at least `min_effect` to fail.
ACCEPTANCE_REGRESSION_KEYS: tuple[str, ...] = ("test", "alpha", "min_effect")
ACCEPTANCE_BLOCK_KEYS: tuple[str, ...] = (*ACCEPTANCE_WORKLOAD_KEYS, "regression")
# What one unit's entry in `accuracy.acceptance` may carry:
#   min_absolute        the absolute floor used when the `=off` leg cannot run
#   measured_reference  what a reference drafter measured on this workload; the
#                       floor is this x 0.95 when `min_absolute` is not given
#   floor_source        where that measurement was taken. A floor with neither
#                       a measured reference nor a source is used, and every
#                       verdict says it is not tied to a measurement
ACCEPTANCE_UNIT_KEYS: tuple[str, ...] = (
    "min_absolute",
    "measured_reference",
    "floor_source",
    *ACCEPTANCE_WORKLOAD_KEYS,
)

TARGET_FIELD = "target"
RUNTIME_FIELD = "runtime"
SCOPE_FIELD = "scope"
SOURCE_EMBARGO_FIELD = "source_embargo"
# Budgets for the bounded UNCOVERED remediation cycle. Optional: the
# defaults in `remediation.RemediationBudget` are the policy, and a spec
# only states one when this target is known to need a wider one.
REMEDIATION_FIELD = "catalog_remediation"

# ------------------------------------------------ envelope / matrix / certification
# What the target is certified FOR, declared before the run: the serving
# envelope (engine limits, capture policy, every member claimed) and the
# feature combinations that must have engine evidence. Both are optional so an
# existing spec still validates -- its certified set is then the hull of its
# gate tuples, which every rendering states. The validators, the derivation and
# the migration command live in `envelope.py`.
SERVING_ENVELOPE_FIELD = "serving_envelope"
FEATURE_MATRIX_FIELD = "feature_matrix"

# Runtime switches the gates must exercise, and which therefore have to be
# declared rather than left to whatever the engine defaults to.
#
# The values are what the *user* wants true of the run, not what the flag
# says: two of these are spelled as negations on the command line
# (``--disable_overlap_scheduler``, ``--disable_kv_cache_reuse``), so a
# reader translating "block reuse on" into a flag has to remember to emit
# nothing. Doing that translation once, in Python, is the point —
# :func:`~.gates.runtime_flags` owns it and no agent ever types it.
RUNTIME_SWITCHES: dict[str, bool] = {
    "cuda_graph": False,
    "chunked_prefill": False,
    "overlap_scheduler": True,
    "kv_cache_reuse": True,
}

# Scopes a target may be built at. ``text_only`` is the useful one: a
# multimodal checkpoint whose vision tower is out of scope still has to
# load, route and gate as a language model, and saying so up front is what
# keeps an assembler from treating the absent vision path as a defect.
SCOPES: tuple[str, ...] = ("full", "text_only")
DEFAULT_SCOPE = "full"

# Every block below is closed; see :func:`reject_unknown_keys` for why.
# Listed as data rather than derived from the validators so that adding a
# field and forgetting to accept it is a test failure, not a silent drop.
TOP_LEVEL_KEYS: tuple[str, ...] = (
    *REQUIRED_PATH_FIELDS,
    ARCH_FIELD,
    PARALLEL_FIELD,
    MEMBERS_FIELD,
    FEATURES_FIELD,
    ENVIRONMENT_FIELD,
    ACCURACY_FIELD,
    TARGET_FIELD,
    RUNTIME_FIELD,
    SCOPE_FIELD,
    SOURCE_EMBARGO_FIELD,
    REMEDIATION_FIELD,
    SERVING_ENVELOPE_FIELD,
    FEATURE_MATRIX_FIELD,
    # Per-role backend, model and reasoning effort; see `roles.py`.
    AGENTS_FIELD,
    COMMIT_POLICY_FIELD,
)

REMEDIATION_KEYS: tuple[str, ...] = (
    "max_rounds",
    "max_attempts_per_item",
    "max_new_items",
    "max_parallel_verifiers",
)

TARGET_KEYS: tuple[str, ...] = ("family", "checkpoint_segment")

ACCURACY_KEYS: tuple[str, ...] = (
    # The absolute `acceptance_length` floor a unit falls back to when the
    # `=off` leg cannot exist -- a genuinely new architecture has no built-in
    # implementation to compare a speculator against. See
    # `gates.resolve_acceptance_floor`. Also the acceptance gate's declared
    # workload; see `ACCEPTANCE_WORKLOAD_KEYS`.
    "acceptance",
    "benchmark",
    "reference_yaml",
    "model_key",
    "tolerance",
    "dataset_path",
    "llm_api_options",
    # Which ROW of reference_yaml this run anchors to. The references key a
    # model to a list discriminated by quant_algo / kv_cache_quant_algo /
    # spec_dec_algo, and the match is exact -- so a quantized checkpoint with
    # no discriminators here anchors to the bf16 row, or to nothing at all.
    "discriminators",
    # Per-unit configuration for the two-leg WITNESS gates, which exist for
    # the features that are output-identical by construction. Supplies the one
    # thing the orchestrator cannot derive: the environment that turns the
    # feature OFF, and (for host_offload) the memory delta predicted from the
    # checkpoint's own shapes. The bounded-replay features also declare their
    # workload here -- see `REPLAY_WITNESS_REQUIRED`.
    "witness",
    # `trtllm-eval` takes these two as its OWN flags rather than through the
    # options YAML, so they have to be declarable here -- see
    # `gates.eval_command`, which reads exactly these names. They were
    # validated below but missing from this list, which made every task.yaml
    # that set them fail as "unknown key" once the block was closed.
    "max_seq_len",
    "max_batch_size",
    "benchmark_args",
    # How the accuracy floor is derived; see `FLOOR_KEYS`.
    "floor",
    # The scheduler-equivalence witness; see `SCHEDULER_EQUIVALENCE_KEYS`.
    "scheduler_equivalence",
    # The boot gate's workload, declared here; see `BOOT_KEYS`.
    "boot",
    # The launch-path smoke; see `LAUNCH_KEYS`.
    "launch",
    # The benchmark's sample count, scalar or per unit, for the standard
    # error behind the statistical floor. Needed only for a benchmark whose
    # count the gate does not know (`stats.KNOWN_BENCHMARK_SAMPLES`) and whose
    # eval does not print one.
    "samples",
)

# The boot gate's workload, `accuracy.boot`: run-wide keys, plus optional
# per-unit entries (unit name -> the same keys) layered over them. With no
# block the workspace `boot.yaml` is read as before.
#
#   prompts       strings, or mappings {prompt, keywords, keywords_any, images,
#                 render}: a mapping's keywords are checked against THAT
#                 prompt's generated text; `images` are files sent with it
#                 (relative paths resolve against the spec's directory when the
#                 file is there, else against the tree under test, where the
#                 repository's test assets are); `render` is `loader` (the
#                 tree's multimodal input loader applies the chat template) or
#                 `raw` (the prompt is pre-rendered, placeholders included)
#   prompts_file  a YAML/JSON list of plain prompts, or boot.yaml's shape
#   keywords      each must appear in SOME generated text
#   max_tokens    per prompt (default 32)
#
# Keywords are matched against the generated texts only, never the log. A unit
# whose target implements `multimodal` and declares no image prompt sends the
# repository's own `tests/integration/test_input_files/merlion.png`.
BOOT_KEYS: tuple[str, ...] = ("prompts", "prompts_file", "keywords", "max_tokens")
BOOT_PROMPT_KEYS: tuple[str, ...] = ("prompt", "keywords", "keywords_any", "images", "render")
BOOT_RENDERS: tuple[str, ...] = ("loader", "raw")

# The launch-path smoke, `accuracy.launch`: a small accuracy workload through
# the unit's OWN deployment entry -- `trtllm-serve` by default, a declared
# launcher for the disagg unit -- because the accuracy gate's in-process LLM
# API never parses an HTTP request, renders a served chat template or loads
# served media. Run-wide keys, plus per-unit entries layered over them.
#
#   enabled        default true
#   units          units it runs in (default: every unit)
#   members        all (default) or first
#   num_samples    questions from the accuracy dataset (default 32)
#   max_tokens     per request (default 512)
#   endpoint       chat (default: /v1/chat/completions) or completions
#   dataset / text_field / answer_field / prompt_template / split /
#   dataset_config the workload, as for the acceptance gate
#   scorer         gsm8k / contains / exact / none (default: the benchmark's)
#   min_accuracy   the floor; default anchor - k * SE at the smoke's count
#   command        a launcher for an entry trtllm-serve alone is not, REQUIRED
#                  for the disagg unit: it starts the deployment, prints
#                  STAIRCASE_ENDPOINT=<base url> once it serves, and stops
#                  everything it started on SIGTERM
#   timeout        seconds for the entry to become healthy
LAUNCH_KEYS: tuple[str, ...] = (
    "enabled",
    "units",
    "members",
    "num_samples",
    "max_tokens",
    "endpoint",
    "dataset",
    "text_field",
    "answer_field",
    "prompt_template",
    "split",
    "dataset_config",
    "scorer",
    "min_accuracy",
    "command",
    "timeout",
)
LAUNCH_ENDPOINTS: tuple[str, ...] = ("chat", "completions")
LAUNCH_SCORERS: tuple[str, ...] = ("gsm8k", "contains", "exact", "none")

# The `accuracy.scheduler_equivalence` block: the witness that runs the same
# prompts greedily at batch 1 with the overlap scheduler on and off (and with
# speculation off) and requires token identity where the configuration is
# output-preserving. Every key optional; see `step_check.py`.
#
#   enabled               default true
#   units                 units it runs in (default: every unit)
#   prompts / prompts_file / dataset / num_prompts / text_field /
#   prompt_template       the workload, as for the acceptance gate; default the
#                         first 8 questions of the accuracy dataset
#   max_tokens            per prompt, default 128, at least 64
#   ignore_eos            default true, so every prompt reaches max_tokens
#   min_generated_tokens  default 64; a shorter continuation fails as too short
#   legs                  contrast legs, a subset of [no_overlap, no_spec]
#   spec_pair             how no_spec is judged: report (default), agreement
#                         (first difference at or past `agreement_tokens`) or
#                         exact
#   agreement_tokens      default 16
#   pin_autotuner         default true: one AutoTuner cache across the legs
#   timeout               seconds per leg
#   step_trace            {reference_command: ...}, the optional reference
#                         adapter for the step-trace hook
SCHEDULER_EQUIVALENCE_KEYS: tuple[str, ...] = (
    "enabled",
    "units",
    "prompts",
    "prompts_file",
    "dataset",
    "num_prompts",
    "text_field",
    "prompt_template",
    "max_tokens",
    "ignore_eos",
    "min_generated_tokens",
    "legs",
    "spec_pair",
    "agreement_tokens",
    "pin_autotuner",
    "timeout",
    "step_trace",
)
SCHEDULER_EQUIVALENCE_LEGS: tuple[str, ...] = ("no_overlap", "no_spec")
SCHEDULER_EQUIVALENCE_PAIRS: tuple[str, ...] = ("exact", "agreement", "report")

# The `accuracy.floor` block.
#
#   rule               `statistical` (default): floor = anchor - k * SE, the SE
#                      binomial at the anchor over the benchmark's sample count.
#                      `tolerance`: floor = anchor - `accuracy.tolerance`, the
#                      fixed width this workflow used before; an explicit choice
#   k                  default 2.0. GSM8K's 1319 questions at 92 % give SE 0.74,
#                      so the floor sits ~1.5 points under the anchor, where a
#                      fixed 5.0 sat ~7 SE under it
#   shift_min_members  default 2. When every member of a unit scores more than
#                      1 SE under the anchor, that is a FINDING that blocks
#                      delivery, even though each member passed its own floor --
#                      but only for a unit gated over at least this many members
#   acknowledge        unit -> reason. A finding on that unit is recorded with
#                      the reason and no longer blocks delivery
FLOOR_KEYS: tuple[str, ...] = ("rule", "k", "shift_min_members", "acknowledge")

ENVIRONMENT_KEYS: tuple[str, ...] = (
    "kind",
    "python",
    "env",
    "spool",
    "partition",
    "account",
    "container_image",
    "mounts",
    "time_limit",
    "gpus_per_node",
    # Pyxis reuses an already-imported container of this name on the node,
    # which is the difference between paying 4-8 minutes per job for a 26 GB
    # image and paying it once.
    "container_name",
    # `auto` (default) picks the MPI plugin by task count. Set it only when
    # the cluster reports a versioned PMIx plugin -- `srun --mpi=list`.
    "mpi",
    # A checkpoint the STOCK modeling path already supports, used by
    # `preflight.substrate_gate` when the run's own checkpoint has an
    # architecture that path has never heard of. Optional, because a
    # checkpoint the stock path supports needs no proxy -- and declared
    # rather than guessed, because this block is closed and because which
    # model stands in for another is a judgement nothing here can make.
    "substrate_checkpoint",
    # Per-workload substrate overrides. See `_validate_workloads`.
    "workloads",
    # The compute sessions the run stands on, the one the orchestrator lives
    # in, and how a successor allocation is requested. Validated by
    # `substrate.sessions.validate_sessions`.
    "sessions",
    "orchestrator",
    "allocator",
)

#: Environment keys that describe the run's sessions rather than one launch,
#: and so have no meaning inside a per-workload override.
SESSION_ENVIRONMENT_KEYS: tuple[str, ...] = ("sessions", "orchestrator", "allocator")

ENVIRONMENT_KINDS: tuple[str, ...] = ("local", "slurm", "slurm_broker", "session_spool")
ENVIRONMENT_REQUIRED_FIELDS: tuple[str, ...] = ("kind", "python")
SLURM_REQUIRED_FIELDS: tuple[str, ...] = ("partition", "container_image")

# Accuracy tolerance, in benchmark points, under `accuracy.floor.rule:
# tolerance` only. The default rule is statistical (`FLOOR_KEYS`): 5.0 points
# is about seven standard errors on GSM8K, wide enough that a defect moving
# every member by half a point passed every gate of the run that had it.
DEFAULT_ACCURACY_TOLERANCE = 5.0


class TaskSchemaError(ValueError):
    """Raised when ``task.yaml`` fails staircase-onboard schema validation."""


def reject_unknown_keys(
    block: Mapping[str, Any],
    known: Sequence[str],
    errors: list[str],
    *,
    where: str = "",
) -> None:
    """Append an error for any key in ``block`` that ``known`` does not list.

    Every block in this schema is closed, and the reason is the same one
    ``runtime`` has always stated: a misspelled key that is silently ignored
    produces a run that gates a configuration nobody asked for, and the
    report still says it was covered.

    That was not a hypothetical. Before this check existed, ``feature:``
    instead of ``features:`` validated cleanly and resolved to ``[]`` — the
    run onboarded core only, gated it green, and reported success for a
    feature set it never built. ``runtme:`` instead of ``runtime:`` dropped
    ``max_seq_len`` and reproduced the ``Impossible to fit in any sequence
    in kvCache`` failure that had already cost most of a run's wall clock.

    The nearest known key is suggested, because the whole class of error is
    off-by-one-character and a suggestion turns the message into the fix.
    """
    prefix = f"{where}." if where else ""
    for key in sorted(set(block) - set(known)):
        close = difflib.get_close_matches(str(key), [str(k) for k in known], n=1, cutoff=0.6)
        hint = f" — did you mean '{prefix}{close[0]}'?" if close else ""
        errors.append(
            f"unknown key '{prefix}{key}'{hint} Known keys here: {sorted(str(k) for k in known)}"
        )


def parse_parallel_segment(segment: str) -> tuple[str, int | None]:
    """Return ``(kind, world_size)`` for a parallel segment.

    ``world_size`` is ``None`` for ``unified``, whose member set — not the
    segment string — decides how many devices the run claims.
    """
    if segment == PARALLEL_UNIFIED:
        return PARALLEL_UNIFIED, None
    match = PARALLEL_PATTERN.match(segment)
    if match is None:
        raise TaskSchemaError(
            f"'{PARALLEL_FIELD}' must be 'tp<N>', 'tep<N>', 'dep<N>' or "
            f"'{PARALLEL_UNIFIED}'; got {segment!r}"
        )
    world_size = int(match.group(2))
    if world_size < 1:
        raise TaskSchemaError(f"'{PARALLEL_FIELD}' world size must be >= 1; got {segment!r}")
    return match.group(1), world_size


def members_for(data: Mapping[str, Any], unit: str | None, world_size: int) -> list[str]:
    """The concrete parallel configurations ``unit`` is gated over.

    One function and one code path for both forms on purpose. A concrete
    segment is the **degenerate** member set — itself, once — so the gate
    loop in :meth:`~.workflow.StaircaseOnboardWorkflow._assemble_one_unit`
    never branches on whether the run is unified, and no caller can forget
    the unified case the way :func:`~.gates.topology_flags` did.

    ``members`` takes the same two shapes ``runtime`` and
    ``accuracy.llm_api_options`` take: a list applies to every unit, and a
    mapping from assembly unit to a list is layered over ``default``. The
    per-unit form is what a run uses to buy a cheaper answer honestly — the
    ``multimodal`` unit gated at one member, named, rather than every unit
    gated at one member and the target documented as covering three.

    Validation lives in :func:`_validate_members`; this resolves an
    already-validated spec and falls back to the default set.
    """
    segment = str(data.get(PARALLEL_FIELD) or DEFAULT_PARALLEL)
    if segment != PARALLEL_UNIFIED:
        return [segment]
    block = data.get(MEMBERS_FIELD)
    if isinstance(block, dict):
        chosen = block.get(unit)
        block = block.get("default") if chosen is None else chosen
    if isinstance(block, list) and block:
        return [str(member) for member in block]
    return [f"{kind}{world_size}" for kind in DEFAULT_MEMBER_KINDS]


def derive_slug(data: Mapping[str, Any]) -> str:
    """Return ``<checkpoint>-<arch>-<segment>`` for a validated spec.

    The checkpoint segment is the checkpoint directory's own name,
    lowercased with non-alphanumerics folded to hyphens — it names the
    branch, the workspace subdirectory and the run's reports, so it has to
    survive being a path component on any filesystem.
    """
    checkpoint = Path(str(data["checkpoint_path"]).rstrip("/")).name
    checkpoint = re.sub(r"[^a-zA-Z0-9]+", "-", checkpoint).strip("-").lower()
    return f"{checkpoint}-{data[ARCH_FIELD]}-{data[PARALLEL_FIELD]}"


def derive_family(architecture: str) -> str:
    """``GptOssForCausalLM`` -> ``gpt_oss``: the modeling_v2 family directory.

    The family groups every target for one architecture, and in tree it is
    snake_case with the ``For<Task>`` suffix dropped — ``gpt_oss``,
    ``deepseek_v3``. Derived rather than declared because the checkpoint
    already names its architecture, and a hand-typed family that disagrees
    with the routing index is a target nothing can reach.
    """
    name = re.sub(r"For(Causal|Conditional|Seq2Seq|Sequence)?LM$", "", architecture)
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    name = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", name)
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def derive_checkpoint_segment(checkpoint_path: str | Path) -> str:
    """``DeepSeek-R1-0528-NVFP4`` -> ``deep_seek_r1_0528_nvfp4``.

    **A guess, and the one identity component worth overriding.** In tree the
    segment sits *under* the family directory, so the family's own name is
    usually redundant in it — ``deepseek_v3/r1_0528_nvfp4__sm_103__dep4`` rather
    than ``deepseek_v3/deepseek_r1_0528_nvfp4__sm_103__dep4``. Nothing can derive
    which part is redundant, so set ``target.checkpoint_segment`` in
    task.yaml when the default reads badly.
    """
    name = Path(str(checkpoint_path).rstrip("/")).name
    # Split camelCase only at a genuine word boundary — a lowercase letter
    # followed by a capital. Allowing a digit on the left would break the
    # size suffixes these names are full of: `Qwen3-30B-A3B` would come out
    # as `qwen3_30_b_a3_b`.
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", name)
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def _validate_target(block: Any, errors: list[str]) -> None:
    """Validate the optional ``target`` block, which overrides the derivation."""
    if not isinstance(block, dict):
        raise TaskSchemaError(f"'{TARGET_FIELD}' must be a mapping, got {type(block).__name__}")
    reject_unknown_keys(block, TARGET_KEYS, errors, where=TARGET_FIELD)
    for name in TARGET_KEYS:
        value = block.get(name)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            errors.append(f"'{TARGET_FIELD}.{name}' must be a non-empty string")
        elif not re.match(r"^[a-z0-9_]+$", value):
            errors.append(
                f"'{TARGET_FIELD}.{name}' is a Python package directory name, so it must "
                f"be lowercase with underscores; got {value!r}"
            )


def _validate_runtime_switches(block: dict, errors: list[str], where: str, *, fill: bool) -> None:
    """Validate one runtime block. ``fill`` writes the defaults in."""
    unknown = sorted(set(block) - set(RUNTIME_SWITCHES) - {"extra_flags"})
    if unknown:
        errors.append(
            f"'{where}' has unknown switch(es) {unknown}; known switches are "
            f"{sorted(RUNTIME_SWITCHES)} plus 'extra_flags'"
        )

    for name, default in RUNTIME_SWITCHES.items():
        value = block.get(name)
        if value is None:
            if fill:
                block[name] = default
        elif not isinstance(value, bool):
            errors.append(f"'{where}.{name}' must be a boolean, got {type(value).__name__}")

    extra = block.get("extra_flags")
    if extra is None:
        if fill:
            block["extra_flags"] = []
    elif not isinstance(extra, list) or not all(isinstance(f, str) for f in extra):
        errors.append(f"'{where}.extra_flags' must be a list of strings")


def _validate_runtime(block: Any, errors: list[str]) -> None:
    """Validate the optional ``runtime`` block in place, filling defaults.

    Unknown keys are an error rather than a warning. A misspelled switch
    that is silently ignored produces a run that gates a feature set
    nobody asked for, and the report still says the feature was covered —
    which is the expensive kind of wrong.
    """
    if not isinstance(block, dict):
        raise TaskSchemaError(f"'{RUNTIME_FIELD}' must be a mapping, got {type(block).__name__}")

    # Per-unit form: a mapping whose values are themselves mappings. The units
    # of one run can genuinely DISAGREE about a switch -- `core` and `mtp` were
    # gated with block reuse ON and passed, while the multimodal path requires
    # it OFF -- and no run-wide value is right for all three. A per-unit block
    # is layered over `default`, so a unit states only what differs.
    if any(isinstance(v, dict) for v in block.values()):
        nested = {k: v for k, v in block.items() if isinstance(v, dict)}
        if len(nested) != len(block):
            errors.append(
                f"'{RUNTIME_FIELD}' mixes the flat form (cuda_graph: true) with the "
                f"per-unit form (core: {{...}}); pick one"
            )
        if "default" not in block:
            errors.append(
                f"'{RUNTIME_FIELD}' is a per-unit mapping, so it needs a 'default' key "
                f"-- a unit nobody listed would otherwise be gated under the engine's "
                f"defaults rather than this run's"
            )
        for unit, sub in nested.items():
            # Only `default` is filled in: a per-unit block is an OVERLAY, so
            # filling its absent switches with the schema defaults would
            # silently override `default` with them.
            _validate_runtime_switches(
                sub, errors, f"{RUNTIME_FIELD}.{unit}", fill=(unit == "default")
            )
        return

    _validate_runtime_switches(block, errors, RUNTIME_FIELD, fill=True)


def _validate_members(block: Any, parallel: Any, errors: list[str]) -> None:
    """Validate the optional ``members`` block: what a unified run is gated over.

    Three rules, each preventing a distinct silent wrong:

    - **Only under ``parallel: unified``.** A concrete segment already names
      one configuration, so a ``members`` list beside it is a second and
      quieter declaration of the same thing, and the two can disagree. There
      is no reading of ``parallel: tep4`` plus ``members: [dep4]`` that is
      not a mistake in one of them.
    - **No ``unified`` member.** A member is a configuration to *serve at*;
      ``unified`` is the family, not a member of it. Accepting it would
      reintroduce the TP-only gate under a new name — which is precisely the
      defect this key exists to close.
    - **One world size.** One allocation serves every member, so a member
      asking for a different device count cannot be run by this run at all.
      The mismatch is reported with both values rather than quietly re-sized
      to the allocation, because a member silently served at half its
      declared width is a measurement of something nobody asked for.
    """
    if parallel != PARALLEL_UNIFIED:
        errors.append(
            f"'{MEMBERS_FIELD}' is only legal with '{PARALLEL_FIELD}: {PARALLEL_UNIFIED}'; "
            f"this spec says '{PARALLEL_FIELD}: {parallel}', which already names the one "
            f"configuration the run is gated at"
        )
        return

    # Same two shapes `runtime` and `accuracy.llm_api_options` take: a list
    # applies to every unit; a mapping from unit to list is layered over
    # `default`.
    if isinstance(block, dict):
        if "default" not in block:
            errors.append(
                f"'{MEMBERS_FIELD}' is a per-unit mapping, so it needs a 'default' key "
                f"-- a unit nobody listed would otherwise fall back to the built-in "
                f"member set and be gated over configurations this spec did not choose"
            )
        lists = {f"{MEMBERS_FIELD}.{unit}": value for unit, value in block.items()}
    elif isinstance(block, list):
        lists = {MEMBERS_FIELD: block}
    else:
        errors.append(
            f"'{MEMBERS_FIELD}' must be a list of parallel segments, or a mapping from "
            f"assembly unit to one; got {type(block).__name__}"
        )
        return

    # Keyed by the segment string so the mismatch message can name the two
    # members that disagree rather than only the two numbers.
    world_sizes: dict[str, int] = {}
    for where, entries in lists.items():
        if not isinstance(entries, list) or not entries:
            errors.append(
                f"'{where}' must be a non-empty list of parallel segments; an empty one "
                f"means a unit that is built and never gated"
            )
            continue
        for index, entry in enumerate(entries):
            if not isinstance(entry, str):
                errors.append(f"'{where}[{index}]' must be a string, got {type(entry).__name__}")
                continue
            try:
                kind, size = parse_parallel_segment(entry)
            except TaskSchemaError as exc:
                errors.append(f"'{where}[{index}]' is not a parallel segment: {exc}")
                continue
            if kind == PARALLEL_UNIFIED or size is None:
                errors.append(
                    f"'{where}[{index}]' is '{PARALLEL_UNIFIED}', which is the family "
                    f"rather than a configuration to serve at. A member is concrete: "
                    f"'tp4', 'tep4', 'dep4'."
                )
                continue
            world_sizes[entry] = size

    if len(set(world_sizes.values())) > 1:
        widest = max(world_sizes, key=lambda m: world_sizes[m])
        narrowest = min(world_sizes, key=lambda m: world_sizes[m])
        errors.append(
            f"'{MEMBERS_FIELD}' mixes world sizes: {narrowest} wants "
            f"{world_sizes[narrowest]} device(s) and {widest} wants {world_sizes[widest]}. "
            f"One allocation serves every member, so they must all declare the same "
            f"world size."
        )


def _validate_source_embargo(block: Any, errors: list[str], repo: Any) -> None:
    """Validate ``source_embargo``: paths agents may not read.

    Two shapes, and the split is on where the path points rather than on how
    it is spelled:

    - **repo-relative**, resolved under ``trtllm_repo_path``. This is the
      case the field was written for: TensorRT-LLM already ships an
      implementation of this checkpoint, so name it and the writing roles
      cannot open it.
    - **absolute and outside the repo.** The tempting file is often not in
      this repository at all — a sibling vLLM or SGLang checkout on the same
      filesystem implements the same checkpoint, is one ``Read`` away from
      every agent, and is *more* likely to be found because the run
      deliberately starts one of those engines as its accuracy oracle. The
      argument that made this a mechanism rather than a discipline does not
      depend on which repository the file lives in; it depends on the file
      being reachable.

    An absolute path that resolves *inside* the repo is still rejected: that
    is the case the original error was actually about, and the fix is to
    spell it repo-relative so the list reads the same from any working
    directory.

    Either way the entry must **exist**. An embargo naming a path that is not
    there reads as protection while protecting nothing, and the typo it
    usually is would leave the real file readable. A **directory** entry is
    legal and embargoes the tree beneath it — a sibling engine checkout is a
    tree, and asking a spec to enumerate its model files is asking it to go
    stale.
    """
    if not isinstance(block, list) or not all(isinstance(p, str) for p in block):
        raise TaskSchemaError(
            f"'{SOURCE_EMBARGO_FIELD}' must be a list of paths: repo-relative, absolute for a "
            f"tree outside the repo, or `{PACKAGE_PREFIX}<path>` inside the workflow package"
        )
    if not isinstance(repo, str):
        return
    repo_root = Path(repo).resolve()
    for entry in block:
        if entry.startswith(PACKAGE_PREFIX):
            problem = package_entry_problem(entry)
            if problem:
                errors.append(f"'{SOURCE_EMBARGO_FIELD}' names {entry!r}, which {problem}.")
            elif not resolve_entry(entry).exists():
                errors.append(
                    f"'{SOURCE_EMBARGO_FIELD}' names {entry!r}, which does not exist under "
                    f"the workflow package. An embargo on a path that is not there protects "
                    f"nothing."
                )
            continue
        if entry.startswith("/"):
            resolved = Path(entry).resolve()
            if resolved == repo_root or repo_root in resolved.parents:
                errors.append(
                    f"'{SOURCE_EMBARGO_FIELD}' names {entry!r}, which is inside "
                    f"{repo}. Spell an in-repo path repo-relative, so the list means "
                    f"the same thing from any working directory."
                )
            elif not resolved.exists():
                errors.append(
                    f"'{SOURCE_EMBARGO_FIELD}' names {entry!r}, which does not exist. "
                    f"An embargo on a path that is not there protects nothing."
                )
        elif not (Path(repo) / entry).exists():
            errors.append(
                f"'{SOURCE_EMBARGO_FIELD}' names {entry!r}, which does not exist under "
                f"{repo}. An embargo on a path that is not there protects nothing."
            )


def _validate_remediation(block: Any, errors: list[str]) -> None:
    """Validate the ``catalog_remediation`` budgets.

    Every field is a positive integer, and stating one is optional -- the
    defaults in :class:`remediation.RemediationBudget` are the policy. A
    zero is rejected rather than read as "unlimited": an unbounded
    remediation cycle is the thing this block exists to prevent, so the
    spelling that means it should not be the easiest one to type by
    accident.
    """
    if block is None:
        return
    if not isinstance(block, dict):
        errors.append(f"'{REMEDIATION_FIELD}' must be a mapping, got {type(block).__name__}")
        return
    reject_unknown_keys(block, REMEDIATION_KEYS, errors, where=REMEDIATION_FIELD)
    for name in REMEDIATION_KEYS:
        value = block.get(name)
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            errors.append(
                f"'{REMEDIATION_FIELD}.{name}' must be an integer >= 1, got {value!r}. "
                f"There is no spelling for 'unlimited' on purpose."
            )


def _validate_workloads(block: Any, errors: list[str]) -> None:
    """Per-workload launch substrate: ``environment.workloads.<name>``.

    **The right substrate depends on how long the job runs.** A shared session
    imports a 26 GB container once and costs nothing per job afterwards, which
    is correct for a 1-minute entry test -- one onboarding ran 443 of them. It
    is wrong for a 10-15 minute gate, where the import is 30-50% overhead and
    isolation is the point: a shared session has **one failure domain**, so a
    gate that leaks a server, kills a process group or collides on PMIx takes
    every other job and the orchestrator with it. All three happened in the run
    this was written for, and two more session deaths followed in the next one
    -- see :func:`~.environment.shares_allocation_with_spool`.

    The workload name is checked against
    :data:`~.environment.WORKLOAD_KINDS` rather than trusted, because an
    override that never applies is **invisible**: the run simply uses the base
    environment and nothing says why. That is the same class of silent wrong
    :func:`reject_unknown_keys` exists for, one level down.
    """
    if block is None:
        return
    if not isinstance(block, dict):
        errors.append(f"'{ENVIRONMENT_FIELD}.workloads' must be a mapping of workload to block")
        return
    # Imported here rather than at module scope: `environment` is the consumer
    # of this schema, so a top-level import would make the two modules circular.
    from .environment import WORKLOAD_KINDS

    for name, override in block.items():
        where = f"{ENVIRONMENT_FIELD}.workloads.{name}"
        if name not in WORKLOAD_KINDS:
            errors.append(
                f"'{where}' is not a workload the orchestrator dispatches under; "
                f"expected one of {WORKLOAD_KINDS}"
            )
            continue
        if not isinstance(override, dict):
            errors.append(f"'{where}' must be a mapping")
            continue
        # `workloads` itself is excluded from the legal keys: an override that
        # carries its own overrides is a nesting nothing resolves.
        reject_unknown_keys(
            override,
            tuple(
                k
                for k in ENVIRONMENT_KEYS
                if k != "workloads" and k not in SESSION_ENVIRONMENT_KEYS
            ),
            errors,
            where=where,
        )
        kind = override.get("kind")
        if kind is not None and kind not in ENVIRONMENT_KINDS:
            errors.append(f"'{where}.kind' must be one of {ENVIRONMENT_KINDS}; got {kind!r}")
        env = override.get("env")
        if env is not None and not isinstance(env, dict):
            errors.append(f"'{where}.env' must be a mapping")


def _validate_environment(block: Any, errors: list[str]) -> None:
    """Validate the ``environment`` block in place, appending to ``errors``."""
    if not isinstance(block, dict):
        raise TaskSchemaError(
            f"'{ENVIRONMENT_FIELD}' must be a mapping, got {type(block).__name__}"
        )

    reject_unknown_keys(block, ENVIRONMENT_KEYS, errors, where=ENVIRONMENT_FIELD)

    _validate_workloads(block.get("workloads"), errors)

    from .substrate.sessions import validate_sessions

    validate_sessions(block, errors, where=ENVIRONMENT_FIELD)

    for name in ENVIRONMENT_REQUIRED_FIELDS:
        value = block.get(name)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"'{ENVIRONMENT_FIELD}.{name}' must be a non-empty string")

    kind = block.get("kind")
    if isinstance(kind, str) and kind not in ENVIRONMENT_KINDS:
        errors.append(
            f"'{ENVIRONMENT_FIELD}.kind' must be one of {ENVIRONMENT_KINDS}; got {kind!r}"
        )

    env = block.get("env")
    if env is None:
        block["env"] = {}
    elif not isinstance(env, dict):
        errors.append(f"'{ENVIRONMENT_FIELD}.env' must be a mapping, got {type(env).__name__}")
    else:
        for key, value in env.items():
            if not isinstance(value, (str, int, float)):
                errors.append(
                    f"'{ENVIRONMENT_FIELD}.env.{key}' must be a scalar the shell can "
                    f"carry, got {type(value).__name__}"
                )

    # Same must-exist rule the embargo list carries, for the same reason: a
    # proxy that is not there is discovered by the substrate gate, which is
    # the longest step in preflight and the one the proxy exists to rescue.
    proxy = block.get("substrate_checkpoint")
    if proxy is not None:
        if not isinstance(proxy, str) or not proxy.strip():
            errors.append(
                f"'{ENVIRONMENT_FIELD}.substrate_checkpoint' must be a path to a "
                f"checkpoint directory the stock modeling path supports"
            )
        elif not Path(proxy).exists():
            errors.append(
                f"'{ENVIRONMENT_FIELD}.substrate_checkpoint' does not exist: {proxy}. "
                f"The substrate gate falls back to it only after the run's own "
                f"checkpoint has already failed, so a missing proxy turns the "
                f"longest step in preflight into two failures instead of one."
            )

    if kind == "session_spool":
        # The spool is the whole contract. It must already exist, because a
        # spool nobody is watching accepts work forever and completes none
        # of it — the run would hang on its first gate with no error.
        spool = block.get("spool")
        if not isinstance(spool, str) or not spool.strip():
            errors.append(f"'{ENVIRONMENT_FIELD}.spool' is required when kind is 'session_spool'")
        elif not Path(spool).is_dir():
            errors.append(
                f"'{ENVIRONMENT_FIELD}.spool' does not exist: {spool}. Start the compute "
                f"session first — an unwatched spool accepts work and never completes it."
            )

    if kind == "slurm_broker":
        # The broker is an MCP service, so only a Claude Code process can
        # reach it — the orchestrator cannot. `account` is mandatory here
        # and not for plain `slurm` because the broker rejects every
        # submit that omits it, which costs a round trip to discover.
        for name in ("partition", "account", "container_image"):
            value = block.get(name)
            if not isinstance(value, str) or not value.strip():
                errors.append(
                    f"'{ENVIRONMENT_FIELD}.{name}' is required when kind is "
                    f"'slurm_broker' and must be a non-empty string"
                )
        gpus = block.get("gpus_per_node")
        if gpus is None:
            errors.append(
                f"'{ENVIRONMENT_FIELD}.gpus_per_node' is required when kind is "
                f"'slurm_broker'. NVL72 clusters reject job-scoped --gpus, so the "
                f"per-node count is the only form that submits."
            )
        elif not isinstance(gpus, int) or isinstance(gpus, bool) or gpus < 1:
            errors.append(f"'{ENVIRONMENT_FIELD}.gpus_per_node' must be an integer >= 1")
        mounts = block.get("mounts")
        if mounts is None:
            block["mounts"] = []
        elif not isinstance(mounts, list) or not all(isinstance(m, str) for m in mounts):
            errors.append(f"'{ENVIRONMENT_FIELD}.mounts' must be a list of strings")
        if not isinstance(block.get("time_limit"), str):
            errors.append(
                f"'{ENVIRONMENT_FIELD}.time_limit' is required when kind is "
                f"'slurm_broker' (HH:MM:SS); the partition's own maximum is not a default "
                f"the broker will supply."
            )

    if kind == "slurm":
        for name in SLURM_REQUIRED_FIELDS:
            value = block.get(name)
            if not isinstance(value, str) or not value.strip():
                errors.append(
                    f"'{ENVIRONMENT_FIELD}.{name}' is required when kind is 'slurm' "
                    f"and must be a non-empty string"
                )
        mounts = block.get("mounts")
        if mounts is None:
            block["mounts"] = []
        elif not isinstance(mounts, list) or not all(isinstance(m, str) for m in mounts):
            errors.append(f"'{ENVIRONMENT_FIELD}.mounts' must be a list of strings")


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_contrast_env(entry: Mapping[str, Any], errors: list[str], where: str) -> None:
    contrast_env = entry.get("contrast_env")
    if not isinstance(contrast_env, dict) or not contrast_env:
        errors.append(
            f"'{where}.contrast_env' is required and must be a non-empty mapping: "
            f"a witness with no changed variable compares a run to itself"
        )
    elif not all(isinstance(v, (str, int, float, bool)) for v in contrast_env.values()):
        errors.append(f"'{where}.contrast_env' values must be scalars the shell can carry")


def _validate_witness_timeout(entry: Mapping[str, Any], errors: list[str], where: str) -> None:
    timeout = entry.get("timeout")
    if timeout is not None and not _positive_int(timeout):
        errors.append(f"'{where}.timeout' must be a whole number of seconds >= 1, got {timeout!r}")


def _validate_witness_workload(entry: Mapping[str, Any], errors: list[str], where: str) -> None:
    """Validate a token-comparing witness block's declared workload."""
    _validate_acceptance_workload(
        {k: v for k, v in entry.items() if k in ("prompts", "prompts_file", "max_tokens")},
        errors,
        where,
    )
    minimum = entry.get("min_generated_tokens")
    if minimum is not None and not _positive_int(minimum):
        errors.append(f"'{where}.min_generated_tokens' must be an integer >= 1")


def _validate_witness_batch_size(entry: Mapping[str, Any], errors: list[str], where: str) -> None:
    value = entry.get(WITNESS_BATCH_SIZE_KEY)
    if value is not None and not _positive_int(value):
        errors.append(
            f"'{where}.{WITNESS_BATCH_SIZE_KEY}' must be an integer >= 1 (1 gives both legs "
            f"the same schedule), got {value!r}"
        )


_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _validate_legs_env(entry: Mapping[str, Any], errors: list[str], where: str) -> None:
    """Validate the optional ``legs_env``: shell variables, none of them the contrast."""
    legs_env = entry.get(WITNESS_LEGS_ENV_KEY)
    if legs_env is None:
        return
    key = f"{where}.{WITNESS_LEGS_ENV_KEY}"
    if not isinstance(legs_env, dict) or not legs_env:
        errors.append(f"'{key}' must be a non-empty mapping from variable to value")
        return
    for name, value in legs_env.items():
        if not isinstance(name, str) or not _ENV_NAME.fullmatch(name):
            errors.append(f"'{key}' names {name!r}, which is not a shell variable name")
        if not isinstance(value, (str, int, float, bool)):
            errors.append(f"'{key}.{name}' must be a scalar the shell can carry")
    contrast_env = entry.get("contrast_env")
    if isinstance(contrast_env, dict):
        shared = sorted(str(name) for name in set(legs_env) & set(contrast_env))
        if shared:
            errors.append(
                f"'{key}' and '{where}.contrast_env' both set {', '.join(shared)}: a variable "
                f"leg B sets differently is the contrast, not something the legs share"
            )


def _validate_equivalence_witness(entry: Mapping[str, Any], errors: list[str], where: str) -> None:
    """Validate an equivalence witness block: leg B must be its own launcher."""
    reject_unknown_keys(entry, EQUIVALENCE_WITNESS_KEYS, errors, where=where)
    _validate_contrast_env(entry, errors, where)
    _validate_legs_env(entry, errors, where)
    command = entry.get("contrast_command")
    if not isinstance(command, str) or not command.strip():
        errors.append(
            f"'{where}.contrast_command' is required and must be a non-empty string: leg B "
            f"is a context engine, a generation engine and a proxy, which no environment "
            f"variable over the aggregated command becomes, so without it the witness "
            f"compares a run to itself"
        )
    lengths = entry.get("prompt_tokens")
    if lengths is not None:
        if not isinstance(lengths, list) or not lengths:
            errors.append(f"'{where}.prompt_tokens' must be a non-empty list of integers")
        else:
            for value in lengths:
                if not _positive_int(value):
                    errors.append(
                        f"'{where}.prompt_tokens' entries must be integers >= 1, got {value!r}"
                    )
                elif value > MAX_WITNESS_PROMPT_WORDS:
                    errors.append(
                        f"'{where}.prompt_tokens' entry {value} is over "
                        f"{MAX_WITNESS_PROMPT_WORDS}; each prompt travels as one argv string, "
                        f"which Linux caps at 128 KiB"
                    )
    _validate_witness_timeout(entry, errors, where)
    _validate_witness_batch_size(entry, errors, where)
    _validate_witness_workload(entry, errors, where)


def _validate_replay_witness(
    feature: str, entry: Mapping[str, Any], errors: list[str], where: str
) -> None:
    """Validate a bounded-replay witness block: the workload is DECLARED here.

    Every size check below exists because the opposite value turns the witness
    into one that passes for a target that implements nothing: a prompt no
    longer than the window makes replay-on and replay-off do identical work, so
    "the decoder saw at most `window` rows" is true of both.
    """
    reject_unknown_keys(
        entry,
        (*REPLAY_WITNESS_REQUIRED[feature], *REPLAY_WITNESS_OPTIONAL[feature]),
        errors,
        where=where,
    )
    for key in REPLAY_WITNESS_REQUIRED[feature]:
        if entry.get(key) is None:
            errors.append(
                f"'{where}.{key}' is required: the {feature} witness sizes its workload "
                f"from the spec, never from the run it judges"
            )
    _validate_contrast_env(entry, errors, where)

    window = entry.get("window")
    if window is not None and not _positive_int(window):
        errors.append(f"'{where}.window' must be an integer >= 1, got {window!r}")
        window = None
    for key in ("prompt_tokens", "prefix_tokens"):
        value = entry.get(key)
        if value is None or key not in REPLAY_WITNESS_REQUIRED[feature]:
            continue
        if not _positive_int(value):
            errors.append(f"'{where}.{key}' must be an integer >= 1, got {value!r}")
        elif window is not None and value <= window:
            errors.append(
                f"'{where}.{key}' is {value}, which does not exceed window={window}. At or "
                f"under the window, replay on and replay off do the same work, and the "
                f"witness passes for a target that implements nothing."
            )
        elif value > MAX_WITNESS_PROMPT_WORDS:
            errors.append(
                f"'{where}.{key}' is {value}; the witness prompt travels as one argv "
                f"string, which Linux caps at 128 KiB, so the limit is "
                f"{MAX_WITNESS_PROMPT_WORDS}"
            )
    alignment = entry.get("reuse_alignment")
    if alignment is not None and not _positive_int(alignment):
        errors.append(f"'{where}.reuse_alignment' must be an integer >= 1, got {alignment!r}")
    _validate_witness_timeout(entry, errors, where)


def _validate_kv_quant_witness(entry: dict[str, Any], errors: list[str], where: str) -> None:
    """Validate the ``kv_quant`` witness block, filling ``exact``'s default.

    ``exact`` is written back explicitly because it changes what the unit
    claims, and the resolved spec in the workspace is what a reader checks: a
    unit gated without its identity leg should say so there, not by omission.
    """
    reject_unknown_keys(entry, KV_QUANT_WITNESS_KEYS, errors, where=where)
    expected = entry.get("expected_bytes_per_token")
    if not _positive_int(expected):
        errors.append(
            f"'{where}.expected_bytes_per_token' is required and must be an integer >= 1, "
            f"computed from the checkpoint's storage formats before the run; got {expected!r}"
        )
    exact = entry.setdefault("exact", True)
    if not isinstance(exact, bool):
        errors.append(f"'{where}.exact' must be true or false, got {exact!r}")
    elif exact or entry.get("contrast_env") is not None:
        # The identity leg needs a second storage to compare against; without
        # one it compares native storage with itself and passes.
        _validate_contrast_env(entry, errors, where)
    tolerance = entry.get("bytes_tolerance")
    if tolerance is not None and (
        not isinstance(tolerance, (int, float))
        or isinstance(tolerance, bool)
        or not 0 <= tolerance < 1
    ):
        errors.append(f"'{where}.bytes_tolerance' must be a fraction in [0, 1), got {tolerance!r}")
    _validate_witness_timeout(entry, errors, where)
    _validate_witness_batch_size(entry, errors, where)
    _validate_witness_workload(entry, errors, where)


def load_prompts_file(path: str | Path) -> tuple[str, ...]:
    """Read a prompt list for the acceptance gate's declared workload.

    Two shapes, both already on disk in a run: a YAML or JSON **list** of
    strings (a harness's ``prompts.json``), or a **mapping** with a
    ``prompts:`` list (``boot.yaml``'s shape). Raises
    :class:`TaskSchemaError` naming the file for anything else, including an
    empty list: a workload with no prompts measures nothing.
    """
    source = Path(path)
    try:
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TaskSchemaError(f"cannot read the prompts file {source}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise TaskSchemaError(f"the prompts file {source} is not valid YAML/JSON: {exc}") from exc
    if isinstance(data, dict):
        data = data.get("prompts")
    if not isinstance(data, list) or not data:
        raise TaskSchemaError(
            f"the prompts file {source} must hold a non-empty list of strings, or a "
            f"mapping with one under 'prompts'"
        )
    if not all(isinstance(p, str) and p.strip() for p in data):
        raise TaskSchemaError(f"every prompt in {source} must be a non-empty string")
    return tuple(data)


def _validate_acceptance_workload(entry: Mapping[str, Any], errors: list[str], where: str) -> None:
    """Validate the workload keys one level of ``accuracy.acceptance`` declares."""
    prompts = entry.get("prompts")
    prompts_file = entry.get("prompts_file")
    if prompts is not None and prompts_file is not None:
        errors.append(
            f"'{where}' declares both 'prompts' and 'prompts_file'; keep one, so the "
            f"workload a verdict was measured on has a single source"
        )
    if prompts is not None and (
        not isinstance(prompts, list)
        or not prompts
        or not all(isinstance(p, str) and p.strip() for p in prompts)
    ):
        errors.append(f"'{where}.prompts' must be a non-empty list of non-empty strings")
    if prompts_file is not None:
        if not isinstance(prompts_file, str) or not prompts_file.strip():
            errors.append(f"'{where}.prompts_file' must be a non-empty path")
        elif not Path(prompts_file).is_file():
            errors.append(f"'{where}.prompts_file' does not exist: {prompts_file}")
        else:
            try:
                load_prompts_file(prompts_file)
            except TaskSchemaError as exc:
                errors.append(f"'{where}.prompts_file': {exc}")
    sources = [k for k in ("prompts", "prompts_file", "dataset") if entry.get(k) is not None]
    if "dataset" in sources and len(sources) > 1:
        errors.append(
            f"'{where}' declares 'dataset' beside {sources[0]!r}; keep one, so the workload "
            f"a verdict was measured on has a single source"
        )
    dataset = entry.get("dataset")
    if dataset is not None and (not isinstance(dataset, str) or not dataset.strip()):
        errors.append(
            f"'{where}.dataset' must be a local path or a cached dataset id (no download)"
        )
    for key in ("text_field", "prompt_template", "split", "dataset_config"):
        value = entry.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            errors.append(f"'{where}.{key}' must be a non-empty string")
    template = entry.get("prompt_template")
    if isinstance(template, str) and "{text}" not in template:
        errors.append(f"'{where}.prompt_template' must name the question as {{text}}")
    for key in ("max_tokens", "min_draft_iterations", "num_prompts", "min_prompts"):
        value = entry.get(key)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 1
        ):
            errors.append(f"'{where}.{key}' must be an integer >= 1")


def _validate_acceptance_regression(block: Any, errors: list[str], where: str) -> None:
    """Validate ``accuracy.acceptance.regression``."""
    if not isinstance(block, dict):
        errors.append(f"'{where}' must be a mapping with keys {list(ACCEPTANCE_REGRESSION_KEYS)}")
        return
    reject_unknown_keys(block, ACCEPTANCE_REGRESSION_KEYS, errors, where=where)
    test = block.get("test")
    if test is not None and test not in stats.PAIRED_TESTS:
        errors.append(f"'{where}.test' must be one of {stats.PAIRED_TESTS}, got {test!r}")
    alpha = block.get("alpha")
    if alpha is not None and (not _number(alpha) or not 0 < alpha < 1):
        errors.append(f"'{where}.alpha' must be a probability in (0, 1), got {alpha!r}")
    effect = block.get("min_effect")
    if effect is not None and (not _number(effect) or effect < 0):
        errors.append(f"'{where}.min_effect' must be a number >= 0, got {effect!r}")


def _validate_acceptance(block: Any, errors: list[str]) -> None:
    """Validate ``accuracy.acceptance``: per-unit floors plus the gate's workload.

    Every key that is not a workload key names a unit, and its value is the
    unit's floor -- a number, or a mapping with ``min_absolute`` -- optionally
    with that unit's own workload beside it.
    """
    where = f"{ACCURACY_FIELD}.acceptance"
    if not isinstance(block, dict):
        errors.append(
            f"'{where}' must be a mapping from unit to floor, plus the optional "
            f"workload keys {list(ACCEPTANCE_WORKLOAD_KEYS)}"
        )
        return
    _validate_acceptance_workload(block, errors, where)
    if block.get("regression") is not None:
        _validate_acceptance_regression(block["regression"], errors, f"{where}.regression")
    for unit, entry in block.items():
        if unit in ACCEPTANCE_BLOCK_KEYS:
            continue
        unit_where = f"{where}.{unit}"
        if isinstance(entry, dict):
            reject_unknown_keys(entry, ACCEPTANCE_UNIT_KEYS, errors, where=unit_where)
            _validate_acceptance_workload(entry, errors, unit_where)
            floor = entry.get("min_absolute")
            floor_where = f"{unit_where}.min_absolute"
            measured = entry.get("measured_reference")
            if measured is not None and (not _number(measured) or measured < 1.0):
                errors.append(
                    f"'{unit_where}.measured_reference' must be a number >= 1.0; got {measured!r}"
                )
            source = entry.get("floor_source")
            if source is not None and (not isinstance(source, str) or not source.strip()):
                errors.append(f"'{unit_where}.floor_source' must be a non-empty string")
        else:
            floor = entry
            floor_where = unit_where
        if floor is None:
            continue
        # Acceptance length counts the target's own token on every step, so
        # it is at least 1.0 by construction and a floor below that passes
        # everything.
        if not isinstance(floor, (int, float)) or isinstance(floor, bool) or floor < 1.0:
            errors.append(f"'{floor_where}' must be a number >= 1.0; got {floor!r}")


def _validate_boot_level(level: Mapping[str, Any], errors: list[str], where: str) -> None:
    """Validate one level of ``accuracy.boot``."""
    prompts = level.get("prompts")
    if prompts is not None:
        if not isinstance(prompts, list) or not prompts:
            errors.append(f"'{where}.prompts' must be a non-empty list")
        else:
            for n, entry in enumerate(prompts):
                at = f"{where}.prompts[{n}]"
                if isinstance(entry, str):
                    if not entry.strip():
                        errors.append(f"'{at}' must be a non-empty string")
                    continue
                if not isinstance(entry, dict):
                    errors.append(f"'{at}' must be a string or a mapping")
                    continue
                reject_unknown_keys(entry, BOOT_PROMPT_KEYS, errors, where=at)
                if not isinstance(entry.get("prompt"), str) or not entry["prompt"].strip():
                    errors.append(f"'{at}.prompt' is required and must be a non-empty string")
                for key in ("keywords", "keywords_any", "images"):
                    value = entry.get(key)
                    if value is not None and (
                        not isinstance(value, list)
                        or not all(isinstance(v, str) and v.strip() for v in value)
                    ):
                        errors.append(f"'{at}.{key}' must be a list of non-empty strings")
                if entry.get("render") is not None and entry["render"] not in BOOT_RENDERS:
                    errors.append(f"'{at}.render' must be one of {BOOT_RENDERS}")
    prompts_file = level.get("prompts_file")
    if prompts_file is not None:
        if not isinstance(prompts_file, str) or not Path(prompts_file).is_file():
            errors.append(f"'{where}.prompts_file' does not exist: {prompts_file}")
    keywords = level.get("keywords")
    if keywords is not None and (
        not isinstance(keywords, list) or not all(isinstance(k, str) for k in keywords)
    ):
        errors.append(f"'{where}.keywords' must be a list of strings")
    tokens = level.get("max_tokens")
    if tokens is not None and not _positive_int(tokens):
        errors.append(f"'{where}.max_tokens' must be an integer >= 1")


def _validate_boot(block: Any, errors: list[str]) -> None:
    """Validate ``accuracy.boot``: the run-wide keys, then each unit's entry."""
    where = f"{ACCURACY_FIELD}.boot"
    if not isinstance(block, dict):
        errors.append(f"'{where}' must be a mapping")
        return
    _validate_boot_level(block, errors, where)
    for unit, entry in block.items():
        if unit in BOOT_KEYS:
            continue
        if not isinstance(entry, dict):
            errors.append(
                f"'{where}.{unit}' is neither a boot key ({list(BOOT_KEYS)}) nor a unit's "
                f"mapping of them"
            )
            continue
        reject_unknown_keys(entry, BOOT_KEYS, errors, where=f"{where}.{unit}")
        _validate_boot_level(entry, errors, f"{where}.{unit}")


def _validate_launch_level(level: Mapping[str, Any], errors: list[str], where: str) -> None:
    """Validate one level of ``accuracy.launch``."""
    for key in ("enabled",):
        if level.get(key) is not None and not isinstance(level[key], bool):
            errors.append(f"'{where}.{key}' must be true or false")
    for key in ("num_samples", "max_tokens", "timeout"):
        if level.get(key) is not None and not _positive_int(level[key]):
            errors.append(f"'{where}.{key}' must be an integer >= 1")
    if level.get("members") is not None and level["members"] not in ("all", "first"):
        errors.append(f"'{where}.members' must be 'all' or 'first'")
    if level.get("endpoint") is not None and level["endpoint"] not in LAUNCH_ENDPOINTS:
        errors.append(f"'{where}.endpoint' must be one of {LAUNCH_ENDPOINTS}")
    if level.get("scorer") is not None and level["scorer"] not in LAUNCH_SCORERS:
        errors.append(f"'{where}.scorer' must be one of {LAUNCH_SCORERS}")
    floor = level.get("min_accuracy")
    if floor is not None and (not _number(floor) or not 0 <= floor <= 100):
        errors.append(f"'{where}.min_accuracy' must be a percentage in [0, 100]")
    for key in (
        "dataset",
        "text_field",
        "answer_field",
        "prompt_template",
        "split",
        "dataset_config",
        "command",
    ):
        value = level.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            errors.append(f"'{where}.{key}' must be a non-empty string")
    units = level.get("units")
    if units is not None and (
        not isinstance(units, list) or not all(isinstance(u, str) for u in units)
    ):
        errors.append(f"'{where}.units' must be a list of unit names")


def _validate_launch(block: Any, errors: list[str]) -> None:
    """Validate ``accuracy.launch``: the run-wide keys, then each unit's entry."""
    where = f"{ACCURACY_FIELD}.launch"
    if not isinstance(block, dict):
        errors.append(f"'{where}' must be a mapping")
        return
    _validate_launch_level(block, errors, where)
    for unit, entry in block.items():
        if unit in LAUNCH_KEYS:
            continue
        if not isinstance(entry, dict):
            errors.append(
                f"'{where}.{unit}' is neither a launch key ({list(LAUNCH_KEYS)}) nor a unit's "
                f"mapping of them"
            )
            continue
        reject_unknown_keys(entry, LAUNCH_KEYS, errors, where=f"{where}.{unit}")
        _validate_launch_level(entry, errors, f"{where}.{unit}")


def _validate_scheduler_equivalence(block: Any, errors: list[str]) -> None:
    """Validate ``accuracy.scheduler_equivalence``."""
    where = f"{ACCURACY_FIELD}.scheduler_equivalence"
    if not isinstance(block, dict):
        errors.append(f"'{where}' must be a mapping with keys {list(SCHEDULER_EQUIVALENCE_KEYS)}")
        return
    reject_unknown_keys(block, SCHEDULER_EQUIVALENCE_KEYS, errors, where=where)
    _validate_acceptance_workload(
        {k: v for k, v in block.items() if k in ACCEPTANCE_WORKLOAD_KEYS and k != "max_tokens"},
        errors,
        where,
    )
    for key in ("enabled", "ignore_eos", "pin_autotuner"):
        if block.get(key) is not None and not isinstance(block[key], bool):
            errors.append(f"'{where}.{key}' must be true or false")
    units = block.get("units")
    if units is not None and (
        not isinstance(units, list) or not all(isinstance(u, str) and u for u in units)
    ):
        errors.append(f"'{where}.units' must be a list of unit names")
    for key in ("min_generated_tokens", "agreement_tokens", "timeout"):
        if block.get(key) is not None and not _positive_int(block[key]):
            errors.append(f"'{where}.{key}' must be an integer >= 1")
    tokens = block.get("max_tokens")
    if tokens is not None and (not _positive_int(tokens) or tokens < 64):
        errors.append(
            f"'{where}.max_tokens' must be an integer >= 64: a defect in generation steps "
            f"needs steps to show in"
        )
    legs = block.get("legs")
    if legs is not None and (
        not isinstance(legs, list) or not set(legs) <= set(SCHEDULER_EQUIVALENCE_LEGS)
    ):
        errors.append(
            f"'{where}.legs' must be a list drawn from {list(SCHEDULER_EQUIVALENCE_LEGS)}"
        )
    pair = block.get("spec_pair")
    if pair is not None and pair not in SCHEDULER_EQUIVALENCE_PAIRS:
        errors.append(f"'{where}.spec_pair' must be one of {SCHEDULER_EQUIVALENCE_PAIRS}")
    trace = block.get("step_trace")
    if trace is not None:
        if not isinstance(trace, dict):
            errors.append(f"'{where}.step_trace' must be a mapping")
        else:
            reject_unknown_keys(trace, ("reference_command",), errors, where=f"{where}.step_trace")
            command = trace.get("reference_command")
            if command is not None and (not isinstance(command, str) or not command.strip()):
                errors.append(f"'{where}.step_trace.reference_command' must be a non-empty string")


def _validate_floor(block: Mapping[str, Any], errors: list[str]) -> None:
    """Validate ``accuracy.floor`` and that its rule can be computed for every unit.

    The statistical rule needs a sample count per benchmark. Asked here
    rather than at the gate: a floor that cannot be computed is found at the
    end of a unit's first accuracy run otherwise, which is the most expensive
    place to learn that a number was never declared.
    """
    where = f"{ACCURACY_FIELD}.floor"
    floor = block.get("floor")
    if floor is not None:
        if not isinstance(floor, dict):
            errors.append(f"'{where}' must be a mapping with keys {list(FLOOR_KEYS)}")
            floor = {}
        reject_unknown_keys(floor, FLOOR_KEYS, errors, where=where)
        rule = floor.get("rule")
        if rule is not None and rule not in stats.FLOOR_RULES:
            errors.append(f"'{where}.rule' must be one of {stats.FLOOR_RULES}, got {rule!r}")
        k = floor.get("k")
        if k is not None and (not _number(k) or k <= 0):
            errors.append(f"'{where}.k' must be a number > 0, got {k!r}")
        members = floor.get("shift_min_members")
        if members is not None and not _positive_int(members):
            errors.append(f"'{where}.shift_min_members' must be an integer >= 1, got {members!r}")
        acknowledge = floor.get("acknowledge")
        if acknowledge is not None and (
            not isinstance(acknowledge, dict)
            or not all(isinstance(v, str) and v.strip() for v in acknowledge.values())
        ):
            errors.append(
                f"'{where}.acknowledge' must map a unit to the reason its systematic shift "
                f"is accepted, as a non-empty string"
            )
    else:
        floor = {}

    samples = block.get("samples")
    if samples is not None:
        values = samples.values() if isinstance(samples, dict) else [samples]
        if not all(_positive_int(v) for v in values):
            errors.append(
                f"'{ACCURACY_FIELD}.samples' must be an integer >= 1, or a mapping from unit "
                f"to one; got {samples!r}"
            )
            return

    if (floor.get("rule") or stats.FLOOR_STATISTICAL) != stats.FLOOR_STATISTICAL:
        return
    # Imported here: `gates` imports this module.
    from .gates import resolve_samples

    benchmarks = block.get("benchmark")
    pairs = (
        list(benchmarks.items())
        if isinstance(benchmarks, dict)
        else [(None, benchmarks or "gsm8k")]
    )
    for unit, benchmark in pairs:
        if not isinstance(benchmark, str):
            continue
        count, source = resolve_samples(block, None if unit == "default" else unit, benchmark)
        if count is None:
            errors.append(
                f"'{ACCURACY_FIELD}' uses the statistical floor, and there is {source} "
                f"(unit {unit or 'every unit'}); the floor is anchor - k * SE and the SE "
                f"needs it"
            )


def _validate_accuracy(block: Any, errors: list[str]) -> None:
    """Validate the ``accuracy`` block in place, appending to ``errors``."""
    if not isinstance(block, dict):
        raise TaskSchemaError(f"'{ACCURACY_FIELD}' must be a mapping, got {type(block).__name__}")

    reject_unknown_keys(block, ACCURACY_KEYS, errors, where=ACCURACY_FIELD)

    # Both of these accept the per-unit mapping form, for the same reason
    # `llm_api_options` does. The `multimodal` increment's release criterion
    # is a multimodal benchmark in a different reference file; gating it on
    # the text benchmark core was gated on is not a weaker check, it is a
    # check that cannot see the vision tower and passes whatever it does.
    for name in ("benchmark", "reference_yaml"):
        value = block.get(name)
        if isinstance(value, str):
            if not value.strip():
                errors.append(f"'{ACCURACY_FIELD}.{name}' must be a non-empty string")
            continue
        if isinstance(value, dict):
            if "default" not in value:
                errors.append(
                    f"'{ACCURACY_FIELD}.{name}' is a per-unit mapping, so it needs a "
                    f"'default' key -- otherwise a unit nobody listed silently has no "
                    f"{name} and the gate falls back to a hard-coded one"
                )
            for unit, entry in value.items():
                if not isinstance(entry, str) or not entry.strip():
                    errors.append(f"'{ACCURACY_FIELD}.{name}[{unit}]' must be a non-empty string")
            continue
        errors.append(
            f"'{ACCURACY_FIELD}.{name}' must be a non-empty string, or a mapping "
            f"from assembly unit to one"
        )

    tolerance = block.get("tolerance")
    if tolerance is None:
        block["tolerance"] = DEFAULT_ACCURACY_TOLERANCE
    elif not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
        errors.append(f"'{ACCURACY_FIELD}.tolerance' must be a number")
    elif tolerance < 0:
        errors.append(f"'{ACCURACY_FIELD}.tolerance' must be >= 0; got {tolerance}")

    # ``dataset_path`` is optional but load-bearing for MMLU: trtllm-eval
    # fetches that one dataset itself from a URL that now 404s, writes the
    # error page to data.tar, and dies inside tarfile.open — so the failure
    # reads as a corrupt archive rather than a dead link. Every other
    # benchmark comes through lm-eval and is unaffected.
    dataset_path = block.get("dataset_path")
    if dataset_path is not None and (not isinstance(dataset_path, str) or not dataset_path.strip()):
        errors.append(f"'{ACCURACY_FIELD}.dataset_path' must be a non-empty string when present")

    # `trtllm-eval` takes none of the quickstart runtime flags, so the
    # configuration the gate measures under has to arrive as a YAML. It must
    # exist: a missing file means the gate silently measures the engine's
    # defaults, which is how a run compares a number against an anchor taken
    # under a different configuration and calls it a regression.
    options = block.get("llm_api_options")
    # A mapping unit -> path, because a feature unit is not measured under
    # the same configuration as core: MTP needs a `speculative_config`, and
    # without one the accuracy gate re-measures the base path while the
    # acceptance gate compares 1.0 against 1.0.
    if isinstance(options, dict):
        for unit, path in options.items():
            if not isinstance(path, str) or not Path(path).is_file():
                errors.append(f"'{ACCURACY_FIELD}.llm_api_options[{unit}]' does not exist: {path}")
        options = None
    if options is not None:
        if not isinstance(options, str) or not options.strip():
            errors.append(f"'{ACCURACY_FIELD}.llm_api_options' must be a non-empty path")
        elif not Path(options).is_file():
            errors.append(
                f"'{ACCURACY_FIELD}.llm_api_options' does not exist: {options}. Without it "
                f"the gate measures the engine's defaults, not the configuration the "
                f"anchor was measured under."
            )

    for name in ("max_seq_len", "max_batch_size"):
        value = block.get(name)
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            errors.append(f"'{ACCURACY_FIELD}.{name}' must be an integer >= 1")

    if block.get("acceptance") is not None:
        _validate_acceptance(block["acceptance"], errors)

    _validate_floor(block, errors)

    if block.get("scheduler_equivalence") is not None:
        _validate_scheduler_equivalence(block["scheduler_equivalence"], errors)

    if block.get("boot") is not None:
        _validate_boot(block["boot"], errors)

    if block.get("launch") is not None:
        _validate_launch(block["launch"], errors)

    witness = block.get("witness")
    if witness is not None:
        if not isinstance(witness, dict):
            errors.append(f"'{ACCURACY_FIELD}.witness' must be a mapping from unit to a block")
        else:
            for unit, entry in witness.items():
                where = f"{ACCURACY_FIELD}.witness.{unit}"
                if not isinstance(entry, dict):
                    errors.append(f"'{where}' must be a mapping")
                    continue
                # Closed per feature: `window` under `host_offload` is a key
                # nothing reads, and `expected_mib` under a replay feature reads
                # as a check that is not being made.
                if unit in REPLAY_WITNESS_REQUIRED:
                    _validate_replay_witness(str(unit), entry, errors, where)
                    continue
                if unit == "kv_quant":
                    _validate_kv_quant_witness(entry, errors, where)
                    continue
                if unit in EQUIVALENCE_WITNESS_FEATURES:
                    _validate_equivalence_witness(entry, errors, where)
                    continue
                reject_unknown_keys(entry, WITNESS_KEYS, errors, where=where)
                _validate_witness_timeout(entry, errors, where)
                _validate_witness_batch_size(entry, errors, where)
                _validate_witness_workload(entry, errors, where)
                command = entry.get("contrast_command")
                if command is not None and (not isinstance(command, str) or not command.strip()):
                    errors.append(f"'{where}.contrast_command' must be a non-empty string")
                _validate_contrast_env(entry, errors, where)
                # A scalar, or per member with a `default`: the witness runs
                # at every member, and the delta can differ between them.
                predicted = entry.get("expected_mib")
                values = predicted.values() if isinstance(predicted, dict) else [predicted]
                if predicted is not None and (
                    not all(_number(v) and v > 0 for v in values)
                    or (isinstance(predicted, dict) and not predicted)
                ):
                    errors.append(
                        f"'{where}.expected_mib' must be a positive number, or a mapping from "
                        f"member (and `default`) to one"
                    )

    # A mapping of scalars (the whole run) or of unit -> mapping of scalars.
    # Validated rather than trusted because a discriminator the reference file
    # does not use produces `no entry matching {...}` at the END of an
    # assembly, which is the most expensive place to learn about a typo.
    discriminators = block.get("discriminators")
    if discriminators is not None:
        if not isinstance(discriminators, dict):
            errors.append(f"'{ACCURACY_FIELD}.discriminators' must be a mapping")
        else:
            nested = [v for v in discriminators.values() if isinstance(v, dict)]
            if nested and len(nested) != len(discriminators):
                errors.append(
                    f"'{ACCURACY_FIELD}.discriminators' mixes the flat form "
                    f"(quant_algo: ...) with the per-unit form (core: {{...}}); pick one"
                )
            for group in nested or [discriminators]:
                for key, value in group.items():
                    if not isinstance(value, (str, int, float, bool)):
                        errors.append(
                            f"'{ACCURACY_FIELD}.discriminators.{key}' must be a scalar "
                            f"matching a reference-file discriminator, got "
                            f"{type(value).__name__}"
                        )

    # The evaluation *protocol* -- chat template, few-shot handling, seed --
    # is expressed as options of the benchmark subcommand, so it cannot live
    # in the llm_api_options YAML. It has to be declared, because a protocol
    # mismatch between the anchor and the gate is indistinguishable from a
    # 27-point accuracy regression, and that is measured, not hypothetical.
    benchmark_args = block.get("benchmark_args")

    def _is_arg_list(value: Any) -> bool:
        return isinstance(value, list) and all(isinstance(a, str) for a in value)

    if benchmark_args is None:
        block["benchmark_args"] = []
    elif isinstance(benchmark_args, dict):
        # Per unit for a harder reason than the other fields: these are
        # options of the BENCHMARK subcommand, and the subcommands do not
        # share an option set. `--apply_chat_template` is a gsm8k option and
        # not an mmmu one, so a run-wide list makes the multimodal unit's
        # gate fail to start rather than fail to measure.
        if "default" not in benchmark_args:
            errors.append(
                f"'{ACCURACY_FIELD}.benchmark_args' is a per-unit mapping, so it needs "
                f"a 'default' key"
            )
        for unit, value in benchmark_args.items():
            if not _is_arg_list(value):
                errors.append(
                    f"'{ACCURACY_FIELD}.benchmark_args[{unit}]' must be a list of strings"
                )
    elif not _is_arg_list(benchmark_args):
        errors.append(
            f"'{ACCURACY_FIELD}.benchmark_args' must be a list of strings, or a "
            f"mapping from assembly unit to one"
        )


def _require_launch_command(data: Mapping[str, Any], errors: list[str]) -> None:
    """The disagg unit's deployment entry is a launcher the spec must declare.

    Asked here because the launch-path smoke runs at the END of the unit: an
    undeclared launcher found there costs the unit's whole gate sequence.
    """
    features = data.get(FEATURES_FIELD)
    accuracy = data.get(ACCURACY_FIELD)
    if not isinstance(features, list) or "disagg" not in features or not isinstance(accuracy, dict):
        return
    launch = accuracy.get("launch")
    launch = launch if isinstance(launch, dict) else {}
    own = launch.get("disagg") if isinstance(launch.get("disagg"), dict) else {}
    enabled = own.get("enabled", launch.get("enabled", True))
    units = own.get("units", launch.get("units"))
    if not enabled or (units and "disagg" not in units):
        return
    if not (own.get("command") or launch.get("command")):
        errors.append(
            "feature 'disagg' needs 'accuracy.launch.disagg.command': the unit's launch-path "
            "smoke goes through the disaggregated proxy, which trtllm-serve alone does not "
            "start. Give a launcher that prints STAIRCASE_ENDPOINT=<base url> once it serves "
            "(or set 'accuracy.launch.disagg.enabled: false' and say why in the plan)"
        )


def _require_declared_witnesses(data: Mapping[str, Any], errors: list[str]) -> None:
    """A feature whose witness judges against a DECLARED value needs it here.

    Here and not at the unit boundary, where `host_offload` and `disagg` are
    checked: these witnesses compare the target's own counters against a
    workload size or a byte figure, and the one place such a declaration is
    certainly made before the run it judges is this one. Found missing at the
    gate, it would be written after the assembler's turns -- and chosen,
    however honestly, by someone who has seen the target's numbers.
    """
    features = data.get(FEATURES_FIELD)
    accuracy = data.get(ACCURACY_FIELD)
    if not isinstance(features, list) or not isinstance(accuracy, dict):
        return
    witness = accuracy.get("witness")
    witness = witness if isinstance(witness, dict) else {}
    for feature in features:
        if feature not in DECLARED_WITNESS_FEATURES or feature in witness:
            continue
        keys = REPLAY_WITNESS_REQUIRED.get(feature) or ("expected_bytes_per_token", "exact")
        errors.append(
            f"feature {feature!r} needs 'accuracy.witness.{feature}' ({', '.join(keys)}): "
            f"a unit that implements nothing passes both boot and accuracy, so the "
            f"witness is the only gate that can tell, and what it judges against has to "
            f"be declared before the run."
        )


def _validate_certification(data: Mapping[str, Any], errors: list[str]) -> None:
    """Validate ``serving_envelope`` and ``feature_matrix``; see `envelope.py`.

    Imported here rather than at module scope: `envelope` consumes this
    schema's constants and helpers, so a top-level import would be circular.
    """
    from .envelope import validate_certification_blocks

    validate_certification_blocks(data, errors)


def _resolve_spec_relative_paths(data: dict[str, Any], spec_dir: Path) -> None:
    """Make paths the user writes NEXT TO the spec portable, in place.

    ``accuracy.llm_api_options``, ``accuracy.witness.*.boot_spec`` and
    ``accuracy.acceptance[.<unit>].prompts_file`` point at files the author
    writes alongside ``task.yaml`` — the per-unit runtime YAMLs, prompt
    sets. Written absolute, as they had to be, **the spec stops working on any
    other machine**, which is the opposite of what a spec is for. One run
    produced a 220-line task.yaml with six absolute paths in it; the most
    reusable artifact of that run was unusable anywhere else. This run's own
    ``mimo-v2.6-flash-onboard/task.yaml`` repeats it: every one of its five
    per-unit ``llm_api_options`` entries names a file that sits in a directory
    beside the spec, spelled from the filesystem root.

    Resolved against the **spec's own directory**, because that is where the
    author put them, and written back as absolute so every consumer downstream
    is unchanged — the gates take a string and pass it to ``trtllm-eval``, and
    none of them learns anything new.

    Absolute paths are untouched: a site that wants one keeps it, and a
    workspace copy of an already-resolved spec re-resolves to itself, which is
    the case that matters because the workflow copies ``task.yaml`` into the
    workspace and reads it again.

    Deliberately NOT applied to ``checkpoint_path``, ``trtllm_repo_path`` or
    ``environment.spool``. Those genuinely differ per machine and are read by
    processes that never see the spec's directory; making them relative would
    move the portability problem rather than solve it.
    """

    def absolute(value: Any) -> Any:
        if not isinstance(value, str) or not value.strip():
            return value
        candidate = Path(value)
        return value if candidate.is_absolute() else str((spec_dir / candidate).resolve())

    accuracy = data.get(ACCURACY_FIELD)
    if not isinstance(accuracy, dict):
        return

    options = accuracy.get("llm_api_options")
    if isinstance(options, dict):
        accuracy["llm_api_options"] = {k: absolute(v) for k, v in options.items()}
    elif options is not None:
        accuracy["llm_api_options"] = absolute(options)

    witness = accuracy.get("witness")
    if isinstance(witness, dict):
        for block in witness.values():
            if isinstance(block, dict) and block.get("boot_spec") is not None:
                block["boot_spec"] = absolute(block["boot_spec"])
            if isinstance(block, dict) and block.get("prompts_file") is not None:
                block["prompts_file"] = absolute(block["prompts_file"])

    # The acceptance workload's prompt file, at the block level and per unit.
    # A `dataset` may be a cached dataset id ("openai/gsm8k"), which reads like
    # a relative path, so it is resolved only when the spec's directory holds it.
    acceptance = accuracy.get("acceptance")
    if isinstance(acceptance, dict):
        for level in (acceptance, *(v for v in acceptance.values() if isinstance(v, dict))):
            if level.get("prompts_file") is not None:
                level["prompts_file"] = absolute(level["prompts_file"])
            dataset = level.get("dataset")
            if isinstance(dataset, str) and dataset.strip() and not Path(dataset).is_absolute():
                if (spec_dir / dataset).exists():
                    level["dataset"] = absolute(dataset)

    # Boot prompt files, and boot images that sit beside the spec. An image
    # that does not is left relative: the gate resolves it against the tree
    # under test, where the repository's own test assets are.
    boot = accuracy.get("boot")
    if isinstance(boot, dict):
        for level in (boot, *(v for v in boot.values() if isinstance(v, dict))):
            if level.get("prompts_file") is not None:
                level["prompts_file"] = absolute(level["prompts_file"])
            for entry in level.get("prompts") or ():
                if isinstance(entry, dict) and isinstance(entry.get("images"), list):
                    entry["images"] = [
                        absolute(i)
                        if isinstance(i, str)
                        and not Path(i).is_absolute()
                        and (spec_dir / i).exists()
                        else i
                        for i in entry["images"]
                    ]

    equivalence = accuracy.get("scheduler_equivalence")
    if isinstance(equivalence, dict):
        if equivalence.get("prompts_file") is not None:
            equivalence["prompts_file"] = absolute(equivalence["prompts_file"])
        dataset = equivalence.get("dataset")
        if isinstance(dataset, str) and dataset.strip() and (spec_dir / dataset).exists():
            equivalence["dataset"] = absolute(dataset)


def load_and_validate_task_yaml(path: str | Path) -> dict[str, Any]:
    """Parse ``path`` as YAML, validate it, and fill in the workflow's defaults.

    Returns the resolved mapping. Raises :class:`TaskSchemaError` with
    **every** detected problem batched into a single message.
    """
    task_path = Path(path)
    if not task_path.is_file():
        raise TaskSchemaError(f"task file not found: {task_path}")

    text = task_path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise TaskSchemaError(f"{task_path} is not valid YAML: {exc}") from exc

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TaskSchemaError(
            f"{task_path} must be a YAML mapping at the top level, got {type(data).__name__}"
        )

    # BEFORE validation, deliberately. `accuracy.llm_api_options` has an
    # existence check, and that check is a *consumer* of the resolved path:
    # resolving afterwards makes a perfectly good relative path fail as
    # "does not exist". Resolving first also means the error message names
    # the absolute path the run would really have used.
    _resolve_spec_relative_paths(data, task_path.resolve().parent)

    errors: list[str] = []

    # First, before anything reads a field: a key this schema does not know
    # is an error, not a no-op. Placed here so the message a user gets names
    # their typo rather than the consequence of it — `feature:` silently
    # resolving to no features reads downstream as "the run built core and
    # said it was done", which is a much longer trip back to the cause.
    _validate_remediation(data.get(REMEDIATION_FIELD), errors)
    validate_commit_policy_block(data.get(COMMIT_POLICY_FIELD), errors)
    reject_unknown_keys(data, TOP_LEVEL_KEYS, errors)
    errors.extend(validate_agents(data, ROLES))
    errors.extend(validate_role_defaults(data))

    for name in REQUIRED_PATH_FIELDS:
        if name not in data:
            errors.append(f"missing required field '{name}'")
            continue
        value = data[name]
        if not isinstance(value, str) or not value.strip():
            errors.append(f"'{name}' must be a non-empty string, got {type(value).__name__}")
            continue
        if not Path(value).exists():
            errors.append(f"'{name}' points to a non-existent path: {value}")

    arch = data.get(ARCH_FIELD)
    if arch is None:
        errors.append(
            f"missing required field '{ARCH_FIELD}'. It is never inferred: catalog "
            f"receipts are per-architecture, so the wrong value gates this target "
            f"against another machine's certification."
        )
    elif not isinstance(arch, str) or not ARCH_PATTERN.match(arch):
        errors.append(f"'{ARCH_FIELD}' must look like 'sm_100' / 'sm_103'; got {arch!r}")

    parallel = data.get(PARALLEL_FIELD)
    if parallel is None:
        data[PARALLEL_FIELD] = DEFAULT_PARALLEL
    elif not isinstance(parallel, str):
        errors.append(f"'{PARALLEL_FIELD}' must be a string, got {type(parallel).__name__}")
    else:
        try:
            parse_parallel_segment(parallel)
        except TaskSchemaError as exc:
            errors.append(str(exc))

    # After `parallel` has been defaulted, since legality depends on it.
    if MEMBERS_FIELD in data:
        _validate_members(data[MEMBERS_FIELD], data.get(PARALLEL_FIELD), errors)

    features = data.get(FEATURES_FIELD)
    if features is None:
        data[FEATURES_FIELD] = []
    elif not isinstance(features, list):
        errors.append(
            f"'{FEATURES_FIELD}' must be a list of strings, got {type(features).__name__}"
        )
    else:
        for index, feature in enumerate(features):
            if not isinstance(feature, str):
                errors.append(
                    f"'{FEATURES_FIELD}[{index}]' must be a string, got {type(feature).__name__}"
                )
            elif feature not in FEATURE_SKILLS:
                errors.append(
                    f"'{FEATURES_FIELD}[{index}]' is {feature!r}, which no assembler skill "
                    f"covers. Known features: {sorted(FEATURE_SKILLS)}"
                )
        readers = [f for f in MTP_NAMESPACE_FEATURES if f in features]
        if len(readers) > 1:
            errors.append(
                f"'{FEATURES_FIELD}' lists {readers}, and all of them read the checkpoint's "
                f"`mtp.*` weights, which are MTP modules or DSpark stages, never both. "
                f"Keep the one the checkpoint ships. An `mtp.*` that holds DSpark stages "
                f"(`dspark_*` keys in its config) is spelled `dspark`, even when the "
                f"request says 'MTP'."
            )

    if ENVIRONMENT_FIELD not in data:
        errors.append(
            f"missing required field '{ENVIRONMENT_FIELD}'. It declares how GPU work is "
            f"launched; without it the agents would have to discover that themselves, "
            f"which is exactly what this workflow refuses to do."
        )
    else:
        try:
            _validate_environment(data[ENVIRONMENT_FIELD], errors)
        except TaskSchemaError as exc:
            errors.append(str(exc))

    if TARGET_FIELD not in data:
        data[TARGET_FIELD] = {}
    else:
        try:
            _validate_target(data[TARGET_FIELD], errors)
        except TaskSchemaError as exc:
            errors.append(str(exc))

    if RUNTIME_FIELD not in data:
        data[RUNTIME_FIELD] = dict(RUNTIME_SWITCHES) | {"extra_flags": []}
    else:
        try:
            _validate_runtime(data[RUNTIME_FIELD], errors)
        except TaskSchemaError as exc:
            errors.append(str(exc))

    scope = data.get(SCOPE_FIELD)
    if scope is None:
        data[SCOPE_FIELD] = DEFAULT_SCOPE
    elif scope not in SCOPES:
        errors.append(f"'{SCOPE_FIELD}' must be one of {SCOPES}; got {scope!r}")

    if SOURCE_EMBARGO_FIELD not in data:
        data[SOURCE_EMBARGO_FIELD] = []
    else:
        try:
            _validate_source_embargo(
                data[SOURCE_EMBARGO_FIELD], errors, data.get("trtllm_repo_path")
            )
        except TaskSchemaError as exc:
            errors.append(str(exc))

    if ACCURACY_FIELD not in data:
        errors.append(
            f"missing required field '{ACCURACY_FIELD}'. The accuracy gate is the release "
            f"criterion, and its anchor comes from a written source — never from the run."
        )
    else:
        try:
            _validate_accuracy(data[ACCURACY_FIELD], errors)
        except TaskSchemaError as exc:
            errors.append(str(exc))
        _require_declared_witnesses(data, errors)
        _require_launch_command(data, errors)

    _validate_certification(data, errors)

    if errors:
        bullet = "\n  - "
        raise TaskSchemaError(
            f"{task_path} failed staircase-onboard schema validation:{bullet}{bullet.join(errors)}"
        )

    return data


def assembly_units(data: Mapping[str, Any]) -> list[str]:
    """Return the assembly units in dispatch order: ``core`` then each feature.

    Core is always first and always green before the first feature: the
    accuracy gate's tolerance is wide enough that a core defect and a
    feature defect are indistinguishable when they land together. It also
    means a first assembly with ``features`` follows exactly the same path
    as a core assembly plus later increments, so there is only one path to
    get right.
    """
    return ["core", *[str(f) for f in data.get(FEATURES_FIELD, [])]]


__all__ = [
    "ACCEPTANCE_UNIT_KEYS",
    "ACCEPTANCE_WORKLOAD_KEYS",
    "ACCURACY_FIELD",
    "ACCURACY_KEYS",
    "ARCH_FIELD",
    "ARCH_PATTERN",
    "DEFAULT_ACCURACY_TOLERANCE",
    "DEFAULT_MEMBER_KINDS",
    "DEFAULT_PARALLEL",
    "DEFAULT_SCOPE",
    "ENVIRONMENT_FIELD",
    "ENVIRONMENT_KEYS",
    "ENVIRONMENT_KINDS",
    "DECLARED_WITNESS_FEATURES",
    "DEFAULT_BYTES_TOLERANCE",
    "FEATURES_FIELD",
    "FEATURE_SKILLS",
    "KV_QUANT_WITNESS_KEYS",
    "MAX_WITNESS_PROMPT_WORDS",
    "MEMBERS_FIELD",
    "MTP_NAMESPACE_FEATURES",
    "REPLAY_WITNESS_OPTIONAL",
    "REPLAY_WITNESS_REQUIRED",
    "EQUIVALENCE_WITNESS_FEATURES",
    "EQUIVALENCE_WITNESS_KEYS",
    "FEATURE_MATRIX_FIELD",
    "SERVING_ENVELOPE_FIELD",
    "WITNESS_BATCH_SIZE_KEY",
    "WITNESS_KEYS",
    "WITNESS_LEGS_ENV_KEY",
    "PARALLEL_FIELD",
    "PARALLEL_UNIFIED",
    "REQUIRED_PATH_FIELDS",
    "RUNTIME_FIELD",
    "RUNTIME_SWITCHES",
    "SCOPES",
    "SCOPE_FIELD",
    "SLURM_REQUIRED_FIELDS",
    "reject_unknown_keys",
    "SOURCE_EMBARGO_FIELD",
    "TARGET_FIELD",
    "TARGET_KEYS",
    "TOP_LEVEL_KEYS",
    "TaskSchemaError",
    "assembly_units",
    "derive_checkpoint_segment",
    "derive_family",
    "derive_slug",
    "load_and_validate_task_yaml",
    "load_prompts_file",
    "members_for",
    "parse_parallel_segment",
]
