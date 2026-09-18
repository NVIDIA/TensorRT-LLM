# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-denoising-step activation precision for static-FP8 Cosmos3.

A ModelOpt-calibrated checkpoint carries one activation scale per projection,
taken as a max over the whole sampling trajectory. That single scale fits the
outer denoising steps worst, so this module lets those steps run the resident
FP8 weights through a 16-bit GEMM instead: the weight is dequantized with its
own ``weight_scale`` and ``input_scale`` goes unused. Middle steps keep the
checkpoint's fully quantized path.

No extra weights are read -- the weights and scales are the ones already
loaded, and no second checkpoint or persistent dequantized copy is kept. What
the checkpoint does supply is the policy itself, under
``quantization_config.runtime.diffusion_step_policy``: which steps take the
16-bit path, and what the understanding tower does. Only the builds whose
calibration needs it carry one, so a checkpoint without a policy runs fully
quantized and there is no default to substitute.

Precision is selected once per denoising step, before any transformer call for
that step, so the conditional and unconditional CFG branches of one step always
agree.
"""

from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Tuple

import torch

from tensorrt_llm._torch.modules.linear import FP8QDQLinearMethod, Linear, W8A16FP8LinearMethod

# The policy the checkpoint publishes under
# ``quantization_config.runtime.diffusion_step_policy``. Every field is
# required and every value is checked: a policy shape we do not implement must
# fail loudly rather than be half-honoured, since silently ignoring a field the
# producer set is indistinguishable from the feature not working.
_POLICY_FIELDS = frozenset(
    {
        "schema_version",
        "type",
        "index_space",
        "scope",
        "default_mode",
        "first_steps",
        "last_steps",
        "overlap",
        "reasoner",
    }
)
_STEP_RANGE_FIELDS = frozenset({"count", "mode"})
_SCOPE_COMPONENT = "transformer"


@dataclass(frozen=True)
class StepPrecisionPolicy:
    """A validated ``diffusion_step_policy``.

    ``reasoner`` is not step-scoped. The understanding tower runs once per
    request -- on the first transformer call, then cached -- so a step-indexed
    rule would describe it only by accident of which step that call lands on.
    The policy states its precision directly instead.
    """

    first_steps: int
    last_steps: int
    reasoner_high_precision: bool


def parse_diffusion_step_policy(quantization_config: Any) -> Optional[StepPrecisionPolicy]:
    """Read the checkpoint's step policy, or None if it declares none.

    Absence is meaningful, not a default to fill in: the producer ships this
    only for the checkpoints whose calibration needs it (the multi-step video
    builds), and deliberately omits it for the image and distilled 4-step
    builds, whose output does not show the artifact it targets.
    """
    if not isinstance(quantization_config, Mapping):
        return None
    runtime = quantization_config.get("runtime")
    if not isinstance(runtime, Mapping) or "diffusion_step_policy" not in runtime:
        return None
    policy = runtime["diffusion_step_policy"]
    if not isinstance(policy, Mapping):
        raise TypeError(
            "quantization_config.runtime.diffusion_step_policy must be a mapping, "
            f"got {type(policy).__name__}"
        )

    unknown = set(policy) - _POLICY_FIELDS
    if unknown:
        raise ValueError(f"Unknown diffusion_step_policy fields: {sorted(unknown)}")
    missing = _POLICY_FIELDS - set(policy)
    if missing:
        raise ValueError(f"Missing diffusion_step_policy fields: {sorted(missing)}")

    schema_version = policy["schema_version"]
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != 1
    ):
        raise ValueError(
            f"diffusion_step_policy.schema_version must be the integer 1, got {schema_version!r}"
        )
    for field, expected in (
        ("type", "first_last_n"),
        ("index_space", "denoising_loop_iteration"),
        ("default_mode", "native"),
        # Which mode wins where the first and last windows meet. "a16" is what
        # the ``or`` below already yields, so this is checked rather than acted
        # on; any other value would need a different predicate.
        ("overlap", "a16"),
    ):
        if policy[field] != expected:
            raise ValueError(
                f"diffusion_step_policy.{field} must be {expected!r}, got {policy[field]!r}"
            )

    scope = policy["scope"]
    if not isinstance(scope, list) or not scope or not all(isinstance(s, str) for s in scope):
        raise TypeError("diffusion_step_policy.scope must be a non-empty list of strings")
    if _SCOPE_COMPONENT not in scope:
        return None

    reasoner = policy["reasoner"]
    if reasoner not in ("native", "a16"):
        raise ValueError(
            f"diffusion_step_policy.reasoner must be 'native' or 'a16', got {reasoner!r}"
        )

    return StepPrecisionPolicy(
        first_steps=_parse_step_range(policy["first_steps"], "first_steps"),
        last_steps=_parse_step_range(policy["last_steps"], "last_steps"),
        reasoner_high_precision=reasoner == "a16",
    )


def _parse_step_range(value: Any, name: str) -> int:
    if not isinstance(value, Mapping):
        raise TypeError(f"diffusion_step_policy.{name} must be a mapping")
    unknown = set(value) - _STEP_RANGE_FIELDS
    if unknown:
        raise ValueError(f"Unknown diffusion_step_policy.{name} fields: {sorted(unknown)}")
    missing = _STEP_RANGE_FIELDS - set(value)
    if missing:
        raise ValueError(f"Missing diffusion_step_policy.{name} fields: {sorted(missing)}")
    if value["mode"] != "a16":
        raise ValueError(f"diffusion_step_policy.{name}.mode must be 'a16', got {value['mode']!r}")
    count = value["count"]
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise TypeError(f"diffusion_step_policy.{name}.count must be a non-negative integer")
    return count


class StepPrecisionController:
    """Holds the activation precision selected for the current denoising step."""

    def __init__(self, first_steps: int, last_steps: int) -> None:
        if first_steps < 0 or last_steps < 0:
            raise ValueError(
                f"first_steps/last_steps must be non-negative, got {first_steps}/{last_steps}"
            )
        self.first_steps = first_steps
        self.last_steps = last_steps
        self.high_precision = False
        # One shared instance serves every Linear: these methods hold no
        # state, weights live on the module and are passed to apply().
        self._a16_method = W8A16FP8LinearMethod()
        # (module, its own FP8 method) for the projections this controller
        # steers. The FP8 instance is kept rather than rebuilt so restoring is
        # exact for anything that compares identity.
        self._governed: List[Tuple[Linear, FP8QDQLinearMethod]] = []

    def pin_unquantized(self, module: Linear) -> None:
        """Bind *module* to the 16-bit path for good.

        The reasoner's precision is stated by the policy rather than derived
        from a step, so it is bound once and never revisited.
        """
        module.quant_method = self._a16_method

    def govern(self, module: Linear, fp8_method: FP8QDQLinearMethod) -> None:
        self._governed.append((module, fp8_method))
        self._bind(module, fp8_method, self.high_precision)

    def _bind(self, module: Linear, fp8_method: FP8QDQLinearMethod, a16: bool) -> None:
        module.quant_method = self._a16_method if a16 else fp8_method

    def _rebind_all(self) -> None:
        for module, fp8_method in self._governed:
            self._bind(module, fp8_method, self.high_precision)

    def set_step(self, step_index: int, num_steps: int) -> None:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if step_index < 0 or step_index >= num_steps:
            raise IndexError(f"step_index must be in [0, {num_steps}), got {step_index}")
        # A single-step schedule is the warmup probe rather than a real
        # request, and treating every step as an edge step would make warmup
        # exercise a path the measured run never takes.
        if num_steps == 1:
            self.high_precision = False
            self._rebind_all()
            return
        self.high_precision = (
            step_index < self.first_steps or step_index >= num_steps - self.last_steps
        )
        self._rebind_all()

    def reset(self) -> None:
        self.high_precision = False
        self._rebind_all()


def linear_runs_high_precision(module: Optional[Linear]) -> bool:
    """Whether this Linear is currently on the 16-bit path.

    Activation-sharing callers quantize once above several Linears, which
    would defeat the 16-bit step, so they consult this before doing so. The
    bound method answers: only W8A16FP8LinearMethod asks for an unquantized
    activation, and it is bound exactly on the steps that want one.
    """
    if module is None:
        return False
    return module.requires_unquantized_activation


def install_step_precision(
    roots: Iterable[torch.nn.Module],
    controller: StepPrecisionController,
    always_high: bool = False,
) -> int:
    """Put every static-FP8 Linear under *roots* under step control.

    Must run after weight loading: binding earlier would put the 16-bit method
    in the path of the loader's ``isinstance`` checks. Returns the number of
    governed modules; 0 means this is not a static-FP8 model.

    ``always_high`` serves the reasoner. The understanding tower builds its KV
    cache on the first transformer call of a request and is cached after, so a
    step-indexed decision would only match the policy while that call happens
    to land inside a window -- true for the published 3/3 policy, false the
    moment one ships ``first_steps: 0``. Those projections are bound to the
    16-bit method once and never rebound.
    """
    governed = 0
    for root in roots:
        for module in root.modules():
            if not isinstance(module, Linear):
                continue
            existing = module.quant_method
            if isinstance(existing, W8A16FP8LinearMethod):
                # Already bound by an earlier install; nothing to re-derive.
                governed += 1
                continue
            if not isinstance(existing, FP8QDQLinearMethod):
                continue
            if always_high:
                controller.pin_unquantized(module)
            else:
                controller.govern(module, existing)
            governed += 1
    return governed
