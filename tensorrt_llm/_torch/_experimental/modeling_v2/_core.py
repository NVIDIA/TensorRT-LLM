# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The one piece of a model's core that is genuinely not model-specific.

Models in this tree share no code with each other -- duplication is the
accepted price of each being readable in isolation. `ModelingV2Core` is the
one exception, and a narrow one: every model's core derives from it, under
its own model-specific name (`GptOssModelingV2`, ...), but what it provides
is the contract-check *skeleton* only, not a policy any model's checkpoint,
weights or forward would need to differ over.

Everything a core actually does -- `__init__`, weight declaration, `forward`,
`build_layer_views` / `derive_after_load` -- stays on the subclass. This base
holds `_check_step_contract` and the two hooks it calls, plus the one line of
`__init__` every subclass would otherwise repeat verbatim.
"""

from typing import Any

from tensorrt_llm._torch._experimental.modeling_v2._router_index import step_contract_enabled
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModel


class ModelingV2Core(DecoderModel):
    """Shared base for a model's core.

    Holds the opt-in step-contract check and nothing else: the fields a
    model's step consumes differ by model, so the check is a skeleton here
    (`_probe_step_surface`, `_after_contract_check`) that each subclass fills
    in, not a body this class runs itself.
    """

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__(model_config)
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks for it: read once here
        # rather than per forward, and False for the whole life of a served
        # engine. "pending" rather than "enabled" because the check runs once
        # -- everything it looks at is fixed at engine construction.
        self._contract_pending = step_contract_enabled()

    def _check_step_contract(self, md: Any) -> None:
        """Opt-in first-forward fail-fast, run when TRTLLM_MODELING_V2_VALIDATE
        asks for it: the metadata fields a model consumes must exist (they
        are private trtllm surface). Everything checked is fixed at engine
        construction -- once per model instance is sound, and off in a served
        engine, where the only thing this could still do is fail.

        Called unconditionally every forward; the early return below is the
        gate. That costs one Python call per forward, not per layer, against
        a decode step measured in milliseconds -- accepted in exchange for
        not scattering the `_contract_pending` check across every call site.
        """
        if not self._contract_pending:
            return
        # Calling the projection is the check: it reads every metadata field
        # the model consumes, so a rename or removal upstream surfaces here
        # rather than mid-forward. Deriving it this way is the point -- a
        # hand-kept list of the same names drifts silently the first time
        # `_build_step_args` gains a field and nobody updates the copy.
        try:
            self._probe_step_surface(md)
        except AttributeError as exc:
            raise AssertionError(f"attention metadata surface drifted: {exc}") from exc
        self._after_contract_check(md)
        self._contract_pending = False

    def _probe_step_surface(self, md: Any) -> None:
        """Read every metadata field this model consumes, so a rename
        upstream surfaces here rather than mid-forward. Subclasses call
        their own `_build_step_args`."""
        raise NotImplementedError

    def _after_contract_check(self, md: Any) -> None:
        """Anything else a model wants done once, on the first forward."""
