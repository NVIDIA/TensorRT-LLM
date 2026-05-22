# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-family adapter contract for VisGen-Auto.

A family adapter describes a Diffusers transformer family enough for the
auto path to capture and rewrite it: which Diffusers class it targets, how
to build example inputs for `torch.export`, which dims are dynamic, which
rewrite policy applies, and (optionally) Ulysses + custom-pass hooks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from ..config import DiffusionModelConfig
    from .pass_manager import PassManager
    from .policy import RewritePolicy


class VisGenFamilyAdapter(ABC):
    """Abstract base class for family adapters."""

    family: str = ""
    diffusers_transformer_cls: str = ""

    @abstractmethod
    def example_inputs(
        self,
        cfg: "DiffusionModelConfig",
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> tuple[tuple, dict[str, Any]]:
        """Return `(args, kwargs)` for `torch.export.export`.

        Tensors must be materialized on `device` with floating-point dtype
        `dtype`; integer tensors (timestep, ids) keep their natural dtype.
        Returning `args=()` and using kwargs for everything is conventional
        because Diffusers transformers take all inputs as kwargs.
        """

    @abstractmethod
    def dynamic_shapes(self, cfg: "DiffusionModelConfig") -> dict[str, Any]:
        """Return the `dynamic_shapes` arg for `torch.export.export`.

        Keys are kwarg names from `example_inputs`; values are dim specs as
        accepted by `torch.export` (e.g. ``{0: torch.export.Dim("B"),
        1: torch.export.Dim("S_img")}`` or ``None`` for fully static).
        """

    @abstractmethod
    def rewrite_policy(self, cfg: "DiffusionModelConfig") -> "RewritePolicy":
        """Return the rewrite policy for this family + config combination."""

    def default_quant_exclude_modules(self) -> list[str]:
        """Glob patterns for `nn.Linear` modules that should NOT be quantized
        when this family runs through the auto path.

        Default is empty; families with known quantization-sensitive layers
        override this. The patterns are merged with the user-supplied
        `quant_config.exclude_modules` (union — user can only add, never
        remove a family default by passing their own list).
        """
        return []

    # ------------------------------------------------------------------
    # Multi-GPU (Ulysses) hooks
    # ------------------------------------------------------------------
    @property
    def uses_internal_seq_shard(self) -> bool:
        """When True, the family shards the flat sequence INSIDE the captured
        model.forward (after patch+flatten) and all-gathers before output
        projection. `_GraphModuleAsTransformer` skips its pipeline-boundary
        shard/gather hooks for these families.

        Image-DiT families (FLUX, SD3, ...) have already-token-flat
        `(B, S_img, C)` inputs and stick with boundary sharding — default
        `False`. Video-DiT families (WAN, ...) have 5-D `(B, C, T, H, W)`
        pre-patch inputs where the flat-sequence half doesn't map cleanly
        to any single 5-D axis, so they override this to `True` and provide
        a `pre_capture_patch` that bakes the slice + gather into the
        captured graph.
        """
        return False

    def customize_passes(self, pm: "PassManager") -> None:
        """Optional hook called by ``apply_rewrites`` after the default pass
        list (built from ``RewritePolicy``) is constructed but before it runs.

        Adapters can splice in family-specific passes via the manager's
        ``insert_before`` / ``insert_after`` / ``replace`` / ``remove`` /
        ``append`` primitives. Default is a no-op — only families with a
        structural pattern not covered by the shared passes need this.

        Example::

            def customize_passes(self, pm):
                from .my_family_passes import fuse_my_thing

                pm.insert_after("fuse_qkv", Pass("fuse_my_thing", fuse_my_thing))
        """
        return None

    def pre_capture_patch(self, model, visual_gen_mapping) -> None:
        """Optional hook called by `AutoDiffusersPipeline` right before
        `torch.export.export`. Families with `uses_internal_seq_shard=True`
        monkey-patch the model's `forward` here to inject the per-rank
        sequence slice + `visgen_auto.all_gather_seq` call. Other families
        leave this as a no-op.

        `visual_gen_mapping` may be `None` (single-GPU); in that case the
        patch should be a no-op (no slicing needed).
        """
        return None
