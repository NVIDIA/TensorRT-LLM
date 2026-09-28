# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experimental FA4 dense forward tuning using per-call dependency controls.

Uses the source-built FA4 package shipped in the TRT-LLM wheel. Import this
module after flash_attn4 has installed the CUTLASS compatibility shims.
"""

from functools import lru_cache
from importlib.metadata import version

import torch

from ...autotuner import AutoTuner, OptimizationProfile, TunableRunner, TuningConfig

# Stable indices are persisted by AutoTuner. Bump the runner version if this changes.
_TACTICS = tuple((cta, freq) for cta in (False, True) for freq in (None, 0, 4, 8, 16, 32))
_TUNING_CONFIG = TuningConfig()


@lru_cache(maxsize=1)
def _require_tuning_api() -> None:
    from trtllm_flash_attn.interface import _flash_attn_fwd

    if getattr(_flash_attn_fwd, "visual_gen_tuning_api", None) != 1:
        raise RuntimeError(
            "The bundled FA4 package is missing its per-call tuning API. "
            "Rebuild TRT-LLM with scripts/build_wheel.py or install a matching TRT-LLM wheel."
        )


@lru_cache(maxsize=None)
def _device_identity(device: torch.device) -> tuple[str, tuple[int, int]]:
    return torch.cuda.get_device_name(device), torch.cuda.get_device_capability(device)


@lru_cache(maxsize=1)
def _cutlass_version() -> str:
    return version("nvidia-cutlass-dsl")


def can_tune(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> bool:
    """Limit the demo to dense SM100/SM103 FP16/BF16 head-dim-128 MHA."""
    if causal or any(t.ndim != 4 or t.device != q.device or t.dtype != q.dtype for t in (q, k, v)):
        return False
    if q.device.type != "cuda" or q.dtype not in (torch.float16, torch.bfloat16):
        return False
    if any(t.shape[-1] != 128 or t.stride(-1) != 1 or t.requires_grad for t in (q, k, v)):
        return False
    if k.shape != v.shape or q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        return False
    if min(q.shape[:3]) == 0 or k.shape[1] == 0:
        return False
    return _device_identity(q.device)[1] in ((10, 0), (10, 3))


class Fa4Runner(TunableRunner):
    """Time CTA count and exp2 emulation jointly without modifying FA4 globals."""

    def __init__(self, inputs: list[torch.Tensor], scale: float) -> None:
        from trtllm_flash_attn import utils

        _require_tuning_api()
        self.scale = scale
        self.device_name, self.capability = _device_identity(inputs[0].device)
        self.disable_2cta = utils._get_disable_2cta_default(is_fwd=True)
        self.clc = utils._get_use_clc_scheduler_default()
        self.tensor_metadata = tuple((str(t.dtype), tuple(t.stride())) for t in inputs)

    def unique_id(self) -> tuple:
        from trtllm_flash_attn._build_info import BUILD_ID

        # Shapes are keyed by AutoTuner; strides/dtypes and runtime controls are not.
        return (
            1,
            BUILD_ID,
            torch.__version__,
            torch.version.cuda,
            _cutlass_version(),
            self.device_name,
            self.capability,
            self.disable_2cta,
            self.clc,
            self.scale,
            self.tensor_metadata,
        )

    def get_valid_tactics(
        self, inputs: list[torch.Tensor], profile: OptimizationProfile, **kwargs: object
    ) -> list[int]:
        # Include the unmodified FA4 heuristic (including automatic split-KV).
        tactics = [-1]
        for idx, (use_2cta, freq) in enumerate(_TACTICS):
            if use_2cta and (self.disable_2cta or inputs[0].shape[1] <= 256):
                continue
            # SM103 has faster hardware exp2; retain the FA4 default as the control.
            if self.capability == (10, 3) and freq is not None and freq != 0:
                continue
            tactics.append(idx)
        return tactics

    def forward(
        self,
        inputs: list[torch.Tensor],
        *,
        tactic: int = -1,
        do_preparation: bool = False,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from trtllm_flash_attn.interface import _flash_attn_fwd

        tuning_kwargs = {}
        if tactic != -1:
            use_2cta, freq = _TACTICS[tactic]
            tuning_kwargs = {"use_2cta": use_2cta, "ex2_emu_freq": freq}
        output, lse, *_ = _flash_attn_fwd(
            *inputs,
            softmax_scale=self.scale,
            causal=False,
            softcap=0.0,
            return_lse=True,
            num_splits=0 if tactic == -1 else 1,
            **tuning_kwargs,
        )
        return output, lse


def tuned_forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use warmup's AutoTuner context; cache misses outside warmup use FA4 defaults."""
    inputs = [q, k, v]
    runner = Fa4Runner(inputs, scale)
    tuner = AutoTuner.get()
    # Profiling uses CUDA graphs itself. Never start a search inside an outer capture.
    if tuner.is_tuning_mode and torch.cuda.is_current_stream_capturing():
        return runner(inputs)
    selected_runner, tactic = tuner.choose_one(
        "visual_gen::fa4_dense", [runner], _TUNING_CONFIG, inputs
    )
    return selected_runner(inputs, tactic=tactic)
