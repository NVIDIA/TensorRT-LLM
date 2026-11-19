# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact shared final-head cast for one frozen pipeline."""

from __future__ import annotations

import types

import torch
from round5_shared_runtime import SharedRuntime, restore, same


class Runtime(SharedRuntime):
    def __init__(self, pipeline, *, timer, module, proof: dict) -> None:
        super().__init__(pipeline, timer=timer)
        assert proof["unary_qualified"] and proof["resources"]["cast"]["driver"]["local"] == 0
        self.blocks = 848
        self.threads = 256
        self.module = module
        self.proof = proof
        runtime = self

        def helper(model, projection, hidden: torch.Tensor) -> torch.Tensor:
            assert runtime.enabled
            if projection is model.proj_in or projection is model.audio_proj_in:
                runtime.counts["original_input_projection_calls"] += 1
                return runtime.original_helper(projection, hidden)
            eligible = (
                projection is runtime.video
                and hidden.dtype == torch.bfloat16
                and hidden.is_cuda
                and hidden.device == runtime.video.weight.device
                and hidden.is_contiguous()
                and tuple(hidden.shape) in ((1, 37776, 5376), (1, 37820, 5376), (1, 38169, 5376))
                and hidden.data_ptr() % 16 == 0
            )
            if eligible:
                value = runtime.module.cast(hidden, 1, runtime.blocks, runtime.threads)
                runtime.counts["fast_calls"] += 1
            else:
                value = runtime.original_helper(projection, hidden)
                runtime.counts["fallback_calls"] += 1
            if runtime.validate and projection is runtime.video:
                assert runtime.current is None and eligible
                expected = runtime.original_helper(projection, hidden)
                equal = same(value, expected)
                runtime.checks.append(
                    dict(kind="native_video_cast", byte_equal=equal, shape=list(hidden.shape))
                )
                assert equal
                runtime.current = (hidden, value)
                runtime.counts["video_source_checks"] += 1
            return value

        self.bound_helper = types.MethodType(helper, self.model)

    def configure(self, enabled: bool, *, validate: bool = False) -> None:
        assert self.model_stamp() == self.owner, "Frozen originalFP32 outputheads changed"
        assert (
            not self.pipeline.pipeline_config.cuda_graph.enable
            and int(self.model.sharder.size) == 1
        )
        if enabled:
            assert (
                not self.enabled
                and self.model.forward == self.original
                and self.model._projection_input == self.original_helper
            )
            assert not self.video._forward_pre_hooks and not self.audio._forward_pre_hooks
            self.validate = bool(validate)
            self.enabled = True
            self.model.forward = self.bound
            self.model._projection_input = self.bound_helper
            if validate:
                self.audio_handle = self.audio.register_forward_pre_hook(self.audio_pre)
        else:
            if self.enabled:
                assert (
                    self.model.forward == self.bound
                    and self.model._projection_input == self.bound_helper
                )
                if self.audio_handle is not None:
                    self.audio_handle.remove()
                    self.audio_handle = None
                restore(self.model, "_projection_input", self.helper_previous)
                restore(self.model, "forward", self.forward_previous)
            self.enabled = False
            self.validate = False
            self.current = None
