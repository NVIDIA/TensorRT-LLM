# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load strategies: the ways weights reach a model that `ModelLoader` built.

`ModelLoader.load()` reads the model config, builds the model, and then needs
to fill its parameters. A *checkpoint loader* (`BaseCheckpointLoader`) knows
the checkpoint format: how to read the config, which tensors the files hold,
and how their names map onto the model. A *load strategy* decides how those
weights get into the model's GPU memory:

============================  ==============================  ===================================
Load strategy                 Chosen when                     Where the weights come from
============================  ==============================  ===================================
`CheckpointLoadStrategy`      ``load_format="auto"``          checkpoint files, via the loader
`MxLoadStrategy`              ``checkpoint_format="MX"``      a running donor over RDMA, or files
`ModelStreamerLoadStrategy`   ``checkpoint_io_policy=``       files, streamed by Run:ai Model
                              ``"model_streamer"``            Streamer (Kimi K3 only)
`GmsLoadStrategy`             ``load_format="gms"``           a GPU memory pool shared on a node
`DummyLoadStrategy`           ``load_format="dummy"``         random values, for tests
`VisionOnlyLoadStrategy`      ``load_format="vision_only"``   nothing; vision weights load early
============================  ==============================  ===================================

`select_weight_load_strategy()` picks the strategy, and `WeightLoadStrategy` lists the steps
`ModelLoader.load()` runs on it. Helpers shared by several strategies live in
`model_hooks` (module post-load hooks) and `post_transform` (whether MX and
GMS loads may share already-transformed weights). `model_streamer` also holds
`ModelStreamerPlan`, which a model's ``plan_model_streamer_load()`` returns to
say which tensors it needs and how to load them.
"""

from tensorrt_llm.checkpoint_format import CheckpointFormat
from tensorrt_llm.llmapi.llm_args import CheckpointIOPolicy, LoadFormat

from .base import ModelLoaderMetricNames, WeightLoadContext, WeightLoadStrategy
from .checkpoint import CheckpointLoadStrategy
from .dummy import DummyLoadStrategy, VisionOnlyLoadStrategy, initialize_dummy_weights
from .gms import GmsLoadStrategy
from .model_streamer import ModelStreamerLoadStrategy
from .mx import MxLoadStrategy

__all__ = [
    "CheckpointLoadStrategy",
    "DummyLoadStrategy",
    "GmsLoadStrategy",
    "ModelLoaderMetricNames",
    "ModelStreamerLoadStrategy",
    "MxLoadStrategy",
    "VisionOnlyLoadStrategy",
    "WeightLoadContext",
    "WeightLoadStrategy",
    "initialize_dummy_weights",
    "select_weight_load_strategy",
]


def select_weight_load_strategy(context: WeightLoadContext) -> WeightLoadStrategy:
    """Pick the load strategy for ``context.llm_args.load_format``.

    For ``load_format="auto"``, ``checkpoint_io_policy="model_streamer"``
    gets `ModelStreamerLoadStrategy`. Otherwise, for ``load_format="auto"``
    and the GMS writer, MX checkpoints get `MxLoadStrategy` and every other
    format gets `CheckpointLoadStrategy`.
    Under GMS the MX strategy raises when the checkpoint cannot be
    fingerprinted, because a GMS reader has no disk fallback.

    Raises:
        NotImplementedError: For a load format without a load strategy.
    """
    load_format = context.llm_args.load_format
    if load_format == LoadFormat.AUTO:
        if context.llm_args.checkpoint_io_policy == CheckpointIOPolicy.MODEL_STREAMER:
            return ModelStreamerLoadStrategy(context)
        return _checkpoint_load_strategy(context, fallback_on_artifact_error=True)
    if load_format == LoadFormat.GMS:
        return GmsLoadStrategy(
            context, inner=_checkpoint_load_strategy(context, fallback_on_artifact_error=False)
        )
    if load_format == LoadFormat.DUMMY:
        return DummyLoadStrategy(context)
    if load_format == LoadFormat.VISION_ONLY:
        return VisionOnlyLoadStrategy(context)
    raise NotImplementedError(f"No load support for load format: {load_format}")


def _checkpoint_load_strategy(
    context: WeightLoadContext, *, fallback_on_artifact_error: bool
) -> CheckpointLoadStrategy:
    """Return the strategy that reads ``context.checkpoint_loader``'s format."""
    if context.checkpoint_loader.checkpoint_format == CheckpointFormat.MX:
        return MxLoadStrategy(context, fallback_on_artifact_error=fallback_on_artifact_error)
    return CheckpointLoadStrategy(context)
