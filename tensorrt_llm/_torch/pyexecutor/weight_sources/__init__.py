# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight sources: the ways weights reach a model that `ModelLoader` built.

`ModelLoader.load()` reads the model config, builds the model, and then needs
to fill its parameters. A *checkpoint loader* (`BaseCheckpointLoader`) knows
the checkpoint format: how to read the config, which tensors the files hold,
and how their names map onto the model. A *weight source* decides how those
weights get into the model's GPU memory:

============================  ==============================  ===================================
Weight source                 Chosen when                     Where the weights come from
============================  ==============================  ===================================
`CheckpointWeightSource`      ``load_format="auto"``          checkpoint files, via the loader
`MxWeightSource`              ``checkpoint_format="MX"``      a running donor over RDMA, or files
`GmsWeightSource`             ``load_format="gms"``           a GPU memory pool shared on a node
`DummyWeightSource`           ``load_format="dummy"``         random values, for tests
`VisionOnlyWeightSource`      ``load_format="vision_only"``   nothing; vision weights load early
============================  ==============================  ===================================

`select_weight_source()` picks the source, and `WeightSource` lists the steps
`ModelLoader.load()` runs on it. Helpers shared by several sources live in
`model_hooks` (module post-load hooks) and `post_transform` (whether MX and
GMS loads may share already-transformed weights).
"""

from tensorrt_llm.checkpoint_format import CheckpointFormat
from tensorrt_llm.llmapi.llm_args import LoadFormat

from .base import ModelLoaderMetricNames, WeightLoadContext, WeightSource
from .checkpoint import CheckpointWeightSource
from .dummy import DummyWeightSource, VisionOnlyWeightSource, initialize_dummy_weights
from .gms import GmsWeightSource
from .mx import MxWeightSource

__all__ = [
    "CheckpointWeightSource",
    "DummyWeightSource",
    "GmsWeightSource",
    "ModelLoaderMetricNames",
    "MxWeightSource",
    "VisionOnlyWeightSource",
    "WeightLoadContext",
    "WeightSource",
    "initialize_dummy_weights",
    "select_weight_source",
]


def select_weight_source(context: WeightLoadContext) -> WeightSource:
    """Pick the weight source for ``context.llm_args.load_format``.

    For ``load_format="auto"`` and the GMS writer, MX checkpoints get
    `MxWeightSource` and every other format gets `CheckpointWeightSource`.
    Under GMS the MX source raises when the checkpoint cannot be
    fingerprinted, because a GMS reader has no disk fallback.

    Raises:
        NotImplementedError: For a load format without a weight source.
    """
    load_format = context.llm_args.load_format
    if load_format == LoadFormat.AUTO:
        return _checkpoint_source(context, fallback_on_artifact_error=True)
    if load_format == LoadFormat.GMS:
        return GmsWeightSource(
            context, inner=_checkpoint_source(context, fallback_on_artifact_error=False)
        )
    if load_format == LoadFormat.DUMMY:
        return DummyWeightSource(context)
    if load_format == LoadFormat.VISION_ONLY:
        return VisionOnlyWeightSource(context)
    raise NotImplementedError(f"No load support for load format: {load_format}")


def _checkpoint_source(
    context: WeightLoadContext, *, fallback_on_artifact_error: bool
) -> CheckpointWeightSource:
    """Return the source that reads ``context.checkpoint_loader``'s format."""
    if context.checkpoint_loader.checkpoint_format == CheckpointFormat.MX:
        return MxWeightSource(context, fallback_on_artifact_error=fallback_on_artifact_error)
    return CheckpointWeightSource(context)
